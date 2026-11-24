# World placement and streaming data

What is between the exported models/textures and a rendered world: the world metadata resources, the map
placement records, their transforms, bounds and LOD/streaming fields. Everything below was measured on the
Xbox Series 1.0.2.1 package. No game data is in this file; counts, aggregate bounds and format constants only.

## How much was read

`levels.rpf` is 76,155,065,056 bytes (70.92 GiB), so it was sampled in bounded windows rather than scanned
whole. `packs.rpf`, `data.rpf` and `common.rpf` are small enough to index completely. All indexes were made
with the corrected RPF9 carver from PR #90 (`--index-only`), which advances by the actual stream length. Every
stream is zlib; "unread" is non-zero bytes between streams that did not inflate, not a second storage format.

### First pass (four windows)

| Area | Bytes covered | Streams | Stored | Inflated | Zero | Unread | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `packs.rpf` (whole archive) | 4,798,213,264 | 144,364 | 4.269 GiB | 9.067 GiB | 3.08% | 1.39% | 95.5% |
| `levels.rpf` 0–2 GiB | 2,147,483,648 | 3,185 | 1.994 GiB | 13.063 GiB | 0.28% | 0.02% | 99.7% |
| `levels.rpf` 20–21 GiB | 1,073,741,824 | 16,683 | 0.982 GiB | 2.359 GiB | 1.48% | 0.42% | 98.2% |
| `levels.rpf` 40–41 GiB | 1,073,741,824 | 14,314 | 0.985 GiB | 1.774 GiB | 1.17% | 0.35% | 98.5% |
| `levels.rpf` 70–70.8 GiB | 858,993,459 | 14,445 | 0.787 GiB | 1.923 GiB | 1.50% | 0.09% | 98.4% |
| `data.rpf` (whole archive) | 481,544,304 | 28,509 | 0.394 GiB | 1.307 GiB | 7.88% | 4.21% | 87.9% |
| `common.rpf` (whole archive) | 6,642,192 | 954 | 0.005 GiB | 0.047 GiB | 5.85% | 8.64% | 85.5% |

### Second pass (bounded map-band scan)

The second pass used `tools/world_index.py` on the gaps and on every 2 GiB around the two known map areas. The
requested 72–76 GiB window does not exist: the file ends at 70.92 GiB. The tail window is 70.8–70.92 GiB
(134,143,917 bytes). Coverage is scanned compressed bytes / window span; streams and `0x7a` resources are from
the same tool.

| Window (GiB) | Streams | Coverage | 0x7a resources | Map files | Entities |
|---|---:|---:|---:|---:|---:|
| 4–5 | 22,991 | 97.5% | 648 | 0 | 0 |
| 6–7 | 23,433 | 97.0% | 914 | 0 | 0 |
| 7–8 | 12,897 | 98.9% | 1,892 | 498 | 36,480 |
| 8–9 | 17,619 | 97.8% | 4,430 | 1,398 | 88,228 |
| 9–10 | 18,886 | 97.9% | 4,107 | 1,068 | 75,344 |
| 10–11 | 10,239 | 99.0% | 1,042 | 164 | 18,306 |
| 16–18 | 35,216 | 97.9% | 9,449 | 2,174 | 145,118 |
| 22–23 | 16,824 | 98.2% | 3,522 | 821 | 50,438 |
| 26–27 | 17,974 | 98.0% | 4,737 | 992 | 68,492 |
| 30–31 | 16,946 | 98.2% | 2,959 | 152 | 10,533 |
| 34–35 | 13,082 | 98.6% | 2,361 | 98 | 4,754 |
| 38–40 | 28,331 | 98.5% | 4,591 | 209 | 13,476 |
| 42–43 | 10,725 | 98.7% | 2,874 | 111 | 9,566 |
| 46–47 | 9,085 | 99.4% | 2,906 | 105 | 7,999 |
| 50–51 | 13,634 | 98.6% | 1,737 | 121 | 6,086 |
| 52–54 | 29,483 | 98.5% | 5,062 | 214 | 11,942 |
| 54–56 | 28,185 | 98.5% | 4,715 | 226 | 10,658 |
| 56–58 | 33,248 | 98.3% | 7,346 | 276 | 16,731 |
| 58–60 | 30,811 | 98.5% | 5,136 | 215 | 9,842 |
| 60–62 | 30,411 | 98.5% | 5,041 | 178 | 9,766 |
| 62–64 | 39,004 | 98.1% | 2,372 | 104 | 4,006 |
| 64–66 | 59,986 | 97.3% | 127 | 0 | 0 |
| 66–68 | 48,706 | 97.5% | 8,203 | 0 | 0 |
| 68–70 | 50,412 | 97.6% | 137 | 0 | 0 |
| 70.8–70.92 | 2,353 | 97.7% | 0 | 0 | 0 |
| **new windows** | **620,481** | **98.2%** | **86,308** | **9,124** | **597,765** |

The second pass scanned 37,026,725,270 bytes (34.48 GiB) in 35.125 GiB of windows. Adding the three first-pass
`world_index` windows gives 39,984,950,733 bytes (37.24 GiB, 52.5% of the archive); adding the carver-indexed
0–2 GiB window gives 42,126,097,955 bytes (39.23 GiB, 55.3%). This is still a sample, not a census.

The first four `levels.rpf` windows cover 5,153,960,755 bytes, 6.77% of the archive. The sampled stream split
is ~2.36 GiB in 20–21 GiB, 1.77 GiB in 40–41 GiB and 1.92 GiB in 70–70.8 GiB; the first two GiB are anomalous
(see the metadata family below). Positions and counts are sample measurements, not whole-map totals.

## Composition

Resource classes after joining contiguous graphics pages to their system page. "model" means a system page that
carries a `.#dr`, `.#ft` or `.#dd` name; it is an upper bound on model/fragment resources, not proof that
`rpf9dump` decodes geometry from it. "respage" is a standalone `42 40 00 00` graphics page. "system" is a
zero-head `0x5...` system page with no drawable name found in the stream. Counts are resources; bytes are
inflated.

| Area | model | respage | system | 0x7a meta (first page) | 0x7a map (first page) | PSIN | other/text |
|---|---:|---:|---:|---:|---:|---:|---:|
| `packs.rpf` | 27,224 / 5.350 GiB | 205 / 0.082 | 47,933 / 0.417 | 4,678 / 0.530 | 0 | 586 / 0.010 | 35,229 / 2.678 |
| `levels` 0–2 GiB | 0 | 0 | 0 | 2,855 / 13.054 | 0 | 8 / <0.001 | 322 / 0.009 |
| `levels` 20–21 GiB | 2,956 / 1.170 | 500 / 0.798 | 2,207 / 0.214 | 2,728 / 0.053 | 630 / 0.019 | 46 / <0.001 | 5,218 / 0.106 |
| `levels` 40–41 GiB | 1,172 / 0.442 | 782 / 0.672 | 1,984 / 0.305 | 3,282 / 0.026 | 96 / 0.002 | 32 / <0.001 | 6,014 / 0.328 |
| `levels` 70–70.8 GiB | 5,782 / 0.931 | 395 / 0.877 | 271 / 0.065 | 67 / 0.001 | 0 | 10 / <0.001 | 1,622 / 0.049 |
| `data.rpf` | 2,917 / 0.390 | 5 / 0.001 | 229 / 0.065 | 15,768 / 0.375 | 0 | 2,281 / 0.035 | 2,758 / 0.427 |
| `common.rpf` | — | — | — | — | — | — | 954 / 0.047 (257 XML docs) |

`data.rpf` also has 1,263 `RBF0` metadata resources (0.008 GiB) and 87 `GFX*` shader blobs (0.002 GiB).
`common.rpf` is XML config, not placement: 3,315 `<filename>` paths, 2,267 unique `.rpf` paths, including
519 `_metadata.rpf`, 117 `_merged.rpf`, 93 `_interiors.rpf`, 51 `_occl.rpf` and 2 `_occl_2.rpf` map-cell
archives. Those names explain the cell layout, but the outer `levels.rpf` does not expose them; its directory
table stays encrypted.

The 20–21 GiB map count needs a correction: a first system page of exactly 65,536 bytes is often followed by a
small continuation stream (usually 1–2 KiB) that the next stream scan alone would miss. `tools/world_index.py`
joins that continuation when the schema still has unresolved pointers. The join turns 630 first-page maps into
698 complete map files and 55,564 entities into 75,551. The other first-pass windows did not change. This is a
format rule, not a heuristic: the continuation is reached by system pointers in the first page, and the joined
schema then validates.

### Where the maps are

Map placement is not clustered in a few page ranges. It occupies one continuous band, bounded at the tested
window granularity:

- 0–2, 4–5 and 6–7 GiB: **0 maps**; 2–4 and 5–6 GiB were not sampled;
- 7–8 GiB: the band starts, 498 maps; the first map resource is at `0x1e0bed010` (7.51 GiB);
- 8–64 GiB: every sampled window has maps; density ranges from 96 to 1,398 maps per GiB (96 to 2,174 per
  sampled window);
- 62–64 GiB: the last sampled map is at `0xfb883d010` (62.88 GiB);
- 64–70.92 GiB: sampled 64–66, 66–68, 68–70, 70–70.8 and 70.8–70.92 GiB, **0 maps**.

That is a 57 GiB map band. The first pass's 794 files were not a 20/40 GiB cluster: 20–21 and 40–41 GiB are
inside the band, while its 0–2 and 70–70.8 GiB windows are outside it. The revised answer is that maps are spread
through roughly 80% of `levels.rpf`, with local density varying by more than 14×.

Within a window the maps are also packed in runs, not spread evenly. Median spacing between consecutive map
resource offsets is 8–24 KiB; runs are separated by hundreds of MiB (largest gaps 0.4–1.0 GiB). For example,
8–9 GiB has 1,398 maps over an 842 MiB span and 16–18 GiB has 2,174 over 1,331 MiB, while 58–60 GiB has 215
maps spread over 1,356 MiB in 30 runs.

## Signatures and strings

The class names are not stored as text: `CMapData` and `CInstance` had zero literal hits in `common.rpf` and
the sampled `levels.rpf` streams. The map roots are found by their struct hashes (`d3593fa6`, `ce501483`),
which are the case-sensitive joaat of `CMapData` and `CEntityDef`. Readable strings in the world streams are
model/material names (`.#dr`, `.#ft`) and, in `data.rpf`, UI/dialogue text; `ipl`, `stream`, `lod`, `cell`,
`zone` hits were substrings of those names or of unrelated text, not chunk tags. No repeated 4×4 or 3×4 matrix
table was found.

Byte offsets used: the bounded windows in the two tables above. The first map resource in the 20–21 GiB window
is at 21,588,586,512 (`0x506c7b010`); the first in the 40–41 GiB window is at 42,953,629,712
(`0xa003c6010`). The `0DRP` signature is at +0x10 of a map/meta stream; in a `PSIN` resource the same tag is in
the footer.

## The 0x7a resource

World metadata and map placement are **self-describing `0x7a` resources**, not `PSIN` and not `RBF0`. A carved
stream that starts a resource has 8 zero bytes, a `0x5...` page-map pointer at +0x08, then:

```
+0x10  "0DRP" + u8 0x7a          magic + version
+0x20  u32 root object count
+0x28  u64 -> struct table       32 bytes per struct
+0x30  u64 -> enum table
+0x38  u64 -> instance table     16 bytes per block
+0x48  u16 structs, u16 enums, u16 blocks, u16 reserved
```

Struct record (32 bytes): name hash, signature, flags, 0, `u64 -> member table`, size, 0, member count.
Member record (16 bytes): name hash, byte offset, type, subtype, reference, extra. Instance block (16 bytes):
struct hash, byte size, `u64` pointer. The pointers are `0x50000000 | file offset`; a `0x6...` pointer would
reach an appended graphics page. Fields are not named on disk. Their 32-bit hashes are the **case-sensitive
joaat** of the engine names; matching those hashes is how the fields below were identified.

The member `type` byte uses the PSIN type enum ([psin.md](psin.md#layouts)). Measured in the `CMapData`
(`d3593fa6`) and `CEntityDef` (`ce501483`) member tables of three carved `levels.rpf` maps: `0x34` is the
`4×f32` quaternion (`rotation`), `0x32` a 12-byte vec3 (`position`), `0x33` its 16-byte padded form (the four
extent triples), `0x21` f32 (`scaleXY`, `scaleZ`, `lodDist`, `childLodDist`), `0x12` i16 (`parentIndex`,
`parentInstance`), `0x15` u32 (`flags`, `numChildren`), `0x08` u64 (`entity id`), `0x60` an 8-bit enum
(`lodLevel`, `priorityLevel`), `0x4A` a u32 name hash, `0x52` and `0x5C` the two array forms, and `0x05` a struct
reference whose `extra` word is the referenced struct hash. `0x0C` is a 16-byte vector in the PSIN table but did
not occur in these schemas. 6stars independently reads these members with the same codes; no code or table was
copied, and its repository has no licence.

Every 0x7a header is checked the same way: the struct/instance table pointers must lie inside the stream, every
struct's member table must lie inside the stream, and at least one block must have a known struct and a
resolvable pointer. A block whose struct hash is not in the table is kept opaque rather than guessed (some
resources use a page block there). Entity blocks must divide cleanly by the entity struct size before entities
are counted. A stream whose tables point past its end is treated as a split resource and its continuation is
joined, rather than rejected.

Two structs identify a map-placement file: root `d3593fa6` (`CMapData`, 0x460 bytes in the sample) and entity
`ce501483` (`CEntityDef`, 0x90 bytes). Both are the case-sensitive joaat of those class names, which confirms
both the names and the hash function. Files with no entity/root struct are counted as `meta`, not placement.

`0DRP` appears in one other place: at the end of a `PSIN` resource it is the page-directory chunk, followed by
the same `0x7a` u32 and a `CHKS` chunk. It is not a map file; it describes the pages of the model/texture
resource. `data.rpf` contains 15,768 such 0x7a/page-directory resources (most are page directories),
`packs.rpf` 4,678, and none of them has a `CMapData` or `CEntityDef` struct. The task's "0DRP page
directories" and a map file share the magic, not the payload type.

### The first two GiB are a different family

The 0–2 GiB window is 89.6% 0x7a resources (2,855 first pages; 2,824 validate completely), 13.05 GiB inflated,
and **none contain an entity block**. All 2,824 carry one identical 7-struct schema:

```
6c4d9553 71ca4525 ba51bd1e c2b80676 c69192ec e33df1c6 fb42c06a
```

Their blocks include 9,546 blocks of exactly 1,048,576 bytes, which at 16 bytes per slot is 65,536 slots: a
hash-table family, like the slot tables in the text containers. There are also many 16/576/1152/1728-byte
blocks. The header and schema are readable; the field semantics are not. This family is **not** placement and
should not be decoded as entity tables. It is the main reason the first two GiB look so different from the rest.

## Map placement records

Decoded from the entity schema; offsets below are the observed member offsets in the sampled files. `position`
is RAGE world space in metres, Z up. `rotation` is the stored quaternion and is the inverse of the entity's
world rotation (conjugate it for rendering). `scaleXY` applies to X/Y, `scaleZ` to Z.

| Field | Hash | Offset | Type | Meaning |
|---|---|---:|---|---|
| `archetypeName` | `a023a02c` | +0x44 | u32 | hash of the archetype/model name |
| `position` | `01166164` | +0x38 | 3×f32 | world position, metres |
| `rotation` | `77dd4010` | +0x10 | 4×f32 | quaternion x,y,z,w, stored inverse |
| `scaleXY` | `9ca32637` | +0x54 | f32 | XY scale |
| `scaleZ` | `10fb7c42` | +0x58 | f32 | Z scale |
| `flags` | `67d26872` | +0x48 | u32 | entity flags |
| entity id | `1b60404d` | +0x08 | u64 | per-entity id (not a name key; not unique, see world_names.md) |
| archetype key | `c8a7b90a` | +0x20 | u64 | per-archetype key: MurmurHash3 x64 128 low 64 of the name |
| `lodLevel` | `6c8f1715` | +0x7e | u8 | 0 HD, 1 LOD, 2 SLOD1, 3 SLOD2, 4 SLOD3, 5 ORPHAN_HD, 6 SLOD4 |
| terrain parent-map selector | `84799847` | +0x7f | i8 | zero-based slot in the source map's parent-map reference array, or -1 in the measured terminal rows |
| `parentIndex` | `d89231bd` | +0x5c | i16 | parent entity index (signed; -1 = none) |
| `parentInstance` | `e74c3c23` | +0x5e | i16 | parent batch instance |
| `numChildren` | `a687ac89` | +0x68 | u32 | linked children |
| `lodDist` | `bf74dfa7` | +0x60 | f32 | LOD distance; -1 = use archetype default |
| `childLodDist` | `ca974bcd` | +0x64 | f32 | child LOD distance |
| `priorityLevel` | `2691f019` | +0x7c | u8 | priority |

The 2,716 entity schemas sampled for [world_names.md](world_names.md) all use the two 64-bit members above.
The earlier `guid` reading (`9a7b6e6d`, +0x20) was not a member of those schemas: +0x08 changes per placement,
while +0x20 is one value per archetype. [world_names.md](world_names.md) gives the measured formulas and the
name-resolution coverage.

The map root carries `name` (`e6e17f14`, +0x08, u32 hash), `parent` (`a99c4651`, +0x68, u32 hash),
`streamingExtentsMin` (`dd22821f`) / `streamingExtentsMax` (`a22ea335`), `entitiesExtentsMin` (`1c75bcf1`) /
`entitiesExtentsMax` (`6d074437`), `flags` and `contentFlags`. The extents are reached through the root's member
table, not fixed offsets.

### Name resolution

`archetypeName` and map `name`/`parent` are the lower-cased joaat of the model or map name. The per-archetype
+0x20 key and the map +0x10 key are MurmurHash3 x64 128 (seed 0, low 64); they confirm a 32-bit match instead of
assuming it. A local 79,738-name dictionary resolved 5,107/24,105 distinct archetype hashes (8,263/203,799
entities) in two 1 GiB windows, every one confirmed by the 64-bit key, and rejected one 32-bit-only candidate.
The same dictionary resolved 5/23,535 map names and 1/2,089 parent rows, all confirmed by the map key. This is a
name-source lower bound, not a formula limit: with the names `tools/rpf9_names.py` lists in `levels.rpf`, 99.85%
of the 1,670,435 entities a whole-archive `world_index.py` run reads resolve, all verified by the 64-bit key. See [world_names.md](world_names.md) and
`tools/world_names.py`.

## Archetype definitions (`CMapTypes`)

The `.ytyp` role. A `CMapTypes` resource (root `d98bb561`, 0x50 bytes) is a `0x7a` metadata resource like a map
file, counted as `meta` by `world_index.py`. Its archetypes are stored the way entities are: one block of
fixed-size instances per struct, `CBaseArchetypeDef` (`82d6fc83`, 0xb8 bytes) and `CMloArchetypeDef`
(`10506455`, 0x220 bytes). The interior struct repeats the base members at the same offsets and adds its own
from +0xb8. `CTimeArchetypeDef` (`76b0c56c`) did not occur in the measured window. All three names, and the
member names below, are case-sensitive joaat. `tools/world_archetypes.py` finds every field by hash in the
resource's own member table.

| Field | Hash | Offset | Type | Meaning |
|---|---|---:|---|---|
| `lodDist` | `bf74dfa7` | +0x38 | f32 | default LOD distance (an entity's `lodDist` of -1 uses it) |
| `specialAttribute` | `6c1523e4` | +0x3c | u32 | |
| `bbMin` / `bbMax` | `d9ef8236` / `e78aa618` | +0x40 / +0x4c | 3×f32 | model-space bounding box |
| `bsCentre` / `bsRadius` | `d61872ff` / `82e8c5de` | +0x58 / +0x64 | 3×f32 / f32 | bounding sphere |
| `flags` | `67d26872` | +0x68 | u32 | |
| `hdTextureDist` | `ad5d5b4c` | +0x70 | f32 | |
| `name` | `e6e17f14` | +0x78 | u32 | lower-cased joaat of the archetype name; what `CEntityDef.archetypeName` holds |
| `clipDictionary` | `19471791` | +0x7c | u32 | hash |
| `physicsDictionary` | `d3c717fc` | +0x80 | u32 | hash |
| `assetType` | `069004b2` | +0x84 | enum (u32) | see below |
| `assetName` | `9a77c7c5` | +0x88 | u32 | hash of the model resource name |
| `extensions` | `f17e7f28` | +0x90 | array | |

Unnamed members: five u64 (type `0x4f`) at +0x08..+0x28, `c067a638` (+0x30), `8f215e15` (u32, +0x6c),
`66d9cb52` and `96b4753e` (u32 hashes, +0x74 and +0x8c), `9a7b6e6d` (+0xa0), `1b60404d` (u64, +0xa8, the same
member hash as the entity id) and the enum `07d164a8` (+0xb0).

Measured in the 8–9 GiB window of `levels.rpf`: 119
`CMapTypes` resources hold 31,800 definitions (31,785 base, 15 interior) for 14,955 distinct names. Where a
name is defined more than once, every copy has the same box.

`assetType`, measured by name: each distinct `name` of the window against the `name.#dr` / `name.#ft` strings
that `tools/rpf9_names.py --types dr,ft,dd` lists in the whole `levels.rpf` (266,456 `.#dr`, 50,801 `.#ft`, no
`.#dd`):

| `assetType` | names | listed as `.#ft` | listed as `.#dr` | not listed |
|---:|---:|---:|---:|---:|
| 1 | 3,614 | 3,611 | 0 | 3 |
| 2 | 8,689 | 0 | 8,681 | 8 |
| 3 | 2,637 | 0 | 2,327 | 310 |
| 0 | 15 | 0 | 0 | 15 |

1 is a fragment and 2 a drawable; no name crosses over. 3 is named like a drawable but is a different type;
that it is the drawable-dictionary type is an inference (no `.#dd` string confirms it). The 15 type-0
definitions are the interiors. For types 1–3, `assetName` equals `name` in every definition; for the
interiors it does not.

### Checking transforms against the map extents

A map root's `entitiesExtentsMin`/`Max` is a box the map's own entities fill. With the definitions, that box can
be rebuilt independently: transform each entity's archetype box and take the union. `world_archetypes.py
--check-extents` does this for every map whose entities all have a definition in the window, with a 5 cm
tolerance per bound. In the 8–9 GiB window, 1,088 of 1,398 maps are fully defined:

| 827 maps with a rotated entity | stored quaternion | conjugated quaternion |
|---|---:|---:|
| union equals the stored extents | 236 | **616** |
| median of the largest bound error | 0.627 m | **0.00016 m** |

This confirms the conjugate rule of [Map placement records](#map-placement-records) on box geometry, not only
on positions. The 211 maps that match neither have extents larger than the union, never smaller (827/827
inside): something the entity list does not hold also widens them.

**The stored extents ignore the entity scale.** Of the 179 fully-defined maps with a scaled entity, 155 match
the union built without scale and 101 the union built with `scaleXY`/`scaleZ`; with scale, 41 unions are larger
than the stored extents, without scale none is. This says how the extents were computed, not how the renderer
applies scale; the scale rule of [Map placement records](#map-placement-records) stays as it is.

```powershell
python -I tools\world_archetypes.py <game>\levels.rpf --start 8589934592 --limit 1073741824 --check-extents --out <work>\world\archetypes_8-9.json
```

### Measured placement summary

| Scope | Map files | Entities |
|---|---:|---:|
| 0–2 GiB (first pass) | 0 | 0 |
| 20–21 GiB (first pass) | 698 | 75,551 |
| 40–41 GiB (first pass) | 96 | 6,912 |
| 70–70.8 GiB (first pass) | 0 | 0 |
| **first-pass subtotal** | **794** | **82,463** |
| **second-pass new windows** | **9,124** | **597,765** |
| **measured total** | **9,918** | **680,228** |

The 20–21 GiB LOD split: HD 53,551; LOD 2,119; SLOD1 230; SLOD2 40; SLOD3 11; SLOD4 10; ORPHAN_HD 19,590.
`lodDist` is -1 for 74,375 entities (98.4%, meaning archetype default); the rest are 10–1200 m.
`parentIndex` is -1 for 19,874 entities (26.3%); the rest point at another entity. The 40–41 GiB window splits
HD 5,354; LOD 1,248; SLOD1 278; SLOD2 25; SLOD3 6; SLOD4 1.

Every sampled map file's three streaming extents are equal, i.e. a cube. Cube sizes range from 20 m to
42,744 m; common values in the 20–21 GiB window are 240, 300, 500, 600 and 2000 m. That per-file cube is the
cell/zone metadata a streamer needs. **No global cell table, zone table or 4×4/3×4 matrix array was found** in
the sampled streams. Placement is per-entity position + quaternion + scale, not a matrix; the repeated 16-byte
pattern in the 0–2 GiB family is a hash-table slot, not a matrix row.

### How many are there?

A full sharded census of the `levels.rpf` data area now gives a measured total: **23,535 map resources /
1,670,442 placement entities**. Every map row is exact against its declared entity size and every expected
shard completed. The earlier bounded passes measured 9,918 files and 680,228 entities in 55.3% of the archive;
that was a lower bound, and the full pass settles the extrapolation. See
[world_census.md](world_census.md) for the shard layout, the exactness condition and the 98.32% stream
coverage.

The full count is lower than SixNG's published 30,785 map files / 2,249,490 entities
(https://github.com/holdmysocks/SixNG), and the whole difference is now accounted for map by map (see
[world_census.md](world_census.md#the-sixng-difference-map-by-map)). The 23,528 maps both count agree
on every entity count. SixNG also counts 7,257 `0x7a` resources that have `CEntityDef` objects but no
`CMapData` struct: 961 interior archetype resources (`CMloArchetypeDef`, 540,665 entities in interior
space) and 6,296 `35abc248` entity groups (39,622 entities in group-local space). It misses 7 maps
(1,239 entities) whose entity data is in a continuation page. 7,257 − 7 = 7,250 maps and 580,287 −
1,239 = 579,048 entities. The 23,535 / 1,670,442 census figure is the count of world placement
lists; interiors and entity groups need the transform of whatever places them.

## What is readable, what is hash-only

Readable:
- the 0x7a header, schema, struct/member tables and instance blocks;
- map file count, per-file streaming/entity extents;
- archetype definitions in `CMapTypes`: name, asset name and type, bounding box and sphere, default LOD
  distance and flags ([Archetype definitions](#archetype-definitions-cmaptypes));
- every entity's position, rotation, scale, flags, per-entity id, per-archetype 64-bit key, archetype hash and LOD
  fields;
- in `common.rpf`, the full XML content-definition path list and cell names.

Hash-only or opaque:
- the RPF directory table and therefore original file names; resource names must be recovered from payloads;
- map file `name`/`parent` (32-bit hashes; `parent` is non-zero in 2,089 of the 23,535 census maps). A local
  dictionary resolves 5 names and 1 parent; all five names are confirmed by the map root's 64-bit key. The
  remaining map-cell names are not in the local model/XML exports;
- `archetypeName` values are 32-bit hashes, but nearly all of them resolve. With the `name.#dr`/`name.#ft` strings
  that `tools/rpf9_names.py` lists in `levels.rpf` as the dictionary, 1,667,978 of 1,670,435 entities (99.85%,
  227,422 of 228,681 distinct hashes) resolve, every one verified by the entity's 64-bit key; see
  [world_names.md](world_names.md#whole-archive-coverage). The earlier figures (5,107 of 24,105 hashes, 8,263 of
  203,799 entities = 4.05%, in two 1 GiB windows) used only names already exported locally;
- texture keys do not appear in placement records. They are 64-bit asset keys in model materials, so a runtime
  resolves placement → archetype → model → material asset key;
- the 0–2 GiB 7-struct family's field semantics and most of its payload;
- `PSIN` page-directory internals beyond the chunk tags (`MNEL`/`LSNS`, `LGID`/`LRSC`, `0DRP`/`0x7a`, `CHKS`);
- texture dictionaries: no `.#td` system page was found in the sampled `levels.rpf` windows. World textures in
  this package are the loose RDS5 pools under `data/ttdb` plus `textures.rpf`, so placement records do not
  carry texture names, only material/asset keys in the model resources.

## Location, interior, river and road names (`GameShape__Data__LocationNameLoopkup`)

The place names shown to the player come from one `GameShape__Data__LocationNameLoopkup` XML document in
`common.rpf` (the root name keeps its `Loopkup` spelling). Counts only; no names or keys.

| Section | Items | Content |
|---|---:|---|
| `LocationNames/Item key=...` | 2,318 | `Name` (empty in 382); 46 items also name a `TextureDictionary` (2 distinct) and a `Texture` (46 distinct) |
| `InteriorNames/Item key=...` | 1,358 | 2,280 inner `Item`s, each `Name` (empty in 434) and `Pos` (`x y z`); 1 position in 1,192 interiors, 2 or more in the other 166 |
| `RiverNames/Item key=...` | 66 | the name as item text |
| `RoadNames/Item key=...` | 409 | `Name` |

- Keys are plain text, distinct within each section, and no key is shared between the location, river and road
  sections.
- **All 2,280 interior positions lie inside the world rectangle** shared by the `PAMH` heightmap and the region
  tint map (x -13,312..5,120, y -9,728..11,264); none is at the origin. Heights run from -78 to 821 (median 5).
- **Interior keys are interior archetype names.** 1,357 of the 1,358 `InteriorNames` keys, lower-cased, are the
  joaat `name` of a `CMloArchetypeDef` in `levels.rpf`, and all 1,357 are confirmed by that archetype's 64-bit key
  (MurmurHash3 x64 low 64 of the lower-cased key at +0x08, member `8bfb6095`). The remaining key names no interior
  and its hash occurs in no hash member of `levels.rpf`. The earlier "own namespace" reading came from a 1 GiB
  window holding 15 of the 1,706 interior archetypes and from the model catalog, which has no interiors. See
  [Interior keys and interior placements](#interior-keys-and-interior-placements).
- None of the location, river or road keys is an archive entry name or the joaat of a model archetype name in the
  local catalog; the two texture-dictionary names match archive entries but not `.ytd` ones. The 46 textures
  match no texture-dictionary item in the catalog either.

### Interior keys and interior placements

`tools/world_interior_names.py` reads every interior definition and every interior placement in `levels.rpf`
and joins them with the `InteriorNames` section (full archive, `--jobs 15`, about 2 minutes):

| Measured | Count |
|---|---:|
| `CMloArchetypeDef` definitions / distinct names | 1,706 / 1,706 |
| `CMloInstanceDef` placements / distinct interiors placed | 2,590 / 1,704 |
| placements of a name with no `CMloArchetypeDef` | 0 |
| keys naming an interior (confirmed by the 64-bit key) | 1,357 (1,357) |
| keys whose interior is placed | 1,356 |
| ... with as many positions as placements | 1,268 |
| ... with more positions than placements | 64 |
| ... with fewer positions than placements | 24 |
| ... whose placements have no `position` to measure from | 0 |
| inner-item `Pos` values that are not three numbers (dropped) | 0 |

So a key selects one interior archetype, and its inner items are, in most cases, one per placed copy of that
interior: the `Name` of each item labels that copy.

Interiors are not placed by `CEntityDef`: no entity in the 1,670,435 rows of `world_index.py --entities-out`
uses an interior name. They are placed by `CMloInstanceDef` (`09cc28be`, 0x250 bytes), which carries the entity
members `rotation` (+0x10), `position` (+0x38), `archetypeName` (+0x44), `scaleXY` (+0x54) and `scaleZ` (+0x58)
plus interior-specific members. These 2,590 placements are separate from the 23,535 map / 1,670,442 entity
census, which counts `CEntityDef` only.

The positions sit at the edge of their own interior, not at its origin or centre. Distance from each of the 2,278
positions to the nearest placed copy of its own interior's box (0 = inside):

| Rotation | Inside | ≤ 1 m | ≤ 2 m | ≤ 5 m | ≤ 25 m | ≤ 100 m | Median | In a non-`limbo` room box |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stored quaternion | 315 | 1,015 | 1,189 | 1,443 | 1,867 | 2,199 | 1.72 m | 215 |
| conjugated | 571 | 950 | 1,121 | 1,416 | 1,863 | 2,199 | 2.10 m | 484 |

The interior and room boxes are scaled by each placement's `scaleXY` and `scaleZ` before measuring. Only 6 of
the 2,590 placements are scaled at all (`scaleZ` 0.998), so this changes two room hits per convention and none
of the distances. A point within 1e-6 m of an interior or room face counts as inside it (`EPS`), since the
inverse of a float32 rotation can leave a face point about 1e-17 m outside. Positions are float32 and resolve to
about 1 mm near the map edge, so a tolerance this far below that decides as exact arithmetic would; on the full
archive it changes no count above.

The interior box's largest side has a median of 16.7 m. Most positions are therefore just outside or just inside
the interior shell, which fits an entry or label point; that they are doors is a hypothesis (portals were not
compared). The conjugated rotation puts more positions inside rooms, consistent with the entity rule in
[Map placement records](#map-placement-records); it is not a separate proof of it. 79 positions are more than
100 m from every copy of their interior.

Interior rooms (`CMloRoomDef` `0b1811f1`, 0x70 bytes: `name` +0x00, `bbMin` +0x10, `bbMax` +0x20) are reached
through the interior's array members, which use the `InstanceList` reference of
[Instanced props and grass](#instanced-props-and-grass) (1-based block index, offset from bit 20, u16 count). Room
names are lower-cased joaat; all 1,706 interiors have rooms, and in every one the first room is `limbo`
(`interiors_first_room_limbo`). A room's hash member `a261e0a9` (+0x34) holds the joaat of 267 interior keys
(`keys_in_a_room_key_member`).

```powershell
python -I tools\gamedata_reader.py <work>\common --root GameShape__Data__LocationNameLoopkup --out <work>\gamedata
python -I tools\world_interior_names.py <game>\levels.rpf --names <work>\gamedata\GameShape__Data__LocationNameLoopkup_<offset>.json `
    --jobs 15 --out <work>\world\interior_names.json
python -I tests\test_world_interior_names.py
```

## What a runtime needs

1. **RPF9 page reader.** Read the `9FPR` header; the data area starts at the u32 at +0x04. Streams start on a
   16-byte boundary and are zlib. Join a system page and its contiguous graphics pages; for 0x7a resources also
   join the small continuation page when the schema's pointers reach it.
2. **0x7a schema parser.** Validate the header and tables, then read instance blocks. Do not assume fixed
   offsets: field offsets come from the resource's member table. The map root/entity hashes above identify
   placement files.
3. **Transforms.** Apply `position`, conjugate the stored quaternion for world rotation, and use `scaleXY` /
   `scaleZ`. The source is RAGE Z-up; glTF-style Y-up is `(x, z, -y)` with the vector part rotated accordingly.
4. **Archetype keys.** Resolve the 32-bit `archetypeName` hash to an exported model. Confirm the match with the
   per-archetype 64-bit key at +0x20 (MurmurHash3 x64 low 64); do not use the +0x08 per-entity id as a name key.
   Hash-only records are normal; the runtime needs a hash→model table, with the bbox/archive-order fallback for
   names that were not recovered.
5. **Streaming cells.** Load map files by their streaming AABB (cube) and use `lodLevel`, `lodDist`,
   `childLodDist`, `parentIndex`, `parentInstance` and `numChildren` to switch HD/LOD/SLOD. Load the HD model
   first, then its parent/children; do not use one global LOD distance.
6. **Model/texture layer.** The existing manifest gives names/bounds; texture keys are 64-bit asset keys in
   model materials. Placement itself contains neither geometry nor texture names.

## Region tint map (`RegionTintMap`)

One `RegionTintMap` XML document in `common.rpf` (see [gamedata.md](gamedata.md)) places a single streamed,
tiled tint texture over the world. Counts only; no names.

| Field | Value |
|---|---|
| `RegionTintMaps/Item` | 1 |
| `WorldAABBMin`, `WorldAABBMax` | x -13,312 to 5,120, y -9,728 to 11,264 (18,432 x 20,992 units) |
| `StreamingDesc/NumberOfTilesX`, `Y` | 36 x 41 |
| `StreamingDesc/TileSize` | 512 x 512 |
| `TintMapType`, `RegionTintMapFilename` | one token each |
| `channelRed`, `channelGreen`, `channelBlue`, `channelAlpha`, `settings/pairs` | empty |

Two exact matches tie this to the rest of the world data:

- The tiles cover the box exactly: 36 x 512 = 18,432 and 41 x 512 = 20,992, so there is no partial tile.
- The box is identical to the six-float bounds of the `PAMH` max/min heightmap in [gamedata.md](gamedata.md),
  `(-13312, -9728)` to `(5120, 11264)` in x and y. The tint map and the coarse heightmap therefore share one
  world rectangle, and that rectangle also has the same minimum x as the four `GridAtlasInfo` grids.

Entity positions from a local index of map resources (2,036,542 placements, 2,011,519 of them in `levels.rpf`)
lie within x -9,955 to 4,577 and y -8,096 to 9,440 at the 0.1th/99.9th percentiles, inside this rectangle.

The texture named by `RegionTintMapFilename` is not an archive entry or texture-dictionary item in the local
catalog, so the tile contents are unknown. With all four channel names empty, which tint each channel carries
is not stated here either.

## Tool

`tools/world_index.py` scans an RPF9 archive or a directory of carved streams and writes JSON/CSV with
`offset, size, signature, version, kind, structs, blocks, entities, extents, position bounds, LOD counts`,
map name/parent hashes and the map root's 64-bit name key. It implements the header/schema validation above
and the continuation join. It does not decrypt anything and does not extract models/textures. Its JSON/CSV
output is aggregate (one row per resource). `--entities-out` adds a local-only JSON-lines file with one row per
complete entity (archetype hash and key, position, stored rotation, scales, LOD fields), the input of
`world_build.py --placements` below.

For a complete placement uniquely referenced by the map root, `map_name` is the source map's 32-bit name
hash (eight lowercase hex digits) and `entity_index` is its zero-based ordinal in the authored `entities`
array. Other classes may occupy slots in that array. The JSONL keeps its physical record order.
For the ordinary terrain case (`parentInstance == -1`, a nonnegative `parent_index` and a valid signed
map selector), `parent_map_name` holds the selected map hash in the same format. Match
`(parent_map_name, parent_index)` to a unique `(map_name, entity_index)` across the input maps.
These fields are optional. Older exports, unavailable references and unsupported parent forms can lack
the corresponding keys. A missing or ambiguous target stays unresolved, including when a scan covers only
part of the map set.

`tools/world_census.py` scans the whole `levels.rpf` data area in shards and counts every validated
`0x7a` map resource and placement entity. The measured full pass is 23,535 maps / 1,670,442 entities
with 98.32% stream coverage; the exactness condition and byte accounting are in
[world_census.md](world_census.md).

`tools/world_names.py` builds the hash→name dictionary from local model manifests, level-scan paths, name
lists and XML, then resolves `world_index`/`world_census` map rows and aggregate archetype counts. It
reports 32-bit-only candidates separately from matches confirmed by the 64-bit key. The measured dictionary
and coverage are in [world_names.md](world_names.md).

```powershell
python -I tools\world_index.py <levels.rpf> --start 21474836480 --limit 1073741824 `
    --out work\world\levels-20-21.json --csv work\world\levels-20-21.csv
python -I tests\test_world_index.py
python -I tests\test_world_names.py
```

For a 76 GB archive always bound the run with `--start/--limit`; the tool records the window in its summary.
A 0x7a resource split beyond the first continuation, or a map whose entity block lives in a graphics page, is
reported with fewer entities or as `meta` rather than guessed.

## Building a placed world

`tools/world_build.py --placements` builds the viewer's chunks from map placements instead of each model's own
coordinates. It covers points 3 and 4 of *What a runtime needs*, and approximates point 5 with `--lod-sets`
(below).

1. `world_index.py <levels.rpf> --entities-out <work>\world\entities.jsonl` writes one row per complete entity.
2. rpf9dump exports the models (`models/manifest.json` + OBJ).
3. `world_build.py <assets> --out <world> --placements <entities.jsonl>`. Each selected entity is matched to an
   exported model by the lower-cased joaat of the manifest `Name` without its `.#dr`/`.#ft` suffix. When the
   entity row carries the +0x20 key, the model's Murmur3-64 must equal it, as in `world_names.py`; a 32-bit-only
   match is a `key_mismatch` and is not placed. The model is scaled by (`scaleXY`, `scaleXY`, `scaleZ`), rotated
   by the **conjugate** of the stored quaternion, translated to `position`, and converted from RAGE Z-up to
   glTF Y-up `(x, z, -y)`.

`--lod-levels` picks the `lodLevel` values to place (default `0,5`: HD and ORPHAN_HD) and `--region
MINX,MINY,MAXX,MAXY` bounds the RAGE world X/Y. `world.json` then has `"layout": "placements"` and a
`stats.placements` block: `rows`, `malformed`, `other_lod_level`, `outside_region`, `selected`, `placed`,
`placed_unverified` (no key in the row), `key_mismatch`, `unresolved_archetype` (no exported model has that name),
`model_not_loadable`, `degenerate_scale`, `mirrored` and `distinct_models`.

`--max-tex-size` limits the longest edge of embedded DDS previews to 512 pixels by default.
When shrinking a DDS tagged sRGB, the converter averages RGB in linear light, weights each texel by its
alpha, and encodes the result as sRGB again. An opaque black and white pair averages to 188 in sRGB
and 128 in an untagged RGBA8 texture.

### Shared geometry (`--instancing`)

A city has 1.55 M HD placements of about 120 k models, and most models are placed many times. By default
(`--instancing gpu`) each model's OBJ is parsed once into a packed cache file, each chunk stores every model it uses
once, and the placements in the chunk are instance transforms (`EXT_mesh_gpu_instancing`: one node per model with
`TRANSLATION`/`ROTATION`/`SCALE` accessors; listed in `extensionsRequired`). three.js's `GLTFLoader`, which the
viewer uses, turns each such node into `InstancedMesh`es. `--instancing nodes` writes one plain glTF node per
placement instead, for loaders without the extension (a draw call per placement). `--instancing bake` keeps the
first version's output, where every placement is a transformed copy of the geometry.

Each chunk is written as soon as it is built, so memory holds about 100 bytes per placement plus the models of
the chunks being written, not the whole map. `--jobs N` packs models and writes chunks in N processes.

Every instance transform has a positive determinant, so the winding is right in any loader:
- the TRS is the baked transform split into node form (checked against it on random rotations and scales);
- a negative `scaleXY` is a 180° turn about the model's Z axis and is folded into the rotation;
- a negative `scaleZ` mirrors. Those placements draw a mirrored copy of the mesh (glTF Y negated, winding
  reversed) with a positive scale, counted as `mirrored`;
- a zero scale cannot be inverted for normals; those rows are counted as `degenerate_scale` and not placed.

Chunk records add `distinct_models` (meshes stored) and `stored_triangles`; `triangles` stays the drawn count
(each mesh's triangles times its placements). `world.json` adds `instancing`, `stats.stored_triangles`,
`stats.chunk_bytes` and `stats.max_chunk_bytes`.

### LOD sets (`--lod-sets`)

`--lod-sets` places every `lodLevel` (unless `--lod-levels` narrows it) and builds one chunk set per tier, ids
`c_<tier>_<x>_<z>` with tiers `hd`, `lod`, `slod1`, `slod2`, `slod3`, `slod4` and `orphan_hd`. Each chunk record
names its `lod` tier and its grid `cell` (`[minx, minz, maxx, maxz]`, glTF metres). `world.json` gets
`lod_bands`: for each tier, the camera distance band in which it is drawn, from the selected entities:

- a tier starts at its own median `childLodDist` (the distance inside which a parent hands over to its children);
  HD starts at 0;
- it ends where the next coarser tier present starts; the coarsest tier ends at its median `lodDist`;
- ORPHAN_HD has no parent and runs from 0 to its median `lodDist`.

`stats.lod_tiers` records the medians and each tier's chunk and placement counts. The viewer loads a tiered chunk
while the horizontal distance from the camera to its cell is inside the band, and keeps it up to 32 m past the
band, so each cell shows one tier. Worlds without `lod_bands` stream as before.

Tier medians over all 1,670,435 entities (`lodDist` ≤ 0 means "use the archetype's default" and is left out):

| Tier | Entities | Median `childLodDist` | Median `lodDist` | Band |
|---|---:|---:|---:|---|
| HD | 1,153,122 | — | 124 (97,625 set) | 0–130 |
| LOD | 100,953 | 130 | 450 | 130–450 |
| SLOD1 | 16,828 | 450 | 1,000 | 450–1,000 |
| SLOD2 | 1,708 | 1,000 | 2,000 | 1,000–1,976 |
| SLOD3 | 601 | 1,976 | 4,000 | 1,976–6,043.5 |
| SLOD4 | 352 | 6,043.5 | 20,000 | 6,043.5–20,000 |
| ORPHAN_HD | 396,871 | — | 1,000 | 0–1,000 |

From HD to SLOD2, each tier's median `lodDist` is within 5% of the next tier's median `childLodDist` (124/130,
450/450, 1,000/1,000, 2,000/1,976), so the two fields agree on the bands. For SLOD3 they differ (4,000 and
6,043.5), and the band follows `childLodDist`.

The viewer approximates point 5 with distance bands per cell and tier. To follow an individual parent link,
a loader needs the selected map's name and the target placement's original ordinal in that map's `entities` array.

Two terrain regions on Xbox 1.0.2.1 give a cross-map example. In each tested HD and LOD map, `CMapData.parent` is zero.
The schema member `75d16adf` at +0x58 is an array with one 32-bit hash in a block of type `0x4a`. That hash
matches the next map's `CMapData.name`. The child's signed `parentIndex` selects an ordinal in **that map's**
`entities` reference array. Looking up the index in the HD map itself returns the child in all 127 cases.

For all 127 HD/LOD pairs, the same name stem with the respective tier suffix reproduced both stored archetype
keys (JOAAT and Murmur3-64). Following the reference from LOD grouped the 127 children under 32 SLOD1 entities.
At both steps, every parent's incoming count matched its `numChildren`, and its `childLodDist` matched each
child's `lodDist`.

Four terrain maps with two references show how an entity selects its parent map. Schema member `84799847` at
+0x7f is a signed byte (type `0x10`). Use its value as a zero-based index in the stored `75d16adf` array,
then apply `parentIndex` to that map's `entities` array. In these maps, 64 children select slot 0 and 18
select slot 1. For 42 children the same parent ordinal, tier and distance fields fit both maps. In every
one of these cases the selected parent is closer to the child's XY position. Reversing
the array while keeping the selector fixed sends those 42 to the other map and the remaining 40 to an
out-of-range ordinal. Remapping the selector along with the array restores all 82 links.

Including the adjacent single-reference maps, all 324 links agree with the declared child counts of all 34
parents. Always selecting slot 0 gives incorrect child counts for ten parents. The six terminal entities
have an empty map reference array, selector -1 and `parentIndex` -1.
All 330 entities have `parentInstance` -1. These checks establish authored associations for the measured
terrain records. Nondefault batch instances and the runtime switching distance remain open.

### Measured

On the 1.0.2.1 package, `levels.rpf`, Linux, 88 cores, `--no-textures`, 256 m chunks. Models were exported with
rpf9-explorer at the baseline revision (`917e9b9`) and with breadbbbb/rpf9-explorer#8 and #10 applied. Each placement was
counted against the first export and built from the second.

| Step | Result |
|---|---|
| `--entities-out` over the whole data area (24 parallel `--start/--limit` windows) | 1,670,435 rows, 66 s |
| `lodLevel` split | 0: 1,153,122 · 5: 396,871 · 1: 100,953 · 2: 16,828 · 3: 1,708 · 4: 601 · 6: 352 |
| HD + ORPHAN_HD placements with a recovered model (joaat + Murmur3-64) | 640,080 of 1,549,993 (41.3%) at `917e9b9`; 1,549,347 (99.96%) with #8/#10, from 121,411 resources |
| 700 m × 700 m downtown box, 10,729 placements, `--jobs 16` | `bake` (#1183): 5 min, 22 GB peak RSS. `gpu`: 35 s; 0.43 GB in the largest process, 1.93 GB for all 17 processes; same 16 chunks and 29,331,138 triangles |
| **whole city**, HD + ORPHAN_HD, `gpu`, `--jobs 40` | 1,549,343 placed (646 `unresolved_archetype`, 4 `model_not_loadable`, 0 `key_mismatch`, 0 `mirrored`), 121,407 distinct models → 2,940 chunks; 7.31 G triangles drawn, 3.34 G stored; 125 GB of GLB, median chunk 17 MB, largest 926 MB; 21 min; 3.4 GB in the largest process, 38 GB for all 41 processes |
| **whole city, all tiers, `--lod-sets`, `gpu`, `--jobs 24`** | 1,669,771 placed (660 `unresolved_archetype`, 4 `model_not_loadable`), 228,566 distinct models → 12,340 chunks: HD 2,911 (median 16.9 MB), LOD 2,836 (0.6 MB), SLOD1 2,811, SLOD2 1,222, SLOD3 501, SLOD4 284 (all ≤ 0.1 MB median), ORPHAN_HD 1,775 (3.0 MB); bands as in the table above; 33 min; 2.6 GB in the largest process, 30 GB for all 25 |
| what the viewer loads at the densest HD cell (camera at street level in its centre) | with `lod_bands`: 577 chunks, 180 M triangles, 3.1 GB of GLB (5 HD, 69 ORPHAN_HD, the rest LOD/SLOD); the same city without LOD sets at the viewer's 1,400 m radius: 119 chunks, 1,060 M triangles, 16.3 GB |

Checks:
- Every chunk of the whole-city build passes a structural check: accessors inside their bufferViews, indices
  below the vertex count, POSITION `min`/`max` equal to the data, the instancing attributes' counts equal, unit
  rotations, and positive finite scales. Recomputed from the GLBs, the chunk bounds, placement counts (1,549,343)
  and drawn triangles (7,311,221,858) equal `world.json`.
- three r160's `GLTFLoader` (the viewer's loader), run under Node, parses 65 of the city chunks: 60 random and the
  5 largest, including the 926 MB one. It builds `InstancedMesh`es whose drawn triangles equal `world.json`, with
  bounds inside the chunk's bounds. It also parses all 16 chunks of the box and 35 chunks of the LOD-set build
  (4 random and the largest per tier).
- The LOD-set build's 12,340 chunks pass the same structural check; 1,669,771 placements and 7,527,211,045 drawn
  triangles equal `world.json`.
- On the downtown box, the conjugate of every stored rotation equals SixNG's independently derived world rotation:
  |dot| ≥ 0.9999 on 29,124 of 29,125 entities matched by position and archetype, and the minimum is 0.99894.
  Scales are equal on all 29,125. The instance TRS equals that transform (see the tests).

The pin's 41.3% is a model-recovery gap, not a naming gap: almost every archetype name resolves (99.85%, see
*What is readable* above), but only recovered models can be placed. The archetypes that #8/#10 add are `.#ft`
fragments. The pinned `LooksLikeFragment` now accepts the null optional pointers covered by
`breadbbbb/rpf9-explorer#8`, so that header filter no longer explains the remaining model-recovery gap.
Although 99.85% of archetype names resolve (see *What is readable* above), only 41.3% of HD and ORPHAN_HD
placements had a recovered model at `917e9b9`. The additional archetypes recovered with #8/#10 were `.#ft`
fragments. The pinned explorer (`e2939175`) includes both changes. Its fragment check accepts null pointers
at header offsets 16, 32 and 40 and requires system pointers at offsets 8, 24 and 168.
[Fragment check at the pinned revision](https://github.com/breadbbbb/rpf9-explorer/blob/e2939175c36a02d6ed25fa2c217b9d6ddcb8b516/Core/RageModelDecoder.cs#L66-L84)
The 41.3% at the earlier pin (917e9b9) was a model-recovery gap, not a naming gap: almost every archetype name
resolves (99.85%, see *What is readable* above), but only recovered models can be placed. The archetypes that
#8/#10 add are `.#ft` fragments, whose headers `LooksLikeFragment` rejected before breadbbbb/rpf9-explorer#8 (it
now accepts null optional pointers at `+16`/`+32`/`+40` and range-checks the 16-bit fields at `+48..+54`).
Leonida's pin is now e2939175, which contains #8 and #10, so the pinned `LooksLikeFragment` accepts these headers;
the 99.96% was measured on 917e9b9 with both applied and has not been recounted at e2939175.

**Not verified or limited:**
- Nothing here was looked at in a browser. The checks above are structural and numeric.
- Memory grows with `--jobs` (each worker holds the models of one chunk), not with the map: 38 GB at 40 jobs, 30
  GB at 24 jobs for a bigger build, 1.9 GB at 16 jobs for the box.
- With LOD sets, most of what is still loaded near the centre is ORPHAN_HD: it has no parent tier, so its band is
  0 to 1,000 m.
- Chunk sizes follow model detail. A few dense downtown cells exceed 100 MB, and the largest stores 24 M
  triangles of 2,204 distinct models, so a smaller `--chunk-size` there helps a browser.
- No placement in this package has a negative `scaleZ`, so the mirrored path is checked by the synthetic tests
  only.
- Textures were not part of these runs (`--no-textures`).

```powershell
python -I tools\world_index.py <levels.rpf> --out work\world\index.json --entities-out work\world\entities.jsonl
python -I tools\world_build.py work\assets --out work\world\placed --no-textures --jobs 16 `
    --placements work\world\entities.jsonl
python -I tools\world_build.py work\assets --out work\world\lod --no-textures --jobs 16 `
    --placements work\world\entities.jsonl --lod-sets
python -I tests\test_world_place.py
python -I tests\test_world_instancing.py
node --test tests\test_viewer_streaming.cjs
```

The entity rows and the placed chunks come from your own copy of the game; keep them in the local work folder.
## Trains (`CTrainConfigs`)

Trains are configured by one `CTrainConfigs` XML document in `common.rpf` (see [gamedata.md](gamedata.md)).
Counts only; no names.

| Path | Count | Content |
|---|---:|---|
| `trainConfigs/Item` | 46 | `name` (46 distinct), station stop durations, announce/beep/hang/swing flags, `carriageGap`, `defaultMaxSpeed`, `maxAcceleration`, `maxDeceleration`, `populateTrainDist`, `deathToReviveTime`, `carriages` |
| `.../carriages/Item` | 314 | `modelName` (304), `modelSetName` (300), `maxPedsPerCarriage`, `flipModelDir`, `doInteriorLights`; 1 to 17 per train (median 8) |
| `trainGroups/Item` | 5 | `name` and `trainConfigRefs` (24, 4, 5, 1 and 2 refs) |
| `.../trainConfigRefs/Item` | 36 | `trainConfigsName` and `trainConfigIndex` |

**Group references: resolve by name, not by index.** All 36 `trainConfigsName` values name a config and all 36
`trainConfigIndex` values are in range, but only 25 indices equal the position of the named config. The other 11
are exactly one less. The split is clean: every reference to a config at (0-based) position 23 or earlier agrees, and every
reference to a config at position 26 or later is off by one. That is what one config inserted at position 24 or
25 after the indices were written would produce, so `trainConfigIndex` looks stale: a loader that trusts it
would pick a different config for 11 of the 36 references. 12 of the 46 configs are in no group.

Carriage models: the 304 `modelName` values use 13 distinct names; 9 are `.yft` base names in the local archive
catalog and all 9 are among the vehicle distant-light models, while 4 match no archive entry. The 23 distinct
`modelSetName` values resolve to an archive entry name in 9 cases. Every carriage has a model name, a model set,
or both (290 have both).

Ranges: `defaultMaxSpeed` 8 to 15 (median 8), `maxAcceleration` 1 and `maxDeceleration` 2 in all 46,
`populateTrainDist` 40 in all 46, `carriageGap` -0.5 to 0.1.

## Instanced props and grass

Map resources can carry two instance lists besides `CEntityDef`. All four struct names are the case-sensitive
joaat of the GTA V class names, and every member below is named by the resource's own member table.
`tools/instance_lists.py` reads them through that table (by name hash) and writes counts only.

| Hash | Struct | Size | Members (offset) |
|---|---|---:|---|
| `0xBA049770` | `rage__fwPropInstanceListDef` | 0x170 | `CEntityDef` members, `InstanceList` 0x90, `BatchAABB` 0xA0, `ScaleRange` 0x160, `InstanceLodDist` 0x168 |
| `0xEA6B3A96` | `rage__fwPropInstanceListDef__InstanceData` | 0x40 | `InstMat` 0x0 (3x4 floats, translation in the last column), `Tint` 0x34 |
| `0x7C475B5D` | `rage__fwGrassInstanceListDef` | 0x1A0 | `CEntityDef` members, `BatchAABB` 0x80, `ScaleRange` 0xA0, `LodFadeStartDist` 0xB8, `InstanceLodDist` 0xBC, `OrientToTerrain` 0xC0, `slodID` 0xB0, `IsSLODList` 0x150, `SLODArchetypeList` 0x158 (array), `InstanceList` 0x178 |
| `0xED86F522` | `rage__fwGrassInstanceListDef__InstanceData` | 0x10 | `Position` 0x0 (3 x u16 in `BatchAABB`), `NormalX` 0x6, `NormalY` 0x7, `Color` 0x8 (3 bytes), `Scale` 0xB, `ScaleZ` 0xC, `RotationIdx` 0xD |

Two more grass-list members are named by joaat from a brute force over common engine words: `slodID` (0xB0, member hash `9fe78fd1`) and `SLODArchetypeList` (0x158, `b451fc98`), so grass lists carry their own link to an SLOD archetype set next to `IsSLODList`. The other unnamed members stay as hashes in the schema.

An `InstanceList` is a u64 (low 12 bits: 1-based block index; bits 20+: offset in that block) and a u16 count.
Instance blocks are usually in the physical buffer, so the tool joins each resource's virtual and physical
streams from its block map (+0x08 -> +8 virtual count, +9 physical count).

Full run over `levels.rpf`, `story000.rpf` and `shared_minigame000.rpf` (stream occurrences, duplicates
included):

| Archive | Map resources | With lists | Prop lists | Prop instances | Grass lists | Grass instances |
|---|---:|---:|---:|---:|---:|---:|
| `levels.rpf` | 188,447 | 135,560 | 690,779 | 24,778,818 | 202,143 | 219,453,097 |
| `story000.rpf` | 48,084 | 2,683 | 4,833 | 1,652,980 | 5,504 | 5,641,657 |
| `shared_minigame000.rpf` | 787 | 472 | 3,317 | 51,795 | 592 | 437,849 |
| total | 237,318 | 138,715 | 698,929 | 26,483,593 | 208,239 | 225,532,603 |

Struct sizes and every member offset in the table match the schema in all 98,722 resources with prop lists
and all 57,650 with grass lists. 29,482 prop lists and 8,015 grass lists point to a block outside the
streams the tool joined; their instances are not counted.

Props (26,483,593 instances): every matrix is right-handed; 26,046,778 have orthonormal axes (within 0.01),
25,936,781 are uniformly scaled and 4,692,108 have unit scale; the translation is inside the list's
`BatchAABB` for 26,386,624. List `lodLevel`: LOD 424,012, HD 116,660, ORPHAN_HD 93,958, SLOD1 64,299. Every
list has an ordered `ScaleRange` and `0 < InstanceLodDist <= lodDist`.

Grass (225,532,603 instances): 225,379,032 `NormalX`/`NormalY` pairs (byte / 127.5 - 1) are inside the unit
disc; `RotationIdx` is 0 to 72; `ScaleZ` equals `Scale` in 52,021,564 and `Scale` is 0 in 47,679,734, so
`ScaleZ` is its own value, not ambient occlusion. `OrientToTerrain` is 1.0 in all 208,239 lists, `IsSLODList`
is set in 5,802, and `lodLevel` is ORPHAN_HD in 205,181 and SLOD1 in 3,058.

## Procedural objects and plants (`CProceduralInfo`)

In GTA V, `procedural.meta` scatters small objects and plants over ground by collision material: a tag table maps
material procedural tags to entries in an object list and a plant list. This build's equivalent, one
`CProceduralInfo` XML document in `common.rpf`, has the same three parts but only the table is filled. Counts
only; no names.

| Path | Count | Content |
|---|---:|---|
| `procObjInfos` | 1 | empty |
| `plantInfos` | 1 | empty |
| `procTagTable/Item` | 255 | `name`, optional `procObjTag` (66, all distinct) and `plantTag` (69, 57 distinct); 30 items have both |

- **The two lists the tags point into are empty**, so no procObj or plant tag resolves inside this document. The
  procedurally scattered objects and plants are therefore either stored elsewhere in this build or replaced by
  the authored grass and prop instance lists above (225.5 million grass and 26.5 million prop instances).
- The 255 table slots carry 107 distinct names. One name fills 149 slots, spread through the table (positions
  67 to 254, not contiguous), and none of those 149 has a tag: they read as unused placeholder slots, which would
  keep the slot index (the value a material would store) stable. The other 106 names occur once each, and
  only one of them has neither tag.
- Five retained fragments of `MaterialDataList` contain 399 complete material records with 1,433 nested tag
  entries. One of the 107 procedural table names occurs as a substring in the retained text, but none matches
  a field value in those records exactly. The fragments cover only part of the list (see the snow and mud
  section below). A numeric value in the slot range (0–254) alone does not identify a slot reference.

Not measured here (hypothesis): whether materials index this table by slot, as in GTA V, and where the
object and plant definitions live.

## Deformable snow and mud (`CSnowTrackData`)

The tracks left in snow, mud and similar ground are configured by one
`CSnowTrackData` XML document in `common.rpf` (see [gamedata.md](gamedata.md) for how it is carved). Counts
only; no values or names.

| Path | Count | Content |
|---|---:|---|
| `CommonDeepSurfaceParams/Item` | 16 | `Name`, `DebugColor`, `PhysicsMaterials`, and three parameter blocks: `HeightMapGenerationParams` (14 fields: depth, berm, noise, viscosity, lifetime, fade, blur, `POMScale`, plus `SupportPropFlatten` on 2), `TerrainRenderParams` (height, bump-map and displacement blend, normal scale), `PathRegenAndFlattenParams` (berm height, trail width, vegetation and explosion flattening) |
| `.../PhysicsMaterials/Item` | 24 | one material name each; 0 to 6 per surface (0 on 6 surfaces, 1 on 5) |
| `MudParams` | 1 | 23 fields: soak and puddle-fill ramps, puddle scale/ripple/slope/mask/specular, mip bias, POM and height-map puddle scale, two tint colours |
| `TrackTypeData/Item` | 9 | `Name`, `Type` (3 distinct, used 5, 2 and 2 times), 17 numeric fields (sphere and berm scale, depth offset and bias, trail width/V-scale/depth/velocity ranges, grass flattening on land and in water), and three texture names: trail, print, mud displacement |
| `SnowNormalMapName` | 1 | one texture name |
| `GlobalDeepSurfaceParams` | 1 | tessellation-mask mip bias, deformation and parallax world-space widths, path-segment cull distance and time |
| `TrailmapParams` | 1 | width and side offsets, and water-push sphere and trail strengths |

So the system is a world-space height map stamped by moving spheres and trails, rendered with tessellation,
parallax occlusion mapping and a normal map, with a separate mud/puddle layer; that is the reading of the field
names, and the values were not tested in a renderer.

Cross-checks:

- **Physics materials.** The 24 names are distinct. `MaterialDataList`, the material table, survives only from
  the middle of the document (its first block is in the unread region of `common.rpf`); 13 of the 24 names occur
  in the retained part. The other 11 may be in the lost part, so no name is shown to be missing.
- **Textures.** The 9 distinct texture names (8 track textures and the snow normal map) match no archive entry
  name and no texture-dictionary item name or its joaat hash in the local catalog (118,925 named items). The
  catalog is incomplete for names, so this does not show the textures are absent.

### The 16 surface entries and the deformation map (ASSET-305)

Measured on a 1.0.2.1 package (one `CSnowTrackData` document, one complete stream, found with
`tools/gamedata_xml.py` on a `rpf9_carve.py` carve of `common.rpf`). Classes and counts only; the entry and
material strings are not reproduced.

- **11 surfaces are defined and 5 slots are empty.** Entries 0 to 10 each have a `Name`, a distinct opaque
  `DebugColor` and their own parameter set (11 distinct parameter sets). Entries 11 to 15 have no `Name`
  text, a `DebugColor` of `0x00000000`, no physics material, and one shared parameter set that equals none of
  the 11 (their terrain render scales are 0). The array is 16 long whatever the content, which fits a
  fixed-size table indexed by a 4-bit surface id; the index is not read in a shader here.
- **The 11 surfaces, by class** (from the `Name` and the physics-material names of each entry):

  | Entry | Class | Physics materials |
  |---:|---|---:|
  | 0 | snow | 1 |
  | 1 | quicksand | 1 |
  | 2 | mud (grass, wet, sticky, soft periphyton) | 4 |
  | 3 | scree / gravel | 2 |
  | 4 | sand and loose dirt | 4 |
  | 5 | compact dirt, clay, dry mud | 3 |
  | 6 | algae and soft debris (seaweed, rubbish, bagasse, tar pit) | 6 |
  | 7 | plastic mulch | 1 |
  | 8 | parallax-only (named after POM), no physics material | 0 |
  | 9 | gravel trap | 1 |
  | 10 | wet sand | 1 |

  So the 24 physics materials go to 10 of the 11 surfaces; the 6 surfaces without a material are entry 8 and the
  5 empty slots. `SupportPropFlatten` is set on the 2 sand entries (4 and 10).
- **Cross-check with the material table.** The retained part of `MaterialDataList` is one stream without its
  head, kept as five contiguous fragments (four 64 KiB blocks and a short tail, 403 `name` elements in all).
  All 14 `isDeformableTerrain` items of the stream are in the first fragment, and all are `"true"`. 13 of them
  belong to exactly the 13 of the 24 surface materials whose names occur in the stream (all 13 names are in the
  first fragment); the 14th belongs to a record whose name is before the start of the stream. No other record
  carries the flag, and no surface material lacks it. The other 11 surface materials do not occur in the
  retained stream.
- **Deformation map size.** `GlobalDeepSurfaceParams` gives `DeformationWorldSpaceWidth` = **96 m** and
  `ParallaxWorldSpaceWidth` = **48 m**: one world-space width each. The document gives no second dimension, so a
  square window is an assumption, not a reading. No field in the document gives a texel count, and no other carved XML in
  `common.rpf` names `DeformationWorldSpaceWidth`; the map resolution is therefore not in the data files read
  here (it would be in the executable or the render-target setup, neither of which is in the package), so the
  size in texels, and where the renderer sets it, stay open.
- **Track types.** The 9 `TrackTypeData` entries are 1 human, 6 animals and 2 vehicles (a car class and a
  cycle). `Type` is `kSnowTracksBiped` on 5 (the human and 4 animals, including the horse),
  `kSnowTracksQuadraped` on 2 animals and `kSnowTracksVehicle` on the 2 vehicles. Which entry a given ped model
  uses is not in this document.

Not measured here (hypothesis): the texel resolution and the second dimension of the deformation map, where it is centred (camera or
player), and how an entry is chosen at run time (the material's flag, the surface index, or both).

## Map lights and the vehicle lane graph

Light emitters on archetypes and entities, and the vehicle path nodes, links and junctions, are in
[world_lights_paths.md](world_lights_paths.md) (`tools/world_lights_paths.py`).

## No-sharing rule

Everything here is aggregate. Do not commit archives, carved streams, extracted models/textures, entity
dumps, or logs that quote them. `work/` and `reports/` are git-ignored; the tool output contains only resource
headers, counts and aggregate bounds, but it is still generated from the user's copy and stays local.
