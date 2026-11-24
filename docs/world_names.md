# Placement and archetype names

What a locally recovered name can resolve in the `0x7a` map resources of `levels.rpf`, and what it cannot.
Measured on the Xbox Series 1.0.2.1 package. This page contains counts, field hashes and formulas only; no
game names, entity rows or carved bytes.

## Result

Two keys are stored for most names, and they are independent enough to reject a same-hash collision:

| Field | Member hash | Offset | 32-bit key | 64-bit confirmation |
|---|---|---:|---|---|
| `archetypeName` | `a023a02c` | +0x44 | joaat, lower-cased | entity `c8a7b90a` at +0x20 |
| map `name` | `e6e17f14` | +0x08 | joaat, lower-cased | map `8bfb6095` at +0x10 |
| map `parent` | `a99c4651` | +0x68 | joaat, lower-cased | none in the root |
| entity id | `1b60404d` | +0x08 | none | per-entity, not a name key |

The 64-bit keys are not a second joaat. Both are the **low 64 bits of MurmurHash3 x64 128 with seed 0** over
the exact UTF-8 name bytes. In the two sampled windows every archetype name that matched the 32-bit hash also
matched this 64-bit key, and the one 32-bit-only candidate was rejected by it. The same formula reproduces the
map root's +0x10 key for all five map names that the local dictionary resolves.

The 32-bit field hashes themselves are the case-sensitive joaat of the engine member names (`joaat("CMapData")
= 0xd3593fa6`); the *values* are the lower-cased joaat of the model or map name.

## Dictionary

The measured dictionary was built only from names already recovered locally:

| Source | Input rows |
|---|---:|
| rpf9dump model manifests (`data.rpf`, `packs`, `core_restofworld`, `story000`) | 7,370 |
| rpf9dump levels scan paths (`0-8`, `4-4.5`, `12-16` GiB) | 43,187 |
| recovered name lists (levels and packs streams) | 16,844 |
| `common.rpf` XML `<filename>` strings | 6,448 |
| `common.rpf` XML `<name>`/`<Name>` strings | 15,378 |
| **merged exact names** | **79,738** |
| exact hash values / 32-bit collision groups (names) | 79,709 / 28 (57) |
| derived `_lod`/`_slod` spellings | 157,724 |
| derived hash values / collision groups (names) | 157,664 / 58 (118) |

Derived spellings are **not recovered strings**. They are LOD variants of recovered names whose 32-bit and
64-bit hashes both matched in the sample; the tool keeps them in a separate table and reports them separately.
The full 149,811-model `levels.rpf` export was not in the local artifact set used for this measurement, so the
dictionary is smaller than the project's model total.

## Archetype coverage

Two bounded 1 GiB windows were measured with entity aggregation only (counts per hash, no entity rows):

| Window | Maps | Entities | Distinct archetype hashes | Exact names resolved | Share | 32-bit-only rejected | With derived LOD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8–9 GiB | 1,398 | 88,228 | 12,400 | 1,942 hashes / 3,246 entities | 3.68% | 1 | 2,961 / 4,364 (4.95%) |
| 13–14 GiB | 1,318 | 115,571 | 11,705 | 3,165 hashes / 5,017 entities | 4.34% | 0 | 4,788 / 6,640 (5.75%) |
| **total** | **2,716** | **203,799** | **24,105** | **5,107 / 8,263** | **4.05%** | **1** | **7,749 / 11,004 (5.40%)** |

Every exact resolution above is confirmed by the +0x20 key: 5,107/5,107 hashes, 0 ambiguous. The one rejected
candidate had a valid lower-cased joaat but the wrong 64-bit key, which is exactly the same-hash case the
confirmation exists for. Derived `_lod`/`_slod` names are confirmed the same way; they add 2,741 entities.

This is a **dictionary-coverage lower bound**, not a claim that the other archetypes have no names (see
[Whole-archive coverage](#whole-archive-coverage)). The
unresolved majority is dominated by models whose names are not in the local exports, and the same 32-bit
formula is proven by every resolved entry.

## Whole-archive coverage

The windows above use only names that were already exported locally. The names of almost all the other
archetypes are in `levels.rpf` itself: `tools/rpf9_names.py` lists the `name.#dr` and `name.#ft` strings of every
stream. A full pass over the data area (76.15 GB, 5,255 s) inflates 1,116,073 zlib streams (98.32% of
compressed bytes) and finds 235,814 named streams / 318,266 unique names (267,463 `.#dr`, 50,803 `.#ft`);
with that list as the dictionary nearly every placement resolves. Whole `levels.rpf`, one
`world_index.py --archetypes-out` run (23,532 placement maps and 1,670,435 entities: 3 maps and 7 entities fewer
than the census in [world.md](world.md#how-many-are-there), which also joins the raw continuation page that 3
placement maps use and `world_index.py` does not; 228,681 distinct `archetypeName` hashes, one 64-bit key per hash, no conflicting keys,
no incomplete entity records):

| Dictionary (`world_names.py build --rpf9-names`) | Archetype hashes resolved | Entities resolved |
|---|---:|---:|
| `rpf9_names.py --types dr,ft` (default `--max-inflate`, 1 MiB) | 227,422 (99.45%) | 1,667,978 (99.85%) |
| `rpf9_names.py --types dr,ft --max-inflate 268435456` | 228,430 (99.89%) | 1,669,373 (99.94%) |
| `rpf9_names.py --types dr` only (default) | 217,491 (95.11%) | 508,600 (30.45%) |

Every resolved hash is **verified** by the entity's 64-bit key (Murmur3-64, see Result): 0 unverified, 0
unconfirmed, 0 ambiguous. Two observations explain the earlier lower bound:

- **Fragments carry most placed objects.** Fragment (`.#ft`) names resolve only ~10,000 more hashes than
  drawable names alone, but those archetypes account for about 70% of the entities.
- **Map LOD models are drawable-dictionary members.** A drawable dictionary is one resource with several member
  drawables, each with its own `name.#dr` string; a model export that takes one name per resource misses the
  others, but `rpf9_names.py` lists every name string. In a local structural check (member = named drawable
  root with a valid LOD table), dictionary member names account for 99,987 of the 228,681 hashes (43.7%) but
  only 110,650 entities (6.6%): many distinct LOD models, each placed a few times. Some member names lie deep
  in large resources: raising `--max-inflate` from 1 MiB to 256 MiB adds 1,008 hashes, 992 of them members.

The 251 hashes (1,062 entities) left have no match in the 256 MiB name list.

## Map-name coverage

The full `levels.rpf` census has 23,535 placement maps:

| Measure | Value |
|---|---:|
| Map rows / distinct `name` hashes | 23,535 / 23,535 |
| Names resolved by the dictionary | 5 (0.021%) |
| Of those, confirmed by the map +0x10 key | 5 / 5 |
| 32-bit-only candidates rejected | 0 |
| Rows with a non-zero `parent` | 2,089 |
| Distinct `parent` hashes | 2,053 |
| Parent hashes resolved | 1 (0.048%) |
| `parent` values that equal some map's `name` | 1,304 |
| Parent links whose parent name resolves | 1 |

The five map-name matches are not collisions: their map root's +0x10 value equals the Murmur3-64 of the same
name, and one of the maps also contains an entity whose archetype hash is the same name. The single resolved
parent equals a map name that resolves, so it is a real parent link. The remaining 23,530 names are unresolved
because those map-cell names are not in the local model/XML exports, not because a different hash formula was
observed. A runtime still needs a map-cell name source; this measurement only proves the formula.

## 64-bit fields

| Field | Window | Unique values / entities | Reading |
|---|---|---:|---|
| entity `1b60404d` (+0x08) | 13–14 GiB | 115,563 / 115,571 | per-entity id; 8 duplicate pairs |
| entity `1b60404d` (+0x08) | 8–9 GiB | 86,932 / 88,228 | per-entity id; duplicates follow duplicate payloads |
| entity `c8a7b90a` (+0x20) | 13–14 GiB | 11,705 / 11,705 | one key per archetype |
| entity `c8a7b90a` (+0x20) | 8–9 GiB | 12,400 / 12,400 | one key per archetype |

No name-only formula was found for +0x08: it changes per placement, so it cannot be a per-archetype key. The
earlier reading of +0x08 as a per-archetype hash and +0x20 as the GUID is reversed for this schema. The
`guid` member `9a7b6e6d` was not present in the entity schemas inspected in the 8–9, 13–14 and 20–21 GiB
windows.

## Tool

`tools/world_names.py` builds the dictionary and resolves map rows and aggregate archetype counts:

```powershell
python -I tools\world_names.py build --manifest <work>\assets --scan <work>\levels-scan.json `
    --names <work>\names.txt --xml <work>\common-concat.txt --out <work>\world\names.json

python -I tools\world_names.py resolve --dict <work>\world\names.json `
    --maps <work>\world\census\maps.json --archetypes <work>\world\archetypes.json `
    --out <work>\world\names-report.json

python -I tests\test_world_names.py
```

`build` accepts model manifests (a file or a tree containing `models/manifest.json`), rpf9dump scan JSON,
one-name-per-line text, `rpf9_names.py` index JSON (`--rpf9-names`) and `common.rpf` XML. `--derive-lod` adds the
separate `_lod`/`_slod` table.

The whole-archive numbers above come from these four commands (no other inputs):

```powershell
python -I tools\rpf9_names.py <game>\levels.rpf <work>\world\rpf9_names.json --types dr,ft --max-inflate 268435456
python -I tools\world_index.py <game>\levels.rpf --out <work>\world\world_index.json `
    --archetypes-out <work>\world\archetypes.json
python -I tools\world_names.py build --rpf9-names <work>\world\rpf9_names.json --out <work>\world\names.json
python -I tools\world_names.py resolve --dict <work>\world\names.json `
    --archetypes <work>\world\archetypes.json --out <work>\world\names-report.json
```

`resolve` adds the counts of several `--archetypes` files, so make one whole-archive file rather than one per
window; `--rpf9-names` may be given once per window.
When several `--maps` files cover different windows of the same world, a child in one can link
to a parent map in another. Each child row counts once regardless of file order.

`resolve` accepts `world_index`/`world_census` map JSON (or a row list) and an aggregate archetype file:

```json
{"archetypes": {"a023a02c": 12}, "keys": {"a023a02c": "0011223344556677"}}
```

`keys` is optional. With it, a match is reported as verified only when the 32-bit candidate's Murmur3-64 equals
the stored key; candidates that match only the 32-bit hash are reported as `unconfirmed`, not guessed. Map rows
may carry `map_name64` to get the same confirmation. `tools/world_index.py` now emits that field, and a fresh
`world_census` run carries it through; the 23,535-row census JSON used above was produced before the field was
added, so the plain census run reported the five map-name matches as resolved but unverified. A five-row recheck
supplied the map root +0x10 keys and verified 5/5.

The dictionary and detailed reports contain game-derived names. Keep them in the local work folder; only
aggregate counts belong in git, issues or pull requests.
