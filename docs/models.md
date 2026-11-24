# Model recovery

Measured state of `tools/rpf9dump models` on the Xbox Series 1.0.2.1 package, and the
upstream `rpf9-explorer` fixes that close it. The fixes are merged upstream and the submodule is
pinned at `e293917`, where a full ten-archive re-count recovers **382,781 models** (**0 decode
failures**), up from 350,251 at `917e9b9` and 190,792 at `c3708fb`.

Before the fix, the exporter did not reject any vertex or index layout in the tested archives. It
dropped valid models in three validation paths, all in `third_party/rpf9-explorer` at commit
`48557ea`.

## Upstream status

[Leonida issue #29](https://github.com/ItsNotRexed/Leonida/issues/29) tracks this work. The decoder
fix in [rpf9-explorer PR #1](https://github.com/breadbbbb/rpf9-explorer/pull/1) is merged, and
`third_party/rpf9-explorer` is now pinned at upstream `e293917` (was `917e9b9`, originally
`48557ea`). Commits up to `917e9b9` added the `packs.rpf` drawable-dictionary roots (one drawable
per entry), the TTPF compact/array/cubemap/volume layouts, and the external drawable-skeleton
reader. Commits `917e9b9..e293917` add upstream `v1.2` model-discovery changes (PRs #8–#12:
optional fragment pointers, skipping texture dictionaries in drawable-dictionary scans, 16-bit
bone-palette indices, probe-count parity, and zero-radius drawables) plus BCPack BC1/BC3/BC4/BC5
decoding (PR #7). A full ten-archive re-count at `e293917` recovers **382,781 models** (**+32,530**
over `917e9b9`'s 350,251) with **0 decode failures**, **55,417 fragments** and **2,636,248,990
triangles** (the `levels.rpf` cells from a full export, see the table note), and the earlier
`917e9b9`, `c3708fb`, and `48557ea` measurements are kept below as controls. This page records the
measured state; it is not a competing patch.

PR #1 reports 190,792 of 190,792 recognized candidates across the ten archives configured in
Leonida, including 169 previously rejected candidates (also verified after archive readback). A
later full-archive run at the same head independently reproduced the ten-archive total: 190,792
candidates, 190,792 recovered, read back and decoded, 0 decode or readback failures, no duplicate
model names, and 2.22 billion triangles (the pre-fix export was 155,598 models / 1.95 billion
triangles). Issue #29's run uses its own recognized-candidate set and covers primary-LOD geometry;
the pre-fix counts below use `rpf9dump`'s page-scan candidates and export every recovered
drawable/LOD variant. The two sets are different scopes, not contradictory.

### `packs.rpf` after the pin bump

`packs.rpf`'s `.#dr` pages are drawable-dictionary roots, not drawables, so the old candidate
scan never saw them. The dictionary walk recovers **23,720 dictionary children** from that
archive at both `917e9b9` and `e293917`; the fragment count grew from **314** to **2,865** when
upstream `v1.2` began recovering zero-radius fragments. The archive total is therefore
**24,034 models at `917e9b9`** (23,720 children + 314 fragments) and **26,585 at `e293917`**
(23,720 + 2,865), against 314 at `c3708fb`. A full `rpf9dump models` export at `917e9b9`
reported **`ok=24,034 fail=0 skip=0` in 263 s** (0 embedded textures); the OBJs and manifest are
local. The 314 `48557ea`/`c3708fb` entries in the table below are the pre-pin control, not the
current `packs.rpf` result.

### Full-archive re-count at `e293917`

A fresh scan, recovery, and full `RageModelDecoder.Decode` readback across all ten archives in
`MODEL_ARCHIVES` (no index cache; `RpfPreview.Core` at `e293917`, .NET 8/10 Release) measures the
effect of upstream `v1.2` PRs #8–#12:

| Archive | Candidates | Models at `e293917` | Dict children | Standalone | Fragments | Bone-bound | Triangles | Rejects | Delta vs `917e9b9` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `data.rpf` | 2,921 | **3,175** | 268 | 2,907 | 1,758 | 2,891 | 3,387,016 | 0 | +594 |
| `pack/shared_minigame000/...` | 1,058 | **1,294** | 317 | 977 | 10 | 18 | 16,843,903 | 0 | +6 |
| `pack/core_restofworld/...` | 924 | **936** | 808 | 128 | 53 | 849 | 1,239,596 | 0 | +51 |
| `pack/story000_restofworld/...` | 1 | **3** | 3 | 0 | 0 | 0 | 2,620 | 0 | 0 |
| `pack/story000/story000.rpf` | 4,195 | **7,461** | 4,156 | 3,305 | 0 | 0 | 39,861,747 | 0 | +1 |
| `packs.rpf` | 26,389 | **26,585** | 23,720 | 2,865 | 2,865 | 26,585 | 64,966,572 | 0 | +2,551 |
| `levels.rpf` | 235,330 | **343,327** | 130,239 | 213,088 | 50,731 | 59,544 | 2,509,947,536 | 71 | +29,327 |
| `textures.rpf` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `common.rpf`, `core_restofworld001` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **270,819** | **382,781** | **159,511** | **223,270** | **55,417** | **89,887** | **2,636,248,990** | **72** | **+32,530** |

`levels.rpf` was first counted with `count.exe recover --verify`, which reports candidates, models,
dictionary children, standalone models, rejects and decode failures only, so its fragment,
bone-binding and triangle cells were open. A full `rpf9dump models <levels.rpf> <out>` export at
`e293917` (25 min, about 340 GB of OBJ, deleted afterwards) fills them from its manifest: 235,330
candidates, 235,259 recovered, 71 not recovered, 343,327 entries, `fail=0`, the recovery count to
the row. Its `Drawable` field is set on 130,239 entries and null on 213,088, the dictionary-children
and standalone cells above (3,694 dictionaries have a single child, so counting shared offsets
instead gives 126,545 / 216,782). `IsFragment` is true on 50,731 entries and `HasBoneBindings` on
59,544; `Tris` sums to 2,509,947,536. The same counting on a fresh `data.rpf` export gives that
row's 268 / 2,907, 1,758, 2,891 and 3,387,016. The other archives measure 4,686 fragments, 30,343
bone-bound and 126,301,454 triangles.

Every recovered entry was read back from its archive segments and decoded through
`RageModelDecoder.Decode`: **382,781 / 382,781 decoded, 0 decode failures**. The other archives
measure **236,311 meshes**, **104,183,438 vertices**, **126,301,454 triangles** and **512 embedded
textures**; `levels.rpf` adds 3,854,587 meshes, 2,364,218,711 vertices, 2,509,947,536 triangles and
1,293 embedded textures (the export's own summary line), for **4,090,898 meshes**, **2,468,402,149
vertices**, **2,636,248,990 triangles** and **1,805 embedded textures** over all ten archives.

What moved between `917e9b9` and `e293917`:

- **Standalone fragments (+32,505 standalone models total).** Upstream
  [rpf9-explorer#8](https://github.com/breadbbbb/rpf9-explorer/pull/8) (optional fragment pointers)
  accounts for the bulk of the gain across `levels.rpf` (+29,313 standalone), `packs.rpf` (+2,551
  fragments, all bone-bound), `data.rpf` (+594), `core_restofworld` (+40 standalone), and
  `shared_minigame000` (+6).
- **Texture-dictionary false positives eliminated (1,242 → 1).** Upstream
  [rpf9-explorer#9](https://github.com/breadbbbb/rpf9-explorer/pull/9) removes all 1,242 texture
  dictionaries that were previously proposed as model candidates in `levels.rpf` (1,009 → 0),
  `data.rpf` (75 → 0), `core_restofworld` (140 → 0), and `shared_minigame000` (18 → 0). Only one
  texture-dictionary candidate remains across all ten archives (in `textures.rpf`; rejected cleanly,
  and recovered by `Rpf9TextureRecovery`).
- **Probe caps and zero-radius drawables (+25 dictionary children plus capped standalones).**
  Upstream [rpf9-explorer#11](https://github.com/breadbbbb/rpf9-explorer/pull/11) recovers the 29
  previously capped candidates (+1 in `story000.rpf`, and +14 dictionary children plus capped
  standalones in `levels.rpf`). Upstream
  [rpf9-explorer#12](https://github.com/breadbbbb/rpf9-explorer/pull/12) recovers all 11 zero-radius
  dictionary children in `core_restofworld` (797 → 808); `story000_restofworld.rpf` already
  recovered its 3 models at `917e9b9`, so its delta is 0.
- **Remaining 72 rejects (down from 1,282 at `917e9b9`).** Aside from the single texture-dictionary
  candidate in `textures.rpf`, all 71 remaining rejects are in `levels.rpf` (0 texture dictionaries,
  0 zlib checksum errors). Sampled headers are fragment candidates (`LooksLikeFragment == true`)
  whose `ProbeRequiredExtents` reports `MetadataComplete == false` on the first zlib block with
  `RequiredSystemBytes` extending into subsequent 16-byte-aligned zlib continuation blocks.

Method checks: sharded 4 MiB `FindCandidateOffsets` was verified exact-equal against the
single-threaded scan on `packs.rpf` (`26,389 == 26,389`), and `tools/rpf9dump models` end-to-end
OBJ + manifest export was spot-checked on `story000_restofworld.rpf` (`ok=3 fail=0 skip=0`) and
`data.rpf` (`ok=3175 fail=0 skip=0`).

### Other archives at `917e9b9`

A fresh scan/recover at the new pin (no cache; `models/3`) shows the same pattern in every
archive with drawable dictionaries: the `c3708fb` recovery is the non-child count, and the new
entries are dictionary children.

| Archive | Candidates | Models at `917e9b9` | Dictionary children | `c3708fb` |
|---|---:|---:|---:|---:|
| `data.rpf` | 2,402 | **2,581** | 268 | 2,313 |
| `pack/shared_minigame000/...` | 1,070 | **1,288** | 317 | 971 |
| `pack/core_restofworld/...` | 1,024 | **885** | 797 | 88 |
| `pack/story000/story000.rpf` | 4,195 | **7,460** | 4,156 | 3,305 |
| `packs.rpf` | 23,838 | **24,034** | 23,720 | 314 |
| `levels.rpf` | 206,981 | **314,000** | 130,225 | 183,801 |
| `story000_restofworld` | 1 | **3** | 3 | 0 |
| `common`, `textures`, `story000_script`, `core_restofworld001` | 0 | 0 | 0 | 0 |
| **Total** | **239,511** | **350,251** | **159,486** | **190,792** |

### Re-count at `e293917`

A full `rpf9dump models` re-run over the same archives at the current pin (the manifests are
local; `levels.rpf` was exported separately, see the table note above):

| Archive | Rows | Non-fragment | Fragments | Delta vs `917e9b9` |
|---|---:|---:|---:|---:|
| `packs.rpf` | **26,585** | 23,720 | 2,865 | +2,551 |
| `data.rpf` | **3,175** | 1,417 | 1,758 | +594 |
| `pack/shared_minigame000/...` | **1,294** | 1,284 | 10 | +6 |
| `pack/core_restofworld/...` | **936** | 883 | 53 | +51 |
| `pack/story000/story000.rpf` | **7,461** | 7,461 | 0 | +1 |
| `story000_restofworld` | **3** | 3 | 0 | 0 |
| `common`, `textures`, `story000_script`, `core_restofworld001` | 0 | 0 | 0 | 0 |
| **Total, excluding `levels.rpf`** | **39,454** | **34,768** | **4,686** | **+3,203** |
| `levels.rpf` | **343,327** | 292,596 | 50,731 | +29,327 |
| **Total** | **382,781** | **327,364** | **55,417** | **+32,530** |

The gain outside `levels.rpf` is fragments: upstream `v1.2` PRs #8-#12 recover zero-radius
fragments that the earlier export skipped (2,551 in `packs.rpf`, 594 in `data.rpf`), plus one
drawable in `story000`. `story000_restofworld` has the same 3 models at both pins, so its delta
is 0.

`levels.rpf` was re-counted at `e293917` (10 shards over 235,330 candidates): **343,327 models**
(130,239 dictionary children + 213,088 standalone), against 314,000 at `917e9b9` (206,981
candidates; 130,225 children + 183,801). The +29,327 is a recovery count, not a row-for-row delta.
The full OBJ export at `e293917` (table note above) gives the same 343,327 rows.

The last column is the `c3708fb` recovery for comparison. `story000`'s standalone count is
3,304 vs 3,305; the full new-pin standalone total is 190,765, 27 below the `c3708fb` 190,792.
Those 27 are not merged duplicates: they are the 27 models with more than 256 geometries that the
probe rejects at `917e9b9` (26 in `levels.rpf`, 1 in `story000`; see
[rejected candidates](#rejected-candidates-at-917e9b9)). The new entries are **159,486
drawable-dictionary children**: one model per embedded drawable in a dictionary root, not new
pages. `core_restofworld` recovers 873 of its 1,024 candidates (885 models); the 151 rejects are
140 texture dictionaries and 11 drawable dictionaries whose drawable has radius 0.

`levels.rpf` is the sum of ten parallel recovery shards over 206,981 candidates: **314,000
models**, 130,225 dictionary children, 183,775 standalone. 1,037 candidates are rejected: the
1,009 in the last shard are texture dictionaries, not models, and 28 more in shards 1 to 7 are
models the probe caps reject (see [rejected candidates](#rejected-candidates-at-917e9b9)). The
`levels.rpf` full export was not re-run, so its triangle total is not re-run here.

### Rejected candidates at `917e9b9`

Every candidate of each model-bearing archive was run through `RecoverAtOffsets` on its own and,
for each reject, through a copy of `TryResource` that records which check drops it (it agrees
with the real code on all 206,981 `levels.rpf` candidates). Measured on the 1.0.2.1 package:

| Archive | Rejects | Texture dictionaries | Geometry cap (`:351`) | List 4 cap (`:387`) | Radius 0 |
|---|---:|---:|---:|---:|---:|
| `levels.rpf` | 1,037 | 1,009 | 26 | 2 | 0 |
| `data.rpf` | 75 | 75 | 0 | 0 | 0 |
| `pack/core_restofworld/...` | 151 | 140 | 0 | 0 | 11 |
| `pack/shared_minigame000/...` | 18 | 18 | 0 | 0 | 0 |
| `pack/story000/story000.rpf` | 1 | 0 | 1 | 0 | 0 |
| `packs.rpf` | 0 | 0 | 0 | 0 | 0 |

Line numbers are in `third_party/rpf9-explorer/Core/RageModelDecoder.cs`.

- **Texture dictionaries (1,242).** `LooksLikeDrawableDictionary` (`:78`) checks only the
  dictionary header, which texture dictionaries share, so they become model candidates and fail
  at `:314` because their values are not drawables. `Rpf9TextureRecovery` recovers all 1,242.
  In `levels.rpf` they are the archive's last 0.12 GiB, each a 256- to 880-byte system block
  and a graphics block. None of them overlaps a recovered model. Upstream
  [rpf9-explorer#9](https://github.com/breadbbbb/rpf9-explorer/pull/9) stops the scan from
  proposing them.
- **Geometry cap (27).** The probe allows at most 256 geometries per model (`:351`), while
  `Decode` allows `MaxMeshes`, 1,024 (`:535`). These models have 263 to 377 geometries.
- **List 4 cap (2).** The probe caps renderer list 4 at 4,096 entries (`:387`) but never uses
  that count for sizing. These two dictionaries have 5,078 and 5,461.
- **Radius 0 (11).** Drawable dictionaries whose drawable has a zero radius and non-empty bounds,
  rejected by `LooksLikeDrawableHeader` (`:124`) through `:314`. Whether they decode otherwise
  has not been checked.

[rpf9-explorer#11](https://github.com/breadbbbb/rpf9-explorer/pull/11) raises both probe caps.
With it, the 29 capped candidates recover (41 models, 3.25 million triangles), and every
candidate already recovered decodes exactly as before in all six archives above.

One `levels.rpf` copy measured for this showed a 1,038th reject: a zlib block whose Adler-32
fails because of a single flipped bit in that copy (flipping it back restores the stored
checksum). It is local corruption, not part of the package, and is left out of the table.

## Independent validation

Scoped validation at PR #1 head `c3708fb` with .NET 8.0.428 on the real package:

| Scope | Candidates | Recovered | Rejects | Readback |
|---|---:|---:|---:|---:|
| `data.rpf` full | 2,313 | 2,313 | 0 | 2,313 / 2,313 |
| `common.rpf` | 0 | 0 | 0 | 0 / 0 |
| `levels.rpf` 0-8 GiB | 28,748 | 28,748 | 0 | 28,748 / 28,748 |
| `levels.rpf` 4-4.5 GiB | 4,010 | 4,010 | 0 | 4,010 / 4,010 |
| `levels.rpf` 12-16 GiB | 10,429 | 10,429 | 0 | 10,429 / 10,429 |

The `data.rpf` export reported `ok=2,313 fail=0 skip=0`, 2,313 unique OBJ names, and 0 duplicate
stems. All 963 pre-PR misses were recovered, and all 1,350 pre-PR OBJs were byte-identical to their
pre-PR versions. The PR smoke test passes; v1/v2 cache round-trip and two raw-segment entries were
verified.

Caveat: this scoped run did not scan the full `levels.rpf`; the full pass below covers it. The
`data.rpf` OBJ text is the only byte-identical file comparison; the full pass counted decoded
models, not OBJ bytes. No counterexample appeared in the tested scopes.

### Full-archive run

A full pass at the same head over `levels.rpf` (70.9 GiB) and the other model-bearing archives
reproduced the ten-archive total. The `48557ea` column is the pre-fix control:

| Archive | Candidates | `48557ea` | `c3708fb` |
|---|---:|---:|---:|
| `levels.rpf` (full, 70.9 GiB) | 183,801 | 149,811 | **183,801** |
| `pack/story000/story000.rpf` | 3,305 | 3,303 | **3,305** |
| `data.rpf` | 2,313 | 1,350 | **2,313** |
| `pack/shared_minigame000/shared_minigame000.rpf` | 971 | 783 | **971** |
| `packs.rpf` | 314 | 314 | 314 |
| `pack/core_restofworld/core_restofworld.rpf` | 88 | 37 | **88** |
| `common`, `textures`, `story000_restofworld`, `core_restofworld001` | 0 | 0 | 0 |
| **Total** | **190,792** | **155,598** | **190,792** |

The `packs.rpf` row is the pre-dictionary-root control: at `917e9b9` that archive recovers
24,034 models (26,585 at `e293917`), and the other archives' new-pin counts are in the table above.
The rows here are the `c3708fb` measurements (190,792 models); the `917e9b9` re-count is 350,251,
and the `e293917` re-count outside `levels.rpf` is 39,454. The `e293917` triangle totals of all ten
archives, `levels.rpf` included, are in the table above.

At `c3708fb`, every candidate is recovered, read back and decoded: 0 decode failures, 0 readback
failures, no duplicate model names, and 2.22 billion triangles (1.95 billion before). The
169-candidate subset is a different scope and was not isolated here.

## How the exporter works

```
rpf9dump models <archive.rpf> <outdir> [--start BYTES] [--length BYTES]
```

`FindCandidateOffsets` scans 4 MiB blocks for a page-aligned 16-byte zero prefix followed by a zlib
stream at page offset +16. It inflates the first 176 bytes and keeps the offset when the result has
a drawable or fragment header (`RageModelDecoder.LooksLikeResource`). `RecoverAtOffsets` follows
the following zlib blocks, splits the inflated bytes into the system segment and the graphics
segment, probes the extents, and calls `RageModelDecoder.Decode`.

`--start`/`--length` scan one window of the file instead of the whole archive. Neither `models`
nor `textures` needs the archive directory: the entry list comes from this run's scan or from the
index cache, and every resource is read from its own compression blocks
(`tools/rpf9dump/Segments.cs`). A window is therefore a raw slice - a copy of `levels.rpf` from an
offset, which has no `9FPR` header - and the scan covers exactly the slice. A sub-range is never
written to the index cache, which describes a whole file; the manifest records `Scan.Start` and
`Scan.Length` for it. One measured 1 GiB slice of `levels.rpf`: 1,228 models scanned, recovered,
decoded and exported in about 2 s (497 MB of OBJ and manifest).

To locate a named drawable before decoding, `tools/rpf9_names.py` indexes the measured
`name.#dr`/`#ft`/`#dd` spellings in bounded zlib prefixes without the encrypted TOC; see
[resource_names.md](resource_names.md).

For each returned entry, `tools/rpf9dump/Program.cs` reads the segments, splits at
`ResourceSystemBytes`, decodes the model, writes one OBJ, and writes the manifest. The manifest is
`<outdir>/<archive>/models/manifest.json` with `src`, `Count`, and `Items[]`. Each item records the
offset, OBJ name, model name, mesh/vertex/triangle counts, fragment and bone-binding flags, bounds,
per-mesh shader indices, embedded textures, and material texture references.

Three arrays describe the geometries one by one, in the same order as the OBJ's `o` groups:
`MeshTris` (triangles), `MeshModels` (the position of the model in the drawable's primary-LOD model
list) and `MeshGeoms` (the position of the geometry in that model). That pair is the geometry index
a dehydrated BLAS leaf stores ([bvh.md](bvh.md) §4a), which is what `tools/bvh_match.py` matches on;
`MeshShaders` maps each geometry to its entry in `Materials`, and therefore to its shader name hash.

The manifest has no failure, skip, or layout fields. `Program.cs` increments `fail` only when a
model entry returned by the carver fails `RageModelDecoder.Decode`; `skip` is not used in models
mode. A printed `fail=0 skip=0` therefore means "no decoded entry failed", not "nothing was
rejected".

## Pre-fix measured counts

Counts below are from submodule `48557ea` before the upstream fix:

| Archive | Candidates | Recovered | Not recovered |
|---|---:|---:|---:|
| `common.rpf` | 0 | 0 | 0 |
| `textures.rpf` | 0 | 0 | 0 |
| `data.rpf` | 2,313 | 1,350 | 963 |
| `levels.rpf`, 0-8 GiB sample | 28,748 | 26,184 | 2,564 |
| `levels.rpf`, 4-4.5 GiB sample | 4,010 | 3,821 | 189 |

`levels.rpf` was sampled, not scanned end to end. Its first 3.9 GiB held no model candidates.
All counts come from the same package and the same `48557ea` submodule.

The 2,313 candidates in `data.rpf` decode to 2,313 unique model names once the checks below are
relaxed, so the 963 missing entries are valid models, not duplicate or false candidates.

## Pre-fix rejection classes

Observed in `data.rpf` with the unmodified carver:

| Class | Count | Location |
|---|---:|---|
| Drawable bounds do not contain a geometry's bounds | 231 | `Core/RageModelDecoder.cs:377` |
| Graphics block has non-zero bytes after `RequiredGraphicsBytes` | 259 | `Core/Rpf9ModelRecovery.cs:140` |
| Fragment system block with trailing bytes; next block not requested | 388 | `Core/Rpf9ModelRecovery.cs:132-133` |
| A following block was found, but no split decoded | 67 | `Core/Rpf9ModelRecovery.cs:141-143` |
| No following block was found | 18 | `Core/Rpf9ModelRecovery.cs:112-113` |

In the `levels.rpf` 0-8 GiB sample the same diagnostic classes were 978 bounds failures, 190
graphics-trailing cases, 1,200 cases where a system block was not followed to the next block, and
196 cases where a block was found but no split decoded.

The last two classes are symptoms of the splitter. The first three are the actionable causes.

Cause isolation on `data.rpf`:

| Change | Recovered |
|---|---:|
| Unmodified carver | 1,350 |
| Relax system trailing-byte check only | 1,408 |
| Relax graphics trailing-byte check only | 1,609 |
| Relax drawable bounds check only | 1,581 |
| Truncate both spans to the probed extents | 1,969 |
| Truncate both spans and relax drawable bounds | 2,313 / 2,313 |

`RequiredSystemBytes` and `RequiredGraphicsBytes` are the ends of the last structures the probe
actually validated. Bytes after those ends are legal: system blocks can carry inline strings, and
a graphics zlib block can carry trailing data. Decoding the full spans and then requiring every
trailing byte to be zero rejects valid resources.

## Measured diagnosis and the upstream fix

This is our measured diagnosis, not a competing patch. The two hunks below record the minimum our
runs isolated; PR #1 implements this area more broadly, including cluster marker and palette
handling, packed-position bounds, and mixed raw/zlib block handling, and its exact code may differ.
The hunks are against `breadbbbb/rpf9-explorer` at `48557ea` and are not applied to this checkout.

```diff
--- a/Core/Rpf9ModelRecovery.cs
+++ b/Core/Rpf9ModelRecovery.cs
@@ -132,3 +132,2 @@
-            if (extent.RequiredSystemBytes > prefixLength ||
-                system.AsSpan(extent.RequiredSystemBytes).IndexOfAnyExcept((byte)0) >= 0) continue;
+            if (extent.RequiredSystemBytes > prefixLength) continue;
             int graphicsLength = total - prefixLength;
@@ -140,3 +139,4 @@
-            if (graphics.AsSpan(extent.RequiredGraphicsBytes).IndexOfAnyExcept((byte)0) >= 0) continue;
-            RageModel? decoded = RageModelDecoder.Decode(system, graphics);
+            RageModel? decoded = RageModelDecoder.Decode(
+                system.AsSpan(0, extent.RequiredSystemBytes),
+                graphics.AsSpan(0, extent.RequiredGraphicsBytes));
             if (decoded is null) continue;
```

```diff
--- a/Core/RageModelDecoder.cs
+++ b/Core/RageModelDecoder.cs
@@ -373,8 +373,6 @@
                     var mesh = ReadMesh(geometry, $"model_{mi}_geometry_{gi}", partMin, partMax, boneCount);
                     if (shaderMapping is int mapping)
                         mesh = mesh with { ShaderIndex = U16(sys, mapping + gi * 2) };
-                    float tolerance = Tolerance(min, max);
-                    Require(Contains(min, max, partMin, tolerance) && Contains(min, max, partMax, tolerance));
                     meshes.Add(mesh);
                 }
             }
```

The bounds check is safe to remove because `ReadMesh` still validates every referenced index and
every decoded vertex against the geometry's own part bounds (`Core/RageModelDecoder.cs:643-646`).
The deleted check compared each geometry with the drawable-level bounds, which is not a valid
envelope for a drawable with multiple LOD models.

The narrow two-hunk experiment in a temp copy recovered:

- `data.rpf`: 2,313 / 2,313 entries.
- `levels.rpf` 4-4.5 GiB: 4,010 / 4,010 entries.
- `levels.rpf` 0-8 GiB: 28,746 / 28,748 entries.

The two remaining 8 GiB cases are not tag rejections: one graphics extent is 48 bytes larger than
the assembled block and the next zlib stream starts 3,419 bytes past it; one model with 32
geometries throws inside `ReadMesh`. PR #1's broader patch resolves both; at `c3708fb` the same
sample has 0 rejects.

## Vertex and index tags

The pre-fix tag survey (`data.rpf` plus the 0-8 GiB `levels.rpf` sample) contains 198,286 geometry
records. No unhandled layout tag appears:

| Item | Observed |
|---|---|
| Position `(size, format)` | `(8, 16)` on 195,182 geometries; `(12, 6)` on 674; no other pair |
| Index buffer stride | 4 on every geometry |

`(12, 6)` and `(8, 16)` are the only position encodings accepted by
`Core/RageModelDecoder.cs:244` and `:548-571`. The index stride must be 4
(`Core/RageModelDecoder.cs:249` and `:576`). Other values would be rejected, but none occur in
`data.rpf` or the `levels.rpf` samples. `data.rpf` uses `(8, 16)` exclusively; the 674
`(12, 6)` geometries are spread over 256 models in the levels sample.

## Vehicle ratings and performance estimates

`common.rpf` has one `CVehicleBaseRatingsAndEstimatesMap` XML document (see [gamedata.md](gamedata.md)): a
`Map` of 252 `Item key="<model>"` entries, each with four blocks. Counts and ranges only; no model names.

| Block | Fields |
|---|---|
| `estimatesRoad`, `estimatesOffroad` | `fTopSpeed`, `fCornerSpeed`, `fAccelerationTime`, `fBrakingTime`, `fStandingQuarterTime`, `fStandingQuarterSpeed`, `fAvgLatTraction` |
| `ratingsRoad`, `ratingsOffroad` | `fOverallRating`, `fSpeedRating`, `fHandlingRating`, `fAccelerationRating`, `fBrakesRating` |

- **Keys are vehicle model names.** The 252 keys are distinct plain names (no hex hashes), and all 252 are the
  base name of a `.yft` in the local archive catalog. 245 of them are also among the 319 models of the
  distant-light recipes; 74 of those 319 have no rating entry.
- **Ratings are not fractions.** All ten rating fields have a minimum of exactly 10 and medians of 142 to 585,
  with maxima up to 1,327 (`fAccelerationRating`). `fOverallRating` is not the mean of the other four: the two
  differ by a median of 38 and agree within 0.01 in only 10 of 504 cases.
- **Estimates look like SI units.** Road `fTopSpeed` has a median of 44.3 and a maximum of 77.4 (m/s would make
  that 160 and 279 km/h); `fStandingQuarterTime` has a median of 15.8, the right size for seconds over a
  quarter mile; `fAvgLatTraction` has a median of 1.42 on road and 0.92 off road (lateral g would fit). Offroad
  speeds are lower than road ones (median top speed 19.3).
- **Sentinels.** On each surface, 11 entries have every estimate 0 and 10 more have zero speeds with an
  infinite time (`inf`). `inf` occurs in 38 road and 99 offroad blocks overall, so a parser must accept `inf`
  in these fields (it reads as "never reaches the test speed or distance").
- **`fOverallRating` is a weighted mean of the other four ratings** (measured, confirmed to float noise) on the
  345 of the 504 rating blocks whose four parts are all above 10: there `fOverallRating` = 0.21
  `fSpeedRating` + 0.38 `fHandlingRating` + 0.40 `fAccelerationRating` + 0.01 `fBrakesRating`, with a residual
  of at most 7e-5. The four weights sum to 1, so handling and acceleration weigh about twice speed and brakes is 1%.
  Differencing a key's road and offroad blocks gives the same 0.2100 / 0.3800 / 0.4000 / 0.0100 on the 139 keys
  that are above 10 on both surfaces, so the weights do not depend on the surface.
- **The overall has the same observed minimum of 10.** The other 159 rating blocks each have at least one part
  at exactly 10, and the formula does not hold for them: they sit up to 4.1 below the weighted mean, and 24 of
  them carry an `fOverallRating` of exactly 10. No block without a part at 10 deviates. The exports show only
  this minimum; they cannot tell where, or whether, the engine clamps.
- **Scale.** The ratings are continuous floats, not a 0-1 or 0-100 score: 342 to 451 distinct values per field
  over the 504 blocks, a floor of exactly 10, and `fAccelerationRating` the only field past 1,000 (22 blocks,
  maximum 1,327).

Not measured here (hypothesis): whether the estimates are precomputed from handling data or measured by
simulation, and what fixes the top of the rating scale.
- **The estimates are a separate stored table, not the performance maps.** `common.rpf`'s data area at `0x9580`
  also holds `CVehiclePerformanceMaps` (285 rows, 10 result fields) and `CTyreDataSet` (40 tyre types). 251 of the
  252 estimate keys join the performance rows, but no field reproduces an estimate exactly: the best OLS over
  all ten gives R^2 0.872, and `fAccelerationTime` against `accelZeroSixty` is R^2 0.811 with a median |residual|
  of 0.54 s. The clearest relation is internal to the estimates: over the 231 rows with a finite top speed,
  `fStandingQuarterTime = 6.73 + 401.0 / fTopSpeed` (R^2 0.99998, median |residual| 1.01 s; R^2 0.82 over the
  209 rows under 30 s). For the seven top-speed-limited rows, `fTopSpeed * fStandingQuarterTime` is
  402.339 +/- 0.0003 m, a quarter mile; the residual tracks the 0-60 time.
- **The ratings are deterministic functions of the estimates.** Each of the 223 distinct road top speeds gives
  one `fSpeedRating`; handling/corner has 221 levels, acceleration/acceleration time 204, brakes two collisions,
  and `fOverallRating` uses more than one field. The rating scale and the overall weighting are still not fitted.
- **The handling inputs are not in the readable archives.** No `CHandlingData`/`handling.meta`, `fMass`,
  `fInitialDriveForce`, `fInitialDragCoeff`, `nInitialDriveGears`, `fTractionCurve*`, `fBrakeForce` or
  `fSuspensionForce` occurs in `common.rpf`, `data.rpf`, `packs.rpf` or `script.rpf`; the executable is absent.
  `levels.rpf` is map data and was not fully scanned. The table is therefore consistent with an offline
  estimator fed by handling-derived scalars, but whether it was fitted to handling or copied from a simulation
  cannot be verified from this package.

## Character skeletons

World drawables do not carry the shared 597-name humanoid skeleton. Character `.#ft` fragments
store that name/hash table, and each character `#dr` model points at a local bone-hash array that
maps into it. The measured name table, the following 597-record hierarchy/reference-TRS table,
the model mapping and the texture-parameter split are in
[character_skeleton.md](character_skeleton.md). The character inverse bind matrices are at
skeleton header `+0x30` and agree with the default bone transforms (see that page);
`tools/anim_glb.py` exports a skinned part with them.

The character-part inventory (Lucia's named records and decoded parts, the `player_story000_00`
drawable set, slot-name counts and texture coverage) is in
[character_models.md](character_models.md).

### Character census

`tools/character_census.py` totals a decoded model tree or an `rpf9dump models` manifest:

```powershell
python -I tools/character_census.py <models-dir-or-manifest> [--json FILE] [--csv FILE]
```

It reads every nested `model.json` (the decoded-pack layout: `type`, `vertex_count`,
`triangle_count`, `materials[].shader_name_hash`, `materials[].textures[].hash64`,
`drawable.skeleton`/`drawable.models[].bone_palette` and `meshes[].skin_influences`), or the
`Items[]` of a manifest (`Verts`, `Tris`, `IsFragment`, `HasBoneBindings`, `Materials[]`). The
local exporter does not write an index count, so the index column and total normally stay unknown;
an input that supplies `index_count` is counted, but the field is optional and not part of the
decoded layout. It prints one row per model with kind, vertex/triangle counts, bones, 8-influence
skin, shader hashes and texture keys, then totals by kind and by shader hash. A missing or wrongly typed field
is reported on that row instead of ending the run. An `rpf9dump models` manifest records only
`HasBoneBindings`, not a bone count or influence width, so those cells stay `?`. Report paths that
alias an input are refused. The tests are synthetic: `python -I tests/test_character_census.py`.

## Shader render buckets

`rpf9dump` writes two raw shader fields per material, `RenderBucket` (byte at shader +0x39) and
`RenderBucketMask` (u32 at +0x3C; `RageModelDecoder.cs:651`). Measured on a random sample of 3,000 `.ydr`/`.yft`
models (27,435 shaders):

- `RenderBucketMask == 0xFF00 | (1 << (RenderBucket & 0x7F))` in all 27,435. Bit 7 of `RenderBucket` is a
  separate flag (set in 438) that does not change the mask.
- `RenderBucket & 0x7F` takes the values 0-6, the GTA V drawable bucket list. Model-name tokens (counted per
  shader, from the model's name) line up with it:

| Bucket | GTA V meaning | Shaders | Name evidence (share of shaders whose model name matches) |
|---:|---|---:|---|
| 0 | opaque | 17,826 | baseline: water 0.6%, vegetation 0.5%, glass/window 0.2%, decal 0.1% |
| 1 | alpha | 725 | glass/window 7.6% (38x baseline) |
| 2 | decal | 7,282 | decal/graffiti/dirt 3.1% (31x) |
| 3 | cutout | 1,311 | vegetation 2.8% (5.6x) |
| 4 | no splash | 6 | one program (`fc34f08b`) |
| 5 | no water | 6 | the same program |
| 6 | water | 343 | water/river/pool/canal 80.2% |

Names are per model and a building holds glass, decal and opaque shaders alike, so the token shares are low
in absolute terms; the enrichment against bucket 0 is the signal. For export: bucket 1 needs blending, 2 is
a projected decal, 3 needs alpha testing, 6 is a water surface.

`common.rpf` carries the matching preset-to-effect table (`SPS1`, decoded by `tools/gamedata_sps1.py`,
[`gamedata.md`](gamedata.md#sps1-shader-preset-table)): `858` `.sps` presets map onto `539` `SGD5` Database 0 effects
with a default `u8` render bucket (`0..7`), `63` effects accept multiple buckets, and the single bucket-`4`
and bucket-`5` presets both resolve to effect `fc34f08b`.

## Material shader hashes and the carved programs

`rpf9dump` also writes each material's shader hash (`manifest.json` → `Materials[].Shader`, u32 as 8 hex digits).
It is the joaat (lower-cased) of the material's **effect** name. The carve index (`<work>/shaders/dxbc/index.json`)
names compute programs `<effect>:<entry>_<8 hex>_Wrapped` after a one-byte length prefix (the `:` becomes `_` in
file names), so the effect is recoverable from them. Measured on a 1.0.2.1 copy, `data.rpf` models and the full
26,503-program carve:

| | Count |
|---|---:|
| Materials (3,175 models) | 3,340 |
| Distinct material shader hashes | 18 |
| Compute programs named `<effect>:<entry>` | 2,689 (every CS), 564 effects |
| Effects with `RaytracingResolve` CS programs | 341 (1,004 programs; 3 each for 324) |
| Materials whose hash is the joaat of a carved effect name | 3,211 (96%), 8 of the 18 hashes |
| Materials whose hash is the joaat of a native effect record's name (both SGD5 databases, below) | 3,340 (100%), 18 of the 18 hashes, one record each |

So a material leads to its effect. Through the carved compute names it reaches only 8 of the 18 hashes; the
effect records' own names reach all 18 (@jeremygold02, `tools/sgd5_effects.py` in #3036). The draw (VS/PS/MS)
programs are not reachable through the carved programs' names, order or suffixes:

- None of the 23,619 graphics programs (2,236 VS, 12,101 PS, 9,282 MS; the 195 GS are left out here) carries an
  effect name; they are named `<stage>_<technique or entry>_<8 hex>`.
- `shaders.rpf` stores programs grouped by kind, not by effect (the 1,004 `RaytracingResolve` programs sit next
  to each other), so position doesn't group an effect's programs either.
- Every one of the 23,619 has an 8-hex suffix. The suffixes group them: there are 1,228 distinct ones, the largest
  shared by 1,806 programs and 53 used once. The suffix is not the joaat of the program name, and only 1 of the
  896 suffixes on the effect-named CS programs appears on any graphics program.

They are reachable through the **effect records** in `shaders.rpf` (found by @jeremygold02, #2519). All fields
are little-endian:

1. **The effect record.** It starts with a one-byte name length and the name, including its terminator, then a
   u32 (below). Six stage lists follow, each a u32 count and that many u32 ordinals.

   The u32 is a size computed from the record's own counted fields (@jeremygold02). Let G be the ordinals in the
   six lists, T the techniques, P the passes, and W the post-pass word count. Let A and B be the two block
   lengths after them, E the post-pass entry count, I the payload bytes of the entries whose tag is a length
   (not `0x80000000`), and S the string-pool length. Lengths don't count their own prefixes. Then, with
   `x = 4W + A + B`:

   ```text
   word = 48 + 8(G + T + E) + 32P + I + S + (round_up(x, 8) if E else x)
   ```

   This holds for all 932 records of this build (910 + 22, recomputed here). The value never equals the record's
   serialized length (0 of 932), so the next record is found by the sequential walk, not by this word. What the
   runtime uses it for is unknown. Groups 3 and 4 are empty here, so their share of it is unmeasured.

   `shaders.rpf` holds **two shader databases** (SGD5), and an ordinal counts within one of them (found by
   @jeremygold02). Each database stores six program groups, each a u32 count followed by its programs. A program
   is a u8-length label including its NUL, one tag byte, a u32-length payload, 152 fixed bytes and a u32-length
   trailer. The u32 after group 5 is the database's effect count. An ordinal is the zero-based position in the
   group of the effect's own database, so a lookup needs the database, the group and the ordinal.

   | Database (group walk starts at) | 0: VS + MS | 1: PS | 2: GS | 3 | 4 | 5: CS | Effects declared |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | `0x10E39` | 11,018 | 10,944 | 195 | 0 | 0 | 2,647 | 910 |
   | `0x10289C81` | 500 | 1,157 | 0 | 0 | 0 | 42 | 22 |

   The two databases add up to the whole carve: 11,518 VS + MS, 12,101 PS, 195 GS and 2,689 CS. Groups 3 and 4 are
   empty and their stages are not identified. The first database's programs come first in the archive, so for its
   effects the ordinal also equals the program's position among the carve index's programs of that group, sorted
   by `offset`. The three records here are all in the first database: the wiper's 11017 is the last entry of its
   group 0.

   On the rim and the tire, list 5's three entries are exactly the effect's own `<effect>:RaytracingResolve`
   programs, the ones the material hash leads to in the table above. In the second database all 42 group-5
   references resolve to programs that carry their effect's name; resolved against the archive-wide CS list
   instead, none of them do (@jeremygold02).
2. **The technique table.** A u32 count, then one 4-byte entry per technique: the low 15 bits of its first u16 are
   the name's offset in a string block, and byte 3 is the pass count. A u32 pass total follows, then 24-byte pass
   records. Where the table and its string block sit relative to the record is not derived yet: the three
   tables were located by offset.
3. **The pass records.** The first six bytes of each pass select an entry of each stage list: 0 is unused,
   otherwise `stage_list[selector - 1]`.

The group walk and its counts above were reproduced on a 1.0.2.1 copy (the same `shaders.rpf`, SHA-256
`0bc16345…`). So was @jeremygold02's census of both databases' effects with `tools/sgd5_effects.py
--list-effects` (#3036, `0029eba7`): 932 effect records with 932 distinct name hashes, 913 of them with a
technique table, 25,037 techniques and 28,798 passes. Since `c0439966` the tool also **walks** the records
forward from each database's effect count, using only stored lengths: every record's string pool, the next
record, and at the end the terminal word and zero padding. Its tests, run against this package
(`LEONIDA_GAME_DIR` set, `tests/test_sgd5_effects.py`), pass: the walk and the earlier search agree on every
record, technique and pass. The record and pass format below was also checked by hand on three records:

| Record | Stage list sizes | Techniques | Passes (total = sum of counts) | First pass list 0 / list 1 ordinal |
|---|---|---:|---:|---|
| windshield wiper mask | 1, 1, 0, 0, 0, 0 | 1 | 1 | 11017 / 10943 |
| stock rim | 41, 53, 0, 0, 0, 3 | 106 | 108 | 9143 / 8925 |
| stock tire | 41, 53, 0, 0, 0, 3 | 106 | 108 | 9089 / 8835 |

Two things are still open:

- **The u32 after an effect's name.** It is reproduced from the record's fields; how the runtime uses it is
  still unknown.
- **Which technique and pass the game draws with.** That is a run-time choice; it is presumably made in the
  executable (RT-5).

Until both are settled, a real model is drawn with a chosen pair ([first_light.md](first_light.md)). The effect
records already give that choice a pair that can be checked independently.

So a material leads to its effect. Through the carved compute names it reaches only 8 of the 18 hashes; the
effect records' own names reach all 18 (@jeremygold02, `tools/sgd5_effects.py` in #3036). The draw (VS/PS/MS)
programs are not reachable through the carved programs' names, order or suffixes:

- None of the 23,619 graphics programs (2,236 VS, 12,101 PS, 9,282 MS; the 195 GS are left out here) carries an
  effect name; they are named `<stage>_<technique or entry>_<8 hex>`.
- `shaders.rpf` stores programs grouped by kind, not by effect (the 1,004 `RaytracingResolve` programs sit next
  to each other), so position doesn't group an effect's programs either.
- Every one of the 23,619 has an 8-hex suffix. The suffixes group them: there are 1,228 distinct ones, the largest
  shared by 1,806 programs and 53 used once. The suffix is not the joaat of the program name, and only 1 of the
  896 suffixes on the effect-named CS programs appears on any graphics program.

They are reachable through the **effect records** in `shaders.rpf` (found by @jeremygold02, #2519). All fields
are little-endian:

1. **The effect record.** It starts with a one-byte name length and the name, including its terminator, then a
   u32 (below). Six stage lists follow, each a u32 count and that many u32 ordinals.

   The u32 is a size computed from the record's own counted fields (@jeremygold02). Let G be the ordinals in the
   six lists, T the techniques, P the passes, and W the post-pass word count. Let A and B be the two block
   lengths after them, E the post-pass entry count, I the payload bytes of the entries whose tag is a length
   (not `0x80000000`), and S the string-pool length. Lengths don't count their own prefixes. Then, with
   `x = 4W + A + B`:

   ```text
   word = 48 + 8(G + T + E) + 32P + I + S + (round_up(x, 8) if E else x)
   ```

   This holds for all 932 records of this build (910 + 22, recomputed here). The value never equals the record's
   serialized length (0 of 932), so the next record is found by the sequential walk, not by this word. What the
   runtime uses it for is unknown. Groups 3 and 4 are empty here, so their share of it is unmeasured.

   `shaders.rpf` holds **two shader databases** (SGD5), and an ordinal counts within one of them (found by
   @jeremygold02). Each database stores six program groups, each a u32 count followed by its programs. A program
   is a u8-length label including its NUL, one tag byte, a u32-length payload, 152 fixed bytes and a u32-length
   trailer. The u32 after group 5 is the database's effect count. An ordinal is the zero-based position in the
   group of the effect's own database, so a lookup needs the database, the group and the ordinal.

   | Database (group walk starts at) | 0: VS + MS | 1: PS | 2: GS | 3 | 4 | 5: CS | Effects declared |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | `0x10E39` | 11,018 | 10,944 | 195 | 0 | 0 | 2,647 | 910 |
   | `0x10289C81` | 500 | 1,157 | 0 | 0 | 0 | 42 | 22 |

   The two databases add up to the whole carve: 11,518 VS + MS, 12,101 PS, 195 GS and 2,689 CS. Groups 3 and 4 are
   empty and their stages are not identified. The first database's programs come first in the archive, so for its
   effects the ordinal also equals the program's position among the carve index's programs of that group, sorted
   by `offset`. The three records here are all in the first database: the wiper's 11017 is the last entry of its
   group 0.

   On the rim and the tire, list 5's three entries are exactly the effect's own `<effect>:RaytracingResolve`
   programs, the ones the material hash leads to in the table above. In the second database all 42 group-5
   references resolve to programs that carry their effect's name; resolved against the archive-wide CS list
   instead, none of them do (@jeremygold02).
2. **The technique table.** A u32 count, then one 4-byte entry per technique: the low 15 bits of its first u16 are
   the name's offset in a string block, and byte 3 is the pass count. A u32 pass total follows, then 24-byte pass
   records. Where the table and its string block sit relative to the record is not derived yet: the three
   tables were located by offset.
3. **The pass records.** The first six bytes of each pass select an entry of each stage list: 0 is unused,
   otherwise `stage_list[selector - 1]`.

The group walk and its counts above were reproduced on a 1.0.2.1 copy (the same `shaders.rpf`, SHA-256
`0bc16345…`). So was @jeremygold02's census of both databases' effects with `tools/sgd5_effects.py
--list-effects` (#3036, `0029eba7`): 932 effect records with 932 distinct name hashes, 913 of them with a
technique table, 25,037 techniques and 28,798 passes. Since `c0439966` the tool also **walks** the records
forward from each database's effect count, using only stored lengths: every record's string pool, the next
record, and at the end the terminal word and zero padding. Its tests, run against this package
(`LEONIDA_GAME_DIR` set, `tests/test_sgd5_effects.py`), pass: the walk and the earlier search agree on every
record, technique and pass. The record and pass format below was also checked by hand on three records:

| Record | Stage list sizes | Techniques | Passes (total = sum of counts) | First pass list 0 / list 1 ordinal |
|---|---|---:|---:|---|
| windshield wiper mask | 1, 1, 0, 0, 0, 0 | 1 | 1 | 11017 / 10943 |
| stock rim | 41, 53, 0, 0, 0, 3 | 106 | 108 | 9143 / 8925 |
| stock tire | 41, 53, 0, 0, 0, 3 | 106 | 108 | 9089 / 8835 |

Two things are still open:

- **The u32 after an effect's name.** It is reproduced from the record's fields; how the runtime uses it is
  still unknown.
- **Which technique and pass the game draws with.** That is a run-time choice; it is presumably made in the
  executable (RT-5).

Until both are settled, a real model is drawn with a chosen pair ([first_light.md](first_light.md)). The effect
records already give that choice a pair that can be checked independently.

## Vehicle distant-light recipes

`common.rpf` has one `CVehicleDistantLightData` XML document (see [gamedata.md](gamedata.md)) that ties
vehicle models to far-away light recipes. Counts only; no model names.

| Path | Count | Content |
|---|---:|---|
| `distantLights`, `presetLights` | 1 each | empty |
| `recipeLights/Item` | 1,980 | `vehicleModelName`, `secondaryRecipes`, `lights` |
| `recipeLights/Item/secondaryRecipes/Item` | 2,383 | one token each; 0 to 3 per recipe (0: 312, 1: 1,029, 2: 563, 3: 76) |
| `recipeLights/Item/lights` | 1,980 | empty in all |

- **Every `vehicleModelName` is a real fragment model.** The 1,980 recipes name 319 distinct models, and all
  319 exist as `.yft` entries in the local archive catalog, each also with a `<name>_hi.yft` high-detail
  variant. None is a `.ydr`. A loader can therefore key the recipes by the vehicle's `.yft` base name.
- A model has 1 to 45 recipes (93 models have one); in 304 of the 319 models every recipe has a different set
  of secondary tokens, so a recipe is selected by its secondaries, not by the model alone.
- The 780 distinct secondary tokens are not catalog entries (2 coincide with an entry name) and not texture
  dictionary items, and are never another recipe's model name. 1,242 of the 2,383 contain the recipe's own
  model name.
- **No light data is stored**: `lights` is empty in all 1,980 recipes and both top-level lists are empty. In
  this build the document is an index from model and variant to recipe with the light positions absent,
  so distant vehicle lights cannot be reconstructed from it.

To reproduce the joins, locate the XML roots in the [common carve catalog](gamedata.md) with
`python -I tools/gamedata_xml.py <common-carve-dir> --json xml.json`. Trim and lowercase the field
text, then match each nonempty `CVehicleDistantLightData/recipeLights/Item` by `vehicleModelName`
and its sorted list of `secondaryRecipes/Item` values against `vehicleName` and the same list on
`CPaletteIndexedRecipeTintsArray` rows at `.//recipeCombinations/Item`. For the carriage check,
expand the light lists to distinct pairs and compare with `CTrainConfigs` rows at `.//carriages/Item`.

The sampled `CPaletteIndexedRecipeTintsArray` document reuses 746 of the 780 distinct distant-light
secondary tokens in its own `secondaryRecipes` field. Of the 1,668 distant-light rows with at least one
token, 1,339 match a palette recipe on both vehicle name and the complete token list, ignoring case and
token order. Replacing each vehicle name with the next of the 319 distinct names from all light rows in sorted order
(wrapping at the end) and keeping its tokens fixed gives 56 matches among the same 1,668 rows.

For a single-token comparison, the distant-light rows give 1,547 distinct (vehicle, secondary token)
pairs. Four match the `modelName` and `modelSetName` fields on the same `CTrainConfigs` carriage row,
after trimming whitespace and ignoring case. These four pairs cover two vehicle models and four model
sets. Rotating the 319 vehicle names as above gives zero matching pairs. Keeping the vehicle fixed and
rotating the 780 distinct tokens in sorted order, with wraparound, gives two matching pairs.

In the 34-row `CVehicleBoneOffsetsArray` document, 0 of the 780 distinct tokens match a whole text or
attribute value case-insensitively. In the 2,447-row document of the same type, 178 of the 780 match
`boneOffsets/Item/name`. All 2,447 rows have empty `vehicleName` fields, so those name matches provide
no same-row vehicle association.
Which physical parts or liveries a token selects, and where the runtime obtains light positions, remain open.

## Skinned GLB export

`tools/skinned_glb.py` writes a skinned GLB from decoded pack components and the shared ped
skeleton:

```
python -I tools/skinned_glb.py out.glb --skeleton <dir> --component <dir> [--component ...]
```

It reads the 8-influence `JOINTS_0/1` and `WEIGHTS_0/1` streams from the graphics buffer, resolves
the drawable's local joint palette through the drawable and ped tables, and emits `skins` and
`inverseBindMatrices` with the ped bind pose. The measured stream offsets and joint mapping are
in [character_skeleton.md](character_skeleton.md#skinned-glb-export). On the 1.0.2.1 package, the
head (8,983 v) and biker jacket (9,818 v) produce one skin with 596 ped joints (703 with the
jacket's `GM_U01_*` helpers); every weight sums to 255 and the rest-pose reconstruction error is
6e-8 m. `python -I tests/test_skinned_glb.py` runs 24 synthetic tests and needs no game files.

`tools/glb_inspect.py` reads such a GLB back without a library, so an export can be checked
before it is opened in a viewer:

```
python -I tools/glb_inspect.py out.glb [--json report.json]
```

It prints the chunk layout and padding, the `counts` of scenes, nodes, meshes, materials, images,
skins and animations, each mesh's primitive, vertex and triangle counts, each material's
`baseColorTexture`, each skin's joint count, each animation's channels and duration, and the
POSITION bounding box (marked `partial` when a POSITION accessor has no `min`/`max`). It reports
what the JSON chunk says — a field the exporter dropped reads `(missing)` — and never infers
geometry from the BIN chunk. Standard library only; no game files.

## What to do next

1. **Done:** [rpf9-explorer PR #1](https://github.com/breadbbbb/rpf9-explorer/pull/1) is merged upstream.
2. **Done:** the pin is `e293917` (was `917e9b9`, originally `48557ea`).
3. **Done:** all ten `MODEL_ARCHIVES` were re-counted and read-back decoded at `e293917` (**382,781
   models**, **0 decode failures**; table above, `levels.rpf` geometry from a full export).
4. Investigate the 71 remaining unrecovered fragment candidates in `levels.rpf` (and the 1
   texture-dictionary candidate in `textures.rpf`) if further upstream recovery is wanted.
5. Keep issue #29 in sync with the `e293917` counts.
3. **Done:** `tools/rpf9dump models` was re-run over the model-bearing archives at `e293917`;
   the six non-`levels.rpf` archives total 39,454 rows (+3,203, almost all recovered fragments).
   `levels.rpf` has the same 343,327 rows in a full export (50,731 fragments, 2,509,947,536
   triangles).
4. Keep the README and issue #29 in sync with the new counts.

### Predict model offsets for placements

Run `python -I tools/world_drawable_lookup.py entities.jsonl archetypes.json names.json`
with the placement rows from `world_index.py --entities-out`, the definitions from
`world_archetypes.py`, and a bounded `rpf9_names.py --types dr,ft` report. The command
prints local scratch JSON. Supply `--manifest path/to/models/manifest.json` for each
existing export whose listed identities should be excluded.

The lookup requires agreement between the placement archetype, the definition's name and
assetName, the resource stem's joaat and its 64-bit placement key. It considers type 1
fragments and type 2 drawables at page offset +16. Multiple offsets for an identity in
the indexed window are marked ambiguous. Even a single hit is a prediction: a stream
can mention another model. Export a window around the predicted offset with
`rpf9dump models <archive> <outdir> --start BYTES --length BYTES`, compare the manifest offset and
model identity and run `world_build.py --placements` with the exported geometry.

Save the lookup JSON and run
`python -I tools/world_drawable_verify.py lookup.json first/models/manifest.json second/models/manifest.json`
to verify single-offset predictions. Supply completed normal model manifests from the
same source archive. The verifier reads cached JSON only. It uses the world builder's
name hashes and counts each identity and its placements once. A matching offset and both
hashes confirm an observation, including dictionary children sharing an offset. The local
report counts repeat exact observations and lists other offsets for the same identity.
Without an exact match, a different name at the offset takes precedence over a matching
identity elsewhere. The coverage flag independently records whether the predicted offset
falls in a supplied scan (start inclusive, end exclusive). A full scan covers all offsets.
A covered prediction with no named evidence remains missing. Geometry loadability is
checked separately by running the normal world builder on confirmed placement rows.

Add `--window-bytes BYTES` to recommend one scan in the name index's archive.
The optional `export_window` field gives the decimal `start` and `length` for
`rpf9dump models`. It maximizes the sum of unmatched placement rows with a
single predicted offset in that index. Each identity contributes its placement
count once. Ambiguous identities contribute zero. For a tie it takes the earliest
last offset, then trims the interval to its first and last included bytes.
An empty selection yields JSON null. Run the lookup separately for each archive.

The budget bounds the half-open range of candidate **starts**. The exporter can
inspect prefix bytes and read model segments beyond the range. The report
predicts coverage. Confirm identities and offsets in the exported manifest and
check the resulting placement count with `world_build.py`.
