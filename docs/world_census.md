# Census of `levels.rpf`

`levels.rpf` is 76,155,065,056 bytes. Its table of contents is encrypted, so the archive cannot be
enumerated by name. `tools/world_census.py` measures the data area directly: it inflates every zlib
stream, validates each `0DRP` / version `0x7a` resource, joins continuation streams, and counts the
map resources and placement entities they contain.

## Result

| Measure | Value |
|---|---:|
| Data area scanned | 76,154,954,352 bytes (start `0x1B070`) |
| Map resources (`kind=map`) | **23,535** |
| Placement entities | **1,670,442** |
| Complete / validated map rows | **23,535 / 23,535** |
| Exact entity counts | **1,670,442 / 1,670,442** |
| Shards complete | **36 / 36** |
| Zlib streams inflated | **1,117,857** |
| Stored stream bytes | 74,877,281,159 |
| Inflated bytes (all streams) | 174,941,755,029 |
| Rejected zlib candidates | 840 |
| Metadata streams | 164,914 |
| Duplicate content groups | 0 |
| Scan time | 290.8 s |

A map row is a resource whose schema includes `CMapData`. `maps_exact` means every entity block
counted against the declared `entity_size` and no entity block was left in a graphics page.
`entities_exact` is the same check summed over all rows. Six entities of the 23,535 maps were split
across a continuation. The stream and byte totals don't depend on the shard size: each shard scans an
8 MiB warm-up window before its start only to find a stream boundary, and the streams there are counted by
the shard they belong to. The stored total therefore equals the accepted stream payload in Coverage below.

### Placement maps vs metadata

The 23,535 rows above are **placement maps**: a `CMapData` root and at least one countable
`CEntityDef` block. The same pass also validates **138,839 `CMapData`-only metadata resources** with
no `CEntityDef` block, plus **26,075 other `0x7a` metadata resources**. Counting every `CMapData`
root regardless of entity schema gives **162,374** resources, but only the 23,535 with entities are
placement maps in this census. Keeping the two classes separate avoids inflating the map count with
map-definition metadata.

| Detail | Count |
|---|---:|
| Placement maps (`CMapData` + `CEntityDef`) | 23,535 |
| `CMapData`-only metadata (no `CEntityDef`) | 138,839 |
| Other `0x7a` metadata | 26,075 |
| Placement maps joining a continuation | 2,868 |
| Placement maps using a raw continuation page | 3 |
| Metadata resources using a raw continuation page | 1 |
| Byte-identical duplicate map payloads | 0 |
| Duplicate 32-bit map-name hashes | 0 |
| Entity blocks in graphics pages | 0 |
| Unresolved or truncated map resources | 0 |


## Coverage

| Byte class | Bytes | Share of data area |
|---|---:|---:|
| Accepted stream payload | 74,877,281,159 | 98.32% |
| Raw graphics pages (`0x6`) | 2,762,232 | 0.004% |
| Raw system continuation | 44,304 | 0.0001% |
| Zero padding | 1,058,806,784 | 1.39% |
| Unread non-zero bytes | 216,059,873 | 0.28% |
| **Accounted** | **76,154,954,352** | **100%** |

The unread bytes are mostly small alignment and metadata gaps between streams. The gap audit found
**23** `0DRP` signatures in unread ranges, but none resolved to a map or metadata resource
(`raw_maps = 0`, `raw_meta = 0`), so the census is not missing a map class. No unaligned zlib stream
that inflates to a `0DRP` header was found in a gap (`missed_zlib_meta = 0`). The 906 rejected
candidates are zlib streams that do not form a complete `0x7a` resource; they are ordinary archive
payload, not counted maps.

## Other archives

The same walk was run over the ten other non-radio RPF9 archives in the package, in 1 GiB shards
(32 shards total) with 12 jobs; `--merge-only` then rebuilt each summary from its shard files. All
shards completed and every byte of the scanned data areas is accounted for. A `CMapData`-only
metadata row has a `CMapData` root but no `CEntityDef` table; the `Other 0x7a meta` column is every
other non-map `0x7a` resource.

| Archive | Size (bytes) | Data span (bytes) | Maps | Entities | `CMapData`-only meta | Other `0x7a` meta rows | Shards | Stream coverage | Accounted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `data.rpf` | 481,544,304 | 481,474,576 | 0 | 0 | 0 | 15,768 | 1/1 | 87.9% | 100% |
| `packs.rpf` | 4,798,213,264 | 4,798,195,520 | 0 | 0 | 0 | 4,678 | 5/5 | 95.5% | 100% |
| `common.rpf` | 6,642,192 | 6,603,920 | 0 | 0 | 0 | 0 | 1/1 | 85.5% | 100% |
| `textures.rpf` | 192,585,024 | 192,583,088 | 0 | 0 | 0 | 0 | 1/1 | 97.8% | 100% |
| `anim.rpf` | 4,407,626,384 | 4,407,608,368 | 0 | 0 | 0 | 22,823 | 5/5 | 95.4% | 100% |
| `navmeshes.rpf` | 6,707,512,496 | 6,707,510,576 | 0 | 0 | 0 | 9,603 | 7/7 | 97.5% | 100% |
| `shaders.rpf` | 411,695,600 | 411,661,232 | 0 | 0 | 0 | 0 | 1/1 | 33.4% | 100% |
| `movies.rpf` | 8,832,481,776 | 8,832,454,448 | 0 | 0 | 0 | 0 | 9/9 | 0.0% | 100% |
| `script.rpf` | 4,358,960 | 4,358,352 | 0 | 0 | 0 | 0 | 1/1 | 59.2% | 100% |
| `appdata0.rpf` | 8,880 | 8,016 | 0 | 0 | 0 | 0 | 1/1 | 0.0% | 100% |
| **Total** | **25,842,668,880** | **25,842,458,096** | **0** | **0** | **0** | **52,872** | **32/32** | **62.2%** | **100%** |

Outside `levels.rpf`, **no `0x7a` resource has a `CMapData` root: 0 placement maps / 0
entities**. The 52,872 metadata rows describe other `0x7a` resources. `tools/world_index.py`
independently parsed the ten archives and also returned 0 maps / 0 entities. `audio.rpf` (radio) was
not scanned.

`movies.rpf` and `appdata0.rpf` have 0% stream coverage because their data areas contain no complete
zlib streams; `movies.rpf` holds Bink video (`KB2` at file offset `0x7000`) and `appdata0.rpf` is an
8,016-byte data area. Their negative result rests on the whole-gap audit, which looked for a raw
`0DRP` + `0x7a` header and for an unaligned zlib stream hiding one. `shaders.rpf` (33.4%) and the
unread remainders of `data`, `packs`, `anim` and `navmeshes` are non-zlib payload. Across the ten
archives the gap audit found **63** `0DRP` + `0x7a` signatures in unread ranges, but none resolved to
a map or metadata resource (`gap_0drp_other = 63`), and no unaligned zlib stream inflated to a
`0DRP` header (`missed_zlib_meta = 0`). All ten are valid `9FPR` archives with complete shards.
`movies.rpf` and `appdata0.rpf` are not supported by the zlib stream walk, so their negative result
is a signature audit of the data area rather than a full stream census; neither archive is malformed.
The other eight have stream censuses with 33–98% accepted payload and audited gaps.

Adding these archives does not change the measured total: **23,535 maps / 1,670,442 entities**, all
in `levels.rpf`. SixNG publishes 30,785 map files / 2,249,490 entities, leaving 7,250 maps / 579,048
entities. The difference is not explained by maps outside `levels.rpf` (zero found here) or by
duplicates (the levels census reports zero byte-identical map payloads and zero duplicate map-name
hashes). `levels.rpf` also contains 138,839 `CMapData`-only metadata resources and 26,075 other `0x7a`
metadata resources.

### The SixNG difference, map by map

A local SixNG `placements` run on the same package writes one file per map, keyed by the resource's
stream offset, which is the `offset` of a census row. Matched by offset against a fresh census at
48ad23f:

| | Maps | Entities |
|---|---:|---:|
| in both | 23,528 | 1,669,203 (equal on every map) |
| SixNG only: `CEntityDef` objects, `CMloArchetypeDef` (`10506455`) objects, no `CMapData` struct | 961 | 540,665 |
| SixNG only: `CEntityDef` objects, one `35abc248` root object, no `CMapData` struct | 6,296 | 39,622 |
| census only: entity table or schema past the first 64 KiB system page | 7 | 1,239 |
| **SixNG total / census total** | **30,785 / 23,535** | **2,249,490 / 1,670,442** |

The two SixNG-only classes are the "other" metadata resources above. SixNG counts a `0x7a` resource
as a map when it has at least one `CEntityDef` object. The census also requires a `CMapData` struct
in the schema, and no resource in either class has one. Neither class is a world placement list:

- The 961 interior resources hold `CMloArchetypeDef` objects (1,358 in total). Their entities lie
  within 173 m of the origin on every axis (median 5.9 m), so the coordinates are interior space.
  This is consistent with interior entity lists that get their world transform from an interior instance in a
  map. SixNG's `interiors` step reads them that way.
- The 6,296 `35abc248` resources each have one root object and entities within 30.4 m of the
  origin (median 0.9 m). SixNG's own source describes them as named, prefab-like entity groups in
  group-local space, placed by nothing in the map data.

The 7 census-only maps have a 65,536-byte first system page, and their `CEntityDef` block or the
schema that the entity table needs lies in the continuation page. In 4 of them the continuation is a
second zlib stream; in 3 it is a raw page. `world_census.assemble` joins that page. SixNG's output
has no file for these 7.

The net difference is 7,257 − 7 = 7,250 maps and 580,287 − 1,239 = 579,048 entities, the whole
published gap. The 580,287 extra entities have no world position of their own: an interior instance
or another placement has to supply the transform.

The census's own join already tracks each continuation's real end (`out["end"] = nxt + mused`), so it
does not have the padding miss that #1084 fixes in `world_index.scan_rpf`; the census numbers above
do not depend on that fix.

## Exactness condition

`summary.json` sets `census_exact` only when all of these hold:

- `complete` is true and every expected shard was written;
- `maps_exact == maps` and `entities_exact == entities`;
- `entity_gfx == 0` and `entity_unresolved == 0`;
- `incomplete == 0` and `invalid == 0`;
- `raw_maps == 0` and `missed_zlib_meta == 0`.

The run above satisfies every condition. `accounted_ratio` is 1.0 because every byte is assigned to
accepted payload, a raw page, zero padding or counted unread data.

## Method

The scan starts at the data-area offset from the `9FPR` header. It walks 16-byte-aligned zlib
candidates in independent shards and:

1. inflates each stream, keeping the first bytes needed to recognize `0DRPz`;
2. validates the little-endian `0x7a` header, struct table, block table and member offsets;
3. follows a contiguous split resource, either another zlib page or a raw system page whose length
   the resource's pointers require;
4. counts `CEntityDef` blocks against the declared entity size, and records `0x6` graphics extents;
5. merges shard rows, checks duplicate names and content hashes, and audits every gap for a missed
   `0DRP` signature or unaligned zlib stream.

Each shard is written to `shard_<index>_<start>_<end>.json`; `summary.json` and `maps.json` are
rebuilt from the shards. Keep the input stable during a run and use one census per output directory.
Each shard records a local archive fingerprint based on file size, timestamps, filesystem identifiers
and up to three 4 KiB samples. If filesystem identifiers are unavailable, the resolved path also
contributes to the hash. Verifying every byte would require reading the whole archive. Copies or metadata changes can require a rescan. `--resume` rescans missing,
older-format and mismatched shards. `--merge-only` reports their ranges as missing and exits 1.
Use `--resume` to refresh format 4 shards in place, since their stream totals can include warm-up streams twice.

```powershell
python -I tools\world_census.py <levels.rpf> --out work\world\census --shard-size 2147483648 --jobs 12
python -I tools\world_census.py <levels.rpf> --out work\world\census --merge-only
python -I tests\test_world_census.py
```

The default shard size is 2 GiB, the default job count is half the logical CPUs, and the scan is
bounded by `--start` / `--limit` when a partial pass is wanted. `--jobs 1` runs the shards in-process
instead of a process pool, which is useful for small synthetic archives and tests.

## Limits

- The census counts resources, not names. It does not decrypt the RPF9 TOC, so it cannot say which
  archive entry owns a map; `docs/world.md` correlates the measured bands with the placement files
  it could carve.
- Only `0x7a` resources are validated. Other resource types contribute to stream and byte counts but
  are not classified.
- A map row proves the resource has a `CMapData` root and a self-consistent entity table. It does not
  prove the map is reachable, loaded, or complete in the runtime.
- Output contains resource offsets, sizes, counts and content hashes only. It is aggregate metadata;
  keep it out of git along with the archive and carved streams.
