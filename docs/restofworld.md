# Rest-of-world and script archives

This page measures the six rest-of-world/script archives in the Xbox 1.0.2.1 package:
`core_restofworld.rpf`, `story000_restofworld.rpf`, `story000_script.rpf`,
`core_restofworld001.rpf`, `appdata0.rpf` and `core_licenses_data.rpf`. The two story archives also
appear in the story report; this page re-measures them with the same corrected carver and adds the
1.21 GiB `core_restofworld`, the three small archives, and the cross-checks. It is a container survey,
not a resource decoder. The RPF9 tables of contents are still encrypted, so every count comes from the
data area or from a fixed header, never from a path or file name.

All six archives were scanned end to end with the corrected carver from PR #90 (`--index-only`, which
advances by the actual stream length). The largest is 1.21 GiB, so no archive needed a bounded window.
Everything below is aggregate; no game payload, asset name or path is stored in this repository.

## Archives and full index

Sizes are exact bytes from the package. `data offset` is the little-endian u32 at file offset 4. The
readable fraction is the carver's `stored / span`; Bink 2 streams count as readable.

| Archive | File size | Data offset | Streams | Stored | Inflated | Zero padding | Unread | Readable | Elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `core_restofworld.rpf` | 1,295,760,752 (1.21 GiB) | 0xAA0 | 2,381 | 1,293,131,069 | 1,355,861,220 | 2,410,297 | 216,666 | 99.8% | 1.9 s |
| `story000_restofworld.rpf` | 198,354,992 (189.17 MiB) | 0xA40 | 631 | 188,163,116 | 255,343,231 | 862,330 | 9,326,922 | 94.9% | 1.3 s |
| `story000_script.rpf` | 78,023,728 (74.41 MiB) | 0x2B0 | 2,632 | 20,408,094 | 74,729,504 | 4,585,959 | 53,028,987 | 26.2% | 3.9 s |
| `core_restofworld001.rpf` | 791,984 | 0x3B0 | 33 | 734,650 | 1,575,059 | 47,232 | 9,158 | 92.9% | 0.1 s |
| `appdata0.rpf` | 8,880 | 0x360 | 0 | 0 | 0 | 0 | 8,016 | 0.0% | 0.1 s |
| `core_licenses_data.rpf` | 13,536 | 0x330 | 3 | 1,074 | 2,004 | 19 | 11,627 | 8.4% | 0.1 s |

The six archives total 1,572,953,872 bytes. The three larger scans are all full-file; the three small
archives are also full, including the 8,016-byte data area of `appdata0`. "Unread" is the non-zero gaps
between streams that did not inflate, not empty space.

## What each archive is

| Archive | Byte-dominant payload | Models | Maps | Scripts |
|---|---|---|---:|---:|---:|
| `core_restofworld` | Bink 2 video, 95.8% of the data span; 958 streams start with a graphics block, 13 more contain one later, and 924 system pages carry model names | 37 decoded in the existing manifest; 433 unique name strings measured here | 0 | 0 |
| `story000_restofworld` | Bink 2 video 60.6% of the data span, then system pages and 101 `0DRP` metadata resources | 1 page with 3 `.#dr` strings; no decoded model | 0 | 0 |
| `story000_script` | `PSIN` metadata, 79.2% of inflated bytes | 0 | 0 | 0 validated container |
| `core_restofworld001` | 23 zero-prefix system pages plus 4 XML content manifests | 0 | 0 | 0 |
| `core_licenses_data` | 3 XML content manifests | 0 | 0 | 0 |
| `appdata0` | no readable stream; 100% unread | 0 | 0 | 0 |

"Models" and "maps" are counts of validated resources, not guesses from a magic. The 37
`core_restofworld` models are the existing model-manifest count in `docs/design.md`; the 924 name-bearing
pages are a larger upper bound on the page population, not 924 decoded models.

## Payload composition

### `core_restofworld.rpf`

| Class | Streams | Stored | Inflated | Share of inflated |
|---|---:|---:|---:|---:|
| Bink 2 (`KB2n`) | 11 | 1,241,514,796 | 1,241,514,796 | 91.6% |
| Streams starting with a `42 40 00 00` block | 958 | 37,556,463 | 77,862,128 | 5.7% |
| zero-prefix system pages | 1,230 | 12,920,541 | 34,278,992 | 2.5% |
| other raw/page starts | 162 | 1,045,600 | 1,960,826 | 0.1% |
| `PSIN` | 6 | 50,402 | 142,056 | <0.1% |
| `0DRP` metadata | 1 | 35,489 | 65,536 | <0.1% |
| `RBF0` | 3 | 3,380 | 23,364 | <0.1% |
| XML | 10 | 4,398 | 13,522 | <0.1% |

- **Video.** The 11 `KB2n` streams are all 1920x1080, with 450 to 9,776 frames; together they are
  1,241,514,796 bytes, or 95.8% of the archive's data span. They are the largest payload by far.
  No Bink 2 stream from this archive is byte-identical to one in `story000_restofworld` (0 common
  SHA-256 over the 11 + 7 stored bodies).
- **Models and pages.** 924 zero-prefix system pages carry 936 model-name strings (883 `.#dr`, 53
  `.#ft`; 433 unique strings), which is the same name-bearing system-page pattern used by the model
  exporter. There are no `.#dd`, `.#td` or `.#txd` strings. The 958 streams starting with a
  `42 40 00 00` block range from 256 bytes to 9.4 MiB each and total 37.6 MB stored / 77.9 MB inflated. The existing
  model manifest reports 37 decoded `core_restofworld` models; the 924 name pages are not all complete
  resources, so this page does not raise that count.
- **Metadata.** One standalone `0DRP` resource at offset 53,899,280 has version 122 and 16 at +0x20.
  `world_index.py` classifies it as `meta`, not a map: 1 resource, 0 maps, 0 entities. Six `PSIN` streams
  follow: five complete six-section chains and one partial `LSCH`+`LENM` stream before an unread gap.
  Joining the standalone resource's continuation stream completes its `0x7a` tables (18 structs, 2 enums,
  18 blocks), while the five `LRSC` payloads are 80-byte `0DRP` + 122 summary records with zero at
  `+0x08..+0x0f`, `0xbb`-filled pointer slots and 1-based `root_block` indices 7, 5, 7, 2 and 2 at `+0x10`
  (within block counts 7, 5, 8, 3 and 2). The `u32` at `PSIN+8` is 257 on four of the six; it is not called
  a version because it varies and the story report already found it unstable.
- **Schemas and text.** The three `RBF0` records name `CMovieSubtitleContainer` (2) and
  `VfxTextureAtlasMetadataInfoMgr` (1). The 10 XML streams have the roots
  `CDataFileMgr__ContentChangeSetDef` (8), `CDataFileMgr__ContentDef` (1) and `SSetupData` (1). They are
  content definitions, not UI layout.
- **Containers.** The file has 11 `9FPR` headers that pass the measured check: the outer archive plus
  10 embedded header candidates; one sits at the data start (2,720), eight are 4 KiB-aligned, and one
  is at 54,154,944. There is no `RSC8` or `RSC7` header anywhere in it. The 216,666 unread bytes include
  the non-zero gaps around those headers.

### `story000_restofworld.rpf`

| Class | Streams | Stored | Inflated | Share of inflated |
|---|---:|---:|---:|---:|
| Bink 2 (`KB2n`) | 7 | 120,187,756 | 120,187,756 | 47.1% |
| zero-prefix system pages | 400 | 57,694,475 | 92,678,320 | 36.3% |
| `0DRP` metadata | 101 | 8,152,633 | 38,632,912 | 15.1% |
| other raw/page starts | 99 | 1,903,036 | 3,436,561 | 1.3% |
| clip streams | 3 | 94,250 | 137,024 | 0.1% |
| `PSIN` | 3 | 40,679 | 104,080 | <0.1% |
| text | 3 | 28,330 | 60,064 | <0.1% |
| `42 40 00 00` resource page | 1 | 44,718 | 58,880 | <0.1% |
| type-hash page | 1 | 11,532 | 31,760 | <0.1% |
| XML | 13 | 5,707 | 15,874 | <0.1% |

- The seven Bink 2 streams are 120,187,756 bytes, 60.6% of the data span: five 720x1280 and two
  1920x1080, with 425 to 3,192 frames. This is a vertical-video/extra-content archive as much as a
  resource archive.
- The 101 `0DRP` streams are all version 122. Their +0x20 counts range from 1 to 37 and sum to 1,662
  pages. `world_index.py` classifies all 101 as `meta`, not maps: 101 resources, 0 maps, 0 entities.
  The 400 zero-prefix system pages account for 92.7 MB inflated. Only one page has model-name strings
  (3 `.#dr` names), so this is not a model archive.
- The three `PSIN` stream starts are 2 single-stream chains and 1 multi-stream container that completes
  across four contiguous continuation streams (321,992 inflated bytes total), giving 3 complete `LRSC`
  payloads alongside the 101 standalone `meta` resources (36 single-stream, 65 joined with one continuation
  stream). The 13 XML streams are again content definitions (11 change-set, 1 content-def, 1 `SSetupData`);
  the three text streams are 59,848, 146 and 70 bytes; the three clip streams have the measured
  `u32 2/3 + 01 01 00 00` header.
- The 19 `9FPR` headers are the outer archive plus 18 nested headers on 4 KiB boundaries. The 9.33 MB
  unread region holds nine `RSC8` headers (version 8001, flags 1, declared size 4,852-13,764, sum
  71,380) between offsets 70,697,336 and 70,877,704, but no `RSC7`. In these nine headers the bytes at
  +0x30 are zero and the first `0x50xxxxxx` pointer is at +0x48; this page does not claim a name field
  at +0x30.

### `story000_script.rpf`

| Class | Streams | Stored | Inflated | Share of inflated |
|---|---:|---:|---:|---:|
| `PSIN` | 2,309 | 14,484,253 | 59,151,216 | 79.2% |
| other raw/page starts | 258 | 4,704,587 | 11,970,376 | 16.0% |
| zero-prefix system pages | 65 | 1,219,254 | 3,607,912 | 4.8% |

`PSIN` dominates; no compiled-script container was found. The measured header and chain are:

```
+0   "PSIN"
+4   big-endian first-section offset (points at LSCH)
+8   u32, 261 on 1,771 streams; 77/265/264 on the next 517; 21 other values
+12  u32, the hash-table byte count
...
record: 4-byte tag + big-endian u32 total size (includes the 8-byte record header)
```

| Tag | Sections | Size range | Median |
|---|---:|---:|---:|
| `LSCH` | 2,160 | 2,440-7,464 | 4,200 |
| `LENM` | 2,156 | 376-5,968 | 728 |
| `LSNS` | 2,118 | 68-17,204 | 1,804 |
| `LGID` | 2,118 | 8 | 8 |
| `LRSC` | 2,118 | 88 | 88 |
| `CHKS` | 2,118 | 20 | 20 |

Of the 2,309 containers, 2,118 have all six sections in a single stream, 38 have `LSCH`+`LENM`, 4 have
`LSCH` only and 149 cross a stream boundary before their first section. An all-offset scan finds 2,275
`0DRP` occurrences in the inflated set, and all 2,275 are `PSIN` `LRSC` records (0 standalone `0x7a`
resources): 2,118 in single-stream containers, 152 in contiguous multi-stream `PSIN` continuations and 5
in `LRSC`+`CHKS` stream tails after an unread gap. The tag/length chain is contiguous and ends at `CHKS`,
which is the same layout measured for `script.rpf`. The `LSNS` payload is high-entropy (5.228-7.990
bits/byte, median 7.888) and none of the 2,118 payloads decompresses with zlib; its block layout (a block table
and 16-byte records) is decoded in [psin.md](psin.md#lsns), the record contents are not. No `SCRP`, `YSC`,
`RSCC`, `RSC8`, `RBF0`, `FSSM` or Bink magic occurs anywhere in the inflated set. One of the 258 raw
streams begins with `LSCH` but has no `PSIN` header; it is a fragment, not a container.

The 65 zero-prefix streams are 3.6 MB inflated. The remaining 258 raw streams are 226 distinct
first-eight-byte classes, 4.7 MB stored / 12.0 MB inflated; no recognised container starts them.
The archive has 5 `9FPR` headers: the outer archive plus four nested headers at 4,096; 98,304;
51,539,968 and 77,819,904. The 53,028,987 unread bytes (68.0% of the data span) contain those nested
archives and their high-entropy payloads. They are not decoded and are not guessed to be bytecode.

### `core_restofworld001.rpf`, `appdata0.rpf` and `core_licenses_data.rpf`

- `core_restofworld001.rpf` has 33 streams: 23 zero-prefix system pages (686,469 stored / 1,507,952
  inflated), 6 other raw streams (46,599 / 63,872) and 4 XML streams (1,582 / 3,235). The XML roots are
  2 `CDataFileMgr__ContentChangeSetDef`, 1 `CDataFileMgr__ContentDef` and 1 `SSetupData`. There is one
  nested `9FPR` at 4,096. There is no `PSIN`, `0DRP`, Bink, model-name string or map. It is a small
  content-manifest/page supplement.
- `appdata0.rpf` has the outer `9FPR` header and an 8,016-byte data area with no complete zlib stream,
  no Bink and no nested `9FPR`. The data area is 2,751 zero bytes and 2,036 printable bytes with an
  entropy of 6.155 bits/byte; no `<?xml`, `RSC8`, `PSIN`, `0DRP`, `KB2` or other known magic occurs.
  It is left opaque rather than labelled a save, a table of contents or anything else.
- `core_licenses_data.rpf` has 3 XML streams, 1,074 stored / 2,004 inflated, with one each of
  `CDataFileMgr__ContentChangeSetDef`, `CDataFileMgr__ContentDef` and `SSetupData`. One nested `9FPR`
  sits at 4,096. The other 11,627 bytes (91.4% of the data span) are unread; no other signature is
  present.

## Signatures actually observed

Only a stream start counts as a container signature. A four-byte magic inside an inflated page or XML
text is recorded as a substring, not as a container.

| Magic | Offset | Validation used | Measured count |
|---|---|---|---:|
| `9FPR` (RPF9) | +0 | data offset u32 at +4, header sizes below EOF | 11 core (10 embedded), 19 story-rest (18), 5 script (4), 2 core001, 2 licenses, 1 appdata |
| zlib `78 01/5e/9c/da` | +0 | inflates cleanly to the stream end | 2,370 core, 624 story-rest, 2,632 script, 33 core001, 3 licenses |
| `KB2[a-z]` (Bink 2) | +0 | size u32 +4, width +0x14, height +0x18, valid bounds | 11 core (all `KB2n`), 7 story-rest (all `KB2n`) |
| `PSIN` | +0 | tag/length chain below EOF, six measured tags | 6 core, 3 story-rest, 2,309 script |
| `0DRP` | +0x10 or +0 | version 122 at magic+4; `0x50xxxxxx` pointers + `root_objects` at +0x20 (standalone `+0x10`), or `0xbb` pointer fill + `root_block` at +0x10 (`LRSC` `+0`) | 1 core, 101 story-rest standalone; plus 5 `LRSC` records in core, 3 in story-rest, 2,275 in script |
| `RBF0` | +0 | printable schema string after the 8-byte header | 3 core |
| `RSC8` | +0 | version 8001 at +4, flags 1 at +12, declared size below EOF | 0 at stream starts; 9 raw headers in the story-rest unread gap |
| `RSC7` | +0 | - | 0 everywhere |
| `42 40 00 00` | +0 | 0x4042 vertex-block header | 958 core stream starts, 1 story-rest stream start |
| zero-prefix page | +0 | first 8 bytes zero | 1,230 core, 400 story-rest, 65 script, 23 core001 |
| XML | +0 | optional UTF-8 BOM + `<?xml` | 10 core, 13 story-rest, 4 core001, 3 licenses |
| text | +0 | printable sample, at least 64 bytes | 3 story-rest |
| type-hash header | +0 | u32 type, u32 8, u32 count, u32 0 | 1 story-rest |
| clip stream | +0 | u32 2/3 + `01 01 00 00` | 3 story-rest |

Not observed as a container anywhere in the six archives: `RSC7`, `SCRP`, `YSC`, `RSCC`, `AWC`,
`ADAT`, `FSSM`, `DDS `, `DXT1`, `DXT3`, `DXT5`, `ATI2`, `BC5`, `BC7` and `TXTR`. Naive byte scans do
produce false positives, which is why the table is restricted to stream starts:

- `ADAT` occurs inside the XML word `METADATA` in two story-rest streams and two core streams.
- `AWC\x00` occurs inside a printable name table in one story-rest zero-prefix page.
- `ATI2` occurs once inside a high-entropy story-rest page, but it is not a validated texture header.
- `KB2` occurs inside two story-rest pages as a four-byte coincidence; only the seven validated
  `KB2n` streams are Bink.

The `0DRP` version is 122 in every validated header. The `RSC8` version is 8001 in all nine raw
headers. No format is inferred beyond these bytes.

## Cross-checks

### `story000_script` against `script.rpf` and `story000.rpf`

The fixed walker finds 1,012 `PSIN` streams in `script.rpf`, of which 1,009 are complete six-section
chains. That matches the 1,012 complete records the `rpf9dump` path reports; the earlier 671-stream
carve figure is not reproduced by the corrected walk. `story000_script` has 2,309 `PSIN` streams and
2,118 complete chains. The two sets are not the same records:

| Archive | `PSIN` | Complete | `LSCH` size | `LENM` size | `LSNS` size | `u32@8` | Common inflated SHA-256 |
|---|---:|---:|---:|---:|---:|---|---:|
| `story000_script.rpf` | 2,309 | 2,118 | 2,440-7,464 | 376-5,968 | 68-17,204 | 261 (1,771), 77/265/264 (517) | - |
| `script.rpf` | 1,012 | 1,009 | 472-552 | 128 | 52-19,412 | 22 (1,010) | 0 |
| `story000.rpf` first 256 MiB | 0 | 0 | - | - | - | - | 0 |
| `story000.rpf` 14-14.25 GiB | 0 | 0 | - | - | - | - | 0 |

The section shapes differ by an order of magnitude, and no inflated stream is byte-identical. The
`story000` samples are 33,315 streams / 2,024,500,480 inflated bytes in the first 256 MiB (16,977
`0DRP` pages) and 2,547 streams / 472,245,792 inflated bytes at 14 GiB (780 `42 40 00 00` pages); neither
has a `PSIN` stream or a hash in common with `story000_script`. That is sample evidence, not a full
20 GB proof. It supports the reading that `story000_script` is its own script-metadata corpus, not a
copy of the base `script.rpf` records; no compiled-script container was found in it. The `LSNS` record
contents remain opaque.

### `core_restofworld` against `packs.rpf` and `levels.rpf`

A full hash pass over `packs.rpf` found 144,364 streams, 4,583,737,693 stored / 9,735,393,187
inflated, matching `docs/world.md`. Against all 2,370 zlib streams in `core_restofworld` (2,278 distinct
hashes; 92 are internal duplicates, including 16 repeated `42 40 00 00` page groups), four hashes are
common with packs and all four are 240-byte zero-prefix pages; there is no common `42 40 00 00` page
and no common model system page. The levels samples are three windows: 20-21 GiB (16,683 streams),
40-41 GiB (14,314) and 70-70.8 GiB (14,445); they share **zero** inflated hashes with core.

| Comparison | Scope | Common inflated streams |
|---|---|---:|
| `core` vs `packs` | full packs, 144,364 streams | 4 (all 240-byte zero pages) |
| `core` vs `levels` 20-21 GiB | 1 GiB window, 16,683 streams | 0 |
| `core` vs `levels` 40-41 GiB | 1 GiB window, 14,314 streams | 0 |
| `core` vs `levels` 70-70.8 GiB | 0.8 GiB window, 14,445 streams | 0 |

The model-name check is consistent: `core` has 433 unique base name strings; the packs sample has
11,991 and the levels sample 4,853, with **zero** exact or case-insensitive base-name intersection.
The samples are derived name lists, not a table of contents, so this does not prove the full archives
never share a model. It does show that the measured core pages are private copies, not the same page
sets as full packs or the three levels windows.

Map placement is absent: all 101 story-rest `0DRP` resources and the one core `0DRP` resource are
`meta`, with 0 `CMapData`/`CEntityDef` structs and 0 entities. The map-placement records remain in
`levels.rpf` (`docs/world.md`), not in these archives.

### Cross-archive duplicates

No inflated stream SHA-256 is shared across `core_restofworld`, `story000_restofworld`,
`story000_script`, `script.rpf`, `core_restofworld001` and `core_licenses_data`. The core and story-rest
Bink sets also have 0 common stored-body SHA-256. These are distinct corpora, not copies at different
block boundaries.

### Standalone `0DRP`/`0x7a` versus `PSIN` `LRSC` (ASSET-156)

Both placements serialize the same 80-byte (`0x50`-byte) `0DRP` + `u32 122` (`0x7a`) header frame in two
mutually exclusive forms:

| Frame offset (`magic+`) | Standalone `0x7a` resource (`stream +0x10`, `0x60` B header) | `PSIN` `LRSC` record (`LRSC +0x08`, `88` B section) |
|---|---|---|
| prefix (`stream +0x00..+0x0f`) | `8` zero bytes + `0x50000000`-based page-map pointer at `+0x08` | none (`"LRSC"` + big-endian `88` precedes `magic+0x00`) |
| `+0x00..+0x07` | `"0DRP"` + `u32 122` (`0x7a`) | `"0DRP"` + `u32 122` (`0x7a`) |
| `+0x08..+0x0f` | `8` zero bytes (`stream +0x18..+0x1f`) | `8` zero bytes (`LRSC` payload `+0x0c` is always `0`) |
| `+0x10..+0x17` | `u32 root_objects` (`stream +0x20`), `u32 0` | `u32 root_block` (`1 <= root_block <= blocks`), `u32 0` |
| `+0x18..+0x37` | four `u64` table slots (`+0x28` structs, `+0x30` enums, `+0x38` blocks, `+0x40`): `0` or `0x50000000`-based virtual pointers | `32` bytes of `0xbb` (`0xbbbbbbbbbbbbbbbb` x 4); schemas and blocks live in `LSCH`, `LENM` and `PSIN+0x08` |
| `+0x38..+0x3f` | four `u16`: `(structs, enums, blocks, unknown_4e)` (`stream +0x48`) | four `u16`: `(structs, enums, blocks, 0)` matching `(len(LSCH), len(LENM), len(block_table), 0)` |
| `+0x40..+0x47` | `0` or `0x50000000`-based virtual pointer (`stream +0x50`) | `8` bytes of `0xbb` (`0xbbbbbbbbbbbbbbbb`) |
| `+0x48..+0x4f` | `8` zero bytes (`stream +0x58..+0x5f`) | `8` zero bytes, followed by `CHKS` (`20` bytes) |

`tools/odrp_7a.py` validates both placements, joins contiguous 16-byte-aligned continuation streams when a
standalone `0x7a` resource or `PSIN` container crosses a stream boundary, and detects unjoined `PSIN` tail
streams ending with `LRSC` (`88` bytes) + `CHKS` (`20` bytes) after an unread gap. Across the six archives
in this note plus `script.rpf`, `data.rpf` and `shared_minigame000.rpf`, all 22,235 records validate as
complete (`0` incomplete):

| Archive | Records | Standalone (`meta` / `map`) | Standalone joined | `PSIN` `LRSC` | `LRSC` joined / `psin_tail` |
|---|---:|---:|---:|---:|---:|
| `core_restofworld.rpf` | 6 | 1 (`1` / `0`) | 1 | 5 | 0 / 0 |
| `story000_restofworld.rpf` | 104 | 101 (`101` / `0`) | 65 | 3 | 1 / 0 |
| `story000_script.rpf` | 2,275 | 0 | 0 | 2,275 | 152 / 5 |
| `core_restofworld001.rpf`, `appdata0.rpf`, `core_licenses_data.rpf` | 0 | 0 | 0 | 0 | 0 / 0 |
| `script.rpf` (cross-check) | 1,012 | 0 | 0 | 1,012 | 3 / 0 |
| `data.rpf` (cross-check) | 17,978 | 15,768 (`15,768` / `0`) | 159 | 2,210 | 118 / 11 |
| `shared_minigame000.rpf` (cross-check) | 860 | 787 (`685` / `102`, 20,339 entities) | 13 | 73 | 3 / 0 |

```powershell
python -I tools/odrp_7a.py <archive.rpf> <work>\odrp.json
python -I tests/test_odrp_7a.py
```

## Readable versus opaque

Readable: the RPF9 header and data offset, zlib streams, Bink 2 headers, XML content definitions,
`PSIN` tag/length chains, `0DRP` version and count fields, `RBF0` schema strings, and model-name
strings in system pages.

**Vertex blocks on graphics pages (ASSET-57).** In core rest-of-world, 958 inflated streams
start with the target header and another 13 reach it after earlier chunks. In all 971, the
first 23 little-endian words contain tagged stream locations: for a populated slot,
the low byte is the tag and twice the upper 24 bits gives the byte offset from the block
start. Slot 0 starts at +0x80. Sorting the populated slots by offset gives positive spans.

The independent check is the meshlet group following the streams. To locate a candidate
group, take the greatest common divisor of the adjacent stream spans, try one through four
times that unit for the final span, then round up to 16 bytes across zero padding.
In 553 blocks the following group fits and validates: meshlet base offsets match the running
vertex-reference and triangle counts, and every packed triangle identifies its meshlet and
uses local indices within range. In 535 of these blocks the referenced indices cover
every integer from zero through the maximum, and each adjacent span divides exactly by
the size of that range. For all 535 the measured span per referenced vertex is **8 bytes
for tag 0x42** and **4 bytes for tags 0x21, 0x25, 0x26, 0x27 and 0x28** wherever
present with a measured successor. This supports separate vertex streams. It leaves the
component types, final stream width and blocks without an independent count unresolved.

Across all 22 accessible RPF9 archives, a full data-area scan using the four zlib
headers from #752 accepted 2,241,411 streams at checksum-validated EOF. There were
309,007 target page starts and 41,342 streams in which a chunk walk reached the
target later. Each stream contributes at most its first target block. Of these
350,349 blocks, 350,348 passed the slot bounds and positive-span checks, 108,342
had a validated following meshlet group, and 62,207 also had dense references and
integral span quotients. The recurring widths below account for 61,896 of those
blocks. Each row counts blocks with a measured successor for that tag.

| Slot tag | Bytes per referenced vertex | Supporting blocks |
|---|---:|---:|
| 0x42 | 8 | 61,896 |
| 0x21 | 4 | 59,308 |
| 0x25 | 4 | 21,837 |
| 0x26 | 4 | 60,480 |
| 0x27 | 4 | 18,030 |
| 0x28 | 4 | 31,164 |
| 0x8c | 16 | 3,391 |

Of the 61,896, core rest-of-world contributes 535, minigame 263, full story 110,
story rest-of-world 1 and the other archives 60,987. Full story alone has 5,589
target starts and 1,523 continuations. The 311 other integral cases give different
quotients. A dense range measures the vertices used by the following group and can
miss vertices used elsewhere. The outlier analysis below checks those counts. The survey counts
stream occurrences, including duplicates, and excludes opaque data and encrypted
directory contents.

**Integral outliers.** Comparing each measured adjacent span with the following
group's dense reference count times the recurring width for its tag gives a
single ratio per block: all measured tags in that block scale together. The
61,896 controls have ratio 1. Among the 311 outliers (62 distinct sets of
tag/quotient pairs), 193 have ratio 1/4, 59 have 1/2, eight have 3/4 and
51 have a ratio above 1. These counts include every integral case once.

For the 260 ratios below 1, the first stream alone cannot hold all vertices
referenced by the following group at eight bytes each. Missing references or
positive padding cannot explain that shortfall. For the 51 above 1, a group
using only part of a block is possible. In both directions, a common scale
across tags also fits a different vertex count. It does not establish a new
encoding for any tag. The final stream has no measured successor and is
excluded from this comparison.

Dividing each adjacent span by its tag's recurring width gives the same
positive integer count within each of the 311 blocks. A scan of every other
16-byte-aligned group header in the same inflated page, using the meshlet
checks above, found a second group whose references cover exactly that count
in **230 blocks** (222 of the 260 short cases and eight of the 51 long cases).
This independently corroborates the proposed count on those pages. It leaves
**81 blocks** without a second count match: 48 have another validated group
but no match, and 33 have no other validated group on the page. Even an exact
match does not identify which block a group draws from.

For each of those 81, scan the immediately preceding and following
checksum-validated zlib streams in the same archive, in stored order, without
skipping intervening stream types. Use the same aligned group validation and
require dense references from zero through the proposed count minus one.
All 81 have an exact match on one side only: 66 in the preceding stream and
15 in the following stream. As a distance check, repeating the test two streams
away on either side yields matches for four of the same 81.

Together, 230 same-stream and 81 adjacent-stream matches corroborate the
recurring-width count for **all 311 outliers**. The 61,896 original controls
remain a separate count because their immediately following group supplies
the matching range. No outlier had a control with the same header and bytes
through the last measured span. Applying the recurring widths to the final
slot as a separate check lets a chunk walk consume the whole stream in 219
outlier cases. Only six of those walks have a one-to-one count match between
all blocks and dense groups, with a unique match for the target block.
Count agreement therefore supports the widths across the outlier set while
leaving the draw-to-block mapping unresolved.

Rounding each measured stream's size up to 16 bytes using the following
group's count and the recurring widths fits zero of the 311 blocks. Padding
after the final stream cannot change the earlier spans. In three blocks, the
four candidate final lengths lead to more than one distinct validated group
offset. The original first-match rule is retained for the counts above.
The page-only survey supplies no reason to introduce an alternate per-tag format.

**Draw ownership from system metadata.** A geometry record supplies two independent
resource-relative addresses. Following the pinned `RageModelDecoder.ReadMesh` layout,
geometry +8 points to the vertex buffer and geometry +64 points to the meshlet
descriptor, whose first pointer selects the index buffer. Each buffer's +24 pointer
addresses the complete inflated graphics segment, including when its target is in
another compressed stream. The vertex layout supplies the count separately. The
descriptor supplies meshlet, triangle and reference counts that must agree with the
pointed-to group's header. This check also validates the packed triangles, vertex
reference bounds and padded index-buffer length.

The original check found one distinct validated group address for each of the
311 target blocks. The regular drawable LOD lists reached 303. For the other
eight, a search of aligned system records found the same pointer layout and a
geometry list containing the record. The layout count and the group's dense range
agreed in all 311. This includes the six uniquely paired page walks and seven
walks with a repeated target block count.

**The eight auxiliary root paths (ASSET-273).** An independent scan of all 22
archives reproduced 311 outliers and verified all 311 draw bindings. The regular
drawable lists reached 264 blocks through LOD 0 and another 39 through LODs 1–3.
The eight records reached through auxiliary paths each have a verified geometry
record and graphics-group address:

- **Seven standalone drawables:** follow the root drawable's pointer at `+0xD0`.
  In the auxiliary header, `+0x30` points to a table of system pointers and
  `+0x38` holds its 16-bit count, which matched capacity in these cases. Each
  relevant table entry points to a record whose `+0x68` field leads to the
  structure with the geometry list at `+0x50`.
- **One fragment:** start with the fragment root's pointer at `+0x90`, then
  follow pointer fields at `+0x130`, `+0x20` and `+0x70`. In that table, entry 1
  (zero-based) reaches the record whose `+0x68` pointer leads to the structure
  with the geometry list at `+0x50`. These offsets are one measured route.

One regular LOD 0 binding needs a compressed final system page that inflates to
48 bytes. The diagnostic scan's 64-byte minimum excluded that page. When the
first page's map declares the final system continuation, the archive diagnostic
accepts a nonempty short page at the next 16-byte boundary with zero padding
and a valid zlib checksum. That restores the normal draw binding. Both the
diagnostic and the pinned model reader recover the containing dictionary's 11
children.

`rpf9_carve.py` and the carved-folder model join use the same short final-system-page
exception, so a carve retains that page and can reproduce the archive join. Unrelated
streams still need at least 64 inflated bytes. When a carve has an `index.json`, the folder join uses its file list, offsets,
compressed sizes and preceding bytes to check root alignment and zero padding just
as the archive join does. An omitted stream between pages therefore interrupts the
join. Extra files are ignored, and missing indexed files or size mismatches are errors.
The index describes the original archive and cannot authenticate a same-size replacement
payload. Without an index, folder mode still uses file order and inflated file sizes
and can join across bytes the carver omitted.
The folder reader follows the carver's file types: `.bin` contains an inflated zlib
payload and `.bk2` contains raw video. A `KB2` prefix inside a `.bin` remains resource
data, while a raw `.bk2` interrupts the model's continuation run.

| Pointed-to group start relative to the target stream (original survey) | Blocks |
|---|---:|
| Preceding stream | 68 |
| Same stream | 228 |
| Following stream | 15 |

Two of the 230 same-stream count matches have their descriptor-selected group
in the preceding stream. A count-only association would therefore choose a
different group even with an exact match in hand.

For an already separated system/graphics resource pair, the following command
walks the primary LOD from the drawable offset reported by the model manifest.
Use `--lod 1`, `--lod 2` or `--lod 3` with the same drawable offset to select a
lower LOD, or use `--geometry-offset` for a known individual geometry record.
Both offsets are relative to the start of the system segment. Unsupported or
inconsistent bindings are reported as unresolved.

```powershell
python -I tools/meshlet_page.py <graphics-segment> --draw-system <system-segment> --drawable-offset <offset>
```

**System-to-graphics model-page join (ASSET-157).** `tools/restofworld_model_join.py` joins model system pages to
their owning graphics pages directly from an RPF9 archive or a carved stream directory without the encrypted TOC.
The earlier 37-model figure came from `rpf9dump` at `48557ea`, before `pgDictionary<rmcDrawable>`, optional-pointer
fragment and zero-radius drawable support (`e293917`; see [models.md](models.md)). Four in-band structural rules
pair every model resource unambiguously:

1. **Top-level framing and root header.** In an RPF9 data area, every top-level resource starts on a 4,096-byte
   boundary plus `0x10` (`(offset & 0xFFF) == 0x10`) preceded by 16 zero bytes, with `u64@+0x00 == 0` and a
   `0x50000000`-based system pointer at `+0x08` (`+0x10..+0x13 != "0DRP"`). Root headers separate into standalone
   `rmcDrawable`, `fragType` (`fragment`), `pgDictionary<rmcDrawable>` (`drawable_dict`), standalone
   `pgDictionary<grcTexture>` (`140` in `core_restofworld.rpf`: `115` `1+1`, `25` `1+2`), and `44` other non-model
   system resources.
2. **In-band `datResourceMap` page counts.** At `page_map = u64@+0x08 - 0x50000000`, `u32@+0x00 == 0`,
   `u32@+0x04 == 0`, `u8@+0x08 = system_pages`, `u8@+0x09 = graphics_pages`, `u16@+0x0A == 0`, and `u32@+0x0C == 0`,
   followed at `+0x10` by `system_pages + graphics_pages` zero-initialised `u64` runtime page-base slots (`924 / 924`
   in `core_restofworld.rpf`, `1 / 1` in `story000_restofworld.rpf`). When `system_pages > 1`, `page_map` lies in the
   final system stream after concatenating preceding 64 KiB system streams.
3. **Contiguous 16-byte-padded runs and 64 KiB blocking.** Each resource owns `system_pages` contiguous zlib streams
   followed immediately by `graphics_pages` contiguous zlib streams, with `0..15` zero padding bytes at every
   internal boundary (`924 / 924`). Every non-final system stream (`2 / 2`) and every non-final graphics stream
   (`96 / 96`) inflates to an exact multiple of 64 KiB (`(size & 0xFFFF) == 0`); the trailing `graphics_pages == 2`
   continuation stream ranges from `656` to `63,520` bytes.
4. **Cross-page LOD, vertex-block and meshlet validation.** Across all four LOD slots (`+0x50..+0x68`) of every
   drawable (`583` 1-LOD, `287` 2-LOD, `25` 3-LOD, `41` 4-LOD), `vb+0x18` (`0x60000000`-based) lands on a `0x42`
   (`(8, 16)`) or `0x63` (`(12, 6)`) vertex block whose byte span covers `u32@(fvf+0x138) >> 10` vertices, and
   `geo+0x40` lands on a meshlet index buffer whose cluster count (`countFlags & 0xFFFF` when bone-palette bit
   `0x40000000` is set, else `countFlags & 0x7FFFFFFF`), 8-bit per-cluster triangle counts (`((packed >> 7) & 0xFF) + 1`),
   running offsets, packed triangle indices and vertex references all match the system descriptor and reach into the
   final declared graphics page (`924 / 924`).

| `core_restofworld.rpf` model class | Resources | `1+1` | `1+2` | `2+2` | System streams | Graphics streams | Drawables | Primary-LOD tris | All-LOD tris |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `drawable_dict` (`795` x 1 child, `1` x 13) | 796 | 740 | 56 | 0 | 796 | 852 | 808 | 743,663 | 1,158,313 |
| Standalone `drawable` | 75 | 58 | 16 | 1 | 76 | 93 | 75 | 312,068 | 369,031 |
| `fragment` | 53 | 30 | 22 | 1 | 54 | 75 | 53 | 183,865 | 301,456 |
| **Total joined (`0` unjoined)** | **924** | **828** | **94** | **2** | **926** | **1,020** | **936** | **1,239,596** | **1,828,800** |

The join accounts for both page populations in full:

- **System pages (`926` streams, `924` name-bearing):** all `922` single-system-page resources carry their `.#dr`/`.#ft`
  name strings in `sys_page0`, while the two `2+2` resources (`1` standalone drawable `[131,072, 3,728]` and `1`
  fragment `[65,536, 9,680]`) hold root and geometry structures in a nameless 64 KiB-multiple `sys_page0` and place
  their name strings and `datResourceMap` tails in `sys_page1` (`922 + 2 = 924` name-bearing zero-prefix pages;
  `936` drawables = `883` `.#dr` + `53` `.#ft`, `849` bone-bound, `75` with an empty embedded texture dictionary).
- **Graphics pages (`1,020` streams):** all `924` `gfx_page0` streams start with `42 40 00 00`, and the `96`
  `gfx_page1` continuation streams split into `34` `42 40 00 00` starts plus `62` meshlet-led starts (`924 + 34 = 958`,
  claiming **958 / 958** `42 40 00 00`-start streams in the archive). All **13 / 13** streams that reach a
  `42 40 00 00` vertex block after earlier meshlet groups (`8` standalone drawables, `5` fragments) sit inside those
  `62` meshlet-led `gfx_page1` continuations (`47` `gfx_page1` streams carry a pointed-to `42 40 00 00` block; `49`
  hold meshlet groups only). Primary-LOD geometry totals (`1,177` geometries, `1,001,667` vertices, `1,239,596`
  triangles) match [models.md](models.md) to the triangle; all LODs total `2,021` geometries and `1,598,782` vertices.
- **Other rest-of-world archives:** `story000_restofworld.rpf` joins **1 / 1** model resource (`drawable_dict`, `1+1`,
  `3` child drawables, `3` geometries, `1,808` vertices, `2,620` triangles, claiming **1 / 1** `42 40 00 00`-start
  stream); `core_restofworld001.rpf` has `0` model candidates and `0` `42 40 00 00` streams.

```powershell
python -I tools/restofworld_model_join.py <archive.rpf> <work>\model_join.json
python -I tools/restofworld_model_join.py <work>\carve\<name> <work>\model_join.csv
python -I tests/test_restofworld_model_join.py -v
```

**Component types (full corpus).** `tools/vertex_types.py` (at `1aef69e`, with the running-sum fix to
`group_edges`) was run over the seven archives that hold target blocks: `data`, `packs`, `levels`, `story000`,
`core_restofworld`, `shared_minigame000` and `story000_restofworld`. It measures a block only when the
following meshlet group covers it densely, so it uses **60,442 blocks** (138,929,728 vertices,
282,704,362 triangle edges) out of the 309,007 target streams; the rest are skipped as "no group" (225,795)
or "not dense" (22,769), plus one single-stream block. The run before that fix gave the same block and vertex counts and 3,300 fewer
edges, so none of the conclusions below depend on it.

| Measure | Result |
|---|---|
| `+0x5C` flags word | `0x200` in all 60,442 |
| `+0x7C` float | 4.0 in all 60,442 |
| `+0x6C` axis word | 2: 30,017, 1: 16,718, 0: 13,707 |
| Position header (bias, scale) finite, scale >= 0 | 60,442 / 60,442 |
| Every triangle edge no longer than the block extent | 48,150 / 60,442 |

**Slots 6/7 (tag `0x26`) and 8/9 (tag `0x25`) are skin weights and joint indices.** Where slot 8 holds tag
`0x25`, each `0x25` slot comes with a `0x26` slot two below it (slot 6 with 8, slot 7 with 9), and no block in
the sample has one without the other. Read as four unorm8 weights per `0x26` slot and four joint indices per
`0x25` slot, `skin_checks` in `tools/vertex_types.py` counts, per vertex over all weight slots
(core_restofworld, first 500 streams: 612 blocks, 212,564 vertices; 454 blocks with two pairs, 158 with one):

| Check | Vertices |
| --- | --- |
| Weights sum to 255 | 203,243 (95.6%) |
| Weights sum to 247-255 | 204,758 (96.3%) |
| Weights in descending order | 211,902 (99.7%) |
| No joint repeated among the non-zero weights | 212,564 (100%) |
| Every zero weight has joint index 0 | 156,808 (73.8%) |
| Control: slot 1 colour bytes sum to 255 | 0 |

The `0x25` bytes are small: the highest joint used by a non-zero weight is below 16 in 488 blocks, 16-31 in
117 and 48-63 in 7. So the indices point into a short per-block (or per-mesh) joint list, not a whole
skeleton; which list was not traced here.

Of the 9,321 core_restofworld vertices whose weights do not sum to 255, the twelve most common sums (254
down to 233) all come from one-pair blocks: a four-influence vertex whose weights fall short, as
`tools/skin_streams.py` also finds for one packed four-influence population of the drawable skin streams.
levels (first 400 streams) has 9 skinned blocks (3,910 vertices), all one pair, and every vertex there
passes: one weight of 255 on joint 0. The colour control never sums to 255 in either sample.

Slot usage (block counts): slot 0 tag `0x42` in all 60,442; slot 2 `0x21` in 57,925; slot 1 `0x26` in
57,873; slot 3 `0x28` in 49,294 and slot 4 `0x28` in 29,379; then slot 12 `0x26` (25,239), slot 6 `0x26`
(24,876), slot 8 `0x25` (21,519), slot 13 `0x26` (17,928), slot 11 `0x27` (11,120) and 21 rarer slot/tag
pairs, including `0x84` and `0x4b` in slot 17 (5,452 and 3,714).

**What the skinned mesh shaders read from slots 6, 8 and 17 (shader reading; slot 17 is only partly
checked on the data).** The 363 mesh shaders that contain a `BUF_FMT_16_16_16_16_UINT` load take each
stream's type from the low four bits of a per-slot descriptor word in a constant buffer (one dword per
slot: `+0x18` is slot 6, `+0x20` slot 8, `+0x44` slot 17; stride `((w >> 3) & 60)` bytes). They choose
the load from that type:

| Slot | Type | Load |
| --- | --- | --- |
| 6 | 6 (tag `0x26`) | `tbuffer_load_format_xyzw ... BUF_FMT_8_8_8_8_UNORM` |
| 6 | other | `buffer_load_dwordx4` (four raw dwords) |
| 8 | 5 (tag `0x25`) | `tbuffer_load_format_xyzw ... BUF_FMT_8_8_8_8_UINT` |
| 8 | other (type 11 is tag `0x4b`) | `tbuffer_load_format_xyzw ... BUF_FMT_16_16_16_16_UINT` |

198 of the 363 also read slot 17 with `buffer_load_dwordx2` (the first 8 bytes):

- u16 0 bit 15 set: that vertex skips the step.
- Otherwise u16 0 is an element index, checked against a count from a table.
- That element's entry holds eight u16 point indices.
- u16 1-3 are three coordinates, each `v / 65535` (`v_mul_f32 0x37800080`).
- The shader fetches the eight points and blends them with the trilinear weights `(1-a)(1-b)(1-c)` ... `abc`.

That is a deformation-lattice binding.

On the data (core_restofworld, first 500 streams), slot 17's width follows the skin influence count:

- All 338 blocks with tag `0x84` (16 bytes) have two weight/index pairs (slots 6-9, eight influences).
- All 121 blocks with `0x4b` (8 bytes) have one (slots 6 and 8, four influences).
- Bit 15 of u16 0 is set on 20,898 of 186,727 `0x84` vertices (11.2%) and 5,973 of 12,336 `0x4b` vertices
  (48.4%).
- For `0x84`, bit 15 of the second 8 bytes' first u16 is set on 117,067 (62.7%). Those 8 bytes may be a
  second binding, but no shader read of them was traced.

Element indices are mostly unique per vertex (115,101 distinct non-flagged values over the 186,727 `0x84`
vertices), so few elements are shared by enough vertices to test the coordinates. In the 39 elements with
8 or more vertices, positions are an affine function of the three coordinates: the median rms residual is
1.6% of the block extent, against 25% with the coordinates shuffled within the element. That is consistent
with the reading, but too few elements to confirm it. In these samples no block puts tag `0x4b` in
slot 8.

**Tag `0x28` is two signed 16-bit values.** Over the triangle edges of the slot 3 stream, a difference of more
than 0x8000 between neighbouring vertices occurs 1,597,530 times when the values are read as unsigned and
18,791 times when they are read as signed (85 times fewer); slot 4 gives 1,680,055 against 5,570. Read as
half floats, 1,391,311 slot 3 values are NaN or infinite. So the stream is a pair of `int16`s, continuous across
the surface, which fits the `x / 32767 * f32@0x7C` texture-coordinate reading.

**Tag `0x27` is a second tangent frame: a unit tangent along `u` of another `0x28` stream, with a sign.**
The mesh shaders fetch it with `tbuffer_load_format_xyzw ... BUF_FMT_2_10_10_10_UNORM` (one of the
four typed vertex formats they use, in 2,162 of the 9,282 mesh shaders) and map every component by
`v * 2 - 1`. They transform xyz by the same matrix as the normal and cross it with the normal, scaled by
w. `tangent_0x27` in `tools/vertex_types.py` is that read: xyz = bits 0-9, 10-19, 20-29 over 1023, and
w = bits 30-31 over 3. Per `0x27` slot, `tangent27_checks` counts unit length over all vertices; over
the sampled triangles it counts vectors perpendicular to the face normal (`|cos| < 0.2`) and, for each
`0x28` stream, vectors within `cos > 0.8` of `dP/du` and the w side:

| Sample | Slot | Vertices | Unit | w = 3 / 0 | Perpendicular to face | Best `0x28` stream: along u | w > 0 exactly when `cross(face normal, xyz)` runs along `-v` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| levels (400 streams) | 11 | 637,604 | 100% | 634,431 / 3,173 | 86.1% | slot 4: 96.7% (slot 3: 69.5%) | 99.8% |
| core_restofworld (500 streams) | 18 | 182,208 | 100% | 180,282 / 1,926 | 40.3% | slot 14: 79.3% (slot 4: 73.6%, slot 3: 21.0%) | 99.2% |
| core_restofworld | 19 | 12,679 | 100% | 12,445 / 234 | 61.7% | slot 14: 69.0% (slot 3: 54.1%) | 98.5% |

The w side is counted among the along-u vertices of that stream. The w field only ever holds 0 or 3 in
these samples. Against the vertex normal of tag `0x21` (read as the mesh shaders read it) instead of the
face normal, the slot-18 vectors are perpendicular (`|cos| < 0.1`) for 174,942 of 182,208 (96.0%) and the
slot-19 and levels slot-11 vectors for all of them. Where the face-normal column is low, the face normals
disagree with the vertex normals, as for tag `0x21`. The tangent matches the `0x21` tangent (`cos > 0.99`)
for only 10-17% of vertices, so it is a different frame, the one for a second texture-coordinate stream.

**Tag `0x26` (slot 1) is four bytes with an alpha-like byte 3.** Byte 3 is 255 on 62.4% of the 137.9 million
vertices and 0 on 17.2%; each of bytes 0-2 is constant within a block in 36,544 to 38,854 blocks. The bytes
change smoothly over edges (block-weighted median edge/random difference ratio 0.18 to 0.27), as a colour or
blend weight would. As a packed normal it scores no better than random directions (below).

**Tag `0x21` (slot 2) is an octahedral normal whose width depends on bit 31.** The face-normal check: for each
block the tool reads 256 sampled triangles and counts the vertices whose decoded vector is within `|cos| > 0.8`
of the face normal; random directions score 0.2. None of the 19 fixed decodings scores well over the whole
corpus (best block-weighted medians: unorm10 xyz 0.479, octahedral unorm8 bytes 0-1 0.331, the rest 0.13 to
0.31), because the stream mixes two encodings. The mesh shaders show which one each vertex uses (6,285 of the
9,282 mesh shaders contain the octahedral `v_sub_f32 ..., 1.0, |x|`): after
`buffer_load_dword` of the vertex word they branch on its sign. With bit 31 set they take x = bits 0-9 and
y = bits 10-19 (`v_and_b32 0x3ff`, `v_bfe_u32 ..., 10, 10`), map each by `v * 2/1023 - 1`, and unfold the
octahedron (`z = 1 - |x| - |y|`, the `-z` clamp added back with each sign, then `v_rsq_f32`). With bit 31
clear they take bytes 0 and 1 (`v_cvt_f32_ubyte0/1`), map each by `v / 127.5 - 1`, and unfold the same way.
`normal_0x21` in `tools/vertex_types.py` is that decode (candidate `oct_bit31_10_or_8`).

Scored separately on the first streams of two archives (core_restofworld: 120 streams, 143 blocks;
levels: 200 streams, 187 blocks), each class matches the face normals only under its own branch:

| Vertices | Decode | core_restofworld p25 / median / p90 | levels p25 / median / p90 |
| --- | --- | --- | --- |
| bit 31 set | 10-bit octahedral | 0.53 / 0.73 / 0.98 | 0.89 / 1.00 / 1.00 |
| bit 31 set | 8-bit octahedral | 0.19 / 0.20 / 0.30 | 0.12 / 0.23 / 0.57 |
| bit 31 clear | 8-bit octahedral | 0.44 / 0.61 / 1.00 | 0.67 / 1.00 / 1.00 |
| bit 31 clear | 10-bit octahedral | 0.00 / 0.16 / 0.26 | 0.00 / 0.00 / 0.75 |
| all | both, by bit 31 | 0.53 / 0.63 / 0.96 | 0.89 / 1.00 / 1.00 |

Bit 31 is set on 48,046 of the 60,600 sampled core_restofworld vertices (79.3%). Model axis order is the one
`+0x6C` gives for positions; the other five axis orders and sign flips of the decoded vector score 0.27 to
0.36 there. Some core_restofworld blocks score lower under the matching decode than levels blocks do; this
check does not say why.

**The rest of the `0x21` word is the tangent frame.** The same mesh shaders go on to build a tangent.
With bit 31 set, bits 20-29 are an angle, `v * 2pi/1023 - pi` (`v_fma_f32` by 2pi/1023 and -pi, then
`v_sin_f32`/`v_cos_f32`), that turns a basis made from the normal alone (`s` = sign of z,
`a = 1/(s + z)`, `b1 = (s x^2 a - 1, s x y a, s x)`, `b2 = (x y a, y^2 a - s, y)`), giving
`tangent = b1 cos - b2 sin`. With bit 31 clear, bits 16-22 and 23-29 are a 7-bit octahedral tangent
(`v_bfe_u32 ..., 16, 7` and `23, 7`, each `v / 63.5 - 1`). Bit 30 (`v_lshrrev_b32 30` then `& 1`) is the
bitangent sign. Of the 6,254 mesh shaders with the bit-31 branch, 5,196 read bits 20-29, 5,191 read the
two 7-bit fields and 528 read bit 30. `tangent_0x21` in `tools/vertex_types.py` is that decode.

The tool checks it against the first `0x28` stream (texture coordinates): per sampled triangle it projects
`dP/du` into each vertex's tangent plane and counts tangents within `cos > 0.8` of it ("along u"), the same
tangent turned by a random angle about the normal (chance level), and, of the along-u vertices, how often
bit 30 matches the side of `dP/dv` that `cross(normal, tangent)` points to:

| Archive sample | Form | Vertices checked | Along u | Random angle | Bit 30 = bitangent side |
| --- | --- | --- | --- | --- | --- |
| levels (187 blocks) | 10-bit | 59,448 | 98.0% | 23.2% | 99.4% |
| levels | 8-bit | 411 | 83.2% | 22.6% | 98.8% |
| core_restofworld (143 blocks) | 10-bit | 52,153 | 86.3% | 22.3% | 97.8% |
| core_restofworld | 8-bit | 6,992 | 56.4% | 15.8% | 92.6% |

Bit 30 set means `cross(normal, tangent)` runs along `+v`. The shaders turn a set bit 30 into the factor
`-1.0` (`v_cndmask_b32 ..., 1.0, -1.0`), so their bitangent is that factor times their own cross product;
which operand order they use was not traced here. The 8-bit form in core_restofworld fits least;
this check does not say why.

An independent `data.rpf` sample says the tangent decode itself is not the weak part. On one mesh with all
809 triangles usable, the decoded 8-bit vector pointed toward increasing UV0 U on **1,992 of 1,992** eligible
corner uses; its median |dot| with the U tangent was **0.999598** (0.000025 with V and 0.027633 with the face
normal), and permuting the candidate vectors within the same mesh lowers the U median to 0.697877. Four
bit-31-clear resources from the first 12 qualifying ones in archive order gave **114 of 114** positive U
comparisons (median +0.987002). Bit 30 agreed with the side of `dP/dv` that `cross(normal, tangent)` points
to on **101 of 104** corner uses. All three exceptions are one indexed vertex whose five adjoining triangles
have three negative and two positive UV determinants while its stored bit stays set; a single per-vertex bit
cannot represent that vertex, so a consumer reconstructing a tangent frame should take handedness per
triangle or accept that documented exception. This does not by itself explain the lower core_restofworld
8-bit score above, whose per-block UV coverage was not measured here; the mixed-determinant vertex is one
candidate.

Opaque: the encrypted RPF9 tables of contents, the Bink 2 bodies, the `LSNS` record contents, the bodies of the
`RSC8` headers, the nested RPF9 payloads, and the whole `appdata0` data area. Across the 971
core graphics blocks, final-slot widths remain unresolved (`0x21`, the last unread component type,
is decoded above). The stride check
Opaque: the encrypted RPF9 tables of contents, the Bink 2 bodies, the `LSNS` record contents, the nested RPF9
payloads, and the whole `appdata0` data area. The bodies of the nine `RSC8` headers are decoded in
[rsc8_audio.md](rsc8_audio.md): facial/lip-sync clips. Across the 971
core graphics blocks, final-slot widths remain unresolved, and of the component types the non-normal
bits of `0x21` (see above). The stride check
also leaves 436 blocks unclassified, including 18 with a validated following meshlet group.
"Unread" in the tables above is not a claim that those bytes are empty: in
`story000_restofworld` the 9.33 MB unread gap contains 15 `ADAT` (AWC) banks (8,908,216 bytes), which hold the
nine `RSC8` headers as tag `0x5c` clips, and nested RPF9 headers, and
in `story000_script` the 53 MB unread region contains four nested RPF9 headers.

## Likely format family

The bytes support the existing **RPF9/RAGE family**, not a new container:

- RPF9 data areas are zlib streams with embedded RPF9 sub-archives (10 nested in core, 18 in
  story-rest, 4 in script; most sit on 4 KiB boundaries).
- Bink 2 video is `KB2n` and is separate from the resource pages; core is video-first, story-rest is
  video plus metadata/pages.
- `0DRP` version 122 reuses one 80-byte header frame across two distinct placements: in a standalone
  stream (`+0x10` after the 16-byte system prefix) it is a `0x7a` metadata resource with `0x50000000`
  table pointers, while in a `PSIN` `LRSC` section it is an 80-byte root/count summary record with
  `0xbb`-filled pointer slots. Neither form carries map-placement structs in these six archives.
- `PSIN` is a sectioned metadata container, not a compiled-script container. The six tags and the
  big-endian length chain are measured; the `LSNS` block layout is decoded, its record contents are not.
- `RBF0` records carry schema names (movie subtitles, VFX texture-atlas metadata); XML records carry
  content definitions.
- `RSC8` appears only in the story-rest unread gap, as the tag `0x5c` facial/lip-sync clips of its 15
  `ADAT` (AWC) banks ([rsc8_audio.md](rsc8_audio.md)). No `RSC7` was found.

Models are present (37 decoded in core, plus 924 name-bearing pages), but the measured bytes do not
support a claim that these archives hold map placement, compiled scripts or UI layout. Their only audio banks
are those 15 in story-rest, with 76 streams: 13 carry a streamed-music `0x48` tag and 9 are dialogue lines with
a clip. A raw `tools/rsc8_audio.py` scan of the other five archives finds no `ADAT` bank.
The only UI-adjacent evidence is content-manifest XML and text; there is no UI container signature.

## Content-definition reader (ASSET-149)

`tools/content_definitions.py` reads the three measured XML roots:
`CDataFileMgr__ContentChangeSetDef`, `CDataFileMgr__ContentDef` and `SSetupData`.
It accepts local XML files or an existing `rpf9_carve.py` directory and prints only aggregate counts:

```powershell
python -I tools/content_definitions.py <work>/carve/story000_restofworld
python -I tools/content_definitions.py <local-content.xml> <local-setup.xml>
python -I tests/test_content_definitions.py
```

The Python API `parse_content(byte_chunks)` returns a `ContentDocument` only after the entire document
parses successfully. Its `kind` is `change_set`, `content` or `setup`; its immutable `root` record preserves
ordered children, attributes, text and tails, including unknown fields, after normal XML normalization.
Comments and processing instructions are not configuration fields. The convenience views are:

- `files`: ordered `fileList/Item` entries with `filename`, `file_type` and the original record. `contents`,
  flags and future fields remain accessible through that record.
- `included_changesets`: ordered `includedChangesets/Item` records, including `loadPath` and include flags.
- `included_sets`: ordered `includedSets/Item` records.
- `change_set_groups`: setup `contentChangeSetGroups/Item` records, preserving `NameHash` and the ordered
  `ContentChangeSets/Item` children.

`record.select("field", "Item")` selects direct-child paths without collapsing repeated fields;
`record.scalar("field")` reads a single text or `value`-attribute field. Missing fields return `None`, empty
fields return `""`, and ambiguous repeated/nested scalars raise `ContentError`. Text scalar access strips
surrounding whitespace; the underlying record retains it. Values stay strings: hashes, bit flags, phases,
macros, paths and references are not converted or resolved. No include is opened and no mount is performed.
The reader does not infer runtime execution order, resource boundaries or a relationship between an XML
filename and a carved resource. Those still require independent evidence.

`carve_documents(directory)` uses the existing game-data grouper and yields `(offset, stored_bytes, document)`;
the first two values are the first recovered block offset and sum of compressed block sizes, **not** a named
RPF entry or a contiguous span including padding. Non-content XML yields `None`. Missing carved blocks raise
an error instead of substituting the index's header bytes. This path inherits the grouper's recovery limits.
The parser retains one complete XML document, limited to 64 MiB and 128 nested elements by default (`max_bytes`
can change the byte limit); the grouping step itself is not subject to that parser limit. DTDs are refused.
The CLI reports failed input counts and returns nonzero on failure; successful documents earlier in a failed
carve remain in the counts. It emits no input values, paths or exception text. API records contain game data
and must stay local, as must the carve and its index.

Full scans of the four archives in this note recovered **30 content XML documents**, all accepted, with
no record differences against an independent ElementTree parse of each reassembled payload:

| Archive | Change sets | Content definitions | Setup documents | File entries | Included change sets | Setup groups |
|---|---:|---:|---:|---:|---:|---:|
| `core_restofworld` | 8 | 1 | 1 | 33 | 8 | 4 |
| `story000_restofworld` | 11 | 1 | 1 | 28 | 11 | 2 |
| `core_restofworld001` | 2 | 1 | 1 | 3 | 2 | 1 |
| `core_licenses_data` | 1 | 1 | 1 | 1 | 1 | 1 |
| **Total** | **22** | **4** | **4** | **65** | **22** | **8** |

There are zero `includedSets/Item` entries in these documents. Counts cover the four complete archives,
not a sample of the separate 20 GB `story000.rpf`. The check compares tags, attributes, text, tails and
child order; it establishes parsing fidelity, not the game's interpretation of flags or references.
Synthetic tests cover the three schemas, unknown fields, symbolic references, duplicate/ambiguous fields,
chunk boundaries, UTF-16, malformed/DTD input, limits, split carve payloads, missing blocks and the CLI.

## Textures (`ASSET-148`)

The stream-start scan above found zero `DDS`, `TXTR`, `DXT` or `BC` signatures because rest-of-world textures are
not raw stream-start images. Measured on the 1.0.2.1 package, they live in two places:

1. **Embedded RAGE9 `0x980` texture descriptors inside `zero-prefix` CPU-segment streams** (the same `grcTexture`
   layout `tools/tex_headers.py` reads in the main archives: `q[k-1] == 0x980`, 256-byte-aligned GPU pointer
   `0x60000000 + offset` at `q[k]`, `(width, height, depth, byte6, dxgi_format)` at `q[k+4]`, `flags | (mips << 16)`
   at `q[k+5]`), paired with the following GPU-segment stream chain. `tools/restofworld_index.py` now detects these
   descriptors inside `zero-prefix` streams (`tex980=`, `dict=`, and summary fields `texture_streams`,
   `texture_dictionaries`, `texture_headers`):

   | Archive | Texture streams | Standalone `pgDictionary` | Total `0x980` headers | GPU chain fit | Formats and tiling |
   |---|---:|---:|---:|---:|---|
   | `core_restofworld.rpf` | 145 | 144 (72×1, 72×2) | 278 | 278 / 278 | 216 tiled (`flags & 0xffff = 0x5`, 1–10 mips: `BC1_UNORM` 146, `BC1_UNORM_SRGB` 31, `BC7_UNORM` 26, `BC5_UNORM` 9, `BC7_UNORM_SRGB` 4) in the 144 dictionaries; 62 linear (`0x0`, 1 mip, `R16G16B16A16_FLOAT`, e.g. `16x4`–`16x10`) in one 576 KiB resource page |
   | `story000_restofworld.rpf` | 1 | 0 | 51 | 51 / 51 | 51 linear (`0x0`, 1 mip, `R16G16B16A16_FLOAT`, `16x4`–`16x10`) in one 320 KiB resource page |
   | `core_restofworld001.rpf` | 0 | 0 | 0 | — | textures are in the companion TTPF pool below |
   | `story000_script.rpf` | 0 | 0 | 0 | — | none |

   All 329 embedded descriptors across these archives have `dtype = 0x980`, `depth = 1`, `byte6 = 1`, and 100% fit
   within their following GPU-segment stream chain (`data_off + mip0_bytes <= gpu_chain_bytes`).

2. **Companion RDS5/TTPF texture pools** alongside the archives under `pack/core_restofworld*/data/ttdb/` (6 pools,
   88,795,904 bytes total, **635** 2D textures, already walked and exported by `make_game.py`'s `assets` stage
   under `ttpf_pack/core_restofworld/` and `ttpf_pack/core_restofworld001/`):

   | Pool | Bytes | Textures | Formats |
   |---|---:|---:|---|
   | `pack/core_restofworld/data/ttdb/props/ttdb_props.ttpf` | 35,286,784 | 211 | `BC1_UNORM` 110, `BC1_UNORM_SRGB` 72, `BC7_UNORM_SRGB` 11, `BC7_UNORM` 9, `BC3_UNORM_SRGB` 8, `BC6H_UF16` 1 |
   | `pack/core_restofworld/data/ttdb/textures_loose.ttpf` | 21,081,856 | 194 | `BC1_UNORM` 89, `BC4_UNORM` 67, `BC7_UNORM` 17, `BC7_UNORM_SRGB` 13, `BC5_UNORM` 5, `BC6H_UF16` 2, `BC3_UNORM_SRGB` 1 |
   | `pack/core_restofworld/data/ttdb/peds/ttdb_peds.ttpf` | 24,529,408 | 165 | `BC1_UNORM` 67, `BC1_UNORM_SRGB` 46, `BC7_UNORM` 35, `BC5_UNORM` 9, `BC7_UNORM_SRGB` 5, `BC3_UNORM_SRGB` 1, other 2 |
   | `pack/core_restofworld/data/ttdb/ui/ttdb_ui.ttpf` | 4,729,600 | 43 | `BC1_UNORM` 37, `BC7_UNORM` 6 |
   | `pack/core_restofworld/data/ttdb/vfx/ttdb_vfx.ttpf` | 3,129,600 | 17 | `BC7_UNORM_SRGB` 5, `BC1_UNORM` 5, `BC5_UNORM` 5, `BC4_UNORM` 1, `BC3_UNORM_SRGB` 1 |
   | `pack/core_restofworld001/data/ttdb/ui/ttdb_ui.ttpf` | 38,656 | 5 | `BC7_UNORM` 3, `BC3_UNORM` 2 |

### Which source the drawables' texture references reach

The drawables name textures by the 64-bit texture IDs in their shader parameter records
([textures.md](textures.md)). Measured over the 1,732 `.ydr`/`.yft`/`.ydd` entries of the two core archives in
a local RPF9 index (all decoded; 17,208 texture references to 675 distinct IDs), against the ID tables of all 57
`.ttpf` pools in the package and against the two sources above:

| Where the ID is | Distinct IDs | References |
|---|---:|---:|
| a pool under `pack/core_restofworld*/data/ttdb/` | 285 | |
| a shared pool under `data/ttdb/` | 258 | |
| subtotal: in a pool | 543 | 6,403 |
| a key of an embedded `pgDictionary`, or any u64 inside the 145 embedded-texture streams | 0 | 0 |
| nowhere | 132 | 10,805 |

The 144 embedded dictionaries hold 166 keys; 119 of them have a zero high half, and none equals a referenced ID
or either 32-bit half of one. So these 64-bit IDs reach the pools only; how a drawable binds the embedded
`0x980` textures was not traced here.

The unresolved share is not spread over the materials. By shader texture slot (slots named by role where the
material's own textures show it; the others are unnamed and listed by size):

| Slot | References in a pool / nowhere | Distinct IDs in a pool / nowhere |
|---|---:|---:|
| colour | 1,818 / 136 | 166 / 15 |
| normal | 1,758 / 223 | 145 / 14 |
| specular | 1,675 / 306 | 149 / 19 |
| a fourth material slot | 1,049 / 931 | 52 / 24 |
| five slots with one fixed ID each | 0 / 1,463 to 1,568 each | 0 / 1 each |
| a sixth such slot | 6 / 1,517 | 1 / 2 |
| one slot that never resolves | 0 / 184 | 0 / 57 |

The per-asset material textures resolve for about nine references in ten (colour 93%, normal 89%, specular 85%).
Most unresolved references come from six slots that each use one fixed ID on 1,463 to 1,568 shaders: textures
shared by the whole engine and bound at run time rather than stored with an asset (hypothesis: lookup tables,
noise or environment maps).

Pool entry formats of the 543 resolved IDs (DXGI numbers): 71/72 (BC1) 391, 98/99 (BC7) 120, 83 (BC5) 14,
78 (BC3) 14, 80 (BC4) 3, 65 (A8) 1; 22 are 4 x 4 placeholders.

Method: drawables decoded with a local RPF9 model decoder (the shader parameter layout of textures.md); pool ID
tables read from each pool's `ETPF` section (count at +8, 16-byte records, then the sorted u64 IDs); embedded
streams found with `tools/restofworld_index.py`'s `find_rage_textures` / `is_pg_tex_dictionary` (keys at the
`+0x20` pointer, count at `+0x28`). Counts only; no IDs, names or texture data.

## What a future decoder needs

1. The RPF9 table-of-contents key/schema for names, paths and exact entry boundaries. This project
   does not search for or distribute keys.
2. A recursive RPF9 walker. The flat carve crosses 10/18/4 embedded archives; a decoder needs their
   spans and TOCs to group pages into resources.
3. A Bink 2 decoder for the 11 core and 7 story-rest videos (`KB2n`; see `docs/movies.md`).
4. An `0DRP`/`0x7a` parser (`tools/odrp_7a.py`, ASSET-156) now distinguishes and validates standalone
   `0x7a` metadata/map resources (`0x50000000` table pointers at stream `+0x10`) against `PSIN` `LRSC`
   root/count summary records (`0xbb`-filled pointer slots at `LRSC+0x08`) across single and
   continuation-joined streams (see [world.md](world.md) and [psin.md](psin.md)); decoding resource payload
   fields beyond schema/block layout still needs per-type definitions.
5. A `PSIN` section walker for `LSCH`, `LENM`, `LSNS`, `LGID`, `LRSC` and `CHKS`. The chain is
   measured; `LSNS` is a block table and 16-byte records ([psin.md](psin.md#lsns)), not zlib. Do not call it
   bytecode without evidence.
6. Model-page joining is implemented in `tools/restofworld_model_join.py` (and `rpf9dump` at `e293917`):
   the `+0x08` `datResourceMap` page counts, 64 KiB stream blocking and `0x60000000` buffer validation join
   all 924 `core_restofworld` model resources (936 drawables across 926 system and 1,020 graphics streams)
   and the single `story000_restofworld` drawable dictionary (3 drawables) without the TOC. Archive path
   names and external texture-dictionary links still need the TOC or hash matching.
7. A vertex decoder for `42 40 00 00` blocks. ASSET-57 measured recurring stream widths;
   decoding vertex values still requires component types, final-stream widths and a way to
   locate the owning geometry when only a graphics page is available. (Texture identification is
   resolved above under **Textures (`ASSET-148`)**: embedded `0x980` descriptors in `zero-prefix`
   CPU-segment streams plus the companion `ttdb/*.ttpf` pools.)
8. Runtime consumers for the XML content-definition records read by `tools/content_definitions.py`
   (see above). Their roots describe configuration, not UI layout; reference resolution and mapping
   to resources remain unimplemented.
9. A false-positive-aware magic scanner. `ADAT` inside `METADATA`, `AWC` inside names and `ATI2` inside
   a page are measured examples; only validated stream starts and fixed headers count.

## No-sharing rule

Do not commit archives, carved streams, inflated payloads, asset names, path strings, hashes of assets
or name dumps. This page and its test contain aggregate counts and synthetic bytes only. Keep carves in
a local work directory, never in the repository or a PR. `tools/restofworld_index.py` writes offsets,
sizes, signatures and header fields; it does not write resource payloads or carved bytes.

## Reproduce

```powershell
python -I tools/rpf9_carve.py <archive.rpf> <work>\carve\<name> --index-only
python -I tools/restofworld_index.py <archive.rpf> <work>\index.csv
python -I tools/restofworld_index.py <archive.rpf> <work>\index.json
python -I tests/test_restofworld_index.py
```

`--start` must be 0 or more, and `--limit` and `--max-inflate` at least 1 (decimal or `0x` hex). An out-of-range
value stops with a one-line error and exit code 2 before an index is written.

For the map/meta split, run the `world_index.py` from `docs/world.md` over the same archive; it reports
`maps`, `meta`, `entities` and the `0x7a` version without decoding geometry. The index tool walks readable
streams; the nine `RSC8` headers in the story-rest unread gap come from a raw-file scan, not from the
tool. The cross-check hashes in this report came from a local SHA-256 pass over the inflated streams and
are not committed. The carver and index tool are stdlib-only and read the archive in place; neither
decrypts the TOC.

## See also

- [docs/rpf.md](rpf.md) - RPF9 carving and recovery tools.
- [docs/design.md](design.md) - the model manifest and the archive-stage table.
- `docs/story.md` - the `story000`/`shared_minigame000` archive family (pending story PR).
- `docs/world.md` - map placement and the `0x7a` metadata resource (PR #117).
- `docs/psin.md` - the `PSIN` container measured on `script.rpf` and `data.rpf` (PR #110).
