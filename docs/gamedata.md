# Game data in `common.rpf`

This note measures the `common.rpf` archive from the Xbox Series 1.0.2.1 package. It records aggregate
counts, sizes and schema/root names only. No archive, carved file, XML value, entry name or per-file path is
committed. Everything below comes from a local copy and can be reproduced on any copy of the same version.

`common.rpf` is 6,642,192 bytes. Its RPF9 header is `RPF9` (`9FPR` little-endian) and the data area starts at
byte 38,272 (`0x9580`). The table of contents is encrypted/opaque; nothing here decrypts it or looks for keys.
The measurements cover the data area only.

## How to carve it

Use the carver with the short-final-slice fix (advance by the bytes actually read, not by the 1 MiB request):

```powershell
python -I tools/rpf9_carve.py common.rpf <work>\carve\common
```

Its `index.json` has `off`, `stored`, `size`, `pre` and `head` per stream. The `.bin` files are named by
offset and are **zlib streams, not files**. RPF9 splits a large file into 64 KiB inflated blocks, each with
its own zlib stream. A logical XML document is the concatenation of consecutive streams until its root element
closes; 178 of the 257 header-bearing documents are one stream, and the largest document spans 218.

`tools/gamedata_index.py` groups those blocks into payloads and writes an offset/size/type/root index:

```powershell
python -I tools/gamedata_index.py <work>\carve\common --csv common.csv --json common.json
```

The CSV/JSON contain offsets, sizes, the coarse type and the XML root element. They contain no element values
or entry names, but they are still local work products and must not be committed or attached to an issue.
`--csv` and `--json` refuse a path that would replace the carve's `index.json` or one of its
`<offset>.bin`/`.bk2` streams, so a report cannot be written over a carved input.

### Coverage

| Measure | Value | Share of data span |
|---|---:|---:|
| Data span (file size minus header) | 6,603,920 B | 100.0% |
| zlib streams | 954 | - |
| Stored stream bytes | 5,647,255 B | 85.5% |
| Inflated bytes | 50,231,123 B | 8.9x the stored bytes |
| Zero padding | 386,138 B | 5.8% |
| Unread bytes | 570,527 B | 8.6% |

The earlier 781-stream / 39.5 MB / 87.3% figure came from the carver that advanced by the 1 MiB read request.
It counted skipped regions as part of the preceding stream, so its stored-byte total was too high and its
inflated total was too low. The fixed carver finds 173 more streams and 10.7 MB more data; its stored total is
lower because it measures each zlib stream's real end. Treat the old figures as superseded.

The 8.6% unread region is the non-zero gaps between streams: high-entropy bytes that do not decode as complete
zlib streams. Some contain a zlib header but fail the Adler check or are truncated. This is why 9 XML documents
survive only from their closing half and one 64 KiB XML block has no recoverable root. The carve does not guess
their content.

## Type split

Stream-level split, using the requested classes. `other` is 2 `MHIU` payloads (38 streams), 1 `PAMH` and 1
`SPS1`; there are no `RBF0`, `PSIN` or Bink 2 streams in this archive.

| Type | Streams | Stored B | Inflated B | Logical payloads |
|---|---:|---:|---:|---:|
| XML (`<?xml` / `<`) | 844 | 3,764,523 | 43,284,925 | 267 (266 docs + 1 fragment) |
| `RBF0` | 0 | 0 | 0 | 0 |
| `PSIN` | 0 | 0 | 0 | 0 |
| text / INI-like | 4 | 30,162 | 142,036 | 3 |
| binary | 66 | 1,396,087 | 4,312,136 | 2 |
| other (`MHIU`/`PAMH`/`SPS1`) | 40 | 456,483 | 2,492,026 | 4 |
| **total** | **954** | **5,647,255** | **50,231,123** | **276** |

By inflated bytes: 86.2% XML, 8.6% binary, 5.0% `other`, 0.3% text. XML is only 66.7% of stored bytes,
which is why a 6.6 MB archive inflates to 50.2 MB. XML grouping is exact because each document closes on its
root element. Non-XML grouping follows the 64 KiB block rule; none of the nine non-XML payloads here has a size
that is an exact multiple of 64 KiB, so their counts are unambiguous too.

The four non-XML structured payloads are the only readable binary metadata here: two `MHIU` blobs (about
1.19 MB and 1.21 MB inflated), one `PAMH` blob (48.6 KB) and one `SPS1` shader-preset table (43,791 B).
The two large opaque blobs (1.04 MB and 3.27 MB) have no recognised header and are left as binary. No model,
texture or audio signature is present in `common.rpf`; the two opaque blobs are not identified.

## MHIU and PAMH heightmaps

`tools/gamedata_maps.py` decodes both payloads. They are **heightmaps**, not
generic square arrays: each has a 44-byte header, a row table and u8 heights,
and both scale a value into the header's Z box with

```
z = bounds.min.z + value / 255 * (bounds.max.z - bounds.min.z)
```

`common.rpf`'s manifest names `common:/data/levels/americas/heightmap.dat` as a
`WORLD_HEIGHTMAP_FILE`, and the `PAMH` magic is the GTA V `HMAP` magic stored
little-endian. The row-table layout matches the published GTA V
`heightmap.dat` reader (`HMAP`, 44-byte header, 8-byte row headers, max/min
u8 arrays). The measured file is version 1.2; GTA V writes 1.1.

### `MHIU`: 2048 x 2048 single-layer heightmap

| Header | Offset | Type | Measured |
|---|---|---|---|
| magic | 0x00 | 4 bytes | `MHIU` |
| version | 0x04 | `u16` | 1 |
| width | 0x06 | `u16` | 2048 |
| height | 0x08 | `u16` | 2048 |
| pad | 0x0A | `u16` | 0 |
| six bounds | 0x0C | 6 x `f32` | `(-13000, -9000, -50)` to `(7000, 11000, 500)` |
| data size | 0x24 | `u32` | bytes after the 44-byte header |
| default height | 0x28 | `u8` | 23 in both (0 m quantised) |
| pad | 0x29 | 3 bytes | differ per file; not a checksum |

At 0x2C there are `height` records of eight bytes
`{u16 start, u16 count, i32 offset}`, then the packed u8 heights. A row with
`count = 0` keeps the default height from byte 0x28 (**23** in both payloads); the other rows fill cells
`start .. start + count - 1` from `pixels[offset + start + i]`.

| Payload | Filled rows | Empty rows | Packed bytes | Height range |
|---|---:|---:|---:|---|
| `0x144000` | 1,724 | 324 | 1,172,743 | 19..209 |
| `0x16E000` | 1,727 | 321 | 1,194,048 | 0..209 |

The row spans cover the packed bytes exactly: **0 gaps, 0 overlaps** in both
payloads. The default 23 is `-0.39 m`, and 209 is `400.8 m`, so the default is
the sea surface: `(0 - min.z) / (max.z - min.z) * 255 = 23.18`, so 23 is 0 m
quantised to the header's Z box.

The encoder crops each row to its first and last non-default cell. Over the
3,451 filled rows of both payloads, **0 of the 6,902 span ends** hold 23,
although 23 is 82% of all cells and occurs 849,544 times *inside* spans. The
ends are 24 (6,402), 22 (343) and other neighbours of 23. Empty rows are rows
that are entirely default. The decoder reads the default from byte 0x28 and
reports `span_ends_at_default` so a different build can be checked the same way.

The earlier "zero-filled prefix plus square byte grid" reading (1,090 x 1,090
and 1,100 x 1,100) was an artifact. The prefix is the zeroed row headers of
the empty rows, and the grid is 2048 x 2048. The largest square that ends at
the payload end happened to fit the byte count, but it is not the format.

### `PAMH`: `HMAP` max/min heightmap (`common.rpf` 144 x 164, tiles 64 x 64)

| Header | Offset | Type | Measured |
|---|---|---|---|
| magic | 0x00 | 4 bytes | `PAMH` (little-endian `HMAP`) |
| version | 0x04 | 2 x `u8` | 1.2 in `common.rpf`; overwritten in the `levels.rpf` tiles |
| pad | 0x06 | `u16` | 0 in `common.rpf`; text in the tiles |
| word at the GTA V `compressed` offset | 0x08 | `u32` | `0x1BBC8CED` in `common.rpf`. Level tiles contain label bytes with bit 0 set. |
| width, height | 0x0C | 2 x `u16` | 144, 164 in `common.rpf`; 64, 64 in every tile |
| six bounds | 0x10 | 6 x `f32` | `(-13312, -9728, -302)` to `(5120, 11264, 404)` |
| data size | 0x28 | `u32` | 48,544 in `common.rpf`; 8,704 in a tile |

At 0x2C there are 164 row records, then `width * height` **max** bytes and
`width * height` **min** bytes. The measured max layer runs 110..255 and the
min layer 0..238; `min <= max` in all 23,616 cells. The max layer is a flat
110 (2.5 m) over most of the map - the sea surface - and rises to 255 (404 m)
on land. The min layer is the ground or sea floor.

The earlier u16 reading interleaved the two arrays: every measured `0x6E6E`
was a `(110, 110)` max/min pair, not the value 28,270.

### The 1,476 `PAMH` tiles in `levels.rpf`

`levels.rpf` holds 1,476 more `PAMH` payloads. All of them decode with the
same rules: a 44-byte header, 64 x 64 cells, 64 full-width row records with
`start = 0`, `count = 64` and `offset = y * 64`, then 4,096 **max** bytes and
4,096 **min** bytes. That is 6,045,696 cells in total.

- The tiles are a 36 x 41 rectangle with **512 m** spacing, so they cover the
  same `(-13312, -9728)` to `(5120, 11264)` box as the `common.rpf` payload,
  with the same Z box `-302` to `404`.
- Max values span 0..255 and min values 0..254. `min <= max` in all but 62
  cells, all in the tile at `(-5120, 2048)`, where the encoder wrote a min
  above its max.
- Use the tile's minimum XY corner to recover its column and row. Subtract the
  grid minimum and divide each axis by 512. The 1,476 distinct coordinate pairs
  produce only 954 distinct seven-byte labels at 0x05..0x0B.
- To reproduce the label, right-align `height` in those seven bytes, then write
  the one-based `column_row` and its NUL at 0x05. This predicts 819 headers
  exactly. Setting bit 0 at **0x08** in that prediction matches all seven bytes
  on **1,476/1,476** tiles. In 81 tiles the bit occupies the expected NUL.
  Every stored 0x08 word is odd. The bit's runtime meaning is unknown.
- The census groups by grid size. Byte 0x04 stays 1, 0x05 holds the first
  digit of the column, and the 0x08 word takes 28 distinct values.

```powershell
python -I tools/gamedata_maps.py <work>\carve\levels --json levels.json
```

The tool prints one `PAMH census 64x64: ...` line and writes the per-group
counts. Only header fields and counts are written, never the height bytes.

### Cross-check and Z mapping

MHIU and PAMH cover slightly different boxes at different resolutions.
Sampling MHIU at the PAMH cell centres over their overlap gives a Pearson
correlation of **r = 0.861** for payload `0x144000` and **r = 0.870** for
`0x16E000` against the PAMH max layer. Mirrored or transposed orientations
give `r <= 0.64`. The land masks (`MHIU > 23` and `PAMH max > 110`) overlap
with **IoU 0.61**. Row 0 is therefore `bounds.min.y`, column 0 is
`bounds.min.x`, and the linear Z formula above is the same for both.

The HMAP header and row-table rules were cross-checked against the public
[rage-formats](https://github.com/VIRUXE/rage-formats) reader (`Unlicense`)
and [gta-op/heightmap](https://github.com/gta-op/heightmap) (`MIT`); no code is
copied. The MHIU field offsets, the span checks and the Z correlation are
measured on this package.

```powershell
python -I tools/gamedata_maps.py <work>\carve\common --json maps.json
```

### Checksum and unknown fields

- The `common.rpf` `PAMH` payload holds `0x1BBC8CED` at GTA V's `compressed`
  offset, 0x08. The level tiles overlap this word with the coordinate label
  described above. The common value is not CRC32, Adler32, FNV-1a, joaat32 or a hashlib
  digest of the max/min arrays; it is reported as a flag word, not a proven
  checksum.
- `MHIU` 0x28 is a `u8` default height, not a 32-bit word (ASSET-48). Its
  value is 23 in both payloads, the same value the empty rows were measured
  to hold, and the row cropping above uses it. A full scan of all 12 archives
  of package 1.0.2.1 (every zlib stream inflated far enough to read its magic)
  finds **2 MHIU payloads**, both in `common.rpf`, so N = 2 for the header
  values; the cropping rule is measured over all 6,902 span ends.
- `MHIU` 0x29..0x2B differ between the two files and match none of CRC32,
  Adler32, FNV-1/FNV-1a, joaat32 or an MD5/SHA-1/SHA-256/SHA3/BLAKE2 digest
  (any 3-byte window, either byte order, or a 24-bit slice of the 32-bit
  hashes) over the grid, the packed bytes, the row table, the header, the
  body or the payload with 0x28..0x2B zeroed. They do not equal any count or
  sum of the grid. They are most likely uninitialised struct padding after the
  `u8` and are reported as `pad_29`; with N = 2 this is a hypothesis.

## `SPS1` shader-preset table

The single `SPS1` payload in `common.rpf` (`0x650000`, `43,791` B inflated, 1 stream) is the material shader-preset
lookup table (`tools/gamedata_sps1.py`), not an `"SPS1Z"` five-byte magic:

| Offset | Type | Meaning | Measured in 1.0.2.1 `common.rpf` |
|---|---|---|---|
| `0x00` | `char[4]` | Magic | `"SPS1"` |
| `0x04` | `u32` | `entry_count` | `858` (`0x0000035A`; its low byte `0x5A == 'Z'` caused the old `"SPS1Z"` reading) |
| `0x08` | `entry[858]` | `{u8 preset_len, char preset[], u8 effect_len, char effect[], u8 render_bucket}` | `43,783` B consuming the entire payload to EOF (`0` extra bytes) |

Measured over all **858** records and cross-checked against `shaders.rpf`'s two `SGD5` databases:

- **Presets:** all `858` carry a unique case-insensitive `.sps` path (`8`..`51` bytes), split into `846` root
  names and `12` one-folder paths with zero basename stem collisions.
- **100% `SGD5` target resolution:** all **858 of 858** records (`539 of 539` distinct target effects, `4`..`47`
  bytes) match an effect in `shaders.rpf` `SGD5` Database 0 (`0x10E39`, `910` effects; `0` in Database 1).
- **Aliases:** `532` presets have `preset_stem == effect` and `326` alias a preset name onto a shared effect; only
  `589` preset stems match an `SGD5` effect name directly (`269` preset stems are defined only in `SPS1`).
- **Default render buckets (`0..7`):** `0` (`401`), `1` (`161`), `2` (`101`), `3` (`162`), `4` (`1`), `5` (`1`),
  `6` (`30`), `7` (`1`). `63` target effects span multiple buckets across their presets (`58` in `{0, 3}`, `2` in
  `{0, 1, 3}`, `1` in `{1, 3}`, `1` in `{0, 1, 2, 3, 6, 7}`, and `1` in `{4, 5}` — the single bucket-`4` and bucket-`5`
  presets both point to effect hash `0xFC34F08B`, matching [`models.md`](models.md#shader-render-buckets)).

```powershell
python -I tools/gamedata_sps1.py <work>\carve\common --shaders shaders.rpf --json sps1.json
```

Only counts, bucket distributions and `--preset-hash` hashes are written; preset and effect names stay local.

## Grid atlases

Four `GridAtlasInfo` documents describe world-space texture grids streamed by position. All four have the
same numeric fields; the rest are names (`type`, `clientSystemName`, tile prefix/suffix, fallback texture).

| Field | Measured (identical in all 4) |
|---|---|
| `width`, `height` | 16,384 x 33,792 world units |
| `tileSize` | 512, so **32 x 66 = 2,112 tiles** |
| `origin` | the grid **centre** (see below) |
| `streamRadius` | 512 (one tile) |
| `zInflation` | 1,000 |
| `hasOwnGlobalFallback`, `useGlobalFallback` | true, true |
| `sectorOverrides/Item` | 1, with one sector at start (0, 0) and span (32, 66): the whole grid |

The tile count is the only reading under which the sector override's span matches: 16,384 / 512 = 32 and
33,792 / 512 = 66, so `width`/`height` are world units, not tile or texel counts.

Which corner `origin` names was tested against entity positions from a local index of map resources: 2,036,542
placements from 23,361 resources at every LOD level, 2,011,519 of them in `levels.rpf` and the rest in the
minigame and story packs (x 0.1th to 99.9th percentile -9,955 to 4,577, y -8,096 to 9,440). This is a
different count from the 1,670,442 entities of the levels census in [world_census.md](world_census.md); it was
only used for the share below. With `origin` as the grid centre, 99.83% fall inside the grid; as the minimum
corner, 1.0%; as the top-left corner with y growing downwards, 82.5%. The `levels.rpf` subset alone gives the
same picture: 99.83%, 1.0% and 83.0%. The centre
reading also gives a minimum x of exactly -13,312, the `PAMH` heightmap's minimum x above.

Tile names: the tile textures do exist, but none is named after the tile. Every tile carries only the grid's
`tileTextureNameSuffix` as its texture name, one tile per texture dictionary, and the dictionary names are not in
the data. That is why `tileTextureNamePrefix` spellings and the fallback names matched no archive entry or item name
in the earlier catalog search. `tools/world_grid_atlas.py` (below) finds them in the texture manifests of
`make_game.py --stages assets`.

### What the four grids hold

All four are water data. Every count below is in the tool's printed summary or is the named `--json` field;
`levels` and `textures` are `levels.rpf` and `textures.rpf`.

| Grid (`clientSystemName`) | Tile texture | Tile dictionaries | Fallback (pixels per tile) | Placed |
|---|---|---:|---|---:|
| `WaterSystemHeightMap` | `watersystemheightmap`, 512 x 512 `R16_UNORM` | 738 in `levels` | 1024 x 2112 `R16_UNORM` (32) | 737 |
| `shorelinegrid1` | `shore_wave_placement_set1`, 256 x 256 `B8G8R8A8_UNORM` | 268 in `levels` | 1024 x 2112 `BC7_UNORM` (32) | 262 |
| `waterfogatlas` | `waterfog`, 128 x 128 `BC6H_UF16` | 1,313 in `textures` | 512 x 1056 `BC6H_UF16` (16) | 592 |
| `waterfogatlas2` | `waterfog2`, 128 x 128 `BC6H_UF16` | the same 1,313 | 512 x 1056 `BC6H_UF16` (16) | 592 |

- Each `globalFallbackTextureName` is a mosaic of the whole 32 x 66 grid. Fallback image row 0 is the grid's north
  edge (largest y, see the fit below), and cell = row * 32 + column.
- The two fog grids share their dictionaries: each of the 1,313 holds one `waterfog` and one `waterfog2`.
- `shared_minigame000` holds the `mapswap_tblux` sector-override fallback of all four grids (`sector_fallbacks`)
  and a few more tiles (`tiles_in_other_archives`: 10 water height, 2 shoreline, 1 fog pair). They are counted
  and printed, not placed.
- The water-height fallback has 1,134 constant cells of 2,112; its most common value, 20,480, covers 55.74% of the
  pixels, and the values run from 20,470 to 49,247.

### Placing the tiles

A tile's cell is recovered by comparing it with its grid's fallback mosaic, then checked against archive order.

- Water height and fog (`method: content`): each tile is point-sampled to the fallback's pixels per tile and
  compared with every cell by mean absolute difference; the two fog textures of a dictionary are matched together.
  An anchor is a tile with detail (standard deviation above 0.5) whose runner-up distance is at least twice the
  best and 0.5 above it. 737 of 738 water-height tiles are anchors; the one flat tile stays unplaced. For fog,
  617 dictionaries have detail in either texture and 561 are anchors.
- Shoreline (`method: mask`): the fallback's most common colour, `(0, 128, 0, 128)`, is background. Tiles and cells
  are compared by intersection over union of their feature masks (pixels more than 32 from the background), 264
  fallback cells have features and 6 tiles have none. The tiles, in archive order, are aligned to cells in text
  order (next point) by a dynamic programme that maximises the summed overlap. A tile is placed when it overlaps
  the cell it is aligned to: 262 are (median overlap 0.83, `overlap_median`), and the 6 unplaced ones are the 6
  without features. 208 tiles match one cell clearly without the order (best overlap more than 0.2 above the
  runner-up); these are the shoreline anchors, and the alignment moves none of them
  (`anchors_moved_by_alignment` 0).
- **Archive order:** the tile dictionaries are stored in the order of their cells sorted as text of the decimal
  cell index (`sorted(cells, key=str)`). Ascending steps of the anchors in archive order (`anchor_order`):

| Order of the cells | Water height | Fog | Shoreline |
|---|---:|---:|---:|
| text of the cell index | 736 / 736 | 560 / 560 | 207 / 207 |
| cell index as a number | 735 / 736 | 559 / 560 | 206 / 207 |
| text of `row_column` | 709 / 736 | 540 / 560 | 189 / 207 |
| text of `column_row` | 700 / 736 | 526 / 560 | 173 / 207 |

Number order differs from text order only where the cell index changes its digit count. The one break of
number order in each grid is there: in the `--json` placements the three-digit cells (from 912 in water height)
are stored after the last four-digit cells, where text order puts them. The dictionary names most likely
contain the decimal cell index; with one deciding step per grid this is a strong hint, not a read name.
- Between two anchors, tiles are placed by archive order when their number equals the free cells between them
  (`placed_by_order`): 31 fog tiles, none for water height. 721 fog dictionaries stay unplaced; 677 of them have
  no detail in either texture (`unplaced_without_detail`), which content cannot place.
- Cross-check (`in_water_height_cells`, `anchors_in_water_height_cells`, `placed_by_order_in_water_height_cells`):
  all 262 placed shoreline tiles and all 561 fog anchors lie in a cell that also has a water-height tile; of the
  31 fog tiles placed by order, 16 do.

### Water-height values and `zInflation`

The water-height values follow the ground. Over the fallback pixels above 36,000, a least-squares fit against the
`MHIU` terrain heights (`water_fit`) gives:

| `MHIU` payload | Units per metre | Value at 0 m | r (north up) | r (south up) |
|---|---:|---:|---:|---:|
| `0x144000` | 40.81 | 32,947 | 0.9457 | -0.2193 |
| `0x16E000` | 40.97 | 32,815 | 0.9935 | -0.2187 |

Reading row 0 as the south edge destroys the fit, which is why row 0 is north. Under the better fit, the
fallback's 20,470 to 49,247 is -301.3 to 401.1 m (`water_metres`), close to the `PAMH` Z box of -302 to 404 m. The
fit is consistent with a `R16_UNORM` value spread over -800 to 800 m (65,536 / 1,600 = 40.96 units per metre, 0 m
at 32,768). Under that reading the dominant 20,480 is -300 m, at the floor of the `PAMH` Z box (-302 m) and below
all ground in `MHIU` (-50 m), which would make it a "no water" value. Both readings are hypotheses: the scale comes from a fit, not from a field.

`zInflation` was not observed at runtime. All four documents have origin z = 0 and `zInflation` 1,000, so if the
runtime turns each 2D tile into a box from origin z - 1,000 to origin z + 1,000 for its stream test, that box
covers every height the world data describes. The tool reports (`z_check`) that all 3 heightmap Z ranges in
`common.rpf` (`PAMH` -302 to 404 m, both `MHIU` -50 to 500 m) lie inside -1,000 to 1,000, and the water heights
above fall inside it as well. This is a consistency check of that reading, not a measurement of the runtime:
nothing here shows whether the box is used for the stream test, the camera, or something else.

```powershell
python -I tools/rpf9_carve.py <game>\common.rpf <work>\carve\common
python -I tools/world_grid_atlas.py <work> --carve <work>\carve\common --json <scratch>\grid_atlas.json
python -I tests/test_world_grid_atlas.py
```

The tool imports numpy and Pillow (Pillow decodes `BC6H`); its placement, ordering and fitting logic is plain
Python, and the tests of that part run on the standard library alone. The forward check in the tests builds four
synthetic grids with tiles at known cells, stored in text order, and requires every tile with content to come back
at its cell; stored in number order instead, the text-order check reports breaks. The `--json` file holds the cell
of every placed tile, derived from your own copy of the game: keep it local.

### `TextureGridInfo`

In the 1.0.2.1 exports, one content change set lists four grid metadata registrations and three candidate
archive paths as siblings. The list gives no per-grid archive assignment. The two fog grids share a tile
prefix, while the shore-wave and water-height grids each have a distinct prefix. A case-insensitive search
of the 30,516 other plain export blobs (30,520 total, excluding the four grid documents) found none of the
three prefixes or the grids' global or sector fallback names in ASCII or UTF-16LE. None of the three
archive filenames appeared as a loose file in the local package. The single catalog hash match above
remains uncorroborated. The tile payloads and the mapping from tile coordinates to archive entries remain
unverified.

A fifth grid uses the same base fields: one `TextureGridInfo` document has `origin`, `width`, `height`,
`tileSize`, `streamRadius` and `zInflation`, plus a low- and a high-detail fallback name, but no sector
overrides. It is 32,768 x 32,768 units with the same 512 tile, so **64 x 64 tiles**, the same 512 stream
radius and the same 1,000 `zInflation`, around a different origin. Tested the same way, 100.00% of the
placements fall inside with `origin` as the centre, 0.01% as the minimum corner and 11.8% as the top-left
corner, so the centre convention holds for both document types.

The tool looks up both fallback names of this document (`texture_grids`): `fallen_snow_cov_fallback_low` is in
`textures.rpf` and `fallen_snow_cov_fallback_high` is in none of the exported manifests. The document has no tile
name field, so the tool does not look for its tiles, and what this grid holds is not measured.

## XML documents

257 documents start with an XML declaration and all 257 parse as well-formed XML with a standard parser.
Nine more complete documents are visible only from their closing root tag because their first block was in
the unread region; 2 of those 9 still parse from their first retained byte, and 7 are truncated at the start.
One 64 KiB XML block has no root at all. That gives 266 recoverable documents and 118 distinct root elements.

Size distribution of the 257 header-bearing documents: minimum 88 B, median 13,435 B, mean 165,603 B,
maximum 14,262,150 B. 59 are at most 1 KiB, 178 are at most 64 KiB, 3 exceed 1 MiB and the largest spans 218
streams. 41 start with a UTF-8 BOM; 255 declare UTF-8 (214 uppercase, 41 lowercase), 2 omit the encoding
attribute, 11 say `standalone="yes"`, and the 412 comments are hand-edit notes.

### Root elements

| Root element | Documents |
|---|---:|
| `timecycle_modifier_data` | 49 |
| `CVehiclePoints` | 32 |
| `timecycle_keyframe_data` | 27 |
| `CDataFileMgr__ContentChangeSetDef` | 23 |
| `CTuningFile` | 10 |
| `CDamageDataManager` | 5 |
| `GridAtlasInfo` | 4 |
| `CCoverProcessorTuning` | 3 |
| `CDataFileMgr__ContentDef` | 2 |
| `CVehiclePerformanceMaps` | 2 |
| `CVehicleBoneOffsetsArray` | 2 |
| 107 other root elements | 107 |
| **total** | **266** |

The repeated roots matter: a root element alone does not identify a file. There are 49 timecycle modifier
documents, 32 vehicle-point documents and so on. Original paths and names would have to come from the missing
table of contents.

### Functional categories

The 118 root elements group into these categories. Counts and bytes cover the 266 recoverable documents
(41.2 MiB), not the one unattributed block.

| Category | Files | Inflated B | Roots | Representative root elements |
|---|---:|---:|---:|---|
| time, weather, sky, water | 87 | 17,339,011 | 13 | `timecycle_modifier_data`, `timecycle_keyframe_data`, `CContentsOfWeatherXmlFile`, `ShoreWavePlacementData` |
| vehicles, tyres, handling data | 45 | 19,618,605 | 12 | `CVehiclePoints`, `CVehicleBoneOffsetsArray`, `CTyreDataSet`, `CVehiclePerformanceMaps` |
| peds, schedules, appearance | 29 | 2,029,321 | 29 | `CPedScheduleInfoManager`, `CPedVariationBodyTypeManager`, `CPairedPedDataBlock`, `CEnergyConfigInfos` |
| tuning, parameters, effects | 28 | 343,601 | 14 | `CTuningFile`, `CCoverProcessorTuning`, `GridAtlasInfo`, `PatchConstants` |
| mount / content manifests | 25 | 450,316 | 2 | `CDataFileMgr__ContentChangeSetDef`, `CDataFileMgr__ContentDef` |
| world, navigation, map | 19 | 1,660,133 | 19 | `GameShape__Data__LocationNameLoopkup`, `PathingStyleData`, `CTrainConfigs`, `CHazardNodes` |
| combat and AI behaviour | 13 | 809,074 | 13 | `CCombatInfoMgr`, `CFiringPatternInfoManager`, `CFleeDecisionMakerData`, `Combat__GoalData__Store` |
| animation, poses, weapons | 6 | 162,778 | 6 | `WeaponAnimations__FullbodyControllerData`, `WeaponAnimations__ConfigData`, `CAimUpperBodyDataBlock` |
| damage | 5 | 485,482 | 1 | `CDamageDataManager` |
| audio, video, speech | 4 | 13,741 | 4 | `LandingPageVideos`, `CTVPlaylistContainer`, `naSpeechEventTypeBatchRequestParams` |
| missions, scenarios, respawn | 3 | 57,327 | 3 | `CDynamicScenarioManager`, `CMissions`, `Respawn__MetaData` |
| conversation | 2 | 250,000 | 2 | `CConversationList`, `CConversationDictionary` |
| **total** | **266** | **43,219,389** | **118** | |

This archive has no root element for popcycle, weapon stats or handling. Weapon-related documents here cover
loadouts, combat references and animation configuration. Vehicle handling-style values appear under the
vehicle/tyre roots, and the remaining gameplay tables are not XML. Do not infer a full gameplay schema from
`common.rpf` alone.

## Fog volumes

Volumetric fog comes in two documents: `FogVolumeProfileList` (the media) and
`CScatteringTransmittance__FogVolumeListData` (placed volumes). Counts; no values or names.

**Profiles** (`ProfileList/Item`, 43): `Name`, `UUID`, `Anisotropy`, `WaterDropletDensity`,
`DiffusionIntensity`, `ScatteringColorBlend`, `AbsorptionColorBlend`, `Scattering` and `Absorption` (`x y z`),
`IntExtMode`, plus optional `WaterDropletVariance` (8) and `WindDensityInfluence` (15). Names and UUIDs are
unique. `Scattering` and `Absorption` are within 0..1 per channel in 41 of 43. `IntExtMode` is 0 in 31 and 1 in 12.

**Volumes** (`FogVolumeList/Item`, 14), 37 fields each: shape (`IsBox`, `Position`, `PositionOffset`, `Scale`,
`Rotation`, `Shear`, `Wedge`), falloff (`Falloff`, `FalloffCurve` and per-axis versions), two 3x3 matrices
(`RotationMtx`, `TransformMtx`, 9 numbers each), ambient/emissive intensity, noise and wind, `Scattering`, `Density`,
`Anisotropy`, `WaterDropletDensity`, `Color`, `BlendMode`, `IntExtMode`, `SortKey`, `ProfileIndex`,
`SecondaryProfileIndex`, `ProfileInterpolation`, `uuid` and `Name`.

| Check | Result |
|---|---|
| `ProfileIndex` and `SecondaryProfileIndex` in `0..42` | 14 of 14 each: profiles are referenced by **list position**, not by UUID |
| Volume `uuid` equal to a profile `UUID` | 0 of 14 (separate id spaces) |
| Distinct profiles referenced | 2 of 43; primary equals secondary in 12; `ProfileInterpolation` is 0 in all 14 |
| `RotationMtx` and `TransformMtx` | identity in all 14, while `Rotation` is zero and `Scale` ranges 0.6 to 28 |
| `Shear` | zero in all 14 |
| `IsBox` | true in 13, false in 1 |
| `BlendMode` / `IntExtMode` | 0 or 1, seven each / 0 in all 14 |
| Positions | 7 at exactly (0, 0, 0); 8 distinct; all within x -2,093..30, y 0..803, z 0..109 |

Two consequences for a loader. The stored matrices do not include `Scale`, so the volume transform has to be
built from `Position`, `Rotation` and `Scale` rather than read from `TransformMtx`. Profile indices are
positions in `ProfileList`, so its order must be kept.

**Package-wide census (ASSET-308, `tools/fog_volumes.py`).** Scanning all `0x7a` resources (assembled via
`world_census.assemble`, up to 64 MiB and multi-stream, without a 256 KiB per-stream cutoff) in `packs.rpf`,
`levels.rpf`, `data.rpf`, and `story000.rpf`, plus bounded zlib-stream scans across the remaining package
archives (`common.rpf`, `anim.rpf`, `navmeshes.rpf`, `textures.rpf`, `script.rpf`, `story000_script.rpf`,
`core_restofworld.rpf`, `core_restofworld001.rpf`, `story000_restofworld.rpf`, `shared_minigame000.rpf`,
`core_bvh4.rpf`, `shared_minigame000_bvh4.rpf`, `appdata0.rpf`, `core_licenses_data.rpf`) for XML, RBF0, and
`0x7a` schemas matching `CScatteringTransmittance__FogVolumeListData` (`0x03073767`) or
`CScatteringTransmittance__FogVolume` (`0x81907918`) finds:

- **`levels.rpf`** (`188,449` complete `0x7a`, `0` incomplete), **`data.rpf`** (`15,768` complete `0x7a`, `0`
  incomplete — matching `docs/world.md` and `docs/page_directory.md` — plus `1,263` `RBF0`), **`story000.rpf`**
  (`48,084` complete `0x7a`, `0` incomplete, plus `1` `RBF0`), and all other archives except `common.rpf` and
  `packs.rpf`: **0** `FogVolumeListData` / `FogVolume` resources.
- **`common.rpf`**: remains the only archive with the XML `FogVolumeProfileList` (`0x236000`, 43 profiles) and
  world-coordinate `CScatteringTransmittance__FogVolumeListData` (`0x64c000`, 14 volumes).
- **`packs.rpf`** (`4,678` complete `0x7a` resources assembled, `0` incomplete): **231 complete `0x7a`
  resources** (`0x8ae9010..0xc120010`) containing **770 stored** `CScatteringTransmittance__FogVolume`
  (`0x81907918`, 512 bytes, 45 members) records and **384 distinct serialized records**:
  - **230 smaller resources** (`{1: 80, 2: 146, 3: 3, 4: 1}`, median `2`) carrying **385** stored records (`384`
    distinct).
  - **1 larger resource** carrying **385** stored records (`384` distinct) that match the 385 records from the 230
    smaller resources byte-for-byte (including multiplicities).

| Check (`packs.rpf` 0x7a, 770 stored / 384 distinct across 231 resources) | Result (stored occurrences) |
|---|---|
| `Position` | `(0, 0, 0)` in all 770 (local-space attachments) |
| `PositionOffset` | non-zero in 766 of 770: X `-2.46..2.46`, Y `-9.20..20.62`, Z `-6.50..6.66` m |
| `Scale` | X `0.43..6.00`, Y `0.13..28.00`, Z `0.13..2.63` m |
| `Rotation` / `Shear` | non-zero in 82 / 388 of 770 |
| `RotationMtx` and `TransformMtx` | identity in all 770 (even where `Rotation` or `Shear` is non-zero) |
| `IsBox` (`+0x000`, `ty=0x01`) | `1` in 730, `0` in 40 |
| `BlendMode` / `IntExtMode` | `1` in all 770 / `1` in 760, `0` in 10 |
| `ProfileIndex` / `SecondaryProfileIndex` | `{0: 2, 10: 4, 12: 764}` (all in `0..42`) / `0` in all 770; `ProfileInterpolation` is 0 in all 770 |
| `Type` / `SortKey` | `{0: 768, 2: 2}` / `{0: 768, 10: 2}` |

This resolves the earlier hypothesis: across all 188,449 complete `0x7a` resources in `levels.rpf`, none carries
`FogVolumeListData` or `FogVolume`. Instead, `packs.rpf` holds 231 resources totalling 770 stored local-space
records (384 distinct serialized 512-byte records: 385 in 230 per-asset resources duplicated by 385 in one larger
resource; `Position = 0`, non-zero `PositionOffset` and `Shear`, `IntExtMode = 1` in 760/770), and
`RotationMtx`/`TransformMtx` stay identity there too — confirming a loader must compose the volume transform from
`Position`, `PositionOffset`, `Rotation`, `Shear`, and `Scale`. Runtime placement identity, resource-relative
reference targets, and what `IntExtMode` and `BlendMode` select at runtime remain untested.
Not measured here (hypothesis): with half the volumes at the origin, two profiles in use and no interpolation,
this list looks like a sample or default set; the placed fog of the map is probably elsewhere (map resources
or another archive).

**`IntExtMode` and `BlendMode` against placement and interiors (ASSET-312).** Measured on the same 14 volumes
(`common.rpf`, `CScatteringTransmittance__FogVolumeListData` at `0x64c000`) and on the interior positions of
`GameShape__Data__LocationNameLoopkup` (`InteriorNames`, 2,280 `Pos`; see [world.md](world.md)), both read with
`tools/gamedata_reader.py <carve> --root <root> --out <dir>`. That document gives interior points, not bounds; no
document in `common.rpf` stores interior extents (the `CMloArchetypeDef` bounds in `levels.rpf` are in interior
space and need the placing instance). So containment cannot be tested: what is measured is only the distance
from a volume centre (`Position` + `PositionOffset`) to the nearest interior point, and a point nearby would not
show that a volume is inside an interior.

| Field value | Volumes | `Position` | Nearest interior point | Within 100 m of one |
|---|---|---|---|---|
| `BlendMode` 0 | 7 | world coordinates, none at the origin | 117.48 to 230.16 m | 0 |
| `BlendMode` 1 | 7 | exactly (0, 0, 0) in all 7, offset only | unknown (parent transform not known) | unknown |
| `IntExtMode` 0 | 14 | both groups above | as above | 0 of the 7 world-placed |

- `BlendMode` splits the list exactly by placement: 0 on the 7 world-placed volumes, 1 on the 7 at the origin.
  The world position of those 7 is unknown: their offsets are local to a parent this document does not name.
  #3873 reports `BlendMode` 1 on the 385 local-space records of `packs.rpf`; that counts record occurrences in
  one representation of the pack block, so it agrees with the split without being independent placement
  evidence. Hypothesis: 1 marks a volume attached to (and blended with) a parent transform; what it does in the
  renderer is not measured.
- `IntExtMode` cannot be correlated here: it is 0 in all 14, and none of the 7 world-placed centres is near an
  interior point. The 2 volumes that use a profile with `IntExtMode` 1 (`ProfileIndex` 10) are origin volumes
  with `IntExtMode` 0, so the volume field does not copy the profile's. Whether 1 means interior stays a
  hypothesis; deciding it needs world-placed volumes with `IntExtMode` 1 and placed interior bounds.

## Schema style

Four patterns cover almost everything:

1. **Mount/content manifests.** `CDataFileMgr__ContentChangeSetDef` has `<id>`, optional registration/map
   fields, a `<fileList>` of `<Item>` with `<filename>` and `<fileType>`, plus flags and map phases.
   `CDataFileMgr__ContentDef` is the companion content description. There are 25 such documents and 3,315
   `<filename>` elements in total. The path tokens are `$(common)`, `$(target)` and `platform:/`; no XInclude
   is used.
2. **Typed object graphs.** `CCombatInfoMgr` and the conversations/damage roots use `<Item type="...">`.
   Fields are usually child elements, scalars use `value="..."`, and enums are bare text. Across the 259
   parseable documents there are 428,611 elements, 5,293 distinct tag names, 318 distinct `type` values
   (7,181 `type` attributes) and 33 distinct attribute names. The most common are `value` (84,585), `x`/`y`
   (20,825 each), `content` (13,449), `key` (9,180), `type` (7,181), `name` (4,688) and `ref` (924).
3. **Compact numeric tables.** `CVehicleBoneOffsetsArray` stores `<packedIndices content="short_array">`
   and `<palette content="float16_array">` as whitespace-separated numbers, with `numEntries` and
   `bitsPerIndex`. `content` values seen: `short_array` (10,309), `float16_array` (2,481),
   `float_array` (536), `matrix34` (46), `vector2_array` (40), `matrix33` (28), `vec3v_array` (9).
   `CTyreDataSet` uses nested groups of named floats. These tables are why the largest single document is
   14.26 MB and spans 218 streams.
4. **Timecycle arrays.** `timecycle_modifier_data` has a version root attribute, `<modifier>` records with
   name/description/apply-tag/index/user-flag attributes, and parameter elements whose text is a
   space-separated vector (`a b`, sometimes `a b c d`). Keyframe documents add `<cycle>`/`<region>`/`<axis>`.
   This is not the `name=value` attribute style of pattern 2.

A runtime consumer needs:

- a UTF-8 XML parser that tolerates a BOM, CRLF, tabs, comments and inconsistent whitespace around `=`;
- a root-element dispatch table (118 roots, 5,293 tag names), or an RTTI/class-metadata bridge if the
  executable's class names are available;
- a polymorphic `Item` type registry for `type=`;
- enum and symbolic-reference resolution: this archive uses symbolic `ref`/`key`/`name` strings
  (924 `ref`, 9,180 `key`); only 28 `key` values look like 8-digit hex hashes, so a name registry matters more
  than a joaat table here;
- whitespace-array and float16/packed-index decoders for the compact vehicle tables;
- a mount resolver for `$(common)`, `$(target)` and `platform:/` prefixes;
- streaming rather than whole-file parsing for the 14 MB table;
- strict handling of the 9 tail-only and 1 unattributed blocks: they are incomplete evidence, not documents
  to guess at.

## Custom clouds

One `CustomClouds` document places individual weather clouds over the world. Counts and ranges; no values or
names:

| Path | Count | Content |
|---|---:|---|
| `CustomCloudGroupList/Item` | 26 | `Name` (26 distinct), `Position` (`x`, `y`), `CustomCloudList` |
| `.../CustomCloudList/Item` | 630 | `Name` (429 distinct), `Position` (`x`, `y`), `Density`, `Rain`, `Fog`, `FogHeight`, `StrikeChance`, `BurstChance`, `CoverageMinMax` (`x`, `y`), and five booleans `Nimbus0`, `Nimbus1`, `Cumulus`, `Stratus`, `CirroStratus` |

Measured over the 630 clouds:

- **Positions are absolute world x/y.** Every group `Position` is (0, 0), so the group offset cannot be
  distinguished from none, and cloud positions span x -33,515 to 28,596 and y -31,920 to 37,612. 386 of the 630
  fall inside the 2D box of the world grid atlases (x -13,312..3,072, y -9,216..24,576; #1180); 136 lie more than
  20,000 units from the origin, far outside the map placements. There is no z component.
- **Groups** hold 1 to 74 clouds (median 18).
- **Cloud types are flags, not weights.** Per cloud the number of `true` flags is 0 (91 clouds), 1 (418), 2 (55),
  3 (26), 4 (3) or 5 (37). Clouds per type: `Cumulus` 357, `Nimbus0` 158, `Nimbus1` 131, `CirroStratus` 84,
  `Stratus` 73.
- **`CoverageMinMax` is a distance range, not a fraction**: min <= max in all 630, values 0 to 30,000, medians 200
  (min) and 2,000 (max), so it reads as an inner/outer radius in world units.
- `StrikeChance` and `BurstChance` are within 0..1 in all 630. `Rain` and `Fog` are within -1..1 (median 0),
  `Density` within -100..100 (median 0.5) and `FogHeight` within -100..300.

The cloud renderer's settings sit in one `STS_Config` document with 44 top-level sections: four cloud layers
(`Nimbus`, `Cumulus`, `Stratus`, `Alto`) that share one set of 13 fields (noise weights and blends, erosion,
density, lighting and multi-scattering density scale, shape fade, transition and altitude ranges, dilation,
noise displacement); six media (`GlobalFog`, `LocalFog`, `Rain`, `Clouds`, `CloudsCirrus`, `CloudsHaze`) that
share another set of 9; and 34 single-value or vector settings. Of the five cloud flags above, `Cumulus` and
`Stratus` match a layer name, `Nimbus0`/`Nimbus1` would both fall on the one `Nimbus` layer, and
`CirroStratus` has no layer of its own (`CloudsCirrus` is a medium, not a layer). The weather data's
`CloudType` has 14 distinct fractional values from 0.01 to 0.67 over the 25 weathers.

### Custom-cloud map draw (ASSET-337)

One atmospheric technique in the 1.0.2.1 shader dictionary lists a cloud-map compute pass, a custom-cloud
vertex/pixel draw and a cloud-density compute pass in consecutive entries, in that order. The density
program has two distinct resource rows whose compiled and authored name hashes match `CloudMapA` and
`CloudMapB`. Across the resource and constant-buffer member rows of all 932 effects and 26,503 programs,
neither hash column has an exact match for `Rain`, `Fog`, `Density`, `CoverageMinMax` or `CloudType`.

The draw reads three consecutive words from a cloud record. They hold a half-precision coefficient and a
16-bit selector, two half-precision radii, and a half-precision center pair. For distance `d` from that
center in map coordinates and `outer > inner`, the pixel contribution before render-target blending is
`coefficient * clamp((outer - d) / (outer - inner), 0, 1)`. Selectors 0 through 3 route it to R, G, B
or A of the first target. Selector 4 writes it to both R and G of the second target.

A Vulkan replay used the translated draw with synthetic records and blending disabled. Switching only the
coefficient from +0.75 to -0.75 reversed all 1,804 nonzero pixels exactly. A coefficient of -1 also produced
a negative contribution. Equal positive radii gave a hard edge at the sampled distances, all strictly inside
or outside the radius. Across fifteen 64×64 readbacks, the largest absolute error against the expected
pixel values was below 0.00048.

The missing link is the producer that converts an XML cloud into these draw records. The world-to-map conversion
for `CoverageMinMax`, the fields that supply the coefficient and selector (including the two `Nimbus` flags),
and any handling of negative `Rain`, `Fog` or `Density` before the draw remain open. The replay also leaves
the live target formats, overlap blending, and the relationship between the weather's `CloudType` and the
two maps unmeasured.

## XML catalog

`tools/gamedata_xml.py` turns a carve into a per-file and per-root catalog. It is stdlib-only and rebuilds
multi-stream documents with `gamedata_index.py`, so it reads the same 267 XML payloads the index reports:

```powershell
python -I tools/gamedata_xml.py <work>\carve\common --json xml.json --csv xml.csv --check
```

The file rows carry offset/bytes/chunks, root, category, element count, maximum depth, attribute names, `$(name)`
and `platform:/` counts, include-like counts and the well-formed/fragment flags. The root rows add the
element-name and attribute histograms and the reference targets. `--check` separates a rootless fragment, a
tail-only document and a malformed document instead of treating all three as parse failures.

Measured over the 267 payloads (43,284,925 B):

| Measure | Value |
|---|---:|
| Well-formed | 259 (257 with a declaration, 2 complete without a declaration) |
| Truncated at the start | 7 |
| Rootless fragment | 1 |
| Malformed | 0 |
| Named roots / catalogued roots | 118 (+1 unattributed) / 118 |
| Elements / distinct element names | 428,611 / 5,293 |
| Attribute names / `type` values / `content` kinds | 33 / 318 / 7 |
| Maximum depth | 17 |
| Comment starts (all / declaration-bearing) | 507 / 412 |
| `filename` / `parentPath` / `parentFile` / XInclude | 3,315 / 40 / 3 / 0 |
| `platform:/` references (levels / other) | 2,674 (2,346 / 328) |
| `$(common)` / `$(root)` / `$(target)` references | 588 / 28 / 23 |

Macro counts cover all element text; 587 of the 588 `$(common)` hits are in `<filename>` elements, which is the
filename-only bucket in the table above.

The root and element distribution is the one in the tables above: the five repeated roots hold 141 of the 266
documents, and the element histogram is dominated by the general node names `Item` (118,179), `item` (16,316)
and `Name` (12,850). The long tail is per-system parameter names; the manifest and timecycle names are a small
part of it. A category is a dispatch bucket, not a schema: for the 266 documents the catalog reproduces the
file and root counts in the functional table and assigns each root one category. No root has more than one
handler.

Reference handling is narrow. Every `$(...)` and `platform:/` string is element text, not an attribute; no
document uses XInclude. The macro alphabet is `$(common)`, `$(target)` and `$(root)`, and the platform targets
split into `platform:/levels/` and everything else. `filename` is the only include-like element with a large
population; `parentPath`/`parentFile` are the only relative references outside it. A resolver therefore needs
four prefix rules and a content-set/mount walk, not a general include processor.

A runtime dispatcher needs the 118-root table (or an RTTI class bridge), the 12 category buckets, the 318-value
`type` registry, the 33-name attribute whitelist, the `content=` array decoders and a streaming reader for the
14 MB document. The mount/reference half of that layer is measured separately in
[gamedata_graph.md](gamedata_graph.md): 25 manifest documents, 30 content-set ids, 12 resolved include edges
and 3,343 reachable target references. It also needs to mark the 9 tail-only documents and the one rootless
block as incomplete evidence: `gamedata_xml.py --check` is the reference for that split. The catalog is a local work product. Only
aggregate counts and schema/root/element/attribute names belong in git; the CSV/JSON, the element values, the
`type`/`content` values and per-file paths stay on the local machine. The tests are synthetic.

## Weather types

One `CContentsOfWeatherXmlFile` document defines the weather system. Element tree (counts per path; no values
or names):

| Path | Count | Content |
|---|---:|---|
| `WeatherTimeCycleFileNames/Item` | 25 | one timecycle file path each, all with the same path prefix |
| `WeatherTypeDatas/Item` | 25 | `Name`, `TimeCycleName`, 72 numeric `value=` fields (wind, gusts, rain, snow, fog, sandstorm, ripples, ocean and shore waves, foam, puddles, accumulation), one-token `LocalizedDensityMap`, `CloudMapA` and `CloudMapB`, an empty `Variations` on all 25, and `PedDirtDefinition` / `DirtCannedColor` (set on 1 of 25) |
| `WeatherTypes/Item` | 168 | `Name`, `ShortName`, `DataName`, eight `Mod*` tags, `WeatherAppDataName` (151), `SkipDebugCycling` |
| `WeatherAppTypeDatas/Item` | 26 | `Name`, `Icon`, `IconNight` (23), `Background`, `Description` |
| `Mod*List/Item` | 73 | allowed tags: Atmosphere 17, Precipitation 16, Temperature 9, Humidity 5, Wind 11, Fetch 5, Misc 7, Niceness 3 |
| `WeatherCycles/Item` | 2 | `CycleName`, `TimeMult` |
| `WeatherTypeIndirectionList/Item` | 15 | `From`, `To` |
| `WeatherMiscData` | 1 | four `Honor*` thresholds and times |

How the pieces link, checked by name (case-insensitive) over the archive:

- **25 weather datas = 25 timecycles.** Every `WeatherTypeDatas/Item` has `TimeCycleName` equal to its own
  `Name`, the 25 are distinct, and each matches the `cycle` `name` of one of the 27 `timecycle_keyframe_data`
  documents. The 25 matched documents are exactly the ones with 1,142 variables per region; the two keyframe
  documents with 515 and 520 variables are not referenced. All 25 file paths contain one of the 25 names;
  19 are in the same order as the data list.
- **168 weather types over 25 datas.** Every `WeatherTypes/Item` has a distinct `Name` and a `DataName` that
  resolves to one of the 25 datas, and every data is used. A weather type is therefore a named variant of a
  data entry, not a separate timecycle. 151 types carry all eight `Mod*` tags and a `WeatherAppDataName`; the
  other 17 carry none, and 16 of those have `Name == DataName` (the plain base weathers).
- **Mod tags resolve to their lists**, with three exceptions: Atmosphere, Precipitation, Temperature, Humidity,
  Fetch and Niceness resolve 151/151; Wind 150/151 (the miss is two characters from a list entry); Misc only
  14/151, because 137 types use one of two values that are not in `ModMiscList`. A loader should not reject a
  type over an unknown `ModMisc`.
- `WeatherAppDataName` resolves 151/151 to `WeatherAppTypeDatas`. In the 15 indirection pairs, 14 `From`
  and 14 `To` values are weather type names, and 13 pairs have both.
- The two `WeatherCycles` names occur in other `common.rpf` documents (2 and 1 documents), not as weather types.

Measured further in [weather.md](weather.md): the `Mod*` axes against the numeric fields, the exact-case
vocabulary mismatches, `TimeMult`, and the `Honor*` values.

## `OceanSimPresets`

`tools/gamedata_xml.py` lists `OceanSimPresets` among the time/weather/water roots. There is one such document.
Its element tree (counts per path; no values or names):

| Path | Count | Content |
|---|---:|---|
| `Presets/Item` | 10 | one preset per sea region |
| `Presets/Item/Name`, `Id` | 10 each | region label and id |
| `Presets/Item/Location` | 10 | `x`, `y`, `z` attributes in world units. The 10 elevations are distinct. Horizontal extents are x -7,995 to 3,183 and y -4,572 to 8,645. |
| `Presets/Item/Direction` | 10 | wave direction |
| `Presets/Item/Octaves/Item` | 30 | exactly 3 per preset |
| `Presets/Item/Octaves/Item/ControlPoints` | 30 | 4 `(x, y)` pairs each, both in 0..1 |
| `WavefrontShapeA`, `B`, `G`, `R` `/Item/ControlPoints` | 1 each | one curve each |

The presets supply three octave curves per world position. A separate timecycle input,
`water_ocean_beaufort_scale`, occurs 676 times across 26 keyframe documents and ranges from 0 to 10.

The authored curves and locations provide two controls for a runtime capture.

- Hold position fixed at an authored site and compare the octave amplitudes at Beaufort 10. All 30 curves
  have strictly increasing x and endpoints at x = 0 and x = 1. Under piecewise-linear interpolation, 19
  differ by more than 0.000001 between inputs 1 and 5/6, the predictions for division by 10 and by 12.
  The other 11 are unsuitable as controls at that weather setting. The interpolation is an assumption of this test.
- Hold the normalized input at 1. In the horizontal (XY) plane, 11 of the 45 pairs of sites have a midpoint
  closer to the pair than to any third site. At 49% and 51% along each of those 11 segments, the nearest site switches. In every
  case at least one octave's endpoint amplitudes differ by more than 0.000001 across the pair. A hard
  nearest-site rule predicts a step there. Samples approaching the midpoint from both sides can distinguish
  that prediction from a continuous blend.
- To test whether elevation enters the selection, hold XY at one of those midpoints. Evaluate the 3D distance
  first at one site's elevation and then at the other's. The unique nearest site switches for all 11 pairs.
  A selector using only XY sees identical inputs for both measurements.

The ocean simulation technique has six compute programs. Four declare the same nine constant-buffer members
in the same order and at the same byte offsets. In all four, the pipeline reflection binds one constant buffer
at `b5, space1`. The other two declare zero members and zero constant buffers. To reproduce the program
selection, run `python -I tools/sgd5_effects.py <shaders.rpf> --effect OceanWaveSim --technique oceanWaveSim`.
Compare the member keys and offsets in each program's counted trailer against the other passes, then check
the constant-buffer register in its `PSV0` reflection.

Which preset applies at a position, how Beaufort maps to the curve input, and what the four `WavefrontShape`
curves drive remain open. The reflected member keys give a concrete target for tracing the writer or comparing
dispatch inputs with position and weather varied independently.

## Timecycle documents

Two of the three most repeated roots are the timecycle tables: 27 `timecycle_keyframe_data` and 49
`timecycle_modifier_data` documents. All 76 parse. Counts below; no values, modifier names or region names.

**Keyframe documents** (root attribute `version`, the same value in all 27):

| Path | Count | Content |
|---|---:|---|
| `cycle` | 27 | exactly one per document; attributes `name`, `description`, `regions` |
| `cycle/region` | 54 | exactly two per cycle; attribute `name` |
| `cycle/region/<variable>` | 59,170 | one element per variable, text = **13** space-separated numbers |

Every one of the 59,170 variables has 13 values, the same as GTA V's 13 time-of-day samples, so a keyframe
document reads as one weather's day curve per variable and region. In 25 documents each region has
**1,142 variables**; the other two have 515 and 520. Within every document the two regions list the same
variables in the same order. The 1,142-variable set and the two short documents together use 1,301 distinct
names: 362 of the 521 short-document names are in the 1,142 set and 159 are not, so the short documents
follow an older or separate variable list rather than a prefix of the current one. A loader should therefore
bind variables by element name, not by position.

**Modifier documents** (root attribute `version`, the same value in all 49):

| Path | Count | Content |
|---|---:|---|
| `modifier` | 4,593 | attributes `name`, `description`, `groupApplyTag`, `localApplyIndex`, `userFlags` on all; 4,592 distinct names |
| `modifier/<variable>` | 29,738 | 29,727 hold 2 numbers, 11 are empty |
| `modifier/submodifiers/item` | 8,043 | in 1,928 modifiers (4 items is the usual count: 1,252); attributes `flags` (29), `alpha` (14), `blendSpeed` (3) |
| `modifier/axis` | 405 | in the 302 modifiers that carry `numAxes` and `numKeyframes`; attribute `controlValue` on all |
| `modifier/axis/keyframe_t_value` | 302 | exactly one per axis modifier (on one of its axes), `numKeyframes` numbers |
| `modifier/axis/<variable>` | 2,713 | `numKeyframes` numbers each |

Modifiers come in two shapes:

- **Plain** (4,291): variables are direct children with two numbers each, as in GTA V's modifier files.
  Per modifier: 0 to 407 variables. 1,025 have none: 759 of those only list submodifiers and 266 are empty. The 1,054 modifier variable
  names include 947 keyframe variables and 107 that occur only in modifiers.
- **Axis-driven** (302): every variable sits inside an `axis` and has exactly `numKeyframes` values; the axis's
  `keyframe_t_value` lists the input positions (one list per modifier, even with several axes). All 2,713 axis variables match their modifier's
  `numKeyframes`, and all 302 `keyframe_t_value` lists have that length and are sorted ascending.
  `numAxes` is 1 (235), 2 (49) or 4 (18) and always equals the number of `axis` children
  (`(numAxes, numKeyframes)` most often `(1, 6)`: 93), and `numKeyframes` ranges from
  2 to 25 (2 to 7 in 271 of the 302; 6 alone in 107). `controlValue` takes 43 distinct symbolic names, so the input is a named game
  quantity rather than time of day. 61 axes add `input_min`/`input_max`, and 7 add `blendType`; 23 of the
  302 position lists fall outside `[input_min, input_max]` or, without those attributes, outside 0..1, so
  the range is not a clamp on the stored positions.

`submodifiers/item` text is one token. 8,042 of the 8,043 items equal the `name` of a modifier in the same
49 documents, so a submodifier is a by-name reference to another modifier, with its own blend attributes.
The one unresolved item points outside this archive or at a document lost to the unread region.

**Sample hours come from data.** One `time_data` document (root attributes `version` and `mode`) has exactly 13
`sample` elements, each with `name`, `hour`, `uw_tc_mod` and `interpolation_type`, plus one `region`. The 13
hours are integers from 0 to 22 in strictly increasing order, with gaps of 1 hour (7 times), 2 (twice), 3 (once)
and 4 (twice); `interpolation_type` is `EASE_LINEAR_IN` on 9 samples and `EASE_QUARTIC_INOUT` on 4, and all 13
`uw_tc_mod` values are the same name, which resolves to one of the 4,592 modifier names. The `region` name equals one of the two keyframe
region names. So unlike GTA V, where the 13 sample times are compiled in, this build stores them in `time_data`;
their count matches the 13 values of every keyframe variable. The `interpolation_type` labels are stored metadata:
how the runtime selects the interval between two samples, and whether it applies the label, is not verified.

The `hour` attributes of the 13 `sample` elements, in document order, are **0, 4, 5, 6, 7, 9, 12, 16, 18, 19, 20,
21, 22**. They come from the one `time_data` payload of the `common.rpf` carve, at archive offset 1,773,568
(`0x1b1000`): a single zlib stream, 268 B stored and 1,370 B inflated. The samples at hours 4, 20, 21 and 22
carry the `EASE_QUARTIC_INOUT` label. That value *i* of a keyframe variable belongs to sample *i* is a
hypothesis: it rests on the equal count and the document order, not on the executable. To export the sample
records as JSON from a carve (the CLI prints summary counts; the JSON is game data and stays local):

```
python -I tools/gamedata_reader.py <work>/carve/common --root time_data --out <scratch>
```

Not measured here (hypothesis): how the two regions are chosen at runtime, what the two plain-modifier numbers
are (GTA V reads them as a value pair blended by strength), and how `controlValue` maps onto game state.

**The two plain-modifier numbers.** Measured over the 29,727 two-number variables of the plain modifiers
(1,051 variable names, 3,257 modifiers with at least one such variable):

- The second number is `0` in 26,186 pairs (88 %) and `1` in 3,104; only 437 take another value. In 2,300 of
  the 3,257 modifiers it is `0` on every variable, in 21 it repeats the first number on every variable, and
  936 mix. Per variable name, 624 names never have a non-zero second number (8,410 pairs), 83 always do
  (1,289 pairs) and the other 344 vary from one modifier to the next.
- The pair is not an ordered range: the first number is larger in 20,833 pairs, smaller in 4,604 and equal in
  4,290.
- Where the second number is non-zero (3,541 pairs) it does not follow the first: Pearson correlation 0.009,
  and the first is `0` in 570 of them.
- Against the keyframe documents (25,732 pairs with a second number of `0` whose name is also a keyframe
  variable): `0` lies outside every value that variable takes in the 27 keyframe documents in 17,041 pairs,
  but the first number is outside that range too in 15,183 of the same pairs. Plain-modifier values are not
  bounded by the keyframe ranges, so this comparison does not tell what either number means.

So `0` is the most frequent stored value of the second number. What it means, and how the runtime blends the
two numbers by strength, is not settled by these counts.

Not measured here (hypothesis): the hour of each of the 13 samples (GTA V's are fixed in code), how the two
regions are chosen at runtime, what the two plain-modifier numbers mean and how the runtime blends them (see
above), and how `controlValue` maps onto game state.
**Plain-modifier number pairs `(a, b)`** (`tools/timecycle_pairs.py`, measured over all **29,727** two-number
elements in the 4,291 plain modifiers across all 49 `timecycle_modifier_data` documents):

| Measure | First number (`a`) | Second number (`b`) |
|---|---:|---:|
| Range (`min .. max`, median) | `-31,869.0 .. 400,000.0` (`0.789439`) | `-3.0 .. 5,000.0` (`0.0`) |
| Distinct values (rounded to 6 dp) | 4,566 | 71 |
| `< 0` | 2,972 (10.0%) | 7 (0.02%) |
| `== 0` | 3,590 (12.1%) | 26,186 (88.1%) |
| `(0, 1)` | 10,230 (34.4%) | 88 (0.30%) |
| `== 1` | 6,973 (23.5%) | 3,104 (10.4%) |
| `> 1` | 5,962 (20.1%) | 342 (1.15%) |

Three patterns establish what the pair stores (matching RAGE's two-float `tcModData` slot):

1. **`a` carries the variable's value; `b` is zero (`0.0`) on 88.1% of entries.** Across the 1,051 distinct plain
   variables, 624 always have `b == 0.0`, 75 always have `b == 1.0`, and 343 appear with both `b == 0.0` and
   `b == 1.0` (272 of them mixed within a single document, and 579 modifiers mix `0` and `1` across their own
   variables) with the same physical range in `a`. A loader that applied `lerp(base, a, strength * b)` uniformly
   would discard 88.1% of all modifier entries.
2. **Fractional `b` in `(0, 1)` is a shared per-colour blend weight.** Only 18 variables ever take `b` outside
   `{0, 1}`; 15 of them stay in `[0, 1]` (88 entries total). Across all **1,875 / 1,875** RGB triplets
   (`*_r`/`*_g`/`*_b`) present in the same modifier, `b` is identical across all three channels (`1,554` with `0`,
   `301` with `1`, `20` fractional — `0` mismatches).
3. **Only three variables take `b` outside `[0, 1]` (349 entries, all keyframe variables), where `(a, b)` is a
   same-unit value/range pair:** `far_clip` (`172/172` with `b > 1`, `a` in `0..5,000`, `b` in `40..5,000`,
   `a == b` in 110, `a == 0` in 54, `0 < a < b` in 7, `a > b` in 1), `fog_start` (`161` with `b > 1`, `7` with
   `b < 0`, `a` in `-1,000..5,000`, `b` in `-3..5,000`, `a == b` in 94, `a < b` in 89, `a > b` in 7), and
   `fog_isotropic_phase` (`a == b` in all `10/10`, `1.0..6.0`).

Not measured here (hypothesis): the hour of each of the 13 samples (GTA V's are fixed in code), how the two
regions are chosen at runtime, the exact runtime rule distinguishing `b == 0` from `b == 1` on the 343 variables
where both occur, and how `controlValue` maps onto game state.
## Vehicle and tyre XML documents

Four roots account for 37 of the 45 documents (4,547,932 inflated bytes) in the vehicles/tyres category:
`CVehiclePoints` (32), `CVehicleBoneOffsetsArray` (2), `CVehiclePerformanceMaps` (2), and `CTyreDataSet` (1).
All 37 parse cleanly with no truncation errors. Only counts, element/attribute names, encodings, and numeric
ranges are recorded below; no vehicle names or entry values are included.

### `CVehiclePoints` (32 documents, 33,516 B)

Each document fits in one stream (370 to 10,425 B, median 592 B) and holds one vehicle entry (31 distinct
`DefinitionName` values across the 32 documents; `DefinitionName` equals `ModelName` in all 32). Element tree:

| Path | Count | Content |
|---|---:|---|
| `DefinitionName`, `ModelName` | 32 each | text identifier (equal in all 32) |
| `Flags`, `iNumBoardingPoints` | 32 each | `value="0"` in all 32 |
| `Points`, `Props`, `AuthoredMeshName` | 32 each | empty in all 32 |
| `fNavOverrideRecastCellSize` | 32 | `value=` float `0.0000..0.0625` (median `0.0312`, 4 distinct values) |
| `fMinPatchAreaBeforeCulling` | 31 | `value=` float `0.1000..4.0000` (median `1.0000`, 6 distinct values) |
| `fNavMaxHeightChange`, `fNavMinZDist` | 30 each | `value=` constant `0.35` and `0.50` |
| `fNavOverrideAgentRadius`, `fNavOverrideAgentHeight` | 30 each | `value=` `0.00..0.15` (4 values) and `0.00..1.20` (2 values) |
| `fNavOverrideRecastCellHeight` | 26 | `value=` `0.00..0.04` (2 values) |
| `ModDefs` / `ModDefs/Item` | 6 / 49 | 1 to 32 mod entries per document (median 4) |

Every `ModDefs/Item` carries `ChassisModName` (49/49 non-empty), `PropsetExteriorModName` (35/49 non-empty),
`UtilityModName` (32/49 non-empty), `InteriorModName` (4/49 non-empty), and `CargoModName` (0/49 non-empty);
17 items also include `RoofModName` (8 non-empty). Cross-referencing `ModelName` against the two
`CVehiclePerformanceMaps` documents matches 20 keys in the watercraft map and 4 in the land/air map, matching
the navmesh-override fields used for walkable/boardable hulls and large vehicles.

### `CVehicleBoneOffsetsArray` (2 documents, 4,303,883 B)

Two documents share the `CVehicleBoneOffsetsArray` root (`boneOffsets/Item`, 2,481 entries total) with a clean
split between identifier fields:

| Document | Inflated B | Streams | `boneOffsets/Item` | Identifier field | `attachBones/Item` per entry | `numEntries` per entry |
|---|---:|---:|---:|---|---:|---:|
| Doc 0 | 4,283,195 | 66 | 2,447 | `name` (2,447/2,447; `vehicleName` empty) | 8..3,200 (median 8; 66,128 total, 9,345 non-empty) | 24..9,600 (median 24; 740,376 total) |
| Doc 1 | 20,688 | 1 | 34 | `vehicleName` (34/34; `name` empty) | 4 in all 34 (136 total, all non-empty) | 12 in all 34 (408 total) |

33 of Doc 1's 34 `vehicleName` entries match keys in `CVehiclePerformanceMaps` Doc 0 (0 in Doc 1).

Each `boneOffsets/Item/offsets` block stores a palette-compressed float16 table with four elements: `numEntries`
(`value=`), `bitsPerIndex` (`value=`), `packedIndices` (`content="short_array"`), and `palette`
(`content="float16_array"`). Measured over all **2,481 / 2,481** tables (740,784 unpacked entries):

- **`numEntries` divisibility:** exact multiple of `len(attachBones)` in all 2,481 entries (remainder 0) and exact
  multiple of 12 in all 2,481 entries. `numEntries / len(attachBones)` is `3` in 2,319 entries (including all 34
  of Doc 1 — one 3-component vector per bone slot); the remaining 162 entries take multiples of 3 from `9` to
  `1,200` (`9, 12, 27, 36, 48, 75, 108, 300, 1200`).
- **Bit width and palette:** `bitsPerIndex` is `1` (253 tables, palette length 2), `2` (1,951 tables, length
  3..4), `3` (258 tables, length 5..8), or `4` (19 tables, length 9..14) — always minimal for `len(palette)`.
  Palette values span `-30.90625` to `2.542969`, with `0.0` present in 2,119/2,481 palettes.
- **Exact packing rule:** `packedIndices` holds 107,229 non-negative unsigned 16-bit integers (`0..65535`)
  encoding a continuous **LSB-first bitstream** across 16-bit words. The array length equals
  `floor(numEntries * bitsPerIndex / 16) + 1` in **2,481 / 2,481** tables (when `numEntries * bitsPerIndex` is a
  multiple of 16 — 2,043 tables — the writer appends one trailing `0` word; all 2,043 trailing words and all
  unused tail bits across all 2,481 tables are `0`).
- **Bounds check:** unpacking all 740,784 indices produces **0 out-of-bounds** palette indices.

### `CVehiclePerformanceMaps` (2 documents, 133,848 B)

Two documents with disjoint `Map/Item[@key]` sets (285 distinct keys total, 0 overlap). Every `Map/Item` carries
the exact same 10 scalar `value=` children: `topSpeed`, `accelZeroTwenty`, `accelZeroSixty`, `brakingSixtyZero`,
`turnRadius`, `maxGforceLateral`, `maxGforceSteering`, `idlePitchAngle`, `buoyancyOffsetZ`, and `takeOffSpeed`.

| Document | Inflated B | Streams | Entries | Non-zero road braking/turn/G | Non-zero `buoyancyOffsetZ` | Non-zero `takeOffSpeed` | `topSpeed` range (median) |
|---|---:|---:|---:|---:|---:|---:|---|
| Doc 0 (land/air) | 117,808 | 2 | 251 | 203..223 | 0 | 10 (max `57.64`) | `0.00..115.96` (`45.32`) |
| Doc 1 (watercraft) | 16,040 | 1 | 34 | 0 | 34 (`-1.11..1.91`) | 1 (`21.54`) | `0.00..55.05` (`16.82`) |

### `CTyreDataSet` (1 document, 76,685 B, 2 streams)

One document containing 40 `TyreDataEntries/Item` records (40 distinct `key` attributes, max tree depth 5). Each
entry combines up to four nested parameter groups with 12 direct scalar `value=` fields:

| Path under `TyreDataEntries/Item` | Present | Schema |
|---|---:|---|
| `roadPerformance` | 40 / 40 | 8 scalars: `fPeakSlipAngle`, `fAdhesionLimitAngle`, `fLateralFrictionMult`, `fPeakSlipRatio`, `fAdhesionLimitRatio`, `fLongitudinalFrictionMult`, `fPeakForceCoefficient`, `fSlidingForceCoefficient` |
| `offRoadPerformance` | 38 / 40 | identical 8-scalar schema (omitted on 2 entries) |
| `roadPenalties`, `offRoadPenalties` | 40 / 40 each | identical 6-scalar schema: `fMaterialTractionLoss`, `fMaterialRollingDragGain`, `fMaterialSinkDragGain`, `fMaterialSurfaceDragGain`, `fWetTractionLoss`, `fWetDragGain` |
| Direct scalars (required) | 40 / 40 each | `fLoadShapeVarianceScale`, `fDamageShapeVarianceScale`, `fDamageCoefficientScale`, `fOptimumTyreTemp` (all `100.0`), `fOptimumTyreTempRange` (all `100.0`), `fGripDropOffWithTemp` (`0.01..0.05`) |
| Direct scalars (optional) | 36..39 / 40 | `fPuncturedTyreGripReduction`, `fBurstTyreGripReduction`, `fTractionStiffnessSpeedMult`, `fTractionStiffnessSpeedCap` (39 each); `fLoadStability` (37); `fLoadSensitivity` (36) |
**Sample hours come from data.** One `time_data` document (root attributes `version` and `mode`) has exactly 13
`sample` elements, each with `name`, `hour`, `uw_tc_mod` and `interpolation_type`, plus one `region`. The 13
hours are integers from 0 to 22 in strictly increasing order, with gaps of 1 hour (7 times), 2 (twice), 3 (once)
and 4 (twice); `interpolation_type` is `EASE_LINEAR_IN` on 9 samples and `EASE_QUARTIC_INOUT` on 4, and all 13
`uw_tc_mod` values are the same name, which resolves to one of the 4,592 modifier names. The `region` name equals one of the two keyframe
region names. So unlike GTA V, where the 13 sample times are compiled in, this build reads them (and the easing
between them) from `time_data`; their count matches the 13 values of every keyframe variable.

Not measured here (hypothesis): how the two regions are chosen at runtime, the exact runtime rule distinguishing
`b == 0` from `b == 1` on the 343 plain-modifier variables where both occur (see *Plain-modifier number pairs*
above), and how `controlValue` maps onto game state.

## Relationship to `data.rpf` and `script.rpf`

| Archive | Size | Streams | Inflated | Dominant payloads | XML | `RBF0` | `PSIN` |
|---|---:|---:|---:|---|---:|---:|---:|
| `common.rpf` | 6.6 MB | 954 | 50.2 MB | XML/config, text, `MHIU`, `PAMH`, `SPS1` | 844 streams | 0 | 0 |
| `data.rpf` | 481.5 MB | 28,509 | 1,403.3 MB | models/GPU pages, 1,263 `RBF0`, 2,281 `PSIN`, 3,132 resource pages | 0 | 1,263 | 2,281 |
| `script.rpf` | 4.4 MB | 714 | 5.0 MB | 671 `PSIN` plus binary | 0 | 0 | 671 |

There is **no XML, `RBF0` or `PSIN` duplication** between them: `common.rpf` contains no `RBF0`/`PSIN`, and
`data.rpf`/`script.rpf` have no XML documents at all — no stream starts as XML and no inflated stream contains
an XML declaration (the ten `<`-leading `data.rpf` streams are binary). This is a type-level and stream-start
check, not a byte hash of all 1.4 GB of inflated `data.rpf`; without the table of contents a duplicate stored
at a different block boundary would not be recognised.

The manifests in `common.rpf` reference the other archives by path token, not by archive handle. Counted over
the 3,315 `<filename>` elements:

| Reference bucket | Count |
|---|---:|
| `platform:/levels/` | 2,346 |
| `$(common)/` | 587 |
| other `platform:/` | 328 |
| relative/other | 31 |
| `$(target)/data/` | 23 |
| `$(target)/script/` | 0 |

So `common.rpf` is the XML/config layer that mounts levels content and references 23 `data.rpf` paths; none
of its `<filename>` entries use the `script` token. It may still reference script content through non-path
content names, which this count does not cover. The 587 `$(common)/` entries are mostly movie paths (537 end
in `.bk2`), not XML. Of the 57 `.xml` path entries, 50 are under `$(common)/` and 7 are relative/other; none
are under `data/` or `script/`. A further 40 `parentPath` attributes are XPath-style
`../Item[@key='...']` steps that all resolve to sibling items inside their own document, so they need no table
of contents (see [gamedata_graph.md](gamedata_graph.md#what-the-locators-point-at)). No XInclude (`xi:include`) elements occur.

## Shore wave placement

The shore-break waves are placed by one `ShoreWavePlacementData` document, the coastline counterpart of
`OceanSimPresets`. Element tree (counts per path; no values):

| Path | Count | Content |
|---|---:|---|
| `Context` | 1 | empty |
| `Patches/item` | 434 | `loop` and `waveType` (`value=`), then `ControlPoints` |
| `Patches/item/ControlPoints/item` | 7,839 | `position` and `direction` (`x`, `y`), `lateralCoord`, `maxDistance`, `amplitude` (`value=`) |

Measured over the 434 patches and 7,839 points:

- **Patches are open 2D polylines.** 2 to 124 points per patch (median 14), median spacing 9.5 world units.
  `loop` is false on all 434; 11 patches still repeat their first point at the end. Positions span x -9,602 to
  2,335 and y -7,955 to 9,474, a larger box than the ten ocean preset locations. Positions have no z
  component.
- **`waveType`** takes 5 values, used by 181, 147, 69, 31 and 6 patches.
- **`direction` is a unit vector** at every point (length 1.000 to three decimals). It runs across the polyline
  rather than along it: the median |cos| between it and the local polyline tangent is 0.24 (quartiles 0.11 and
  0.44), consistent with a wave travelling across the line towards the shore.
- **`amplitude` is 1.0 at all 7,839 points**, so it carries no per-point variation in this build. The height
  presumably comes from elsewhere, for example the per-weather `ShoreWaveAmplitude`, `ShoreWaveMinAmplitude`,
  `ShoreWaveMaxAmplitude` and `ShoreWaveWindScale` fields of the `CContentsOfWeatherXmlFile` weather data
  (its layout is documented in #1175).
- **`maxDistance`** ranges 3.0 to 120.4 (median 74.9): how far the wave runs from the line.
- **`lateralCoord`** is a float in -2.94..3.50 with 7,820 distinct values. It is a coordinate *along* the line,
  not across it: within a patch it is linear in arc length (distance along the polyline from the first point),
  `lateralCoord ≈ a + b × arc length`.
  - The least-squares fit leaves a median largest residual of 0.012 per patch (90th percentile 0.059).
  - The per-patch correlation with arc length has magnitude at least 0.994 on more than three quarters of the
    patches.
  - The rate is the same everywhere: |b| has a median of 0.00200 per world unit, so it changes by 1.0 every
    500 world units (median 500.4; 10th to 90th percentile 473 to 544). The median rate per `waveType` is 0.00199
    to 0.00202, so the wave type doesn't set it.
  - It *decreases* along the point order: strictly on 416 of the 434 patches, and b < 0 on 426. Only 197 of the
    7,405 point-to-point steps go against the patch's direction, by a median of 0.018. (It is monotonic, just not
    increasing; an earlier count here only tested increasing.)
  - On those 426 patches, `direction` points to the left of the polyline, so the coordinate runs the same way
    relative to the waves everywhere.
  - The offset a spreads over -2.0 to 3.4 (median 0.70), and a whole patch changes it by a median of -0.33. So
    neighbouring patches can continue one another rather than each starting at 0 (10 start within 1e-4 of 0).

  Hypothesis, from the shape rather than a consumer: a phase or texture coordinate along the coast that repeats
  every 500 world units.
- **`waveType`** doesn't change `amplitude` (1.0 at every point of every type) or the `lateralCoord` rate, and the
  types overlap across the map. What differs:
  - Type 3 (147 patches) has the shortest reach: its `maxDistance` median is 64.6, with the only values below
    17 (down to 3.0). Types 0, 1 and 4 have medians of 72.4 to 78.1.
  - The six type 2 patches lie in one area (patch centres x 358 to 1,560, y -1,685 to 6,041). The other types
    span the whole coastline box.

Not measured here: what the five `waveType` values select. Reach is the only measured difference besides type 2's
location; a consumer of the field would settle it.

## Readable vs binary

Readable: the 266 XML documents, 3 text/INI-like files, and the two `MHIU` 2048 x 2048
heightmaps plus the `PAMH` max/min heightmap and the `SPS1` shader-preset table. The XML is ordinary text
and can be consumed by any parser. `tools/gamedata_maps.py` decodes the `MHIU` row table and
packed heights and the `PAMH` row table and max/min arrays, and `tools/gamedata_sps1.py` decodes the
`858`-entry `SPS1` preset table and joins it to `shaders.rpf`'s `SGD5` effect records.

Not fully readable: the 8.6% unread span, the two opaque binary blobs, and the missing first blocks of the 9
tail-only documents (their retained tails are readable but incomplete). The encrypted table of contents is
outside this work by rule; names and exact file boundaries stay unknown until a documented table reader exists.

## Text and binary tables (`tools/gamedata_blobs.py`)

`tools/gamedata_blobs.py` decodes all five non-XML text and binary payloads in `common.rpf` (counts and structural
bounds only; zero names or raw bytes):

| Payload | Inflated B (streams) | Format & measured structure |
|---|---:|---|
| `0x64e000` (`.fx` preload list 0) | 24,285 (1) | `965` unique `<effect>.fx` lines (`0` comments, `0` folder separators): all **910 of 910** `shaders.rpf` `SGD5` Database 0 effect records appear in exact index order (`0..909`), plus `55` unshipped `.fx` names |
| `0x64f1c0` (`.fx` preload list 1) | 575 (1) | `22` unique `<effect>.fx` lines (`0` comments): all **22 of 22** match `SGD5` Database 1 (`0x10289C81`) in exact index order (`0..21`) |
| `0x640000` (`VFX_SETTINGS_FILE` `.dat`) | 117,176 (2) | `3,198` lines: `2,566` data lines (`2,537` `<dotted.key> <scalar>` pairs, `2` `key = value` lines, `27` multi-value rows of `5..24` tokens), `248` comments, `384` blanks (`13` of `122` distinct key prefixes, `381` data lines, match an `SGD5` Database 0 effect name) |
| `0xad000` (fixed table) | 1,038,100 (16) | `u32 count = 801` followed by `801 x 1,296` B records to EOF (`4 + 801 x 1,296 = 1,038,100` B, `0` trailing bytes). Every record opens with `{u32 w0, u32 w1, u32 w2, u32 w3}` where `(w0, w1) == (w2, w3)` in **801 of 801** (`801` distinct pairs) before an undecoded `1,280`-byte record body (arithmetic permits `64 x 20` B; sub-record layout is not decoded by the tool) |
| `0x23c000` (3-region spatial grid) | 3,274,036 (50) | **Region 1** (`0x000000..0x07FFFF`, `131,072` `u32` cells): `32` empty `0xC0000000` sentinels and `131,040` populated cells whose low 23 bits point (`131,040 / 131,040` aligned) into **Region 3** (`0x166934..0x31F533`, EOF `0x31F534`, `451,328` `u32` words partitioned by the `46,019` distinct cell offsets into spans of `1..24` words); located as the trailing run of 4-byte-aligned offsets matching `min(cell_offs)` (`0x166934`), with every word (`451,328 of 451,328`) falling in **Region 2** (`0x080000..0x166933`, `944,436` B partitioned with zero gaps into `21,662` records: `21,273` of `44` B, `313` of `24` B, `76` of `12` B) |

```powershell
python -I tools/gamedata_blobs.py <work>\carve\common --shaders shaders.rpf --json blobs.json
```

## Dirt and canned colours (`CGraphicsDefinitionsData`)

One `CGraphicsDefinitionsData` document holds the dirt/grime presets for peds, vehicles and props and the
colour table they draw from. Seven sections, each a list of `Item key="..."` records (the table gives counts,
no values or names):

| Section | Items | Fields per item | `key` |
|---|---:|---:|---|
| `CannedColors` | 44 | none: `x`, `y`, `z`, `w` attributes | text |
| `AlphaChannels` | 7 | `Name`, `Channels` (1 or 2 `Probability`/`Channel` pairs; 10 in total) | equals `Name` in 6 of 7 |
| `DirtDefinitions` | 24 | 25 (alpha curves and thresholds, vertex multipliers, normal/roughness/material contribution, spawn range, 6 `ClothHideDirtModifiers`) | equals `Name` in 22 of 24 |
| `VehDirtDefinitions` | 1 | 12, including a `DirtTunings` block of tile, detail-map, mud and modulate start/end values | equals `Name` |
| `VehDirtControls` | 5 | 13 | equals `Name` |
| `VehSpawnIntensities` | 31 | Name, `IntensitySets` (7 `Item`s of `DirtMin`/`DirtMax`/`MudMin`/`MudMax`), `DirtRainChance`, `DirtRainMin`, `DirtRainMax` | equals `Name` |
| `PropDirtDefinitions` | 9 | 4 | equals `Name` |

The keys are plain text, not joaat hashes, and a loader can index each section by `key`.

**Canned colours are 8-bit RGB.** In all 44, `x`, `y` and `z` are integers from 0 to 255 and `w` is -1, so
`w` is not alpha; read it as an unused component or "no alpha".

**References** (case-insensitive equality with another section's `key` or `Name`):

| Field | Resolves to | Count |
|---|---|---:|
| `DirtDefinitions/Item/ColorLookup` | `CannedColors` | 24 of 24 |
| `DirtDefinitions/Item/AlphaMasksLookup` | `AlphaChannels` | 18 of 24 (the other 6 are empty) |
| `VehDirtDefinitions/Item/ColorLookup` | `CannedColors` | 1 of 1 |
| `PropDirtDefinitions/Item/Color0Lookup`, `Color1Lookup`, `Color2Lookup` | `CannedColors` | 9 of 9 each |
| weather data `PedDirtDefinition` (see the weather document) | `DirtDefinitions` | 1 of 1 |
| weather data `DirtCannedColor` | none | 0 of 1 |

So every dirt preset gets its colour from the canned table, and one weather selects a ped dirt preset. The
weather's `DirtCannedColor` is a single non-numeric token that matches no canned-colour key, even as a
substring; where it resolves is unknown. Other `common.rpf` documents share only a few short generic words
with the `VehSpawnIntensities` keys, which is not evidence of a reference.

`AlphaChannels` probabilities do not sum to 1 in any of the 7 entries, so they are not a probability
distribution over the listed channels (independent chances or weights would fit). The `Channel` values are 0, 1 or 2.

**`Channel` numbers the mask texture's channels in component order: 0 = R, 1 = G, 2 = B** (ASSET-304). The
dirt presets are ped presets, and `textures.rpf` holds one texture whose name marks it as the ped dirt masks
(1024×1024 BC7, 11 mips, next to the ped dirt normal/roughness map). `tools/mask_channels.py` on its export:

| Channel | Range | Mean | Cover ≥ 64 / 128 | Blobs ≥ 64 / 128 | Largest blob ≥ 64 |
|---|---|---:|---:|---:|---:|
| R | 0-255 | 124.1 | 66.5% / 49.3% | 4,612 / 9,167 | 92.0% |
| G | 90-255 | 193.6 | 100% / 99.5% | 1 / 7 | 100% |
| B | 0-255 | 88.4 | 50.7% / 34.9% | 35,175 / 49,374 | 21.0% |
| A | 255 only | 255 | 100% | 1 | 100% |

Alpha is constant, so the three `Channel` values can only be R, G and B; an order that counts alpha leaves one
value on a channel with nothing in it. G is a soft layer that never drops below 90, and the two entries whose names
describe a light dusty or urban grime use only `Channel` 1. R is a broad layer, one connected region at ≥ 64, and
the snow entry uses only 0. B is the speckled one, with 7.6 times R's blobs at ≥ 64, and the entry named for spots
uses only 2. The reversed order (0 = B) would put the spots on the broad layer and the snow on the speckled one.
The three mixed entries combine 0 and 2. Each tie is a name read against measured texture content, so this is
measured, not confirmed against the shader that samples the mask.

Not measured here (hypothesis): how `VehSpawnIntensities` are chosen at spawn.
**Spawn intensities are keyed by vehicle class.** Measured on 2026-11-23 from one retail Xbox Series
1.0.2.1 install with `tools/veh_spawn_intensities.py` (numbers and ranges only; the tool never prints
entry names). The 31 entries are 30 generic vehicle-class tokens plus one entry whose rain fields differ
from every other chance-0.5 entry; each carries 7 intensity bands, so the section's own "5 fields" summary
is top-level children only:

| Measure | Value |
|---|---|
| `IntensitySets` bands per entry | 7 in all 31 |
| distinct band signatures | 9, carrying 12, 8, 3, 2, 2, 1, 1, 1, 1 entries |
| band 7 ranges over the signatures | `DirtMin` 0.353400..0.804305, `DirtMax` 0.450900..1.240000 |
| mud (a `MudMin`/`MudMax` above zero) | in band 6 of 1 signature, band 7 of 6 signatures; everywhere it exists the range is 0.8..1.0 |
| bands overlap | yes: each band is a draw range, not a partition (band 1's `DirtMax` exceeds band 2's `DirtMin` in the largest signature) |
| `DirtRainChance` groups | 4: 0.5 in 24 entries (`DirtRainMin` 0.818644 except one at 0.7), 0.0 in 6, 0.3 in 1 |

Band 1 to band 7 in the largest signature ladder through the same recurring endpoints
(0.007828, 0.174168, 0.391389, 0.485323, 0.659491, 0.804305), so a run draws its dirt intensity from one
of the seven ranges, and rain applies another random factor gated by `DirtRainChance`. Each entry has 7
bands while `AlphaChannels` has 7 entries and `DirtDefinitions` references 5 of them; whether a spawn picks
one band per `AlphaChannels` entry is open.

Not measured here (hypothesis): whether a spawn selects one `IntensitySets` band per `AlphaChannels` entry.

## Entity component sets and component LOD

Four documents describe which runtime components an entity gets and how far away each stays alive. Counts only;
no set or type names.

| Root | Content |
|---|---|
| `CEntityComponentLodManagerData` | `EntityComponentCounts/Components/Item` x 33: `Type` (33 distinct) and `DesiredCount` (0 to 250, 2,684 in total; one type is 0), plus optional distance-scale factors for not-visible (5 and 2 items), interior (1) and deep-interior (1) entities and per-frame create/remove limits (4 and 2) |
| `CVehicleComponentSetManager` | 7 `CVehicleComponentSetInfo` sets with 11 to 14 components each (90 in all): `Type`, `MinDistance`, `MaxDistance` |
| `CPedComponentSetManager` | 15 `CPedComponentSetInfo` sets with 8 to 17 components each (173 in all), plus feature flags: `HasFacial` (true in 12), `HasRagdollConstraints` (13), `HasLookAt` (10), `HasInventory` (8), `HasPhone` (2), `HasReins` (1), `HasDistraction` (2 of 3 sets that carry it), `IsRidable` (false in all 15) |
| `CObjectComponentSetManager` | 1 `CObjectComponentSetInfo` set with an empty component list |

Each manager has a `DefaultSet ref`, and all three refs name one of their own sets.

**The component types form one closed list.** Vehicle sets use 14 distinct `Type`s and ped sets 18, with none
in common, and every one of the 32 is among the 33 types of `CEntityComponentLodManagerData`; only one LOD-manager
type is used by no set. So a runtime can treat the LOD manager's list as the component registry, with
`DesiredCount` as the pool size per type.

**Distances.** `MinDistance <= MaxDistance` in all 263 components. Vehicle components reach up to 3,000 units
and ped components up to 200 (both medians 100); the natural reading is the range within which the component is kept
alive.

Not measured here (hypothesis): whether the distances are metres from the camera or the player, and what the
not-visible and interior scale factors multiply.

## Damage and tuning XML documents

Three repeated roots in the top-ten table above cover the damage and tuning tables: `CDamageDataManager` (5 documents,
485,482 B), `CTuningFile` (10 documents, 27,718 B), and `CCoverProcessorTuning` (3 documents, 34,380 B). All 18 parse
cleanly. Counts and schemas only; no entry names or parameter values.

### `CDamageDataManager` (5 documents, 485,482 B, 10 streams)

Rather than repeating one table type, the five documents partition the damage system by sub-table:

| Document | Inflated B | Streams | Top-level section | Measured structure |
|---|---:|---:|---|---|
| Doc 0 | 257,008 | 4 | `DamageCleanlinessDatas/Item` (5) | `25` `RegionData/Item` blocks (5 per entry) containing `528` keyed `WeaponData/Item` records (`key=`), with `616` `CleanlinessResults/Item` and `328` `CleanlinessResultsLegendary/Item` pairs `{MaxBulletHits, MaxDamageCleanliness}` |
| Doc 1, Doc 4 | 104,082 / 107,078 | 2 / 2 | `DamageModifiers/Item` (147 / 96) | `243` top-level modifiers across 19 polymorphic `CDamageModifier*` classes (`CDamageModifierMultiply` 63, `CDamageModifierMultiReference` 58, `CDamageModifierClamp` 39, `CDamageModifierFirstSuccessfulReference` 36, `CDamageModifierMulti` 19, `CDamageModifierInstantKill` 5, 13 others) and `336` nested `Modifiers/Item` entries, gated by `Condition` trees across 65 `CDamageCondition*` classes (`CDamageConditionAnd` 144, `CDamageConditionNot` 126, `CDamageConditionOr` 72, `CDamageConditionFromDamageType` 60, `CDamageConditionFromWeaponType` 49, `CDamageConditionIsPlayerBuddy` 44) |
| Doc 2 | 12,498 | 1 | `DismembermentData` (1) | Dismemberment thresholds and bone/region rules |
| Doc 3 | 4,816 | 1 | `HitReactData` (13), `HitReactGripData` (2) | Hit-reaction and weapon-grip impulse parameters |

### `CTuningFile` (10 documents, 27,718 B) and `CCoverProcessorTuning` (3 documents, 34,380 B)

- **`CTuningFile` (10 single-stream documents, 266 to 11,727 B, median 1,875 B):** each wraps a `<Tunables>` list of
  polymorphic `<Item type="...">` blocks (`14` items total across 14 distinct `type` classes, including
  `CConversationManager__Tunables`, `CDeadeyeSpecialAbility__Tunables`, `CLasso__Tunables`, and
  `CPlayerFocusSpecialAbility__Experimental__Tunables`). Sub-structures include `36` keyed
  `EntityMaskConfig/EntityMaskProfiles/Item` records (`MaskValue` 4D `x/y/z/w`, `MaskValueExtras` 2D `x/y` on 33,
  `BlendOutCurveType` on 4) and a nested `ClipSetTree` (`5` `TransportMotionHelpers__ClipsetTransportConditionNode`
  branches over `8` `GameplayHelpers__ClipsetLeafNode` leaves).
- **`CCoverProcessorTuning` (3 single-stream documents: 11,554, 11,422, and 11,404 B):** flat scalar tables with
  `212`, `211`, and `210` direct `value=` children per document (`fAdditionalDistanceFromTreeToRejectAdditionalCoverpoints`,
  `fAICanAimSweepCount`, `fCornerCoverPointDotThreshold`, `fCornerDistanceCheckResolutionFine`, etc.).

## Combat, AI behaviour and weapon animation documents

The representative roots of the combat/AI (13 documents, 809,074 B) and weapon-animation (6 documents, 162,778 B)
categories use typed polymorphic `<Item type="...">` and keyed map tables:

| Root | Inflated B (streams) | Status | Measured schema |
|---|---:|---|---|
| `CCombatInfoMgr` | 237,621 (4) | well-formed | `CombatInfos/Item` × 66 (`type="CCombatInfo"`), each with `Name`, `CombatMovement`, `BehaviourFlags`, `CombatAbility`, `AttackRanges`, `CoverHeightPreference`, `WeaponShootRateModifier` (`value=`), `FiringPatternSetHash`, `AccuracyProfile`, and `CombatFloats` (`BlindFireChance` and related scalars) |
| `CFiringPatternInfoManager` | 208,240 (4) | tail-only | One of the 7 start-truncated XML payloads whose first block fell in an unread RPF9 gap; identified by its closing tag |
| `CFleeDecisionMakerData` | 17,561 (1) | well-formed | `Profiles/Item` × 8 (`key=`), each defining `{Weight, Weights}` blocks for `Cower`, `RunAway`, `Mount`, and `TakeVehicle` (8 each), plus `Fly/Weights` (14 entries) |
| `Combat__GoalData__Store` | 4,550 (1) | well-formed | `RoleSettings` (`Default` + `DataMap/Item` × 5 keyed `StyleMod` entries) and `PriorityCalculators/DataMap/Item` × 8 (`type="Combat__GoalData__EntityDistanceBasedPriorityCalculator"` with `13` `{MaxDistSq, Priority}` pairs) |
| `WeaponAnimations__FullbodyControllerData` | 21,281 (1) | well-formed | `AnimRequestConfigMap/Item` × 11 (`type="WeaponAnimations__FBAnimRequestConfig"`, `DefaultRequestType` on 11, polymorphic `RequiredCondition` on 9, `7` `Variants/Item`) and `AnimRequestDataMap/Item` × 17 (`type="WeaponAnimations__FBRequestData"`, `LocoParams` on 16, `ControlGraphParams` on 11), plus `Timeouts` (1) |
| `WeaponAnimations__ConfigData` | 1,733 (1) | well-formed | `AIHandPoseRules` and `BuddyHandPoseRules` (`type="WeaponAnimations__HandPoseRules"`), with nested `CAIConditionOr` / `CAIConditionAnd` / `CAIConditionBlackboardBool` trees under `PosePrefOverrides/ForceTwoHanded` |
| `CAimUpperBodyDataBlock` | 1,813 (1) | tail-only | Start-truncated single-stream payload identified from its closing root tag |

## Ped schedules, body variation, formations and energy

Four representative roots account for 1,032,473 B (18 streams) of the 29-document peds/schedules/appearance category:

| Root | Inflated B (streams) | Top-level sections and counts |
|---|---:|---|
| `CPedScheduleInfoManager` | 784,294 (12) | `Schedules/Item` × 4,861 (`{Name, Pack}` pairs), `PedScheduleTuning` (18 children), `PedDynamicScheduleTuning` (19 children), `Volumes/Item` × 9 (`VolumeType`, 3D `Position`/`Orientation`/`Scale`, `IsPenVolume`, and `SpawnPositions` `content="vec3v_array"`), `Routes` (1) |
| `CPedVariationBodyTypeManager` | 77,462 (2) | `BodyShapePresets/Item` × 16 containing `432` `Expressions/Item` records (`27` per preset, `{Id, Values content="float_array"}`), `ScaleVariationProfiles/Item` × 26 (`key=`, `34` `ScaleVariations/Item` with `{Min, Max, SelectionWeight, ValueType}`), `BodyShapeVariationProfiles/Item` × 22 (`key=`, `53` `BodyShapeVariations/Item` with `{BodyShapePreset, SelectionWeight}`), `ScaleProfiles` (2), `HeightPresets` (1) |
| `CPairedPedDataBlock` | 10,011 (1) | `SpacingData/Item` × 4 (seven 2D `x/y` distance pairs per item: `Spacing`, `MiddleSpacing`, `MinSpacing`, `MinSpacing_Crossing`, `MinMiddleSpacing`, `MinMiddleSpacing_Crossing`, `MaxSpacing`), `BespokeSpacingData` (2), `VehicleSpacingData` (1), `AmbientFormationData` (3, 6 2D `Pos` slots), `BespokeFormationData` (3, 11 2D `Pos` slots), `VehicleFormationData` (1) |
| `CEnergyConfigInfos` | 160,706 (3) | `HealthConfig/Item` × 63 (`key=`, `parentFile` on 3, `MeleeProperties/KnockedOutWakeUpCount` on 62, `FireVulnerability` on 61, `EnergyReserveProperties/Reserves` `content="float_array"` on 59), `EnergyReserveConfigs/Item` × 58 (`153` `EnergyReserves/Item` with `{Type, EnergyAmount, Enabled, RechargeRate, RechargeEnabled, RechargeThreshold}`), `EnergyConfig` (54), `StaminaConfig` (8), `EnergyHealthStateMap` (4), `HealthRechargeConfig` (3), `SpecialAbilityConfig` (2) |

In `CEnergyConfigInfos`, **60 of 60** `HealthConfig/Item/EnergyReserveConfig` references resolve case-insensitively to
an `EnergyReserveConfigs/Item` entry in the same document.

## World locations, pathing styles, trains and hazard nodes

Four representative roots in the world/navigation/map category (826,397 B across 15 streams; all 4 well-formed):

- **`GameShape__Data__LocationNameLoopkup` (1 document, 671,393 B, 11 streams):** (root spelling preserves the double
  `o` in `Loopkup`.) Contains four keyed lookup tables (`key=`):
  - `LocationNames/Item` × 2,318 (`Name` on all 2,318; `TextureDictionary` and `Texture` on 46).
  - `InteriorNames/Item` × 1,358, holding `2,280` nested `<Item>` records with `{Name, Pos}`. Their 2,280 3D positions
    span `x = -9,673.0..4,705.3`, `y = -7,308.0..8,589.9`, `z = -78.0..820.9` — their 2D `x`/`y` coordinates fall 100%
    inside the `x`/`y` extent of the `PAMH` heightmap (`-13,312..5,120`, `-9,728..11,264`) and `GridAtlasInfo` grids
    above, while `z` reaches `820.9` (above the `PAMH` `404` m ceiling).
  - `RoadNames/Item` × 409 (`Name` on all 409) and `RiverNames/Item` × 66.
- **`PathingStyleData` (1 document, 5,993 B, 1 stream):** `StyleProfiles/Item` × 12 (`key=`), each defining
  `CurvatureStyle`, `CorneringStyle`, `ApproachStyle`, `ContinuityStyle`, `SlowdownStyle`, `SplineConfig`, and
  `AvoidanceStyle`, plus optional `Flags` (9) and `DoorPathingStyle` (5).
- **`CTrainConfigs` (1 document, 125,494 B, 2 streams):** `trainConfigs/Item` × 46, each carrying `name`,
  `populateTrainDist`, three station stop-duration scalars (`stopAtStationDurationNormal`,
  `stopAtStationDurationPlayer`, `stopAtStationDurationPlayerWithGang`), `announceStations`, `doorsBeep`, and `1..17`
  `carriages/Item` entries (median 8, **314 carriages** total, each with `maxPedsPerCarriage`, `flipModelDir`,
  `doInteriorLights`, `modelName` on 304, and `modelSetName` on 300).
- **`CHazardNodes` (1 document, 23,517 B, 1 stream):** a 20-state hazard state machine (`Nodes/Item` × 20,
  `type="CHazardNode"`, `key=`, `Name`, `ClipSet`, `ClipVictim`, `ClipThreat`, `UseMTH`, `RumbleCurve`,
  `RumbleCurveStart`) linked by `43` `Successes/Item` transitions (`type="CHazardTransition"`, gated by
  `CAIConditionGatorPit` [36], `CAIConditionAmbientEventDirection` [5], `CAIConditionOr` [1], `CAIConditionAnd` [1]).
  All **18 of 18** non-empty transition target `<Node>` references resolve to one of the 20 `CHazardNode` keys.

## Door interaction, ped loadouts, climbing, lasso and AI rule schemas

Fourteen previously undocumented well-formed `common.rpf` documents span **1,682,656 inflated bytes** across **32**
streams. Counts, schemas, `type` classes and internal/cross-document key resolutions only; no entry names or values:

### `CDoorInteractionDataList` (404,957 B, 7 streams, 5,831 elements)

Fourteen top-level keyed sections (all 14 named across the reference paths and target sections below; `372` `key=` attributes, `979` `type=` attributes across `31` classes, `44` `content="matrix34"` transforms, `133` vectors, `791` `value=` scalars) form a self-contained internal lookup graph:

| Section / reference path | Count | Resolves to / notes |
|---|---:|---|
| `DoorTypeToDataSetCollectionsMapping` / `List/Item` | 7 / 80 | `CDoorAssetList`; **80 of 80** resolve to `DataSetCollections` (`18` distinct) |
| `DataSetCollections` / `List/Item` | 21 / 60 | `CDoorAssetList`; **60 of 60** resolve to `DoorInteractionDataSets` (`56` of `56` sets) |
| `DoorInteractionDataSets` (`key=`, `type="CDoorInteractionDataSet"`) | 56 | `40` carry `parentPath`, all **40 of 40** resolving via `../Item[@key='...']` (`gamedata_graph.md`); `DoorType` (**20 of 20** → `DoorTypeToDataSetCollectionsMapping`), `OverlayDataSet` (**9 of 9** → `DoorInteractionAnimOverlayDataSets`), `PedHeadingData` (**11 of 11** non-empty → `DoorInteractionHeadingDatas`), `Interactions/Item` (**93 of 93** → `DoorInteractionDatas`), `PostAlignInteractions/Item` (**11 of 11** → `DoorInteractionPostAlignDatas`) |
| `DoorInteractionDatas` (`91`) & `DoorInteractionPostAlignDatas` (`11`) | 102 | `DoorType` (**91 of 91** on interactions), `BreakoutData` (**91 of 91** + **11 of 11** → `DoorInteractionBreakoutDatas` [`6`]), `UpperbodyRequestData` (**91 of 91** + **11 of 11** → `DoorInteractionUpperbodyPoseDatas` [`8`]), `MTHWeightData` (**73 of 73** non-empty → `DoorInteractionMTHDatas` [`3`]), `InterruptToIdleBlendout` (**7 of 7** → `DoorInteractionBodyBlendOutDatas` [`1`]), `PedHeadingData` (**11 of 11** on interactions; **73 of 73** non-empty across all `.//PedHeadingData` → `DoorInteractionHeadingDatas` [`5`]), `AnimData/UpperbodyBodyBlendFilterTemplateOrSelector` (**4 of 4** → `HashSelectors` [`1`]) |
| `DoorInteractionAnimOverlayDataSets` (`1`) & `DoorInteractionAnimOverlayDatas` (`8`) | 9 | `OverlayDatas/Item`: **8 of 8** resolve to `DoorInteractionAnimOverlayDatas` |
| `DoorInteractionUpperbodyPoseDatas` (`8`) & `UpperbodyPoseRequiredBlackboardVariables` (`4`) | 12 | `WeaponData1H/DesiredPose` (**8 of 8**) and `WeaponData2H/DesiredPose` (**8 of 8**) resolve to `UpperbodyPoseRequiredBlackboardVariables` |

### Ped loadouts, task tuning, headbob, speeds, motivation and motion prediction

| Root | Inflated B (streams) | Measured schema and key resolution |
|---|---:|---|
| `CPedInventoryLoadOutManager` | 411,101 (7) | `8,103` elements, `5` `type` classes (`1,692` nested `Items/Item` across the document): `LoadOuts/Item` × `569` (`Name`, `Items`, holding `1,013` direct items: `601` `CLoadOutWeapon`, `302` `CLoadOutRef`, `110` `CLoadOutRandom`) and `NamedLoadoutItems/Item` × `191` (`key=`: `136` `CLoadOutRandom`, `45` `CLoadOutWeapon`, `10` `CLoadOutRef`). `DefaultLoadout` resolves **1 of 1** to `LoadOuts`, and **521 of 521** `CLoadOutRef/Name` elements (`184` distinct) resolve case-insensitively to `NamedLoadoutItems` (`518 of 521` / `185` distinct under exact case); `CLoadOutWeapon` carries `551` `WeaponRecipe` and `546` `WeaponPreset` (`231` distinct) fields |
| `CTaskDataInfoManager` | 192,988 (3) | `3,532` elements (`2,990` `value=`): `aTaskData/Item` × `49` (`type="CTaskDataInfo"`, `27` distinct child fields) and `DefaultSet ref` (**1 of 1** → `aTaskData`). In `FleeStyleSets/{Base,InVehicle,OnMount}`, **47 of 49** entries each (`41` of `42` distinct) resolve case-insensitively to `aTaskData` (`16 of 49` / `13 of 42` under exact case) |
| `STPedNavigationPredictiontPresetContainer` | 92,999 (2) | `1,648` elements (`1,597` `value=`; preserves `Predictiont` spelling): `mapPredictionPresets/Item` × `14` (`key=`), holding `21` `arPredictionSetConfigs/Item` blocks (`1..8` per preset, `71..75` fields each, `81` distinct field names across the 21 blocks covering collision, capsule, drops, step-ups, walls, cover shadows, corners, slope and climb flags) |
| `STPedMotionPredictionSetContainer` | 50,327 (1) | `898` elements (`864` `value=`): `mapPredictionPresets/Item` × `10` (`key=`), holding `12` `arPredictionSets/Item` blocks (`1..3` per preset, `67` fields each from the same prediction-flag schema) |
| `CPedHeadbobSettingsManager` → `CHeadbobConditionManager` | 88,190 (2) + 16,610 (1) | `CPedHeadbobSettingsManager` (`1,347` elements) defines `OnFootSettings` (`12` `IntensitySettingsMap/Item`, `6` `BodyBlendSettingsMap/Item`) and `InTransportSettings` (`12` and `0`) across `CPedHeadbobSettings` (`2`), `CPedHeadbobIntensitySettings` (`24`) and `CPedHeadbobBodyBlendSettings` (`6`). All **84 of 84** `pCondition ref` links (`15` distinct) resolve **across documents** to `CHeadbobConditionManager/Conditions/Item/@key` (`18` `CHeadbobCondition` entries built from `CAICondition*` condition trees) |
| `CPedSpeedDataBlock` | 86,390 (2) | `2,248` elements (`1,455` `value=`, `0` `type=`): `Data/Item` × `16` (distinct `Name` entries, each with `WanderSpeeds`, `Default`, `Combat`, `Flee` over `{Walk,Jog,Run,Sprint}{Min,Max}` plus `QuadBike`, `EScooter`, `MobScooter`, `Caddy` × `16` and `Bike` × `15`, and `1` `ItemSpeeds/Item`) plus `DefaultItemSpeeds/Item` × `1` |
| `CPedMotivationInfoManager` | 32,388 (1) | `872` elements (`499` `value=`): `EventMotivationImpactProfiles/Item` × `7` (`key=`, `35` `EventTypes/Item` across `5` keys, `51` `MotivationImpacts/Item` pairs `{Motivation, ImpactParams}`), `PedMotivationDefinitions/Item` × `14`, and `aFearResponseProfiles/Item` × `4`; `DefaultFearProfileName` resolves **1 of 1** to `aFearResponseProfiles` |

### Lasso, climbing, animal attack behaviours, relationships and AI search POIs

| Root | Inflated B (streams) | Measured schema and key resolution |
|---|---:|---|
| `CLassoDataManager` | 126,639 (2) | `2,718` elements across `11` top-level sections (`99` `key=`, `692` `value=`, `380` 3D vectors with `7` `w`): `LassoHogtieFlagSets/Item` (`8`, referenced **24 of 24** by `ModelToLassoHogtieFlagsKey` plus all **3** default flag fields), `EntityLassoAttachInfo/Item` (`18`, referenced **26 of 26** by `ModelNameToEntityLassoAttachInfo` plus `DefaultHumanEnityLassoAttachInfo` and `DefaultHorseEnityLassoAttachInfo` [`Enity` preserved in XML]), and `LassoAttachments/Item` (`23`, referenced **18 of 18** by `DefaultAttachment` and **20 of 20** by `Attachments/Item`), plus `LassoWeaponLoopBoneTags/Item` × `15` |
| `CAnimalAttackBehaviours` | 55,388 (1) | `877` elements (`378` `type=`): `Behaviours/Item` × `6` (`type="CAnimalAttackBehaviour"`, `{Name, AnimalTuning, Task, Triggers}`) holding `64` `CAnimalAttackTrigger` entries, `58` gated by `CAICondition*` boolean trees (`308` nodes across `32` condition classes, led by `CAIConditionAnd` [`53`], `CAIConditionNot` [`31`], `CAIConditionDistanceBetweenEntities` [`21`]) |
| `CClimbTuneMgr` | 54,137 (1) | `1,102` elements across `12` top-level sections (`TraversalData`, `LandData`, `PlayerClimbData` [20 scalars], `PlayerStepUpData` [15 scalars], `PlayerLedgeGrabData` [16 entries], `PlayerJumpData` [11 entries], `AnimalJumpData` [11 scalars], `InputControlData` [4 scalars], `aClimbParams/Item` × `2` [`CClimbParams`], `aDropdownDetectorParams/Item` × `2` [`CDropDownDetectorParams`, with `matOverride` `content="matrix34"`]); `DefaultClimbParams ref` and `DefaultDropDownParams ref` resolve **1 of 1** each |
| `Relationships__GroupsData` | 35,555 (1) | `554` elements: `Groups/Item` × `64` (`key=`) containing `423` `Relationships/Item` entries across `4` relationship-level text values (`62` distinct targets); all **423 of 423** resolve internally to `Groups/Item/@key` |
| `CSearchPointOfInterestRuleSet` | 34,987 (1) | `684` elements (`47` `CAICondition*` nodes across `14` classes): `PointOfInterestDefinitions/PointOfInterest` × `12` (`key=`) and `SearchSituations/Situation` × `13` (`key=`). All **112 of 112** `AllowedPointsOfInterest/Type/@key` (`12` of `12` POIs) resolve to `PointOfInterestDefinitions`, and all **7 of 7** non-empty `NextSituation` links (`6` distinct) resolve to `SearchSituations` |

## No-sharing rule

The archive, every carved `.bin`, `index.json` (its `head`/`pre` fields contain payload bytes), XML
excerpts, entry names, values, paths, screenshots and per-file reports stay on the local machine. Only
aggregate counts, sizes and schema/category names belong in git, issues or pull requests. The unit test uses
synthetic data; CI never reads a game folder. This is the same rule as `AGENTS.md` and `CONTRIBUTING.md`, and
it applies to this note as well.

## Streaming reader (`tools/gamedata_reader.py`, ASSET-139)

`tools/gamedata_reader.py` reads the carved documents with `iterparse` and turns each into plain Python values,
without a schema per tag. Each top-level field is converted when its end tag arrives and then dropped, so memory is
bounded by the largest field. The generic conversion follows the parser's own conventions:

- `value=` is a scalar: bool, decimal or `0x` integer, float (a C `f` suffix allowed), else text.
- `x/y/z/w` is a vector tuple.
- `content=` is the array body. `short_array`, `float_array` and `float16_array` give number lists;
  `vector2_array`, `vec3v_array`, `matrix33` and `matrix34` give tuples.
- `type="Class"` is a polymorphic object `{"$type": ...}`, and `type="NULL"` is `None`.
- `ref=` is `{"$ref": ...}`.
- All-`Item` children are a list, or a dict when every one carries `key=`. A repeated tag is a list. A text-only
  element is its text. Any other attribute is kept as `"@name"`.

`register_root(tag)` and `register_type(name)` plug in a handler where a schema is known. Everything else stays
generic.

```
python -I tools/gamedata_reader.py <work>/common [--root TAG] [--out DIR]   # counts; --out: one JSON per document (local only)
```

On the 1.0.2.1 carve of `common.rpf` the run takes about 5 s, with 0 conversion errors:

| | count |
|---|---:|
| documents read | 259 of the 267 payloads |
| roots | 111 |
| not well-formed (the tail-only and incomplete carves counted above) | 8 |
| scalars | 84,585 |
| vectors | 20,825 |
| arrays | 13,449 |
| lists | 13,465 |
| keyed maps | 526 |
| polymorphic objects | 6,490 (317 `type` classes over 6,903 typed elements) |
| `NULL` pointers | 278 |
| references | 924 |
| elements with extra attributes | 5,262 |

The 7 roots missing from the 118 occur only in documents that are not well-formed. Counts only: the JSON that
`--out` writes is the game's data and stays local. `tests/test_gamedata_reader.py` checks every convention,
both dispatch tables, that streaming equals a whole-tree conversion, and the CLI, on synthetic XML.

## Next step

The most useful next step is the other half of the **schema bridge**: check whether the 118 XML root tags (and the
317 `type` classes) match RTTI class names in the executable, then register typed handlers for them in
`tools/gamedata_reader.py`. That turns the measured XML into the runtime's config layer without hand-writing 5,293
tags. The streaming reader that dispatches on root and `type` exists now (above). The `MHIU`/`PAMH`
heightmaps are decoded by `tools/gamedata_maps.py`; every MHIU header field is identified (0x28 is the
default height), and the row-table layouts and the value-to-Z mapping are measured.
