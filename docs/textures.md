# Textures: TTPF pools and BCPack decoding

Measured on 2026-11-21 from one retail Xbox Series install (1.0.2.1). The corpus is all 57
RDS5 texture pools: 46 under `data/ttdb/**` and 11 under `pack/<name>/data/ttdb/**`, 57.94 GiB
of files. Nothing below reproduces game data; every number is an aggregate or a descriptor
field.

## Outcome

| | Initial baseline | After layout/format fixes (zlib/raw) | Projected with BCPack (BC1–BC7)* |
|---|---:|---:|---:|
| catalog entries | 357,115 | 357,115 | 357,115 |
| exported with the complete source mip chain | 98,603 | 305,959 | 357,115 |
| exported with fewer mips | 210,292 | 16,423 | 0 |
| exported at least one mip | 308,895 | 322,382 | 357,115 |
| skipped | 48,220 | 34,733 | 0 |

\* *Codec-availability projection from `tools/ttpf_mips.py scan` across all 57 pools, not a full
DDS export run. In full-corpus chunk checks with locally calibrated history tables
(`--allow-provisional --history <path>`), BC6H decodes 268 / 268 chunks (0 strict, 100%
provisional) and BC7 decodes 5,150 / 5,167 chunks (583 strict, 99.67% provisional), while
**17 / 5,167 BC7 chunks (0.33%)** still cleanly refuse in provisional mode when unknown built-in
history references exceed the consistency budget (see §5 below).*

A skipped row has `File: null` and a `Limitation`. Either the catalog marks the texture as
unextractable, or its content does not decode (`NotSupportedException`, `InvalidDataException`).
Both are properties of the pool, so `rpf9dump` still exits 0 and publishes the manifest. A failure
to read the pool or write a DDS is different. It exits 1 and withholds `manifest.json`, and a rerun
resumes from the DDS files already written.

**Bottom line:** the compact mip chain is decoded and validated below, and a clean-room
stdlib-only BCPack decoder (`tools/bcpack.py`, `tools/bcpack_tables.py`,
`tools/bcpack_tables_bptc.py`) now decodes the rANS entropy layer and rebuilds BC blocks
for all six active modes (BC1, BC3, BC4, BC5, BC6H and BC7 — 100% of BCPack codecs in the
corpus). To inspect or decode one raw BCPack stream from the command line, `tools/bcpack_tool.py`
wraps the same decoder: `inspect <in.bin> --codec <1..7> --size <uncompressed_bytes> [--json]` prints
the stream's header, entropy statistics and validation results; `decode <in.bin> <out.bin> --codec <1..7>
--size <uncompressed_bytes> [--provisional]` writes the rebuilt BC blocks; `profile <in.bin> --codec <1..7>
--size <uncompressed_bytes>` reports per-symbol frequency statistics. Strict confirmed-history mode is
the default; `--allow-provisional` enables the
provisional and last-resort history entries, and `--history <path>` supplies locally
calibrated BC6H/BC7 initial history JSON tables at run time (no fitted BC6H/BC7 history
data is embedded in the repository). `rpf9dump`'s C# `TtpfReader` (submodule pin `e293917`,
`breadbbbb/rpf9-explorer#7`) now decodes compact 2D mip tails, multi-slice layouts, DXGI 11,
and BCPack BC1, BC3, BC4 and BC5 chunks in strict mode; BC6H, BC7 and provisional-history
chunks remain skipped by `rpf9dump` and can be completed by `tools/ttpf_assets.py`.

The compact layout rule moves 206,661 2D textures into complete exports and 1,218 more into
partial exports (`305,264` complete, `16,419` partial, `35,432` skipped under zlib/raw).
Resolving the 697 multi-slice descriptors (127 2D arrays — including the 19 two-slice arrays
formerly misread as DXGI 156/199/211/226 — 567 cubemaps, and 3 3D volumes; see
`docs/textures_arrays.md`) adds 690 more complete exports and 4 partials under zlib/raw, and
adding DXGI 11 (`R16G16B16A16_UNORM`, 8 BPP) resolves the last 5 format skips, bringing the
zlib/raw totals to **305,959 complete**, **16,423 partial** and **34,733 skipped** (33,507 2D
reader BCPack, 1,223 compact all-BCPack and 3 array all-BCPack; **0** remaining layout or
format skips).

Adding all six BCPack codecs (BC1, BC3, BC4, BC5, BC6H and BC7) in `tools/ttpf_mips.py`
unlocks **51,156** more complete chains across the 57 pools (51,149 2D + 7 multi-slice 2D
arrays), bringing the codec-availability projection to **357,115 complete** (100% of catalog
entries), **0 partial** and **0 skipped** (**0** remaining BCPack codec skips, **0**
layout/format skips and **0** skipped multi-slice descriptors). On actual chunk decode over all
57 pools, BC6H requires `--allow-provisional --history <path>` (268 / 268 chunks) and
**17 / 5,167 BC7 chunks (0.33%)** still cleanly refuse in provisional mode (see §5 below).

This is the same 308,895 / 48,220 baseline already recorded in `docs/design.md`. The recount
splits the 48,220 by cause: the old "(BCPack or unmapped)" label hides 14,713 skips whose
recorded limitation is a layout/format check, not the codec, and overstates unmapped, which
is zero. Together, the compact 2D, multi-slice and DXGI 11 rules resolve all 697 multi-slice
descriptors, 14,011 2D compact descriptors and 5 DXGI 11 descriptors, leaving only BCPack
under zlib/raw.

## Skip breakdown

| Cause | Before | After (zlib/raw) |
|---|---:|---:|
| BCPack blocks every usable mip (`TtpfReader`: "All source mips require…") | 33,507 | 33,507 |
| compact mip mapping not in the validated profile table | 14,030 | 0 |
| compact layout resolved, but every mip is BCPack | — | 1,223 |
| former "compact element size unknown" (19 two-slice 2D arrays; see `docs/textures_arrays.md`) | — | 0 |
| 2D array / cubemap / volume descriptor (`dimension != 1` or `depth > 1`; see `docs/textures_arrays.md`) | 678 | 0 |
| array layout resolved, but every mip is BCPack | — | 3 |
| element format not in the parser table (DXGI 11, `R16G16B16A16_UNORM`) | 5 | 0 |
| no mapped allocation ("unmapped shard") | 0 | 0 |
| **total** | **48,220** | **34,733** |

The "before" compact row becomes the measured rows below it after the layout rules: records
the rule sizes but cannot decode under zlib/raw because every mip is BCPack. The 1,218 mixed
compact records and 4 mixed array records that become partial exports are not skips; they are
counted in the partial totals below.

The compact row is not one thing:

- 13,979 of the 14,030 carry the descriptor's auxiliary-mapping bits. 13,954 of those are
  mode 4 or 6; the other 25 are modes 5, 7, 36 and 38. The other 51 compact records are modes
  1, 32 or 34 without the aux flag.
- 11,570 of the 14,030 have no BCPack chunk anywhere. They fail because
  `ValidatedCompactProfile` has no entry for their
  `(mode, width, height, mips, nonTail, bytes)` tuple, not because of a codec.
- 19 mode-1 records (64 KiB) originally appeared as unknown DXGI formats 156 (7), 199 (6),
  211 (5) and 226 (1) at 256×256 with 7 mips (2) or 512×512 with 8 mips (17). With the
  corrected 7-bit format / 11-bit depth bitfield (`docs/textures_arrays.md`), all 19 are
  two-slice 2D arrays (`depth = 2`) with standard DXGI formats 28/71/83/98 (18 complete
  zlib/raw, 1 partial).

The five format-11 records are 4×4, 8×4 or 128×4 `R16G16B16A16_UNORM`; the parser's
element-size table has no 8-byte entry for format 11. The 678 records formerly skipped as
array/volume descriptors (plus the 19 two-slice arrays above, 697 total) now have validated
layouts in `tools/tex_headers.py` and `tools/ttpf_mips.py` (`docs/textures_arrays.md`).

## The BCPack footprint

The skip count is only the fully blocked textures. Counting mip 0 and the lower mips
independently of the parser's profile checks gives the full footprint:

| mip 0 | lower mips | Textures |
|---|---|---:|
| zlib/raw | zlib/raw | 305,264 |
| BCPack | at least one zlib/raw | 14,935 |
| zlib/raw | at least one BCPack | 9,999 |
| BCPack | every mip BCPack | 26,215 |
| no usable layout | — | 702 |
| **total** | | **357,115** |

- **41,150 textures have BCPack in mip 0** (26,215 + 14,935).
- **26,215 have no zlib/raw mip at all**, so they cannot export anything.
- 5,845 export from a lower mip because mip 0 is BCPack (`BaseMipLevel > 0`).
- 9,090 of the 14,935 are still skipped: 8,515 because no usable mip is entirely zlib/raw
  (the zlib block is mixed with BCPack or outside the usable range), 575 because the compact
  profile is not validated.
- 9,356 export the top mip but stop before at least one BCPack lower mip; 643 more are
  skipped for the compact profile before an export can happen.

Cross-tabulated against the skip table:

| Parser cause | all zlib | mip 0 zlib, lower BCPack | mip 0 BCPack, lower zlib | all BCPack | no layout | total |
|---|---:|---:|---:|---:|---:|---:|
| BCPack | 0 | 0 | 8,515 | 24,992 | 0 | 33,507 |
| compact profile | 11,570 | 643 | 575 | 1,223 | 19 | 14,030 |
| array/volume | 0 | 0 | 0 | 0 | 678 | 678 |
| format 11 | 0 | 0 | 0 | 0 | 5 | 5 |
| **total** | **11,570** | **643** | **9,090** | **26,215** | **702** | **48,220** |

The partial exports are a separate gap. Of the 210,292:

| Partial reason | Textures |
|---|---:|
| compact profile, standalone mips exported, packed tail missing | 196,257 |
| lower mips BCPack | 8,052 |
| catalog top-only record | 314 |
| other (lower-mip export, mixed ranges) | 5,669 |
| **total** | **210,292** |

The packed-tail gap affects more textures than the BCPack skip count (196,257 partial
exports vs 33,507 skips), but it loses only the small mips. It is also not a BCPack problem:
196,257 textures stop at the descriptor's `non_tail` field because `UsableMipLimit =
NonTailMipCount` for compact profiles, regardless of codec. The next section measures that
field against the layout and finds that it is not the tail boundary: most of the "packed
tail" is standalone mips the reader drops.

## The compact mip chain

The 196,257 partial exports and the 14,030 compact-profile skips are the same layout
problem, not a codec problem. `tools/tex_headers.py` now carries the layout rule as pure
functions, `tools/ttpf_mips.py` measures and decodes with it, and `tests/test_tex_headers.py`
covers the rule on synthetic chains. The counts below come from `scan` and `check` over the
same 57 pools.

### The descriptor mode is a flag plus a tile mode

`TtpfReader` reads the descriptor mode as one number (`b >> 23`) and selects a 64 KiB tile
only for mode 0. The observed values split into a low flag bit and a 4-bit Gfx10 tile mode at
bit 1:

| `(mode >> 1) & 0xF` | Gfx10 mode | Tile | Descriptors |
|---:|---|---:|---:|
| 0 | `SW_64KB_D` | 64 KiB | 39,283 |
| 1 | `SW_4KB_S` | 4 KiB | 70,497 |
| 2 | `SW_4KB_S` | 4 KiB | 195,957 |
| 3 | `SW_4KB_S` | 4 KiB | 50,811 |
| 9–12 | array/volume | — | 567 |

Modes 9–12 are exactly the 567 cubemap descriptors; the other 130 multi-slice records (127 2D
arrays and 3 volumes) use tile mode 0 or 1. Bit 0 of the field is independent: modes 4/5 share
tile mode 2, 6/7 share tile mode 3, and 36/38 share tile modes 2/3. The reader's `mode == 0`
test therefore gives a 4 KiB layout to modes 1 and 32, which are 64 KiB. Fifteen single-slice
mode 1/32 descriptors have a known element size (4 mode 1, 11 mode 32); all fifteen match a
64 KiB allocation span and one (`maps_4_7` #6034) decodes at 0.00–0.33 levels (see below). The
other 19 mode 1 records formerly counted as unknown-format 2D entries are two-slice 2D arrays
(`depth = 2`, DXGI 28/71/83/98; see `docs/textures_arrays.md`).

Where one descriptor's mode is still in doubt, `tools/tex_mode_probe.py <textures index.json>
<carve dir> [--fmt BC1] [--n 12]` scores every 4 KiB and 64 KiB addrlib colour pattern of the
element size, crossed with both placements of the top mip (at the data offset, or at the end of
the chain, since GFX10 stores mips smallest first), by the smoothness of the decoded BC1/BC2/BC3
average colours. It prints the best and second-best pattern with the placement and the linear
(untiled) score per texture, then a tally of the winners and how many textures no candidate fit.
It needs numpy, like `tests/test_tex_mode_probe.py`, which covers it on synthetic blobs with no
game files.

### The layout is the AMD mip tail

The compact profile is the Gfx10 mip chain: standalone mips are padded to whole blocks and
stored smallest-first; a mip whose element count fits half a tile width and a tile height,
among the last 8 (4 KiB) or 12 (64 KiB) mips, goes into one shared tile at offset 0 at the
AMD entry offset (`m = max_tail - 1 - k`; `entry = m > 6 ? 16 << m : m << 8`).

On all 356,413 single-slice 2D descriptors with a known element size — the 232,789
already-trusted compact exports, the 14,011 layout-capable records the profile table rejects,
and every non-compact mode — the computed size equals `next_allocation - base`. There are
**0 mismatches**. Of the remaining 702 descriptors, 697 (127 2D arrays, 567 cubemaps, and 3
volumes) are validated with 0 mismatches in `docs/textures_arrays.md`, leaving only the 5 DXGI
11 (`R16G16B16A16_UNORM`) records without a format entry.

`tools/tex_headers.py` exposes the rule without any container dependency:

- `gfx10_tile_mode(mode)` / `gfx10_tile_bytes(mode)` split the descriptor field into the
  flag bit and tile mode and pick 64 KiB only for tile mode 0.
- `parse_etpf_texture_record(record)` decodes the 16-byte ETPF record, including the 4-bit
  packed mip count plus the 3-bit `non_tail` field that make up the total mip count.
- `compact_surface_layout(width, height, mips, element_bytes, block, tile)` computes the
  allocation size and every mip's offset, region and in-tile origin. It rejects a mip count
  that cannot describe the surface, an element size outside 1/2/4/8/16 bytes, a block size
  other than 1 or 4, and any tile other than 4 or 64 KiB.
- `compact_layout_for_dxgi` applies `FMT_INFO` to that function; `compact_profile_key`
  returns the exact `(mode, width, height, mips, non_tail, bytes)` tuple; and
  `compact_span_matches` accepts an allocation span only when it is byte-exact.

`tools/ttpf_mips.py` uses those functions for `scan`, `layout`, `check` and `export`, so the
measurement and the proposed reader rule cannot drift apart.

The rule is only claimed for 2D descriptors with a 1/2/4/8/16-byte element. The 12-byte DXGI
formats 6/7/8 do not occur in the measured 2D set, so `compact_surface_layout` rejects them
rather than guessing a Gfx10 log pattern. 2D uncompressed tiling is validated across all
2,911 non-volume uncompressed records in the 57 pools (1/2/4/8 BPP under `SW_4KB_S` and
RDNA2/RbPlus `SW_64KB_D`, confirmed against Mesa `src/amd/addrlib/src/gfx10/gfx10SwizzlePattern.h`
`GFX10_SW_4K_S_RBPLUS_PATINFO` and `GFX10_SW_64K_D_RBPLUS_PATINFO`): for 64 KiB 4 BPP
(`DXGI 28/29/87/91`), RDNA2/RbPlus `SW_64KB_D` swaps intra-tile address bits 6 and 7 relative
to Navi 1x (`(3, -3)` = `X2, Y2` instead of `(-3, 3)` = `Y2, X2`), while 1/2/8/16 BPP are
unchanged between Navi 1x and RDNA2. `tools/ttpf_mips.py` decodes and exports the validated
uncompressed formats (`10/11/28/29/49/54/56/61/65/87/91`) as DX10 DDS with `DDSD_PITCH`.

### `non_tail` is not the layout boundary

The descriptor's 3-bit `non_tail` field (`b & 7`) is where the reader stops, and its note calls
the remaining mips "packed tail". The bytes do not follow that split:

- 177,111 of the 232,789 compact exports have `non_tail < first_tail`; 214,298 standalone
  mips lie beyond the limit, 204,108 of them in zlib/raw chunks.
- 196,257 partial exports stop exactly at `non_tail`; 158,272 of them stop **before** the
  computed tail boundary, and only 37,985 stop at it.
- Of the 1,072,068 missing mips, 1,065,138 are in zlib/raw chunks. That decodable part splits
  into 193,019 standalone mips and 872,119 packed-tail mips; the other 6,930 missing mips (in
  1,166 textures) touch BCPack.

`UsableMipLimit = NonTailMipCount` therefore drops decodable standalone mips, not only the
tile. The allocation spans and the decoded content both put the tail at the computed index.

### The 14,030 profile-table records

| | Textures |
|---|---:|
| same layout, new tuple (4 KiB) | 13,956 |
| same layout, modes 5/7/34/38 | 40 |
| same layout, 64 KiB (modes 1/32) | 15 |
| **2D compact layout resolved** | **14,011** |
| two-slice 2D arrays (formerly misread as DXGI 156/199/211/226; see `docs/textures_arrays.md`) | 19 |
| **total** | **14,030** |

The rejected 2D records contain 84 distinct six-tuples with a known element size; the 19
remaining records are two-slice 2D arrays handled by `array_surface_layout`. The ten largest
known-size 2D tuples are:

| `mode` | Width×height | Mips | `non_tail` | Element bytes | Records |
|---:|---|---:|---:|---:|---:|
| 4 | 512×512 | 3 | 3 | 16 | 8,823 |
| 4 | 256×256 | 2 | 1 | 16 | 1,831 |
| 4 | 256×256 | 3 | 1 | 8 | 1,744 |
| 4 | 512×256 | 3 | 1 | 8 | 344 |
| 4 | 256×128 | 3 | 1 | 8 | 319 |
| 4 | 1024×512 | 3 | 3 | 8 | 191 |
| 4 | 512×256 | 9 | 1 | 8 | 68 |
| 4 | 256×256 | 2 | 1 | 8 | 66 |
| 4 | 512×32 | 4 | 1 | 8 | 62 |
| 4 | 512×512 | 10 | 3 | 16 | 58 |

None of these needs a new table row: every one has zero allocation-span mismatches under the
computed rule. `scan --tuples` emits the complete list without texture bytes.

What the layout change does to the 14,030:

| | Textures |
|---|---:|
| complete zlib/raw chain (2D compact) | 11,570 |
| partial export: some readable contiguous mip range (2D compact) | 1,218 |
| still skipped: every mip is BCPack (2D compact) | 1,223 |
| two-slice 2D arrays (resolved in `docs/textures_arrays.md`: 18 full, 1 partial) | 19 |
| **total** | **14,030** |

- The 11,570 complete records hold all 35,727 mips (2.91 GiB decompressed).
- 469 of the 1,218 partial records have a readable tail tile while at least one larger mip is
  BCPack; the other 749 export some contiguous mip range.
- The 1,223 BCPack skips are compact records whose every mip is BCPack; they were counted in
  the 14,030 before and are not part of the 33,507.

The five DXGI 11 (`R16G16B16A16_UNORM`, 8 BPP, all 1-mip `SW_4KB_S`) records (`maps_2_7`
#11324, `maps_5_7` #2425 and #13651, `maps_7_7` #7490 and #13893) are now mapped in
`ELEMENT_BYTES` (`11: 8`). All five match `surface_layout` byte-exact, place 100% of their
active 8-byte elements strictly at the `detile()` addresses (`0` non-zero bytes in padding
slots across the 4 KiB tiles), and export cleanly as DX10 DDS.

### Sample validation

`tools/ttpf_mips.py check` detiles one chain, decodes BC1/BC3/BC4/BC5/BC6H/BC7 and validated
2D uncompressed formats (`10/11/28/29/49/54/56/61/65/87/91`; pure stdlib; HDR float16 formats
`10/54/95/96` are compared after a signed Reinhard tone-map `255 * v / (1 + |v|)`) and compares
each mip with a 2x2 box downsample of its parent (at 4x4 block resolution for BC, or pixel
resolution for uncompressed). A correct chain is 0–4 levels, as in the SixNG notes; the
control is the first tail mip detiled at origin (0,0).

| Pool # | Descriptor | Mips | non_tail / first_tail | Max error | Tail control |
|---|---|---:|---:|---:|---:|
| `interiors_0_3` #19 | 512×256 BC1 | 7 | 1 / 3 | 0.38 | 126.06 |
| `interiors_0_3` #10 | 512×512 BC1 | 8 | 3 / 3 | 0.46 | 57.94 |
| `graphics` #43 | 64×512 BC1 | 10 | 1 / 3 | 1.08 | 132.31 |
| `interiors_2_3` #468 | 512×512 BC1 | 8 | 3 / 3 | 0.60 | 113.83 |
| `maps_0_7` #3217 | 128×128 BC1 | 6 | 0 / 1 | 0.79 | 114.71 |
| `peds_1_3` #527 | 512×512 BC1 | 8 | 3 / 3 | 3.87 | 83.83 |
| `maps_1_7` #1500 | 128×256 BC4 | 6 | 1 / 2 | 0.34 | 72.08 |
| `decorations_0_3` #818 | 128×128 BC5 | 6 | 1 / 2 | 0.02 | 85.48 |
| `interiors_1_3` #507 | 256×256 BC3 | 7 | 1 / 2 | 1.21 | 51.84 |
| `terrain_4_7` #23 | 512×512 BC7 (`98`) | 3 | 3 / 3 | 0.49 | — |
| `maps_4_7` #6034 | 1024×1024 BC7 (`98`, 64 KiB) | 9 | 3 / 3 | 0.33 | 94.76 |
| `interiors_0_3` #3 | 512×512 BC7 sRGB (`99`) | 8 | 3 / 4 | 1.15 | 68.09 |
| `interiors_2_3` #3268 | 1024×1024 BC6H UF16 (`95`, 64 KiB) | 9 | 3 / 3 | 0.35 | 110.21 |
| `lighting` #346 | 512×512 BC6H SF16 (`96`) | 10 | 3 / 4 | 0.68 | 15.41 |
| `terrain_0_7` #5207 | 512×512 `R8G8B8A8_UNORM` (`28`, 64 KiB) | 8 | 3 / 3 | 0.29 | 160.44 |
| `peds_3_3` #4922 | 256×256 `R8G8B8A8_UNORM` (`28`, 4 KiB) | 7 | 3 / 3 | 0.28 | 13.23 |
| `vfx` #654 | 1024×1024 `R8_UNORM` (`61`, 64 KiB) | 3 | 3 / 3 | 0.01 | — |
| `vfx` #869 | 1024×512 `R8_UNORM` (`61`, 4 KiB) | 3 | 3 / 3 | 0.11 | — |
| `peds_0_3` #2958 | 512×512 `A8_UNORM` (`65`, 4 KiB) | 8 | 3 / 3 | 0.00 | 255.00 |
| `interiors_3_3` #1033 | 1024×1024 `A8_UNORM` (`65`, 64 KiB) | 9 | 3 / 3 | 0.50 | 58.31 |
| `graphics` #647 | 128×128 `B8G8R8A8_UNORM` (`87`, 4 KiB) | 8 | 1 / 2 | 0.00 | 61.09 |

`interiors_0_3` #19 is a partial export: the reader emits one mip, while all seven decode and
match. `graphics` #43 and `interiors_2_3` #468 are tuples outside the table; `maps_0_7` #3217
is mode 34 with `non_tail = 0`, which would leave zero standalone mips even if the mode gate
passed.

The same BC6H/BC7 decoders also cover multi-slice chains and BCPack chains in `check`:
`graphics` #236 (256×256×48 BC7 2D array, 9 mips, 64 KiB) passes all 48 slices (`max_error =
0.37` vs slice-major control `40.74`), `lighting` #474 (256×256×6 BC6H UF16 cubemap, 7 mips)
passes all 6 faces (`max_error = 0.25` vs tail control `95.82`), and `graphics` #86 (64×64×6
BC7 cubemap, 7 mips) passes all 6 faces (`max_error = 3.46` vs tail control `40.1`).
Note that prefiltered lighting profiles whose lower mips intentionally fade toward `1.0` or
convolve with a specular lobe will naturally exceed the box-downsample threshold even when
detiled accurately.

Across all 57 pools, the 2,911 non-volume uncompressed records break down as `fmt 10` (1),
`fmt 11` (5), `fmt 28` (1,766 2D + 8 2D arrays), `fmt 29` (1 2D + 1 2D array), `fmt 54`
(946 2D), `fmt 61` (8 2D), `fmt 65` (121 2D), `fmt 87` (49 2D + 1 cubemap) and `fmt 91` (4 2D);
`fmt 49` and `fmt 56` have `0` entries in the TTPF pools (both are still decoded in `check`
and covered by synthetic unit tests). Full-corpus uncompressed validation:

- **64 KiB 4 BPP (`SW_64KB_D @ RbPlus`):** with the Navi 1x bit order `(-3, 3)` at bits 6..7,
  only 11 of 167 multi-mip 64 KiB `DXGI 28` textures passed (median `12.16` on 2D, `16.40` on
  2D arrays, max `36.64`). Switching `PATTERNS_64K[2]` to RDNA2/RbPlus `(3, -3)` (Mesa
  `GFX10_SW_64K_D_RBPLUS_PATINFO`) brings **167 / 167** to pass: 155/155 mode-0 2D (`0.17–0.35`,
  median `0.26`), 4/4 mode-1 2D (`0.02–0.15`), and 8/8 2D arrays (`0.25–0.30`; e.g. `peds_1_3`
  #2581 `max_error = 0.30` vs slice-major control `45.28`).
- **4 KiB 4 BPP (`SW_4KB_S`):** 3/3 multi-mip `DXGI 28` (`0.13–0.28`), 6/6 multi-mip `DXGI 87`
  (`0.00–5.52`), plus 1-mip `DXGI 87` cubemap `graphics` #597 (3.2× smoother neighbor gradient
  after detiling).
- **1 BPP (`DXGI 61/65`, 4 KiB & 64 KiB):** 7/7 multi-mip `DXGI 61` (`0.01–0.12`), 15/15
  64 KiB `DXGI 65` (`0.00–0.50`), and 82/83 4 KiB `DXGI 65` (`0.00–0.28`; the single outlier
  `props_1_3` #8392 is an alpha mask whose mip 5 intentionally fades to all zeros, with mip 2
  tail error `8.65` vs tail-origin control `33.25`).
- **2 BPP (`DXGI 54`, all 1-mip) and 8 BPP (`DXGI 10/11`, all 1-mip):** `DXGI 54` shows
  4.7×–13.8× lower neighbor gradient after detiling on both 4 KiB and 64 KiB (with `PATTERNS_64K[1]`
  matching Mesa `GFX10_SW_64K_D_RBPLUS_PATINFO` and beating alternative bit permutations across
  8 sampled 64 KiB maps); all 5 `DXGI 11` records and `DXGI 10` (`graphics` #832) have `0`
  non-zero bytes outside their detiled active-element slots.

For `interiors_0_3` #19 the tool reports the byte layout: mip 0 (512×256, 128×64 blocks) at
24,576, mip 1 (256×128, 64×32) at 8,192, mip 2 (128×64, 32×16) at 4,096, then the 4 KiB tile
at 0 with mip 3 (64×32 px, 16×8 blocks) at origin (16,0), mip 4 (32×16 px, 8×4) at (8,8),
mip 5 (16×8 px, 4×2) at (0,12) and mip 6 (8×4 px, 2×1) at (0,8). The mip byte sizes 65,536 /
16,384 / 4,096 / 1,024 / 256 / 64 / 16 are exactly the BC1 block counts, and the exported DDS
is 87,524 bytes (148 + 87,376).

### What this unlocks

"Decompressed" is the block-linear BC payload size (`blocks × block bytes`), not the inflated
chunk size, which includes tile padding.

| | Textures | New mips | Decompressed |
|---|---:|---:|---:|
| partial exports that become complete | 195,091 | 1,065,138 | 1.69 GiB |
| compact skips that become complete | 11,570 | 35,727 | 2.91 GiB |
| **newly complete** | **206,661** | **1,100,865** | **4.59 GiB** |

The total export split before and after the 2D compact, multi-slice and DXGI 11 rules (zlib/raw
only; see `docs/textures_arrays.md` for the +690 complete / +4 partial multi-slice breakdown,
plus +5 DXGI 11 complete):

| | Before | After 2D compact | After + multi-slice + DXGI 11 | Change |
|---|---:|---:|---:|---:|
| complete chain | 98,603 | 305,264 | 305,959 | +207,356 |
| partial | 210,292 | 16,419 | 16,423 | −193,869 |
| skipped | 48,220 | 35,432 | 34,733 | −13,487 |
| **total** | **357,115** | **357,115** | **357,115** | |

The 34,733 remaining zlib/raw skips are strictly BCPack: 33,507 2D reader BCPack + 1,223 compact
all-BCPack + 3 array all-BCPack (**0** layout or format skips). The 16,423 partials are
1,166 compact records that still miss BCPack mips, 14,035 other partials, 1,218 compact
records converted from a skip to a partial export, and 4 partial 2D arrays. Nothing in this
section decodes BCPack.

## What BCPack is

BCPack is the Xbox Series hardware decompression codec for BCn texture data. The public GDK
enum exposes seven modes, one per BCn format (BC1–BC7); DirectStorage applies it after DEFLATE
and before the optional swizzle. It is not a replacement for BCn: it is an entropy-coded
preconditioner whose output is a standard BCn block stream. That is why a BCPack chunk has a
BCn codec id, not a separate texture format, and why the container can mix zlib and BCPack
chunks inside one texture.

## What the container says

The RDS5 index is readable without game code:

| Field | Value |
|---|---|
| magic | `RDS5` at 0x00 |
| virtual size | 56 bits at 0x08; top byte 0x16 |
| entry count | u32 at 0x04; `chunks = count − 2`, and the table's last entry is the sentinel |
| chunk table | 0x18, 8 bytes per entry, `(count − 1) * 8` bytes |
| ETPF index | `align(0x18 + (count − 1) * 8, 0x100)`; version 634 (`0x27A`) |
| texture records | a 16-byte ETPF header, then `area` bytes of records (at least 16 × textures) |
| asset keys | one u64 per record after the descriptor array; the array continues into the ID region when it is longer than the record area leaves room for (the area is at least 16 × textures), so a reader must load it across that boundary. IDs are 32 bytes each. In the 57 pools of this install, 15 have a straddling array and 28,666 of the 357,115 keys sit past the record area |

Chunk entry `(u64 value)`:

| Field | Bits | Meaning |
|---|---:|---|
| physical offset | 0–25 | × 256; 256-byte aligned |
| codec | 26–29 | 0 raw, 1–7 BCPack BC1–BC7, 8 zlib |
| virtual offset | 30–51 | × 4096; 4 KiB aligned |
| padding | 52–63 | 12-bit byte count at the end of the physical range |

The parser reads 357,115 texture records and 1,026,833 chunks from the 57 pools with no
exceptions. The chunk table's offsets are 256-byte aligned (physical) and 4 KiB aligned
(virtual), and the virtual ranges cover every allocation the parser could lay out. The
12-bit padding on BCPack chunks is zero-filled (see the header section).

### Codec ids are the BCPack mode

The on-disk codec ids are `0x04, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20`; shifting right by 26
normalizes them to 1, 3, 4, 5, 6, 7, 8. Across **1,026,833 chunks there is not one
format/codec mismatch**:

| codec | BCn mode | DXGI formats seen | Chunks | Packed MiB | Decoded MiB | Saving |
|---:|---|---|---:|---:|---:|---:|
| 0 | raw | 83, 98, 99 | 5 | 0.1 | 0.1 | 0.0% |
| 1 | BC1 | 71, 72 | 102,725 | 10,637.9 | 14,630.9 | 27.3% |
| 2 | BC2 | — | 0 | — | — | — |
| 3 | BC3 | 77, 78 | 35 | 7.4 | 10.1 | 26.3% |
| 4 | BC4 | 80 | 77 | 1.4 | 1.5 | 6.2% |
| 5 | BC5 | 83 | 333 | 60.6 | 74.6 | 18.8% |
| 6 | BC6H | 95, 96 | 268 | 67.8 | 79.1 | 14.2% |
| 7 | BC7 | 98, 99 | 5,167 | 1,097.3 | 1,221.5 | 10.2% |
| 8 | zlib (RFC 1950) | all | 918,223 | 47,179.3 | 128,529.3 | 63.3% |
| **total** | | | **1,026,833** | **59,051.8** | **144,547.1** | **59.1%** |

BCPack is 11.6 GiB of the 57.7 GiB chunk payload (20.1%), and BC1 is 10.4 GiB of that
(89.6%). The saving column is `1 − Σ packed / Σ decoded`; it includes tile padding, so it is
not the codec's saving on block data alone.

This mapping is the strongest evidence that the codec field is the GDK's
`DSTORAGE_BCPACK_MODE` (1 = BC1 … 7 = BC7), not a generic compression id. The field is per
chunk, so a texture can mix zlib and BCPack mips; 15,201 partial exports do.

### From a chunk entry to `bcpack.decode`

`tools/bcpack.py` takes the codec as the BCPack mode (1–7), not the on-disk tag. Per codec:

| Blocks | Tag in the chunk entry | `decode` codec | Lane widths in stream order (bits) | Symbols per lane per block |
|---|---:|---:|---|---:|
| BC1 | `0x04` | 1 | 8,8,5,6,5,5,6,5,8,8,8,8 | 1 |
| BC3 | `0x0c` | 3 | 4,4,8,6,6,6,6,5,6,5,8,8 | 2 |
| BC4 | `0x10` | 4 | 8,8,8,8,6,6,6,6,6,6,6,6 | 1 |
| BC5 | `0x14` | 5 | 4,4,8,8,6,6,6,6,6,6,6,6 | 2 |
| BC6H | `0x18` | 6 | 4,4,8,8,8,5,5,5,6,6,6,8,8,8,8,5 | 2 |
| BC7 | `0x1c` | 7 | 4,4,8,8,8,8,8,7,7,5,6,8,8,8,8,5 | 2 |

The widths are `LANE_WIDTHS` in `tools/bcpack.py`. The tag is `(v >> 24) & 0x3c`, the codec field
above shifted left by 2; tags `0x00` (raw) and `0x20` (zlib) are not BCPack and are handled
before it. Each lane emits `decoded_bytes / 8` symbols (`ELEMENT_BYTES = 8`), so the decoded
byte count of the chunk must be passed as `out_size` and must be a whole number of blocks.

For entry `v` and the next entry `nxt` (this is what `TtpfReader` in `tools/ttpf_mips.py` does):

- input: the bytes from `(v & 0x3FFFFFF) * 256` up to `(nxt & 0x3FFFFFF) * 256 − (v >> 52)`;
- `out_size`: `((nxt >> 30) & 0x3FFFFF) * 4096 − ((v >> 30) & 0x3FFFFF) * 4096`.

The decoded blocks are still in the surface's tiled storage order; `ttpf_mips.py` detiles them
before writing a linear DDS.

A synthetic one-block BC7 chunk (no game data; the encoder is in `tests/test_bcpack.py`) with a
byte-exact result in strict mode, which `tests/test_bcpack.py` runs as
`test_bc7_fixed_stream_strict`:

```python
import sys
sys.path.insert(0, "tools")
import bcpack

packed = bytes.fromhex("""
101010101010101004040303030304040101010101010140
000000000000024002027546281800000101df5f4b7c0341
020202411010101004040403101010104004100103030304
4001080001010104000000080000000041020994e4db27a4
410511fe6b020005
""")
blocks, provisional = bcpack.decode_with_info(packed, 7, 16)
assert blocks == bytes.fromhex("40 31 72 04 a5 68 19 be 02 34 56 78 9a bc de f0")
assert not provisional   # strict mode: nothing unconfirmed was used
```

`decode_with_info` returns the set of unconfirmed history entries used; it is empty in
strict mode, which raises `BcpackProvisionalError` instead of using one.
(Byte-exact blocks, a clean entropy end state and a DDS with fitted colours are three
different levels of confirmation; this example shows the first.)

### The descriptor `mode` is tiling, not codec

The ETPF descriptor's `mode` field (`b >> 23`) is the mip/tile family: mode 0 uses 64 KiB
blocks (`SW_64KB_D`), every other observed mode uses 4 KiB blocks (`SW_4KB_S`). It is
independent of the chunk codec. Formats 71/72 use 8-byte BC1 blocks, 74/75/77/78/83/84/95/96/
98/99 use 16-byte blocks, and 80/81 use 8-byte BC4 blocks. No BC2 (codec 2) texture exists in
this install.

### Character material texture parameters

Character drawables store material texture references as a 32-bit parameter hash immediately before
the 64-bit asset key. In the measured character corpus the texture content separates by parameter
rather than by array order:

| Parameter | Common formats | Measured role |
|---|---|---|
| `0xC8E8C282` | BC1/BC3/BC7 sRGB, some BC1/BC7 UNORM | **sRGB albedo** when present |
| `0xC6F46E79` | BC7 UNORM, BC1 UNORM | `speculartex`: specular/gloss (packed), not a normal map |
| `0x1E77BF4C` | BC5 UNORM, some BC1/BC7 | `bumptex`: tangent-space normal/detail normal |
| `0x739E3D0D` | BC7 UNORM, some BC1 | `speculartex2`: ORM/mask (packed) |
| `0x8ADB8480` | BC4 UNORM | `materialid`: ID/palette map |

The counts in the 246-record character corpus are 215 references each for `0xC6F46E79`,
`0x1E77BF4C` and `0x739E3D0D`, 164 for `0x8ADB8480`, and 64 for `0xC8E8C282`. The garment
shaders (`d5ed50f4`, `ea8d09d0`, `39ddb9ca`) do not declare a `0xC8E8C282` slot at all, so the
jacket, the crop tee and every other garment on them have no albedo by shader design, not because
a texture is missing. Not every garment is on them: the bikini top and the undies are on other
shaders and account for nine of the `0xC8E8C282` references, six and three material rows
([character_models.md](character_models.md)). Exporters must **not** fall back to the first
texture when a material has no such slot; a `speculartex`
sheet painted as a base colour is the usual symptom. The hashes are treated as measured corpus
semantics, not as a claim that they are globally unique across every shader.

`tools/character_textures.py` classifies a decoded `model.json`: it prints each material's slot
hash, role and texture name, resolves only the five measured character slots, and reports any
other hash as `unknown 0x...`. A garment-shader material (`d5ed50f4`, `ea8d09d0`, `39ddb9ca`)
with no `0xC8E8C282` binding is flagged `garment-no-albedo`, so a renderer keeps that part clay
instead of painting `speculartex` as colour. The `texture` column retains the display value,
including the eight-digit `Name` hash from `rpf9dump models`. The appended `asset_key` column
holds the 16-digit `Id64` for matching against a texture pool's `AssetKey`. An absent or malformed
ID appears as `-` in the terminal, an empty CSV cell and JSON `null`. Report paths must be
distinct from the inputs:

    python -I tools/character_textures.py <model.json|dir> [--json FILE] [--csv FILE]

### Map material texture parameters

The character table above covers 246 records. Map drawables (buildings, roads, terrain, props) use more
parameters. Five more names come from the case-sensitive joaat of the parameter name, the same check that
names `0xC8E8C282` (`diffusetex`) in rpf9-explorer's `ModelTextureBindings`:

| Parameter | Name (joaat) |
|---|---|
| `0x1E77BF4C` | `bumptex` |
| `0xC6F46E79` | `speculartex` |
| `0x739E3D0D` | `speculartex2` |
| `0x3FB7964F` | `tinttexture` |
| `0xC075F9B6` | `lookuptexture` (terrain materials) |

Content check on map drawables: a random sample of 1,500 `.ydr`/`.yft` models (seed 7), each referenced
texture counted once per parameter. A texture is "normal-map-like" when its mean colour (BC1 colour-0
endpoints, BC7 mode-6 endpoints) has blue > 190 and red and green within 45 of 128; BC5 is two-channel.

| Parameter | Normal-map-like | Not normal-like | BC5 | Formats (DXGI: count) |
|---|---:|---:|---:|---|
| `0x1E77BF4C` `bumptex` | 1,394 | 74 | 131 | 71: 1,659, 83: 131, 98: 106 |
| `0xC6F46E79` `speculartex` | 69 | 1,469 | 0 | 71: 1,650, 98: 372, 28: 4 |
| `0x739E3D0D` `speculartex2` | 5 | 851 | 0 | 98: 738, 71: 225, 72: 1 |
| `0xC8E8C282` `diffusetex` | 28 | 1,887 | 0 | 72: 1,322, 99: 709, 78: 145 |
| `0x3FB7964F` `tinttexture` | 0 | 61 | 0 | 72: 61, 99: 2, 78: 1 |

So on map materials `bumptex` is the tangent-space normal map and `speculartex` is not a normal map.
The character corpus re-check resolves the same way: `0xC6F46E79` is `speculartex` on characters too,
and `0x1E77BF4C` (`bumptex`) is the normal map. The earlier character row that said otherwise has been
corrected.

Three more parameters hold sRGB colour-like textures on map materials and are not in `ModelTextureBindings`
(names not found): `0x17D3B5C2` (72: 47), `0x6994EF3F` (99: 25) and `0x7485DAFB` (99: 162, 78: 15, 72: 12;
a tiling detail layer next to the two-layer `lyr0diffusetex`/`lyr1diffusetex` blend). `tinttexture` holds an
ID/palette map on most materials and is not a base colour outside terrain.

## BCPack block structure and limits

A BCPack chunk is not a self-describing container. The container gives it a decoded (virtual)
size and a codec id; the stream itself has one 48-byte lane header and one interleaved rANS
byte stream. There is no magic, version, total length or format at offset 0.

| Layer | Bytes | Meaning |
|---|---:|---|
| lane header | 48 | 8 lanes × 6 rounds; byte `48r + l` is round `r`, lane `l` |
| lane 0 record | 6 or 7 | mode, length, 4-byte state; the 7th byte is at offset 48 for the 3-byte form |
| lane 1 record | 6 | constant lane on BC1/BC4: mode, `n−1`, 4-byte state |
| lanes 2–7 | 6+ | endpoint planes; a `0x10` mode is a 1-symbol uniform prefix |
| index lanes 8–11 | 6–7 each | 2×2 quadrant symbols, one per block; start at round 6 |
| entropy payload | rest | one shared stream; active lanes read one byte per round in lane order |
| padding | 0–255 | zero fill to the next 256-byte boundary |

The lane roles and the 6/7-byte records are the public research model (SixNG and bcpack-rs);
only lane 1 and the entropy budget are independently exact here.

The first byte of a lane (`hdr0`) is `[z:2][length form:1][flush:1][q:4]` in the public
patent/research model. The length form is the part that changes the record size:

| `hdr0` bit 5 | Header | First-fragment budget |
|---|---|---|
| 0 | `[hdr0][n−1]` | `n+1` (1–256 bytes) |
| 1 | `[hdr0][lo][hi]` | `(lo \| hi<<8) + 257` (257–65,792 bytes) |

The 4-byte rANS state follows the header, most-significant byte first. With the 3-byte form,
the state's last byte is read in round 6 after the 48-byte interleave; the observed BC1 lane-0
modes use both length forms. Later fragments are in-band: a lane that exhausts its budget with output
left returns to `hdr0` and reads a new 2/3-byte header. The public MDU tables use `M = 2^16`,
`L = 2^24`, 16 models (`q = 0..15`) and legal end states `L`, `L >> 8`, `L >> 16`.

The header is followed by a single byte stream, not per-lane streams: every lane that still
needs input reads one byte in lane order each round. Entropy on 300 sampled chunks per codec is
7.96–7.99 bits/byte. No general-purpose wrapper was found; zlib, gzip, bzip2 and LZMA reject
the first bytes, and raw-deflate attempts on random data are not a valid stream.

### Block sizes and counts

Every BCPack chunk decodes to an exact whole number of BCn blocks: 8 bytes for BC1/BC4 and 16
bytes for BC3/BC5/BC6H/BC7. The 108,605 BCPack chunks cover 17 block counts, from 256 to
524,288:

| Blocks | Chunks | Blocks | Chunks |
|---:|---:|---:|---:|
| 4,096 | 29,160 | 2,048 | 2,519 |
| 16,384 | 25,929 | 32,768 | 2,244 |
| 65,536 | 16,822 | 262,144 | 638 |
| 1,536 | 15,412 | 131,072 | 288 |
| 1,024 | 11,561 | 2,560 | 175 |
| 8,192 | 3,352 | 512 | 168 |
| 3,072 | 166 | 768 | 135 |
| 524,288 | 27 | 3,584 | 8 |
| 256 | 1 | | |

All 108,605 divisions `virtual_size / block_bytes` are exact.

### The one exact lane record

The lane-1 record is the part of the stream that is byte-exact from public facts. On BC1 and
BC4 it is `[0xde][n−1][state: 4 bytes big-endian]`; re-encoding `N` copies of the constant
symbol from `x = L` with `M = 2^16`, `K = 32` reproduces `n−1` and the 32-bit state exactly.
The full corpus:

| Codec | Chunks | Block divisor | Matches |
|---|---:|---:|---:|
| BC1 | 102,725 | 1 | 101,833 |
| BC1 | 102,725 | 2 | 268 |
| BC1 | 102,725 | 4 | 597 |
| BC1 | 102,725 | 8 | 27 |
| BC4 | 77 | 1 | 77 |

The record's block count is `N/k`; the `k > 1` cases are a subset of the 131,072-, 262,144-
and 524,288-block chunks, consistent with `k` equal streams. No BC1 or BC4 first record failed
the check. BC3 and BC5 use `0xdb` (`K = 2`), but the first lane-1 record matches the constant
model in only 7/35 and 39/333 chunks (BC5: 36 at `N`, 3 at `N/4`); BC6H and BC7 had no matching
constant lane 1 for `K` in {2, 4, 8, 16, 32}. The exact result is therefore a BC1/BC4 header
check, not a general BCPack header parser.

Padding is zero-filled: every padding run starts with zero bytes (checked `min(16, pad)` bytes
of all 108,229 padded BCPack chunks: 102,368 BC1, 35 BC3, 76 BC4, 332 BC5, 268 BC6H, 5,150
BC7). That is how the physical ranges reach the next 256-byte boundary.

Three skipped entries, with the fields the container actually exposes:

| Pool / index | Descriptor | Allocation | Chunks |
|---|---|---:|---|
| `graphics/ttdb_graphics.ttpf` #309 | 128×128 BC1_UNORM, 8 mips, mode 2 (4 KiB) | 12,288 B | 1 × codec 1: packed 7,648, pad 544 |
| `ui/ttdb_ui.ttpf` #0 | 512×512 BC1_UNORM, 1 mip, mode 0 (64 KiB) | 131,072 B | 1 × codec 1: packed 87,933, pad 899 |
| `terrain/ttdb_terrain_0_7.ttpf` #1537 | 512×1024 BC7_UNORM, 8 mips, mode 4 (4 KiB) | 700,416 B | 4 × codec 7; top chunk packed 499,734 of 524,288, pad 234 |

The first two bytes of those payloads are lane 0 and lane 1 modes: `0x1f 0xde`,
`0x35 0xde`, and `0x30 0x30`. The full-corpus counts are the useful part:

| Codec | Chunks | Byte 1 value | Count |
|---|---:|---|---:|
| 1 (BC1) | 102,725 | `0xde` | 102,725 / 102,725 |
| 3 (BC3) | 35 | `0xdb` | 35 / 35 |
| 4 (BC4) | 77 | `0xde` | 77 / 77 |
| 5 (BC5) | 333 | `0xdb` | 333 / 333 |
| 6 (BC6H) | 268 | `0x30`–`0x36` | 268 / 268 |
| 7 (BC7) | 5,167 | `0x30`–`0x33` | 4,978 / 5,167 |

The byte-1 value is constant within a codec (BC1/BC4 use `0xde`, BC3/BC5 use `0xdb`), and it
is the only fixed-offset value found. Byte 0 is often `0x30`–`0x37` on BC1, but the rest
includes `0x14`–`0x1f`, `0x5f` and `0xde`, so it is not a magic. BC6H uses only `0x30`–`0x36`;
BC7 has no constant byte 1 and its remaining 189 records use `0xdb`, `0x1c`–`0x1e` and
similar lane modes. The 256-byte dumps are not committed because they are game bytes; the
tables above and the container descriptors are the readable facts.

### Entropy framing and the pixel subset

The public MDU tables and a research lockstep lane schedule reproduce the entropy layer on the
200-chunk test set without guessing pixel semantics. The set is 200 textures whose mip 0 is
exactly one BC1 BCPack chunk and whose mip 1 is fully zlib/raw, in store order. All 200 chunks
passed the hard check:

- all 12 lanes finished with exactly `N` symbols;
- every lane ended at `L`, `L >> 8` or `L >> 16`;
- the decoder consumed exactly the stripped packed length, with no wrapper.

This is a byte-exact convergence check of the entropy layer; it does not identify the endpoint
or index semantics. A public research repository, [`bcpack-rs`](https://github.com/Ahmet-Dedeler/bcpack-rs)
(MIT OR Apache-2.0), publishes candidate BC1 endpoint and index rules. An independent
implementation of that rule, using the public MDU tables and its published codebook, on the
same 200 chunks gives (PSNR: each 2×2 mip0 quadrant against the mip1 pixel at that position):

| Subset | Chunks | Candidate BC1 vs mip 1 (PSNR) | Flat mip 1 |
|---|---:|---|---:|
| no pre-start back-reference | 40 | 30.92–45.69 dB, median 40.27 | 23.78 dB |
| exactly one pre-start reference | 44 | median 7.50 dB with a zero window | 21.81 dB |
| two or more | 116 | not decoded | — |

This is **mip consistency, not a byte-exact decode**. The 160 chunks with a pre-start reference
need the fixed 256-slot predictor window for a full decode; the research branches fit only
part of it and the full table is not published. The 40 clean chunks agree with mip 1 at 31–46
dB, which is strong evidence the candidate rule is close; this is not a held-out score, and
mip 1 is a downsampled image, not the source blocks. No plaintext BCPack/BC block pair was
found publicly. The repository's own `master` (commit `3534739`, read 2026-11-21) claims BC1
decodes at 36.6 dB median on held-out textures and says 621 of 3,276 pair textures are fully
known; the gap is the unpublished window and the unresolved codebook entries, not a
byte-exact validation. The code is not adopted here.

At the time of that initial probe (before the clean-room decoder below was implemented), a
decoder still needed:

1. the 256-slot pre-start predictor window (the largest blocker: 160/200 pair chunks);
2. a stable 256-entry index codebook (two fits agree on 248/256; 8 rare mirror pairs are open);
3. 3-colour block handling (the candidate table is fitted on 4-colour blocks);
4. record layouts and semantics for BC3, BC5, BC6H and BC7 (BC4 lane 1 is exact; the rest are
   not);
5. a reference plaintext pair or a published bitstream specification for a byte-exact check.

Items 1–4 are now resolved by the clean-room decoder documented in
[Clean-room BCPack bitstream specification and decoder](#clean-room-bcpack-bitstream-specification-and-decoder)
below (`tools/bcpack.py`, `tools/bcpack_tables.py`, `tools/bcpack_tables_bptc.py` and
`tools/bcpack_history_calib.py`); item 5 remains external because no plaintext reference
encoder ships with the package.

### The shipped binaries

The install folder contains 16 PE DLLs and `gameos.xvd`; the game executable is not present.
`tools/pe_inspect.py` and a plain ASCII/UTF-16 strings search find no BCPack/BCn/DStorage
import, export or exact string in any of the 16 DLLs. The only decompression strings are
Bink2's video path; `XboxBvhBuilder` contains `RANS` only inside `TRANSFORM` and a PDB path.
`gameos.xvd` has no printable `BCPack`, `DStorage` or `DirectStorage`; its `MDU`/`BCn` byte
sequences occur as isolated 3-byte sequences, not as printable identifiers. No shipped decoder
or dispatcher was found.

## Public sources and licences

| Source | Facts used | Licence / use |
|---|---|---|
| [DSTORAGE_BCPACK_MODE](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/dstorage/enums/dstorage_bcpack_mode) | Modes 1–7 are BC1, BC2, BC3, BC4, BC5, BC6H, BC7; Xbox Series only | [Microsoft Learn Terms of Use](https://learn.microsoft.com/en-us/legal/termsofuse): informational, personal/non-commercial use; linked and paraphrased, no text or code copied |
| [DirectStorage overview](https://learn.microsoft.com/en-us/gaming/gdk/docs/features/console/storage/directstorage/directstorage-overview) | BCPack is a custom entropy coder for BCn data; colour endpoints are separated from palette indices and coded with rANS; decompression order is DEFLATE, BCPack, swizzle | Same Microsoft Learn terms; linked and paraphrased |
| [US Patent 11,234,023](https://patents.google.com/patent/US11234023B2) | Public MDU rANS architecture, fragment-header/model/lane-schedule description, and banded exponential frequency model structure | Public patent disclosure; concepts only, linked and paraphrased |
| [silicon-rans](https://github.com/microsoft/silicon-rans) | Exact MDU rANS tables (`mdu_minirans.v`) and legal end states used for the 200-chunk entropy check | [MIT](https://github.com/microsoft/silicon-rans/blob/main/LICENSE); tables used as facts, no Verilog copied into this repository |
| [Khronos Data Format Specification (BPTC / S3TC / RGTC)](https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html) | Standard BC1, BC3, BC4, BC5, BC6H and BC7 block layouts and partition/anchor tables | Public specification |
| [SixNG, `formats/06-bcpack.md`](https://github.com/walkthestars/GTA6-Asset-Info/blob/main/formats/06-bcpack.md) | Independent container/header observations and the exact-end-state hard check | [GPL-3.0](https://github.com/walkthestars/GTA6-Asset-Info/blob/main/LICENSE); notes only, no code copied |
| [SixNG, `formats/05-ttpf.md`](https://github.com/walkthestars/GTA6-Asset-Info/blob/main/formats/05-ttpf.md) | Independent codec-id table and pool structure | GPL-3.0; notes only |
| [rpf9-explorer](https://github.com/breadbbbb/rpf9-explorer) | The `TtpfReader` catalog checks, limitation strings, and lack of a BCPack decoder | [MIT](https://github.com/breadbbbb/rpf9-explorer/blob/main/README.md); no code copied |
| [bcpack-rs](https://github.com/Ahmet-Dedeler/bcpack-rs) | Candidate BC1 entropy/endpoint/index research cited in the prior 200-chunk check above | [MIT OR Apache-2.0](https://github.com/Ahmet-Dedeler/bcpack-rs/blob/master/LICENSE-MIT); research only, no code copied |
| SCARAB | No public GTA VI source found; the public SCARAB repositories are an unrelated GPL-3.0 Hollow Knight mod manager | Search on 2026-11-21; no facts used |

Microsoft's documentation gives the algorithm class and the mode enum, not the bitstream. The
public enum is documented for Xbox Series consoles; this note does not cover an Xbox One
variant.

## Clean-room BCPack bitstream specification and decoder

The unknowns noted in the earlier measurements are resolved in `tools/bcpack.py`,
`tools/bcpack_tables.py` and `tools/bcpack_tables_bptc.py` (pure Python standard library,
no GDK/NDA code, no keys, no game assets). The console-side hardware decompressor and GDK
implementation are not linked, copied or referenced; only plain retail chunk bytes were
analysed against public BCn and rANS specifications.

### 1. Entropy layer (all six active codecs)

Each BCPack chunk encodes `N = decoded_size / 8` synchronized lane elements across `nl`
interleaved 16-bit rANS lanes (`nl = 12` for BC1/BC3/BC4/BC5; `nl = 16` for BC6H/BC7).
16-byte block formats (BC3, BC5, BC6H, BC7) consume two lane elements per BC block.

Each lane reads one or more fragments until it has emitted `N` symbols:

```
hdr0 = z:2 | h3:1 | flush:1 | q:4
hdr1                               n = hdr1 + 1                (h3 = 0)
hdr1, hdr2                         n = hdr1 + 256*hdr2 + 257   (h3 = 1)
```

- Symbol width `w = default_width[lane] - z` (`1 <= w <= 10`).
- `q` (`0..15`) selects the banded exponential probability model for width `w`: band `0`
  covers symbol `0` and band `p >= 1` covers `[2^(p-1), 2^p)`, with total weight `65536`.
  `q = 0` is uniform; `q = 15` concentrates mass on symbol `0`. All 160 tables live in
  `tools/bcpack_tables.py`.
- rANS state: 32-bit unsigned integer with lower bound `L = 1 << 24`. Slot `0` is reserved
  as the end marker (symbol `0` owns cumulative range `[1, f0)`). While `state < L`, one
  byte is merged (`state = (state << 8) | byte`); while `state > L`, one symbol is decoded
  via `cf = state & 0xFFFF` and advanced via `f * (state >> 16) + cf - c`.
- `flush = 1`: terminates the fragment's rANS stream. Trailing zero bytes are omitted from
  the payload; once `n` bytes are consumed, the decoder shifts in up to 8 virtual zero
  bytes until `state >= L`, continues decoding, and requires the fragment to land on the
  exact home state `state == L`. The next fragment starts from `state = 0`.
- `flush = 0`: once `n` bytes are consumed and `state < L`, the next fragment header is
  read and decoding continues with the carried-over state under the new `(w, q)` model.
- After all lanes finish, any remaining bytes in the chunk must be zero.

### 2. Lane-array prefetch timing

Byte-to-lane interleaving is fixed by a deterministic per-cycle scheduler:

1. A lane holding a decoded symbol whose output slot is still full is output-stalled.
2. Non-stalled lanes in `PROC` advance their rANS state for any held symbol.
3. Lanes with an empty 1-byte input buffer request one byte (header or payload bytes while
   `in_rem > 0`). At most **8 bytes per cycle** are served in lane index order.
4. Any lane that requested a byte and received none freezes for the cycle (no state merge,
   no symbol release).
5. Served lanes (and lanes needing no byte) release any held symbol to their output slot,
   merge at most one byte when `state < L`, and decode the next symbol when `state > L`.
6. Once every active lane has a symbol in its output slot, one synchronized element is
   emitted and all slots clear.

### 3. Per-codec lane layouts and block rebuilders

| Codec | Format | Lanes | Default lane widths | Elements / block | Block rebuilder |
|---:|---|---:|---|---:|---|
| 1 | BC1 | 12 | `8 8 5 6 5 5 6 5 8 8 8 8` | 1 (8 B) | Implemented (`tools/bcpack.py`) |
| 3 | BC3 | 12 | `4 4 8 6 6 6 6 5 6 5 8 8` | 2 (16 B) | Implemented (`tools/bcpack.py`) |
| 4 | BC4 | 12 | `8 8 8 8 6 6 6 6 6 6 6 6` | 1 (8 B) | Implemented (`tools/bcpack.py`) |
| 5 | BC5 | 12 | `4 4 8 8 6 6 6 6 6 6 6 6` | 2 (16 B) | Implemented (`tools/bcpack.py`) |
| 6 | BC6H | 16 | `4 4 8 8 8 5 5 5 6 6 6 8 8 8 8 5` | 2 (16 B) | Implemented (`tools/bcpack.py`) |
| 7 | BC7 | 16 | `4 4 8 8 8 8 8 7 7 5 6 8 8 8 8 5` | 2 (16 B) | Implemented (`tools/bcpack.py`) |

For BC1, BC3, BC4 and BC5:

- **Lane 0** encodes a signed 8-bit backward distance `d` (zig-zag coded; split into two
  bit-interleaved 4-bit nibbles for 2-element formats BC3/BC5). `d = 0` predicts from the
  format's zero/midpoint default (`0` for RGB, `128` for 8-bit alpha/RG); `0 < d <= i`
  references previously decoded block `i - d`; `d > i` references entry `256 + i - d` of
  the format's fixed 256-entry built-in endpoint history.
- **Lane 1** is zero in all observed BC1/BC3/BC4/BC5 chunks (non-zero is rejected).
- **Endpoint lanes** store zig-zag deltas modulo each channel's bit depth (`32`/`64` for
  RGB565, `256` for 8-bit BC3 alpha / BC4 / BC5 endpoints).
- **Index lanes** store ranks into fixed codec lookup tables: 256-entry 2×2 quadrant tables
  for BC1 colour and BC3 colour, and 64-entry 3-bit index-pair tables for BC3 alpha, BC4
  and BC5.

For BC6H and BC7 (16 lanes, 2 elements per 16-byte block):

- **Lanes 0 and 1** carry independent 8-bit backward reference distances (bit-interleaved
  4-bit nibbles): `di = interleave(lane1[2i], lane1[2i+1])` for the index predictor and
  `de = (di + unzig(interleave(lane0[2i], lane0[2i+1]))) & 255` for the endpoint predictor.
  As with BC1–BC5, `d = 0` predicts from a fixed zero/midpoint state, `0 < d <= i` from
  block `i - d`, and `d > i` from entry `256 + i - d` of the format's built-in 256-entry
  endpoint or index history.
- **BC6H (`codec 6`):** lane 15 carries `(mode - 1, partition)` (modes 1–14 mapping to the
  14 D3D11/Khronos BC6H mode codes; 2-region modes use the first 32 BC7 2-subset partitions
  and anchors). Lanes 2–4 carry 16-bit interleaved base endpoint (`w`) deltas, lanes 5–7
  carry 10-bit interleaved `x` deltas, and lanes 8–10 carry 6-bit `y`/`z` deltas; modes 13
  and 14 remap the upper bits of `w` into standard BC6H bit order. Lanes 11–14 carry
  de-interleaved 4-bit row index deltas (rows 0–2 put even pixels on odd bits; row 3 swaps
  bit parity).
- **BC7 (`codec 7`):** lane 15 carries a pair `(a, b)` ranked by `((a + 1) * (b + 1), a)`
  into a 286-entry `(mode, partition, rotation, index_selection)` signature table. Lanes
  2–10 carry per-subset RGBA endpoint deltas (with p-bits stored in the LSB of channel 0
  when present; modes 4–5 reuse lanes 6, 7, 8, 10 for the secondary 2-bit index set). Lanes
  11–14 carry the primary row index deltas with the same de-interleave parity as BC6H.
  Anchor index top bits and mode-1 shared p-bits provide an exact per-block consistency
  check.
- For all six codecs, blocks are decoded in the surface's native tiled memory order, so
  standard AMD Gfx10 detiling (`tools/ttpf_mips.py`) applies unchanged after BCPack
  decompression.

### 4. Codec tables and history loading (strict vs. provisional)

`tools/bcpack_tables_bptc.py` embeds **only** public D3D11 / Khronos BPTC specification
data (2- and 3-subset partition masks and anchor pixel indices) and the 286-entry BC7
lane-15 signature permutation (`BC7_CONFIG`, built programmatically from closed-form
descending ranges for modes 1 and 3–7 plus the mode-0/mode-2 partition order, with 10
inferred-only mode-2 ranks flagged in `BC7_CONFIG_INFERRED`).

BCPack's 256-entry initial endpoint/index history buffers for BC6H and BC7 are unpublished
codec constants that cannot be derived from public specifications alone; they are **not**
embedded in `tools/bcpack_tables_bptc.py`. Self-contained BC6H/BC7 chunks (delta chains
staying within the chunk) decode strictly with `history=None`. Chunks that reference
built-in initial history cleanly refuse with a `--history <path>` hint until a locally
generated JSON file or directory (`bc6h.json` / `bc7.json`, e.g. from
`tools/bcpack_history_calib.py`) is supplied at run time:

- **BC1–BC5 JSON schema (optional `bc1.json`..`bc5.json` or `"bc1"`..`"bc5"` keys):**
  `codec`, `confirmed` and `provisional` maps from entry index (`"0"`..`"255"`) to endpoint
  tuples (`6` ints for BC1, `8` for BC3, `2` for BC4, `4` for BC5), plus optional
  `"replace": true` (default `false`, merging onto `tools/bcpack_tables.py`).
- **BC6H JSON schema (`bc6h.json` or `"bc6h"` key):** `endpointHistory` (256-list or
  index-keyed dict of `3×4` ints), `indexHistory` (16 ints), `endpointClass` and
  `indexClass` (`0` = unknown, `1` = fitted/provisional, `2` = last-resort, `3` =
  reference-confirmed at the recorded precision). Optional `endpointConfirmedBits`
  (`3×4` leading-bit counts, bounded by `16,10,6,6` per channel) and
  `indexConfirmedBits` (16 leading-bit counts, each `0..4`) use the same 256-list
  or index-keyed dict forms. Strict decoding requires class 3 and enough confirmed
  bits for every field the mode reads; provisional decoding labels insufficient
  precision. Omitted rows in an explicit depth table confirm no bits. Without a
  depth table, legacy class-3 entries retain full precision. Regenerate older
  calibration outputs to record their actual precision: exact references confirm
  only the bits their modes observe, including three-bit versus four-bit indices.
- **BC7 JSON schema (`bc7.json` or `"bc7"` key):** `historyEndpoints`,
  `historyEndpointBits`, `historyEndpointConfirmedBits` (24 ints or `[3][2][4]`),
  `historyIndices`, `historyIndexBits`, `historyIndexConfirmedBits` (32 ints or `[2][16]`),
  and `historyEndpointLastResort` (`[k, sub, e, c]` or `[k, slot]` pairs).

For BC1–BC5, `tools/bcpack_tables.py` (added earlier) holds the rANS probability tables and
default BC1/BC3/BC4/BC5 rank/history tables (extendable or replaceable via `--history`):

| Format | History source | Strict vs. provisional status | Notes |
|---|---|---|---|
| BC1 | `tools/bcpack_tables.py` (+ optional `--history`) | 248 confirmed / 7 provisional | Entry 0 unreferenced; provisional entries (6, 9, 67, 68, 71, 74, 122) uncertain by ≤1 green step |
| BC3 | `tools/bcpack_tables.py` (+ optional `--history`) | 0 confirmed / 35 provisional | All 256 colour-quadrant and 64 alpha-pair rank entries populated |
| BC4 | `tools/bcpack_tables.py` (+ optional `--history`) | 0 confirmed / 55 provisional | 64 pair entries populated; UI pool chunks reference additional unfitted entries unless supplied via `--history` |
| BC5 | `tools/bcpack_tables.py` (+ optional `--history`) | 3 confirmed / 211 provisional | 64 pair entries populated |
| BC6H | external `--history <path>` | Empty by default (`history=None`) | Self-contained chunks decode strictly; external JSON supplies class 1/2/3 entries |
| BC7 | external `--history <path>` | Empty by default (`history=None`) | Self-contained chunks decode strictly; 286 signature ranks (10 inferred-only refused in strict mode); partial-depth endpoints (`>= 4` bits) mid-range filled in provisional mode |

`tools/bcpack.py` and `tools/ttpf_mips.py` run in **strict mode by default**: any chunk
referencing an unconfirmed/unknown built-in history entry or inferred BC7 signature raises
`BcpackProvisionalError`. Pass `--history <path>` for locally calibrated history tables
(`tools/bcpack_history_calib.py`) and `--allow-provisional` to opt in to
provisional/last-resort entries (inspectable via `bcpack.decode_with_info`).

### 5. Validation on the 57-pool corpus

- **Full-corpus projection (`tools/ttpf_mips.py scan` across all 57 pools, including multi-slice layouts in `docs/textures_arrays.md` and DXGI 11):**
  - Baseline (2D compact + multi-slice + DXGI 11, zlib/raw only): **305,959** complete, **16,423** partial, **34,733** skipped (33,507 2D BCPack + 1,223 compact BCPack + 3 array BCPack; 0 format skips and 0 layout skips).
  - With all six BCPack codecs (BC1/BC3/BC4/BC5/BC6H/BC7): **357,115** complete (**+51,156**, **100%** of catalog entries), **0** partial, **0** skipped (0 BCPack skips, 0 format skips, 0 layout skips and 0 multi-slice skips).
- **Previously skipped examples above (`tools/ttpf_mips.py`):**
  - `graphics/ttdb_graphics.ttpf` #309 (128×128 BC1, 8 mips): `check` **pass** in strict mode (`max_error = 0.67`, `tail_origin_control = 102.3`).
  - `ui/ttdb_ui.ttpf` #0 (512×512 BC1, 1 mip): `check` **pass** in strict mode.
  - `terrain/ttdb_terrain_0_7.ttpf` #1537 (512×1024 BC7, 8 mips): `export` without `--history` cleanly refuses unknown built-in history with a `--history <path>` hint; `export --history <local_dir> --allow-provisional` writes the full 8-mip DX10 DDS (699,188 bytes) and `check --history <local_dir> --allow-provisional` passes (`max_error = 0.95`, `tail_origin_control = 129.91`).
- **Cross-codec verification:** 20/20 BC1 strict (`median_err = 0.68`), 7/7 BC3 provisional (`median_err = 1.44`, `max_err = 4.18`), 2/2 fitted BC4 provisional (`max_err = 0.43`; unfitted UI entries cleanly refused), 10/10 BC5 provisional (`median_err = 0.17`, `max_err = 0.85`).
- **Full-corpus BC6H/BC7 chunk verification (all 57 pools, pure-Python `tools/bcpack.py`, using locally calibrated `bc6h.json` / `bc7.json` passed via `--history`):**
  - **BC6H (`codec 6`):** **268 / 268** chunks (100%) decode in provisional mode (`0` strict because every chunk references built-in history, `0` refused).
  - **BC7 (`codec 7`):** **583 / 5,167** chunks (11.3%) decode in strict mode; **5,150 / 5,167** (99.67%) decode in provisional mode; **17 / 5,167** (0.33%) cleanly refuse when unknown built-in history references exceed the consistency budget.

## What `textures.rpf` holds

The 192.6 MB archive has no composition table elsewhere in the docs (only census rows). `tools/rpf9_carve.py`
reads 97.8% of its data span as 2,749 checksum-validated zlib streams (188.3 MB stored, 487.9 MB inflated;
1.8% zero padding, 0.5% unread). By inflated size the streams fall into two groups:

| Group | Streams | Inflated | Content |
|---|---:|---:|---|
| 400-byte + 32,768-byte pairs | 1,313 + 1,313 (+2 lone 400-byte) | 43.6 MB | a small texture-dictionary resource: system page, then graphics page |
| everything else | 121 | 444.4 MB (91%) | 11 streams of 16 MB or more (largest 32 MB), 16 of exactly 64 KiB, 62 under 64 KiB |

**The pairs are one template, filled 1,313 times.** Every 400-byte stream is immediately followed by a 32,768-byte
one, and the 8 bytes stored just before 1,312 of the large streams are identical. The 400-byte system pages
have only 4 distinct contents, and 1,312 of the 1,315 are byte-identical: a dictionary of two textures, each
described as 128 x 128 with format code `0x5f` (95, `DXGI_FORMAT_BC6H_UF16` if the codes are DXGI numbers, as
for the BC1/BC7 codes elsewhere in this document), one name per texture. Two 128 x 128 BC6H textures are
2 x 16,384 = 32,768 bytes, exactly the graphics-page size. The graphics pages differ: 1,117 distinct contents
among the 1,313 (the most repeated one 60 times). So the archive stores 1,313 instances of one HDR texture pair
with per-instance data, the shape of a per-tile or per-cell map. One of the other three header variants describes
172 x 172 textures with code `0x62` (98, BC7).

**The 121 other streams** start with no known magic (`RDS5`, `0DRP`, `TTPF`, `RSC8`, `PSIN`, `RBF0`, DDS) in
their first 4 KiB, and the large ones are high-entropy (7.3 to 7.7 bits per byte over the first 64 KiB, except
one at 4.1). They are left as opaque raw data here.

**They are not copies of the texture pools.** Each of the 121 streams was compared with every chunk of all 57
`.ttpf` pools in the package (1,026,833 chunks: 918,223 zlib, 102,725 BCPack BC1, 5,167 BC7, 718 other BCPack
codecs, 5 raw), as stored and, for zlib chunks, inflated: the SHA-1 of the whole stream against the SHA-1 of each
chunk, and the stream's first 64 bytes against each chunk's first 64 bytes (85 of the 121 streams start with more
than four distinct byte values; the other 36 start with near-constant bytes and only take the whole-content test).

| Result | Streams |
|---|---:|
| Identical to a pool chunk (stored zlib bytes and inflated content) | 1 (4 KiB, a shared `data/ttdb` pool) |
| Same first 212,464 to 582,338 bytes as a 4 MiB `data/ttdb` chunk, then different | 3 (6, 12 and 32 MiB) |
| Same first 1,010 to 2,800 bytes as hundreds of chunks (a common leading pattern, not a copy) | 3 |
| No match | 114 |

Sizes of the 121: 62 below 64 KiB, 16 of exactly 64 KiB, 32 from 64 KiB to 16 MiB, 11 of 16 MiB or more. So,
apart from one 4 KiB stream, the opaque data of `textures.rpf` is not pool texture data, and three of the large
streams share a long identical start with a pool chunk and then diverge.

Following them past the divergence with an in-order alignment (extend identical runs; on a mismatch, resync on a
64-byte piece of the chunk taken up to 64 KiB ahead, found within the next 256 KiB of the stream): in the 6 MiB and
32 MiB streams, five different 4 MiB `data/ttdb` chunks each reappear in order for 65.8% to 97.5% of their bytes,
in short runs broken by 164 to 1,007 gaps, while the stream advances about 1.4 to 1.5 times the chunk's length; in
the 12 MiB stream only 8.6%. That five different chunks align this well with the same two streams says the chunks
are alike among themselves, so repetitive content may account for part of it: this measures shared runs, not a
copy, and what the streams are remains open. Method: the carve's
inflated streams and `tools/ttpf_mips.py`'s `Pool` chunk table; counts only.

**Measured keying and format codes (ASSET-274):**
All 1,315 system streams are RAGE `pgDictionary<grcTexture>` pages containing descriptor pointers and a 32-bit
`joaat32` hash table:
- **Format codes are 1:1 DXGI format enums:**
  - `0x5f` (95) = `DXGI_FORMAT_BC6H_UF16` (1,313 pairs, 128 x 128).
  - `0x62` (98) = `DXGI_FORMAT_BC7_UNORM` (1 pair, 172 x 172: UI switch portraits `switch_jason` and `switch_lucia`).
  - `0x4d` (77) = `DXGI_FORMAT_BC3_UNORM` (1 pair, 512 x 512: UI indicators `qrcode_logo` and `busy_spinner`).
- **Keying:** Each dictionary indexes textures by the lower-case ASCII asset name hash (`joaat32`).
  - The 1,313 pairs are each keyed by `waterfog2` (`0x9b9b31df`) and `waterfog` (`0xf1389e16`).
  - The count of 1,313 matches the water-bearing cells of the 36 x 41 (1,476 total) world streaming grid
    (`StreamingDesc/NumberOfTilesX, Y` in `docs/world.md:361`); whether the remaining 163 cells are inland/dry
    cells with no waterfog dictionary is a hypothesis.

## Next step

1. **Port BC6H/BC7 BCPack decoders to `TtpfReader` (`rpf9-explorer` / `tools/rpf9dump`).**
   Compact 2D mip tails, multi-slice layouts, DXGI 11, and strict BCPack BC1/BC3/BC4/BC5
   decoding are now live in `rpf9-explorer` (carried by the `e293917` submodule pin,
   `breadbbbb/rpf9-explorer#7`; `ASSET-44`). Remaining C# follow-ups are BC6H/BC7 block
   rebuilders (plus optional external history tables), preserving lower-mip zlib/raw
   fallback when a higher mip is refused in strict mode, and fixing `Core/TtpfReader.cs:70`
   (`Patterns[2]`), which currently uses Navi 1x's `(-3, 3)` at bits 6..7 instead of
   RDNA2/RbPlus `(3, -3)` (`GFX10_SW_64K_D_RBPLUS_PATINFO`).
2. **Wire optional external `--history` tables into C# `TtpfReader`.** `tools/bcpack_history_calib.py`
   now derives local `bc1.json`..`bc7.json` history files into `<work>` for the pure-Python
   decoder (`tools/bcpack.py` / `tools/ttpf_mips.py --history <path>`), either by exact
   algebraic inversion from reference block/DDS pairs (`--ref-dds`, `--ref-pairs`, yielding
   confirmed strict-mode entries) or by read-only pool self-calibration across mixed
   BCPack + raw/zlib mip chains (yielding provisional entries). Once BC6H/BC7 land in
   `TtpfReader`, it can consume the same local directory.

## Reproduce

The normal `make_game.py` assets stage now fills missing and partial BCPack DDS entries in the same pool manifests used for model texture lookup. It uses strict decoding by default and discovers an optional local reconstruction profile at `<work>/bcpack-calibration.json`. The pass stamp in each pool manifest hashes the decoder modules (`ttpf_assets.DECODER_SOURCES`) and the profile, so a pool is done again after either changes. The single-entry exporter accepts the same profile through `--calibration`. Build instructions, the JSON schema and the game-independent test are in the [calibrated export guide](../tools/bcpack_calibration/README.md).

`tools/ttpf_mips.py` is the read-only measurement and decode tool (with `tools/bcpack.py`
enabled in strict confirmed-history mode by default), and `tools/bcpack_history_calib.py`
derives local `--history <path>` JSON tables using only the Python standard library:

```
python -I tools/bcpack_history_calib.py <game root>/data/ttdb --out <work>/bcpack_history [--limit N] [--check]
python -I tools/bcpack_history_calib.py --verify <work>/bcpack_history [<game root>/data/ttdb]
python -I tools/ttpf_mips.py scan <game root> --json scan.json [--tuples]
python -I tools/ttpf_mips.py layout <pool.ttpf> <index>
python -I tools/ttpf_mips.py check <pool.ttpf> <index> [--allow-provisional] [--history <path>]
python -I tools/ttpf_mips.py export <pool.ttpf> <index> <out.dds> [--allow-provisional] [--history <path>]
python -I tests/test_bcpack.py
python -I tests/test_bcpack_history_calib.py
python -I tests/test_bcpack_tool.py
python -I tests/test_tex_headers.py
python -I tests/test_ttpf_mips.py
```

`scan` reproduces the 357,115 / 98,608 / 210,292 / 48,215 baseline split (the pre-DXGI-11
98,603 / 210,292 / 48,220 split with the 5 DXGI 11 records moved into `reader_full`), reports
both the zlib/raw layout projection (`after_full` / `after_partial` / `after_skip`) and the
full 6-codec BCPack projection (`after_bcpack_full` / `after_bcpack_partial` /
`after_bcpack_skip`), and adds the layout and recovery counters; `--tuples` adds the 84
known-size rejected tuples. `check` reports the per-mip downsample error. `export` writes a
full BC or validated 2D uncompressed chain when every source mip uses a supported codec. None
of the tests need game files: `tests/test_bcpack.py` builds synthetic rANS BCPack streams from
scratch for all six codecs (including synthetic external history JSONs for BC6H/BC7),
`tests/test_bcpack_history_calib.py` tests exact reference inversion (`--ref-pairs`,
`--ref-dds`), prefix rANS lane decoding, and mixed-mip pool self-calibration on synthetic
inputs across all six codecs, `tests/test_tex_headers.py` tests the layout rule on synthetic
headers, and `tests/test_ttpf_mips.py` builds synthetic zlib, uncompressed 2D, and BCPack (BC1
and BC7) RDS5 pools.

The earlier BCPack header counts and 200-chunk entropy check came from a read-only
aggregation over the same `Pool` parser using the public MDU tables from `silicon-rans`;
those probes are not committed because `tools/bcpack.py` supersedes them.

`rpf9dump ttpf` writes one `manifest.json` per pool with a `Limitation` field for every
catalog record, and its summary line has the same `ok` / `skip` totals. The first split above
came from a read-only mirror of `TtpfReader`'s catalog checks: it mmapped each pool and read
only the 24-byte RDS5 header, the chunk table, the ETPF records and the chunk ranges, and it
wrote no texture data. `tools/ttpf_mips.py scan` is that mirror, now committed with the
compact-layout and BCPack work; it reproduces the baseline split and adds the recovery
columns.

A pool's `manifest.json` is published only when the DDS of every `CanExtract` record was
written or already present. If any write fails, `rpf9dump ttpf` keeps the DDS it did write,
removes an older `manifest.json` for that pool and exits 1. Records with `File: null` are
deliberate skips and do not fail the pool. The assets stage exports a pool again when its
manifest is missing, unreadable or lists a `File` that is not on disk, so a manifest from an
older build that names an absent DDS is repaired on the next run.
