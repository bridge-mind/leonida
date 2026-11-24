# TTPF arrays, cubemaps, and 3D volumes

Measured across all 57 retail Xbox Series RDS5 texture pools (1.0.2.1; 357,115 TTPF descriptors,
1,026,833 chunks) via `python -I tools/ttpf_mips.py scan`. The layouts and decoders described
below are implemented in `tools/tex_headers.py` and `tools/ttpf_mips.py`. No texture bytes, DDS
files, or carved payloads are included; every number below is an aggregate, a descriptor field,
a byte offset, or a validation metric.

## Outcome

| | Records |
|---|---:|
| TTPF descriptors scanned | 357,115 |
| multi-slice descriptors (`dimension != 1` or `depth > 1`) | 697 |
| 2D arrays (`dimension = 1`, `depth > 1`, `mode = 1`) | 127 |
| cubemaps (`dimension = 2`, `depth = 6`, `mode = 18/20/22/24`) | 567 |
| 3D volumes (`dimension = 3`, `depth > 1`, `mode = 2`, `mips = 1`) | 3 |
| **layout-validated (byte-exact span)** | **697 / 697 (0 mismatches)** |
| **complete / partial / skipped (zlib/raw only)** | **690 / 4 / 3** |
| **complete / partial / skipped (with BCPack BC1/BC3/BC4/BC5)** | **694 / 3 / 0** |
| **43-bit `(meta, extra)` chunk-page metadata validated** | **357,115 / 357,115 (0 failures)** |

All 697 multi-slice descriptors — the 678 formerly flagged as arrays/volumes plus the 19
two-slice 2D arrays formerly miscounted as unknown-format 2D records — have exact, validated
layouts and DX10 DDS export support (`Texture2DArray`, `TextureCube`, and float16 `Texture3D`)
in `tools/ttpf_mips.py`.

## Descriptor word `b` bitfields

The 16-byte ETPF descriptor record is decoded by `parse_etpf_texture_record` in
`tools/tex_headers.py`. Word `b` packs five fields:

| Field | Expression | Bits | Meaning |
|---|---|---:|---|
| `non_tail` | `b & 7` | 0..2 (3) | streaming snapshot count (cubemaps) / non-tail mip count |
| `format` | `(b >> 3) & 0x7F` | 3..9 (7) | DXGI format ID (`0..127`) |
| `depth` | `((b >> 10) & 0x7FF) + 1` | 10..20 (11) | slice count, face count (`6`), or volume depth |
| `dimension` | `(b >> 21) & 3` | 21..22 (2) | `1` = 2D / 2D array, `2` = cubemap, `3` = 3D volume |
| `mode` | `b >> 23` | 23..31 (9) | low flag bit plus 4-bit Gfx10 tile mode `(mode >> 1) & 0xF` |

### Why the earlier reading saw `format >= 128` and halved depths

An earlier reading took 8 bits for `format` (`(b >> 3) & 0xFF`) and started `depth` at bit 11,
which stole bit 10 (bit 0 of `depth - 1`). Whenever the true `depth` was even (`depth - 1` odd,
bit 10 = `1`), `format` gained `+128` and `depth` was halved:

- **19 two-slice 2D arrays** (`depth = 2`, `depth - 1 = 1`) appeared as `depth = 1` with raw
  formats `156/199/211/226` (`28/71/83/98 | 0x80`).
- **99 three-slice 2D arrays** (`depth = 3`, `depth - 1 = 2`) appeared as `depth = 2` with low
  formats (`28/71/83`) and "3 chains".
- **9 even-depth 2D arrays** (`depth` in `{8, 16, 32, 48, 64}`) appeared as `depth` in
  `{4, 8, 16, 24, 32}` with `format >= 128` and `2 * depth` chains.
- **567 cubemaps** (`depth = 6`, `depth - 1 = 5`) appeared as `depth = 3` with `format >= 128`.
- **3 volumes** (`depth = 8` or `32`) appeared as `depth = 4` or `16` with `format >= 128`.

With the 7-bit `format` and 11-bit `depth` fields, every multi-slice descriptor carries a valid
DXGI format ID in `0..127` and its true slice/face/depth count.

The observed `(mode >> 1) & 0xF` tile modes partition the 697 multi-slice descriptors cleanly:

| `(mode >> 1) & 0xF` | `mode` | Shape | Records |
|---:|---:|---|---:|
| 0 | 1 | 2D array (`dimension = 1`, `depth` in `{2, 3, 8, 16, 32, 48, 64}`) | 127 |
| 1 | 2 | 3D volume (`dimension = 3`, `depth` in `{8, 32}`, `mips = 1`) | 3 |
| 9 | 18 | cubemap (`dimension = 2`, `depth = 6`, `non_tail = 0`) | 90 |
| 10 | 20 | cubemap (`dimension = 2`, `depth = 6`, `non_tail = 1`) | 75 |
| 11 | 22 | cubemap (`dimension = 2`, `depth = 6`, `non_tail = 2`) | 122 |
| 12 | 24 | cubemap (`dimension = 2`, `depth = 6`, `non_tail = 3`) | 280 |

## 2D arrays (`dimension = 1`, `depth > 1`, `mode = 1`) — 127 records

### Layout

Implemented in `array_surface_layout` (`tools/tex_headers.py`):

- Tile mode is `SW_64KB_D` (64 KiB tiles), in **mip-major** order.
- Total allocation span is exact `depth * single_slice_size` (127 / 127 byte-exact, 0
  mismatches).
- When `first_tail < mips`, the packed mip tail sits at offset `0` and occupies `depth * 65536`
  bytes, with slice `s` at `s * 65536`.
- Standalone mips follow from `first_tail - 1` down to `0`. Each standalone mip occupies
  `depth * region` bytes, with slice `s` at `mip_offset + s * region`.

### Corpus breakdown

| True `depth` | Records | Full (zlib/raw) | Partial (zlib/raw) | All-BCPack (zlib/raw) |
|---:|---:|---:|---:|---:|
| 2 | 19 | 18 | 1 | 0 |
| 3 | 99 | 99 | 0 | 0 |
| 8 | 2 | 2 | 0 | 0 |
| 16 | 1 | 0 | 1 | 0 |
| 32 | 2 | 0 | 1 | 1 |
| 48 | 2 | 1 | 1 | 0 |
| 64 | 2 | 0 | 0 | 2 |
| **Total** | **127** | **120** | **4** | **3** |

Under zlib/raw only, 120 arrays export complete chains, 4 are partial (mip 0 touches BCPack on
some slices while lower mips are zlib/raw), and 3 are all-BCPack. With `tools/bcpack.py`
(BC1/BC3/BC4/BC5), the 4 partial arrays complete (**124 complete**) and the 3 all-BCPack arrays
move to partial (**3 partial, 0 skipped**).

### Pixel validation

Verified with `tools/ttpf_mips.py check` (BC5 downsample MAE per slice vs. a slice-major
negative control):

| Pool / index | Dimensions | Format / mips | Per-slice max error | Overall max | Slice-major control |
|---|---|---|---:|---:|---:|
| `ttdb_peds_0_3.ttpf` #4429 | 512×512×2 | BC5, 8 mips | 0.24 / 0.24 | 0.24 | 57.81 |
| `ttdb_peds_0_3.ttpf` #299 | 512×512×3 | BC5, 8 mips | 0.17 / 0.22 / 0.17 | 0.22 | 56.93 |
| `ttdb_graphics.ttpf` #300 | 256×256×8 | BC5, 9 mips | `[0.18, 0.33, 3.67, 2.67, 0.17, 4.1, 2.12, 0.0]` | 4.10 | 87.73 |

`ttdb_peds_0_3.ttpf` #4429 is one of the 19 former "format 211" 2D records, confirming that
those 19 entries are standard 2-slice mip-major 2D arrays.

## Cubemaps (`dimension = 2`, `depth = 6`, `mode = 18 + 2 * non_tail`) — 567 records

### Layout

Implemented in `cubemap_surface_layout` (`tools/tex_headers.py`):

- Each cubemap stores `non_tail + 1` self-contained full-mip cubemap snapshots back-to-back in
  the RDS5 chunk stream. Earlier chunks hold lower-resolution streaming snapshots.
- The **final chunk** holds the full-resolution 6-face cubemap in **slice-major `SW_4KB_S`**
  order: face `f` (`0..5`) starts at `data_offset + f * face_size`.
- When `non_tail > 0` and `6 * face_size >= 36 * 4096`, the final chunk is zero-padded to a
  64 KiB boundary; otherwise its size is exact `6 * face_size`.
- All 567 cubemap spans are byte-exact (567 / 567, 0 mismatches), and **567 / 567 are 100%
  zlib/raw (0 BCPack)**.

| `mode` | `non_tail` | Tile mode | Snapshots | Records |
|---:|---:|---:|---:|---:|
| 18 | 0 | 9 | 1 | 90 |
| 20 | 1 | 10 | 2 | 75 |
| 22 | 2 | 11 | 3 | 122 |
| 24 | 3 | 12 | 4 | 280 |
| **Total** | | | | **567** |

An earlier note mistakenly argued that 512×512 mode-24 cubemaps could not fit six full faces
because it overcounted BC7 and BC1 top-mip sizes by 4×. A 512×512 BC7 top mip is
`128 * 128 * 16 = 262,144` bytes (not `1,048,576`), and a 512×512 BC1 top mip is
`128 * 128 * 8 = 131,072` bytes (not `524,288`). Six full 512×512 BC7 mip chains take
`6 * 352,256 = 2,113,536` bytes (padded to `2,162,688` bytes = 528 4 KiB pages) plus `835,584`
bytes of earlier snapshots, totaling `2,998,272` bytes exact.

### Pixel validation

Verified with `tools/ttpf_mips.py check` (BC1_SRGB 6-face downsample MAE vs. a tail-origin
negative control):

| Pool / index | Dimensions | Format / mode / mips | Snapshots / `data_offset` | Per-face max error | Overall max | Tail control |
|---|---|---|---|---:|---:|---:|
| `ttdb_maps_7_7.ttpf` #7150 | 128×128×6 | BC1_SRGB, mode 20, 8 mips | 2 / 24,576 | 1.82 / 2.50 / 2.05 / 1.81 / 0.68 / 1.58 | 2.50 | 83.61 |
| `ttdb_maps_0_7.ttpf` #4965 | 512×512×6 | BC1_SRGB, mode 24, 10 mips | 4 / 425,984 | 3.20 / 1.83 / 4.55 / 2.16 / 1.71 / 4.26 | 4.55 | 109.37 |

Prefiltered reflection cubemaps can show slightly higher single-face MAE (~9 levels) on tiny
mips because cubemap prefiltering convolves across adjacent faces, while still beating the
tail-origin control (~130–140) by more than 13×.

## 3D volumes (`dimension = 3`, `depth > 1`, `mode = 2`, `mips = 1`) — 3 records

### Layout

Implemented in `volume_surface_layout` (`tools/tex_headers.py`) and `detile_volume`
(`tools/ttpf_mips.py`):

- Standard AMD Gfx10 3D `SW_4KB_S` micro-tiling with a single mip level (`mips = 1`).
- **3 / 3 are 100% zlib/raw (0 BCPack)** and **3 / 3 match their allocation spans byte-exact**.

### Validation

Each volume was detiled in 3D and compared against a 2D stacked-slice control using mean
adjacent-Z voxel difference:

| Pool / index | Dimensions | DXGI format | B/voxel | 3D tile | `padded_d` | Span (B) | 3D Z-diff | 2D stacked control | Ratio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `ttdb_graphics.ttpf` #901 | 256×32×8 | 54 `R16_FLOAT` | 2 | 8×16×16 | 16 | 262,144 | 0.004238 | 0.124305 | 29.3× smoother |
| `ttdb_graphics.ttpf` #575 | 32×32×32 | 10 `R16G16B16A16_FLOAT` | 8 | 8×8×8 | 32 | 262,144 | 0.036721 | 0.565428 | 15.4× smoother |
| `ttdb_graphics.ttpf` #200 | 32×64×32 | 10 `R16G16B16A16_FLOAT` | 8 | 8×8×8 | 32 | 524,288 | 0.016793 | 0.110316 | 6.6× smoother |

The intra-tile bit order (`VOLUME_PATTERNS_4K` in `tools/ttpf_mips.py`) and the tile extents
(`_VOLUME_4K_TILE` in `tools/tex_headers.py`) are Mesa addrlib's `GFX10_SW_4K_S3_PATINFO` (the RbPlus table is
identical), the same as rpf9-explorer's `VolumePatterns`. Until this was corrected, the rows for 2-, 4- and
16-byte elements matched no GFX10 pattern in addrlib, and the 2-byte tile was taken as 16×8×16 instead of
8×16×16. Only #901 uses one of those rows. The adjacent-Z check above could not catch it (the wrong order still
kept most Z neighbours together: 5.3× against the control), but the X/Y neighbours could: as a share of the mean
difference between random voxel pairs, adjacent-Y differences were 0.357 with the old order and are 0.067 now, and
adjacent-Z 0.029 and 0.007 (adjacent-X 0.007 and 0.015). The two 8-byte volumes use the unchanged row.

## 43-bit `(meta, extra)` streaming chunk-page metadata

Decoded by `decode_etpf_chunk_pages` and `etpf_streaming_padding` in `tools/tex_headers.py`, and
verified across all **357,115 / 357,115** descriptors (`chunk_metadata_ok = 357,115`,
`chunk_metadata_fail = 0`):

- Combine `val43 = (extra << 11) | meta`.
- The low 20 bits pack up to four zero-terminated 5-bit chunk page-count codes (in 4 KiB pages):

| Code | Pages | Code | Pages | Code | Pages |
|---:|---:|---:|---:|---:|---:|
| 1..8 | 1..8 | 14 | 48 | 23 | 96 |
| 10 | 12 | 15 | 64 | 24 | 144 |
| 11 | 16 | 16 | 128 | 26 | 272 |
| 12 | 24 | 21 | 18 | 27 | 384 |
| 13 | 32 | 22 | 80 | 28 | 528 |

- Bits `9..18` of `extra` are always `0`.
- The upper bits `extra >> 19` encode `(align_group, pad_pages) = ((extra >> 19) & 3, extra >> 21)`
  for `non_tail == 3` compact 2D textures:
  - `align_group == 1`: `pages[0] + pad_pages == 16`
  - `align_group == 2`: `pages[0] + pages[1] + pad_pages == 16`
  - `align_group == 3`: `pages[0] + pages[1] + pad_pages == 8`
  - `0` for cubemaps and `non_tail == 1`.

## Updated 57-pool corpus totals

With 2D compact layouts plus 2D arrays, cubemaps, and 3D volumes enabled in
`tools/ttpf_mips.py`:

| Configuration | Complete | Partial | Skipped | Remaining skip breakdown |
|---|---:|---:|---:|---|
| Zlib/raw only (2D compact) | 305,264 | 16,419 | 35,432 | 33,507 2D BCPack + 1,223 compact BCPack + 697 multi-slice + 5 DXGI 11 |
| Zlib/raw only (+ multi-slice) | **305,954** (+690) | **16,423** (+4) | **34,738** (−694) | 33,507 2D BCPack + 1,223 compact BCPack + 3 array BCPack + 5 DXGI 11 (**0** layout skips) |
| With BCPack BC1/BC3/BC4/BC5 (2D only) | 353,146 | 2,515 | 1,454 | 752 BC6H/BC7 BCPack + 697 multi-slice + 5 DXGI 11 |
| With BCPack BC1/BC3/BC4/BC5 (+ multi-slice) | **353,840** (**99.08%**, +694) | **2,518** (+3) | **757** (−697) | 752 BC6H/BC7 BCPack + 5 DXGI 11 (**0** layout skips, **0** multi-slice skips) |

`tools/ttpf_mips.py` exports DX10 DDS for 2D arrays (`Texture2DArray`), cubemaps
(`TextureCube`), and validated float16 3D volumes (`Texture3D`). Upstream C# `TtpfReader` parity
has landed in `breadbbbb/rpf9-explorer` PRs #5, #6, and #7 (compact layouts, multi-slice shapes,
and BCPack BC1/BC3/BC4/BC5) and is carried by the `e293917` submodule pin.

## Preview a DDS plane

The header lists the mip count, array size, volume depth and cube faces available
to the PNG decoder. A cube array's `array_size` counts whole cubes.

```powershell
python -I tools/dds_png.py cubemap.dds --info
python -I tools/dds_png.py cubemap.dds face.png --layer 1 --face nz --mip 2
python -I tools/dds_png.py array.dds layer.png --layer 3 --max-size 256
python -I tools/dds_png.py volume.dds slice.png --mip 1 --z 2
```

Indices start at zero. Faces are `px`, `nx`, `py`, `ny`, `pz` and `nz`
(positive and negative X, Y and Z). A legacy cube can omit faces. The default
preview uses mip 0 of the first layer, first stored face or first volume slice.
For volumes, `--z` is relative to the selected mip, whose depth halves at each
level down to one. `--max-size` downsizes that plane after decoding. For a
seekable file, the converter reads the header and the selected plane.

For uncompressed data, a declared top-level row pitch applies to each layer
or slice. Lower mips use tightly packed rows, as written by the project exporter.

## Reproduce

```powershell
python -I tools/ttpf_mips.py scan <game root> --json scan.json
python -I tools/ttpf_mips.py layout <pool.ttpf> <index>
python -I tools/ttpf_mips.py check <pool.ttpf> <index> [--allow-provisional]
python -I tools/ttpf_mips.py export <pool.ttpf> <index> <out.dds> [--allow-provisional]
python -I tests/test_tex_headers.py
python -I tests/test_ttpf_mips.py
```

`scan` verifies all 357,115 descriptors (including all 697 multi-slice layouts and 43-bit chunk
page metadata) and emits both the zlib/raw (`after_full` / `after_partial` / `after_skip`) and
BCPack BC1/BC3/BC4/BC5 (`after_bcpack_full` / `after_bcpack_partial` / `after_bcpack_skip`)
totals. `tests/test_tex_headers.py` and `tests/test_ttpf_mips.py` exercise the 2D array,
cubemap, 3D volume, and chunk-metadata rules on synthetic inputs without game files.
