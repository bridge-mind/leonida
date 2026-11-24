# RPF9 archives and the carve tools

GTA VI ships its assets in RPF9 archives. The **table of contents** (names, sizes, offsets and flags) is
encrypted/packed; this project does not decode it and does not search for keys. The **data area** is mostly
zlib streams laid end to end, so it can be measured and carved without the TOC. Names in the recovery view
are offset-based (`recovered/...`), not the original paths.

## What is readable

- The header: magic `RPF9` (stored as `9FPR` little-endian), sizes/counts, and the data start at `u32 +4`.
  The TOC area immediately after the header is high-entropy and opaque.
- The data area: zlib streams, Bink 2 video (`KB2[a-z]`) in `movies.rpf`, and raw AWC banks in `audio.rpf`.
- From the existing survey in [docs/design.md](design.md): most archives are 93-99% plain zlib
  (`levels.rpf` 98.3%, about 1.1M files); `audio.rpf` is 99.9% raw AWC, not zlib; about half of
  `story000.rpf` and `shared_minigame000.rpf` is non-zlib. Unread bytes are recorded as ranges (the `unread`
  list in `rpf9_carve.py`'s `index.json`), never guessed.

## Whole-package name coverage (reported in #1790)

The recovery view names streams by offset, but some original paths can be read from the payload itself:
mount XMLs and similar text in the data area list them in plain text, directory form included. #1790 ran one
name-recovery walker over all 22 archives of a retail package (150.2 GiB) and reports how far that gets. These
counts are as reported there; they were not re-run for this page.

| | Count |
|---|---:|
| Streams | 1,489,677 |
| Streams with a name | 177,087 (11.89%) |
| Unique paths | 155,214 |
| Unique identifiers | 200,184 |
| Path mentions | 526,581 |

| Stream type (walker's classes) | Named / streams | Share |
|---|---:|---:|
| `sgd5_shader_dxbc` | 25,043 / 25,043 | 100.00% |
| `xml` | 191 / 332 | 57.53% |
| `rsc50` | 123,889 / 531,370 | 23.32% |
| `xml_chunk` | 109 / 504 | 21.63% |
| raw | 26,766 / 690,003 | 3.88% |
| `PSIN` | 465 / 15,380 | 3.02% |
| `0DRP` | 624 / 227,045 | 0.27% |

Three findings from the same run:

- The paths come from the payload, not the TOC. One 8 KB stream in `common.head.bin` declares 473 `.bk2` paths on its
  own.
- Stream alignment is not universal. Inflated streams sit on 4 KiB boundaries in `data.rpf` (295 of 301 checked)
  but not in `levels.rpf` (8 of 26), `packs.rpf` (0 of 301) or `textures.rpf` (0 of 3), so a walker that assumes
  4 KiB steps misses most streams in three of those four. `tools/rpf9_carve.py` does not assume an alignment: it
  continues after the bytes each stream consumed.
- `anim.rpf` is the richest archive for names (64,785 named streams, 367,190 path mentions, almost all `.clip`).
  `movies.rpf` produced no zlib streams, and `script.rpf` and `story000_script.rpf` no names.

The 11.89% is the ceiling of this method, not a progress figure: the rest of the names are in the encrypted TOC,
which this project does not decode. #1790 also confirms the TOC is not ECB (no duplicate 16-byte blocks in 62,487).
The full name index is not published; the issue offers salted hashes of the paths and identifiers so that another
run can be compared without publishing a name.

## Stream roles

In the measured carve indexes, a resource's system prefix can be separated from its graphics pages without
decoding the resource: the prefix ends at the first stream whose inflated size is a multiple of `0x10000`, plus at
most one following non-multiple stream (a spill reached by a `0x5…` pointer). This reproduces the first
`42 40 00 00` graphics page on all 36,911 resources with one in `packs.rpf` and the 20–21, 40–41 and 70–70.8 GiB
`levels.rpf` windows, and on all 2,478 two-stream `0DRP` navmesh pages. It is a boundary, not a page test: only
17,994 of the 40,038 graphics pages measured (44.9%) are multiples of `0x10000`, so the stream at the boundary is
often the last system page. `42 40 00 00` stays the direct graphics-page marker.

An inflated size of exactly `0x800000` is a chunk head, not a continuation. All 43 in `core_bvh4.rpf` follow a
208–544-byte header stream and are followed by a separate continuation stream; the eight in the other measured
indexes are resource or graphics pages. [bvh.md](bvh.md) records the same 8 MiB-first relationship from the payload
side.

The boundary rule is independently read by meetox80/6stars. No 6stars code, prose or table was used; the
repository has **no licence** (all rights reserved), and every count above was re-measured from this package. See
[related_projects.md](related_projects.md).

## Carving

```powershell
python -I tools/rpf9_carve.py <archive.rpf> <out_dir> [--start N] [--limit BYTES] [--index-only]
```

The walker starts at `--start` or the header's data offset, finds a zlib header (`78 01/5e/9c/da`) or a Bink 2
magic, inflates the stream, and continues after the bytes it actually consumed. `--index-only` records the
streams without writing the inflated files; `--limit` bounds a sample; `--start` starts at a known offset.

`--start` must be 0 or more and `--limit` at least 1 (decimal or `0x` hex). An out-of-range value stops with a
one-line error and exit code 2 before any output is written, rather than silently scanning everything or nothing.

`<out_dir>/index.json` contains:

- `summary`: archive, files, compressed/inflated bytes, Bink 2 bytes/files, zero padding, unread bytes, the number of
  unread ranges (`unread_ranges`), span;
- `files`: `off`, `stored`, `size`, `pre` (bytes before the stream), `head` (first 16 inflated bytes), and for
  Bink 2 the `w`/`h`/`frames` fields;
- `unread`: one row per gap that is not all zero bytes, between two streams, before the first or after the last:
  `off`, `size`, and `entropy`, the Shannon entropy in bits/byte of its first `sampled` bytes (at most 1 MiB). A
  gap is the whole region between two streams, so a mostly-zero gap with a few other bytes is one unread range. An
  all-zero gap counts toward `zero_padding` only. The sizes add up to `unread_bytes`.

Measured on `data.rpf` (481,544,304 bytes; span 481,474,576), the 20,268,240 unread bytes (4.2%) are **377
ranges**: median 7,452 bytes, largest 1,149,226; 7 below 1 KiB, 297 from 1 KiB to 64 KiB, 73 of 64 KiB or more
(16,565,898 bytes). By entropy:

| Bits/byte | Ranges | Bytes |
|---|---:|---:|
| 7.5 or more | 278 | 18,257,882 |
| 6 to 7.5 | 32 | 1,350,501 |
| 4 to 6 | 37 | 460,464 |
| below 4 | 30 | 199,393 |

So 90% of the unread bytes look like compressed or encrypted data, with no zlib or Bink 2 header the walker accepts.
The rest is lower-entropy structure worth a look with a hex editor at the listed offsets. The list holds offsets
and sizes only; keep the index in your work folder like the carved files.

The walker advances by `len(chunk)`, not by the 1 MiB read request. A short final slice used to make `p`
overshoot EOF, so `used` exceeded the real stream size and every trailing stream was skipped; the regression
test `tests/test_rpf9_carve.py` builds a two-stream archive to keep that fixed.

A gap between streams counts as zero padding when every byte in it is zero, otherwise as unread.
`rpf9_carve.py` checks a gap in 1 MiB pieces. The other scanners that keep these counts (`rpf9_names`,
`bvh_index`, `navmesh_index`, `navmesh_decode`, `rbf0_index`, `anim_index`, `restofworld_index`, `story_index`)
use `tools/zero_span.py`, which compares the gap with zeros 4 MiB at a time and stops at the first window with a
non-zero byte. They used to slice the gap out of the mapped archive, which copies all of it at once. The largest
gap in `story000.rpf` is 8.9 GiB, so that needed 8.9 GiB of memory. `tests/test_zero_span.py` runs all nine
scanners on a 48 MiB gap and checks that each one's Python heap peak stays under 24 MiB.

The carve is read-only and works on any `.rpf`, `.ttpf` or archive-like file. It does not decrypt the TOC, use
keys, or write anything outside the output directory.

## Recovery tools

`tools/rpf9dump` is a bulk exporter built on the Core library from
[breadbbbb/rpf9-explorer](https://github.com/breadbbbb/rpf9-explorer) (a `third_party` submodule; the
related-projects page records the licence boundary). It carves validated resources; it does
not decrypt the TOC either.

```powershell
dotnet build tools/rpf9dump/rpf9dump.csproj -c Release
rpf9dump ttpf    <pool.ttpf>  <out>
rpf9dump models <archive.rpf> <out>
rpf9dump textures <archive.rpf> <out>
rpf9dump audio  <audio.rpf>  <out>
rpf9dump xma    <audio.rpf>  <out>
```

`audio` writes the AWC banks plus `audio_manifest.json` (stream ids, codec, channels, rate, duration); `xma`
writes RIFF XMA2 WAVs that ffmpeg decodes. `models`/`textures` write OBJ/DDS plus a manifest. The tool's
`ArchiveReader` recovery names are offset-based, and its structured path validates XML/PSIN, models and
textures; the original RPF paths still require the missing executable/key.

`tools/manifest.py` is independent of RPF: it hashes a directory tree and verifies it later, with `--jobs` on
both paths. It is used to check exported assets without keeping a second copy.

`tools/rpf9_names.py` is the lightweight lookup pass: it walks the same data area, inflates a bounded prefix
of each zlib stream and indexes the measured `name.#dr`/`#ft`/`#dd` spellings with their archive offsets. It
does not decode a resource or need the TOC; see [resource_names.md](resource_names.md).

### The trailer at EOF

The header's `u32 +12` is the size of a block that ends the file: the last `+12` bytes of an RPF9 are not part of
any stored file (ASSET-318, measured on the 22 RPF9 archives of the 1.0.2.1 package, offsets and counts only).
`+12` is a multiple of 16 in all 22 (64 to 119,520 bytes); `+8` is always larger than `+12` except in `levels.rpf`
and `core_bvh4.rpf`, and its role is not established.

| Evidence | Archives |
|---|---|
| The data before it ends in zeros up to a 4 KiB boundary, and exactly `+12` non-zero bytes follow to EOF | 10: `anim`, `audio`, `common`, `data`, `levels`, `movies`, `packs`, `shared_minigame000`, `story000`, `core_bvh4` |
| Files are packed at 16-byte alignment; walking the zlib streams to the end leaves exactly `+12` bytes after the last one | 4: `core_licenses_data`, `core_restofworld`, `shared_minigame000_bvh4`, `story000_bvh4` |
| Undecided: the last stored file is not zlib (or the walk stops at a non-zlib file), so its end is not known without the TOC | 8: `appdata0`, `navmeshes`, `script`, `shaders`, `textures`, `core_restofworld001`, `story000_script`, `story000_restofworld` |

No archive contradicts the rule, but 8 don't test it. In the 14, the trailer starts with no zlib or Bink 2 header and is
high-entropy like the TOC after the header: 7.95 to 8.00 bits per byte in the nine of 3 KiB or more, and 6.6 to 7.9
in the five smaller ones (random 144-byte blocks average about 6.7); what they hold is not
decoded here. `tools/bink_index.py` counts it as `rpf_trailer` instead of unread bytes, only for an `RPF9` header
whose `+12` is a multiple of 16 and fits in the tail after the last stream; in `movies.rpf` it is the
18,928 bytes after 3,884 zeros that [movies.md](movies.md) listed as unread.

### Viewing a header in ImHex

[tools/imhex/rpf9.hexpat](../tools/imhex/rpf9.hexpat) is a pattern for the [ImHex](https://imhex.werwolv.net/) hex
editor. It names the magic, the data offset at `+4` and the trailer size at `+12`, shows the u32 at `+8` unnamed,
and shows the range from `+0x10` to the data offset (the rest of the header and the TOC) as one opaque byte array.
It decodes nothing in that range. `python -I tests/test_imhex_patterns.py` runs it on a synthetic header when ImHex is installed.

## Limits

- The TOC is the hard boundary. File names, original paths and the exact file boundaries come from the TOC,
  so they are unavailable until the executable (or a documented key/schema) is part of the project. This
  repository does not search for or distribute keys.
- Some archives are not plain zlib (`audio.rpf`, parts of `story000.rpf`/`shared_minigame000.rpf`); the
  recovery tools handle the formats they know and report the rest as unread.
- Bink 2 video is carved as whole `.bk2` streams. FFmpeg builds generally expose only Bink 1; the package's
  `Bink2Scarlett.dll` decodes Bink 2 (see the movie notes).
- Extracted assets are for local use with your own copy. Never commit or share them.

## See also

- [docs/design.md](design.md) - the asset-stage table, coverage numbers and the recovery approach.
- [docs/page_directory.md](page_directory.md) - the `0DRP` page directory layout, `q38` resource boundaries and
  `tools/page_index.py`.
- [docs/story.md](story.md) - measured `story000`/`shared_minigame000` composition, nested RPF9 sub-archives and the `PSIN`/`RSC8` headers.
- [docs/restofworld.md](restofworld.md) - measured composition of the rest-of-world and script archives.
- `docs/audio.md` - AWC layout and the audio exporters, once the audio guide is merged.
- the related-projects page - the rpf9-explorer/PercyStar/SCARAB licence boundaries.
