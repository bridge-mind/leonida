# Story and minigame archives

This page measures the `story000` and `shared_minigame000` archive family in the Xbox 1.0.2.1
package. It is a container survey, not a resource decoder: the RPF9 table of contents is still
encrypted, so counts come from the data area and from fixed headers, never from names or paths.
Everything below is aggregate. No game payload, path or name is stored in this repository.

## Archives and index

Sizes are exact bytes from the package; `data offset` is the little-endian u32 at file offset 4.
A full index of `story000.rpf` was not completed (see the layout section); its headline row is the
first 1 GiB of the data area, and the full-file non-inflating signature scan is reported separately.

| Archive | File size | Data offset | Scope | Streams | Stored | Inflated | Zero padding | Unread | Readable | Elapsed |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `story000.rpf` | 20,022,508,672 (18.65 GiB) | 0xA120 | first 1 GiB of data area | 55,284 | 1,002,413,457 | 3,963,061,696 | 67,544,007 | 3,803,929 | 93.4% | 44.0 s |
| `shared_minigame000.rpf` | 810,581,008 (773.03 MiB) | 0xFB0 | full | 5,073 | 443,771,951 | 928,713,043 | 4,723,814 | 362,081,227 | 54.7% | 24.5 s |
| `story000_script.rpf` | 78,023,728 (74.41 MiB) | 0x2B0 | full | 2,632 | 20,408,094 | 74,729,504 | 4,585,959 | 53,028,987 | 26.2% | 4.5 s |
| `story000_restofworld.rpf` | 198,354,992 (189.17 MiB) | 0xA40 | full | 631 | 188,163,116 | 255,343,231 | 862,330 | 9,326,922 | 94.9% | 1.0 s |
| `story000_bvh4.rpf` | 5,632 | 0x340 | full | 4 | 1,795 | 16,677 | 2,849 | 156 | 37.4% | 0.004 s |
| `shared_minigame000_bvh4.rpf` | 61,987,040 (59.12 MiB) | 0x4F0 | full | 2,560 | 59,142,787 | 970,044,012 | 2,672,073 | 170,916 | 95.4% | 2.65 s |

Readable is the carver's `stored / span`; Bink 2 streams count as readable. The first `story000`
row scans file offsets `0xA120`-`0x4000A11F`.

A first full 20 GB walk was stopped at 72% (about 14.4 billion bytes, 106,000 streams) with one process at
8.5 GB in memory. A later full walk with `tools/story_index.py` (after #2726 stopped the scanners copying the
gaps between streams) completes in about 10 minutes and is the full index below; no zlib candidate in the
archive inflates past 512 MiB.

**Full index.** `python -I tools/story_index.py story000.rpf out.json` over all 20,022,508,672 bytes:
**141,385 streams**, 10,231,531,520 bytes stored (51.1% of the span) and 17,094,635,373 inflated, 182,253,050
bytes of zero padding and 9,608,682,854 unread (the raw `RSC8` region). By class: zero-prefix 56,228, `0DRP`
48,084, `42 40 00 00` 5,589, `PSIN` 5,348, type-hash 1,393, clip-stream 875, Bink 2 264, MoVE 119, XML 32, and
23,327 streams in 12,463 other first-word classes. By 1 GiB bucket (a stream counts where it starts, so a bucket
whose streams run into the next one is shown as 100%):

| GiB | Streams | Stored | Inflated | Readable | Main classes |
|---:|---:|---:|---:|---:|---|
| 0 | 55,282 | 1,002,378,056 | 3,963,018,176 | 93.4% | zero-prefix 22,679, `0DRP` 21,980 |
| 1 | 18,210 | 1,045,208,660 | 1,484,948,432 | 97.3% | zero-prefix 15,640 |
| 2 | 13,901 | 1,055,312,755 | 1,880,941,536 | 98.3% | zero-prefix 7,464, `0DRP` 3,067, type-hash 459 |
| 3 | 2,020 | 1,068,996,430 | 1,487,761,024 | 99.6% | zero-prefix 1,302 |
| 4 | 2,340 | 463,305,742 | 617,702,940 | 43.1% | `PSIN` 735, zero-prefix 732, MoVE 119 |
| 5 | 0 | 0 | 0 | 0.0% | — |
| 6 | 0 | 0 | 0 | 0.0% | — |
| 7 | 0 | 0 | 0 | 0.0% | — |
| 8 | 0 | 0 | 0 | 0.0% | — |
| 9 | 0 | 0 | 0 | 0.0% | — |
| 10 | 0 | 0 | 0 | 0.0% | — |
| 11 | 0 | 0 | 0 | 0.0% | — |
| 12 | 0 | 0 | 0 | 0.0% | — |
| 13 | 31,714 | 638,135,783 | 1,356,633,857 | 59.4% | `0DRP` 17,584, `PSIN` 4,537, zero-prefix 3,658 |
| 14 | 13,803 | 1,055,814,600 | 1,879,292,779 | 98.3% | zero-prefix 4,725, `42 40 00 00` 3,282, `0DRP` 1,751 |
| 15 | 3,928 | 1,060,254,688 | 1,582,210,186 | 98.7% | `0DRP` 3,702, Bink 2 78, XML 31 |
| 16 | 169 | 1,155,442,136 | 1,155,442,136 | 100.0% | Bink 2 169 |
| 17 | 10 | 1,107,140,984 | 1,107,140,984 | 100.0% | Bink 2 10 |
| 18 | 8 | 579,541,686 | 579,543,323 | 83.4% | Bink 2 7, XML 1 |

The bounded windows below were the measurements before the full index; they agree with it where they overlap.

## `story000.rpf` layout

The archive is not uniform. A 64 MiB magic-count scan of the whole 20 GB file and the full index above separate
these regions:

| Region | zlib carver result | Full-file signature result |
|---|---|---|
| GiB 0-4 | 91,753 streams (full index); GiB 0-3 93.4-99.6% readable, GiB 4 43.1%; `0DRP` pages, `zero-prefix` pages, type-hash and clip headers, `PSIN` from GiB 4 | no RSC8 |
| GiB 5-12 | 0 streams in every 1 GiB bucket (full index) | 225,281 `RSC8` headers (in GiB 6-13), all version 8001, u32@8 sum 2,677,370,916 |
| GiB 13 | 31,714 streams, 59.4% readable (full index): `0DRP` 17,584, `PSIN` 4,537, zero-prefix 3,658 | the last 6,736 `RSC8` headers |
| GiB 14 | 13,803 streams, 98.3% readable (full index); 3,282 `42 40 00 00` pages (780 in the first 256 MiB) | no RSC8 |
| GiB 15 | 3,928 streams (full index): `0DRP` 3,702, Bink 2 78, XML 31 | first Bink 2 bucket |
| GiB 16-18 | Bink 2 only: 169, 10 and 7 streams (plus one XML at 18 GiB) | 264 valid Bink 2 streams in all, 3,327,650,728 bytes |

Window samples (each starts at the stated file offset and scans 256 MiB unless noted):

| Start | Streams | Stored | Inflated | Readable | Dominant class |
|---:|---:|---:|---:|---:|---|
| 0 (1 GiB) | 55,284 | 1,002,413,457 | 3,963,061,696 | 93.4% | `0DRP` 21,980; zero-prefix 22,681 |
| 2 GiB | 8,837 | 255,734,022 | 744,610,656 | 95.3% | `0DRP` 3,067 |
| 3 GiB | 476 | 268,264,094 | 341,274,256 | 99.9% | large zero-prefix pages |
| 4 GiB | 502 | 273,507,830 | 342,088,096 | 100% (crosses window) | large zero-prefix pages |
| 5 GiB | 0 | 0 | 0 | 0% | raw, no zlib |
| 6 GiB | 0 | 0 | 0 | 0% | raw, no zlib |
| 8 GiB | 0 | 0 | 0 | 0% | raw `RSC8` |
| 10 GiB (1 GiB) | 0 | 0 | 0 | 0% | raw `RSC8`; 25,685 headers in that GiB |
| 12 GiB | 0 | 0 | 0 | 0% | raw `RSC8` |
| 14 GiB | 2,547 | 268,670,234 | 472,245,792 | ~100% (crosses window) | `42 40 00 00` 780; zero-prefix 1,100 |
| 16 GiB | 42 | 277,746,976 | 277,746,976 | 100% | Bink 2 |
| 18 GiB | 2 | 208,322,272 | 208,322,272 | 77.6% | Bink 2 |

`RSC8` headers by 1 GiB bucket: 6 GiB 33,846; 7 GiB 41,391; 8 GiB 33,334; 9 GiB 29,687;
10 GiB 25,685; 11 GiB 25,983; 12 GiB 28,619; 13 GiB 6,736. The u32 at +8 sums to 2.49 GiB; the raw
region itself is about 7 GiB, so that field is not the total stored size of each resource. Every one
of the 225,281 headers has version 8001 (`0x1F41`); no `RSC7` header was found anywhere in the file.

Bink 2 by 1 GiB bucket: 15 GiB 78; 16 GiB 169; 17 GiB 10; 18 GiB 7. The 264 streams carry valid
`KB2[a-z]` headers and dimensions from 540x960 to 2560x1440. They account for 3.10 GiB of the
archive and are the only video class measured here.

## Readable payload classes

### `story000.rpf`, first 1 GiB

| Class | Streams | Stored | Inflated | Measured field |
|---|---:|---:|---:|---|
| `0DRP` page directory at +16 | 21,980 | 124,984,888 | 2,336,795,552 | version 122; page count u64 at +0x20 (top: 2 = 10,603, 3 = 5,996, 1 = 1,231, 4 = 646) |
| zero-prefix page | 22,681 | 737,279,203 | 1,234,168,960 | first 8 bytes zero |
| `80 80 80 80` buffer page | 3,594 | 43,479,838 | 203,792,288 | 16 bytes `0x80`, then zeros |
| type-hash header | 934 | 2,247,133 | 5,946,240 | u32 type, u32 8, u32 count, u32 0 |
| clip stream | 804 | 7,326,470 | 13,957,328 | u32 2 or 3, `01 01 00 00` |
| other raw starts | 5,291 (3,137 classes) | 87,095,925 | 168,401,328 | 3,036 classes occur once |

No `PSIN`, `RBF0`, `RSC7`, `RSC8`, `AWC`, `FSSM`, `SCRP`, `YSC` or valid `KB2` stream starts in
this 1 GiB. A prefix scan of all 55,284 inflated streams found no `RSC7`/`RSC8`, `DDS `, `DXT1`,
`DXT3`, `DXT5`, `ATI2`, `BC5`, `BC7` or `TXTR` magic in the first 4 KiB; the only `0DRP` matches
were the 21,980 page directories.

### `shared_minigame000.rpf`, full

| Class | Streams | Stored | Inflated | Measured field |
|---|---:|---:|---:|---|
| `42 40 00 00` resource page | 1,255 | 304,089,245 | 656,802,672 | 0x4042 marker; four u32s then float records |
| zero-prefix page | 1,782 | 104,870,151 | 188,050,712 | first 8 bytes zero |
| `0DRP` page directory at +16 | 787 | 6,251,535 | 19,538,880 | version 122; page counts 1 = 578, 2 = 192, 4 = 4, 6 = 3 |
| `PSIN` | 73 | 164,290 | 556,280 | 70 full six-section chains; 3 without a chain |
| `63 30 00 00` mesh page | 45 | 4,470,231 | 14,894,128 | vertex blocks and meshlet groups; see [the section below](#63-30-00-00-mesh-pages) |
| XML text | 16 | 11,500 | 57,317 | UTF-8 BOM + `<?xml` |
| other text | 10 | 131,472 | 6,985,204 | printable NUL-terminated strings |
| raw `RSC8` header | 1,957 | - | - | version 8001; u32@8 sum 32,904,308 |

The 362 MB unread region is not empty: a 360 MB gap holds five page-aligned nested `RPF9`
headers and the 1,957 `RSC8` headers; the archive has 49 nested headers in total. The `RSC8`
fixed header is 0x30 bytes; a NUL-terminated name follows it before a pointer into the
`0x50xxxxxx` range. The name field was present in the sampled headers. No `RSC7`, `AWC`, `FSSM`,
`SCRP` or `YSC` container was validated.

### `63 30 00 00` mesh pages

The 45 `shared_minigame000.rpf` streams that start with `63 30 00 00` are mesh geometry, not loose float records: each is a run of
16-byte-aligned chunks of two kinds, a **vertex block** and a **meshlet group**. `tools/meshlet_page.py` walks
them; over all 45 pages it parses every chunk and leaves 0 bytes unaccounted for. The same walk over the whole
archive finds 290 more streams that start with a meshlet group instead (continuation pages, below). Counts are
from package 1.0.2.1; no value from the pages is stored here.

Confidence: **high** = holds in every chunk of all 45 pages and is checked by the tool; **medium** = consistent
over all pages, but the meaning is inferred; **open** = not decoded.

**Vertex block** (137 in the 45 pages, 301,916 vertices; 10 to 27,280 vertices per block):

| Offset | Field | Measured | Confidence |
|---:|---|---|---|
| +0x00 to +0x58 | 23 slot words: low byte = tag, high 24 bits = byte offset / 2 from the block start; 0 = slot unused | slots 0-4 used, 5-22 zero in all 137 | high |
| +0x00 | slot 0 = `63 30 00 00`: tag 0x63, offset 0x60 | all 137; this is the page magic | high |
| +0x5C | u32 | 0x200 in all 137 | high (value), open (meaning) |
| +0x60 | vertex streams, stored in slot order 0, 2, 1, 3, 4 | sizes below in all 137 | high |

| Slot | Tag | Bytes/vertex | Content | Blocks | Confidence |
|---:|---:|---:|---|---:|---|
| 0 | 0x63 | 12 | float3 position | 137 | high (layout); medium (position): all 301,916 finite and within ±300 of the origin |
| 1 | 0x26 | 4 | 4 bytes, alpha-like 4th byte | 120 | medium: 4th byte is 255 for all 300,074 vertices; RGBA8 colour is the likely reading |
| 2 | 0x63 | 12 | float3 normal | 137 | high: length 1 ± 0.02 for all 301,916 |
| 3 | 0x42 | 8 | float2 | 137 | medium: all finite; texture coordinates are the likely reading |
| 4 | 0x42 | 8 | float2 | 120 | medium: all finite; a second texture coordinate set is the likely reading |

The vertex count is (slot 2 offset − 0x60) / 12; every other slot's size is its stride times that count, and the
block ends 8 × count bytes after its last slot. Blocks come in two layouts: 120 with all five slots and 17 with
slots 0, 2 and 3 only. 811,377 of the 905,748 position floats have a zero low byte, so most positions look
quantised to 16 mantissa bits. The positions are local (model space): every coordinate lies within ±300 of the
block's origin, so they are not compared with the map bounds in `docs/world.md`.

**Meshlet group** (107 in the 45 pages):

| Offset | Field | Measured | Confidence |
|---:|---|---|---|
| +0 | u32: bit 31 set, bits 0-30 = meshlet count K | 2,119 meshlets | high |
| +4 | u32 triangle count T | 254,180 triangles | high |
| +8 | u32 vertex reference count V | 263,895 references | high |
| +12 | K × 16-byte meshlet records | see below | |
| +12 + 16K | T × u32 triangles: three 7-bit meshlet-local indices (bits 0-6, 7-13, 14-20), meshlet index (bits 21-31) | every index below its meshlet's vertex count; every meshlet index matches its record | high |
| +12 + 16K + 4T | V × u16 vertex references into a vertex block | | high |

Meshlet record:

| Offset | Field | Measured | Confidence |
|---:|---|---|---|
| +0 | u32: bits 0-6 = vertices − 1, bits 7-13 = triangles − 1 | sums equal V and T in all 107 groups; 1,971 of 2,119 meshlets have 128 vertices | high |
| +0 | bits 14-31 | not decoded; vary between meshlets | open |
| +4 | u16 first vertex reference, u16 first triangle | running sums in all 2,119 | high |
| +8 | u32 | 0x007FF7FF in 2,092 of 2,119 | open (likely culling bounds) |
| +12 | u32 | 0x7FF in 2,087 of 2,119 | open |

So the meshlets hold at most 128 vertices and 128 triangles, which fits mesh-shader geometry. In every group the
references cover 0 to max exactly once or more (no gaps), so a group indexes one vertex block of that many
vertices. 78 of the 107 groups match a block of the same vertex count in the same page and 104 in the same or an
adjacent page (in archive order); the other 3 are small and match no block in those pages. A page can therefore hold a group
whose block is in the page before it.

**Pages.** 18 pages are one block and one group; the others interleave several blocks and groups (one page has
3 blocks and no group). 39 of the 45 pages are distinct by SHA-256 (one page occurs twice and one six times).

**Continuation pages.** A full-archive walk also accepts streams that start with a meshlet group which passes
every check above: 290 such streams (8,237,552 bytes inflated), with 748 groups, 7,015 meshlets, 742,141 triangles
and 5 vertex blocks. 109 parse to the end, 16 end inside a group that continues in the next stream, and 165 run
into a `42 40 00 00` chunk. That chunk uses the same slot-word header (slot 0 is tag 0x42 at offset 0x80) with other
tags (0x21, 0x28), so the `42 40 00 00` resource pages are probably the same vertex-block family with other vertex
formats. Their strides are not measured here, and the tool stops at them instead of guessing.

| Scope | Streams | Blocks | Vertices | Groups | Meshlets | Triangles |
|---|---:|---:|---:|---:|---:|---:|
| `63 30 00 00` pages | 45 | 137 | 301,916 | 107 | 2,119 | 254,180 |
| group-led pages | 290 | 5 | 608 | 748 | 7,015 | 742,141 |
| total | 335 | 142 | 302,524 | 855 | 9,134 | 996,321 |

**`42 40 00 00` vertex blocks (ASSET-146).** These are sized now, and `tools/meshlet_page.py` walks past them
instead of stopping. The header is 0x80 bytes:

| Offset | Field | Measured | Confidence |
|---:|---|---|---|
| +0x00..+0x58 | 23 slot words, encoded as in the `63 30` block; slot 0 is tag 0x42 at +0x80 | all blocks | high |
| +0x5C | u32 | 0x200, the same word as the `63 30` block's +0x5C | high (value), open (meaning) |
| +0x60..+0x68 | f32 × 3, position bias | all blocks | high (see below) |
| +0x6C | u32, axis order of the stored position | 0, 1 or 2 in every block | high (see below) |
| +0x70..+0x78 | f32 × 3, position scale | all positive | high (see below) |
| +0x7C | f32, scale of the type-8 (snorm16) UV slots | 4.0f in every block | high (value); meaning from rpf9-explorer |

**The slot tag byte encodes stride and type.** Bits 5-7 times 4 are the stride and the low nibble is the type,
`tag = (stride / 4) << 5 | type`. This is how `third_party/rpf9-explorer/Core/RageModelDecoder.cs` (lines 700-748)
reads vertex slots. It sizes all 99,184 slot words of the archive's `63 30` and `42 40` blocks:

| Tag | Type | Stride | Slots (blocks) | Reading |
|---|---:|---:|---|---|
| 0x63 | 3 | 12 | `63 30` 0, 2 | 3 × f32 (position, normal) |
| 0x42 | 2 | 8 | `63 30` 3, 4; `42 40` 0 (13,545), 3, 4, 14, 15 | 2 × f32; in `42 40` slot 0 the packed position |
| 0x26 | 6 | 4 | `63 30` 1, 12; `42 40` 1, 12, 13 | RGBA8 colour |
| 0x28 | 8 | 4 | `42 40` 3, 4, 14, 15 | 2 × snorm16 UV × the +0x7C scale (rpf9-explorer) |
| 0x21 | 1 | 4 | `42 40` 2 (13,119) | octahedral normal, 10 + 10 bits (below) |
| 0x27 | 7 | 4 | `42 40` 2, 11, 18 | 10:10:10 unit vector: the normal in slot 2, a tangent in slots 11 and 18 (below) |

The slots used in `42 40` blocks are 0-4, 11-15 and 18; slot 15 occurs in 78 blocks and slot 18 in 770.
`tools/meshlet_page.py` sizes slots with this rule (`tag_stride`), for the types above only, and reads the same
1,569 streams with identical totals. The vertex count is the slot-0 span divided by 8, and every stream is
stride × count bytes. Of every 16-byte-aligned `42 40 00 00` word in `shared_minigame000.rpf`,
14,671 have a header that fits this rule exactly. Each of them is followed by another `42 40` block (9,234), a
meshlet group (5,109), a group that continues in the next stream (216), the end of the stream (110) or a `63 30`
block (2), so nothing is left between chunks. 246 more hits have slot tags with no stride; they are not block
headers and are left alone. One `63 30` block also has a slot 12 (tag 0x26, 4 bytes) between slots 1 and 3; it
sizes with the same tag rule.

**Slot 0 is a packed position.** It's the `(8, 16)` position form that
`third_party/rpf9-explorer/Core/RageModelDecoder.cs` (lines 772-790) already reads for models. Take the 8 bytes as one
u64 `q`:
- `a` = bits 8-31 (24 bits);
- `b` = bits 32-47 shifted up 8, or'd with bits 0-7 (24 bits);
- `c` = bits 48-63 (16 bits).

Then `(a, b, c) = bias + scale × (a, b, c)`. The axis word says how the three are stored: 0 is (z, y, x), 1 is
(y, z, x), 2 is (x, y, z). `tools/meshlet_page.py` decodes the positions with this rule (`positions_4240`) and reports
counts only.

I checked the rule independently over `shared_minigame000.rpf`, counts only:
- **Field ends.** Across triangle edges of the 757 groups in the first 300 streams, the chance that a bit of `q`
  differs between the two ends falls steadily towards bits 31, 47 and 63 (0.008, 0.008, 0.020) and jumps back to
  about 0.45 at bits 32 and 48. So the fields end exactly there. Bits 0 and 8 differ about half as often as their
  neighbours (0.24); they are the lowest bits of `b` and `a`.
- **Scales.** Over 1,517 blocks, the edge-length ratio between the first two 16-bit fields follows the ratio of
  +0x70 to +0x74 (r = −0.59 on the logs). For x and y the bias plus half of `scale × 2^24` lands on 0 for many
  objects: within 5% of |bias| for 32% of axes, against at most 1.3% for any other power of two tried.
- **Neighbouring faces.** These are the counts `--faces` gives over all 1,569 streams:

| Positions | Finite | Within 300 m | Shared edges | Faces within ~25° (dot > 0.9) | Opposed (dot < −0.9) |
|---:|---:|---:|---:|---:|---:|
| 15,956,983 | 15,956,983 | 15,945,428 | 17,335,198 | 14,501,371 (83.7%) | 44,414 (0.26%) |

Deliberately wrong reads of the first 300 streams don't hold up:
- **The three u16 at bytes 0-5 as a position:** 40.9% within 25° and 20.0% opposed. The rule gives 79.3% and 0.09%.
- **The low bytes swapped between `a` and `b`:** the same at 25°, but fewer neighbours within 0.1° than with no low
  bytes at all. Within 0.1° the counts are 7.56% swapped, 8.59% with the high 16 bits only, and 9.19% with the rule.

The 4-byte attribute streams (tags 0x21, 0x26, 0x27, 0x28) are still not decoded.

With these blocks sized, the archive walk accepts 1,569 streams and leaves 0 unparsed:

| Scope | Streams | `63 30` blocks | `42 40` blocks | Vertices | Groups | Meshlets | Triangles |
|---|---:|---:|---:|---:|---:|---:|---:|
| before | 335 | 142 | not sized | 302,524 | 855 | 9,134 | 996,321 |
| now | 1,569 | 145 | 13,545 | 302,584 + 15,956,983 | 12,697 | 159,387 | 16,868,805 |

1,234 streams start with a `42 40` block. 7,759 groups have the vertex count of a `42 40` block in the same
stream. 520 streams end inside a group that continues in the next one; 5 stop at a `42 40` word whose header
doesn't fit. The `63 30` checks are unchanged on the 3 extra blocks: all positions finite and all normals of unit
length.

**Slot 2 is the vertex normal.** With tag 0x21 (type 1) it is octahedral: u = bits 0-9 and v = bits 10-19, each
read as unorm `x / 511.5 − 1`, unfolded as usual (z = 1 − |u| − |v|, and where z < 0, x and y become
(1 − |v|)·sign(u) and (1 − |u|)·sign(v)). Bits 20-31 are not decoded; their top two bits are mostly 2 or 3, which
would fit a tangent angle and sign. With tag 0x27 (type 7, 363 blocks) it is x, y, z in bits 0-9, 10-19 and 20-29,
in the same unorm. The normals are in world order whatever the +0x6C axis word says, which also confirms the
position reorder above. `tools/meshlet_page.py --normals` compares each stored normal with the sum of its vertex's
face normals:

| Archive | Vertices | Within 8° | Within 25° | Opposed |
|---|---:|---:|---:|---:|
| `shared_minigame000.rpf` | 14,411,107 | 12,394,835 (86.0%) | 13,271,947 (92.1%) | 210,301 (1.5%) |
| `core_restofworld.rpf` | 451,593 | 299,356 (66.3%) | 331,910 (73.5%) | 51,184 (11.3%) |

On a sample of 79,157 minigame vertices, the readings tried before this one stay well below it, at a mean dot of
0.56-0.8: plain 10:10:10, a smallest-three quaternion, and 15- or 16-bit octahedral. In `core_restofworld`, 70%
of the 347 groups with a type-1 normal have fewer than 5% opposed, and 2% are mostly opposed (more than half),
which fits mirrored parts. The rest may be groups that the span rule binds to the wrong block.

**Meshlet record words +8/+12 hold the meshlet's centre.** Three 12-bit fields give x, y and z: w2 bits 0-11,
w2 bits 12-23 and w3 bits 0-11. Each counts steps around 2047: the centre is `(field − 2047) × step`, with a step of
2^e / 16 m, where e is w2 bits 28-31: 1/16 m (±128 m) for most meshlets, coarser for larger ones further out (below). The default `0x007FF7FF / 0x7FF` is the origin.
`tools/meshlet_page.py --centres` compares each stored
centre with the centre of its meshlet's vertex box, using the `42 40` positions above. Over all 1,569 streams:

| Meshlets | Default words | Axes | Same step | Within one step |
|---:|---:|---:|---:|---:|
| 142,613 | 68 | 427,839 | 214,263 (50.1%) | 307,248 (71.8%) |

The rest are further off. The stored centre may be a bounding-sphere centre rather than the box centre: the mean
of the vertices fits worse (about 30% in the same step on the first 400 streams, against 51.8% for the box).
Within a block of 40 or more meshlets, the fields are almost perfectly ordered by the box centre: median rank
correlation 0.997 (x), 0.997 (y) and 0.983 (z) over 387 blocks. Readings that are deliberately wrong fit far worse on the first 400 streams:

| Reading | Same step |
|---|---:|
| the rule | 51.8% |
| offset 2048 | 11.3% |
| 1/8 m steps | 0.9% |
| x and y swapped | 17.0% |

The earlier test on `63 30` blocks found no correlation, and the reason is that their records hardly vary: the
groups matched to a `63 30` block keep the default words in 2,417 of 2,452 meshlets.

**Word 0 bits 16-31 hold the meshlet's bounding-sphere radius, as an f16**, measured around the stored centre. Bits
14-15 are clear in all 61,170 records of the first 400 streams. Against the radius from the box centre, the
correlation is 0.999 on those streams. `--centres` checks the radius against the farthest vertex from the stored
centre, three ways: it covers that vertex; it covers it once the centre's rounding of up to √3/32 m is allowed for;
and it is tight, at most 1.25 × that distance plus the rounding:

| Archive | Meshlets | Covers | Covers within rounding | Tight |
|---|---:|---:|---:|---:|
| `shared_minigame000.rpf` | 142,613 | 137,777 (96.6%) | 140,881 (98.8%) | 142,518 (99.9%) |
| `core_restofworld.rpf` | 4,899 | 4,771 (97.4%) | 4,895 (99.9%) | 4,883 (99.7%) |

So the record carries a culling sphere (centre plus radius) for each meshlet.

**Word +12 bits 14-30 hold the meshlet's normal cone.** The axis is octahedral, 6 + 6 bits: u = bits 14-19 and
v = bits 20-25, each read as unorm `x / 31.5 − 1`, unfolded like the normals above. Bits 26-30 give the spread:
sin(half-angle) = (31 − f) / 31. All-zero bits 12-31 (76% of records) is no cone. Bits 12-13 and bit 31 are always
clear. On the first 400 streams, the median smallest dot between a face normal and the stored axis follows
√(1 − ((31 − f)/31)²) within 0.033 on average; linear, square-root, squared and sine readings of the field miss by
0.14-0.32. `--cones` counts:

| Archive | Cones | Axis within 25° of the mean normal | Every face inside | … within 0.05 |
|---|---:|---:|---:|---:|
| `shared_minigame000.rpf` | 43,415 | 35,615 (82.0%) | 18,483 (42.6%) | 42,454 (97.8%) |
| `core_restofworld.rpf` | 1,221 | 914 (74.9%) | 463 (37.9%) | 1,217 (99.7%) |

So the cone is rounded to the nearest step, not widened to be conservative. With the centre and radius it completes
a meshlet's culling data.

**w2 bits 28-31 are an exponent for the centre's step.** The step is 2^e / 16 m; bits 24-27 are always clear.
@zzEmmazz's count over the minigame archive and `story000` found e = 0 to 4 (review on #2991; the first version of
this section saw only e = 0 and 1 in the first 400 streams). In the minigame archive, e = 1 is set on 1,732
meshlets and e = 2 on 157; `core_restofworld` has none. These meshlets are large (median radius 33 m), face upward,
and sit 100 m or more from the origin. Within the centre's rounding, the radius around the stored centre covers
the farthest vertex as follows:

| e | Meshlets | Step 2^e/16 | 1/16 m | 1/8 m |
|---:|---:|---:|---:|---:|
| 1 | 1,732 | 1,730 | 0 | 1,730 |
| 2 | 157 | 157 | 0 | 0 |

A least-squares fit of box centre against field on the e = 1 meshlets gives slopes of 1/8.07, 1/8.05 and 1/8.08
per axis. `--centres` counts the meshlets with e > 0 as `coarse_steps`. The centre and radius tables above use each
meshlet's own step.

**Slots 11 and 18 are tangents.** Their type-7 words are unit vectors in the same 10:10:10 unorm, perpendicular to
the vertex's stored normal. Bits 30-31 are always 3 or 0, which fits the bitangent sign +1 or −1. Over the minigame
archive (`--normals`, counts only):
- 7,520,441 tangent words, 7,435,001 (98.9%) of them within |dot| < 0.05 of perpendicular;
- sign bits 6,457,714 × 3 and 1,062,727 × 0, and never 1 or 2.

`core_restofworld.rpf` has no tangent slots.

**The rules hold in a second archive.** `core_restofworld.rpf` with `--faces --centres --normals`:
- all 151 mesh streams parse, with 502 `42 40` blocks, 488,265 positions (all finite, all within 300 m) and
  6,564 meshlets;
- 69.4% of 699,618 shared edges have neighbouring faces within ~25°, and 0.37% are opposed;
- 10,555 of 14,697 centre axes (71.8%) are in the same 1/16 m step, and 87.6% within one.

**And in `story000.rpf`, the main story archive.** All 7,577 mesh streams parse with every check on (`--faces
--centres --normals --cones`, counts only). Six of them are 256 KiB pages where texture-like data follows the mesh
chunks on a 256-byte boundary behind zero padding; the walk stops there (`stopped_at_other_data`). They hold:

| `42 40` blocks | Positions | Meshlets | Triangles |
|---:|---:|---:|---:|
| 173,877 | 29,602,924 | 405,418 | 38,626,760 |

| Check | Count |
|---|---:|
| positions finite | all |
| positions within 300 m | 29,572,450 (99.9%) |
| shared edges with neighbouring faces within ~25° | 33,342,470 of 36,707,506 (90.8%); 1.4% opposed |
| normals within 8° / 25° of their faces | 76.5% / 94.6% of 20,801,089; 2.1% opposed |
| normals within 8° / 25° of their faces | 76.5% / 94.5% of 20,801,089; 2.1% opposed |
| tangents perpendicular to the normal | 30,473,135 of 30,569,246 (99.7%) |
| radius covers the farthest vertex from the stored centre | 257,091 of 275,832 meshlets (93.2%); 94.8% within rounding |
| cones: axis within 25° / every face inside within 0.05 | 89.5% / 94.2% of 228,244 |

The stored centres match the box centre less often here: 25.7% of axes in the same step, 41.5% within one,
against 50.1% / 71.8% in the minigame archive. 625 meshlets use the 1/8 m step. Since the radius still covers the farthest vertex from the stored
centre in 93.2%, the decoding holds. These assets seem to store a centre other than the box midpoint; a sphere
centre would fit. The centre and radius counts in this table were taken before the step exponent was read in
full: they read only w2 bit 28. `story000` also has e = 2 to 4 on about 456 meshlets (@zzEmmazz's count), so those
few rows will move slightly when recounted. Most of the remaining misses come from the group-to-block binding:
several blocks in a page share a vertex count (@Pipodi's count on #2991).

**Which `42 40` block a group is checked against (ASSET-294).** `--faces` and `--centres` decode a group with
a `42 40` block of its vertex count in the same page. When several blocks have that count, the order in the
page doesn't decide it. In `Q0 Q1 G0 G1` pages, G0 is built on Q0 in some pages and on Q1 in others, and taking
the k-th block for the k-th group fits only 9,295 of the 15,997 `story000` groups that have several blocks and a
fitting one. The link is in the drawable's pointers, outside the page. So the walker takes the block whose vertex
boxes are closest to the group's stored centres (`centre_distance`, summed over its meshlets), keeping the nearest
block before the group on a tie. Over all of `story000.rpf` (all 7,577 mesh pages parsed), before and after:

| Pairing | Groups with several blocks | Block changed | Shared edges | Faces within ~25° | Opposed |
|---|---:|---:|---:|---:|---:|
| nearest block before | 16,059 | — | 36,707,506 | 33,342,470 (90.8%) | 518,120 (1.41%) |
| closest centres | 16,059 | 8,668 | 36,719,379 | 34,037,683 (92.7%) | 16,415 (0.045%) |

The face check doesn't use the centres, so it is the independent evidence. The centre counts rise too (same step
212,479 → 222,572 of 827,496 axes), but for those 8,668 groups they no longer measure the centres on their own.
The tables above were counted with the old pairing; the first row is the story000 face count above.

Still open: the +0x5C word, the type-1 normal's
bits 20-31, and what the `42 40` colour and UV slots hold beyond rpf9-explorer's reading.

### `story000_script.rpf`, full

`PSIN` dominates: 2,309 of 2,632 streams, 14,484,253 stored / 59,151,216 inflated. The 16-byte
header is:

| Offset | Field | Measurement |
|---:|---|---|
| +0 | `PSIN` | magic |
| +4 | big-endian first-section offset | points at `LSCH` in 2,173 of 2,309 script streams and 70 of 73 minigame streams |
| +8 | little-endian u32 | 261 on 1,771 script streams, but 77/265/264 and larger values elsewhere; not stable in minigame, so this page does not label it a version |
| +12 | little-endian u32 | 1,568 on 1,227 script streams; not interpreted here |

The section chain follows the measured layout:

```
+0  "PSIN"
... section records: 4-byte tag, big-endian u32 length (includes the 8-byte record header)
```

The chain is contiguous. In one 22,952-byte stream it runs from +0x2EE0 to +0x59A8:

| Section | Length | Next section |
|---|---:|---:|
| `LSCH` | 0x14C8 | +0x43A8 |
| `LENM` | 0x358 | +0x4700 |
| `LSNS` | 0x1234 | +0x5934 |
| `LGID` | 8 | +0x593C |
| `LRSC` | 0x58 | +0x5994 |
| `CHKS` | 0x14 | end |

Chain completeness: 2,118 streams have all six sections; 38 have `LSCH` and `LENM`; 4 have
`LSCH` only; 149 have no parseable chain. The other 323 streams are pages and raw starts.
No `SCRP` or `YSC` container appears, so the script archive's metadata is `PSIN`, not a named
script container.

The 53 MB unread region contains four nested `RPF9` headers (three in a 51.7 MB gap). Their data
is high-entropy and shows no `PSIN`, `RSC8`, `0DRP` or `SCRP`/`YSC` header. That payload is opaque
to this pass; it is not guessed to be bytecode.

### `story000_restofworld.rpf`, full

631 streams, 94.9% readable. Bink 2 is the bulk: 7 streams, 120,187,756 bytes, dimensions
720x1280 (5) and 1920x1080 (2). The rest is 101 `0DRP` pages (version 122), 400 zero-prefix pages,
3 `PSIN`, 13 XML, 3 text and 3 clip streams. Nine `RSC8` headers (version 8001) sit in an 8.96 MB
unread gap; the archive has 18 nested `RPF9` headers. This is the video/extra-content companion to
the story archive.

## `_bvh4` cross-check

| Archive | Streams | Inflated | Content | Overlap with main |
|---|---:|---:|---|---:|
| `story000_bvh4.rpf` | 4 | 16,677 | 4 XML streams (2 ids); no binary BVH payload | 0 of 4 SHA-256 |
| `shared_minigame000_bvh4.rpf` | 2,560 | 970,044,012 | 2,555 binary zero-prefix pages + 5 XML streams (3 ids) | 0 of 2,560 SHA-256 |

The XML streams identify themselves as environment-BVH change sets and carry 61 file-like entries
in `story000_bvh4` and 9 in `shared_minigame000_bvh4`. One filename string is common to both, but
the inflated stream sets have **no common SHA-256**. The binary pages start with zeroes, `0x50xxxxxx` pointers and identity matrices,
consistent with BVH node pages.
These are independent BVH archives, not empty indexes and not shared page sets. `story000_bvh4` is
metadata-only; `shared_minigame000_bvh4` has the payload.

The main-archive comparison scope was the first 1 GiB of `story000.rpf` and all of
`shared_minigame000.rpf`; zero overlaps at those scopes do not prove zero overlap with the
remaining 19 GB.

## Signatures actually observed

| Magic | Offset | Measured fields | Where |
|---|---|---|---|
| `9FPR` (RPF9 little-endian) | +0 | data offset u32 +4; next u32 +8; third u32 +12 | 442 headers in `story000`, 50 in minigame, 5 in script, 19 in restofworld, including the outer archive |
| zlib `78 01/5e/9c/da` | +0 | inflates cleanly to stream end | 93.4% of the first `story000` GiB; 54.7% of minigame |
| `PSIN` | +0 | section chain with big-endian lengths | 2,309 in script; 73 in minigame; 3 in restofworld |
| `0DRP` | +16 | version 122 at +0x14; page count u64 at +0x20 | 21,980 first `story000` GiB; 787 minigame; 101 restofworld |
| `RSC8` | +0 | version 8001 at +4; u32 at +8; flags u32 at +12; name at +0x30 | 225,281 in `story000`; 1,957 minigame; 9 restofworld |
| `RSC7` | +0 | - | 0 in every full-file scan |
| `KB2[a-z]` | +0 | size u32 +4; width +0x14; height +0x18 | 264 in `story000`; 7 in restofworld |
| `42 40 00 00` | +0 | 0x4042 marker; u32s and float records | 1,255 minigame; 780 in the 14 GiB `story000` window |
| `RBF0` | +0 | schema string | 0 at stream starts; 2 raw occurrences in 20 GB (not container starts) |
| `SCRP`, `YSC` | +0 | - | 0 validated |
| `AWC\0`, `FSSM` | +0 | - | 0 validated |
| XML | +0 | UTF-8 BOM + `<?xml` | 16 minigame; 13 restofworld; 4 + 5 in the bvh4 archives |

The `RSC8` version is the same in every validated header: `0x1F41` (8001). The `0DRP` version is
the same in every inflated page: `0x7A` (122). No format is inferred beyond these bytes.

## Readable versus opaque

Readable: zlib streams, Bink 2 streams, XML/text streams, `PSIN` section headers, `0DRP` page
headers and `RSC8` resources. Opaque: encrypted RPF9 tables of contents and the nested
high-entropy payloads in the script archive.

The raw region of `story000` that has no zlib stream holds unencrypted `ADAT` (AWC) audio banks:
63,861 of them, 9,423,830,212 bytes. Every `RSC8` resource in this family is an AWC tag `0x5c`
facial/lip-sync clip inside one of those banks, next to its Opus dialogue line
([rsc8_audio.md](rsc8_audio.md#resolution-of-the-docsstorymd-raw-gaps), decoded by
`tools/rsc8_audio.py`).

"Unread" in the index tables means bytes the zlib walk could not inflate. It is not a claim that
those bytes are empty or unimportant; in minigame and restofworld they contain `ADAT` banks with
their `RSC8` clips, and nested RPF9 archives.

## Likely format family

The bytes support a **new RAGE resource family in an RPF9 container**:

- RPF9 data areas are zlib streams, with page-aligned embedded RPF9 sub-archives (437 of the 442
  `story000` headers are page-aligned).
- `0DRP` is a page header, not a GTA V `RSC7` header: magic at +16, version 122, page count at
  +0x20, and payload pages after it.
- `RSC8` is the resource header in the raw story region: version 8001, a u32 at +8, flags and a
  name string. It is the payload of AWC tag `0x5c`, a facial/lip-sync clip dictionary, and the u32
  at +8 is its body size ([rsc8_audio.md](rsc8_audio.md)). No `RSC7` header was found in 20 GB.
- `PSIN` is a sectioned metadata container, not a script opcode stream. Repository notes already
  describe it as PSO metadata; this pass confirms the section names and the big-endian length
  chain, not the section semantics.
- Bink 2 video is present and separate from the resource pages.

No `RSC7`, `SCRP` or `YSC` container was observed. Models and textures are not claimed from a
magic here: prior model recovery already reports 3,303 models in `story000` and 783 in
`shared_minigame000` (`docs/design.md`), which is consistent with the `0DRP` page population, but
this page does not decode drawables or textures. No DDS, DXT1/3/5, ATI2, BC5, BC7 or TXTR magic
was found in the inflated sample.

## What a decoder needs

1. The RPF9 TOC key/schema. This project does not search for or distribute keys; without it,
   names, paths and exact entry boundaries remain unavailable.
2. A recursive RPF9 walker. The flat zlib carve does not group streams by the 441 additional RPF9
   headers in `story000`; a decoder needs their spans and TOCs to group resources.
3. An `RSC8` decoder for version 8001: done. `tools/rsc8_audio.py` decodes the header and the
   facial/lip-sync clip payload ([rsc8_audio.md](rsc8_audio.md)).
4. A `PSIN` section walker. The tag/length chain is measured; `LSCH`, `LENM`, `LSNS`, `LGID`,
   `LRSC` and `CHKS` semantics are still open.
5. `0DRP` page grouping. The page count at +0x20 is the only measured grouping field; a decoder
   still needs the following stream boundaries.
6. Bink 2 decoding for the 264 story streams and 7 restofworld streams (see `docs/movies.md`).
7. The existing model exporter for drawables; texture identification needs a real container
   signature, which this pass did not find.

## No-sharing rule

Do not commit archives, carved streams, inflated payloads, asset names, path strings, hashes of
assets or name dumps. This page and the test contain aggregate counts and synthetic bytes only.
Keep carves in a local work directory, never in the repository or a PR. The tool writes offsets,
sizes, signatures and header fields; it does not write resource payloads or carved bytes.

## Reproduce

```powershell
python -I tools/story_index.py <archive.rpf> <out.csv|out.json> [--start N] [--limit BYTES]
python -I tools/rpf9_carve.py <archive.rpf> <out_dir> --index-only [--start N] [--limit BYTES]
python -I tests/test_story_index.py
python -I tools/meshlet_page.py <archive.rpf | carved-dir | page.bin> [...] [--json out.json] [--quiet]
python -I tests/test_meshlet_page.py
```

`meshlet_page.py` writes counts and layout checks only (chunk kinds, vertex, meshlet and triangle counts, and
how many values pass each check), never vertex or index values. It exits 1 if a page fails to parse and 2 if an
input does not exist.

`--start` must be 0 or more and `--limit` at least 1 (decimal or `0x` hex). A value outside that, or a `--start`
past the end of the archive, stops `story_index.py` with a one-line error and exit code 2 and writes nothing.
An empty archive gives an index with no files.

`story_index.py` is stdlib-only. It classifies the same zlib/Bink walk as `rpf9_carve.py`, parses
the `PSIN` section chain, and reports `0DRP`/`RSC8`/`9FPR` headers when they start a stream or a
carved file. It does not recurse into the raw embedded archives. A `clip-stream` row is the tagged
auxiliary table of a `0DRP` second system page, checked by the same `aux_table` as `anim_index.py`
(#2496): u32 count 2 or 3, count {u32 tag, u32 payloadBytes} pairs with first tag `0x101`, and every
payload inside the stream. Its `count` is the entry count and its detail lists the tags and payload end;
there is no version and no fixed 24-byte header. The "clip stream" row in the 1 GiB table above was
counted with the older eight-byte prefix check (u32 2 or 3, then `01 01 00 00`).
