# RPF9 `0DRP` page directories

**Status: measured layout and a stdlib indexer.** This note decodes the RPF9 directory page that carries `0DRP`
at +0x10, version 122. It is present and measured in `data.rpf`, `levels.rpf`, `navmeshes.rpf` and `anim.rpf`;
the sampled `*_bvh4.rpf` windows contain no such page. It does not decrypt the table of contents, search for keys
or name a game asset. The numbers come from a read-only scan of one Xbox Series 1.0.2.1 package. No archive,
payload, carved stream or name list is in this document or the repository.

## Bottom line

| Question | Measured answer |
|---|---|
| Where is the header? | `0DRP` at +0x10, version 122 at +0x14, fixed header length **0x60**. |
| Is there a 64-bit key at +0x60? | **No.** The first resource payload starts at +0x60 on 29,297 of 29,298 measured pages, and on 48,193 of the 48,194 pages of the three archives read whole. The bytes called a key are payload. |
| What does +0x38 point to? | A table of **16-byte `{u32 hash, u32 size, u64 self-pointer}` records**, one per resource segment. The count is `u16@0x4c`. |
| What does a record map? | A virtual offset and exact byte size inside the logical resource. The offset is `pointer - 0x50000000`, not an archive offset. |
| Do the records cover the page? | Yes on 15,767 of 15,768 `data.rpf` pages and on every navmesh and animation page (all 9,603 and all 22,823, see *Whole-archive pass*): sorted by target they tile from +0x60 with **zero overlaps**. The 14.8 MB, 89-count page starts with the +0x40 table at +0x60; its `q38` records still tile with zero overlaps. |
| Can a resource span zlib streams? | **Yes.** When a table or target crosses the inflated stream end, the next zlib stream in archive order continues the same resource. All 29,298 core pages resolved with at most **one** continuation stream; the 107 sampled `levels.rpf` pages did the same, and so did all 48,194 pages of `data.rpf`, `navmeshes.rpf` and `anim.rpf` read whole. |
| Are resource boundaries available without the TOC? | **Yes for offsets, sizes and stream membership; no for names or asset identity.** The TOC is still required for paths and the hash-to-name mapping. |

## Method and coverage

The walk is the fixed `tools/rpf9_carve.py` zlib walk from PR #90: find a zlib header, inflate to `eof`, advance by
the bytes actually consumed, and never read the encrypted TOC. A page is counted when the inflated stream has
`30 44 52 50` at +0x10.

| Archive | Scope | Streams | `0DRP` pages | Version |
|---|---:|---:|---:|---:|
| `data.rpf` | full, 481,544,304 bytes | 28,509 | 15,768 | 122 |
| `levels.rpf` | first 64 MiB | 142 | 107 | 122 |
| `navmeshes.rpf` | first 64 MiB | 1,378 | 1,132 | 122 |
| `anim.rpf` | first 256 MiB | 24,232 | 12,398 | 122 |
| `navmeshes.rpf` | 64 MiB at +1 GiB | 956 | 0 | - |
| `core_bvh4.rpf` | first 64 MiB | 5,816 | 0 | - |
| `core_bvh4.rpf` | 64 MiB at +4 GiB | 14,823 | 0 | - |
| `core_bvh4.rpf` | 32 MiB near EOF | 654 | 0 | - |

The +1 GiB navmesh slice is a control: 527 of its 956 streams are typed `0x00010015` data pages, not directory
pages. It confirms that the directory pages cluster in the first part of that archive, as `docs/navmesh.md` (PR
#97) already reported. The `levels.rpf` row is the cross-archive check: the same header, version, table counts and
continuations are present there.

The BVH samples are a negative control. `core_bvh4.rpf` is RPF9, but none of the three sampled windows contains a
`0DRP` stream. This work therefore does **not** establish that the `*_bvh4.rpf` packs use the page-directory
format; their streams are BVH payloads and descriptors (see `docs/bvh.md`, PR #109), not directory pages. The
"generic" claim in this note covers `data.rpf`, `levels.rpf`, `navmeshes.rpf` and `anim.rpf`.

## Whole-archive pass

The `navmeshes.rpf` and `anim.rpf` rows above stop at 64 MiB and 256 MiB. `tools/page_index.py <archive> pages.json`
without `--limit`, on the same 1.0.2.1 package, reads both archives to the end (counts only; every row is taken from
the report's `summary` or counted over its `pages`):

| | `data.rpf` | `navmeshes.rpf` | `anim.rpf` |
|---|---:|---:|---:|
| zlib streams | 28,509 | 90,295 | 130,470 |
| `0DRP` pages, all version 122, all complete | 15,768 | 9,603 | 22,823 |
| `q38` records | 60,880 | 54,898 | 527,439 |
| a `q38` record targets +0x60 | 15,767 | 9,603 | 22,823 |
| pages with overlapping `q38` records | 0 | 0 | 0 |
| gap runs / gap bytes between sorted records | 27,377 / 8,411,848 | 11,461 / 5,058,136 | 136,013 / 10,259,744 |
| pages with one continuation stream (none has more) | 158 | 2,170 | 11,760 |
| `q38` table wholly in the continuation (none is split) | 2 | 270 | 10,218 |
| span (target + size): minimum / median / maximum | 296 / 21,760 / 14,848,448 | 360 / 186,208 / 2,985,724 | 2,732 / 189,072 / 5,634,144 |
| `+0x20` | 1-3, 89 | 1-6 | 1-5, 13-41 |
| `n28` / `n30` / `n38` / `u16@0x4e` | 2-182 / 0-38 / 2-186 / 0-1,077 | 2-12 / 0-4 / 2-6 / 0-602 | 25-103 / 11-57 / 7-79 / 0-1,382 |
| `q28` records declared (`sum n28`) | 60,901 | 108,890 | 1,015,222 |
| `q28` records read | 60,897 | 106,202 | 952,850 |
| `q28` records with a null pointer | 0 | 0 | 3,672 (on 3,511 pages) |
| pages whose `q28` table ends past the streams read | 1 | 225 | 1,355 |

The stream and page totals agree with the other whole-archive readers: 90,295 streams and 9,603 directory pages in
`docs/navmesh.md`, 130,470 streams in `docs/animation.md`. What the samples said about the layout holds on all
48,194 pages: version 122, the first payload at +0x60 (one `data.rpf` exception), `q38` tables 16-byte aligned,
`q28` tables 8-byte aligned, no overlap between `q38` records, at most one continuation stream, and a `q28`
`hash_a` that matches a `q38` hash on every page where both tables were read (15,767, 9,378 and 21,468 pages).

What the samples did not show:

- **Header ranges are wider.** `+0x20` reaches 6 in `navmeshes.rpf` (1: 1 page, 2: 779, 3: 7,650, 4: 932, 5: 22,
  6: 219) and 41 in `anim.rpf` (1: 1,883, 2: 7,571, 3: 9,241, 4: 2,022, 5: 70, 13-41: 2,036); `n30` reaches 57 and
  `u16@0x4e` 1,382, both in `anim.rpf`.
- **A `q28` pointer can be null.** 3,672 `anim.rpf` records on 3,511 pages have a zero pointer with `hash_a`,
  `count_or_size` and the flags set; 116,277 ordinary records follow them in the same tables. The tool used to stop
  a table at the first one. It now keeps the record with `target` null and reads on; a pointer with any other
  non-`0x50000000` value still ends the table with `q28_error`. `data.rpf` and `navmeshes.rpf` have none.
- **A `q28` table can lie past the resource's `q38` boundary.** The tool reads only the continuation streams the
  `q38` table and its records need. On 1,355 `anim.rpf`, 225 `navmeshes.rpf` and 1 `data.rpf` pages the `q28`
  table ends beyond those streams. Such a page reports no `q28` records and names the reason in `q28_error`
  (`q28_tables_unread` in the summary) instead of looking like a page without the table. Whether the next stream
  holds the table was not followed up.
- **The navmesh type hashes are not on every page.** `0xde5f1345` is in 8,412 of the 9,603 pages (every page of the
  first 64 MiB has it); the most common hash, `0x0bcb00e4`, is in 9,585 records.

The per-archive tables further down (common sizes, common hashes, payload classes) are still the 64 MiB and 256 MiB
slices where they say navmesh or animation; the `data.rpf` columns were complete from the start.

### `levels.rpf` in full

`levels.rpf` was counted whole with a temp-only streaming runner that calls the same `scan_rpf`, `parse_page` and
`resource_of` functions and accumulates the summary one page at a time (the stock tool keeps every page; this
archive reached 17.4 GB RSS and had to be stopped). The runner reproduces the stock `data.rpf` row exactly
(28,509 / 15,768 / 60,880 / 158), so the counts are the same fields:

| | `levels.rpf` |
|---|---:|
| zlib streams | 1,116,073 |
| `0DRP` pages, all version 122, all complete | 188,449 |
| `q38` records | 465,611 |
| pages with one continuation stream (none has more) | 3,281 |
| `+0x20` | 1-44 |
| first `q38` payload at +0x60 | **52,776 of 188,449** |

The page total agrees with the existing levels census (188,449). The +0x60 rule is the one place where this
archive differs from the other three: only 52,776 of its pages put the first `q38` payload at +0x60. The rest are
counted and parsed, but this check does not establish the +0x60 key claim for them; their first target or table
sits elsewhere. This was a count run (17.7 min), so it does not fill the payload-class, span or q28-declared
tables below; those keep their labelled sample scopes.

## Fixed header (0x60 bytes)

The core sample is 15,768 `data.rpf` pages plus 1,132 navmesh and 12,398 animation pages = **29,298 pages**.
The `levels.rpf` and BVH windows are cross-checks and are not included in that total.

Offsets are from the start of the inflated stream. The "Observed" column is the measured range over the core
sample; the meaning column separates decoded fields from unresolved ones.

| Offset | Size | Observed | Meaning |
|---:|---:|---|---|
| +0x00 | 8 | zero in all pages | page prefix |
| +0x08 | 8 | `0x0000000050xxxxxx`, inside the resource or exactly at its end | self-pointer/end pointer; role unresolved (4 animation pages point at the end) |
| +0x10 | 4 | `30 44 52 50` | `0DRP` |
| +0x14 | 4 | 122 in all 29,298 | version |
| +0x18 | 8 | zero in all pages | unused |
| +0x20 | 4 | 1-89 in `data.rpf`, 2-3 in navmesh, 1-38 in anim (whole archives: 1-6, 1-41; 1-44 in `levels.rpf`) | presumed page count; **unit unresolved** |
| +0x24 | 4 | zero in all pages | high half of the +0x20 qword |
| +0x28 | 8 | `0x5000xxxx`, 8-byte aligned | pointer to the `q28` table |
| +0x30 | 8 | zero or `0x5000xxxx` | pointer to the `q30` table |
| +0x38 | 8 | `0x5000xxxx`, 16-byte aligned in all pages | pointer to the `q38` resource table |
| +0x40 | 8 | `0x5000xxxx` | pointer to the `q40` table; layout unresolved |
| +0x48 | 2 | 0-182 | `n28`: number of +0x28 records |
| +0x4a | 2 | 0-38 (0-57 over whole `anim.rpf`) | `n30`: number of +0x30 records |
| +0x4c | 2 | 2-186 | `n38`: number of +0x38 records |
| +0x4e | 2 | 0-1,382, varying | **unresolved**; not a dense count |
| +0x50 | 16 | zero in all pages | unused |
| +0x60 | ... | first payload, or the `q40` table on one page | not header |

The header read for the worked example below (`data.rpf`, archive offset `0x11f4010`, inflated 131,072 bytes):

```text
00 00 00 00 00 00 00 00 10 87 01 50 00 00 00 00
30 44 52 50 7a 00 00 00 00 00 00 00 00 00 00 00
02 00 00 00 00 00 00 00 40 87 01 50 00 00 00 00
00 00 00 00 00 00 00 00 d0 86 01 50 00 00 00 00
60 88 01 50 00 00 00 00 04 00 00 00 04 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
```

That is: `0DRP`, version 122, page count 2, `q28=0x18740`, `q30=0`, `q38=0x186d0`, `q40=0x18860`, counts
`4,0,4,0`, then 16 zero bytes. The first payload starts at +0x60.

### The +0x60 "key" is payload

`docs/navmesh.md` (PR #97) reports a 64-bit key at +0x60 and a `u32` at +0x68. The bytes are real but they are not
header fields on these pages. The `q38` table gives the counter-example:

- On the navmesh page at archive offset `0x11ca630`, the first `q38` entry is
  `{hash 0xd7e9c53b, size 0x140, target 0x60}`. It covers +0x60 through +0x19f. The reported key `49` is the
  first byte of that payload, the ASCII character `1`; the reported `u68=0` is the next four payload bytes.
- On `data.rpf` page `0x11f4010`, the first `q38` entry targets +0x60 with size `0x11000`.
- The same holds for 15,767 of 15,768 `data.rpf` pages, all 107 sampled `levels.rpf` pages, all 1,132 sampled
  navmesh pages and all 12,398 sampled animation pages.

The one exception is the 14.8 MB, 89-count directory page at `data.rpf` offset `0x1c396010`: its `q40` table starts
at +0x60, and the first `q38` payload is elsewhere. Even there, +0x60 is a table, not a key.

So the navmesh report's 9,603-page key census (9,409 nonzero) is a census of payload prefixes, not a header field. They cannot be
used as a name or asset id. Names still require the TOC.

## Pointer arithmetic

Every measured `q38` pointer, and every `q28` pointer that is not null (3,672 `anim.rpf` records have a zero
pointer, see *Whole-archive pass*), is a zero-extended 32-bit value in a 64-bit field:

```text
d0 86 01 50 00 00 00 00  ->  0x000000005000186d0  ->  0x186d0
```

The rule is:

```text
target = pointer - 0x50000000      if the high 32 bits are zero and 0x50000000 <= pointer < 0x60000000
```

No other base was observed in a directory table. The subtraction is **not** an archive offset. It is a virtual
offset into the logical resource:

- `target < inflated_size`: the byte is in this inflated stream.
- `target >= inflated_size`: the byte is in the next stream in archive order, at
  `target - inflated_size` into that stream's inflated bytes.
- A `q38` table can lie wholly in a continuation stream (2 `data.rpf` pages, 32 navmesh pages, 6,978 animation
  pages). No partially split table was observed. The tool concatenates the continuation before reading it.

`q38` tables are 16-byte aligned in all 29,298 pages; `q28` tables are 8-byte aligned. `q38` targets are 16- or
8-byte aligned; `q38` sizes are arbitrary.

## The `q38` resource table (solid)

`n = u16@0x4c`. The table is at `q38 = u64@0x38 - 0x50000000`, and each record is 16 bytes:

| Offset | Size | Meaning |
|---:|---:|---|
| +0x00 | 4 | 32-bit type hash |
| +0x04 | 4 | exact payload byte size |
| +0x08 | 8 | self-pointer: `0x50000000 + virtual target` |

Worked table from `data.rpf` page `0x11f4010`, table at `0x186d0`:

```text
89 88 47 a8 70 76 00 00 60 10 01 50 00 00 00 00
89 10 33 99 10 00 00 00 30 87 01 50 00 00 00 00
01 01 00 00 00 10 01 00 60 00 00 50 00 00 00 00
10 00 00 00 8b fa 00 00 00 00 02 50 00 00 00 00
```

| Hash | Size | Pointer | Target |
|---|---:|---|---:|
| `0xa8478889` | `0x7670` | `0x50011060` | `0x11060` |
| `0x99331089` | `0x10` | `0x50018730` | `0x18730` |
| `0x00000101` | `0x11000` | `0x50000060` | `0x60` |
| `0x00000010` | `0xfa8b` | `0x50020000` | `0x20000` |

The records are stored in hash-table order, not target order. Sorted by target, the segments are disjoint. On this
page:

| Virtual range | Contents |
|---|---|
| `0x000060`-`0x01105f` | `q38` hash `0x00000101`, size `0x11000` |
| `0x011060`-`0x0186cf` | `q38` hash `0xa8478889`, size `0x7670` |
| `0x0186d0`-`0x01870f` | `q38` table, 4 x 16 bytes |
| `0x018710`-`0x01872f` | 32-byte gap, not decoded |
| `0x018730`-`0x01873f` | `q38` hash `0x99331089`, size `0x10` |
| `0x018740`-`0x0187bf` | `q28` table, 4 x 32 bytes |
| `0x0187c0`-`0x01885f` | `q28` target records |
| `0x018860`-`...` | `q40` table |
| `0x020000`-`0x02fa8a` | `q38` hash `0x00000010`, size `0xfa8b` |

The last segment starts exactly at the inflated stream end (`0x20000`) and is continued by the next zlib stream,
at archive offset `0x11f9620`, inflated size **64,144** (`0xfa90`). The declared `q38` size is **64,139**
(`0xfa8b`); the continuation has five trailing bytes outside the declared payload. This is the direct
page-to-payload correlation: target `0x20000`, size `0xfa8b`, and the next stream's bytes line up exactly. The
continuation starts with printable ASCII, so that segment is text, not a model, texture or nested page.

### Coverage and tiling

| Archive | Pages | `q38` records | First target +0x60 | Overlap pages | Gap runs | Gap bytes |
|---|---:|---:|---:|---:|---:|---:|
| `data.rpf` | 15,768 | 60,880 | 15,767 | 0 | 27,377 | 8,411,848 |
| `navmeshes.rpf` (64 MiB) | 1,132 | 6,645 | 1,132 | 0 | 1,318 | 270,520 |
| `anim.rpf` (256 MiB) | 12,398 | 269,320 | 12,398 | 0 | 64,356 | 5,088,441 |

"Gap runs" count individual gaps between sorted records, not pages. The gaps are the metadata tables and padding.
Classifying the gap bytes: most fall inside the `q38` table or the `q28`/`q30`/`q40` ranges; the rest are
`0x00`/`0x80` fill and short 16-40 byte alignment runs. No record overlaps another record in any sampled page.
On 2 `data.rpf`, 32 navmesh and 6,978 animation pages the `q38` table itself is in the continuation stream; each
was parsed after concatenation.

Largest/smallest declared span (target + size):

| Archive | Minimum | Median | Maximum |
|---|---:|---:|---:|
| `data.rpf` | 296 | 21,768 | 14,848,448 |
| `navmeshes.rpf` | 5,186 | 144,960 | 2,422,688 |
| `anim.rpf` | 2,732 | 206,064 | 2,319,504 |

### Most common `q38` sizes

| `data.rpf` | Count | `navmeshes.rpf` | Count | `anim.rpf` | Count |
|---:|---:|---:|---:|---:|---:|
| `0x10` | 15,320 | `0x130` | 1,148 | `0x10` | 11,750 |
| `0x2200` | 5,616 | `0x1000` | 1,136 | `0x2e00` | 9,522 |
| `0x220` | 4,658 | `0x20` | 125 | `0x120` | 5,436 |
| `0x4400` | 2,172 | `0x40` | 111 | `0x140` | 4,444 |
| `0x28` | 1,270 | `0x80` | 82 | `0x240` | 4,251 |
| `0x440` | 858 | `0x60` | 76 | `0x40` | 3,567 |
| `0x880` | 768 | `0xc0` | 66 | `0x80` | 3,459 |
| `0x20` | 552 | `0x100` | 65 | `0xc0` | 3,458 |

### Most common `q38` hashes

These are 32-bit hashes; whether they are type ids is open. Repeated values are stable across pages.

| `data.rpf` hash | Count | `navmeshes.rpf` hash | Count | `anim.rpf` hash | Count |
|---:|---:|---:|---:|---:|---:|
| `0x00000010` | 14,520 | `0xde5f1345` | 1,132 | `0x00000007` | 12,398 |
| `0xa8478889` | 14,482 | `0x2283b2cb` | 1,132 | `0xa90b7f7e` | 12,398 |
| `0x99331089` | 14,482 | `0x00000013` | 1,132 | `0x0000004a` | 12,398 |
| `0x00000101` | 14,482 | `0xbb5a70d8` | 1,132 | `0x64e8ac67` | 12,398 |
| `0xc90d1c34` | 1,247 | `0x0bcb00e4` | 1,132 | `0xad5dfe3c` | 12,366 |
| `0xfc3dbc38` | 1,247 | `0x00000100` | 985 | `0x64d97283` | 11,686 |
| `0xd7e9c53b` | 38 | | | `0x00000100` | 11,686 |
| `0x8731ec16` | 38 | | | `0x29f72aea` | 11,670 |

No name or type string is present in the page; these hashes remain opaque until the TOC or the executable supplies a mapping.

### One page lacks the recurring `0x00000007` tag

The Xbox Series 1.0.2.1 export identifies `story000.rpf` as its source and contains 16,977 complete
directory pages. A bounded source check used 17 evenly spaced rows (including the first and last) and
the six-record exception described below. At the recorded compressed spans, the inflated length, page
signature and both record lists (`q38` and `q28`) matched for 18/18 rows in `story000.rpf` and 0/18 in
`data.rpf`. The counts below apply only to this export. Each selected tag occurred once per page that contained it.

| Selected `q38` tag | Pages containing it |
|---|---:|
| `0xa90b7f7e`, `0x0000004a`, `0x64e8ac67` | 16,977 each |
| `0x00000007` | 16,976 |
| `0x00000100` | 16,519 |
| `0x00000101` | 593 |
| `0x00000010` | 19 |
| `0xde5f1345` | 0 |

The one complete page without tag 7 had six `q38` records. The six nearest pages by record count had 10–12
records and carried tag 7 in positions 2–7 (counting from one). Four shared five of the exceptional page's tags
and two shared all six. Across the export, all 81 other pages containing all six tags also contained tag 7.
None preserved the relative order of all six. Tag 7 is therefore optional in this sample. There is no second
six-record table for a comparison that could distinguish omission from substitution.

Of the 15,775 pages with exported `q28` records, all had at least one `hash_a` equal to a same-page `q38`
tag. None had a matching `hash_b`. The export also reported 131 pages with `q28_error`. On the exceptional
page, five of the 26 readable `q28` records each matched a different local tag through `hash_a`. Neither
field equalled 7. On the six nearest pages, `hash_a` also never equalled 7 even though `q38` contained it.
These comparisons establish equality of field values. Their role at runtime and the meaning of the tags remain open.

## The `q28` table (layout measured, semantics partial)

`n = u16@0x48`; table at `q28 = u64@0x28 - 0x50000000`; each record is 32 bytes:

| Offset | Size | Confidence | Meaning |
|---:|---:|---|---|
| +0x00 | 4 | measured | `hash_a`; at least one record on every page matches a `q38` hash |
| +0x04 | 4 | unresolved | `hash_b`; no match to a `q38` hash observed |
| +0x08 | 8 | unresolved | `field_08`; observed `0`, `0x100`, `0x200`, `0x300`, `0x30c`, `0x400`, `0x424`, ...; not a measured byte range |
| +0x10 | 8 | measured | self-pointer, or zero on 3,672 `anim.rpf` records; 60,862 of 60,897 `data.rpf` targets resolve in the first stream |
| +0x18 | 4 | unresolved | `count_or_size`; small values, not consistently the linked `q38` size |
| +0x1c | 4 | measured | flags; only high 16 bits are set, observed `0x10000`-`0x6b0000` |

Worked first `q28` record from the same page (`data.rpf` `0x11f4010`, table `0x18740`):

```text
89 10 33 99 a4 68 55 bf 00 03 00 00 00 00 00 00
00 88 01 50 00 00 00 00 10 00 00 00 00 00 03 00
```

That is `hash_a=0x99331089` (the `q38` entry of size 0x10), `hash_b=0xbf5568a4`, `field_08=0x300`,
`target=0x18800`, `count_or_size=0x10`, `flags=0x00030000`.

The table points at hash-table-like records of 16 bytes, `{u32 hash, u32 0, u32 index, u32 0}`, plus small
dictionaries. The exact meaning of `field_08`, `count_or_size` and `hash_b` is not decoded, so the tool exports
them under those names and does not use them for boundaries.

Counts: 60,897 records in `data.rpf`, 13,090 in the navmesh slice, 511,435 in the animation slice (106,202 and
952,850 over the whole archives). `hash_a`
matched a `q38` hash on 15,767/15,767 `data.rpf` pages, 1,103/1,103 navmesh pages and 11,558/11,558 animation
pages where both tables were present.

## `q30`, `q40` and `+0x4e`

- `q30` is present when `u16@0x4a` is nonzero. Its target begins with a 16-byte hash-like header; in 980 of 1,132
  navmesh pages the first eight bytes are `01 2e 3b 5c 3e b2 ee 6e`. The internal layout is not decoded.
- `q40` points at a table whose common prefix is `01 00 00 00 01 01 00 00 20 00 00 00 ...` (1, 0x101, 32, then
  hash words). The 89-count page starts `05 00 00 00 01 01 00 00 40 46 4e 00 0f 01 ...`. It is not used for
  resource boundaries and its fields are unresolved.
- `u16@0x4e` ranges 0-1,382 (0-1,077 in `data.rpf`) and does not match any table count. It is left raw in the tool JSON as `counts[3]`.

## What a page maps

A `0DRP` stream starts a **logical resource**. The fixed header is followed by payload segments described by
`q38`; the first segment normally starts at +0x60. The page does not map to a zlib block (the zlib framing is
outside it), to another page, or to a named RBF0/PSIN/model/texture resource:

- Every `q38` target's first 0x18 bytes were classed. In `data.rpf`: 46,184 "other" and 14,696 zero-prefix; in
  navmesh: 6,194 "other" and 451 zero-prefix; in anim: 80,697 "other", 188,617 zero-prefix and 6 too short.
  None started with `PSIN`, `RBF0`, `0DRP`, `RSC7` or `RSC8`.
- Continuation streams were classed as zero-prefix or other payload in every case; none was a `0DRP` page. A
  resource therefore extends forward by payload, not by nesting another directory.
- `PSIN` (2,281) and `RBF0` (1,263) are separate top-level stream classes in `data.rpf`, not `q38` payloads.

### Cross-check: `0x00010015` data pages

A typed data page has a different prefix: eight zero bytes, a `0x5000xxxx` pointer at +0x08, a small `u32` at
+0x10 (9 in 355, 11 in 121, 3 in 46, 1 in 5 of the sampled pages), and `15 00 01 00` at +0x14. The same
subtraction applies: 99 of 527 sampled typed pages point inside their own inflated stream; 428 point beyond it,
consistent with a continuation stream. The typed page has **no `q38` table**, so it supplies no resource
boundaries by itself.

This is the cross-check the navmesh report asked for: the `0x50000000` base is shared by directory pages and typed
data pages, but only the directory page carries the offset/size table. The typed prefix is not a `q38` payload
magic; 0 of 60,880 `data.rpf` records and 0 of the 6,645 navmesh and 269,320 animation records in their sampled
windows started with `0x00010015` at +0x14.

## Distribution of the +0x20 count

| Value | `data.rpf` | `navmeshes.rpf` (64 MiB) | `anim.rpf` (256 MiB) |
|---:|---:|---:|---:|
| 1 | 1,247 | 0 | 480 |
| 2 | 14,485 | 147 | 5,240 |
| 3 | 35 | 985 | 5,144 |
| 4 | 0 | 0 | 1,407 |
| 5 | 0 | 0 | 45 |
| 18-38 | 0 | 0 | 82 |
| 89 | 1 | 0 | 0 |

The count does not equal the `q38` record count or the stream count. A `data.rpf` page with count 2 can have four
records; a navmesh page with count 3 has six; an animation page with count 3 can have 22. The unit is **unresolved**.
The earlier label "presumed page count" is kept only as a name for `u32@0x20`.

### The `+0x20` count does not mark stream boundaries

`story000.rpf` was measured to decide whether the field can group a resource's streams. In a 256 MiB carve
(33,315 streams, 16,977 `0DRP` directories) a directory is followed by **exactly one** non-directory stream in
16,128 cases, and that is the same for counts 2, 3 and 4 (9,616 / 5,663 / 633), so the field is not a stream count.
Compared against the 64 KiB units the resource group actually occupies (directory plus the continuations needed to
reach the `q38` table and its last record), the declared count matches on **9,010 of 16,977 pages (53.1%)**. Growing
the group until the `q28`/`q30`/`q38`/`q40` tables also fit raises that only to 9,622 (56.7%). Against
the directory stream alone the best unit is 32 KiB, at 7,660 pages (45.1%). Against counts derived from the `q28`
table (`n28`, `sum(flags >> 16)`, `max(flags >> 16)`, `sum(field_08)`) the match is **0%** on all 16,977 pages, and
the `q38` records tile the resource with zero overlaps and 138,583 gaps in the 1 GiB slice, so they are entries, not
pages either.

The boundary is therefore taken from the tables, not from `+0x20`: a resource is the directory stream plus the
following streams needed to reach the `q38` table and its last record. `tools/page_index.py` reports that group per
page as `resource` (`streams`, `bytes`, `page_units`, `declared_pages`, `declared_matches_units`), and the summary
counts how often the declared count agrees, so the disagreement is recorded rather than used.

## Readable versus opaque

**Readable from bytes:** the `0DRP` magic, version 122, header length and pointers, the four `u16` counts, the
`q38` hash/size/target table, the raw `q28` fields, the first payload offset, continuation stream boundaries, and
the payload's coarse class (zero-prefix or other).

**Opaque:** the hash-to-name mapping (TOC or executable), the meaning of each type hash, `q28` `hash_b`,
`field_08` and `count_or_size`, the `q30` and `q40` layouts, `u16@0x4e`, and the unit of `u32@0x20`. No name,
path or resource type is present in a `0DRP` page.

## Remaining unknowns

1. **Names and types.** Join a `q38` hash to the TOC or the executable's resource registry. Do not guess from a
   four-byte value.
2. **`+0x20` unit.** Tested and rejected as a boundary on `story000.rpf`: no page-size reading fits (best 56.7% at
   64 KiB, 45.1% at 32 KiB over the directory stream, 0% against the `q28`-derived counts). The measured `data.rpf`
   page `0x11f4010` has span 195,211 bytes and count 2 (1.49 x 128 KiB), while a navmesh page with span below 8 KiB
   has count 3. A loader or a decoded dictionary is needed before the field itself is named.
3. **`q28` fields.** `hash_b`, `field_08` and `count_or_size` need a loader or a decoded dictionary. The hash
   tables it points at look like `{hash, 0, index, 0}` records, but the index semantics are not proven.
4. **`q30`/`q40`.** Both contain hash-like tables and are not needed for boundaries. Decode them only with a
   consumer.
5. **Type-hash semantics.** The same hashes recur across pages (`0xde5f1345` in every sampled navmesh page and 8,412 of all 9,603,
   `0x00000101` in 14,482 `data.rpf` records). Whether they are class ids, schema hashes or builder tags is
   **open**.
6. **Full-corpus runs.** The tool is bounded by `--limit`; run it on a whole archive before quoting a count for
   that archive. `data.rpf`, `navmeshes.rpf` and `anim.rpf` are read whole in *Whole-archive pass*; `levels.rpf`
   is read whole in *`levels.rpf` in full*, and its map contents are in `docs/world_census.md`. The common-size,
   common-hash and payload-class tables are still slices for navmesh and animation.
7. **`q28` tables past the `q38` boundary.** 1,581 pages declare a `q28` table that ends beyond the streams the
   `q38` records need. Reading the following stream for it would settle whether a resource can be longer than its
   `q38` records say.

## Tool

`tools/page_index.py` is stdlib-only and reads the same data area as the carver. It never writes payloads and
never touches the TOC.

```powershell
python -I tools/page_index.py <archive.rpf> pages.json --limit 268435456
python -I tools/page_index.py <archive.rpf> pages.csv
python -I tools/page_index.py <carve-dir> pages.json
python -I tests/test_page_index.py
```

`--start` must be 0 or more and `--limit` at least 1; both take decimal or `0x` hex. A value outside that, an output
that does not end in `.json` or `.csv`, or a `--start` past the end of the archive stops the tool with a one-line
error and exit code 2, before anything is scanned or written. An empty archive, or one shorter than its 8-byte
header, gives an index with no pages.

JSON has a `summary` and one object per page: offset, stored/inflated size, version, `page_count` (`u32@0x20`),
the four counts, table offsets, `q38_records`, raw `q28_records` (`target` is null for a null pointer; `q28_error`
names a table that was cut short or not read), continuation streams, `span`,
`payload_at_0x60`, `complete`, and a `resource` group (its `streams` with offsets and inflated sizes, `bytes`,
`page_units` at 64 KiB, `declared_pages` and `declared_matches_units`). CSV is one row per `q38` record:
`page_offset,stored,inflated,version,page_count,q28_count,q30_count,q38_count,span,complete,record_index,hash,size,target,target_stream`.
`target_stream` is 0 for the directory stream and 1..N for a continuation.

Validation on the package used here:

| Run | Streams | Pages | `q38` records | Complete | Continuations | Time |
|---|---:|---:|---:|---:|---:|---:|
| `data.rpf` full | 28,509 | 15,768 | 60,880 | 15,768 | 158 | 21 s |
| `levels.rpf` first 64 MiB | 142 | 107 | 871 | 107 | 14 | 1.1 s |
| `navmeshes.rpf` first 64 MiB | 1,378 | 1,132 | 6,645 | 1,132 | 210 | 1.6 s |
| `anim.rpf` first 256 MiB | 24,232 | 12,398 | 269,320 | 12,398 | 7,865 | 25 s |
| `navmeshes.rpf` full | 90,295 | 9,603 | 54,898 | 9,603 | 2,170 | 2.8 min |
| `anim.rpf` full | 130,470 | 22,823 | 527,439 | 22,823 | 11,760 | 2.5 min |
| `levels.rpf` full (streaming runner) | 1,116,073 | 188,449 | 465,611 | 188,449 | 3,281 | 17.7 min |

The summary also counts the `q28` side: `q28_declared` (the sum of `n28`), `q28_records` (read), `q28_null_targets`
and `q28_tables_unread`.

The synthetic test builds its own RPF9 header and pages. It covers an in-page resource, a payload in the next
stream, a `q38` table that itself straddles the stream boundary, the carved-directory mode, the CSV shape, a
`q28` record with a null pointer in the middle of a table, and a `q28` table past the streams read.

## No-sharing rule

The package is read-only. Do not commit, upload or share the archive, a carved page, a payload excerpt, a name
list or a report that quotes payload bytes. Header bytes, table records and aggregate counts belong in this
repository; the bytes a table points at do not.
