# PSO/shader cache

One cache blob was found in the package data, stored as a single resource in the data area of a top-level RPF9
archive. It is a chain of 4,205 adjacent zlib streams: 4,204 full 64 KiB streams and a 27,436-byte last one, with
31,437 bytes of zero padding between the streams. Concatenating the inflated bytes in archive order gives
**275,540,780 bytes** (0x106C6B2C); the first stream starts at archive offset 0x10579000 and the chain is the last
resource in the archive. This page describes that file; `tools/psocache_inspect.py` checks a carved copy of it.

The blob is little-endian. In the tables, *measured* means the field's value and the region arithmetic were checked
on this file. *Unproven* means the field is only located and described, not decoded.

## Header

The header is 0x80 bytes; the first 0x70 record of the next region starts exactly at 0x80.

| Offset | Size | Value measured | Status |
|---:|---:|---:|---|
| 0x00 | 4 | `50 53 00 09` | magic `PS\0\x09`, measured |
| 0x04 | 10 | NUL-terminated ASCII string | meaning unproven |
| 0x24 | 11 | NUL-terminated ASCII string | meaning unproven |
| 0x48 | 8 | 8 bytes | 8-byte value, meaning unproven |
| 0x50 | 4 | 25,471 | count of the 0x70-byte records, measured; the "graphics PSO" role is inferred from the record shape, unproven |
| 0x54 | 4 | 2,693 | count of the 0x18-byte records, measured; the "compute PSO" role is likewise inferred, unproven |
| 0x58 | 4 | 54,576 | meta-index entries, measured (54,576 × 8 ends where the next region starts) |
| 0x5c | 4 | 21,522 | shader-index entries, measured |
| 0x60 | 4 | 31,120,020 | meta blob size, measured (the meta index covers exactly this) |
| 0x64 | 4 | 240,894,464 | code blob size, measured (the last shader entry ends 24 bytes before it) |
| 0x68 | 0x18 | zero | unused in this file |

Two cross-checks tie the counts to the records. The u32 fields at +0x08, +0x0C and +0x50..+0x5C of the 0x70 records
are 0xFFFFFFFF or below 54,576 (50,942 values that are not 0xFFFFFFFF, maximum 54,441). All 10,772 fields at
+0x08..+0x14 of the 0x18 records are below 54,576 (maximum 54,575, the last meta-index entry). A wrong count or
stride breaks both.

## Regions

| Region | Offset | Size | End |
|---|---:|---:|---:|
| header | 0x0 | 0x80 | 0x80 |
| 0x70 records, 25,471 | 0x80 | 2,852,752 | 0x2B8810 |
| 0x18 records, 2,693 | 0x2B8810 | 64,632 | 0x2C8488 |
| meta index, 54,576 × 8 | 0x2C8488 | 436,608 | 0x332E08 |
| shader index, 21,522 × 8 | 0x332E08 | 172,176 | 0x35CE98 |
| meta blob | 0x35CE98 | 31,120,020 | 0x210A92C |
| code blob | 0x210A92C | 240,894,464 | 0x106C6B2C |

The regions are laid end to end and add up to the file size exactly:

```
0x80 + 25,471 * 0x70 + 2,693 * 0x18 + 54,576 * 8 + 21,522 * 8 + 31,120,020 + 240,894,464
   = 128 + 2,852,752 + 64,632 + 436,608 + 172,176 + 31,120,020 + 240,894,464
   = 275,540,780 = 0x106C6B2C
```

## Indexes

Both tables are arrays of `(u32 offset, u32 size)` pairs, 8 bytes per entry, relative to their blob.

- Meta index: 54,576 entries over the meta blob. The first offset is 0 and each entry starts where the previous one
  ends; the last ends at 31,120,020, so there is no gap, overlap or entry past the blob.
- Shader index: 21,522 entries over the code blob, offsets strictly ascending and all distinct. The largest
  `offset + size` is 0xE5BC1E8, 24 bytes before the declared code size 0xE5BC200.

A shader-index entry is not one program. Counting `s_endpgm` inside each entry's range gives 16,992 entries with one,
4,508 with two, 21 with three or more, and one with none. The code blob holds 26,092 `s_endpgm` (0xBF810000) and
1,070,769 `s_code_end` (0xBF9F0000), and 21,463 of the 21,522 entries end on an `s_code_end` dword (58 end on a zero
dword and one on another word). The run after the last `s_endpgm` is filler, so `s_code_end` is not a program
terminator and the `s_endpgm` count does not equal the shader count. A whole-file marker scan also finds two
`s_endpgm` words outside the code blob (one in the meta blob) and no `s_code_end`, so the marker count has to be
bounded to the code region.

## Meta blob

The blob spans [0x35CE98, 0x210A92C) and is 31,120,020 bytes. A meta-index entry starts with a short header and, in
20,318 of the 54,576 entries, continues with 32-byte chunks. The header length is the first multiple of 4 from 0 to
0x40 for which `(size - header) % 32 == 0` and every chunk's first dword is 0x41, 0x42 or 0x43. Measured headers:
0x10 on 17,634 entries, 0x14 on 2,684, 0x18 on 2,684 and 0x1c on 665. Another 3,349 entries fit that shape with zero
chunks. The remaining 30,909 entries do not parse as chunk lists at all; their layout is not decoded here.

| Chunk | Count | 32-byte body |
|---|---:|---|
| 0x41 | 402,328 | first dword 0x41, then seven dwords |
| 0x42 | 371,387 | first dword 0x42, then seven dwords |
| 0x43 | 20,318 | first dword 0x43, a 16-byte value, then 12 zero bytes (all 20,318) |

The second dword of a 0x41 chunk takes six measured values. An external layout note names the first four; that
mapping is not proven here.

| Value | Count | Meaning |
|---:|---:|---|
| 0x300 | 289,590 | named SRV by the external note, unproven |
| 0x302 | 46,186 | named CBV by the external note, unproven |
| 0x303 | 11,328 | named UAV by the external note, unproven |
| 0x002 | 13,897 | named sampler by the external note, unproven |
| 0x401 | 33,596 | no name |
| 0x100 | 7,731 | no name |

The third dword of a 0x42 chunk takes 26 distinct values shaped like 0x000SSIIRR (0x00070001, 0x000D0201 and so on),
which is consistent with space/index/register slots, but the fields are not decoded.

## Meta records: what the chunks decode to

The chunk lists have a fixed order: all 0x41 chunks, then all 0x42 chunks, then exactly one 0x43. That holds for
20,318 of 20,318 records, so one record belongs to one stage. The decode below is measured shape plus an
inference per field.

### 0x41: user-data entries

| Word | Role measured | Confidence |
|---:|---|---|
| +0 | 0x41 kind | high |
| +4 | offset, always a multiple of 8, 0..0x820; zero for 0x2, 0x100 and 0x401 | high that it is 8-byte-granular; the table it points into is not identified |
| +8 | key 0..0x1fd; repeats, 0 is common | low, it is not a unique binding id |
| +0xc | count 0..0x800, 73 distinct for 0x300 | low |
| +0x10 | three bytes: `a` (bits 16..31), `b` (bits 8..15), size (bits 0..7) | high on the split, meaning below |
| +0x14 | 0xffffffff | measured |
| +0x18 | 0 | measured |

Per kind: 0x401 has offset 0, `b`=0, size 4 and `a`=1..24, consistent with a number of 32-bit constants
(medium); 0x2 has offset 0, `b`=5..13 and size 4 (sampler-like, medium); 0x100 has offset 0, `b`=6..11 and
size 2 (low); 0x300/0x302/0x303 carry the nonzero offsets, `b`=4..13, and size 4 or 8 (0x300, 0x303) or 4
(0x302). Reading `b` as a register space and `a` as a base register for descriptors, or as a count for 0x401,
fits the values but has no independent check (medium/low).

### 0x42: register slots

The slot splits as 0x00SS_IIRR: SS 1..16, II 0..3, RR 1..2. The slots follow the 0x41 descriptors in order. In
the 5,230 records where their counts are equal, positional pairing gives II=0 for 23,047 of 23,049 0x300
entries, II=1 for 4,144 of 4,146 0x303 entries and II=2 for all 6,100 0x302 entries. That is the order
SRV=0, UAV=1, CBV=2 of D3D12_DESCRIPTOR_RANGE_TYPE, so II is probably the range type (medium-high). SS is
probably a register space or table index (medium) and RR (1 or 2) a width (low).

The 0x2 samplers and the 0x401 constants take no slot at all (0 of 13,897 and 0 of 33,596). Static samplers
taking no user-data dwords matches the layer's contract in [d3d12_vulkan.md](d3d12_vulkan.md) 3.5 (medium-high).
Another 15,068 records have more slots than descriptors, usually one or two; those extras all have RR=2, a
zero key and w4=0, with (SS,II) mostly (14,3), (2,0) or (10,0). They are not decoded (low).

### Relation to the 32 user-data dwords

Within one record the cost is small: a graphics chunk record carries 11 constant dwords and one or two tables,
15 dwords under two dwords per table. The 32-dword block is therefore the union of the tables and constants,
not one dword per 0x41 entry (medium). The 0x41 offsets also exceed 128 bytes for half the entries (range
0..0x820, 49.5% at or below 0x78), which fits a per-table descriptor index better than a dword offset. The
records that do not fit are elsewhere: 4,519 of the 20,318 chunk records exceed 32 dwords under two dwords per
table (4,202 under one), and none of them is on the graphics path. Of those 4,519, 1,017 are named by a compute
PSO and the other 3,502 are not named by the graphics or compute records measured here. The extreme record has
231 constant dwords; the largest slot count is 281. No mapping from a slot to a user-SGPR index was confirmed.

### Which PSOs reference which records

- A graphics PSO's u32 at +8 names a stage record (sizes 380..520) and +0xC an input-layout record; +0x50..+0x5C
  are 0xFFFFFFFF in this corpus.
- The stage record names chunk records at +4 (entry 4 on 13,409 PSOs, entry 6 on 12,062), at +0x2C (entry 2 on
  25,469) and further at +0x78, +0x90, +0x150 and +0x1A4. Graphics PSOs share eight chunk records in total, not
  one per PSO.
- A compute PSO's +8 and +0xC name 20- and 28-byte hash records, +0x10 a mixed record and +0x14 a chunk record
  (2,693 references, 2,684 distinct records).

## Graphics user signature

`tools/psocache_usersig.py` resolves a graphics PSO to its user data and prints, per chunk, the kind, register
space, base register, count, element size and the slot bytes. A PSO uses 3 to 5 distinct chunk records after
duplicates are removed (four on 23,625 of 25,471 PSOs); the eight layouts are shared across the corpus.

| Chunk or field | Reading | Confidence |
|---|---|---|
| 0x300 / 0x302 / 0x303 | descriptor entries | high that they are descriptors, medium for the SRV/CBV/UAV names |
| 0x002 | sampler entry, no slot | medium-high; none of the eight graphics records carries one |
| 0x401 | 32-bit constants, `a` = count | medium |
| 0x100 | unknown, not consumed | low |
| 0x41 offset (w2) | descriptor index inside the table, 8-byte granular | medium |
| 0x41 `b` | register space | medium |
| 0x41 `a` | base register for descriptors | medium |
| 0x41 size | element size 2, 4 or 8 | high |
| 0x41 key (w3) | unknown | low |
| 0x41 count (w4) | descriptor count | low |
| 0x42 SS | descriptor-table / root-parameter index | medium-high |
| 0x42 II | range type: 0 SRV, 1 UAV, 2 CBV, 3 sampler | medium-high |
| 0x42 RR | width, 1 normally and 2 on extra slots | low |

The kind-to-range-type correspondence is exact in the positional pairing: 23,047 of 23,049 0x300 entries sit on
II=0, 4,144 of 4,146 0x303 on II=1 and all 6,100 0x302 on II=2. One distinct SS is read as one
`D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE`, each (SS, II) pair as one `D3D12_DESCRIPTOR_RANGE`, and each 0x401
as one `D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS`. That mapping is one-to-one for the measured graphics
records, but it is an inference: SS could be a register space and `b` the table index instead, and the cache
does not say whether a table costs one or two user-data dwords.

Under both table models all 25,471 graphics PSOs fit 32 dwords (15 dwords with two per table). A later
`CreateGraphicsPipelineState` path would group the slots by SS into tables, take II as the range type and the
0x41 entries as the range contents, take each 0x401 as a constants parameter, and pack them in root-parameter
order. The record does not carry that order, so it would have to come from SS or from the shader's reads, which
is not measured.

## What remains unknown

- The meaning of the two header strings, the 0x48 value and the six 0x41 second-dword values. The SRV/CBV/UAV/sampler
  names are the external note's, not this file's.
- The `a` and `b` bytes of 0x41 w5 and the SS/II/RR meanings of the 0x42 slot word: the readings above are
  inferences from value ranges and order, not confirmed by the shader code.
- The encoding of the w3 key and w4 count, and the 15,068 records with extra slots.
- The 0x43 16-byte value was not matched against any hash in the package.
- The 30,909 meta entries that are not chunk lists.
- Whether the code blob is raw RDNA2. The evidence is the two marker words and the `s_code_end` filler; no
  disassembly or translation was run on this blob.

## Checking and carving a copy

`tools/psocache_extract.py` finds the cache in an RPF9 archive without the TOC: it walks the archive's stream
table in order, inflates each stream to learn its extent, keeps only the first 4 bytes, and stops at the first
stream whose output starts with the magic. It then appends the following contiguous streams (up to 1 MiB of zero
padding) until the header's declared total is reached, writes the file under the temporary directory and runs
the checks below. `--budget` bounds the compressed bytes examined and `--max-streams` the number of streams; a
source folder is scanned in sorted order and the report names the archive index and the stream offset, never the
path.

```
python -I tools/psocache_extract.py <archive.rpf|folder> <cache-in-tmp>
```

`tools/psocache_inspect.py` reads the cache file only:

```
python -I tools/psocache_inspect.py <cache> --meta
```

It checks the magic, reads the four counts and two sizes above, bounds all six regions against the file, checks that
the meta index is contiguous over the meta blob and that every shader entry stays inside the code blob, parses the
chunk lists and counts `s_endpgm` and `s_code_end` in the code blob. `--meta` adds the aggregate 0x41/0x42 decode
described above, `--entry N` lists one meta index entry and `--limit N` bounds how many records `--meta` reads. It
prints offsets, sizes and counts, names every field that does not fit instead of stopping, and exits 0 when the
regions add up to the file size, 1 on a disagreement and 2 when the file is not a cache. `--json` prints the same
report as JSON. It does not decode a chunk payload, disassemble the code or verify a hash.

`tools/psocache_usersig.py` turns one or more graphics PSOs into the per-record user signature and the 32-dword
fit described above:

```
python -I tools/psocache_usersig.py <cache> [--entry N] [--limit N] [--json]
```

It reuses the parser from `tools/psocache_inspect.py`, prints the aggregate over the first `--limit` graphics PSOs
(all by default) or one PSO with `--entry N`, and never prints a hash.
