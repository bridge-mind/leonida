# PSIN metadata containers

PSIN is a small metadata container used by the game's data and script archives. It appears at the
start of carved zlib streams in `data.rpf` and `script.rpf`; it is not a compiled script program and
it is not a model, texture or audio resource. The archive table of contents is still encrypted, so
the original names are unavailable; PSIN is identified by its own magic.

## Where it appears

- `script.rpf` carve: **671 streams** start with `PSIN`. **670 are complete**; one is
  the first 64 KiB of a larger record (its first-section offset points past the carved block).
- `data.rpf` carve: **2,281 streams** start with `PSIN` (the same project's carve also reports 1,263
  `RBF0` and 3,132 resource pages). The project's existing `ArchiveReader`/`PsinMetadata` path
  validates **1,045 complete PSIN records** in `data.rpf` and **1,012** in `script.rpf` after joining
  adjacent 64 KiB blocks.
- `common.rpf` has no PSIN. `anim.rpf`, `navmeshes.rpf` and the BVH archives have no PSIN at stream
  starts in the measured samples.

## Header

Every measured file begins with the same 16-byte header:

| Offset | Size | Field |
|---:|---:|---|
| +0 | 4 | `50 53 49 4e` (`PSIN`) |
| +4 | 4 | big-endian absolute offset of the first section (`LSCH`) |
| +8 | 4 | type of the first block-table row (see *Instance data*) |
| +12 | 4 | size of the first block |

`psin_index.py` reports +8 as `version` and +12 as `table_bytes`. They are the first row of the block table that
starts at +8. The value 22 (`0x16`, a block of 64-bit integers) appears on 1,010 of the 18,640 PSIN streams, nearly
all of them in `script.rpf`. On the other 17,630 the field holds a struct name hash or another type code.

## Sections

Sections start at the offset in the header and run to the end of the file. Each section is:

| Offset | Size | Field |
|---:|---:|---|
| +0 | 4 | ASCII tag |
| +4 | 4 | big-endian **total** size, including this 8-byte header |

The next section starts at `offset + size`; the last section ends at EOF. A section with size 8 is
empty. The measured tag counts over the 670 complete `script.rpf` records are:

| Tag | Count | Measured payload |
|---|---:|---|
| `LSCH` | 670 | 464-544 bytes; the type schema (structs and members), decoded below |
| `LENM` | 670 | the schema's enums and their values, decoded below |
| `LSNS` | 670 | block table and 16-byte records; layout decoded below, record contents opaque |
| `LGID` | 670 | empty (8-byte section) |
| `LRSC` | 670 | starts with `0DRP` + version 122; resource page directory link |
| `CHKS` | 670 | file size and a 32-bit checksum |

The `LSCH` sizes on the measured set are 472-552 bytes including the 8-byte header. `LRSC` is the
link to the resource-page world: its payload begins `30 44 52 50` (`0DRP`), then a little-endian 122,
then the page fields used by the generic RPF pager. `CHKS` carries the file size in big-endian and a
32-bit hash; the tool reports the section, not a verified checksum.

## `LSCH` and `LENM`: the type schema

`LSCH` describes the structures of the container's data and `LENM` the enums those structures use. It is a
reflection schema of the kind RAGE's PSO format keeps in `PSCH`, with names reduced to 32-bit hashes. Neither
section carries strings.

### Corpus

Measured over **every PSIN stream in every archive** of package 1.0.2.1. Each `.rpf` except `audio.rpf` (raw
AWC, not zlib) was walked whole with the `rpf9_carve.py` stream walker; on `script.rpf` the PSIN-only walk gives
byte-for-byte the same 1,012 streams as `rpf9_carve.py`. Archives with no PSIN stream: `common`, `shaders`,
`textures`, `movies`, `appdata0`, `core_bvh4`, `core_licenses_data`, `core_restofworld001`,
`shared_minigame000_bvh4`, `story000_bvh4`.

| Archive | PSIN | complete | whole `LSCH` | `LSCH` decoded | whole `LENM` | `LENM` decoded | structs | members | enums | values |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `levels` | 6,982 | 6,650 | 6,680 | 6,680 | 6,671 | 6,671 | 51,414 | 296,370 | 13,913 | 178,110 |
| `story000` | 5,348 | 5,075 | 5,101 | 5,101 | 5,097 | 5,097 | 30,684 | 161,449 | 6,840 | 125,837 |
| `data` | 2,281 | 2,135 | 2,108 | 2,108 | 2,104 | 2,104 | 45,896 | 298,356 | 13,814 | 156,999 |
| `story000_script` | 2,309 | 2,118 | 2,160 | 2,160 | 2,156 | 2,156 | 54,683 | 465,992 | 8,587 | 83,740 |
| `script` | 1,012 | 1,009 | 1,009 | 1,009 | 1,009 | 1,009 | 3,879 | 25,763 | 1,009 | 6,054 |
| `packs` | 586 | 489 | 517 | 517 | 515 | 515 | 5,432 | 52,253 | 1,438 | 48,950 |
| `shared_minigame000` | 73 | 70 | 70 | 70 | 70 | 70 | 277 | 1,430 | 56 | 624 |
| `navmeshes` | 22 | 1 | 1 | 1 | 1 | 1 | 3 | 15 | 0 | 0 |
| `anim` | 18 | 10 | 10 | 10 | 10 | 10 | 43 | 200 | 5 | 20 |
| `core_restofworld` | 6 | 5 | 6 | 6 | 6 | 6 | 51 | 323 | 3 | 28 |
| `story000_restofworld` | 3 | 2 | 2 | 2 | 2 | 2 | 9 | 33 | 0 | 0 |
| **all** | **18,640** | **17,564** | **17,664** | **17,664** | **17,641** | **17,641** | **192,371** | **1,302,184** | **45,665** | **600,362** |

- 157 truncated streams still hold a whole `LSCH`; they are counted. 919 streams are cut off before their first
  section and hold neither.
- 57 complete streams (54 in `data`, 2 in `packs`, 1 in `anim`) use the PSO-style `PMAP`/`PSCH`/`PSIG`
  sections instead and have no `LSCH`.
- 6,421 `LENM` sections are empty (8 bytes): those schemas use no enum.
- `LSCH` sizes, header included: 472-552 bytes on `script`, 2,440-7,464 on `story000_script`, 56-17,496
  on `data`.
- `psin_index.py` reports no schema error on any of them (see the checks under *Tool*).
- Distinct name hashes: 3,175 structs, 10,943 members, 664 enums, 13,946 enum values.

### Shape of both sections

Each section stores **all headers first, then all elements**, in header order. The header count is not stored:
it is the one count *k* for which *k* headers plus the elements they announce fill the payload exactly. That
count exists and is unique on 17,664 of 17,664 `LSCH` and 17,641 of 17,641 `LENM` sections.

### `LSCH` struct header (32 bytes)

| Offset | Size | Field | Confidence |
|---:|---:|---|---|
| +0 | 4 | name hash (case-sensitive joaat) | high: see *Hashes* |
| +4 | 4 | layout signature | high: see below |
| +8 | 1 | flags: 0, 0x0c, 0x1c or 0x24 on 150,352 / 34,361 / 89 / 7,569 structs | field: high; meaning: unknown |
| +9 | 1 | log2 of the alignment, 0-4 | medium: the size is a multiple of 2^n on 192,371 of 192,371 |
| +10 | 2 | 0 on every struct | high |
| +12 | 4 | 0 on every struct | high |
| +16 | 8 | fill, `aa` × 8 on every struct | high |
| +24 | 4 | byte size of the struct | high: every member offset is below it |
| +28 | 2 | 0 on every struct | high |
| +30 | 2 | member count | high: the counts fill the section exactly |

The **layout signature** depends on the layout, not the name. A struct name hash that appears with two
signatures always has two layouts, and the other way round (3,175 of 3,175 names; 49 names have two layouts).
The 199 signatures that occur under more than one name give the same layout (size, offsets, types, sub-codes
and indices) under every name. Its hash function is not known.

### `LSCH` member (16 bytes)

The members follow the struct headers, each struct's members in a row.

| Offset | Size | Field | Confidence |
|---:|---:|---|---|
| +0 | 4 | name hash (case-sensitive joaat), or `0x100` for an array element descriptor | high |
| +4 | 4 | byte offset in the struct (0 for an element descriptor) | high: below the size on 1,302,184 of 1,302,184 |
| +8 | 1 | type code | field: high; per-type meaning: see below |
| +9 | 1 | sub-code: non-zero on every member of types 0x4d and 0x4f, 0 on all others | field: high; meaning: unknown |
| +10 | 2 | element index: for an array (type 0x5c), the position in this struct's member list of its `0x100` element descriptor | high: 49,375 of 49,375 arrays point at a descriptor |
| +12 | 4 | reference hash: a struct or enum name for the types below, else 0 | high for 0x05 and the enum types |

Type codes, with the measured references and the *gap*: the distance to the next member or to the end of the
struct, most common value first. The gap is an upper bound on the member's size, so every reading below is a
hypothesis until a struct is matched against its instance data.

| Type | Members | Reference | Gap | Reading |
|---:|---:|---|---|---|
| 0x05 | 192,944 | a struct name: 155,495 in the same file, 36,277 in another file, 1,172 not found | varies | embedded struct (high) |
| 0x40 | 2,311 | the length in bytes (256, 128 or 16; equal to the gap on every member measured) | 256, 128 | `char[ref]` (high) |
| 0x5c | 49,375 | none; element index instead | 16 | array `{u32 count, u32 capacity, pointer}` (high) |
| 0x60 | 3,887 | an enum (3,866 same file, 21 other file) | 1 | 8-bit enum (medium) |
| 0x62 | 40,024 | an enum (39,758 / 253; 13 not found) | 4 | 32-bit enum (medium) |
| 0x63, 0x64 | 47, 85 | an enum | 1, 2 | 8/16-bit enum variants (low) |
| 0x65 | 16,217 | an enum (16,099 / 115; 1 not found) | 8, 4 | enum-indexed bit set (low) |
| 0x68 | 98 | an enum | 8 | 64-bit enum (medium) |
| 0x50, 0x53 | 20,777, 367 | the element count in the low 16 bits | varies | fixed array in place (high) |
| 0x71, 0x73, 0x75 | 114,700, 111, 12,888 | the key and value element indices (`ref & 0xffff`, `ref >> 16`) | 24, 16, 16 | maps (high), see *Instance data* |
| 0x76, 0x77, 0x54, 0x5a | 4, 2, 1, 3 | varies | varies | 0x54: inline array; the others are not decoded |
| 0x01, 0x11 | 93,811, 14,761 | - | 1 | 1-byte scalars (medium) |
| 0x13, 0x20 | 7,372, 7,356 | - | 2 | 2-byte scalars (medium) |
| 0x14, 0x15, 0x21, 0x4a | 35,088, 41,144, 110,948, 194,191 | - | 4 | 4-byte scalars (medium) |
| 0x07, 0x08, 0x30, 0x42 | 52,656, 2,232, 1,112, 1,084 | - | 8 | struct pointer, 64-bit int, 2×f32, string pointer (high) |
| 0x4d, 0x4f | 152,193, 15,076 | - (sub-code set) | 8, 4, 16 | 4- and 8-byte values, read as u32/u64 hashes (medium: size measured, meaning not) |
| 0x0d, 0x31, 0x33, 0x34, 0x52 | 582, 7,750, 50,665, 6,072, 50,156 | - | 16 | 16-byte vectors; 0x52 is the list `{pointer, u16 count, u16 capacity}` (high) |
| 0x36 | 402 | - | 64 | 64-byte value (medium) |

The rest (0x0c, 0x10, 0x12, 0x16, 0x17, 0x32, 0x44, 0x4b) occur 3-1,551 times each. `psin_index.py --schema`
lists every member. The "(high)" readings were checked against the instance data in the next section. The gap
alone is only an upper bound.

### `LENM` enum header (24 bytes) and value (16 bytes)

| Offset | Size | Field | Confidence |
|---:|---:|---|---|
| +0 | 4 | name hash; a member of the same file's `LSCH` refers to every enum (45,665 of 45,665) | high |
| +4 | 4 | layout signature: it changes exactly when the value list changes (664 of 664 names; 9 have two lists) | high |
| +8 | 8 | fill, `ee` × 8 on every enum | high |
| +16 | 4 | value count | high: the counts fill the section exactly |
| +20 | 4 | 0 on every enum | high |

The values follow the enum headers, each enum's values in a row:

| Offset | Size | Field | Confidence |
|---:|---:|---|---|
| +0 | 8 | signed value: 0, 1, 2, ... in 38,870 of 45,665 enums; 1,270 enums have a negative value | high |
| +8 | 8 | name hash (case-sensitive joaat) in the low half; the high half is 0 on every value | high; unique within each enum (45,665 of 45,665) |

### Hashes

Names are the **case-sensitive joaat** (`world_names.joaat(text, case_sensitive=True)`), as for the engine
member hashes in [world_names.md](world_names.md). Candidates: every identifier-like string in the inflated
`data.rpf` and `script.rpf` streams (3,386,171 strings), against the 32,313 distinct hash values above:

| Function | struct names | member names | enum names | enum value names | signatures |
|---|---:|---:|---:|---:|---:|
| joaat, case-sensitive | 26 / 3,175 | 733 / 10,943 | 0 / 664 | 1,023 / 13,946 | 1 / 3,750 |
| joaat of the lower-cased string only | 0 | 8 | 2 | 4 | 0 |
| control: CRC-32; CRC-32 xor a constant | | | | | 28; 30 over all |

1,782 name hits against a chance level of about 30 confirm the function. The low coverage only means that the
engine's type and field names are mostly not in the data files. The one signature hit is at chance level: the
signatures are not joaat of a name. No name or hash from the game is committed here.

### Links to the rest of the container

- Struct references (type 0x05) resolve in the same `LSCH` for 155,495 of 192,944 members and in another
  file's `LSCH` for 36,277 more. Enum references (types 0x60, 0x62-0x65, 0x68) resolve in the same file's
  `LENM` for 59,945 of 60,354 and in another file for 395 more.
- The region between the 8-byte header and `LSCH` is the instance data these structs describe: a block table
  and its data blocks, read from the root block that `LRSC` names. See *Instance data*.
- `LSNS` values occur nowhere else in the container (see *`LSNS`*).

## `LSNS`

`LSNS` is a block table followed by the blocks it lists:

| Offset | Size | Field |
|---|---|---|
| +0 | 4 | little-endian block count n |
| +4 | 8 × n | per block: u32 kind, u32 byte count |
| +4 + 8n | | the n blocks back to back |

The byte counts fill the section exactly, and every block is a whole number of 16-byte records. A section with
n = 0 is 12 bytes long (tag, size, count) and is followed directly by `LGID`. The kind's high byte is 1 or 2 and
reads like the word width (4- or 8-byte words), but the 16-byte rule holds for both. The low byte varies.

Measured with `psin_index.py` over every complete container in five carved archives:

| Archive | `LSNS` sections | Empty | 16-byte records | Errors |
|---|---:|---:|---:|---:|
| `data.rpf` | 2,083 | 886 | 164,512 | 0 |
| `script.rpf` | 1,009 | 0 | 56,382 | 0 |
| `story000_script.rpf` | 2,118 | 0 | 285,674 | 0 |
| `core_restofworld.rpf` | 5 | 0 | 605 | 0 |
| `story000_restofworld.rpf` | 2 | 0 | 1,975 | 0 |
| **Total** | **5,217** | **886** | **509,148** | **0** |

Blocks by kind across these sections: `0x101` 4,200, `0x11c` 2,118, `0x207` 1,235, `0x10f` 26, `0x204` 14,
`0x113` 7, and 1-2 each of `0x102`, `0x205`, `0x208`, `0x109`, `0x11e` and `0x211`. Section sizes run from 12 to
36,756 bytes (median 836).

### The records

The record contents are opaque. A wider pass over 16,169 containers that have an `LSNS` block (the archives
above plus `levels.rpf`, `story000.rpf`, `packs.rpf` and `anim.rpf`; 22,101 blocks, 1,957,739 records) measured:

- **Uniform bytes.** 8.0 bits/byte over the pooled records, and no record byte position favours any value.
  (The per-section 5.2-8.0 bits/byte in [restofworld.md](restofworld.md) is the small-sample limit of short
  sections.) Records are not version-4 GUIDs: no fixed version or variant nibble.
- **No references.** No 32-bit or 64-bit record word occurs anywhere else in its container, `LSCH` included (a
  300-container sample, 107,594 words). Instance data stores its own hashes directly (member types `0x4a`-`0x4f`).
- **No name hash.** Against up to 3.3 million harvested identifiers and model names, each as is, lower-cased and
  upper-cased, the records match none of these functions. Each search was checked against a planted known value.
  - 32-bit words: joaat, FNV-1, FNV-1a, CRC-32, MurmurHash3-32 and xxHash32 all stay at the level of a salted
    control (about 1,350 chance hits each; this test used a third of the dictionary, 1.0 million words,
    against 5.8 million distinct 32-bit words).
  - 64-bit words: MurmurHash3 x64 (either half; the function behind the 64-bit map keys in
    [world_names.md](world_names.md)), xxHash64, XXH3-64, FNV-1 and FNV-1a 64 have 0 matches.
  - Whole records: MD5, XXH3-128, BLAKE2b-128, truncated SHA-1 and SHA-256, and full MurmurHash3 x64 128 have
    0 matches.
- **Repeats.** 18,692 of the 22,101 blocks are distinct. Beyond repeated whole blocks, 99,071 records still repeat
  across different blocks.

Uniform bytes in exact 16-byte units that repeat record by record fit a deterministic 128-bit block cipher. They
also fit 128-bit hashes of inputs outside the tested dictionary and functions. Which one it is has not been
established. Per the project rules, no decryption is attempted.

## `PMAP`, `PSCH`, `PSIG` and `STNS`: the PSO-style variant

The 57 complete `PSIN` streams without `LSCH` (54 in `data.rpf`, 2 in `packs.rpf`, 1 in `anim.rpf`) replace
`LSCH`/`LENM`/`LSNS`/`LRSC` with a big-endian PSO schema chain: `PMAP`, `PSCH`, `PSIG`, optional `STFN`/`STDN`, `STNS`
and `CHKS`. Measured with `tools/psin_psch.py` over all **54** containers in `data.rpf` (`54/54` valid):

- **Header and instance region:** bytes `+0x08..+0x0F` hold `pppppppp` (`0x70` x 8) and instance data blocks sit at
  `+0x10..first_section` (in contrast to `LSCH` containers, which place the block table at `+8`).
- **`PMAP` (287 blocks across 54 files):** starts with a 16-byte big-endian header (`u32 root_block_id`, `u16 flags`,
  `u16 0`, `u16 block_count`, 6-byte `pppppp` fill), followed by `block_count` x 16-byte rows `{u32 name_hash, u32 offset,
  u32 0, u32 length}`. On **54 of 54**, blocks begin at `0x10` and end at exact `first_section`; 237 blocks reference a
  struct in `PSCH` (`237/237` have `length % struct_size == 0`) and 50 are anonymous (`0x1`: 41, `0xc`: 7, `0x8`: 1,
  `0x6`: 1).
- **`PSCH` (`689` structs / `69` distinct, `3,250` members / `500` distinct, `146` enums / `17` distinct, `662` values):**
  `u32be entry_count` followed by `entry_count` directory pairs `(u32be name_hash, u32be section_offset)`. Kind `0`
  entries are structs (`u8 0`, `u8 flags`, `u16be member_count`, `u32be byte_size`, `u32be 0`, then `12`-byte big-endian
  members `{u32 hash, u8 type, u8 sub, u16 offset, u32 ref}`); kind `1` entries are enums (`u8 1`, `u8 flags`, `u16be
  value_count`, then `16`-byte values `{u32 hash, u32 0, i64be value}`). Every array member (`0x0d`, 226/226) and map
  member (`0x10`, 3/3 key/value pairs) points at `0x100` element descriptors inside its struct; all 185 enum members
  (`0x0e`) resolve locally, and 718 struct members (`0x0c`) hold `671` local hashes, `11` zero references, `33`
  cross-file `data.rpf` references and `3` external references.
- **`PSIG` and `STNS`/`STDN`/`STFN`:** `PSIG` stores `entry_count` big-endian `u32` layout signatures (`83` distinct).
  `STNS` mirrors `LSNS` as a little-endian `u32 block_count` table of `(u32 kind, u32 bytes)` pairs and 16-byte-aligned
  records (`867` total across `54` blocks) padded with zeros to a 16-byte payload boundary
  (`align16(4 + 8 * block_count + data_bytes)` on all 54 containers: 4 zero bytes for `1`-block sections and 12 zero
  bytes for `0`- and `2`-block sections). Where present, `STDN` (`2/2`) matches the prefix of `STNS`'s block data
  byte-for-byte.

```text
python -I tools/psin_psch.py <file_or_carve_dir_or_rpf> [--carve-dir DIR] [--json FILE]
```

`tests/test_psin_psch.py` builds synthetic `PMAP`/`PSCH`/`PSIG`/`STDN`/`STNS` containers and tests block validation,
member descriptor checks, carve-directory filtering and `--json` collision refusal.

## Instance data

The region between the 8-byte header and `LSCH` is a block table followed by its data blocks. The schema in
`LSCH`/`LENM` describes the blocks, and `LRSC` names the root. `tools/psin_data.py` reads it.

### Block table and blocks

| Offset | Size | Field |
|---:|---:|---|
| +0 | 4 | block type: a struct name hash from `LSCH`, a member type code below 0x100 (for example `0x10` for string bytes, `0x16` for 64-bit integers) or `0x100 + n` for an anonymous block |
| +4 | 4 | data size in bytes |
| +8 | 8 | fill, `aa` × 8 |

The rows start at +8. The table therefore ends 8 bytes short of a 16-byte boundary, and those 8 bytes are `70` × 8
(`pppppppp`) on 17,507 of 17,507 files. The blocks follow in table order, each padded to 16 bytes, and the last one
ends exactly at the first section (17,507 of 17,507). Blocks are numbered from 1.

The root is the u32 at +0x10 of the `0DRP` header in `LRSC`. It is a struct block on 17,507 of 17,507 files.

A **pointer** is a u64. Its low 20 bits are the block number, with 0 meaning null, and the bits above are the byte
offset into that block.

### Layouts

The tables below give the layouts the instance data confirms. Arrays name their element with the member's element
index, as in `LSCH`. Maps name a key element at `ref & 0xffff` and a value element at `ref >> 16`.

| Type | Bytes | Layout |
|---:|---:|---|
| 0x07 | 8 | pointer to a struct block; the block's type is the struct |
| 0x42 | 8 | pointer to a NUL-terminated string in a type-`0x10` block |
| 0x44 | 16 | string with an in-place form: when bit 7 of byte 15 is set, its low 7 bits are the length of the string in bytes 0..14; otherwise bytes 0..7 are a string pointer as for 0x42 |
| 0x40 | ref | `char[ref]`, NUL-padded |
| 0x50, 0x53 | ref × element | `ref & 0xffff` elements in place |
| 0x54 | 8 + capacity × element | `{u32 capacity, u32 count}`, then the elements in place |
| 0x5c | 16 | `{u32 count, u32 capacity, u64 pointer}` |
| 0x52 | 16 | `{u64 pointer, u16 count, u16 capacity, u32 0}` |
| 0x71 | 24 | `{u64 1, u64 pointer, u16 count, u16 capacity, u32 0}`: `count` (key, value) pairs sorted by key |
| 0x75 | 16 | `{u32 groups, u32 count, u64 pointer}`: `groups` groups of 16 control bytes followed by 16 (key, value) slots. A control byte with bit 7 set marks an empty slot. |
| 0x73 | 16 | `{u32 capacity, u32 count, u64 pointer}`: `capacity` (key, value) slots, where key 0 marks an empty slot |
| 0x60, 0x63 / 0x64 / 0x62, 0x65 / 0x68 | 1 / 2 / 4 / 8 | enum value or enum bit set |

**Alignment.** In a (key, value) slot the value starts at the key size rounded up to the value's alignment, and the
slot size is rounded up to the larger of the two alignments. A struct aligns to 2^(+9 of its header). The 16-byte
vectors (0x0c, 0x0d, 0x31, 0x33, 0x34, 0x35) and 0x36 align to 16, and 0x32 aligns to 4. Every other type aligns to
its size, capped at 8. With these rules the 24-byte 0x71 value sits at +8 after a u32 key, in a 32-byte pair.

### Measured coverage

The measurement walks every value reachable from the root, through embedded structs, pointers, arrays and maps,
and marks each byte it reads.

| Archive | PSIN | with `LSCH`+`LRSC` | tables fill the region | fully reached | block bytes | reached |
|---|---:|---:|---:|---:|---:|---:|
| `levels` | 6,982 | 6,650 | 6,650 | 6,650 | 43,591,021 | 43,591,021 |
| `story000` | 5,348 | 5,075 | 5,075 | 5,075 | 36,404,701 | 36,404,701 |
| `data` | 2,281 | 2,081 | 2,081 | 2,079 | 14,773,021 | 14,772,189 |
| `story000_script` | 2,309 | 2,118 | 2,118 | 2,118 | 30,677,748 | 30,677,748 |
| `script` | 1,012 | 1,009 | 1,009 | 1,009 | 1,323,348 | 1,323,348 |
| `packs` | 586 | 487 | 487 | 486 | 1,924,622 | 1,923,342 |
| `shared_minigame000` | 73 | 70 | 70 | 70 | 227,855 | 227,855 |
| `navmeshes` | 22 | 1 | 1 | 1 | 28,688 | 28,688 |
| `anim` | 18 | 9 | 9 | 6 | 102,040 | 86,344 |
| `core_restofworld` | 6 | 5 | 5 | 5 | 59,136 | 59,136 |
| `story000_restofworld` | 3 | 2 | 2 | 2 | 5,632 | 5,632 |
| **all** | **18,640** | **17,507** | **17,507** | **17,501** | **129,117,812** | **129,100,004** (99.986%) |

The other 1,133 streams are not counted. 1,076 are cut off (see *Corpus*), and 57 use the `PMAP`/`PSCH` form, which
has no `LSCH`.

Checks over the 17,507 containers:

- **Pointers.** 238,681 struct pointers (0x07) and 81,680 string pointers resolve; none points outside its block
  table or past the end of its block. 4,305 more 0x44 strings are stored in place, and every one has a NUL at its
  stated length.
- **Arrays.** 0x5c and 0x52 arrays resolve 177,675 and 628,888 times. Count ≤ capacity holds on all 1,352,211 arrays
  and maps that store both. The target block's type is the element type (or struct) on 806,520 of them. The other
  43 are arrays of enums stored in anonymous `0x100+n` blocks. 24,949 more arrays name an element struct that is
  not in their own file's `LSCH`, and all 24,949 are empty.
- **Maps.** 0x71: the first word is 1 on 159,409 of 159,409. The 69,954 non-empty maps hold 267,161 pairs, and
  69,951 of those maps have ascending keys. 0x75: on all 16,035 non-empty tables the group count is a power of two
  and the used control bytes equal `count` (92,903 slots). In `data`, `levels` and `story000_script`, the control byte of every
  used slot with a u32/u64 hash key (0x4a, 0x4d, 0x4f) is the key's low 7 bits. It is not for other key types. When the target block is larger than `groups` groups, the rest holds further tables. 0x73: on 167 of 167
  the non-zero keys equal `count`, and the capacity is a power of two.
- **Enums.** 533,100 of 533,927 enum values are listed in their `LENM` enum, and 827 are not. A 0x62 value is a
  32-bit field, and `LENM` lists values on both sides of the sign bit for it: of the 33,564 that are negative as `s32`, 2,767
  are listed only as negative numbers and 30,797 only as values of 2^31 or more, so a value counts as listed under either
  reading. Reading it only as `s32`, as the first version of this tool did, leaves 31,624 unlisted. 235,486 of 235,489 bit sets (0x65) use only bits their enum defines. 1,431
  values name an enum that is not in their file's `LENM`.
- **Not decoded.** 0x5a (81 members), 0x76 (12) and 0x77 (1) are counted and skipped. They are the only reason the
  6 partly reached containers are partial: the 0x5a and 0x76 members point at `f32` and u32-hash blocks that
  nothing else reaches.

These numbers cover layout only. What each struct means is not decoded here. The engine names behind the schema
hashes are covered under *Hashes*, and the resource names behind hash values under *Name hashes in the data*.

### Tool

```text
python -I tools/psin_data.py <PSIN file or carve directory> [--json] [--limit N]
python -I tools/psin_data.py <PSIN file> --dump
```

The default output is the counts above, for one file or every `.bin` in a carve directory. A single file that does
not start with the `PSIN` magic is one `ERROR:` line and exit 2; inside a directory such a file is skipped, since a
carve folder holds other streams too. `--dump` prints one container's decoded tree (hashes as hex) on stdout, for
local inspection only. It writes no file and must not be pasted into an issue or a pull request.
`tests/test_psin_data.py` builds containers from invented names and covers every layout above, the end padding, a
bad fill, pointers out of range, an unlisted enum value, a skipped type and the CLI.

### Name hashes in the data

Members of type `0x4a`, `0x4d` and `0x4b` hold a u32 hash and type `0x4f` a u64 hash. Many of these values name other
resources: drawables, fragments and maps. They use the two rules that [world_names.md](world_names.md) found for
placement maps:

| Member type | Value |
|---|---|
| `0x4a`, `0x4d`, `0x4b` (u32) | joaat of the lower-cased name |
| `0x4f` (u64) | low 64 bits of MurmurHash3 x64 128, seed 0, over the name's bytes |

`tools/psin_names.py` walks every container with `psin_data.py`, collects each hash value with the struct and member
that hold it, and looks it up in a dictionary of names you already have: `rpf9_names.py` JSON (the `.#dr`/`.#ft`/`.#dd`
suffix dropped) or plain name lists. Each run also hashes every name with `#zq` appended, through the same rules. Those
names cannot be right, so their matches show how many real matches chance alone would give.

Measured on the 1.0.2.1 package (2026-11-22). The data was 18,640 `PSIN` streams carved whole from 11 archives, so the
container counts differ slightly from those above. The dictionary was 341,954 distinct names from `rpf9_names.py` over
the same archives with default options; 317,257 of them come from `levels.rpf`.

| Archive | Containers | With a resolved name | u32 resolved | u32 control | u64 resolved | u64 control |
|---|---:|---:|---:|---:|---:|---:|
| `data.rpf` | 2,200 | 116 | 9,939 | 52 | 28,905 | 0 |
| `levels.rpf` | 6,951 | 957 | 7,310 | 51 | 34,309 | 0 |
| `packs.rpf` | 543 | 335 | 2,755 | 2 | 10,190 | 0 |
| `script.rpf` | 1,012 | 31 | 189 | 0 | 0 | 0 |
| `navmeshes.rpf` | 21 | 1 | 95 | 0 | 209 | 0 |
| `anim.rpf` | 10 | 1 | 3 | 0 | 0 | 0 |
| `story000.rpf` | 5,316 | 2,648 | 403 | 25 | 90,612 | 0 |
| `story000_script.rpf` | 2,270 | 1,264 | 19,894 | 169 | 23,612 | 0 |
| `shared_minigame000.rpf` | 73 | 2 | 2 | 0 | 0 | 0 |
| `core_restofworld.rpf`, `story000_restofworld.rpf` | 8 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **18,404** | **5,355** | **40,590** | **299** | **187,837** | **0** |

- **The u64 rule is general.** 187,837 of the 443,968 non-zero u64 hash values resolve, with no control match.
  They are spread over 119 of the 223 u64 fields, so the MurmurHash3 keys are not specific to placement maps. The
  other u64 fields match nothing in this dictionary.
- **The u32 rule is precise in name fields.** 40,590 matches against 299 for the control, about 0.7% expected false.
  122 of the 306 fields with a match have at least 20 resolved values and a control under 1% of them. Those fields
  hold 227,504 of the 228,427 resolved values (u32 and u64 together), with no control match. The other 184 fields
  mix a few real names with chance hits on non-name hashes.
- **Coverage.** 5,355 of 18,404 containers (29%) reference at least one known resource, and 23,049 distinct names
  occur. Only 5,495 of the matches are in the placement fields that `world_names.py` already resolves.
- The remaining values are not names in the dictionary. They may be names of other kinds (sounds, animations,
  scenarios), or resources whose own name string `rpf9_names.py` does not find.

```text
python -I tools/psin_names.py <PSIN file or carve directory> --names <rpf9_names.json> [--names ...]
                              [--json] [--fields N] [--refs OUT.tsv] [--limit N]
```

The text output gives the totals and the fields with the most resolved values, as hashes. `--json` adds the whole
field table. `--refs` writes one row per resolved value: file, struct, field, type, value and name. These are game names,
so keep them in your local work folder; only the counts belong in git, issues or pull requests. On the data above the
run takes about 50 s. `tests/test_psin_names.py` builds containers from invented names.

#### Vehicle names in the navmesh PSIN map

The version 256 PSIN at `navmeshes.rpf` offset `0x1749fe000` has a root map of type `0x73`. Its pointer
resolves to an anonymous `0x100` block of 28,672 bytes. The schema gives a `0x4a` key at slot offset
`+0x00` and a 48-byte struct value at `+0x08`, for a stride of 56 bytes. The map declares 512 slots and
209 entries. Counting distinct nonzero keys gives 209.

On the 1.0.2.1 package, compare the lower-case joaat hashes of `ModelName` and the nonempty
`ModDefs/Item` names in the 32 `CVehiclePoints` documents in `common.rpf` with every occupied slot.
Of 31 distinct model names, 25 match the map key. Of 55 distinct part names, 20 match `0x4a` members
in the value: 6 at slot `+0x18`, 2 at `+0x1c`, 4 at `+0x28`, and 8 at `+0x34`. Appending `#zq`
to the model names gives zero key matches. All the original matches lie in the instance block before
the first section. The container's 493 `LSNS` records have zero matches for either set of hashes.
The map identifies vehicles and parts in the container. Which entry refers to each of the 161
local-space navmeshes remains open.

## What it does not give

- No plaintext script code, function names or string table. `LSNS` has a decoded block layout, but its
  16-byte records are uniformly random and match no tested name hash (see *`LSNS`*).
- No station/track names or file paths. Model names occur only as hashes; a name list from elsewhere resolves
  them (see *Name hashes in the data*). The archive TOC is still the only source for original paths.
- No names. `LSCH` and `LENM` give the schema as 32-bit name hashes only; resolving them needs a name
  list from elsewhere (see *Hashes*).

## Tool

```text
python -I tools/psin_index.py <PSIN file or carve directory> [--json] [--schema] [--limit N]
```

`--limit N` stops after N PSIN containers and must be at least 1 (0 or a negative value is refused with exit
code 2). Leave it out to index every container.

- `parse(blob)` returns the magic, version, table size, the ordered section list
  (`tag`, `offset`, `size`, `payload`) and a `complete` flag with the first structural error. When the
  `LRSC` payload starts with `0DRP`, the `0DRP` version is reported; `CHKS` contributes the file size
  and a 32-bit checksum. Text output and the summary include those fields.
- `decode_lsch(payload)` and `decode_lenm(payload)` return the structs (with members) and the enums (with
  values) laid out above, and raise `ValueError` when no header count fills the payload. `parse()` adds
  `structs`, `members`, `enums`, `enum_values` and `schema_error` (an undecodable section, a member past its
  struct's size, or an array whose element index is not an element descriptor) to every record; with
  `--schema` (or `parse(blob, schema=True)`) the decoded schema is included, hashes as hex. Every record
  then has the same `schema` entry, `{structs, enums, lsch, lenm, reason}`, also when the walk stopped before
  reaching the sections (a truncated stream): `lsch`/`lenm` say whether each was decoded and `reason` why
  not ("" when both were); text mode prints the reason as a `schema:` line.
- `collect(src)` scans a single file or every `.bin` in a carve directory; non-PSIN files are
  skipped.
- Text output prints one line per container and a tag summary; `--json` writes `{files, summary}`.
- The parser validates the first-section offset, each tag's printable range, each section size and
  the exact EOF; a 64 KiB split is reported as incomplete rather than guessed.

`tests/test_psin_index.py` covers complete records, an empty section, a bad first offset, a zero-size
section, truncated data, trailing bytes and the JSON CLI, and the `LSCH`/`LENM` layouts, empty sections,
a count that fits nothing and both schema checks. It uses synthetic headers and joaat hashes of invented names
only.

### Viewing a container in ImHex

[tools/imhex/psin.hexpat](../tools/imhex/psin.hexpat) is a pattern for the [ImHex](https://imhex.werwolv.net/) hex
editor that lays out one container as this page describes it:
- the header, the instance-data block table with its `pppppppp` end, and the blocks, as raw bytes;
- the section chain;
- the `LSCH` struct headers and members, and the `LENM` enums and values, with the header count found by the same
  fill-exact rule as `psin_index.py`;
- the `LSNS` block table and its opaque 16-byte records;
- the `LRSC` `0DRP` version and root block, and the `CHKS` file size.

A section that runs past the end of a truncated stream is shown as raw bytes. `tests/test_imhex_patterns.py` builds a
synthetic container, checks it with `psin_index.py` and `psin_data.block_table`, and when ImHex is installed compares
the pattern's output with both.

## Non-PSIN object-tree table in `script.rpf`

Besides its 1,012 `PSIN` containers (1,015 streams, 3,477,872 inflated bytes), `script.rpf` contains one large non-`PSIN`
binary container carved at offset `0x53000` (`42` contiguous deflate streams: 41 x 65,536 + 15,408 = **2,702,384**
inflated bytes / `0x293C30`, 461,333 stored bytes, 43.7% of `script.rpf`'s inflated output). It is not `PSIN`, `RBF0`
or `0DRP`, and carries no strings: it is a dense 483-tree forest serialized in preorder with `0x71717171` (`'qqqq'`)
relocatable pointers, measured with `tools/script_qtable.py` on package 1.0.2.1.

### Layout and header (`0x000000..0x293C2F`)

| Region | Offset | Size | Layout |
|---|---:|---:|---|
| Header | `0x000000` | 40 B | ten `u32` words: `record_count = 30,683` (`0x000077DB`) at `+0x14`; all other nine words `0` |
| Offset table | `0x000028` | 245,464 B | `30,683` strictly ascending, 16-byte-aligned `u64` offsets (`0x03BF00` to `0x293C00`) |
| Record pool | `0x03BF00` | 2,456,880 B | `30,683` 16-byte-aligned records (`32` to `6,704` bytes), ending at exact EOF (`0x293C30`) |

Each record `i` begins with an 8-byte prefix before its body at `offsets[i] + 8`:

- **`+0x00` `u32 root_key_hash`**: non-zero on **483** records (`483` distinct values) and `0` on the other **30,200**
  records.
- **`+0x04` `u32 root_flags`**: `0` on all 30,200 non-root records; on the 483 roots, `0xFFFF0000` (385), `0` (86) or
  `(1..8) << 16` (12).

### Relocatable pointers and tree bijection

Every 8-byte-aligned `u64` word from `+0x08` onward whose high 32 bits equal `0x71717171` (`'qqqq'`) is an internal
pointer whose low 32 bits equal `offsets[target] + 8` (the start of record `target`'s body):

- **30,200 of 30,200** `0x71717171` pointers resolve to a valid record (`0` unresolved, `0` outside the pool).
- **Exact root bijection:** the **483** records with `root_key_hash != 0` are the exact **483** records with
  in-degree `0`; all **30,200** records with `root_key_hash == 0` have in-degree `1`.
- **Contiguous preorder forest:** all **30,200** edges point strictly forward (`target > source`, `0` back-edges or
  cycles), remain inside their root's contiguous index slice `[root_k, root_{k+1})` (`0` cross-tree edges), and match
  strict depth-first preorder in stored pointer-field sibling order across all **30,683** positions. Tree sizes range
  from `1` to `1,131` nodes (median `53`, depths `0..9`).

### Array and struct records

| Kind | Count | Header at `+0x08` | Notes |
|---|---:|---|---|
| Array records | 10,294 | `+0x08 = 0x14`, `+0x0C = 0`, `+0x10 = +0x14 = count` (`0..220`), `+0x18 = stride`, `+0x1C = flags`, `+0x20 = 0` (`u64`), `+0x28 = element_type_hash` (35 distinct, none 0), `+0x2C = 0` | Elements start at `+0x30`; `align16(48 + count * stride) == len(rec)` holds on **10,294 of 10,294**. `flags = 0` (8,211 value/struct arrays, strides `4, 8, 12, 16, 20, 24, 32, 48, 64, 128`) and `flags = 0x10000` (2,083 pointer arrays, all stride `8`, holding 7,297 `0x71717171` pointers). |
| Typed struct records | 20,389 | `+0x08 = struct_type_hash` (66 distinct hashes), `+0x0C = 0` on all 20,389 | Sizes `32` B (17,106: 12,332 wrapper nodes pointing at `+0x10` to `i + 1` with `+0x18 = 0`, 3,540 single-pointer at `+0x18`, 3 two-pointer, 1,231 leaves), `48` B (1,642: 1,641 two-pointer, 1 leaf) and `64` B (1,641 leaves). |

```text
python -I tools/script_qtable.py <carve_dir_or_rpf_or_bin> [--root-hash HEX] [--type-hash HEX] [--json FILE]
```

Accepts an `rpf9_carve.py` carve directory (`script`), a `script.rpf` archive or a joined `.bin` payload.
Default stdout and `--json` output emit aggregate counts, histograms and structural invariant checks only; `--root-hash`
and `--type-hash` report local tree/type counts in hex. `tests/test_script_qtable.py` covers synthetic multi-tree parsing,
64 KiB stream reassembly across a carve folder, pointer validation and output collision guards. A target that does not
exist, or that cannot be read, stops the tool with one `ERROR:` line and exit 2 before anything is parsed; a readable
file that is not an object-tree payload is exit 1, the scan's verdict, and a `--json` file that cannot be written is
exit 2 as well.

## Limits

- The 64 KiB split is not joined automatically. A caller that has the archive order can join adjacent
  blocks first; the tool then parses the complete container.
- `LSNS` record contents and the `CHKS` payload are opaque to this tool; only the `LSNS` block layout is read. `LSCH` and `LENM` are decoded as above, and the instance
  data by `tools/psin_data.py`. Struct, member and enum names stay 32-bit hashes, and most of them have no known
  name.
- No game data, names, hashes or payloads are committed. The carve directories are local.
