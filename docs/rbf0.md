# RBF0 resources in `data.rpf`

`RBF0` is RAGE's binary form of an XML data file. It is a single record stream with an inline descriptor table, not
a section container. This note measures the 1,263 `RBF0` resources carved from `data.rpf` of the Xbox Series
1.0.2.1 package, using the fixed zlib walk from PR #90. It records format facts and aggregate counts only: no
payload, carved stream, path list or string dump is committed here.

## Bottom line

| Question | Result |
|---|---|
| How many `RBF0` streams? | **1,263 logical resources**: 1,243 single-stream and 20 split over 2-37 zlib chunks. The magic is exactly `RBF0` at offset 0 in all 1,263. |
| Is the header stable? | **Yes.** A 4-byte prologue `FB FF 01 00` in 1,260 streams; absent in 3. The record grammar is identical in both. |
| Is there a section directory? | **No.** No section count, offset table or size table exists. All 1,263 resources parse to their exact last byte; the only structure is the record sequence. |
| Is there a name or hash table? | **No.** Schema names are inline UTF-8 with a u16 length. There is no hash table and no readable hash string. The 64-bit integers are mostly small codes; none is identified as a name hash. |
| What do the streams carry? | UI framework and UI data schemas, movie-subtitle containers, and three tuning/config schemas. No animation, mesh or audio payload. |
| Is it the same format as the `anim.rpf` rig blob? | **Yes, same container and grammar.** That blob is one `rage__crRigSetManager` resource; its 9 descriptors and 28,962 elements parse with the same rules. |
| Can it be decoded to a tree? | **Yes.** `tools/rbf0_tree.py` decodes all 1,263 resources to element/value/text trees. Unknown descriptors and record types fail closed. |
| Compression inside `RBF0`? | **None observed.** The only compression is the outer RPF9 zlib framing. |

## 1. Method and coverage

The corrected `rpf9_carve.py` walks the 481,544,304-byte `data.rpf` data area and inflates 28,509 zlib streams:
423,279,355 stored bytes and 1,403,250,069 inflated bytes. Of those streams, **1,263 start with `RBF0`** and 96
more are 64 KiB continuations of the 20 chunked resources, so 1,359 streams belong to `RBF0` resources.

The 1,263 starts are 0.48% of the carved stored bytes (2,040,469) and 1.03% of the inflated bytes (14,476,545).
The RBF0 count is the same as the earlier scan; the carve fix changes trailing streams, not this class.

Each start was inflated, joined to its continuations by archive order, and parsed with the grammar in section 3.
**1,263 of 1,263 resources parse exactly to the last byte** after joining. The parser is `tools/rbf0_index.py`;
the unit test builds only synthetic headers.

## 2. Sizes and chunking

| Measure | Value |
|---|---:|
| Logical resources | 1,263 |
| Inflated bytes | 14,476,545 |
| Stored bytes | 2,040,469 |
| Minimum | 98 |
| First quartile | 1,287 |
| Median | 3,429 |
| Third quartile | 7,433 |
| Maximum | 2,372,401 |
| Mean | 11,462 |

| Inflated size | Resources |
|---|---:|
| < 1 KiB | 259 |
| 1-4 KiB | 456 |
| 4-16 KiB | 419 |
| 16-64 KiB | 109 |
| 64 KiB-1 MiB | 18 |
| >= 1 MiB | 2 |

A single RPF9 zlib stream in this package tops out at 65,536 inflated bytes. When a resource is larger, the
archive stores another zlib stream whose first bytes continue the previous record, not a new `RBF0` magic or a
continuation header. The split can fall inside any field: the 20 multi-chunk resources use 2, 3, 4, 8, 10, 17 and
37 chunks, with 96 continuation streams in total. The record stream carries no length field and no self-size, so a decoder
must append by archive order and stop when the root close leaves the buffer exactly consumed (or at the next
known signature). All 20 multi-chunk resources are in the 299.6-302.7 MB archive range. All 20 are wide; the three
narrow resources are single-stream.

## 3. Header and record grammar

There is no established version field and no section count. The only header is:

```
0x00  4 bytes   "RBF0"                          all 1,263
0x04  4 bytes   FB FF 01 00                     1,260; the other 3 start the records here
0x08  ...       first record                    (0x04 in the three streams without the prologue)
```

Measured first bytes of the 98-byte stream at archive offset `0x12035690`:

```
52 42 46 30  fb ff 01 00  00 00 11 00  75 69 50 72 6f 67 72 65 73 73 ...
"R B F 0"    prologue      idx 0, elem  name length 17  "uiProgressDataset" ...
```

The prologue's meaning is not established from the bytes. It is perfectly correlated with value width: the 1,260
streams with it use 8-byte integers and floats, and the 3 without it use 4-byte integers and floats. Whether it is
a version, a build mode or a width flag is not proven. The three narrow resources are at archive offsets
472,271,120 (271 bytes, `CTuningFile`), 472,498,176 (15,432 bytes, `InputContextStyleStore`) and 473,161,728
(18,317 bytes, `rage__SafeframeConfig`).

Every record is a 1-byte descriptor index and a 1-byte type:

| Index / type | Meaning | Payload |
|---|---|---|
| `idx < descriptor_count` | use an existing descriptor | none |
| `idx == descriptor_count` | new descriptor | u16 name length, then UTF-8 name |
| `idx > descriptor_count` | invalid | none |
| `0xFF 0xFF` | close the open element | none |
| `0xFD 0xFF` | text for the open element | u32 length, then that many bytes (NUL-terminated in this corpus) |
| type `0x00` | open an element | three u16 words |
| type `0x10` | integer | 8 bytes (wide) or 4 bytes (narrow) |
| type `0x20` / `0x30` | true / false | none |
| type `0x40` | float | 8 bytes (wide) or 4 bytes (narrow) |
| type `0x50` | three floats | 3 x width (not present in this corpus) |
| type `0x60` | string | u16 length, then UTF-8 bytes |

The three u16 element words are `0, 0, attribute_count` in every stream. Across **475,461 elements**, word 0 and
word 1 are always zero and word 2 is 0 (389,525), 1 (84,541) or 2 (1,395). The declared attribute count matches
the number of leading value records after the open in **475,461 of 475,461** elements, with zero mismatches. This
is the same rule the public `rage_engine_formats` RBF reader describes for the GTA V form; the measurements here
are independent.

One difference from the GTA V reader matters: this corpus **reuses a descriptor with more than one value type**.
The GTA V reader rejects a record whose type differs from the descriptor's first type; this corpus does that in
**6,131 of 591,064 reused-descriptor records (1.0%), spread over 536 of the 1,263 resources**. The descriptor is a
name index, not a type declaration, and the index tool does not enforce the GTA V rule.

### Aggregate record counts

| Record | Count |
|---|---:|
| Descriptors (names) | 44,887 |
| Distinct descriptor names | 698 |
| Elements | 475,461 |
| Scalar values | 160,490 |
| Text records | 254,935 |
| String values (`0x60`) | 85,924 |
| Integers (`0x10`) | 29,400 |
| Floats (`0x40`) | 38,304 |
| True / false (`0x20` / `0x30`) | 3,410 / 3,452 |
| Vectors (`0x50`) | 0 |
| Maximum element depth | 36 |

Per resource: descriptors 4 (minimum), 29 (median), 204 (maximum); elements 1 (minimum), 106 (median), 68,398
(maximum). The integer values are mostly small codes: 337 distinct, 29,221 fit in 32 bits, 13 are
`0xFFFFFFFFFFFFFFFF`, and the most common value is 1 (7,979 occurrences). The 38,304 floats have 7,814 distinct
values and range from -3,840.0 to 65,536.0. The index tool reports counts and samples; mapping these numbers to
fields needs the schema, which the format does not carry.

### What is absent

- **No section table.** A complete parse consumes every byte; there is no region left for a directory. The
  element words are not offsets.
- **No hash table.** Descriptor names are literal. There are no 8- or 16-hex-digit strings in the corpus, and no
  hash field adjacent to a name.
- **No nested compression.** No byte-blob value type exists and every record payload is consumed by the parser.
- **No file name or path.** A resource does not store its own archive path; that would come from the RPF9 TOC.

## 4. Readable and opaque

The 1,263 resources contain **340,859 string occurrences** and **29,986 distinct strings**:

| Category | Count |
|---|---:|
| Strings with `/` | 4,087 occurrences, 1,057 distinct |
| Identifier-shaped (letters, digits, `_.:@+-`) | 333,176 occurrences |
| Empty | 22 |
| Longest path-like string | 126 bytes |

The identifiers are UI field names, control names and enum values. The path-like strings are UI resource
references (control and panel paths), not archive paths: they are relative, with no `pack:/` prefix. A keyword
scan over all strings gives:

| Term | Occurrences | Distinct | Reading |
|---|---:|---:|---|
| `anim/` | 0 | 0 | no animation paths |
| `pack:/` | 0 | 0 | no pack-root paths |
| `SKEL_` | 0 | 0 | no bone names |
| `RBF0` | 0 | 0 | magic is not repeated in strings |
| `xml` | 0 | 0 | no XML text |
| `nav` | 2,019 | 247 | substring of UI identifiers, not nav data |
| `map` | 919 | 203 | substring of UI identifiers |
| `ipl` | 207 | 36 | substring of UI identifiers |
| `handling` | 1,054 | 3 | enum/control names |
| `meta` | 20 | 16 | substring |
| `hud` | 3,836 | 532 | UI names |
| `vehicle` | 9,429 | 2,102 | UI names |
| `weapon` | 5,492 | 859 | UI names |
| `credits` | 965 | 211 | UI names |
| `phone` | 2,788 | 693 | UI names |
| `rage__` | 35,287 | 173 | engine schema names |
| `fwui` | 26,783 | 155 | engine UI schema names |

The counts for `nav`, `map`, `ipl`, `handling` and `meta` are substring matches in identifiers, not evidence of
navigation, map, IPL or handling files. No `RBF0` stream in `data.rpf` carries archive paths, animation paths,
bone names or XML text. There is no readable hash string; whether any integer value is a hash is not established.

## 5. Composition by root schema

| Group | Resources | Elements | Strings | Root schemas (count) |
|---|---:|---:|---:|---|
| `rage__fwui*` UI framework | 895 | 210,513 | 146,339 | `rage__fwuiPrefab` 603, `rage__fwuiAnimationNodeGraph` 190, `rage__fwuiFlowBlock` 61, `rage__fwuiSubscreenPrefab` 14, `rage__fwuiCoreControlPrefab` 11, `rage__fwuiPanel` 8, `rage__fwuiGrid` 6, `rage__fwuiCanvas` 1, `rage__fwuiGpsDefs` 1 |
| Screens | 201 | 170,029 | 119,953 | `UIScreen` 136, `UIMovieView` 65 |
| Movie subtitles | 125 | 18,359 | 14,517 | `CMovieSubtitleContainer` 125 |
| UI data and settings | 39 | 76,420 | 59,813 | `uiFeedbackContext` 28, `uiProgressDataset` 5, `uiSafezone` 4, `uiSettingsGamepadControlsData` 1, `InputContextStyleStore` 1 |
| Tuning and config | 3 | 140 | 237 | `CTuningFile` 1, `CCombatMeleeGroup__Tunables__Tuning` 1, `rage__SafeframeConfig` 1 |
| **Total** | **1,263** | **475,461** | **340,859** | 20 root schemas |

The four largest resources are `uiProgressDataset` at 2,372,401 bytes, `UIMovieView` at 1,094,224 and 653,738
bytes, and `rage__fwuiSubscreenPrefab` at 480,604 bytes. Together they are 4,600,967 bytes (31.8% of the inflated
total), and all four are multi-chunk. The 20 multi-chunk resources are not a separate kind; they are the same
schemas at the large end of the size range.

The most frequent descriptor names are `type`, `Item`, `Name`, `Children`, `Path`, `LayoutParams`, `Style`,
`Id` and `key`. These are schema vocabulary, not asset names.

## 6. Cross-check with the `anim.rpf` rig blob

`anim.rpf` ends in one large `RBF0` resource. The 21 zlib chunks (58,391 stored, 1,348,436 inflated) join to a
1,348,436-byte blob that parses to the exact last byte with the same grammar:

```
52 42 46 30  fb ff 01 00  00 00 15 00  "rage__crRigSetManager" ...
```

| Measure | `data.rpf` (1,263 resources) | `anim.rpf` tail (1 resource) |
|---|---:|---:|
| Inflated bytes | 14,476,545 | 1,348,436 |
| Stored bytes | 2,040,469 | 58,391 |
| Chunks | 1,243 x 1; 20 x 2-37 | 21 |
| Layout | 1,260 wide, 3 narrow | wide |
| Descriptors | 44,887 (698 distinct) | 9 |
| Elements | 475,461 | 28,962 |
| Text records | 254,935 | 20,031 |
| Strings | 340,859 | 21,157 |
| Integers / floats | 29,400 / 38,304 | 6,677 / 0 |
| Maximum depth | 36 | 6 |
| Root schema | 20 classes | `rage__crRigSetManager` |
| Exact parse | 1,263 / 1,263 | 1 / 1 |

The rig blob's nine descriptors are `rage__crRigSetManager`, `RigSet`, `Item`, `key`, `Entries`, `Dictionary`,
`Rig`, `Group` and `Priority`. It has 1,126 `anim/rigset/` keys, 6,677 `anim/rig/` keys and 4,815 distinct
strings. The element words are again `0, 0, attrs` (27,836 with 0, 1,126 with 1) and there are no attribute
mismatches.

The grammar-driven tree decoder in section 9 reproduces those counts and resolves the nesting. The root contains
one `RigSet`, whose 1,126 `Item` elements each carry one `key` string attribute and one `Entries` element. Under
`Entries`, the 6,677 entry `Item` elements contain a `Dictionary` text, a `Rig` text, a `Group` text and a
`Priority` integer after the three child elements. Maximum element depth is 6. The decoder does not resolve the
rig-to-skeleton link.

So the container, prologue, record grammar and element header are shared. The difference is the root schema:
`anim.rpf` carries a rig-set manager, while `data.rpf` carries UI and config schemas. No `data.rpf` resource has
the rig-set root, an `anim/` string or a `SKEL_` string. The `data.rpf` RBF0s are not animation data.

## 7. Relationship to PSIN and the model set

- **PSIN.** `data.rpf` also contains 1,045 complete PSIN metadata records. PSIN is a different container with its
  own section tags; an `RBF0` stream is not a PSIN section. The two classes are disjoint at stream start.
- **Models.** The package's model recovery found 2,313 model entries. `RBF0` carries no mesh, vertex, index or
  texture payload: it has zero vector records and no binary blob value type, and every byte belongs to the record
  grammar. The two sets do not overlap in content.
- **Animation.** See section 6: same format, different root schema. The `RBF0` in `anim.rpf` is the rig-set
  manager; the `data.rpf` set is UI/config.

## 8. What a future decoder needs

1. **Join chunks before parsing.** A logical resource can span 2-37 zlib streams. Append in archive order; a
   continuation starts mid-record. Stop when the root close consumes the buffer exactly, or at the next known
   signature. There is no length field to trust.
2. **Parse the descriptor table and the type byte together.** A new descriptor is introduced when the index
   equals the current descriptor count. Do not enforce the GTA V "type must match the declaration" rule; this
   corpus reuses names with different value types.
3. **Select the value width from the prologue.** `FB FF 01 00` means 8-byte integers and floats; its absence means
   4-byte. Do not guess from the root schema.
4. **Use the attribute count.** The third u16 of an element header is the number of leading value records that
   are attributes; the rest are children. The measured range is 0-2, and the rule holds for every element.
5. **Build the element tree, then map to XML.** Text records attach to the open element. The public
   `rage_engine_formats` RBF reader documents the same mapping for the GTA V form; the `data.rpf` corpus follows
   its record grammar with the wider scalar values.
6. **Do not look for a section directory or a hash table.** There is none. Names are literal; integers are
   values. Mapping a value to a field requires a schema, not a hash lookup.
7. **Keep the compression boundary explicit.** The RBF0 bytes are uncompressed after the outer RPF9 zlib inflate.
   There is no byte-blob value type and the parse consumes every byte, so there is no place for a nested frame.

The chunk join, descriptor/type pairing, width selection, attribute count and tree build are implemented in
`tools/rbf0_tree.py`; section 9 records what that decoder measured and what it still cannot name.

## 9. Tree decoder

`tools/rbf0_tree.py` turns the record stream into a tree. It reads a carve directory or one complete resource,
joins the 64 KiB continuations, and writes one JSON file with one object per resource, plus optional per-resource
generic XML files. It is stdlib-only and uses no schema file.

The full `data.rpf` carve decodes **1,263 of 1,263 resources with status `ok`**, and the totals are identical to the
index counts in section 3: 475,461 elements, 254,935 text records, 85,924 string values, 160,490 value records,
44,887 descriptors (698 distinct names) and maximum element depth 36. No unknown record type occurs in the corpus.

### 9.1 Grammar used by the decoder

| Bytes | Meaning |
|---|---|
| `RBF0` + `FB FF 01 00` | magic and the wide (64-bit integer/float) prologue; without the prologue values are 32-bit |
| `idx type` | one record; `idx < descriptor_count` reuses a descriptor, `idx == descriptor_count` adds one |
| `u16 len` + UTF-8 | descriptor name, present only when the record adds a descriptor |
| `0x00` | element open; three u16 words follow, words 0 and 1 are zero and word 2 is the attribute count |
| `0x10` | unsigned integer, 8 bytes wide or 4 bytes narrow |
| `0x20` / `0x30` | true / false, no payload |
| `0x40` | float, 8 bytes wide or 4 bytes narrow |
| `0x50` | three floats, 3 x width (not present in this corpus) |
| `0x60` | string; u16 length followed by that many UTF-8 bytes |
| `FD FF` | text for the open element; u32 length followed by that many bytes (NUL-terminated in this corpus) |
| `FF FF` | close the open element |

A descriptor is a name index, not a type declaration. The decoder records the type that introduced the descriptor
and all later types used with the same index.

In every one of the 1,263 resources, the declared attributes are the first value records of their element: no child
element or text record precedes them (0 violations). Values after the attribute prefix are content in stream order,
and a value can follow a child element. A text record attaches to the open element.

### 9.2 Measured tree facts

| Measure | Value |
|---|---:|
| Resources decoded | 1,263 / 1,263 `ok` |
| Elements | 475,461 |
| Elements with 0 / 1 / 2 declared attributes | 389,525 / 84,541 / 1,395 |
| Two-attribute type pairs | float/float 128, string/int 1,111, string/float 40, string/string 116 |
| Text records | 254,935, all NUL-terminated |
| String values | 85,924; all are attributes and none is NUL-terminated |
| Text / string length | 2-1,006 / 1-126 bytes (the text length includes its trailing NUL) |
| Descriptor instances reused with more than one type | 564 |
| Root with an attribute | 1: `rage__SafeframeConfig` |
| Descriptor names not usable as XML element names | 1: `xmlns:xi` |
| Unknown record types | 0 |

The single root attribute is `xmlns:xi` on the narrow `rage__SafeframeConfig` resource at archive offset
`0x1c33e000` (18,317 inflated bytes). The same name is declared as descriptor 1 at offset `0x23` and is used 14 times
in that resource. It is also the only descriptor name that cannot be used as an XML element name (the `xmlns`
prefix is reserved), which is why the XML writer uses fixed tags and puts the exact name in a `name` attribute.

Of the 26 text payloads that are not valid UTF-8, all are text records; no string value or descriptor name failed
a strict UTF-8 decode. The decoder replaces invalid bytes with U+FFFD and sets `utf8_valid: false`; the original
encoding is not established.

### 9.3 Measured tree shape

Every resource has one root. The corpus has **1,263 roots**, **474,198 parent-child edges** and **890,886 tree
nodes** in total: 475,461 elements, 160,490 values and 254,935 texts. The depth distribution is:

| Depth | Elements | Depth | Elements |
|---:|---:|---:|---:|
| 1 | 1,263 | 19 | 24,523 |
| 2 | 3,722 | 20 | 31,186 |
| 3 | 15,537 | 21 | 21,989 |
| 4 | 19,248 | 22 | 11,765 |
| 5 | 18,377 | 23 | 10,702 |
| 6 | 24,965 | 24 | 10,247 |
| 7 | 22,295 | 25 | 15,399 |
| 8 | 19,530 | 26 | 8,563 |
| 9 | 18,599 | 27 | 7,918 |
| 10 | 17,738 | 28 | 13,470 |
| 11 | 16,822 | 29 | 5,721 |
| 12 | 14,247 | 30 | 4,177 |
| 13 | 14,740 | 31 | 3,433 |
| 14 | 12,670 | 32 | 1,612 |
| 15 | 14,584 | 33 | 1,139 |
| 16 | 28,395 | 34 | 363 |
| 17 | 18,917 | 35 | 38 |
| 18 | 21,559 | 36 | 8 |

The child-count distribution is concentrated at the leaves: 280,469 elements have no child element, while the
largest element has 156. The mean is 0.9973 child elements per element.

| Child elements | Elements |
|---:|---:|
| 0 | 280,469 |
| 1 | 76,379 |
| 2 | 54,236 |
| 3 | 20,118 |
| 4 | 27,568 |
| 5 | 7,731 |
| 6 | 2,594 |
| 7 | 2,277 |
| 8 | 1,615 |
| 9 | 968 |
| 10 | 191 |
| 11 | 661 |
| 12-19 | 407 |
| 20-29 | 69 |
| 30-39 | 71 |
| 40-49 | 25 |
| 50-59 | 17 |
| 60-69 | 8 |
| 70-79 | 18 |
| 80-89 | 10 |
| 90-99 | 9 |
| 100-109 | 14 |
| 124-127 | 5 |
| 156 | 1 |

The tree also separates attribute values from content values. 87,331 values are attributes (85,924 string, 1,111
integer, 296 float); 73,159 are content (38,008 float, 28,289 integer, 6,862 boolean). 22,977 content values follow
a child element rather than preceding it. 14,423 elements have an empty content list (attributes do not count),
and 254,935 have text as their only content.

A synthetic four-element tree shows the stored fields without using game names:

```
Root         depth 1  attrs 0/0  children 2  content 2
  Item       depth 2  attrs 1/1  children 1  content 2
    value "id" (string, attribute)
    Child    depth 3  attrs 0/0  children 0  content 1
      text "example"
  Empty      depth 2  attrs 0/0  children 0  content 0
```

`attrs` is `declared/actual`; `children` counts child elements only, while `content` counts every ordered content
record (child element, value or text). The root's `end` is the byte after its close; a complete resource consumes
the buffer exactly.

### 9.4 Output and failure behavior

JSON carries one object per resource: `offset`, `size`, `chunks`, `layout`, `root_name`, `status`, `error`,
`descriptors`, `counts` and `tree`. Every node has a `kind` and a byte `offset`. Elements carry `descriptor`,
`attrs_declared`, `attrs_actual`, `children`, `content_count`, `attributes`, ordered `content` and `end`; values
carry `descriptor`, `type`, `payload_offset`, `size` and `value`; text nodes carry `length`, `nul_terminated`,
`text` and `utf8_valid`. The full-corpus JSON is about 0.6 GB because every node keeps its offset and child counts;
`--root`, `--offset` and `--limit` select samples. The JSON summary adds the aggregate tree shape: `nodes`,
`elements_by_depth`, `children_histogram`, `content_length_histogram`, `attribute_count_histogram`,
`leaf_elements`, `empty_elements`, `text_only_elements`, `parent_child_edges`, `max_children` and `mean_children`.

`--csv PATH` writes one flat row per node. The `parent` column is the parent element's byte offset, `slot` is
`attribute` or `content`, `sibling` is the index in that list, and `child_index` is the index among child elements;
`attrs`, `children`, `content_count`, `type`, `length`, `nul`, `utf8` and `value` describe the node. The CSV is a
convenience view of the same structural tree, not a second parse. Use `source` (the starting filename) with
`resource` (the archive offset) to group rows: two files in a carve directory can specify the same offset.

XML is a rendering, not a round-trip; the JSON tree is the form that encodes back to bytes (9.6). Fixed tags avoid descriptor names that cannot be used as XML element names:
`<node name="...">`, `<attr>`, `<value>` and `<text>`. A string containing XML markup is escaped by the writer. The
`<rbf0>` root carries `status` and, on failure, `error`. JSON, CSV and XML all contain names or payloads and are
game data.

The decoder fails closed. An index beyond the current descriptor table, an unknown record type, a non-zero element
header word, a close or text with no open element, and a truncated payload stop that resource with `status` `bad`
or `truncated`; the partial tree is kept and no value is invented. `status` `trailing` means the root closed before
the last byte. The corpus has 0 bad, 0 truncated and 0 trailing resources.

### 9.5 Remaining unknowns

- There is no schema: descriptor names are fields, but value meaning, enum spelling and units are not stored.
- The prologue's meaning is still not established; the decoder uses it only to select 4- or 8-byte values.
- Element header words 0 and 1 are zero in all 475,461 elements. Their intended meaning is unknown; the decoder
  rejects a non-zero value instead of guessing.
- The attribute rule is measured, not proven universal: every corpus attribute leads its element and every text
  element is a text-only leaf. The decoder marks `attribute_order_violation`, `multiple_text` and
  `text_after_child` if a future resource departs from it; the corpus has none of these.
- The 26 invalid UTF-8 text payloads have no established encoding. The JSON, CSV and XML writers replace bad bytes,
  so they are lossy for those records, and `rbf0_encode.py` refuses the 23 resources that contain them (9.6).
- `xmlns:xi` is a literal field name. Its namespace meaning is not decoded.
- `0x50` (three floats) is accepted by the grammar but never occurs in this corpus.
- No external schema or XSD validates the tree; JSON, CSV and XML output are structural renderings only.
- A resource does not store its archive path. The source label is the carve offset or file name, not a game path.

### 9.6 Encoder and round trip

`tools/rbf0_encode.py` writes a tree document back to bytes with the grammar of section 3. It emits each descriptor
name at its first use, then the element header words `0, 0, attribute count`, the attributes, and the content in tree
order. Text and strings keep their recorded bytes and terminating NUL. `--check` decodes every resource of a carve,
joining continuations as `rbf0_tree.py` does, encodes it, and compares the result with the original bytes. It exits 0
only when every resource is byte-identical: a resource that encodes to different bytes or is refused makes it exit 1,
as `rbf0_tree.py` does for a resource that is not `ok`.

The encoder fails closed: it refuses a tree that cannot hold the original bytes instead of writing other bytes. A
refusal is one of these: an incomplete parse, invalid UTF-8 that the decoder replaced, a NaN float (infinities are
exact), an attribute after content, a descriptor out of order, a string or text whose byte length changed, or a
value the record cannot hold: an integer out of range for the layout, a float that overflows it, an unknown type, or
an element with more attribute records than its attribute count. An element short of its attribute count with a
value in its content is refused as an attribute after content, because that value would decode as an attribute.

Measured on the same 1,263 `data.rpf` resources:

| Result | Resources |
|---|---:|
| Byte-identical after decode and encode | **1,240** |
| Encoded to different bytes | 0 |
| Refused: invalid UTF-8 | 23 |
| Wide layout, exact | 1,237 of 1,260 |
| Narrow layout, exact | 3 of 3 |

The 23 refused resources are exactly those that hold the 26 invalid UTF-8 text payloads in 9.5. No other reason
occurs, and with them `--check` exits 1 on this corpus. So the tree loses no byte anywhere else: element words, attribute counts, descriptor order, value widths,
float bits and NUL terminators all survive. The encoder also writes an edited tree, for example a JSON report
changed by hand. It then checks each string and text against the recorded byte length when one is present, so a
change that alters a length needs that field dropped or updated. Report-to-folder export also exits 1 if any resource is refused,
even when it writes the valid resources; on the measured corpus it writes 1,240 resources and refuses 23.
An export with no refusals exits 0.

## 10. Reproduction

```powershell
# Index the archive: CSV or JSON, root/counts/strings; no payload is written
python -I tools/rbf0_index.py <package>\data.rpf <work>\rbf0.json
python -I tools/rbf0_index.py <package>\data.rpf <work>\rbf0.csv --quiet

# Index a directory of already-carved streams (offsets from leading hex file names)
python -I tools/rbf0_index.py <work>\data-carve <work>\rbf0.json

# Decode trees from the same carve; JSON plus optional CSV node table and per-resource XML
python -I tools/rbf0_tree.py <work>\data-carve <work>\rbf0-tree.json
python -I tools/rbf0_tree.py <work>\data-carve <work>\rbf0-tree.json --csv <work>\rbf0-nodes.csv
python -I tools/rbf0_tree.py <work>\data-carve <work>\rbf0-tree.json --xml <work>\rbf0-xml
python -I tools/rbf0_tree.py <work>\data-carve <work>\subtitle.json --root CMovieSubtitleContainer --limit 1

# Round trip: decode and re-encode every resource and compare with the original bytes (counts only)
python -I tools/rbf0_encode.py --check <work>\data-carve
# Write the resources of a (possibly edited) tree report back as .rbf files into a new folder
python -I tools/rbf0_encode.py <work>\rbf0-tree.json <work>\rbf0-encoded

# Unit tests: synthetic headers and trees only
python -I tests/test_rbf0_index.py
python -I tests/test_rbf0_tree.py
python -I tests/test_rbf0_encode.py
python -I tests/test_imhex_patterns.py
```

To look at one resource in a hex editor, open the inflated and joined resource (not the archive) in
[ImHex](https://imhex.werwolv.net/) with [tools/imhex/rbf0.hexpat](../tools/imhex/rbf0.hexpat). The pattern follows
the grammar in section 3, picks the 8- or 4-byte value width from the prologue, and stops where the root element
closes. `tests/test_imhex_patterns.py` builds wide and narrow streams with every record type, checks them with
`rbf0_index.parse_rbf0`, and when ImHex is installed (on `PATH`, in its default Windows folder, or named by
`LEONIDA_IMHEX`) compares the pattern's output with the parser's.

`--start` must be 0 or more and `--limit` at least 1 (decimal or `0x` hex); an out-of-range value stops with a
one-line error and exit code 2 before an index is written.

The JSON summary carries the corpus totals used above (`streams`, `compressed_bytes`, `inflated_bytes`,
`rbf0.resources`, `rbf0.multi_chunk`, `rbf0.by_root`, `rbf0.size_buckets`, record counts, string categories and
keyword occurrences), so the numbers can be re-measured without committing an index. `--all-strings` writes the
full readable string list to JSON; that file is game data and must stay out of git. The tree summary adds the
structural counts `elements_by_depth`, `children_histogram`, `content_length_histogram`,
`attribute_count_histogram`, `leaf_elements`, `empty_elements`, `text_only_elements`, `parent_child_edges`,
`max_children` and `mean_children`, so the tree shape can be re-measured without keeping the trees. The tree JSON,
CSV and XML outputs contain names and payloads and are game data too.

## 11. No-sharing rule

Game files, carved streams, inflated RBF0 blobs, readable string lists, parsed trees and reports that quote them
stay on the machine that owns the package. Do not commit them, attach them to an issue or pull request, or paste
them into a chat. Short header excerpts (a few bytes) and aggregate counts are allowed, as in `CONTRIBUTING.md`.
This document contains no archive data, no asset path and no string dump.
