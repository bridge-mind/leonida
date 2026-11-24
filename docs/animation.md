# Animation archive (`anim.rpf`)

Measured state of the animation archive in the Xbox Series 1.0.2.1 package. Everything here was read from a
local copy; no archive, payload, name list or byte dump is committed. The tables are counts and short header
excerpts only.

The table of contents of an RPF9 is encrypted. This work does not read or decrypt it. It indexes the data area
the same way `tools/rpf9_carve.py` does: every zlib stream that inflates cleanly to its end is one **stream** in
the tables below. A logical RAGE resource can span several streams, so "stream" is not "file".

## Measured archive composition

```
archive             4,407,626,384 bytes
RPF9 magic          b"9FPR"
data start          0x4660 (18,016)
data span           4,407,608,368 bytes
```

The requested command (`python -I tools/rpf9_carve.py <package>/anim.rpf <work>/anim-index --index-only`) prints:

| Measure | rpf9_carve |
|---|---:|
| validated zlib streams | 130,449 |
| stored (compressed) bytes | 4,204,677,247 (3.92 GiB) |
| inflated bytes | 11,592,111,696 (10.80 GiB) |
| zero padding | 180,165,556 (4.09% of span) |
| unread bytes | 23,664,707 (0.54% of span) |
| readable | 95.40% of span |
| Bink 2 | 0 streams |
| elapsed | 141 s |

`rpf9_carve.py` overreads its last stream: it advances a 1 MiB read cursor by 1 MiB even when the final mmap
slice is shorter. That makes the last `PSIN` stream look 974,742 bytes stored instead of 1,766 and skips 21
streams at the tail. `tools/anim_index.py` advances by the actual slice length. Its corrected walk finds:

| Measure | corrected walk |
|---|---:|
| streams | 130,470 |
| stored bytes | 4,203,762,662 (3.915 GiB) |
| inflated bytes | 11,593,460,132 (10.80 GiB) |
| zero padding | 180,167,209 (4.09%) |
| unread bytes | 23,678,497 (0.54%) |
| readable | 95.38% |
| elapsed | ~140 s |

The 21 extra streams are the 1.35 MB `RBF0` rig metadata at the end of the data area (below). The class table
uses the corrected counts: the `PSIN` group is 213,747 stored in total, and the tail is its own row.

| Class (first bytes) | Streams | Stored | Inflated | Median | Max | Where |
|---|---:|---:|---:|---:|---:|---|
| `0DRP` at +16 (page directory) | 22,823 | 169,144,668 (161 MiB) | 3,883,044,624 (3.62 GiB) | 196,608 | 5,636,096 | first 1/8 of data area |
| starts with 8 zero bytes | 78,688 | 3,569,562,146 (3.32 GiB) | 6,918,674,880 (6.44 GiB) | 46,864 | 20,578,304 | throughout |
| other (no match below) | 28,646 | 464,618,867 (443 MiB) | 789,033,180 (752 MiB) | 23,984 | 65,536 | throughout |
| `RBF0` rig blob (1 header + 20 blocks) | 21 | 58,391 (57 KiB) | 1,348,436 (1.29 MiB) | 65,536 | 65,536 | tail |
| `pack:/` name table at 0 | 273 | 164,654 (161 KiB) | 711,536 (695 KiB) | 896 | 61,056 | throughout |
| `PSIN` at 0 | 18 | 213,747 (209 KiB) | 646,580 (631 KiB) | 35,768 | 65,536 | scattered |
| `SKEL_` bone list at 0 | 1 | 189 | 896 | 896 | 896 | early |

Only the first `RBF0` stream has the header; the other 20 are continuation blocks, so `anim_index.py` reports one
`rbf0` and 20 `raw:<hex>` records for that tail.

Two observations matter for a decoder:

- Every `0DRP` stream is in the first 1/8 of the data area (about 551 MB of span). The remaining 7/8 contains no
  `0DRP` header at +16.
- The 78,688 zero-prefixed streams are not opaque blobs. In a 200-stream sample, 78% have a 64-bit pointer into
  the `0x50000000` range at +8; the rest are mostly zero or other tables. Some pages carry a name table and
  high-entropy blocks later in the same stream.

## Container signatures actually observed

### `0DRP` page marker at +16

22,823 streams. The fixed part of the header is:

```
+0x00  8 zero bytes
+0x08  u64 pointer into 0x50xxxxxx
+0x10  "0DRP"                  (bytes 30 44 52 50)
+0x14  u32 122                 (same value in all 22,823)
+0x20  u64 page count          (observed 1..41, 32 distinct values)
+0x28  four u64 pointers       (0x50xxxxxx)
+0x48  four u16 values
```

Top page-count values: 1 (1,883), 2 (7,571), 3 (9,241), 4 (2,022), 19–26 (1,592). The largest sampled page
(5,636,096 bytes) is 94.6% zero bytes, with sparse pointers. That is a page directory/relocation table, not
keyframe data. The marker is reported as a page marker, not as a known GTA V resource magic. Each `0DRP` stream is
the first system page of a resource; the page map at the `+0x08` pointer gives its page count (see
[Keyframe decoding](#keyframe-decoding)).

### `MoVE` container

324 resources start with `"MoVE"` (325 zlib streams: 323 fit in a single RPF9 stream of 114–58,452 inflated bytes, and
one at `0x105a7d000` spans a 65,536-byte stream plus a 5,563-byte continuation stream at `0x105a7f9c0`, for 71,099
bytes). All 324 share the same 32-byte fixed header prefix:

```
+0x00  "MoVE"                  (bytes 4D 6F 56 45)
+0x04  u32 19                  (version 19 in all 324)
+0x08  three u32 0             (reserved zeros in all 324)
+0x14  u32 network_hash        (324 distinct values; not a size)
+0x18  u32 body_size           (68..71,048; equals total_bytes - 46 - 5 * N_desc in all 324)
+0x1c  u32 0                   (reserved zero in all 324)
```

What earlier looked like a fixed 46-byte header with "5/10/15/20 bytes of padding" on 140 streams is a **variable
preamble at `+0x20`** containing a table of `N_desc` 5-byte `{u32 hash; u8 kind}` descriptors that `body_size` does not
count (`total_bytes == 46 + 5 * N_desc + body_size` across all 324 resources, with zero tail padding). The full
preamble, state-machine hierarchy (`Type 1` = `StateMachine`, `Type 3` = `InlinedStateMachine`, `Type 22` =
`MotionTree`), transitions and parameter blocks are decoded in [`MoVE` state-machine containers](#move-state-machine-containers)
below (`tools/anim_move.py`, ASSET-86).

### Name tables (`pack:/...`)

273 streams start directly with NUL-terminated ASCII paths; there is no container header before the names. They
hold 711,536 inflated bytes and 2,298 distinct paths:

| Suffix | Distinct paths |
|---|---:|
| `.clip` | 2,012 |
| `.anim` | 274 |
| `.motion` | 12 |

Paths are padded to 16-byte boundaries. The records include a 32-bit name hash; a second 32-bit value recurs as
`0x98e1750c` in some tables. It is `joaat32("bonerotation")`, the track-type hash of rotation tracks (see
[Keyframe decoding](#keyframe-decoding)); these tables are second system pages of clip dictionaries. The name hash
is **joaat32 of the path stem**:
strip `pack:/`, strip the final extension, lowercase. In one 161-name table all 161 hashes matched exactly;
across the 273 tables, 3,229 of 3,637 name occurrences (88.8%) and 248 of 273 tables matched every name. The
remaining 11.2% are not explained (duplicate names, missing records or a different record layout). A separate scan of the first 4 KiB of the 130,449 `rpf9_carve` streams found 7,661 distinct
`pack:/...` paths and 31,020
occurrences; 5,880 of those streams contain at least one path, bone name or animation word in that prefix. The
union of the two name sources is 8,020 distinct paths. Names also appear embedded at non-zero offsets in
zero-prefixed pages (for example at +0x497e0 and +0x4aa8 in one 327,680-byte page), not only in the 273
start-of-stream tables.

### Skeleton and bone names

A full-corpus prefix scan found 31 distinct `SKEL_*` tokens, 170 occurrences, in the first 4 KiB of the streams;
the count includes fragments. The four full-name streams give the 30-name union below. Two forms were seen in
carved samples:

- a plain NUL-terminated bone list (896-byte stream), and
- a front-coded table (2,064 bytes) whose entries are length-prefixed name fragments plus parent indices.

The `SKEL_<region>_<part>` shape is the conventional RAGE bone naming. The animation pages carry no model name next
to the bones, but a skeleton carries its own names (`joaat32` at bone record `+0x1C`, name at `+0x30`,
[character_skeleton.md](character_skeleton.md)), so the match is measured: see
[Bone tables and motion matchers](#bone-tables-and-motion-matchers).
Measured across all `SKEL_*` streams in
`anim.rpf` (`177` streams, `118` distinct `SKEL_*` tokens) against the decoded skeleton name tables in
`packs.rpf` (`2,866` `SKEL_`-bearing streams, `691` distinct `SKEL_*` tokens) using `tools/anim_skeleton_match.py`:
`117` of `118` (`99.2%`) match by exact case and all `118` of `118` (`100%`) match case-insensitively (and by
lowercased `joaat32`, since `SKEL_HEAD` corresponds to `SKEL_Head`); `166` of `177` streams match all of their
tokens by exact case. The animation pages still carry no model name next to the bones.

### Type-hash containers

1,371 streams share this 16-byte header shape:

```
+0x00  u32 type id
+0x04  u32 8
+0x08  u32 chunk count (1..4096)
+0x0c  u32 0
```

The most frequent type ids are `0x88560988` (721 streams), `0x51e5606c` (220), `0x51404a31` (133),
`0x8db79d92` (119) and `0x98a13c55` (7). Their payload is chunk-like, entropy 2.57–4.26 bits/byte. `PSIN` is a
separate 18-stream group (`PSIN` at offset 0, 0.21 MB stored / 0.65 MB inflated); `docs/design.md` already
describes `PSIN` as PSO metadata, and these bytes do not show that it is animation data.

### Packed clip streams

1,781 streams start with:

```
02 00 00 00 01 01 00 00 ...     or     03 00 00 00 01 01 00 00 ...
```

The historical `clip-stream` scan label selects **second system pages starting at the parent's auxiliary
table** (ASSET-158). The leading u32 counts tagged entries. The 1,781 pages occupy 18,168,950 bytes stored
and 32,691,776 bytes inflated, with 1,767 distinct page contents.

To reproduce this on 1.0.2.1, walk `anim.rpf` with `anim_index.scan_rpf`. For every `clip-stream` row,
inflate the preceding `page-directory` row and the selected row. Check that the second compressed stream starts
at the next 16-byte boundary after the first. In **1,781/1,781** pairs the u64 at first-page `+0x40`, minus
`0x50000000`, equals the inflated first-page length. At that offset, read a u32 count, then that many
`{u32 tag, u32 payloadBytes}` entries, then the payloads in order.

| Entries | Pages | Tags in order |
|---:|---:|---|
| 2 | 528 | `0x101`, `0x207` |
| 3 | 1,235 | `0x101`, `0x113`, `0x207` |
| 3 | 10 | `0x101`, `0x204`, `0x207` |
| 3 | 8 | `0x101`, `0x10F`, `0x207` |

In every pair the declared payload lengths fit within page two. The instance blocks referenced by the
[`+0x38` instance block table](navmesh.md#schema-tables-q28-q30-q38-q40) fit wholly within page one. Schema tables also occupy page two after the auxiliary payloads,
and none overlaps the auxiliary table or its payloads. The same framing is measured independently for
[navigation resources](navmesh.md#schema-tables-q28-q30-q38-q40). The meaning of the individual payloads
in the animation archive remains open.
The leading u32 is 2 or 3; the next four bytes are constant. Stored 18,168,950 bytes, inflated 32,691,776 bytes.
1,767 distinct payloads (duplicates up to 10 copies). Entropy 4.03–7.79 bits/byte, so these are packed/quantized
bitstreams rather than zlib streams. The 24-byte header has five small fields after the type; the measured
version split and the track side are below. No keyframe values are claimed. Each of these streams is the second
system page of a `0DRP` resource, the one right after its `0DRP` stream (see [Keyframe
decoding](#keyframe-decoding)).

### Partial clip decode

Two named clips were followed from their `pack:/...` name-table records to page-0 payloads. The name record
stores `joaat32(stem)`, a second hash, and two 64-bit `0x50xxxxxx` pointers; the pointer rule is page-0 offset
when it is below the page-0 size, otherwise a name-table page offset.

| Clip | Name hash | Page-0 stream | Clip record | Duration | Tracks |
|---|---|---:|---:|---:|---:|
| `breakout_quick_front` | `0xDC5BEF77` | `0x1D284010` | page0 `0x1F8A0` | 3.2333 s | 10 |
| `idle` | `0x71C21326` | `0x9F0E0010` | page0 `0xF580` | 6.0000 s | 8 |

The durations are `f32` at record `+0x78` (breakout) and `+0x68` (idle). Track arrays are `0x48`-byte records
with `type = 0xFFFF0001` and two hashes per record.

Bone masks:

- Breakout: a 63-entry body mask and a 38-entry finger mask.
- Idle: a 12-entry head mask and the same 38-entry finger mask.
- The 38-entry mask is byte-identical between the clips and **all 38 hashes resolve** to
  `SKEL_{L,R}_Finger{00..02,10..13,20..23,30..33,40..43}`.
- The 63-entry mask maps 25 body/`PH_` names; its other 38 entries are unresolved (likely facial or accessory
  bones, not assumed).

The 2,064-byte motionmatcher table also re-parses cleanly: 15 names, parent entries, and all 8 record hashes
resolve.

Header fields measured over the 1,781 streams:

- `+0x04` is always `0x00000101`.
- Version 3 (1,253 streams): `+0x0C = 0x0113` on 1,235, `+0x14 = 0x0207` on all 1,253.
- Version 2 (528 streams): `+0x0C = 0x0207` on all 528; `+0x14` is a 32-bit hash.
- `+0x08` is the leading packed-region length; the first 16-byte `(hash, 0, index, 0)` table follows a short
  sub-header (for example at `0x2C8`, `0x4C8`, `0x6C8` in one page).

**Still missing:** the page → packed-stream join. None of the 1,781 streams contains any of the 30 known bone
hashes or the nine tested clip-name hashes, as 32-bit or 64-bit values; only the shared track-type hash
`0x9A7B6E6D` crosses over. The track records and quantization-like blocks are visible, but the exact component
count/order/delta coding did not survive a unit-quaternion norm test. No keyframe value is claimed for these
`0DRP` pages. The clip dictionaries' keyframes are decoded in [Keyframe decoding](#keyframe-decoding).

### Tail `RBF0` rig-set manager

The last 21 streams appear to be one logical `RBF0` blob split into 64 KiB blocks (20 × 65,536 plus 37,716
bytes); the first block ends mid-path (`...anim/rig`) and the next starts `/canonical/...`:

```
52 42 46 30 fb ff 01 00 00 00 15 00  "rage__crRigSetManager"
```

They hold 58,391 stored / 1,348,436 inflated bytes. The blob is a rig-set manager, not clip keyframes. Parsing
the `03 60` key strings, `fd ff` value strings and `ff ff 06 00`/`ff ff 07 00` field markers gives:

| Measure | Value |
|---|---:|
| Rig-set keys | 1,126 |
| Source → rig entries | 6,677 |
| Distinct rig paths | 3,669 |
| Distinct names (rig sets + rigs) | 4,795 |
| Empty rig sets | 56 (object 53, vehicle 3) |
| Rig-set categories | ped 664, vehicle 346, object 115, system 1 |
| Max entries / distinct rigs in one set | 45 / 41 |
| Entries with a group string | 6,677 (16 distinct groups) |

Source entries, with distinct rig paths in each source:

| Source | Entries | Distinct rigs |
|---|---:|---:|
| `generated@source` | 3,616 | 1,188 |
| `pipelined@source` | 2,135 | 1,931 |
| `canonical@source` | 888 | 534 |
| `canonical@legacy` | 38 | 16 |

Every distinct rig appears under exactly one source category. Every source tag is followed by exactly one rig
key before the next source (0 exceptions), and every entry is assigned to a rig-set by its offset.

The first rig-set block starts at 0x0045. The tag/length layout is visible directly:

```
0x0045  03 60 03 00 "key"
0x004c  1f 00 <31-byte rig-set path>
0x009a  fd ff 11 00 00 00 "canonical@source" 00
0x00b1  ff ff 06 00 03 00 "Rig" ...
0x00c0  fd ff 30 00 00 00 <48-byte rig path> 00
0x00f6  ff ff 07 00 05 00 "Group" ...
0x0107  fd ff 05 00 00 00 "pose" 00
0x0112  ff ff 08 10 08 00 "Priority" ...  a0 0f 00 00
```

- Rig-set key: `03 60` + u16 length + ASCII (no NUL). The first block is special and puts the schema name `key`
  before the first path; the following 1,125 blocks use `02 00 00 00 00 00 01 00 03 60` + u16 length + path.
- Entry: `fd ff` + u32 length including NUL + source, then `ff ff 06 00` (Rig), the rig key in the same
  `fd ff` form, `ff ff 07 00` (Group), and the group string in the same form. A `ff ff 08 10` field and a mostly
  zero u32 follow; the field semantics are not mapped.
- A non-empty rig-set ends with six `ff` bytes (1,070 occurrences). An empty rig-set has
  `04 00` + six zero bytes + four `ff` bytes and no entry (56 occurrences).
- The group names are the rig's channel/group field; the observed values are `deformation` (2,169),
  `deformation[2]` (1,702), `identity` (1,109), `pose` (938), `pose.ragdoll.post` (160), `facial` (152), and
  ten smaller values.

The grammar-driven tree decoder (`tools/rbf0_tree.py`; `docs/rbf0.md` section 9) parses the same 1,348,436-byte
blob to its last byte and confirms the counts above: 9 descriptors, 28,962 elements, 20,031 text records, 1,126
string values, 6,677 integer values and maximum element depth 6. The 1,126 string values are the `key` attributes
on the rig-set `Item` elements. Each of the 6,677 entry `Item` elements contains `Dictionary`, `Rig` and `Group`
text children followed by a `Priority` integer. The record framing is consumed exactly; the numeric field
semantics remain unresolved. The `Dictionary` text names one of the four rig dictionaries and `Rig` is a key
there; see [Rig dictionaries](#rig-dictionaries).

Mapping to the name tables:

- The 2,298 distinct `pack:/...` paths resolve to a stored joaat32 stem hash for **1,901** names (82.7%);
  **401** stay unresolved. By occurrence, 3,229 of 3,637 (88.8%) match, and 248 of 273 name tables are complete.
- The RBF0 names do **not** overlap the clip/anim/motion namespace: direct name overlap is 0 and stem-hash
  overlap with the 2,298 paths is 0.
- The RBF0 table contains 4,795 distinct rig and rig-set names. An earlier version of the indexer reported
  zero stored joaat32 matches for every input. The count for this corpus needs to be measured again with the
  corrected `--rigs` report. A match indicates that the four hash bytes occur somewhere in the blob.

### Bone tables and motion matchers

Four small resources in this index carry full bone names. One is a standalone name list; three are motion-matcher
resources found in the prefix scan that embed a bone table and a `pack:/...motionmatcher` path:

| Offset | Inflated | Distinct names | Occurrences | joaat32 present | Kind |
|---|---:|---:|---:|---:|---|
| `0x101a32160` | 896 | 28 | 68 | 0 | standalone name list |
| `0x101c379a0` | 4,272 | 28 | 55 | 14 | motion matcher |
| `0x101c41d50` | 2,064 | 15 | 15 | 8 | motion matcher |
| `0x101ea66c0` | 2,848 | 30 | 50 | 19 | motion matcher |

The three motion matcher streams contain four distinct `pack:/...motionmatcher` paths. The union of all four
streams is 30 names (28 `SKEL_*` and 2 `PH_*`); 19 of the 30 have a joaat32 present in at least one stream; 11
do not. The standalone 896-byte stream has no hash table.

The 2,064-byte stream shows the embedded record layout most clearly: header `0f 00 e0 00 0c 00 16 00`, eight
0x30-byte records from 0x0528 to 0x0678, a u32 hash at +0x10, a parent-index array at 0x06d0, bind-pose floats
from 0x0100, the full name list at 0x0750, and a `pack:/...motionmatcher` path at 0x04a0:

```
0x0528  0c 00 01 00 00 00 00 00 00 00 00 00 19 de 33 81 0b 00 00 00
0x0558  0c 00 00 00 06 00 01 00 00 00 00 00 00 00 00 00 55 a9 3f 2b 03 00 00 00
```

All 8 of its +0x10 hashes match joaat32 of full names in its own 0x0750 list. The 2,848-byte stream has a
14-entry pointer table at 0x0000 (targets in the `0x5005xxxx` range), records at 0x02d0, and two motionmatcher
paths at 0x0210 and 0x0290; 19 of its 30 names hash-confirm. The 4,272-byte stream has 28 names and 14 hashes.

**Which skeleton (ASSET-231).** `tools/anim_skel_match.py` reads every skeleton `anim_glb.read_skeleton` reads in
`packs.rpf` (14,212, 199 distinct joint-hash lists) and looks each name up by `joaat32` of its lowercased spelling:

| Stream | Names | Joints of some skeleton | Same spelling as the joint | Skeletons holding all of them |
|---|---:|---:|---:|---|
| `0x101a32160` | 28 | 28 | 28 | 2 (the 596- and 560-bone humanoids) |
| `0x101c379a0` | 28 | 28 | 28 | 2 (596, 560) |
| `0x101c41d50` | 15 | 15 | 15 | 69 distinct, 124 copies (78 to 596 joints) |
| `0x101ea66c0` | 30 | 30 | 30 | 2 (596, 560) |
| union | 30 | 30 | 30 | 2 (596, 560) |

So every bone name in these pages is a joint of the female (596) and male (560) humanoid skeletons that the clip
dictionaries already bind to ([character_skeleton.md](character_skeleton.md)), under the same name; no other
skeleton holds the 30. The 15-name table is a subset many smaller rigs share too.

This is a measured **clip → bone-table binding for the four motionmatcher paths**: the path and the bone table
are in the same stream. It is not a rig → skeleton link for RBF0:

- RBF0 contains no `SKEL_` token and no lowercase `bone`/`skeleton` string; the only `Bones`/`Skel` text is
  inside two rig path names.
- 0 of the 3,669 rigs and 0 of the 1,126 rig sets match any of the 30 bone names or any of the 19 hashes present.
- The four motionmatcher paths are not in the 2,298 `.clip`/`.anim`/`.motion` name-table paths.

So RBF0 gives rig-set → source → rig, and the motion matcher resources show how a clip can carry its own bone
table. The RBF0 rig itself resolves into a rig dictionary ([Rig dictionaries](#rig-dictionaries)); what it does not
give is a whole skeleton for the clips.

### Magics not found

At stream starts across the 130,449 `rpf9_carve` streams, the counts of `YCD`, `YDD`, `YLD`, `YFD`, `RSC7`,
`RSC8`, `ADAT`, `AWC`, `RBF0`, `FSSM` and `KB2` are all zero. The corrected walk adds 21 tail streams whose first
starts with `RBF0`; the rest start inside the same blob. One three-byte `YDD` sequence occurs at +0xE1 inside a
high-entropy stream; it is not at a header boundary and is treated as coincidence, not as a YDD file.

## Keyframe decoding

`tools/anim_clips.py` decodes the clip dictionaries down to per-frame track values. It starts from the same stream
walk as `anim_index.py`, so the stream numbers below are the corrected walk's 130,470. Run on the 1.0.2.1 copy:

```powershell
python -I tools/anim_clips.py <package>\anim.rpf <work>\anim-clips.json --jobs 8
python -I tools/anim_clips.py <package>\anim.rpf --decode <dictionary offset> [--frames N]   # values to stdout
```

The JSON holds counts and one row per dictionary (offset, hash and counts, plus the name read from your copy).
It holds no values. `--decode` prints one dictionary with its decoded values and writes no file.

Provenance: channel types 0–8 follow the GTA V `crAnimation` channel set, as named by community tools such as
CodeWalker's YCD reader (static, quantized, indirect-quantized and cached-quaternion channels). No code was taken
from them. The resource framing, the GTA VI-only channel type 10, the type 3 and type 8 rules below and every
count in this section were measured on the local copy.

### Clip dictionaries are RAGE resources

A clip dictionary is a system-only RAGE resource. Its **root stream** is one of the `starts with 8 zero bytes`
streams above:

```
+0x00  8 zero bytes
+0x08  u64 page-map pointer (0x50xxxxxx)
+0x10  zero up to +0x38
```

At the page map, the byte at +8 is the number of system pages and the byte at +9 the number of graphics pages.
Every dictionary has 1 or 2 system pages and no graphics pages. Its pages are that many back-to-back streams,
each 16-byte aligned after the one before it. Join them, and a pointer `p` is the offset `p - 0x50000000` into
the joined bytes.

A continuation can begin with an animation pack whose header also looks like a dictionary root.
When that header has a verifiable page-local name and hash, the grouper requires the preceding pages
to contain the owner's complete pack table and an entry matching both the boundary pointer and the
hash. A table on the candidate or any later page could belong to the next dictionary. Such a layout
leaves the owner unresolved, even for a legitimate pack. This check applies only to recognized named
boundaries. A malformed reference already present in the prefix can still match.

Measured: 61,510 root streams. 61,510 join into resources: 35,750 with one system page and 25,760 with two.
In 36 roots the page map says two pages, but the next listed stream does not start where the first one ends. Their
second page is right after the first stream, but `anim_index.py` does not list it: in 28 it is a zlib stream that
inflates to 16–48 bytes (`anim_index.py` lists streams of 64 bytes or more), and in 8 it is a 16-byte raw page,
like rpf9-explorer's raw system tail. `anim_clips.py` joins such a last page only when the joined pages then parse
as a clip dictionary; all 36 do, so no root is left unresolved. That recovery runs only after the page map itself
has been read from listed streams: a root whose page map is in such an unlisted last page stays unresolved ("page
map not in the next stream"). No anim.rpf root has that layout. In `story000.rpf` 5 roots do, with the page map in
a 32- or 48-byte zlib tail, and none of the 5 is a clip dictionary once that tail is joined. Of the joined resources, 61,505 are clip
dictionaries. 4 have `RIG` plus a version at +0x70 instead of a pack array (rig dictionaries, [decoded
below](#rig-dictionaries) by `tools/anim_rigs.py`), and
1 has some other value there. Together the dictionaries use 87,234 listed streams plus the 36 unlisted pages
(3,802,124,713 stored / 7,077,351,104 inflated bytes in the listed streams). By `anim_index.py` class those
streams are: zero-prefix 66,115, raw 20,845, name-table 273, skeleton-names 1.

The other classes fit the same framing:

- **The 273 `pack:/` name tables** at the start of a stream are the second system page of a clip dictionary. All
  273 sit inside dictionaries.
- **The `0DRP` (version 122) streams** are a different resource kind with the same page map. Their system pages are
  one stream in 2,892 resources and two in 19,931. In 19,909 of the two-page ones the second page is the next stream;
  in the other 22 it is not. Their second pages include all 1,781 "2/3 packed clip streams" and all 1,371 type-hash
  streams. So those classes are not stand-alone clip payloads: each one is the page right after a `0DRP` stream in
  the same resource. The `0DRP` root objects are not decoded here. The 324 `MoVE` resources are stand-alone flat binary
  containers rather than `0DRP` or clip-dictionary pages; see [`MoVE` state-machine containers](#move-state-machine-containers).

### Objects

All offsets are page offsets. An `atString` is 16 bytes: `{char *ptr; u16 length; u16 capacity}`, or, when byte 15
has bit 7 set, up to 15 characters stored inline with `byte 15 & 0x7F` as the length (909 clip names and 3 dictionary
names). An array is `{T *ptr; u16 count; u16 capacity}`, and an `atMap` is `{Entry **buckets; u32 bucket count (11);
u32; u32 count}` with `Entry {u64 key; T *value; Entry *next}`. Hashes are `joaat32` of the lowercased name.

| Object | Field | Meaning |
|---|---|---|
| dictionary (page offset 0) | `+0x38` atString | dictionary name |
| | `+0x48` atMap | clips; key = `joaat32(dictionary name) << 32 \| clip hash` |
| | `+0x68` u32 | `joaat32(dictionary name)` (all 61,505 dictionaries) |
| | `+0x70` array, 16-byte items | `{pack *ptr; u32 hash; u32 0}`. With a pointer, a local pack and its `joaat32(name)`. With a null pointer, the hash is another clip dictionary's `joaat32(name)` (see `.motion` clips below) |
| animation pack | `+0x38` atString | pack name (`<dictionary>-N`) |
| | `+0x48` atMap | animations, key = `joaat32` of the animation name |
| | `+0x68` u32 | `joaat32(pack name)` |
| `.clip` (`+0x20` u32 < 16) | `+0x28` atString | `pack:/<name>.clip` |
| | `+0x68` pointer | list `{entry *ptr; u16 count; u16 capacity}`, entries 0x38 bytes: `+0` animation pointer, `+8` animation hash, `+0x28`/`+0x2C` start/end (s), `+0x30` rate |
| | `+0x78` f32 | duration (s) |
| `.motion` (`+0x20` is a pointer) | `+0x18` pointer | the dictionary root (page offset 0) |
| | `+0x28` u32, `+0x2C` u32 | the clip hash (the low half of its `atMap` key) and `joaat32` of the dictionary name |
| | `+0x30` atString | `pack:/<name>.motion` |
| | `+0x58` u32 | 1 or 2; selects where the parameter node is (see `+0x120`, `+0x138`); most likely the number of blend parameters (not verified) |
| | `+0x60` array, 16-byte items | children `{u64 0; u32 clip hash; u32 dictionary hash}`: the clips it blends |
| | `+0x70` array, 0x28-byte items | one item per child: `{f32 child duration; f32 duration in the blend; {ptr; u16 count} curve of (child time, blend time) f32 pairs; 16 zero bytes}` |
| | `+0x120` pointer (`+0x58` = 2) | a block whose `+0x18` points at the parameter node |
| | `+0x138` pointer (`+0x58` = 1) | the parameter node: `{u64 0; u32 type; u32 0; ...}` |
| parameter node, type 3 | `+0x10` array of u32, `+0x20` array of 8-byte items, `+0x30`/`+0x34` f32 | one sampled axis: its name hash(es), then `{u32 sample index; f32 value}` sorted by value, then min and max |
| `.motionmatcher` (`+0x20` is a pointer) | `+0x20` pointer | the feature object (below) |
| | `+0x28` 16 bytes, `+0x38` atString | an id, then `pack:/<name>.motionmatcher` |
| | `+0x58` pointer, u32 count at `+0x54` | records `{f32; f32; f32 −1.0; u32 clip index; u64 0}`, 24 bytes. A `.motionmatcher` array keeps its u32 count just before the pointer, and a u32 capacity before that |
| | `+0x68` pointer, count at `+0x64` | u32 offsets, one more than the records, from 0 and never decreasing |
| | `+0x78` pointer, count at `+0x74` | u16, one per clip and then the record count: each clip's first record |
| | `+0xB8` pointer, count at `+0xB4` | u8, 0 or 1, one per clip |
| | `+0x98` pointer, count at `+0x94` | on 36 of 180: one `{u64 0; u32 clip hash; u32 dictionary hash}` per clip, like `.motion` children |
| | `+0xA0` u32 groups, `+0xA4` u32 entries, `+0xA8` pointer | a hash table of the clips: per group 16 control bytes (0x80 empty, below 0x80 full), then 16 slots `{u32 clip hash; u32 dictionary key; u32 clip index; u32 0}` |
| | `+0x170`/`+0x174`/`+0x178` | the same kind of table with 8-byte slots `{u32 hash; u32 index}` |
| | `+0x188` u32, `+0x18C` u32 | the clip's own hash and the dictionary's `joaat32(name)`, as `.motion` has at `+0x28`/`+0x2C` |
| | `+0x48` pointer, count at `+0x44` | one 40-byte item, zero except a u32 1 at its `+0x20` (`+0x40` holds the name's length and capacity, measured separately) |
| | `+0x118` pointer, count at `+0x114` | one u32: the pose sample count |
| | `+0x128` pointer, count at `+0x124` | one u16, 0 |
| matcher feature object | `+0x18` u32, `+0x28` pointer, `+0x30` pointer or null | 1 (measured separately), then the skeleton and the `+0x30` object (both below) |
| | `+0x40` pointer, count at `+0x3C` | pointers to the features, 0x30 bytes each: `{+0x08 u32 column; +0x0C u16 width; +0x0E u16 (1 or 5); +0x18 u32 name hash (a bone with 1); +0x1C u32 kind bits}` |
| | `+0x48` u32 groups, `+0x4C` u32 entries, `+0x50` pointer | a hash table like the clip table, 8-byte slots `{u32 hash; u32 feature index}` |
| | `+0xF0` pointer, count at `+0xEC` | the pose matrix, f16, in 32-byte chunks: one row of feature values per sample (layouts below) |
| | with `+0x30`: `+0xF8` u32, `+0xFC` u32, `+0x100` u32 | samples, columns, 32-byte chunks per row (rows are padded to 16 columns) |
| | with `+0x30`: `+0xA8` pointer, count at `+0xA4` | one 48-byte `{f32 quaternion[4]; f32 vec3; f32 0; f32 1, 1, 1; f32 0}` per sample |
| | without `+0x30`: `+0xFC` u32, `+0x128` u32 | columns, samples (the matrix is in blocks of 8 samples) |
| | `+0x98` pointer, count at `+0x94` | one u32 per sample: a bit mask over the matcher's key table indices, the same for every sample of a record |
| | `+0xB8` pointer, count at `+0xB4` (32-byte chunks) | one f32 weight per pose column, 8 to a chunk, zero after the last column |
| | `+0xC0` f32, `+0xC4` f32, `+0xC8` u32 | not decoded, then 1.0, then 1 with rows and 3 with blocks of 8 |
| matcher `+0x30` object (112 bytes, measured separately) | `+0x08` u32 | its form: 1 or 2 |
| | form 2: `+0x10`/`+0x14` u32, `+0x18`/`+0x1C` f32 | two bones (`joaat32` of their names) and their weights |
| | form 2: `+0x30` 0x30 bytes, `+0x68` u32, `+0x6C` u16 | a copy of the clip's first feature (its kind at `+0x4C`), that feature's width, a flag (6 or 9) |
| | form 1: `+0x10` f32[4], `+0x60` u32 | a unit quaternion (a reference rotation), a flag (6 or 7) |
| | form 1: `+0x40` {u32 n; u32 n; pointer}, `+0x50` the same | n u32 bones (`joaat32` of their names), then n f32 weights |
| matcher skeleton (feature object `+0x28`) | `+0x18` u32 slots, `+0x1C` u32 entries, `+0x20` pointer | tag table, 8-byte slots `{u32 tag; u32 bone}`, empty tag 0xFFFFFFFF |
| | `+0x28` pointer, `+0x6C` u16 count (`+0x6E` the same, measured separately) | the bones, 80 bytes each: `{f32 rotation quaternion[4]; f32 translation[3]; 4 bytes; f32 1.0[4]; char *name at +0x30; s16 parent at +0x46; u16 index at +0x48 and +0x4C; u16 bone id at +0x4A}` (the 1.0s and the two indices on 180 of 180, measured separately) |
| | `+0x30`, `+0x38` pointers | per bone a 4x4 f32 matrix: the inverse bind matrix, and the local matrix (row vectors, translation in row 3) |
| | `+0x40` pointer | the parents, one s16 per bone, −1 for the root |
| | `+0x48` pointer | a prefix tree of the bone names: `{u16 n; u16 size}`, nodes from `+4` `{u16 next sibling; u16 first child; s16 value; char text[]}` (offsets relative to the node), then n u16 bone indices at `+size` |
| animation | `+0x24` u16 | frame count |
| | `+0x28` f32 | duration = (frames − 1) / fps |
| | `+0x50` array | sequence pointers |
| | `+0x60` array, 12-byte items | tracks `{u32 bone hash; u32 track type; u32 format}` |

A clip plays its animations at the same time, each over its own `[start, end]` window at its rate: animation time
`t` is clip time `(t − start) / rate`, and `(end − start) / rate` equals the clip's `+0x78` duration: over all
61,505 dictionaries of `anim.rpf` (`tools/anim_clips.py`, `clip_windows`), 784,562 of the 785,174 animation entries of
ordinary clips match to float32 precision (relative 1e-5) and the other 612 within 2.2e-4 relative, 581 of them in
one-frame (0.033 s) clips where the rounding shows most; none has rate 0 (the first check, 300 sampled
dictionaries, had 4,471 of 4,471). In the samples, the animations of one clip covered different bones and could
have different frame rates. `tools/anim_glb.py` plays clips on a fragment skeleton and writes them as glTF
animations; see [character_skeleton.md](character_skeleton.md#animated-gltf-export).

**Track types** are `joaat32` of their names. This is the meaning of the recurring `0x98e1750c` noted above:

| Hash | Name | Tracks |
|---|---|---:|
| `0x98e1750c` | `bonerotation` | 7,673,334 |
| `0xf791a95f` | `bonetranslation` | 2,440,527 |
| `0x2deefadd` | `genericcontrol` | 1,454,457 |
| `0xf5c3a5b6` | `generictranslation` | 69,844 |
| `0x6ddef121` | `cameratranslation` | 69,645 |
| `0x364d3bdb` | `facialtranslation` | 63,394 |
| `0x7aaa9f60` | `bonescale` | 24,590 |
| `0xd9ad1a74` | `camerarotation` | 476 |
| 36 other hashes | not named | 882,732 |

A bone hash is `joaat32` of the lowercased bone name, for example `skel_pelvis`. In one checked rig, 64 of 66
track bones matched the skeleton's names; bone hash 0 is the whole-body (mover) track. **Formats:** 0 =
quaternion `(x, y, z, w)`, 2 = vec3, 4 = float (7,935,044, 2,907,303 and 1,824,430 tracks; format 3 is used by 12,222
tracks and is not mapped).

### Sequences

An animation's frames are split over one or more sequences. Consecutive sequences **share one frame**, so
`sum(sequence frames) = frames + sequences − 1` (true for 191,042 of 191,042 decoded animations). The repeated frame
decodes to the same pose on both sides: 814,011 of 814,880 quaternion boundaries are the same rotation (|q₁·q₂| >
0.9999, so q and −q count as equal), and 696,853 of 708,648 vec3/float boundaries agree within 1e-3. The cause of the
remaining 0.1% and 1.7% was not investigated.

A sequence is a 0x20-byte header and `length` bytes of data:

| Header | Type | Meaning |
|---|---|---|
| `+0x04` | u32 | data length (excluding the header) |
| `+0x0C` | u32 | frame-data offset in the data |
| `+0x16` | u16 | frames in this sequence |
| `+0x18` | u16 | bytes per frame |
| `+0x1C` | u16 | sum of the type-4 bit widths (true in 212,306 of 212,306 sequences) |
| `+0x1E` | u8 | 11 (the number of channel types) |

The data holds, in order:

1. **Channel definitions**, grouped by type 0 → 10. They end exactly at the frame-data offset in every decoded
   sequence.
2. **Frame data**: `frames × bytes per frame`. Each frame is an LSB-first bit field holding the type-3 values,
   then the type-4 values, then the type-5 indices, in channel order.
3. `u16 count[11]`, the number of channels of each type.
4. 11 lists of `u16 (track << 2 | component)`, one per type, each padded to a multiple of 4 entries. Item `i` of
   list `t` names the track and component of the `i`-th channel of type `t`.

### Channel types

| Type | Definition | Value | Channels |
|---|---|---|---:|
| 0 | 3 × f32 x, y, z | constant quaternion, w = +√(1 − x² − y² − z²) | 3,579,477 |
| 1 | 3 × f32 | constant vec3 | 2,533,788 |
| 2 | f32 | constant | 5,428,131 |
| 3 | none | 32 frame bits = an f32 per frame | 40 |
| 4 | u32 bits, f32 quantum, f32 offset | `bits` frame bits × quantum + offset | 7,600,074 |
| 5 | u32 frame bits, u32 value bits, u32 words, f32 quantum, f32 offset, then the table (`words` u32, value-bits entries) | frame bits index the table; entry × quantum + offset | 484,616 |
| 6 | — | not seen | 0 |
| 7 | none | rebuilds one quaternion component (below) | 4,942,792 |
| 8 | none | marks a quaternion with four stored components (below) | 227,655 |
| 9 | — | not seen | 0 |
| 10 | variable, see below | second-order delta coding (GTA VI) | 6,836,654 |

**Type 10** (new in this build, and the most common animated channel):

```
u32 word count (whole channel, header included) | u8 c1, c2, c3, 0 | f32 quantum | f32 offset | bits (LSB-first):
  nc = ceil(frames / 64) chunks:  nc × c1 bits   chunk payload positions (redundant, for random access)
                                  nc × c2 bits   chunk start values
  ceil((frames − 1) / 8) unary width codes: k zero bits then a one = bit width of the next 8 deltas
  ... padding ...
  last 8 × c3 u32 words: 8 lanes, interleaved (word w·8 + j belongs to lane j);
  lane j holds delta j of every 8-delta group, k bits each, low bits first.
decode, per chunk:  x = start; v = 0
                    for each of the next 63 frames: v += zigzag(delta); x += v
                    value = x × quantum + offset;   zigzag(u) = u / 2 for even u, −(u + 1) / 2 for odd u
```

Integrating a full chunk's 64th delta lands exactly on the next chunk's start value. That holds for
all 48,142,174 of 48,142,174 chunk boundaries. There is one width code per 8 *deltas*: reading `ceil(frames / 8)`
codes runs into the lanes.

**Quaternion tracks.** A track with a type-0 channel is a constant `(x, y, z, +√(1 − x² − y² − z²))`. Otherwise
its components come from scalar channels:

- **Type 7** appears on a track that stores three components (comp 0–2). The missing component is
  `+√(1 − sum of squares)`, placed at the type-7 channel's comp position, and the stored three fill the other
positions in order. Measured: 4,942,792 sequence tracks; the largest |x, y, z| overshoot before normalising is
0.0971.
- **Type 8** always sits at comp 0 on a track that stores all four components. Nothing is rebuilt; the four values
  are normalised. Measured: 227,655 sequence tracks; |q| of the four stored values is within 0.0190 of 1.

Static quaternions: 3,579,477, and 0 of them have |x, y, z| > 1. The (x, y, z, w) order and the rebuilt sign fit the
boundary check above. An earlier local comparison against one skeleton's bind rotations also fits (dot products
0.97–1.0); that comparison is not part of the tool.

### Coverage

191,042 animations in 52,923 local packs; 191,042 decode with every check above passing and 0 fail (0 skipped by the
time budget). 8,582 dictionaries have no local pack; dictionaries refer to 27,559 packs by hash only (18,595 distinct
hashes), and none of those hashes is the hash of one of the 52,923 local packs: they are dictionary hashes. In
anim.rpf, 16,693 of the 18,595 distinct hashes are the `joaat32(name)` of a clip dictionary here (21,014 of the
27,559 references). Clips: 312,263, of which 17,247 are
`.motion` and 180 are `.motionmatcher` (`+0x20` is a pointer in both; the name is at `+0x30` for `.motion` and at
`+0x38` for `.motionmatcher`, after a 16-byte id); `.motion` clips are described below. Sequences: 212,306. Before
the gap-page join, a full run with `--jobs 44` took 5 min 44 s on 44 cores; the slowest dictionary takes about 32 s.

### `.motion` and `.motionmatcher` clips

A `.motion` clip has no animation of its own; it lists other clips (its children), which it presumably blends. Its
fields are in the object table above. Measured on all 17,247 `.motion` clips:

- `+0x18` points at the dictionary root, `+0x28` is the clip's own hash (the low half of its `atMap` key) and `+0x2C`
  is the dictionary's `joaat32(name)`. `+0x58` is 1 on 8,932 and 2 on 8,315.
- They have 159,584 children `{clip hash, dictionary hash}`. For every child, the dictionary hash is one of the
  root's hash-only `+0x70` references (159,584 of 159,584), so those references are the dictionaries the
  `.motion` clips draw from.
- 114,925 children name a dictionary in anim.rpf, and 111,948 of those (97.4%) name a `.clip` of that
  dictionary. The other 2,977 name a hash that is not a clip there.
- The remaining 44,659 children name one of the 1,902 referenced dictionaries that are not in anim.rpf. A local
  search of every clip-dictionary-like root page in `levels`, `packs`, `story000`, `story000_script`,
  `story000_restofworld`, `core_restofworld`, `core_restofworld001`, `shared_minigame000`, `data`, `common` and
  `script` found none of those hashes, so this package does not resolve them.
- **Child times (`+0x70`).** One 0x28-byte item per child on all 17,247: the child's duration, its duration in the
  blend, and a piecewise-linear curve from child time to blend time. Of the 159,584 items, 158,991 have equal
  durations and the two points (0, 0) and (d, d), so the child plays at its own speed. 402 have a 5–82 point curve
  from (0, 0) to (duration, blend duration), increasing in both coordinates, so the child is time-warped. The other
  191 have two points, but 170 don't start at (0, 0) and 21 don't end at (d, d). For every child that names a
  `.clip` in anim.rpf, the first float is bit-for-bit that `.clip`'s duration (`+0x78`): 111,948 of 111,948.
- **Parameter node.** When `+0x58` is 1 the node is at `+0x138`; when it is 2, `+0x120` points at a block whose
  `+0x18` is the node. Every `.motion` has one, and it starts `{u64 0, u32 type, u32 0}`. By type: 3 on 10,171, 1 on
  3,557, 4 on 2,644, 5 on 482, 6 on 295, 8 on 98.
- **Axes by type.** Types 3, 4, 5, 6 and 8 have 1, 2, 3, 4 and 5 axes: `+0x10` is a `{ptr, count}` array of 8-byte
  `{u32 name hash, u16, u16}` entries, with that count on every node of each type (the two u16 are not decoded).
  Type 1 has no axis array and a sample count at `+0x20`: 1 on 3,552, 2 on 5. Types 4, 5, 6 and 8 keep their
  sample count at `+0x90`, `+0x90`, `+0x50` and `+0xA0`: on all 2,281 of them that are a `.motion`'s only node, it
  equals the number of `+0x90` sample lists below. The word list resolves 1,135 of their 8,404 axis names, the same
  names as type 3 (`verticalheight`, `yawangle`, `surfaceincline` the most); `anim_clips.py` counts them as
  `motion_node_axis_name:<type>:<name or unresolved>`. Types 4–8 have a second `{ptr, count}` array at `+0x20`
  (found by @ismailchaka1): null on all 2,644 type-4 nodes and on 431 of the 482 type-5 nodes, present on the other
  51 type-5 nodes and on every type-6 (295) and type-8 (98) node. On all 393 type-6 and type-8 nodes its 16-byte
  items read `{u32, u32 0, u32, u32 0}` and none of those words is one of the node's axis names or the `.motion`'s
  child or dictionary hashes; on type 5 only 2 of the 51 arrays have that shape. The item count is 5 on 184 type-6 nodes, 3 on 57, 2 on 44; 9 on 47 type-8 nodes and 6 on 42.
  It equals the child count on only 5 of the 444 nodes. Not decoded; `anim_clips.py` counts the item counts as
  `motion_node_pairs:<type>:<count or null>`.
- **Type 3: one sampled axis.** It has one axis entry, then 1–16 `{u32 sample index, f32 value}` items and the
  axis min and max. On all 10,171 (29,123 samples), the items are sorted by value, min and max are the first and
  last values, and the sample indices are 0..n−1 in some order. They are not child indices: the sample count
  equals the child count on 4,589 of them. The name hashes are `joaat32` of a lower-case name. 88 distinct hashes
  occur, and a word-list search resolved 10 of them, covering 5,604 axes: `yawangle` 2,524, `rotationz` 1,691,
  `facingdirection` 638, `speed` 284, `surfaceincline` 271, `roll` 66, `verticalheight` 58, `pitch` 25,
  `pitchangle` 25, `height` 22. The ranges fit the names: `speed` is within [0, 1], `surfaceincline` within
  ±0.698 rad (40°), and `roll` within ±3.14. `anim_clips.py` reports every name as `motion_axis_name:<name or hash>`.
- **Samples to children (`+0x90`).** A `{ptr, count}` array of one `{ptr, count}` list per sample of the parameter
  node; a list item is 12 bytes, `{u32 child index, f32, f32}`. On all 8,932 `.motion`s whose only node is at
  `+0x138` (`+0x58` 1), the number of lists equals the node's sample count, every child index is below the child
  count, and every child is played by at least one sample. So sample index k plays the child in list k: the type-3
  sample index is not a child index, it goes through this list. 8,903 have one item in every list; the other 29
  (types 5, 6 and 8) have lists of several children. On the 6,970 where the sample count equals the child count,
  list k names child k; on the other 1,962 there are more samples than children and some children are played by
  several samples. The first float is 1.0 on every item of types 1, 3 and 4 (36,830) and on all but 576 of the
  57,043 items of types 5, 6 and 8; the second is 1.0 on 76,604 of the 93,873 items. Neither is decoded. By type
  (one-node motions, samples equal to children / more): 1: 1,176 / 0, 3: 4,545 / 930, 4: 1,009 / 479, 5: 154 /
  271, 6: 44 / 226, 8: 42 / 56. `anim_clips.py` counts these as `motion_samples:<type>:*`.
- Not decoded: where types 4, 5, 6 and 8 place their samples in parameter space (no array of theirs has one entry
  per sample; most have a pointer at `+0x30`), the two u16 of an axis entry and the two floats of a sample item, how a
  `.motion` with `+0x58` 2 (a block array at `+0x120` with several nodes) shares the `+0x90` lists among its
  nodes, and the rest of the object from `+0x80`.

The 180 `.motionmatcher` clips sit in 142 dictionaries that hold only `.motionmatcher` clips and no local pack.
The three motion matcher streams in [Bone tables and motion matchers](#bone-tables-and-motion-matchers)
(`0x101c379a0`, `0x101c41d50`, `0x101ea66c0`) are the second system pages of three of those dictionaries, with 2, 1 and 2
`.motionmatcher` clips.

A `.motionmatcher` object has fields up to `+0x18F`; they are in the object table above, and `anim_clips.py` decodes them
(`parse_matcher`) and counts them as `matcher_*`. Measured by `parse_matcher` on the pages of all 142 dictionaries
(180 clips):

- **Arrays** keep a u32 count at pointer − 4 and a u32 capacity at pointer − 8, not the `{ptr; u16 count}` of the
  rest of the dictionary.
- **Records (`+0x58`).** 4,562 records (1–346 per clip, measured separately). The third float is −1.0 in all of
  them, and the u32 is the record's position on 175 of 180 clips. The first float is 0 or 0.03–2.37 and the second
  one of 2.5, 5, 6.667, 10, 15, 20, 30 or 40 (also measured separately); that they are a start time and a sample
  rate is a hypothesis.
- **Offsets (`+0x68`).** One more than the records on 180 of 180, starting at 0 and never decreasing, so an offset
  table into data not decoded here.
- **Clip table (`+0xA0`).** An open-addressing hash table: the group count is a power of two (2 on 151 clips,
  measured separately), every control byte is 0x80 or below 0x80, the last word of every slot is 0 (measured
  separately), and the full slots equal the entry count on 180 of 180. Its 4,381 slots hold a clip hash that is a
  clip in anim.rpf (4,381 of 4,381) and a clip index; the indices are 0..n−1 over the entries on 180 of 180. The
  matcher's clips are these n entries, and its records are cut from them: a record's fourth word is its clip's
  index and the record lies in that clip's `+0x78` range on 180 of 180 (`matcher_record_clip_in_its_u16_78_range`).
  There is one clip per record on 175; on the other 5 some clips have several records. The second word names the
  clip's dictionary: measured separately against the 60,778 dictionary names of anim.rpf, it is `joaat32` of the
  dictionary's name behind a fixed 8-character prefix on 4,356 of 4,381 entries (the slot's clip is in that
  dictionary on 4,349), the plain dictionary hash on 18, and neither on 7. 177 of 180 matchers resolve all their
  entries this way.
- **References (`+0x98`).** 36 clips have one `{clip hash, dictionary hash}` per clip (439). All 439 resolve to a
  clip of that dictionary in anim.rpf, and the item at each table slot's clip index names that slot's clip on all
  36. On those 36, the dictionary the second word names is the reference's on 438 of 439 (measured separately), so
  the table gives the dictionary for the other 144 too.
- **Key table (`+0x170`).** The same table with 8-byte slots, 2 groups on all 180: 403 entries of `{u32 hash, u32
  index}`, the indices 0..n−1 on 180 of 180. 38 entries are clip hashes in anim.rpf; there are 23 distinct hashes
  (measured separately).
- **Where an entry sits.** Every entry is where its hash h puts it: control byte h & 0x7F, in group
  (h >> 7) & (groups − 1), or, when that group is full, in a later one with every group in between full (linear
  probing), on 180 of 180 for each of the three tables (`table_placed_by_key`). The key table and the features'
  `+0x48` table hash an entry by its own first u32. The clip table hashes the u64 of its first two words (clip
  hash low, second word high) with MurmurHash3's 64-bit finaliser (`fmix64`, `clip_slot_hash`): 4,351 of its
  4,381 entries sit in their home group and the other 30 after full groups (measured separately).
- **Self (`+0x188`).** The clip's own hash and the dictionary's `joaat32(name)` on 180 of 180.
- **`+0x78` and `+0xB8`.** The u16 array has one item per clip and then one more, on 180 of 180: it starts at 0,
  never decreases and ends at the record count, so item i is clip i's first record. The u8 array has one item per
  clip on 180 of 180 and is 0 or 1; 16 items are 1. (With one clip per record, on 175, both also fit "one per
  record".)
- **Features (`+0x20`).** Each matcher points to a feature object with a list of features: 1,873 in all, distinct
  name hashes within each clip on 180 of 180 (1–21 per clip and 53 distinct names overall, measured separately). Widths are multiples of 3 on 180 of 180
  (3, 6, 9 or 12, measured separately), and the columns chain from 0 (each feature starts where the one before
  ends) on 180 of 180, so a clip's features fill 6–126 columns (11,616 in all; the range measured separately),
  and the pose matrix below has one column per feature column. The `+0x48` table has one slot per feature, with the indices 0..F−1 on
  180 of 180; its hashes are not the `+0x18` name hashes (0 of 180, measured separately). `+0x0E` is 1 on 1,842
  features and 5 on 31, all 31 in clips with the blocks layout (measured separately). With 1, the name hash is
  `joaat32` of a bone name of the clip's skeleton on 1,842 of 1,842, and `+0x1C` is a set of bits that gives the
  width: 3 columns for bit 0, 3 for bit 1 and 6 for bit 3, on 1,842 of 1,842 (kinds 1, 3, 9 and 11 are seen,
  measured separately). That bit 0 is the bone's position, bit 1 its velocity and bit 3 a rotation written as two
  vectors is a hypothesis; the pose check below supports it for position and rotation. With 5, `+0x18` names no
  bone (0 of 31): `+0x1C` is 0 on 20 and `+0x18` is 0 on the other 11. Measured separately, the nonzero words of
  the 30 three-column ones are `joaat32` of names that are not bones (a word list resolves some of them).
- **Skeleton (feature object `+0x28`).** 3,918 bones over the 180 clips. One root and every parent before its
  child on 180 of 180, the bones' parent field is the `+0x40` array, and the tag table holds exactly two tags per
  bone, `joaat32` of its name and its u16 id, on 180 of 180. The root's id is 0 and every other bone's is the ELF
  hash of its upper-cased name, mod 0xFE8F, plus 0x170, on 180 of 180 (3,738 bones; `bone_id`). The local
  matrix is the bone's rotation and translation, and the inverse bind matrix times the product of the local
  matrices up the chain is the identity (to 2e-3), on 180 of 180. `+0x48` is a prefix tree of the bone names
  (shared prefixes stored once): walking it, a node whose value lands in the u16 array after the tree names the
  bone in that slot, and every bone's name leads to that bone, on 180 of 180 (`skeleton_name_tree`). The array
  lists the bones in tree order (measured separately). `parse_matcher` returns it all as `skeleton`.
- **Per sample (`+0x98`).** In both layouts, one u32 per sample, the same for every sample of a record, on 180 of
  180. It is a bit mask over the key table's indices: the bits set over all samples are exactly bits 0..K−1, with
  K the key table's entries, on 180 of 180 (`matcher_sample_u32_bits_are_key_indices`), so each record carries the
  set of keys (tags) that apply to it.
- **Pose matrix (feature object `+0xF0`).** One row of f16 values per sample, one column per feature column. The
  sample count equals the last `+0x68` offset on 180 of 180 (36,436 samples), so record i owns the rows from
  offset i to offset i+1, and the column count is the features' total on 180 of 180. Two layouts, told apart by
  the feature object's `+0x30` object (149 with it, 31 without):
  - **Rows** (149): row-major, each row padded with zero columns to the next multiple of 16 (`+0x100` chunks of 16).
  - **Blocks of 8** (31): 8 samples at a time, column-major inside a block, so column c of sample r is at
    `(r // 8) · D' · 8 + c · 8 + r mod 8`, with D' the column count rounded up to even (the spare column is zero).
    The lanes after the last sample repeat it.

  The sizes fit these layouts on 180 of 180 (`parse_matcher` refuses one that doesn't), every value is finite and
  the padding is as described on 180 of 180. On the 149 with rows, `+0xA8` has one 48-byte item per sample: a
  unit quaternion, a vec3 and a scale of (1, 1, 1) on 149 of 149. It is the sample's reference frame: the
  quaternion is a rotation about z only on 30,946 of the 30,962 samples, and the frame comes from the `+0x30`
  object's weighted bones (below). `anim_clips.matcher_pose` reads one row.

  Measured separately by decoding the clips' own keyframes at each sample's time (up to two clips per matcher,
  285 clips; record start + j / the record's rate), a feature holds its bone's pose in that frame:
  - position columns are the bone's model-space position minus the frame's vec3, rotated by the frame's inverse,
    on 11,469 of 12,873 samples (to 0.01 m);
  - the two vectors of bit 3 are rows 0 and 1 of the bone's model-space rotation, in the same frame, on 2,233 of
    2,544 (to 0.02);
  - the frame's x and y are the weighted centroid of the object's bones on 1,148 of 1,233 samples; its z is the
    centroid's z with flag 6, and −1.0 on every sample on 29 of the 32 form 1 matchers with flag 7 (all 149
    matchers counted);
  - its yaw is the heading from the first to the second bone with form 2 (682 of 752), and the z-twist of the
    reference rotation's inverse times the bone's rotation with a one-bone form 1 (89 of 89).

  Velocity columns (bit 1) fit a plain difference of the model-space position on only 1,960 of 9,366 samples, so
  they are not decoded. A record's rows cover part of its clip, not all of it:
  with rows 1 / (the record's second float) seconds apart, they fit inside the clip's duration on 2,108 of the
  2,155 records whose clip duration is known without ambiguity (measured separately), which fits the sample-rate
  hypothesis above. Which part of the clip they cover is not decoded.
- **One-item arrays.** The matcher's `+0x48` is one 40-byte item, zero except a u32 1 at `+0x20`; `+0x118` is
  one u32 equal to the pose sample count; `+0x128` is one u16, 0; each on 180 of 180.
- **`+0x30` object.** On the 149 clips that have it, its `+0x08` is 2 on 93 and 1 on 56. It names the bones the
  pose frame is built from (above). With 2: two distinct bones of the clip's skeleton by `joaat32` of their names
  at `+0x10`/`+0x14` (93 of 93), and at `+0x30` a copy of the clip's first feature (its `+0x0E`, name hash and
  kind, 93 of 93) whose width is at `+0x68` (by the bit rule above, 93 of 93); the u16 at `+0x6C` is 6 on 92 and
  9 on 1. With 1: the arrays at `+0x40` and `+0x50` have the same count n (1, 3, 4 or 6, measured separately), and
  the first holds bones of the skeleton by `joaat32` of their names (56 of 56); `+0x10` is a unit quaternion (56
  of 56), and the u32 at `+0x60` is 6 on 24 and 7 on 32. Its weights (form 2: the f32s at `+0x18`/`+0x1C`; form
  1: the second array) are positive and sum to 1 on 149 of 149.
- **Column weights (`+0xB8`).** One f32 per pose column, stored 8 to a 32-byte chunk with zeros after the last
  column (the count at `+0xB4` is the chunks), on 180 of 180. They are positive and sum to 1, and the three
  columns of each vec3 have the same weight, on 180 of 180 (`matcher_column_weights_*`). They are not the
  reciprocal, the deviation or the variance of their columns over the samples (measured separately), and a
  clip's weights and `+0xC0` are the same for every clip with the same features on 20 of 23 feature sets
  (measured separately), so they look like settings, not statistics. `+0xC4` is 1.0, and `+0xC8` is 1 with rows
  and 3 with blocks of 8, on 180 of 180.
- Not decoded: the hash of the features' `+0x48` table. It is a function of the feature's `+0x0E` and `+0x18`
  only (53 distinct pairs, no conflict), and, measured separately, it is not case-sensitive or case-folded
  joaat32 (with any constant seed or prefix, or a suffix of up to 3 characters), one-at-a-time, FNV-1/1a 32 or 64,
  ELF, CRC-32, MurmurHash3 32, MurmurHash64A, or `fmix32`/`fmix64` of the name hash or the bone id. Also not
  decoded: the velocity columns, the yaw rule of a form 1 object with several bones (the z-twist of their
  weighted inverse-bind rotations fits only to 0.15 rad, 353 of 392 samples), why form 1's quaternion is not
  the blend of its bones' bind rotations (15 of 56), the width-9 feature with `+0x0E` = 5, and the f32 at
  `+0xC0`.

### Other archives

The same tool reads `story000.rpf` with no change: of 34,533 roots with the dictionary header, 34,518 join (7 with a
gap page) and 9,190 are clip dictionaries. The other 25,328 roots have some other value at `+0x70` and are not clip
dictionaries; 15 roots do not join. All 59,593 animations of those dictionaries decode with every check passing, and
all 54,352,969 type-10 chunk boundaries close. Its 14 `.motion` clips have 75 children, and all 75 name a `.clip` of a
dictionary in `story000.rpf`. None of its 9,188 distinct dictionary hashes is a dictionary in anim.rpf, and none is
one of the 1,902 hashes anim.rpf refers to without having them.

```
python -I tools/anim_clips.py <package>\pack\story000\story000.rpf <work>\story000-clips.json --jobs 20
```

## Rig dictionaries

`tools/anim_rigs.py` reads the four `RIG` roots above (ASSET-20). Run on the 1.0.2.1 copy:

```powershell
python -I tools/anim_index.py <package>\anim.rpf <work>\anim-index.json --rigs <work>\rigs.json
python -I tools/anim_rigs.py <package>\anim.rpf <work>\anim-rigs.json --rigs <work>\rigs.json
```

The output holds counts and one row per rig dictionary (offset, name, hash, entry count), no table items.

**Layout.** The root is the clip dictionary's: `+0x38` atString name, `+0x48` atMap, `+0x68` `joaat32(name)` (4 of
4), and at `+0x70` the bytes `RIG` and a version byte (2 in all four). Every map key has a zero high half (3,897 of
3,897). Every map value has a pointer at `+0x20` to a **`RIG` block** (3,897 of 3,897): `RIG`, a version byte (2),
a u32 length of what follows, then tables `{u32 kind hash; u32 size = 0xC + 8 × count; u32 count; u32 word}`, each
followed by `count` 8-byte items `{u32 hash; u32 word}`, with other bytes between the tables. The other bytes and
the item words are not mapped.

**The four dictionaries are the four RBF0 sources.** Their names are the four source strings of the rig-set
manager:

| Name | Entries | Keys that are `joaat32` of an RBF0 rig |
|---|---:|---:|
| `canonical@legacy` | 22 | 16 |
| `canonical@source` | 543 | 534 |
| `generated@source` | 1,256 | 1,188 |
| `pipelined@source` | 2,076 | 1,931 |

These are the distinct-rig counts per source in [the RBF0 table above](#tail-rbf0-rig-set-manager). The other 228
keys are not named by any RBF0 entry. **All 6,677 RBF0 entries resolve**: the entry's `Dictionary` text is the
rig dictionary's name, and `joaat32(lowercase Rig path)` is a key in it (6,677 of 6,677; 0 in another dictionary,
0 missing). So a rig set reaches its rigs' `RIG` blocks through rig-set → `{Dictionary, Rig}` → rig dictionary →
key → `+0x20`.

**Tables.** 6,606 tables are found in the 3,897 blocks. Four kind hashes hold all but 5 of them; the 5 are single
hits whose "kind" is an f32 near 1.0 (`0x3f7f....`, `0x3f76....`) or a small integer, so they are float data that
happens to fit the header rule. The kind names are not resolved:

| Kind | Tables | Items | Items that are a clip track's bone hash |
|---|---:|---:|---:|
| `0x74d00508` | 2,642 | 13,210 | 0 |
| `0x2316833f` | 2,437 | 118,853 | 91,559 |
| `0xb688b77e` | 1,427 | 72,528 | 39,880 |
| `0x5e25d3bf` | 95 | 4,416 | 0 |

"A clip track's bone hash" means the hash is the bone of some track in the 191,042 decoded animations.

**Rigs are not the clips' skeletons.** The clip side was measured with a local script over the same
`anim_clips.py` parse: for each of the 52,752 clip dictionaries with bone tracks, the union of its
`bonerotation`/`bonetranslation` bone hashes (mover track 0 left out).

- Against the 596 hashed names of the shared humanoid skeleton ([character_skeleton.md](character_skeleton.md)),
  44,359 dictionaries have at least 90% of their bones in it (16 have all of them).
- Against the rigs (the items of the two bone-bearing kinds above), no rig set covers the clips of a humanoid
  dictionary. Even the union over all the rigs of one rig set reaches 80% of a dictionary's bones for 11 of those
  44,359. Only 4 dictionaries, all of objects, are covered completely by a rig set.

So humanoid clips bind to the fragment skeleton, not to a `RIG` block. The rig tables look like per-rig bone or
channel subsets (the RBF0 groups are `deformation`, `pose`, `identity`, `facial` and others); that reading is not
measured. The auxiliary table on the second `0DRP` page is [framed above](#packed-clip-streams). Which model
or skeleton each non-humanoid clip dictionary drives remains open.

### Can the fragment bone names identify a clip's model? (ASSET-158)

In the ASSET-158 scan, bone coverage identified **zero unique fragments** for the 8,393 dictionaries below the 90% humanoid
threshold above. This comparison used the named fragment tables recovered from `packs.rpf` in Xbox 1.0.2.1.
The threshold selects dictionaries for investigation; some may still animate humanoids.

| Coverage of a dictionary's nonzero rotation and translation bones | Dictionaries |
|---|---:|
| At least one bone absent from the union of the recovered fragment tables | 7,817 |
| Every bone in the union, but no individual table contains them all | 573 |
| One or more individual tables contain every bone | 3 |

Each of the last three dictionaries animates just one bone. Each matches **82 fragment pages** spanning
**50 distinct bone sets**. Choosing the first matching page would assign a model arbitrarily.

To reproduce the comparison:

1. Walk all 144,364 streams returned by `anim_index.scan_rpf` for `packs.rpf`. In each inflated page, find
   the first NUL-terminated `.#ft` suffix before the last NUL-terminated `skel` marker. Between them, hash
   the NUL-terminated ASCII runs matching `[A-Za-z0-9_]+` with `anim_index.joaat32`.
2. Read aligned little-endian `{u32 hash, u32 index}` pairs in the page prefix ending before the suffix.
   The archive scan retained a hash in the name set when its index was less than or equal to the number
   of distinct name hashes. This inclusive filter selects candidates across pages. The reference
   hash table itself uses zero-based indices 0 through 595. This recovers
   596 hashes on the [known reference page](character_skeleton.md#the-skeleton-name-table). Across the archive,
   2,849 pages yield a nonempty set, with 524 distinct sets. This check covers pages with that name and
   hash layout; other skeleton layouts remain open.
3. Group and parse the `anim.rpf` dictionaries with `anim_clips.group` and `anim_clips.parse_dictionary`.
   Union the nonzero rotation and translation bone hashes across each dictionary's local packs. Keep
   dictionaries for which fewer than nine tenths of those hashes occur on the reference page.
4. Require **all** bones of a retained dictionary to occur on the **same** fragment page. Count matching
   pages before deduplicating identical bone sets, so repeated fragments remain distinguishable.

### Explicit model-to-dictionary references in `levels.rpf`

The ASSET-158 scan of archetype metadata linked **1,324 model asset names and 159 clip dictionaries**
in Xbox 1.0.2.1. Of those dictionaries, **63** carry local rotation or translation tracks below the humanoid
coverage threshold. Their archetypes name **488 distinct model assets**. This identifies which dictionaries
the archetypes reference; the per-bone mapping onto each model's skeleton still needs independent evidence.

1. Walk the complete version-122 `0DRP` metadata resources in `levels.rpf` with `world_census.assemble`
   and the `Meta` reader in `world_lights_paths.py`. For each `CBaseArchetypeDef` or `CMloArchetypeDef`
   block with the fields, find `assetName` and `clipDictionary` through the struct's own member table
   using their case-sensitive joaat hashes. Read the two u32 name hashes from the same instance.
   Across 188,449 complete metadata resources, 15,394 records carry a nonzero dictionary reference,
   giving 161 distinct dictionary hashes.
2. Search **the same archive** for clip dictionary roots. At each 16-byte-aligned zlib stream start,
   check `anim_clips.is_root` and the dictionary hash at `+0x68` against those references. Join consecutive
   system pages as needed to parse the dictionary. Require a nonempty clip map and a name at `+0x38`
   whose lowercase joaat equals the stored hash. The scan confirmed 159 of the 161 referenced hashes.
   A direct hash comparison against the 52,752 bone-bearing dictionaries in `anim.rpf` found zero matches.
   Applying the same root and clip-map checks to the other 21 RPF archives (85,152,072,416 bytes)
   yielded no additional matches. The two references unresolved by that scan appeared in 43 and 64 archetype records,
   covering two and 16 distinct asset names respectively. This search covers readable dictionary roots
   at aligned zlib stream starts. Other storage layouts remain possible.
3. Deduplicate the resulting `(assetName, clipDictionary)` pairs. Of the 159 confirmed dictionaries,
   55 are referenced by one distinct asset name each. The remaining 104 are shared. Across their local
   animation packs, 63 dictionaries have nonzero rotation or translation bone hashes and 96 have none.
   All 63 fall below the 90% reference-skeleton threshold used above.
measured. Still open in ASSET-20: the `0DRP` "2/3 packed clip streams" ([above](#packed-clip-streams)) and which
model or skeleton each non-humanoid clip dictionary drives.

## `MoVE` state-machine containers

`tools/anim_move.py` decodes all 324 `MoVE` (version 19) resources in `anim.rpf` (ASSET-86):

```powershell
python -I tools/anim_move.py <package>\anim.rpf <work>\anim-move.json
```

All 324 resources (325 zlib streams, 331,771 stored / 1,494,188 inflated bytes once the 64 KiB stream at
`0x105a7d000` is joined with its 5,563-byte continuation stream at `0x105a7f9c0`) decode with 0 errors. Unlike clip
dictionaries and `0DRP` resources, a `MoVE` resource has no virtual `0x50xxxxxx` pointers or RAGE page map: every
internal cross-reference is a signed 32-bit self-relative byte offset (`target = field_offset + rel_i32`).

### Header and variable preamble

After the fixed 32-byte header (`"MoVE"`, `u32 version = 19`, three zero `u32`s, `u32 network_hash` at `+0x14`,
`u32 body_size` at `+0x18`, `u32 0` at `+0x1C`), a variable-length preamble starts at `+0x20`:

| Field | Type | Measured across 324 resources |
|---|---|---|
| `N_req`, `requests` | `u32`, `N_req × u32` | `0..16` request hashes (`> 0` in 149 resources, 543 entries total) |
| `N_flag`, `flags` | `u32`, `N_flag × u32` | `0..27` flag hashes (`> 0` in 123 resources, 687 entries total) |
| `preamble_mode` | `u32` | `0` in 225 resources, `1` in 91, `2` in 8 |
| `N_desc`, `descriptors_5b` | `u32`, `N_desc × {u32 hash; u8 kind}` | `0` in 184 resources, `1..15` in 140 (272 entries total; `kind` = `0`: 20, `1`: 92, `2`: 74, `4`: 1, `7`: 85) |
| `tail` | `{u16 sub_flag; u32 ref_hash_0; u32 ref_hash_1}` | 10 zero bytes whenever `preamble_mode` is `0` or `1` (316/316); `sub_flag = 1` with two non-zero reference hashes when `preamble_mode == 2` (8/8) |

The root state container starts immediately after the 10-byte tail at
`node_start = 48 + 4 * (N_req + N_flag) + 5 * N_desc` (`58..244` bytes). `body_size` at `+0x18` counts the
request/flag arrays, three `u32` preamble words and the entire state graph (`body_size == 12 + 4 * (N_req + N_flag) +
graph_bytes`), excluding the 46 fixed/tail bytes and the `5 * N_desc` descriptor table:

$$\text{total\_bytes} = 46 + 5 \cdot N_{\text{desc}} + \text{body\_size}$$

holds exactly for **324 of 324** resources.

### Recursive state containers (`Type 1`, `Type 3`, `Type 22`)

Every node in the graph starts with `{u16 type; u16 index; u32 name_hash}`. Three container types form the recursive
state-machine and motion-tree hierarchy (**1,616** containers across the 324 resources). All three share a 36-byte
(`0x24`) common prefix:

```
+0x00  u16 type                (1 = StateMachine, 3 = InlinedStateMachine, 22 = MotionTree)
+0x02  u16 index               (0 on root/child states; node index inside an enclosing MotionTree)
+0x04  u32 name_hash           (joaat32; e.g. joaat32("statemachine") = 0x1997df3c, joaat32("motiontree") = 0x8d398f99)
+0x08  i32 init_or_root_rel    (target = off + 0x08 + init_or_root_rel)
+0x0c  u32 flags               (0 or 1)
+0x10  u8 b10, u8 b11, u8 n_children, u8 n_transitions
+0x14  four u32 words          (0 in all measured Type 1 and Type 3 headers)
```

| Container type | Instances | Header & body layout after `+0x24` |
|---|---:|---|
| `Type 1` (`StateMachine`) | 230 (63 at file root, 167 nested) | `0x58`-byte header: `+0x24` `i32 transitions_rel` (`off + 0x58 + 8 * n_children` when `n_transitions > 0`, or `0`), `+0x44` `f32 1.0`, `+0x50` `u32 n_default_transitions`, `+0x54` `i32 default_transitions_rel`. At `+0x58`: `n_children` 8-byte child directory entries `{u32 child_hash; i32 child_rel}` (`target = entry + 4 + child_rel`), then `n_transitions` outgoing transitions, `n_default_transitions` default transitions, and the `n_children` child state containers (`Type 1` or `Type 22`). In 230 of 230 instances, `init_target` is in the child directory and every child's `+0x04` hash matches its directory `child_hash`. |
| `Type 3` (`InlinedStateMachine`) | 132 (all inside a `MotionTree`, 332 child states) | Same header as `Type 1` through `+0x54`, plus `u32 n_inputs` at `+0x58` (`1..2`, 151 inputs total) and `n_inputs × i32 input_rel` at `+0x5c` (`target = (off + 0x5c + 4*i) + input_rel`, pointing to the input pose node in the enclosing `MotionTree`). The `n_children × 8`-byte child directory follows at `off + 0x5c + 4 * n_inputs`, then `n_default_transitions` (114 total) and the `n_children` child states. |
| `Type 22` (`MotionTree`) | 1,254 (261 at file root, 993 nested) | `0x30`-byte header: `+0x24` `i32 transitions_rel` (always `0x0c`, so transitions start at `off + 0x30`), `+0x28` `i32 param_blocks_rel`, `+0x2c` `u32 n_param_blocks`. Followed by `n_transitions` outgoing transitions, `n_param_blocks` length-prefixed parameter blocks, and the motion-tree node arena (`3,875` declared children; first node has `index == 0`, root expression node is at `off + 0x08 + init_or_root_rel`). |

### Transitions, parameter blocks and motion-tree nodes

- **Outgoing state transitions (1,097 total: 77 on `Type 1`, 1,020 on `Type 22`):** each transition record carries its
  own byte length as `u32 size` at `+0x10` (`size >= 0x4c`; `76`, `84`, `88`, `92`, `96`, ... bytes), a signed relative
  offset to the target state at `+0x48`, and (when `size >= 0x50`) a `u32 condition_kind` at `+0x4c` followed by
  `size - 0x50` bytes of condition hashes and thresholds. Summing `size` across `n_transitions` matches the exact
  transition block span in **100%** of containers.
- **Default / any-state transitions (491 total: 377 on `Type 1`, 114 on `Type 3`):** each record starts with
  `{u16 size; u16 condition_count; i32 target_rel}` (`size >= 16`). Summing `size` across `n_default_transitions`
  matches `first_child - def_tr_start` in **100%** of containers.
- **MotionTree parameter blocks (2,104 total across the 1,254 `MotionTree`s):** each block starts with
  `{u32 block_size; u8 index, b1, b2, b3, b4, b5, b6, b7}` (`b7 == 0` in all 2,104), followed by
  `(b1 + b2 + b3 + b4)` 16-byte parameter/signal bindings (`3,683` total), `b5` 8-byte output/event bindings (`591`
  total), and `b6` variable-length curve/expression sub-records (`632` total, each carrying `u16 sub_size` at `+0x04`).
  The byte identity
  $$\text{block\_size} = 12 + 16(b_1 + b_2 + b_3 + b_4) + 8b_5 + \sum_{k=1}^{b_6} \text{sub\_size}_k$$
  holds in **2,104 of 2,104 (100%)** blocks.
- **MotionTree root expression nodes (1,254 total):** the root node at `init_target` is most commonly `Type 11`
  (clip leaf, 216), `Type 4` (2-way blend, 193), `Type 3` (`InlinedStateMachine`, 132), `Type 2` (12-byte output leaf,
  116), `Type 28` (filter/modifier, 94), `Type 9` (directional blend, 83), `Type 1` (nested `StateMachine`, 81),
  `Type 10` (N-way blend, 72), `Type 12` (parameterized motion, 55), `Type 19` (additive blend, 49), `Type 15`
  (20-byte unary node, 48), `Type 5` (44-byte blend, 42), and `Type 23` (8-byte identity pose leaf, 20), plus 13
  rarer operator types (`6, 8, 18, 25, 27, 31, 32, 34, 36, 37, 39, 40, 41`, 53 total).

## `MoVE` state-machine containers

`tools/anim_move.py` decodes all 324 `MoVE` (version 19) resources in `anim.rpf` (ASSET-86):

```powershell
python -I tools/anim_move.py <package>\anim.rpf <work>\anim-move.json
```

All 324 resources (325 zlib streams, 331,771 stored / 1,494,188 inflated bytes once the 64 KiB stream at
`0x105a7d000` is joined with its 5,563-byte continuation stream at `0x105a7f9c0`) decode with 0 errors. Unlike clip
dictionaries and `0DRP` resources, a `MoVE` resource has no virtual `0x50xxxxxx` pointers or RAGE page map: every
internal cross-reference is a signed 32-bit self-relative byte offset (`target = field_offset + rel_i32`).

### Header and variable preamble

After the fixed 32-byte header (`"MoVE"`, `u32 version = 19`, three zero `u32`s, `u32 network_hash` at `+0x14`,
`u32 body_size` at `+0x18`, `u32 0` at `+0x1C`), a variable-length preamble starts at `+0x20`:

| Field | Type | Measured across 324 resources |
|---|---|---|
| `N_req`, `requests` | `u32`, `N_req × u32` | `0..16` request hashes (`> 0` in 149 resources, 543 entries total) |
| `N_flag`, `flags` | `u32`, `N_flag × u32` | `0..27` flag hashes (`> 0` in 123 resources, 687 entries total) |
| `preamble_mode` | `u32` | `0` in 225 resources, `1` in 91, `2` in 8 |
| `N_desc`, `descriptors_5b` | `u32`, `N_desc × {u32 hash; u8 kind}` | `0` in 184 resources, `1..15` in 140 (272 entries total; `kind` = `0`: 20, `1`: 92, `2`: 74, `4`: 1, `7`: 85) |
| `tail` | `{u16 sub_flag; u32 ref_hash_0; u32 ref_hash_1}` | 10 zero bytes whenever `preamble_mode` is `0` or `1` (316/316); `sub_flag = 1` with two non-zero reference hashes when `preamble_mode == 2` (8/8) |

The root state container starts immediately after the 10-byte tail at
`node_start = 48 + 4 * (N_req + N_flag) + 5 * N_desc` (`58..244` bytes). `body_size` at `+0x18` counts the
request/flag arrays, three `u32` preamble words and the entire state graph (`body_size == 12 + 4 * (N_req + N_flag) +
graph_bytes`), excluding the 46 fixed/tail bytes and the `5 * N_desc` descriptor table:

$$\text{total\_bytes} = 46 + 5 \cdot N_{\text{desc}} + \text{body\_size}$$

holds exactly for **324 of 324** resources.

### Recursive state containers (`Type 1`, `Type 3`, `Type 22`)

Every node in the graph starts with `{u16 type; u16 index; u32 name_hash}`. Three container types form the recursive
state-machine and motion-tree hierarchy (**1,616** containers across the 324 resources). All three share a 36-byte
(`0x24`) common prefix:

```
+0x00  u16 type                (1 = StateMachine, 3 = InlinedStateMachine, 22 = MotionTree)
+0x02  u16 index               (0 on root/child states; node index inside an enclosing MotionTree)
+0x04  u32 name_hash           (joaat32; e.g. joaat32("statemachine") = 0x1997df3c, joaat32("motiontree") = 0x8d398f99)
+0x08  i32 init_or_root_rel    (target = off + 0x08 + init_or_root_rel)
+0x0c  u32 flags               (0 or 1)
+0x10  u8 b10, u8 b11, u8 n_children, u8 n_transitions
+0x14  four u32 words          (0 in all measured Type 1 and Type 3 headers)
```

| Container type | Instances | Header & body layout after `+0x24` |
|---|---:|---|
| `Type 1` (`StateMachine`) | 230 (63 at file root, 167 nested) | `0x58`-byte header: `+0x24` `i32 transitions_rel` (`off + 0x58 + 8 * n_children` when `n_transitions > 0`, or `0`), `+0x44` `f32 1.0`, `+0x50` `u32 n_default_transitions`, `+0x54` `i32 default_transitions_rel`. At `+0x58`: `n_children` 8-byte child directory entries `{u32 child_hash; i32 child_rel}` (`target = entry + 4 + child_rel`, 747 child states total), then `n_transitions` outgoing transitions, `n_default_transitions` default transitions, and the `n_children` child state containers (`Type 1` or `Type 22`). In 230 of 230 instances, `init_target` is in the child directory and every child's `+0x04` hash matches its directory `child_hash`. |
| `Type 3` (`InlinedStateMachine`) | 132 (all inside a `MotionTree`, 332 child states) | Same header as `Type 1` through `+0x54`, plus `u32 n_inputs` at `+0x58` (`1..2`, 151 inputs total) and `n_inputs × i32 input_rel` at `+0x5c` (`target = (off + 0x5c + 4*i) + input_rel`, pointing to the input pose node in the enclosing `MotionTree`). The `n_children × 8`-byte child directory follows at `off + 0x5c + 4 * n_inputs`, then `n_default_transitions` (114 total) and the `n_children` child states. |
| `Type 22` (`MotionTree`) | 1,254 (261 at file root, 993 nested) | `0x30`-byte header: `+0x24` `i32 transitions_rel` (always `0x0c`, so transitions start at `off + 0x30`), `+0x28` `i32 param_blocks_rel`, `+0x2c` `u32 n_param_blocks`. Followed by `n_transitions` outgoing transitions, `n_param_blocks` length-prefixed parameter blocks, and the motion-tree node arena (`3,875` declared children; first node has `index == 0`, root expression node is at `off + 0x08 + init_or_root_rel`). |

### Transitions, parameter blocks and motion-tree nodes

- **Outgoing state transitions (1,097 total: 77 on `Type 1`, 1,020 on `Type 22`):** each transition record carries its
  own byte length as `u32 size` at `+0x10` (`size >= 0x4c`; `76`, `84`, `88`, `92`, `96`, ... bytes), a signed relative
  offset to the target state at `+0x48`, and (when `size >= 0x50`) a `u32 condition_kind` at `+0x4c` followed by
  `size - 0x50` bytes of condition hashes and thresholds. Summing `size` across `n_transitions` matches the exact
  transition block span in **100%** of containers.
- **Default / any-state transitions (491 total: 377 on `Type 1`, 114 on `Type 3`):** each record starts with
  `{u16 size; u16 condition_count; i32 target_rel}` (`size >= 16`). Summing `size` across `n_default_transitions`
  matches `first_child - def_tr_start` in **100%** of containers.
- **MotionTree parameter blocks (2,104 total across the 1,254 `MotionTree`s):** each block starts with
  `{u32 block_size; u8 index, b1, b2, b3, b4, b5, b6, b7}` (`b7 == 0` in all 2,104), followed by
  `(b1 + b2 + b3 + b4)` 16-byte parameter/signal bindings (`3,683` total), `b5` 8-byte output/event bindings (`591`
  total), and `b6` variable-length curve/expression sub-records (`632` total, each carrying `u16 sub_size` at `+0x04`).
  The byte identity
  $$\text{block\_size} = 12 + 16(b_1 + b_2 + b_3 + b_4) + 8b_5 + \sum_{k=1}^{b_6} \text{sub\_size}_k$$
  holds in **2,104 of 2,104 (100%)** blocks.
- **MotionTree root expression nodes (1,254 total):** the root node at `init_target` is most commonly `Type 11`
  (clip leaf, 216), `Type 4` (2-way blend, 193), `Type 3` (`InlinedStateMachine`, 132), `Type 2` (12-byte output leaf,
  116), `Type 28` (filter/modifier, 94), `Type 9` (directional blend, 83), `Type 1` (nested `StateMachine`, 81),
  `Type 10` (N-way blend, 72), `Type 12` (parameterized motion, 55), `Type 19` (additive blend, 49), `Type 15`
  (20-byte unary node, 48), `Type 5` (44-byte blend, 42), and `Type 23` (8-byte identity pose leaf, 20), plus 13
  rarer operator types (`6, 8, 18, 25, 27, 31, 32, 34, 36, 37, 39, 40, 41`, 53 total).

## Readable, compressed and still opaque

- **Readable:** all 273 name tables (2,298 paths), the `0DRP` version/page count, 1,371 type-hash headers,
  1,781 clip-stream headers with their version split, two named clip records (durations and 0x48-byte track
  arrays), the 63/38 and 12/38 bone masks (all 38 finger hashes resolve), the RBF0 mapping (1,126 rig sets /
  6,677 entries / 3,669 rigs), the four `RIG` dictionaries (3,897 entries / 6,606 tables), all 324 `MoVE` v19
  state-machine and motion-tree containers (1,616 state containers, 1,097 outgoing transitions, 491 default
  transitions, 2,104 parameter blocks), one standalone bone-name list, and three motion matcher streams with
  embedded bone tables (30-name union; 19 bone hashes present).
- **Outer compression:** the only compression this work inverts is the RPF9 zlib framing. Inflating the 130,470
  streams accounts for the 11.59 GB above.
- **No nested zlib:** in eight sampled zero-prefixed pages, none of the `78 01/5E/9C/DA` byte pairs started a
  valid nested zlib stream. The outer zlib stream is the only compression layer observed.
- **No known codec at the sampled headers:** no zstd or LZ4 frame magic was found in the first 4 KiB of the
  130,449 indexed streams. gzip, bzip2 and lzma byte patterns occur, but none started a valid stream at the
  sampled positions.
  No Oodle signature is known from these bytes; none is guessed.
- **Still opaque:** high-entropy blocks inside `other` and zero-prefixed pages (4 KiB blocks up to 7.96
  bits/byte), the `0DRP` root objects and the meaning of their auxiliary payloads, the
  56 empty rig sets, the RBF0 numeric fields, and
  the rig→skeleton link (0 measured matches). Of the 30 bone names, 19 have a confirmed stored hash and 11 do
  not; the four motionmatcher paths bind to embedded tables, but not to an RBF0 rig.

## Cross-check with the model and XML work

- `models-out/data/models/manifest.json` (the only model manifest available here) has 1,350 models: 1,341 with
  `HasBoneBindings`, 689 fragments, all with material entries. It exposes model names and 64-bit texture
  `Id64` keys, but no bone names or skeleton hashes.
- None of the 1,350 model names occurs in the 8,020 animation paths collected here, and none occurs in the
  `RBF0` rig metadata either. The sampled clip hash `0x25ed0554e58ba1fc` and the recurring `0x98e1750c` are
  absent from all manifest names and integer/string values. A real cross-check needs the full model export
  (382,781 models at `e293917` per `docs/design.md`) and the rig table.
- `common.rpf` is the readable metadata layer. It carved to 781 streams, 710 of them XML-ish. Strict keyword
  occurrences: `clip` 734, `anim` 1,131, `motion` 237, `gesture` 2, `bone` 428 (214 `<Bones>` open/close
  pairs), `ped_` 1,308, `blend` 40; `facial`, `ycd` and `skeleton` 0. The tags include `<AnimData>`,
  `<ClipSet>`, `<ClipSelectionRule>` and `<Bones>`, and a path to `platform:/anim/animscenes/gesture/gesture.rpf`.
  These are configuration references, not clip payloads.
- A 64 MiB sample of `data.rpf` (6,359 streams, 3 XML-ish) yielded no animation catalog. Its strict keyword
  hits are dialogue or model/decal names; the five `ycd#` hits are part of a model hash table, not a container
  header.

## Likely format family

The bytes support **new RAGE resource pages**, not the GTA V `YCD`/`RSC7` family:

- RPF9 container, `0DRP` page directories with `0x50xxxxxx` virtual pointers and page counts.
- `pack:/...` path tables with 16-byte-aligned names and 16-byte hash/pointer records.
- `SKEL_*` bone tables in plain and front-coded forms.
- `RBF0` rig-set metadata (`rage__crRigSetManager`): 1,126 rig sets, 6,677 source→rig entries and 3,669 distinct
  rigs, with source tags and group strings. The rig→skeleton key is not in RBF0 or the skeleton streams.
- `MoVE` (version 19) and the 2/3 clip streams as the animation payloads; neither is zlib inside the page.
- No `YCD`, `YDD`, `YLD`, `YFD`, `RSC7` or `RSC8` at any stream start.

`MoVE` (version 19) is a flat relative-offset state-machine and motion-tree graph (`tools/anim_move.py`).

### What the bytes do not prove

- `0DRP` is a page marker with a version, page count and pointers. Its role is inferred from the page layout, not
  decoded, and the marker is not proven to be animation-specific.
- `MoVE` v19's container layout, preamble tables, recursive state containers (`Type 1`, `Type 3`, `Type 22`),
  transitions and parameter blocks are decoded (`tools/anim_move.py`), while the unhashed source names of most
  network/request/flag hashes and the evaluation semantics of the rarest leaf operator node types remain unmapped.
- The two- and three-entry auxiliary tables bound their tagged payloads. Their contents have yet to be linked
  to a clip. The clip dictionaries' keyframes are decoded in [Keyframe decoding](#keyframe-decoding).
- The `RBF0` rig-set/rig/group entries are parsed. The numeric fields, the schema type table and the
  rig→skeleton reference are not; the measured skeleton match is 0.
- The three motion matcher streams embed bone tables, but whether those define a full skeleton or only the
  motion-matching bones is not established. 19 of 30 names have a matching joaat32 in at least one stream.

## What a future decoder needs

1. **Use the parsed RBF0 rig mapping.** It is 1,126 rig sets, 6,677 source→rig entries and 3,669 distinct rigs;
   56 rig sets are empty. The key/tag layout is above. The field after `ff ff 08 10` and the rest of the schema
   still need mapping.
2. **Resolve rig → skeleton.** None of the 3,669 rigs or 1,126 rig sets matches a stored skeleton name or one of
   the 19 bone hashes present. Three motion matcher streams bind four paths to embedded bone tables (30-name
   union; parent indices and bind data in the 2,064-byte example). Parse those records to learn the clip→bone
   binding, then find how an RBF0 rig points at the same table. The missing key is likely a separate rig asset
   or an unparsed numeric field, not these pages. Do not infer it from the category names.
3. **Group streams into resources.** The TOC is encrypted, so use the `0DRP` page header: count at +0x20,
   pointers at +0x08 and +0x28..+0x40, four u16 values at +0x48. The sampled layout places page headers among
   zero-prefixed pages; a logical resource can span several streams. Do not assume one stream is one file. The
   page map at the `+0x08` pointer gives the system page count, and that is how `tools/anim_clips.py` joins the
   clip dictionaries.
4. **Parse the name/hash tables.** Names are NUL-terminated and 16-byte aligned; the name hash is joaat32 of
   the stem (path without `pack:/` and without the final extension). The record stride varies (16 or 32 bytes in
   sampled tables), so do not assume one layout. The recurring `0x98e1750c` is `joaat32("bonerotation")`, a
   track type.
5. **Trace the auxiliary payloads to their consumers.** Use the parent's `+0x40` table to bound each payload
   before checking its fields against the root object's schema. The table framing alone establishes no
   association with the two named page-0 clips. (`MoVE` v19 is now decoded separately by `tools/anim_move.py`.)
6. **Keep the compression boundary explicit.** Outer RPF9 zlib is understood. Do not add a zlib pass inside a
   page without evidence; the sampled pages had no valid nested stream.
7. **Re-run the model cross-check with the full export.** The current 1,350-model data.rpf manifest has bone
   bindings but no bone names or hashes. The full 382,781-model export plus the `RBF0` rig table is needed to
   match skeletons to models.

## Reproduction

```powershell
# Full stream index and coverage (writes index.json only)
python -I tools/rpf9_carve.py <package>\anim.rpf <work>\anim-index --index-only

# Signature index (CSV or JSON; --limit/--start for a bounded scan)
python -I tools/anim_index.py <package>\anim.rpf <work>\anim-index.csv
python -I tools/anim_index.py <work>\anim-carve <work>\anim-index.json

# Keyframe decoding: counts and one row per clip dictionary; --decode prints one dictionary's values
python -I tools/anim_clips.py <package>\anim.rpf <work>\anim-clips.json --jobs 8

# Rig dictionaries and MoVE v19 state-machine containers
python -I tools/anim_rigs.py <package>\anim.rpf <work>\anim-rigs.json --rigs <work>\rigs.json
python -I tools/anim_move.py <package>\anim.rpf <work>\anim-move.json
```

`tools/anim_index.py` reports `MoVE`, `0DRP`, `pack:/`, `SKEL_`, `RBF0`, `PSIN`, type-hash and 2/3 clip headers,
plus `raw:<hex>` for anything it does not recognise. It writes no payloads. `--start` must be 0 or more and
`--limit` at least 1 (decimal or `0x` hex); an out-of-range value stops with a one-line error and exit code 2
before an index is written.

## No-sharing rule

Game files, carved streams, name lists, payload dumps and reports that quote them stay on the machine that owns
the package. Do not commit them, attach them to an issue or PR, or paste them into a chat. Short header excerpts
(a few bytes) and aggregate counts are allowed, as in `CONTRIBUTING.md`. The work folder is git-ignored; this
document contains no archive data.
