# `RSC8` audio-bank resources (AWC tag `0x5c`)

**Status: decoded at container, object-graph, property, animation-block, track-directory and keyframe level; stdlib decoder in [`tools/rsc8_audio.py`](../tools/rsc8_audio.py).**

[story.md](story.md) measured 225,281 raw `RSC8` (version `8001` / `0x1F41`) headers in `story000.rpf`
(plus 1,957 in `shared_minigame000.rpf` and 9 in `story000_restofworld.rpf`), and the mission-music
search recorded in [audio.md](audio.md#mission-music-home-sweet-home) (#716) left `RSC8` as the only
undecoded mission-side container.

Full-package scanning resolves what `RSC8` is:

- **`RSC8` v8001 is not an audio codec or standalone audio container.**
- **641,428 of 641,428** `RSC8` resources across the package (`8,183,359,968` bytes total) sit inside
  standard little-endian **`ADAT` (AWC)** banks as **chunk tag `0x5c`**. Zero `RSC8` headers occur
  outside `ADAT` anywhere, zero `0x5c` chunks hold non-`RSC8` data, and zero `RSC8` resources appear
  under any other AWC chunk tag.
- **641,428 of 641,428** `0x5c` entries belong to an AWC stream that also carries a sibling **`0x55`**
  audio payload and **`0x76`** format chunk declaring **48 kHz mono Opus** (`codec = 0x0B`, `channels = 1`,
  `sample_rate = 48000`).
- Inside tag `0x5c`, the payload is an unencrypted RAGE **`crClipDictionary` + `crClipAnimation` +
  `crAnimationDictionary` + `crAnimation`** facial / lip-sync animation resource whose clip duration
  tracks the companion Opus dialogue line (typically with a short lip-sync blend margin). Zero `RSC8`
  resources contain an embedded audio container (`ADAT`, `OggS`/`OpusHead`, or `RIFF WAVE`).
- Consequently, `RSC8` cannot hold undecoded mission music, closing that residual item from #716.

Every figure below is an aggregate measurement from one local Xbox Series 1.0.2.1 package. No game
strings, asset names, dialogue lines, audio streams or raw payloads are committed here or emitted by
`tools/rsc8_audio.py`.

## Corpus counts

| Archive | `ADAT` banks | `ADAT` streams | Bytes in `ADAT` | `RSC8` (`0x5c`) | `RSC8` outside `ADAT` | `RSC8` tag bytes | `u32@+0x08` sum | Paired `0x55` Opus (48 kHz mono) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `audio.rpf` | 7,162 | 1,606,241 | 28,505,691,532 | 414,181 | 0 | 5,470,286,400 | 5,465,316,228 | 414,181 / 414,181 |
| `story000.rpf` | 63,861 | 234,458 | 9,423,830,212 | 225,281 | 0 | 2,680,074,288 | 2,677,370,916 | 225,281 / 225,281 |
| `shared_minigame000.rpf` | 638 | 4,334 | 359,078,504 | 1,957 | 0 | 32,927,792 | 32,904,308 | 1,957 / 1,957 |
| `story000_restofworld.rpf` | 15 | 76 | 8,908,216 | 9 | 0 | 71,488 | 71,380 | 9 / 9 |
| **Total (4 archives)** | **71,676** | **1,845,109** | **38,297,508,464** | **641,428** | **0** | **8,183,359,968** | **8,175,662,832** | **641,428 / 641,428** |

Eight additional archives were scanned in full and contain **0** `RSC8` headers: `anim.rpf`,
`packs.rpf`, `core_restofworld.rpf`, `data.rpf`, `textures.rpf`, `story000_script.rpf`, `common.rpf`,
and `script.rpf`.

Two of `audio.rpf`'s 7,162 banks are **zero-stream banks**: a complete 20-byte header with stream
count `0`, flags `0xff210001` (the `0x200000` table's `(count + 1) * 4` bytes fill the header exactly)
and no tags. The encrypted TOC lists both as `.awc` entries, and rpf9-explorer's
`AwcRecovery.ReadValidated` accepts them, which is why [design.md](design.md)'s audio row has 7,162
banks. Until the parser learned them, this table said 7,160: `parse_adat_header_bytes` rejected any
`count == 0` header as malformed. A zero-stream bank is now accepted when the flag-selected tables
end exactly at the declared header size (no stream ids or tag rows can follow, so any leftover bytes
mean the header is not what the flags say); header slop, truncation and out-of-bounds sizes are still
rejected, covered by synthetic tests in `tests/test_rsc8_audio.py`. The other three archives have no
zero-stream bank, so only `audio.rpf`'s bank and byte counts changed (banks 7,160 → 7,162, +40 bytes).

## The undocumented `0x5a`/`0xd9` companion tags

`audio.rpf` carries two tag types no other archive has, mostly on streams that hold no `0x5c` clip:
**1,185 streams** with a `0x5a` tag (804 of them also with `0xd9`; shapes `{0x36,0x55,0x5a,0x76,0xa3,0xd9}` × 803,
`{0x36,0x55,0x5a,0x76,0xa3}` × 381, and one `{0x36,0x55,0x5a,0x76,0xa3,0xbd,0xd9}`), all declaring **32,000 Hz**
(one 32,007).

`tools/rsc8_audio.py` decodes both structurally (`decode_grid20_table`, `decode_index_runs`) and validates
them per stream in its scan tallies (`grid20_*`, `indexrun_*` summary keys): over the whole archive
**387,233 `0x5a` entries / 0 invalid** and **4,853 `0xd9` groups / 0 invalid / 0 index overruns**. Every
invariant below was measured on that full population; the field names describe structure, and the meaning of
`p`, `z`, `s`, `t` and the trailer stays a labelled hypothesis.

`0x5a` — `{u32 12345678; N × 20-byte entries; f32 trailer}` (`(size − 8) % 20 == 0` always). The framing is
@kotorvr's ([finding](https://github.com/ItsNotRexed/Leonida/issues/3232#issuecomment-7084750136)), recounted
for ASSET-256; read 4 bytes later, as before, each entry paired one entry's fields with the next one's
position, and the trailer became the last "counter":

```
0    u32  position       (a sample position at the stream's rate, see below)
4    f32  value          (3.658 .. 146.789 measured; spans up to 133 within one stream)
8    u16  p              (a multiple of 512 on every measured entry)
10   u16  z              (0 .. 16)
12   u16  index_lo
14   u16  index_hi       (index pairs tile 0..129 / 1..130 measured)
16   u16  s
18   u16  t
trailer: f32             (2/3 = 0x3f2aaaab on 1,182 streams, 0.666812 on 1, 0.725624 on 2)
```

The position is a sample position in the stream (ASSET-256, all 1,185 streams and 387,233 entries;
`tools/rsc8_audio.py`'s `grid20_position_past_samples` = 0). It ascends strictly and stays below the `0x76`
sample count: the largest is 0.9986 of the stream's samples, and the last one ends p10/p50/p90 0.8/1.85/5.4
mean entry steps before the stream's end. One entry lasts one period of its `value` read as a frequency:
`(position[k+1] − position[k]) × value[k] / rate` is within 1% of 1 on 365,803 of the 386,048 entry pairs
(within 5%: 383,993), so `value` reads as a pitch in Hz. Read as a u32, the trailer is about 1.06 × 10⁹,
2^9.9–2^15.2 times the sample count, which made the old "counter" look finer than a sample. Every one of these
streams is mono XMA2 (`0x76` codec 5) at 32,000 Hz (one at 32,007 Hz).

`0xd9` — `{u32 K; K × {u32 2; u32 len; u32 hash; len × u32}}`: the run values ascend strictly inside each
group, exactly one distinct `hash` appears per stream, the group count always equals `K`, and **every run
value is smaller than the sibling `0x5a` table's entry count** (804/804), so the runs index `0x5a` entries.

### Resolution of the `docs/story.md` raw gaps

The uninflatable raw regions noted in [story.md](story.md) are unencrypted `ADAT` (AWC) audio banks:

- **`story000.rpf` GiB 5–13**: `63,861` `ADAT` banks spanning `9,423,830,212` bytes (`8.78 GiB`),
  containing `753` streamed-music (`0x48`) banks (matching #716) and `225,281` spoken dialogue streams
  with `0x5c` facial clips. The `u32@+0x08` sum (`2,677,370,916` bytes = `2.49 GiB`) matches
  [story.md](story.md) to the exact byte; the remainder of the `8.78 GiB` region is the sibling Opus
  speech audio (`0x55`), streamed-music chunks (`0x48`), and per-stream audio metadata tags.
- **`shared_minigame000.rpf` 360 MB unread gap**: `638` `ADAT` banks spanning `359,078,504` bytes
  (`99.2%` of the `362.1 MB` unread span), containing `7` streamed-music (`0x48`) banks and `1,957`
  `0x5c` facial clips (`u32@+0x08` sum `32,904,308` bytes, matching [story.md](story.md)).
- **`story000_restofworld.rpf` 8.96 MB unread gap**: `15` `ADAT` banks spanning `8,908,216` bytes
  (`99.4%` of the gap), containing `13` streamed-music (`0x48`) tags and `9` `0x5c` facial clips.

## Companion AWC speech tags

Beyond the three tags documented in [audio.md](audio.md) (`0x55` stream data, `0x76` format, `0x48`
streamed-music descriptor), speech streams carrying `0x5c` consistently carry these sibling tags:

| Tag | Role | Measured layout |
|---:|---|---|
| `0x55` | Audio stream payload | 2048-byte Opus packet blocks (`codec = 0x0B`, mono, 48 kHz) |
| `0x76` | Audio format | 32 bytes (`+0x00 u32` samples, `+0x08 u16` sample rate, `+0x1C u8` codec, `+0x1D u8` channels) |
| `0x5c` | Facial / lip-sync clip | Unencrypted `RSC8` resource (`crClipDictionary` + `crAnimation`) |
| `0x36` | Peak envelope | Array of `u16` amplitude peaks |
| `0xa3` | Opus seek table | `u32` cumulative sample offsets per 2048-byte Opus block (step `0x2D00 = 11,520` = `12 × 960` samples) |
| `0xb2` | Loudness curve | `(u32 sample_index, f32 dB)` pairs sampled every `4,800` samples (`100 ms` at 48 kHz) |
| `0xbd` | Cue list (`markers`) | 16-byte records `{u32 name hash, u32 value, u32 sample offset, u32 extra}`, no header; see [below](#the-0xbd-cue-records) |

### The `0xbd` cue records

An earlier version of this table called `0xbd` "a 16-byte header starting with magic `0xE5C74063`". It is a list
of 16-byte records with no header, and `0xE5C74063` is one cue name among 167. `tools/rsc8_audio.py` decodes the
list (`decode_marker_records`) and counts it in every scan (`marker_*` summary keys: 420,168 streams, 781,350
records, 0 invalid on `audio.rpf`). Over all of `audio.rpf`:

| Check | Result |
|---|---:|
| `0xbd` tags, records | 420,168 / 781,350 |
| tag size a multiple of 16 | 420,168 / 420,168 |
| sample offsets never decrease inside a tag | 420,168 / 420,168 |
| offset at most the stream's `0x76` sample count (the 419,223 tags whose stream has a `0x76` tag) | 750,020 / 750,020 records |
| records per tag | 1 on 88,542, 2 on 330,777, 3 or more on 849 |
| distinct name hashes | 167 |
| first record's name | `0xdbedd43c` on 330,756 tags, `0xE5C74063` on 88,082, 150 other values on 1,330 |

```
+0    u32  name     joaat of the cue name
+4    u32  value    an f32 for most names; a second name hash for region cues
+8    u32  offset   sample position in the stream
+12   u32  extra    0 on 363,166 records, 1..6 on the rest
```

The name word is 0 on 96 records (78 tags start with one); what a nameless record means is unknown.

- **The name is a joaat of a lowercase word.** Four of the stored values are the hashes of `beat` (`0xe89ae78c`,
  17,770 records), `trackid` (`0xa6d93246`, 1,042), `intro` (`0xda8f5f53`, 835) and `outro` (`0x5d40b28d`, 825),
  as the [GTA6-Asset-Info](https://github.com/TrinitronClient/GTA6-Asset-Info) byte reference reports
  (`structure/03-audio-awc.md`, commit `029e4c3`); the hashes were computed here, not copied. The names behind
  `0xE5C74063` (418,838 records, 402,357 of them on speech streams with a `0x5c` clip) and `0xdbedd43c` (330,756,
  all on such streams and all with `extra = 0`) are not known.
- **`value`** is a whole number stored as `f32` on every record of those six names. On five of the other names it
  is `joaat("start")` (`0x84dc271f`) or `joaat("end")` (`0xb0c485d7`). Matched record by record, 418 of the 420
  `start` records are followed by their `end`; see [Region cues](#region-cues-start-and-end-records).
- The same reference states that an AWC tag id is the low byte of the joaat of the chunk name. That holds for
  `markers` (`0xbd`), `data` (`0x55`), `peak` (`0x36`), `seektable` (`0xa3`) and `streamformat` (`0x48`), which
  `tests/test_rsc8_audio.py` checks, and by the same rule `0x5a` and `0xd9` are `granulargrains` and
  `granularloops`. Those spellings are this document's: the reference prints the last two as `seek_table` and
  `stream_format`, and with the underscore the rule fails (`0xb8` and `0xf9`). It does not hold for `0x5c`,
  `0x76` and `0xb2` under the names `animation`, `format` and `metadata` either, so those three names stay open.

#### Region cues: `start` and `end` records

The five names whose `value` is the hash of `start` or `end` have 841 records in 22 tags of `audio.rpf`: 420
`start`, 420 `end` and one with a third value. An earlier version of this page read the near-equal counts per name
(196/197, 101/101, 54/54, 49/49, 20/19) as region pairs without matching them. `tools/rsc8_audio.py` now matches
them (`pair_marker_regions`, the `marker_regions` block of every scan summary): inside one tag it takes the
records of one name in stored order, and a pair is a `start` whose next record of the same name is an `end` at the
same or a later offset. Every other record of such a name is counted as an exception.

| Name | Tags | `start` | `end` | Pairs | Not paired | Pair length in samples (min / median / max) |
|---|---:|---:|---:|---:|---|---:|
| `0xda559149` | 22 | 196 | 197 | 196 | 1 `end` | 172,602 / 857,823 / 3,899,077 |
| `0x1df219ec` | 1 | 101 | 101 | 101 | none | 25,289 / 95,237 / 497,606 |
| `0x77edd3a2` | 1 | 54 | 54 | 53 | 1 `start`, 1 `end`, the record with the third value | 44,984 / 90,546 / 209,958 |
| `0x994ee645` | 1 | 49 | 49 | 49 | none | 22,233 / 93,644 / 212,156 |
| `0xccb465d1` | 1 | 20 | 19 | 19 | 1 `start` | 36,146 / 92,928 / 186,375 |
| **Total** | **22** | **420** | **420** | **418** | **5 records** | |

So the reading holds for 418 pairs, and equal counts did not prove it: the 54/54 name has two unpaired records.
20 of the 22 tags pair completely (a tag holds 1 to 181 pairs, median 3). The five exceptions sit in two tags:

- `0xda559149` has a second `end` 16,717 samples after the `end` that closes the region before it (record 744 of
  its tag's 1,435). Nothing opens it.
- A `0x77edd3a2` `start` (record 40 of 834) is followed, 118,171 samples later, by a record of the same name whose
  value is `0xdaa2d5ec`, the only value on these names that is neither word. Hypothesis: that record closes the
  region under another word; the distance is inside the name's pair lengths.
- A `0xccb465d1` `start` (record 773 of the same tag, the last record of its name) is directly followed by a
  `0x77edd3a2` `end` that no `start` opens (record 774), 114,697 samples later. Hypothesis: one region whose two
  records carry different names; the distance is inside the pair lengths of both.

If both hypotheses hold, every one of the 420 `start` records has a closing record and only the second `end`
is left over. None of the five is a region left open at an edge of its tag: none is the tag's first or last
record, the nearest to an edge is 40 records in, and the unclosed `0xccb465d1` `start` is 18,219,408 samples
before its tag's last offset.

Also measured on the 418 pairs:

- No pair has a zero length, and `extra` is the same on a pair's two records (418 / 418).
- Regions of one name cannot overlap under this matching; an overlap would show as an unpaired `start` followed
  by a `start`, and there is none. Regions of different names do overlap: 54 pairs of regions share samples, 43
  with one region inside the other and 11 crossing (each starts or ends inside the other). All 54 are between
  `0xda559149` and one of the other four names (`0x1df219ec` 13 inside / 7 crossing, `0x77edd3a2` 19 / 4,
  `0x994ee645` 2 / 0, `0xccb465d1` 9 / 0); the other four never overlap each other. So a sample can lie in
  regions of two names at once.
- None of the 22 streams has a `0x76` tag (`tags_with_sample_count` is 0), so this parser has no sample count to
  compare a region's end with.

The other three archives with `ADAT` banks have `0xbd` tags as well (226,788 in `story000.rpf`, 1,959 in
`shared_minigame000.rpf`, 22 in `story000_restofworld.rpf`) and not one record with either value.

Not known: the words behind the five names and behind `0xdaa2d5ec`, what a region does in the game, and whether
the two hypotheses above are right. The lengths are in the unit of `offset`; they were not converted to seconds,
because the rate of these 22 streams is not read here.

## Binary format of `RSC8` (version `8001`)

All integers and IEEE-754 floats are little-endian. Every `RSC8` tag payload is 16-byte aligned.

### 1. Outer 16-byte wrapper (`+0x00..+0x0F`)

```
+0x00  4s   "RSC8"
+0x04  u32  version     0x00001F41 (8001) on 641,227 clips; 0x01001F41 on 201 audio.rpf clips
+0x08  u32  body_size   exact byte length after +0x0C: tag_size == body_size + 12 (641,428 / 641,428)
+0x0C  u32  flags       always 1 (641,428 / 641,428)
+0x10  ...  virtual segment (`virtual_size == tag_size - 0x10 == body_size - 4`)
```

Every internal 64-bit pointer uses virtual base **`0x50000000`** relative to **`+0x10`**:

```
file_offset = 0x10 + (u64_ptr - 0x50000000)
```

The `201` `audio.rpf` clips with `version = 0x01001F41` (`version & 0xFFFF == 8001`, flag bit 24 set)
use the identical binary structure, pointer base, animation block format, and sorted track directory
as `0x00001F41`.

### 2. Root `crClipDictionary` (`virt+0x00..virt+0xAF`)

| Virtual offset | Size | Type | Meaning |
|---:|---:|---|---|
| `+0x00` | 8 | `u64` | `0` (vtable placeholder) |
| `+0x08` | 8 | `vptr` | Block-map node pointer (`+0x08 u32 = 1`) |
| `+0x38` | 16 | `atString` | 16-byte SSO dictionary name (see §3) |
| `+0x48` | 8 | `vptr` | Clip `atMap` bucket array (`+0x50 u32 = 11` buckets, `+0x58 u32 = 1` entry) |
| `+0x68` | 4 | `u32` | `stream_hash` (`stream_hash & 0x1FFFFFFF == AWC stream_id` on 100% of `story000` / `minigame` / `restofworld` and ~99% of `audio.rpf`) |
| `+0x6C` | 4 | `u32` | Dictionary flags (`2`) |
| `+0x70` | 12 | `atArray` | `vptr` (`+0x70`), `u16 count`, `u16 capacity` (`+0x78`) of 16-byte `(vptr anim_dict, u32 hash, u32 1)` entries |
| `+0x80` | 8 | `vptr` | Secondary hash `atMap` (`+0x88 u32 = 11` buckets, `+0x90 u32 = 1` entry) |
| `+0xA0` | 8 | `vptr` | Pointer to `crProperties` table |
| `+0xA8` | 4 | `u32` | Exact byte size of the `crProperties` table |

### 3. 16-byte SSO `atString` (`virt+0x38` and `crClipAnimation+0x28`)

The same 16-byte short-string-optimized layout appears for both the dictionary name and the clip name:

- **Inline mode (`raw[15] & 0x80 != 0`)**: `length = raw[15] & 0x7F` (`<= 14`), NUL-terminated ASCII
  stored directly in `raw[0 : length + 1]`.
- **Heap mode (`raw[15] & 0x80 == 0`)**: `+0x00 vptr` to NUL-terminated ASCII string in the virtual
  segment, `+0x08 u16 length`, `+0x0A u16 capacity` (`>= length + 1`), `+0x0C u32 flags = 1`.

### 4. `crProperties` (`props_ptr`, length `props_size`)

| Relative offset | Size | Type | Meaning |
|---:|---:|---|---|
| `+0x10` | 8 | `vptr` | Pointer to the `crClipAnimation` object |
| `+0x18` | 4 | `u32` | `prop_count` (measured range `3..7`, predominantly `4` or `5`) |
| `+0x1C` | 4 | `u32` | `prop_capacity` |
| `+0x70` | `prop_count * 0x48` | descriptors | Fixed `0x48`-byte property descriptors |
| tail | variable | attributes | Contiguous typed attribute records tiling cleanly to `props_size` |

Typed attribute records begin with `(u32 key_hash, u32 type_id)`:

| `type_id` | Record size | Payload | Common key hashes |
|---:|---:|---|---|
| `0xFFFF0004` | `0x40` | String attribute (`+0x18 u64` string length) | |
| `0xFFFF000E` | `0x20` | 16-byte GUID attribute | |
| `0xFFFF0003` | `0x0C` | Boolean (`+0x08 u32`) | |
| `0xFFFF0002` | `0x0C` | Integer (`+0x08 u32`) | `0x10BF2F94` = frame rate (`60` Hz) |
| `0xFFFF0001` | `0x0C` | Float (`+0x08 f32`) | `0xAAA6AF06` = lead-in / blend (`0.166667` = `1/6` s) |

### 5. `crClipAnimation` (`clip_ptr`, `0xA0` bytes) and `crAnimation` descriptor (`anim_desc_ptr`, `0x70` bytes)

`crClipAnimation`:

| Relative offset | Size | Type | Meaning |
|---:|---:|---|---|
| `+0x28` | 16 | `atString` | 16-byte SSO clip name |
| `+0x48` | 8 | `vptr` | Back-pointer to `crProperties` |
| `+0x58` | 4 | `u32` | `clip_hash` (matches root `stream_hash`) |
| `+0x68` | 8 | `vptr` | Pointer to `crAnimation` descriptor |
| `+0x94` | 4 | `f32` | Clip duration in seconds (`== (total_frames - 1) / fps`) |
| `+0x98` | 4 | `f32` | Playback rate (`1.0`) |

`crAnimation` descriptor:

| Relative offset | Size | Type | Meaning |
|---:|---:|---|---|
| `+0x18` | 8 | `vptr` | Pointer to `crAnimationDictionary` |
| `+0x30` | 8 | `vptr` | Pointer to NUL-terminated animation name string |
| `+0x48` | 4 | `u32` | `max_block_size`: maximum `anim_size` across all packed animation blocks |
| `+0x50` | 12 | `atArray` | `vptr` (`+0x50`), `u16 block_count`, `u16 block_capacity` (`+0x58`) of `vptr` to packed `crAnimation` blocks |
| `+0x60` | 12 | `atArray` | `vptr` (`+0x60`), `u16 track_count`, `u16 track_capacity` (`+0x68`) of 12-byte track directory entries |

### 6. Packed `crAnimation` blocks and sorted track directory

Each pointer in `anim_desc+0x50` targets a packed `crAnimation` block (typically at `virt+0xB0`,
though the resource allocator occasionally places the track directory at `virt+0xB0` and the first
block after it, or splits long dialogue animations across `2` or `3` blocks):

```
+0x00  u32  signature
+0x04  u32  payload_size       always anim_size - 32
+0x08  u32  reserved           0
+0x0C  u32  chunk_data_offset  <= payload_size; where the channel definitions end and the frame data starts
+0x10  u32  anim_size          total packed block size in bytes (>= 32)
+0x14  u16  format_version     always 2
+0x16  u16  frame_count        frames in this block
+0x18  u32  extra_flags        low u16: bytes per frame; high u16 not decoded
+0x1C  u16  sub_count          sum of the type-4 channel bit widths
+0x1E  u16  track_format_tag   always 11: a u8 11 (the number of channel types) and a zero byte
+0x20  ...  one keyframe sequence (`payload_size` bytes), see section 7
```

The field names are the ones `tools/rsc8_audio.py` reports; the notes on the right are what section 7 measured.

Multi-block animations share their boundary frame between consecutive blocks, so across `N` blocks:

```
total_frames   = sum(block.frame_count) - (N - 1)
clip_duration  = (total_frames - 1) / fps
max_block_size = max(block.anim_size)
```

The sorted track directory (`anim_desc+0x60`) holds `track_count` (`90` on all `story000`, `minigame`,
and `restofworld` clips; `90`, `88`, or `87` on `audio.rpf`) × 12-byte records in ascending
(non-decreasing) unsigned `track_id_hash` order:

```
+0x00  u32  track_id_hash   sorted ascending (non-decreasing) across the directory
+0x04  u32  schema_hash     constant 0x364D3BDB = joaat("facialtranslation"), a track type
+0x08  u32  track_type      the value format: 4 = one value per frame, 3 = two values per frame
```

This is the 12-byte track entry of the clip dictionaries in `anim.rpf` (`{id, track-type hash, value format}`,
[animation.md](animation.md#objects)); `facialtranslation` is one of the track types `tools/anim_clips.py` names.

### 7. Keyframes

A packed block is one **sequence** as [animation.md](animation.md#sequences) describes it for `anim.rpf`: the
0x20-byte header, the channel definitions grouped by type, `frames x bytes per frame` of frame data, `u16
count[11]` and the per-type `(track << 2 | component)` lists. Nothing new had to be decoded: `rsc8_audio.py
--curves` hands every block to `anim_clips`' sequence decoder and counts the outcome. It writes counts only.

| Archive | Clips | Decoded | Type-10 chunk boundaries closed |
|---|---:|---:|---:|
| `story000_restofworld.rpf` | 9 | 9 | 593 / 593 |
| `shared_minigame000.rpf` | 1,957 | 1,957 | 439,313 / 439,313 |
| `story000.rpf` | 225,281 | 225,281 | 29,144,718 / 29,144,718 |
| `audio.rpf` | 414,181 | 414,181 | 58,428,044 / 58,428,044 |
| **Total** | **641,428** | **641,428** | **88,012,668 / 88,012,668** |

Over those 641,428 clips:

- The decoded frame count equals the clip's on 641,428, and the header's bit sum equals the
  type-4 widths on 641,428.
- Channels by type: type 2 (constant) 21,683,337, type 4 (quantised) 9,171,169, type 5 (table) 162,508 and
  type 10 (second-order delta) 33,239,783. No other type occurs, so a facial clip has no quaternion channel.
- All 57,706,322 directory entries get values, and the component count follows the entry's format on every one:
  format 4 decodes to one value per frame (51,159,853 tracks) and format 3 to two (6,546,469 tracks).

`decode_facial_curves` returns the per-frame values to a caller; the tool itself never prints or stores them.

## Full-corpus validation results

`tools/rsc8_audio.py` validates every resource against 24 structural integrity checks (wrapper size,
version, flags, 16-byte alignment, SSO strings, `crProperties`, `crClipAnimation`, `crAnimation`
descriptor, all packed `crAnimation` blocks, `max_block_size`, exact frame-to-duration agreement,
sorted track directory, and absence of embedded audio containers).

| Archive | Valid `RSC8` | Invalid `RSC8` | Version `0x00001F41` / `0x01001F41` | `stream_id` match / mismatch | Blocks `1` / `2` / `3` | Dict name heap / inline | Frame range |
|---|---:|---:|---:|---:|---:|---:|---:|
| `story000_restofworld.rpf` | 9 | 0 | 9 / 0 | 9 / 0 | 9 / 0 / 0 | 9 / 0 | 50–246 |
| `shared_minigame000.rpf` | 1,957 | 0 | 1,957 / 0 | 1,957 / 0 | 1,957 / 0 / 0 | 1,638 / 319 | 46–937 |
| `story000.rpf` | 225,281 | 0 | 225,281 / 0 | 225,281 / 0 | 225,273 / 8 / 0 | 216,814 / 8,467 | 17–1,706 |
| `audio.rpf` | 414,181 | 0 | 413,980 / 201 | 409,881 / 4,300 | 414,151 / 28 / 2 | 377,008 / 37,173 | 11–1,399 |
| **Total** | **641,428** | **0** | **641,227 / 201** | **637,128 / 4,300** | **641,390 / 36 / 2** | **595,469 / 45,959** | **11–1,706** |

*(Note: in `audio.rpf`, 4,300 structurally valid speech clips — 1.04% of `audio.rpf` — assign an
internal `RSC8` clip-name hash whose low 29 bits differ from the enclosing AWC `stream_id`;
`tools/rsc8_audio.py` reports `stream_id_matches` separately from structural validity. On the full
641,428-clip corpus, all 24 structural checks pass with **0** failures.)*

## What remains undecoded

- What the curves drive. The keyframes decode (section 7), but which facial control each `track_id_hash` names,
  the unit of a value and how the rig evaluates them are not established here; that belongs to the animation
  runtime, not audio extraction.
- The high u16 of a block's `+0x18` word.
- The semantic difference (if any) of version flag bit 24 (`0x01001F41`) on the 201 `audio.rpf` clips;
  structurally all 201 parse identically to `0x00001F41`.

## Reproduce

```powershell
# Run synthetic unit tests (stdlib only, no game files needed)
python -I tests/test_rsc8_audio.py

# Validate RSC8 resources in an archive (summary only, zero game strings emitted)
python -I tools/rsc8_audio.py <package>\pack\story000_restofworld\story000_restofworld.rpf --summary-only
python -I tools/rsc8_audio.py <package>\pack\shared_minigame000\shared_minigame000.rpf --summary-only
python -I tools/rsc8_audio.py <package>\pack\story000\story000.rpf --summary-only
python -I tools/rsc8_audio.py <package>\audio.rpf --summary-only

# Every summary also has the cue counts (marker_* keys) and the region pairing (marker_regions): pairs and
# exceptions per name hash, pair lengths, overlaps between names and where each exception sits in its tag.
# The numbers of "Region cues" above are the audio.rpf summary's (57 to 87 s over three runs of the 28.5 GB archive).

# Also decode every clip's keyframes and count the outcome (slower: about 40 clips per second per process;
# split a large archive with --start/--limit and add the summaries)
python -I tools/rsc8_audio.py <package>\pack\shared_minigame000\shared_minigame000.rpf --summary-only --curves

# Optional per-clip structural JSON/CSV export to a local work directory
python -I tools/rsc8_audio.py <package>\pack\shared_minigame000\shared_minigame000.rpf <work>\rsc8.csv
```

## No-sharing rule

Do not commit archives, extracted `.awc` banks, `.rsc8` blobs, Ogg Opus streams, or asset-name dumps.
`tools/rsc8_audio.py` never writes game strings or raw payloads; `tests/test_rsc8_audio.py` uses
synthesized bytes only.
