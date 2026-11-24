# Audio config music records

`audio.rpf` starts with a zlib-compressed **audio config**, then the raw AWC banks begin at archive
offset `0x129B000` (19,509,248). On the package measured here the config is 837 zlib streams and
54,143,493 inflated bytes. It is not a script resource and it is not part of the encrypted table of
contents. It holds the game's music/radio record table, the station record blocks and the mixer
configuration.

Carve the config without touching the AWC banks. `--until-adat` ends the scan at the first raw AWC bank
(an `ADAT` whose header parses as a bank, not the four bytes alone, which also occur in `METADATA`), so the
boundary isn't specific to one build; the summary in `<out>/index.json` records it as `first_adat_bank`
(`0x129B000` on the package measured here, the same window as `--limit 0x129B000`):

```powershell
python -I tools/rpf9_carve.py audio.rpf <out> --until-adat
python -I tools/audio_config.py <out> --join --json
```

`--join` scans the carved streams in name order as one stream. That matters because the config is
one logical blob split into zlib blocks, and a record can straddle a block boundary. On the
measured carve the per-file scan finds 1,588 records and the joined scan finds 1,589. The parts are
read in blocks and never concatenated (see Limits).

Under `--join` each record has two positions. JSON `offset` is the byte position in the joined
stream; `file_offset` is the position inside `file`, the part that holds the record's `0c` marker
byte. Text output prints `file:file_offset`, so a reference such as `middle.bin:0` points at the
marker in that file even when the duration sits at the end of the previous part (so `file_offset`
can be 0-3 under `--join`). Without `--join`, `offset` and `file_offset` are both the position
inside the file.

## Record layout

The music records share one 46-byte layout. A record's marker is at `+4`; the bytes before it carry
the duration:

| Offset | Size | Field |
|---:|---:|---|
| -4 | 4 | duration, milliseconds (u32) |
| 0 | 2 | `0c 01` marker |
| +2 | 2 | config id (u16) |
| +4 | 4 | `02 00 0c 00` |
| +8 | 4 | hash A (unresolved) |
| +12 | 4 | hash B: `joaat(normalized name)` |
| +16 | 4 | `00 00 02 23` |
| +20 | 22 | fixed flags/parameters (constant on this package) |

`hash B` is Jenkins one-at-a-time over the lowercased name, with runs of spaces and punctuation
collapsed to `_`. The AWC stream id is `hash B & 0x1FFFFFFF`. The duration is the real song/ad
length. `awc_manifest.py` now reports the per-channel duration: for a 1-channel streamed descriptor it
divides the interleaved sample count by the Opus packet channel count (2 when the TOC stereo bit is
set), so it matches the config instead of reporting twice.

The records are not necessarily contiguous: they are 46 bytes apart inside a run, but omitted
records leave gaps. The tool finds records by the marker and the duration range, not by a fixed
stride or a hard-coded stream offset.

## What the table gives

On the measured package:

- **1,589 records**, 1,589 distinct ids, range `0xB422`–`0xBFC6`, 272,442.8 s total.
- **1,584 of 1,589** durations matched the AWC stream length within 0.01 s. The original
  comparison halved the legacy manifest values to get the per-channel duration. The current
  manifest reports the per-channel duration directly. The five exceptions are duplicate-title
  or placeholder hash collisions.
- **All 1,589** records map to an AWC stream id; the ids are the music streams exported by
  `tools/awc_opus.py`.
- The records are grouped by contiguous id ranges that follow station order. The same config also
  holds station records and station-local content-id lists; they are not part of this record table
  and this tool does not embed the package's ranges.

It does **not** give per-track start offsets. Id order is not mix order, and the schedule-like
structures in the config carry constant template fields, not cue times. A continuous radio mix must
be matched in the audio (a music page under docs/ if one is added, or the local radio-timing tools), except where
the bank itself carries track markers: see [Marker events](#marker-events-awc-tag-0xbd) below.

## Marker events (AWC tag `0xBD`)

The cue times the config lacks are partly in the banks. AWC tag `0xBD` is a headerless array of
16-byte cue records; [rsc8_audio.md](rsc8_audio.md#the-0xbd-cue-records) has the layout over all 420,168 tags of
`audio.rpf`, counted by `tools/rsc8_audio.py`. This section is about the 945 of them in the streamed-music banks,
which that count does not separate, and about what their records mark. The fields are the same four; the names
used here (with the other page's in brackets) are type (`name`), value, time (`offset`) and flags (`extra`).

In a streamed-music bank (one with a `0x48` descriptor, [audio.md](audio.md#streamed-music-0x48))
the tag is the only tag of a channel's own stream-table entry: the entry whose id is the descriptor's channel id.
The `0x48` stream itself (id 0) never has one. Those entries have no `0x76` format tag, so the check of the
offsets against the `0x76` sample count on the other page does not reach them; here they are checked against
the descriptor's length.

| Offset | Size | Field | Confidence |
|---:|---:|---|---|
| 0 | 4 | type (u32 name hash) | measured: 17 distinct values over the music banks, table below |
| +4 | 4 | value (f32; for six types a second u32 name hash) | measured per type; what the number means is a hypothesis except where the table says otherwise |
| +8 | 4 | time (u32), 48 kHz frames per channel from the start of the stream | measured: never decreases inside a chunk and never passes the stream length (945 of 945); the unit follows from that, see below |
| +12 | 4 | flags (u32) | measured: 0 in all 31,330 music events; meaning unknown |

One event, with made-up values (type `0x11111111`, value 1234.0, time 96,000 = 2.0 s, flags 0):

```text
11 11 11 11  00 40 9a 44  00 77 01 00  00 00 00 00
```

**The time unit.** The stream length is the descriptor's interleaved sample count divided by the packet
channel count (the record table's duration). Against that length the last event of a chunk is at a median
0.997 of the stream and at 0.99 or more in 771 of 945 chunks, on 934 stereo and 11 mono streams, and it is
never past the end. Interleaved samples would run to twice the length on the stereo streams, and
milliseconds would stop at 2% of it, so the unit is per-channel frames at 48 kHz. All 3,116 descriptor
channels declare 48,000 Hz.

### Whole-corpus counts

Every `ADAT` bank of `audio.rpf` (7,162 banks, 1,606,241 streams), split by whether the bank has a `0x48`
descriptor:

| | Streamed-music banks | Other banks |
|---|---:|---:|
| Banks | 3,008 | 4,154 |
| Banks with a `0xBD` chunk | 938 | 1,019 |
| `0xBD` chunks | 945 | 419,223 |
| Events | 31,330 | 750,020 |
| Chunk size not a multiple of 16 | 0 | 0 |
| Events per chunk (min / median / max) | 1 / 23 / 1,435 | 1 / 2 / 3 |
| Chunks where the time decreases | 0 | 0 |
| Chunks whose last time is past the stream length | 0 of 945 | 0 of 419,222 |
| Distinct types | 17 | 153 |
| Events with non-zero flags | 0 | 418,184 |

- 928 of the 2,958 one-channel music banks have a chunk, and 10 of the 50 multi-channel banks (17 chunks; in the
  four banks with more than one, every channel holds the same events).
- 923 of the 945 music chunks sit on a stream whose id is one of the 1,589 config records.
- The streams with a chunk run from 25.5 s to 7,231.7 s (median 224.0 s; 17 are under 60 s). Of the 2,070
  music banks without one, 1,713 are under 60 s and 106 are over 120 s: the short idents and stings
  ([awc_shorts.md](awc_shorts.md)) mostly have no markers.
- The finding that reported this chunk counted 873 of 2,684 banks, 16 types, 16,207 events of type
  `0xE89AE78C` and 981 of type `0xA6D93246` in 849 banks. Those six numbers are exactly the streamed-music
  banks 323 to 3,006 of the 3,008 in archive order, so that count seems to have left out the first 322 banks and
  the last 2 (why is not known);
  the layout it gives is the one measured here.

### Event types in the music banks

| Type | Events | Chunks | Chunks with one | Most in a chunk | Value |
|---|---:|---:|---:|---:|---|
| `0xE89AE78C` | 17,770 | 809 | 21 | 868 | integers 0 to 14,459 (314 distinct; 8, 4 and 16 are the most common) |
| `0x74E06FAC` | 3,139 | 778 | 67 | 96 | 0.0 |
| `0x08DBA0F8` | 2,969 | 755 | 0 | 4 | one of 4 name hashes |
| `0x3A423FE8` | 2,322 | 681 | 51 | 81 | 0.0 |
| `0x0C801D6C` | 1,561 | 656 | 250 | 45 | 0.0 |
| `0xA6D93246` | 1,042 | 910 | 905 | 41 | integers 1,001 to 15,102 (865 distinct): a track number |
| `0xDA8F5F53` | 835 | 804 | 801 | 16 | 0.0 |
| `0x5D40B28D` | 825 | 797 | 795 | 15 | 0.0 |
| `0xDA559149` | 393 | 22 | 0 | 161 | one of 2 name hashes |
| `0x1DF219EC` | 202 | 1 | 0 | 202 | the same 2 name hashes |
| `0x77EDD3A2` | 109 | 1 | 0 | 109 | the same 2, and a third once |
| `0x994EE645` | 98 | 1 | 0 | 98 | the same 2 name hashes |
| `0xCCB465D1` | 39 | 1 | 0 | 39 | the same 2 name hashes |
| `0x00000000` | 15 | 13 | 12 | 3 | 0.0 |
| `0x7A495DB3` | 6 | 6 | 6 | 1 | 113.0, 131.0 or 158.0 |
| `0x2AC28E32` | 4 | 4 | 4 | 1 | 0.0 |
| `0x12067E90` | 1 | 1 | 1 | 1 | 0.0 |

The types are `joaat` name hashes (`tools/audio_config.py`, the helper the record table uses). Hashing plain
lowercase words: the seven the finding proposed all hit (7 of 7), and 89 further words tried here gave two
more type hits, so 9 of the 16 non-zero types resolve and 7 do not; type 0 is the hash of the empty string.
The value hashes of the six hash-valued types are 7 distinct values, and 6 of them resolve the same way; the
seventh occurs once. This page keeps to the hashes: the words are generic marker labels, none is a track or
station name, and the tool prints none.

### Track markers (`0xA6D93246`)

The type is the hash of `trackid`, one of the four names [rsc8_audio.md](rsc8_audio.md#the-0xbd-cue-records) gives.
Measured:

- 905 chunks carry exactly one, at time 0 in 900 and within 0.35 s of it in the other five. 35 chunks carry none.
- Five chunks carry 22, 24, 25, 25 and 41, on streams of 2,386.0 s, 3,004.7 s, 5,014.1 s (two streams with
  different ids and different other events) and 7,231.7 s. These are the long mixes; the first marker is at
  0.0 s in four and at 0.9 s in one, and consecutive markers are 3.3 s to 325.6 s apart.
- 64 of the 137 markers in those five chunks have the `value` of a one-marker bank (32 of 41, 22 of 22,
  10 of 24, 0 of 25, 0 of 25), so the value links a stretch of a mix to the standalone bank of the same track
  number. 101 values occur in more than one chunk.

Not confirmed:

- **That a marker is the audible start of the track.** The finding checked that against the decoded audio
  (most markers within 3 s of a strong spectral change); that was not repeated here.
- **The length of a stretch.** For the 64 linked markers, the stretch up to the next marker (or the end of the
  stream) is within 1 s of the linked bank's length for none and within 5 s for 6; it is shorter for 59, at a
  median 0.63 of it. Without the link, 41 of the 137 stretches are within 0.05 s of *some* config duration, but
  so are 47 of 137 stretches lengthened by a random 20 to 40 s, so that is chance. A mix plays part of a
  track, or the marker is a cue and not a cut: the record table's durations do not give the mix positions,
  and the markers do not come from them.
- What the other types mark, and the meaning of their values.

So the statement above stands for the config, and for the mixes without markers. For the five marked mixes a
runtime can read the track boundaries and track numbers from the bank.

### Other banks

The 419,223 tags outside the music banks are the ones [rsc8_audio.md](rsc8_audio.md#the-0xbd-cue-records)
describes, and the counts here agree with that page where they overlap (945 + 419,223 = 420,168 tags,
31,330 + 750,020 = 781,350 records, 17 + 153 types with three in both groups = 167, 363,166 records with
flags 0). Two things it does not have:

- In the 418,838 records of type `0xE5C74063` the value is an integer 0.0 to 6.0 and the flags field equals
  that integer (418,838 of 418,838). These are the only non-zero flags in the archive.
- The 330,756 records of type `0xDBEDD43C` are all at time 0, with integer values 33 to 1,467.

The three types found in both groups are `0x00000000`, `0x0C801D6C` and `0x7A495DB3`.

### Reproduce

```powershell
python -I tools/awc_markers.py <game>\audio.rpf --out <temp>\awc_markers.json
```

`tools/awc_markers.py` walks the archive for `ADAT` banks with the header parser of `tools/rsc8_audio.py`
(the same 7,162 banks), jumps over each parsed bank and reads only the `0xBD` chunks and the stream lengths:
under a minute for the 28.5 GB archive on an NVMe drive (10 s once the file is cached). It prints the two
groups and their 20 most frequent types; the JSON has every type with its events, chunks, value range and,
for a type with at most 8 distinct values, the value bits. It also takes one `.awc` bank or a directory of
them. The report is counts, hashes and numeric ranges; it has no names and no audio.
`tests/test_awc_markers.py` builds its banks by hand.

Not reproduced by the tool: the track-marker cross-checks (the link by `value`, the stretch lengths, the
comparison with the config durations) and the name-hash trials. They joined the decoded events with
`audio_config.py --join --json` in a local script that is not committed, because it works on per-event rows:
link = equal `value`, stretch = next marker time minus this one, tolerances as stated.
be matched in the audio itself; no tool in this repository does that yet. The station graphs in
[radio_stations.md](radio_stations.md#why-this-is-not-a-start-table) are not a start table either.

## Tool

```text
python -I tools/audio_config.py <carve dir or blob> [--json] [--names FILE] [--join]
                                 [--min-ms N] [--max-ms N]
```

- `scan(blob)` returns `offset`, `id`, `duration_ms`, `duration_s`, `hash_a`, `hash_b`,
  `stream_id`. `scan_stream(chunks)` yields the same records for byte chunks read in order,
  without joining them; its `offset` is the joined position.
- `collect(src, names=..., join=...)` scans a blob or a directory and adds `file` and
  `file_offset`; `names` maps hash B to a display string when a NUL-separated local name pool is
  supplied.
- `joaat` and `normalized_name` are the shared hash helpers. `joaat` is the lowercased **latin-1** byte hash
  that `tools/hashdict.py` and `tools/awc_names.py` use, so a name hashes alike in every tool; a non-ASCII
  character is one byte (`é` = `0xE9`), not its UTF-8 pair. `tests/test_joaat_agreement.py` holds every copy in
  `tools/` to that rule.
- Text output prints one `file:file_offset` line per record and a summary; `--json` writes
  `{records, summary}`.

`--names` is optional. The package's debug name stream and the localization credits are local game
files; nothing from them is committed. Its strings are read as latin-1 bytes, the same bytes `joaat`
hashes, so a title with an accented character resolves to the hash a record carries.

## Limits

- The table is record-level. It does not say which record is a song, a DJ line, an ad or a jingle;
  duration and the contiguous station ranges are the only built-in signals.
- Hash A is unresolved, and hash B resolves only through a supplied name pool or an AWC manifest.
- The first ~19.5 MiB bound is specific to this package. A different build can put the AWC start at
  another offset; `rpf9_carve.py --until-adat` finds it (the first AWC bank whose header parses).
- No start offsets, no play order and no playlists are stored here. The config's station content
  lists reference records, but the mix order is assembled at runtime. Five long mixes carry track
  markers in their own bank (tag `0xBD`, above).
- A marker is reported only when at least 20 bytes follow it in the scanned input, as before.
- `--join` memory: at most two read blocks of input (`JOIN_BLOCK_BYTES`, 1 MiB each) plus a
  24-byte carry, one name and start offset per part, and the record list. The non-joined scan
  reads one whole part at a time, and a single blob is read whole.
