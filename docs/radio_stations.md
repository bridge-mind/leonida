# Radio station records and content-id lists

**Status: layout measured, station ownership partly inferred.** This page is the companion to
[audio_config.md](audio_config.md), which covers the 1,589 music/radio records. Here are the station
descriptors and the content-id groups that reference those records.

Both structures live in the same inflated audio config at the start of `audio.rpf`: 837 zlib streams,
54,143,493 bytes. They are plain config, not scripts and not AWC audio. The companion record table
groups 1,589 music/radio records into contiguous station runs, with shared ad/DJ/jingle libraries
interleaved. Offsets below are relative to one inflated 64 KiB stream. The stream is named by its
`audio.rpf` stream offset, so `0x762160` means "the stream whose zlib record starts at archive offset
`0x762160`", not a position in the concatenated config. On this package's concatenation the same
streams begin at `0x160B419` (`0x762160`) and `0x3C55D5` (`0xDAEB0`); those concatenation offsets
change with the carve order.

Only the record side has a committed reader (`tools/audio_config.py`, described in the companion
page). Station identification needs a local hash dictionary, so this page records the measured
offsets and counts instead of adding a scanner that would have to guess station names.

## Station descriptors in stream `0x762160`

The stream holds 19 eight-byte pairs:

```text
offset  +0                 +4
0x524C  [u32 station hash]  e8 03 00 00     end = 0x03E8
0x3B94  [u32 station hash]  38 4a 00 00     end = 0x4A38
```

The second u32 is a **byte end offset in the same 64 KiB stream**, not a count, a duration or a start
time. `0x4A38` is 19,000 bytes, not 19 seconds and not 19,000 tracks. The 19 end values are exactly:

```text
0x03E8 0x07D0 0x0BB8 0x0FA0 0x1388 0x1770 0x1B58 0x1F40 0x2328 0x2710
0x32C8 0x36B0 0x3A98 0x3E80 0x4268 0x4A38 0x4E20 0x5208 0x55F0
```

They partition `0x0000`-`0x55F0` (22,000 bytes) into 19 spans. Every value is a multiple of 1,000,
so the first 22,000 bytes behave like a slot arena of 1,000-byte units: 17 spans are one unit,
one is two (`0x4268`-`0x4A38`) and one is three (`0x2710`-`0x32C8`). That slot reading is inferred;
the end offset itself is measured.

The 19 record starts run from `0x2F4C` to `0x6230`. Each record begins with the 8-byte pair and
continues with a variable-length graph of config nodes; only the next known station pair gives a clean
upper bound. The pairs are therefore not a fixed-stride array. The graph mixes content ids,
`FFFFFFFF` separators, floats and nested nodes; ids from the station's local range appear next to
shared ids. It is not a clean ordered playlist. The u32 field is not the length of this graph: it is
an end offset into the 22,000-byte span area described below, which is logically separate from the
graph after the pair.

### The hash is not `joaat("Core.RadioStation.*")`

Each station hash matches Jenkins one-at-a-time over a **bare identifier** of the form `radio_<slug>`.
The plaintext debug list at stream `0xB01000` contains `Core.RadioStation.<slug>` strings, but
`joaat("Core.RadioStation.vcrr")` is not the descriptor hash; `joaat("radio_vcrr")` is. On this
package all 19 descriptors resolve to a bare `radio_*` identifier with a candidate list. Fourteen of
the 19 bare suffixes also occur in the debug list; the other five use a different suffix, so the debug
list is a vocabulary, not a lookup table. A station-name dictionary is still required.

A second table is easy to confuse with this one. At `0x202C`-`0x20DC` there are 22 contiguous
`[u32 station hash][u32 ident]` pairs with ident values `0x2634`-`0x27CC`. Those idents are content
references, not byte offsets. Only 17 hashes occur in both tables; two descriptor hashes are absent
from the ident table and five ident hashes are absent from the descriptor table. The two tables do
not enumerate the same station set, and the 19 descriptors do not cover every station group in the
record table.

## The 19 spans

The counts below are u32 values in the `0x2000`-`0x2FFF` band, the band the local content references
use on this package. They are **reference counts, not track counts**. Each descriptor's u32 picks the
span that ends at that value; the record's own offset does not select the span. "Descriptor at" is
where the 8-byte pair is stored, not the start of the span; the pair for S01 is stored at `0x524C`,
inside the S19 range. The last column counts station-table pairs stored inside the span: 22 ident
pairs for S09, boundary descriptors for the others. That shows the spans are not exclusive station
payloads.

| Span | Bytes | Descriptor at | Size | `0x2xxx` refs | Unique | Station-table pairs inside |
|---|---:|---:|---:|---:|---:|---:|
| S01 | `0x0000`-`0x03E8` | `0x524C` | 1,000 | 8 | 8 | 0 |
| S02 | `0x03E8`-`0x07D0` | `0x3D80` | 1,000 | 4 | 4 | 0 |
| S03 | `0x07D0`-`0x0BB8` | `0x358C` | 1,000 | 9 | 9 | 0 |
| S04 | `0x0BB8`-`0x0FA0` | `0x6230` | 1,000 | 4 | 4 | 0 |
| S05 | `0x0FA0`-`0x1388` | `0x3E60` | 1,000 | 17 | 17 | 0 |
| S06 | `0x1388`-`0x1770` | `0x2F4C` | 1,000 | 19 | 19 | 0 |
| S07 | `0x1770`-`0x1B58` | `0x3F44` | 1,000 | 4 | 4 | 0 |
| S08 | `0x1B58`-`0x1F40` | `0x3664` | 1,000 | 86 | 86 | 0 |
| S09 | `0x1F40`-`0x2328` | `0x5860` | 1,000 | 112 | 89 | 22 (ident) |
| S10 | `0x2328`-`0x2710` | `0x304C` | 1,000 | 53 | 18 | 0 |
| S11 | `0x2710`-`0x32C8` | `0x4130` | 3,000 | 210 | 95 | 2 |
| S12 | `0x32C8`-`0x36B0` | `0x5784` | 1,000 | 95 | 68 | 3 |
| S13 | `0x36B0`-`0x3A98` | `0x60C0` | 1,000 | 89 | 63 | 1 |
| S14 | `0x3A98`-`0x3E80` | `0x41FC` | 1,000 | 99 | 76 | 3 |
| S15 | `0x3E80`-`0x4268` | `0x5B24` | 1,000 | 115 | 80 | 3 |
| S16 | `0x4268`-`0x4A38` | `0x3B94` | 2,000 | 121 | 79 | 0 |
| S17 | `0x4A38`-`0x4E20` | `0x37EC` | 1,000 | 41 | 16 | 0 |
| S18 | `0x4E20`-`0x5208` | `0x33E4` | 1,000 | 60 | 41 | 0 |
| S19 | `0x5208`-`0x55F0` | `0x547C` | 1,000 | 110 | 71 | 2 |

S01-S08 and S10 contain no station table. S09 contains the 22-entry ident table at
`0x202C`-`0x20DC`; that table contributes 29 of its 112 `0x2xxx` occurrences. S11-S15 and S19
contain boundary descriptors. The counts for S09, S11-S15 and S19 are therefore upper bounds, not
list lengths. The other spans are not upper-bounded by a station table, but they are still not
playlists. S16-S18 contain no station table yet remain mixed node graphs; a span being
station-table-free does not make it a playlist.

S06 shows a clean pattern: repeated `3f 80 00 00` (float 1.0) followed by a content id:

```text
0x1388  3f 80 00 00  22 27 00 00  3f 80 00 00  23 27 00 00
0x1398  3f 80 00 00  24 27 00 00  3f 80 00 00  25 27 00 00
```

That is 19 references for S06: 15 ids from its local run plus four shared ids. The rest of the stream
mixes floats, `FFFFFFFF` separators, zeros and small integers with the ids. The spans preserve local
order where a list is present, but they are not station-wide playlists.

The spans are logically separate from the record payloads. The u32 field points into the first
22,000 bytes; the graph after a pair starts at the next bytes in the stream. A span can contain other
stations' descriptors, so its content-id count is not a playlist length. The clean 80-id VCRR list
is not in this stream; it is in `0xDAEB0`.

## Content-id groups in stream `0xDAEB0`

The actual count-prefixed groups are in a second 64 KiB stream. A group is:

```text
[u32 count][count x u32 content id]
```

The start of the VCRR range is a clean example:

```text
0x668C  08 00 00 00  f2 28 00 00  f3 28 00 00  f4 28 00 00  f5 28 00 00
0x669C  f6 28 00 00  f7 28 00 00  f8 28 00 00  f9 28 00 00
0x66B0  03 00 00 00  fa 28 00 00  fb 28 00 00  fc 28 00 00
0x66C0  da 00 00 00  93 15 00 00  2b 00 00 00  10 00 01 aa
```

The count is a count of ids: `0x28F2` is stored as `f2 28 00 00`, so the ids are u32, not u16.

There are 33 count-prefixed groups in the stream, with counts 1-9 and ids in the `0x2822`-`0x29A5`
band. `0xDA` and `0x1593` appear together after 20 of the 33 groups; the exact
`0xDA, 0x1593, 0x2B` tail appears in 19, one uses `0x31`, and the other 13 have different tails. It is
a common template, not a universal constant. `0xDA` and `0x1593` do not vary with a record's duration
or with its position in the group.

The other node family links one content id to the next:

```text
[1][0x1594 or 0x15A8][0][0][0][0x2A][0xAAAA2609][u32 hash][next content id]
```

The hash on a link is not the record's `+8` hash, not its `+12` name hash and not the stream id. It
is another unresolved 32-bit value. The nodes form a graph of content references, not a table of
times.

### Why this is not a start table

No field in either family is a per-track start offset:

- The count is the number of ids that follow; it is not a byte length or a duration.
- The tail values `0xDA`, `0x1593`, `0x2B`/`0x31`, `0xAA010010`, `0x20`, `0x28` are template
  values. The same values follow groups with different ids and different total durations.
- The per-link hash is a new hash per node; it does not grow with time.
- None of the 80 prefix sums of the VCRR record durations in milliseconds occurs as a u32 anywhere
  in stream `0xDAEB0` or `0x762160`. The two non-zero known VCRR cue times (86.24 s and 647.12 s,
  stored in milliseconds) also do not occur anywhere in the 54 MiB config.
- The values that do vary are floats and small parameters around the nodes, not monotonic
  millisecond counts.

The groups may preserve a local order, and the VCRR ids are listed in ascending order, but order is
not a start time. A runtime still has to get the starts from the audio or from the missing
executable.

## VCRR: exact mapping at one base

The VCRR content range is `0x28F2`-`0x2941`: 80 consecutive ids. All 80 occur in stream `0xDAEB0`
(83 aligned occurrences; two ids repeat, one twice and one once). Six count-prefixed groups hold 33 of
them (8 + 3 + 8 + 4 + 2 + 8 ids); the other 47 occur in other node types, including 21 as linked-node
`next` values.

At base `0x9481`:

```text
0x28F2 + 0x9481 = 0xBD73
0x2941 + 0x9481 = 0xBDC2
```

Every one of the 80 ids maps to a record in `0xBD73`-`0xBDC2`, and every record in that range is
covered: **80/80, measured**. The first group maps `0x28F2`-`0x28F9` to `0xBD73`-`0xBD7A`; the
second maps `0x28FA`-`0x28FC` to `0xBD7B`-`0xBD7D`; the linked chain continues `0x28FD` to `0xBD7E`.
The base is constant across this local range.

The uncertainty is in the anchor, not the arithmetic:

- The base is chosen because `0x28F2` is the first content id and `0xBD73` is the first record of the
  presumed VCRR block. That block is contiguous, 80 records long, and its 70 longest durations fit the
  mix, but the config does not label it.
- A pure "all 80 ids resolve" search is **not unique**: many 80-record runs exist in the 1,589-record
  table, and shifting the base by one count also resolves all 80 into a neighbouring run. The
  arithmetic alone does not identify VCRR; the station-context anchor does.
- Therefore "80/80 at base `0x9481`" is measured, and "this block is VCRR's track set" is inferred.

The 80 records sum to **13,013.088 s**. The 70 longest sum to **12,043.287 s**, 39.7 s more than the
measured 12,003.6 s VCRR mix, while the 10 shortest sum to 969.8 s. That is consistent with the mix
using about 70 of the 80 records, but the config does not say which ten are omitted or trimmed. The
80-record total is 1,009.5 s longer than the mix, so no cumulative table over all 80 can be the mix.

## Shared and best-fit clusters

Six content ranges are best read as shared libraries or shared blocks rather than station-local track
blocks. All bases in this table are best-fit, not decoded:

| Content range | Record range at that base | Records present/span | Base | Evidence |
|---|---|---|---:|---|
| `0x27D1`-`0x289C` | `0xBA05`-`0xBAD0` | 204/204 | `0x9234` | best-fit shared library |
| `0x27DA`-`0x280D` | `0xB82E`-`0xB861` | 51/52 | `0x9054` | best-fit shared library |
| `0x27D6`-`0x2804` | `0xB8EC`-`0xB91A` | 47/47 | `0x9116` | best-fit shared library |
| `0x281E`-`0x283D` | `0xBD2C`-`0xBD4B` | 28/32 | `0x950E` | arithmetic exact; use inferred |
| `0x28AE`-`0x28D6` | `0xBDBC`-`0xBDE4` | 41/41 | `0x950E` | arithmetic exact; use inferred |
| `0x27D7`-`0x27F0` | `0xBF8C`-`0xBFA5` | 25/26 | `0x97B5` | best-fit shared library |

The two `0x950E` rows are the strongest: the ids map onto contiguous record runs, with four absent
ids in the first row (`0xBD3E`-`0xBD41`) and none in the second. The four large rows are best-fit
because content ids are reused across spans; the mapped ranges are compact, but another base can
produce another compact range.

The same local id range can therefore mean different records in different contexts. `0x281E`, for
example, occurs in the VCRR span, but at base `0x950E` it lands in the shared block, not in the VCRR
block at base `0x9481`. This is consistent with a station graph mixing its own local range with
shared ranges, each range keeping its own base; that reading is inferred from the ranges, not decoded
from a per-range base field.

## Title-keyed stream ids

The low 29 bits of `hash B` are the AWC stream id, and `hash B` is `joaat(normalized title)`. That
turns the record table into a title-keyed lookup even when the config's display-name field is blank.
On the current 800 resolved rows, the unmodified normalized title matches the stream id exactly for
**702 (87.75%)**; trying case, punctuation, bracket, dash and feature-credit variants raises this to
**746 (93.25%)**. The 54 misses are not evidence of absence: the stored title can differ from the
in-game name, the hash can cover a different mix or edition, and a few records are duplicate-title or
placeholder collisions.

This rule resolved four previously untitled CircoLoco records whose config names were empty; one of
the four is assigned to CircoLoco by record-id and AWC-bank adjacency rather than by a station field.
The title hash identifies a bank, not a station, not a play order and not a start offset. Keep the
station assignment and cue times as separate, evidence-labelled steps.

## What a runtime needs

| Need | Source | Status |
|---|---|---|
| Record id, duration, name hash, stream id | record table, `docs/audio_config.md` | measured, 1,589 records |
| AWC bank for a stream id | AWC manifest; `stream_id = hash B & 0x1FFFFFFF` | measured |
| Station set and span ends | 19 descriptors in `0x762160` | measured |
| Local content ids | count-prefixed groups in `0xDAEB0` | measured for VCRR; group-to-station ownership inferred elsewhere |
| Content id -> record id | base per range | VCRR base measured only at its anchor; shared bases best-fit |
| Display names | hash dictionary for `joaat` | missing; the `0xB01000` list does not hash-match |
| Play order and start offsets | not stored | missing |

To build a station, a runtime can read the descriptors, take the content-id groups once the
group-to-station mapping is known, resolve each local range with its own base, and use
`hash B & 0x1FFFFFFF` to find the AWC bank. It cannot get the mix order or the cue times from this
config. It also cannot assume that every station in the record table has a descriptor: the 19
descriptors and the 22-entry ident table are different sets.

The graph after each station pair is not a substitute for the groups: it mixes local and shared ids and
has no count or order guarantee. The groups are the only place where the VCRR range is complete.

The u32 hash at `+8` (hash A) remains unresolved. It is not the normalized-name hash at `+12`, not
the stream id, and it did not match any content id checked. The name hash at `+12` is only useful
with a name dictionary; no such dictionary is committed here.

## Measured vs inferred

Measured:

- 19 station records in `0x762160`, each starting with an eight-byte `[station hash][byte end]` pair;
  the end values are the 19 multiples of 1,000 from `0x03E8` to `0x55F0`.
- The spans and `0x2xxx` reference counts in the table, including the boundary-descriptor and
  S09 ident-table overlaps.
- 80 unique VCRR content ids, 83 aligned occurrences; 80/80 map to `0xBD73`-`0xBDC2` at base
  `0x9481`; 13,013.088 s total, 12,043.287 s for the 70 longest.
- 33 count-prefixed groups; `0xDA` and `0x1593` together in 20, the exact three-field tail in 19,
  25 groups with `0x2B` somewhere and 2 with `0x31`.
- No VCRR prefix sum appears as a u32 in the two streams; the known cue times do not occur in the
  54 MiB config.
- The six cluster ranges and their present/span counts.
- The title-keyed stream-id check: 702/800 exact `joaat(normalized title)` matches and 746/800
  with the tested case/punctuation/feature variants; 54 remain unmatched.

Inferred:

- The 1,000-byte slot arena, and that a span is an allocation range rather than an exclusive
  station payload.
- That the 80-record block at `0xBD73` is VCRR's track set, and that about 70 of the 80 are used.
- The station ownership of every non-VCRR group and every shared cluster base.
- That the `0x202C` ident table is a station-ident map rather than another content table.
- That the four title-resolved CircoLoco records belong to that station; one assignment is by
  record-id/AWC-bank adjacency, not by a station field.

## Reproduce

```powershell
python -I tools/rpf9_carve.py audio.rpf <out> --until-adat
```

(`--until-adat` stops at the first AWC bank, `0x129B000` on the measured package; `--limit 0x129B000` gives the
same window there.)

The carve writes `000000762160.bin` and `0000000daeb0.bin`, both 65,536 bytes. The record table is
`python -I tools/audio_config.py <out> --join --json`. The station scan is a u32 read of the two
streams; no station scanner is committed, because the station hash set needs a local dictionary.

## No-sharing rule

The package is read-only. Keep carved streams, the name list and any extracted names or hashes in a
local temp folder. This page and its PR contain offsets, counts, numeric ranges and the hash formula
only: no station-name list, no track title, no hash value, no carved bytes, no audio.

The same rule applies to resolution: a content id is not a global record id, and a base is not
portable. Do not publish a table that gives one global meaning to a content id, and do not reuse one
station's base for another station or for a shared block. Resolve per range and keep the station
context.
