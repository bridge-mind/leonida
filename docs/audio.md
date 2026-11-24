# Audio

`audio.rpf` is one of the few archives that is not mostly zlib. It holds **AWC** banks, and each bank uses the
little-endian **`ADAT`** container. The build exports the banks whole, then repackages or decodes their streams:

- Opus (codec `0x0B`) becomes standard **Ogg Opus**, losslessly: the stored packets are rewrapped, not re-encoded.
- PCM (codec `0` and codec `2`) becomes **WAV**.
- XMA2 (codec `5`) becomes a RIFF XMA2 file, which **ffmpeg** decodes to WAV.

This page describes the container, what each tool writes, and the bank types that are easy to misread when listening.
The corpus counts and verification results are in [design.md](design.md#asset-status-no-exe-needed).

## The AWC (`ADAT`) container

All integers are little-endian.

```
offset  size  field
0x00    4     "ADAT"
0x04    u32   flags
0x08    u32   stream count (n)
0x0C    u32   header size
```

Optional tables follow the header, selected by `flags`. `tools/awc_opus.py` skips the ones seen in the package:

| Flag | Extra bytes before the stream table |
|---|---|
| `0x00400000` | 4 |
| `0x00010000` | `n * 2` |
| `0x00200000` | `(n + 1) * 4` |

Then comes the stream table, one `u32` per stream:

```
bits 31:29  tag count (0-7)
bits 28:0   stream id
```

Each stream's tags follow, 8 bytes each:

```
bits 63:56  tag type
bits 55:28  size
bits 27:0   offset (from the start of the bank)
```

The exporters use three tag types:

| Tag | Meaning |
|---|---|
| `0x55` | stream data |
| `0x76` | format |
| `0x48` | streamed-music descriptor |

`tools/rsc8_audio.py` additionally decodes the speech companions `0x36` (peaks), `0xa3` (seek
table), `0xb2` (loudness), `0xbd` (cue records) and the two structural tags below; every other tag type
is skipped. The `0xbd` record layout is in [rsc8_audio.md](rsc8_audio.md#the-0xbd-cue-records); the music banks
have the tag too (945 tags in 938 of the 3,008 streamed-music banks, counted by `tools/awc_markers.py`), and
what their records mark is in [audio_config.md](audio_config.md#marker-events-awc-tag-0xbd).

| Tag | Meaning |
|---|---|
| `0x5a` | 20-byte entries, a sample position and a pitch each (`12345678` marker; see [rsc8_audio.md](rsc8_audio.md#the-undocumented-0x5a0xd9-companion-tags)) |
| `0xd9` | grouped ascending index runs into the sibling `0x5a` entries |

Both appear only in `audio.rpf`, on 1,185 streams (32,000 Hz; 804 of them carry both).

### Format tag (`0x76`)

The fields the exporters read:

```
+0    u32  sample count (Opus: 48 kHz frames, interleaved across channels)
+8    u16  sample rate
+12   u32  Opus per-channel frame count (codec 0x0B; +0 for mono, +0/2 for stereo)
+28   u8   codec
+29   u8   channel count
```

The exporters ignore the remaining bytes.

### Codecs

| Codec | Format | Export |
|---|---|---|
| `0` | PCM, 16-bit | `.wav`, 16-bit PCM |
| `2` | PCM, 32-bit float (4 bytes per sample; values can exceed ±1, e.g. impulse responses) | `.wav`, IEEE float |
| `0x0B` (11) | Opus, 48 kHz | `.ogg`, Ogg Opus, lossless repackage |
| `5` | XMA2 | `.xma.wav`, RIFF XMA2; ffmpeg decodes it to PCM WAV |

Opus data is stored in 2048-byte blocks. Each block is a sequence of `[u16 length][packet]`; a zero length pads
the rest of the block. The first 312 decoded samples are the encoder's lookahead. Measured on 30 real streams from
four banks: decoded with pre-skip 0, the 5% and 10% attack points are at 312 samples (median 312; the 10% point
ranges 278..868, none near 1920), so 312 is the exporter's default. The AWC does not store the value, so
`--pre-skip` overrides it for a stream encoded with a different lookahead.

The exporter writes that value as `OpusHead` pre-skip and emits the EOS page at the first packet that reaches
`pre-skip + declared samples`; later packets are encoder padding and are dropped. Clamping only the last page
would let its granule go backwards when the trim point falls before that page starts, and decoders would keep the
excess: a 27-packet real-libopus fixture decoded 24,960 frames for 24,000 declared, while the reaching-packet
form decodes exactly 24,000. An oversized `--pre-skip` is clamped to the stream length so the final granule never
falls below it.

The format tag's sample count is already in 48 kHz frames even when its rate field is 32 kHz, and `+0x0C` is the
per-channel frame count (equal to `+0` for mono and half of it for stereo in all 7,451 measured streams, 0.5% of
the 1.6M-stream corpus). The exporter uses `+0x0C` only when it agrees with `+0` (`frames * channels == samples`)
and otherwise falls back to `+0 / channels`, so a malformed block cannot silently truncate audio. It never scales
either count by the rate field, because scaling a 32 kHz stream by 48000/32000 leaves padding untrimmed.

Codec `5` is not handled by `tools/awc_opus.py`. `rpf9dump xma` writes it as RIFF XMA2, and ffmpeg does the decode.

### Streamed music (`0x48`)

A `0x48` tag describes a streamed mix. Its descriptor is:

```
+0    u32  block count
+4    u32  chunk size
+8    u32  channel count
then 16 bytes per channel:
+0    u32  channel id (low 29 bits)
+4    u32  sample count
+10   u16  sample rate
+12   u8   codec
```

At the `0x55` data offset, each chunk starts with a 24-byte record per channel. The record's `+4` word is the
channel's subblock count and `+8` is a repeated-prefix sample count to skip. The channel data are its run of
2048-byte subblocks after the 2048-aligned header. The skipped samples repeat the previous chunk's tail. The
exporter verifies that the skip ends on a packet boundary and that the repeated packets match the previous
packets for that channel group, then writes each verified overlap once. Each descriptor channel becomes one Ogg
Opus file. Invalid repeat counts or payloads reject that music descriptor so independent streams can still export.
The chunk size gives the distance between blocks. The final block may use fewer bytes. The channel table and
all declared subblocks must fit within their own block and the `0x55` tag's data range.

The descriptor's `channels` field is the number of channel groups, not necessarily the packet channel count.
An Opus packet's TOC stereo bit (`packet[0] & 0x04`) gives the real count, and the descriptor's `samples` field
counts interleaved samples across those packet channels. The exporter therefore writes a stereo Ogg with
`OpusHead channels=2` and `samples / 2` as the end granule when the packets are stereo; mono groups keep their
count unchanged. Reading `samples` as a frame count produces a file that reports twice its real duration.

## What the exporters write

`rpf9dump` puts output under `<out>/<archive basename>/`. For `audio.rpf`, that is `<out>/audio/`.

| Command | Output |
|---|---|
| `rpf9dump audio <audio.rpf> <out>` | `<out>/audio/awc/<path>.awc` (the original banks) and `<out>/audio/audio_manifest.json` |
| `rpf9dump xma <audio.rpf> <out>` | `<out>/audio/xma/<bank>/<stream id>.xma.wav` for every codec-5 stream |
| `python -I tools/awc_opus.py <bank or dir> <out> [--jobs N] [--pre-skip N]` | `<out>/<bank>/<stream id>.ogg`, or `.wav` for codec 0/2; `--jobs` processes that many banks at once |
| `python -I tools/opus_tags.py <ogg dir> --names <csv> [--dialogue <csv>]` | the same `.ogg` files, with names, dialogue labels and subtitles in their `OpusTags` header (see *Tags: names and subtitles*) |
| `python -I tools/audio_wav.py <ogg dir> --out <wav dir> [--jobs N]` | 16-bit PCM WAV mirroring the Ogg tree; `--jobs` ffmpeg processes at once |

When exporting a directory, `awc_opus.py` keeps the path to each bank below the input root. For example,
`a/music.awc` writes under `<out>/a/music/`. Passing that file on its own writes under `<out>/music/`.
To regenerate a flat export from an earlier version, run `awc_opus.py` with the original bank directory and a fresh output directory.

To import music cues into Audacity, run
`python -I tools/awc_cue_labels.py <bank>/cue_regions.json <new-label-directory>` on a sheet
exported by `awc_opus.py --cue-sheet`. Open the channel Ogg in Audacity at time zero, then use
**File > Import > Labels** to load the matching `.txt` by filename stem. Each region shows its
hexadecimal cue hash followed by its one-based row number in the bank sheet. Overlapping and
repeated regions keep individual labels. Start and end times use nine decimal places in seconds
so rounding back to the nearest 48 kHz frame recovers both endpoints.
[Audacity label file format](https://manual.audacityteam.org/man/importing_and_exporting_labels.html)

`audio_manifest.json` is the bank index: each bank's `Path`, `Offset`, `Size`, and its `Streams` with `Index`,
`Id` (hex), `Codec`, `Channels`, `SampleRate`, `DurationSeconds` and `IsStreamed`. For streamed music,
`Channels` is the Opus packet channel count and `DurationSeconds` is per channel: the tag `0x48` descriptor's
interleaved sample count is divided by that channel count.

`audio_wav.py` runs ffmpeg with `-c:a pcm_s16le` at the stream's own rate and channel count. For a single stream
it trims to the final Ogg Opus granule minus pre-skip. Chained streams have a separate granule timeline for each
link, so ffmpeg handles their end trimming. Existing WAVs are normally skipped on resume. If the destination
filesystem treats two output names as the same file, the converter keeps the input extensions and adds a counter
where needed. It leaves a previously shared WAV untouched. Before reusing a name ending in `.ogg.wav` or `.opus.wav`
(including counters such as `.ogg~2.wav`), it decodes the source again and compares the PCM, rate and channel
layout. These checks read the full streams serially during planning. If the audio differs (including decoding
differences between ffmpeg versions) or the existing WAV cannot be verified, it stays in place and the converter
tries the next counter. Plain existing WAVs are
trusted by name, so resuming into the same output folder assumes the same set of inputs.
With no `--out`, a directory writes to `<src>/wav` and a single file writes next to itself.

## How `make_game.py --wav` uses them

The `assets` stage runs these in order:

```
rpf9dump audio   -> <work>/assets/audio/awc + audio_manifest.json
awc_opus.py      -> <work>/assets/audio/ogg
audio_wav.py     -> <work>/assets/audio/wav        (only with --wav)
rpf9dump xma     -> <work>/assets/audio/xma/<bank>/<stream id>.xma.wav
ffmpeg           -> <work>/assets/audio/xma/<bank>/<stream id>.wav
```

`--wav` is opt-in because 16-bit PCM is several times larger than Ogg Opus. It only adds PCM copies of the Ogg Opus
output; the XMA2 decode is separate and runs independently when ffmpeg is on PATH. If ffmpeg is missing, the
`.xma.wav` files are kept and the build says so. The stage skips work that is already present, including existing
WAVs.

## Songs, station mixes and venue stems

A bank is an asset container, not a radio station. The same `ADAT` framing holds several different things:

| What you see | What it is |
|---|---|
| One stream, typically around five minutes | An individual song or music asset, not a station feed |
| A long continuous stream | A station-style mix, much longer than a single song |
| A bank with 3 or 8 channels | **Venue stems**: the music plus the audience/ambience channels for one venue |

The multi-channel banks are the easy mistake. A 3-channel bank is not three radio stations, and an 8-channel bank is
not an eight-station lineup: it is one venue's mix split into stems, so the game can balance the music against the
crowd. `awc_opus.py` writes one mono Ogg per channel and does not mix them. Listening to a single stem can sound like
a dry music bed or like pure audience noise; the in-game result is the mix.

Use the bank name, stream count, duration and channel count together before deciding what a bank is. Station
programming is assembled by the game from many assets, so a lone AWC is rarely a complete station.

The music/radio **config table** is a separate structure: 1,589 records with exact durations, AWC stream ids,
contiguous station grouping and no per-track start offsets. See [audio_config.md](audio_config.md) for the record
layout, the station-content references and the marker events (tag `0xBD`) that mark track boundaries in five
long mixes.

## Mission music: Home Sweet Home

The credited "Burn This Shit Down" is not a radio-station track. It is the song for the
**Home Sweet Home** mission (Chapter 1, mission 21/25): Lucia at Jason's house in Key Lento
during the house fire and the escape. The mission and its localized objective/fail strings are
present in `data.rpf`.

The song audio is in 1.0.2.1 too, as stems, not as one stream (found by @jefftern4 in the coordination thread,
PROC-178). It is the plain (not streamed) `audio.rpf` bank at offset `0x300695000`, 9,481,280 bytes, bank 5,573 of
7,162 in the exporter's `audio_manifest.json` order:

| Field | Value |
|---|---|
| Streams | 73, all stereo, all codec `0x0B` (Opus, 48 kHz) |
| Common length | 47 streams declare 771,213 samples (16.067 s); the manifest's rounded `DurationSeconds` gives 771,213 for 32 of them and 771,214 for 15 |
| Other lengths | 6 × 1,156,820, 2 × 2,313,640 and 2 × 385,607 samples (1.5, 3 and 0.5 times the loop), and one 24,100-sample stream |
| Range | 0.439 s to 114.477 s, 1,297.2 s in total |

The 47 equal-length streams are loop stems meant to play together (vocals, melodic, sub-bass and ad-lib layers per
slot, by the finder's listening), so no single stream is the song and a whole-file fingerprint of each stream
misses it. That is why the searches below came back empty. The bank's shape (field values above) was re-measured
from `audio_manifest.json` on a second 1.0.2.1 copy; the stem roles are the finder's and were not re-checked here.

Measured coverage of the earlier search, which looked for the song as one stream. The `audio.rpf` all-banks row
is the one the stems fall under:

| Container | Searched | Result |
|---|---|---|
| `story000.rpf` streamed-music banks | 753 banks / 2,642 channels | none |
| `story000.rpf` plain banks | 201 banks / 1,950 streams | none |
| `story000.rpf` Bink | 264 streams / 162 with audio / 165 tracks / 5,090 s | none |
| `movies.rpf` Bink | 229 tracks / 12,401 s | none |
| `audio.rpf` streamed cues under 30 s | 1,149 channels / 5.78 h | none |
| `audio.rpf` all banks | 7,162 banks, all streams | none as a single stream; the stems are bank `0x300695000` (above) |
| loop/chunked banks | 1,099 banks / 118,180 streams | none |
| `packs.rpf` inflated | 144,364 files / 9.74 GB | 0 AWC, 0 Bink2 |
| `data.rpf` inflated | 28,509 files / 1.41 GB | mission text, 0 song strings |
| `script.rpf`, `common.rpf`, `story000_script.rpf` | 1,058 / 955 / 2,633 files | none |
| `levels.rpf` raw | 94 `ADAT` magics, 2,820 `KB2` patterns | 0 valid AWC/Bink |

Fingerprint search: about 19,807 files / 19.9 M Chromaprint frames / 653 fingerprint-hours, with a
0.50-2.00 tempo sweep over the story-music and long-AWC targets. No candidate passes the vote
threshold. The positive controls keep their earlier results: Sensimilla 2,284 votes / 0.911 bits
/ 100 %; Don't Break My Heart 88 votes / 7.046 bits / 94.5 %.

A `data.rpf` hash table contains 12 occurrences of `joaat("burn")`, but it is not the song: the
config blob has none, no AWC stream ID matches its low 29 bits, and the table has no audio
reference.

Residual uncertainty: `levels.rpf` (76 GB) was not fully inflated; the retail track may be a third
master or patch content not covered by the public references; sub-30 s stings were not exhaustively
fingerprinted. No game audio is stored in the repository; the searches ran on a local copy and
`radio-dump-v3` was not modified.
Residual uncertainty: how the game sequences the stems (which layers play in which slot, and the
order of the longer streams) is not decoded; the mission's audio config is the likely place (a
hypothesis, not measured). No game audio is stored in the repository; the searches ran on a local
copy and `radio-dump-v3` was not modified.

This search first listed the `RSC8` resources of the story and minigame archives as the only
mission-side container not decoded. They are decoded in [rsc8_audio.md](rsc8_audio.md), and none of
them holds the song: all 641,428 raw `RSC8` headers in the package (225,281 in `story000.rpf`, 1,957
in `shared_minigame000.rpf`, 9 in `story000_restofworld.rpf` and 414,181 in `audio.rpf`) are AWC tag
`0x5c` facial/lip-sync clips. Each sits in an `ADAT` bank next to the 48 kHz mono Opus dialogue line
it animates, none lies outside a bank, and none contains an `ADAT`, Ogg Opus or RIFF WAVE header.

## Stream names

The stream table stores only `joaat(lowercase name) & 0x1FFFFFFF`, so the exporters name files by id.
`tools/awc_names.py` recovers names by hashing dictionary words against those ids. With about 1.8 million distinct
ids in a 2^29 space, roughly 0.3% of arbitrary words collide by chance, so a hit is accepted only under one of two
rules:

| Rule | Accepted when |
|---|---|
| `dialogue` | the word is shaped like a dialogue wave, `<root>au_<4 letters>_<NN>`, and is in the dictionary exactly |
| `family` | one base (a name without its trailing `_<number>`) hits ids in the same bank with several numbered suffixes (`''`, `_00`-`_40`, `_1`-`_9`): at least 3 members, or 2 with adjacent numbers. Every token of the base occurs in at least 5 dictionary words, and at least one member is itself a dictionary word |

Each run repeats the attack with `zq` prepended to every word. Those names can't be right, so what the control
accepts estimates the false names. Per source, the dialogue rule is kept only when its control is under 2% of its
real count, and the family rule takes the smallest family size whose control is under 2%.

The dictionary comes from two kinds of input:

- `--words`: plain word lists, or a `hashdict.py` JSON (its values).
- `--conversations`: conversation data, which pairs a dialogue key with its line labels as
  `<ROOT>AU_<XXXX>\0<LABEL>\0`. A key with N distinct labels yields the waves `<key>_01` .. `<key>_<N>`. Wave `_01`
  speaks the first label, `_02` the second, and so on. In 1.0.2.1 the pairs are in 594 carved `story000.rpf`
  streams, 4 `shared_minigame000.rpf` streams, 2 `data.rpf` streams and 1 `story000_restofworld.rpf` stream, and
  they give 138,104 wave names.

Measured on the 1.0.2.1 package (2026-11-22) with 2,455,498 distinct lowercase words: identifiers harvested from the
carved archives (by a scan like `hashdict.py`, not `hashdict.py` itself) plus the conversation waves. The columns are
the ones the tool prints per source, so the second one is its distinct bank/ID pairs: an id that appears in two banks
counts once in each (`Index.streams` in `tools/awc_names.py`):

| Archive | Banks | Distinct bank/ID pairs | Named | Dialogue | Family | Control (expected false) |
|---|---:|---:|---:|---:|---:|---:|
| `audio.rpf` | 7,162 | 1,606,237 | 92,043 | 0 | 92,043 | 352 |
| `story000.rpf` | 63,861 | 234,458 | 173,800 | 120,511 | 53,289 | 62 |
| `shared_minigame000.rpf` | 638 | 4,334 | 1,957 | 1,957 | 0 | 3 |

`story000.rpf` is 74.1% named: dialogue is named almost entirely, because conversation data lists every key.
`audio.rpf` is 5.7% named. Its sound effects, ambience and music have no list, so only numbered families are
found. Without the one `story000_restofworld.rpf` conversation stream, 1,019 dialogue streams stay unnamed. When two
accepted families claim the same id in one bank (3 rows), the first family in sorted base order keeps it.

The output names game files: keep it local, and do not commit it or attach it to an issue.

```sh
python -I tools/hashdict.py <carve dir> --out <temp>/hashes.json
python -I tools/awc_names.py <work>/assets/audio/awc --words <temp>/hashes.json \
    --conversations <carve dir>/story000 --out <temp>/awc_names.csv
```

Each positional source is reported and thresholded on its own. A source is a directory of `.awc` banks, a single
bank, or an `audio_manifest.json` from `rpf9dump audio` (read without opening the banks). Pure Python; the run above
took about 3 minutes on one core.

The dictionary is also readable the other way, which needs no carve folder:
`python -I tools/hashdict.py --lookup <temp>/hashes.json <hash>...` takes a dictionary path followed by one or more
32-bit hashes in hexadecimal and prints each with the name the dictionary holds for it, or `?` when it holds none
(`tools/hashdict.py:70,82-93`). That answers "which wave is this id" when another tool prints a hash and no name.

You can also pass the JSON inventory written by `awc_manifest.py --out inventory.json`. The naming tool reads
the plain stream IDs and streamed-music channel IDs directly from its rows. Bank labels keep the nested path
and drop the `.awc` extension to match the export folders. Banks with no stream rows have no names in this inventory.

## Tags: names and subtitles

`awc_opus.py` writes an empty `OpusTags` header, so a player shows only the stream id. `tools/opus_tags.py` fills
that header in place from the `awc_names.py` CSV and, for dialogue, the `text_tables.py` dialogue CSV
([text.md](text.md#dialogue)):

| Comment | Value |
|---|---|
| `TITLE` | the stream name |
| `LABEL`, `SPEAKER` | the dialogue line label and the speaker code in it |
| `SUBTITLE` | the English text, or the first language that has text |
| `SUBTITLE_<LANG>` | the text in each language that has it (`SUBTITLE_DE`, `SUBTITLE_ZH-HANS`, ...) |

Subtitle markup (`~n~`, `~sl:...~`, `~COLOR_...~` and other `~...~` tokens) becomes a space, and empty values are left
out. A file is matched by its stream id and its bank: the folder below the Opus folder, or that path with leading
folders removed, so one Opus folder can hold several archives' exports. `--source` limits the names to one source
label of the names CSV.

Dialogue wave names are matched without case. Repeated rows are accepted when they produce identical tags,
including after subtitle markup is removed. Conflicting labels, speakers or subtitles stop the pass with a CSV
line diagnostic before any Ogg file is modified; this also applies to `--dry-run`.

```sh
python -I tools/opus_tags.py <work>/assets/audio/ogg --names <temp>/awc_names.csv --dialogue <temp>/dialogue.csv
```

Only page 1, which holds the `OpusTags` packet alone, is rewritten. The audio pages, page sequence and granule
positions stay byte for byte as `awc_opus.py` wrote them, and a second run replaces the tags of the first. Unnamed
streams are not touched. A tags packet that would not fit in one Ogg page (65,024 bytes) is skipped and counted;
dialogue tags are a few kilobytes. `--dry-run` counts without writing.

Checked on the 1.0.2.1 package (2026-11-22) with all 638 `shared_minigame000.rpf` banks and the first 3,000
`story000.rpf` banks (83,088 Ogg Opus files): 45,311 named streams tagged, exactly the names CSV rows for those
banks; 1,980 of them are dialogue lines and 1,703 have a subtitle. A fresh `awc_opus.py` export of
`shared_minigame000.rpf` matches the tagged files on every page except page 1 (4,315 of 4,315 files). ffprobe
reads the comments, and ffmpeg decodes a sample of 8,299 tagged files without errors. The run takes about 4 s.

The tagged files hold game text as well as game audio: they stay in the work folder like any other export.

## Never commit or share extracted audio

Extracted banks, Ogg Opus, WAV, XMA2 and the manifest are game data. Keep them in the work folder, never in the
repository, an issue, a pull request or a file-sharing link. Format notes, counts and a few header bytes are fine;
the audio itself is not. See the ground rules in [design.md](design.md#ground-rules) and
[CONTRIBUTING.md](../CONTRIBUTING.md).

## What has been checked

- 20 real Opus streams (6 at 48 kHz, 14 at 32 kHz) decode with ffmpeg to exactly their declared
  per-channel frame counts after the EOS fix; 300 random streams decode with no errors. The earlier
  "lengths match" check read the granule-derived duration, not decoded frames, so it did not catch
  the backwards-EOS case.
- All 1,818 XMA2 streams decode after `rpf9dump xma`.
- A 4-channel streamed-music bank gives four 42.0 s mono channels with real levels and no decoder errors.
- A 1-channel streamed-music descriptor can carry stereo Opus packets; it exports as one stereo Ogg with the
  interleaved sample count halved, and a 3-channel descriptor with stereo packets exports three stereo Oggs.
- In the 3,008-bank / 3,116-channel streamed-music census, 2,501 of 2,958 one-channel descriptors carry stereo
  packets; the 50 multi-channel descriptors split into 20 all-stereo and 30 all-mono banks, with no channel
  mixing TOC stereo bits. 61 representative exports decode cleanly.
- All exported WAVs decode. Codec 2 values are stored as-is, so a float WAV can exceed ±1.

The tools export the streams and channels as stored. They do not reconstruct runtime mixing, looping, ducking or
music-segment selection; that belongs to the game's audio system.

## Streamed-music cue sheets

Run `python -I tools/awc_opus.py <bank-or-directory> <output> --cue-sheet` to write
`cue_regions.json` next to the Ogg files for each bank with verified regions. Each row names
a sibling audio file and the hexadecimal cue-name hash. The integer interval
`[start_sample, end_sample)` counts 48 kHz frames per channel from the start of playback.
Overlapping regions keep their original endpoints. The exporter pairs consecutive same-name
start/end records within one ordered marker tag. It requires a unique stream entry and
descriptor channel with the same ID, exported packets, and endpoints within the declared
and exported audio. A rerun with the option refreshes an existing sheet even if it becomes empty.
