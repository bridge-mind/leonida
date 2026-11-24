# Text: string tables and dialogue subtitles

**Status: decoded.** Every string table the descriptor search finds in `data.rpf`, `story000.rpf` and
`shared_minigame000.rpf` is read, and every dialogue line is joined to its subtitles. Labels are stored only as
hashes. Dialogue labels are named by the conversation data; other labels are named only when a dictionary holds them.
The tables have no language field, so the language is a guess (below).

Measured on the Xbox Series 1.0.2.1 package on 2026-11-22. The outputs quote game text, so they are local work
products: do not commit them or attach them to an issue. The test fixtures are synthetic.

## String tables

A string table is a `0DRP` logical resource ([page_directory.md](page_directory.md)): the `0DRP` stream joined with
the 64 KiB streams that continue it. Its payload holds a hash table, an index array and a string blob. A descriptor
in the resource locates the three. All pointers are `0x5xxxxxxx` resource pointers; the low 28 bits are offsets
into the logical resource.

```
descriptor   ptr index | u32 | u32 0x10 | ptr | 01 01 00 00 | u32 hash_bytes | ptr hash
             | u32 0x10 | u32 blob_bytes | ptr blob | ... more entries (u32 tag, u32 bytes, ptr)
ptr          u32 0x5xxxxxxx, u32 0
hash entry   16 bytes: u32 joaat(lowercase label), u32 0, u8 1, u8[2], u24 index (at +11), u8
index entry  16 bytes: u64, u64 (offset in its blob chunk << 20 | flags); the chunk is flags & 3
blob         NUL-terminated UTF-8 strings, in one or more chunks
```

A hash entry whose first word is 0, whose second word is not 0, or whose byte at +8 is not 1 is an empty slot. Text
keeps the game's inline markup (`~o~`, `~s~`, `~n~`, `~sl:0.05:0.9~`, `~COLOR_SUBTITLE_FOREIGN_DIALOGUE~`, ...).

### Blob chunks

From the blob on, the descriptor is a run of 16-byte entries: u32 tag, u32 bytes, pointer. The run ends at the first
16 bytes whose pointer word is not a `0x5xxxxxxx` pointer. A table with more than 1 MiB of text has several tag
`0x10` entries instead of one. Each is a blob chunk of at most 1 MiB, and the chunks need not be contiguous. An
index entry's second word names its chunk: its low 20 bits are always `0b1cc`, where `cc` is the chunk, and the bits
above them are the offset inside that chunk.

Measured on `data.rpf` (counts only):

| | |
|---|---:|
| Tables with more than one chunk | 23 (3 of them contiguous) |
| Chunks per such table | 2-4 |
| Index entries in them | 649,246 |
| Low 20 bits `0b100` / `0b101` / `0b110` / `0b111`, all tables | 3,914,583 / 220,161 / 66,416 / 6,721 |
| Index entries pointing at the start of a string, chunk = `flags & 3` | 4,207,881 of 4,207,881 |

Before this rule, `text_tables.py` read every offset from the first chunk, which is a full 1 MiB. The offset then
landed inside some other string, so 293,292 strings in those 23 tables came out with the wrong text and no error.
The other 14,459 `data.rpf` tables read the same as before. So do all 15,951 tables of `story000.rpf` and all 78
of `shared_minigame000.rpf`.

| Archive | Tables |
|---|---:|
| `data.rpf` | 14,482 |
| `story000.rpf` | 15,951 |
| `shared_minigame000.rpf` | 78 |
| **Total** | **30,511** tables, 7,530,381 strings |

The same 14,482 `data.rpf` tables come out of a directory carved by `rpf9_carve.py`, the archive read directly, and
an older carve that wrote each resource as one file.

### Language

The tables carry no language field. Each table's language is guessed from the first 400 strings:

1. **Script.** Kana is `ja`, hangul is `ko`, CJK is Chinese and Cyrillic is `ru`.
2. **Latin script.** Common words and distinctive letters (`ąćęłńśźż`, `äöüß`, `ãõ`, `ñ¿¡`, ...) score each of `en`,
   `de`, `es`, `fr`, `it`, `pl` and `pt`. No score, or a tie, gives `xx`.
3. **Variants.** Spanish and Chinese ship two variants each:
   - Chinese splits by characters that differ between Traditional and Simplified script (`這`/`这`).
   - Spanish splits by Spain's second-person plural (`vosotros`, `habéis`) and vocabulary against Latin America's
     (`ustedes`, `celular`, `ahorita`).
   - A table without a marker keeps the plain `es` or `zh` code.

The table below counts these per-table guesses. Since the nested-archive rule (next section), they are the `guess`
column of `--tables`.

| Language | Tables | Distinct label hashes | Hashes with a second, different text |
|---|---:|---:|---:|
| `en` | 2,190 | 545,145 | 337 |
| `de` | 2,236 | 552,563 | 127 |
| `es-ES` | 1,649 | 525,316 | 4,542 |
| `es-419` | 1,568 | 509,876 | 5,220 |
| `es` (no marker) | 1,214 | 37,802 | 18,852 |
| `fr` | 2,031 | 535,263 | 210 |
| `it` | 2,063 | 544,165 | 56 |
| `ja` | 2,326 | 552,111 | 35 |
| `ko` | 2,334 | 554,576 | 35 |
| `pl` | 2,280 | 547,423 | 58 |
| `pt` | 2,395 | 548,445 | 5,392 |
| `ru` | 2,327 | 545,820 | 33 |
| `zh-Hans` | 2,221 | 553,483 | 81 |
| `zh-Hant` | 2,206 | 554,441 | 34 |
| `zh` (no marker) | 250 | 839 | 252 |
| `xx` (undecided) | 1,059 | 3,652 | 2,246 |

The last column counts label hashes whose text differs between two tables of the same language. A clean split
leaves a few dozen of them (about 35 for `ja`, `ko`, `ru` and `zh-Hant`). Above that there are two causes:

- **Variants mixed.** The plain `es`, `zh` and `xx` rows hold small UI tables without enough text to decide. In the
  marked Spanish rows about 1% of hashes still disagree. The `pt` row holds some tables of another language.
- **Labels with two texts.** Some labels exist in two tables of one language with different text. For example, a
  line spoken in Spanish has a plain subtitle in one table and an `In Spanish: …` subtitle under
  `~COLOR_SUBTITLE_FOREIGN_DIALOGUE~` in another. This accounts for most of the `en` and `fr` counts.

The dialogue table keeps one text per label and language: the last table read wins. `--strings` writes every table
separately, so nothing is lost.

### Language from the nested archive

In `data.rpf`, `story000.rpf` and `shared_minigame000.rpf` every string table sits in one of 13 nested RPF9 archives:
`9FPR` headers at 4 KiB-aligned offsets, whose data offset at +4 stays inside the file. Each of those archives holds a
single language. `text_tables.py` therefore gives every table in a nested archive the guess that most of that archive's
tables share (`xx` left out), when that guess covers more than half of them. A table outside a nested archive, or in an
archive with no majority, keeps its own guess. The tool finds the archive in an `.rpf` source, or in the archive that an
`rpf9_carve.py` directory's `index.json` names. Without it, every table keeps its own guess.
`story000_restofworld.rpf` has 18 nested archives, but the descriptor search finds no string table in it: none of its
101 `0DRP` resources matches the table descriptor.

Measured with the per-table guess (counts only):

| | `data.rpf` | `story000.rpf` | `shared_minigame000.rpf` | `story000_restofworld.rpf` |
|---|---:|---:|---:|---:|
| Nested archives (all) | 41 | 436 | 48 | 18 |
| Nested archives holding string tables | 13 | 13 | 13 | 0 |
| Tables in them / all tables | 14,482 / 14,482 | 15,831 / 15,831 | 78 / 78 | 0 / 0 |
| Tables per archive | 1,109-1,113 | 1,214-1,220 | 6 | - |
| Per-table guess equal to its archive's language | 57.7%-99.9% | 72.2%-100% | 66.7%-100% | - |
| Tables whose language changes | 2,075 | 849 | 5 | 0 |
| `xx` before / after | 869 / 0 | 188 / 0 | 2 / 0 | 0 / 0 |
| Plain `es` before / after | 659 / 0 | 553 / 0 | 2 / 0 | 0 / 0 |
| Plain `zh` before / after | 203 / 0 | 47 / 0 | 0 / 0 | 0 / 0 |

The three archives that hold tables share the 13 languages in the same order: `en`, `pt`, `zh-Hans`, `zh-Hant`, `fr`,
`de`, `it`, `ja`, `ko`, `es-419`, `pl`, `ru`, `es-ES`. The lowest agreement is in the Spanish archives: in `data.rpf` and
`story000.rpf` 208 to 314 tables per archive had no variant marker, and in `shared_minigame000.rpf` the `es-ES` archive
agrees on 4 of its 6 tables.

Reading `story000.rpf` this way needs the `rpf9_carve.py` that checks a gap in chunks (#2117): one unread gap there is
9.56 GB, and slicing it whole ran out of memory.

## Dialogue

Conversation data (PSO streams) pairs a dialogue key with its line labels as `<ROOT>AU_<XXXX>\0<LABEL>\0`. A key with N
distinct labels has the audio waves `<key>_01` .. `<key>_<N>`, in order: wave `_01` speaks the first label, `_02` the
second. These are AWC stream names. A label's `joaat` is its hash in the string tables, so each wave gets its text
in every language. A label ending `_<A-Z{1,3}>_<n>` names the speaker code.

| Where the pairs are | Streams |
|---|---:|
| `story000.rpf` | 594 |
| `story000_restofworld.rpf` | 1 |
| `shared_minigame000.rpf` | 4 |
| `data.rpf` | 2 |

Together they give **118,351 keys and 138,104 lines**. Labels can contain lowercase letters (`..._90s_1`). Some
keys reuse another mission's lines, so a key's root and its labels' prefix can differ. Lines with text:

| `en` | `de` | `es-ES` | `es-419` | `fr` | `it` | `ja` | `ko` | `pl` | `pt` | `ru` | `zh-Hans` | `zh-Hant` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 125,240 | 126,451 | 123,738 | 121,624 | 126,423 | 128,101 | 128,063 | 125,083 | 128,060 | 126,391 | 121,400 | 126,483 | 125,072 |

About 7-12% of lines have no text in a given language: the label's hash is in none of that language's tables.

## Tables and label names

`--labels` takes word lists or a `hashdict.py` JSON. Each table is then grouped by the longest `_`-prefix (up to three
parts) that at least half of its resolved labels share, usually a mission or feature code. With about 3 million
harvested identifiers as labels, 1,869,883 of 7,172,826 strings (26%) resolve, and 9,409 tables get a
group.

## Usage

```sh
python -I tools/text_tables.py <carve>/data <carve>/story000 <carve>/story000_restofworld <carve>/shared_minigame000 \
    --labels <temp>/hashes.json \
    --dialogue <temp>/dialogue.csv --tables <temp>/tables.csv --strings <temp>/strings.csv
```

A source is an `rpf9_carve.py` directory, an `.rpf` archive (read through `gamedata_index.py`; nothing is written),
or files and directories of whole resources. All sources are read before anything is written, because dialogue
keys and their text sit in different archives.

| Output | Columns |
|---|---|
| `--dialogue` | `wave`, `key`, `label`, `speaker`, `text_<lang>` per language |
| `--tables` | `source`, `offset`, `lang`, `guess` (the per-table guess), `archive` (nested archive offset), `group`, `strings`, `resolved` |
| `--strings` | `source`, `offset`, `lang`, `hash`, `label`, `text` |

Standard library only. The four archives above take about 2 minutes on one core from a carve.

`tools/opus_tags.py` writes the `--dialogue` rows into the exported Ogg Opus files as `LABEL`, `SPEAKER` and
`SUBTITLE` comments (see [audio.md](audio.md#tags-names-and-subtitles)).

## Open questions

- **Language per table.** The nested archive now groups the tables of `data.rpf`, `story000.rpf` and
  `shared_minigame000.rpf` by language, but the language code of each archive is still the majority guess of its
  tables. A field or a naming rule outside the tables (the TOC or the executable) would name it directly. The
  descriptor search finds no string table in `story000_restofworld.rpf`, so no nested archive there has a table
  language to name.
- **Which text the game picks.** When a label has two texts in one language, which table wins at run time
  (load order, mission scope, the foreign-dialogue subtitle option) is not established.
- **Hash entries whose byte at +8 is 5, 7 or 8.** `data.rpf` has 3,026 of them, in 62 tables, and none of those tables
  is guessed as English. They are not strings: their u24 at +11 runs 0, 2, 4, ... in every table. That is, in units
  of 16 bytes, an index into a 32-byte record array, the descriptor entry tagged `0x956a9df5`, whose size is exactly
  32 bytes times the number of such entries. Another entry, tagged `0x10075946`, holds 32-byte records whose first
  word is a blob reference like an index entry's second word. All 7,345 of them point at the start of a string.
  A group record holds a u32 count equal to the u32 after it (1-7). Read as (first record, count), its first word
  and that count cover the `0x10075946` array exactly in only 34 of the 62 tables, so the layout is not decoded.
  `text_tables.py` skips these entries.
