# Named RPF9 resources

RPF9's table of contents is encrypted, so resource names cannot be read from the directory. They
are still present in the inflated stream data: a drawable is spelled `name.#dr`, a fragment
`name.#ft`, a drawable dictionary `name.#dd`, and so on. `tools/rpf9_names.py` walks the data area
like `rpf9_carve.py`, inflates a bounded prefix of each zlib stream, and indexes those names with
their archive offsets.

This is a lookup pass, not a decoder. It does not need the TOC, does not decode models or
animations, and does not write any game data into the repository. The JSON index is local scratch.

## Usage

```text
python -I tools/rpf9_names.py <archive.rpf> <out.json> [--start N] [--limit BYTES]
                              [--max-inflate BYTES] [--types dr,ft,dd] [--quiet]
```

- `--start` defaults to the data start in the RPF9 header.
- `--limit` bounds the pass; no limit scans to the end of the archive.
- `--max-inflate` is the prefix inspected per stream (default 1 MiB).
- `--types` selects the resource suffixes; the default is `dr,ft,dd`.

The output has a `summary` and one `files` row per stream that contains a matching name:

| Field | Meaning |
|---|---|
| `offset` | archive offset of the zlib stream |
| `stored` | compressed stream length |
| `size` | inflated stream length |
| `names` | matching full names in that stream |
| `streams` | every zlib stream accepted by the walk |
| `streams_with_names` | streams with at least one matching name |
| `name_hits` | total matching name occurrences |
| `unique_names` | distinct names across the pass |
| `by_type` | distinct names per suffix |
| `zero_padding`, `unread_bytes` | gap accounting, as in the carve tool |
| `bink2_files` | Bink 2 videos skipped whole |
| `bink2_bytes` | total bytes of those Bink 2 videos |

## Measured pass

Bounded runs against the 1.0.2.1 package validate the lookup:

| Scope | Streams | Streams with names | Unique names |
|---|---:|---:|---:|
| `packs.rpf`, full | 144,364 | 27,292 | 12,338 (9,470 `#dr`, 2,865 `#ft`, 3 `#dd`) |
| `data.rpf`, first 256 MiB | 17,717 | 24 | 274 (`#dr`) |
| `packs.rpf`, player head window (2 MB) | 44 | 21 | 15 (`#dr`) |
| `packs.rpf`, brabikini window (2 MB) | 33 | 14 | 6 (`#dr`) |
| `packs.rpf`, Lucia head window (2 MB) | 36 | 16 | 6 (`#dr`) |
| `packs.rpf`, hand window (2 MB) | 21 | 9 | 4 (`#dr`) |
| `packs.rpf`, feet window (2 MB) | 62 | 28 | 11 (`#dr`) |
| `packs.rpf`, undies window (2 MB) | 74 | 33 | 12 (`#dr`) |

The character windows contain the same head/eyes/eyecap, hand, feet, brabikini and undies drawable
records found by the earlier manual scan, which is the validation that matters for targeted
extraction. The generated index is local scratch because it contains game names.

## Limits

- Only the first `--max-inflate` bytes of each stream are inspected; a name after that prefix is
  not found. The default is deliberately larger than the measured name tables.
- The scan is a zlib-stream walk, not a TOC read. Streams that are not clean zlib (raw blocks)
  are skipped and counted in the gap accounting. Bink 2 videos are skipped whole and counted in
  `bink2_files` / `bink2_bytes`.
- Names are data, not proof that a resource is usable: a matching stream still has to be decoded by
  `rpf9dump` or `RageModelDecoder`.
- Counts are package/version-specific. The output index is local and must not be committed.
