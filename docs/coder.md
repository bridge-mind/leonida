# The compressed mass in the story script archives

`story000_script.rpf` and `script.rpf` hold data in a framing the rest of the package does not use.
Most of it is ordinary zlib, but a large part of `story000_script.rpf` is compressed by an encoder that
no standard decoder reproduces. This page records what was measured about it, so the next attempt starts
from the numbers instead of from a guess. Everything below is counts, offsets, constants and
statistics over the user's own copy — no payload is stored here.

Subject: `story000_script.rpf`, region `0x2000..0x3127000` (51,531,776 bytes), 905 page-blocks.

## Two populations

| population | blocks | bytes | share | decoder |
|---|---:|---:|---:|---|
| `78 DA` zlib at block start | 105 | 893,535 | 1.79 % | standard zlib, raw deflate at +2 with a 32 KiB window |
| anything else | 767 | 49,095,163 | 98.21 % | unidentified |

The two are **interleaved**, not separated by region: 211 population switches across the first 120
blocks. The zlib blocks yield ~2 MB of plaintext that is plainly structured data (8-byte aligned records,
`0x50000000`-space pointers, `FFFFFFFF` sentinels, ascending u32 tables). The other 767 do not decode yet.

## Entry framing

Each entry is:

```
u16 tag + 6 class-biased header bytes │ zero pad │ payload at the next 16-byte FILE boundary
```

i.e. `payload_offset = align16(entry_offset + 8)`, which is `+16` for 766 of the 800 entries measured.
The rule was confirmed 10/10: the predicted pad length equals the observed zero run every time, and the
absolute payload offset is `%16 == 0` every time. This is why earlier scans that assumed payloads start at
the entry start, or that stopped at skip +12, found nothing.

Header shape (distinct byte values per position over the 800 unknown entries):
`52, 65, 191, 198, 171, 148, 176, 188, 219, 239, 243, 243, …` — i.e. bytes 0..7 are structured and byte 8
onwards is noise. There is no size field in the header (60 width/endianness combinations tested, all
`|r| < 0.10` against the block length).

The first two bytes form a **32-value tag alphabet**: 75 distinct u16 in total, but 32 codes carry nearly
all 800 entries at 15–33 each (800/32 = 25 expected). It is not a bitfield (byte 0 has 52 values, byte 1
has 65, joint 75 ≪ 3380), and it correlates with nothing measured: length (`r = -0.035`), gap to the next
entry (`-0.036`), offset (`+0.035`), or neighbours' population. Codes never cluster in file order (881 runs
over 905 entries), so it is not a file or stream id.

## What the payload is not

Every decoder below was run at the **correct** `align16(entry+8)` offset, with a positive control that
passes on data compressed by that same decoder, and against a same-geometry random baseline.

| family | result |
|---|---|
| deflate (raw, and raw with a 32 KiB zero window) | 0 of 800 entries give even 4 KiB of partial output |
| zlib, both framing and preset-dictionary (all 32 fcheck-valid CMF/FLG, FDICT set) | 0 — payloads never begin with a `78 xx` header at all |
| zstd (magicless and framed), brotli, snappy, lz4 (two decoders), LZAV, FastLZ, LZF, LZJB, bz2, xz, lzma-alone, lzma-raw | 0 |
| Oodle (oo2core 2.9.10, positive control bit-exact) | 0 over 11 offsets × 19 destination sizes × fuzzSafe 0/1 |
| 21 byte/bit transforms (bit-reverse, nibble-swap, rotates, delta, xor-prev, transposes, block reversal) × 4 decoders | 5 hits against a 2-hit random baseline, 4 degenerate |

## What the payload is

Statistical shape, measured against the same statistics computed on data compressed by each family:

| statistic | zlib level 9 | lzma | bz2 | this payload |
|---|---:|---:|---:|---:|
| popcount correlation | +0.676 | −0.065 | −0.432 | **+0.632** |
| bit-plane bias | +0.0138 | −0.0007 | −0.0110 | **+0.0083** |

So it is **deflate-shaped but proprietary**: an LZ+entropy coder in the deflate family, not LZMA and not a
BWT coder. AES-class output is excluded because all eight bit planes are biased the same way at ~40σ,
which ciphertext cannot do. Bit autocorrelation is +0.006 *inside* a byte and exactly 0 at byte boundaries,
so the stream is bit-domain with no byte framing.

The code is **variable-length and LSB-first**. That was established from runs of one repeated symbol: a run
emits the same code repeatedly, so the bit spacing of a run equals the code length. Across the region,
21 non-zero period-7 runs show a uniform **7-bit** spacing and 6 show **8 bits**, in LSB-first order
(MSB-first gives 9, 9, 9, 9, 9, 2, 7 on the same patterns — inconsistent).

The only visible structure in the 331,760-byte single-block sample at `script.rpf+0x2010` is a 496-byte
field at payload+0x14460, 16-byte aligned at both ends, made of exact period-7 repeats in two groups
(54 and 16 copies). Decomposed as 8 symbols × 7 bits, each group is **one plaintext u32 repeated**:

```
group A:  63 30 48 16 | 63 30 48 16
group B:   2  7 115 102 |  2  7 115 102
```

at 28 bits per 4 bytes = **7.00 bits/byte**. So the field is a table of ~140 identical 4-byte entries
(default records), the repeats are literals rather than matches, and the coder achieves roughly 1/8
compression on repeated data. The eight 7-bit code values read out are `2, 7, 16, 30, 48, 63, 102, 115`.

## The one missing piece

The coder stores **no table**: the longest run of identical bytes anywhere in 331 KB is the 32-byte zero
tail, and a canonical-Kraft scan over 8,192 offsets finds nothing. With variable-length codes of 7 and 8
bits and no stored table, the symbol→length mapping is a **static array in the executable**.

That array is narrow enough to search for. A Kraft-exact length vector for 7- and 8-bit codes satisfies
`2·n7 + n8 = 256`; with the largest observed code value 115 requiring `n7 ≥ 116`, the candidate families
are `n7 = 116..127`, i.e. a **129–140 byte vector of 7s and 8s**, most likely `n7 = 116, n8 = 24`
(140 symbols). Search the executable for a 129–140 byte run of values in `{7,8}` satisfying that equation.
It is not in the package: the 16 bundled DLLs and the whole 348 MB `gameos.xvd` contain no such run
(byte-exact search), and no `oo2core`/`Oodle`/codec string exists in either.

## Recovered plaintext, for calibration

Two independent extractions recovered 1,961,559 bytes and 855,961 bytes of payload (the second from
entries whose zlib part is buried behind the aligned header). The 0x88-byte record structure in it is
described in the mission notes; the useful calibration here is that a correct decode of the remaining
767 entries should produce the same record discipline: 8-byte alignment, pointers of the form
`0x50000000 | offset` with the high dword 0, and 96.7 % of pointers at even dword indices.
