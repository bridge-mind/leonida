# Bink2Scarlett ABI over `movies.rpf`

`bink_abi_test.cpp` is an optional Windows test. It maps the package's `Bink2Scarlett.dll` with `rt::Loader`, as
[`bink_decode_test`](bink_decode_test.md) does, and opens every stream listed in a `tools/bink_index.py` CSV in place
inside `movies.rpf`: `BinkSetFileOffset`, then `BinkOpen` with flag `0x20`. Nothing is extracted.

For each stream, it:

- reads the first 256 bytes of the `HBINK`;
- calls `BinkGetSummary` on a sentinel-filled 1,024-byte buffer;
- keeps the 32-bit word offsets whose value equals the stream header's width, height, frame count, frame rate and
  audio track count;
- tracks the current frame number across `BinkOpen`, `BinkNextFrame` and `BinkGoto`;
- tallies the return values of `BinkWait`, `BinkDoFrame` and `BinkCopyToBuffer`.

It then reopens the first stream with a forced `BinkSetFrameRate(24000, 1001)` to tell the playback rate fields apart
from the file's. It checks the measured offsets in [`docs/movies.md`](../../docs/movies.md#abi-over-the-archive).
Output is offsets, counts and return values only: the decoded frames stay in memory and are never written.

It runs only when both variables are set:

```sh
python -I tools/bink_index.py C:\path\to\package\movies.rpf --out %TEMP%\movies.csv
set LEONIDA_GAME_DIR=C:\path\to\package
set LEONIDA_BINK_INDEX=%TEMP%\movies.csv
set LEONIDA_BINK_LIMIT=8      # optional: only the first 8 streams
runtime\build\bink_abi_test.exe
```

With either required variable unset, it prints `SKIP (no package/index)` and exits 77, so `ctest` stays green on a
machine without the package.

## Measured on 1.0.2.1

All 515 streams opened and `RESULT PASS`. It ran in 46 s on a hard disk: Windows 11, MSVC 19.52, RelWithDebInfo. No
GPU is involved. The field table is in [`docs/movies.md`](../../docs/movies.md#abi-over-the-archive).

## Gaps

- Only fields whose value appears in the stream header can be found this way. The rest of the `HBINK` (after
  `+0x38`) and of the bytes `BINKSUMMARY` writes (+0x00..+0x8F) is not named.
- The forced frame rate was checked on one stream.
- The async, GPU and audio paths are not exercised; the audio track API is covered by
  [`bink_audio_test`](bink_audio_test.md).
