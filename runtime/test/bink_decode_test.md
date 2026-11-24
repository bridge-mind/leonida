# Bink 2 decode through `rt::Loader`

`bink_decode_test.cpp` is an optional Windows test that maps the package's `Bink2Scarlett.dll` with `rt::Loader`,
checks that `xmem.dll!XMemAlloc`/`XMemFree` bind to `runtime/shims/xmem.cpp` (the IAT slot is compared with
`rt::shims::find`), and decodes a `.bk2` through the loader's export map.

It runs only when both variables are set:

```sh
set LEONIDA_GAME_DIR=C:\path\to\package
set LEONIDA_MOVIE=C:\path\to\stream.bk2
set LEONIDA_GOTO=200          # optional
set LEONIDA_RUN_DLLMAIN=1     # optional; not needed for the measured streams
runtime\build\bink_decode_test.exe
```

With either required variable unset it prints `SKIP (no package/movie)` and exits 0, so `ctest` is green on a machine
without the package. Frames are written as raw BGRA and PPM under `%TEMP%\leonida_bink`; they are never committed.

## Measured on 1.0.2.1

- `rt::Loader` maps `Bink2Scarlett.dll` at `0x180000000` with 2 shim imports (`xmem.dll!XMemAlloc`,
  `xmem.dll!XMemFree`), 58 host imports, no bundled-module imports and **0 unimplemented imports**. Both xmem IAT
  slots equal `rt::shims::find("xmem.dll", ...)`; `Bink2Scarlett.dll` is mapped by `rt::Loader`, not loaded with
  `LoadLibrary`, and no msvcrt redirection is involved.
- `BinkOpen`, `BinkDoFrame`, `BinkCopyToBuffer`, `BinkNextFrame`, `BinkGoto` and `BinkClose` resolve from
  `Module::exports`. The decoded BGRA frames are bit-identical to the earlier msvcrt-redirection experiment for the
  1280x720 frame 0 and 3840x2160 frame 200.
- The loader does **not** run `DllMain`/TLS, and decode still works. The image has no TLS directory. Calling the
  entry point as `DllMain(DLL_PROCESS_ATTACH)` returns TRUE and also decodes; it is not required for these streams.
- 1280x720 (40 frames): frame 0 and frame 1 differ.
- 1920x1080 (150 frames): frame 0 and frame 1 are identical in the sampled stream; both replace the sentinel buffer.
- 3840x2160 (450 frames): frame 0 and frame 1 differ; `BinkGoto(200, 0)` differs from frame 0 and matches the
  earlier 4K frame-200 decode.
- No `BinkSetSoundSystem` or XAudio2 call is made. A 3840x2160 stream with one audio track and a 1920x1080
  stream with four audio tracks both open and decode video without the sound system.
- Independent run on a second machine (Windows 11, Ryzen 7 7800X3D, own 1.0.2.1 package): a `KB2j`-revision
  512x512 15-frame stream decodes the same way (2 shim imports, 58 host, 0 unimplemented; frames 0 and 1 differ), so `KB2j` also decodes through the loader, on a second 1.0.2.1 copy (the `KB2n`
  numbers above are from the first). The 10 `KB2j` streams are the nine 512x512 15-frame files plus one
  1920x1080 60-frame file ([docs/movies.md](../../docs/movies.md) section 1).

## Gaps

Audio tracks are only read as header metadata; no Bink Audio packet is decoded and no audio system is opened. The
async and GPU decode paths and subtitles are not tested; the `HBINK`/`BINKSUMMARY` fields are measured by
[`bink_abi_test`](bink_abi_test.md). `DllMain`/TLS
are not run by the loader by default; they were not needed for this video path.
