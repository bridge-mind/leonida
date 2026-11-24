# Bink Audio track extraction through `rt::Loader`

`bink_audio_test.cpp` is an optional Windows diagnostic. It maps `Bink2Scarlett.dll` with `rt::Loader`, checks that
`xmem.dll!XMemAlloc`/`XMemFree` bind to `runtime/shims/xmem.cpp`, enumerates the movie's audio tracks, and decodes
one selected track to 16-bit PCM/WAV.

```sh
set LEONIDA_GAME_DIR=C:\path\to\package
set LEONIDA_MOVIE_AUDIO=C:\path\to\stream.bk2
set LEONIDA_AUDIO_TRACK=0        # optional, default 0
set LEONIDA_AUDIO_FRAMES=100     # optional, default all frames
set LEONIDA_RUN_DLLMAIN=1        # optional, not needed
runtime\build\bink_audio_test.exe
```

With either required variable unset it prints `SKIP (no package/audio movie)` and exits 0. Output goes to
`%TEMP%\leonida_bink_audio`; no movie, packet or PCM is committed.

## Measured API shape (1.0.2.1)

- Loader binding is the same as the video path: **2 shim** (`xmem.dll!XMemAlloc`, `XMemFree`), **58 host**,
  **0 unimplemented**; both xmem IAT slots equal `rt::shims::find`.
- `BinkGetTrackID(h, index)` returns the header track ID (0, 1, 2, 3 in the four-track sample).
- `BinkGetTrackType(h, index)` returns `(audio_flags << 16) | sample_rate`: `0x7000bb80` for `0x7000`/48 kHz and
  `0x3000bb80` for `0x3000`/48 kHz.
- `BinkGetTrackMaxSize(h, index)` returns 146,432 for these tracks; the header's max-size field is 145,920 (512
  bytes smaller).
- `BinkOpenTrack(h, index)` returns a handle whose observed prefix is `Frequency=48000`, `Channels=2`,
  `MaxSize=154624`, then `HBINK`, an internal pointer, and `trackindex`. This is not the public Bink 1
  `Frequency, Bits, Channels, MaxSize` order.
- **`BinkGetTrackData(track, dest)` returns decompressed 16-bit PCM bytes, not compressed Bink Audio packets.**
  A full 15 s stereo 48 kHz track is exactly 2,880,000 bytes. The first call can return a pre-roll (145,920 bytes
  on the 4K stream, 19 x 7,680); later inspected calls returned 7,680 or 0. The return is a byte count, not a fixed
  packet size. `BinkSetSoundSystem` and `BinkOpenXAudio2` are never called; `BinkSetSoundTrack` is used only to
  select the one track being extracted.
- A `BinkCopyToBuffer` call per frame followed by `Sleep(1)` is required by this build. Without the delay,
  `BinkGetTrackData` intermittently faults inside `Bink2Scarlett.dll` at `+0x1EE1B`; `BinkWait` and `BinkService`
  return before the audio path is ready. This is a DLL race, not a loader/xmem failure.

## Measured tracks

| Stream | Track | Frames | PCM bytes | Duration |
|---|---|---:|---:|---:|
| 3840x2160, 1 track | 0 | 450 | 2,880,000 | 15.000 s |
| 1920x1080, 2 tracks (`0x7000` each) | 0 | 151 | 966,400 | 5.033 s |
| 1920x1080, 2 tracks (`0x7000` each) | 1 | 151 | 966,400 | 5.033 s |
| 1920x1080, 4 tracks | 0 | 900 | 5,760,000 | 30.000 s |
| 1920x1080, 1 track `0x3000` | 0 | 4350 | 27,840,000 | 145.000 s |
| 1920x1080, 2 tracks (`0x3000` + `0x7000`) | 1 | 450 | 2,880,000 | 15.000 s |
| 1280x720, 1 track `0x7000` (`binkaudio_dct`, 48 kHz stereo) | 0 | 181 | 1,158,400 | 6.033 s |

All decoded WAVs are 48 kHz, stereo, `pcm_s16le`. The sampled `0x3000`-only track is entirely silent (peak 0); the
`0x7000` track in the mixed two-track file has normal signal. This does not establish what each track is used for.

The 1280x720 row is an independent run on a second machine (Windows 11, Ryzen 7 7800X3D, own 1.0.2.1 package):
every one of the 181 calls returned 6,400 bytes (no 7,680 or 0 calls, so no pre-roll on this stream), and the
total matches the API shape above (a byte count per call).

## FFmpeg

FFmpeg has the `binkaudio_dct` and `binkaudio_rdft` decoders, but its Bink demuxer misparses the `KB2n` audio header:
it reports 14,848 Hz mono RDFT and rejects the first packet. Changing the revision byte to `k` makes the demuxer skip
the extra header word, but the decoder then rejects the packets as "Incomplete packet". FFmpeg is not a usable path
for this package's Bink 2.2 audio; the DLL's track API returns PCM directly.

## Gaps

The harness does not extract compressed packets; the DLL returns PCM directly. One raw packet was probed manually for
the FFmpeg check, but its exact layout is not mapped. No sound system/XAudio2 output is opened, and the
async/GPU/subtitle paths are untested. Which track corresponds to which language or mix is not known.
