# Short streamed-music idents

`tools/awc_shorts.py` exports short streamed-music channels from AWC banks: station idents, stings,
DJ links and similar clips. It selects each `0x48` channel by its decoded duration and repacks the
stored Opus packets into Ogg Opus without re-encoding.

```powershell
python -I tools\awc_shorts.py <bank.awc | directory> <out> `
    --min-seconds 5 --max-seconds 60 --jobs 8
```

If two banks would share an output folder (for example `clip.awc` and `clip.AWC`), the scan is refused
before writing clips or replacing the catalog. Rename a bank or export the two to separate output roots.

The output is `<out>\<bank>\<stream id>.ogg` plus `<out>\shorts.csv` with bank, stream id, duration,
packet channel count, packet count and relative path. For a directory scan, `<bank>` keeps the
path relative to the source directory, without the `.awc` extension. For example, a nested
`left/shared.awc` has a `bank` value of `left/shared`. Existing files are overwritten, and malformed
banks are reported and skipped.

## Duration and channels

A streamed-music descriptor stores an interleaved sample count. The tool reads the real packet channel
count from the Opus TOC stereo bit and uses `samples / packet_channels` as the per-channel frame count.
It handles both the older `awc_opus.music_channels` five-field result and the newer six-field result.

On the measured package there are **3,008** banks with streamed music and **3,116** descriptor channels.
Of the 2,958 one-channel descriptors, **2,501** carry stereo packets. The 50 multi-channel descriptors
split into 20 all-stereo and 30 all-mono banks. A 5-30 s filter selects **1,288** channels and a 30-60 s
filter selects **336** channels.

## Scope

The tool does not transcribe, classify or deduplicate the clips. A local Whisper pass is a separate
step; the measured Stockyard ident extraction used it without committing the transcripts or audio.
No game data belongs in the repository.
