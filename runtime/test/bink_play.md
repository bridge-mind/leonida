# `bink_play`: watch a Bink 2 movie through the package's decoder

`bink_play.cpp` maps the package's `Bink2Scarlett.dll` with `rt::Loader` (the path `bink_decode_test` checks) and plays
a Bink 2 movie in a window at the movie's frame rate. A movie inside an archive plays where it is: nothing is
extracted and nothing it decodes is written to disk.

```sh
bink_play                                   # no movie: plays Stinger/RSG_STINGER.bk2 (so a double-click plays it)
bink_play stinger                           # a movie's path in the game, or any unique part of it
bink_play TV/<name>.bk2 --loop
bink_play movies.rpf@OFFSET | file.bk2      # an offset in an archive, or a loose .bk2
```

Options: `--game-dir DIR`, `--frames N`, `--headless`, `--loop`. The game folder comes from `--game-dir`,
`LEONIDA_GAME_DIR`, or the archive the name table was built from.

Names come from the table `tools/bink_play.py` keeps in `%LOCALAPPDATA%\Leonida\bink_names.csv` (names, offsets
and frame sizes from `tools/bink_names.py` and `common.rpf`'s mount lists; no movie data), rebuilt when the archive's
size changes. Without a table, `bink_play` runs that tool once to build it, which takes about half a minute and needs
the game folder the first time: `bink_play stinger --game-dir C:\path\to\package`. After that a double-click
is enough. `python tools/bink_play.py --list [TEXT]` lists the names.

For `archive@OFFSET` it calls `BinkSetFileOffset(OFFSET)` and opens the archive with `BinkOpen`'s `BINKFILEOFFSET`
flag (0x20); without the flag the offset is ignored and `BinkOpen` reports "Not a Bink file". An error in a windowed
run is shown in a message box too, since a double-clicked console closes with the process.

| Key | Action |
|---|---|
| Space | pause / resume |
| Left / Right | one second back / forward (`BinkGoto`) |
| Home | back to the first frame |
| L | loop on / off |
| Esc, closing the window | quit |

The window title shows the frame number and the rate frames are being shown. The frame is scaled to the window with
its aspect ratio kept (GDI `StretchDIBits`, no GPU path).

`--headless` decodes without a window as fast as it can and prints the decode rate; `--frames N` stops after N
frames. CTest runs `bink_play --headless --frames 30` as `bink_play_headless`, which skips (exit 77) unless
`LEONIDA_MOVIE` names a movie (a headless run never falls back to the stinger).

## Measured on 1.0.2.1

`Stinger/RSG_STINGER.bk2` (3840x2160, 450 frames, 30 fps), played from inside `movies.rpf`: 90 frames decode
headless in about 1 s (81-118 fps over runs) on a Ryzen 7 5700X (the decoder is CPU-only), so the 4K stinger plays in
real time; the windowed run shows all 450 frames.

An independent run on a second machine (Windows 11, Ryzen 7 7800X3D, own 1.0.2.1 package) plays a 1280x720 181-frame
`KB2n` stream headless in 0.27 s (**672 fps**), consistent with the decoder being CPU-only and the 4K rate above
being resolution-bound rather than machine-bound.

## Gaps

No audio: the movie's Bink Audio tracks are not decoded and no sound system is opened (as in `bink_decode_test`).
Playback uses the synchronous `BinkDoFrame` path only.
