# compare.py - pixel conformance checker and renderer diff

`compare.py` verifies rendered PNGs against the pixel checks a renderer's
`manifest.json` declares, and measures how far one renderer's scenes are from
another's. It reads PNG and JSON only: it never parses C++ or `scenes.h`. Every
expectation comes from the manifest, so the same tool works for the native
D3D12 renderer, the Vulkan-backed shim and any future scene.

Standard library only. Tested with Python 3.12.10 on Windows (the conformance
box). Python 3.8 or newer is needed for the language features used.

```
D:\conform\tools\
    compare.py      the tool
    README.md       this file
```

On the conformance box:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" D:\conform\tools\compare.py --help
```

Elsewhere, `python compare.py --help` is enough.

## Manifest contract

The tool consumes exactly the shape the renderers write:

```json
{
  "renderer": "native",
  "adapter": "NVIDIA GeForce RTX ...",
  "scenes": [
    {
      "name": "clear",
      "png": "clear.png",
      "checks": [
        {"x": 0, "y": 0, "r": 10, "g": 20, "b": 30, "a": 255,
         "why": "top-left is the clear colour"}
      ]
    }
  ]
}
```

The manifest must be UTF-8; a UTF-8 BOM (which PowerShell may add) is accepted.
Each JSON object must have unique keys. A repeated field, including a channel in a pixel check,
is an input error rather than a last-value-wins override.

Validation rules, all reported with the manifest path and the exact scene/check
position:

- `renderer` and `adapter` are required strings (`adapter` may be empty).
- `scenes` is a non-empty list. Each scene needs a unique `name`, a unique `png`
  file name beside the manifest (no directories), and a non-empty `checks` list.
  A scene with no checks is rejected: it would prove nothing.
- A check needs integer `x`, `y` (0 or more) and `r`, `g`, `b`, `a` (0-255).
  `why` is optional; when present it is quoted in a failure line.
- The number of checks is not assumed. The current scene contract uses 2 to 5
  per scene; `expect` evaluates however many are declared.
- Unknown extra keys are ignored so a newer renderer can add metadata without
  breaking this tool.

The scene `name` is also used for the heatmap file `<scene>.diff.png`, so names
that are Windows-illegal or contain a path separator are rejected at load with
an explanation rather than failing later during a write.

## Orientation and channel order

These are contract conventions and the reader follows them literally:

- Image row 0 is the **top** of the image and corresponds to NDC y = +1. The
  reader does not flip rows, and the writer does not flip them either.
- Channels are stored and interpreted in **R, G, B, A** order, 8 bits each. The
  reader does not reorder channels.
- **A vertical flip is the single most common reason a GPU diff looks like a
  total mismatch.** `inspect` always prints labelled top-left and bottom-right
  values so a flip is obvious at a glance, and `selftest` includes an image whose
  top and bottom halves differ so the tool's own orientation is proven.

## expect - verify declared pixels

```powershell
python compare.py expect --manifest D:\conform\native\manifest.json
python compare.py expect --manifest D:\conform\shim\manifest.json --dir D:\conform\shim\renders
```

`--dir` is optional; without it, PNGs are read from the manifest's directory.

Output is one line per declared check, then a summary. This is an excerpt from a
shim run against an 18-check manifest:

```
PASS  fullscreen   (128,128)     expected 200,30,60,255      got 200,30,60
FAIL  fullscreen   (8,8)         expected 200,30,60,255      got 10,20,30  (top-left corner is covered)
FAIL  blend        (128,128)     expected 105,25,45,255      got 200,30,60  (0.5*(200,30,60) + 0.5*(10,20,30) = (105,25,45))

6 scenes, 18 checks: 8 passed, 10 failed
```

- `got` prints the channels actually stored: three for an RGB file, four for
  RGBA. An RGB file has no alpha channel, so it is compared as if alpha were 255.
  A check that declares any other alpha against an RGB file fails and says why.
- A pixel outside the image is a failure that names the coordinate and the
  image's size. The scene's other checks still run.
- A missing, truncated or corrupt image is one scene-level FAIL carrying the
  reader's message (path, byte offset or CRC), and the scene's declared checks
  are counted as failed in the summary.
- Exit code is 0 only when every check passed.

`expect` does not verify that the manifest's checks are the right expectations.
That is the scene contract's job; this tool checks the pixels against whatever
the manifest declares.

## diff - compare two renderers

```powershell
python compare.py diff --reference D:\conform\native\manifest.json --candidate D:\conform\shim\manifest.json
```

Both manifests must declare the same scene names; each manifest's own `png`
field is used to find its image, so the two renderers may name files
differently. Images must have equal dimensions.

Per scene the row reports:

- **differing**: number of pixels where any R, G, B or A channel differs, and
  that count as a percentage of width x height.
- **mean**: mean absolute channel delta over every channel of every pixel. The
  denominator is `width * height * 4`, so an RGB image contributes zero alpha
  deltas and a small area failing hard is distinguishable from a large area
  failing softly.
- **max**: largest single-channel delta in the scene.
- **first**: first differing pixel in row-major order (top row first, left to
  right within a row), or `-` when identical.

Thresholds are strict `>` comparisons and both default to `0.0`, so any
difference fails:

```
--max-differing-percent FLOAT   fail above this percent of differing pixels
--max-mean-delta FLOAT          fail above this mean absolute channel delta
```

NaN, infinity and negative values are rejected by the argument parser rather
than silently passing everything.

Example (the native and shim fixture run, shortened):

```
PASS  clear        differing 0/65536 (0.000%)  mean 0.0000  max 0  first -
FAIL  fullscreen   differing 256/65536 (0.391%)  mean 0.2246  max 190  first (8,8)  (differing pixels 0.391% exceed --max-differing-percent 0.000%)
FAIL  depth_order  differing 34648/65536 (52.869%)  mean 46.2601  max 180  first (6,6)  (differing pixels 52.869% exceed --max-differing-percent 0.000%)
FAIL  scissor      differing 57344/65536 (87.500%)  mean 78.7500  max 180  first (0,0)  (differing pixels 87.500% exceed --max-differing-percent 0.000%)
FAIL  blend        differing 65536/65536 (100.000%)  mean 28.7500  max 95  first (0,0)  (differing pixels 100.000% exceed --max-differing-percent 0.000%)
      scene 'blend': every pixel differs - usually the scene's tint (constant buffer b0) or clear colour, before chasing rasterisation
FAIL  indexed      differing 10230/65536 (15.610%)  mean 18.7317  max 200  first (51,51)  (differing pixels 15.610% exceed --max-differing-percent 0.000%)

6 scenes compared, 5 exceed the thresholds. Heatmaps written beside the candidate images as <scene>.diff.png
```

When **every** pixel differs, the tool prints a second, indented line naming the
scene and pointing at the most likely cause:

```
      scene 'fullscreen': every pixel differs - usually the scene's tint (constant buffer b0) or clear colour, before chasing rasterisation
```

Each scene's tint is distinct in the current contract, so a whole frame of the
wrong colour is usually a constant-buffer value, not rasterisation. The scene
name is already at the start of the row; the extra line makes that diagnosis
impossible to miss.

### Heatmap

For every scene, `diff` writes `<scene>.diff.png` beside the candidate image,
same size as the candidate:

- unchanged pixels: dark grey `(48, 48, 48)`, not black, so a black render pixel
  cannot hide in the background;
- differing pixels: `delta` is the largest absolute channel difference at that
  pixel. Red is `round(255 * (delta - 1) / 254)`, green is 0, blue is
  `255 - red`. Delta 1 is pure blue `(0, 0, 255)`, delta 255 is pure red
  `(255, 0, 0)`, and the ramp passes through purple. Alpha differences count.

Heatmaps are written for passing scenes too, so a clean run leaves a complete
visual record. Existing files are overwritten. The tool is deterministic: the
same inputs produce byte-identical heatmaps (verified with SHA-256 over two
runs).

### diff failures that are not pixel differences

- Missing or extra scene names in the candidate manifest: error names them and
  exits 2. There is nothing meaningful to compare.
- Different image sizes: error names both paths and their sizes and exits 2.
- Unreadable or unsupported image: error names the file and exits 2.

A pixel comparison that ran and exceeded a threshold exits 1.

## inspect - look at one image

```powershell
python compare.py inspect --image D:\conform\shim\clear.png
python compare.py inspect --image D:\conform\shim\clear.png --pixel 0,0 --pixel 128,128
```

Top-left and bottom-right are always printed with explicit coordinates. Without
`--pixel`, top-right, bottom-left and the centre are printed too. With
`--pixel X,Y` (repeatable), the requested pixels follow the two labelled corners.

```
fixtures/native/clear.png
  256 x 256, RGB, 8 bits per channel, non-interlaced
  top-left     (0,0)      = 10,20,30
  bottom-right (255,255)  = 10,20,30
  no --pixel given; also showing top-right, bottom-left and centre
  top-right    (255,0)    = 10,20,30
  bottom-left  (0,255)    = 10,20,30
  centre       (128,128)  = 10,20,30
```

RGB prints three values, RGBA four. A pixel outside the image is an error naming
the coordinate and the image size; exit code 2.

## selftest - prove the tool without external files

```powershell
python compare.py selftest
```

It creates a temporary directory, writes fixture PNGs, asserts the behaviours
below, removes the directory, and exits 1 on the first mismatch (naming the
assertion, expected value and actual value).

```
selftest: fixtures in C:\Users\Hussain\AppData\Local\Temp\compare-selftest-wf7w_kkg
ok    PNG round-trip: RGB pixels and alpha-free pixel reads
ok    PNG round-trip: RGBA rectangle at (2,1)-(5,4), alpha 200 preserved
ok    PNG writer is deterministic: two writes of one raster match byte for byte
ok    failure path: unwritable output directory names the path
ok    failure path: corrupt IDAT reports the CRC and byte offset
ok    failure path: truncated file reports the byte offset
ok    PNG filters 0-4 decode: hand-computed streams for a 3x2 RGB fixture
ok    failure path: short filter data and undefined filter type are rejected
ok    arguments: thresholds reject negative, NaN, inf and text; pixels need x,y
ok    orientation: top and bottom halves survive; inspect labels both corners
ok    expect: a five-check manifest exits 0 and prints five PASS rows
ok    manifest reader accepts a UTF-8 BOM, as PowerShell writes
ok    expect: wrong pixel and out-of-range coordinate fail with explanations
ok    expect: a missing image fails once for the scene and names the path
ok    failure path: a scene with no checks is rejected, not silently passed
ok    diff metrics: 2 known pixels give exact count, percent, mean, max and first
ok    heatmap: grey unchanged, red at delta 255, blue at delta 1
ok    diff edge cases: row-major first difference, alpha-only delta, tint hint
ok    failure path: mismatched image sizes refuse to compare and say so
ok    diff: changed scene exits 1 and writes clear.diff.png beside the candidate
ok    diff: generous thresholds pass the same scene without hiding its delta
selftest passed
```

### What selftest proves

- **PNG writer and reader**: an 8x6 RGB image round-trips byte for byte; an 8x6
  RGBA image round-trips with a known rectangle at x 2..5, y 1..4 and alpha 200
  preserved at its corners and clear outside it. A second write of the same
  raster is byte-identical.
- **PNG filters**: all five filter types decode a 3x2 RGB fixture whose filtered
  streams were computed independently from the filter definitions; filter 0, 1,
  2, 3 and 4 each reconstruct the same pixels. Short filtered data and a filter
  type of 5 are rejected with the reason (and the row for an undefined type).
- **PNG failure paths**: writing into a missing directory fails with the path;
  a flipped byte inside IDAT fails with a CRC mismatch and byte offset; a
  truncated file fails naming the truncation. None of these crash or write a
  zero-byte file.
- **Arguments**: thresholds accept 0 and 1.5 and reject negative, NaN, infinity
  and text; `--pixel` requires x,y and rejects malformed values.
- **Orientation and channel order**: a 4x4 image with a red top half and a blue
  bottom half round-trips with row 0 still red and row 3 still blue, channels
  still R,G,B, and `inspect` labels both corners with those values. A row flip
  or a BGR swap would fail this step.
- **expect**: a manifest with five checks (one per corner plus the centre)
  passes, prints five PASS rows and exits 0; a wrong expectation and an
  out-of-range coordinate both fail with explanations and exit 1; a missing
  image produces one scene-level failure naming the path; a scene with an empty
  check list is rejected at load. A manifest with a UTF-8 BOM still loads.
- **diff metrics**: a known two-pixel change in a 4x3 image produces exact
  differing count, total pixels, percent, mean absolute delta, maximum delta and
  first coordinate. Identical images report zero. A 100%-different frame gets
  the wrong-tint hint naming the scene; a partial difference gets none.
- **diff edge cases**: first difference follows row-major order (a change at
  (3,0) wins over (0,1)); an alpha-only change against an RGB reference is
  counted with delta 1; mismatched image sizes refuse to compare with both sizes
  in the message.
- **diff command**: a changed scene exits 1, writes `clear.diff.png` beside the
  candidate, and the written heatmap marks a delta-255 pixel pure red. The same
  scene passes with generous thresholds while still printing its pixel count.
- **heatmap colours**: unchanged is dark grey, delta 255 is red, delta 1 is blue
  in the built raster before it is written.

### What selftest does not prove

- **Independent PNG encoders.** Selftest's PNGs come from the tool's own writer,
  and the filter fixtures are hand-built byte streams. Separately, during
  development, the reader was cross-checked on macOS against Pillow 13.0.0: 16
  images (RGB and RGBA; 1x1, 2x3, 17x9, 256x256; `optimize` on and off) decoded
  byte for byte, and their IDAT streams were observed to use filter types 0, 1,
  2, 3 and 4. Pillow is not installed on the conformance box and is not a
  dependency of this tool, so that check is not part of `selftest`.
- **The full command-line parser.** Selftest calls the command functions and the
  two argument type functions directly. The transcripts in this README exercise
  the real `argparse` wiring on Windows.
- **Unsupported PNG variants.** 16-bit, palette, grayscale and interlaced files
  are rejected by the reader with specific messages, but selftest does not
  construct each one.
- **Huge or hostile inputs.** There is no decompressed-size cap beyond the
  expected length check; a deliberately enormous or adversarial PNG is not
  tested.
- **Performance.** 256x256 is fast; a multi-megapixel image is not benchmarked.
- **That the manifest's checks are correct.** It proves the tool, not the scene
  contract.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Everything the mode checked passed. |
| 1 | The mode ran and found a mismatch: a failed check, a threshold exceeded, or a selftest assertion failed. |
| 2 | The mode could not run: bad arguments, missing/unreadable/invalid manifest, unreadable image, mismatched sizes, or an unwritable heatmap path. |

`argparse` also exits 2 for malformed command lines, which matches the table.

## Known limits and deliberate omissions

- **No enforced 256x256.** The manifest has no size field, so `expect` only
  fails when a declared coordinate falls outside the image. A wrong-size image
  whose checks all happen to be in range passes. `diff` does refuse to compare
  images of different sizes. If a size gate is wanted, add width/height to the
  manifest rather than hard-coding 256 in the tool.
- **PNG support is exactly the contract**: 8-bit RGB (type 2) or RGBA (type 6),
  non-interlaced, filter types 0-4. Everything else is rejected with a message.
  RGB must be opaque: `tRNS` transparency is rejected rather than silently treated
  as opaque. Re-export such images as RGBA with an explicit alpha channel.
- **No colour management.** Ancillary chunks such as gAMA, sRGB and iCCP are
  ignored; pixels are compared exactly as stored. This matches the scene
  contract's "no sRGB conversion".
- **APNG** is read as its default image; animation chunks are ancillary and
  ignored.
- **Exact matching in expect.** No per-channel tolerance. Use `diff` thresholds
  when measuring drift.
- **A global diff threshold**, not per-scene or per-channel limits.
- **Mean is always over four channels.** For RGB files the alpha term is zero,
  which is deliberate and consistent.
- **Heatmap is RGB** and encodes the per-pixel maximum channel delta, not the
  sum, and not a perceptual metric (no SSIM or Delta-E).
- **Manifest names are file-name constrained** because they become
  `<scene>.diff.png`; `png` must be a base name beside the manifest. Unknown
  extra JSON keys are ignored.
- **No batch/report output**: no recursive directory scan, no JSON or JUnit
  output, no parallelism. Run it once per manifest pair.
- **Stale heatmaps are not removed.** Scenes deleted from a manifest leave their
  old `<scene>.diff.png` beside the candidate until deleted by hand.
- **The writer always uses filter 0.** The reader handles all five, which is the
  part that matters for reading other encoders.
- **Python speed.** Plain nested loops are used for clarity; 256x256 is
  immediate, very large images will be slower than a vectorised decoder.

## What changed in review

The review standard asks for findings, fixes, evidence. These are the real
changes made after the first working version.

| Finding | Fix | Evidence |
| --- | --- | --- |
| `build_argument_parser` grew to 92 lines, past the one-screen rule. | Split into `add_expect_command`, `add_diff_command`, `add_inspect_command`, `add_selftest_command`; the builder is now 29 lines. | AST check reports no function over 60 lines. |
| 57 functions had no summary of what they do. | Added one-line docstrings to every function, plus fuller docstrings where a decision needs explanation. | AST check reports no function without a docstring. |
| `step_png_error_paths` read `solid.png` created by an earlier selftest step, an invisible ordering dependency. | The step now writes its own `error-fixture.png`. | Selftest passes; no step reads a file another step was assumed to have left. |
| Two paeth fixture bytes were wrong, so selftest failed on its own fixture rather than on the decoder. | Recomputed all five filtered streams with a separate throwaway encoder and corrected the fixture bytes (251 -> 30, 85 -> 30). | Filter step passes; reader cross-checked against Pillow on adaptive-filter PNGs using all five filters. |
| The heatmap endpoint test asserted pure red at deltas of 245 and 190, so the red end of the ramp was never tested. | Fixtures changed to reference red 0, candidate red 255, giving delta 255. | "heatmap delta 255 pixel is red" and the written-file check both pass. |
| Manifests written by PowerShell with a UTF-8 BOM failed to parse. | Decode with `utf-8-sig`; selftest writes and loads a BOM manifest. | Selftest line "manifest reader accepts a UTF-8 BOM". |
| `Raster.colour_name` would have called any unknown colour type "RGBA". | It now asks `channels_for_colour_type`, which validates. | Code path is shared with the reader; invalid types are rejected before a Raster exists. |
| `diff` used the reference manifest's PNG file name to locate the candidate image, so different file names in the two manifests were ignored. | `diff` now looks up the candidate scene by name and uses the candidate manifest's own `png` field. | Selftest's candidate file is `shim-clear.png` while the reference is `clear.png`, and diff passes. |
| `--max-mean-delta nan` parsed as a float; every `>` comparison against NaN is false, so all scenes would pass. | Threshold parser rejects non-finite values; selftest rejects NaN and infinity. | Selftest line "arguments: thresholds reject negative, NaN, inf and text". |
| The revised scene contract allows 2-5 checks and makes `expect` handle any count. | No code assumption existed; selftest now uses five checks and asserts five PASS rows. | Selftest line "expect: a five-check manifest exits 0". |
| The revised contract makes row order and channel order explicit. | Added an orientation selftest and labelled corner output in `inspect`; README states the flip failure mode. | Selftest line "orientation: top and bottom halves survive". |
| The revised contract wants whole-frame colour failures blamed on the tint. | `diff` prints a scene-named hint whenever every pixel differs. | Selftest asserts the hint names the scene and mentions the tint; manual wrong-tint transcript shows one line per scene. |
| A diff that failed after printing its header showed the stderr error above the header when output was captured. | `diff` and `inspect` flush stdout after the header, before a later read can fail. | The Windows wrong-size transcript now shows the header and then the error. |

## Review pass 2 — corrections

The legibility review found three claims in this README that the code did not support. They are corrected
here rather than silently rewritten above:

- **Exit codes are not identical in every mode.** `expect` treats an unreadable or missing image as a failed
  scene and exits **1**; `diff` exits **2** when it cannot compare (missing manifest, size mismatch). Both
  mean "the run did not complete", but only `diff` returns 2 for them. A crash that the tool did not
  anticipate now exits **2** with a traceback, never 1, so a calling script can tell "the pixels differ"
  from "this program failed".
- **The whole-frame mean does not separate "small area failing hard" from "large area failing softly".**
  One pixel off by 255 and 255 pixels off by 1 give the same mean. Read the mean together with
  `differing` and `max`.
- **The whole-frame hint names pixel values now.** It previously blamed the tint. An unblended draw of the
  correct colour also differs everywhere, and so does a scene that was never drawn; the first differing
  pixel's reference and candidate RGBA values are printed so a reader can tell those apart.

Also fixed: a failing `selftest` keeps its fixture directory and prints the path instead of deleting the
evidence, and a `ToolError` raised inside selftest is reported as a selftest failure (exit 1) rather than a
usage error.
