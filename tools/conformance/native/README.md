# conform_native

Native D3D12 golden-image renderer for the conformance suite. It renders every scene in
`../spec/scenes.h` into an offscreen 256x256 `R8G8B8A8_UNORM` colour target, reads the
colour back through a readback heap, checks every `conform::PixelCheck` against the
pixels, and writes `<scene>.png` plus `manifest.json`.

No swapchain, no window, no present. One process run produces one complete set of images.

The contract lives in `../spec` and is read, never modified:

| Input | Used for |
| --- | --- |
| `../spec/scenes.h` | scene list, per-scene tint, clear values, scissor rect, geometry, PixelChecks |
| `../spec/shaders/fullscreen_color.hlsl` | `vs_6_6`/`ps_6_6`, entries `VSMain`/`PSMain` |
| `../spec/shaders/vertex_color.hlsl` | `vs_6_6`/`ps_6_6`, entries `VSMain`/`PSMain` |

## Build

```bat
cd tools/conformance/native
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build -G "Visual Studio 17 2022" -A x64
"C:\Program Files\CMake\bin\cmake.exe" --build build --config Release --parallel
```

Equivalently: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1`.

Requirements, all already present on this machine:

* Visual Studio 2022 x64 toolset (`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`).
* Windows SDK 10.0.26100 (`d3d12`, `dxgi`, `windowscodecs`; no d3dx12 helpers, no third-party code).
* `dxc.exe` at `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe`.
  Override with `-DDXC_EXECUTABLE=<path>` if the SDK moves.

The build compiles the four shader entry points into `build/shaders/*.cso`
(`-T vs_6_6` / `-T ps_6_6`, `-O3 -WX -Qstrip_debug -Qstrip_reflect`) and copies them next
to the executable. The program loads those `.cso` files at startup; if they are missing it
stops with the filename and the target that produces them rather than falling back to
anything.

## Run

```bat
tools/conformance/native\build\Release\conform_native.exe tools/conformance/native\out\run1
```

```
usage: conform_native <output-directory> [options]
  --shaders <directory>  where the .cso files live (default: <exe>\shaders)
  --adapter <index>      adapter from the printed list (default: most dedicated
                         memory, non-software)
  --debug-layer          enable D3D12 validation and fail on validation errors
  --help                 print this text
```

Exit codes: `0` every check passed, `1` a check failed or the run could not complete,
`2` bad command line. The output directory is created if absent.

Each run prints one line per check:

```
PASS readback scene=blend (128,128) expected 105,25,45,255 actual 105,25,45,255  0.5*(200,30,60) + 0.5*(10,20,30) = (105,25,45)
```

`readback` verifies the pixels copied out of the render target. `png` re-decodes the PNG
that was just written with WIC and runs the same checks against it, so a channel swap or a
truncated file in the encoder path fails the run instead of silently shipping.

## Baseline state conformance

The header's baseline table is not documentation in this renderer; it is the state that is
set. Where each row lives:

| Baseline row | Set in |
| --- | --- |
| topology `TRIANGLELIST` | `BindPipelineAndDraw` — `IASetPrimitiveTopology` |
| cull `NONE`, fill `SOLID`, `FrontCounterClockwise FALSE` | `MakeBasePipelineDescription` |
| sample count 1, sample mask `0xFFFFFFFF`, no MSAA | `MakeBasePipelineDescription` |
| `RenderTargetWriteMask = ALL` | `MakeBasePipelineDescription` |
| viewport `(0,0,256,256)`, depth 0..1 | `Renderer::Initialize` |
| scissor full target unless the scene says otherwise | `MakeDrawPlan` |
| root signature: one root CBV at b0, visibility ALL, space 0, `ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT` | `CreateSceneRootSignature` |
| constant buffer: one float4 at offset 0, `tint/255.0f` | `CreateTintConstants` (one 256-byte slot per scene), bound in `BindPipelineAndDraw` |
| depth only for `depth_order`: `D32_FLOAT`, clear 1.0, LESS, write ALL, stencil off | `ClearAndDrawScene` (DSV bound only there) + `CreateVertexColorPipeline` |
| vertex format `ClipVertex`, stride 16 | `kVertexColorInputLayout` plus two `static_assert`s on size and offset |
| index format `R16_UINT` | `CreateGeometryBuffers` |

The vertex-colour shaders do not read b0; the constant buffer is bound for those draws
anyway so all three pipelines share the single signature the header names.

## Output

Six PNGs, 8-bit RGBA (`colour type 6`), 256x256, top-left origin, plus `manifest.json`:

```json
{
  "renderer": "native",
  "adapter": "NVIDIA GeForce RTX 5070 Ti",
  "scenes": [
    {
      "name": "clear",
      "png": "clear.png",
      "checks": [
        {"x": 0, "y": 0, "r": 10, "g": 20, "b": 30, "a": 255, "why": "top-left is the clear colour"}
      ]
    }
  ]
}
```

Every check is copied from `scenes.h` (`checkCount`, up to the header's five); the diff
tool never parses C++. The manifest carries no timestamp, run counter or build path, so it
is byte-comparable between runs.

## Verified

Transcripts from this machine: RTX 5070 Ti (default pick) and AMD Radeon iGPU
(`--adapter 0`), one Windows install.

**All checks pass, both adapters.** `scripts\verify.ps1` prints `VERIFY: PASS` and exits 0:

```
ok      run A exits 0 (all PixelChecks pass)
ok      run A wrote 7 files (6 PNGs + manifest.json), found 7
ok      identical bytes: blend.png 75AD77065361FAF4951974525D00B1AB92154E04EB375FD87DB883BC10E2217F
ok      identical bytes: clear.png 28DDD65A36CF858D64E12F07D3BD910AC06B7EA9C45C20E8EB8C6F22FF5857DC
ok      identical bytes: depth_order.png B01EA5355093A8AEDF4B625422A46EE3647214C51371E4B04D6ECC4630C97C5F
ok      identical bytes: fullscreen.png 7AFB53BFBBF7FE54D59B9BFA12D46F118E81D25C1DD951596CA466D8B2A5FFEC
ok      identical bytes: indexed.png 52043F2B0F9C79DCFF1A85873642FE6DA11E3357B17D2471E8F726AAD85DFD04
ok      identical bytes: manifest.json 8029372FD2147D99F965F1C0AE31E8E051ADA1C897F7ED0D3E6C90310AC51420
ok      identical bytes: scissor.png 394DAD5B4BF2D8AF0B79132A79750A8AE40305DBA443BFACA5D84DF332028E48
ok      render on adapter 0 exits 0
ok      adapter 0 passes all 18 readback checks
ok      adapter 0 passes all 18 PNG checks
...
VERIFY: PASS
```

| Scene | Checks | Readback | PNG |
| --- | --- | --- | --- |
| clear | 3 | pass | pass |
| fullscreen | 3 | pass | pass |
| depth_order | 2 | pass | pass |
| scissor | 5 | pass | pass |
| blend | 2 | pass | pass |
| indexed | 3 | pass | pass |
| **total** | **18** | **18/18** | **18/18** |

Rendered identically on `NVIDIA GeForce RTX 5070 Ti` (default) and `AMD Radeon(TM)
Graphics` (`--adapter 0`). Same PNG hashes on the NVIDIA adapter across two runs.

**The scenes are not vacuous.** A throwaway copy with three pieces of state disabled
(`scripts/negative-control.ps1`, build discarded afterwards) produced exactly the failures
each scene claims to catch, and nothing else:

```
FAIL readback scene=depth_order (128,128) expected 40,160,90,255 actual 220,40,40,255
FAIL readback scene=scissor (40,64)  expected 10,20,30,255 actual 70,140,210,255
FAIL readback scene=scissor (180,64) expected 10,20,30,255 actual 70,140,210,255
FAIL readback scene=scissor (112,24) expected 10,20,30,255 actual 70,140,210,255
FAIL readback scene=scissor (112,104) expected 10,20,30,255 actual 70,140,210,255
FAIL readback scene=blend (128,128) expected 105,25,45,255 actual 200,30,60,128
FAIL readback scene=blend (8,8)     expected 105,25,45,255 actual 200,30,60,128
readback checks: 11 passed, 7 failed
```

That is: with `DepthEnable = FALSE` the far red quad drawn second wins (proving the depth
test is what keeps green); with the scissor widened to the full target the four outside
checks see the tint; with `BlendEnable = FALSE` the source colour lands unchanged, alpha
128 included. Clear, fullscreen and indexed still passed, so the failures are attributable.

**The images are the expected shapes.** Read the PNGs back and looked at them:
`depth_order.png` is a full-height red rectangle with a green rectangle over its left two
thirds and a clear border where the quads stop at ±0.95; `scissor.png` is a filled blue
rectangle inset from every edge (the 48..176 × 32..96 rect); `indexed.png` is two right
triangles, yellow and blue, with clear space between and below them.

**PNG structure.** IHDR of all six files: `width 256 height 256 bitdepth 8 colortype 6`.

**Shader bytecode.** `dxc -dumpbin` on the four `.cso` files reports
`EntryFunctionName: VSMain` / `PSMain`, `Vertex Shader` / `Pixel Shader`, and the input
signatures the code assumes: `SV_VertexID` for the fullscreen VS, `POSITION`/`COLOR` for
the vertex-colour VS, `SV_Position`/`COLOR` for its PS, and a `cbuffer` in the fullscreen
PS.

**Failure paths exercised** (`scripts/failure-log.cmd`, merged stdout+stderr, all exit
non-zero, all naming the file and the next thing to check):

```
=== output path is an existing file
FAIL tools/conformance/native\src\main.cpp:154 std::filesystem::create_directories -> could not create tools/conformance/native\out\failures\blocked: Cannot create a file when that file already exists. Check: the path is a directory the process can write to, not an existing file and not a read-only volume
EXITCODE 1

=== shader directory with no .cso files
FAIL tools/conformance/native\src\renderer.cpp:420 std::ifstream::open -> could not open tools/conformance/native\out\failures\no-shaders\fullscreen_color.vs.cso. Check: the build ran the conform_shaders target and the shaders directory sits next to conform_native.exe; use --shaders to point elsewhere
EXITCODE 1

=== PNG path occupied by a directory
PASS readback scene=clear (0,0) expected 10,20,30,255 actual 10,20,30,255  top-left is the clear colour
FAIL tools/conformance/native\src\png_io.cpp:61 IWICStream::InitializeFromFilename -> HRESULT 0x80070005. Check: the output directory is writable and nothing else holds tools/conformance/native\out\failures\png-blocked\clear.png open
readback checks: 3 passed, 0 failed
png checks:      0 passed, 0 failed
RESULT: FAIL (run did not complete)
EXITCODE 1
```

The third case is an unwritable output path (a directory occupies `clear.png`): rendering
and readback verification still ran and passed for that scene; the run exits non-zero
because no PNG was written. Command-line errors exit 2 and print usage (no arguments,
unknown flag, missing option value, bad `--adapter` value, two positional directories).

## Decisions a reviewer should agree with, or overrule

1. **Three pipeline states, not two.** D3D12 bakes blend state into a PSO and has no
   `OMSetBlendState`, so the blend scene needs its own. Fullscreen and blend share the root
   signature and shaders and differ only in `D3D12_BLEND_DESC`, so the split isolates blend
   rather than duplicating anything.
2. **Scissor is per-scene state, not pipeline state.** `D3D12_RASTERIZER_DESC` has no
   `ScissorEnable` (that is D3D11). D3D12 applies the scissor rect whenever one is set, so
   the scissor scene is the fullscreen pipeline plus `RSSetScissorRects(48,32,176,96)`.
   The negative control above shows the rect is load-bearing: widen it and four checks fail.
3. **One DSV binding, for `depth_order` only.** The header's baseline says every other
   scene binds no DSV, so `OMSetRenderTargets` passes `nullptr` for them. The indexed scene
   uses the depth-enabled vertex-colour PSO with no DSV bound, which leaves depth testing
   inert; that is the same PSO the contract asks for, not a fourth state.
4. **One 256-byte constant-buffer slot per scene.** A root CBV must be 256-byte aligned and
   each scene now carries its own tint. Each slot's float4 sits at byte offset 0 of its
   slot, so the shader sees exactly "a single float4 at offset 0". Slots are filled once
   before any draw and never rewritten, so no CPU write can race a GPU read.
5. **No sRGB anywhere.** Target format is `R8G8B8A8_UNORM` (not `_SRGB`), clears are
   `byte/255.0f`, and WIC writes the bytes it is given. Nothing converts.

## Not covered

* **D3D12 validation was not available on this machine.** `--debug-layer` reports
  `the D3D12 debug layer is not installed (HRESULT 0x887A002D)` and continues without it.
  The plumbing (`ID3D12InfoQueue` drain, failure on validation errors) is implemented but
  has never validated anything here. Install the Graphics Tools optional feature to
  exercise it. This is the largest gap in the verification.
* Only two adapters were tried, both on one Windows install. The software adapter
  `Microsoft Basic Render Driver` (index 2) was listed but not rendered on.
* Device-removed and GPU-hang paths (`GetDeviceRemovedReason`, 60 s fence timeout) are
  coded but never triggered; forcing a TDR needs a debugger or a fault-injection build.
* PNG determinism is verified on this WIC build only. WIC's zlib output is stable here, but
  that is an observation about this machine, not a documented Windows guarantee. The
  pixels are deterministic; the compressed bytes are a WIC implementation detail.
* The depth buffer is never read back, so depth correctness is inferred from colour. The
  negative control closes that gap for this scene (colour changes when depth is disabled),
  but there is no direct depth-value check.
* 16-bit indices are configured (`DXGI_FORMAT_R16_UINT`) and the shuffled index buffer
  draws the expected triangles, but no scene has more than 8 vertices, so nothing proves
  behaviour past 65 535 indices.
* `scenes.h` allows at most 5 checks per scene (`PixelCheck checks[5]`); the manifest and
  verifier iterate `checkCount` and would read past the array if the contract raised it
  without changing the struct.
* No swapchain, present, resize or multi-frame path is implemented or tested; out of scope
  for a golden-image tool.
* Nothing here validates the Vulkan shim's output. That comparison needs the diff tool.

## Contract revision (05:07)

The shared header was replaced mid-work. Changes and how they landed:

| Header change | What the renderer does now |
| --- | --- |
| baseline pipeline table | one root signature (CBV b0, visibility ALL, `ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT`); see the conformance table above |
| `depth_order` draw order reversed | unchanged code path: indices are drawn in header order, so the near green quad is now first and depth rejects the later red one |
| `indexed` vertices/indices shuffled, third check | unchanged code path: the vertex and index arrays are uploaded as declared |
| scissor rect `{48,32,176,96}`, five checks | `MakeDrawPlan` reads `kScissor*`; checks at all four outside edges now run |
| per-scene tint (`tintR/G/B/A`) | one CB slot per scene, filled from each `Scene` |
| `checks[5]`, 2..5 per scene | verifier and manifest already iterate `checkCount` |
| blend alpha `ONE/INV_SRC_ALPHA` | PSO updated from the previous `ZERO/ONE`; `SrcBlendAlpha = ONE`, `DestBlendAlpha = INV_SRC_ALPHA` |

After the revision: 18/18 readback and 18/18 PNG on both adapters, and determinism holds
(`verify.ps1`: `VERIFY: PASS`). Only `scissor.png` and `manifest.json` changed bytes
relative to the pre-revision run; the other five PNGs are byte-identical, which is what the
contract predicts — the reversed depth order and shuffled indices produce the same image
when implemented correctly, and 128/255 with `ONE/INV_SRC_ALPHA` resolves to the same
8-bit blend result the header says it should.

## What changed in review

Three passes in the order `REVIEW.md` asks for. Every fix was re-verified by the build and
`scripts\verify.ps1` after the change.

**Pass 1 — functional.**

* Finding: first build failed with `C2065: 'D3D12_COMMAND_QUEUE_TYPE_DIRECT'` and
  `C2039: 'ScissorEnable' is not a member of 'D3D12_RASTERIZER_DESC'`.
  Fix: queue type is `D3D12_COMMAND_LIST_TYPE_DIRECT`; the scissor PSO was deleted because
  D3D12 has no scissor enable bit, and the scissor rect became per-scene state.
  Evidence: `BUILD EXIT 0`; scissor checks pass and fail correctly in the negative control.
* Finding: the first run passed everything, which is not evidence by itself.
  Checked the blend arithmetic by hand and changed the alpha channel to the header's
  `ONE/INV_SRC_ALPHA`; then built the negative control to prove the passing scenes react to
  their state. Evidence: negative-control transcript above.
* Finding: a stray closing brace after `CreatePipelines` (bad edit) broke the build with
  `C2059`.
  Fix: removed. Evidence: `BUILD EXIT 0`.

**Pass 2 — legibility.**

* Finding: FAIL diagnostics appeared above the program banner because stdout was buffered
  and stderr was not. Fix: `setvbuf(stdout, nullptr, _IONBF, 0)` in `main`. Evidence: the
  merged failure transcript now reads banner → `FAIL …` → `RESULT: FAIL`.
* Finding: `Check: could not create <path>` read as a broken sentence under the
  `FAIL file:line call -> detail. Check:` shape. Fix: the hint is now a testable claim
  ("the output directory is writable and nothing else holds `<path>` open").
* Finding: a doubled period from concatenating a Win32 message that already ends in `.`
  with the `. Check:` separator. Fix: `TrimTrailingPeriod`. Evidence: `… already exists.
  Check: …`.
* Finding: `Renderer::full_frame_scissor` and `GpuContext::adapter_index` were written and
  never read. Fix: both removed. Evidence: clean rebuild under `/W4 /WX`.
* Finding: `#include <cstring>` in `gpu.cpp` with no use. Fix: removed.
* Finding: the marker `TintKind` and two hard-coded tint addresses became dead state when
  the header moved to per-scene tints. Fix: replaced by the scene's tint slot; `grep` for
  `TintKind`/`kTint*` now returns nothing.

**Pass 3 — verify and fix.** Clean rebuild from an empty `build\`, then
`scripts\verify.ps1`: 30 claims, `VERIFY: PASS`, exit 0, plus the negative control and the
failure-path log above.

## Layout

```
CMakeLists.txt                 build, shader compilation, shader copy
scripts/build.ps1              configure + build
scripts/run.ps1                run one output directory with extra arguments
scripts/verify.ps1             re-checks every claim in this README, exits 0 only if all hold
scripts/failure-log.cmd        exercises the three failure paths and prints one merged log
scripts/negative-control.ps1   throwaway build with depth/blend/scissor disabled, then deleted
src/main.cpp                   arguments, COM, run loop, tally, exit code
src/gpu.h/.cpp                 adapter choice, device, queue, fence, validation drain
src/renderer.h/.cpp            targets, buffers, root signature, PSOs, recording, readback
src/scene_plan.h/.cpp          Scene -> pipeline/geometry/tint slot/depth binding/scissor
src/png_io.h/.cpp              WIC encode and decode
src/verify.h/.cpp              PixelCheck comparison and PASS/FAIL output
src/manifest.h/.cpp            manifest.json with JSON escaping
src/pixel_surface.h            R8G8B8A8 pixel buffer shared by readback and decode
src/diagnostics.h/.cpp         failure lines: file, call, HRESULT/error in hex, what to check
```

## Review pass 2 — legibility fixes applied

A second review pass read every file against the review notes in the pull request. These were fixed and re-verified;
each line names the failure it prevents.

| fix | file | the failure it prevents |
|---|---|---|
| Severity comparisons inverted | `gpu.cpp` `ReportValidationMessages` | `D3D12_MESSAGE_SEVERITY` counts **down** as severity rises (CORRUPTION 0 … MESSAGE 4), so `>= WARNING` printed only INFO/MESSAGE and `>= ERROR` counted everything except corruption. Real errors were printed as nothing and corruption was ignored. |
| `--debug-layer` now fails when the layer is absent | `main.cpp`, `gpu.cpp`, `gpu.h` | It printed `validation errors: 0` and could end `RESULT: PASS` on a machine where validation never ran. It now prints `validation: NOT RUN` and exits 1. |
| Enum switches can no longer fall through | `CMakeLists.txt` (`/we4062`), `scene_plan.cpp` | `DrawPlan` defaults to `{Fullscreen, None}`, so a new `SceneKind` would have rendered as a clear-only scene with no message. C4062 is off under `/W4`, so `/WX` never caught it. |
| Renderer released before the device | `gpu.h`, `gpu.cpp`, `main.cpp` | `DestroyGpuContext` ran while `Renderer` still held resources, PSOs and command lists. A `GpuContext` destructor now runs after the renderer's, and the manual teardown calls are gone. |
| Readback pitch derived, padding stripped | `renderer.cpp` | `kReadbackRowPitch` was hard-coded to `width*4` and the copy used the pitch as the row length, so any width whose pitch is padded would have produced sheared rows. |
| Index type pinned at compile time | `renderer.cpp` | The format (`R16_UINT`) and `sizeof/sizeof` counts were independent of the array types; a switch to `uint32_t` would have doubled every count with no build error. |
| Check counts validated at compile time | `verify.cpp` | A scene with `checkCount == 0` passed with nothing checked; a count above 5 read past `checks[5]`. Now a `static_assert`. |
| Validation messages that cannot be read are counted | `gpu.cpp` | `GetMessage` failures were skipped silently, hiding queue entries. |
| Negative control asserts its own result | `scripts/negative-control.ps1` | The script printed and hoped. `Replace` calls that silently did nothing produced no evidence, and the README quoted output that did not match what the current code prints. It now throws when a replacement does not match exactly once, and fails unless exactly `blend`, `depth_order` and `scissor` fail. |

Deferred, and why: `HrSucceeded` still carries the `__FILE__`/`__LINE__` pairs instead of taking a
`std::source_location`, because renaming it touches 69 call sites and the functional pass would have to be
re-run entirely; the name is misleading but the behaviour is correct.
