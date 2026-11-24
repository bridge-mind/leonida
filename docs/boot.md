# First boot: the runbook and the two traces

**Status: `--dry-run` and `--init` have been run on the retail executable; `--run` has not.** The package in the
repository's test corpus does not contain `game_scarlett_master.exe` (RT-5); the dry-run on an executable supplied
separately maps it and leaves 30 imports unimplemented, and `--init` runs its TLS callbacks and the package DLLs'
`DllMain`s without an exception
([`runtime_imports.md`](runtime_imports.md#the-retail-executable-rt-5)), so this page is still the procedure and the
trace tool for the first boot. The executable's `ws2_32.dll` imports bind to the host's, so run `--run` with the
network off. Nothing here quotes game data; a real trace stays on the player's machine.

## 1. What the first boot produces

`gta6rt` has three passes over the executable (`runtime/README.md`, "Dry run" and "Current limits"):

| Pass | What it does | What a clean result shows |
|---|---|---|
| `--dry-run` | maps the image, binds its imports, applies TLS | every module in `map`, no exception |
| `--init` | runs TLS callbacks and `DllMain(PROCESS_ATTACH)` | `tls_callback` and `dllmain` events, no exception |
| `--run` | calls the entry point on the game thread and returns its exit code | `exit` (or a deliberate `error`) |

Two traces are written independently:

* **startup trace** (`--trace FILE`): one JSON object per line, flushed per event, so it survives a crash. Kinds
  and fields are in [`runtime/README.md`](../runtime/README.md) ("Startup trace").
* **D3D12 trace** (`--trace-d3d12`): one `[d3d12-trace] #<seq> <method> <outcome>` line per call, on stderr. The
  ordinary `[rt]` diagnostics stay separate.

`tools/boot_trace.py` reads both and prints the triage units: modules, unimplemented calls and callers,
DllMain/TLS results, exceptions with their stack frames, the exit, and the D3D12 calls by outcome and first call
per method. The `[rt]` lines in the same stderr log carry the device parameters, feature ids and warnings; the
summary leaves them as a count, so grep `^\[rt\]` for those.

## 2. Commands

Windows (from the repository root):

```powershell
$env:LEONIDA_D3D12 = 'vulkan'
$exe = "C:\path\to\game_scarlett_master.exe"
$pkg = "C:\path\to\the\package"
$trace = Join-Path $env:TEMP boot.jsonl
$log = Join-Path $env:TEMP d3d12.log
.\runtime\build\gta6rt.exe --run $exe --game-dir $pkg --trace $trace --trace-d3d12 2>&1 |
    ForEach-Object { $_.ToString() } |
    Out-File -LiteralPath $log -Width 4096 -Encoding utf8
python -I tools\boot_trace.py $trace --d3d12 $log
```

The capture pipeline writes each message as text to avoid PowerShell 5.1 wrapping the first stderr record at a
narrow output width. The tool also reads existing UTF-16LE logs saved with `2>` in PowerShell 5.1 and recognizes
the `program.exe : ` prefix on the first stderr line.

Linux under Wine (a cross build made with xwin, or the `runtime-windows` artifact; the backend must be Vulkan):

```sh
export LEONIDA_D3D12=vulkan
wine runtime/build/windows/gta6rt.exe --run /path/to/game_scarlett_master.exe \
     --game-dir "Z:\\path\\to\\the\\package" \
     --trace "Z:\\tmp\\boot.jsonl" --trace-d3d12 2> /tmp/d3d12.log
python3 -I tools/boot_trace.py /tmp/boot.jsonl --d3d12 /tmp/d3d12.log
```

`--trace FILE` **refuses an existing file** (exit 2): it is created before the loader reads anything, so use a
fresh name or delete the old one. `LEONIDA_D3D12=vulkan` selects the Vulkan-backed device, the only backend that
writes the `[d3d12-trace]` lines; without it the layer uses the Windows D3D12 device and the trace has nothing to
read. Keep the traces out of the repository: they can quote whatever the game prints.

Useful switches:

* `LEONIDA_UNIMPLEMENTED=log` (default) logs every unimplemented import once and returns 0; `fail` returns
  `E_NOTIMPL` for the `HRESULT` imports; `stop` ends the process with exit code 3 on the first such call, after
  printing the caller's `module+RVA` and the argument registers (with a debugger attached it breaks instead). Start
  with `log`, then use `stop` to pin the first blocker and `fail` to see how the game's own failure path reacts.
  This switch covers imports only; it does not affect the D3D12 outcomes.
* `LEONIDA_SKIP_INT3=1` (with `--trace`) steps over each int3 in guest code and logs it once per address as an
  `int3_skipped` event, so a boot can be followed past soft asserts (an int3 followed by code that carries on).
  Diagnostic only: a correct run reaches none, so each skipped address is a finding, not a fix. A failed guest file
  open is always traced as `file_open_failed` (path, error, caller) when `--trace` is on.
* `LEONIDA_VK_VALIDATION=1` enables the Khronos validation layer with the Vulkan backend (startup overhead; it is a
  diagnostic).
* `DISABLE_RTSS_LAYER=1` and `DISABLE_VULKAN_OBS_CAPTURE=1` when MSI Afterburner or OBS is installed; their
  implicit layers add swapchain usage that shows up as validation errors ([testing.md](testing.md), "Known
  problems").
* `--overlay DIR` sends the guest's writes to a private folder instead of the package.

## 3. Triage order

Read the startup trace first; the first event that is not the expected one is the run's first fact.

1. **`map`** — is the executable at its preferred base (`relocated: false`), and are the expected package DLLs
   mapped? A missing module is an import-table or search-path problem.
2. **`unimplemented_call`** — the first `dll!name` and its `caller` (module+RVA). Add a shim, a host binding or a
   deliberate failure in `runtime/shims/registry.cpp`; rerun with `LEONIDA_UNIMPLEMENTED=stop` to halt at the first
   remaining blocker.
3. **`tls_callback` / `dllmain`** — a failed DllMain, or a TLS callback that faults, usually points at the CRT or
   TLS index. The summary shows the address and phase of the last observed TLS callback
   (`last_tls_callback` in JSON). The loader records it before invoking the callback, so the record gives no
   completion result. The loader's own TLS is checked by `run_tiny_game_tls`.
4. **`exception`** — code, `at`, access/address and the stack frames. The loader's vectored handler only records,
   so the process behaviour is the same with or without the trace. When the report reaches `--limit`, it reserves
   the final slot for the last observed exception. The trace also includes handled exceptions, so the last
   observation alone does not establish what caused the run to stop.
5. **`exit` / `error`** — a clean exit is not a correct frame; continue to the D3D12 trace.
6. **D3D12 trace** — the `failed`, `unsupported`, `refused` and `partial` lines name the method. A `partial`
   submission ran the supported commands from a list. The runtime's `[rt]` diagnostic names the command it left out.
   Compare the method's position with the public SDK vtable. An unknown slot is expected until the executable names it
   (Q2, Q6). The `[rt]` lines in the same log carry the 24 device words, feature IDs and warnings.
   The summary expects `#1` first, then one more than the last observed number. It counts each mismatch once and
   shows up to `--limit` expected/observed pairs. A break can reflect an excerpt, restart, duplicate or reordered
   line; it is not a count of missing calls.
7. **First frame** — after the device and queues, the first `CreateRootSignature`, PSO, `Draw*` and `PresentX` (or
   DXGI) calls are the frame's critical path. The layer's M0–M4 tests already exercise the offscreen parts; the
   present slots are the part that cannot be measured from the package alone.

The trace tool prints the first call per method and the `unsupported`/`failed`/`refused`/`partial` lists. The `--json`
option gives the same summary for a script. Share only the summary's **counts and error classes**.

If startup stops before the first D3D12 call, the combined command still prints the startup report. A missing
or invalid requested trace is reported on stderr and sets exit code 2. In a partial JSON report that section
is `null`; if neither input yields a report, stdout is empty.

## 4. The open questions, and the trace signal for each

The questions are in [`d3d12_vulkan.md`](d3d12_vulkan.md) §8.2; that table repeats 16, 19 and 20, and this one is
the de-duplicated 1–22. It maps each question to the event to look for; it does not answer them.

| # | Question | Signal in the first trace |
|---|---|---|
| 1 | `D3D12XboxCreateDevice` 24-word params and flags | first `D3D12XboxCreateDevice` in the D3D12 trace; the shim prints the 24 words as an `[rt]` line in the same log, with or without `--trace-d3d12` (grep `^\[rt\]`); accept unknown flags |
| 2 | Xbox device/heap/queue/command-list slots | queue, fence and allocator method names in the D3D12 trace; device slots past the PC vtable print an `[rt] d3d12_xs: device slot …` line instead and record nothing; compare with the SDK vtable |
| 3 | Root signature → `s0..s31` | `CreateRootSignature`, `SetGraphicsRootSignature`, then the shader's user-SGPR reads; mask `0x03000000` on a copy |
| 4 | Descriptor bytes and heap increments | first `CreateShaderResourceView`/`CreateSampler` and the first draw; dump the heap bytes and compare with the T#/S# parser |
| 5 | VA range and arena capacity | `GetGPUVirtualAddress` and every descriptor base; high-water mark |
| 6 | DXGI, `PresentX`, or both | first present/queue call; `PresentX` is a queue method past the public chain, so an unknown slot is the expected first finding |
| 7 | Standard AS build vs `XboxBvhBuilder` | first `BuildRaytracingAccelerationStructure`/`Copy...` vs slot-62 PSOs; see [embedded_shaders.md](embedded_shaders.md) |
| 8 | XDXR desc layouts | measured in [xdxr.md](xdxr.md); the live signal is a non-empty stream desc at slot 62 |
| 9 | Tiled textures / DCC | slot 76, `CopyTextureRegion`, resource descs; a `0x101` layout or tiled address |
| 10 | Binning, DCC, VRS, wave MMA, stream output | shader-side answered by `tools/feature_scan.py`; the trace adds PSO/stream subobjects |
| 11 | `CheckFeatureSupport` answers | the call's `ok`/`failed` and the `[rt]` feature-id line; the answer itself is not logged yet |
| 12 | Async compute / copy queues | queue creation and cross-queue fences |
| 13 | 144-byte push constants | device creation `[rt]` line; `push_constants_need_fallback` when the device reports fewer |
| 14 | Two code-base stages in one PSO | PSO creation log; the index metadata (`uses_code_base`) |
| 15 | `ExecuteIndirect` signatures | `CreateCommandSignature` and `ExecuteIndirect` calls |
| 16 | UAV counters and typed buffer formats | UAV descs and V# format logs; formats outside `Gen.FORMATS` read 0 |
| 17 | Upload/readback resources reaching shaders | `CreateCommittedResource` heap types and the VB/IB/descriptor that points at one |
| 18 | Placed/reserved/tiled resources | `CreatePlacedResource`, aliasing barriers, tile mappings |
| 19 | `0x03000000` root-signature bits | the root-signature blobs; the mask is a workaround, not the meaning |
| 20 | Which BVH builder programs the game uses | slot-62 PSOs on the first RT frame; map each to its `ILDN` container |
| 21 | PIX `Metadata` blob | metadata values; 0 and 1 are decoded silently, so an `unsupported` line means an unexpected value; if it is 2, decode the console blob from the executable's PIX calls, not the PC WinPixEventRuntime layout |
| 22 | `CheckFeatureSupport(OPTIONS5)` RaytracingTier | the `[rt]` feature id; the tier answer itself is not logged yet, and `DispatchRays` stays unimplemented |

## 5. What a good first run looks like

* `map` lists the executable at its base, no `relocated`, and the package DLLs the imports need.
* `dllmain` is all `1`; no `exception` before the entry point.
* The D3D12 trace starts with the device, queue, allocator, list and fence, then the first resource/PSO/draw.
* The first `unsupported` is a named method, not a crash; it is the next D3D12 call to fix. `LEONIDA_UNIMPLEMENTED`
  does not affect these outcomes.
* `exit` is the game's own code, not the loader's. A crash after the first frame is the next layer question.

## 6. Limits

The retail executable is not in the package, so this page is untested against it; the tool and the trace formats
are tested on synthetic fixtures and on `tiny_game` (which has no D3D12 calls). The present slots (Q6) and the
Xbox device parameters (Q1) remain the first things the executable itself has to name. The D3D12 trace needs the
Vulkan backend; a build without it runs the game's calls through the PC device and records nothing. The runbook
does not make the game boot; it makes the first real trace readable.
