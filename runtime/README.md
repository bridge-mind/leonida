# gta6rt

`gta6rt` is the native Windows runtime for the Xbox Series build. It is deliberately small: the game
executable is still missing from the package used here, so this directory contains the machinery that
can be built and tested without it — the PE loader, the measured import shims, and the GPU/Vulkan test
tools.

## Build

The normal path is the repository build script:

```powershell
powershell -File scripts/build_runtime.ps1
```

It uses CMake + Ninja from Visual Studio when available, otherwise `cl.exe`, and writes
`runtime\build\gta6rt.exe` plus `gta6rt_tls_host.dll` (the loader's TLS-index helper; it must stay next to
the executable). The direct CMake path works too:

```powershell
cmake -S runtime -B runtime/build
cmake --build runtime/build --config RelWithDebInfo
```

Requirements: Windows x64, Visual Studio 2022 with the C++ desktop workload. The GPU tools need the
Vulkan SDK; the runtime and CPU tests do not. The loader is x64-only and the PE mapper rejects any
other machine.

## Dry run

```powershell
gta6rt --dry-run <image.exe|image.dll> [--game-dir <package>] [--overlay <dir>]
```

`--overlay` is where a mapped title's writes go (default `<game-dir>\overlay`); `GAME:` paths read from
`--game-dir`. See [Console paths and the write overlay](../docs/runtime_imports.md#console-paths-and-the-write-overlay).

The dry run maps the image and the bundled modules it imports, applies base relocations, binds every
import through the shim registry, a host DLL, or another bundled module, parses exports, installs static
TLS for the calling thread, applies section protections, and
prints a per-module report. It does **not** call `DllMain`, TLS callbacks, or the entry point, so no
code from the mapped images runs. TLS reserves its module index through the runtime's own
`gta6rt_tls_host.dll`, and host imports load system DLLs, as they did before.

`gta6rt --init <image>` runs the same mapping and then `Loader::initialise()`: each module's TLS
callbacks and each DLL's `DllMain(DLL_PROCESS_ATTACH)`, dependencies first. An EXE's entry point is not
run. When the loader is destroyed, initialised modules receive TLS callbacks and DLL entry-point calls with
`DLL_PROCESS_DETACH`, in reverse initialisation order. All mapped images, unwind tables and static TLS stay
available until these notifications finish, so DLLs can release resources such as FLS callbacks safely. The reserved
argument is null (explicit unload). A map-only load receives no detach notifications.

`--dry-run`, `--init` and `--run` are separate modes: give exactly one. Two together (`--dry-run --run`)
print the usage and exit 2 before anything is mapped.

The report line is:

```
<name>  base=<addr> size=<image size> entry=+<rva>  shim=N host=N module=N unimplemented=N
```

Shim and unimplemented imports are listed by name. A non-zero `unimplemented` count is the useful
signal: it names the next shim that is missing. The measured result for the 16 DLLs shipped in the
package is recorded in the runtime-imports documentation.

After the totals the report counts the unimplemented imports by return class, and marks the HRESULT ones in the list:

```
    UNIMPLEMENTED xgameruntime.dll!XUserAddAsync  [returns HRESULT]
1 module(s) mapped, 1 unimplemented import(s)
unimplemented by return policy: hresult=1 other=0
```

### Unimplemented imports at run time: `LEONIDA_UNIMPLEMENTED`

An import with no shim, host function or bundled module is bound to a thunk. What a call to it does is set by the
environment variable `LEONIDA_UNIMPLEMENTED`, read once per process:

| value | a call to the thunk |
|---|---|
| `log` (default) | prints `[rt] unimplemented: dll!name` once and returns 0 in rax and xmm0. This is what the runtime always did. |
| `fail` | the same, but returns `E_NOTIMPL` (`0x80004001`) for imports known to return an `HRESULT`, and 0 for all others. |
| `stop` | on the first call prints the import, the caller's return address as `module+RVA`, and rcx, rdx, r8 and r9 as the game passed them; then breaks if a debugger is attached (continuing returns the `fail` value), else ends the process with exit code 3. |

No mode writes to an out parameter: a function that fills one in on success leaves it as the caller had it. Under
`log` an `HRESULT` function therefore returns `S_OK` with its outputs unset, which is why `fail` exists; with `fail`
the caller sees an error and can take its own failure path. Any other value is reported and treated as `log`.

Whether an import returns an `HRESULT` is decided by DLL and name alone (`classify_import` in
`shims/registry.cpp`, where each entry names its source). The list is short on purpose and every entry was checked
against the public Microsoft documentation: `D3D12Create*`/`D3D12Serialize*` and the D3D12 debug/experimental
entries, `CreateDXGIFactory*` and two DXGI functions, `XGameRuntimeInitialize`, and the GDK families `XAsync*`,
`XTaskQueue*`, `XUser*`, `XStore*`, `XGameSave*`, `XPackage*`, `XGameUi*` minus the functions the same pages document
as `void`/`bool`/`int`. A function that is not on the list gets 0 under `fail`, exactly as under `log`.

The thunk keeps the Win64 argument registers and the return address intact (it is entered by the game's `call`), and
reaches a shared stub that saves them, so a floating-point caller still reads 0.0 from xmm0 and a debugger can unwind
through it.

## PE loader

`src/pe_loader.cpp` maps x86-64 images at their preferred base, or elsewhere (with relocations) when
that base is taken, copies headers and sections, applies `IMAGE_REL_BASED_DIR64` relocations, binds
imports by name or ordinal, parses the export table, sets section protections, and registers the
image's x64 unwind table (`.pdata`) with `RtlAddFunctionTable`, so exceptions raised in or passing
through a mapped image are dispatched and stack walks can step over its frames. The table is validated
first (entries in bounds, sorted and non-overlapping) and removed when the loader is destroyed. Modules
are deduped by basename, so an import graph that reaches the same DLL twice maps it once. The identity
key uses the filesystem's native path representation with ASCII case folded; it does not require a
Windows filename to fit the ANSI code page. A lossless legacy narrow filename alias retains the existing
bytewise case behavior for narrow API and import-name lookups. `Module::name`, mapped address labels,
and path diagnostics use UTF-8. The `LoadLibrary*`/`GetModuleHandle*` shims look a wide name up by its native
basename when it has a character outside ASCII (`Loader::find_loaded_module_w`, `load_runtime_module_w`), and
module names compare case-insensitively with the system's own table (`CompareStringOrdinal`), as Windows
compares them; a narrow name outside ASCII compares the same way through the ANSI code page.

`module_names` checks native and mapped renamed DLLs, distinct Unicode identities, basename cache
reuse, existing ACP aliases, the wide and case-insensitive lookups, export execution, and path errors. On native Windows with Python available,
`run_module_names` compares original renamed EXEs through native execution and `--dry-run`, `--init`,
and `--run`, including a Unicode parent directory and an ACP-named imported DLL. It checks strict UTF-8
trace names and unchanged fixture bytes. Both tests use only locally built synthetic images.

MSVC C++ exceptions need one more piece. `_CxxThrowException` stores the throwing image's base in the
exception record, from `RtlPcToFileHeader`, and the catch side adds the throw's RVAs to it; Windows' own
`RtlPcToFileHeader` only knows images its loader mapped, so for a mapped image the base is 0 and no
`catch` matches. When the loader binds a host copy of the MSVC runtime (`vcruntime140.dll`,
`msvcp140.dll` and the rest of that list), it points that DLL's import of `RtlPcToFileHeader` at a
version that asks Windows first and then the modules a live `Loader` maps. The redirect stays for the
process lifetime and answers as Windows does when no `Loader` is alive.

`load()` also installs static TLS for the calling thread. `__declspec(thread)` reads `TEB->ThreadLocalStoragePointer[_tls_index]`, which is a
different index space from the `TlsAlloc` slots, and only the OS loader can allocate a module index.
The runtime therefore ships a minimal `gta6rt_tls_host.dll`; the loader reuses the index the OS assigns
it, writes that index into the image's `AddressOfIndex`, and copies the image's TLS template into the
slot. This reserves one index; every additional simultaneous TLS module gets its own by mapping a fresh
clone of the helper from `%TEMP%\leonida_tls_host_<pid>_<slot>.dll`, so up to 16 modules can be live at
once and a further one is rejected with an error instead of sharing a slot. Images without a TLS
directory (or with an all-zero one) load normally.

Current limits, by design:

- `load()` runs no image code. Static TLS is data, not code. The TLS index helper is the runtime's own
  `gta6rt_tls_host.dll`; host imports still load system DLLs as before.
- `--dry-run` never runs image code. `initialise()` runs TLS callbacks and a DLL's `DllMain`;
  `run_tls_callbacks()` is the seam for callers that do not use `initialise()`. An EXE's entry point is
  the caller's to run.
- Every thread started after `load()` gets its own copy of the TLS template, from
  `gta6rt_tls_host.dll`'s `DLL_THREAD_ATTACH`, and gives it back at `DLL_THREAD_DETACH`, also when the
  thread outlives its `Loader`. From `initialise()` until `~Loader` starts the process detach, those
  events also run the module's TLS callbacks, with the thread's copy already in place. A thread that
  was already running when `load()` ran, such as a worker of the process thread pool that the task queue later runs a
  title's callback on, gets its copy from `load()` itself, the way Windows gives one to running threads for a DLL it
  loads: a block, no `DLL_THREAD_ATTACH`, `DLL_THREAD_DETACH` when it ends, and `~Loader` takes the block back.
  `install_tls_for_thread()` and `uninstall_tls_for_thread()` are no-ops on such a thread. Mapped DLLs' `DllMain` is not called with thread events. Destroy the `Loader` on the thread that
  called `load()`.
- The TOC of the game's RPF archives is encrypted and is not read here. The loader works on the PE
  files and on any images the caller passes it.

## Shims

`shims/registry.cpp` holds the static tables. The current shims are:

- `xmem.dll` — `XMemAlloc`, `XMemFree`, the measured two-argument `XMemGetWorkingSetStatistics(kind, out)`, and the
  measured three-argument
  `XMemVirtualQuery(process, address, out)`. Allocations are 64-byte aligned; the working-set and
  virtual-query outputs fill the observed 64-byte structures and return `FALSE`/`0` on a null or bad
  address. `XMemAllocDefault`/`XMemFreeDefault` share that heap. `XMemVirtualAlloc` accepts the attribute word,
  drops the console's `PAGE_GRAPHICS_*` protection bits, commits 2 MiB guest pages 2 MiB-aligned,
  and passes `MEM_RESERVE_PLACEHOLDER`/`MEM_REPLACE_PLACEHOLDER` through. Physical pages
  (`XMemAllocatePhysicalPages`, `XMemFreePhysicalPages`, `XMemMapPhysicalPages`, `XMemMapPhysicalPagesScatter`) are
  64 KiB pages of pagefile-backed sections, mapped one view per page into the placeholder that a reserve-only
  64 KiB-page `XMemVirtualAlloc` becomes, with that reservation's protection. `XMemIsToolsMemAvailable` is `FALSE`.
- The `kernel32.dll` file API — `CreateFileA/W`, `FindFirstFileA/W`/`FindNextFileA/W`/`FindClose`,
  `GetFileAttributesA/W`, `GetFileAttributesExA/W` and `DeleteFileA/W`: a `GAME:` path maps under `--game-dir`, an
  open that can write and a delete go to `--overlay DIR` (default `<game-dir>\overlay`), which reads prefer, and a path
  that climbs above `GAME:` is refused. The package is never modified. See
  [docs/runtime_imports.md](../docs/runtime_imports.md#console-paths-and-the-write-overlay).
- `d3d12_xs.dll` — `D3D12SerializeVersionedRootSignature` and
  `D3D12CreateRootSignatureDeserializer`, with the console root-signature high byte (`0x03000000`)
  cleared on a copy before forwarding to the PC `d3d12.dll`. `D3D12XboxCreateDevice` wraps a device in
  the measured-slot Xbox-ABI proxy: the host PC `ID3D12Device` by default, or the Vulkan-backed device
  (`shims/d3d12_vulkan_device.cpp`) when `LEONIDA_D3D12=vulkan` is set and the runtime was built with the
  Vulkan SDK and Python 3. The Vulkan device covers command submission, descriptor heaps, render-target and
  depth views, committed resources and copies (uploads, readbacks, BC textures), root signatures, graphics and
  compute pipeline states, draws, dispatches and `ExecuteIndirect`; see `docs/d3d12_vulkan.md` for what is
  missing.

- `gameinput.dll` / `xgameruntime.dll` — `GameInputCreate`: the v0 GameInput API for gamepads on SDL3
  (`shims/gameinput.cpp`), so any controller SDL supports works, with rumble, Xbox trigger rumble, force feedback
  and trigger haptics where SDL opens the pad as a haptic device, and on a DualSense adaptive-trigger effects and the
  lightbar. Optional and CMake-only: built when an installed SDL3 is found,
  or with `-DLEONIDA_SDL3_FETCH=ON` to download the pinned release; nothing is downloaded otherwise, and the
  `scripts/build_runtime.ps1` cl.exe fallback never builds it. See [docs/gameinput.md](../docs/gameinput.md) for the mapping, the `LEONIDA_DS_*` settings and what is
  unverified.

### What `xgameruntime.dll` reports: environment overrides

The `xgameruntime` shim (`shims/xgameruntime.cpp`) answers the title's questions about the console and its package
with fixed values. Five environment variables change an answer, for trying another of the title's code paths. Each
is read on every call, and a value that is empty or too long for its buffer is ignored.
`tests/test_docs_env_vars.py` fails when the runtime or `make_game.py` reads a `LEONIDA_*` variable that no
document names.

| variable | default | what it changes |
|---|---|---|
| `LEONIDA_DEVICE_TYPE` | `7` | The `XSystemDeviceType` that `XSystemGetDeviceType` returns: `0` Unknown, `1` Pc, `2` XboxOne, `3` XboxOneS, `4` XboxOneX, `5` XboxOneXDevkit, `6` XboxScarlettLockhart (Series S), `7` XboxScarlettAnaconda (Series X), `8` XboxScarlettDevkit. A number outside 0..8 is ignored; text that is not a number counts as `0`. `6` also makes `XSystemGetAnalyticsInfo` report the form `Xbox Series S` instead of `Xbox Series X`. |
| `LEONIDA_ANALYTICS_FORM` | unset | The `form` string of `XSystemGetAnalyticsInfo`, in place of the one the device type gives (up to 63 characters). |
| `LEONIDA_PACKAGE_ID` | `LeonidaPackageIdentifier00000000` | The identifier `XPackageGetCurrentProcessPackageIdentifier` returns (up to 32 characters). It is also the one identifier `XPackageMount` refuses with `E_ACCESSDENIED`, as the game's own package. |
| `LEONIDA_CONTENT_PACKAGES` | unset | The content packages `XPackageMount` can mount: `id1=path1;id2=path2` (up to 1,023 characters). The mount path of a listed identifier is its path, as written; any other identifier is `ERROR_NOT_FOUND`, and a path longer than 259 characters is `ERROR_INSUFFICIENT_BUFFER`. |
| `LEONIDA_PACKAGED` | `1` | `0`, or text that is not a number, makes `XPackageIsPackagedProcess` return false. Any other number keeps it true. |

The two serializer functions and `D3D12XboxCreateDevice` are the only `d3d12_xs` imports used by the
shipped DLLs (measured; see `docs/design.md` and `docs/d3d12_vulkan.md`). The game executable's own
imports are unknown until it is available.

## Tests

For a Visual Studio build, pass CTest the same configuration you built above.

```powershell
ctest --test-dir runtime/build -C RelWithDebInfo --output-on-failure
```

On Windows the build configures the CPU tests that do not need a GPU or game files, including the
root-signature forwarding test (`root_sig_test`) and the clear-command test. The Vulkan tools and
D3D12-over-Vulkan COM targets are built when Vulkan headers are found (via `VULKAN_SDK`, an installed
SDK, or `-DLEONIDA_VULKAN_HEADERS_FETCH=ON` to download the pinned Khronos Vulkan-Headers release):

- `vk_harness` — hand-written RDNA2 shader programs on a real Vulkan device.
- `vk_texture_test`, `vk_compute_test`, `vk_store_test` — texture, LDS/compute, and buffer-store paths.
- `vk_command_replay_test`, `d3d12_com_test`, `d3d12_vulkan_device_test` — the D3D12 command path on
  Vulkan: replay, the COM command objects, and the Vulkan device behind the Xbox-ABI proxy.
- GPU tests count only validation-type errors. Loader and layer messages, such as a third-party implicit layer
  that fails to load, are printed but don't fail the run; to silence one, set
  `VK_LOADER_LAYERS_DISABLE=<layer name pattern>`.
- `d3d12_draw_test` — milestone M1: a VS+PS pipeline found by FSSM hash, the input layout matched by semantic,
  and five draws plus a culled triangle, compared pixel by pixel. They also cover an unbound render target and
  DSV, a rebound root signature, instance step rate 0, an unbound vertex slot, and PSOs, buffers and heaps that
  must be refused. A second pass draws a full-screen triangle with no vertex input and an XOR logic op.
- `d3d12_translated_draw_test` — milestone M1.5: the same with translator output. `test/shaders/vs_m15.asm` (vertex
  fetch through a V#) and `ps_m15.asm` (constants read through the code base) are built by `test/shaders/build.py`
  and found by FSSM hash; three draws into a float target, compared bit for bit. Needs `llvm-mc` (`LLVM_MC`) and the
  Vulkan SDK's `spirv-as` and `spirv-val` at build time.
- `d3d12_compute_test` — the compute path: root signature, a compute PSO found by its FSSM hash in a
  translator-style index, root arguments through the GPU address arena, and two dispatches. A second program is
  found the same way in `<shaders>/bvh/index.json`, where the BVH builder's programs go (`docs/embedded_shaders.md`).
- `d3d12_fspx_compute_test` — compute PSOs from synthetic FSPX programs (`docs/fspx.md`): an `ACCC` container
  through slot 62 and a bare kind-6 program, found by `fspx_code_sha256` in `<shaders>/fspx/index.json`, one
  dispatch each that reads its kind-6 program as the code base; malformed directories and untranslated programs
  are refused with `E_INVALIDARG`. Also slot 67 with all four pairs (picked by kind 0), refused pairs, and the
  pipeline's console slot 11.
- `d3d12_set0_compute_test` — set 0 from the compute bind point: a compute PSO whose index entry lists set-0
  bindings, and four dispatches that sample two SRVs, bytes no view was made for and a clamping sampler through
  the bindless set the layer binds.
- `d3d12_present_test` — milestone M0: opens a window and presents 60 cleared frames through the proxy on
  Vulkan, resizes it and presents 10 more, and checks the colour in the presented images (the presenter's
  capture) and on screen where GDI can read it. A driver double then injects an
  out-of-date and a suboptimal present, a failed semaphore during a swapchain recreation and a failed blit
  submission, which the presenter must survive. Set
  `LEONIDA_VK_VALIDATION=1` to run them under the Khronos validation layer.
- `vk_pipeline_check <spirv dir> [cache.bin] [--jobs N] [--start I] [--count N] [--max-bytes N]` -
  creates a shader module and, for compute shaders, a pipeline for every entry in the translator's
  `index.json`; reports `MODULE FAIL` / `PIPELINE FAIL` per shader and a summary. `--max-bytes`
  names and skips modules larger than the limit, which is how the corpus run works around the
  driver's compile time on multi-megabyte ray-tracing shaders. The measured partial pass is in
  [docs/pipeline_check.md](../docs/pipeline_check.md).

`gameinput_test` (only with SDL3; otherwise CTest lists `gameinput` as not run) drives the GameInput shim on SDL virtual joysticks (no controller needed): enumeration, readings
and the button/axis mapping, rumble and trigger rumble, DualSense effect bytes, a dispatcher, and hot-plug.
`gameinput_probe [--seconds N] [--rumble] [--force-feedback]` is the manual check on a real pad: it prints live
readings, plays each motor and trigger, and holds a constant force on a pad that reports a force-feedback motor.

`unimplemented_policy_test` builds a small image whose two imports bind to nothing and checks `log`, `fail` and
`stop` on it: return values, untouched out parameter, xmm0 = 0, environment parsing, and (in a child process) the
`stop` message with the caller's module+RVA and the argument registers. It also checks that the stub keeps the stack
16-byte aligned at the call into the runtime, that rbx/rbp/rdi/rsi/r12-r15 and xmm6-xmm15 survive a call
(`test/fixtures/abi_probe.asm`, MASM; skipped without `ml64`), and that `RtlVirtualUnwind` from inside the call walks
back to the caller.

The PE-loader and Bink-decode tests that need the retail package are optional: when
`LEONIDA_GAME_DIR` (and, for Bink, `LEONIDA_MOVIE`) is unset they print `SKIP` and exit 0. They read
the files in place and never copy or commit game data.

`shader_library_corpus_test` runs the D3D12 layer's shader lookup (`src/shader_library.h`) over a work folder
that `make_game.py --stages shaders` built from your copy. With `LEONIDA_SHADERS=<work>/shaders/spirv` it takes
every container in `<work>/shaders/dxbc` (or `LEONIDA_SHADER_DXBC`), inflates its FSSM program and looks it up
under the stage its DXIL program header names. Each container has to find its own `index.json` entry and a
readable `.spv`, every entry has to be reached, and in containers with more than one 0x0D payload no earlier
payload may be the indexed program. Without `LEONIDA_SHADERS` it prints `SKIP` (CTest reports it as skipped).
It prints counts only; on the 1.0.2.1 corpus all 26,503 containers resolve.

## Current limits

`gta6rt --run <exe> [-- args]` runs an executable's entry point inside the runtime (`shims/process.cpp` gives it its
own command line and turns its `exit` into the end of the run); CTest's `run_tiny_game` does that with a stand-in game.
The game's `atexit` handlers and static destructors go to a table of the run's own (the CRT's process-wide one is
shared with gta6rt): `exit` and a return from `main` run them newest first before the run ends, `_cexit` runs them and
returns, and `_exit`, `quick_exit`, `ExitProcess` and `abort` skip them, as natively. `run_tiny_atexit` runs a stand-in
program natively and through `--run` for each of those paths (and for `_exit` from a second thread during `exit`) and
holds both to the Windows sequence (under Wine, whose CRT differs, a native difference is only reported). Not covered
yet: `at_quick_exit` handlers (`quick_exit` runs none) and the thread-local destructor callback
(`_register_thread_local_exe_atexit_callback`); both still reach the host CRT.
`run_tiny_frame` does it with a stand-in that imports `d3d12_xs.dll!D3D12XboxCreateDevice` and, with
`LEONIDA_D3D12=vulkan`, clears a render target and reads it back on the Vulkan device: the loader, the game thread,
the Xbox-ABI proxies and the device in one process (`test/fixtures/tiny_frame.cpp`; `test/check_tiny_frame.py` checks
the verdict, the `shim` binding and the call order in the `--trace-d3d12` output, and skips without a Vulkan GPU).
The game does not boot yet. The missing executable blocks the import-driven work: the real device
entry point, command-list/queue/fence extensions, `PresentX`, and most of the `xgameruntime` surface (an offline
shim covers `XGameRuntimeInitialize`, device/console queries, `XTaskQueue` and `XAsync`; see `xgameruntime_test`). The
D3D12-to-Vulkan design and the measured console ABI are in
The game does not boot yet, and none of this has been measured against the real executable. The missing executable
blocks the import-driven work:

- The device entry point: `D3D12XboxCreateDevice` exists and wraps the PC or the Vulkan-backed device in the
  measured-slot proxy, but its parameter and flag layout is unmeasured, and the Xbox device vtable is only partly
  measured: the FSR DLL's calls confirm device slots 14, 15, 16, 18, 19, 27, 38, 62 and 76, slot 67 is a
  console-only pipeline call, and the rest is open (`docs/d3d12_vulkan.md`, question 2).
- The command-queue and fence extensions: the proxies forward the public slots, and the console's own methods past
  them (`PresentX`, `SuspendX`, `ResumeX`) have unknown slots, so they are refused with a log.
  `PresentX` and the frame events themselves are implemented (`shims/xbox_present.*`, `xbox_present_test`) but
  nothing reaches them yet: their queue and device slots are unmeasured.
- Most of the `xgameruntime` surface: an offline shim covers `XGameRuntimeInitialize`, device/console queries,
  `XTaskQueue` and `XAsync`, and `QueryApiImpl` serves `XThreadingImpl`, `XErrorImpl`, `XNetworkingImpl`,
  `XPersistentLocalStorageImpl` and `XSystemImpl` (see `xgameruntime_test`); any other class is refused with a log.

The D3D12-to-Vulkan design and the measured console ABI are in
[docs/d3d12_vulkan.md](../docs/d3d12_vulkan.md); the shader translator and its GPU checks are described
in [docs/design.md](../docs/design.md), and the differential-validation design is in the shader-validation
work.

No game data, keys, or extracted assets belong in this directory or in the repository.

### First-boot D3D12 trace

On a Vulkan-enabled build, select the Vulkan backend to trace device, command-list and queue calls handled
by Leonida. Put `--trace-d3d12` before the separator for game arguments.

```powershell
$env:LEONIDA_D3D12 = 'vulkan'
.\runtime\build\gta6rt.exe --run $gameExe --trace-d3d12 2>&1 |
    Select-String '^\[d3d12-trace\]'
```

Each trace line contains a sequence number, a fixed API name and an outcome. `called` marks a void API call
whose validation may be deferred. `ok`, `failed` and `unsupported` report known results. `ignored` marks a no-op.
Share only the lines prefixed with `[d3d12-trace]`. Other runtime diagnostics are separate.

### Startup trace: `--trace FILE`

`--trace FILE` (with `--dry-run`, `--init` or `--run`, before the `--` separator) writes one JSON object per line
to FILE, flushed per event so the file survives a crash ([shims/startup_trace.h](shims/startup_trace.h)). FILE must
not exist yet: the trace is created before the loader reads anything, so gta6rt refuses an existing path (exit code 2)
rather than risk truncating an input such as the image. The file is UTF-8; `OutputDebugStringA` text is converted
from the ANSI code page.

| `kind` | When | Fields |
|---|---|---|
| `trace_open` | first line | `pid` |
| `map` | a module is mapped, bound and protected | `module`, `base`, `size`, `relocated` |
| `tls_callback` | before each TLS callback the Loader runs (process attach and detach; thread events run under the Windows loader lock and are not traced) | `module`, `at` (module+RVA), `reason` |
| `dllmain` | after `DllMain(DLL_PROCESS_ATTACH)` returns | `module`, `result` |
| `unimplemented_call` | first call of each unimplemented import | `import` (`dll!name`), `caller` (module+RVA) |
| `xmem_attributes` | the first `XMemAlloc`, `XMemFree`, `XMemAllocDefault`, `XMemFreeDefault` and `XMemVirtualAlloc` for each distinct raw attribute word (not every allocation) | `api`, `attributes`, `size` (not for the frees), `caller` (module+RVA) |
| `xmem_physical_pages` | the first `XMemAllocatePhysicalPages` for each distinct attribute word | `api`, `attributes`, `pages`, `caller` |
| `file_open_failed` | a guest `CreateFileW`/`CreateFile2` that fails (at most 256 events) | `api`, `path` (as the title passed it), `error` (Win32), `caller` (module+RVA) |
| `int3_skipped` | with `LEONIDA_SKIP_INT3` set: first int3 at each address, which is then stepped over | `at` (module+RVA) |
| `debug_string` | `OutputDebugStringA/W` with no debugger attached | `text` |
| `exception` | any exception, before handlers run | `code`, `at`, `access`/`address` for access violations, `cpp` for C++ throws, `stack` (up to 16 frames) |
| `exit` / `error` | `--run` returns, or the loader throws | `code` / `what` |

Every line also has `seq` and `t` (seconds since the trace opened). Addresses are `module+0xRVA` for mapped images,
`host:name.dll+0xRVA` for host modules. The exception handler only records: it returns `EXCEPTION_CONTINUE_SEARCH`,
so the process behaves the same with or without the trace (except under the diagnostic `LEONIDA_SKIP_INT3`, which
steps over int3). It works with any `LEONIDA_UNIMPLEMENTED` policy.

The file can quote what the game prints, so it stays on your machine. In issues and pull requests, share counts or
lines from synthetic fixtures only.

```powershell
.\runtime\build\gta6rt.exe --run $gameExe --trace startup.jsonl
Get-Content startup.jsonl | ConvertFrom-Json | Group-Object kind | Select-Object Count, Name
```

[tools/boot_trace.py](../tools/boot_trace.py) prints the first-boot summary of this file and of the
`--trace-d3d12` log (modules, unimplemented calls with their callers, DllMain/TLS results, exceptions with their
stack, the exit, and the D3D12 calls by outcome and first call per method):

```powershell
python -I tools\boot_trace.py startup.jsonl --d3d12 d3d12.log
```

[docs/boot.md](../docs/boot.md) is the runbook around it: the commands, the triage order and the open questions'
trace signals.

### Typed clear regression

`clear_color_test` checks numeric UINT conversion and unchanged float clears without a GPU.
`d3d12_clear_test` compares full/rectangle readbacks with Windows WARP across UNORM, SNORM, FLOAT, UINT and SINT
render-target formats, plus depth and stencil clears. Fixtures are synthetic; no shaders or game data are needed.
The GPU test runs explicitly, while the CPU test is registered in CTest.

```powershell
ctest --test-dir runtime/build -C RelWithDebInfo -R '^clear_color$' --output-on-failure
$env:LEONIDA_VK_VALIDATION = '1'
$env:VK_VALIDATION_VALIDATE_SYNC = '1'
.\runtime\build\RelWithDebInfo\d3d12_clear_test.exe
```

### RGBA8 typeless RTV clear regression (phase 1)

`typeless_rtv_test` checks typed-view recording and replay without a GPU. It is registered in CTest.
`d3d12_typeless_rtv_test` compares synthetic full and rectangular clears with Windows WARP through explicit
UNORM/SRGB/UINT RTVs on 2D RGBA8 typeless resources, including mip slices, array-layer ranges and 4x MSAA.
SINT/SNORM clears are compared where both Vulkan and WARP report render-target support. It checks descriptor
snapshots, alternating view formats, SRGB with UAV resource usage, and invalid view rejection.

```powershell
ctest --test-dir runtime/build -C RelWithDebInfo -R '^typeless_rtv$' --output-on-failure
$env:LEONIDA_VK_VALIDATION = '1'
$env:VK_VALIDATION_VALIDATE_SYNC = '1'
.\runtime\build\RelWithDebInfo\d3d12_typeless_rtv_test.exe
```

### 96-bit R32G32B32 textures (D3D-62)

`d3d12_rgb96_test` compares footprints, uploads, readbacks, boxes, texture-to-texture copies and CPU copies of
`R32G32B32_FLOAT/UINT/SINT/TYPELESS` textures (2D arrays with mips, and volumes) with Windows WARP, bit for bit,
checks that WARP and the hardware adapter report no resource tiling for all four formats, and prints how long an
upload and a readback take. `d3d12_descriptor_test` samples them (alpha reads as one). It
is registered in CTest (`d3d12_rgb96`).

```powershell
$env:LEONIDA_VK_VALIDATION = '1'
$env:VK_VALIDATION_VALIDATE_SYNC = '1'
.\runtime\build\RelWithDebInfo\d3d12_rgb96_test.exe
```

RGBA8 typeless RTV clears support explicit UNORM/SRGB/UINT/SINT/SNORM views of 2D mip slices and contiguous
array-layer ranges, and MSAA. Other typeless color families also accept same-family typed RTVs, subject to Vulkan
color-attachment support. Typed-resource RTV casts are limited to R8G8B8A8_UNORM resources whose castable-format
list allows the view. A 1D texture cannot be created as a render target or depth buffer (`CreateCommittedResource`
refuses `ALLOW_RENDER_TARGET`/`ALLOW_DEPTH_STENCIL` on it), and 3D typeless view reinterpretation remains refused.

### Resource descriptions

`d3d12_resource_desc_test` creates the same committed and reserved resources on the Vulkan-backed device and on
Windows D3D12 (the default adapter and WARP, whichever exist) and compares every field of `ID3D12Resource::GetDesc`:
the `Alignment` and `MipLevels` a description left at 0 must read back as the host fills them in. A description the
host refuses is skipped, and one only the layer refuses is listed and not compared. It needs a
Vulkan GPU and runs explicitly.

```powershell
$env:LEONIDA_VK_VALIDATION = '1'
.\runtime\build\RelWithDebInfo\d3d12_resource_desc_test.exe
```

`d3d12_placed_dimensions_test` places 1D and 3D textures (arrays, mips, BC1, UAV and render targets, plus a 2D
control) in heaps big enough for both sides' `GetResourceAllocationInfo`, at offset 0 and at 64 KiB, on the
Vulkan-backed device and on Windows D3D12 (the default adapter and WARP), and compares the answers and `GetDesc`. A
dimension outside the enum must be refused alike. It also copies a pattern through a buffer into every subresource of a placed
3D texture and a placed 1D array and back, and checks the bytes. It needs a Vulkan GPU.

```powershell
$env:LEONIDA_VK_VALIDATION = '1'
.\runtime\build\RelWithDebInfo\d3d12_placed_dimensions_test.exe
### 1D render targets and depth buffers

`d3d12_texture1d_target_test` creates `TEXTURE1D` textures with `ALLOW_RENDER_TARGET` (10 color formats) or
`ALLOW_DEPTH_STENCIL` (4 depth formats), one texture and a 6-layer array each, on the Vulkan-backed device, the
hardware adapter and WARP. Where every reference creates one and the Vulkan device can make that 1D attachment image,
the layer must create it and `GetResourceAllocationInfo` must answer a size. It then clears through `TEXTURE1D` and
`TEXTURE1DARRAY` RTVs and DSVs (whole and with rectangles, on mips and slice ranges) and through a 1D UAV with a
rectangle, and compares every subresource's bytes with WARP. It is registered in CTest (`d3d12_texture1d_target`).

```powershell
$env:LEONIDA_VK_VALIDATION = '1'
.\runtime\build\RelWithDebInfo\d3d12_texture1d_target_test.exe
```

### Depth and stencil resolves

`d3d12_depth_resolve_test` resolves one plane of a 4x D32 or D24S8 target into a 1x one with
`ResolveSubresourceRegion` (MIN and MAX of depth or stencil, AVERAGE of depth by its own format; whole and by
rectangle) on the Vulkan-backed device and on Windows D3D12 (the default adapter and WARP), and compares the
destinations read back. Calls the layer refuses must fail `Close` and leave the destination as cleared. Clears give
every sample of a pixel one value, so the modes over differing samples are not checked. It needs a Vulkan GPU.

```powershell
$env:LEONIDA_VK_VALIDATION = '1'
.\runtime\build\RelWithDebInfo\d3d12_depth_resolve_test.exe
```

### Predicate capture lifetime

`vk_predicate_capture_test` checks full-width predicate snapshots, offset rebinding and reset with 32 gated submissions, then reuses completed command-buffer slots. Run explicitly on a Vulkan GPU with conditional rendering; it uses only synthetic data.

### A synthetic guest draws its first offscreen frame

On a Windows Vulkan build, configure CMake with `-DLEONIDA_TEST_GUEST_DRAW=ON` and build the
`tiny_guest_draw` and `gta6rt` targets. With llvm-mc, spirv-as and spirv-val available at configure
time, run `ctest --test-dir runtime/build -C Release -R "^d3d12_translated_draw_guest$" -V`.
The loader maps the test executable. It imports the Xbox device factory, submits three draws using two
handwritten shaders, reads back a 64×64 offscreen texture and checks every pixel.

The guest also builds independently of the shader tools. To produce the same synthetic fixture on
another machine, run `python -I runtime/test/shaders/build.py m15_shaders vs_m15 ps_m15` from the
checkout with llvm-mc, spirv-as and spirv-val available. Copy the whole resulting folder to the test
machine. For Wine, make the runtime executable, its runtime DLLs, the guest executable and the fixture
accessible in the same Wine prefix. For example, from their containing directory with the guest and
fixture beside the loader:

```sh
fixture=$(winepath -w "$PWD/m15_shaders")
shaders=$(winepath -w "$PWD/m15_shaders/spirv")
guest=$(winepath -w "$PWD/tiny_guest_draw.exe")
LEONIDA_D3D12=vulkan LEONIDA_SHADERS="$shaders" wine ./gta6rt.exe --run "$guest" -- "$fixture"
```

A passing run exits zero and prints `M1.5 guest: PASS, 4064 of 4096 pixels exact, 32 edge pixels the clear colour`.
The 32 diagonal pixels must all match the clear colour exactly. The fixture does not create a window.

### A 600-frame headless guest scene

Configure a Windows Vulkan build with `-DLEONIDA_GUEST_SCENE=ON` and static SDL3
(`SDL3_DIR` or `-DLEONIDA_SDL3_FETCH=ON`). Build `gta6rt` and `tiny_guest_draw`.
With the shader tools and Python available at configure time, CTest runs the existing
one-frame test and the scene with `ctest -R "^(d3d12_translated_draw_guest|d3d12_guest_scene)$" -V`.
For an externally generated M1.5 shader fixture, use the runner directly:

```sh
python -I scripts/run_guest_scene.py --loader path/to/gta6rt.exe --guest path/to/tiny_guest_draw.exe \
    --shaders path/to/m15_shaders --out new-scene-run --negative-controls
```

For Wine, add `--wine path/to/wine`. The runner launches the mapped guest for an
uninterrupted 600-frame run, a 300-frame save, and a fresh process that restores the
checkpoint. This opt-in build links a fixture input source into the loader. Scripted keyboard changes feed the SDL virtual pad. GameInput readings
drive the position of one quad. The guest streams ten 16-byte synthetic palette pages from
`GAME:\scene.bin` and changes its shader constants. A software reference independently
predicts the GPU pixel digest for each frame and checks six captured images.

Each run writes frame records, a state file, selected PPM readbacks, and a checkpoint
where applicable. At the first A-button edge the guest generates a quarter-second PCM
WAV. Playback is off by default. `--play-audio` requests playback through host WinMM
and records its result. The runner uses a fresh output directory, or a temporary one
when `--out` is omitted. The negative controls independently corrupt a scene byte and
a saved-state byte and require the expected guest errors. This fixture renders offscreen.
