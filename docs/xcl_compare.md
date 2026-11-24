# Comparing the runtime with XCL's test title

`tools/xcl_compare.py` runs one synthetic console executable under `gta6rt --run` and under
[XCL](https://github.com/ivall/xcl), and lists where the two runtimes disagree. XCL already runs retail Xbox Series
titles, so a check that passes there and fails here points at a loader, shim or D3D12 gap. It is a lead to
investigate, not proof of a bug: some of the title's checks test XCL's own design choices.

## Licence boundary

XCL is GPL-2.0-only, which is not compatible with this repository's GPL-3.0-or-later (see
[related_projects.md](related_projects.md)). The tool contains no XCL code, headers or tables. It starts programs
built from a separate XCL checkout that you point it at, reads what they print, and writes a report outside the
repository. Do not copy the test title, its sources or its output into this repository.

## What the test title is

XCL's test title (the `testtitle` and `testgamedll` crates in XCL's repository) is a fake console executable. Its
import table names `kernelx.dll`, `xgameruntime.dll`, `xmem.dll` and a bundled game DLL, so plain Windows cannot
start it. It runs about 50 checks (title memory, TLS, exceptions, modules, console paths, XGameRuntime, task
queues, users, storage, D3D12.X, DirectStorage, XAudio2) and prints `[PASS] <check>` or `[FAIL] <check>: <reason>`
for each. No game files are involved.

## Run it

1. Build XCL in its own checkout: `cargo build --release --workspace` (Rust and the Windows SDK; XCL looks up
   `dxcompiler.dll` in the Windows SDK).
2. Build the runtime: `scripts/build_runtime.ps1`, or CMake as in [testing.md](testing.md).
3. Run:

```powershell
python -I tools/xcl_compare.py --xcl C:\src\xcl
python -I tools/xcl_compare.py --xcl C:\src\xcl --gui      # the checks live in a window
```

The work folder defaults to `%TEMP%\leonida_xcl_compare`. It must not be on `D:`: XCL maps the console drive `D:`
to the game folder, so a real `D:\` path is not reachable inside XCL and the title fails to load. The report is
`report.json` in the work folder (`--out` to change it). Everything each title printed is in `xcl_output.txt` and
`leonida_output.txt` beside it. A check counts as a difference when its status or its reason differs between the
two runs, so a check that fails in both for different reasons is still listed.

A check that prints nothing for `--timeout` seconds (15 by default) is a hang. The tool records the stacks of the
title's threads with `cdb.exe` from the Windows SDK debugging tools (when installed), stops the title, and starts it
again with that check skipped, so a hang does not hide the checks after it. `--skip <fragment>` skips checks by
name. Both need two small local changes to the title's `check` function, which upstream XCL does not have: print
`[RUN] <check>` before each check, and return early with `[SKIP] <check>` when `XCL_TESTTITLE_SKIP` holds part of
its name. Without them the check in progress is taken from XCL's order, and a hang ends the run (the tool notices
the repeat and stops).

## Result, 2026-11-22

Windows 11, RTX 5070 Ti (driver 616.92), MSVC 19.44, Windows SDK 10.0.26100, runtime at `f847748c`, XCL at
`055b0ad` with the two local changes above in its test title.

| Runtime | Pass | Fail | Hang |
|---|---:|---:|---:|
| XCL | 50 | 0 | 0 |
| gta6rt | 5 | 43 | 2 |

XCL passes all 50 on this NVIDIA GPU, so its open gap on 32-wide-wave GPUs (wave64 compute, geometry and mesh
programs refused) does not affect this title. The D3D12 call traces were recorded (`--trace-d3d12`), but the runtime
never reaches a D3D12 call, so there is nothing to compare there yet.

What the differences come down to:

| Difference | Reading |
|---|---|
| Both hangs: `XTaskQueueTerminate(queue, wait = true)` on a manual queue, from the thread that dispatches it (`TaskQueueImpl::Terminate` waits on its condition variable) | Not a runtime bug. The reference implementation in `third_party/xtaskqueue` behaves this way, and Microsoft's `XTaskQueueTerminate` documentation warns that waiting from the dispatching thread can deadlock. XCL returns there; the title relies on that. |
| `GetModuleHandle(NULL)` and `current_exe` name `gta6rt.exe`, not the title | Covered by open #2075 (the mapped executable's module identity). |
| `G:\` and `T:\` console paths | RT-218 (claimed). |
| Persistent local storage | RT-219 (claimed). |
| XUser, XGameSave folder | #279 and #3363 (open). |
| `XAudio2Create` | RT-197, #3831 (open). |
| `d3d12_xs.dll!D3D12CreateDevice` not found (16 D3D12.X checks) | The runtime serves the console's `D3D12XboxCreateDevice`; the title asks for `D3D12CreateDevice`. Whether the console's `d3d12_xs.dll` exports that name is not measured here. |
| `kernelx.dll` imports `XMemAlloc`, `XMemVirtualAlloc`, `TitleMemoryStatus`, `GetConsoleType`, `SetThreadName`/`GetThreadName`; `xmem.dll` physical pages; `xcrdapi.dll!DStorageGetFactory`; `MFResetDXGIDeviceManagerX` | Unimplemented here. None is in the measured imports of the shipped modules ([runtime_imports.md](runtime_imports.md)), and the executable is not available to measure, so they are leads, not known needs. `xmem.dll!XMemAlloc` is served; the same name under `kernelx.dll` is not. |
| The title's window is not a hidden 1920x1080 tool window | An XCL design choice (its console screen), not a console fact. |
