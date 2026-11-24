# D3D12 command objects: the measured queue, list, fence and `PresentX` surface

Status: gap analysis, sourced. This page records what can be measured about the Xbox Series
`ID3D12CommandQueue`, `ID3D12GraphicsCommandList` (and 4/6/8), `ID3D12Fence`,
`ID3D12CommandAllocator` and `PresentX` from the shipped package and from the Windows SDK, and
separates that from prior-art claims. It does not propose a vtable and does not copy code.

Confidence legend, consistent with [d3d12_vulkan.md](d3d12_vulkan.md) and
[d3d12_xs_surface.md](d3d12_xs_surface.md):

- **Measured** — produced in this tree by a read-only inspection (PE import/export scan, GUID
  byte scan, static disassembly) or by a test in this repository.
- **SDK** — checked against the installed Windows SDK 10.0.26100.0 headers.
- **Prior art** — a claim by another project, not independently reproduced here.
- **Derived** — follows from a documented rule plus stated anchors; the derivation is shown.
- **Open** — no evidence.

Scope limit: no game data, keys or decryption was used; no vendor binary was executed. The
package DLLs are read-only inputs. The game executable that would answer most of the remaining
questions is not in the package (section 1).

## 1. What the package contains, and what it does not

### 1.1 Inventory

The package root contains **16 DLLs** and no D3D12 runtime DLL:

| Group | Modules |
|---|---|
| Middleware | `amd_fidelityfx_loader_xbox.dll`, `amd_fidelityfx_upscaler_xbox.dll`, `Bink2Scarlett.dll`, `libHttpClient.GDK.dll`, `XboxBvhBuilder.dll`, `XCurl.dll` |
| CRT | `concrt140.dll`, `msvcp140.dll`, `msvcp140_1.dll`, `msvcp140_2.dll`, `msvcp140_atomic_wait.dll`, `msvcp140_codecvt_ids.dll`, `vccorlib140.dll`, `vcomp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll` |

The following files are **absent** from the package root and from every subdirectory:

- `d3d12_xs.dll` — the console D3D12 runtime that owns the queue/list/fence/allocator vtables
  and the `PresentX` method;
- `xgameruntime.dll` — the GDK runtime;
- `game_scarlett_master.exe` — the executable that imports the runtime and calls the command
  objects.

The package layout XML lists the executable at
`layout_3202719e-b4b4-421c-9f8f-70d1c4a7d60b_xs.xml:14`, but the file is not in the folder. The
same XML lists the 16 DLLs; it does not list `d3d12_xs.dll` or `xgameruntime.dll`, which are
system/GDK dependencies rather than package payload. This confirms
[d3d12_vulkan.md §2](d3d12_vulkan.md) and Appendix A.4, which already record the absent
executable.

**Consequence:** the export table, import table and vtable of `d3d12_xs.dll` cannot be read from
this package. Any statement about the console queue/list/fence/allocator slot order in this page
is either SDK-derived, prior art, or a measurement of a *client* of those interfaces (the shipped
BVH builder), never a measurement of the runtime itself.

### 1.2 What the 16 DLLs import

A PE import scan (normal and delay-load) of all 16 DLLs gives the complete static D3D12 picture:

| DLL | `d3d12_xs.dll` import | `d3d12.dll` / `dxgi.dll` import | `xmem.dll` import |
|---|---|---|---|
| `XboxBvhBuilder.dll` | `D3D12SerializeVersionedRootSignature` (1) | none | `XMemVirtualQuery` (1) |
| `amd_fidelityfx_upscaler_xbox.dll` | `D3D12CreateRootSignatureDeserializer` (1) | none | `XMemAlloc`, `XMemFree`, `XMemGetWorkingSetStatistics` (3) |
| `Bink2Scarlett.dll` | none | none | `XMemAlloc`, `XMemFree` (2) |
| the other 13 DLLs | none | none | none |

No shipped module imports `d3d12.dll`, `dxgi.dll` or `xgameruntime.dll`. No shipped module
imports a queue, list, fence, allocator or `PresentX` free function. The only two `d3d12_xs`
imports are the root-signature calls already recorded in
[d3d12_vulkan.md §7](d3d12_vulkan.md) and Appendix A.2/A.3. This is consistent with the
prior-art statement that the bundled middleware surface is tiny and the command-object surface
belongs to the executable.

Reproduce: `python tools/pe_inspect.py <dll> --game-dir <package> --out <dir>` for each DLL, or
the import scan described in section 8.2.

### 1.3 What this rules out

The package cannot tell us:

- the `d3d12_xs.dll` export names or ordinals for `D3D12CreateDevice`/`D3D12XboxCreateDevice`,
  `D3D12SerializeVersionedRootSignature`, the `DXGIX*` present functions or
  `InternalDriverCallX`;
- the Xbox queue, command-list, fence or allocator vtable layout;
- the `PresentX` slot, signature or plane struct;
- any return-convention quirk on those interfaces.

It can tell us which interfaces the shipped BVH builder was compiled against, and it contains
one client that actually calls a command list. That is the useful measurement in section 3.

## 2. The public slot baseline

This section is the **SDK** baseline a replacement runtime can implement today. It is not a
statement that the Xbox build uses these slots; section 3 gives the few Xbox call sites that
have been measured.

Slots are absolute vtable indices including `IUnknown` (0–2) and `ID3D12Object` (3–6). A byte
offset in an 8-byte-slot vtable is `slot * 8`. The counts below were produced by parsing
`d3d12.h` in the installed SDK; each interface's declaration line is given.

### 2.1 Device creation slots

| Slot | Byte offset | Method | SDK line |
|---:|---:|---|---|
| 8 | `0x040` | `ID3D12Device::CreateCommandQueue` | `d3d12.h:9006` |
| 9 | `0x048` | `ID3D12Device::CreateCommandAllocator` | `d3d12.h:9011` |
| 12 | `0x060` | `ID3D12Device::CreateCommandList` | `d3d12.h:9026` |
| 36 | `0x120` | `ID3D12Device::CreateFence` | `d3d12.h:9178` |

The Xbox device is **not** uniformly the public order. Prior art measured slots
1/14/15/16/18/19/27/38 in desktop order, plus a compute-PSO method at 62 and an extra
texture-creation method at 76 where the PC interface has `CreateStateObject`/something else
([d3d12_vulkan.md §7](d3d12_vulkan.md),
[`d3d12_xs_surface.md §2.4`](d3d12_xs_surface.md)). **Slots 8, 9, 12 and 36 were not measured
by that work.** They are the four calls a game must make before it can have a queue, allocator,
list or fence, so they are the first thing a trace must confirm.

**They cannot be measured from the package either (D3D-350).** The package ships no D3D12 runtime:
the 16 DLLs in [`game/xbox-1.0.2.1.json`](../game/xbox-1.0.2.1.json) do not include
`d3d12_xs.dll`, which is part of the console OS, and the device clients that are shipped never call
these four slots. Measured on 1.0.2.1 (counts only):

| Image | `ID3D12*` IIDs it contains | Imports from `d3d12_xs.dll` | Calls at `0x40`/`0x48`/`0x60`/`0x120` on the device |
|---|---|---|---|
| `XboxBvhBuilder.dll` | `PipelineState`, `RootSignature` | `D3D12SerializeVersionedRootSignature` | none: no indirect call at any of the four offsets |
| `amd_fidelityfx_upscaler_xbox.dll` | `CommandSignature`, `DescriptorHeap`, `Device`, `Heap`, `PipelineState`, `Resource`, `RootSignature`, `RootSignatureDeserializer` | `D3D12CreateRootSignatureDeserializer` | none: the 3 calls at `0x40` are `ID3D12Resource::Map` on a resource it has just created, the 29 at `0x48` are `Close`/`Unmap`-shaped calls with no `(type, &IID, &out)` arguments, the 4 at `0x60` are a function-pointer field in a plain struct (no vtable load), and there are none at `0x120` |
| `amd_fidelityfx_loader_xbox.dll`, `Bink2Scarlett.dll` | none | none | — |

Neither image contains the IID of `ID3D12CommandQueue`, `ID3D12CommandAllocator`,
`ID3D12CommandList`, `ID3D12GraphicsCommandList`, `ID3D12Fence` or `ID3D12Fence1`, and neither
can take one from `d3d12_xs.dll`: each imports one function there and no GUID data (ordinals
23–115). A creator called through `IID_PPV_ARGS` needs that IID, so these clients do not create
queues, allocators, lists or fences; they record into lists the game gives them. The interfaces
that *are* present are the positive control: they are what each client is known to create
(compute PSOs through slot 62, root signatures, descriptor heaps, and resources through slots 27
and 76). The IID column is reproduced by `python -I tools/d3d12_iid_scan.py <package>/*.dll`
(`tests/test_d3d12_iid_scan.py`); the call-site column comes from `dumpbin /disasm` listings,
following each `0x40`/`0x48`/`0x60`/`0x120` call back to its receiver.

So the four slots' callers are in the game executable, which the package does not contain (the
same blocker as RT-5). Confidence for slots 8, 9, 12 and 36 stays **Open**: the table in §2.1 is
still the SDK baseline. Once the executable is available, `tools/d3d12_iid_scan.py` run on it
shows whether it carries these IIDs, and its call sites through them give the slots.

### 2.2 Queue, allocator and fence

`ID3D12CommandQueue` is declared at `d3d12.h:8701`. It has 19 slots (0–18); its own methods start
at 8 because `IUnknown` + `ID3D12Object` + `ID3D12DeviceChild` + `ID3D12Pageable` contribute
eight slots and `ID3D12Pageable` adds none.

| Slot | Byte offset | Method | SDK line |
|---:|---:|---|---|
| 8 | `0x040` | `UpdateTileMappings` | `d3d12.h:8704` |
| 9 | `0x048` | `CopyTileMappings` | `d3d12.h:8716` |
| 10 | `0x050` | `ExecuteCommandLists` | `d3d12.h:8724` |
| 11 | `0x058` | `SetMarker` | `d3d12.h:8728` |
| 12 | `0x060` | `BeginEvent` | `d3d12.h:8733` |
| 13 | `0x068` | `EndEvent` | `d3d12.h:8738` |
| 14 | `0x070` | `Signal` | `d3d12.h:8740` |
| 15 | `0x078` | `Wait` | `d3d12.h:8744` |
| 16 | `0x080` | `GetTimestampFrequency` | `d3d12.h:8748` |
| 17 | `0x088` | `GetClockCalibration` | `d3d12.h:8751` |
| 18 | `0x090` | `GetDesc` (returns the 16-byte desc by value on MSVC) | `d3d12.h:8756` |

`ID3D12CommandAllocator` is declared at `d3d12.h:5193`; it has 9 slots (0–8), with
`Reset` at slot 8 (`d3d12.h:5196`).

`ID3D12Fence` is declared at `d3d12.h:5323`; it has 11 slots (0–10):

| Slot | Byte offset | Method | SDK line |
|---:|---:|---|---|
| 8 | `0x040` | `GetCompletedValue` | `d3d12.h:5326` |
| 9 | `0x048` | `SetEventOnCompletion` | `d3d12.h:5328` |
| 10 | `0x050` | `Signal` | `d3d12.h:5332` |

`ID3D12Fence1` (`d3d12.h:5477`) adds `GetCreationFlags` at slot 11 (`d3d12.h:5480`), for 12
slots total. The console runtime's exported IID table stops at `IID_ID3D12Fence1` and
`IID_ID3D12GraphicsCommandList2`; that is a property of the **exported IID set**, not a bound on
the interfaces the game uses ([`d3d12_xs_surface.md §2.4`](d3d12_xs_surface.md)).

### 2.3 `ID3D12GraphicsCommandList` and 1–8

`ID3D12GraphicsCommandList` is declared at `d3d12.h:6337`. Its own methods occupy slots 9–59
(51 methods), after `IUnknown` (3), `ID3D12Object` (4), `ID3D12DeviceChild` (1) and
`ID3D12CommandList` (1). The chain then grows as follows:

| Interface | Own slots | Total slots | Last byte offset | SDK line |
|---|---|---:|---:|---|
| `ID3D12GraphicsCommandList` | 9–59 | 60 | `0x1d8` | `d3d12.h:6337` |
| `ID3D12GraphicsCommandList1` | 60–65 | 66 | `0x208` | `d3d12.h:7216` |
| `ID3D12GraphicsCommandList2` | 66 | 67 | `0x210` | `d3d12.h:7982` |
| `ID3D12GraphicsCommandList3` | 67 | 68 | `0x218` | `d3d12.h:20165` |
| `ID3D12GraphicsCommandList4` | 68–76 | 77 | `0x260` | `d3d12.h:21255` |
| `ID3D12GraphicsCommandList5` | 77–78 | 79 | `0x270` | `d3d12.h:30275` |
| `ID3D12GraphicsCommandList6` | 79 | 80 | `0x278` | `d3d12.h:31123` |
| `ID3D12GraphicsCommandList7` | 80 | 81 | `0x280` | `d3d12.h:31965` |
| `ID3D12GraphicsCommandList8` | 81 | 82 | `0x288` | `d3d12.h:32816` |

The methods that matter for this page:

| Slot | Byte offset | Method | SDK line |
|---:|---:|---|---|
| 9 | `0x048` | `Close` | `d3d12.h:6340` |
| 10 | `0x050` | `Reset` | `d3d12.h:6342` |
| 14 | `0x070` | `Dispatch` | `d3d12.h:6362` |
| 25 | `0x0c8` | `SetPipelineState` | `d3d12.h:6418` |
| 26 | `0x0d0` | `ResourceBarrier` | `d3d12.h:6421` |
| 33 | `0x108` | `SetComputeRoot32BitConstant` | `d3d12.h:6446` |
| 39 | `0x138` | `SetComputeRootShaderResourceView` | `d3d12.h:6476` |
| 45 | `0x168` | `SOSetTargets` | `d3d12.h:6500` |
| 59 | `0x1d8` | `ExecuteIndirect` | `d3d12.h:6580` |
| 68 | `0x220` | `GCL4::BeginRenderPass` | `d3d12.h:21258` |
| 69 | `0x228` | `GCL4::EndRenderPass` | `d3d12.h:21264` |
| 72 | `0x240` | `GCL4::BuildRaytracingAccelerationStructure` | `d3d12.h:21276` |
| 73 | `0x248` | `GCL4::EmitRaytracingAccelerationStructurePostbuildInfo` | `d3d12.h:21281` |
| 74 | `0x250` | `GCL4::CopyRaytracingAccelerationStructure` | `d3d12.h:21286` |
| 75 | `0x258` | `GCL4::SetPipelineState1` | `d3d12.h:21291` |
| 76 | `0x260` | `GCL4::DispatchRays` | `d3d12.h:21294` |
| 79 | `0x278` | `GCL6::DispatchMesh` | `d3d12.h:31126` |
| 80 | `0x280` | `GCL7::Barrier` (enhanced barriers) | `d3d12.h:31968` |
| 81 | `0x288` | `GCL8::OMSetFrontAndBackStencilRef` | `d3d12.h:32819` |

**Not a bound on the game.** The console runtime's 93 GUID exports include only
`IID_ID3D12GraphicsCommandList`, `…1`, `…2`; the shipped BVH builder names `…4`, `…6` and `…8`
in its exported signatures (section 3.1). The export table's IID set is therefore not the game's
interface set, a point already made in [`d3d12_xs_surface.md §2.4`](d3d12_xs_surface.md).

## 3. Direct measurements from the shipped binaries

### 3.1 `XboxBvhBuilder.dll`'s interface types

`XboxBvhBuilder.dll` is 884,224 bytes, PE32+ x86-64, with 12 named exports (ordinals 1–12).
The mangled export names are direct evidence of the C++ parameter types the DLL was compiled
against; no symbols were guessed. The relevant exports:

| Ordinal | RVA | Export (demangled shape) | Interface named |
|---:|---:|---|---|
| 1 | `0x21330` | `XboxBvhBuilder(ID3D12Device11*, bool)` | `ID3D12Device11` |
| 3 | `0x21610` | `BuildRaytracingAccelerationStructure(ID3D12GraphicsCommandList4*, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC*, UINT, const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC*)` | GCL4 |
| 4 | `0x216a0` | `BuildRaytracingAccelerationStructureBatched2X(ID3D12GraphicsCommandList6*, UINT, const D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2*)` | GCL6 |
| 5 | `0x21980` | `CopyRaytracingAccelerationStructure(ID3D12GraphicsCommandList4*, UINT64, UINT64, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE)` | GCL4 |
| 6 | `0x21b40` | `CpuValidateResultX(ID3D12GraphicsCommandList4*, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC*)` | GCL4 |
| 8 | `0x219d0` | `RehydrateRaytracingAccelerationStructureBatched2X(ID3D12GraphicsCommandList8*, UINT, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC*, const UINT64*)` | GCL8 |
| 9 | `0x219a0` | `RehydrateRaytracingAccelerationStructureBatchedX(ID3D12GraphicsCommandList8*, UINT, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC*)` | GCL8 |
| 10 | `0x219f0` | `RehydrateRaytracingAccelerationStructureX(ID3D12Device11*, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC*)` | `ID3D12Device11` |

The DLL imports only `d3d12_xs.dll!D3D12SerializeVersionedRootSignature` and
`xmem.dll!XMemVirtualQuery`; it does not import a queue, list, fence or allocator function.
This is the same import result recorded in
[d3d12_vulkan.md Appendix A.2](d3d12_vulkan.md), re-measured here.

**What this proves:** the helper's public signatures require GCL4, GCL6, GCL8 and Device11
objects. **What this does not prove:** that the helper calls the public vtable slots for those
interfaces. Section 3.3 measures the call sites.

### 3.2 The shipped modules do not contain the command-object IIDs

A byte scan for the 80 `IID_ID3D12*` values defined by the installed `d3d12.h`, plus the two
console GUIDs recorded in [`d3d12_xs_surface.md §4`](d3d12_xs_surface.md), found:

| Binary | IIDs found |
|---|---|
| `XboxBvhBuilder.dll` | `IID_ID3D12RootSignature` at `0xc9ee8`, `IID_ID3D12PipelineState` at `0xc9ed8` |
| `amd_fidelityfx_upscaler_xbox.dll` | `IID_ID3D12Device` `0x472f0`, `IID_ID3D12Resource` `0x47310`, `IID_ID3D12RootSignatureDeserializer` `0x47320`, `IID_ID3D12PipelineState` `0x47330`, `IID_ID3D12Heap` `0x47340`, `IID_ID3D12RootSignature` `0x47350`, `IID_ID3D12DescriptorHeap` `0x472e0`, `IID_ID3D12CommandSignature` `0x47300` |
| the other 14 DLLs | none of the scanned IIDs |

No shipped module contains `IID_ID3D12CommandQueue`, `IID_ID3D12GraphicsCommandList` (any
version), `IID_ID3D12Fence`/`Fence1`, `IID_ID3D12CommandAllocator` or
`IID_ID3D12ConsDmaCommandList`. The helper receives command lists through its exported
parameters; the corresponding IID constants are not in its own data. That is a fact about the
shipped modules, not a claim that the game never queries for them.

The `GraphicsCommandList` ASCII string occurs 30 times in `XboxBvhBuilder.dll` (type/debug
names); `PresentX` does not occur in any of the 16 DLLs.


### 3.2a No shipped module passes an unknown GUID at a call site

Section 3.2's byte scan can only recognise *known* IIDs; a console-only GUID (the four list IIDs of
[`d3d12_xs_surface.md §4`](d3d12_xs_surface.md)) would scan as anonymous data. The call-site scan closes
that hole: `tools/d3d12_guid_sites.py` enumerates every 16-byte constant a module's code can name —
RIP-relative `lea` targets in data sections, absolute pointers stored in data, two adjacent
`movabs` immediates that inline a GUID without a constant pool, and `mov reg, [rip+...]` loads
through import slots (a GUID imported as data) — and reports which are GUID-shaped. Over all 16
modules of the 1.0.2.1 package:

| Binary | GUID-shaped constants the code references |
|---|---|
| `XboxBvhBuilder.dll` | 2: `IID_ID3D12RootSignature`, `IID_ID3D12PipelineState` (the same pair as §3.2) |
| `amd_fidelityfx_upscaler_xbox.dll` | 8: the §3.2 set, nothing else |
| `amd_fidelityfx_loader_xbox.dll` | 0 |
| the other 13 modules | 0–44 GUIDs each (six have none); where present they are HTTP, WinRT and CRT interfaces — none of these modules imports `d3d12_xs.dll`, so none can call D3D12 at all |

No module loads a GUID through an import slot, and no `movabs` pair inlines one. As a control, the
upscaler's single `IID_ID3D12Device` reference (rva `0x482f0`) is the IID of a real slot-0
`QueryInterface` call (`(this, &IID, &out)`, decompiled at rva `0xde66`). So the only
interfaces any shipped module can name at a `QueryInterface` or a creation call are the ten it
already uses; a console compute/copy list IID cannot appear in the package, statically, in any
form. The same scan looks for code-referenced static `D3D12_COMMAND_QUEUE_DESC` windows
`{Type, Priority, Flags, NodeMask}`: 10 windows in 3 modules survive the shape filter, every one with
`Type` 1 — a PC enum value — and all of them offset tables read by stride arithmetic, not queue
descriptions (checked by decompiling the referencing function of the upscaler's six). None has a
value past `VIDEO_DECODE` (4), so the DMA type's numeric value has no static source in the package
either.

Reproduce: `python -I tools/d3d12_guid_sites.py <package>/*.dll --known
IID_ID3D12ConsDmaCommandList=30bc77b9-8379-4b25-95de-b8588c42275d` (prints GUID values and counts
only). The four list IIDs and the DMA type remain reachable only through the running title, as
[`d3d12_xs_surface.md §4`](d3d12_xs_surface.md) already requires for the IIDs.

### 3.3 The BVH builder's command-list call sites

The exported `BuildRaytracingAccelerationStructure` (ordinal 3, RVA `0x21610`) copies the input
description into a stack structure and **delegates to ordinal 4**
(`BuildRaytracingAccelerationStructureBatched2X`) at RVA `0x2166d`, passing the same command-list
pointer in RDX and a count of 1. The batched entry therefore carries both the non-batched and the
batched paths.

Disassembling the exported bodies and the functions they call directly gives these calls through
the object received in the command-list parameter:

The console list is not in PC order past slot 26 (section 3.5a), so the method column reads each call in the
console order, from the arguments at the call and from the inline write it falls back from (section 3.5a, last
paragraph). Every call below except the two barriers is the fallback branch of an inline write of the same values.

| RVA | Vtable offset | Console slot → PC method | Observed arguments | Confidence |
|---:|---:|---|---|---|
| `0x2171e` | `+0xd0` | 26 → 26 `ResourceBarrier` | `NumBarriers=1`, `&{Type=2 (UAV), Flags=0, pResource=null}` | **Measured**: the shape is a global UAV barrier; slot 26 is the SDK `ResourceBarrier`; the site is gated (section 3.3b), so the call never fires in the shipping DLL |
| `0x1d9cb` | `+0x70` | 14 → 14 `Dispatch` | `(1, 1, 1)` | **Measured**: exact `Dispatch` shape |
| `0x1d85e` | `+0xc8` | 25 → 25 `SetPipelineState` | `RCX=list`, `RDX=[rdi+8]`: loaded at `0x1d7fa` and not changed before the call; the inline branch (`0x1d822`–`0x1d850`) copies 28 bytes from that object's `+0x10` | **Measured**: RDX is the pipeline the inline branch reads |
| `0x1d7f4` | `+0x108` | 33 → 29 `SetComputeRootSignature` | one argument, `RDX=[rcx]`: the pointer stored next to the pipeline (`[rdi]`; the pipeline is `[rdi+8]`) | **Measured** shape; agrees with slot 33 in section 3.5a |
| `0x1d8b9` | `+0x138` | 39 → 35 `SetComputeRoot32BitConstants` | `(3, 1, &value, offset)`: `EDX=3`, `R8D=EDX-2=1`, `R9=rsp+0x30`, fifth argument `EDI` | **Measured**: four arguments with a count and a source pointer; a root SRV takes two |
| `0x1d956` | `+0x138` | 39 → 35 `SetComputeRoot32BitConstants` | `(3, count, source, offset)`: `R8D=EDI` (the count of the inline copy loop above it), `R9=RSI` | **Measured** |
| `0x1d8f9` | `+0x168` | 45 → 41 `SetComputeRootUnorderedAccessView` | `(0, VA)`: `EDX=0`, `R8=RBP`; the inline branch writes a header word and the same 64-bit `RBP` (12 bytes) | **Measured**: a root index and a 64-bit address |
| `0x1da02` | `+0x490` | 146 → GCL7 80 `Barrier` | `(1, &group)`, group `{Type=0, NumBarriers=1, pGlobalBarriers}`; the barrier is the 16 bytes at RVA `0xcdf60`: `{0x80, 0x80, 0x10, 0x10}` | **Measured**: the public `Barrier(UINT32, const D3D12_BARRIER_GROUP*)` shape with public enum values |

The `+0x490` call is after the `Dispatch(1,1,1)` at `0x1d9cb`, in the per-descriptor path. It is
**beyond the last public GCL8 slot** (slot 81 is `0x288`), but its arguments are the public enhanced-barrier
call: one `D3D12_BARRIER_GROUP` of type `D3D12_BARRIER_TYPE_GLOBAL` (0) holding one `D3D12_GLOBAL_BARRIER` with
`SyncBefore = SyncAfter = D3D12_BARRIER_SYNC_COMPUTE_SHADING` (`0x80`) and
`AccessBefore = AccessAfter = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS` (`0x10`), values as in SDK 10.0.26100
`d3d12.h`. The public PC equivalent is GCL7 slot 80.

@MoonYueya's static enumeration of the Xbox Series 1.0.2.1 builder found 29 calls through console slot 146
across seven functions. It counted 11 direct calls through the vtable entry and 18 calls through a register
loaded from that entry. All 29 passed one global barrier group containing one `D3D12_GLOBAL_BARRIER` with
the same compute-shading and unordered-access arguments described above. The counts cover the two
enumerated dispatch forms in the builder's `.text`. [Static corroboration and coverage details](https://github.com/ItsNotRexed/Leonida/issues/1626)

The same two inner functions (`0x1d540` and `0x1d7c0`) contain the `+0x70`, `+0xc8`, `+0x108`,
`+0x138`, `+0x168` and `+0x490` calls; the outer batched function contains the `+0xd0` barrier.
The exported `CopyRaytracingAccelerationStructure` (ordinal 5) and the rehydrate exports
(ordinals 8–9) do not call the passed command list in their exported bodies; they forward to
internal functions using objects cached in the builder (`[this+0x10]` and `[this+8]`). Their
signatures still require GCL4/GCL8 objects, but those bodies do not measure command-list slots.

### 3.3a Other builder call sites past the PC order

The rest of `.text` has further calls through the list pointer. The table shows examples, including one of the
slot-146 sites counted in section 3.3. Each name rests on the call shape. “Inline” means the call is the fallback
of an inline write of the same values.

| RVA | Vtable offset | Console slot → PC method | Observed arguments | Confidence |
|---:|---:|---|---|---|
| `0x1a1e9` | `+0x128` | 37 → 33 `SetComputeRoot32BitConstant` | `(2, value, 0)`: `EDX=R9D+2`, `R9D=0` (inline: header + one dword) | **Measured** shape |
| `0x14e81` | `+0x138` | 39 → 35 `SetComputeRoot32BitConstants` | `(0, 19, rsp+0xc0, 0)` (inline: header + 19 dwords) | **Measured** shape |
| `0x14e11` | `+0x158` | 43 → 39 `SetComputeRootShaderResourceView` | `(3, VA)`: `EDX=3`, `R8=R13` (inline: header + the 64-bit `R13`) | **Measured** shape |
| `0x1a3b0` | `+0x168` | 45 → 41 `SetComputeRootUnorderedAccessView` | `(0, VA)` (inline: header + a 64-bit value) | **Measured** shape |
| `0x20ed6` | `+0x4a0` | 148 → none | `(list, count * 40, 16)` returns a pointer; on null the DLL prints `D3D12 XDXR Error: Failed to allocate upload buffer` and throws `E_FAIL` | **Measured**: an upload allocator `(size, alignment)` |
| `0x20f99` | `+0x158` | 43 → 39 `SetComputeRootShaderResourceView` | `(0, p)`, `p` the pointer slot 148 returned at `0x20ed6`, unchanged | **Measured**: the CPU pointer is used as the GPU address |
| `0x149ba` | `+0x280` | 80 → none | `(x, 1, 1, 0x8020)`: `EDX=EDI`, `R8D=R9D=1`, fifth argument `0x8020` (inline: the five-dword packet `0x4003cc00, x, 1, 1, W` of section 3.5a; the calls at `0x1525d` and `0x153b4` pass `0x20`, the one at `0x1d047` `0x8020`) | Inferred: a `Dispatch` with one extra word |
| `0xd6de` | `+0x238` | 71 → none | `(&label, counter, 0 or 0x1000, 0)`: the counter at `[rbx+0x9c]` is incremented first; `0x1000` on the last item of a batch | Inferred: a GPU write of a 32-bit value to memory |
| `0xd679` | `+0x248` | 73 → none | `(&label, 3, reference, 0xffffffff, 0)` | Inferred: a GPU wait until the masked value compares (function 3) with the reference |
| `0xd7b4` | `+0x490` | 146 → GCL7 80 `Barrier` | the same global UAV barrier as `0x1da02` | **Measured** |

So the compute root-binding group keeps its PC order four slots later (33, 37, 39, 43 and 45 here; 35 and 41 in
section 3.5a), and the console list has methods well past the PC tail: 71, 73, 80, 146 and 148 here, 141 in the
FSR DLL.

Slot 148 matters beyond the builder: the address it returns is written by the CPU and passed unchanged to the GPU
as a root SRV address, so console upload memory has the same address for the CPU and the GPU. A replacement must
return memory whose CPU pointer is also a valid GPU address (for this layer: inside the arena of
[d3d12_vulkan.md §3.3](d3d12_vulkan.md)).

The function at `0xd580` uses the 71/73 pair instead of the slot-146 barrier when the byte at RVA `0xd773c` is set.
That byte is one of seven bools at `0xd7738`–`0xd773e` in zero-initialised `.bss`. The only accesses to them in
`.text` are about 50 RIP-relative reads (`cmp`, `movzx`); there is no write and no export that points there. So in
the shipping DLL the label path is unreachable unless something writes that byte through a computed pointer, and
the barrier path is the one that runs.

### 3.3b The batched build validates the geometry descriptors on the CPU first

A logging-proxy harness (section 7, item 1) never reached the first list-vtable call: both runs of the
batched export faulted at RVA `0x142d3`, before the object received in the command-list parameter was
touched once. The reason is a CPU-side validation of the build items that the section 7 descriptor
guidance did not cover. The batched entry (`0x216a0`) reaches it through `0xe6a0` (call at `0x216dc`),
the packet processor `0xcc00` (call at `0xe781`) and a per-record validator at `0x14270` (call at
`0xce18`; a second call site exists at `0x11c25`). The processor walks the batched descriptor's 80-byte
build items (item `i` at descriptor `+ i*0x50`) and routes each by the dword at its `+0x08` — the
pointer's `+0x00` at the validator: kind 1 reaches the validator, kind 0 goes through `0xc820`/`0xc780`
instead, other values take an error path (`0xd52f`). The validator reads, from the pointer it receives
(all **measured** in the static disassembly):

| Offset | Field | Check |
|---:|---|---|
| `+0x08` | descriptor count | loop bound |
| `+0x0c` | elements layout (`D3D12_ELEMENTS_LAYOUT`) | 0 = inline array of 0x38-byte records, 1 = array of pointers to records; other values print `D3D12 XDXR Error: Unexpected value for D3D12_ELEMENTS_LAYOUT` and throw `E_INVALIDARG` |
| `+0x10` | descriptor base | mode 0: record `i` at `base + i*0x38`; mode 1: record `i` at `[base + i*8]` |
| record `+0x00` | geometry type (`D3D12_RAYTRACING_GEOMETRY_TYPE`) | 0 = triangles, 1 = AABBs; others print `D3D12 XDXR Error: Unrecognized D3D12_RAYTRACING_GEOMETRY_TYPE` and throw `E_INVALIDARG` |
| record `+0x08` | AABB count (type 1) | accumulated |
| record `+0x10` | index format (type 0) | at most `0x39` and in the bitset `0x0220040000000000` — `DXGI_FORMAT_R32_UINT` (42), `DXGI_FORMAT_R16_TYPELESS` (53), `DXGI_FORMAT_R16_UINT` (57); others print `D3D12 XDXR Error: Unsupported index buffer format provided` and throw `E_NOTIMPL` |
| record `+0x18` | index count (type 0, format ≠ 0) | exact multiple of 3; otherwise prints `D3D12 XDXR Error: Triangle index count must be a multiple of 3 when an index buffer is provided` and throws `E_INVALIDARG` |
| record `+0x1c` | vertex count (type 0, format = 0) | checked instead of the index count, also an exact multiple of 3; otherwise prints `D3D12 XDXR Error: Triangle vertex count must be a multiple of 3 when no index buffer is provided` and throws `E_INVALIDARG` |

The fault that stopped the dynamic runs is the base dereference itself: at `0x142d3` (`mov ecx,[rdx]`,
`rdx = base + i*0x38` in the inline mode) the record's type word is read with a CPU load, so a descriptor
whose base is a GPU address that is not CPU-readable faults there. For a replacement runtime this is a
measured host-side contract: the geometry descriptors must be backed by CPU-readable memory, not only by
the GPU arena. The validator returns the summed primitive count (the triangle counts ÷ 3 plus the AABB
counts), which the packet processor stores (`0xce1d`). It reads nothing past record `+0x1f`; what the
`+0x20..+0x37` tail of a record is used for is open — the dynamic runs stopped at the first record read.

The same static pass settles what the first list calls would have been, and finds that three of the sites
section 3.3 and 3.3a list or miss are gated by the zero-initialised bytes of section 3.3a, so in the
shipping DLL they never run:

- The `+0xd0` `ResourceBarrier` at `0x2171e` (section 3.3) needs a non-zero item count (`0x216e1`) and the
  byte at `0xd773c` (`0x216e9`). Its shape stays as measured; in the shipping DLL the call never fires.
- Slot 90 (`+0x2d0`, call at `0xe836`) is new — it is not in the section 3.3a tail list. Its shape is
  `(list, p, 0, last)`: `RDX = [item+0x28] + [item+0x54]*4`, where `item` is one 160-byte element of the
  processor's output array, selected through the dword array beside it, `R9 = 1` on the last iteration,
  and the element's dword at `+0x9c` is cleared first. The loop is behind the same `0xd773c` byte
  (`0xe7d0`) and the count check, so it does not run in the shipping DLL either.
- The direct slot-146 barrier at `0xea99` is behind the adjacent byte `0xd773b` (`0xea41`, after the
  per-item callback reports true); the wrapper calls at `0xebbb`, `0xec79`, `0xed37` and `0xee17` are not
  gated by these bytes, and each ends in the measured slot-146 global UAV barrier (section 3.3a).

The path also reads one byte past the seven bools of section 3.3a: `0xd773f` at `0xce3f` and `0x11c35`.
A linear sweep of `.text` finds no writes to `0xd7738`–`0xd773f` (only `cmp`/`movzx` reads; the two `lea`
sites take the address of `0xd7740`, one past the last byte), so the read-only range is eight bytes.

With the gated sites removed, the shipping-DLL order for one item is (measured with a postbuild-info
type of 2, which routes to the `0x1d7c0` worker): the validation above → slot 146 four times through
the `0xd580` wrapper → the upload function `0x20ea0` (called at `0xee85`; a second site at `0xf0f9`):
slot 148 for `count * 40` bytes at alignment 16 (`0x20ed6`; the size is
`lea edx,[rsi+rsi*4]; shl edx,3`), root signature (33), pipeline (25, or its inline write), the
slot-148 pointer as root SRV (43, or inline), the count as a 32-bit root constant (37), and
`Dispatch((count+31)/32, 1, 1)` (`0x21054`, or its inline four-word packet) — then the per-entry
worker `0x1d7c0` (called at `0x217d0`) whose sequence section 3.3 lists, ending in the slot-146
barrier at `0x1da02`. Postbuild-info types 0 and 3 route to the mirrored worker `0x1d540` (called at
`0x217e2`). Slot 148 is therefore asked before the per-entry dispatch, and the first dispatch on the
path is the upload function's own — the two refinements to section 7's prediction. The `+0x490` barrier
arguments (one global UAV group `{0x80, 0x80, 0x10, 0x10}`) and slot 148's `count * 40`/16 are confirmed
by both passes (D3D-293's static disassembly and this section's independent iced-x86 re-check).

### 3.4 The object is wider than the public vtable

The inner function at `0x1d7c0` receives the command-list pointer in RDX, copies it to RBX
(`0x1d7da`–`0x1d7dd`), and then does more than call public methods:

- writes 32 bytes at `[rbx+0x544]` (`0x1d803`, `0x1d810`: the pipeline object's `+0x10`..`+0x2f`);
- writes a byte at `[rbx+0x140]` (`0x1d841`, `0x1d884`);
- reads `[rbx+0x350]` (`0x1d877`) and `[rbx+0x344]` (`0x1d8c9`);
- maintains a pointer at `[rbx+0x18]` against limits at `[rbx+0x30]` (`0x1d818`, `0x1d871`);
- calls a method at vtable offset `+0x490` (`0x1da02`), past the end of the public GCL8 chain.

A client that knows only the public COM interface cannot legally read or write those fields.
The helper is therefore compiled against a **wider contract** than the public vtable: either the
Xbox command-list object is an internal class whose layout first-party code may use, or the game
passes a builder-specific object that also implements the interface. Section 3.5a explains most of the fields
(the inline command buffer), and section 3.3 identifies `+0x490` as `Barrier` at console slot 146.

This is the most important compatibility finding on this page. A replacement command list that
implements only the public GCL7/GCL8 vtable is **not** safe to hand to `XboxBvhBuilder`'s build path. (Section 3.5a
shows it is not safe for console-compiled callers' public calls either: the console list order differs from the PC
order past slot 26.) The helper will touch fields the public
object does not define and call `+0x490` (console slot 146, `Barrier`), which a public-only vtable does not have.

### 3.5a The FSR DLL as a second device client

`amd_fidelityfx_upscaler_xbox.dll` is a second shipped module compiled against the console device, the second
measured client of the console **command list** after the BVH builder (section 3.3), and the first one run. It is run
by `fsr_dispatch_test` with its measured Xbox backend
descriptor ([d3d12_vulkan.md Appendix A.3](d3d12_vulkan.md)), on a PC device behind the Xbox-ABI proxy, and its
command recorder (RVA `0x59a0`) was read with Ghidra 12.2 through REA. RVAs are from image base `0x180000000`. The
run's command-list calls are the same 13, in the same order, on an RTX 2000 Ada (driver 597.06) and on WARP.

**Device.** At context creation it calls slots 14, 15, 18, 19, 27, 38 and 76 and heap slot 9, all with the shapes the
PC order and the measured overrides predict; slot 76 takes the seven `CreateCommittedResource` arguments. During
`ffxDispatch` it adds 16 `CreateRootSignature` and heap slot 10 (the recorder also has a path that calls 24
`CopyDescriptorsSimple`, not taken in this run), plus its two compute-pipeline paths below.

Which descriptor-copy path runs is decided by a **byte at `+0x18` of the device object**, read before each copy at
`0x5bb8`, `0x5dd1`, `0x5f8c` and `0x617e`. When it is 0 (the run), the DLL copies the 32 bytes at the source CPU
handle to the destination CPU handle itself (two 16-byte moves, e.g. `0x5bc0`–`0x5bce`), so CPU descriptor handles
must be pointers to at least 32 writable bytes. When it is set, the same copy is device slot 24,
`CopyDescriptorsSimple(1, dst, src, 0)` (calls at `0x5bed`, `0x5e06`, `0x5fc2`, `0x61b3`). A device proxy that sets
that byte therefore sees this DLL's descriptor copies as calls. Like the list's fields (below), the byte is past the
vtable pointer of a public object.

- **Slot 67** (call at `0x56d9`) is **not** the PC `CreateProtectedResourceSession1`: it takes `(desc, riid, out)`
  with `IID_ID3D12PipelineState`. The DLL's own messages name it `DeserializeComputePipelineStateX`. Its desc is the
  root signature at `+0x00`, then pointer and size pairs for the shader's `FSPX` entries: kind 0 at `+0x08`, kind 1
  at `+0x18`, a copy of the kind-6 program at `+0x28` with its exact size at `+0x30`, and kind 7 at `+0x38`
  ([fspx.md](fspx.md)). The proxy maps the console desc to `D3D12_COMPUTE_PIPELINE_STATE_DESC` and routes it to
  `CreateComputePipelineState` (PC slot 11). On a PC device that call fails (`E_FAIL`) because the code is not a PC
  shader, so the DLL falls back to slot 62; its message for that failure calls the fallback "NewBE". With kind 0 in
  the description (always, from this DLL) the proxy passes all four pairs on as one `FSPX` payload, checking each pair
  before reading it. The Vulkan device then finds the translated module by `fspx_code_sha256` and `fspx_state_sha256`
  ([fspx.md](fspx.md), "Build and run-time lookup"), for the programs the build translated.
  - **The layout is pinned from the DLL's code:** its FSPX reader (`0x1a20`), the same loop inline for a container,
    and the copy into the description. The kind 0 and kind 6 of one call come from the same payload. With the
    reader run on all 29 programs, every translated program resolves to its own entry ([fspx.md](fspx.md), "How
    the table was pinned").
  - **A failed call is logged with its number** (`slot 67 call N failed`). The DLL creates its pipelines one after
    another, so a miss is not necessarily the first call's.
- **Slot 62** (call at `0x5825`) takes the measured `(D3D12_COMPUTE_PIPELINE_STATE_DESC*, 0, 0, riid, out)`. The PC
  runtime refuses the DLL's code (`E_FAIL`): it is not a PC shader. The DLL's pipeline function then returns
  `0x8000000d` (`0x4b85`), but the dispatch goes on, records the null pipeline and faults (`0x65ff`), which is where a
  PC device stops. The Vulkan device (`LEONIDA_D3D12=vulkan`) takes the container: it hashes the kind-6 bytes of the
  `ACCC` part and creates the pipeline from the translated module, for the programs the build translated; others,
  and a malformed `FSPX` directory, are refused with `E_INVALIDARG`.
- **Pipeline slot 11 sizes the compute scratch.** After either path succeeds (slot 67 jumps to `0x5833`), the DLL
  calls the new pipeline's slot 11 as `(0, 0, &obj)` (`0x586d`); the public `ID3D12PipelineState` ends at slot 8, so
  this is a console-only method. On success it reads a dword through `obj`'s slot 3 (`0x5881`), multiplies it by
  1024 (`shl ecx, 0xa`), stores it in the pipeline's record at `+0x44` and releases `obj` (slot 2). The context keeps
  the largest value (`+0x10f8`). Before a dispatch whose pipeline has a non-zero value (`0x64d3`; zero skips the
  binding), the DLL creates the scratch buffer once, `largest * 0xd0` bytes rounded up to 64 KiB, through device
  slot 27 in state `UNORDERED_ACCESS` (`0x65ab`), names it `FFX_XBox_ComputeScratch`, and binds it with list slot
  141 `(2, GPU VA, buffer size, this pipeline's value)` (`0x65f8`). Measured: the calls and the arithmetic.
  - **Failure and zero (measured, `0x585c` to `0x58b1`).** The value at `+0x44` is set to 0 before the call. A failed
    HRESULT or a null `obj` skips to the function's success return with the value still 0, so pipeline creation does
    not depend on slot 11. At dispatch a value of 0 skips both the scratch buffer and list slot 141 (`0x64d6`).
  - **The object (measured shape, hypothesis about its type).** Slot 3 is called with no arguments and its result is
    read as a pointer to the dword; slot 2 releases it. That is the shape of `ID3DBlob::GetBufferPointer` and
    `Release`.
  - **No per-program size (measured on the 29 programs).** `COMPUTE_PGM_RSRC2.SCRATCH_EN` is clear in all 29, no
    program has a `scratch_*` or `flat_scratch` instruction, and kind-0 words 0, 1 and 2 are each one value across
    all 29.
  - **Implemented.** The Vulkan device's pipeline answers slot 11 with a blob holding 0; console slots 9 and 10
    return `E_NOTIMPL`. A translated module keeps its private data in SPIR-V, not in a console scratch buffer.
  - **Open.** What the console returns there for these programs, the unit of the dword (the factor 1024 suggests
    KiB), the meaning of `0xd0`, and of the leading `2`.
- The DLL's 29 shaders are `DXBC` containers with `SFI0`, `ISG1`, `OSG1`, `PSV0`, `RTS0`, `HASH` and `ACCC` parts
  and no `DXIL`. The `ACCC` payload starts with the tag `FSPX` in all 29; the game's own native shaders are `FSSM`
  inside `XBCH` ([design.md](design.md)). `FSPX` is a directory around an uncompressed RDNA2 program and its
  dispatch state; `tools/fspx_unpack.py` unpacks it for the translator, which takes 17 of the 29 today, and
  `make_game.py`'s `shaders` stage does both for the package's DLL. What stops the other 12 is listed in
  [fspx.md](fspx.md), "Translation status".

**Command list.** The console list is **not** the PC order past `ResourceBarrier`. Each row is backed by the
argument shape at the call and, where marked "run", by the call seen in the run. Rows marked "inferred" fit the
shape and the shift around them, but the shape alone does not single the method out:

| Console slot (offset) | PC slot | Method | Evidence |
|---|---|---|---|
| 14 (`+0x70`) | 14 | `Dispatch` | the vtable path of the inline dispatch at `0x59a0`; the BVH builder's `Dispatch(1,1,1)` at `+0x70` |
| 15 (`+0x78`) | 15 | `CopyBufferRegion` | run: `(resource, 0, resource, …)` |
| 25 (`+0xc8`) | 25 | `SetPipelineState` | `(pipeline)` at `0x59a0` |
| 26 (`+0xd0`) | 26 | `ResourceBarrier` | run: `(1, barriers)`; the BVH builder's `+0xd0` |
| 32 (`+0x100`) | **28** | `SetDescriptorHeaps` | run: `(1, &heap)`, the heap a device-proxy descriptor heap |
| 33 (`+0x108`) | **29** | `SetComputeRootSignature` | one pointer argument, the first call at `0x59a0` (null in the run, where pipeline creation had failed); in the BVH builder the pointer stored next to the pipeline (section 3.3) |
| 35 (`+0x118`) | **31** | `SetComputeRootDescriptorTable` | `(index, heap GPU start + offset)` at `0x59a0` |
| 41 (`+0x148`) | **37** | `SetComputeRootConstantBufferView` (inferred) | `(index, 64-bit GPU VA from the constant allocator)` at `0x6738`; the root SRV/UAV setters take the same shape |
| 53 (`+0x1a8`) | **49** | `ClearUnorderedAccessViewUint` | run: `(GPU handle, CPU handle, resource, …)` |
| 54 (`+0x1b0`) | **50** | `ClearUnorderedAccessViewFloat` | `(GPU handle, CPU handle, resource, values, 0, null)` at `0x7296`; chosen over the slot-53 call at `0x72ee` by a compare of a local with `0x12` (`0x7266`), with the four values passed in place where the 53 path copies them to the stack |
| 55 (`+0x1b8`) | **51** | `DiscardResource` | `(resource, null)` at `0x6dcb`, after the pending barriers are flushed through slot 26 |
| 59 (`+0x1d8`) | — | console only (shape) | four calls, two shapes: `(1, blob, size)` at `0x6b5a` and `0x6d53`, the blob a run of qwords whose first is `0xf3080 \| (qword count & 0x7f)`; and `(0, &qword, 8)` with the qword 1 at `0x6e52` and `0x6eb8`, only when a profiling bit of the context is set (`0x6e2a`). The first argument is set as a byte (`mov dl, 1` / `xor edx, edx`). It looks like one slot for both ends of a GPU event; no public method takes a bool first. With 55 at +4 and 61 at +2, the two slots the console lacks are among 56–60 |
| 61 (`+0x1e8`) | **59** | `ExecuteIndirect` | `(signature, 1, argument buffer, offset, null, 0)` at `0x59a0` |
| 85 (`+0x2a8`) | — | `CopyTextureRegion` shape (inferred) | run: `(stack pointer, 0, 0)`, twice per dispatch. Statically six arguments, `(&dst, 0, 0, 0, &src, null)`: the public `CopyTextureRegion` (PC 16) with two `D3D12_TEXTURE_COPY_LOCATION`s, not one pointer. At `0x7147` dst is `{texture, SUBRESOURCE_INDEX, 0}` and src `{buffer, PLACED_FOOTPRINT, footprint}`, the footprint from device slot 38 (`GetCopyableFootprints`, `0x708b`); at `0x7191` both are subresource 0 (texture to texture). The caller also clears the next stack argument (`0x70e7`); whether the method takes a seventh is open, as is what console slot 16 is |
| 141 (`+0x468`) | — | console only | `(2, GPU VA, size, n)` at `0x65f8`: binds a buffer the DLL names `FFX_XBox_ComputeScratch`; `n` is the bound pipeline's scratch value from pipeline slot 11 (see above) |

So the console list has four more methods than the PC list between `ResourceBarrier` (26) and `SetDescriptorHeaps`
(console 32), and two fewer between `ClearUnorderedAccessViewUint` (console 53) and `ExecuteIndirect` (console 61).
Mapping the console list as "PC plus a constant" is wrong. The BVH builder adds 37, 39, 43 and 45 (the rest of the
+4 root group) and the console-only 71, 73, 80, 146 (`Barrier`) and 148 (sections 3.3 and 3.3a).

**The list object is wider than its vtable, and this is what that width is for.** Before a root-argument or dispatch
call the DLL checks a write cursor at `+0x18` against a limit (`+0x30` for root arguments, `+0x28` for a dispatch). While there is room,
it writes the command into that buffer itself instead of calling: a root argument is a header word from the
per-root-parameter table at `+0x344` and then the value (32 bits for a descriptor table, 64 for the slot-41 GPU VA),
a dispatch four words (`0x4002c400`, or `(flag | 0x1000ba00) << 2`, then x, y, z).
The BVH builder also writes a five-word dispatch, `0x4003cc00, x, 1, 1, W`.
- **Sites:** `0x14972`, `0x15215`, `0x1536a` and `0x1d002`, each checked against the dispatch limit `+0x28`. The
  cursor advances 20 bytes and the flag byte at `+0x140` is cleared.
- **It is the inline form of console slot 80** (section 3.3a). When the buffer is full, each site calls slot 80
  (`+0x280`) instead, as `(x, 1, 1, W)` with the same `W`.
- **`W` is `0x8020`** at `0x14972` and `0x1d002`, and **`0x20`** at `0x15215` and `0x1536a`. What bit 15 selects is
  open.

The FSR DLL never writes this form.
**Byte `+0x143` is console-runtime state:** the BVH builder reads it at 31 sites and the FSR DLL at one (all `cmp
byte ptr [reg+143h]`), but neither DLL writes it. Every store into `+0x140`..`+0x143` in either DLL is a byte store to
`+0x140` or `+0x141`, and no wider store reaches `+0x143`. A replacement list can leave it 0.
It sets a flag byte at `+0x140`, stores a flag at `+0x141` from the
pipeline, and copies 32 bytes of the pipeline object (from its `+0x10`) to `+0x544`. These are the fields
section 3.4 measured in the BVH builder's helper without a meaning (`+0x18`, `+0x30`, `+0x140`, `+0x344`, `+0x544`).
**With the cursor at the limit, every command goes through the vtable**: that is how `fsr_dispatch_test` records
them, and it is the simplest contract for a layer that does not want to decode these packets. Whether the game's
executable makes the same check before writing is unknown until it is available; it is compiled against the same
console headers.

**Every inline write in the package's DLLs checks first.** `tools/inline_scan.py` finds each cursor store-back
(`mov [list+18h],reg` after stores through the cursor) and looks for the compare in front of it. On the 1.0.2.1
DLLs (dumpbin 14.44 listings):

| Image | Inline writes | Checked | Not checked | Limit compared | Fallback slot (sites) |
|---|---:|---:|---:|---|---|
| `XboxBvhBuilder.dll` | 196 | 196 | 0 | `+0x30` ×165, `+0x28` ×30, a register loaded in another block ×1 | 45 ×71, 37 ×35, 25 ×34, 14 ×24, 39 ×16, 43 ×9, 80 ×4; 3 reach theirs through a shared jump the tool does not follow |
| `amd_fidelityfx_upscaler_xbox.dll` | 9 | 9 | 0 | `+0x30` ×8, `+0x28` ×1 | 35 ×6, 25, 41, 14 |

**Measured.** The packet headers it reads back are the two dispatch forms above: `0x4002c400` (26 four-word
dispatches in the builder, one in FSR) and `0x4003cc00` (the builder's four five-word ones). The other 14 DLLs have
no inline write, except one checked site in `libHttpClient.GDK.dll`, which imports nothing from D3D12: read a
result as an upper bound of about one site per image. Removing the compare and branch in front of the builder's
`0x14972` write turns exactly that site into "not checked", so a missing check is visible.

On the executable, `python -I tools/inline_scan.py <exe> --sites` answers the question in one pass (see the day-one
list in `docs/design.md`): no unchecked site means a cursor-at-limit layer sees every command the game records; each
unchecked site is listed by RVA and is a write that layer never sees. Its fallback-slot line is the set of console
list methods the game reaches through the vtable. Limit offsets other than `+0x28`/`+0x30` would be command kinds
the DLLs never write (draws, input-assembler and output-merger state); read a few of those sites by hand.

**The five-word dispatch.** The BVH builder also writes `0x4003cc00, x, 1, 1, W`:

- **Sites:** `0x14972`, `0x15215`, `0x1536a` and `0x1d002`, each checked against the dispatch limit `+0x28`. The
  cursor advances 20 bytes and the flag byte at `+0x140` is cleared.
- **Its vtable fallback is console slot 80** (section 3.3a), read from the code: when the buffer is full, each site
  calls slot 80 (`+0x280`) instead, as `(x, 1, 1, W)` with the same `W` (`0x149ba`, `0x1525d`, `0x153b4`, `0x1d047`).
  So the packet is slot 80's inline form; that slot 80 is a `Dispatch` with one extra word stays inferred.
- **`W` is `0x8020`** at `0x14972` and `0x1d002`, and **`0x20`** at `0x15215` and `0x1536a`. What bit 15 of `W`
  selects is not known.

The FSR DLL never writes this form.

**Byte `+0x143` is console-runtime state:** the BVH builder reads it at 31 sites and the FSR DLL at one (all `cmp
byte ptr [reg+143h]`), but neither DLL writes it. Every store into `+0x140`..`+0x143` in either DLL is a byte store to
`+0x140` or `+0x141`, and no wider store reaches `+0x143`. A replacement list can leave it 0.

**The root-argument header word** is the root parameter's table entry `table[root]` (`+0x344 + 4·root`), adjusted
only for root constants:

- **Several 32-bit constants at once:** `table[root] + ((count − 1) << 16)`, then `count` dwords. Examples:
  `+ 0x120000` for 19 constants (`0x14e3b`), `+ 0x70000` for 8 (`0x1d324`), `+ 0x30000` for 4.
- **One constant at a non-zero offset:** `table[root] + 4 · destOffset`. At `0x19fca` the builder adds `0xc` and its
  vtable fallback is slot 37 `(2, value, 3)`; at `0x1a346` it adds `0x14` and the fallback is `(2, value, 5)`.
- **Otherwise the entry is written unchanged:** one constant at offset 0, a descriptor table (then 4 bytes), a root
  CBV/SRV/UAV (then the full 8-byte GPU address, e.g. `0x19fff`).

No site combines both adjustments. Before writing, every reader tests the entry against `0xC0000000` (98 sites in the
builder, 7 in the FSR DLL), and sets the flag byte at `+0x140` only when both of those bits are clear.

### 3.5 What the package does not measure

There is no shipped client for `ID3D12CommandQueue`, `ID3D12Fence`, `ID3D12CommandAllocator`,
`PresentX`, `CreateCommandQueueX`, `SuspendX` or `ResumeX`:

- the BVH builder's constructor takes a device, and its build functions take a command list; no
  call at the public device creation slots 8/9/12/36 appears in its `pdata` functions, and it
  does not create a swapchain;
- no shipped module imports `d3d12_xs.dll` beyond the two root-signature calls;
- no shipped module contains a queue/fence/allocator IID or the string `PresentX`.

The queue/fence/allocator Xbox surface is therefore **not measured here at all**. The only
command-object facts from the package are the command-list call sites in section 3.3.

## 4. Prior-art claims

Prior art is cited for facts only; no code is copied. Licence details are in section 8.

### 4.1 SCARAB (GPL-3.0-only)

SCARAB measured the console `d3d12_xs.dll` export surface and reports the console-only
extensions. The parts relevant to this page:

| Claim | Status | Detail |
|---|---|---|
| `PresentX` is an `ID3D12CommandQueue` method | **Prior art** | Class-attributed by a UMD validation message and by 367 call sites in the console SDK samples ([`d3d12-xs-wrapper.md:69-97`](https://github.com/p-portela/SCARAB/blob/d304fa2/docs/d3d12-xs-wrapper.md)). |
| `PresentX` carries at most 2 planes and 8 views per plane | **Prior art** | `D3D12CONS_PRESENT_PLANES_MAX = 2`, `D3D12CONS_PRESENT_VIEWS_MAX = 8`; the UMD message names `pPlaneParameters[%d].ResourceCount <= 2`. |
| Present flags `NONE`, `IMMEDIATE`, `PERSIST`, `UNIQUE_VIEW_RESOURCES` exist | **Prior art** | Names are recovered; the numeric values are **not** measured. `console_extensions.h` assigns `0x1/0x2/0x4` while `present_adapter.h` says the values are unknown and uses zero; the two contradict each other ([`d3d12_xs_surface.md §7`](d3d12_xs_surface.md)). |
| `PresentX` signature and slot | **Open** | No signature, slot or call sequence is recovered. |
| `SuspendX` / `ResumeX` are queue methods | **Prior art** | 122 sample call sites each; no signature or slot. |
| `CreateCommandQueueX` is a device method | **Prior art** | UMD message ties it to `ID3D12Device`; the inferred shape is `(pDesc, pShaderQueueParams, riid, out)`; the UMD says a shader-based queue supports only `D3D12_COMMAND_LIST_TYPE_COMPUTE`; SCARAB returns `E_NOTIMPL`. Slot and parameter layout are **Open**. |
| `ID3D12ConsDmaCommandList` IID `30bc77b9-8379-4b25-95de-b8588c42275d` | **Prior art** | Exported at ordinal 100; the DMA list rule is quoted from the UMD. `D3D12CONS_COMMAND_LIST_TYPE_DMA = 6` is **Derived** by appending to the PC enum, not measured. |
| Command-list extension order | **Assumed** | SCARAB's own header says the order within the appended block is not recovered; every appended slot logs its name and raw arguments ([`console_extensions.h:13-31`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/extension-wrapper/console_extensions.h)). Do not copy it. |
| `OMSetFrontAndBackStencilRef` is public GCL8 | **SDK fact, not an extension** | Same signature at `d3d12.h:32819`; the SCARAB catalogue's placement after GCL6 would land in the wrong slot ([`d3d12_xs_surface.md §5.1`](d3d12_xs_surface.md)). |

The `DXGIXPresentArray`, `DXGIXSetFrameNotification`, `DXGIXSetVLineNotification` and
`DXGIXGetFrameStatistics` names are **free functions** at ordinals 20/21/22/19, not queue
methods; no parameter layouts are recovered. `DXGIXGetFrameStatistics` is reported to write 64
bytes, so a zero-argument stub is not safe ([`d3d12_xs_surface.md §7`](d3d12_xs_surface.md)).

### 4.2 `americas-recomp` (GPL-2.0-only)

The prior art that ran the package DLLs against a PC D3D12 device measured the **device**
surface, not the command objects:

- device slots 1/14/15/16/18/19/27/38 in desktop order; slot 14 is `CreateDescriptorHeap`;
- device slot 62 is called as `CreateComputePipelineState` (the PC interface's slot 62 is
  `CreateStateObject`), and slot 76 is Xbox texture creation;
- descriptor-heap `GetCPU/GPUDescriptorHandleForHeapStart` (heap slots 9/10) return the 8-byte
  handle in RAX;
- Xbox root-signature descriptors carry flags `0x03000000`, masked on a copy;
- the BVH builder's 53 compute shaders are Xbox DXIL.

Its own summary explicitly stops before **presentation (`PresentX` and plane parameters),
command-list extensions, and queue/fence creation slots**
([`d3d12_vulkan.md §7`](d3d12_vulkan.md),
[`PLATFORM_SURFACE.md`](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md)).
So it adds **no** measured queue/list/fence slot to this page. The device slots it did measure
are useful context: they are the reason a replacement cannot assume the whole device vtable is
public order, and the reason the four creation slots 8/9/12/36 still need a trace.

### 4.3 What prior art does not close

Even taking both projects at face value, the following remain **Open**:

- the Xbox slots for `CreateCommandQueue`, `CreateCommandAllocator`, `CreateCommandList`,
  `CreateFence`;
- the Xbox queue vtable beyond the public chain (`PresentX`, `SuspendX`, `ResumeX`);
- the `PresentX` signature, plane struct, resource/view arrays and swapchain parameters;
- the command-list extension slots other than the ones sections 3.3a and 3.5a measured (71, 73, 80, 85, 141, 146, 148);
- the DMA list type's real numeric value and the compute/copy console list IIDs, which the package
  itself cannot supply (§3.2a): they need the running title or a vendor runtime;
- any Xbox-specific return convention on these interfaces.

## 5. What a runtime can do without the executable

### 5.1 Implementable and testable now

The repository already implements the public-chain objects on Vulkan:

- `ID3D12CommandQueue`, `ID3D12CommandAllocator`, `ID3D12GraphicsCommandList7` and
  `ID3D12Fence1` are defined in `runtime/shims/d3d12_com.cpp:41-352`, on the Windows SDK vtable
  order, with generated stubs for every method they do not implement
  (`runtime/shims/d3d12_com.h:1-5`);
- `CreateCommandQueue`/`CreateCommandAllocator`/`CreateCommandList`/`CreateFence` are wired at
  the public device slots in `runtime/shims/d3d12_vulkan_device.cpp:345-382`;
- `runtime/test/d3d12_com_test.cpp:77-153` creates all four, records clears and both barrier
  kinds, rejects an open list at `ExecuteCommandLists`, executes behind a queue `Wait`, and
  checks fence signalling and `SetEventOnCompletion`. The test is source-level evidence; this
  page did not rebuild the Vulkan runtime.

The builder's `ResourceBarrier` call at `+0xd0` and `Dispatch(1,1,1)` at `+0x70` are consistent
with the SDK slots the runtime already implements. Legacy barriers (slot 26) and enhanced
barriers (slot 80 in GCL7) both have IR and Vulkan replay support
(`runtime/src/command_list.h`, `runtime/src/vulkan_replay.h`; see
[d3d12_vulkan.md §3.2](d3d12_vulkan.md)).

### 5.2 Implementable, but unverified against the Xbox device

A runtime can create the four objects and hand them out at the **public** device slots 8/9/12/36
today. That is an assumption, not a measurement: the Xbox device was measured at
1/14/15/16/18/19/27/38/62/76, and slots 8/9/12/36 are exactly the ones this page needs. If the
Xbox slots differ, a generic forward-to-PC thunk calls the wrong method. The current proxy
(`runtime/shims/d3d12_xs_device.h:1-18`) forwards every unshadowed slot in PC order and logs
first calls; a wrong slot would therefore be visible as a misnamed call, not a silent success.

Queues and fences are wrapped the same way.

- **What is wrapped.** Device slots 8 (`CreateCommandQueue`), 75 (`CreateCommandQueue1`), 36
  (`CreateFence`) and 32 (`OpenSharedHandle`, when it opens a fence) return queue and fence proxies. Each
  proxy forwards the public chain in PC order: 19 queue slots, and 11 or 12 fence slots, in SDK 10.0.26100.
- **Logging.** Each slot logs its first call. `--trace-d3d12` records every call except `AddRef`/`Release`,
  as `XboxID3D12CommandQueue::Signal called` or `XboxID3D12Fence::Slot40 unsupported` (the README's trace
  grammar); on the Vulkan device the layer's own record of the same call follows it. An interface the real
  object implements but the proxy does not model is logged once and refused.
- **The refused tail.** A slot past the public chain, up to the end of the proxy's own table (128 queue and
  64 fence slots, every one a thunk), logs its first four argument words once and returns `E_NOTIMPL`.
- **The guard region.** Each table sits at the end of the first page of a 64 KiB reservation, and the rest
  of that reservation is never committed. A slot past the table therefore faults on the vtable read itself,
  inside the reservation, and a vectored handler logs the slot before the fault goes on. A vtable read that
  lands beyond the reservation is not caught.
- **Unwrapping.** Wherever a device or queue method hands a heap, queue or fence object to the real
  runtime, the proxy passes the real object instead: queue `Signal`/`Wait` (14/15), and device `CreateSharedHandle` (31), `MakeResident`
  and `Evict` (34/35), `SetEventOnMultipleFenceCompletion` (45), `SetResidencyPriority` (46) and
  `EnqueueMakeResident` (50).
- **QueryInterface.** It answers only the modelled interfaces, so no real object reaches the title.
- **What is not covered.** Command lists are not wrapped yet: `ExecuteCommandLists` passes the lists through,
  and a heap proxy given to a list's `SetDescriptorHeaps` reaches the real list unchanged. The command-list
  proxy of §3.5a adds the unwrap at both places.

Command allocators get the same kind of proxy: device slot 9 (`CreateCommandAllocator`) returns one.

- **What is forwarded.** The public chain, 9 slots: the base chain and `Reset` at 8. A slot past it, up to the
  end of a 64-slot table, is logged and refused, and a read past the table faults in its own guard region.
  The console allocator's extra methods cannot be measured from the package (D3D-360). No shipped module creates,
  receives or calls an allocator (§3.5). The package has no `d3d12_xs.dll` either, and no export of the three
  D3D12 clients takes an allocator. `dumpbin /exports` (MSVC 19.52) lists the BVH builder's 12 exports, which take
  an `ID3D12Device11` or a command list (`docs/d3d12_xs_surface.md`), and the FSR loader's and upscaler's
  `ffxConfigure`, `ffxCreateContext`, `ffxDestroyContext`, `ffxDispatch` and `ffxQuery`. A call past slot 8 could
  only show up in the game executable, which the package doesn't contain, so such a slot stays logged and refused.
- **Where a list takes it.** A list takes an allocator from the title in `CreateCommandList` and `Reset`, so
  the proxy is safe only where the list side resolves it. Slot 9 therefore wraps an allocator only when it
  answers `kLeonidaCommandAllocatorTag` (`d3d12_xs_device.h`). That is the Vulkan layer's own allocator, whose
  `CommandAllocator::from()` resolves the proxy for both list calls.
- **On the host runtime.** A PC runtime allocator comes back unwrapped, logged once, because the PC runtime's
  lists would take a proxy for their own allocator. A command-list proxy that unwraps at `Reset` (§3.5a) is
  what lets the host backend wrap allocators too.
- **Residency.** An allocator is a pageable, so `MakeResident`/`Evict`/`SetResidencyPriority`/
  `EnqueueMakeResident` unwrap allocator proxies like heap, queue and fence proxies.

`allocator_proxy_test` checks this on the Vulkan device and on the host runtime, and the mock test checks it
over allocators with and without the tag. On the Vulkan device it checks forwarding, the refused tail, the guard
region, identity, and D3D12's allocator rules through the proxy, and runs a list created and `Reset` on the
proxy whose copies go through the queue and fence proxies and are read back.

Which of this is measured: the slot numbers and signatures are the SDK's (public). That the console
keeps the public queue and fence slots, and device slots 8/32/36/75, is an assumption the package cannot
test, as above. The proxies make a wrong assumption visible in the log; they do not prove it right.
`queue_proxy_test` checks all of this with no game files, on the host runtime (WARP and a hardware
adapter) and on the Vulkan device; `OpenSharedHandle` of a shared fence only on the host runtime, since the
Vulkan device does not share fences (the test prints "not checked" there). Each error case runs in its own
process and is compared with the real object's HRESULT. The test also includes a console-style client: it makes a heap through slot 14 (handle in
RAX) and buffers through slot 76 (layout word 0x1). It then copies through the queue proxy, finishes through
the fence proxy, reads the data back, and makes residency calls that carry the heap, queue and fence proxies.
`device_proxy_test` gets its queue and fence through the registered `D3D12XboxCreateDevice` shim.

The package-gated `bvh_builder_test` and `fsr_dispatch_test` stay at `SKIP (no package)` without
`LEONIDA_GAME_DIR`, and they would not cover these paths if they ran. Neither creates a queue or a fence:
the BVH builder's constructor calls device slots only, and `fsr_dispatch_test` records into a logging list it
never executes. Neither DLL calls the device slots the proxies change (§3.5, §3.5a).

The same caveat applies to **return conventions**: the public AMD64 COM convention (HRESULT in
EAX, `void` methods, out-parameters through pointers, `GetDesc` returning a 16-byte struct by
value) is what the runtime implements. No Xbox-specific convention on a queue, list, fence or
allocator has been measured. The only measured Xbox return-convention quirk in the graphics
surface is the descriptor-heap start handle in RAX (prior art), which is a heap method, not a
command object.

### 5.3 Blocked on the executable or console driver

| Item | Why it is blocked |
|---|---|
| `PresentX` | No slot, signature, plane struct or swapchain/plane parameters. Prior art gives only names, the 2-plane/8-view limits and four flag names. |
| `CreateCommandQueueX` | No slot or parameter layout; only the UMD message and an inferred shape. |
| `SuspendX` / `ResumeX` | No slot or signature. |
| Xbox queue/list/fence/allocator slot deviations | No runtime binary and no game executable; the package has no queue, fence or allocator client, and its only command-list client is the BVH builder. |
| Builder's list field `+0x350` | Read once (`0x1d877`) and not explained by the inline buffer of section 3.5a, which covers `+0x18`, `+0x28`, `+0x30`, `+0x140`, `+0x141`, `+0x344` and `+0x544`. |
| DMA list use/value | The IID is prior art; type 6 is derived; no shipped module uses it, and no module can name any GUID at a call site beyond its known ten or holds a static queue description with a non-PC type (§3.2a). |
| Compute/copy console list IIDs | Not exported (see #3520) and not present at any call site (§3.2a); only the running title can supply them. |

### 5.4 GCL8 (resolved in PR #164)

The shipped builder's rehydrate exports name `ID3D12GraphicsCommandList8`, and
`OMSetFrontAndBackStencilRef` is public GCL8 at slot 81. **PR #164 adds the public GCL8 interface**
to `runtime/shims/d3d12_com.cpp`: it records the front/back values, resets them in `Reset`, answers
`IID_ID3D12GraphicsCommandList8`, and registers the `d3d12_gcl8` CTest (9/9). The values are recorded
but not applied because this layer still has no pipeline/draw path. The builder's `+0x490` call is the
enhanced-barrier `Barrier` at console slot 146 (section 3.3), which a console-order list routes to GCL7 slot 80.

## 6. Return conventions

| Interface | Public convention | Xbox-specific measurement |
|---|---|---|
| `ID3D12CommandQueue` | `ExecuteCommandLists`, `SetMarker`, `BeginEvent`, `EndEvent` are `void`; `Signal`, `Wait`, `GetTimestampFrequency`, `GetClockCalibration` return `HRESULT`; `GetDesc` returns `D3D12_COMMAND_QUEUE_DESC` by value | None measured |
| `ID3D12CommandAllocator` | `Reset` returns `HRESULT` | None measured |
| `ID3D12Fence`/`Fence1` | `GetCompletedValue` returns `UINT64`; `SetEventOnCompletion`/`Signal` return `HRESULT`; `GetCreationFlags` returns the flags enum | None measured |
| `ID3D12GraphicsCommandList`/1–8 | Most recording methods are `void`; `Close`/`Reset` return `HRESULT`; `GetType` returns the list type; GCL8 `OMSetFrontAndBackStencilRef` is `void` | The builder's `+0xd0` and `+0x70` calls ignore RAX, consistent with `void`; no other convention measured |

The `GetDesc` by-value return is the one place a generic thunk can get the ABI wrong on x64;
the runtime implements it directly (`runtime/shims/d3d12_com.cpp:309`). No Xbox variant of that
return has been measured.

## 7. Next measurable step

The executable is absent, so the next step should use the one shipped client that exists rather
than invent slots.

1. **Confirm sections 3.3, 3.3a and 3.3b with a logging proxy.** Create a test-only command
   list whose public GCL6/GCL8 vtable logs `(slot, raw arguments)` and whose tail is a page of
   logged unknown slots. Pass it to
   `XboxBvhBuilder::BuildRaytracingAccelerationStructureBatched2X` with a minimal non-empty
   descriptor. The proxy ran twice; both runs stopped one step short, at the CPU-side geometry
   validation of section 3.3b (fault at `0x142d3`), before the first list call. The static pass
   settled what the first calls will be: not the `+0xd0` barrier — it is gated and never fires
   in the shipping DLL — but the wrapper's slot-146 barriers, then the upload function's slot
   148 for `count * 40` bytes and its own `Dispatch((count+31)/32, 1, 1)`, and only then the
   per-entry `Dispatch(1,1,1)` of section 3.3; the other new slot, 90, is gated too
   (section 3.3b). Keep the cursor at its limit so every call reaches the vtable, and back the
   geometry descriptors with CPU-readable memory holding valid 0x38-byte records per section 3.3b
   (type 0, index format 0, vertex count a multiple of 3 — prior art measured a 3,000-vertex
   triangle geometry; the prebuild answer is 204,800 bytes / 128,704 scratch). A descriptor shaped
   like the one `GetRaytracingAccelerationStructurePrebuildInfo` accepts is not enough by itself,
   because the validation reads the records with CPU loads.
2. **GCL8 is now implemented** in PR #164 (public slot 81, recorded/reset, CTest 9/9); the
   builder's `+0x490` is `Barrier` (section 3.3), and its slot 148 needs an upload allocator (section 3.3a).
3. **When the executable appears, trace the four creation calls first.** Per-slot counters on
   device slots 8/9/12/36, then queue slots 10/14/15, fence slots 8/9/10, and every slot beyond
   the public chain. The first `PresentX` call and the queue vtable tail are the only way to
   recover the present surface; a DXGI import would make the DXGI path the first
   implementation instead ([d3d12_vulkan.md §8.2 Q6](d3d12_vulkan.md)). The counters exist: device,
   queue and fence proxies count every slot, and the queue and fence tails are logged with their
   first argument words (§5.2).

Do not fill the unknown slots with guesses. A wrong slot is worse than a clean refusal; the
existing device proxy already follows that rule (`runtime/shims/d3d12_xs_device.h:16-18`).

## 8. References and provenance

### 8.1 Licence

Leonida is GPL-3.0-or-later. SCARAB is GPL-3.0-only and `americas-recomp` is GPL-2.0-only; both
are cited for ABI facts only. No code, headers, generated stubs, comments or tables were copied
from either project. The PE and disassembly facts in section 3 were produced locally from the
shipped `XboxBvhBuilder.dll`; the public slot table was parsed from the installed Windows SDK
10.0.26100.0. No game data was read, decrypted or executed. See
[`docs/related_projects.md`](related_projects.md) for the project-level rule.

The XboxBvhBuilder export names quoted in section 3.1 are mangled C++ symbols from the DLL's
export table, reproduced as ABI facts. The GUID values in section 3.2 are 16-byte constants from
the SDK; the console GUIDs are attributed to SCARAB in
[`d3d12_xs_surface.md §4`](d3d12_xs_surface.md).

### 8.2 Reproduction

- Package inventory and absent files: directory listing plus
  `layout_3202719e-b4b4-421c-9f8f-70d1c4a7d60b_xs.xml:14`.
- Imports/exports: `python tools/pe_inspect.py <dll> --game-dir <package> --out <dir>`; the
  16-DLL summary in section 1.2 used `pefile` normal + delay-load directories.
- IID scan: read the 80 `DEFINE_GUID(IID_ID3D12…)` values from
  `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um\d3d12.h`, plus the two console
  GUIDs from [`d3d12_xs_surface.md §4`](d3d12_xs_surface.md), and search each DLL for the
  16-byte little-endian value. Absence is evidence about static constants, not about runtime
  `QueryInterface` calls.
- Builder call sites: Capstone 5.0.7 x86-64 disassembly of the 12 exported functions and the
  functions they call directly; RVAs are PE relative virtual addresses. The
  `0xd0` call is in the batched entry (`0x216a0`, site `0x2171e`); the `0x490` call is in the
  per-descriptor function reached from it (`0x1d7c0`, site `0x1da02`). Section 3.3a: the
  `call [reg+disp]` sites in the rest of `.text` whose object register holds the list pointer,
  each read with the argument registers set before it and the inline write it falls back from;
  the barrier constant is the 16 bytes at RVA `0xcdf60` (file offset from the section table).
- Section 3.3b and the gated sites: a second, independent disassembly pass (iced-x86, run for the
  D3D-293 measurement) of the batched entry, the validation chain (`0xe6a0`, `0xcc00`, `0x14270`),
  the upload function `0x20ea0` and the wrapper call sites; the dynamic side is D3D-293's
  logging-proxy harness, whose two runs faulted at `0x142d3` before the first list call. The
  quoted validation messages are the DLL's own wide strings at the `.rdata` addresses the throw
  paths load. The gate-byte sentence is a linear sweep of `.text` collecting memory operands
  targeting `0xd7738`–`0xd773f`.
- Public slots: parse the `MIDL_INTERFACE`/`virtual` declarations in `d3d12.h`; the counts in
  section 2 include the seven `IUnknown`+`ID3D12Object` slots. Cross-check selected values with
  `python tools/d3d12_xs_surface.py --check --sdk` (116 slots, 83 standard GUIDs) and
  `python -I tests/test_d3d12_xs_surface.py`.

### 8.3 References

- [d3d12_vulkan.md](d3d12_vulkan.md) §3.1, §3.2, §4, §4.4, §4.5, §7, §8.2, Appendix A.2,
  Appendix B — the COM/vtable assumptions, the record/replay design, the known/unknown list and
  the BVH builder evidence.
- [d3d12_xs_surface.md](d3d12_xs_surface.md) §2, §4, §5, §5.1, §6, §7, §9 — the export/IID
  contract, console GUIDs, extension catalogue, GCL8 correction, DMA list rule and `PresentX`
  status.
- [related_projects.md](related_projects.md) — the GPL-2.0-only and GPL-3.0-only provenance
  rules.
- SCARAB, pinned at `d304fa26c0292e2f3ba76f23b15c9f30734bec02`:
  [`docs/d3d12-xs-wrapper.md:69-97`](https://github.com/p-portela/SCARAB/blob/d304fa2/docs/d3d12-xs-wrapper.md),
  [`components/extension-wrapper/console_extensions.h:13-31,45-72,84-191`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/extension-wrapper/console_extensions.h),
  [`docs/d3d12-xs-bridge.md:239-245`](https://github.com/p-portela/SCARAB/blob/d304fa2/docs/d3d12-xs-bridge.md).
- `americas-recomp`, branch `xbox-series`:
  [`docs/xbox/PLATFORM_SURFACE.md`](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md),
  [`tools/xbox/platform/README.md`](https://github.com/realdody/americas-recomp/blob/xbox-series/tools/xbox/platform/README.md).
- Windows SDK 10.0.26100.0:
  `um/d3d12.h:5193,5323,5477,6337,7216,7982,8701,9006,9011,9026,9178,20165,21255,30275,31123,31965,32816`.
