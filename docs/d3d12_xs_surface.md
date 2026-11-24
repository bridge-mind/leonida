# The measured `d3d12_xs` surface, from SCARAB

Status: reference, sourced. This page records what SCARAB measured about the Xbox Series
`d3d12_xs.dll` ABI, checks those claims against the files and the Windows SDK, and states what
Leonida can use. It does **not** adopt SCARAB's forwarding architecture: Leonida translates D3D12
to Vulkan, while SCARAB forwards to PC `d3d12.dll` / vkd3d-proton. Only ABI facts, IID values and
call-sequence constraints are taken from it. No SCARAB code is copied.

Source pin: `p-portela/SCARAB` at commit `d304fa26c0292e2f3ba76f23b15c9f30734bec02`
(2026-11-21), read-only clone. Licence: GPL-3.0-only; see §11.

Verification environment: Windows SDK 10.0.26100.0 headers, plus a read-only inventory of the
local 1.0.2.1 package (the 16 DLL export tables, and the `gameos.xvd` header, entropy and string
scan). No `d3d12_xs.dll` or game data was opened for this page, and no Wine run was made. Where a
statement depends on SCARAB's vendor DLL, it is marked **Reported** and cannot be reproduced from
either repository.

Confidence legend, consistent with [d3d12_vulkan.md](d3d12_vulkan.md):

- **Verified** — checked here against a committed file or the Windows SDK headers.
- **Reported** — SCARAB states it from a vendor binary that is not in either repository.
- **Derived** — follows from a documented rule plus stated anchors; the derivation is shown.
- **Open** — no evidence.

## 1. What the runtime is

SCARAB identifies `d3d12_xs.dll` as a 51,576-byte, PE32+ x86-64 front end
(`FileVersion 10.0.26100.6247`, product string `... D3D12.CONSOLE Runtime Library for Offline
PC`) that loads a user-mode driver at runtime. Its only non-CRT import is `KERNEL32.dll`. That is
why 116 exports fit in 51 KB: the implementation is in the UMD, not in this DLL
([`docs/d3d12_xs-findings.md:5-45`](https://github.com/p-portela/SCARAB/blob/d304fa2/docs/d3d12_xs-findings.md)).
For a replacement DLL, the export table is therefore the complete free-function contract.
SCARAB de-brands platform names throughout, so its quoted strings use "console" where the vendor
string would carry the platform name; §2.3 covers the consequence for export names.

SCARAB also measured that the console runtime and the PC D3D12 runtime define the same COM
interfaces: 28 `IID_*` data exports that exist on both sides are byte-identical. That is the fact
that makes the standard D3D12 surface forwardable rather than reimplementable — and for Leonida it
means a `QueryInterface` for a standard interface can be answered with the PC GUID without a
translation table.

Scope limit: the "28" are the *D3D12* IIDs compared against the D3D12 headers. The runtime also
exports 55 other standard GUIDs (D3D11, DXGI, `ID3D10Blob`, debug GUIDs). §3 checks all 83.

## 2. The export contract: 116 = 23 functions + 93 GUID data

SCARAB's plan records the composition as 23 functions and 93 GUID data exports
([`docs/plans/2026-11-21-translation-layer-architecture.md:30`](https://github.com/p-portela/SCARAB/blob/d304fa2/docs/plans/2026-11-21-translation-layer-architecture.md)).
The 93 is internally consistent: 5 `DXGI_DEBUG_*` + `IID_ID3D10Blob` + 43 D3D11/DXGI IIDs at
29–71 + 28 D3D12 IIDs at 72–99 + 16 IIDs at 100–115 = 93, and
`components/runtime-bridge/src/iid_table.cpp:104` declares `scarab_iid_count = 93` with exactly
93 definitions.

### 2.1 The 23 function exports

Ordinals are documented in
[`docs/d3d12-xs-bridge.md:99-113`](https://github.com/p-portela/SCARAB/blob/d304fa2/docs/d3d12-xs-bridge.md)
and the individual stubs in
[`components/runtime-bridge/src/d3d12_xs_bridge.cpp`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/runtime-bridge/src/d3d12_xs_bridge.cpp).
"Leonida today" describes `runtime/shims/d3d12_xs.cpp` at the time of writing.

| # | Export (SCARAB's spelling) | Kind | Leonida today | Note |
|---:|---|---|---|---|
| 1 | `D3D12ZeroMemoryX` | console | absent | console fast-zero; `memset` is equivalent |
| 2 | `PixGetPixieEntryPoints` | console (PIX) | absent | no PIX path |
| 3 | `D3D10CreateBlob` | console (D3D10) | absent | local `ID3D10Blob`; no D3D10 dependency needed |
| 4 | `D3D12ConfigureVirtualMemoryX` | console | present (returns `S_OK`) | VM policy; PC/Vulkan owns it. Measured: one pointer argument (a zeroed 32-byte block, dword +4 set to a count); the title only tests the result for failure |
| 5 | `D3D12CreateDevice` | standard | **absent** | Leonida exports only the Xbox entry point; see §2.3 |
| 6 | `D3D12CreateRootSignatureDeserializer` | standard | present | forwarded |
| 7 | `D3D12CreateVersionedRootSignatureDeserializer` | standard | absent | forward if imported |
| 8 | `D3D12EnableExperimentalFeatures` | standard | absent | forward if imported |
| 9 | `D3D12GetDebugInterface` | standard | absent | forward if imported |
| 10 | `D3D12SerializeRootSignature` | standard | present | forwarded, flags cleared on a copy |
| 11 | `D3D12SerializeVersionedRootSignature` | standard | present | forwarded, flags cleared on a copy |
| 12 | `D3D12UploadCustomMicrocodeX` | console | absent | **the shader seam**: hands pre-compiled microcode to the driver |
| 13 | `D3D12ConsCreateDevice` | console | present as `D3D12XboxCreateDevice` | same call shape; see §2.3 on the name |
| 14 | `D3D12ConsEnableMemoryProfiling` | console | absent | record / no-op |
| 15 | `D3D12ConsGetProcessDebugFlags` | console | absent | record |
| 16 | `D3D12ConsSetProcessDebugFlags` | console | absent | record |
| 17 | `D3D12ConsUnloadDriverModule` | console | absent | no UMD to unload; no-op |
| 18 | `D3DFlushEntireCpuCache` | console | absent | fence-equivalent |
| 19 | `DXGIXGetFrameStatistics` | console present | absent | no PC/Vulkan equivalent |
| 20 | `DXGIXPresentArray` | console present | absent | presents an *array* of swapchains |
| 21 | `DXGIXSetFrameNotification` | console present | absent | frame notification |
| 22 | `DXGIXSetVLineNotification` | console present | absent | scanline/vblank notification |
| 116 | `InternalDriverCallX` | console private | absent | runtime↔UMD escape hatch; nothing to forward to |

The seven standard functions are 5–11. The other sixteen are console-only. Leonida serves four of
the twenty-three today (6, 10, 11, 13). Any function in the table other than those four is an
unimplemented import if the game imports it; the PE loader's dry-run report (`docs/runtime_imports.md`
on the runtime-imports branch) is the place to see it.

### 2.2 The 93 GUID data exports

Ordinals 23–71 and 100–115 are documented per entry in
[`components/extension-wrapper/src/iid_console.inc`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/extension-wrapper/src/iid_console.inc).
Ordinals 23–28 and 72–99 are derived in §2.4. Names are SCARAB's; the 10 console-only GUIDs are in
§4. "Std" marks a GUID that is also defined by the Windows SDK; all 83 were checked in §3. The
order within 23–27 is inferred and immaterial: those are debug GUIDs, not interfaces a title QIs
for.

The 93 are the documented GUID data exports. They do **not** include the four console
command-list IIDs in §4: SCARAB's provenance note says the runtime does not export them, so they
have no ordinal here and are absent from this export dump.

| # | GUID export | Std | Evidence |
|---:|---|---|---|
| 23 | `DXGI_DEBUG_ALL` | yes | derived (debug block, alphabetical) |
| 24 | `DXGI_DEBUG_APP` | yes | derived |
| 25 | `DXGI_DEBUG_D3D11` | yes | derived |
| 26 | `DXGI_DEBUG_DX` | yes | derived |
| 27 | `DXGI_DEBUG_DXGI` | yes | derived |
| 28 | `IID_ID3D10Blob` | yes | derived (only unplaced name; by elimination) |
| 29 | `IID_ID3D11Asynchronous` | yes | documented |
| 30 | `IID_ID3D11BlendState` | yes | documented |
| 31 | `IID_ID3D11BlendState1` | yes | documented |
| 32 | `IID_ID3D11Buffer` | yes | documented |
| 33 | `IID_ID3D11ClassInstance` | yes | documented |
| 34 | `IID_ID3D11ClassLinkage` | yes | documented |
| 35 | `IID_ID3D11CommandList` | yes | documented |
| 36 | `IID_ID3D11ComputeShader` | yes | documented |
| 37 | `IID_ID3D11Counter` | yes | documented |
| 38 | `IID_ID3D11CounterSampleX` | **no** | documented |
| 39 | `IID_ID3D11CounterSetX` | **no** | documented |
| 40 | `IID_ID3D11DepthStencilState` | yes | documented |
| 41 | `IID_ID3D11DepthStencilView` | yes | documented |
| 42 | `IID_ID3D11Device` | yes | documented |
| 43 | `IID_ID3D11Device1` | yes | documented |
| 44 | `IID_ID3D11DeviceChild` | yes | documented |
| 45 | `IID_ID3D11DeviceContext` | yes | documented |
| 46 | `IID_ID3D11DeviceContext1` | yes | documented |
| 47 | `IID_ID3D11DeviceContextX` | **no** | documented |
| 48 | `IID_ID3D11DeviceX` | **no** | documented |
| 49 | `IID_ID3D11DmaEngineContextX` | **no** | documented |
| 50 | `IID_ID3D11DomainShader` | yes | documented |
| 51 | `IID_ID3D11GeometryShader` | yes | documented |
| 52 | `IID_ID3D11HullShader` | yes | documented |
| 53 | `IID_ID3D11InputLayout` | yes | documented |
| 54 | `IID_ID3D11PerformanceContextX` | **no** | documented |
| 55 | `IID_ID3D11PerformanceDeviceX` | **no** | documented |
| 56 | `IID_ID3D11PixelShader` | yes | documented |
| 57 | `IID_ID3D11Predicate` | yes | documented |
| 58 | `IID_ID3D11Query` | yes | documented |
| 59 | `IID_ID3D11RasterizerState` | yes | documented |
| 60 | `IID_ID3D11RasterizerState1` | yes | documented |
| 61 | `IID_ID3D11RenderTargetView` | yes | documented |
| 62 | `IID_ID3D11Resource` | yes | documented |
| 63 | `IID_ID3D11SamplerState` | yes | documented |
| 64 | `IID_ID3D11ShaderResourceView` | yes | documented |
| 65 | `IID_ID3D11Texture1D` | yes | documented |
| 66 | `IID_ID3D11Texture2D` | yes | documented |
| 67 | `IID_ID3D11Texture3D` | yes | documented |
| 68 | `IID_ID3D11UnorderedAccessView` | yes | documented |
| 69 | `IID_ID3D11UserDefinedAnnotationX` | **no** | documented |
| 70 | `IID_ID3D11VertexShader` | yes | documented |
| 71 | `IID_ID3D11View` | yes | documented |
| 72 | `IID_ID3D12CommandAllocator` | yes | derived |
| 73 | `IID_ID3D12CommandList` | yes | derived |
| 74 | `IID_ID3D12CommandQueue` | yes | derived |
| 75 | `IID_ID3D12CommandSignature` | yes | derived |
| 76 | `IID_ID3D12Debug` | yes | derived; **anchored** by `iid_provenance.json` |
| 77 | `IID_ID3D12DescriptorHeap` | yes | derived |
| 78 | `IID_ID3D12Device` | yes | derived |
| 79 | `IID_ID3D12Device1` | yes | derived |
| 80 | `IID_ID3D12Device2` | yes | derived |
| 81 | `IID_ID3D12Device3` | yes | derived |
| 82 | `IID_ID3D12Device4` | yes | derived |
| 83 | `IID_ID3D12DeviceChild` | yes | derived |
| 84 | `IID_ID3D12Fence` | yes | derived |
| 85 | `IID_ID3D12Fence1` | yes | derived |
| 86 | `IID_ID3D12GraphicsCommandList` | yes | derived |
| 87 | `IID_ID3D12GraphicsCommandList1` | yes | derived |
| 88 | `IID_ID3D12GraphicsCommandList2` | yes | derived |
| 89 | `IID_ID3D12Heap` | yes | derived |
| 90 | `IID_ID3D12Object` | yes | derived |
| 91 | `IID_ID3D12Pageable` | yes | derived |
| 92 | `IID_ID3D12PipelineLibrary` | yes | derived |
| 93 | `IID_ID3D12PipelineLibrary1` | yes | derived |
| 94 | `IID_ID3D12PipelineState` | yes | derived |
| 95 | `IID_ID3D12Resource` | yes | derived |
| 96 | `IID_ID3D12RootSignature` | yes | derived |
| 97 | `IID_ID3D12RootSignatureDeserializer` | yes | derived |
| 98 | `IID_ID3D12Tools` | yes | derived |
| 99 | `IID_ID3D12VersionedRootSignatureDeserializer` | yes | derived |
| 100 | `IID_ID3D12ConsDmaCommandList` | **no** | documented; pinned out of alphabetical order |
| 101 | `IID_ID3DDeviceContextState` | yes | documented |
| 102 | `IID_IDXGIAdapter` | yes | documented |
| 103 | `IID_IDXGIAdapter1` | yes | documented |
| 104 | `IID_IDXGIAdapter2` | yes | documented |
| 105 | `IID_IDXGIDevice` | yes | documented |
| 106 | `IID_IDXGIDevice1` | yes | documented |
| 107 | `IID_IDXGIDevice2` | yes | documented |
| 108 | `IID_IDXGIDeviceSubObject` | yes | documented |
| 109 | `IID_IDXGIFactory` | yes | documented |
| 110 | `IID_IDXGIFactory1` | yes | documented |
| 111 | `IID_IDXGIFactory2` | yes | documented |
| 112 | `IID_IDXGIObject` | yes | documented |
| 113 | `IID_IDXGISwapChain` | yes | documented |
| 114 | `IID_IDXGISwapChain1` | yes | documented |
| 115 | `IID_IGraphicsUnknown` | **no** | documented |

### 2.3 The published names are not all wire names

This matters if the table is used to build a replacement DLL. SCARAB deliberately de-brands the
vendor token:

- `tools/gen_def.py` derives the console-only token from the export table, then emits
  `EXPORTS <real-name>=<internal-name>` using `OUR_TOKEN = "Cons"`
  ([`tools/gen_def.py:2-24,38,149-183`](https://github.com/p-portela/SCARAB/blob/d304fa2/tools/gen_def.py)).
  The generated `.def` is **not committed**; it is built from the user-supplied vendor DLL.
- The bridge source defines `D3D12ConsCreateDevice` and `D3D12Cons*` symbols
  ([`d3d12_xs_bridge.cpp:451-475`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/runtime-bridge/src/d3d12_xs_bridge.cpp)).
- The string `Xbox` does not occur anywhere in SCARAB's tree, and the README contains the
  tell-tale "an console Series" — the tree was mechanically de-branded before the first commit.

Therefore `D3D12ConsCreateDevice` (ordinal 13) and the other four `D3D12Cons*` exports are
SCARAB's internal spellings, not evidence of the runtime's export names. Leonida already uses
`D3D12XboxCreateDevice`, which is the public GDK name and the same call shape
(`adapter, params, riid, out`). That is the likely wire name, but it should be confirmed from the
game's import table. The same caveat applies to `IID_ID3D12ConsDmaCommandList`: the GUID in §4 is
the fact; the spelling may be de-branded.

The other 18 function names and all 83 standard GUID names carry no token and can be used as
published.

### 2.4 How ordinals 72–99 were recovered

SCARAB does not publish a per-ordinal map for the 28 standard D3D12 IIDs. The map is derivable
because the runtime's GUID exports behave as if they were ordered alphabetically by name, with one
deliberate exception:

- Sort the 93 GUID export names. `23 + index` matches **58 of the 59** ordinals documented in
  `iid_console.inc`. The single mismatch is `IID_ID3D12ConsDmaCommandList`: documented at 100,
  alphabetically at 76.
- SCARAB's own provenance correction says the IID at the runtime's table slot 47 is
  `IID_ID3D12Debug`, **exported at ordinal 76**
  ([`components/extension-wrapper/iid_provenance.json:2`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/extension-wrapper/iid_provenance.json)).
  Alphabetical placement of the 28 standard names puts `IID_ID3D12Debug` at 76. The anchor and the
  rule agree.
- With `IID_ID3D12ConsDmaCommandList` pinned at 100, the remaining 15 names at 101–115 are again
  alphabetical, and 72–99 are exactly the 28 standard D3D12 IIDs in alphabetical order.
- 23–28 follow by the same rule: 23–27 are the five `DXGI_DEBUG_*` GUIDs and 28 is
  `IID_ID3D10Blob`, the only GUID name not otherwise placed.

Confidence: **Derived**. The 23 function ordinals, the 59 documented IID ordinals, and the 28+10
IID values are **Verified** (the values against the SDK in §3; the console values are **Reported**
by SCARAB). If a real export table ever contradicts 72–99, the derivation is what is wrong, not
the rest of the table.

A second lesson from the same data: the runtime exports IIDs only up to `ID3D12Device4`,
`ID3D12GraphicsCommandList2`, `ID3D12Fence1` and `ID3D12PipelineLibrary1`. That is a property of
the *exported IID table*, not proof that the console device stops at those interface versions —
the shipped `XboxBvhBuilder.dll` names `ID3D12Device11` and `ID3D12GraphicsCommandList4/6/8`
(see [d3d12_vulkan.md](d3d12_vulkan.md) Appendix B). Do not use the absence of newer IIDs from
this table to bound the game's interface use.

## 3. Standard IIDs: 28/28 holds, and 83/83 is the stronger measured fact

SCARAB's headline is exact: the 28 D3D12 `IID_*` values it extracted from the console runtime are
byte-identical to the PC headers. I checked all 28 against the Windows SDK 10.0.26100.0 on this
machine: 27 are in `um/d3d12.h` and `IID_ID3D12Debug` is in `um/d3d12sdklayers.h`; all 28 match
byte-for-byte. SCARAB's claim that they match "the PC Agility SDK headers" is therefore true, with
the nuance that one of the 28 is a debug-layer IID, not a `d3d12.h` IID.

The check extends further. SCARAB's `iid_table.cpp` also reproduces 49 standard D3D11/DXGI IIDs,
`IID_ID3D10Blob`, and 5 `DXGI_DEBUG_*` GUIDs. All **78 standard IIDs + 5 debug GUIDs = 83/83**
match the SDK headers (`d3d11*.h`, `dxgi*.h`, `dxgi1_*.h`, `d3dcommon.h`, `dxgidebug.h`,
`d3d11sdklayers.h`, `d3d12sdklayers.h`). The 10 console-only GUIDs are absent from the SDK.

The chain of evidence has two links, and only one is checkable here. This page checks
**SCARAB's table against the SDK**. SCARAB's gate G1 checks **its table against the vendor
runtime** (`93/93` GUID bytes); that gate needs the vendor DLL and was not run here. So the
correct statement is: SCARAB's table is exactly the PC SDK's standard GUID set, and SCARAB reports
that the table is byte-identical to the runtime's. The second link is **Reported**, not verified
in this session.

Reproduce the first link:

```
python tools/d3d12_xs_surface.py --check --sdk
python tools/d3d12_xs_surface.py --check --sdk --scarab <SCARAB clone>
python -I tests/test_d3d12_xs_surface.py
```

The first command checks the 116-slot table and the 83 standard GUIDs against the installed SDK.
The second also compares all 93 values and the 59 documented ordinals against a SCARAB checkout.
The third is the standalone test; `pytest tests/test_d3d12_xs_surface.py` works where pytest is
installed.

Consequences for Leonida:

- A QI for any standard interface can be answered with the SDK GUID. No translation table is
  needed, and a captured GUID can be looked up directly in the SDK headers.
- The 28-IID claim understates the evidence. If the game uses D3D11 on the console — SCARAB's
  open question 1 — the existing D3D11/DXGI IIDs are the PC ones too. Leonida still lists D3D11
  as a non-goal, so this is information, not a work item.
- The GUID values are facts about the runtime; for the 83 standard GUIDs, the *names* come from
  the SDK and are not proprietary.

The 28 D3D12 values, for reference (all verified):

| IID | GUID |
|---|---|
| `IID_ID3D12CommandAllocator` | `6102dee4-af59-4b09-b999-b44d73f09b24` |
| `IID_ID3D12CommandList` | `7116d91c-e7e4-47ce-b8c6-ec8168f437e5` |
| `IID_ID3D12CommandQueue` | `0ec870a6-5d7e-4c22-8cfc-5baae07616ed` |
| `IID_ID3D12CommandSignature` | `c36a797c-ec80-4f0a-8985-a7b2475082d1` |
| `IID_ID3D12Debug` | `344488b7-6846-474b-b989-f027448245e0` |
| `IID_ID3D12DescriptorHeap` | `8efb471d-616c-4f49-90f7-127bb763fa51` |
| `IID_ID3D12Device` | `189819f1-1db6-4b57-be54-1821339b85f7` |
| `IID_ID3D12Device1` | `77acce80-638e-4e65-8895-c1f23386863e` |
| `IID_ID3D12Device2` | `30baa41e-b15b-475c-a0bb-1af5c5b64328` |
| `IID_ID3D12Device3` | `81dadc15-2bad-4392-93c5-101345c4aa98` |
| `IID_ID3D12Device4` | `e865df17-a9ee-46f9-a463-3098315aa2e5` |
| `IID_ID3D12DeviceChild` | `905db94b-a00c-4140-9df5-2b64ca9ea357` |
| `IID_ID3D12Fence` | `0a753dcf-c4d8-4b91-adf6-be5a60d95a76` |
| `IID_ID3D12Fence1` | `433685fe-e22b-4ca0-a8db-b5b4f4dd0e4a` |
| `IID_ID3D12GraphicsCommandList` | `5b160d0f-ac1b-4185-8ba8-b3ae42a5a455` |
| `IID_ID3D12GraphicsCommandList1` | `553103fb-1fe7-4557-bb38-946d7d0e7ca7` |
| `IID_ID3D12GraphicsCommandList2` | `38c3e585-ff17-412c-9150-4fc6f9d72a28` |
| `IID_ID3D12Heap` | `6b3b2502-6e51-45b3-90ee-9884265e8df3` |
| `IID_ID3D12Object` | `c4fec28f-7966-4e95-9f94-f431cb56c3b8` |
| `IID_ID3D12Pageable` | `63ee58fb-1268-4835-86da-f008ce62f0d6` |
| `IID_ID3D12PipelineLibrary` | `c64226a8-9201-46af-b4cc-53fb9ff7414f` |
| `IID_ID3D12PipelineLibrary1` | `80eabf42-2568-4e5e-bd82-c37f86961dc3` |
| `IID_ID3D12PipelineState` | `765a30f3-f624-4c6f-a828-ace948622445` |
| `IID_ID3D12Resource` | `696442be-a72e-4059-bc79-5b5c98040fad` |
| `IID_ID3D12RootSignature` | `c54a6b66-72df-4ee8-8be5-a946a1429214` |
| `IID_ID3D12RootSignatureDeserializer` | `34ab647b-3cc8-46ac-841b-c0965645c046` |
| `IID_ID3D12Tools` | `7071e1f0-e84b-4b33-974f-12fa49de65c5` |
| `IID_ID3D12VersionedRootSignatureDeserializer` | `7f91ce67-090c-4bb7-b78e-ed8ff2e31da0` |

## 4. Console-only IIDs

Ten IIDs exported by the runtime are absent from the Windows SDK. Values are **Reported** by
SCARAB from the vendor runtime's export table; the ordinal is from `iid_console.inc`, so it is
**Verified** against SCARAB's file, not against the vendor binary.

| Ordinal | IID | GUID | What it is | Leonida relevance |
|---:|---|---|---|---|
| 100 | `IID_ID3D12ConsDmaCommandList` | `30bc77b9-8379-4b25-95de-b8588c42275d` | DMA command-list interface | **High.** If the game creates a DMA list, this is the IID it passes to `CreateCommandList` |
| 115 | `IID_IGraphicsUnknown` | `aceeea63-e0a9-4a1c-bbec-71b2f485f758` | unknown purpose | Do not invent a vtable; log and refuse |
| 48 | `IID_ID3D11DeviceX` | `177700f9-876a-4436-b368-36a604f82cef` | D3D11 device extension | Out of scope (D3D11 is a non-goal) |
| 47 | `IID_ID3D11DeviceContextX` | `48800095-7134-4be7-9186-b86becb26477` | D3D11 context extension | Out of scope |
| 49 | `IID_ID3D11DmaEngineContextX` | `a6332ddb-8e02-427d-b0b7-34a1e61a6488` | D3D11 DMA engine | Out of scope |
| 54 | `IID_ID3D11PerformanceContextX` | `9458fe06-c78d-47f7-96a0-ec7b727be1e9` | D3D11 performance | Out of scope |
| 55 | `IID_ID3D11PerformanceDeviceX` | `88671610-712e-4f1e-84ab-01b5948bd373` | D3D11 performance | Out of scope |
| 69 | `IID_ID3D11UserDefinedAnnotationX` | `b2daad8b-03d4-4dbf-95eb-32ab4b63d0ab` | D3D11 annotation | Out of scope |
| 39 | `IID_ID3D11CounterSetX` | `f9f6a732-c5f8-4a57-ba36-eae7c27ba74a` | D3D11 counters | Out of scope |
| 38 | `IID_ID3D11CounterSampleX` | `806ceee4-280b-43d9-9008-b4fdf7febee2` | D3D11 counters | Out of scope |

The four command-list IIDs are **not runtime exports** according to SCARAB's provenance, so they
have no row above and no ordinal:
`ID3D12ConsComputeCommandList`, `...6`, `...8`, and `ID3D12ConsCopyCommandList6`. SCARAB's
provenance note says they are "NOT exported by the runtime and are NOT recovered" and that a
`QueryInterface` for one is logged by the wrapper's audit hook
([`iid_provenance.json:13`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/extension-wrapper/iid_provenance.json)).
This 1.0.2.1 package's own DLL export tables do not contain them either: its root has the 16
middleware/CRT DLLs and no D3D12 runtime DLL, its layout XML lists no `d3d12_xs.dll`, the
executable that would call `QueryInterface` with them is absent, and `gameos.xvd` is an encrypted
XVD with a high-entropy payload and no plaintext `d3d12_xs.dll` name in ASCII or UTF-16 (measured
in [d3d12_command_objects.md](d3d12_command_objects.md) §1 and
[d3d12_vulkan.md](d3d12_vulkan.md) Appendix A.1). SCARAB also holds seven candidate v4 UUIDs
found 16-byte-aligned after `IID_ID3D12Device4` in the UMD's `.rdata` and marks them `CANDIDATE`,
not used. Do not use them or assign them by position. A QI for one of those interfaces must be
logged and refused until a title trace or another artifact supplies the GUID.

The DMA IID is the load-bearing one. SCARAB's end-to-end test created a console DMA command list
through a PC D3D12 device and got an object that served both `IID_ID3D12ConsDmaCommandList` and
`IID_ID3D12GraphicsCommandList`
([`docs/d3d12-xs-wrapper.md:134-162`](https://github.com/p-portela/SCARAB/blob/d304fa2/docs/d3d12-xs-wrapper.md)).
That is **Reported** (Wine, no GPU). It proves the wrapper's QI routing, not the game's use of DMA
lists.

## 5. The extension method surface

SCARAB's catalogue has 105 rows labelled `method` and 106 labelled `enum`
([`components/extension-wrapper/docs/console-extension-catalogue.tsv`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/extension-wrapper/docs/console-extension-catalogue.tsv)).
The names were mined from the UMD's validation strings. Read the catalogue as a *name list for a
trace filter*, not as an ABI:

- **At least 11 of the 105 "method" rows are not interface methods**: `UINT_MAX`,
  `UINT64_MAX`, `DXGI_DEBUG_DX`, `D3D12_MESSAGE_ID_INVALID_NODE_INDEX`,
  `D3D12CONS_PRESENT_PLANES_MAX`, `D3D12CONS_PRESENT_VIEWS_MAX` (both also in the enum list),
  `UMDInternalDriverCallX` (a UMD export), `WorldRayOriginX`, `WorldRayDirectionX`,
  `DispatchRaysDimensionsX` (DXR builtin names), and `TopLeftX` (not a method-shaped name). That
  leaves at most 94 plausible methods, which is consistent with the wrapper document's own "~95".
- The counts drift between files: 105/106 in the TSV and the wrapper doc; 104/108 in
  `console_extensions.h:8`; "60 console-specific GUIDs" in `wrap.cpp:84` against 59 entries in
  `iid_console.inc`. None of these is a measurement of the ABI; they are documentation drift.
- Only 10 methods are class-attributed by a UMD message naming the interface, and 6 more by call
  sites in the console SDK samples
  ([`docs/d3d12-xs-wrapper.md:69-97`](https://github.com/p-portela/SCARAB/blob/d304fa2/docs/d3d12-xs-wrapper.md)).
  The rest are unattributed, and the **order within the appended block is not recovered**.
- SCARAB's own honesty rule is right and worth keeping: every appended slot logs its name and raw
  arguments, and unknown slots log their index, so a wrong order is diagnosable.

SCARAB's wrapper class derives from the genuine `ID3D12Device4` and
`ID3D12GraphicsCommandList6` interfaces and marks every forwarded method `override`, so the
compiler proves **53 device and 76 command-list** signatures against the real `d3d12.h`
([`docs/d3d12-xs-wrapper.md:53-63`](https://github.com/p-portela/SCARAB/blob/d304fa2/docs/d3d12-xs-wrapper.md)).
That proves the PC-side vtable layout of its wrapper; it says nothing about the appended console
block, which is the part Leonida must not copy.

Class placement that is evidence-backed:

| Class | Methods | Evidence |
|---|---|---|
| device | `CreateCommandQueueX`, `GetFrameStatisticsX`, `ScheduleFrameEventX`, `SetFrameIntervalX`, `SetDebugErrorFilterX`, `WaitFrameEventX` | UMD messages name the interface; the samples call `SetFrameIntervalX` 122, `ScheduleFrameEventX` 122 and `WaitFrameEventX` 121 times on the device |
| queue | `PresentX`, `SuspendX`, `ResumeX` | UMD names `ID3D12CommandQueue::PresentX`; the samples call `PresentX` 367, `SuspendX` 122, `ResumeX` 122 times on the queue |
| DMA list | `FillRawBuffer`, `CopyResource` | UMD names `ID3D12ConsDmaCommandList::FillRawBuffer` |
| graphics list | `RSSetDepthBiasX`, `OMSetSampleMaskX`, `OMSetFrontAndBackStencilRef`, `FillMemoryWithPattern` | UMD messages |
| unattributed | the other ~90 names | strings only |

**Do not copy the appended order.** Leonida implements the public vtable chain; an unknown X
method should be logged with its slot and arguments and refused, exactly as SCARAB does, until the
game's trace identifies it.

### 5.1 Some "console extensions" are public D3D12 methods

This is the most useful correction in the whole comparison. SCARAB's wrapper derives from
`ID3D12GraphicsCommandList6`, so it treats anything added to D3D12 later as a console extension.
The Windows SDK 10.0.26100.0 has since promoted several of these names:

| SCARAB catalogue name | Public method in SDK 10.0.26100.0 | Consequence |
|---|---|---|
| `OMSetFrontAndBackStencilRef` | `ID3D12GraphicsCommandList8::OMSetFrontAndBackStencilRef(UINT, UINT)` (`d3d12.h:32819`) | identical signature; it is the GCL8 slot, not an appended slot |
| `RSSetDepthBiasX` | `ID3D12GraphicsCommandList9::RSSetDepthBias(FLOAT, FLOAT, FLOAT)` (`d3d12.h:33680`) | SCARAB declares `INT` for the first argument and says "no command-list state to set"; both are wrong for the 26100 header |
| `OMSetDepthBoundsX` | `ID3D12GraphicsCommandList1::OMSetDepthBounds(FLOAT, FLOAT)` (`d3d12.h:7237`) | a base-D3D12 method; the X variant may add something, but "no PC equivalent" needs proof |
| `RSSetShadingRateX` | `ID3D12GraphicsCommandList5::RSSetShadingRate(...)` (`d3d12.h:30278`) | same |
| `DispatchMeshX` | `ID3D12GraphicsCommandList6::DispatchMesh(UINT, UINT, UINT)` (`d3d12.h:31126`) | same |

The lesson is not that the catalogue is useless; it is that a name with an `X` suffix is not proof
of a console-only method. Check every name against the SDK version the game was compiled against
before treating it as new. For `OMSetFrontAndBackStencilRef` the signature is identical, so a
GCL8-compiled game calls it at the public slot; SCARAB's placement after GCL6 would land in the
wrong slot. Leonida's public-chain command list already has it at GCL8.

## 6. Command lists and `CreateCommandList`

The one measured console command-list rule, quoted from the UMD:

> `CreateCommandList: An riid of __uuidof(ID3D12ConsDmaCommandList) is required for
> D3D12CONS_COMMAND_LIST_TYPE_DMA.`

SCARAB records `D3D12CONS_COMMAND_LIST_TYPE_DMA = 6`
([`console_extensions.h:45-47`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/extension-wrapper/console_extensions.h)).
The *name* is evidence-backed; the value 6 is **Derived** from appending to the PC enum (0–5), not
measured. It is a strong inference, but treat it as a hypothesis until the game's
`CreateCommandList` call is traced.

SCARAB's interception logic
([`wrap.cpp:279-326`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/extension-wrapper/src/wrap.cpp)):

- if `type == 6`, map it to PC `D3D12_COMMAND_LIST_TYPE_COPY` and require the DMA IID;
- otherwise create the list normally;
- prefer `CreateCommandList1` and `Reset` to apply the initial state;
- return a wrapper that serves the DMA IID only for DMA lists.

For Leonida, the transferable facts are the rule and the IID, not the wrapper:

- On `CreateCommandList`, log `(nodeMask, type, riid)` once. If `type` is outside 0–5, or `riid` is
  not a standard GCL IID, treat it as a console list.
- If `type == 6` **and** `riid == 30bc77b9-8379-4b25-95de-b8588c42275d`, implement it as a copy
  list (Vulkan transfer queue / transfer-only command buffer). If the IID is anything else, log
  the GUID and refuse; SCARAB reports that the compute/copy console IIDs are not runtime exports
  and they are not known.
- `D3D12CONS_COMMAND_LIST_FLAG_ENABLE_IB_CHAINING` is a recovered name (value unknown). A console
  list flag must not be passed to `vkCmd...` as if it were a D3D12 flag.
- The console also has `ID3D12ConsComputeCommandList` (and 6/8 variants) and
  `ID3D12ConsCopyCommandList6`, plus `D3D12ConsRehydrateRaytracingAccelerationStructure`
  ([`docs/d3d12-xs-bridge.md:239-245`](https://github.com/p-portela/SCARAB/blob/d304fa2/docs/d3d12-xs-bridge.md)).
  Their IIDs are not runtime exports according to SCARAB, so this package's export tables cannot
  supply them; the values and the layouts are **Open**.

What this does not tell us: the slot of `CreateCommandList` in the console device vtable (Leonida
already measures device slot 14/62/76 independently), the real numeric value of the DMA type, or
whether the game uses DMA lists at all. It is a conditional branch to add to the trace, not a
proven path.

## 7. Presentation: `PresentX`, `DXGIX*`, frame events

This is where the expectation that SCARAB closes Leonida's §7 list is **not** met. SCARAB gives
names and constraints, not signatures, slots or a working present path.

What is evidence-backed:

- `PresentX` is an `ID3D12CommandQueue` method. The UMD message is quoted as
  `ID3D12CommandQueue::PresentX - pPlaneParameters[%d].ResourceCount must currently be less than
  or equal to 2`, and the console SDK samples call it on the queue (367 sites)
  ([`docs/d3d12-xs-wrapper.md:79-97`](https://github.com/p-portela/SCARAB/blob/d304fa2/docs/d3d12-xs-wrapper.md)).
- Present flags are named `D3D12CONS_PRESENT_FLAG_NONE`, `_IMMEDIATE`, `_PERSIST`,
  `_UNIQUE_VIEW_RESOURCES`; `D3D12CONS_PRESENT_PLANES_MAX = 2` and
  `D3D12CONS_PRESENT_VIEWS_MAX = 8`
  ([`console_extensions.h:62-72`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/extension-wrapper/console_extensions.h)).
  The **values are not measured**: `console_extensions.h` assigns 0x1/0x2/0x4, while
  `present_adapter.h:38-44` says the values are unknown and sets them all to 0. The two headers
  contradict each other; use neither as an ABI.
- `DXGIXPresentArray`, `DXGIXSetFrameNotification`, `DXGIXSetVLineNotification` and
  `DXGIXGetFrameStatistics` are **free functions** at ordinals 20, 21, 22 and 19 — not queue
  methods. SCARAB's bridge implements them as recorded no-ops; `DXGIXGetFrameStatistics` zeroes
  64 bytes ([`d3d12_xs_bridge.cpp:520-551`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/runtime-bridge/src/d3d12_xs_bridge.cpp)).
  No parameter layouts are recovered.
- Device-side frame pacing names: `SetFrameIntervalX`, `ScheduleFrameEventX`, `WaitFrameEventX`,
  `GetFrameStatisticsX`. The first three are called on the device in the samples; the enum names
  `D3D12CONS_FRAME_EVENT_ORIGIN` and `D3D12CONS_SCHEDULE_FRAME_EVENT_FLAG_DISABLE_SIGNAL` exist.
  Queue-side `SuspendX`/`ResumeX` exist. No signatures or slots.
- SCARAB's `present-adapter` is a seam only: its header proposes a swapchain registry and
  `D3DKMTWaitForVerticalBlankEvent` for vblank, and its flag enum is all zeroes
  ([`present_adapter.h:11-66`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/present-adapter/include/present_adapter.h)).
  Nothing is implemented.

Public samples add the call shapes (not slots). From MIT/zlib code
([Xbox-GDK-Samples](https://github.com/microsoft/Xbox-GDK-Samples) SimpleDeviceAndSwapChain and FramePacing,
[directxtk12modelviewer](https://github.com/walbourn/directxtk12modelviewer), SDL2's Xbox D3D12 backend), as
collected in [d3d12_vulkan.md §4.5](d3d12_vulkan.md):

- `PresentX(planeCount, planes, params)` on the queue, 3 arguments; plane fields `Token`, `ResourceCount`,
  `ppResources`, `pSrcViewRects`, `ColorSpace`; 2 planes for HDR + GameDVR; `params` carries
  `ImmediateThresholdPercent`. `SuspendX(0)` and `ResumeX()` on the queue.
- On the device: `SetFrameIntervalX` (4 args), `ScheduleFrameEventX` (4), `WaitFrameEventX` (5, returns a frame
  token), `GetFrameStatisticsX` (4). Back buffers are committed resources with `ALLOW_DISPLAY`; DXGI is used
  only for `IDXGIDevice1`→`GetAdapter`→`EnumOutputs`. No DXGI swapchain.

Vtable slots and struct layouts are **not** unmeasured any more: public, licensed code and Microsoft's public
reference give most of them. Enum values are still mostly **unmeasured**.

§8.2 question 6 ("DXGI, `PresentX`, or both") is **partly answered**: public GDKX code presents through
`PresentX` and frame events only. That the game does the same is a **hypothesis**; the game is not observed.
The call site must still be found by tracing the queue vtable beyond the public chain, not by guessing a
slot. The plan in [d3d12_vulkan.md §4.5](d3d12_vulkan.md) is now a `PresentX` + frame-event presenter first,
with DXGI factory/adapter/output enumeration kept and the DXGI swapchain as fallback.

The `DXGIX*` free functions are a separate question. If the game imports them from `d3d12_xs.dll`,
Leonida's loader will report them as unimplemented; a Vulkan present-array implementation would
need a swapchain registry, which does not exist yet. SCARAB's no-op behavior is not acceptable for
Leonida because it would silently present nothing.

### 7.1 Public sources for the presentation and device ABI

Microsoft's public GDK graphics reference does not document the console interfaces. The
[graphics index](https://docs.xbox.com/_llms/en/apis/gdk-apis/graphics-api-reference.txt) lists 528 pages,
8 of them under `d3d12_xs` (six enums, `D3D10CreateBlob`, `DXGI_ADAPTER_DESC2`); the `ID3D12CommandQueue`
page lists the PC methods only, and there is no `PresentX`, `SetFrameIntervalX` or `D3D12XboxCreateDevice`
page, while the `D3D12_ROOT_SIGNATURE_FLAGS` pages return 404. The slots and layouts below therefore come from public code
that is not Microsoft documentation: XCL and the Microsoft samples measure other titles' runtimes, ebiten and
Mesa are public code compiled against the GDK headers, and docs.xbox.com is Microsoft's reference.

| Item | Public source and value | Verdict on this repository's provisional value |
|---|---|---|
| Queue `PresentX` slot | `ivall/xcl` (GPL-2.0-only; facts only) dispatches `PresentX(planeCount, planes, params)` at queue slot **30** in the older runtime layout (XDefiant, Minecraft, Fortnite) and slot **32** in a newer one (Unicorn Overlord), with the same SDK version but two methods inserted before `Signal` ([`present.rs`](https://github.com/ivall/xcl/blob/95bda116cfc1b118156901984052ff5698dbf4cc/crates/xcl/src/hle/d3d12x/present.rs), [`real.rs`](https://github.com/ivall/xcl/blob/95bda116cfc1b118156901984052ff5698dbf4cc/crates/xcl/src/hle/d3d12x/real.rs), [`HANDOFF.md`](https://github.com/ivall/xcl/blob/95bda116cfc1b118156901984052ff5698dbf4cc/HANDOFF.md)). | **Confirms a slot, not one number.** The version moves it; trace the game's runtime, do not hard-code either value. |
| Queue `SuspendX` / `ResumeX` slots | No public slot in XCL or the samples. The signatures are `SuspendX(UINT flags)` and `ResumeX()` from the Apache-2.0 ebiten C++ binding ([`api_microsoftgdk_windows.go`](https://github.com/hajimehoshi/ebiten/blob/d438df689b19544c82543ad522b36be86a5a60b7/internal/graphicsdriver/directx/api_microsoftgdk_windows.go)); the samples call `SuspendX(0)` and `ResumeX()`. | **Leaves provisional.** |
| `D3D12XBOX_PRESENT_PLANE_PARAMETERS` | 0x48 bytes on x64: `Token` u64 @0x00, `ResourceCount` u32 @0x08, `ppResources` @0x10, `pSrcViewRects` @0x18, `pDestPlacementBase` @0x20 (four floats), `ColorSpace` int32 @0x28, `ScaleFilter` int32 @0x2C, `ExtendedDescCount` u32 @0x30, `pExtendedDescs` @0x38, `Flags` u32 @0x40. ebiten's Go mirror of the GDK struct ([`d3d12_windows.go`](https://github.com/hajimehoshi/ebiten/blob/d438df689b19544c82543ad522b36be86a5a60b7/internal/graphicsdriver/directx/d3d12_windows.go)); XCL independently reads token @0, count @8 and resource list @0x10 ([`present.rs`](https://github.com/ivall/xcl/blob/95bda116cfc1b118156901984052ff5698dbf4cc/crates/xcl/src/hle/d3d12x/present.rs)). | **Confirms** the field order, types and size in `xbox_present_types.h` and `xbox_present.h`. `ScaleFilter` and `Flags` values stay ours. |
| `D3D12XBOX_PRESENT_PARAMETERS` | 0x20 bytes on x64: `ImmediateThresholdPercent` float @0x00, `ViewCount` u32 @0x04, `ExtendedDescCount` u32 @0x08, `pExtendedDescs` @0x10, `Flags` u32 @0x18. Mesa's MIT Xbox WGL backend sets `Flags = D3D12XBOX_PRESENT_FLAG_IMMEDIATE` for an immediate present ([`d3d12_wgl_framebuffer_xbox.cpp`](https://github.com/openbsd/xenocara/blob/f3893fcf57cec33c2bd4275edd8ca6d865e3adf2/lib/mesa/src/gallium/winsys/d3d12/wgl/d3d12_wgl_framebuffer_xbox.cpp)); ebiten has the field order. The flag's numeric value is not public. | **Corrects** "at least this field": the repository's one-float struct is the first 4 bytes of a 0x20-byte object. Safe as a pointer prefix; never size a copy by it. |
| Device frame calls | Slots **78** `SetFrameIntervalX`, **79** `ScheduleFrameEventX`, **80** `WaitFrameEventX`, **81** `GetFrameStatisticsX` (0-based, `QueryInterface` = 0). XCL's [`frame.rs`](https://github.com/ivall/xcl/blob/95bda116cfc1b118156901984052ff5698dbf4cc/crates/xcl/src/hle/d3d12x/frame.rs) numbers the calls and documents the arguments; ebiten's `ID3D12Device` vtable has `SetFrameIntervalX`/`ScheduleFrameEventX`/`WaitFrameEventX` at 78/79/80 and leaves 81 unnamed ([`d3d12_windows.go`](https://github.com/hajimehoshi/ebiten/blob/d438df689b19544c82543ad522b36be86a5a60b7/internal/graphicsdriver/directx/d3d12_windows.go)). | **Confirms** all four slots, two independent public sources. |
| `D3D12XBOX_FRAME_STATISTICS` | The MIT FramePacing sample names `Type`, `Interval`, `Event`, `Render`/`Present`/`Display` with `PlaneIndex`, and `Display.PercentScanned`; it treats `S_FALSE` or a count below the request as "not available yet" ([`FramePacing.cpp`](https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/FramePacing/FramePacing.cpp)). XCL's implementation writes `Type` at +0, origin at +0x10 and flip at +0x18, so its record is at least 0x20 bytes ([`frame.rs`](https://github.com/ivall/xcl/blob/95bda116cfc1b118156901984052ff5698dbf4cc/crates/xcl/src/hle/d3d12x/frame.rs)). | **Partly confirms** the field meanings and the two timestamp offsets; the full record layout and the union members stay unmeasured. |
| Token and timeout | `D3D12XBOX_FRAME_PIPELINE_TOKEN` is a 64-bit integer with 0 as `..._NULL` (ebiten). Microsoft's Game Streaming page says `WaitFrameEventX` returns the token at frame start and `PresentX` sends it with the frame to the client ([Learn](https://learn.microsoft.com/en-us/gaming/gdk/docs/features/common/game-streaming/game-streaming-latency-compensation-deep-dive)). XCL's implementation issues 48-bit nonzero tokens (mask `0xFFFF_FFFF_FFFF`), i.e. it leaves the top 16 bits unused, returns `S_FALSE` on a finite timeout, writes the token on timeout only with flag 2, and returns `E_INVALIDARG` for an unscheduled event; `ScheduleFrameEventX` returns `S_FALSE` or `0x10000000` when it starts the clock. | **Confirms** the 64-bit storage and the `0` null. **Contradicts** this layer's `HRESULT_FROM_WIN32(WAIT_TIMEOUT)` on timeout (XCL: `S_FALSE`) and `ERROR_INVALID_STATE` for an unscheduled event (XCL: `E_INVALIDARG`); both are recorded here, not changed in this PR. The 48-bit range is XCL's implementation choice, not a documented mask. |
| `D3D12XBOX_CREATE_DEVICE_PARAMETERS` | Public field names and (for ebiten's version) x64 offsets: `Version` @0, `ProcessDebugFlags` @4, `GraphicsCommandQueueRingSizeBytes` @8, `pOffchipTessellationBuffer` @0x10, `GraphicsScratchMemorySizeBytes` @0x18, `ComputeScratchMemorySizeBytes` @0x1C, `DisableGeometryShaderAllocations` @0x20, `DisableTessellationShaderAllocations` @0x24, `DisableDXR` @0x28, `DisableAutomaticDPBBBreakBatchEvents` @0x2C, `pDXRStackBuffer` @0x30, `DXRStackBufferOverrideSizeBytes` @0x38, `CreateDeviceFlags` @0x3C, `AutoHDRPaperWhiteLevelNits` @0x40, `DisableAutomaticCommandSegmentChaining` @0x44 ([`d3d12_windows.go`](https://github.com/hajimehoshi/ebiten/blob/d438df689b19544c82543ad522b36be86a5a60b7/internal/graphicsdriver/directx/d3d12_windows.go)). The MIT samples add `AmplificationShaderIndirectArgsBufferSize` and `AmplificationShaderPayloadBufferSize` by name ([`SimpleDeviceAndSwapChain.cpp`](https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleDeviceAndSwapChain/SimpleDeviceAndSwapChain.cpp)). | **Partly confirms.** Seventeen named fields cover the first 18 words (word 3 is the x64 alignment pad at 0x0C); the samples name two more fields without a public layout. A 24-word (96-byte) block still has at least four words with no public name or meaning. |
| Root-signature flags `0x03000000` | Not documented: the [graphics index](https://docs.xbox.com/_llms/en/apis/gdk-apis/graphics-api-reference.txt) has no `D3D12_ROOT_SIGNATURE_FLAGS` entry, and both URL forms under `docs.xbox.com/reference/graphics/` (`d3d12` and `d3d12_xs`) return 404. | **Leaves provisional.** Keep masking the byte on a copy; no public bit meaning. |
| `CheckFeatureSupport(D3D12_OPTIONS)` with 0x40 bytes | Microsoft publishes the 15-field PC structure, 0x3C bytes, with no extra member ([`D3D12_FEATURE_DATA_D3D12_OPTIONS`](https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_feature_data_d3d12_options_public.md)); its `CheckFeatureSupport` page says a size mismatch is `E_INVALIDARG` ([`CheckFeatureSupport`](https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device/methods/id3d12device_checkfeaturesupport_public.md)). XCL measured XDefiant asking 0x40 and retries with 0x3C, zeroing the tail ([`HANDOFF.md`](https://github.com/ivall/xcl/blob/95bda116cfc1b118156901984052ff5698dbf4cc/HANDOFF.md), [`real.rs`](https://github.com/ivall/xcl/blob/95bda116cfc1b118156901984052ff5698dbf4cc/crates/xcl/src/hle/d3d12x/real.rs)). | **Confirms the 0x40 request size; leaves the extra 4 bytes unknown.** The measurement fixes the size, not the field. No public source names them; `MaxGPUVirtualAddressBitsPerProcess` is a member of the separate `D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT`, not a published member of `D3D12_OPTIONS`. The repository's accept-and-zero-tail behaviour matches the observed failure and is safe. |

The call shapes are the samples' (and XCL's) shapes: `SetFrameIntervalX(output, interval_us, period, flags)`,
`ScheduleFrameEventX(type, offset_us, ancillary, flags)`,
`WaitFrameEventX(type, timeout_ms, ancillary, flags, &token)`,
`GetFrameStatisticsX(token, type_mask, &count, stats)`, and `PresentX(plane_count, planes, params)`. Type 0 is
the frame origin (the samples use only it); XCL also uses 1 and 2 for later points in the frame. The
`D3D12XBOX_SCHEDULE_FRAME_OBJECT_LIST` / `D3D12XBOX_WAIT_FRAME_OBJECT_LIST` ancillary block is
`{ u32 count; HANDLE* objects @0x08; i32* signaled @0x10 }` per XCL.

The meanings the public samples give the create-device words: `Version` is the SDK version (`D3D12_SDK_VERSION`
in `SimpleDeviceAndSwapChain`), `GraphicsCommandQueueRingSizeBytes` is the command-queue ring,
`GraphicsScratchMemorySizeBytes` and `ComputeScratchMemorySizeBytes` are shader scratch memory,
`D3D12XBOX_DEFAULT_SIZE_BYTES` asks for the runtime's default, the `Disable*` BOOLs save title memory when the
stage is unused, and `CreateDeviceFlags` is `D3D12XBOX_CREATE_DEVICE_FLAG_NONE` in the sample. XCL logs word 0
as the SDK version. ebiten also gives the `D3D12XBOX_PROCESS_DEBUG_FLAG_*` values (`DEBUG` 1, `INSTRUMENTED`
0x00100000, `NULL_GPU` 0x00400000, `ENABLE_COMMON_STATE_PROMOTION` 0x01000000, `ENHANCED_VALIDATION`
0x02000000); these are process debug flags, not `CreateDeviceFlags` bits. The `0x03000000` root-signature bit
is separate and has no public meaning.

The provisional layouts in the split `xbox_present_types.h` on the PresentX-capture branch (not yet on main) are
the same as `xbox_present.h`: the plane struct agrees byte for byte with the 0x48 layout above, the
present-parameter struct is the 4-byte prefix of the 0x20 console object, and `uint64_t Token` is the confirmed
width. Its colour-space values are the documented DXGI numbers; `ScaleFilter` and the plane flags are still ours.
The 60/30/120 Hz periods (`16667`/`33333`/`8333` us) are the public ones; this layer does not carry 24/40/50 Hz.

Still unknown after this search: the queue slots of `SuspendX`/`ResumeX`; which of the two queue layouts the
game's runtime uses; the values of `ScaleFilter` and the plane flags; the full `D3D12XBOX_FRAME_STATISTICS`
layout (the samples name `Type`, `Interval`, `Event`, `Render`/`Present`/`Display.PlaneIndex` and
`Display.PercentScanned`, but no public byte layout); the four extra `D3D12_OPTIONS` bytes; the last words of
the create-device block; the `0x03000000` meaning; and the console's behaviour for a null, repeated or
out-of-order token.

### 7.2 What the public samples give Bink, XAudio2 and FidelityFX

- **FidelityFX.** The MIT `Kits/AMDTK` GDK backend includes `<xmem.h>`, calls
  `XMemGetWorkingSetStatistics(XMEM_WORKING_SET_TITLE, ...)` for its memory report, and on Scarlett creates
  resources with `CreateCommittedResourceX` and a `D3D12XBOX_RESOURCE_DESC` carrying DCC options
  ([`ffx_gdk.cpp`](https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Kits/AMDTK/fidelityfx/src/backends/gdk/ffx_gdk.cpp)).
  Its frame-interpolation swapchain calls the title's `PresentX` callback with `framePipelineTokenOriginal`
  and `framePipelineTokenToSubmit`; the FSR3 sample can pass `D3D12XBOX_FRAME_PIPELINE_TOKEN_NULL` for the
  real frame ([`ffx_gdk.h`](https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Kits/AMDTK/fidelityfx/include/FidelityFX/host/backends/gdk/ffx_gdk.h),
  [`DeviceResources.cpp`](https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/AMDFidelityFX_FSR3FrameInterpolation/DeviceResources.cpp)).
  This matches the three `xmem` imports of the shipped `amd_fidelityfx_upscaler_xbox.dll`; only the
  allocation/capability part transfers, because that DLL is an upscaler and the public sample is the
  frame-interpolation swapchain.
- **XAudio2.** `SimplePlaySound` calls `XAudio2Create(&pXAudio2, 0)` in `InitializeXAudio`, after the device
  resources are made, and sets `XAUDIO2_DEBUG_CONFIGURATION` only in debug. XMA2 playback needs the compressed
  audio in APU memory: `ApuAlloc(&m_xmaMemory, nullptr, bytes, SHAPE_XMA_INPUT_BUFFER_ALIGNMENT)`, filled with
  `memcpy`, and freed with `ApuFree`
  ([`SimplePlaySound.cpp`](https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Audio/SimplePlaySound/SimplePlaySound.cpp)).
  That is the console-specific part a title using XMA reaches; the sample does not show a slot or an ABI addition
  beyond it.
- **Bink.** No Bink use exists in the public GDK samples (the tree has none), so there is nothing to borrow;
  the repository's existing `Bink2Scarlett.dll` path stands.

## 8. Memory, heaps, compression and tiling

SCARAB's `memory-adapter` is also a seam only: every flag is declared with value 0 and the header
says the numeric values are not recovered
([`memory_adapter.h:21-57`](https://github.com/p-portela/SCARAB/blob/d304fa2/components/memory-adapter/include/memory_adapter.h)).
The catalogue contributes **names**, which are useful as a checklist:

- Resource flags: `ALLOW_DCC`, `FORCE_DCC_TEXTURE_COMPATIBILITY`, `FORCE_FMASK_TEXTURE_COMPATIBILITY`,
  `FORCE_HTILE_TEXTURE_COMPATIBILITY`, `FORCE_TEXTURE_COMPATIBILITY`, `DENY_COLOR_COMPRESSION_DATA`,
  `DENY_DEPTH_COMPRESSION_DATA`, `DENY_DEPTH_COMPRESSION_EXPCLEAR`,
  `DENY_DEPTH_COMPRESSION_PLANES`, `DENY_DEPTH_STENCIL_HTILE`, `DENY_STENCIL_COMPRESSION_DATA`,
  `DENY_STENCIL_COMPRESSION_EXPCLEAR`, `DENY_STENCIL_COMPRESSION_SMEM`, `ALLOW_INDIRECT_BUFFER`,
  `ALLOW_AUTOMATIC_GAMEDVR_TONE_MAP`, `INTERNAL_TILED`, `PREFER_INDEX_BUFFER`,
  `PREFER_PREDICATION_BUFFER`, `PREFER_STREAM_OUTPUT`.
- Heap flags: `D3D12CONS_HEAP_FLAG_ALLOW_INDIRECT_BUFFERS`.
- Copy flags: `FORCE`, `FORCE_DECOMPRESS`, `FORCE_DECOMPRESS_EXPAND_COLOR`,
  `FORCE_ELIMINATE_DCC`, `FORCE_GATHER_COLOR_FMASK`, `FORCE_PROPAGATE_COLOR_CLEAR`,
  `RESUMMARIZE_HTILE`.
- Resource states: `PRESERVE`, `PRESERVE_COMPRESSED_COLOR/DEPTH/STENCIL`,
  `PRESERVE_EXPANDED_COLOR/DEPTH/STENCIL`, `PRESERVE_SCATTERED_COLOR_FMASK`.
- Layout/alignment names: `D3D12CONS_TEXTURE_LAYOUT_SWIZZLE_MODE_64KB_R_X`,
  `D3D12CONS_TEXTURE_DATA_PITCH_ALIGNMENT`, `D3D12CONS_DCC_COMPRESSION_LEVEL_128B_TEXTURE_COMPATIBLE`.

None of these has a numeric value in SCARAB, and none has a layout. The tiling/DCC question is
therefore **described, not answered**: the names tell Leonida what to look for in the game's
resource descriptions, and the copy/state names tell it which compression transitions the game
may perform, but the actual swizzle and DCC metadata format still has to come from Leonida's own
`tools/amd_swizzle.py` (the mode's equation, read from the public AMD header and evaluated to
detile with), `tools/tile_probe.py` (the block-endpoint feature and smoothness score measured on
a blob) and `tools/swizzle_search.py` (the brute-force search over every bit ordering with the
right x/y split inside a tile, scored with `tile_probe`), or from the console `xg_xs.dll`
functions `XGComputeOptimalSwizzleMode`, `XGEncodeDCC`, `XGDecodeDCC` (SCARAB names them; it does
not call them).

Two names are worth adding to the trace filter because they affect correctness, not just
performance: `D3D12CONS_RESOURCE_STATE_PRESERVE_*` (a barrier that preserves compressed data) and
`D3D12CONS_COPY_FLAG_FORCE_ELIMINATE_DCC` / `RESUMMARIZE_HTILE` (a copy that changes compression).
If the game uses either, a naive Vulkan copy or barrier will read stale or compressed bytes.

## 9. What this answers for Leonida

Mapping to [d3d12_vulkan.md §8.2](d3d12_vulkan.md). "Answered" means a fact Leonida can implement
against; "described" means names or constraints but no layout; "still unknown" means no usable
evidence.

| Leonida unknown | Status after SCARAB (and §7.1) | What is actually new |
|---|---|---|
| #1 device entry point and 24-word params | **described** | The separate console device export exists at ordinal 13 and is named `D3D12ConsCreateDevice` internally / `D3D12XboxCreateDevice` publicly; the params struct is named `D3D12CONS_CREATE_DEVICE_PARAMETERS`. Public sources name 17 fields and their x64 offsets, plus two more field names; a 24-word block still has at least four words unaccounted (§7.1). |
| #2 device/heap/queue/command-list slots | **partly measured** | SCARAB has no console slot map. Its appended-after-GCL6 model conflicts with the SDK's GCL8/GCL9 methods (§5.1). Public §7.1 gives the presentation slots (device 78-81, queue 30/32); the heap and command-list slot map is still unknown. Do not use SCARAB as a slot source. |
| #3 root signature → `s0..s31` | **still unknown** | Only flag names (`D3D12CONS_ROOT_SIGNATURE_FLAG_FORCE_MEMORY_BASED_ABI`, `_RAYTRACING`); no mapping. Leonida's own register-based/spill facts already go further. |
| #4 descriptor bytes and increments | **still unknown** | No measured increment. `D3D12CONS_DESCRIPTOR_HEAP_TYPE_SAMPLER_UNBOUNDED` is a new name, not a size. |
| #5 VA range / arena capacity | **still unknown** | — |
| #6 DXGI vs `PresentX` | **partly answered** | `PresentX` is a queue method; public code gives its queue slot (30 in the older runtime layout, 32 in the newer) and the x64 plane/present layouts (§7.1). Public samples give the call sequence and argument counts (no DXGI swapchain; frame events + `PresentX`), see §7. The game itself is not observed. |
| #7 standard AS vs BVH builder | **still unknown** | New names: `D3D12ConsRehydrateRaytracingAccelerationStructure`, `D3D12CONS_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_DEHYDRATE`, `DispatchRaysX`, `DispatchRaysDimensionsX`. No layouts. |
| #8 XDXR struct layouts | **still unknown** | Nothing from SCARAB. Measured since from `XboxBvhBuilder.dll` itself, see [xdxr.md](xdxr.md). |
| #9 tiled data layout / DCC | **described** | Flag, copy and state names; no values, no swizzle. |
| #10 binning/DCC/VRS/WMMA usage | **described** | `D3D12CONS_PIPELINE_STATE_FLAG_DISABLE_AUTO_DPBB`, `_DISABLE_RB_PLUS`, `RSSetShadingRateX`; usage still unknown. |
| #11 `CheckFeatureSupport` answers | **partly measured** | A console title asks `D3D12_OPTIONS` with 0x40 bytes where the PC struct is 0x3C (XCL/XDefiant); the extra 4 bytes are unnamed. The layer answers the PC fields and zeroes the tail (§7.1). |
| #12 async compute / copy queues | **described** | `CreateCommandQueueX` creates a shader-based queue, which the UMD says only supports `D3D12_COMMAND_LIST_TYPE_COMPUTE`; `SuspendX`/`ResumeX` exist. No params. |
| #18 heap aliasing / reserved | **still unknown** | — |
| console IIDs | **10 exported; 4 not exported** | The DMA list IID is proven and the other nine console GUIDs are reported. SCARAB's provenance reports that the four command-list IIDs are not runtime exports, so they have no ordinal; this package has no D3D12 DLL to read. They stay unresolved until a title trace or another artifact supplies them. |
| standard IIDs | **answered** | 83/83 standard GUIDs match the SDK; the 28 D3D12 ones are exact. |
| export contract | **answered** | 116 = 23 functions + 93 GUIDs; 23 function ordinals and 59 IID ordinals are documented, 34 are derived. The four command-list IIDs are outside the 93 and have no ordinal. |
| free-function dispositions | **answered as a list** | 7 standard + 16 console-only; Leonida serves 4 today. |

The honest summary: SCARAB closes the **export/IID contract** and the **DMA-list rule**, and gives
a **name-level map** of the extension surface. It does not close presentation, slots, struct
layouts, numeric flags or tiling.

## 10. Claims that did not hold up

These are stated plainly because a future reader will otherwise trust the wrong part.

1. **"The 116 export names" are not in the repository.** The `.def` and CSV are generated from the
   user-supplied vendor DLL; the docs publish SCARAB's internal, de-branded spellings for five
   `D3D12Cons*` exports (§2.3). Use the ordinals and the 18 token-free names; confirm the other
   five wire names from the game's import table.
2. **`RSSetDepthBiasX` has a PC equivalent.** SCARAB says depth bias is PSO-only. The 26100 SDK
   has `ID3D12GraphicsCommandList9::RSSetDepthBias(FLOAT, FLOAT, FLOAT)`. SCARAB's declared
   `INT` first argument is also inconsistent with the SDK. Recording the call and dropping it
   would render with the wrong depth bias on a GCL9 title.
3. **`OMSetFrontAndBackStencilRef` is not a console extension in the 26100 SDK.** It is public
   GCL8 with the same signature. SCARAB's wrapper puts it after GCL6; a GCL8-compiled game calls
   the public slot. Do not copy the placement.
4. **The catalogue overcounts methods.** 105 rows include at least 11 non-methods (§5); the real
   recovered method-name count is at most 94, and the docs elsewhere say "~95". The wrapper's
   "105 method names" is not a measured method count.
5. **The present flag values are guesses.** `console_extensions.h` assigns 0x1/0x2/0x4;
   `present_adapter.h` says the values are unknown and uses 0. Neither is a measurement.
6. **The unknown-arity stubs are not all safe.** The stated policy is a zero-argument stub that
   ignores its arguments, which is safe on the x64 ABI. But `DXGIXGetFrameStatistics` writes 64
   bytes through its first pointer, `PixGetPixieEntryPoints` writes 8, and `D3D12ZeroMemoryX`
   writes `Size` bytes. If the arity or the struct size is wrong, these corrupt memory. The policy
   covers only the genuinely zero-argument stubs.
7. **`iid_console.inc` is mislabelled.** It says "console-specific interfaces"; 49 of its 59
   entries are standard D3D11/DXGI. The values are right; the label can mislead a reader into
   treating a standard IID as console-only.
8. **The build claims need artifacts that are not in the repo.** The wrapper's 53+76 `override`
   proof needs `vendor/d3d12/include/d3d12.h`; the generated `gen_device.inc` / `gen_clist.inc`
   are not committed; the 116/116 export check needs the vendor DLL. The measurements are stated
   with commands, but a clean clone cannot reproduce them without the user-supplied files. That is
   a deliberate policy, not a false claim; it does mean "verified" in SCARAB's README means
   "verified on the author's artifact set".
9. **The Wine results are not reproducible here.** "All 7 DLLs load", "the whole console graphics
   stack runs on Linux", and the DMA round-trip are **Reported**. This page did not open
   `d3d12_xs.dll` and did not run Wine. Nothing in this doc depends on them.
10. **The package's D3D12 DLL exports do not contain the four command-list IIDs with 1.0.2.1.**
    The package has 16 DLLs and no D3D12 runtime DLL, and `gameos.xvd` is encrypted; SCARAB's
    provenance says the runtime does not export these four IIDs. There is no export ordinal to
    read in this package or in the vendor DLL. A `QueryInterface` trace from the running title is
    one possible way to establish the mapping; the package evidence does not exclude other
    sources.

The central claim does hold, with one link stated: **SCARAB's table reproduces 28/28 D3D12 IIDs,
and in fact 83/83 standard GUIDs, byte-identically**, checked against the Windows SDK here. That
the table also equals the vendor runtime's exports is SCARAB's G1 measurement (`93/93` GUID bytes),
which needs the vendor DLL and was not reproduced. The DMA IID and the export composition also
hold as files, subject to the name caveat.

## 11. Licence and provenance

SCARAB is **GPL-3.0-only**: its README says so and every first-party file carries
`SPDX-License-Identifier: GPL-3.0-only`. Leonida is GPL-3.0-or-later. The two are compatible in the
direction that matters — GPL-3.0-only code can be combined into a GPL-3.0 work — but the combined
work cannot be relicensed as "or later" for the SCARAB-derived part. This page copies **no SCARAB
code**. It records interface names, ordinal numbers and 16-byte GUIDs, which are ABI facts, with
the source file and line cited. The 28 standard GUIDs are also facts from the Windows SDK; the 10
console GUIDs are reported from the vendor runtime and are attributed to SCARAB.

The public sources in §7.1 are used as facts only: XCL is GPL-2.0-only (no code copied), ebiten is Apache-2.0,
the Mesa WGL file is MIT, and the Microsoft samples are MIT. Only names, numbers, field offsets and slot indices
are taken from them; no code or prose is copied.

If any code is later adapted from SCARAB, it must carry the SCARAB copyright and
`GPL-3.0-only` notice, and the file must say exactly which SCARAB file it came from. The checker
in `tools/d3d12_xs_surface.py` is original code written for this repository; it reads only the
tables above and, optionally, a local SDK or a SCARAB checkout.

## 12. References

SCARAB, pinned at `d304fa2`:

- `README.md:14-39` — the 28/28 claim, component states.
- `docs/ARCHITECTURE.md:169-185` — E1–E17, the evidence table.
- `docs/d3d12-xs-bridge.md:17-28,82-123` — IID identity, console IIDs, the 116-export disposition.
- `docs/d3d12-xs-wrapper.md:69-97,134-162` — what is proven, class placement, DMA round-trip.
- `docs/d3d12_xs-findings.md:47-67` — the export surface composition.
- `docs/plans/2026-11-21-translation-layer-architecture.md:30` — 23 functions + 93 GUIDs.
- `components/extension-wrapper/docs/console-extension-catalogue.tsv` — 105 + 106 names.
- `components/extension-wrapper/iid_provenance.json:2-20` — the Debug@76 correction and 7 candidates.
- `components/extension-wrapper/src/iid_console.inc:9-67` — 59 GUIDs and their ordinals.
- `components/runtime-bridge/src/iid_table.cpp:7-105` — 93 GUID values.
- `components/runtime-bridge/src/d3d12_xs_bridge.cpp:336-576` — the function stubs.
- `components/extension-wrapper/console_extensions.h:45-191` — type/method declarations.
- `components/extension-wrapper/src/wrap.cpp:279-326` — `CreateCommandList` interception.
- `tools/gen_def.py:2-24,38,149-183` — the token alias rule.
- `components/present-adapter/include/present_adapter.h`, `components/memory-adapter/include/memory_adapter.h`
  — the two seams.

Windows SDK 10.0.26100.0:

- `um/d3d12.h:7237,30278,31126,32819,33680` — `OMSetDepthBounds`, `RSSetShadingRate`,
  `DispatchMesh`, `OMSetFrontAndBackStencilRef`, `RSSetDepthBias`.
- `um/d3d12sdklayers.h:4243` — `IID_ID3D12Debug`.
- `um/d3d11*.h`, `shared/dxgi*.h`, `um/d3dcommon.h`, `um/dxgidebug.h` — the other standard GUIDs.

Leonida:

- [d3d12_vulkan.md](d3d12_vulkan.md) §3.5, §4.5, §7, §8.2 — the gaps this page maps onto.
- `docs/runtime_imports.md` (runtime-imports branch) — which imports the shipped DLLs actually bind.
- `runtime/shims/d3d12_xs.cpp` — the four functions Leonida serves today.
