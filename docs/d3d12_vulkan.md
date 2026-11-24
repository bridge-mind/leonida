# Xbox D3D12 (`d3d12_xs`) to Vulkan

**Status: in progress. A Vulkan-backed `ID3D12Device` (queues, allocators, command lists, fences, descriptor heaps,
render-target and depth views, texture views and samplers, committed and reserved buffers, 2D textures and 3D texture transfers and color clears, and query heaps)
runs behind the Xbox-ABI device proxy, and a presenter puts back buffers in a window (section 3.2). The M0
milestone passes in a synthetic test (section 6), M1 with stand-in shaders in `d3d12_draw_test`, M1.5 with
hand-written RDNA2 run through the translator in `d3d12_translated_draw_test`, and the synthetic part of M2
(descriptor heaps, SRVs, a sampler and a translated textured-quad draw, 4,096 exact pixels) in
`d3d12_m2_textured_test`. `D3D12XboxCreateDevice` uses it when `LEONIDA_D3D12=vulkan` is
set and the host PC device otherwise. Root signatures (packed into the 32 user-data dwords), graphics and compute
PSOs, draws, dispatches, `ExecuteIndirect`, and presentation through `PresentX` or a DXGI swapchain are implemented
and have GPU tests; which present path the game uses is still open (question 6). Shader-resource views of buffers,
pipeline-state streams (`CreatePipelineState`), tessellation and stream-output pipelines are not implemented, and
mesh shaders are reported as not supported (stream PSOs and mesh shaders: D3D-29, #1712). The device reports
`RaytracingTier` as `NOT_SUPPORTED`, yet `BuildRaytracingAccelerationStructure` builds XDXR BVH4 structures on the
GPU (section 4.8).** This is the plan for
replacing the console's `d3d12_xs.dll` with an implementation that records D3D12 commands and replays them on
Vulkan, while producing exactly the shader-visible contract that `tools/rdna2spirv.py` already emits against (see
[design.md](design.md#translator-status-and-runtime-contract)).

The document is deliberately conservative about what is known. Every claim is marked:

- **Verified** — observed in this repository or in the game package, with a way to reproduce it.
- **Public** — documented in the Windows SDK, Microsoft's public GDK/D3D12 documentation, or the Vulkan
  specification.
- **Assumption** — a design choice that must be checked against the game's executable when it is available.
- **Open** — unknown; listed with a way to resolve it.

Nothing here uses game data, keys or decryption. The package is read-only; the executable that would answer most
of the open questions is not in it.

## 1. The job

The game is an Xbox Series build (`game_scarlett_master.exe`, `d3d12_xs.h` / `d3d12_xs.lib`). It was compiled
against the Xbox flavour of D3D12 and calls the console driver through COM interfaces. The layer must:

1. **Intercept the D3D12 surface.** Free functions exported by `d3d12_xs.dll` (device creation, root-signature
   serialization) and every COM method the game calls on the returned objects. The objects are our own C++
   objects; the vtables must have the exact method order the game was compiled against.
2. **Preserve D3D12 semantics that affect output.** Command lists record; execution is deferred; descriptors are
   values (GPU virtual addresses or CPU handles); resource states are explicit; root arguments are set at record
   time; allocators and fences control reuse.
3. **Produce the translator's shader contract.** The translated SPIR-V is already written against a bindless set
   0, a 144-byte push-constant block, a descriptor hash table and one GPU address space. Changing that contract
   would invalidate 26,503 translated shaders and the GPU tests that check them.

The layer is not a general D3D12-on-Vulkan compatibility layer. It only needs the calls this game makes, but each
call it accepts must behave like D3D12.

## 2. What is verifiable today

| Fact | Evidence | Confidence |
|---|---|---|
| The public D3D12 interface chain and method order are in the Windows SDK `d3d12.h` (10.0.26100.0 here). The Xbox public reference lists the same base interfaces and method names under `d3d12_xs.h` / `d3d12_xs.lib`; the measured Xbox device ABI is mostly that order, with deviations at slots 62/76 and heap slots 9/10 (section 7). | Local SDK header; [Xbox public graphics reference](https://devdocs.xbox.com/reference/graphics/d3d12); e.g. [CreateCommittedResource](https://devdocs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device/methods/id3d12device_createcommittedresource_public) | Public; measured deviations in section 7 |
| The game's raytracing helper is a shipped DLL, `XboxBvhBuilder.dll`, whose export signatures name `ID3D12Device11`, `ID3D12GraphicsCommandList4`, `ID3D12GraphicsCommandList6` and `ID3D12GraphicsCommandList8`, plus Xbox XDXR batched/streaming acceleration-structure types. | PE export table of `XboxBvhBuilder.dll` (884,224 bytes); see Appendix A | Verified |
| That DLL imports only `d3d12_xs.dll!D3D12SerializeVersionedRootSignature` and `xmem.dll!XMemVirtualQuery` from console libraries. It receives the device and command lists from the game. | PE import table; Appendix A | Verified |
| `amd_fidelityfx_upscaler_xbox.dll` imports only `d3d12_xs.dll!D3D12CreateRootSignatureDeserializer` and contains the base `ID3D12Device`, `ID3D12Resource`, `ID3D12PipelineState`, `ID3D12RootSignature`, `ID3D12Heap`, `ID3D12DescriptorHeap` and `ID3D12CommandSignature` IIDs. | PE import table and GUID byte scan; Appendix A | Verified |
| None of the other scanned non-CRT DLLs (`amd_fidelityfx_loader_xbox`, `Bink2Scarlett`, `libHttpClient.GDK`, `XCurl`) imports `d3d12_xs`. | PE import scan; Appendix A | Verified |
| The package layout lists `game_scarlett_master.exe`, but the file is not in the package folder. | `layout_..._xs.xml`, chunk 1001; directory listing | Verified |
| `gameos.xvd` is an XVD (`msft-xvd` at file offset 0x200, format version 3) with a high-entropy payload and no plaintext PE headers or D3D12 strings. Extracting `d3d12_xs.dll` from it would require keys, which this project does not do. | Entropy and string scan; Appendix A | Verified |
| Prior art runs the package DLLs against a PC D3D12 device and measured the Xbox ABI: 6 Xbox-specific static imports across the 16 DLLs, `D3D12XboxCreateDevice`, device slots 62/76, descriptor-heap start handles in RAX, root-signature flags `0x03000000`, and the BVH builder's Xbox DXIL shaders. | [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md), [platform README](https://github.com/realdody/americas-recomp/blob/xbox-series/tools/xbox/platform/README.md) | Verified by the prior art (running the package DLLs / static imports; section 7) |
| The translator and its GPU tests already define the shader contract: set 0 bindings 0–19, 144-byte push constants, heap/code base, FNV-1a descriptor hash table, buffer device addresses. | [design.md](design.md#translator-status-and-runtime-contract); `runtime/test/vk_texture_test.cpp` | Verified |

What this does **not** tell us: the full `d3d12_xs.dll` export list, the `D3D12XboxCreateDevice` params/flag
layout, the remaining Xbox slot deviations (presentation, command lists, queues, fences), the
root-signature-to-`s0..s31` mapping, the console descriptor layout, the present path, the raytracing path, and the
GPU VA window. Section 8.2 lists each with a way to resolve it.

The first run with the executable should be a **trace build**: every free function and COM method is a logging
stub that returns a failure, and the methods are filled in in the order the game calls them. Section 3.1 describes
the interception; section 8.2 turns the trace into a work list.

## 3. Architecture

### 3.1 Interception and COM

The current shim (`runtime/shims/d3d12_xs.cpp`) forwards only two free functions to the host `d3d12.dll`. That is
enough for the bundled middleware's `d3d12_xs` calls, not for the game. The layer adds:

- **Free functions.** Every `d3d12_xs.dll` export the executable imports. The import table is the source of truth.
  Prior art names the device entry point `D3D12XboxCreateDevice(adapter, params, riid, out)`; its 24-word params
  struct is only logged there, and the signature is taken from public documentation rather than a header. Treat the
  params/flag layout as **Open**, accept the call, log the words, and create the device. Do not assume
  `D3D12CreateDevice` is the console entry point.
- **COM objects.** `Device`, `CommandQueue`, `CommandAllocator`, `GraphicsCommandList`, `RootSignature`,
  `PipelineState`, `DescriptorHeap`, `Resource`, `Heap`, `Fence`, `QueryHeap`, `CommandSignature`,
  `PipelineLibrary`, and the raytracing/lifetime objects if used.
- **Vtable layout.** Use the Windows SDK's `d3d12.h` declarations for the public chain, but treat the vtable as
  **mostly public, not universally public**. Prior art measured device slots 1, 15, 18, 19, 27 and 38 in the
  desktop order, but also an extra texture-creation method at slot 76, a `CreateComputePipelineState` call at
  slot 62 where the PC interface it forwarded to presents `CreateStateObject` (different arity), and descriptor-heap
  start handles returned in RAX at heap slots 9/10. Implement the public chain plus those measured overrides
  explicitly; do not assume every slot. Device 11 adds `CreateSampler2`; command list 8 adds
  `OMSetFrontAndBackStencilRef` (public SDK).
- **Unknown IIDs and slots.** `QueryInterface` must log the IID and return `E_NOINTERFACE` if it is not in the
  public chain. If the game then fails, the IID and the call site identify the Xbox-specific interface to reverse.
  Do not invent a vtable for an unknown IID: a wrong slot is worse than a clean failure.
- **No host D3D12 or DXGI.** The PE loader currently lets unknown system DLLs bind to the host. `dxgi.dll` and
  `d3d12.dll` must be shimmed, not resolved from System32. Host DXGI would hand the game a desktop adapter and
  swapchain that cannot see our Vulkan device, and host D3D12 would create a second, unrelated device. This is an
  integration requirement, not an optimisation.
- **Trace first.** `gta6rt --trace-d3d12` emits numbered records for device entry points, Vulkan device operations,
  command-list recording and queue submission. Records contain fixed API names and outcomes only. See
  [the runtime guide](../runtime/README.md#first-boot-d3d12-trace) for collecting a trace.

**Measured, not assumed:** the Xbox device ABI is not uniformly the desktop order. Slots 1/15/18/19/27/38 matched
in the prior art, while slots 62 and 76 and the heap start-handle return convention did not. Treat every slot as
unverified until traced; the public interface names and IIDs may still be usable, but the slot mapping is not
guaranteed.

### 3.2 Record and replay

D3D12 command lists are recorded, then executed later. Vulkan command buffers are also recorded, but the layer
must not simply translate each D3D12 call into a Vulkan call as it arrives, because:

- descriptor heaps can be updated between recording and execution;
- a command list can be reset and rerecorded;
- a bundle is recorded once and executed many times.

The design is therefore an **IR per command list**:

- `SetPipelineState`, root arguments, vertex/index buffers, viewport/scissor, render targets, descriptor-heap
  bindings and the other sticky state update a state block.
- Each draw/dispatch/copy/barrier appends a compact command with a snapshot of the state that affects it.
  GPU descriptor tables can refer to shader-visible heap entries at execution. CPU descriptors such as RTVs
  are consumed while recording, so their view contents go into the IR. An application can reuse the CPU
  descriptor slot after the call. See Microsoft's [descriptor overview][descriptor-overview].
- `ExecuteCommandLists` replays the IR into a fresh `VkCommandBuffer`, resolving descriptors, uploading dirty
  descriptor ranges and the hash table, and emitting barriers. Bundles are inlined when they are executed (see `ExecuteBundle` below).
- `Reset` on a command list discards its IR; the allocator owns the memory. D3D12 already requires the application
  not to reset an allocator whose lists are in flight; the layer can rely on that and assert in debug builds.
- Command-list and allocator misuse returns what the PC runtime returns (D3D-51). `d3d12_vulkan_device_test` runs
  the same 25 calls on the host runtime and on this layer and compares the HRESULTs. One list at a time records into
  an allocator: creating or resetting a second list onto it gives `E_INVALIDARG`, and resetting the allocator while
  a list records into it gives `E_FAIL`. `Close` twice and `Reset` of an open list give `E_FAIL`. Resetting an
  allocator whose lists are still executing gives `S_OK` on the PC runtime too (keeping that safe is the
  application's job). Two differences remain. `Reset` with a null allocator crashes the PC runtime, while ours
  returns `E_INVALIDARG`, so that call is not compared. Executing a list that is still open removes the PC device
  (`GetDeviceRemovedReason` gives `DXGI_ERROR_INVALID_CALL`), while this layer submits nothing and keeps the device:
  an invalid call does not remove it.
  Creation answers as the host runtime too (`compare_invalid_arguments`, "command allocator:", "command list:" and
  "command list1:", RTX 5070 / 621.14): a type past `VIDEO_ENCODE` and a node other than node 0 are `E_INVALIDARG`,
  checked before the output; a null output is then `E_UNEXPECTED`. Video list types are created as graphics lists
  here, while the host hands out a video list (`E_NOINTERFACE` for `ID3D12GraphicsCommandList`), so they are not
  compared.
- `Close` freezes the IR. `ExecuteBundle` is accepted only on a direct list, with an already-closed bundle as its callee; the bundle's recorded commands and calls are replayed at the call site, so the source can be reused or reset without changing the parent. Command-list creation and reset require an allocator of the same type. A bundle cannot be submitted directly to a command queue.
- Bundle restrictions are checked while recording for the IR operations currently implemented here (barriers, clears, copies, resolves, `SetPredication`, `DiscardResource`, the atomic copies, `RSSetViewports`, `RSSetScissorRects`, `OMSetRenderTargets` and `ClearState`; `WriteBufferImmediate` is allowed in a bundle); those calls are not legal inside a D3D12 bundle and make `Close` return `E_INVALIDARG`. `d3d12_bundle_forbidden_test` measured the last four on the PC runtime without the debug layer: the bundle's `Close` fails (`E_FAIL`, where this layer gives `E_INVALIDARG`) on the default adapter and on WARP, and executing a bundle whose `Close` failed crashes the runtime, so the layer refuses it in `ExecuteBundle` instead. A bundle has no render targets, viewport or root signature of its own, so its state calls (pipeline state, root signature and arguments, vertex and index buffers, topology, blend factor, stencil references) and its draws and dispatches are kept as they were called, and `ExecuteBundle` plays them on the parent through the same members the application would have called. A draw then snapshots the parent's state plus what the bundle set, and what the bundle set stays on the parent afterwards. Commands that need no state (markers, immediate writes) are copied between them in order. `d3d12_bundle_test` measured the rule on the PC runtime, where the default adapter (an RTX 4060 Ti) and WARP drew the same picture in every case it ran. A bundle's draws see the parent's render targets, viewport and scissor, root signature and arguments (also when the bundle sets the same root signature first, so those arguments stay), and vertex and index buffers. A bundle starts with the pipeline state it was created or reset with, not the parent's. After `ExecuteBundle` the parent has that pipeline state and whatever else the bundle set: topology, root arguments, vertex and index buffers, and a different root signature with the bundle's arguments in place of the parent's. Topology: the default adapter takes the parent's when the bundle sets none. That is more lenient than Microsoft's stated rule (topology is undefined when a bundle begins), and WARP follows the rule by removing the device when a bundle draws or hands back without one, so a bundle should set it; the layer follows the default adapter. A bundle that sets a root argument before any root signature crashes the NVIDIA driver (the layer applies it to the parent's signature). The deferred calls are checked when they play, against the parent's root signature, so a bad one makes the parent's `Close` fail, not the bundle's. A per-list state setter must go through the deferred path (`defer()` in `d3d12_com.cpp`, replaying through a private member that does not trace) to work in a bundle. Not deferred yet: `ExecuteIndirect` in a bundle, and bundle `Dispatch` has no GPU test.
- What `Close` returns after one recorded call is compared with the PC runtime over WARP (no debug layer) by
  `d3d12_close_parity_test`, 100 cases on fresh lists, nothing executed. 62 agree; the other 38 are listed in the
  test as known differences. In 23 the layer fails a list WARP closes: draws and dispatches with no pipeline,
  root arguments before a root signature, a `CopyTextureRegion` box that is empty or inverted, footprint to
  footprint or at an unaligned footprint offset or pitch, `CopyResource` of a resource onto itself,
  `ClearDepthStencilView` with no flag, an unknown flag or a depth outside 0..1, vertex-buffer slots past 31,
  17 viewports, an `OCCLUSION` query ended as `BINARY_OCCLUSION`, and `WriteBufferImmediate` at an address that
  is not 4-aligned. Since one failed list fails the whole `ExecuteCommandLists`, each of these drops a submission
  D3D12 would run. In 15 the layer closes a list WARP fails (redundant or out-of-range barriers, copies past a
  resource's end or between formats of another family, resolves the PC runtime refuses, a query index past the
  heap); what the layer does with those at execution is not part of this test. A refused list's code is always
  `E_INVALIDARG` here, where the PC runtime also answers `E_FAIL`, so the test compares the verdict and prints both
  codes.
- A queue's `ExecuteCommandLists` submits in order with one timeline semaphore signal. `Wait`/`Signal` become
  timeline-semaphore waits/signals. `Present` queues after the frame's work.

The command-list surface maps to the IR as follows:

| Recorded call | IR / replay |
|---|---|
| `SetPipelineState` | state; bind the `VkPipeline` before the next draw/dispatch |
| `SetGraphicsRootSignature` / `SetComputeRootSignature` | state; select the pipeline layout and root map |
| `Set*Root32BitConstant(s)` / `Set*Root*View` / `Set*RootDescriptorTable` | update root args; build the push-constant block at the draw/dispatch |
| `SetDescriptorHeaps` | state; scan/populate the bindless arrays and hash table at execute |
| `IASetVertexBuffers` / `IASetIndexBuffer` / `IASetPrimitiveTopology` | state; bind arena buffer + offset, dynamic topology |
| `RSSetViewports` / `RSSetScissorRects` | state; dynamic viewport/scissor |
| `OMSetRenderTargets` / `OMSetBlendFactor` / `OMSetStencilRef` | state; rendering scope and dynamic state |
| `DrawInstanced` / `DrawIndexedInstanced` | command with a state snapshot |
| `Dispatch` / `DispatchMesh` | command with a state snapshot |
| `Copy*` / `Resolve*` / `Clear*` | copy/clear command |
| `ResourceBarrier` / `Barrier` | barrier command; state tracking |
| `DiscardResource` / `WriteBufferImmediate` / `AtomicCopyBufferUINT(64)` | discard, immediate-write and atomic-copy commands (section 4.2) |
| `BeginQuery` / `EndQuery` / `ResolveQueryData` | query commands |
| `ExecuteIndirect` | command; signature + argument buffer |
| `BeginRenderPass` / `EndRenderPass` | rendering scope |
| `ExecuteBundle` | play the bundle's state calls and draws on this list, copy its state-free commands in order |

The IR is the place to add the trace, validation and "unsupported call" reporting. It also lets the layer replay a
frame deterministically from a capture, which is valuable for comparing against the game.

What is implemented (all of it testable without the game):

- **IR** (`runtime/src/command_list.h`): legacy barriers (transition, UAV, aliasing; split barriers record their end
  half), enhanced barriers (global, buffer, texture; the half with `SyncAfter = SPLIT` is dropped and the other is
  replayed as a full barrier), `ClearRenderTargetView` / `ClearDepthStencilView`, whole view or rectangles,
  `CopyBufferRegion` / `CopyTextureRegion` / `CopyResource`, colour resolves,
  `ClearUnorderedAccessViewUint` / `ClearUnorderedAccessViewFloat` (whole view or rectangles), (D3D-26)
  `DiscardResource`, `WriteBufferImmediate` and `AtomicCopyBufferUINT` / `AtomicCopyBufferUINT64`, and `BeginQuery` /
  `EndQuery` / `ResolveQueryData` (section 4.6).
  Views and rectangles are copied while recording, so the CPU descriptor slot can be rewritten afterwards.
  A `ClearDepthStencilView` flag for a plane the view's format doesn't have is ignored, as the PC runtime (WARP)
  does: `STENCIL` on a `D32_FLOAT` view clears nothing and `DEPTH | STENCIL` clears the depth plane. Until 2026-11-23
  `prepare` returned `invalid_clear` for it and the whole list was dropped (D3D-437); `d3d12_clear_test` compares three
  such cases with WARP.
- **Replay** (`runtime/src/vulkan_replay.h`): `prepare` resolves and checks a list without writing anything, `emit`
  writes it. Legacy states map through the section 4.3 table (`vulkan_access`), enhanced barriers through
  `vulkan_sync` / `vulkan_barrier_access` / `vulkan_layout` to `vkCmdPipelineBarrier2`. Whole-view clears are
  transfer clears inside a layout round trip; rectangle clears are `vkCmdClearAttachments` in a dynamic-rendering
  scope, with rectangles clipped to the mip. Copies become `vkCmdCopyBuffer`, `vkCmdCopyBufferToImage`,
  `vkCmdCopyImageToBuffer` and `vkCmdCopyImage`. A placed footprint's row pitch becomes `bufferRowLength`, and
  block-compressed extents stop at the mip edge (D3D12 pads footprints to whole blocks, Vulkan does not; a
  texture-to-texture box of a block-compressed format is rounded up to whole blocks, as on the PC runtime, must fit
  the destination's padded mip, and the Vulkan extent is cut where it reaches an image's edge; Vulkan wants that per
  image, so when no single cut extent is a block multiple or at the edge of both images, for example a 4x4 box into
  a 3x3 mip, the copy goes through transfer scratch (`vulkan_scratch.h`, as depth <-> colour copies do):
  `vkCmdCopyImageToBuffer` with the source's own cut extent into whole blocks, then `vkCmdCopyBufferToImage` with the
  destination's. The box fits the source's padded mip, so the scratch holds every block the destination reads). Each
  range and buffer bound is checked before anything is written. A copy runs in `TRANSFER_SRC`/`TRANSFER_DST`, or
  in place in `GENERAL` when the texture is in `COMMON`: that is D3D12's implicit promotion to a copy state.
  A same-subresource `CopyTextureRegion` uses `GENERAL` for both sides; preparation rejects overlapping regions.
  Cross-format `CopyTextureRegion` from an uncompressed texture into a partial BC edge block also uses transfer
  scratch. Each source texel supplies one block, and the destination's write extent stops at the actual mip edge.
  A 3D mip is one subresource. Boxed volume transfers preserve the full footprint's row and slice strides.
  Whole-resource depth/stencil copies issue one `vkCmdCopyImage` per aspect, with a transfer write-to-read/write
  dependency between them; all mips and layers stay batched within each aspect. This avoids stencil corruption
  observed on RX 5700 XT, Windows driver 32.0.21045.5002, when both aspects share one multi-region copy command.
  A standalone synthetic Vulkan multi-mip array reproduces the corruption with validation clean; this is a
  conservative compatibility measure, not evidence that the Vulkan specification requires this dependency.
  `d3d12_stencil_copy_test` checks D24S8/D32S8 across mip/layer counts and odd dimensions against WARP (CTest)
  or `LEONIDA_HOST_ADAPTER` (manual runs), plus the deterministic upload pattern for both planes and sources.
  **Depth ↔ colour copies** (`CopyResource` / `CopyTextureRegion` between `D32_FLOAT` and `R32_FLOAT/UINT/SINT`,
  or `D16_UNORM` and `R16_FLOAT/UNORM`, single-sampled, single-plane depth) copy the bits as the PC runtime does.
  `vkCmdCopyImage` refuses mismatched aspects, so each region goes image → scratch buffer → image
  (`src/vulkan_scratch.h`: device-local chunks kept until the submission completes, on every GPU). Not covered:
  depth with stencil (two planes), a D24 stored as D32, and multisampled textures; they are still refused. A float
  written into a `D32_SFLOAT` image outside [0, 1] is undefined in Vulkan without `VK_EXT_depth_range_unrestricted`,
  where D3D12 keeps the bits; `d3d12_copy_xfmt_test` uses values in [0, 1).
  **D24_UNORM_S8_UINT without a native D24** (the Radeon 880M, AMD hardware generally, RADV): the layer stores
  `D32_SFLOAT_S8_UINT` and makes it behave as D24 (`src/depth24.h`, `src/vulkan_depth24.h`). A footprint
  of the depth plane holds 24-bit UNORM words (bits 0-23; the top byte is unspecified, the PC runtime reads it as
  nothing and writes 0), the stencil plane is plain bytes. An upload converts the words with a compute pass
  (`src/depth24_convert.comp`, integer arithmetic) into a scratch buffer of floats, the nearest float to
  `d / 16777215`, which `vkCmdCopyBufferToImage` then moves into the image; a readback copies the image into the
  scratch and the pass writes the words (a float is multiplied by 16777215 in float and rounded to nearest, ties to
  even, which is what WARP and AMD's D3D12 driver do; NaN and negatives are 0, 1 and above the maximum). Every
  24-bit word makes the round trip unchanged (checked on all 16777216 by `depth24_convert_test`). Texture to
  texture copies move the floats as they are. `ClearDepthStencilView` of any D24 texture, native or stored as D32, clears with the float of the quantised value (an exact 24-bit value, so no driver tie rounding is left),
  so the texture reads back as D3D's quantisation of the clear value. The scratch buffers and descriptors live
  until the submission that used them completes. A copy larger than the device's `maxStorageBufferRange` (128 MiB at
  least) is split into several dispatches, by slices or by rows of one slice. The dispatch runs even under a
  `SetPredication` that would skip copies: the layer runs copies unpredicated (section 4.7) and the conversion
  must go with them, so conditional rendering is ended around it. The D3D11.3 functional spec (3.2.3.6, FLOAT to
  UNORM) says: NaN is 0, the value is clamped to 0..1, multiplied by 2^n - 1 in float, 0.5 is added and the
  fraction dropped, with a tolerance of 0.6 ULP on the integer side; read literally a tie rounds up, and 3.2.2
  lists this conversion as an exception to round-to-nearest-even without fixing ties. WARP and AMD's D3D12 driver
  were measured to round the float product to nearest even (2.5 gives 2, 0.5 gives 0), which the tolerance
  allows; NVIDIA's D3D12 driver (and a native D24 clear in its Vulkan driver) breaks some of those ties the other way;
  the layer follows WARP and AMD there, a measurement and not a rule of the spec. **Draws** (D3D-309,
  `src/depth24_fragment.h`): a single-sampled graphics pipeline with depth enabled and a D24 DSV stored as D32 has
  its pixel stage write `FragDepth = quantise(z + bias)`, with D3D's bias in D24 units (`DepthBias / (2^24 - 1) +
  SlopeScaledDepthBias * max(|dz/dx|, |dz/dy|)`, clamped by `DepthBiasClamp`) and the same quantisation as
  `depth24_convert.comp`, so depth tests and stored depth match D24; the rasteriser's bias is off for these
  pipelines (so a clamp needs no `depthBiasClamp` there). A pipeline without a pixel shader, with or without a
  geometry shader, gets a built-in one; a shader that writes its own depth has it quantised, unbiased, as in D3D. The slope comes from the derivatives of the
  interpolated depth, so a slope-scaled bias can land up to two 24-bit steps from D24 hardware's plane-exact one
  (`d3d12_depth24_draw_test`). Not covered: multisampled targets (a `FragDepth` write is per pixel; they keep float
  bias and precision, logged), pixel shaders with `EarlyFragmentTests` (which ignore a `FragDepth` write; logged), a
  viewport depth range narrower than the biased depth (D3D clamps before quantising), and early depth testing, which
  a `FragDepth` write turns off for these pipelines. A depth texture sampled in a shader still sees the float.
  `ResolveSubresource` becomes `vkCmdResolveImage` in the same layouts as a copy: a multisampled colour
  subresource into a single-sampled one of the same size and format. Integer colour formats in the requested
  format or either resource are refused at recording; `Close` fails. `ResolveSubresourceRegion` uses the same
  Vulkan operation for non-integer colour with average mode, taking a source rectangle (or the entire subresource)
  and a destination offset. A `ResolveSubresourceRegion` of one depth or stencil plane (the subresource's plane
  slice, the same on both sides) runs in a rendering scope of its own with no draws: the multisampled plane is the
  depth or stencil attachment (load `LOAD`, store `NONE`), the destination its resolve target with the mode, the
  render area the source rectangle. Both move from their resolve layouts to `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` and
  back around it; the destination's other plane is not attached and keeps its contents. A rectangle smaller than the
  subresource has the texels around it copied into a scratch buffer before the scope and back after it, on every
  GPU: Vulkan resolves only inside the render area, but MoltenVK (1.4.2 and 1.4.3, Apple M3 Max) resolves the whole
  destination subresource, because it resolves through Metal's store action, which has no render area. Measured
  with a raw Vulkan program (a 4x D32 attachment, render area (2,1) 5x4 of 8x6, `MIN`, `MAX` and `SAMPLE_ZERO`,
  store `NONE` or `STORE`): MoltenVK writes all 48 destination texels, KosmicKrisp only the 20 inside. No device
  property reports it, so nothing is gated. The format is the depth
  format or the plane's own (`R32_FLOAT`, `R24_UNORM_X8_TYPELESS`, `X24_TYPELESS_G8_UINT`, and for D32S8
  `R32_FLOAT_X8X24_TYPELESS` and `X32_TYPELESS_G8X24_UINT`; all compared with the PC runtime, TEST-847). A depth-only source above array layer 0 is resolved in a
  `VkRenderPass` with a framebuffer of the same two views instead of the rendering scope, on every GPU (D3D-417, a
  driver workaround): AMD's Windows driver (32.0.21042.62, Ryzen 7 9700X iGPU) resolves a dynamic-rendering depth
  attachment from layer 0 of the image when the multisampled view starts at another layer, while it honours the
  resolve view's first layer. Measured with a raw Vulkan program on a 4x D32 image of four layers: a layer range
  cleared with `vkCmdClearDepthStencilImage`, a `CLEAR` load or `vkCmdClearAttachments` reads back right through a
  `texelFetch` of every sample, through a resolve of all four layers in one scope and through a copy of the layer
  into a one-layer image that is then resolved, but a one-layer view resolved in a rendering scope gives layer 0
  for every layer (view type `2D` or `2D_ARRAY`, store `NONE` or `STORE`), and views of layers 1-3 resolved in one
  scope give layers 0-2. The render pass gives the right layer there; an RTX 3070 (610.74) does in every variant.
  `d3d12_view_slices_test` reads a 4x D32 array back slice by slice. A depth-stencil source above layer 0 still
  uses the rendering scope, because its two planes can be in different layouts, which the render pass's attachment
  description cannot say without the planes' layouts at recording; a 4x D24S8 or D32S8 array is still read back
  as its slice 0 on that driver (measured). The format is the depth
  format or the plane's own (`R32_FLOAT`, `R24_UNORM_X8_TYPELESS`, `X24_TYPELESS_G8_UINT`, and the D32S8 pair, untested).
  What the PC runtime accepts was measured on an RTX 5070 (driver 621.24) and on WARP: `MIN` and `MAX` of either
  plane by either format, and `AVERAGE` of the depth plane by its own format only (its `Close` fails for `AVERAGE`
  by the depth format or of stencil); the layer accepts the same. WARP is no reference for two of them: a resolve
  with the format `D32_FLOAT` passes its `Close` but removes the device when run (`DXGI_ERROR_DRIVER_INTERNAL_ERROR`),
  and a one-plane D24S8 resolve writes both planes; the RTX 5070 resolves only the plane named. A scope resolves each pixel in place, so a
  destination offset other than the source rectangle's corner is refused, as is a mode missing from the device's
  `supportedDepthResolveModes`/`supportedStencilResolveModes` (queried at device creation; one plane per scope, so
  `independentResolve` never applies). Refused calls fail `Close` and record nothing. `d3d12_depth_resolve_test`
  compares D32, D24S8 and D32S8 resolves (typed and typeless resources, whole and rectangle) with the PC runtime and
  WARP. Its edge cases (TEST-848) draw a triangle over the cleared 4x source first (depth 0.5, stencil 0x44): the
  pixels on its diagonal edge hold differing samples, and MIN, MAX and AVERAGE over them match the PC runtime on an
  RTX 5070 (driver 621.01) and WARP byte for byte. `DECOMPRESS`,
  `ResolveSubresource` of depth (which D3D12 does not offer) and resolves between two formats of one typeless family
  remain open. A multisampled texture copies only to one with the same sample count
  and has no footprint.
  Discards, immediate writes and atomic copies replay as section 4.2 describes.
- **UAV clears** (`ClearUnorderedAccessViewUint` / `ClearUnorderedAccessViewFloat`). The CPU handle's slot is read
  while recording as the GFX10 descriptor a shader would load (section 3.4), so the clear works with any
  view-creation code that writes them and with descriptors the game builds itself; the GPU handle is not needed.
  A T# gives the view's format (FORMAT plus DST_SEL, which tells BGRA from RGBA), mip and layers; a 3D view clears
  every slice of its mip. A V# gives a byte range: its address minus the resource's `GetGPUVirtualAddress`, and
  NUM_RECORDS bytes (stride 0, no format: raw), records (a stride: structured) or elements (a format: typed). A view
  that reaches past the end of the buffer stops there, as the hardware's bounds check would.
  - Values follow D3D's rules (`runtime/src/uav_clear.h`): Uint copies the low bits of each value into its channel
    with no conversion, whatever the format's type; Float converts (UNORM/SNORM clamp and round to nearest, NaN is
    0; 16-bit and the packed 11/10-bit floats round to nearest even). A raw or structured view takes value 0 for
    every 32-bit word (Float: its bits). D3D11 calls Float on an integer view invalid and D3D12 does not say; here
    the value is converted (toward zero, clamped, NaN 0) so the result is defined. These are checked against
    hand-computed elements in `d3d12_uav_clear_test`.
  - Buffers: elements of up to 4 bytes repeat in every aligned word, so `vkCmdFillBuffer` writes the aligned middle
    and `vkCmdCopyBuffer` copies the unaligned ends from it (an 8-bit view at an odd offset, say). 8- and 16-byte
    elements are written with `vkCmdUpdateBuffer` (up to 64 KiB) and doubled with copies. A rectangle is an
    element range `[left, right)` on row 0. If a range contains no complete aligned word, prepare reserves eight
    transfer-scratch bytes, retained until submission completion. Replay fills that scratch with the repeated
    element and copies only the requested bytes, preserving neighbouring bytes even at resource boundaries.
    Rectangles within one clear reuse that pattern with transfer dependencies between them. Ownerless replay
    callers without `copy_scratch` still refuse these short ranges before emitting anything.
    `d3d12_short_uav_clear_test` compares Uint/Float whole-view and clipped/overlapping rectangle clears against
    WARP (CTest) or `LEONIDA_HOST_ADAPTER` (manual runs), checking the entire buffer against literal patterns.
    Portable native comparisons keep rectangles on row 0. Out-of-row controls run separately against WARP:
    NVIDIA D3D12 was observed to ignore Y where WARP and AMD preserve the buffer, so those controls do not assert
    that native drivers behave uniformly. This test distinction does not change the replay policy.
    Portable UNORM Float clears use `0.25f` (R8: `0x40`, R16: `0x4000`), away from a conversion midpoint;
    `0.5f` remains covered by separate WARP-only midpoint controls. Native NVIDIA was observed to clear that
    midpoint to `0x7f`/`0x7fff` where WARP and AMD give `0x80`/`0x8000`. All comparisons remain byte-exact.
  - Textures: the values become one element of the view's format, and that element is cleared into the image as
    the value that decodes it in the image's own format, so a typeless texture viewed as UINT gets exactly the bits
    asked for. Exact for every typed-UAV format except the SNORM pattern `0x80..` (Vulkan writes -1.0 as `0x81..`)
    and NaN payloads. A whole view is `vkCmdClearColorImage`; rectangles are `vkCmdClearAttachments` in a
    dynamic-rendering scope, so a texture created with `ALLOW_UNORDERED_ACCESS` also gets `COLOR_ATTACHMENT` usage
    when its format supports it. Both run in place in `GENERAL`, the UAV layout. Rectangle clears of volumes, or of
    formats without attachment support, are refused as `unsupported`.
  - The clear is fenced as the UAV state widened to transfers, so a following UAV barrier, transition, or enhanced
    barrier from `SYNC_CLEAR_UNORDERED_ACCESS_VIEW` sees it.
  - Under `SetPredication`, a whole-view texture clear runs as one rectangle through the attachment path, as
    render-target clears do, so conditional rendering applies. Buffer clears, and texture clears that cannot use an
    attachment, are transfers that conditional rendering does not cover; they run unconditionally and are counted
    with the copies (`unpredicated_copies`).
- **Layout tracking** (`LayoutTracker`): the current Vulkan layout of every image subresource, so barriers start
  from the real layout instead of the application's `before`. This is how implicit promotion and decay work: a
  texture in `COMMON` sits in `GENERAL`, which every promotable access can use, so a later barrier naming the
  promoted state still starts from `GENERAL`, and decay changes no layout. Buffers have no layout, and
  simultaneous-access textures stay in `GENERAL` for life. With `separateDepthStencilLayouts` the depth and stencil
  planes are tracked apart, so a depth-only clear or a one-plane transition leaves the other plane where it is.
- **Execution** (`runtime/src/vulkan_backend.h`): `VulkanContext` (registered resources, the tracker, a view cache),
  `VulkanQueue` (`ExecuteCommandLists` prepares every list before writing any, so a bad list submits nothing and
  leaves the tracker as it was; `Signal`; `Wait`, which is attached to the next submission because a Vulkan wait
  only holds back its own batch) and `VulkanFence` (a timeline semaphore).
- **COM** (`runtime/shims/d3d12_com.cpp`): `ID3D12GraphicsCommandList8` (the GCL7 chain plus
  `OMSetFrontAndBackStencilRef`), `ID3D12CommandQueue`, `ID3D12CommandAllocator` and `ID3D12Fence1` (with
  `SetEventOnCompletion` events). Every method they do not implement is a stub generated at build time from the
  SDK's `d3d12.h` by `tools/gen_d3d12_stubs.py`. The stub reports its name once, so the vtables follow the SDK
  order exactly and nothing derived from the SDK is checked in. Unknown IIDs are logged and refused. Resources
  are identified by their `ID3D12Resource` pointer, and RTV/DSV CPU handles point at a `TargetView`.
  `OMSetFrontAndBackStencilRef` records its two `UINT`s on the list (and `Reset` clears them), but nothing applies
  them yet: Vulkan can set the faces with `vkCmdSetStencilReference` and a face mask, but only a pipeline that
  consumes the dynamic state can, and this layer has no pipeline or draw path.
- **Device** (`runtime/shims/d3d12_vulkan_device.cpp`): an `ID3D12Device11` (PC ABI, 80 slots) that the Xbox-ABI
  proxy (`d3d12_xs_device.cpp`) forwards to, so slot 14's heap-handle conversion and slot 76's resource creation
  work unchanged. It creates the objects above, plus:
  - descriptor heaps whose slots are 32 bytes (`GetDescriptorHandleIncrementSize`; the console's increments are
    question 4). RTV/DSV slots hold a `TargetView`; shader-visible heaps return a GPU handle, but nothing reads it
    yet;
  - `CreateRenderTargetView` / `CreateDepthStencilView` for whole resources, `TEXTURE1D`, `TEXTURE1DARRAY` (D3D-63),
    `TEXTURE2D`, `TEXTURE2DMS`, `TEXTURE2DARRAY` and `TEXTURE2DMSARRAY` (a slice range; `ArraySize` -1 is every
    slice from the first on, as on the host, `d3d12_view_slices_test`);
  - `CopyDescriptors` / `CopyDescriptorsSimple`. A descriptor is the bytes of its slot, so a copy is a byte copy:
    the destination and source range lists are walked together, one descriptor at a time, and null size arrays
    mean one descriptor per range;
  - committed buffers (upload and readback heaps mapped persistently and zeroed), 1D textures and 1D arrays (render
    targets and depth-stencils too: their attachment views are `VK_IMAGE_VIEW_TYPE_1D_ARRAY`), and
    2D textures in the common render-target, depth and texture formats (typeless families become mutable-format
    images; `R24G8_TYPELESS` and `R32G8X24_TYPELESS` are depth-stencil images with or without
    `ALLOW_DEPTH_STENCIL`, as Vulkan has no colour format with their layout), plus non-attachment 3D textures with
    mipmapped upload, image copy and readback, and multisampled 2D textures (single-mip render targets or
    depth-stencils, in the sample counts the GPU has for the format);
  - packed `R9G9B9E5_SHAREDEXP` textures map to `VK_FORMAT_E5B9G9R9_UFLOAT_PACK32` for uploads and sampled views.
    When `ALLOW_UNORDERED_ACCESS` is requested on a GPU whose `VK_FORMAT_E5B9G9R9_UFLOAT_PACK32` lacks native
    `STORAGE_IMAGE` support (for example NVIDIA), the layer backs the texture with a mutable `VK_FORMAT_R32_UINT`
    image (`MUTABLE_FORMAT | EXTENDED_USAGE` + `VkImageFormatListCreateInfo` `{R32_UINT, E5B9G9R9_UFLOAT_PACK32}`),
    refusing `ALLOW_RENDER_TARGET`, `ALLOW_DEPTH_STENCIL` and `DENY_SHADER_RESOURCE`: `CheckFeatureSupport` advertises
    `TYPED_UNORDERED_ACCESS_VIEW` and `UAV_TYPED_STORE` (plus `UAV_TYPED_LOAD` with `R32_UINT` storage atomics) while
    keeping `RENDER_TARGET` off; SRV and UAV descriptors emit the `E5B9G9R9` T# (`FORMAT 132`, RGB1) and reject
    incompatible casts, while set 0 creates a `VK_IMAGE_USAGE_SAMPLED_BIT` `E5B9G9R9` sampled view and a
    `VK_IMAGE_USAGE_STORAGE_BIT` `R32_UINT` storage view (bound in `as_uint` / `r32ui`, never `as_float`);
    `ClearUnorderedAccessViewFloat` packs RGB9E5 with round-half-up (clamping negative/NaN to 0 and +Inf/overflow to
    65408, bumping on 512) and `ClearUnorderedAccessViewUint` writes 0, matching WARP (`d3d12_r9g9b9e5_uav_test`,
    `d3d12_uav_clear_test`);
  - the descriptor shader samples `B5G5R5A1_UNORM` through a 2D SRV. Opaque and transparent red check bit 15,
    and mixed-channel texels check the five-bit color fields;
  - `B8G8R8X8_UNORM`, `_UNORM_SRGB` and `_TYPELESS` (D3D-87) are `B8G8R8A8` images: clears, copies and footprints
    keep all four bytes (WARP writes the clear's alpha into the X byte too), and shader views read alpha as 1. Its
    typeless family is its own, so views to or from `B8G8R8A8` are refused, unless the texture's castable-format
    list names the view's format (relaxed casting, which WARP and AMD's D3D12 driver accept both ways). The view's
    format then decides the alpha: a `B8G8R8X8` view of a `B8G8R8A8` texture reads 1, a `B8G8R8A8` view of a
    `B8G8R8X8` one the stored byte. It decides the channel order too: an `R8G8B8A8` texture whose list names
    `B8G8R8A8` or `B8G8R8X8` reads red and blue swapped through those views, as on the host (both orders share one
    T# format code, so the T#'s swizzle carries the order). An RTV in a format the list names stores clears and
    draws in the view's channel order and encoding, as on the host (see the castable-list RTVs under D3D-275 below).
    As on the host runtime, the family has no typed UAV, and a typed `B8G8R8X8`
    texture cannot allow unordered access (the host accepts one whose castable-format list names `B8G8R8A8`; the
    layer doesn't yet). Blending into a `B8G8R8X8` render target reads destination
    alpha as 1, as WARP and AMD's D3D12 driver do: the PSO turns `DEST_ALPHA` into `ONE`, `INV_DEST_ALPHA` into
    `ZERO` and the colour `SRC_ALPHA_SAT` into `ZERO`, where the BGRA8 image would read the X byte. What a draw
    leaves in the X byte is not defined (WARP writes 0, AMD the blended alpha), and no view reads it;
  - `R10G10B10_XR_BIAS_A2_UNORM` maps to `VK_FORMAT_A2B10G10R10_UNORM_PACK32`, the same packed 10:10:10:2
    layout; the bias and scale are the application's shader work. Like the PC runtime, it accepts only a
    single-mip, single-sample 2D texture: extra mips, multisampling and 3D are refused in creation, allocation
    and footprints. Format support reports `TEXTURE2D` and `CAST_WITHIN_BIT_LAYOUT`, not the host's display,
    back-buffer-cast or video-processor bits;
  - `CheckFeatureSupport` for feature levels (up to 12_1), architecture, D3D12 options (`OPTIONS`, `OPTIONS1`–`5`,
    `7`–`9`, `11`–`21`), GPU virtual-address support, format support, format plane counts, multisample quality
    levels, shader model, root signature version, predication (`VK_EXT_conditional_rendering`), command-queue
    priority support, protected-resource session queries, shader cache, existing heaps, serialization, cross-node
    sharing, displayable resources, meta-command queries (`E_INVALIDARG`), application-specific driver state, and
    bytecode bypass hash support. `D3D12_OPTIONS8` reports unaligned block textures when the Vulkan logical device
    enables BC texture compression, and `D3D12_OPTIONS12` reports `RelaxedFormatCastingSupported = TRUE`, so a
    committed or placed texture accepts a castable-format list. Other queries (including `OPTIONS6` and `OPTIONS10`)
    are logged and refused until the trace shows what the game asks (question 11);
  - Single-sampled `A8_UNORM` 2D textures use one byte per texel in `VK_FORMAT_R8_UNORM`. Their SRVs expose
    (0, 0, 0, A) before applying the caller's component mapping. Format support includes `MIP` when the Vulkan
    image supports multiple levels. Render-target, depth-stencil and UAV creation are unsupported for this format.
  - reserved resources and `GetResourceTiling` (section 4.2);
  - query heaps for occlusion, timestamps and pipeline statistics, and `GetTimestampFrequency` on the queue
    (section 4.6);
  - `GetCopyableFootprints` for buffers, 1D and 1D-array textures, 2D textures and volumes, including BC1–7 and the
    two-plane depth-stencil formats.
    The test compares every output with the PC runtime's; that comparison found that planar
    depth-stencil rows are padded to 512 bytes, not 256.
  - the `1`/`2`/`3` variants as wrappers over the methods above (D3D-28): `CreateCommittedResource1/2/3`,
    `CreateCommandQueue1` and `GetCopyableFootprints1`. `D3D12_RESOURCE_DESC1` drops to `D3D12_RESOURCE_DESC`,
    and the initial state or `D3D12_BARRIER_LAYOUT` is not used, because new textures start `UNDEFINED` and move
    on first use. The newer arguments are answered as the PC runtime answers them. `d3d12_method_variants_test`
    compares each answer below with the host runtime live, measured on an RTX 4090 (driver 621.14):
  - `CreateCommandSignature` (`ID3D12CommandSignature`) and `ID3D12GraphicsCommandList::ExecuteIndirect` (including
    GPU-written `CountBuffer` via Vulkan 1.2 core `vkCmdDrawIndirectCount` / `vkCmdDrawIndexedIndirectCount`).
  - Optional `VK_KHR_acceleration_structure` + `VK_KHR_deferred_host_operations` + `VK_KHR_ray_query`:
    `ID3D12Device5::GetRaytracingAccelerationStructurePrebuildInfo`, `ID3D12GraphicsCommandList4::BuildRaytracingAccelerationStructure`
    (BLAS + TLAS), `vkGetBufferDeviceAddress` in `ID3D12Resource::GetGPUVirtualAddress()`, and RT-aware `UAV`
    barriers (cleanly disabled/skipped on GPUs without RT).

- **DXGI Swapchain** (`runtime/shims/dxgi_swapchain.h`, `runtime/shims/dxgi_com.{h,cpp}`): `IDXGIFactory4` and
  `IDXGISwapChain3` over `VK_KHR_win32_surface` + `VkSwapchainKHR`, exposed via `rt::d3d12::create_dxgi_factory`.
  Swapchain backbuffers are real `ID3D12Resource` COM objects (`BackBufferResource` inheriting from
  `ID3D12ResourceStubs`) registered directly with `VulkanContext`, supporting `CreateSwapChainForHwnd`,
  `GetCurrentBackBufferIndex`, `GetBuffer`, `CreateRenderTargetView`, `Present`, and `ResizeBuffers` (preserving
  explicit buffer dimensions even when the HWND client size is unchanged via an offscreen backbuffer + blit path,
  and exposing exactly the requested `BufferCount` even when Vulkan's native image count differs, in which case the
  offscreen ring is the app-visible swapchain and each present blits the current slot into the acquired image).
  The factory answers up to `IDXGIFactory7` (D3D-2). `EnumAdapterByGpuPreference` returns the one Vulkan adapter
  at index 0 for every preference. It accepts a preference outside the enum, as the host does, and gives
  `DXGI_ERROR_INVALID_CALL` for a null `out` and `DXGI_ERROR_NOT_FOUND` past the end. The adapter list never
  changes, so `RegisterAdaptersChangedEvent` never signals its event. It still gives the host's answers: a null
  event or cookie, or an event that is already registered, is `DXGI_ERROR_INVALID_CALL`. Cookies start at 1 and
  are separate from the occlusion cookies, and an unknown cookie is `DXGI_ERROR_NOT_FOUND`. All of these were
  measured on the host's `dxgi.dll` (Windows 11, RTX 3070) and are checked in `dxgi_swapchain_test`.
  explicit buffer dimensions even when the HWND client size is unchanged via an offscreen backbuffer + blit path,
  and exposing exactly the requested `BufferCount` even when Vulkan's native image count differs, in which case the
  offscreen ring is the app-visible swapchain and each present blits the current slot into the acquired image).

    | Argument | Answer |
    |---|---|
    | a protected resource session | `E_INVALIDARG`; the layer reports no `D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_SUPPORT` |
    | `SamplerFeedbackMipRegion` on a texture | `E_INVALIDARG` from `CreateCommittedResource2/3`; `GetCopyableFootprints1` returns its failure answer (every output all ones) |
    | `SamplerFeedbackMipRegion` on a buffer | ignored |
    | a buffer's initial layout other than `UNDEFINED` | `E_INVALIDARG` |
    | a castable-format count with a null list | no list (`S_OK`) |
    | a castable-format list | accepted on a texture when the formats map to Vulkan formats: the image is mutable and carries a `VkImageFormatListCreateInfo`; a buffer or an unmapped format is `E_INVALIDARG`. A cast whose texel size differs from the resource's is also `E_INVALIDARG` — the host refuses other sizes, and a size-mismatched format list is a Vulkan validation error (`VUID-VkImageCreateInfo-pNext-06722`). The list is kept on the resource, so an SRV whose format is neither the resource format nor in the list leaves a null descriptor. The resource's own format is always accepted and is added to the Vulkan format list when the D3D12 list omits it. A listed same-size cross-family cast, such as `R8G8B8A8_UINT` to `R32_UINT`, is accepted. An RTV in a listed RGBA8 family format on a typed `R8G8B8A8_UNORM` texture clears through the cast, the same reinterpretation a typeless RTV uses; without a list a typed resource still refuses a different-format RTV. `D3D12_OPTIONS12` reports `RelaxedFormatCastingSupported = TRUE` (this layer requires Vulkan 1.3, where the format list is core). `CreatePlacedResource2`'s list reaches its placed image through the same rules (D3D-90) |
    | `CreateCommandQueue1`'s creator ID | ignored |

    `CreatePlacedResource1/2` wrap the placed path above; `2` carries the castable-format list. `CreateHeap1`
    remains a stub.
  - `CreateSampler2` as a wrapper over `CreateSampler` (D3D-28, part 2). Without flags it writes the same S#.
    `D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR` reads the union as integers. `(0,0,0,0)`, `(0,0,0,1)` and `(1,1,1,1)`
    take the S#'s transparent-black, opaque-black and opaque-white types. The S# has no separate integer types,
    since the hardware reads them by the image's format. Vulkan does not: an integer image sampled with a
    `VK_BORDER_COLOR_FLOAT_*` border reads 0. Vonsek found this on an RTX 5080 (#538), and `d3d12_descriptor_test`
    reproduces it on an RTX 4090 when the layer uses the float enums. So the layer marks an integer border in the S#
    with `BORDER_COLOR_PTR = 0x800` (`gfx10::kIntegerBorderPtr`). The hardware reads that pointer only for a table
    colour, and Mesa writes it only for `SQ_TEX_BORDER_COLOR_REGISTER`. The mark gives the integer border its own
    key in the descriptor hash table, so it does not share the float border's `VkSampler`, and `vulkan_sampler`
    makes it `VK_BORDER_COLOR_INT_*`. Any other colour is a table colour (`BorderColor::table`). With
    `VK_EXT_custom_border_color` (D3D-73), the layer registers custom float and UINT border colours in a device-wide
    table of up to 4000 entries; the table index is stored in `BORDER_COLOR_PTR [11:0]` of the S#, so distinct custom
    colours produce distinct keys in the descriptor hash table and distinct `VkSampler` objects. On sampler creation,
    `VkSamplerCustomBorderColorCreateInfoEXT` passes the colour components with `VK_BORDER_COLOR_FLOAT_CUSTOM_EXT` or
    `VK_BORDER_COLOR_INT_CUSTOM_EXT` and `format = VK_FORMAT_UNDEFINED` (via `customBorderColorWithoutFormat`). Without
    the extension, custom colours fall back to transparent black with a diagnostic log line, using
    `VK_BORDER_COLOR_INT_TRANSPARENT_BLACK` for integer borders and `VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK` for
    float borders to satisfy Vulkan input validation rules on integer views. An integer border colour
    with the bits of `1.0f` is not white and receives a table colour index.
    `D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES` sets `FORCE_UNNORMALIZED` (word 0, bit 15) and changes nothing
    else; `vulkan_sampler` makes it `unnormalizedCoordinates`. The layer answers `D3D12_OPTIONS17` with
    `NonNormalizedCoordinateSamplersSupported` (core Vulkan, with the same restrictions) and no
    `ManualWriteTrackingResourceSupported`. That is what the RTX 4090 and WARP report. The runtime on a Radeon 680M
    (D3D12 driver 31.0.14084.6002) reports no non-normalized samplers, and leaves every non-normalized description
    unwritten, including those the table below says are written.

    A description the PC runtime refuses leaves the slot as it was. The runtime itself does this, without the
    debug layer, on the RTX 4090 (driver 621.14) and on WARP; the debug layer's message names the rule.
    `d3d12_sampler2_test` compares each row with the host runtime live:

    | `CreateSampler2` input | Answer |
    |---|---|
    | a null description | the slot is left as it was |
    | a flag other than the two | the slot is left as it was |
    | non-normalized, with MIN ≠ MAG, a linear MIP filter, anisotropy or comparison | the slot is left as it was; `MINIMUM`/`MAXIMUM` filters are accepted |
    | non-normalized, with `AddressU` or `AddressV` not `CLAMP`/`BORDER` | the slot is left as it was; `AddressW` is free, as in Vulkan |
    | non-normalized, with `MinLOD`, `MaxLOD` or `MipLODBias` not 0 | the slot is left as it was; `-0` is 0 |
    | non-normalized, a point filter with `MaxAnisotropy` or `ComparisonFunc` set | written; both are ignored (the debug layer warns about `ComparisonFunc`) |
    | both flags | written |

  BC1–7 textures need `textureCompressionBC`, which the device enables when the GPU has it, as it does
  `occlusionQueryPrecise` and `pipelineStatisticsQuery` for the query heaps.
- **Presentation** (`leonida::VulkanPresenter` in `runtime/src/vulkan_present.h`, which
  `rt::d3d12::create_window_presenter` makes for a device and a window). This is the part the
  `PresentX` front end uses; which of DXGI or `PresentX` the game uses is still open (question 6). It takes a back
  buffer, an ordinary texture in the D3D12 `PRESENT` state, and blits it to an acquired `VkSwapchainKHR` image,
  scaling when the window size differs. The blit goes on the same `VkQueue` after the application's work, so the
  frame contains everything submitted before the present. Details:
  - two frames in flight, and a present semaphore per swapchain image;
  - FIFO when vsync is asked for, otherwise MAILBOX or IMMEDIATE; with `VK_KHR/EXT_swapchain_maintenance1` the FIFO
    swapchain can also present one frame in IMMEDIATE or FIFO_RELAXED (`present(..., tear)`, section 4.5);
  - the swapchain is recreated on resize or when the surface reports it out of date or suboptimal, and a minimised
    window skips presents;
  - a back buffer that was never written is zero-filled like any other texture;
  - `presented()` counts the presents the engine took (`VK_SUCCESS` or `VK_SUBOPTIMAL_KHR`), not ones it reported
    out of date;
  - a recreation is published only when the swapchain, its images and every per-image semaphore exist. On a
    failure the partial objects are released and the next present tries again, so no image is acquired without
    its semaphore;
  - if the blit's submission fails after an image was acquired, an empty batch takes the acquire semaphore and
    signals the frame's fence, and the swapchain is recreated next time to get the image back. The next present
    therefore does not wait forever on a fence nothing signals. If even that batch fails, or a fence wait or reset
    fails, the presenter stops (`ok()` is false) and its owner makes a new one.
  - for `PresentX` (section 4.5): a source rectangle, a blank present, an sRGB swapchain format for an sRGB source,
    and an optional capture of each presented frame to host memory (what `xbox_present_test` reads back).

  The device needs `VK_KHR_swapchain` (and the instance the surface extensions). The standalone device enables
  them when present.

  New textures are not touched at creation. They are tracked as `UNDEFINED`, and the first command that uses a
  part of one zero-fills it (D3D12 committed resources start zeroed; `CREATE_NOT_ZEROED` skips this) and moves it
  to the layout that command expects, so creation needs no submission. Block-compressed textures are not
  zero-filled, because Vulkan cannot clear them; they start undefined until the application uploads them. Each
  submission ends with a barrier to the host, so a readback buffer holds the GPU's writes once the fence wait
  returns. Children hold a reference to the proxy, not
  the inner device, so `GetDevice` always returns the Xbox-ABI object. Each live D3D12 queue leases a distinct
  `VkQueue` while one is free (its own type's family first, then the graphics family); the standalone device
  requests up to 16 per family. The queues still share a submit lock for
  replay state, but Wait-before-Signal between them can make progress (D3D-97), and a batch that waits for a Signal
  not yet submitted is held on the CPU so layouts follow dependency order (section 4.4). Caller-owned Vulkan devices can
  supply additional handles through `VulkanDeviceInfo::command_queues`; without them they retain legacy single-queue
  behavior. When every handle is in use, the new D3D12 queue shares the `VkQueue` that has the fewest D3D12 queues
  (section 4.4, "More D3D12 queues than Vulkan queues"); it used to fail with `E_OUTOFMEMORY`.
  Dedicated compute/transfer
  *families* would also require resource queue-family ownership handling and remain future work.

Descriptors and the bindless set (milestone M2, without the pipelines that will bind the set), from the shader's
side inwards:
- **`runtime/src/vulkan_bindless.h`**: set 0 includes the twenty bindings the translator declares, and
  `DescriptorTable` (`runtime/src/descriptor_table.h`, section 5.1) behind binding 2. One set for the device.
  The translator declares every array as a runtime array, so the counts are the runtime's choice. Details:
  - The device sizes the arrays once, when it creates the set. It does not recreate the set, because pipeline
    layouts are built on `bindless_set_layout()`. `bindless::size_bindings` reads the update-after-bind limits
    (`maxPerStageDescriptorUpdateAfterBind*`, `maxDescriptorSetUpdateAfterBind*`,
    `maxPerStageUpdateAfterBindResources`, `maxUpdateAfterBindDescriptorsInAllPools`), or the plain limits for a
    set without update-after-bind. It starts from the default counts below (4096 sampled 2D images, 1024
    samplers, 1024 2D-array, 256 1D, 256 3D and 256 per storage array). Where a limit is lower, it clamps the
    sampler, storage and 2D arrays, in that order, and the device logs it. Next come the MS pairs that fit. Then it
    grows the arrays, in the same order, towards the maximums:
    - samplers: up to 4096, and at most `maxSamplerAllocationCount` less 64 for samplers outside the set. That
      covers a full 2048-sampler D3D12 heap plus room for literal and static S#s.
    - 2D images: up to 65536.
    - storage arrays: up to 4096 each.

    Growth leaves 64 descriptors of each limit to the other sets of a pipeline layout. It also stays within
    `maxPerSetDescriptors`, which never shrinks the default counts, because a layout past it may still be
    supported: past it, the device grows only where `vkGetDescriptorSetLayoutSupport` accepts the grown layout.
    Intel's UHD 770 (101.7092) reports 6159 and accepts the full sizes, so it gets the same set as an RTX (D3D-219).
    MoltenVK (1.4.2 and 1.4.3 on an M3 Max) reports 1212 and accepts only layouts with fewer descriptors than that
    (`MVKDevice::getDescriptorSetLayoutSupport` compares the sum with it), although it creates and samples a set of
    65536 2D images and 4096 samplers correctly (a native repro, and `d3d12_descriptor_test` under Wine with the set
    grown). So on MoltenVK the D3D12 device doesn't ask: the per-type update-after-bind limits, which MoltenVK takes
    from Metal's argument-buffer limits (500000 samplers on an M3 Max), decide alone, and the set grows as on other
    GPUs. Where a set still ends up short of a full 2048-sampler D3D12 heap, the device logs that at startup. Past
    the last free index, `CreateShaderResourceView` and `CreateSampler` log once that the array is full, and a
    shader reads the fallback. `d3d12_descriptor_test` then fills each short array to its last index and checks that
    the next view and sampler are refused.
    The other arrays keep their default counts. On an RTX 5070 (driver 621.24, 1048576 of each type
    and 4000 samplers) the set has 65536 2D images, 3936 samplers and 4096 storage indices in 2D: 93793
    descriptors, 3-6 MiB at 32-64 bytes a descriptor. The device prints the counts it chose, and
    `bindless_sizing_test` checks the rules on synthetic limits.
  - An image has one index in its dimension (2D, 2D array, 3D, 1D, 2DMS or 2DMS array), shared by the
    bindings for that dimension. The runtime adds sampled-image bindings in pairs for 2DMS and 2DMS array:
    20/21 for floating-point and normalized color, 22/23 for unsigned integers, and 24/25 for signed integers.
    The runtime enables pairs that fit the device's descriptor limits and have a multisampled zero fallback
    of the matching numeric class. A missed key reads zero at texel (0, 0), layer 0, sample 0.
  - The storage arrays are shorter (`BindlessSet::storage_capacity`). Each array holds at most as many images
    as its dimension, and the 2D r32ui array is the same size as the other 2D storage arrays. So an image with a
    storage view gets an index the storage arrays hold, and a read-only one goes above where there is room.
  - Index 0 of every array is a fallback (a 1x1 image of zeros, a nearest clamping sampler), because a miss is
    index 0.
  - Changes are queued and written by `flush()`, one write a slot, so a view that is removed before the next
    flush is never written.
  - `flush()` also uploads the table to its buffer: device-local memory, written from a mapped staging copy when
    the buffers are created (first use or growth) and with `vkCmdUpdateBuffer` for slot changes. Adding a key only
    puts a new entry in an empty slot, and the copy of an entry is not atomic; a running lookup that does not know
    the key cannot resolve to it, which is accepted. Erasing a key empties its slot and moves no other entry
    (every lookup, CPU and shader, compares all eight probe slots and never stops at an empty one), so it is
    uploaded like an add. Replacing a live key's index, an insert that relocates a live entry, and a table that
    has grown (a new buffer behind binding 2) change what running lookups see: `flush_replaces_table()` says when a
    flush would do one of those, and the flush then makes a new generation instead of writing in place (D3D-314; see the
    paragraph on the flush at every `ExecuteCommandLists`, below). Erasing used to compact the probe cluster, which
    moved other keys and made every texture release wait for the whole GPU.
- **`runtime/src/gfx10_descriptors.h`**: the T#, S# and V# structures (section 5.3), encoded and decoded, with
  their mapping to Vulkan formats, view types, component mappings and samplers. Field positions are Mesa's
  (`src/amd/registers/gfx10-rsrc.json`) and agree with what the translator and the GPU tests already use. Bits
  that only the console's driver could give a meaning (tiling, compression metadata, performance hints) are
  written as zero.
- **Device**: `CreateShaderResourceView` and `CreateUnorderedAccessView` of textures write a T# into the slot,
  `CreateSampler` an S#; each then makes sure set 0 has the view or sampler and the table has the key. Details:
  - The key is the descriptor's bytes, so a copied descriptor needs nothing more, and an unordered-access view
    and a shader-resource view of the same mip are one entry.
  - A D3D12 comparison sampler with function NEVER and an ordinary sampler both encode zero in the hardware
    depth comparison field. The layer marks the NEVER comparison in S# word 3, bit 12 (unused with
    DERIV_ADJUST_EN clear) to give it a separate lookup key and a comparison-enabled Vulkan sampler.
  - A view the layer cannot show (below) leaves a null descriptor, all zeros, which finds the fallback.
  - A typed multisampled color texture has one mip. Its default SRV covers every layer. An explicit
    `TEXTURE2DMS` view reads layer zero, while `TEXTURE2DMSARRAY` selects a range starting at
    `FirstArraySlice` (`ArraySize = UINT_MAX` selects the remaining layers). The format-support query
    advertises `MULTISAMPLE_LOAD` when the format can be sampled with MSAA and its binding pair is available.
  - A texture is named in its T#s by an address of its own, above 4 GiB and 256 bytes apart, that nothing
    dereferences; its entries leave the set and the table when it is destroyed.
  - `VulkanDeviceInfo::bindless` says the Vulkan device has the features this needs: `runtimeDescriptorArray`,
    dynamic indexing of sampled- and storage-image arrays, and update after bind for sampled images, storage
    images and storage buffers. The standalone device enables them where the GPU has them.
  - Every `ExecuteCommandLists` flushes the set before its commands are written.
  - `rt::d3d12::bindless_set` and `bindless_set_layout` hand out the set and its layout: what a pipeline whose
    shader declares set-0 bindings is created with and binds. Nothing in the layer creates such a pipeline yet.
- **Layouts.** A descriptor in set 0 has one image layout for as long as it lives, but D3D12 lets a texture be
  sampled in a shader-resource state and also while promoted from `COMMON`, which is `GENERAL` here (section
  4.3). So with the bindless set, a texture's shader-resource states use `GENERAL` too
  (`ResourceBinding::sampled_in_general`), and every descriptor is written with `GENERAL`. This follows from the
  Vulkan rule that a descriptor's layout matches the image's; no test here can see it broken, because the
  validation layer does not follow which element of a runtime array a shader reads. A depth texture sampled
  while it is also bound read-only (`DEPTH_READ` with a shader-resource state) is in
  `DEPTH_STENCIL_READ_ONLY_OPTIMAL` and is not handled.

A typed single-sampled 3D texture can use a default SRV or an explicit `TEXTURE3D` SRV of the same format.
An `R8G8B8A8_TYPELESS` volume supports explicit `R8G8B8A8_UNORM` and `R8G8B8A8_UNORM_SRGB` SRVs.
The sRGB view decodes RGB and preserves linear alpha. A typeless volume requires an explicit typed view.
The T# records the base depth, the view's mip range and `ResourceMinLODClamp`. Binding 6 holds the 3D Vulkan view,
which has one array layer. `d3d12_descriptor_test` checks samples at four W coordinates through two views of the same
volume, plus minimum-LOD clamped views (inside the mip range via `VK_EXT_image_view_min_lod`, at or below
`MostDetailedMip`, and above the last mip).

Known gaps in the descriptor path: buffer shader-resource views and constant-buffer views (the V# a shader would
load for them; the structure is in `gfx10_descriptors.h`, the views are not written; buffer UAVs of every kind are,
section 3.4 "Buffer UAVs", `d3d12_m4_compute_test` and `d3d12_buffer_uav_test`);
1D-array SRVs, 1D and 3D storage views, other typeless volume formats, multisampled depth/stencil and typeless
color SRVs, and shader-translator lowering for multisample loads. Single-sampled depth-plane views (`R24_UNORM_X8_TYPELESS`,
`R32_FLOAT_X8X24_TYPELESS`) are a T# of the texture's depth format (8_24_UNORM, X24_8_32_FLOAT) through the
depth aspect (the texture's backing format, so X24_8_32_FLOAT for a D24 texture where the layer stores it as
`D32_SFLOAT_S8_UINT`), each on its own family only (the other family's is a null
descriptor, as D3D12 refuses it), and stencil-plane views (`PlaneSlice` 1, as
`X24_TYPELESS_G8_UINT` or `X32_TYPELESS_G8X24_UINT`) are an 8_UINT T# that set 0 shows through the stencil
aspect, in its float-typed sampled arrays like other single-sampled UINT views. Other gaps include descriptors
the game writes itself, which need
the execute-time scan of section 3.4 (only descriptors made through the device are registered); the literal S#s of shaders (`literal_samplers` in the index),
which `rt::d3d12::register_samplers` registers without a heap slot but which nothing passes to it yet (the call at
pipeline creation comes with the shader index, #291);
custom border colours without VK_EXT_custom_border_color, and minimum/maximum filters, anisotropy or mirror-once addressing without the Vulkan
feature, each logged and approximated (note: in Vulkan, `VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT` is
per-format; because bindless set 0 hashes `T#` and `S#` independently in the shader, `bind_sampler` cannot know
which image format a sampler will be paired with — on Radeon iGPUs `BC*` formats lack `MINMAX` while uncompressed
formats support it, whereas RTX supports all queried formats); the storage arrays hold 255 images a dimension at the default counts, more
where the device has room (4095 2D and 1023 2D-array on an RTX 5070); nothing writes through a
storage view in a D3D12-level test yet (`vk_bindless_test` goes through the storage bindings at the Vulkan
level). The set is flushed at every `ExecuteCommandLists`. A flush that moves the table to a larger buffer, changes
the index of a live key or relocates one (`BindlessSet::flush_replaces_table()`) does not write in place and does not
wait (D3D-314; D3D-76 first waited for the whole device, which stalled every queue and would have at every texture
release of a game that churns textures). It makes a new generation: new table buffers (the whole table uploaded to
them in that submission's command buffer), a new one-set descriptor pool and set (binding 2 names the new table, and
every descriptor written so far is written again from a CPU shadow of the set, in runs, so unwritten elements stay
unwritten under `partially_bound`), and replay binds that set at the dispatches and draws of the submission
(`ReplayFunctions::bindless_set`; the pipelines' own `bindless_set` only marks that they use one). Submissions made
before keep reading the old set and table, which nothing writes again. The old generation is stamped with the
latest submission ticket and freed at a later `ExecuteCommandLists` (and at device teardown) once
`completed_retire_ticket()` has reached it. Those tickets are the layer's per-submission records of (queue timeline
semaphore, value): the stamp is therefore the latest submitted value of every queue that could have bound the
set, and the free waits until each of those timelines has passed its value; nothing is waited for. The completed
prefix is global and in order, so a queue blocked on a fence only delays the free, never another queue. The stamp
is taken under `tracker_mutex_`, which `execute()` holds from the flush to the end of its submit, so no submission
that bound the old set is unsubmitted when it is read. `LEONIDA_BINDLESS_IDLE_WAIT=1` restores the old wait
(a counted `vkDeviceWaitIdle` before the flush) to compare frame times; nothing else uses it. The cost of the new
path is a full upload of the table and a rewrite of the set's descriptors per replacing flush, where the old one
uploaded the changed slots after the stall; `d3d12_bindless_churn_test` prints both.

The pattern is the one vkd3d-proton uses to retire GPU-visible objects without a device idle (approach only, no
code taken): stamp the object with what every queue has submitted and free it at a later submit once those
submissions have completed. Two further rules come with it here. The submission that carries a new generation's
table upload makes every other queue's next submission wait for its timeline value (`bindless_upload_semaphore_`,
like the storage-image initialisation), because that submission is the first to bind the new set and a queue with no
D3D12 fence on it could otherwise run before the copy; the table's buffers need no ownership transfer because every
D3D12 queue of the device is made in `info.queue_family`. And at most 8 retired generations are kept: past that, the
next flush waits (logged once, with the descriptor lock released) for the submissions up to the oldest one's ticket,
which is a wait for those queues' timelines and not for the device. The commit of a submit's result runs under
`tracker_mutex_`, so a later flush never sees an earlier submit's outcome unrecorded. If the new generation cannot be
made (out of memory) the flush leaves the in-use set and the pending writes alone and fails.
Two bindless hazards that vkd3d-proton documents for games and that this change does not remove: descriptors
written for resources allocated after the submit that reads them, and an image and a buffer descriptor aliasing one
heap slot (vkd3d-proton PR #3033, https://github.com/HansKristian-Work/vkd3d-proton/pull/3033). Future path, not
implemented: `VK_EXT_descriptor_heap` (https://www.khronos.org/blog/vulkan-introduces-roadmap-2026-and-new-descriptor-heap-extension)
would replace descriptor sets with application-sized heaps.

A typed 3D color target supports default and explicit RTVs. The default covers mip 0; `FirstWSlice` and
`WSize` select the depth slices in an explicit view's mip. `d3d12_clear_test --volume` checks full and
rectangular clears of overlapping slice ranges against both Windows WARP and expected RGBA bytes.

The compute path, end to end:
- `runtime/src/shader_library.h` finds the translated SPIR-V for a shader the game hands to D3D12 (section 3.6).
  It extracts the FSSM program from the DXBC container (DXIL → XDAT entry 0x0D → XBCH → zlib) and looks up its
  SHA-256 as `fssm_sha256` in `spirv/index.json`. Its own inflate, SHA-256 and JSON reader keep it free of
  dependencies (`shader_library_test`).
  - A container with two different 0x0D payloads (195 of 26,503) is read at the last one, as
    `tools/fssm_unpack.py` does by default and so as `fssm_sha256` was computed.
  - Index entries that share a program are all kept (789 groups of 1,182 entries in the corpus). A PSO takes the
    first entry that fits (for a VS: whose input signature the input layout feeds) and declares no set-0
    bindings (the cheaper pipeline: a binding-free entry has no set 0 in its layout).
  - A malformed index (bad numbers or `\u` escapes) is refused with an error instead of throwing into the shim,
    and an integer field outside 0..2^32−1 reads as 0.
- Root signatures use the model already on `main` (`runtime/src/root_signature.h`, `rt::rootsig`): `parse_rts0`
  reads the serialized blob and `pack()` lays out the 32 user-data dwords by the section 3.5 default mapping,
  refusing anything that would spill. `pack_stage()` lays them out in the console layout of one stage (section 3.5,
  D3D-448); the compute bind point uses it.
- **GPU address arena** (section 3.3, when the device has `bufferDeviceAddress`): one host-visible,
  device-addressable buffer, 256 MiB by default (`LEONIDA_ARENA_MB`, at most 4095 MiB, so every VA fits 32-bit
  table pointers).
  - It is also no larger than the device's `maxBufferSize` (Vulkan 1.3) and, counting the buffer's memory
    requirement, `maxMemoryAllocationSize` (Vulkan 1.1), whichever is smaller (`runtime/src/arena_size.h`). When a
    limit cuts the request, one log line names the limit and the size used. `arena_size_test` (CPU) checks the
    decision on synthetic limits; `d3d12_compute_test` checks the arena against the device's limits and a simulated
    32 MiB `maxMemoryAllocationSize`.
  - The RX 6800 (RDNA2, AMD driver 26.11.2, Windows 11, measured 2026-11-22) reports 2 GiB for both, so
    `LEONIDA_ARENA_MB` above 2048 gets a 2048 MiB arena there.
  - Before the cap, that driver did create a 3000 MiB arena. Buffers placed past 2 GiB in it (VA 0x89820000) were
    written by a compute shader and read back correctly, but the validation layer reports the allocation as over
    `maxMemoryAllocationSize`. Nothing relies on this out-of-spec behaviour.
  - The device logs the arena's high-water mark when it is destroyed: the peak end offset against the size, the
    most bytes live at once (by use: buffers, descriptor heaps, shaders, heaps), allocations and refusals (D3D-55).
    `LEONIDA_ARENA_LOG=1` also logs one line per VA handed out, per refused allocation, and per upload or readback
    buffer VA outside the arena.
  - Default-heap buffers are 64 KiB-aligned ranges in it, zeroed through its mapping; their console VA is the offset.
  - Shader-visible descriptor heaps also live in it: the CPU handle is its host mapping and the GPU handle is the VA,
    so descriptor bytes the application writes itself are what shaders load.
  - Upload and readback buffers keep their own memory, and their VA is their device address minus the heap base.
  - Shader programs (FSSM blobs) are copied in as the code base.
- **`CreateRootSignature`** parses the blob with `parse_rts0` and keeps each parameter's place in two layouts:
  `pack()` for the graphics bind point and `pack_stage(Stage::Compute)` for the compute one (from s4, one dword per
  table; section 3.5, D3D-448). A signature neither can place (more than 32 user dwords: the spill layout is open; the
  translated shaders have no below-base spill-pointer candidates) is refused with both reasons; one only a single layout places is created with a log line, and binding it at the other bind point
  invalidates the list. Before
  that it answers as the host runtime does (`compare_invalid_arguments`, "root signature:", RTX 5070 / 621.14):
  `S_FALSE` for a null output whatever the blob, then `E_INVALIDARG` for a node other than node 0 or a DXBC
  container whose size (the dword at +24) is not the size passed.
  **`CreateComputePipelineState`** (PC slot 11, Xbox slot 62 through the proxy) finds the shader by its FSSM
  hash, builds a pipeline with the 144-byte push-constant range and registers it for replay. Shaders that declare
  set-0 bindings get the device's bindless set 0 in their layout and bound at every dispatch (M2); on a device without
  that set they are refused with a log.
- **Command list**: `Set{Compute,Graphics}RootSignature` (a different signature clears the arguments, as D3D12
  leaves them undefined; setting the bound one again keeps them, as D3D12 does), `SetPipelineState`, the 32-bit
  constant, root view and descriptor-table setters, `SetDescriptorHeaps` and `ClearState`. A wrong parameter type,
  out-of-range constants, or an invalid `SetDescriptorHeaps` call makes `Close` return `E_INVALIDARG`.
  - `SetDescriptorHeaps` tracks the currently bound shader-visible `CBV_SRV_UAV` and `SAMPLER` heaps and their
    arena GPU base VAs, holds references to them until `Reset` or list destruction, unbinds omitted types (clearing
    both on `ClearState` and `Reset`), refuses `COPY` lists, > 2 heaps, null arrays/elements, non-shader-visible or
    `RTV`/`DSV`/duplicate-type/foreign-device heaps, and enforces bundle matching against the parent list (`parent`'s
    currently bound heap for each non-null heap the bundle specifies, `d3d12_set_descriptor_heaps`).
  - The list holds a reference to every root signature and pipeline state it is given, until `Reset` or its own
    release. D3D12 lets an application release them right after setting them, so the root layout the setters read
    and the pipeline `ExecuteCommandLists` looks up must outlive that. Keeping them alive while the GPU runs the
    submission stays the application's job, as in D3D12.
  `Dispatch` records the block as it is at that moment. Replay binds the pipeline, pushes s0..s31, the heap base
  and the code base, and dispatches.
- **Graphics pipelines and draws (M1).** `CreateGraphicsPipelineState` (PC slot 10) finds the VS and PS by FSSM hash.
  - **Vertex input:** each input-layout element is matched to the VS input signature by semantic name and index,
    and its location is the element's register (the translator's vertex-fetch rule). An input the layout leaves
    unfed is refused, as D3D12 refuses it.
  - **Fixed-function state:** rasterizer, blend and depth-stencil state map one to one. Viewport, scissor,
    topology, stencil references and the blend factor are dynamic, and so are vertex strides when the pipeline has
    vertex bindings. A pipeline without them (a full-screen pass from `SV_VertexID`) does not declare the stride
    state, since nothing would set it (`VUID-vkCmdDraw-pStrides-04913`).
    `d3d12_state_sweep_test` checks the mapping against the host on the same adapter: 482 full-screen draws over
    every blend factor pair, blend operation, write mask, comparison function and stencil operation, with the stencil
    plane read back, are byte-identical on an RTX 5080 and a Radeon iGPU.
  - **Logic ops** come from `RenderTarget[0]`, as in D3D12, and apply to every target (Vulkan has one per
    pipeline; it skips float formats). **`DepthClipEnable = FALSE`** is depth clamping: D3D12 clamps output depth
    to the viewport's range anyway.
  - **Conservative rasterization (Tier 1/2):** the standalone device enables `VK_EXT_conservative_rasterization`
    when available; caller-owned devices must set `VulkanDeviceInfo::conservative_rasterization` only if they
    enabled it. OPTIONS reports at least Tier 1 only with at least eight subpixel precision bits and a finite, nonnegative
    primitive overestimation no greater than `0.5 - 1/512` pixels, leaving half a minimum grid step within the
    Tier 1 uncertainty bound. `ConservativeRaster=ON` uses overestimate mode with zero extra overestimation,
    including unbound-attachment variants. OFF keeps the ordinary pipeline. Missing support, unknown modes,
    wireframe and non-triangle PSOs are refused; conservative TIR (`ForcedSampleCount != 0`) remains `E_NOTIMPL`.
    Tier 2 additionally requires `degenerateTrianglesRasterized` and primitive overestimation plus half a
    device subpixel step no greater than 1/256 pixel. Other eligible devices retain Tier 1. The standalone
    device does not expose Tier 3: the Xbox `SV_InnerCoverage` input mapping has not been confirmed. See
    [Microsoft's rules](https://learn.microsoft.com/en-us/windows/win32/direct3d12/conservative-rasterization)
    and [Vulkan's properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceConservativeRasterizationPropertiesEXT.html).
    `d3d12_conservative_tier2_test` checks the tighter geometric uncertainty bound, post-snap and exact
    degenerates with distinct indices, provoking-vertex colour, constant depth, back-facing classification,
    culling, indexed permutations, triangle strips, scissor and unbound-attachment variants. On RX 5700 XT
    (Windows driver 32.0.21045.5002), 16 tight-bound scenes and 480 degenerate draws pass across Vulkan and native
    AMD D3D12. The WARP comparison omits only its pixel-corner fixture: WARP drops that tiny triangle even though
    Vulkan and native AMD cover it. `--warp-corner` retains the strict failing software-reference reproducer;
    Vulkan always checks the corner. Variable `SV_Position.z` is not a provoking-vertex oracle: both native
    references and Vulkan exhibit extrapolation for screen-space degenerates. This test checks constant depth
    and separately varying colour, and does not claim a depth-buffer readback or variable-depth conformance.
    `d3d12_conservative_raster_test` checks geometric required/forbidden coverage, subpixel/sliver and degenerate
    triangles, culling, scissor, 1x/4x MSAA, variants, feature reporting and refusals. Uncertain edge pixels are
    allowed to differ. On RX 5700 XT (Windows 32.0.21045.5002), 48 scenario pairs pass on Vulkan and native AMD;
    24 single-sample pairs pass on WARP. WARP's 4x subpixel fixture covers outside the geometric bound; it remains
    reproducible with `--warp-msaa4`, separately from CTest's 1x software comparison. Vulkan's 4x oracle remains
    strict. On RTX 3060 (Windows 11, NVIDIA 610.88), 48 pairs pass on Vulkan and native NVIDIA. The oracle judges
    the layer alone. A native render that fails it is printed as "host not comparable", with its normal and
    conservative pixel counts, and that scenario isn't compared with the host. For example, Windows 10's WARP
    (`d3d10warp.dll` 10.0.19041, the host CTest selects with `LEONIDA_HOST_ADAPTER=warp`) misses pixel (4,4) of
    scene 4 with conservative rasterization on: seen on an RTX 5060 Ti and an RTX 2060 machine, both of which pass
    with their NVIDIA GPU as the host (TEST-907).
    The GPU test needs the current graphics path's 144-byte push-constant support; CPU capability checks
    do not need a GPU. No game-pass usage is claimed by these synthetic tests.
    `d3d12_conservative_tier3_reference` is a native D3D12 reference probe, not Tier 3 support in the Vulkan layer.
    **Explicit InnerCoverage library contract (experimental):** a caller-owned device can set
    `VulkanDeviceInfo::inner_coverage_abi` only for a shader library whose InnerCoverage input is already lowered
    to the boolean SPIR-V `FullyCoveredEXT` built-in (false maps to uint 0, true to uint 1). Tier 3 then also
    requires all Tier 2 properties and `fullyCoveredFragmentShaderInputVariable`. The default remains false,
    including for the standalone device; this is not an inferred FSSM/VGPR layout or automatic game-shader support.
    Graphics PSOs inspect the actual SPIR-V capability, even if index capability metadata is absent, and refuse
    `FragmentFullyCoveredEXT` with `E_NOTIMPL` before shader-module creation unless the contract and device support
    are present. A module using `PostDepthCoverage` additionally requires
    `conservativeRasterizationPostDepthCoverage`. The tests use explicit synthetic GLSL; no new translator rule
    or retail shader coverage is claimed.
    `inner_coverage_oracle` checks false positives, missed interior values and missing invocations on the CPU.
    `d3d12_inner_coverage_guards` checks that neither opt-in nor metadata omission can bypass missing support.
    `d3d12_inner_coverage_native` checks 48 native D3D12 scenarios on the matching adapter: six triangles,
    1x/4x MSAA, scissor and unbound-target variants. These pass on RX 5700 XT (Windows 32.0.21045.5002) with
    Vulkan validation active. `d3d12_conservative_tier3` runs the same 48 scenario pairs on Vulkan and native D3D12
    when Vulkan has the property and native D3D12 has Tier 3; it skips on this RX 5700 XT. The positive Vulkan
    path still requires independent execution on a capable GPU. InnerCoverage with conservative rasterization
    off is undefined in D3D12 and is not compared. Edge pixels within the uncertainty region are not required
    to match byte-for-byte; both backends must independently satisfy the geometric guarantees.
    To validate the explicit path, configure with Vulkan SDK/glslang available, build
    `d3d12_conservative_tier3_test`, select an adapter with `LEONIDA_VK_DEVICE` if needed, and run
    `ctest --test-dir runtime/build -C RelWithDebInfo -R "^(inner_coverage_oracle|d3d12_inner_coverage_guards|d3d12_inner_coverage_native|d3d12_conservative_tier3)$" -V`.
    Check that the positive test prints 48 Vulkan/native paired scenarios rather than `Skipped`.
    `d3d12_conservative_tier3_reference` remains a native D3D12 reference probe; it does not opt in to that Vulkan contract.
    It reports the selected Vulkan device's `fullyCoveredFragmentShaderInputVariable`, matches that device to its
    native D3D12 adapter by LUID, then checks the adapter's Tier 3 `SV_InnerCoverage` output on four synthetic
    triangles. On Windows with Visual Studio 2022 and Windows SDK 10.0.26100 or newer, run
    `.\scripts\run_conservative_tier3_reference.ps1` from the repository root; by default this uses Vulkan's default
    adapter. Pass `-Device <name substring>` to select a specific one when multiple GPUs are present. It builds only
    this test, needs no glslang, and fetches the pinned Vulkan headers only if none are installed. The verbose CTest
    output reports the capability and result; it skips if the selected device lacks a matching native D3D12 adapter
    or Tier 3.
  - **Formats:** dynamic rendering takes the RTV and DSV formats. A draw that leaves a slot unbound for which the
    PSO names a format (D3D12 allows it and discards those writes) uses a variant of the pipeline with that format
    undefined, built on first use and kept with the PSO. Bit i of the variant key is colour slot i, and bit 8 is
    depth/stencil.
  - **Instance step rate 0:** every instance reads element 0, done with a stride of 0 for that slot at draw time.
  - **Features:** independent blend state, wireframe fill, a depth-bias clamp, SRC1 blend factors, logic ops and
    disabled depth clipping need `independentBlend`, `fillModeNonSolid`, `depthBiasClamp`, `dualSrcBlend`, `logicOp`
    and `depthClamp`. The standalone device enables those the GPU has and reports them in
    `VulkanDeviceInfo::features`. A PSO that needs a missing one is refused with a log.
  - **Shader features:** the standalone device also enables `shaderInt64` (declared by 25,803 of 26,503 index
    entries), `shaderFloat64`, `shaderInt16`, `shaderClipDistance` and `shaderCullDistance` where the GPU has them.
    Without one, `vkCreateShaderModule` reports the capability (`VUID-VkShaderModuleCreateInfo-pCode-08740`). The
    index's `features` list is not read yet, so a shader that needs a feature the GPU lacks is not refused up front.
  - **Stores and atomics outside compute:** a pixel shader that stores to an image or does an image atomic declares
    that storage image without `NonWritable` (of the 1,038 translated 1.0.2.1 modules that contain an image atomic,
    counted as modules with an `OpImageTexelPointer` and split by the index's stage, 1,018 are pixel shaders and 20
    compute),
    which Vulkan allows only with `fragmentStoresAndAtomics` (`VUID-RuntimeSpirv-NonWritable-06340`); a vertex or
    geometry shader needs `vertexPipelineStoresAndAtomics` (`-06341`). The standalone device enables both where the
    GPU has them. A graphics PSO whose stage declares writable storage (read off the module's declarations,
    `src/spirv_stores.h`) on a device without the feature is refused with `E_NOTIMPL` and a log naming the shader.
    `d3d12_stores_atomics_test` checks both under the validation layer.
  - **Geometry shaders:** `desc.GS` is the translator's Vulkan geometry stage (docs/design.md, "Geometry
    shaders": triangles in, a triangle strip out) between the VS and PS. It needs `geometryShader`, which the
    standalone device enables where the GPU has it and reports in `VulkanDeviceInfo::features`; without it such a
    PSO is `E_NOTIMPL`. The PSO's topology type must be `TRIANGLE`, the GS's input (`E_INVALIDARG` otherwise). The
    push-constant range names the geometry stage only on such a device (`d3d12_geometry_test`).
  - **Refused with a log:** hull and domain stages, stream output, MSAA, set-0 bindings, and two stages
    that both read their code base (question 14, section 5.2). More than eight render targets is `E_INVALIDARG`.
  - **Push constants:** both PSO paths declare the 144-byte block as one range. Its stage flags are the whole
    stage table the graphics path accepts, and the replay pushes with the same flags. A device whose
    `maxPushConstantsSize` is below 144 (#224's `push_constants_need_fallback`) has every PSO refused, with the
    limit in the log, until the uniform-buffer tail exists.

  The command list keeps the input-assembler, viewport, scissor, render-target (read at `OMSetRenderTargets`, as
  D3D12 does), blend-factor and stencil-reference state, including CommandList8's front/back references. Each
  `DrawInstanced` / `DrawIndexedInstanced` records a copy of that state. A draw with no viewport (never set, or
  unbound with `RSSetViewports(0, …)`) or an empty one rasterizes nothing in D3D12. Such a draw is not recorded,
  since Vulkan forbids a 0×0 viewport.

  At replay:
  - **Buffers:** vertex and index views resolve to buffers by VA (arena or upload). A view that runs past its
    buffer is cut at the buffer's end. An unbound vertex slot, or an indexed draw with no index buffer, reads the
    arena's reserved first 64 KiB, which is kept zero (stride 0), so it gets the zeros D3D12 gives.
  - **Attachments:** they move to attachment layouts through the tracker, and the draw runs in a
    dynamic-rendering scope. Its layer count is the smallest layer count of the bound views.
  - **Viewport:** it is flipped (negative height), because translated vertex shaders write D3D clip coordinates;
    front-face winding then matches D3D12's.
  - Only the first viewport and scissor are used, and primitive restart is not modelled yet.

  **Resources and heaps:**
  - A buffer of 0 bytes, or one whose size wraps when aligned, is `E_INVALIDARG`. One the arena cannot hold is
    `E_OUTOFMEMORY`, and so is a shader-visible descriptor heap the arena cannot hold (in host memory its GPU
    handle would be 0).
  - If the arena cannot be created, its buffer is released, so other buffers keep their own memory and get no VA.
  - With `VK_EXT_conditional_rendering`, the arena is created with the conditional-rendering usage, and
    `SetPredication` on a default-heap buffer reads at the buffer's offset in the arena plus the call's offset.
  - `CreateHeap` reserves the requested size in the arena at 64 KiB, and a buffer placed in it gets the heap's
    place plus the placed offset as its console VA. A placed texture keeps its own `VkDeviceMemory` and a null VA,
    1D, 2D and 3D alike (arrays and mips included), and a dimension outside the enum is `E_INVALIDARG`, as on the
    host (RTX 2080 Ti and WARP, `d3d12_placed_dimensions_test`);
    tiled (`64KB_UNDEFINED_SWIZZLE` and Xbox `0x101`) 2D/3D placed textures are backed the same way with
    tile-multiple `GetResourceAllocationInfo`, `GetResourceTiling`, and host-matched `E_INVALIDARG` checks
    (D3D-191). `CreatePlacedResource2`'s castable-format list reaches the placed texture's image as the committed
    path's does (D3D-90). `CreateHeap` refuses with `E_INVALIDARG`, as the host does (RX 5700 and WARP, D3D-238),
    an alignment other than 0, 64 KiB and 4 MiB, a heap type outside the enum, a custom heap without a CPU page
    property and memory pool or a non-custom heap with either, and a heap with all three `DENY_*` flags. It also
    refuses `ALLOW_DISPLAY` (committed resources only), as an RTX 4070 SUPER and a UHD 770 host do. It accepts
    `HARDWARE_PROTECTED`, as those hosts do, though the `D3D12_HEAP_FLAGS` documentation calls it unsupported.
    (D3D-191). Multisampled 2D placed textures (`UNKNOWN` and `4x` tiled) share that backing with host-matched heap
    and offset alignment checks (`4 MiB` vs `64 KiB`, `D3D-222`). `CreatePlacedResource2`'s castable-format list
    reaches the placed texture's image as the committed path's does (D3D-90).

Known gaps: set 0 in pipelines (the device keeps the bindless set, but no pipeline declares or binds it yet) and
therefore every translated shader that declares bindings; tessellation, geometry and mesh pipelines,
multisampled pipelines, several viewports, primitive restart; `requiredSubgroupSize = 64` for wave64 compute;
the game's present entry point (`PresentX`; the DXGI factory/swapchain and the presenter above are what it
- **Subgroup size (compute and graphics PSOs).** A translated shader assumes one subgroup size, and a GPU whose
  default differs runs it inexact: a wave64 shader reads lanes 32-63 of a 32-wide subgroup, and a wave32 pixel
  shader on a 64-wide default reads the other half's lanes (AnyPS5 #487 on RDNA4). `runtime/src/subgroup_size.h`
  decides, per stage, from the shader's index entry and what the device offers; the PSO chains the result.
  - **Index fields:** `wave64`, `subgroup_size` (64 for wave64, 32 otherwise, `null` when emulated) and
    `wave64_exchange` (present: translated with `--wave64-emulate`, exact on any size).
  - **Device:** `subgroupSizeControl` and `computeFullSubgroups` are enabled where supported;
    `minSubgroupSize`, `maxSubgroupSize`, `maxComputeWorkgroupSubgroups`, `requiredSubgroupSizeStages` (from
    `VkPhysicalDeviceSubgroupSizeControlProperties`) and the default `subgroupSize` are kept in
    `VulkanDeviceInfo::subgroup` and printed at device creation.
  - **Emulated shaders:** nothing is required.
  - **Native shaders:** `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo{subgroup_size}` when the feature is
    enabled, the stage is in `requiredSubgroupSizeStages`, the size is a power of two in [min, max] and, for
    compute and mesh, the workgroup fits in size x `maxComputeWorkgroupSubgroups` (VUID-VkPipelineShaderStageCreateInfo-pNext-02755
    and -02756, VUID-VkPipelineShaderStageRequiredSubgroupSizeCreateInfo-requiredSubgroupSize-02760..02762).
    A compute stage also gets `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT` when
    `computeFullSubgroups` is enabled and local size X is a multiple of the size (-flags-02785 for the feature,
    -flags-08988 for the stage, -pNext-02757 for X). Wave32 shaders ask for 32 the same way. A device whose only size is the wanted one (NVIDIA,
    wave32) needs no chain. Where the size cannot be required (stage not listed, feature off) but the device default is
    the wanted one, the shader works as it is (`device_default`; nothing guarantees the default, so it is not logged
    as inexact); any other size is `unavailable` and logged.
  - **Which stages can be required:** it is the driver's list, not a fixed rule, and AMD's Windows driver gives
    different lists for different GPUs, on the same 26.11.2 release too:
    - Radeon 880M (RDNA3.5, gfx1150), 26.11.2: FRAGMENT | COMPUTE | TASK | MESH, sizes 32..64, so wave64 and wave32
      pixel shaders run exact there.
    - RX 6800 (RDNA2 like the console; gfx1030), 26.11.2 (Windows driver 32.0.21045.29003), measured 2026-11-22:
      COMPUTE only, sizes 32..64, default 64 (`vulkaninfo`; the device log says `requirable in stages 0x20`). No
      size can be requested for pixel or mesh shaders (`vk_harness` chains none for `VKH_SUBGROUP_SIZE`), and at the
      default they run 64-wide: a GLSL probe pixel shader drawn by `vk_harness` counts 64 live invocations per
      subgroup (`subgroupBallotBitCount`) on a full-screen draw (SPIR-V 1.5 and 1.6), and `ms_w64x_ballot` (both
      halves of a wave64) passes natively. For the translated SPIR-V 1.5 modules, made without
      `ALLOW_VARYING_SUBGROUP_SIZE`, the specification fixes `SubgroupSize` at that default ("Varying Subgroup
      Size"); only the driver's default could change. So wave64 pixel and mesh shaders work at the default
      (`device_default`; the layer has no mesh PSO yet), and a wave32 pixel shader is `unavailable`: two of its
      waves share a subgroup. `ps_bitmask_perm` shows it: lanes 0-31 exact, while the
      other 2,048 of 4,096 pixels read `v_mbcnt_lo` = 32 instead of their lane (the fixture's check judges lanes
      0-31 only). Of the 1,127 wave32 pixel shaders in the corpus, 1,010 contain a `v_readlane`/`v_writelane`/
      `v_readfirstlane`, 146 a `v_mbcnt`, 105 an `s_bcnt1`/`s_ff1` of a lane mask, and all 1,127 a branch on EXEC
      or VCC or an SCC vote on a mask, which the translator decides by a subgroup vote (117 have only those). 926
      also have `v_permlanex16` and row DPP and 826 quad ops, which stay inside a 32-lane half as translated
      (`tools/wave32_pixel_scan.py`; counts of candidates, not verdicts).
    - Ryzen 7 9800X3D iGPU (RDNA2, gfx1036), Windows driver 32.0.21045.5002: COMPUTE only, sizes 32..64; RX 9070 XT
      (RDNA4, gfx1201), Windows driver 32.0.31041.1004: FRAGMENT | COMPUTE | TASK | MESH, sizes 32..64
      (@AC-Storm-YT, `vulkaninfoSDK` from Vulkan SDK 1.4.363.1,
      [#313](https://github.com/ItsNotRexed/Leonida/issues/313#issuecomment-7086133354)). The same two lists on
      AMD 26.11.2 (Windows drivers 32.0.21045.29003 and 32.0.32029.2010), where `ps_bitmask_perm` also shows the
      live fragment width: the gfx1036 iGPU gets the RX 6800's split (2,048 of 4,096 pixels read lane 32), the
      RX 9070 XT is exact on all 4,096 (@AC-Storm-YT,
      [#3982](https://github.com/ItsNotRexed/Leonida/pull/3982#issuecomment-7088523396) and #4225).

    Current Mesa RADV on GFX10+ lists COMPUTE | FRAGMENT | task/mesh (Mesa main at 152b9cde,
    `src/amd/vulkan/radv_physical_device.c`, `requiredSubgroupSizeStages`; older Mesa reported compute only), so
    pixel shaders run exact there. NVIDIA lists every stage but only size 32, so a native wave64 shader cannot get 64
    there. Vertex and geometry stages are in neither list.
  - **A size the device cannot give** is logged once per stage with the first shader's name (and again at 10, 100,
    1000 shaders of that stage): `[rt] d3d12: <name> (<stage>) is a 64-wide shader but this device cannot require that
    subgroup size ...`. For compute and mesh the fix is the build, `--wave64-emulate` (`make_game.py
    --wave64-emulate auto` picks it for this GPU); a wave64 vertex, geometry or pixel shader has no emulation yet
    (the ~10,000 wave64 pixel shaders are the open case on NVIDIA, SHADER-10).
  - **Test:** `subgroup_size_test` (CPU) checks the decision against AMD-like, NVIDIA-like and old-RADV
    properties and the resulting create-info chain; `d3d12_subgroup_test` (GPU) runs `cs_w64x_readlane` natively
    and with `--wave64-emulate`, and `cs_wqm_w32`, through D3D12 PSOs and compares the read-back words.

Known gaps: set 0 in pipelines (the device keeps the bindless set, but no pipeline declares or binds it yet) and
therefore every translated shader that declares bindings; tessellation, geometry and mesh pipelines,
multisampled pipelines, several viewports, primitive restart; subgroup size for mesh pipelines (the decision covers them, no mesh PSO exists to chain it into), and for wave64 vertex, geometry and, on a GPU that cannot require 64 for pixel shaders (NVIDIA), pixel shaders, which have no emulation (SHADER-10); wave32 pixel shaders on a GPU that runs the fragment stage 64-wide and cannot require 32 for it (RDNA2 under AMD's Windows driver, "Subgroup size" above), where two waves share a subgroup;
the game's present entry point (DXGI factory/swapchain or `PresentX`; the presenter above is what either
calls), HDR and fullscreen; buffer and constant-buffer views (UAV clears read the descriptor bytes, so they do
not wait on view creation), typeless volume RTVs, depth-stencil resolves to a destination offset other than
the source rectangle's;
default-heap buffers are not zero-filled yet; a fence cannot be signalled to a lower value (Vulkan timelines only
increase; such a signal is reported and dropped); a CPU `Signal` above a queue `Signal` the GPU has not reached
yet is held until it has (`vkSignalSemaphore` must stay below every pending signal,
`VUID-VkSemaphoreSignalInfo-value-03259`), so `GetCompletedValue` shows it only then, and a fence released
while it holds one drops it; instance step rates above 1 (they need vertex attribute
divisors); read-only DSVs (the depth image always moves to `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`, which matters once
depth SRVs are sampled next to a read-only DSV, M2); the index buffer is bound without its view's size
(`vkCmdBindIndexBuffer2`, Vulkan 1.4, takes one).

Known gaps found in review, not fixed here:
- **Queues:** every D3D12 queue submits to one `VkQueue`, which runs batches in order. So a queue that waits on a
  fence another queue signals later (`B->Wait(f, n)` recorded before `A->Signal(f, n)`) holds back that signal,
  and the GPU never progresses (woreksami's probe). Queues on separate Vulkan queues (#306 for copy and compute)
  or holding back a batch until its wait value is submitted would fix it.
- **Lookups:** `buffer_at` searches every resource for each vertex and index view; an ordered map by VA would do
  when it shows.
- **Fences:** a CPU `ID3D12Fence::Signal` checks only the semaphore's current value, so it can signal below a GPU
  signal that is still pending (`VUID-VkSemaphoreSignalInfo-value-03258`).

Build `runtime/` (Vulkan SDK and Python 3 on `PATH`) and run `ctest --test-dir <build> --output-on-failure` for the
CPU tests:
- `command_list_test` covers the recording rules.
- `d3d12_state_map_test` covers both state tables, subresource and barrier-range decoding, layout tracking, the
  subresource ranges `DiscardResource` prepares, Resizable BAR detection, and the memory type order on a no-ReBAR
  heap layout. On Windows it also checks every D3D12 constant the IR copies against the SDK's `d3d12.h`.
- `gfx10_descriptors_test` covers the T#, S# and V#: every field alone at its bit position, round trips, what
  decoding refuses and ignores, the T#s the image test shaders build, sampler words as radv writes them, and the
  format table. `tests/test_gfx10_descriptors.py` compares that table with `tools/gfx10_formats.json`.
- `subgroup_size_test` covers the subgroup-size decision of section 3.2 without a GPU: synthetic index entries
  (native and emulated wave64, wave32; compute, pixel, mesh, vertex) against AMD-like (32..64, FRAGMENT | COMPUTE |
  TASK | MESH), NVIDIA-like (32 only, every stage) and old-RADV (COMPUTE only) properties, the limits (feature off,
  `maxComputeWorkgroupSubgroups`, a size that is not a power of two), and the create-info chain.

The GPU tests read their results back. With `LEONIDA_VK_VALIDATION=1` they run under `VK_LAYER_KHRONOS_validation`,
and any validation error fails them.
- `vk_command_replay_test`, part A: only Vulkan 1.0 functions and no tracker.
- `vk_command_replay_test`, part B: rectangle clears, a depth clear while the stencil plane is read-only, a promoted
  texture, and enhanced barriers including a discard.
- `vk_command_replay_test`, part C: a 4x4x4 volume uploaded from a footprint, copied as a 2x2x2 block and whole,
  and read back; a 4x multisampled target cleared and resolved; and the copies and resolves that must be refused
  (a z range past the volume or its footprint, a footprint past the buffer, a resolve from a single-sampled
  texture or into another size, a multisampled texture copied to a footprint or to another sample count).
- `vk_command_replay_test`, part D: every D3D12 sync bit against every access bit (360 pairs), each as a global,
  a buffer and a texture barrier. Under the validation layer none may be rejected; before
  `vulkan_barrier_scope` there were 462 validation errors on Intel(R) Graphics.
- `vk_command_replay_test`, part F (D3D-26): `WriteBufferImmediate` in all three modes at console VAs, one of them
  in a binding that is a range inside another buffer (as arena resources are); UINT and UINT64 atomic copies
  ordered between the writes so that a missing barrier would copy a stale or a later value; a buffer discard
  (kept), a texture cleared, discarded and cleared again, and a rectangle discard (kept); and the lists that must
  be refused (an address past a binding or in none, no `vkCmdUpdateBuffer`, a copy or dependent range past its
  buffer, an unknown dependent resource, a subresource the resource does not have, and an explicit discard count of
  `UINT_MAX`, which is a range and not the whole resource).
- `vk_command_replay_test`, part I: a 4x D32 `MIN` resolve of a rectangle; the texels around it must keep the
  destination's clear value (MoltenVK overwrote all 28 of them before the scratch copy). It is skipped where the
  device has no `MIN` depth resolve (llvmpipe), and without `copy_scratch` the list is refused.
- `d3d12_vulkan_device_test` also runs both atomic copies and the discards through COM, and checks that
  `WriteBufferImmediate` is refused at execution while no buffer has a GPU virtual address on `main`.
- `vk_bindless_test` (run by `runtime/test/shaders/check.py` and `scripts/run_gpu_tests.py`): the translator's
  own lookup against the table and the set. `ps_tex.asm`, a hand-written RDNA2 pixel shader built by `build.py`,
  loads a T# and an S# from a descriptor table and samples. The test draws the gradient with the T# at the fourth
  slot probed, the fallback after the key is erased, the gradient again after the table has grown into a new
  buffer, another view at the same index, and the fallback after removal, with and without the
  descriptor-indexing flags. Last, it draws through a 2D image past 5000 others and a sampler past 2100 others.
- `vk_sampler_test` (run the same way): the S# and T# mip fields as `gfx10_descriptors.h` maps them. `ps_ssharp.asm`
  loads a T# and an S# from the heap and samples a 128x128 two-level texture twice, with `image_sample_lz` (LOD 0)
  and `image_sample` (LOD 1 at this size). The sampler is `vulkan_sampler(decode(S#))` and the view's levels come
  from `decode(T#)`. `check.py`'s `SAMPLER_CASES` cover a point mip filter, MAX_LOD 0, MIN_LOD 1, LOD_BIAS ±1,
  MIP_FILTER_NONE with and without MIN_LOD, and a T# whose BASE_LEVEL is 1. Each pixel is judged against levels
  worked out from the S# and T# bits in Python, independently of the header. On Vulkan, the sampler's minLod and
  mipLodBias apply to an explicit LOD as well. The T#'s MIN_LOD (D3D12's ResourceMinLODClamp) is not covered by
  this test.
- **Measured on RDNA2** (SHADER-539): `tools/sampler_lod_probe.py --oracle hw` samples a linear 64x64 R32_SFLOAT
  texture with seven levels at its top-left texel. The texture memory is a dword ramp, so each result names the
  level it was read from, and a calibration row per level finds the level starts. It runs 43 rows of
  `image_sample_l` and `image_sample_lz` on a gfx1036 (Radeon iGPU, Windows 11, AMD 26.10.1, `tools/hw_oracle`
  HIP launcher, rocm-sdk-core 10.0.0 clang). wave32 and wave64 gave the same level in every row, and three runs
  were identical.
  - **The S# fields apply to explicit LODs.** LOD_BIAS (±), MIN_LOD and MAX_LOD apply to `image_sample_l`'s
    explicit LOD and to `image_sample_lz`'s LOD 0, bias before the clamp, as radv assumes. MIP_FILTER none ignores
    them and reads the view's base level.
  - **The T#'s MIN_LOD applies too.** It counts from the resource's level 0, not the view's BASE_LEVEL: BASE_LEVEL
    1 with MIN_LOD 2 reads level 2.
  - **Where the hardware is not Vulkan:**
    - **Rounding.** The point mip filter's level is floor(lambda + 0.5), so an exact .5 rounds up: LOD 0.5, 1.5 and
      2.5 read levels 1, 2 and 3, and so do S# MIN_LOD 1.5 or MAX_LOD 1.5 and LOD 1 with bias +0.5. Vulkan's
      nearest is ceil(lambda + 0.5) - 1, which reads 0, 1 and 2.
    - **The T#'s MIN_LOD clamps the still fractional LOD before that rounding.** MIN_LOD 2.25, 2.5 and 2.75 at
      LOD 0 read levels 2, 3 and 3. `VK_EXT_image_view_min_lod` raises the integer level to floor(minLod), which
      gives 2, 2 and 2.
    - **MIP_FILTER none with the T#'s MIN_LOD.** The level is the base raised to floor(MIN_LOD): 2.5 reads level 2,
      the same as Vulkan.

    So a host whose Vulkan driver follows the spec's nearest rule reads one level lower than the console on 8 of
    the 43 rows: an exact half-level LOD, or a fractional ResourceMinLODClamp. LOD 1.49 reads level 1 on both. The
    AMD driver does not follow it. Through the runtime's own path (`vk_sampler_test`, `vulkan_sampler(decode(S#))`),
    an S# LOD_BIAS of +0.5 on `image_sample_lz` reads level 1, like the console, on the same gfx1036 and on an RX 7800
    XT (both AMD 26.10.1). So on AMD hosts the S# case matches. On other hosts the result splits: Mesa llvmpipe
    (Mesa 23.2.1, LLVM 15.0.7) and an Apple M2 (macOS 26.4, MoltenVK 1.4.2) follow the spec's nearest rule — the
    same `vk_sampler_test` S# with LOD_BIAS +0.5 reads level 0 (on the M2 the same explicit `image_sample_lz` with
    bias 0 and -0.5 also reads level 0; llvmpipe switches to level 1 at 129/256, for LOD_BIAS and MIN_LOD alike),
    while NVIDIA reads level 1, like the console: an RTX 5070 (driver 610.88) switches at exactly 128/256. Intel
    is not measured here.
  - **Layout.** gfx1036 lays a linear mip chain out smallest level first, each row padded to 256 bytes. Levels 6..0
    of the 64-wide R32 chain start at dwords 0, 64, 192, 448, 960, 1984 and 4032.

  `tests/test_sampler_lod_probe.py` checks the descriptor fields against `check.py` and `gfx10_descriptors.h`, and
  the model against these levels, without a GPU.
- `d3d12_com_test`: drives the COM objects through their vtables. It also checks that a queue `Wait` on a fence
  the CPU has not signalled holds the GPU back.
- `dxgi_swapchain_test`: creates a Win32 HWND and exercises `rt::d3d12::create_dxgi_factory`,
  `IDXGIFactory4::CreateSwapChainForHwnd`, `IDXGISwapChain3::GetCurrentBackBufferIndex`, `GetBuffer` + `GetDesc()`,
  backbuffer `CreateRenderTargetView` + clear + readback verification + `Present`, and `ResizeBuffers` with 0
  validation errors.
- `d3d12_indirect_rt_test`: validates `ID3D12Device::CreateCommandSignature` + `ExecuteIndirect` with a GPU
  `CountBuffer` (`vkCmdDrawIndirectCount`, `count = 0` skipped and `count = 1` rendered) and optional
  `VK_KHR_acceleration_structure` + `VK_KHR_ray_query` (`GetRaytracingAccelerationStructurePrebuildInfo`, BLAS + TLAS
  `BuildRaytracingAccelerationStructure`, and a `RayQueryKHR` compute shader verifying hit vs. miss; clean SKIP on
  GPUs without RT).
- `d3d12_gcl8_test`: the public `ID3D12GraphicsCommandList8` surface: `QueryInterface` for its IID,
  `OMSetFrontAndBackStencilRef(1, 2)` with the recorded values, and the reference count around the query. It
  creates only an allocator and a list, so it needs no GPU or Vulkan loader.
- `d3d12_depth_resolve_test`: depth and stencil `ResolveSubresourceRegion` of a 4x D32 or D24S8 target cleared with
  three overlapping values, whole and by rectangle, compared with the PC runtime on the layer's own adapter and WARP;
  the refused calls must fail `Close` and leave the destination as cleared.
- `d3d12_vulkan_device_test`: the Vulkan device behind the Xbox-ABI proxy, called the way the game will call it.
  It creates heaps through slot 14 and reads their start handles in RAX, creates a render target through slot 76
  with the Xbox layout word, then creates views, records clears and barriers, executes and waits on a fence.
  Results come back through D3D12 itself (`CopyTextureRegion` into a readback heap, then `Map`): the render
  target, the depth and stencil planes, and a never-written texture (which must be zero). A second list uploads a
  pattern and copies it on: whole into a texture created in `COMMON`, a boxed part into a smaller texture, a BC1
  round trip, a 1D texture and a 1D-array layer, and a buffer region. A 4x multisampled render target is cleared through a `TEXTURE2DMS` view and
  resolved into a texture that is read back; the multisampled shapes D3D12 refuses are refused, and so is a
  resolve from a single-sampled texture. `GetCopyableFootprints` is compared with the PC runtime's, and a short
  run repeats on a standalone device, the path `LEONIDA_D3D12=vulkan` takes. That standalone run also covers
  heaps and placed resources: a castable list on a placed texture is cleared through an RTV in a cast format and
  read back against the host's answer, and refused lists match the host's `E_INVALIDARG` (D3D-90).
  The comparison tests run against a PC D3D12 device: the layer's own adapter (the default adapter when the Vulkan
  device has no LUID or that adapter gives no D3D12 device), or the one named by
  `LEONIDA_HOST_ADAPTER` (`warp`, or part of an adapter's name; the adapter used is printed). `d3d12_vulkan_device`
  picks it through `runtime/test/d3d12_copy_compare.h`; `d3d12_copy_xfmt`, `d3d12_depth_resolve`,
  `d3d12_fence_events`, `d3d12_indirect_zero`, `d3d12_node_mask`, `d3d12_query_rules` and `d3d12_resource_desc`,
  which also compare with WARP, through `runtime/test/host_adapter.h`. Some answers are the
  driver's choice, so pick the host side when two adapters disagree: `GetResourceTiling` splits standard and packed
  mips differently on an NVIDIA, an AMD and a WARP host, and the test accepts any split.
  - Copies against the host (D3D-46, `d3d12_copy_compare.h`): the same scenario runs on the layer and on the host and
    the bytes read back are compared; a failing case prints its name, the region (subresource, mip, slice, plane)
    and the first differing byte. Every case uploads deterministic bytes into a texture and then runs
    `CopyResource`, `CopyTextureRegion` texture to texture (an inner box to the origin, and the origin box into
    another array slice at an offset), footprint to texture (an inner box at an offset) and texture to footprint
    (an inner box at an offset into a readback buffer prefilled with `0xcd`), and reads every subresource back.
    Formats: R8, R8G8 SNORM, RGBA8 (UNORM, SRGB, typeless), BGRA8, R16 UNORM, RG16 and RGBA16 float, R32 UINT,
    RG32 UINT, RGBA32 float, R10G10B10A2, R11G11B10, R9G9B9E5, B5G6R5, B5G5R5A1, D16, D32, D24S8, D32S8 and BC1-BC7
    (SRGB and signed variants included), as a 36x20 texture with three slices and six mips (odd mip sizes), 3D
    textures with boxes in depth, a single-mip single-slice texture and a one-block BC7. Buffers: `CopyBufferRegion`
    at odd offsets and sizes, a `CopyResource` of a 4093-byte buffer, and an upload buffer straight to a readback
    buffer. Only the bytes inside a footprint row are compared (row padding is the driver's; a difference there is
    printed, not failed); every byte a partial copy must leave alone is compared. It found:
    - `R8G8_SNORM`, `R16_UNORM`, `R32G32_UINT`, `R9G9B9E5_SHAREDEXP`, `B5G6R5_UNORM` and `B5G5R5A1_UNORM` textures were
      refused (no Vulkan mapping); they map to `R8G8_SNORM`, `R16_UNORM`, `R32G32_UINT`, `E5B9G9R9_UFLOAT_PACK32`,
      `R5G6B5_UNORM_PACK16` and `A1R5G5B5_UNORM_PACK16`.
    - A texture-to-texture box on a block-compressed texture that reaches the padded edge of a mip (for example a
      box of whole blocks over a 9x5 mip) was refused as `invalid_copy`; the PC runtime accepts it.

    Copies between different mips of one texture are compared too (mips 24, 12, 6, 3, 1; RGBA8, BC1, BC7): sizes that
    are no block multiple, boxes to a padded edge, and boxes that do not fit the destination, which both sides must
    refuse (the host at `Close` with `E_INVALIDARG`, the layer at execution with `invalid_copy`). It found:
    - **Block-compressed copies into a mip whose size is no block multiple** (a 0..8 box into a 6-wide mip, a 6x6
      mip into a larger one, a 4x4 box into a 3x3 mip) were refused with `invalid_copy`: the host copies whole
      blocks, and one `vkCmdCopyImage` cannot (its extent must be a block multiple or reach each image's edge). They
      are staged through transfer scratch now (above) and match the host byte for byte, as do a box 4..8 of a 6x6
      mip into a 3x3 one and a 3x3 mip into a 6x6 one at (4, 4). Measured on an RTX 5070 Ti and an RTX 3060 (driver
      621.24) and an integrated AMD Radeon(TM) Graphics (`0x1002:0x164e`, Vulkan driver 2.0.353), Windows 11, each
      against its own host. `d3d12_state_map_test` checks the staged regions on the CPU.

    Known gaps shown by it and by the formats, not fixed:
    - **`TEXTURE1D`** resources are not supported by the layer (creation, footprints and allocation info refuse
      them), so 1D copies are not compared.
    - On a GPU with a native D24, a readback leaves the unspecified `X8` byte of the depth plane untouched where the PC
      runtime writes 0; the comparison ignores it. Where the layer stores D24 as D32 (see section 3.2) it writes 0.
    A PC driver's copy flushes denormal float depth to zero, so the test keeps its values normal. A
    `LEONIDA_HOST_ADAPTER` that matches no adapter fails the run instead of skipping the comparisons.
  - D24 against the host (`d3d12_depth24_compare.h`, `depth24_convert_test`): the D24S8 row of the copy test
    above runs with random 24-bit words on every GPU, the Radeon 880M's `D32_SFLOAT_S8_UINT` fallback included
    (72 regions, byte for byte, upload, readback and copies). `d3d12_depth24_compare.h` clears 5215 depth values
    (steps, half steps and the floats around them, powers of two, random) as a whole-view clear of a 4x4 texture
    (the first 160) and as one rectangle clear per value of a wide texture, with a stencil value, and compares both
    planes read back. The layer is checked against the reference (`depth24_quantize`), and against the host byte for byte
    where the host agrees with it; NVIDIA's host rounds some ties the other way (478 of 5215 values) and is logged as a host quirk, not a failure. It also clears an
    `R24G8_TYPELESS` texture through a typed D24 DSV (200 values, compared the same way), and reports whether
    shader-resource views of both planes (`R24_UNORM_X8_TYPELESS`, `X24_TYPELESS_G8_UINT`) write the T# of the
    texture's backing format (D3D-75): X24_8_32_FLOAT for the depth plane where the layer stores the texture as
    `D32_SFLOAT_S8_UINT`, 8_24_UNORM where the GPU has D24, and 8_UINT for the stencil plane. An SRV read is not compared with the host (no shader runs in these tests and the host's
    descriptor is opaque). It also runs an upload and readback under a `SetPredication` that skips copies, which the
    layer must still convert and which is compared with the host's unpredicated run. `depth24_convert_test` runs the
    compute pass on the GPU for all 16777216 words (to float, and back), 16777216 float bit patterns against
    `depth24_quantize` (with a 1 MiB range so the copy splits into 65 dispatches), and a pitched copy at an offset
    split by a 1 KiB range. It tests the pass, which needs no D24 absence, on whatever device
    `VK_LOADER_DRIVERS_SELECT` picks; the layer's own D24 path (the D3D12 tests above) is used only on a GPU without
    D24 depth-attachment support, so run those with the AMD driver selected (the Radeon 880M, RADV).
- `d3d12_translated_draw_test`: milestone M1.5 (D3D-53), the same path with translator output.
  - **Shaders:** `test/shaders/vs_m15.asm` and `ps_m15.asm` are hand-written RDNA2 (gfx1013). The build assembles them
    with `llvm-mc` and translates them with `tools/rdna2spirv.py` through `test/shaders/build.py`. `build.py` also writes
    each one's synthetic FSSM program (`; fssm-at` / `; fssm-data` in the header) to `fssm/` and the translator puts its
    SHA-256 in `spirv/index.json`, so the layer's lookup is the one the game uses; the test loads that index as it is.
    An unknown program is refused.
  - **User data:** root parameter 0 is a descriptor table (`s0:s1`, the VA of its first slot), parameter 1 four root
    constants (`s2..s5`: the draw's clip offset, and a bias for the PS).
  - **Vertex fetch:** the VS has no input layout. It `s_load`s a V# the application wrote into a shader-visible slot and
    fetches position and colour with `buffer_load_dwordx4 ... idxen`; the V# points at a DEFAULT-heap buffer in the arena.
  - **Code base:** the PS starts with `s_getpc_b64` and `s_load`s four floats from offset 0x100 of its own FSSM program,
    so the layer must have uploaded that program at the push-constant code base (dwords 34-35).
  - **Draws:** a quad and a second V# over the same buffer, with different table slots and constants, into a
    `R32G32B32A32_FLOAT` target. Every value is a short binary fraction, so the comparison is bit for bit: 4064 of 4096
    pixels are exact, and the 32 whose centres lie on the triangle's long edge are counted apart.
  - **Zero-based ids:** all draws use `StartVertexLocation` 0 and `StartInstanceLocation` 0, so `v0` is the same with
    and without the translator's zero-based VS ids (#436, not in this tree: with its translator change applied the
    result was identical, checked on both GPUs). Both devices enable `shaderDrawParameters` where supported, which
    #436's shaders need. A non-zero base would differ (hardware: zero-based, the base in an SGPR), so the test avoids one.
- `d3d12_method_variants_test`: each variant against the method it wraps, on the device directly (the proxy's
  slot 76 is the Xbox texture method, not `CreateCommittedResource3`). It runs 12 creation cases (buffers on the
  three heap types, textures, a volume, a multisampled target, and three shapes the base refuses) through every
  `CreateCommittedResource` variant, which must give the same result, description, heap properties and `Map`
  answer. It runs the edge cases in the table above and compares them with the host's PC runtime when there is
  one; protected sessions are checked on the layer alone, and castable lists have their own success/refusal
  checks. `GetCopyableFootprints1` must match `GetCopyableFootprints` field by field, including the failure
  answer. A texture from `CreateCommittedResource3`
  is uploaded, copied back and compared on a queue from `CreateCommandQueue1`.
- `d3d12_sampler2_test`: `CreateSampler2` against `CreateSampler`, on the slots' bytes. It checks:
  - 10 descriptions without flags, which must be byte for byte the same;
  - 6 UINT and float-bit border colours against their float equivalents (the three built-in UINT colours with
    the integer mark, which must decode to `VK_BORDER_COLOR_INT_*`);
  - custom float and UINT border colours registered in the table on `VK_EXT_custom_border_color` (D3D-73),
    verifying index reuse for identical colours and distinct S# descriptors for distinct colours or integer vs
    float representations;
  - 3 non-normalized samplers, which must be `CreateSampler`'s S# with `FORCE_UNNORMALIZED` set;
  - the `D3D12_OPTIONS17` answer;
  - the 22 written-or-refused cases behind the table in section 3.2, compared with the host's PC runtime when
    there is one.
- `xbox_present_test`: `PresentX` and the frame events (section 4.5, "what is implemented"), checked on swapchain pixels.
    there is one. A host that reports no non-normalized samplers is compared only on the 3 cases without the
    flag. The other 19 are then checked against their recorded answers only.
- `d3d12_subgroup_test`: subgroup size control through D3D12 PSOs, on the PSOs of the device proxy.
  - **Shaders:** `cs_w64x_readlane` (a `v_readlane` across both halves of a wave64) translated natively and with
    `--wave64-emulate`, and `cs_wqm_w32`, built by `test/shaders/build.py` (needs `llvm-mc`: `LLVM_MC`).
  - **Check:** each is dispatched through a PSO found by FSSM hash and the eight words it stores are compared. The
    PSO's request (`pipeline_subgroup_requests`) must equal the decision function's. The native wave64 shader
    must be exact where the device can give 64 (AMD: required size 64 plus full subgroups) and is reported, not
    judged, where it cannot (NVIDIA); the emulated one must be exact everywhere.
  - **Control:** a second run with nothing requested is printed. On the Radeon 880M the default compute subgroup is
    already 64, so the wave64 controls are exact as well.
  - **Probes:** GLSL shaders (`cs_sg_probe.comp`, `ps_sg_probe.frag`, `vs_sg_fullscreen.vert`) store `gl_SubgroupSize`,
    so the size the driver used is observed for a wave64 entry, a wave32 entry and an emulated one. Where the PSO
    required a size, the observed size must equal it. For a wave32 compute shader on a device whose default is not 32,
    the control is asserted to differ (the request matters); a wave32 pixel shader on the 880M goes from 64-wide to
    32-wide with the request.
  - **Wave64 pixel shader:** `ps_w64x_readlane.asm` (`v_readlane` of lanes 40, 5, 33 and 63) is drawn over 64x64
    pixels through a graphics PSO into a float target, and every pixel must carry the four values derived in the
    .asm header where the PSO required 64: 4096 of 4096 on the 880M. The control is asserted to differ only where the
    default pixel subgroup is not 64; the 880M runs pixel shaders 64-wide by default, so its control is exact too. On
    the RTX 5070 the shader gets no 64 (0 of 4096 exact, not judged).
  - **RX 6800 (RDNA2, AMD Windows driver 26.11.2, 2026-11-22):** the compute requests take effect (wave32 compute
    goes from 64-wide to 32-wide), but nothing can be requested for the fragment stage: the pixel probe runs at the
    64-wide default for the wave64, wave32 and emulated entries (outcomes `device_default`, `unavailable`, `emulated`), and
    `ps_w64x_readlane` is not judged; its control, at the 64-wide default, is 4096 of 4096 exact.
- `d3d12_descriptor_test`: descriptors through the proxy.
  - Copies, checked on the slots' bytes: one run, range lists of different shapes on each side, null size
    arrays, and a destination longer than its source.
  - What views and samplers write, decoded: a view of an uploaded 8x8 texture and a sampler, mips 1 and 2 of a
    BGRA cube with a component mapping, an unordered-access view (the same bytes as the shader-resource view of
    that mip), and the null descriptors of a missing resource, a texture without `ALLOW_UNORDERED_ACCESS`, a view
    past the mips and a volume.
  - UINT border colours (Vonsek's case on #538): an `R8G8B8A8_UINT` texture sampled one tile to the right
    through `CreateSampler2` samplers with a UINT `(0,0,0,0)` and `(1,1,1,1)` border reads 0 and 1 in every
    channel. A float white border on the UNORM texture reads 255; its S# differs from the UINT white one only in
    the integer mark.
  - That set 0 holds them: a compute shader (`test/glsl/cs_sample.comp`, GLSL, compiled with `glslangValidator`;
    skipped without it) is handed a slot's T# and the sampler's S#, looks them up in the hash table as a
    translated shader does, and samples. The sampler wraps in U and the shader samples one tile to the right,
    so a missed S# shows too: the fallback sampler clamps. The layer has no pipelines yet, so the test creates
    this one in Vulkan and binds the device's set. It reads the texture through the view's own slot and through
    a copy of it, and gets the fallback from a slot of bytes no view was made for and, after the texture is
    destroyed, from the view's own bytes.
  - Capacity: 5000 distinct SRVs of that texture and one more, and 2100 distinct wrapping samplers in two
    2048-slot heaps. The last view through the last sampler reads the texture one tile to the right. With the
    sizes of before, both were past their arrays ("array is full") and the shader read the fallback.
- `d3d12_compute_test`: the compute path through the proxy. It creates a root signature from the PC serializer's
  blob, and a PSO through slot 62 from a container shaped like the game's, found in a translator-style index by its
  FSSM hash (an unknown program is refused). It sets root constants, a root UAV and a descriptor table whose
  application-written entry points at an upload buffer, then runs two dispatches that change s0 and the UAV in
  between. The results come back through a readback buffer. The shader, `test/glsl/cs_user_data.comp`, is GLSL
  written to the translator's push-constant and address contract, compiled with `glslangValidator`. It also checks:
  - the dispatches' root signature and PSO are released right after being set, and the same signature is set
    again after the arguments, so the list must hold both and keep the arguments (without the hold the test
    crashes);
  - the index has an entry with set-0 bindings for the same program ahead of the binding-free one, and the PSO
    must take the binding-free one;
  - a device that reports 128 bytes of push constants refuses the PSO.
- `d3d12_fspx_compute_test`: the FidelityFX DLL's two ways of passing an FSPX program, with synthetic programs
  ([fspx.md](fspx.md), "Build and run-time lookup"). A PSO through slot 62 from a `DXBC` container whose `ACCC` part
  is an FSPX payload, and one through `CreateComputePipelineState` from a bare kind-6 program, as the slot-67
  description carries it. Both are found by `fspx_code_sha256` in `<shaders>/fspx/index.json` and dispatched once with
  `cs_user_data.comp`, which adds dword 1 of its code base: each must read its own kind-6 program. Ten inputs are
  refused with `E_INVALIDARG`: an untranslated program, bare and in a container; kind-6 entries that end past the
  payload, start inside the directory, wrap 32 bits, are too large or are empty; a container without `ACCC`; bytes
  without the prologue; and the program with four bytes appended.
- `d3d12_bindless_churn_test`: the table churn of D3D-314 through D3D12 alone, on a direct and a compute queue with
  three frames in flight. Each of 96 frames makes 24 textures with SRVs in a ring of shader-visible heap slots and
  releases those of seven frames earlier (about 190 live keys, so the table grows), the direct queue uploads, the
  compute queue waits for it on the GPU, and each queue runs the `cs_set0_sample.comp` shader eight times on SRVs
  of the frame and the three before it. All 98,304 sampled texels per run are compared with what the CPU wrote.
  The two D3D12 queues get distinct `VkQueue`s where the device's family has two (`queue_count_wanted = 2`,
  `command_queues`; the test prints which it got and says so when the family has only one). Two queues on one
  `VkQueue` are ordered by submission, which hides a missing wait between them. It runs four times: as the layer is (0
  `vkDeviceWaitIdle` for the bindless set required, at least one generation made, a retired generation seen pending
  right after the submit that made it); with the compute queue never waiting on the direct queue's fence (it samples
  textures the CPU has seen finished, so only the grown table is new to it); and both again with
  `LEONIDA_BINDLESS_IDLE_WAIT=1` (the old wait, counted above 0). Times are printed, not asserted. The async runs
  are what the layer's wait on the uploading submission keeps correct: with that wait removed, the RTX 5070 run
  failed at frame 5 (compute read a zeroed table: the grown binding 2 with the upload not yet run), as @duudiis
  reported on #2740; the old-wait path shares that bug on `main` (the idle wait only covers work already submitted,
  not another queue's later submission that binds the new table). The Radeon 880M did not reproduce it in two
  runs. `d3d12_bindless_churn_validated` runs it under the validation layer.
  The table's growth is the only replacing flush this scenario produces (3 generations). The other kinds (a live
  key's index replaced, a relocation) and the failed-submit retry are checked at the Vulkan level by
  `vk_bindless_test` cases 3 to 6 (the set changes; `release_retired` frees only what its ticket allows). That test
  needs the translated `vs_ids`, `ps_tex` and `ps_imgatomic`; it was not in ctest before D3D-314 (only
  `scripts/run_gpu_tests.py` and `check.py` ran it), and is now, as `vk_bindless_ps_tex` and `vk_bindless_ps_imgatomic`.
  without the prologue; and the program with four bytes appended. It also covers the dispatch state:
  - one program in containers with three kind-0 entries picks its own entry for each, an unknown kind 0 is refused, and
    the bare program is refused as ambiguous;
  - slot 67 with all four pairs picks by kind 0 too (one entry's module is missing on purpose, so the pick shows as
    `E_FAIL`), and four inconsistent pairs are refused before they are read;
  - slot 67 with the description built the way the DLL builds it (`dll_slot67`: the reader's pair order over the
    container's own directory, kinds 0, 1 and 7 pointing into the payload, kind 6 copied into a 256-rounded buffer
    padded with `0xcc`) resolves each program to its own translation with its own state;
  - console slot 11 on a slot-67 and a slot-62 pipeline returns an object whose slot 3 points at the dword 0 and whose
    slot 4 gives 4 bytes; it releases to 0, and the pipeline's reference count is unchanged.
- `d3d12_set0_compute_test`: set 0 from the compute bind point, through the proxy and D3D12 alone. The index entry
  of its shader lists set-0 bindings, so the PSO's layout holds the device's bindless set and replay binds it at
  each dispatch; nothing in the test binds it. Four dispatches of `test/glsl/cs_set0_sample.comp` (GLSL written to
  the translator's contract) load a T# and an S# from two descriptor tables and sample one tile to the right in U:
  - two SRVs give their textures texel for texel, so the T# decides the image;
  - 32 bytes no view was made for give the fallback image, one colour everywhere;
  - a clamping sampler gives the texture's right edge, so the S# decides the sampler;
  - a device without the bindless set refuses the PSO.

  Removing the set-0 bind from the dispatch path makes this test fail (`VUID-vkCmdDispatch-None-08600`, measured on
  llvmpipe); `d3d12_m2_textured_test` covers the bind at draws in the same way.
- `d3d12_draw_test`: milestone M1, through the proxy.
  - **Pipeline:** a VS+PS PSO found by FSSM hash. The input layout is POSITION then COLOR, while the VS has them in
    the opposite register order, so a match by position would fail.
  - **Buffers:** the vertex buffer is copied into the arena, and the index buffer is read from an upload heap.
  - **Draws:** two indexed draws with different root constants and root CBVs, then a back-facing triangle that must
    be culled. Each of these also checks a fix:
    - the first draw uses a PSO with a second RT format and a DSV format, with neither bound (a variant);
    - before the second draw, the same root signature is set again after the arguments, which must keep them;
    - two more draws, through a viewport on the top-right quadrant, read COLOR per instance with step rate 0 (two
      instances that must both read element 0) and from an unbound slot (zeros).
  - **Second pass** on the same target, cleared again: a full-screen triangle from a VS with no inputs
    (`test/glsl/vs_fullscreen.vert`), first in its submission so no vertex buffer was bound before it, through a
    viewport on the bottom-right quadrant; then, where the GPU has `logicOp`, the red quad XORed over the blue
    top-left quadrant, which must read `(255,0,255,0)`.
  - **Refusals:**
    - PSOs that need wireframe fill, independent blend, a depth-bias clamp or depth clamping are refused on this
      device, which enables none of them;
    - nine render targets is refused, at PSO creation and at `OMSetRenderTargets`;
    - 0-byte and wrapping buffer sizes are refused, and a buffer or shader-visible heap larger than the arena runs
      out of memory;
    - a draw with no viewport executes cleanly and draws nothing.
  - **Check:** all 4096 pixels of the read-back target are compared. Where the red and green quadrants land also
    checks the viewport flip and the winding. The test fails with any of these fixes undone: the rebind, step
    rate and logic op show wrong pixels, the variant breaks `dynamicRenderingUnusedAttachments-08912`, and the
    stride state breaks `pStrides-04913`.

  The shaders are GLSL written to the translator's contract (`test/glsl/vs_draw.vert`, `vs_fullscreen.vert`,
  `ps_draw.frag`).
- `d3d12_present_test`: milestone M0. It opens a window and presents 60 frames, each one: reset the list, move
  the back buffer `PRESENT` → `RENDER_TARGET`, clear, move it back, execute, present, signal a fence. It then
  resizes the window and presents 10 more (70 in all), which recreates the swapchain, scales the blit and switches
  to unsynchronised presents. The presenter's capture copies every presented image back, and the test checks
  that the whole image holds the clear colour at the swapchain's size (after the first 60 frames, after the
  resize and at the end). Last, it reads the colour on screen with GDI when the window is in front, no other
  window lies over its centre (an always-on-top one can, even over the foreground window) and GDI can read the
  screen. On a GPU that does not drive the window's monitor (the UHD 770 with the desktop on an RTX 4080 SUPER) the
  desktop never shows the swapchain, so the test prints the pixel, does not assert it and says why. Under Wine's macOS driver it can't: `GetPixel` on the screen DC returns `CLR_INVALID` even
  for a pixel GDI painted itself, and the window's own DC holds Wine's GDI surface (white), not the image
  presented through the Metal layer. There the capture checks are what confirms the colour.
  - The injected failures below make the runtime log `[rt] d3d12: present failed` three times (VkResult
    -1000001004 twice, then -1, `VK_ERROR_OUT_OF_HOST_MEMORY`). The test announces them on stderr first; they
    are expected and are not out-of-date presents from the driver (the closing line counts those).
  - The device and the presenter get their Vulkan functions through a driver double in the test. It forwards to
    the real driver, counts the presents the driver reports out of date (the frame counts allow for them), and
    tracks the binary semaphores alive.
  - Through the double, the test reports one successful present as out of date (not counted) and one as
    suboptimal (counted).
  - It fails the second per-image semaphore of a recreation: only the two acquire semaphores may remain, and a
    retry at the same vsync must make a complete swapchain.
  - It fails one blit submission: the next four presents must go through without hanging.
  - Run against the presenter before these fixes, the test fails at the frame count after the resize, where the
    driver reported a real out-of-date present (RTX 5060 Ti, driver 621.24).

**Open:** whether the game uses command-list flags, bundles, or `ExecuteCommandLists` with many lists at once. All
are supported by the IR; the trace will show which are hot.

[descriptor-overview]: https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptors-overview

### 3.3 One GPU address space

The shader contract is the constraint that shapes everything else. The translator computes:

```
host_address = heap_base + console_va
```

where `heap_base` is push-constant dwords 32–33 and `console_va` is an Xbox GPU virtual address the game writes
into a descriptor, a root argument, or a buffer. This applies to buffer, descriptor-table and shader-code
addresses. Image descriptors (T#) are hashed by the translator, not dereferenced, so an image's base can be a
separate key range (below). Most shaders also force the high dword of a descriptor-table pointer to zero (25,865
of the 26,503 in package 1.0.2.1, counted 2026-11-22; see "Descriptor-table high-dword truncation" below), so
**shader-visible descriptor heaps must live below 4 GB** in this address space.

The layer therefore owns the console VAs. It returns them from `ID3D12Resource::GetGPUVirtualAddress` and from
`GetGPUDescriptorHandleForHeapStart`; the game only sees values the layer chose. The design is:

| Piece | Design |
|---|---|
| Address space | Two ranges. A **BDA arena** holds buffers, descriptor heaps and shader code; its device address is `heap_base`. A separate **image key range** supplies unique console VAs for images; those keys are never added to `heap_base`. The arena's `VkBuffer` is created with all buffer usages the game can bind (uniform, storage, vertex, index, indirect, transfer) plus `SHADER_DEVICE_ADDRESS`, and its memory with `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`. |
| Buffers | A buffer resource is a range in the arena's single `VkBuffer`: offset = its console VA, size = the resource size. Buffer commands bind the arena buffer with that offset, so its device address is exactly `arena_bda + offset`; no per-resource `VkBuffer` object is needed. |
| Images | Each image gets a unique console VA key and lives in its own `VkDeviceMemory` (or a suballocated device-local image heap) with optimal tiling. Shader image access goes through the T#/hash table, so nothing dereferences the key. Keeping images out of the arena avoids forcing optimal-tiled images into the arena's memory type and keeps the arena sized by buffers, not textures. |
| Heaps and placed resources | `CreateHeap` reserves the requested size in the arena, so a buffer placed at a 64 KiB-aligned offset is that offset into the arena and its console VA is the heap's place plus the offset. A placed texture (1D, 2D or 3D) gets its own `VkDeviceMemory` and a null VA, as in D3D12. Tiled (`64KB_UNDEFINED_SWIZZLE` and Xbox `0x101`) 2D/3D placed textures (D3D-191) and multisampled 2D placed textures (D3D-222) share that backing, answer `GetResourceAllocationInfo` and `GetResourceTiling`, and check heap/offset alignments. Reserved buffers can use heap tile mappings (section 4.2). |
| Descriptor heaps | Shader-visible CBV/SRV/UAV and sampler heaps are allocated in the low 4 GB of the arena. RTV/DSV heaps are host-side only. |
| Shader code | Each FSSM blob is uploaded to the arena; its console VA is the code base. |
| Upload/readback | `D3D12_HEAP_TYPE_UPLOAD`/`READBACK` resources are separate host-visible buffers outside the arena. Their console VAs are only used by copy APIs; if the game binds one as a vertex/index buffer or reads it in a shader, the layer biases the view/descriptor by `-heap_base` so the shader's addition lands on the host BDA (**ASSUMPTION**; **measured to fail for a V#**, see below). |

Non-arena buffers (upload/readback) can still be addressed by the shader without putting them in the arena: pass
`console_va = host_bda - heap_base` (mod 2^64) in root descriptors and vertex/index/indirect views, so the shader's
`heap_base + console_va` equals the host BDA. For descriptors, patch the V# base field the same way when the layer
copies or scans them. This relies on the game not using the raw VA for anything except shader addressing
(**ASSUMPTION**); if it does, the resource must move into the arena instead.

**Measured (D3D-53, `d3d12_translated_draw_test`):** the bias works for 64-bit root arguments (`d3d12_compute_test`: a root
CBV and a table slot that point at an upload buffer) but **not for a V#**. A V#'s base field is 48 bits
(dword0 and dword1[15:0]), and an upload buffer's console VA, its device address minus `heap_base`, is negative on both test
GPUs (`0xffffffffffce0000` on the AMD Radeon 880M, `0xffffffffffedc000` on the RTX 5070 Laptop; the buffer's device
address lay below the arena's, an allocation-order fact of these runs, so it is not guaranteed either way). The shader
adds `heap_base` to the truncated 48-bit value, which is a wild address: the draw faults and the whole queue submission is lost
(the clear that came before it is lost as well). Any resource a game can put in a V# (a vertex, constant or structured buffer
behind a descriptor) must therefore be an arena range, whatever the allocator happens to do, upload and readback heaps included. Until the layer does that, the
translated-draw test fetches from a DEFAULT-heap buffer and does not draw from the upload buffer.

**Measured (static, `XboxBvhBuilder.dll`): the BVH builder only takes destination addresses below 2^40.** Its
validation function at `0xc8f0` loads `DestAccelerationStructureData` (the first field of the build description it
reaches through `[rcx+0x68]`) and checks it twice:

| Check | Instruction | On failure |
|---|---|---|
| 256-byte alignment | `test r8b,r8b` | message "…must be 256-byte aligned…" |
| below 2^40 | `cmp r8,10000000000h` / `jae` | message "…can not have more than 40b of address set" |

On a failed check it prints its message through `OutputDebugStringW` after a "D3D12 XDXR Error: " prefix, and then
throws `E_INVALIDARG`: a C++ exception out of the DLL, not an `HRESULT` return.
- The alignment rule is the public `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT`.
- The 40-bit rule has no PC counterpart.

Arena VAs (at most 4095 MiB, so 32-bit) pass. A VA biased from outside the arena does not: the negative upload-heap
VAs above are at or above 2^40 as unsigned values. That is one more reason for the rule above that every buffer
a console client can address is an arena range, whichever heap it comes from.

Why not give each buffer its own BDA and set `heap_base = 0`? Because 25,865 of the 26,503 shaders (97.6%)
truncate descriptor-table pointers to 32 bits, and host BDAs are not guaranteed to be below 4 GB. The arena keeps console VAs low and stable.

Why one arena rather than several? Because the shader gets exactly one `heap_base` for buffer, descriptor-table and
code accesses. A draw can reference a descriptor table in one heap and a vertex buffer in another; both must
satisfy the same `heap_base + va` equation. Several non-contiguous VkBuffers cannot provide that equation.

Image keys are 256-byte-aligned byte addresses. The [RDNA 2 ISA, §8.2.6, Table 45][rdna2-image-resource] defines
the 40-bit T# base field as address bits 47:8. `gfx10_descriptors.h` encodes bits 39:8 in word 0 and bits 47:40
in word 1[7:0]. The largest representable key is `0xFFFFFFFFFF00` (2^48 − 256 bytes). Bit 48 is not stored,
so keys separated by 2^48 bytes produce the same base field.

The runtime allocates image keys from `0x100000000` (4 GiB) in 256-byte steps, above the 4095 MiB arena cap.
**Open:** whether any shader in the game uses an image VA as a raw pointer rather than through its T#.

[rdna2-image-resource]: https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf

**Memory type policy.** The first boot needs correctness, not bandwidth:

- M0–M3: allocate the arena as `HOST_VISIBLE | HOST_COHERENT` (preferably also `DEVICE_LOCAL` when large BAR is
  available). The CPU can then write descriptor heaps, shader blobs and uploaded buffers directly. Images are
  already device-local, so rendering still uses fast memory for textures. This is slow for dynamic buffers but
  removes staging and host/console address translation from the bring-up path.
  - A memory type is only tried when its heap can hold the whole allocation (`memory_type_order` in
    `runtime/src/vulkan_memory.h`, beside D3D-96's `resizable_bar`). Without Resizable BAR the only `DEVICE_LOCAL | HOST_VISIBLE` type sits in a
    heap of 256 MiB or less (214 MiB on an RTX 3090 with driver 616.56), so the 256 MiB default arena lands in
    system memory there. Asking that heap for it broke VUID-vkAllocateMemory-pAllocateInfo-01713 even though the
    driver returned `VK_SUCCESS` (reported by CharlyRien).
- M4+: allocate the arena `DEVICE_LOCAL` and stage CPU writes (descriptor heaps, shader blobs, buffer uploads)
  through a host-visible staging buffer, copied before the commands that use them. The hash table is not in the
  arena either: it is a normal storage buffer in `DEVICE_LOCAL` memory, uploaded from a mapped staging copy before
  the commands that read it (D3D-76).
- Upload-heap buffers prefer `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` memory when the GPU has Resizable BAR
  (D3D-96): the CPU writes them through the BAR and the GPU reads them from VRAM. `leonida::resizable_bar`
  (`runtime/src/vulkan_memory.h`) calls it Resizable BAR when such a type sits in a device-local heap larger than
  256 MiB. Without Resizable BAR that type lives in a window of 256 MiB or less (214 MiB on an RTX 3090) that every
  upload buffer would share, so they stay in system memory. If VRAM is full, the allocation falls back to system
  memory. The device logs which one it uses (`[rt] d3d12: upload heaps in ...`), and `LEONIDA_UPLOAD_SYSMEM=1`
  keeps them in system memory to compare the two. Readback buffers keep preferring `HOST_CACHED`.
- Textures and default-heap buffers outside the arena prefer `DEVICE_LOCAL` and need no host access. When VRAM is
  full they still get memory, from the next type `memory_type_order` lists, which is system memory. The first one
  that does is logged once (`[rt] d3d12: a N-byte allocation did not fit in device-local memory ...`, D3D-299,
  `leonida::spilled_to_host`), so a run on a 4 GB GPU shows that it spilled (#2375).

**Capacity.** The arena is sized by buffers, descriptor heaps and shader code, not by textures. The allocator
tracks the high-water mark; start with a configurable capacity (for example 4 GiB for the loading/menu path) and
fail with a clear message when a console VA would fall outside it. Growing means creating a new arena, copying,
and updating `heap_base` for all subsequent commands; do it at a frame boundary, never with work in flight.

**Open:** the game's real buffer VA range and peak working set are unknown until it runs. `maxBufferSize` (with
`VK_KHR_maintenance4`) and `maxMemoryAllocationSize` bound the arena; a large open-world buffer working set may not
fit on smaller PC GPUs. This is a product requirement, not something the layer can hide.

**Reserved resources.** `CreateReservedResource` and the queue tile APIs use Vulkan sparse binding for buffers and
compatible single-mip RGBA8 2D textures (section 4.2). Other descriptions have fully allocated backing. Wider
texture coverage and the game's use of tile streaming remain open.

### 3.4 Descriptor heaps and the bindless model

The game can use D3D12 descriptors in three ways, and the layer must handle all of them:

1. **Root descriptor tables.** `SetGraphicsRootDescriptorTable` supplies a GPU handle; the shader receives the
   table's console VA in its user data and loads T#/S# structures from it.
2. **Direct heap indexing.** The root signature has `D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`
   and/or `..._SAMPLER_HEAP_DIRECTLY_INDEXED`, and the shader indexes `ResourceDescriptorHeap` /
   `SamplerDescriptorHeap` directly. The console driver supplies the bound heap base and stride; the shader loads
   the descriptor at `base + index * stride`. **Measured (D3D-329): each heap base arrives as a 32-bit pointer in
   an input SGPR the shader reads directly, inside the user data for compute; no stride arrives at all.**
   `tools/heap_index_census.py` over the 1.0.2.1 corpus (26,503 listings; 1,473 with embedded data words skipped)
   follows every descriptor-width `s_load_dwordx4/x8/x16` back to the SGPRs it starts from. Of 770,412 such loads,
   769,366 (99.86%) start from an entry SGPR used as a 32-bit pointer (high word a constant 0), 0 from a pointer
   loaded from memory, and 1,041 could not be followed (0.14%, all but 63 in heap-indexing shaders). That holds for
   the 18,066 shaders whose `SFI0` sets either heap-indexing bit as much as for the 6,964 that set neither. The
   heap-indexing shaders add a base read with a run-time offset (the heap index): for example `s10` in CS, `s23` in
   MS and `s15` in PS for the resource heap, and a separate base for the sampler heap (`s11` in CS, `s24` in MS,
   `s16` in PS). For compute this is settled: every base lies below the shader's `COMPUTE_PGM_RSRC2` `USER_SGPR`
   count, so it is user data. For VS, PS, MS and GS no user-SGPR count is measured; every base lies below `s32` and
   none is the NGG `s3`, so they are user data under the translator's 32-SGPR assumption, and the user count of the
   graphics stages (question 3 in the table below) is what would confirm it. Which SGPR holds which heap varies
   with the stage and the root signature, so placing the bases belongs to the root-signature mapping, not to a new
   translator input. No stride is read at run time: static offsets of T# loads are multiples of 32 and of S# loads
   multiples of 16, so the stride is compiled into the code. In the runtime shim, `CreateRootSignature` records the
   `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and `SAMPLER_HEAP_DIRECTLY_INDEXED` root-signature flags and `SetDescriptorHeaps`
   tracks the bound shader-visible heap GPU bases (`d3d12_set_descriptor_heaps`), while placing those bases into
   per-stage user SGPRs remains open on question 3.
3. **Game-built descriptor memory.** The game writes raw T#/S# bytes into a heap itself, without calling
   `CreateShaderResourceView`. The shaders then read those bytes exactly as in case 1.

The layer's descriptor model:

- A `DescriptorHeap` has a host shadow and, if shader-visible, an arena range. CPU handles point into the shadow;
  GPU handles are arena VAs. `Create*View`, `CopyDescriptors` and `CopyDescriptorsSimple` write the shadow and mark
  the range dirty. The execute-time scan below is the correctness path, so a game that writes raw descriptor bytes
  through a CPU handle still works; dirty tracking is only an optimisation.
- **Heap start-handle ABI.** Prior art measured `GetCPUDescriptorHandleForHeapStart` and
  `GetGPUDescriptorHandleForHeapStart` (heap slots 9/10) returning the 8-byte handle in RAX on the Xbox build,
  while the PC call returns it through a hidden pointer in RDX. A generic sret/IDL thunk must special-case these
  two slots or the handle is garbage. This is a method-level ABI quirk; the heap object is still the public
  `ID3D12DescriptorHeap` interface.
- The bytes written for an image/sampler descriptor are **GFX10 T#/S# structures**, the same structures the
  translator reads. The layer writes them itself in `Create*View`; this does not depend on what the console driver
  wrote. If the game builds descriptors in memory itself, it must use the same format (the shader reads raw T#/S#),
  which is the assumption to verify. If it uses an opaque format, the layer must translate at bind time.
- At execution, for each bound table (or the whole bound heap when direct indexing is enabled), the layer walks
  the declared descriptor range with the heap's stride, parses each T#/S#, and ensures a bindless entry exists:
  - image T# → `VkImageView` (format, view type, mip range, array layers) → bindless image index;
  - sampler S# → `VkSampler` (filter, address modes, LOD, compare, anisotropy) → bindless sampler index;
  - buffer V# / CBV → no Vulkan view; the descriptor already contains a console VA the shader dereferences.
- **Buffer UAVs.** `CreateUnorderedAccessView` on a buffer writes a V# in the first 16 bytes of the slot (the rest of
  the 32 bytes is zero): `BASE_ADDRESS` = the resource's GPU VA + `FirstElement` * the element size, and by view kind
  - *raw* (`R32_TYPELESS` + `RAW`): stride 0, `NUM_RECORDS` = bytes, `OOB_SELECT` 3 (raw);
  - *structured* (`UNKNOWN` + `StructureByteStride`): stride, `NUM_RECORDS` = records, `OOB_SELECT` 0;
  - *typed* (any other format the typed-UAV table has: `uav_clear.h` layouts that `gfx10_descriptors.h`'s
    `format_table` names with a code below 128): `FORMAT` = the `GFX10_FORMAT_*` code, stride = the element size in
    bytes, `NUM_RECORDS` = elements, `DST_SEL` = the format's own channel order (`format_swizzle`: R8 reads as
    (x, 0, 0, 1), B8G8R8A8 as (z, y, x, w)), `OOB_SELECT` 1 (structured: only the index is checked against
    `NUM_RECORDS`, `docs/buffer_bounds.md`). This is what `buffer_load_format_*` / `buffer_store_format_*` / the
    `buffer_atomic_*` of a translated `RWBuffer<T>` read; the translator decodes the element from the V#'s `FORMAT`
    at run time (`rdna2spirv.py` `FORMATS`) and bounds-checks the whole element against `NUM_RECORDS`. A format the
    layer cannot name (a typeless or 96-bit one, a compressed one, anything whose code is 128 or more, `R16G16_SNORM`
    and the other texture formats `vulkan_format` has no row for) leaves a null slot, logged once. Typed clears
    (`ClearUnorderedAccessViewUint/Float`) read the same V# back, so they work on these views and on game-built ones.
  - **UAV counters** (`pCounterResource` + `CounterOffsetInBytes`, structured views only). D3D12 keeps the counter in
    an application-visible buffer: there is no `CopyStructureCount` (that is D3D11); a game copies the counter with
    `CopyBufferRegion` from `pCounterResource` at `CounterOffsetInBytes` (this is "CopyStructureCount" in D3D12 and
    needs nothing from the layer), and resets it with `ClearUnorderedAccessViewUint` on a raw view of that word or a
    copy from an upload buffer. The layer refuses (null slot, logged once) a counter on a non-structured view, a
    counter resource that is not a buffer created with `ALLOW_UNORDERED_ACCESS`, an offset that is not a multiple of
    `D3D12_UAV_COUNTER_PLACEMENT_ALIGNMENT` (4096) or leaves no room for 4 bytes, and a non-zero offset with no counter
    resource, the combinations D3D12 documents as invalid. **Hypothesis for the console contract (not measured):** the
    shader gets the counter as a second V#, raw, 4 bytes (stride 0, `NUM_RECORDS` 4, `OOB_SELECT` 3) at address
    counter VA + offset, in **bytes 16..31 of the same descriptor slot**, which is otherwise unused for a buffer
    (a V# is 16 bytes in a 32-byte slot), and `AppendStructuredBuffer` / `ConsumeStructuredBuffer` compile to
    `buffer_atomic_add` / `buffer_atomic_sub` with `glc` on it, the returned old value (minus one for consume) being
    the record index. Why this guess: the translator has no `ds_append` / `ds_consume` / GDS path and the corpus scan
    (open-questions table below, row 10) found no GDS instruction, so a counter in memory is the only form left; where the
    counter V# sits (a second slot, a user SGPR pair, a spill) is open until a shader that uses one is traced
    (row 16 of the same table). `d3d12_buffer_uav_test` runs `cs_buf_counter.asm`, a hand-written shader of exactly that shape:
    256 lanes append, 128 consume, the counter reads 256 and then 128 through `CopyBufferRegion`, and the 256
    appended records are each present once. If the console puts the counter elsewhere, only `write_buffer_uav` moves.
- Unbounded descriptor ranges (`NumDescriptors = UINT_MAX`) have no end. Scan from the table handle to the end of
  the bound heap, and cache parsed descriptors by key so unchanged slots cost nothing. **Open:** whether the game
  uses unbounded tables; the trace will show the root-signature ranges.
- The heap stride is the value the layer returns from `GetDescriptorHandleIncrementSize`. Prior art forwards device
  slot 15 (`GetDescriptorHandleIncrementSize`) unchanged, but the actual increment values are still **Open**. Choose
  32 bytes for CBV/SRV/UAV (the T# size) and 32 bytes for samplers (16 bytes of S# plus padding) pending the console
  value. The game may hardcode the console increments or infer them from shader address arithmetic; the shader's
  shift amount gives the stride it expects, and the layer must match it.
- `CopyDescriptors` copies raw bytes. If the descriptor was produced by our `Create*View`, the copy is a valid
  T#/S#; if it came from game memory, it is still valid raw data. No re-encoding is needed.
- Handle validation: a GPU descriptor handle is an arena VA; the heap registry maps the containing range to a
  heap. A table handle outside the bound heaps is a validation error; log it and resolve to fallback index 0
  rather than reading arbitrary memory.
- RTV/DSV creation builds a `VkImageView` and stores it in the host-side heap; `OMSetRenderTargets` resolves those
  handles to views for the rendering scope. RTV/DSV descriptors are never shader-visible.
- View creation maps `D3D12_SRV_DIMENSION`/`D3D12_UAV_DIMENSION` to `VkImageViewType`, the mip and layer ranges,
  sample count, and `D3D12_SHADER_COMPONENT_MAPPING` to `VkComponentMapping`. Single-layer 1D textures have default and
  explicit shader-resource views on sampled binding 5. Single-sampled 1D, 2D, 2D-array, cube and 3D colour SRVs apply
  the T# minimum LOD in resource mip coordinates. A clamp inside the exposed mip range uses
  `VK_EXT_image_view_min_lod`. Vulkan allows the image-view minimum to be rounded down to an integer.
  With nearest mip filtering, the UHD 770 and Arrow Lake Xe-LPG returned mip 0 for a 2D minimum of 0.75.
  The Xe-LPG also returned mip 1 for a 2D-array minimum of 1.75. RTX and Radeon returned mips 1 and 2.
  `d3d12_descriptor` accepts either rule for a fractional clamp (all texels of a view following the float or the
  floored minimum), and prints which one the driver used.
  See [Vulkan image-level selection](https://docs.vulkan.org/spec/latest/chapters/textures.html#textures-image-level-selection)
  and the [Intel readbacks](https://github.com/ItsNotRexed/Leonida/pull/1015#issuecomment-7075135647).
  If the logical device lacks the feature, the runtime logs the unsupported view
  and its descriptor resolves to the fallback. A clamp at or below the first mip needs no extension. Above the
  last mip, the view retains its dimensions and maps sampled channels to zero while preserving constant channels.
- Set 0 is a fixed bindless layout, sized once when the device creates it. The defaults are 4096 sampled 2D
  images, 1024 samplers, 1024 2D-array images, 256 1D, 256 3D and 256 per storage dimension/class. The 2D, sampler and
  storage arrays grow to what the device's limits allow, up to 65536, 4096 and 4096 (see `vulkan_bindless.h`
  above). The set is not recreated to grow, because recreating it would invalidate the pipeline
  layouts built on it. Use `UPDATE_AFTER_BIND` and `PARTIALLY_BOUND` so a new view does not require rewriting
  every descriptor; without those features, rebuild the set when it changes.
- The command buffer binds set 0 once; descriptor-heap changes only update the arrays and the hash table.

**Descriptor lifetime.** A bindless index may be referenced by in-flight command buffers after the resource is
destroyed. Views/samplers are retired on a fence-value queue; the hash-table entry can stay (a stale key is only a
miss for a new descriptor, not a correctness problem) until the slot is reused. Reusing a slot for a new resource
updates the index in place; the key match makes the lookup correct.

### 3.5 Root signature to user SGPRs

This is the least verifiable part of the layer. D3D12 gives the application a root signature; the console driver
turns it into the user SGPRs (`s0..s31`) that the shader reads. The translator already copies those 32 dwords from
push constants into `s0..s31`. The layer must produce them.

The default mapping (implemented by `rt::rootsig::pack` in `runtime/src/root_signature.{h,cpp}`) follows root
parameter order:

| Root parameter | User-data dwords |
|---|---|
| `D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS` | one dword per constant |
| `D3D12_ROOT_PARAMETER_TYPE_CBV` / `_SRV` / `_UAV` | two dwords: the 64-bit console VA passed to `Set*Root*View` |
| `D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE` | two dwords: the table's console VA (a `D3D12_GPU_DESCRIPTOR_HANDLE`) |
| static sampler | none; registered in the hash table at root-signature creation |

Before the draw/dispatch, the layer copies the current root arguments into a 32-dword block in that order and
uploads it as push constants. Missing parameters are zero.

Measured on the corpus (`python -I tools/user_sgpr_census.py <spirv/index.json> --base ps=8 vs=8 gs=8 cs=4 ms=16`, all 26,503 shaders). These are physical user SGPRs; the tool cannot see logical root-parameter indices.
- The lowest user SGPR read as a pointer is s8 in most PS (9,830 of 12,101) and VS (1,660 of 2,236), s4 in CS (2,364 of 2,689) and s16 in mesh shaders (7,977 of 9,282).
- Pointers are 32-bit (`ptr32_lo`) in every PS/VS/CS/MS shader that reads one; only the 195 GS read a 64-bit pointer (`ptr64_lo` at s8).
- Consecutive pointer SGPRs in one shader are most often adjacent (distance 1: 22,008 PS, 14,548 MS, 3,059 CS, 1,836 VS), and the SGPR after a 32-bit pointer is never read as a `ptr64_hi`.
- Below those bases only `const32` reads appear, at odd SGPRs in PS/VS/CS (s1, s3, s5, s7), and s2 in GS: a reserved block whose meaning is not established here.

Taken together this disagrees with the packing above (from s0, two dwords per table) and is consistent with a per-stage base (s8 VS/PS, s4 CS, s16 MS) and one dword per table pointer. It is a census of shader reads, not a confirmed mapping: it does not show which root parameter feeds which SGPR, and the GS behave differently. `--expect ps=8 vs=8 cs=4 ms=16` turns it into a check; `tests/test_user_sgpr_census.py` shows that the two packing models give different results.

**Console layout, measured for compute (D3D-448).** `rt::rootsig::pack_stage` lays the parameters out for one stage:
in root order from the stage base, each while it fits whole, a constant taking one dword per value, a root
CBV/SRV/UAV two (its 64-bit VA), a descriptor table **one** (the 32-bit VA of its first descriptor; the heaps live
below 4 GiB). A parameter that does not fit goes, with every later one, to the spill array behind `s[0:1]`
(section 7), which is not modelled, so such a signature is refused. How each part is known:
- **Compute, base s4: measured three ways.**
  - The console compiler's output (section 7).
  - The package's FidelityFX DLL, whose 29 compute containers carry their `RTS0` next to the console-compiled
    program (`tools/rootsig_census.py <fspx> --disasm <fspx>/disasm`, Xbox Series 1.0.2.1). Base s4 plus the
    signature's span in this layout equals the program's RSRC2 `USER_SGPR` count in 29 of 29: 27 signatures
    (table, table, four root CBVs) span 10 dwords and launch with 14 user SGPRs, 2 (table, table, one root CBV)
    span 4 and launch with 8. The `pack()` layout spans 12 and 6, two short of both.
  - The reads: `tools/root_sig_contract.py <fspx index.json> --rootsig <that shape> --model console` finds all 20
    translated programs compatible (tables read at s4/s5 as 32-bit pointers, each root CBV's VA copied whole as
    two dwords from s6). The `pack()` layout contradicts 7 of them, which read the second table at s5, the first
    table's high half in that layout.
- **Vertex/pixel/geometry s8 and mesh s16: a hypothesis.** It rests on the census above only; no signature
  confirms these bases, and visibility is not modelled (every parameter takes its place in every stage).

The runtime uses the console layout for the **compute** bind point only. Graphics keeps `pack()`: every stage of a
pipeline reads `s0..s31` from one push-constant block at offset 0 (`tools/rdna2spirv.py`, `ShaderUserData` in
`runtime/src/vulkan_replay.h`), so a mesh shader (base s16) and a pixel shader (base s8) in one pipeline would need
the same root argument at two offsets. That needs per-stage user-data windows in the translator's push-constant
contract (D3D-448 follow-up). A root signature carries both layouts; one that only one of them can place is
created, and binding it at the other bind point invalidates the command list (`Close` returns `E_INVALIDARG`).
`s0..s3` of a compute dispatch are the driver's block; the layer leaves them zero (109 CS read `s1` as a constant,
42 `s3`; their meaning is open).

**Measured: a real signature beside the count it produced.** `tools/fspx_unpack.py` decodes `COMPUTE_PGM_RSRC2` of
each of the 29 compute programs in `amd_fidelityfx_upscaler_xbox.dll`, and its `USER_SGPR` field is the count the
console launched the dispatch with. Every one of those 29 containers also carries an `RTS0`, which makes this the
only place in the package where a serialized root signature sits beside the dword count it produced
(`tools/rootsig_census.py`). The game's own 26,503 containers carry no `RTS0`, so their read sets cannot be joined
with a signature and cannot settle it either.

All 29 are version 1.1 with `Flags = 0`, 168 parameters in total (58 tables and 110 root CBVs, no 32-bit constants,
SRVs or UAVs), every parameter at `D3D12_SHADER_VISIBILITY_ALL`, and 27 of them carry one static sampler. They come
in two shapes:

| Parameters | Signatures | `USER_SGPR` | Layer model (2 dwords per table) | One dword per table |
|---|---|---|---|---|
| 4 CBV + 2 tables | 27 | 14 | 12 | 10 |
| 1 CBV + 2 tables | 2 | 8 | 6 | 4 |

The surplus is 2 dwords under the layer's model and 4 under one-per-table, and it is the same for both shapes: the
driver put a fixed block below the parameters. Arithmetic alone therefore does not separate the two models. The
disassembly does (`tools/rootsig_census.py --disasm`, over the same 29 programs):

- `s4` is copied into the low half of a pair, the high half zeroed, and passed to `s_load` as the base in 29 of 29;
  `s5` is in 16. The two declared tables therefore sit at s4 and s5, one dword each, in signature order, and the 13
  programs that never read `s5` are using only the first.
- A 64-bit pair is copied whole from `s6:s7` in 27 of the 29. Five of those also copy from `s8:s9`, `s10:s11` and
  `s12:s13`, which matches their four root CBVs at two dwords each. The other two read none of `s6` to `s13`: they
  declare four CBVs and never touch them.
- 27 read none of `s0` to `s3`. In `_05` the program adds `s0` and `s3` to `s4`, and `_07` multiplies by `s0`, so
  the block below the tables is not always dead, but no program here uses one of those dwords as a pointer.

That is `USER_SGPR = 4 + 2 per table + 2 per CBV`, which gives the measured 14 and 8, and it is the one-dword-per-
table model behind a four-dword prefix. The CS census above agrees: its lowest pointer is s4, where a four-dword
block puts the first table, while two dwords per table would put it at s2.

Still open: which CBV index lands in which pair. The four pairs do appear in declaration order, but nothing in the
package proves that order, and every parameter in these 29 is `D3D12_SHADER_VISIBILITY_ALL`, so they say nothing
about the visibility question below.

Spill policy: a signature needing more than 32 user SGPRs, or more than the 64-dword D3D12 cost limit, is
refused loudly (`pack` returns the offending parameter) instead of being truncated. The console spills through
`__XB_GetRootSignatureSpillPointer`; until that layout is measured, refusing is the only correct behaviour.

Measured (D3D-315): zero below-base spill-pointer candidates in the translated shaders.
`python -I tools/root_spill_census.py <spirv/index.json> --carve <dxbc/index.json> --expect-no-spill` exits 0 on
the 26,501 translated shaders (all but the 2 compute shaders refused for workgroup memory under wave64 emulation),
Xbox Series 1.0.2.1:
- **No root signatures in the containers.** 0 of the 26,503 carved containers have an `RTS0` part. The game
  builds its root signatures at run time, so the number that exceed 32 dwords is open until they are dumped from
  the executable.
- **No spill-pointer candidates.** No user-SGPR read used as an address (`ptr32_lo`, `ptr64_lo`, `ptr64_hi`) sits
  below the stage base (ps/vs/gs s8, cs s4, ms s16) in any stage. For compute that block holds the measured spill
  pointer `s[0:1]` (section 7); here `s0` and `s1` are never read as an address, only `s1` as a constant
  (109 CS, 551 PS, 12 VS).
- **Compute stays far below the budget.** The largest RSRC2 `USER_SGPR` count is 14 (31 of 2,687 shaders), so at
  most 10 of the 28 root dwords above s4 are used.
- **The highest reads are not spills.** PS reads reach s31 (8 shaders), s29 (7), s27 (1) and s25 (8); MS s29 (2)
  and s27 (8); VS s25 (2). Each of those top reads is a 32-bit constant after a gap, not an address.

What this does not settle:
- **How many runtime-created signatures exceed the budget.** The containers carry none, so that count waits for
  signatures dumped from the executable.
- **Why a particular `pack()` refusal happens.** `pack()` counts two dwords per table from s0, which the census
  above disagrees with, so its 32-dword limit and the console's budget need not agree. Whether a given refusal is
  that model or a real spill is open until the signature and its shaders are compared.
- **A spill behind a pointer at or above the base.** It reads the same as a table pointer, so the census can't
  tell the two apart.

The spill layout stays unimplemented: no shader in the translated set reads a candidate a spill could sit behind.

Reconciliation: `tools/root_sig_contract.py` checks a candidate root signature against every shader's
`user_sgpr_reads` from `spirv/index.json` (const32 vs constant dwords, 64-bit VAs vs view/table pairs) and fails
the build on the first contradiction. Until the console mapping in §3.5 is confirmed by a PSO, this check — not
a rendered frame — is the acceptance gate for the mapping above.

Blob format: `rt::rootsig::parse_rts0` reads the `RTS0` chunk of a serialized root signature, versions 1.0, 1.1
and 1.2. `root_signature_oracle_test` checks it on Windows against blobs the PC `d3d12.dll` serializes. These
layouts are measured, not assumed:
- a descriptor table's payload is `{num_ranges, ranges_offset}`, with the ranges at that offset;
- ranges are 20 bytes, or 24 with the 1.1/1.2 flags dword, in `D3D12_DESCRIPTOR_RANGE(1)` order:
  `{type, count, base register, space, offset}` in 1.0 and `{type, count, base register, space, flags, offset}` in
  1.1/1.2, so the flags come *before* `OffsetInDescriptorsFromTableStart` (a range with `DATA_STATIC` and offset 7
  serializes as `… 8 7`). The parser keeps all six fields;
- root descriptors carry a flags dword from 1.1 on;
- static samplers are 13 dwords (`D3D12_STATIC_SAMPLER_DESC` order). In 1.2 they are 14: `D3D12_STATIC_SAMPLER_DESC1`
  appends `Flags`, i.e. UINT border colours and non-normalized coordinates.

Static samplers take no user-data dwords, so 1.2 changes nothing in `pack`.

**Root-signature flag quirk.** Prior art measured Xbox root-signature descriptors carrying flags `0x03000000`; the
PC serializer rejects them with `E_INVALIDARG` because its flags live in the low 16 bits. Clear the high byte on a
private copy before calling the host serializer/deserializer; never modify the game's descriptor. The meaning of the
bits is unknown. The v1.0 `D3D12SerializeRootSignature` path keeps flags at offset +32, and a descriptor with
`0x3000001` also fails on the PC runtime.

**Measured: where the bits come from in the package.** Of the two console-compiled clients, only `XboxBvhBuilder.dll`
sets them, and it does so twice.

- **In its shaders:** all 52 `RTS0` parts in its DXBC containers (version 2, i.e. 1.1) carry `Flags = 0x03000000`.
- **At run time:** the builder's only `D3D12SerializeVersionedRootSignature` call (`0xc32f`, import address table
  `0x180024250`, `d3d12_xs.dll`) is in the function at `0xc2d0`, which ten call sites reach. That function copies the
  48-byte `D3D12_VERSIONED_ROOT_SIGNATURE_DESC` to the stack (`[rsp+40h]`) and, with no branch before it, ORs the
  bits into the copy's v1.1 `Flags` (`+0x28`): `or dword ptr [rsp+68h],3000000h` at `0xc30e`.

`amd_fidelityfx_upscaler_xbox.dll` does neither. Its 29 `RTS0` parts carry `Flags = 0`, and no instruction in it uses
the constant `0x3000000`. So the bits are not a property of every console root signature; this builder adds them. The
game executable is not in the package, so whether its root signatures carry them is open.

**Version 1.2 on older runtimes.** The in-box `d3d12.dll` of Windows 10 (measured on 22H2, 10.0.19045, without the
Agility SDK) and of Windows 11 23H2 (#361) serializes root signatures only up to 1.1. On Windows 10 it rejects even an
empty 1.2 description with `E_INVALIDARG`, and `D3D12_FEATURE_ROOT_SIGNATURE` reports `HighestVersion` 1.1 (#528).
Version 1.2 differs from 1.1 only in
`D3D12_STATIC_SAMPLER_DESC1::Flags`. So on such a host, the `d3d12_xs` serializer rewrites a 1.2 description whose
static samplers all have `Flags == 0` as the same root signature in 1.1. It refuses one that uses sampler flags (a
UINT border colour or non-normalized coordinates) with a log line, since 1.1 can't express them. The host is probed
once with an empty 1.2 description. `root_sig_test` checks the rewrite field by field and through the host
deserializer.

Things that can invalidate this simple model, all **Open**:

- **Shader visibility.** The layer starts with one global block in root-parameter order and treats visibility as
  access control, not as a per-stage reordering. If the console driver instead omits invisible parameters per
  stage, a VS and a PS would need different values at the same push-constant offsets, which Vulkan cannot express
  with one block; that would force per-stage blocks or a per-stage indirection. **Measured, VS/PS (D3D-318):**
  consistent with one block in root order. The game's shaders carry no `RTS0` and the executable that pairs them
  into pipelines is not in the package, so `python -I tools/user_sgpr_census.py <spirv/index.json> --pairs` pairs
  a VS with a PS by technique name instead (names alike apart from the `VS_`/`PS_` token) and compares the user
  SGPRs both read. Over the 1.0.2.1 corpus: 166 techniques have both stages (225 VS, 203 PS, 358 combinations).
  - The 379 SGPRs both stages read have the same kind (pointer or constant) in every case. Over all 26,196,976
    VS × PS combinations, the control, 25.5% of the shared reads differ in kind.
  - In 79 combinations one stage reads an SGPR the other skips, below an SGPR both read. If that parameter were
    omitted from the other stage, the parameters above it would move down one SGPR there; all 111 shared reads
    above such an SGPR still agree.

  It can't tell a hidden parameter from a visible one a stage doesn't read, and it doesn't see the root signature,
  so the executable's pipeline table would settle it; `tests/test_user_sgpr_census.py` shows that the two models
  give different counts. Mesh shaders (base s16) and the compute signatures are not covered here.
- **Root signature larger than 32 dwords.** D3D12 allows up to 64 dwords. The console has 32 user SGPRs, so a
  driver must spill the rest to memory, probably behind a pointer in the user data. The current shader contract has
  no representation for that. The likely answer is that this game keeps its root signatures small (bindless
  descriptors, a few constants), but it must be measured. If a large root signature appears, the spill layout has
  to be reverse-engineered from the shader's loads. Measured on the shader side (D3D-315, spill policy above):
  zero below-base spill-pointer candidates in the translated shaders. The root signatures themselves, and how many
  exceed the budget, are still to be dumped from the executable.
- **Direct heap indexing.** The bound heap base and stride are not root parameters. Measured (D3D-329, section
  3.4): each heap's base reaches the shader as a 32-bit pointer in an entry SGPR, inside the user data for compute
  (and below `s32` for the graphics stages, whose user count is not measured yet), and the stride is compiled into
  the shader's offsets. So the translator needs no new runtime value; the layer has to write the bound heap bases
  into the user-data slots the root-signature mapping assigns them. The root signature
  flag tells us direct indexing is enabled; the shader's address arithmetic identifies the base.
- **Static samplers.** D3D12 static samplers do not consume root parameters. The console driver may place their
  S# values in a fixed table. The layer registers every static sampler's S# in the hash table, so a shader that
  loads it by value works; a shader that expects a driver table needs that table's location (**Open**).
  `CreateRootSignature` does the registering, for 1.0, 1.1 and 1.2 blobs: each static sampler is encoded as
  `CreateSampler2` encodes the same description (the five `D3D12_STATIC_BORDER_COLOR` values, two of them UINT, and
  the 1.2 flags), and gets a sampler in set 0 and a key in the hash table without a heap slot. The entry outlives
  the root signature. `d3d12_descriptor_test` checks the key before and after and samples with it.
  loads it by value works; a shader that expects a driver table needs that table's location. **Measured
  (SHADER-735): no sampling instruction in the corpus was found reading an S# from a fixed table** (0 of 366,628;
  0.66% stay undetermined under the conservative reading), so on this evidence the layer has none to emulate; the
  counts and their limits are under "Where the sampled S# comes from" below.
- **Descriptor-table high-dword truncation.** If a shader forces the high dword of a table pointer to zero, the
  table must be below 4 GB; the layer allocates all shader-visible heaps there. If a shader forces a resource VA's
  high dword to zero, the resource must also be low; that would constrain the arena layout. Measured on 2026-11-22
  over package 1.0.2.1 (`python -I tools/ptr_truncation_census.py <work>/shaders/disasm`, all 26,503 shaders, the
  `truncates_high` / `ptr32_lo` rule of `tools/shader_meta.py`): 25,865 shaders truncate at least one pointer
  (VS 2,038 of 2,236, PS 11,666 of 12,101, CS 2,684 of 2,689, GS 195 of 195, MS 9,282 of 9,282), 86,036 truncated
  pointers in all, at most 5 per shader. Every one is the base of a 4/8/16-dword descriptor load (`s_load_dwordx4/x8/x16`,
  a table pointer) from a user SGPR, and no 1–2 dword `s_load` base is truncated. No `global_*` in the corpus
  takes its base from SGPRs: its 10,548 `global_*` instructions (530 shaders) address through VGPR pairs, which the
  classifier does not follow. 25,669 shaders truncate every user-SGPR pointer they dereference, 196 only some (the
  195 GS, whose 64-bit s8 pointer has a high dword the classifier cannot resolve, and one CS), and the other 638
  dereference no user-SGPR pointer. Bases not rooted in a user SGPR (462 shader/access combinations) never have a constant-zero
  high dword. So what is measured to need the low 4 GB is the descriptor heaps; buffer addresses reach the shaders
  inside V#s (48-bit bases) or in VGPRs for `global_*`, and this count sees neither, so it does not show that
  buffers may sit above 4 GB.

**Where the sampled S# comes from** (SHADER-735; counts only). `tools/sampler_source_census.py` follows the four
SGPRs of the sampler operand of every `image_sample*`, `image_gather4*` and `image_get_lod` back to where their
values are made: through constants, scalar loads, SGPRs spilled to a VGPR and read back, and joins, where it keeps
the sources of every path. Over the 1.0.2.1 corpus (26,503 listings, none skipped; the ones with embedded data
words are re-anchored as the translator does) 14,025 shaders sample, with 366,628 such instructions:

| stage | sampling instructions | loaded through user data | literal | held in user SGPRs | other base | undetermined |
|---|---|---|---|---|---|---|
| CS | 120,816 | 63,141 | 57,674 | 0 | 0 | 1 |
| MS | 21,992 | 10,142 | 9,723 | 0 | 0 | 2,127 |
| PS | 221,157 | 115,109 | 106,048 | 0 | 0 | 0 |
| VS | 2,663 | 1,020 | 1,351 | 0 | 0 | 292 |
| all (the 195 GS do not sample) | 366,628 | 189,412 | 174,796 | 0 | 0 | 2,420 |

- **Other base** is what a driver table would be: some dword of the S#, on some path, held in or loaded through a
  constant address, a system SGPR (at or above `USER_SGPR` in CS; `s32` and up in the graphics stages, and the
  NGG `s3` in MS and GS) or a code-relative address. It wins over every other class, the undetermined one
  included, and it is empty in every stage.
- **Loaded through user data**: an `s_load_dwordx4` or wider at an address that starts from a user SGPR, 179,602
  at a static offset (0x540 at most) and 9,810 at a run-time one. No constant is added to any of these pointers,
  so `constant table address + user index`, which is the same arithmetic as `user pointer + constant` and which
  the classes could not tell apart, does not occur either. All 189,412 are in shaders whose `SFI0` sets the
  sampler-heap bit: no shader without it loads an S# from memory. Whether one of these loads goes through a
  descriptor table or through the sampler heap base of section 3.4 the shader cannot say; both are a user SGPR.
- **Literal**: four immediates, 105,235 in shaders with the sampler-heap bit and 69,561 in shaders without it,
  which sample with nothing else. 28,659 of them, in 6,824 shaders, build one dword with `s_brev_b32`.
  `tools/shader_meta.py` does not fold that instruction, so `literal_samplers` in the index does not list those
  S#s yet and the layer cannot register them from it.
- **Undetermined** (0.66%, in 300 shaders): with every edge followed, some path brings something else into the S#
  registers. 2,355 read the S# back from a VGPR it was spilled to, and one path does not save it there (on the
  paths that do, it is a load through user data); in 64 the paths bring a literal and a load through user data;
  in 1 a path brings a value the dataflow does not follow. All of them go away with one kind of edge: the S# is
  made or saved inside a region the compiler jumps over with `s_cbranch_execz` when no lane is active, and the
  sample comes after the region. Without the taken edge of those branches (the region always runs, as it does
  whenever a lane is active) the same analysis gives all 2,420 a class: 2,419 loaded through user data, 1
  literal. The table keeps them apart because that second reading trusts that nothing samples on the skipping
  path; lane masks are not modelled.

So, as far as the shaders can answer it, this corpus gives no evidence of a driver sampler table: no instruction
lands in the other-base class, and the 2,420 undetermined ones are not shown to be free of it by the conservative
reading, only by the one that assumes nothing samples on an exec-zero skip.
What the census does not show: it reads the shaders, not what the driver writes at the addresses they load from;
the graphics stages' user count is the translator's 32, not a measured one (question 3 in section 8.2); and it
cannot say which S# is a root signature's static sampler. The literals are the candidates, since a static sampler
is known when the shader is compiled and half of the sampling instructions build their S# from immediates, but
that is a hypothesis until the root signatures are matched against them. Reproduce with
`python -I tools/sampler_source_census.py <work>/shaders/dxbc/disasm --jobs 4 --expect-none other_base`, which
exits 1 if an instruction lands in that class.

**What the console's own D3D12 libraries show** (names only, read statically from a user's console system-file
dump; nothing from those files is in this repository):

- Two root-signature ABIs exist. The driver's validation messages name a flag
  `D3D12XBOX_ROOT_SIGNATURE_FLAG_FORCE_MEMORY_BASED_ABI`, and require it for ray tracing (XDXR) and for pipelines
  with an amplification shader. So the default is the register-based ABI this section describes, and a root
  signature can opt into a memory-based one. The shader compiler has the matching switch,
  `XBOX_FORCE_MEMORY_BASED_ABI`.
- The ABI is decided when the shader is compiled, not by the driver at bind time. The shader compiler has the
  intrinsics `__XB_GetShaderUserData` and `__XB_GetRootSignatureSpillPointer`. That answers the "larger than
  32 dwords" question above: parameters that don't fit in the user SGPRs spill to memory, and the shader reads a
  spill pointer from its user data. Because the game ships precompiled shaders, each shader's own user-SGPR reads
  (and any loads through a spill pointer) are the authoritative map; the metadata below is how to recover it.
- `d3d12_xs.dll` is a thin front end (about 50 KB). It loads the user-mode driver, `xgs12_10.dll`, which implements
  the device and command lists and exports `UMD*` entry points. `xg_xs.dll` exports the texture layout functions
  (`XGComputeTexture2DLayout`, `XGComputeOptimalSwizzleMode`, ...), the reference for untiling textures.
- The driver keeps the standard D3D12 root-signature cost limit: 64 dwords, with descriptor tables costing 1 and
  root views 2.

**Required build metadata.** To solve the mapping without guessing, `spirv/index.json` should gain, per shader:

- the set of user-SGPR dwords read before first write, and whether each is used as a 32-bit constant or a 64-bit
  pointer;
- the descriptor stride(s) implied by descriptor address arithmetic;
- whether the shader uses the code base (`s_getpc_b64`) and, if so, in which stage;
- the literal S# values it builds from immediates;
- the input signature (semantic name/index, register, mask, component type), not just the register numbers;
- the FSSM content hash (the PSO lookup key, section 3.6);
- the descriptor-table / heap-direct-indexing mode the shader was compiled for.

The runtime can then reconcile the root signature with the shader's reads and fail loudly on a mismatch instead of
rendering silently wrong.

`tools/shader_meta.py` (invoked by `tools/rdna2spirv.py` and verified by `tests/test_shader_meta.py`) now emits these
fields into `spirv/index.json`: `user_sgpr_reads` (`const32`, `ptr64_lo`/`ptr64_hi`, `ptr32_lo`, `vsharp`, `tsharp`,
`ssharp`), `descriptor_strides`, `uses_code_base`, `literal_samplers`, `input_signature` (ISG1 semantic name/index,
register, masks, system value, component type), `fssm_sha256`, and `descriptor_access` (DXBC `SFI0` direct-indexing
bits 25/26 plus table-load and high-dword-truncation flags). A missing or malformed ISG1 signature leaves the vertex shader
without recorded inputs. In that case the runtime accepts any input layout when filtering shader candidates.

### 3.6 Shader identity, PSOs and feature selection

A D3D12 PSO carries DXBC containers, not SPIR-V. The layer identifies a shader by hashing the **FSSM payload** (the
native RDNA2 program inside the container) and looking it up in a build-time table keyed by `(stage, hash)`. The
build emits that hash for every translation; the runtime extracts the FSSM chunk from the DXBC container the same
way `tools/fssm_unpack.py` does. The current `spirv/index.json` is keyed by carve name, which the runtime may not
have; add the content hash to the index.

Pipeline creation:

- `D3D12_GRAPHICS_PIPELINE_STATE_DESC` is supported today (`CreateGraphicsPipelineState`), as is the compute desc. The
  `D3D12_PIPELINE_STATE_STREAM_DESC` subobjects (`CreatePipelineState`) are not implemented yet. The plan: parse the
  stream subobjects in order; unknown types are a hard error with the type number logged.
- Build `VkPipelineShaderStageCreateInfo` from the translated module and the index entry. Mesh shaders use
  `VK_EXT_mesh_shader` (`AS`/`MS` subobjects → task/mesh stages); geometry uses the geometry stage; compute uses a
  compute pipeline.
- The index's `max_vertices`/`max_primitives` already size the SPIR-V mesh output arrays. Vulkan has no
  per-pipeline mesh output limit: check them against
  `VkPhysicalDeviceMeshShaderPropertiesEXT::maxMeshOutputVertices`/`maxMeshOutputPrimitives` and fail at pipeline
  creation if the device is smaller.
- Vertex input comes from the PSO input layout plus the shader's ISG1 signature. The translator's rule is
  **location = D3D input register**, not the semantic's HLSL slot. Match the layout element's semantic name/index
  to the ISG1 element to get the register, then create a `VkVertexInputAttributeDescription` with that location
  and the DXGI format. Resolve `D3D12_APPEND_ALIGNED_ELEMENT` against the preceding element and honour
  `InputSlotClass`/`InstanceDataStepRate`. The index entry needs the semantic map (section 3.5).
- Enable the features the index entry lists: barycentrics, fragment interlock, mesh, geometry, Float64, Int64,
  buffer device address, storage image without format, maintenance8.
- Request the subgroup size each shader was translated for (`requiredSubgroupSize` 64 or 32) where the device lists the stage
  in `requiredSubgroupSizeStages`, and say so in the log where it cannot (section 3.2, "Subgroup size").
- Pipeline caches: graphics and compute PSOs use one `VkPipelineCache` per device. On first use the runtime reads
  the `pipelines` build stage's `pipeline_cache.bin` beside the executable if its Vulkan header matches the device
  and pipeline-cache UUID. When the file is absent or incompatible, the runtime starts with an empty cache.
  The offline checker matches the runtime's layout and subgroup request for compute shaders with no descriptor
  bindings and the standard 144-byte push-constant range.
  `GetCachedBlob` records the vendor, device, driver version, `pipelineCacheUUID` and a SHA-256 of the description
  without `Flags`, sealed by a checksum (`runtime/src/cached_pso.h`). The answers are the host's, measured on an
  RTX 3060 and WARP: the same description is accepted; another
  shader, root signature content or state, a changed byte, a truncated blob or a pointer with size 0 give
  `E_INVALIDARG`; a null pointer with a size is ignored; another adapter gives `D3D12_ERROR_ADAPTER_NOT_FOUND` and
  another driver `D3D12_ERROR_DRIVER_VERSION_MISMATCH`. `ID3D12PipelineLibrary` is an in-memory map plus the Vulkan
  cache. Its arguments answer as the host's (`d3d12_pipeline_library_host_test`, 74 values from 46 calls that need no
  PSO, each run on the host in a process of its own, since a call it faults on leaves its runtime broken and the next
  one failing fast; RTX 3060, driver 610.88):
  - `CreatePipelineLibrary`: a foreign, truncated or corrupt blob, the other device's blob, or a null pointer with a
    size is `E_INVALIDARG`, and `*ppPipelineLibrary` is left as it was. One of the layer's own blobs from another
    format version is `D3D12_ERROR_DRIVER_VERSION_MISMATCH`, the documented code for a cache from another driver.
    A pointer with size 0 is an empty library; a longer blob than the header is accepted; a null output is
    `S_FALSE`.
  - The library is a device child, not a pageable object: `QueryInterface(ID3D12Pageable)` is `E_NOINTERFACE`.
  - `LoadComputePipeline`/`LoadGraphicsPipeline`/`LoadPipeline` with a null output are `E_INVALIDARG`, as is an
    unknown or empty name. `Serialize` with a buffer smaller than `GetSerializedSize()` is `E_INVALIDARG`. The
    serialized size and bytes are each device's own (the host's empty library is 900 bytes, the layer's 16).
  - Not compared: the calls the host runtime faults on (`Serialize(nullptr, n)`, `StorePipeline` with a null PSO,
    `Load*` with a null name or description); the layer answers `E_INVALIDARG` for each.

**Open:** whether the game uses `D3D12_PIPELINE_STATE_STREAM_DESC` for mesh/RT and which subobject types appear.
The trace will enumerate them.

**Implemented (D3D-29): pipeline state streams, mesh PSOs and `DispatchMesh`.**
- **Device:** `VK_EXT_mesh_shader` with `meshShader` is enabled where the GPU has it (`VulkanDeviceInfo::mesh_shader`).
  `taskShader` is not enabled, since the corpus has no amplification shaders. `D3D12_OPTIONS7` reports
  `MeshShaderTier` 1 only then; without the extension it reports `NOT_SUPPORTED` and mesh PSOs are `E_NOTIMPL`.
- **Streams:** `CreatePipelineState` parses the stream in the d3dx12.h layout: a subobject type, then its
  description at that description's alignment, each subobject pointer-aligned.
  - Every type up to `RASTERIZER2` is read.
  - Missing subobjects take the runtime's defaults: depth test on with `LESS`, back-face culling, no blending, sample
    mask `0xFFFFFFFF`, one sample.
  - The stream becomes the same descriptions `CreateGraphicsPipelineState` and `CreateComputePipelineState` take, so
    a VS stream builds the same PSO as the classic call.
  - `E_INVALIDARG`, as on the PC runtime: a type given twice, an unknown type, a truncated stream, MS with VS,
    tessellation or GS, CS with a graphics stage, or no stage at all.
  - `E_NOTIMPL` with a log, for what is not built yet: an AS, a depth-bounds test, separate front and back stencil
    masks, view instancing.
  - `RASTERIZER1`/`2`'s float depth bias is truncated to the integer the PSO path takes, with a log.
- **Mesh PSO:** the MS (`ms` index entry by FSSM hash) and PS on the mesh stage. It has no vertex input or input
  assembly; the input layout, topology type and strip cut are ignored, as D3D12 ignores them for mesh PSOs. Topology
  and strides are not dynamic state there (VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-07065/07066). Creation
  is refused when the index's `max_vertices`/`max_primitives` or the workgroup size exceed the device's
  `VkPhysicalDeviceMeshShaderPropertiesEXT`.
- **`DispatchMesh`:** recorded as a `Draw` with `mesh` set; the thread-group counts go in `count`/`instances`/`first`.
  It takes the same state snapshot as `DrawInstanced`: render targets, root arguments, viewport, scissor, blend
  factor, stencil references. Replay issues `vkCmdDrawMeshTasksEXT`.
  - **Recording errors** (`Close` returns `E_INVALIDARG`): a grid above 65535 per count or 2^22 groups.
  - **Not recorded:** an empty grid.
  - **Execution error** (`ReplayResult::invalid_draw`): a mesh draw with a VS pipeline, or the reverse.
- **Tests:** `d3d12_mesh_test` checks the refusals and three draws pixel by pixel on an RX 9060 XT (RADV, under
  Wine), validation-clean.
  - **The three draws:** a 2×2×1 and a 1×1×2 mesh grid, plus a VS PSO from a stream.
  - **What it doesn't cover:** whether the console flattens a 3D `DispatchMesh` grid the same way. That is an open
    question in `docs/design.md`, "Mesh shaders"; the test shader reads the workgroup ID directly.

## 4. Object model mapping

| D3D12 / Xbox object | Vulkan / layer object | Notes |
|---|---|---|
| `ID3D12Device` … `ID3D12Device11` | one `Device` object with the public chain plus measured Xbox overrides | The shipped BVH builder names Device11; slots 1/15/18/19/27/38 matched desktop in prior art, slot 62 is called as `CreateComputePipelineState`, slot 76 is Xbox texture creation, and heap slots 9/10 return in RAX. Unknown Xbox IIDs/slots are logged and refused. |
| `ID3D12CommandQueue` | one leased `VkQueue` per live queue: direct queues on the graphics family, compute queues on an async-compute family and copy queues on a transfer-only (DMA) family when the GPU has them (`src/queue_families.h`), else graphics-family leases | Distinct Vulkan queues allow cross-queue waits and signals to make progress. With more than one family in use, buffers, images, the arena and staging are created `VK_SHARING_MODE_CONCURRENT` over those families, so no ownership transfers are needed. Barriers on copy and compute queues are narrowed to the stages their family has; render-target and depth clears are refused there (as in D3D12), and the D24/RGB96 copy conversions and bindless flushes stay off a transfer-only queue. Tested by `vk_queue_test`. `PresentX`, `SuspendX`, `ResumeX` are Xbox queue methods, not in the public vtable; public GDKX samples present only through them (§4.5). Call shapes known, slots **Open**. |
| `ID3D12CommandAllocator` | IR arena + `VkCommandPool` | Reset only when the previous submission's fence is complete. |
| `ID3D12GraphicsCommandList` … `8` | `CommandList` with IR | Replay at `ExecuteCommandLists`; bundles inlined. Supports render passes, mesh dispatch, enhanced barriers; records the GCL8 front/back stencil references (applied only once a pipeline path consumes them). `Draw` views are inlined and moved into the command stream (D3D-289), eliminating per-draw heap allocations. |
| `ID3D12RootSignature` | `RootSignature` + `VkPipelineLayout` | Deserialize the blob with the host `D3D12CreateVersionedRootSignatureDeserializer` when possible; clear the measured `0x03000000` flag byte on a copy first; fall back to a local parser. Build the user-SGPR map and static samplers. |
| `ID3D12PipelineState` | `VkPipeline` + a `PipelineInfo` | Shader hash → SPIR-V + index entry; PSO state → Vulkan state; dynamic state where possible. The BVH builder's compute PSOs arrive at device slot 62 with a plain `D3D12_COMPUTE_PIPELINE_STATE_DESC`. |
| `ID3D12DescriptorHeap` | `DescriptorHeap` (shadow + optional arena range) | CBV/SRV/UAV and sampler heaps can be shader-visible; RTV/DSV are host-only. Start handles use the measured RAX return at heap slots 9/10. Increments are layer-chosen (32/32) pending the console value. |
| `ID3D12Resource` | `Resource` (a range in the arena buffer; a separate `VkImage` for textures) | Committed, placed and reserved. Reserved buffers use sparse memory where supported (section 4.2). Xbox textures may arrive through device slot 76. `GetGPUVirtualAddress` returns a console VA (a buffer offset or an image key). |
| `ID3D12Heap` | `Heap` (a reserved VA range) | Buffer ranges are backed by the arena; images get their own `VkDeviceMemory`. Residency calls are no-ops. True buffer/image aliasing inside a heap is **Open**. |
| `ID3D12Fence` / `ID3D12Fence1` | `Fence` over a timeline `VkSemaphore` | `GetCompletedValue` = counter value; `SetEventOnCompletion` = wait thread + Win32 event; CPU `Signal` = host signal. |
| Resource barriers (legacy) | `vkCmdPipelineBarrier` | Per-subresource state tracking; `BEGIN_ONLY`/`END_ONLY` become conservative full barriers at the begin point. |
| Enhanced barriers (`ID3D12GraphicsCommandList7::Barrier`) | `vkCmdPipelineBarrier2` | Direct mapping of sync/access/layout; prefer it when the game uses it. |
| `IDXGISwapChain` / `IDXGIFactory` | Factory/adapter/output: fake objects. `Swapchain` over `VkSwapchainKHR` + Win32 surface: only for the PC-style path | Shim DXGI; do not bind host DXGI. A GDKX title likely creates no swapchain (§4.5); it needs `IDXGIDevice1`→`GetAdapter`→`EnumOutputs` and a `PresentX` + frame-event presenter instead. |
| `ID3D12QueryHeap` | `VkQueryPool` | Occlusion, binary occlusion, pipeline statistics, timestamps. Binary results clamped to 0/1. |
| `ID3D12CommandSignature` | `IndirectLayout` | Simple draw/dispatch signatures map to indirect calls; root-constant/vertex/index/descriptor changes need expansion. |
| `ID3D12PipelineLibrary` | map + `VkPipelineCache` | D3D12 cached blobs are ignored. |
| `ID3D12StateObject` (ray tracing) | likely unused | RT work includes compute traversal (`image_bvh*`) plus `XboxBvhBuilder`; a DXR state-object path is stubbed until the trace shows it. |
| `ID3D12ShaderCacheSession` | `VkPipelineCache` | Stub with logging. |
| DRED, protected sessions, meta commands, shared handles | stubs | Not needed for single-player boot; return a clean failure and log. |

### 4.1 Device and feature queries

`CheckFeatureSupport` is a boot-critical diagnostic. Answer from the Vulkan device, not from a wish list:

- `D3D12_FEATURE_ARCHITECTURE`: report the host GPU's UMA property. The console is UMA; a discrete PC GPU is not.
  Reporting the truth may change the game's upload strategy; reporting a lie can make it map memory it cannot map.
  Start truthful and log the game's reaction.
- `D3D12_FEATURE_LEVELS`: report the highest feature level the Vulkan device can satisfy (12_2 when mesh/RT
  features are present).
- `D3D12_FEATURE_SHADER_MODEL`: answer the shader model the layer can honour, not the driver's maximum. Nothing in
  the runtime compiles DXIL: the game's shaders arrive as SPIR-V from the offline translator, and the BVH
  builder's container also carries a console FSSM program that the same translator handles. A request above 6.0 is
  clamped down to 6.0, a request for 5.1 is answered with 5.1 (D3D12's own baseline, which a caller asking for the
  minimum expects), and a value below 5.1 is refused with `E_INVALIDARG`. Measured against the host PC runtime
  (616.86) on an RTX 5070: 5.1-6.8 echo the request, 6.9 clamps to 6.8, and `0` is `E_INVALIDARG`; `0x5f` crashed
  the host runtime, which the test records in a comment rather than repeating. WARP answers a 5.1 request the same
  way.
- `D3D12_FEATURE_ROOT_SIGNATURE`: 1.1, the version the proxy can always produce. A 1.2 blob is rewritten as 1.1 when
  the host's serializer stops at 1.1 and nothing is lost (`d3d12_xs_rootsig.cpp`), and Windows 10's `d3d12.dll` stops
  there, so 1.1 is what the game can rely on everywhere. A 1.2 request is clamped down to 1.1, and an undefined value
  (0, or anything above 1.2) is `E_INVALIDARG` with the value left as it was. Measured against the host PC runtime
  (616.86): 1.0-1.2 echo the request, and a wrong `DataSize` is `E_INVALIDARG`; the host crashed on a null buffer, so
  the test never gives it one. The refusal above 1.2 was measured on an RTX 4070 Laptop (621.24) and WARP.
- `D3D12_FEATURE_D3D12_OPTIONS1`: answer from the Vulkan device's subgroups. `WaveOps` is true only when every
  operation group the wave intrinsics map to is present (`BASIC`, `VOTE`, `BALLOT`, `SHUFFLE`, `ARITHMETIC`),
  because D3D12 promises them all behind that one bool. `WaveLaneCountMin`/`Max` are the subgroup size range
  (`VkPhysicalDeviceSubgroupSizeControlProperties`, or the single `subgroupSize` without those 1.3 properties).
  The exception is an Intel integrated GPU of the pre-Xe-LPG lines (D3D-121): Vulkan reports 8..32 there, Intel's
  D3D12 driver a fixed 16..16, and the layer answers 16..16 like the host. Measured on Xe-LP: UHD 770
  (32.0.101.6129), Raptor Lake-S Mobile (101.7092) and Tiger Lake 0x9a78 (101.7088). From Xe-LPG on the driver
  answers the same range Vulkan reports (D3D-280, measured Meteor Lake 0x7d55 on 32.0.101.6127: 8..32 on both
  sides), so those lines keep the Vulkan range (Meteor/Arrow Lake device IDs 0x7d40-0x7dc5, Lunar Lake
  0x64a0-0x64b6). Discrete Arc keeps the Vulkan range until it is measured (TEST-57).
  `TotalLaneCount` has no Vulkan equivalent: D3D12 counts every lane of the device, and the layer answers one full
  wave, so a game that sizes work from it plans for less parallelism than the GPU has, never more. Measured
  against the host PC runtime (616.86) on an RTX 5070: `S_OK` with `WaveOps` 1, lanes 32..32 and a total of 6144,
  and `E_INVALIDARG` for a `DataSize` one field too short or too long. The host crashed the process on a null
  buffer, so the layer refuses one without asking the host (as with root signatures).
- `D3D12_OPTIONS` families: report the tier the Vulkan feature set can honour (resource binding tier 3 when
  descriptor indexing and update-after-bind are available), plus wave ops, 16-bit, barycentrics, render passes,
  VRS, mesh shader tier, and RT tier 1.1 only if the layer can honour them. The game may branch on these; a false
  "supported" is worse than a false "unsupported" at boot. For `OPTIONS2`–`5`, `7`, `9`, `11`, `13`–`16`, and
  `18`–`21` (D3D-86; leaving `OPTIONS1` to #745, `OPTIONS6`/`10` to #469, `OPTIONS8` to #715, and `OPTIONS12` to
  #693):
  - `OPTIONS4` reports `MSAA64KBAlignedTextureSupported = TRUE` (implemented in `GetResourceAllocationInfo`);
  - `OPTIONS19` maps `SupportedSampleCountsWithNoOutputs` from `framebufferNoAttachmentsSampleCounts`, ties
    `AnisoFilterWithPointMipSupported` to `VulkanDeviceInfo::sampler_anisotropy`, and reports the baseline heap
    limits (`2048` samplers, `1000000` views);
  - `OPTIONS21` reports baseline `D3D12_EXECUTE_INDIRECT_TIER_1_0`, and unimplemented features/tiers (`DepthBounds`,
    `CastingFullyTypedFormat`, MSAA UAVs, independent stencil masks, render passes, RT, mesh shaders, sampler
    feedback, wave MMA, work graphs, GPU upload heaps) stay `FALSE`/`NOT_SUPPORTED`/`TIER_0`. Checked live against
    the host D3D12 runtime on the same adapter (`NVIDIA GeForce RTX 5080`, driver `616.56`, Windows 11): 15/15 feature
    queries compared with `0` mismatches (`MSAA64KBAlignedTextureSupported = TRUE`,
    `SupportedSampleCountsWithNoOutputs = 0x1d`, `LEONIDA_VK_VALIDATION=1` with `0` validation errors).
- `D3D12_FEATURE_FORMAT_SUPPORT`: map to `vkGetPhysicalDeviceFormatProperties`; multisample render-target
  and resolve bits also require the format's image-usage/sample-count query to succeed. D3D12's per-format
  rules then apply, as native D3D12 reports them (RX 9070 XT, AMD 26.11.2, and WARP):
  - `MIP` wherever the image can have more than one level.
  - `SHADER_GATHER` with `SHADER_SAMPLE`, including `A8_UNORM`. Its swizzled R8 view gathered the same four
    alpha values as native D3D12 and WARP on 49 footprints. `d3d12_descriptor` also verifies a view that forces
    alpha to zero.
  - `UAV_TYPED_LOAD` with every typed UAV. Typed UAV texture reads use the sampled binding (`image_read_typed`),
    not a storage-image read. Typed buffer UAVs: the layer now writes their V# (section 3.4, "Buffer UAVs"); the
    gap left is `rdna2spirv.py` `FORMATS`, which decodes only some buffer formats (the 32-bit SINT ones, but no
    8/16-bit SINT/UINT other than the RGBA8 and RGBA16 UINT ones, and no 10_11_11).
  - The UAV atomic bits other than exchange for `R32_UINT`, whose UAV views get the `r32ui` storage view. Today
    that works on 2D UAVs only: the translator's image atomics and the `r32ui` slots cover 2D and 3D, and the
    layer creates no 3D UAV views yet. There is no `image_atomic_swap`, and `R32_SINT` gets no `r32ui` view.
  - `MULTISAMPLE_RENDERTARGET` for depth formats, from the same usages `MULTISAMPLE_QUALITY_LEVELS` queries.
  - No integer resolve, no 1D block-compressed. Depth formats are 2D and cube only: no 3D, shader reads, UAVs,
    buffer or vertex use, and no 1D, which native D3D12 allows but the layer's resource creation refuses.
  - TYPELESS formats report texture dimensions and `MIP` only.

  `d3d12_vulkan_device_test` checks these bits for every format that both the layer and native D3D12 on the same
  adapter (matched by LUID or PCI ID) support. Some checks compare with the host and some check the layer's own
  consistency; the comment on `compare_format_support` lists which are which. Formats the layer answers with no
  uses, such as D24 without Vulkan D24, are listed but not checked. It skips when there is no host on the same
  adapter (Wine, or no LUID/PCI match).
- `D3D12_FEATURE_FORMAT_INFO`: report the DXGI plane count independently of Vulkan texture support. Both
  depth/stencil families (including their view formats), NV12, P010, P016 and NV11 have two planes. The
  device test compares counts and HRESULTs with native D3D12 on the same adapter. If an older host rejects the
  sampler-feedback formats or `A4B4G4R4_UNORM` with `E_INVALIDARG`, the test still requires the layer to return
  `S_OK` and one plane for them.
- `D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS`: query `vkGetPhysicalDeviceImageFormatProperties` for the matching
  colour or depth/stencil attachment usage, plus the transfer and sampled usages used by the resource implementation.
  Vulkan exposes one standard sample pattern, so a supported sample count reports one quality level (quality index 0);
  only D3D12's legal sample counts (powers of two up to 32) are exposed.
- `D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT` (`D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT`): report the
  arena's limits, not the console's.
- `GetResourceAllocationInfo` and `GetCopyableFootprints`: implement D3D12's rules locally; the game uses them to
  suballocate and to lay out uploads. They are deterministic and public. Validate against a host D3D12 device in
  tests if one is available; do not require one at runtime. `GetResourceAllocationInfo` (and `1`/`2`): a buffer
  takes its width rounded up to 64 KiB, aligned to 64 KiB. A texture's size is the memory its `VkImage` needs
  (`vkGetDeviceImageMemoryRequirements`), rounded up to its D3D12 alignment. That alignment is 64 KiB, or 4 MiB with
  MSAA. The small alignments (4 KiB, or 64 KiB with MSAA) apply only when the application asks for them. 4 KiB
  also needs the texture to fit in 64 KiB and not be a render target or depth buffer; if it doesn't qualify, the
  description is invalid (`UINT64_MAX`), as the PC runtime answers. The MSAA 64 KiB alignment has no size limit
  on the PC runtime, so none here either. Texture sizes are the driver's on both runtimes, so only buffer sizes,
  alignments and invalid descriptions are compared with the host. One known difference: a mip count beyond the
  full chain is invalid here (Vulkan can't create it), while the PC runtime still answers with a size.
  A total over several resources that passes 2^64 saturates at `0xFFFFFFFFFFFF0000` here. Hosts disagree on it:
  WARP and the RTX 5070 host of #1204 saturate too, while on Windows 26300.9974 (D3D12Core 10.0.26100.9818) the
  Intel Raptor Lake-S (101.7092) and RTX 5060 Laptop (621.24) hosts wrap, two 2^63 buffers to 0, and WARP on that
  PC still saturates. A wrapped total could make a caller allocate less than one of its resources, so the layer
  keeps saturating, and the test accepts a host that wraps.
- `GetAdapterLuid` (implemented): `VkPhysicalDeviceIDProperties::deviceLUID`, which on Windows is the DXGI
  adapter's LUID; `d3d12_fence_timing_test` finds it with `IDXGIFactory4::EnumAdapterByLuid` (RTX 4090, driver
  621.14). A driver that reports `deviceLUIDValid = VK_FALSE` gets a zero LUID, logged.
- `GetCustomHeapProperties` (implemented): the D3D12 documentation's table for the architecture
  `D3D12_FEATURE_ARCHITECTURE` reports. On a discrete GPU (`UMA = FALSE`): `DEFAULT` is `NOT_AVAILABLE`/`L1`,
  `UPLOAD` `WRITE_COMBINE`/`L0`, `READBACK` `WRITE_BACK`/`L0`, `GPU_UPLOAD` `WRITE_COMBINE`/`L1`. On an integrated
  GPU (`UMA = TRUE`, `CacheCoherentUMA = FALSE`) the same pages, all in `L0`, and a custom heap in `L1` is refused
  with `E_INVALIDARG` (`CreateHeap`, `CreateCommittedResource`), as the host runtime does on
  a Radeon 780M (AMD 32.0.13030.13001) and on WARP. All answers have type `CUSTOM` and node masks 1. `CUSTOM`,
  unknown types and node masks above 1 get an all-zero answer, logged. The test compares `DEFAULT`, `UPLOAD` and
  `READBACK` byte for byte with the host runtime's answer when the host adapter has the layer's architecture (RTX
  4090: identical); `d3d12_heap_properties` compares the `L1` heaps with the layer's own adapter.
- Residency and power (D3D-33) are no-ops that still check their arguments (`d3d12_vulkan_device.cpp`, tested in
  `d3d12_vulkan_device_test`). Vulkan has no residency API: memory from `vkAllocateMemory` stays usable until
  `vkFreeMemory`, and paging under pressure is the kernel driver's job, so every pageable object is always resident.
  The residency answers are the host runtime's: `d3d12_residency_test` runs the same 58 steps on both devices
  with a committed buffer, a heap, a placed buffer, a descriptor heap, a query heap and a fence, and compares every
  HRESULT, the device-removed reason after each step, and the value `EnqueueMakeResident` leaves the fence at
  (110 values; RTX 3060, driver 610.88). Steps that leave the host runtime unusable run there in a child process with a
  fresh device.
  - `MakeResident` and `Evict` return `S_OK` for any count of this device's pageable objects, the same one twice,
    an evicted one or a resident one. D3D12 documents `Evict` as a hint the runtime may defer, so dropping it is
    allowed.
  - `SetResidencyPriority` (Device1): only objects that own their memory take a priority, committed resources and
    heaps; any `UINT` is a valid priority for them. A placed resource (a reserved one too, by the same rule; not measured), a descriptor heap, a query heap
    or a fence is `E_INVALIDARG`, also inside a longer list. The host runtime then removes its device as well
    (`DXGI_ERROR_INVALID_CALL`); the layer doesn't, since a console title that sets priorities on its placed
    resources would lose its device over a hint.
  - `EnqueueMakeResident` (Device3) signals its fence from the CPU with the given value before returning, since
    the work is already complete. A queue `Wait` on that fence must not hang, which is why the signal stays. A null
    fence is accepted and nothing is signalled. `E_INVALIDARG` for flags other than `NONE` and `DENY_OVERBUDGET`;
    a refused call does not signal. A value below the fence's current one moves the host's fence back; the layer's
    stays where it was (see the fence note above).
  - Not compared: a nonzero count with a null array, and a null entry in an array, which the host runtime
    dereferences (an access violation). The layer answers `E_INVALIDARG` for the null arrays, skips a null entry in
    `MakeResident`, `Evict` and `EnqueueMakeResident`, and refuses one in `SetResidencyPriority`.
  - `SetStablePowerState` returns `S_OK`: Vulkan has no clock control, and the result of a frame does not depend
    on the clocks.
  - `SetBackgroundProcessingMode` (Device6): no Vulkan API steers driver background work, so nothing runs. The
    completion event is set at once and `*pbFurtherMeasurementsDesired` is `FALSE`. A mode or action outside the
    SDK's enum is `E_INVALIDARG`.

### 4.2 Resources, heaps, placed and reserved

- **Xbox texture creation is a separate device method at slot 76**, not `CreateCommittedResource`. Prior art
  reports its shape as heap properties / heap flags / description / initial state / clear value / iid / output,
  forwarded on the PC side to `CreateCommittedResource` slot 27. The description is 56 bytes plus extension words,
  and its layout field is `0x101` (tiled) or `0x1` (row-major buffer). The PC proxy replaces the layout with a
  driver-chosen one; our layer must recognise slot 76 and `0x101` explicitly and must not pass `0x101` to the public
  layout enum. The FidelityFX upscaler creates 10 default-heap resources through this slot, so it is not optional
  for middleware. The console tiled data layout is GFX10 `SW_64KB_D` (Mesa addrlib `GFX10_SW_64K_D_RBPLUS_PATINFO`),
  and the detile and tile path is implemented in `resource_detile.h` and verified by `resource_detile_test` (D3D-316).
- `CreateCommittedResource`: allocate an aligned range in the arena for a buffer, or a separate `VkDeviceMemory`
  for an image; record the initial state. Map `D3D12_HEAP_TYPE_DEFAULT` to device-local memory, `UPLOAD`/`READBACK`
  to separate host-visible buffers. This remains the path for buffers and for any PC-style committed resource.
- `ID3D12Resource::GetDesc` returns the description with its defaults filled in, as the PC runtime does: `MipLevels`
  0 becomes the full chain, and `Alignment` 0 becomes the placement alignment, 64 KiB, or 4 MiB for a multisampled
  texture (an alignment the description asked for is kept). **Implemented** and compared field by field with the
  PC runtime and WARP by `d3d12_resource_desc_test`: 25 committed and reserved buffers and 2D, array, volume and
  multisampled textures (measured on an RTX 3060 host and WARP, Windows 10).
- **Invalid arguments (D3D-49).** `CreateCommittedResource`, `CreateDescriptorHeap`, `CreateCommandQueue`,
  `CreateFence` and `Map` return the host PC runtime's HRESULTs (no debug layer) for the cases in
  `compare_invalid_arguments` (`d3d12_vulkan_device_test`): invalid ones, their valid neighbours, and
  `ALLOW_RENDER_TARGET`/`ALLOW_DEPTH_STENCIL` on every format both devices create. Rules the host keeps that are easy
  to get wrong: a committed resource takes no heap flag except `SHARED`, `ALLOW_DISPLAY` (textures only),
  `HARDWARE_PROTECTED`, `CREATE_NOT_RESIDENT`, `CREATE_NOT_ZEROED` and `ALLOW_SHADER_ATOMICS` (only with
  `ALLOW_UNORDERED_ACCESS`; measured on an RTX 4070 Laptop and WARP); a buffer refuses `ALLOW_SIMULTANEOUS_ACCESS`;
  MSAA takes quality 0 or one of the two standard patterns; a buffer may be `ALLOW_RENDER_TARGET` but not
  `ALLOW_DEPTH_STENCIL`, and is aligned 0 or 64 KiB only; upload heaps start in any part of `GENERIC_READ` (or
  `COMMON`), readback heaps in `COPY_DEST` or `COMMON`, neither with UAVs; a texture starts in exactly
  `GENERIC_READ`, in one write state alone (`RENDER_TARGET`, `UNORDERED_ACCESS` and `DEPTH_WRITE` only with the
  matching resource flag, `COPY_DEST`, `RESOLVE_DEST`) or in a mix of read states (`DEPTH_READ` only with
  `ALLOW_DEPTH_STENCIL`, the shader-resource states not with `DENY_SHADER_RESOURCE`), never with a buffer-only or
  undefined state bit, while a buffer takes any bits except `RAYTRACING_ACCELERATION_STRUCTURE` without UAV access
  (RTX 3060 / 610.88; the same rule applies to placed and reserved resources), or at all where the host reports no
  ray tracing (an integrated Radeon without DXR, @aguedou-34 on #2462): the layer refuses it on a device with
  neither KHR acceleration structures nor the XDXR builds; `ALLOW_DEPTH_STENCIL` is accepted for every
  format of the R32, R16, R24G8 and R32G8X24 families and nothing else; a clear value needs a render target or depth
  buffer and a format of the resource's family; descriptor heaps may have zero descriptors; non-monitored fences are
  refused; a null output is `S_FALSE` (a capability check) except for `CreateCommandQueue`, which answers
  `E_UNEXPECTED`, and `CreateDescriptorHeap` answers it before its flag, type and node rules (`S_FALSE` even for a
  shader-visible RTV heap) where committed resources and fences check their arguments first; a `GPU_UPLOAD` heap is
  refused, since `CheckFeatureSupport` reports `OPTIONS16.GPUUploadHeapSupported` FALSE (a host with GPU upload
  heaps, which needs Resizable BAR, accepts it, so that case is compared only on a host without them); a
  global-realtime queue is `DXGI_ERROR_ACCESS_DENIED` without the privilege; `Map` refuses a
  non-empty read range past the buffer. Not compared, because the host faults or the layer cannot follow it: a null
  heap-properties or descriptor-heap description (the host faults), more mips than the full chain (the RX 9070 XT
  runtime keeps up to 16 on a 16×16 texture and dies on 17), `D32_FLOAT` and `D16_UNORM` as render targets and the R32
  and R16 colour formats as depth buffers (accepted by the host; the layer creates the Vulkan format of the same name,
  and the Vulkan drivers measured can't make that attachment, so the test prints these instead of failing),
  cross-adapter resources (the host takes a row-major texture on a `SHARED | SHARED_CROSS_ADAPTER` heap; the layer has
  no cross-adapter sharing), an unknown descriptor heap type (accepted by the host), and the driver's own answers:
  R32G32B32 render targets, 2D textures wider than 16384, `L1` custom heaps, 16x MSAA and video formats and queues
  differ between the RX 9070 XT, an integrated Radeon and WARP. The rest of the table agrees on all three.
- `CreateHeap`: reserve a VA range with the requested size/alignment. `CreatePlacedResource`: assign the resource
  the range at `heap_va + offset`; a buffer uses that range in the arena, an image uses the offset as a key and
  binds its own memory. Vulkan image alignment can exceed D3D12's 64 KiB; because images do not share the arena,
  this mostly affects true aliasing, which is **Open** until tested on the target GPU.
  Its arguments are answered as the host runtime answers them (`compare_placed_reserved` in
  `d3d12_vulkan_device_test`, RTX 3060 / 610.88): the description, initial-state and clear-value rules of a committed
  resource (but any initial state for a buffer in an upload or readback heap, and no UAVs there); a buffer at a
  64 KiB-aligned offset; a texture at a multiple of its `GetResourceAllocationInfo` alignment (4 KiB for a small
  texture), in a heap aligned at least as much, and whole inside the heap; the heap's `DENY` flags against the
  resource; then a null output is `E_UNEXPECTED` and nothing is made.
- `CreateReservedResource` **implemented** (`d3d12_vulkan_device.cpp`, `src/resource_tiling.h`, checked by
  `d3d12_vulkan_device_test` and `resource_tiling_test`):
  - `CreateReservedResource`, `CreateReservedResource1` (no protected sessions) and `CreateReservedResource2`
    (castable formats ignored) create sparse buffers when the device can bind and alias 64 KiB tiles and guarantees
    zero reads from unbound pages. Reserved textures and buffers on other devices use a fully backed resource.
    D3D12's description rules are kept, as measured on the PC
    runtime: buffers are `ROW_MAJOR`, textures `64KB_UNDEFINED_SWIZZLE` (anything else, or a format without a tile
    shape, is `E_INVALIDARG`; `64KB_STANDARD_SWIZZLE` too, since `CheckFeatureSupport` reports
    `StandardSwizzle64KBSupported` as FALSE), and `GetHeapProperties` on a reserved resource is `E_INVALIDARG`.
    The description, initial-state and clear-value rules are a committed resource's on a default heap, and a null
    output is `E_UNEXPECTED` after them (`compare_placed_reserved`, RTX 3060 / 610.88).
    `GetHeapProperties` answers as the host runtime's (`d3d12_heap_properties_test`: 561 values for committed
    resources in every heap type, custom ones and SHARED / ALLOW_DISPLAY / CREATE_NOT_ZEROED / CREATE_NOT_RESIDENT
    included, placed resources in heaps of each kind, reserved ones, and null arguments; RTX 3060, driver 610.88).
    A committed resource's flags are the caller's without the two creation-only bits, plus the DENY bits of the
    resource classes it isn't; a placed resource reports its heap's flags as they are, so one in an `ALLOW_ALL`
    heap reads back 0.
    `CheckFeatureSupport` still reports `TiledResourcesTier` as not supported, so a game that checks it creates
    committed resources instead. Buffer mappings alone do not implement a full tiled-resources tier.
    The Xbox tiled layout word `0x101` is accepted as `64KB_UNDEFINED_SWIZZLE` (**ASSUMPTION**:
    that the console uses it for reserved resources too).
  - `GetResourceTiling`: the standard 64 KiB tile shapes, standard and packed mips (a mip stays standard while it is
    wider than half the tile or taller than the tile, depth not counting, as AMD's PC runtime does on RDNA2 and RDNA4
    and so, most likely, the console (D3D-65); an NVIDIA GPU packs from the first mip smaller than a tile on any axis.
    D3D11.3 functional spec 5.9.2.8.5 only requires that
    a mip filling a tile in every dimension stays standard, a smaller one "may or may not be packed, depending on the
    IHV"), each standard mip's tile counts, and the tile order (each array slice holds
    its standard mips, then its packed tiles). The test compares the tile shape and the tile counts of the mips both
    sides keep standard with the PC runtime, and checks that no mip filling a tile is packed, for 18
    buffer, 2D, array, block-compressed and volume layouts, and skips a layout the host refuses: which ones a PC
    driver accepts varies (an RTX 3090 on driver 616.56, at tier 3, refuses texture arrays whose mips reach the
    packed tail, measured by @CharlyRien in #295; an RTX 5070 on 621.01 accepts them). How many tiles the packed mips take is the driver's
    choice: here it is one tile per tile-depth slice the tail spans, as AMD reports (below), and never fewer than
    their tight linear size in whole tiles; the PC runtime on an RTX 5070 reported one tile more than the tight size
    for 1000x300 and 100x100 RGBA8 and a 128^3 R8 volume, and the same count for the other layouts probed. Two
    further measured differences: on array slices after the first, the PC runtime
    writes a tile index into packed subresources instead of the documented `D3D12_PACKED_TILE` (ours follows the
    documentation), and a resource that is not reserved has no tiles (all outputs zero), as on the PC runtime.
  - **Measured on AMD** (RX 9070 XT, RDNA4, driver 32.0.31041.1004, Windows 11, `main` `91b36cf`): the AMD PC
    runtime splits standard and packed mips differently, which `d3d12_vulkan_device_test`'s tiling comparison used
    to report as a failure; the spec leaves the split to the driver, so the comparison now takes any split (a Radeon
    880M on 26.11.2 behaves the same, and WARP keeps every mip standard: it packs nothing, down to 1x1).
    Tile shapes agree in all 17 texture layouts, and 11 of them match exactly; in the other 6, AMD keeps more
    mips as standard tiles before the packed tail:

    | Layout | Tile | Standard mips: NVIDIA (the layer before D3D-65) | AMD (the layer now) |
    |---|---|---|---|
    | 1000x300 RGBA8 sRGB, 3 slices | 128x128 | 2 | 4 |
    | 2048x1024 BC7, 2 slices | 256x256 | 3 | 4 |
    | 1000x600 BC3 | 256x256 | 2 | 3 |
    | 100x100 RGBA8 | 128x128 | 0 | 1 |
    | 4096x64 BGRA8 | 128x128 | 0 | 6 |
    | 256x256x64 RGBA8 volume | 32x32x16 | 3 | 4 |

    An RDNA2 GPU (the Ryzen 7 9800X3D's, driver 32.0.21045.5002) gives the same splits as the RDNA4 one
    ([#374](https://github.com/ItsNotRexed/Leonida/issues/374#issuecomment-7071869388)). With 30 more layouts that
    probe the edges (a mip exactly at or just past half a tile, thin and tall textures, non-square 2D and volume
    tiles), the rule that fits all 47 on both GPUs is: **a mip stays standard while it is wider than half the tile or
    taller than the tile; depth does not count**. An earlier "every axis at most half the tile's largest side" fitted
    the first 17 but missed four tall or deep edge cases. All 47 layouts and AMD's values are in
    `runtime/test/resource_tiling_test.cpp`. AMD's packed tile count is one tile per tile-depth slice the tail spans,
    `ceil(depth of the first packed mip / tile depth)`: 1 for every 2D layout, and for 32 further volumes whose tail
    is deeper than a tile it steps at each tile depth (8x8x32 R8 takes 1 tile, 8x8x33 takes 2, 8x8x256 takes 8), even
    where the bytes would fit in fewer (16x16x128 RGBA8: 8 tiles for 3 tiles of data). Those 32 are in the test too,
    79 layouts in all, identical on both GPUs. The console is RDNA2, so the layer follows this rule (D3D-65;
    **ASSUMPTION**: that the console's runtime splits like the PC's AMD driver).
  - `UpdateTileMappings` and `CopyTileMappings` bind pages in reserved buffers to default heaps from D3D-16.
    Sequential, NULL, SKIP and single-tile reuse ranges are supported. Queue timeline waits order binding between
    ordinary submissions, and the previous heap allocation is retained until an in-flight unbind completes.
    `CopyTiles` copies 64 KiB tiles between sparse buffers and linear buffers in either explicit direction.
    `d3d12_buffer_tiles_test` compares three full readbacks with native D3D12, including a fence-gated queue.
  - Drivers that exclude the host-visible arena memory type from sparse buffers (including NVIDIA 621.01 on the
    RTX 3090) use a dedicated device-local allocation per heap for its tiles. A heap with this dedicated backing
    cannot mix placed buffers and sparse mappings.
  - D3D-285 extends these mappings to single-sample `R8G8B8A8_UNORM` reserved 2D textures with one mip, one
    layer and no resource flags. The Vulkan image must support sparse aliases and zero reads from unmapped tiles,
    with 128 × 128 color blocks aligned to 64 KiB. Images requiring an opaque mip tail or separately bound metadata
    keep the fully allocated fallback. The image is transitioned while still unbound and kept in `GENERAL` so
    mapping an alias preserves contents. Image tiles use dedicated device-local heap memory and can alias tiles
    of compatible sparse images. Buffer and placed-resource aliases with texture tiles are outside this path.
  - `CopyTiles` transfers RGBA8 tiles in either direction. Partial edge blocks use the full 512-byte linear row
    pitch and transfer only in-bounds texels. Byte-unaligned offsets use transfer scratch.
    `d3d12_texture_tiles_test` compares initially unmapped tiles, column mappings, aliases, and a fence-delayed
    remap with native D3D12, including clipped edges and byte-unaligned offsets.
- `D3D12_RESOURCE_DESC::Layout`: the public `ROW_MAJOR` and `64KB_STANDARD_SWIZZLE` values plus the Xbox `0x101`
  tiled value from slot 76. Row-major textures need `VK_IMAGE_TILING_LINEAR` (not usable for RT/DS) or a staging
  copy; `64KB_STANDARD_SWIZZLE` and `0x101` have no direct Vulkan equivalent. If the game uploads console-tiled data,
  the layer uses a detile step. `runtime/src/resource_detile.h` (`detile_surface_64k` / `tile_surface_64k`, `resource_detile_test`)
  and `runtime/src/gfx10_tiling.h` provide GFX10/RbPlus `SW_64KB_D` detiling. Which mode `0x101` uses, the mip tail and
  upload wiring remain **Open**.
- Formats: a `DXGI_FORMAT` → `VkFormat` table, including sRGB and typeless variants. Typeless resources get a
  concrete image format plus `VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT` and a `VK_KHR_image_format_list` when views cast
  within the family; unsupported casts are logged and fall back to the resource format. `CreateCommittedResource3`'s
  castable-format lists are mapped to a `VkImageFormatListCreateInfo`; buffers and unmapped formats are refused.
  `CreatePlacedResource2`'s list is mapped the same way on the placed path (D3D-90). Depth/stencil, BCn and packed
  formats map directly; any format the device rejects is a hard error with the DXGI value.
- `Map`/`Unmap`: upload/readback buffers map their host-visible memory; default-heap resources return
  `E_INVALIDARG` as D3D12 does. A texture in a CPU-visible custom heap maps only with a null pointer (its layout is
  the driver's), as on D3D12.
- `WriteToSubresource`/`ReadFromSubresource` (implemented in `d3d12_vulkan_device.cpp`) complete before they
  return, as D3D12 requires; a read cannot wait for a later submit. Textures in CPU-visible custom heaps are
  created with `VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT` when the device supports that usage with the requested
  image format and flags, moved to
  `GENERAL` and zero-filled on the host at creation, and copied with `vkCopyMemoryToImageEXT`/
  `vkCopyImageToMemoryEXT` while they stay in `GENERAL` (D3D12's `COMMON`). Every other case (no extension, a
  texture a barrier has moved, a default-heap texture, which D3D12 does not allow but the layer copies and logs) is
  a `CopyTextureRegion` from or to a host-visible staging buffer, replayed on the device's queue and waited for, so
  the layout tracker and a first use's zero fill behave exactly as for the application's own copies.
  Their arguments answer as the host runtime's do (`d3d12_subresource_copy_test` runs the same 117 steps on both
  devices and compares every HRESULT and the whole destination of every read; RTX 3060, driver 610.88): a
  subresource past the last, a box whose right, bottom or back edge lies past the mip's, or one that cuts a
  compressed block is `E_INVALIDARG`; an empty or inverted box that passes those checks is `S_OK` and copies
  nothing; the row and depth pitches are not checked, so rows or slices that overlap (a pitch smaller than a row,
  or 0) are copied in order. Not compared: null data, which the host faults on (the layer answers `E_INVALIDARG`);
  which bytes win when a read's rows overlap, which is the driver's copy order; and a default-heap texture, which
  the host refuses and the layer copies on purpose.
- 96-bit textures (`R32G32B32_FLOAT`/`_UINT`/`_SINT`/`_TYPELESS`, D3D-62) are `R32G32B32A32` images
  (`ResourceBinding::rgb96`), on every GPU. Most GPUs have no optimal-tiling RGB32 image (an RX 9070 XT reports
  no optimal features for `R32G32B32_SFLOAT`), and one that has would not avoid the repack: D3D12 places
  subresources at multiples of 512 bytes and Vulkan requires a copy's buffer offset to be a multiple of the
  12-byte texel (VUID-vkCmdCopyBufferToImage-dstImage-07975). What the application sees stays D3D12's:
  `GetCopyableFootprints` gives 12-byte texels (compared with WARP in `d3d12_rgb96_test`), and a view's T# names
  `GFX10_FORMAT_32_32_32_*`, whose swizzle reads alpha as one, so the image's fourth channel is never read.
  - `CopyTextureRegion` between a texture and a buffer is repacked by the D24 compute pass
    (`src/depth24_convert.comp` modes 2 and 3, `Depth24Job::rgb96`): the same scratch buffers, parts and lifetime,
    with texels of three words in the footprint and four in the scratch, copied as they are (an upload writes the
    fourth word as 0). The words are not interpreted, so NaN payloads, denormals and integers pass unchanged. On
    an RX 9070 XT (AMD 26.11.2) an upload and a readback take about 5 and 2 ms at 1024x1024 and about 20 and 45 ms
    at 4096x4096 (`d3d12_rgb96_test` with `LEONIDA_RGB96_LARGE` set; single runs that vary by a few ms, each the
    first submission of a new queue). The converter is created on every device for this, whether or not the
    GPU has a native D24.
  - Texture-to-texture copies and `CopyResource` are plain image copies, and only between two 96-bit textures.
  - `WriteToSubresource`/`ReadFromSubresource` always take the staged copy, as a D24 texture stored as D32 does:
    a host image copy would not repack.
  - Not done: render targets. WARP creates 96-bit render targets; this layer refuses `ALLOW_RENDER_TARGET` and
    leaves `RENDER_TARGET` out of `FORMAT_SUPPORT`, since a writable fourth channel changes destination-alpha
    blending. `ALLOW_UNORDERED_ACCESS` and `ALLOW_DEPTH_STENCIL` are refused, as on WARP (measured in the test).
    `GetResourceAllocationInfo` gives no size for a description creation refuses. A typed 96-bit texture has
    views of its own format only; the `TYPELESS` one has `FLOAT`, `UINT` and `SINT` views (sampled in
    `d3d12_descriptor_test`). `GetResourceTiling` reports no tiling for a committed 96-bit texture on both
    Microsoft WARP and an NVIDIA GTX 1660 SUPER: total tiles, the tile shape and every packed-mip field are zero,
    and the returned subresource-tiling count is zero (so its output element is untouched). This holds for
    `FLOAT`, `UINT`, `SINT` and `TYPELESS`; there is no standard tile shape to implement for a 12-byte texel.
- Resource flags map to Vulkan usage: `ALLOW_RENDER_TARGET` → `COLOR_ATTACHMENT`, `ALLOW_DEPTH_STENCIL` →
  `DEPTH_STENCIL_ATTACHMENT`, `ALLOW_UNORDERED_ACCESS` → `STORAGE`, `ALLOW_SIMULTANEOUS_ACCESS` → concurrent
  sharing, `DENY_*` → omit the corresponding usage. Buffers get the union of usages their states and flags imply;
  images copy dimension, format, mip levels, array layers and sample count.
- Copies: `CopyBufferRegion` → `vkCmdCopyBuffer` with arena offsets; `CopyTextureRegion` →
  `vkCmdCopyBufferToImage`/`vkCmdCopyImageToBuffer`/`vkCmdCopyImage`, using the `D3D12_PLACED_SUBRESOURCE_FOOTPRINT`
  row pitch and height; `CopyResource` → a whole-subresource copy; colour resolves → `vkCmdResolveImage`;
  depth and stencil resolves → a `vkCmdBeginRendering` scope with a resolve attachment (section 3.2).
  Partial resolve regions to another offset need a shader or a region copy.
  - A placed footprint must have an `Offset` that is a multiple of 512 and a `RowPitch` that is a multiple of 256
    (`D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT`, `D3D12_TEXTURE_DATA_PITCH_ALIGNMENT`). Any other is refused when it is
    recorded, so `Close` returns `E_INVALIDARG` (`CommandList::valid(const CopyLocation&)` in
    `runtime/src/command_list.h`; `command_list_test` has both cases). That is D3D12's rule for a device that
    reports `UnrestrictedBufferTextureCopyPitchSupported` as `FALSE` in `D3D12_FEATURE_D3D12_OPTIONS13`, which this
    device does on purpose (`d3d12_vulkan_device_test` fails on a `TRUE`). The PC runtime answers differently: on an
    RTX 4090 (driver 621.14) and on WARP it reports the capability as supported, and a list with `Offset` 16 and
    `RowPitch` 260 for an `R8G8B8A8_UNORM` footprint closes with `S_OK` (measured with a stand-alone D3D12 program,
    Windows 11 26200). The layer cannot simply say the same: Vulkan takes a copy's buffer offset as a multiple of
    the texel block size and its row length in texels (VUID-vkCmdCopyBufferToImage-dstImage-07975), while the
    capability lifts D3D12's two alignments and is not documented as keeping offsets and pitches texel multiples
    (not tested here), so a footprint that is not texel-aligned would have to be repacked through scratch memory
    the way a 96-bit one is (section 4.2). The replay already accepts any pitch that is a texel multiple and any
    offset that is one, except that a depth, stencil or 96-bit copy needs an offset that is a multiple of 4
    (`prepare_texture_copy` in `runtime/src/vulkan_replay.h`; anything else is `unsupported` there), so accepting
    those needs only the record-time check changed; reporting the capability needs the repack as well. What the
    console's runtime reports is not measured, and no title code that depends on it has been seen.
- `WriteBufferImmediate` (D3D-26) → `vkCmdUpdateBuffer` of 4 bytes. The address is resolved when the list is
  executed, through `ReplayFunctions::buffer_at`: the buffer whose `ResourceBinding::va` range holds it, at
  `buffer_offset + (address - va)`. `VulkanContext` finds it by binary search over the buffers' VA ranges
  (`runtime/src/buffer_va_index.h`, kept in step with the registered resources); aliasing placed buffers can
  overlap: the one whose range starts closest below the address and also holds the bytes the view or write needs
  wins, or, when none holds all of them, the one that starts closest below (`buffer_va_index_test`). These are the same fields and lookup as #291's arena, and the device fills
  `va` once `GetGPUVirtualAddress` lands (#157/#291); until then, a list with an immediate write is refused as
  `unresolved_resource`. `DEFAULT` adds no barrier: it is ordered like any copy, by the application's barriers.
  `MARKER_OUT` waits for every earlier command to finish and sees its writes (`ALL_COMMANDS`/`MEMORY_WRITE` →
  `TRANSFER`/`TRANSFER_WRITE`). `MARKER_IN` only needs earlier commands to have started. Vulkan cannot express
  that, because `TOP_OF_PIPE` in a first synchronisation scope is no stage at all, so it gets an execution-only
  `ALL_COMMANDS` → `TRANSFER` dependency: stricter than D3D12 requires, never weaker. Recording refuses a null
  or unaligned address and an unknown mode (the debug layer's `WRITEBUFFERIMMEDIATE_*` messages, IDs 1090-1092 in
  `d3d12sdklayers.h`).
- `AtomicCopyBufferUINT` / `AtomicCopyBufferUINT64` (D3D-26) → one `vkCmdCopyBuffer` region of 4 or 8 bytes, after
  a `ALL_COMMANDS`/`MEMORY_WRITE` → `TRANSFER`/`TRANSFER_READ|WRITE` barrier. That barrier covers every earlier
  write, those to the dependent resources among them, so the copy sees them all, as D3D12 requires. The dependent
  ranges are checked against their buffers (`end == UINT64_MAX` resolves to the buffer's size) but add nothing a
  global barrier does not already cover. Recording matches live Windows 11 WARP and hardware D3D12 `Close`
  validation: offsets aligned to and within the buffer for the copy size, zero dependencies allowed (with null
  arrays), buffer dependency ranges with `begin < end` and either `end == UINT64_MAX` or `end <= width`, and
  texture dependency ranges `{0, UINT64_MAX}` on a valid subresource or `ALL_SUBRESOURCES`. **ASSUMPTION:** Vulkan
  promises no atomicity for a transfer, and the layer relies on a naturally aligned 4- or 8-byte copy being a
  single write on the GPU. This matters only to a reader racing the copy on another queue or on the host.
- Clears: a full-subresource clear maps to `vkCmdClearColorImage`/`vkCmdClearDepthStencilImage` (UAV buffers to
  `vkCmdFillBuffer`); a rect clear uses `vkCmdClearAttachments` inside the rendering scope. UAV image clears use
  `vkCmdClearColorImage` in `GENERAL`, and rectangles `vkCmdClearAttachments`. Buffer patterns of other sizes
  need no compute clear: 8/16-bit elements are a 32-bit fill with copied ends, 64/128-bit ones an update doubled by
  copies (section 3.2, UAV clears).
- **Typed UINT RTV clears:** `ClearRenderTargetView` still takes `FLOAT[4]`. For the currently exposed
  `R8G8B8A8_UINT`, `R32_UINT` and `R32G32B32A32_UINT` resources, convert those values numerically into
  `VkClearColorValue::uint32` for both image and rectangle clears; copying float bits gives incorrect stored
  integers without a Vulkan validation error. **Measured:** Windows D3D12 on WARP and AMD RX 5700 XT clamps
  negatives/NaN to zero, positive overflow/infinity to the component maximum, and truncates fractions.
  `d3d12_clear_test` (renamed from `d3d12_uint_clear_test` in #525, D3D-45) compares full and rectangular D3D12
  readbacks with WARP over eleven integer input sets on every UINT and SINT render-target format it lists, beside
  the UNORM/SNORM/FLOAT and depth/stencil cases. ctest runs only its `--rgb9e5` mode (`d3d12_rgb9e5_rtv`), so the
  WARP comparison is run by hand until #521's 0.5 tie rounding lands (`runtime/CMakeLists.txt`).
  `clear_color_test` checks both replay paths without a GPU, including unchanged float values.
  This does not add typed-view reinterpretation of typeless resources or new integer formats.
- **RGBA8 typeless RTV clears:** explicit UNORM, SRGB, UINT, SINT and SNORM RTVs on a
  `R8G8B8A8_TYPELESS` 2D resource with sample count 1, selecting a mip and a contiguous array-layer range.
  `TargetView` captures its format by value; copies and layout tracking keep the resource's base format.
  Reinterpreted full clears use dynamic rendering because `vkCmdClearColorImage` cannot select a view format.
  Rectangle clears use the same typed attachment view. The view-cache key includes the format, and views
  request only color-attachment usage so SRGB does not inherit an unrelated storage requirement.
  Same-format image clears and legacy float-only replay clients retain their existing path.
  On an RX 5700 XT, 13 synthetic full/rectangle readbacks match Windows WARP (UINT exact, normalized
  channels within one byte), including descriptor rewrites, format switches and an SRGB RTV on a UAV resource.
  The same tests fail without this change and when the cache ignores format. CPU replay covers 31 scenarios;
  17 unsupported/invalid frontend views fail `Close` without submission. Typed-resource casts, MSAA and 3D
  typeless RTVs remain unsupported.
  Phase 2 adds 54 mip/array and 12 non-array higher-mip readbacks against WARP, including untouched sentinels.
  D3D-161 adds SINT and SNORM views when the Vulkan format supports color attachments. SINT uses the existing
  integer clear conversion, and SNORM passes floating-point values to the attachment. Signed channels matched
  WARP byte for byte in full and rectangular clears on an RX 9070 XT (driver 32.0.32029.2010) and the integrated
  Radeon in a Ryzen 5 9600X (32.0.21045.29003), including mip and array selections. The SNORM vectors avoid
  half-step rounding ties.
- **Typed RTVs of the other typeless families (D3D-275):** an RTV of any TYPELESS color texture (B8G8R8A8,
  R16G16B16A16, R32G32B32A32, R32G32, R16G16, R10G10B10A2, R8G8, R32, R16, and R8 once it is mapped) names a typed
  format of its family; `TargetView` carries that view's VkFormat (`ColorViewFormat` holds VkFormat numbers) and
  `color_view_format` accepts any format of the resource's family (`typeless_view_family`), so the clear writes
  through a view in that format. Typeless images are already mutable without a format list. Before, only RGBA8 was
  reinterpreted and the others cleared through the resource's own format (an `R8_UINT` RTV cleared to 7 wrote 0xff).
  `d3d12_typeless_families_test` clears every typed RTV format of each family with two value sets and compares the
  bytes with the host D3D12 on the same adapter: 62 of 62 equal on an RTX 3060 (NVIDIA 32.0.16.1714) with
  `LEONIDA_VK_VALIDATION=1`; WARP differs from that host in 8 (rounding of 0.5 to UNORM/SNORM and the
  R10G10B10A2_UNORM alpha). Formats the layer does not map yet (R16G16B16A16_SNORM, R16G16_UNORM/SNORM,
  R16_UINT/SNORM/SINT, the R8 family; #1914) are reported and skipped. Draws need no change: the replay builds a
  draw's attachment view in the PSO's `RTVFormats[slot]` (`GraphicsPipeline::color_formats`, applied in
  `vulkan_replay.h`'s draw setup), which D3D12 requires to equal the RTV's format, so pipeline and attachment
  already agree and the mutable image is written in the typed format. The same test draws a full-screen pass (GLSL
  stand-ins for translated shaders found by FSSM hash; HLSL through d3dcompiler_47 on the host) writing
  `uint4(7, 300, 1000, 3)` or `float4(0.5, -2, 1.5, 0.25)` into R16G16B16A16_UINT/FLOAT/UNORM, R32_UINT/FLOAT,
  R10G10B10A2_UINT/UNORM and R8G8B8A8_UINT views of the matching TYPELESS textures: 8 of 8 equal the host, with no
  validation error, before and after the clear fix.
- **RTVs through a castable-format list in the other channel order:** an `R8G8B8A8` or `B8G8R8A8` texture (UNORM or
  SRGB, `B8G8R8X8` included) whose castable-format list names the RTV's format in the other channel order, across
  `B8G8R8X8` and `B8G8R8A8`, or a typed `B8G8R8A8` texture's listed SRGB format, gets an RTV in that format.
  `CreateRenderTargetView` checks the list as the application gave it (`lists_castable_format`), and
  `color_view_format` lets a castable image's clear cross between the RGBA8 and BGRA8 orders. WARP and AMD's D3D12
  driver (RX 9060 XT, 26.11.2) store clears and draws in the view's channel order and encoding: an `R8G8B8A8` texture
  cleared to (0.2, 0.4, 0.6, 0.25) through a `B8G8R8A8` RTV holds the bytes 153, 102, 51, 64. Before, the RTV was
  refused for an `R8G8B8A8` texture (its clear failed `Close` and its draws were dropped), and a typed `B8G8R8A8`
  texture's RTV kept the image's format whatever the view named, so its clears came out in BGRA order and without
  the sRGB encoding (its draws were right already, as they use the PSO's format). A format the list doesn't name
  is refused, on a typed `B8G8R8A8` or `B8G8R8X8` texture as on an `R8G8B8A8` one (the `B8G8R8A8` and
  `B8G8R8X8` ones kept the image's format before); a typed `B8G8R8A8` texture without a list still ignores the
  RTV's format, as before. `d3d12_clear_test` compares 8 such textures with WARP in full and rectangle clears
  within 1 (the X byte of an X8 view left out: WARP and AMD write different values there) and checks that an
  `R8G8B8A8`, a `B8G8R8A8` and a `B8G8R8X8` texture each take their own and their listed format and refuse 3
  formats their lists don't name, `d3d12_draw_test` draws through the same views, and `typeless_rtv_test` checks
  the replay's crossing and its refusals.
- **Typed RTV clears:** `ClearRenderTargetView` still takes `FLOAT[4]`. For every integer RTV format the device
  exposes (`R8G8`, `R8G8B8A8`, `R16G16`, `R16G16B16A16`, `R32`, `R32G32` and `R32G32B32A32` in `_UINT` and `_SINT`,
  plus `R10G10B10A2_UINT`; `R8`/`R16` `_UINT`/`_SINT` are handled in replay too) convert those values numerically
  into `VkClearColorValue` for both image and rectangle clears; copying float bits gives incorrect stored integers
  without a Vulkan validation error. **Measured:** Windows D3D12 on WARP clamps negatives/NaN to zero, positive
  overflow/infinity to the component maximum (and signed below the minimum), and truncates fractions; signed formats
  use `int32` with the same NaN/overflow rules. `R10G10B10A2_UINT` clamps RGB to 1023 and alpha to 3 (inputs 4, 4.5
  and 2^32 in alpha match WARP). `d3d12_clear_test` compares full and rectangular D3D12 readbacks with WARP across
  UNORM, SNORM, FLOAT, all 15 exposed UINT/SINT render targets (11 value vectors, including 8-/16-/10-/2-bit range
  boundaries) and depth/stencil; `clear_color_test` checks both replay paths for 19 integer formats without a GPU,
  including unchanged float values. SRGB clears are omitted from the comparison: the
  linear→sRGB encoding rounds differently at the `.5` boundary on WARP vs Vulkan (a 1-ulp driver difference).
  **Open:** on NVIDIA RTX 5080 (driver 616.56) clears that land exactly on a `.5` UNORM/SNORM step round down where
  WARP rounds up: `R8G8B8A8_UNORM`/`SNORM` 0.5 store `7f`/`3f` vs WARP `80`/`40`, and a `D16_UNORM` 0.5 depth clear
  stores `0x7fff` vs WARP `0x8000` (all on `main` before this change; same 1-ulp class as SRGB, not addressed here).
  `D32_FLOAT` and `D32_FLOAT_S8X24_UINT` match WARP.
  This does not add typed-view reinterpretation of typeless resources.
- `D3D12XBOX_RESOURCE_FLAG_FORCE_TEXTURE_COMPATIBILITY` (public release note) and DCC: ignore the flag, let the
  Vulkan driver choose compression, and fail loudly if the game aliases a buffer over a compressed image or reads
  raw DCC metadata.
- **G-buffer render targets stay 8-bit UNORM.** Reported by @Silliana in #603 (not reproduced here): the material
  shaders pack data into plain bytes (the normal is octahedral-packed across three bytes of one target, and ID
  bytes sit in another) and the lighting shaders unpack them with a `* 255` and a truncate. If the layer or a test
  gives those targets another format (say f16 for everything), a byte is lost in many pixels: the normals go wrong
  and the specular sparkles. Create them with the format the game asks for (`R8G8B8A8_UNORM` and the like) and don't
  substitute a wider or float format for them.

### 4.3 Barriers and layouts

Track a current state/layout per subresource. D3D12 states are bitmasks; a combined state takes the union of the
mapped stages/accesses, and if no single layout serves all uses, use `GENERAL` and log it. Legacy
`D3D12_RESOURCE_STATE_*` maps to a (stage, access, layout) triple:

| D3D12 state | Vulkan layout | Stage / access |
|---|---|---|
| `COMMON` | `GENERAL` for images, no layout for buffers | Promote on first use |
| `RENDER_TARGET` | `COLOR_ATTACHMENT_OPTIMAL` | `COLOR_ATTACHMENT_OUTPUT` / write |
| `DEPTH_WRITE` / `DEPTH_READ` | `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` / `DEPTH_STENCIL_READ_ONLY_OPTIMAL` | early/late tests |
| `PIXEL_SHADER_RESOURCE` | `SHADER_READ_ONLY_OPTIMAL` | fragment shader / read |
| `NON_PIXEL_SHADER_RESOURCE` | `SHADER_READ_ONLY_OPTIMAL` | vertex/compute/etc. / read |
| `UNORDERED_ACCESS` | `GENERAL` | shader / read+write |
| `COPY_DEST` / `COPY_SOURCE` | `TRANSFER_DST_OPTIMAL` / `TRANSFER_SRC_OPTIMAL` | transfer |
| `RESOLVE_DEST` / `RESOLVE_SOURCE` | `TRANSFER_DST_OPTIMAL` / `TRANSFER_SRC_OPTIMAL` | resolve |
| `STREAM_OUT` | no layout for buffers | transform feedback write (if used) |
| `VERTEX_BUFFER` / `INDEX_BUFFER` / `INDIRECT_ARGUMENT` | no layout for buffers | vertex input / index / draw indirect |
| `PRESENT` | `PRESENT_SRC_KHR` for swapchain images, `GENERAL` otherwise | present |
| `RAYTRACING_ACCELERATION_STRUCTURE` | buffer, no layout | acceleration structure read/write |

- Split barriers (`BEGIN_ONLY`/`END_ONLY`, and enhanced `SYNC_SPLIT`) are an optimisation. The resource is unusable
  between the halves, so one full barrier at either point is correct; the IR records it at the end point
  (`command_list.h`). Use `VK_KHR_synchronization2` events if profiling needs the overlap.
- Implicit promotion and decay need no extra barriers: `COMMON` is `GENERAL`, and the layout tracker makes later
  barriers start from it (section 3.2).
- Aliasing barriers: if image aliasing is implemented (one `VkDeviceMemory` per D3D12 heap range, with
  `VK_IMAGE_CREATE_ALIAS_BIT`), transition the old image to `UNDEFINED`, issue a global memory barrier, then
  transition the new image. Until then, log aliasing and allocate distinct memory; do not silently return the
  wrong contents.
- Enhanced barriers (`D3D12_BARRIER_SYNC_*`, `D3D12_BARRIER_ACCESS_*`, `D3D12_BARRIER_LAYOUT_*`) map directly to
  `VkPipelineStageFlags2`, `VkAccessFlags2` and `VkImageLayout` (`vulkan_sync`, `vulkan_barrier_access`,
  `vulkan_layout`). Queue-specific layouts map like their generic forms; ray tracing, video and split scopes use
  `ALL_COMMANDS`; `LayoutBefore = UNDEFINED` or the discard flag becomes `oldLayout = UNDEFINED`.
- `DiscardResource` (D3D-26): each discarded image subresource gets a barrier from `oldLayout = UNDEFINED` to the
  layout the tracker already has for it, so the driver may drop the contents while later commands see no layout
  change. The other cases are no-ops, which "contents undefined" allows: buffers, discards with rectangles (only
  part of each subresource), a part never used yet (it is still to be zero-filled on first use), a list replayed
  without a tracker, and one plane of a combined depth-stencil image without `separateDepthStencilLayouts`
  (discarding one plane would also discard the other). A null region is recorded as `whole`; a region is always an
  explicit range, even with `NumSubresources = UINT_MAX`, so its `FirstSubresource` is checked like any other.
- Access bits and sync scopes are not one-to-one. D3D12 lets an access be named with a sync scope whose Vulkan
  stages cannot perform it: `SYNC_ALL_SHADING` with `ACCESS_VERTEX_BUFFER` is valid D3D12, but
  `VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT` needs `VERTEX_ATTRIBUTE_INPUT`, which shading stages are not. Vulkan
  rejects such a barrier. `vulkan_barrier_scope` therefore widens each side's stages by the narrowest stage that
  can perform each access without one. The table is the synchronization2 chapter's "supported access types". The
  stages it adds:
  - `DRAW_INDIRECT` for indirect arguments;
  - `INDEX_INPUT` and `VERTEX_ATTRIBUTE_INPUT` for index and vertex buffers;
  - all shading for constant buffers, SRVs and UAVs;
  - `COLOR_ATTACHMENT_OUTPUT` for render targets;
  - the fragment tests for depth/stencil;
  - `ALL_TRANSFER` for copies and resolves. `CLEAR` alone cannot read, and `ALL_GRAPHICS` contains no transfer stage.

  The access itself is never dropped, so the dependency still covers everything the application named, over a
  slightly larger scope.
- Render passes: `BeginRenderPass`/`EndRenderPass` map to `vkCmdBeginRendering`/`vkCmdEndRendering` (dynamic
  rendering). The `D3D12_RENDER_PASS_BEGINNING_ACCESS`/`ENDING_ACCESS` types give the load/store ops and resolve
  targets. Suspending/resuming passes have no direct Vulkan equivalent; split them into two rendering scopes with
  the same attachments and `LOAD`/`STORE`. If the game does not use render passes, build a rendering scope per draw
  from the bound RTVs/DSV and let explicit `Clear*View` calls do the clears.

### 4.4 Fences, queues and frames in flight

- One timeline semaphore per `ID3D12Fence`. Queue `Signal` = `vkQueueSubmit` signal; queue `Wait` = submit wait.
- **Wait before Signal across queues, with resources handed over** (`src/queue_holds.h`, `VulkanQueue`; test
  `d3d12_queue_handover_test`). Vulkan accepts a timeline wait whose signal comes later, but the layer replays every
  `ExecuteCommandLists` against one `LayoutTracker` in the order the calls arrive, so a queue that waits for work
  submitted afterwards would be recorded against the layouts from before that work (a copy list that reads a texture
  a compute list has not yet written was replayed against `UNDEFINED`: zero-filled, and a validation error). The layer
  therefore tracks, per fence, the highest value a Signal has been *submitted* for (or the CPU has set; the creation
  value counts), which is not the completed value. A queue that is asked to run a batch (or a `Signal`) while a
  `Wait` it carries names a value above that, or while an earlier batch of the queue is held, copies the lists
  (`CommandList` is a value; the application may `Reset` its list at once) and holds the operation. Everything the
  queue is asked to do after it (batches, Signals, queue markers) goes behind it, so per-queue order is kept.
  The held operations are recorded and submitted, in order, on their own `VkQueue` as soon as the Signals they wait for
  have been submitted: by the thread that submits that Signal (or makes the CPU `Signal`), after it has dropped its own
  queue's state lock. There is no worker thread and no GPU wait is added, but the call that releases is not free: that
  thread records the held batches (what the `ExecuteCommandLists` would have cost, including whatever `execute` does
  for the lists, such as the bindless upload's wait for the device until D3D-314 removes it) while it still holds the
  mutex of the D3D12 queue whose `Signal` or `Wait` it is in, so other calls on that queue wait meanwhile; and
  `remove_fence`, queue destruction and presenter destruction wait for a release in progress. One thread releases at a
  time (the others leave a note and the running one makes another pass); a release that submits a Signal lets the
  next queue's front run in the same pass, so chains (C waits B, B waits A, in any submission order) and operations
  with several waits resolve in dependency order, and the layout tracker follows that order. A wait whose value has
  been reached by a submitted Signal never blocks again (vkd3d-proton's rule too). The normal case (the Signal
  submitted first) is unchanged: no copy, no hold, the same `vkQueueSubmit`. Holding on the CPU also avoids a Wait and
  its Signal landing in the wrong order on one `VkQueue`, which would not make forward progress on the GPU
  (Vulkan-Docs issue 1262): the test also runs with every D3D12 queue on one `VkQueue` and with two.
  What a held operation keeps alive, as D3D12's queue does: the fences it waits on and signals, until its submission
  has completed on the GPU (a released fence stays until then, which also removes the destroy-while-pending
  validation error the unheld path could hit); the pipeline states and root signatures of its lists (copied from the
  recording before the application may `Reset` it, so releasing a PSO right after `ExecuteCommandLists` is fine).
  Resources are not kept: a resource released while a batch that names it is held fails that batch at release.
  Order against other work: a held `Signal(F, v)` released after an unheld `Signal(F, w)` with `w >= v` leaves the
  Vulkan signal out (timelines only increase; waits for `v` already pass), decided under the submit lock so the
  order of the checks is the order of the submissions, and the value counts as submitted only once the
  `vkQueueSubmit` succeeded (a failed Signal releases nobody and its retry goes through); `Present` is queued behind
  held work too (below). Fence keep-alives are released outside the queue lock, at the next `ExecuteCommandLists`,
  `Signal`, `Wait` or marker once their submission has completed. Edge cases:
  a queue destroyed with held operations reports them once (stderr: `queue destroyed with N held operation(s) never
  submitted: waits for a fence value (V) that no Signal reached`) and goes on without waiting, and the CPU waits on
  the fences of its held Signals are completed, as a released fence completes them; queues that wait for those fences
  stay held and are reported when they go; a fence nobody signals holds back only the queues that wait for it (and
  for what they would have signaled), never the others; a Signal value that is not above the fence's is still
  dropped with a note at call time, as before (vkd3d-proton instead maps virtual fence values to physical timeline
  values; the layer does not yet). A held `ExecuteCommandLists` that fails when it is released does not return its
  result to a call that has returned: it is printed, `last_execute_result` and `VulkanQueue::take_held_failure`
  report it, the `Signal` behind it still signals (the work did not run; nothing waits forever, as for a failure that is
  not held), and its waits stay for the queue's next submission, as after a failed `vkQueueSubmit`. A queue does not
  hold `UpdateTileMappings` / `CopyTileMappings`: they still submit at once, before earlier held work, and take the
  queue's pending waits (documented limit). `VulkanQueue::idle` waits for submitted work only.
  `Present` (`VulkanPresenter`): when any queue holds work, a presented back buffer would be blitted as it was before
  that work, so the Present is queued instead (`QueueHolds::after_held`), returns success as a held
  `ExecuteCommandLists` does, and is submitted, in order with the presents queued before it, by the thread that
  releases the last operation held when it was called. It is not blocked on the calling thread because the Signal that
  releases the work may be one that thread makes after `Present` returns. Held work that is never released holds back
  the presents after it (device-wide, since the layer cannot tell which batch draws into which back buffer). A
  blank present reads nothing and is only ordered behind earlier queued presents. At most 8 presents queue up; the
  next one waits for held work to release (logged once; `set_abort` ends the wait), as a full frame-latency queue
  does. A queued present reports `last_present_tore()` false, which the Xbox PresentX path turns into the default
  vblank sync for the frame pacer. The presenter's state is guarded by one mutex (the counters are atomics, so
  reading them never waits for a present blocked on a swapchain image); `set_abort` has its own.
  Facts from vkd3d-proton (LGPL, approach only, nothing copied): it keeps a CPU-side virtual value and pending signals
  per fence and blocks the queue's submission thread until a signal that satisfies a wait has been submitted, so Vulkan
  never sees a wait without a submitted signal. Resources are `VK_SHARING_MODE_EXCLUSIVE` and every D3D12 queue is in
  the graphics family, so no ownership transfer is needed; with dedicated compute/transfer families they would need
  `CONCURRENT` (vkd3d-proton does that, because D3D12 gives no information for transfers).
  Every mode of the handover test ends with the same cases on the host runtime (the expected data is the CPU's
  pattern in all of them, so that run shows the cases are valid D3D12 and holds the layer's modes to one bar).
  The pipeline-state and root-signature pins are covered end to end by `d3d12_m4_compute_test`: the compute list is
  executed while held, its two PSOs and root signature are released and the list is Reset, the copy that releases
  it is submitted, and the output is compared with the CPU (it reads zeros without the pins).
  Measured: `d3d12_queue_handover_test` (three modes: the standalone device with a `VkQueue` per D3D12 queue, then the
  host runtime; `--two-queues`, a caller-owned device with two `VkQueue`s handed over; `--shared-queue`, one) runs
  592 cases on the layer on the Radeon 880M and the RTX 5070 (also with one shared `VkQueue`; 8 with two handed over) and 587 on the host runtime: pairs and three-stage chains (produce, relay, consume)
  over the queue type assignments that fit and all 6 submission orders (consumer first among them), two producers
  (texture from one queue, buffer from another) with one consumer, a chain released by a CPU `Signal`, a CPU
  `Signal` before and after the `Wait`, a never-signaled fence, a fence released while a batch waits, a held Signal
  of a fence the application released, a held Signal below an unheld one on a shared fence, teardown with a held
  Signal under a CPU wait, and 200 rounds with two threads signalling and releasing at once; results compared byte
  for byte, sync validation clean. With holding disabled the test fails 236 cases and validation reports
  `VUID-vkCmdDraw-None-09600`; with the monotonic check disabled it reports
  `VUID-VkSubmitInfo-pSignalSemaphores-03242`; `vk_presenter_sync_test` and `d3d12_m4_compute_test` (which
  submits the compute list first and releases its PSOs and root signature before the copy that feeds it) fail
  without the Present ordering and the pins.
- **More D3D12 queues than Vulkan queues** (`pick_queue_slot` in `src/queue_families.h`, `CreateCommandQueue`;
  test `d3d12_queue_share_test`). A GPU can expose fewer `VkQueue`s than a title creates D3D12 queues: the Intel
  UHD 770 on Windows and lavapipe have one family with `queueCount` 1, so the second D3D12 queue of any type was
  refused with `E_OUTOFMEMORY` ("no independent Vulkan queue available") and a title with a DIRECT and a COMPUTE
  queue stopped at boot (PingasMech, #4863). A new D3D12 queue takes an unused `VkQueue` of its own type's family,
  then of the graphics family, as before (D3D-97: one `VkQueue` per live D3D12 queue while there are enough, so
  nothing changes on a GPU with enough queues; an RTX 5080 gives 16 graphics, 8 compute and 2 transfer queues); when
  every one is in use it shares the one with the fewest D3D12 queues (its own family's first on a tie) and logs that
  once per type. D3D12 queues on one `VkQueue` stay separate objects: each keeps its own batches in order, its own
  progress timeline and its own held operations; the `VkQueue` runs their batches one after the other, in the order
  they reach it, so they lose concurrency, not correctness (as on one family, below). What would break is a GPU
  wait that no submitted signal satisfies landing on the shared `VkQueue` ahead of the signal it waits for, which
  never makes progress. The holding above already rules that out: a batch reaches `vkQueueSubmit` only once every
  fence value it waits for has a *submitted* Signal (earlier on the same `VkQueue`, or on another one) or has been
  set by the CPU. One case let the CPU half slip: a CPU `Signal` that has to wait behind a queue Signal the GPU has
  not reached yet (`VUID-VkSemaphoreSignalInfo-value-03259`, applied later by the fence wait thread) never told the
  held queues, so a queue waiting for that value was held for ever, on separate `VkQueue`s too. The wait thread now
  reports the value once it has set it, and releases what it satisfies; reporting it earlier, when the CPU called
  `Signal`, would send the waiting batch ahead of the queue Signal it depends on, which on a shared `VkQueue` never
  completes. `vkQueueSubmit`, `vkQueueBindSparse`, `vkQueuePresentKHR`, `vkQueueWaitIdle` and the queue labels
  need the `VkQueue` externally synchronised; every call the layer makes holds the device's one submit lock
  (`VulkanContext::submit_mutex`), which covers shared `VkQueue`s as it covers the staging queue the first DIRECT
  queue already shares. A per-`VkQueue` lock (#3841) must key on the `VkQueue` handle, not the D3D12 queue. Costs on
  a shared `VkQueue`: `GetClockCalibration` without calibrated timestamps and the destruction of a D3D12 queue wait
  for the other D3D12 queues' submitted work too, and queue debug labels (`BeginEvent`/`EndEvent`) of the D3D12
  queues interleave on it. Two test settings make the standalone device look like such a GPU, so the sharing path
  runs anywhere: `LEONIDA_VK_MAX_QUEUES_PER_FAMILY=N` (at most N `VkQueue`s per family) and
  `LEONIDA_VK_GRAPHICS_FAMILY_ONLY=1` (COMPUTE and COPY in the graphics family); both together are one family with
  one queue (`docs/testing.md`, "Choosing the GPU"). Only the test executables apply them: the game's device
  (`d3d12_xs`, `LEONIDA_D3D12=vulkan`) ignores them and says so, so a variable left in the environment cannot put
  every queue of the game on one `VkQueue` (`run_tiny_frame_queue_test_settings`). Measured on an RTX 5080 (Linux, driver 580.65.06, Wine 11)
  and on lavapipe, under the validation layer with synchronization validation: `d3d12_queue_share_test` passes
  with no setting, with one queue per family, with one queue in all and on lavapipe (36 queues, a buffer copied
  exactly through all of them consumer first and producer first, 9 deferred CPU Signal cases, 4 threads on 9 queues:
  2,000 iterations with all 9 on one `VkQueue` of the RTX 5080, 80,000 on lavapipe; a fence released right after
  the GPU passed the queue Signal its deferred CPU Signal waited behind, 900 times per mode, the held batch running
  every time), 0 validation messages; on main
  it fails at the 27th queue on the RTX 5080 and at the 2nd on lavapipe, and the first deferred case hangs. The layer
  half of `d3d12_queue_handover_test` runs its 371 cases exact in all four (on lavapipe main ran 0: one queue).
- `SetEventOnCompletion(value, hEvent)`: a small wait thread calls `vkWaitSemaphores` and `SetEvent`. Reuse one
  thread with a condition variable; do not create a thread per fence.
- `GetCompletedValue` is `vkGetSemaphoreCounterValue`; CPU `Signal` is `vkSignalSemaphore`.
- `SetEventOnMultipleFenceCompletion` (implemented). With a null event it is one blocking `vkWaitSemaphores` over
  every fence's semaphore, with `VK_SEMAPHORE_WAIT_ANY_BIT` for `D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY` and no flag for
  `ALL`. With an event, the device's fence wait thread counts down one shared waiter (1 for `ANY`, the fence
  count for `ALL`, kept as the API's unsigned `UINT`) and the hit that reaches zero sets the event, so no thread is
  added; later hits stay at zero (`d3d12_fence_waiter_test` checks counts past `INT_MAX`). Values already reached are
  counted before the call returns. Zero fences, other flags, a null array or a fence from elsewhere are
  `E_INVALIDARG`, as on the PC runtime. A fence released while an event waits on it counts as reached for that
  wait, as on the PC runtime (RTX 5070 and WARP): `SetEventOnCompletion` and `ANY` set the event, `ALL` counts it
  as one fence (`d3d12_fence_events_test`).
- The swapchain owns acquire/present semaphores and N frames in flight. A present waits for the frame's
  render-finished semaphore; the next acquire waits for the image's present-complete semaphore.
- Queue types: a direct queue maps to a graphics+compute queue, a compute queue to a compute-only family if one
  exists, a copy queue to a transfer-only family. If the host exposes only one family, serialize and log; async
  compute is a performance feature, not a correctness one.
- `GetTimestampFrequency`/`GetClockCalibration` use `VK_EXT_calibrated_timestamps` where available.
  `GetClockCalibration` (implemented) returns the GPU stamp in device ticks (`1e9 / timestampPeriod` per second,
  the units of `GetTimestampFrequency`) and the CPU stamp in `QueryPerformanceCounter` units. When the device has
  `VK_KHR_calibrated_timestamps` or `VK_EXT_calibrated_timestamps` enabled (`VulkanDeviceInfo::calibrated_timestamps`;
  the standalone device enables whichever the GPU has) and offers the `DEVICE` and `QUERY_PERFORMANCE_COUNTER`
  domains, it is one `vkGetCalibratedTimestampsKHR` call. Otherwise the queue waits until idle, submits a
  `vkCmdWriteTimestamp` and waits for it, and answers with the midpoint of the `QueryPerformanceCounter` reads
  around that submission. This stalls the queue, which is acceptable for a call made once per profiling capture. A
  queue family with `timestampValidBits = 0` fails with `E_FAIL`. `d3d12_fence_timing_test` takes two calibrations
  50 ms apart and compares the GPU and CPU intervals. On an RTX 4090 (driver 621.14) the gap is 2 us with calibrated
  timestamps, 20 us with the fallback, and 0.4 ms with the fallback under the validation layer. The test fails
  above 2 ms.

### 4.5 Swapchain and present

The game package does not include `dxgi.dll`, so the executable imports it from the system (or from the GDK).
That import must be shimmed. The shim provides:

- `CreateDXGIFactory`/`CreateDXGIFactory2` → one fake factory with one adapter describing the host Vulkan device.
  `IDXGIAdapter::GetDesc` reports host memory/LUID; the game may use these for settings. **Open:** whether it
  expects Xbox Series X values (16 GB, fixed clocks) instead.
- `IDXGIFactory::CreateSwapChain`/`CreateSwapChainForHwnd` → `VkSurfaceKHR` for the HWND + `VkSwapchainKHR`.
  `GetBuffer(i)` wraps each swapchain image as an `ID3D12Resource` so the game can create RTVs.
- `Present` → `vkQueuePresentKHR`; `ResizeBuffers`/`ResizeTarget` recreate the swapchain; `SetFullscreenState`
  toggles borderless/fullscreen; `GetFrameStatistics` is a stub.
- `DXGI_PRESENT_TEST`/`DO_NOT_WAIT` map to a no-op/`VK_NOT_READY` path.

**What exists.** The DXGI shim's `Present` (`runtime/shims/dxgi_com.cpp`) has its own surface, swapchain and
blit (`leonida::VulkanSwapchain`, `runtime/shims/dxgi_swapchain.h`); `PresentX` (`runtime/shims/xbox_present.cpp`)
is implemented over `leonida::VulkanPresenter` (section 3.2) instead. `PresentX` takes ordinary committed
textures and blits them at present time, which frees the game's buffer count and format from the surface's. The
DXGI shim's `GetBuffer(i)` returns the swapchain images themselves, and switches to offscreen images blitted at
present only when the requested size or format differs from the surface's (`runtime/shims/dxgi_swapchain.h`); its
buffer count is the swapchain's image count.

**PresentX and frame events (GDKX).** Public Microsoft samples present on Xbox without a DXGI swapchain. Facts
below come from MIT/zlib-licensed code and docs, read directly; none is from the game or the SDK. Sources:
[SimpleDeviceAndSwapChain](https://github.com/microsoft/Xbox-GDK-Samples/tree/main/Samples/IntroGraphics/SimpleDeviceAndSwapChain)
(readme: "Instead of using DXGI for presentation, this uses the new PresentX API"),
[FramePacing](https://github.com/microsoft/Xbox-GDK-Samples/tree/main/Samples/Graphics/FramePacing),
[DirectXTK12 wiki, DeviceResources](https://github.com/Microsoft/DirectXTK12/wiki/DeviceResources) ("The GDKX
platform uses PresentX, so the method WaitForOrigin has to be called each frame loop"),
[directxtk12modelviewer `DeviceResourcesGXDK.cpp`](https://github.com/walbourn/directxtk12modelviewer) (MIT) and
SDL2 `src/render/direct3d12/SDL_render_d3d12_xbox.cpp` (zlib).

*Fact (call sequence in those samples):*

1. `D3D12XboxCreateDevice(nullptr, &params, iid, &device)`: adapter pointer null, a params struct (ring and scratch
   sizes, `D3D12XBOX_DEFAULT_SIZE_BYTES` = use default), IID, out pointer. SDL asks for `ID3D12Device1`.
2. Back buffers are ordinary `CreateCommittedResource` calls with `D3D12_HEAP_FLAG_ALLOW_DISPLAY`, initial state
   `D3D12_RESOURCE_STATE_PRESENT`, `ALLOW_RENDER_TARGET`, 2D, one mip (SDL: `B8G8R8A8_UNORM`). They are the
   game's own resources, not objects owned by a swapchain.
3. DXGI is used only to find the output: QI the device for `IDXGIDevice1`, `GetAdapter`, `EnumOutputs(0)`.
4. `device->SetFrameIntervalX(output, D3D12XBOX_FRAME_INTERVAL_60_HZ, n-1, D3D12XBOX_FRAME_INTERVAL_FLAG_NONE)`:
   4 arguments; the third is the allowed frame latency (SimpleDeviceAndSwapChain passes buffer count minus 1,
   SDL passes 1). Then `device->ScheduleFrameEventX(D3D12XBOX_FRAME_EVENT_ORIGIN, 0, nullptr, ..._FLAG_NONE)`:
   4 arguments (event, a `UINT` passed as 0, a pointer passed as null, flags).
5. Per frame: `device->WaitFrameEventX(D3D12XBOX_FRAME_EVENT_ORIGIN, INFINITE, nullptr, ..._FLAG_NONE, &token)`:
   5 arguments, the last an out pointer to a `D3D12XBOX_FRAME_PIPELINE_TOKEN` (SDL stores it in a `UINT64`; the
   null value is `D3D12XBOX_FRAME_PIPELINE_TOKEN_NULL`). Then render, `ExecuteCommandLists`, then
   `queue->PresentX(planeCount, planes, params)`: 3 arguments (`UINT`, pointer to an array of
   `D3D12XBOX_PRESENT_PLANE_PARAMETERS`, optional pointer to `D3D12XBOX_PRESENT_PARAMETERS`; null in the simple
   samples).
6. Plane fields used: `Token` (the value from the wait), `ResourceCount` (1), `ppResources` (pointer to the
   resource pointer(s)), `pSrcViewRects` (pointer to a rect, HDR case only) and `ColorSpace` (a
   `DXGI_COLOR_SPACE_TYPE`). The HDR path uses 2 planes sharing the token: plane 0 `RGB_FULL_G2084_NONE_P2020`
   (HDR10), plane 1 `RGB_FULL_G22_NONE_P709` (the GameDVR/SDR copy, its own back-buffer set). Non-HDR uses 1
   plane. This agrees with the UMD limit of 2 planes in `d3d12_xs_surface.md` §7.
7. `D3D12XBOX_PRESENT_PARAMETERS` has at least `ImmediateThresholdPercent` (float). Per the FramePacing text: a
   late frame is shown immediately only if scanout has not passed that percentage of the screen; 0 disables
   limited tearing, 100 allows a late flip anywhere in the scan.
8. Suspend: `queue->SuspendX(0)` (one argument, passed 0; no other thread may be rendering). Resume:
   `queue->ResumeX()`, then **re-run** steps 4's `SetFrameIntervalX`/`ScheduleFrameEventX`. The samples
   say frame events must be registered again after resume.
9. Shutdown: `PresentX(0, nullptr, nullptr)` presents a blank screen before resources are released.
10. Statistics: `device->GetFrameStatisticsX(token, typeMask, &count, array)`: 4 arguments. Type bits seen:
    `FRAME_INTERVAL`, `FRAME_EVENT`, `RENDER`, `PRESENT`, `DISPLAY`; `count` is in/out; `S_FALSE` or a short
    count means "not available yet"; no statistics for the first presented frame; render/present/display
    entries carry a `PlaneIndex`; display entries carry `PercentScanned`.
11. Xbox titles do not handle `DXGI_ERROR_DEVICE_REMOVED`/`RESET`.

*Public corroboration (not measured here).* XCL, a GPL-2.0-only compatibility layer that runs retail Xbox Series
titles, records `PresentX` at queue slot 30 for the older runtime layout (XDefiant, Minecraft, Fortnite) and 32 for
a newer one (Unicorn Overlord), and the four frame calls at device slots 78-81: `SetFrameIntervalX`,
`ScheduleFrameEventX`, `WaitFrameEventX`, `GetFrameStatisticsX` (0-based, `QueryInterface` = 0). The Apache-2.0
ebiten Xbox binding has the same device order, declares `D3D12XBOX_PRESENT_PLANE_PARAMETERS` as a 0x48-byte x64
struct and `D3D12XBOX_PRESENT_PARAMETERS` as 0x20 bytes, and stores the token in 64 bits. So the plane and
present-parameter layouts and the token width in `runtime/shims/xbox_present.h` are no longer guesses; the enum
values and the `SuspendX`/`ResumeX` slots still are. XCL returns `S_FALSE` from a timed-out `WaitFrameEventX`
where this layer returns `HRESULT_FROM_WIN32(WAIT_TIMEOUT)`. The full comparison is in
[d3d12_xs_surface.md §7.1](d3d12_xs_surface.md#71-public-sources-for-the-presentation-and-device-abi).

*Hypothesis (the game is not observed):* a GDKX title follows this pattern, so the DXGI swapchain shim (D3D-2)
is probably **not** on its boot path. What is on it: `D3D12XboxCreateDevice`, `IDXGIDevice1`/`IDXGIAdapter`/
`IDXGIOutput` enumeration, committed `ALLOW_DISPLAY` resources, the frame events and `PresentX`. Without the
frame events and `PresentX` no frame reaches the screen. To confirm: trace which of `CreateSwapChain*` or the
queue's `PresentX` slot the game calls first; check the executable's `dxgi`/`d3d12_xs` imports.

Layer plan:

- Keep the DXGI factory, adapter and output enumeration (adapter `GetDesc`, `EnumOutputs`); no host DXGI.
- Step 3 of the sequence is on the device proxy: it answers `IDXGIObject`, `IDXGIDevice` and `IDXGIDevice1` as one
  COM object with the device (a PC `ID3D12Device` has no `IDXGIDevice`, so nothing is forwarded), and `GetAdapter`
  and `GetParent` return the adapter of the device behind it: the layer's own on the Vulkan device, the host adapter
  with the device's LUID on the host backend. The 14 slots are the public `dxgi.h` order, an assumption like the
  queue's. `SetMaximumFrameLatency` and `SetGPUThreadPriority` are kept and read back only. `dxgi_device_test`
  walks it on both backends; the output itself is `EnumOutputs` (D3D-313).
- `ALLOW_DISPLAY` committed resources are normal `Resource`s that the presenter may blit to a Vulkan
  swapchain image; `PRESENT` state maps per the resource-state table (`PRESENT_SRC_KHR` only for swapchain images).
- Presentation core: `PresentX` with 0-2 planes presents into the `VkSwapchainKHR` (one plain plane is a blit; scaling,
  placement, HDR10 and an overlay plane 1 go through a compose pass, see "Plane composition" below); `SetFrameIntervalX`/`ScheduleFrameEventX`/`WaitFrameEventX` run on a vblank
  timer and hand the token through to the present; `GetFrameStatisticsX` minimal; `SuspendX`/`ResumeX`; 0
  planes = blank screen. Test the methods directly with a fake clock, before slots are known.
- The DXGI `CreateSwapChain*` path stays as the fallback and for PC-style tests, not as the first-boot target.

**What is implemented (the presentation core).** Everything in the plan above that does not need a slot exists as
methods of `rt::d3d12::XboxPresentation` (`runtime/shims/xbox_present.h`), on top of the presenter of section 3.2 and
`leonida::FramePacer` (`runtime/src/frame_pacer.h`, no Vulkan, no Windows headers). The plane and present-parameter
layouts and the 64-bit token are confirmed on x64 by the public code cited above; the constant values and HRESULTs
are **our own provisional definitions**, because the samples give call shapes and field names only and the enum
values are still unmeasured. The class is the surface the proxy will forward to. The queue-side calls ignore which
queue they came on (one `VkQueue` behind all of them).

- *Frame pacing* (`FramePacer`, time from an injectable clock; real time is a steady timer, a host vblank source
  can replace it). `SetFrameIntervalX` takes the interval as a period in microseconds and the latency, and starts
  the vblank grid. `ScheduleFrameEventX` arms the origin event, delayed by an offset inside the frame.
  `WaitFrameEventX` returns at the next origin with the next token (1, 2, 3, ...; 0 is the null token) and does not
  return while `latency` presented frames are still waiting for their vblank. A frame presented before the origin event
  is armed is not paced (`PresentX` does not hold it for a vblank), so it is shown at once and never waits for a vblank:
  blank presents of a loading screen do not delay the first wait. A wait that comes late returns at once
  with the newest passed origin; the missed ones are skipped. A finite timeout gives
  `HRESULT_FROM_WIN32(WAIT_TIMEOUT)`; nothing registered, or suspended, gives `HRESULT_FROM_WIN32(ERROR_INVALID_STATE)`.
  `GetFrameStatisticsX` keeps the last 64 frames and reports the event, present and display times (not render;
  `PercentScanned` only for a frame shown at once by `ImmediateThresholdPercent`, else 0); `S_FALSE` with count 0 for what it does not have.
- *`PresentX`*: 0 planes presents a black frame (no token needed). 1 plane blits plane 0's first resource (a
  single-sample 2D colour texture in `PRESENT`) to the swapchain image, limited to `pSrcViewRects[0]` when set
  (bilinear, so the rectangle's outer texels blend with their neighbours). The blit converts between formats; an sRGB
  source gets an sRGB swapchain format and every other source a UNORM one, so the encoded bytes go through unchanged.
  A source the GPU cannot blit gives `DXGI_ERROR_UNSUPPORTED` and presents nothing: the checks are
  `vkGetPhysicalDeviceFormatProperties` (`BLIT_SRC` on the source, `BLIT_DST` on the swapchain format, cached per
  format; the scaling filter is linear only with `SAMPLED_IMAGE_FILTER_LINEAR`, else nearest), plus integer formats,
  which a UNORM swapchain cannot take. Multisampled planes are refused the same way: resolving one into a temporary
  image is not done. 2 planes must share one token (see "Plane composition" for what happens to plane 1). More than 2
  planes, null planes, or an empty/outside rectangle give `E_INVALIDARG`. A missing `ALLOW_DISPLAY` heap flag is logged
  once and presented anyway.
- *Plane composition* (M7). *Fact* (public MIT sample, Xbox-GDK-Samples `Samples/Graphics/HDR10/DeviceResources.cpp` and
  its readme): the two-plane form there is **not** an overlay. Plane 0 is the HDR10 back buffer
  (`RGB_FULL_G2084_NONE_P2020`), plane 1 is a second back-buffer set holding the SDR rendering of the same frame
  (`RGB_FULL_G22_NONE_P709`), both with the same full-output `pSrcViewRects`; the readme says the SDR signal "is still
  required for GameDVR and screenshots" and the SimpleHDR readme says the system can generate it itself
  (`D3D12XBOX_RESOURCE_FLAG_ALLOW_AUTOMATIC_GAMEDVR_TONE_MAP`). *Fact* (field names seen in a public Go binding of the
  header, hajimehoshi/ebiten, Apache-2.0; no layout or values): `D3D12XBOX_PRESENT_PLANE_PARAMETERS` is `Token`,
  `ResourceCount`, `ppResources`, `pSrcViewRects`, `pDestPlacementBase` (a `D3D12XBOX_VIEW_RECT*`, four floats
  left/top/right/bottom), `ColorSpace`, `ScaleFilter`, `ExtendedDescCount`, `pExtendedDescs`, `Flags`;
  `D3D12XBOX_PRESENT_PARAMETERS` is `ImmediateThresholdPercent`, `ViewCount`, `ExtendedDescCount`, `pExtendedDescs`,
  `Flags`. *Fact* (GDK reference text for `XGameStreamingSetResolution`, read through a third-party mirror of the docs, not
  on Microsoft's site): `pDestPlacementBase` scale factors of 0.0-1.0 refer to the output size (the streaming resolution,
  else 1920x1080 or 3840x2160), and `pSrcViewRects` with `pDestPlacementBase` is how a plane is scaled. Not found in public
  material (so **hypotheses**, all labelled ours in `xbox_present.h`): the values of `ScaleFilter` and `Flags`, whether
  plane 1 can be a blended overlay at all on the console (the plane limit of 2 and the field set suggest a display-plane
  model, but no public sample uses it that way), premultiplied or straight alpha, what is shown outside a plane's
  placement, the blending space, the SDR white level, and every colour space but the two of the HDR sample.

  What the layer does. A plain single plane (SDR colour space, no placement, default filter) is the old blit, unchanged
  (`composed_presents()` does not move; the test compares the bytes of a blit with the compose pass at 1:1 and they are
  equal). Anything else goes through `VulkanPresenter::present_planes`: a compute pass (`runtime/src/present_compose.comp`,
  SPIR-V embedded as `depth24_convert` is, built with `glslc --target-env=vulkan1.1 -O -mfmt=c`) writes an RGBA16F image of
  the swapchain's size, and a `vkCmdBlitImage` (same size, nearest) puts it in the swapchain image, so the layer writes
  the swapchain only by transfer (no storage usage) and the capture path is unchanged. The swapchain also asks for
  `COLOR_ATTACHMENT`, which the layer never uses: overlays that hook the swapchain (Bandicam's
  `VK_LAYER_bandicam_helper`, measured) draw into its images through a framebuffer, and without it every frame raised
  `VUID-VkFramebufferCreateInfo-pAttachments-00877` and `VUID-VkImageViewCreateInfo-image-04441`. Surfaces always
  report it, and the DXGI swapchain (`dxgi_swapchain.h`) already requested it. The planes are read with `texelFetch` (no sampler, so any
  sampled float/UNORM/sRGB format works and filtering never reads outside the source rectangle); an sRGB-format source is
  decoded by the texture unit and encoded again, so its bytes are read as stored. Plane 0: `pSrcViewRects[0]` (else the
  whole texture) scaled to `pDestPlacementBase` (fractions of the output; null is the whole output; outside it is black),
  `ScaleFilter` point (the texel containing the sample) or linear/default (bilinear, clamped inside the source rectangle).
  Plane 1 is blended over plane 0 (in linear light after conversion, see "Linear-light filtering and blending"; alpha premultiplied by default or
  straight with `xbox_plane_flag_straight_alpha`) **only** when it carries our `xbox_plane_flag_overlay`; without it plane 1
  is not composed (logged once), as before, because in the public sample it is the GameDVR copy of the frame and blending
  a plane of unknown alpha would draw the frame twice (hypothesis: whether the console blends a second plane at all is
  unmeasured). The extra fields are read defensively, since their types and values are unmeasured: a placement that is not
  a sub-rectangle of 0..1 (empty, inverted, outside, NaN), an unknown `ScaleFilter` and `ExtendedDescCount` are ignored and
  logged once, and the plane is shown as if they were absent; a single SDR plane with null/zero extras stays on the plain
  blit. Planes the pass cannot read (multisampled, depth, integer, block-compressed) give `DXGI_ERROR_UNSUPPORTED`. The struct
  in `xbox_present.h` follows the field order of the binding cited above exactly (`Token`, `ResourceCount`, `ppResources`,
  `pSrcViewRects`, `pDestPlacementBase`, `ColorSpace`, `ScaleFilter`, `ExtendedDescCount`, `pExtendedDescs`, `Flags`); the
  types and sizes are ours. The compose target follows the swapchain when it is recreated (resize, out-of-date acquire);
  `xbox_present_test` resizes the window between composed presents.

  Colour. Only `RGB_FULL_G22_NONE_P709` (0, SDR, also the default for a zeroed struct) and `RGB_FULL_G2084_NONE_P2020` (12,
  HDR10) are composed; any other value is composed as SDR and logged once. The swapchain is UNORM. If a plane is HDR10 and
  the surface lists an ST.2084 format (`VK_COLOR_SPACE_HDR10_ST2084_EXT`, `A2B10G10R10` or `A2R10G10B10`; needs
  `VK_EXT_swapchain_colorspace` enabled on the instance, the test device does) the swapchain is created in it and an HDR10
  plane goes through the working space unchanged; else the swapchain is SDR (the logged fallback). The real runtime's
  instance (`create_vulkan_device_standalone`) enables `VK_EXT_swapchain_colorspace` when the loader has it (it did not
  before; only the test device did), so the HDR10 surface formats can be listed there; `xbox_present_test` checks this on
  the standalone path. Limited-range and YCbCr colour spaces are not modelled.

  *Linear-light filtering and blending* (decision from the sources below; the console's own blending space is unmeasured).
  Every texel of a plane is decoded to linear light before it is weighted, the filtered plane is converted to the output's
  working space, plane 1 is blended over plane 0 there, and the sum is encoded once:

  1. Decode. SDR planes: the SDR EOTF (below) gives linear Rec.709 with 1.0 = SDR white. HDR10 planes: PQ inverse to linear
     nits (Rec.2020). Premultiplied data is divided by alpha for the decode and multiplied again after, so what is filtered
     and blended is linear premultiplied colour; straight-alpha colour is premultiplied after the decode, so a transparent
     texel adds no colour of its own (found in review by @Flyander).
  2. Convert, still linear. HDR10 output: SDR to Rec.2020 (the Rec.709 to Rec.2020 matrix from the primaries; rows sum to 1)
     times the SDR white level in nits. SDR output: HDR10 to Rec.709 (the inverse matrix), divided by the SDR white level, then
     tone mapped (below). A plane already in the output's space is not converted.
  3. Blend `over + (1 - alpha) * base` (premultiplied; straight alpha was premultiplied in step 1); black outside plane 0.
  4. Encode once: PQ of the nits for an HDR10 output (argument clamped to 0..10000 nits), the inverse SDR EOTF (clipped to
     0..1) for an SDR one.

  Why. Microsoft's Advanced Color documentation says DWM composes in linear scRGB (FP16, "excellent precision for linear
  gamma blend operations"), that an app which puts SDR and HDR content into one surface must apply the SDR boost itself, in
  linear space (multiply by SdrWhiteLevel / 80), and, for a non-linear space such as HDR10, to "consider converting into
  linear gamma to apply the adjustment"; it describes no blending in PQ, and a weighted sum of PQ values is not a sum of
  light. (Some compositors blend in a gamma space to avoid banding; KDE's KWin is reported to do so since Plasma 6.2,
  https://planet.kde.org/xavers-blog-2024-11-01-hdr-and-color-management-in-kwin-part-4-nonlinear-blending, **unverified**: only
  a search snippet of that post was seen. Our intermediate image is RGBA16F, so banding is not what limits us here.) Each plane costs a decode and an encode per texel; a single SDR plane
  still comes out byte for byte (`xbox_present_test`: within the 8-bit rounding of the RGBA16F intermediate, and equal to the
  plain blit at 1:1). What changes against the first version of this section: bilinear filtering and alpha now weight light,
  not encoded values, so a 50% white overlay on black is byte 188 (piecewise sRGB) or 186 (gamma 2.2), not 128, and a
  bilinear edge between a dark and a bright texel is brighter than the same edge filtered in the encoded values. Whether
  the title's art expects the encoded-space result (UI is often authored that way on SDR consoles; inference) is unmeasured
  and is to be settled by a comparison with the title; the blend space is the one place that would then get a switch.

  *SDR EOTF* (selectable). Default: the piecewise sRGB curve (IEC 61966-2-1: `e / 12.92` up to 0.04045, else
  `((e + 0.055) / 1.055)^2.4`). Option: a pure power of 2.2 (`XboxPresentation::set_sdr_eotf`, or the environment variable
  `LEONIDA_SDR_EOTF=gamma22` (or `2.2`, any case; `srgb` or unset is the default, other values are logged once and ignored) read when the object is made; `PresentPlanes::sdr_eotf` in the presenter). The inverse of the
  same curve encodes an SDR output, so SDR in an SDR output stays the identity. Why piecewise as the default:
  **community-sourced**, not a Microsoft statement. KDE developer Xaver Hugl reports that Windows decodes SDR content with
  the piecewise sRGB curve when HDR is on (raised darks, "washed out") while SDR displays use gamma 2.2
  (https://planet.kde.org/xavers-blog-2025-03-31-on-brightness-and-calibrating-your-displays/), and the ledoge
  win11hdr-srgb-to-gamma2.2-icm project exists to make Windows use 2.2 instead
  (https://github.com/ledoge/win11hdr-srgb-to-gamma2.2-icm). What a PC player compares against is Windows, so that is the
  default; art mastered on a gamma 2.2 display looks closer to its intent with the option. The Xbox's own SDR-in-HDR decode
  is unmeasured. An sRGB-format source is encoded again with the sRGB curve first (the texture unit had decoded it), so its
  stored bytes are decoded by the chosen EOTF exactly as UNORM bytes would be.

  *SDR white level* (the nits that SDR white 1.0 maps to in an HDR10 output, and what HDR10 is divided by for an SDR one).
  `XboxPresentation::set_sdr_white_nits(n)` with n > 0 overrides everything (n = 0 goes back to automatic). Otherwise, while
  the swapchain is HDR10, Windows' SDR content brightness for the output the window is on:
  `DisplayConfigGetDeviceInfo(DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL)` on the active display path whose GDI source name
  is the window's monitor, `nits = SDRWhiteLevel / 1000 * 80` (1000 is 80 nits, the scRGB 1.0;
  https://learn.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-displayconfig_sdr_white_level), read at most once a
  second **by a background thread** (the query is not run on the present path, see below); `PresentX` reads the cached value, and a failed
  query keeps the last good one. Otherwise (an SDR swapchain, or nothing read yet) 203 nits, the HDR reference white of ITU-R BT.2408 (graphics white, 58% PQ;
  https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2408-9-2026-TOC-HTM-E.htm; the W3C CSS Color HDR module uses 203 too,
  https://www.w3.org/TR/css-color-hdr-1; the report's text itself was not parsed here). Windows' setting is a PC desktop's;
  the console's is unknown, which is why the override exists. Why a thread: the query is a `QueryDisplayConfig` (with the documented retry on
  `ERROR_INSUFFICIENT_BUFFER`) plus two `DisplayConfigGetDeviceInfo` calls; measured by `xbox_present_test` on the author's
  machine (12 us mean, 70 us worst over 50 calls on the AMD run; 47 us mean, 361 us worst on the NVIDIA run; this laptop's display setup, HDR off). That is
  small here, but the call goes into the display driver stack and its worst case with several displays or during a mode change
  is not bounded by anything measured (inference), and a stall under the present lock would feed `ImmediateThresholdPercent`
  lateness; so it is kept off the present path. What is
  measured about the value: the query returns 80.0 nits on the SDR desktop of both test GPUs' output, which is only the path
  where the layer ignores it (it uses 203 nits unless HDR10 is active); the HDR value itself has not been read on real HDR
  hardware here. 1000, 2500 and 6000 convert to 80, 200 and 480 (arithmetic only).

  *HDR10 content on an SDR output*: tone mapped, no longer clipped. After step 2, the luminance Y of the Rec.709 value
  (0.2126 R + 0.7152 G + 0.0722 B, 1.0 = SDR white) goes through a shoulder: identity up to a knee at 0.8 of SDR white
  (162 nits at 203), and above it extended Reinhard (Reinhard, Stark, Shirley, Ferwerda, "Photographic Tone Reproduction for
  Digital Images", SIGGRAPH 2002, eq. 4) applied to the excess: with `t = 1 - knee`, `u = (Y - knee) / t`,
  `w = (Lw - knee) / t`, `Yd = knee + t u (1 + u / w^2) / (1 + u)`, where `Lw = peak / SDR white` and `peak` is the content's
  peak (default 1000 nits, `set_hdr_peak_nits`; no metadata is read, so this is an assumption). The colour is scaled by
  Yd / Y. This is continuous in value and slope at the knee (slope 1 there), monotonic, and maps `Y = Lw` to exactly 1.0;
  colours with Y below the knee (SDR-range content, up to 162 nits) are not touched. Above the knee it compresses: 203 nits
  (SDR white) comes out at 0.90 of white at peak 1000 (byte 243), 300 nits at 250. Per-channel values above 1 after the
  scaling (saturated highlights) are still clipped; there is no gamut mapping and no dither. Microsoft's Advanced Color
  document lists tone mapping choices, among them a scaled Reinhard (`x / (1 + x)`) and the BT.2390 EETF, as simple options
  (https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range). The BT.2390 EETF itself is not done
  (its constants were not checked against the Recommendation here).

  *HDR metadata*: not sent. The presenter never calls `vkSetHdrMetadataEXT`. Microsoft's page for
  `IDXGISwapChain4::SetHDRMetaData` (https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_5/nf-dxgi1_5-idxgiswapchain4-sethdrmetadata,
  fetched in the research) says calling it is not recommended, that Windows does not guarantee the metadata is sent to the monitor
  and that monitors handle it inconsistently, and that an app should tone map into the range of `IDXGIOutput6::GetDesc1`
  instead. vkd3d-proton forwards `SetHDRMetaData` to `vkSetHdrMetadataEXT` only for an explicit
  `DXGI_HDR_METADATA_TYPE_HDR10` block (libs/vkd3d/swapchain.c, https://github.com/HansKristian-Work/vkd3d-proton; **partly read**:
  the first ~100k of ~175k characters of the file, which include the HDR10 conversion and the setter), so a title that sets none gets none there either.
  We invent no mastering values (Vulkan treats unknown `VkHdrMetadataEXT` fields as 0). Forwarding a title's explicit HDR10
  block is possible once a `SetHDRMetaData` slot exists.

  Sources for the PQ chain: the constants are the exact rationals of SMPTE ST 2084 / ITU-R BT.2100 (m1 = 2610/16384,
  m2 = 2523/4096 x 128, c1 = 3424/4096, c2 = 2413/128, c3 = 2392/128; https://en.wikipedia.org/wiki/Perceptual_quantizer),
  1.0 = 10,000 nits; the Rec.709 to Rec.2020 matrix is computed from the primaries (ITU-R BT.2087 specifies the same
  conversion; its PDF was not parsed). Microsoft's Advanced Color document:
  https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range.

  *Verified* by `xbox_present_test` against a CPU model written from the above (not from the shader; it decodes, converts,
  blends and encodes in double precision): point filtering between opaque SDR planes is byte exact; bilinear, alpha blending
  (premultiplied and straight, point and bilinear), the sRGB and gamma 2.2 EOTFs and the linear 50% overlay are within 1
  level of 255; the HDR10 to SDR shoulder tone map (peaks 1000 and 600 nits; the expected bytes of an eight-step ramp and of an HDR10 plane 1 over an SDR plane 0 are hand calculations, as are the 50% overlay bytes 188 and 186) and the PQ encoder (an 8-bit swapchain forced to hold
  PQ values with `force_hdr_encoding`, a test hook) are within 2 levels (GPU `pow`; the PQ codes of the source are 10-bit).
  The HDR10 swapchain itself (format choice, 10-bit
  capture, pass-through): **verified on an RTX 5080 (driver 616.56, Windows 11) with an HDR10 surface by @JavierBlanco23
  in #2069** (native ST.2084 swapchain within 1 code, GameDVR plane drop, tearing); **not verified on AMD**: the
  author's Radeon 880M and RTX 5070 Laptop (desktop without HDR enabled) list no ST.2084 surface, where the test logs
  that case as skipped and checks the 10-bit codes only where the surface has the format.
- *`ImmediateThresholdPercent`* (hypothesis from the FramePacing text, item 7 above; not measured on the console):
  a frame is due at the vblank after the origin that handed out its token. One that is presented later than that
  is late, and when the scan would not be further down the screen than the threshold percentage (late time modulo the
  interval, as a fraction of the interval) it is flipped at once, mid-scan, instead of waiting a whole interval. 0,
  negative and NaN never tear; 100 or more tears wherever the scan is; a frame on time never does. The decision is
  the pure `leonida::decide_present_sync(late_us, interval_us, threshold)` in `frame_pacer.h`. A frame behind one
  that still waits for its vblank is not torn (it would overtake it), and neither is an unpaced present. The tear
  is shown in `GetFrameStatisticsX` (the display entry's `PercentScanned` is set and its time is the present time)
  and in `XboxPresentation::immediate_presents()`. In Vulkan: with `VK_KHR_swapchain_maintenance1` (else the EXT
  original; the structures are the same and the tests' GPUs list both; the instance gets `VK_KHR_get_surface_capabilities2`
  and the matching `*_surface_maintenance1`) a vsync swapchain is created with `VkSwapchainPresentModesCreateInfoKHR`
  listing FIFO plus IMMEDIATE (FIFO_RELAXED when the surface does not list IMMEDIATE as compatible with FIFO,
  checked with `VkSurfacePresentModeCompatibilityKHR`), and a present that tears chains `VkSwapchainPresentModeInfoKHR`
  with that mode. The extension and its feature are enabled only where the GPU has them. Without them, or without
  either mode, the threshold is ignored and logged once, and a late frame waits for its vblank as before.
  `LEONIDA_PRESENT_NO_TEAR=1` turns it off. Caveats: FIFO_RELAXED tears only when the queue is empty and a vblank was
  missed (close to the semantics, but the driver decides), and the lateness is measured on the pacer's clock (a
  timer, not the display's real scan), so on a real display the tear position is whatever the driver does at that
  moment. What is checked is that the immediate mode is chosen and presented with 0 VUIDs in our runs with `LEONIDA_VK_VALIDATION=1`; no scan-out
  or tearing was observed.
- *Tokens*: what the console does with a null, repeated or out-of-order token is **unmeasured**, and a title that
  presents a loading screen with `Token = 0`, presents again with one token, or presents from a render thread out of
  order must not freeze. So those are presented anyway and logged once per kind; only a token that was never handed
  out gives `E_INVALIDARG` (nothing presented). Skipping a token is fine.
- *`SuspendX`/`ResumeX`*: suspend drops the interval, the origin event and the frames in flight, wakes blocked waits
  with `ERROR_INVALID_STATE` and refuses presents (a `PresentX` waiting for a swapchain image polls the suspended
  state every 50 ms and gives up, so `SuspendX` never waits on it); resume allows presents again, but waits fail until
  `SetFrameIntervalX` and `ScheduleFrameEventX` are called again, as the samples do. Tokens keep counting. Calling
  `SetFrameIntervalX` again outside suspend keeps the origin event armed and re-bases the vblank grid on that moment.
- *Not done*: the BT.2390 tone map and gamut mapping for HDR10 on an SDR swapchain (a Reinhard shoulder only), scRGB, limited-range
  and YCbCr colour spaces, `ResourceCount` above 1 (refused with `DXGI_ERROR_UNSUPPORTED` before either path, logged once, since only resource 0 would be read; found in review by @Flyander), extended descriptors (`pExtendedDescs`) and `ViewCount`, capturing a
  second (GameDVR) output, resolving multisampled planes, a mutable-format swapchain (an sRGB/UNORM
  alternation recreates the swapchain; `VK_KHR_swapchain_mutable_format` is not enabled on the device), a host-vblank clock (the timer drifts from the
  display; with FIFO presents the swapchain holds the queue to the real refresh anyway), render statistics,
  `PercentScanned` of frames shown at a vblank (a scan position is never measured), and the vtable wiring. The
  public slots are the device's `SetFrameIntervalX`/`ScheduleFrameEventX`/`WaitFrameEventX`/`GetFrameStatisticsX`
  at 78-81 and `PresentX` at queue 30 (older runtime layout) or 32 (newer); `SuspendX`/`ResumeX` have no public
  slot. Nothing here is wired to those slots yet, and the game's own runtime is not observed.
- *Tests*: `frame_pacer_test` (fake clock: interval, offset, late waits, timeouts, latency blocking, tokens,
  statistics, suspend/resume, a real 10 ms timer; no GPU) and `xbox_present_test` (the Vulkan device, reading the
  swapchain image back: blank, solid, two colours, source rectangles, RGBA/BGRA/sRGB sources, 2 planes, plane composition
  (placement, point/bilinear, alpha overlay, HDR10 and PQ conversions, refusals; against a CPU model), token errors,
  statistics, suspend/resume, `ImmediateThresholdPercent` with a late frame at several thresholds (immediate mode
  chosen, or not for 0, below the lateness, on time) and with `LEONIDA_PRESENT_NO_TEAR=1`, 30 frames at 60 Hz in real
  time; skipped, exit 77, without a Vulkan device). `frame_pacer_test` also sweeps `decide_present_sync`.

### 4.6 Queries

Implemented (`QueryHeap` in `runtime/shims/d3d12_vulkan_device.cpp`, the query commands in `command_list.h` and
`vulkan_replay.h`), checked on one GPU by `d3d12_vulkan_device_test` under the validation layer:

- **Heaps.** One `VkQueryPool` per `ID3D12QueryHeap`, registered with the `VulkanContext` by its pointer, as
  resources are. `OCCLUSION` heaps are `VK_QUERY_TYPE_OCCLUSION` pools and hold both `OCCLUSION` and
  `BINARY_OCCLUSION` queries, as in D3D12. `TIMESTAMP` and `COPY_QUEUE_TIMESTAMP` heaps are
  `VK_QUERY_TYPE_TIMESTAMP` pools; they are refused when the queue family's `timestampValidBits` is 0.
  `PIPELINE_STATISTICS` heaps count all eleven Vulkan statistics (`0x7ff`), whose result order is the field order
  of `D3D12_QUERY_DATA_PIPELINE_STATISTICS` (IA vertices, IA primitives, VS, GS invocations, GS primitives,
  clipper invocations and primitives, PS, HS, DS, CS), so one 88-byte result per query is copied as is. They need
  `pipelineStatisticsQuery`. `VIDEO_DECODE_STATISTICS` and `PIPELINE_STATISTICS1` (mesh and amplification counts)
  are logged and refused with `E_NOTIMPL`. Invalid descriptions are answered as the host runtime answers them
  (`compare_invalid_arguments`, "query heap:", RTX 5070 / 621.14): `E_INVALIDARG` for no queries, a node other than
  node 0, or a type past `PIPELINE_STATISTICS1` (7), checked before the output; a valid description with a null
  output is then `E_UNEXPECTED`, not `S_FALSE`.
- **Stream-output statistics** (D3D-302). The layer has no stream output: a graphics PSO with `StreamOutput`
  entries is refused and `SOSetTargets` is not implemented, so no primitive reaches a stream and every
  `SO_STATISTICS_STREAM0..3` query counts 0 written and 0 needed. An `SO_STATISTICS` heap is therefore made without
  a `VkQueryPool`: `BeginQuery`/`EndQuery` record nothing in Vulkan (D3D12's call rules are still kept, and the
  heap's queries track ended or not as the others do), they overlap no other query, and `ResolveQueryData` writes
  zeros (`vkCmdFillBuffer`, 16 bytes per query: `NumPrimitivesWritten`, `PrimitivesStorageNeeded`), with no
  statistics split since nothing is dispatched. When stream output lands, the heap becomes a
  `VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT` pool begun with `vkCmdBeginQueryIndexedEXT` on the stream, whose two
  results are in the same order.
- **Recording.** The list keeps D3D12's rules for the calls: timestamps only have `EndQuery`; every other query is
  begun and ended in the same list with the same type, is not begun again while it runs, and is not resolved while
  it runs; resolve offsets are multiples of 8. A list closed with a query still running reports `E_INVALIDARG` and
  is refused at execution. Heap types, index ranges and the destination buffer are checked at
  `ExecuteCommandLists`, when the heap and buffer are known (`ReplayResult::invalid_query`).
- **Replay.** D3D12 queries need no reset; a Vulkan query does. Each `BeginQuery` and each timestamp `EndQuery` is
  preceded by `vkCmdResetQueryPool` of that one query. Nothing in the layer runs inside a render pass yet, so the
  reset is legal where it stands; a draw path that opens a render pass around queries must reset them before it.
  `OCCLUSION` begins with `VK_QUERY_CONTROL_PRECISE_BIT` when `occlusionQueryPrecise` is enabled. A timestamp is
  `vkCmdWriteTimestamp` at `BOTTOM_OF_PIPE` (every earlier command finished). `ResolveQueryData` is
  `vkCmdCopyQueryPoolResults` with `VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT`. A query keeps its result
  across executions until it is begun or written again, as in D3D12.
- **Binary occlusion.** The queue resolves to a scratch buffer in chunks, reduces each 64-bit value to 0 or 1
  with a compute dispatch, and copies it into the destination. This works with readback buffers, which need
  only transfer-destination usage. The helper runs outside conditional rendering so a predicate cannot suppress
  the resolve. `vk_binary_occlusion` checks more than one scratch chunk, gaps, suballocated destinations,
  consecutive submissions and an ordinary occlusion count on a real Vulkan device.
- **Queries never ended.** D3D12 leaves their result undefined; a Vulkan copy with `WAIT_BIT` would wait for them
  forever. Each heap records on the host which queries were ended, in submission order, and a resolve writes zeros
  (`vkCmdFillBuffer`) for the ones that were not.
- **Overlapping occlusion queries and statistics (D3D-252).** On a graphics queue with compute, the replay opens
  a new physical query whenever the set of active queries changes. Ordinary and binary occlusion share the one
  physical pool (both are `VK_QUERY_TYPE_OCCLUSION`, so one physical query is open at a time, and the shared
  intersection interval is counted by both logical queries); pipeline-statistics queries split the same way into
  the statistics pool, where every non-timestamp resolve and every helper dispatch while a query runs also closes
  and reopens the open segment (D3D-162), for every query open at that point. Each logical query keeps a
  contiguous range in the submission's pool. Resolving it sums the results on the GPU, then copies them to the
  destination; a binary resolve normalizes the summed counts to 0 or 1 (`sum_binary_occlusion`, the sum followed
  by the existing normalize). Both occlusion compute helpers reuse one scratch buffer per queue. Logical results
  and in-flight submissions retain the pools independently, so later resolves and overwrites can be recorded
  before the earlier commands complete. `vk_overlapping_occlusion` checks crossing and nested scopes, more than
  one scratch chunk, alternating binary and ordinary resolves under a predicate, gated submissions, crossing
  statistics queries against three unsplit references word by word, and an ordinary query around a binary one, on
  a GPU; `replay_query_predication` pins the segment layouts, the helper dispatches inside an overlap and the
  binary sum-and-normalize without a GPU. On a device without `pipelineStatisticsQuery` (MoltenVK) the test
  leaves the statistics lists out; `vk_overlapping_occlusion_no_statistics` runs it that way (`--no-statistics`)
  on a device that has the feature, so that path is also run where no such device is at hand.
- **Helper dispatches during a pipeline-statistics query** (D3D-162). The helper dispatches above (binary
  normalisation, occlusion sums), statistics sums, `SetPredication` captures, predicated buffer copies and D24/RGB96 footprint
  conversions would count in the application's `CSInvocations` if they ran inside its query. Such a query is split.
  Its counters go to physical `PIPELINE_STATISTICS` segments in a pool of the submission's own; a segment ends
  before each helper and a new one begins after it. Non-query helpers and resolves share the segment numbering.
  Its result is the sum of its segments, added per counter by a compute dispatch (`statistics_sum.comp`) on the
  same scratch buffer, which is itself kept out of any statistics query still running.
  Whether a resolve dispatches is only known once the list is planned against the heaps, so every such resolve
  splits the query. `vk_binary_occlusion` checks that a binary resolve and a statistics sum inside statistics queries
  leave every counter at 0. `vk_overlapping_occlusion` puts a native draw in each segment and checks the split sums
  against unsplit queries with the same draws, word by word. `vk_predicated_copy` checks captures, rebindings and
  predicated copies; `depth24_convert` checks both conversion directions for D24 and RGB96. Their byte-exact
  results still match and all 11 statistics counters are zero. Without a statistics pool (single-list `replay()` or
  a queue without the summation helper), these three commands are refused before helper scratch allocation or
  recording. The resolve fallback is unchanged: only resolves needing dispatches are refused there.
- **Mesh draws during a pipeline-statistics query** (D3D-291). D3D12 allows `DispatchMesh` inside a statistics query
  and reads its input-assembly, vertex, geometry and tessellation fields as 0. Vulkan forbids `vkCmdDrawMeshTasksEXT`
  while a query with those counters is active (`VUID-vkCmdDrawMeshTasksEXT-pipelineStatistics-07076`), and the heap's
  pool has all eleven. So a statistics query that holds a mesh draw is split as above, and a split before a draw stays
  outside its rendering scope. A segment's first draw decides its pool: the submission's full pool, or a second pool
  with only clipping invocations and primitives, fragment and compute invocations. A later draw of the other kind ends
  the segment and begins one of its own kind; dispatches and segments without draws use the full pool, and
  `ExecuteIndirect` draws count as vertex draws. The sum copies a mesh segment's four counters into fields 5 to 8 of
  its record, and `statistics_sum.comp` adds them as `CInvocations`, `CPrimitives`, `PSInvocations` and
  `CSInvocations`, so the other fields get nothing from it. A query that holds only mesh draws is one mesh segment,
  resolved through the same sum. Without the compute helper there is no segment pool and such a list is refused
  (`unsupported`) rather than run unsplit; so is one on the single-list `replay()` (D3D-399), whose
  `create_segment_pool` gives no mesh pool. Overlapping statistics queries (D3D-252) split the same way inside their
  shared segments: a draw of the other kind than the open segment's opens a new one, so every query that segment
  belongs to sums both kinds. `replay_query_predication_test` (cases 15 and 16) checks the split points, the segment
  kinds and the resolve plan on the CPU. `d3d12_mesh_test` resolves a query around a vertex, two mesh and a vertex
  draw, one around a mesh draw and one around the two vertex draws: the vertex fields count only the two triangles,
  and the mixed query has more clipping and pixel-shader invocations than the vertex-only one. On the RX 9060 XT
  (RADV), the clipping counts add up exactly (2 + 10 = 12).
- **Timestamps.** `GetTimestampFrequency` is `1e9 / timestampPeriod` (1 GHz on the test GPU). Every D3D12 queue
  shares one `VkQueue` for now (question 12), so they share one frequency.
- **Timestamp resolves on MoltenVK.** On Apple GPUs, MoltenVK delays every `vkCmdWriteTimestamp` to the end of
  the current Metal encoder, where it is sampled by a blit pass of its own (`encodeTimestampStageCounterSamples`,
  because Apple GPUs sample only at stage boundaries). `vkCmdCopyQueryPoolResults` resolves the counters with a
  blit that nothing orders after that pass. The copy then reads the previous value, or 0 for a timestamp never
  written before. This happens when the copy is in the timestamp's open blit encoder, in a later encoder, in a later
  command buffer or in the next `vkQueueSubmit`. The spec needs no barrier here: `WAIT_BIT` waits for the query.
  `vkGetQueryPoolResults` on the host is right. A minimal Vulkan program (two timestamps around a
  `vkCmdCopyBuffer`, resolved with `64_BIT | WAIT_BIT`, compared with `vkGetQueryPoolResults`, 200 runs per case
  on an M3 Max, macOS 27.3) got the right values 0 to 13 times out of 200 for each of these placements with
  MoltenVK 1.4.3 (`main` 5189f8de). With a pipeline barrier or a fill before the copy it was 9 and 3 out of 200.
  MoltenVK 1.4.2 got 0 to 43 out of 200, measured while other GPU work ran. Every case was right 200/200 on
  KosmicKrisp (Mesa 26.3.1) and llvmpipe. Two things order the copy on MoltenVK, right in every run (200 or 500
  runs, both versions). One is `vkCmdSetEvent` + `vkCmdWaitEvents` on a fresh event before the copy, which MoltenVK
  encodes as a Metal event signal and wait on the command buffer. The other, across submissions, is a timeline
  semaphore wait for the earlier submission. So where `driverID` is `VK_DRIVER_ID_MOLTENVK` (Wine reports the
  host's driver), the queue sets and waits for an event before each timestamp resolve that copies, and a
  submission with such a resolve waits on the queue's own timeline for the submission before it. The values are
  still the GPU's own timestamps, nothing is emulated. MoltenVK picks at submit time, from the event's host state,
  whether a set or wait does anything. So each resolve gets an event of its own, which goes back to the queue's pool
  after a host `vkResetEvent` once the submission completes. `d3d12_vulkan_device` checks the same list (timestamp,
  clear, copy, timestamp, resolve) and a resolve in the next execution with no wait between them. Under Wine
  (CrossOver 26.3, MoltenVK 1.4.3) the first case failed every run before this change (`t1` read 0).

The single-list `replay()` helper replays overlapping occlusion kinds and statistics queries too (D3D-399) when
the caller passes a `keep` vector and sets `ReplayFunctions::create_segment_pool` with the four sum helpers: it
makes one occlusion and one pipeline-statistics segment pool sized to the list, plans them as the batched path does,
and hands them to `keep`, which the caller holds until the command buffer has completed. Without them such a list is
still refused with `ReplayResult::unsupported`. `replay_query_predication_test` case 14 checks both, on case 13's
statistics list and case 10's occlusion pair (no GPU).

Not implemented yet:

- **Overlapping queries in the single-list `replay()` helper.** It has no segment pool, so a list whose occlusion
  kinds or statistics queries overlap each other is refused there with `ReplayResult::unsupported`; the batched
  path above replays them.
- Mesh-shader statistics (`PIPELINE_STATISTICS1`, D3D-132), and stream-output statistics that count real
  primitives (they wait for stream output itself). (`GetClockCalibration` is implemented: see section 4.4,
  "Fences, queues and frames in flight".)
- Stream-output and mesh-shader statistics. (`GetClockCalibration` is implemented: see section 4.4, "Fences,
  queues and frames in flight".)
- Xbox-specific query heap types (if any) are **Open**; unknown types are logged.

### 4.7 Indirect commands and command signatures

- `D3D12_COMMAND_SIGNATURE_DESC` describes the argument stride and a list of argument types. Simple draw,
  draw-indexed, dispatch and mesh-dispatch signatures map to `vkCmdDraw*Indirect`, `vkCmdDispatchIndirect` and
  `vkCmdDrawMeshTasksIndirectEXT`; use the `*Count` variants (`VK_KHR_draw_indirect_count`, core 1.2) when the
  `ExecuteIndirect` call supplies a count buffer.
- Signatures that change root constants, vertex/index buffers or descriptors per argument cannot be expressed by
  one Vulkan indirect call. Use a compute expansion pass that writes a Vulkan-shaped indirect buffer, or
  `VK_NV_device_generated_commands` on NVIDIA. Start with the simple signatures; the trace will show which
  signatures the game creates.
- Predication (`SetPredication`) maps to `VK_EXT_conditional_rendering` (D3D-18). Done: `GraphicsCommandList::
  SetPredication` records a `Predication` command (`runtime/src/command_list.h`; offset 8-byte aligned, op
  `EQUAL_ZERO` or `NOT_EQUAL_ZERO`, a null buffer turns it off). The replay (`runtime/src/vulkan_replay.h`) begins
  conditional rendering on the captured condition, plain for `EQUAL_ZERO` (Vulkan discards when the word is 0) and
  `VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT` for `NOT_EQUAL_ZERO`, ends it before a new predicate (it cannot
  nest) and at the end of the list (predication does not carry over to the next list). While a predicate is set,
  whole-view clears are replayed as one rectangle with `vkCmdClearAttachments`, because `vkCmdClearColorImage` and
  `vkCmdClearDepthStencilImage` ignore conditional rendering. The standalone device enables the extension and its
  `conditionalRendering` feature when the GPU has them, and then creates buffers with
  `VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT`; without it a predicated list is rejected (`unsupported`).
  `D3D12_RESOURCE_STATE_PREDICATION` is the same bit as `INDIRECT_ARGUMENT` (and `SYNC_PREDICATION` /
  `ACCESS_PREDICATION` the same as `EXECUTE_INDIRECT` / `INDIRECT_ARGUMENT`), so with the extension enabled
  barriers into and out of that state also cover `VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT` at
  `VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT`, where the predicate is read: without it a `ResolveQueryData`
  into the predicate had no dependency on the read (`vulkan_access`, `vulkan_barrier_scope`).
  Checked by part E of `vk_command_replay_test` on an NVIDIA GeForce RTX 3070, and end to end by
  `d3d12_vulkan_device_test` (an occlusion query resolved into a predicate, through legacy and enhanced barriers)
  on an NVIDIA GeForce RTX 5070.
  The 64-bit value is captured on the GPU at each `SetPredication`: an internal eight-byte copy and a
    one-invocation compute pass OR the two dwords into a private 32-bit condition. Later writes to the source do
    not change that snapshot. A null buffer disables predication; rebinding ends the previous conditional scope
    before the capture, so the internal dispatch is never predicated. Bounds checks require all eight bytes.
    Capture arenas and descriptors remain alive until their command buffer's timeline ticket completes. No
    CPU readback or per-predicate wait is used. This helper requires a compute-capable configured queue family;
    unsupported queues fail preparation instead of falling back to the low word. The synthetic GLSL source and
    embedded SPIR-V are shipped together; normal builds do not need a shader compiler.
    Capturing a new predicate inside an active pipeline-statistics query splits the query around the capture
    dispatch (D3D-162, section 4.6), keeping its invocation out of the application's result. Without a statistics
    segment pool, preparation still refuses the capture before allocating its scratch.
    Integration notes for #291 and D3D-27: the capture helper binds its own compute pipeline, descriptor set 0
    and push constants. The compute path must restore those bindings afterward, or bind its application state
    before each dispatch. Copies and compute dispatches cannot execute inside dynamic rendering, so render-pass
    integration must place the capture outside the rendering scope while preserving capture at `SetPredication`.
    These are requirements for those future integrations, not compute-dispatch or render-pass support added here.
  `CopyBufferRegion` under a predicate instead uses a byte-granular compute dispatch inside the captured
  conditional scope; both operations, unaligned offsets and lengths, and the full 64-bit predicate are covered.
  Zero-byte copies are no-ops, as checked against native WARP. Unpredicated copies retain `vkCmdCopyBuffer`.
  The compute helper requires enabled Vulkan 1.2 `storageBuffer8BitAccess` (not `shaderInt8`) and a compute queue;
  without it, preparation rejects the predicated buffer copy rather than executing it unconditionally.
  Descriptor windows respect storage-buffer alignment/range and dispatch limits, including arena offsets.
  Copy descriptors are owned by command-buffer slots until the submission timeline completes. Memory dependencies order prior writes,
  subsequent reads and successive windows. Direct and compute queues support the helper; copy queues reject
  active predication. Preparation is serialized under the tracker lock, preserving resource -> tracker -> submit
  lock order and the copy-then-commit tracker. Readback writes accumulate COMPUTE_SHADER for the helper and COPY
  for ordinary copies; mixed lists accumulate both before main's queue-specific host-stage filtering.
  This helper also binds compute pipeline/set 0/push constants, so the
  application-state restoration requirement above applies. A predicated copy inside an active pipeline-statistics
  query splits that query around the helper dispatches (D3D-162); without a statistics segment pool it is refused
  before copy scratch allocation.
  Synthetic checks: `predicated_copy_test` covers preparation and rejection paths without a GPU;
  `d3d12_predicated_copy_test` compares 180 byte-exact cases against WARP and an independent literal oracle,
  plus 16 query-to-copy cases: real empty ordinary/binary occlusion queries resolve zero into a default-heap
  predicate at offsets 0/16, both predicate operations govern copies, and executing overwrite copies replace zero
  with a high-word-only value. Four paired controls verify that recapturing the same address changes exactly one
  13-byte destination region; without recapture, later copies keep using the captured zero. Predicate-buffer guards
  and all untouched destination bytes are checked too. These query cases do not cover nonzero visible-draw counts;
  `vk_predicated_copy_test` forces small windows and dispatches, arena prefixes, dependent copies, offset rebinding,
  reset, 32 gated plus 32 reused submissions, failed-submit cleanup and host stages on all three queue kinds.
  Binary and overlapping occlusion fixtures also verify that query results resolve under a false predicate while
  a dedicated copy destination remains unchanged; disabling predication copies into a separate region.
  The embedded shader is generated from `predicated_copy.comp`
  with glslang 16.6.0 (`-V --target-env vulkan1.2 -Os`) and validated for Vulkan 1.2.

  Open:
  - Texture copies, `CopyResource` and resolves. Vulkan's conditional rendering does not cover transfer
    commands, so these still run unconditionally and `PreparedList::unpredicated_copies` counts them.

  - Draws, dispatches and `ExecuteIndirect` are covered by conditional rendering as soon as they are replayed
    (D3D-3, D3D-4, D3D-17); `ClearUnorderedAccessView*` and `ResolveSubresource` need the same check when added.

### 4.8 Ray tracing and XDXR

This is the largest Xbox-specific area, and the shipped `XboxBvhBuilder.dll` gives concrete evidence. The DLL's
own names use the `XDXR` prefix for Xbox's raytracing path:

- Its constructor takes `ID3D12Device11*`; its build/copy/rehydrate methods take
  `ID3D12GraphicsCommandList4/6/8*`.
- Prior art measured its constructor creating 53 compute pipelines through device slot 62, called with the
  compute-PSO shape (a plain `D3D12_COMPUTE_PIPELINE_STATE_DESC`, two zero words, iid, out); on the PC interface used
  there slot 62 is `CreateStateObject` (a different arity), so the proxy overrides it. The layer must implement
  slot 62 as compute-PSO creation, not as a state object.
- It exposes standard `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC` builds, batched Xbox builds
  (`D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2`), XDXR stream descriptions
  (`D3D12XBOX_XDXR_STREAM_DESC`), post-build offline statistics, and a `DEHYDRATE` build flag. The Xbox types are
  not public. **Measured** from the DLL's own reads ([xdxr.md](xdxr.md), `runtime/src/xdxr_descs.h`): `BATCHED_DESC2`
  is the public build desc, the postbuild count and pointer, and a 16-byte `XDXR_STREAM_DESC` (`0x50` bytes); the
  stream desc is a size and a pointer to 16-byte typed records. The DLL rejects the offline-statistics postbuild type
  (`0x10`) for every build, so that layout is not in the package.
- The translator's `image_bvh_intersect_ray` / `image_bvh64_intersect_ray` already emulate the AMD hardware BVH
  node format: node id = address >> 3 | type, box16/box32/triangle nodes, 256-byte base units. That is the format
  the runtime must produce for the game's compute traversal.

**The builder's own shaders are console FSSM, not Xbox DXIL.** Each of the 52 containers carries an LLVM
bitcode blob and an `XDAT`/`XBCH` record; the PC experiment below compiled the bitcode, but the console runtime
executes the `XBCH` record. Measured from that record: 52/52 unpack to `FSSM`, 51 translate and all 51 pass
`spirv-val` with the same translator the game's 26,503 shaders use; the one left (`Box16Rehydration`) needs
`buffer_store_short` (#1864) ([embedded_shaders.md](embedded_shaders.md)). The
raw 64-bit addresses are the corpus's existing raw-address path, not a separate lowering. Open: whether the game
uses the builder at all (RT-5) and whether the translated programs match a reference on a GPU.

The design choice follows from that: **build AMD-format BVH data, not a `VkAccelerationStructureKHR`, unless the
shaders are changed to `OpTraceRayKHR`.** The game's 64 RT compute shaders traverse the nodes themselves; a
Vulkan AS would be invisible to them. Two paths:

1. If `XboxBvhBuilder` does the build with its own compute dispatches (the likely case, given it ships as a DLL
   with a device and command lists), the layer must translate and run its compute shaders. The console programs
   are `FSSM` and 51 of the 52 already pass `spirv-val`; only `Box16Rehydration` still needs `buffer_store_short` (#1864). Buffers,
   UAVs, descriptors, barriers and indirect dispatch are necessary but not sufficient. The standard
   `BuildRaytracingAccelerationStructure` calls may never reach our command list.
2. If the game calls `ID3D12GraphicsCommandList4::BuildRaytracingAccelerationStructure`, the layer must implement
   it. Produce the AMD node layout in the arena, with the same geometry/instance rules the translator assumes,
   and make `GetRaytracingAccelerationStructurePrebuildInfo` return sizes for that layout.
   The layout is the XDXR BVH4 runtime form ([bvh.md](bvh.md) §3.1). `tools/xdxr_bvh.py` encodes it on the
   CPU, with a reference traversal and a validator ([bvh.md](bvh.md) §7a). `runtime/src/xdxr_build.h` builds it on
   the GPU and is checked against that reference. The standalone device uses it for `BuildRaytracingAccelerationStructure`
   and answers `GetRaytracingAccelerationStructurePrebuildInfo` from its layout, exactly. Acceleration-structure
   buffers then keep their console VAs, the addresses the records name each other by. `LEONIDA_RT_AS=khr` keeps
   #743's `VkAccelerationStructureKHR` builds (for `RayQueryKHR`), which the game's shaders can't read. Updates refit
   the source structure instead of rebuilding it, `CopyRaytracingAccelerationStructure` clones and compacts, and the
   postbuild compacted and current sizes are written ([bvh.md](bvh.md) §7a). A bottom-level build with
   `MINIMIZE_MEMORY` gets BOX16 nodes (half boxes, about a fifth smaller; top-level builds stay BOX32), and one
   without `ALLOW_UPDATE` pairs sibling triangles that share an edge into TRI0/TRI1 leaves. Serialization and the Xbox
   #743's `VkAccelerationStructureKHR` builds (for `RayQueryKHR`), which the game's shaders can't read. A top-level
   build with `D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS` is supported there: the console VAs in the pointer list are
   translated to Vulkan device addresses in a staged copy, without changing the guest's input. The list must be in
   CPU-writable memory (an UPLOAD or write-back CUSTOM heap; WRITE_COMBINE CUSTOM is CPU-writable too); GPU-written
   DEFAULT and READBACK lists are refused as `unsupported` before submission, since their mapped bytes can still hold
   an earlier frame's pointers when the CPU prepares the build. Completed staging buffers
   are reused, so a per-frame rebuild does not allocate memory each frame. Updates refit the source structure instead of rebuilding it, `CopyRaytracingAccelerationStructure` clones and compacts, and the
   postbuild compacted and current sizes are written ([bvh.md](bvh.md) §7a). Serialization and the Xbox
   batched/streaming builds are not implemented. The
   reported `RaytracingTier` is unchanged (`NOT_SUPPORTED`). Checked by `runtime/test/d3d12_xdxr_rt_test.cpp` through
   the D3D12 API.

**Open:** which path the game uses. Resolve by logging the first call to the standard AS methods and by tracing the
builder's command-list usage. If XDXR batched/streaming builds are used, their structs and compression formats need
to be recovered from call sites (or the NDA GDK headers, if a developer has access); they cannot be guessed.

### 4.9 Xbox-specific surface

| Feature | Public evidence | Design |
|---|---|---|
| No ESRAM | Xbox Series X\|S uses unified GDDR6; ESRAM was Xbox One | Do not implement ESRAM-era APIs. The `d3d12_xs.cpp` comment about "ESRAM-era extensions" should be read as Xbox One, not Series. |
| Binning | The GDK topic index names Xbox binning modes (NDA content) | Should be a no-op if the game uses it only as an optimisation: it changes how a render pass uses on-chip memory, not the result. Log and ignore for the first boot; revisit for performance. **Open** whether the game relies on binning for correctness. |
| DCC / texture compatibility | Public release note for `D3D12XBOX_RESOURCE_FLAG_FORCE_TEXTURE_COMPATIBILITY` | Let Vulkan choose compression; fail if the game aliases/reads raw compressed data. |
| PresentX / present planes / frame events | Public GDK samples (MIT/zlib), §4.5 | Likely the only present path of a GDKX title. Presenter + vblank-timed frame events; 1-2 planes composed into the Vulkan swapchain (placement, filter, alpha overlay, HDR10 conversions). Call shapes known; slots/layouts **Open**. |
| Device slots 48/73/78/79/80 | Public HLE `Freminet`/`ScarlettX` (third-party, not Microsoft documentation) | `d3d12_xs_device.cpp` routes `CreateCommandQueueX` (48), `CreateCommandListX` (73), `SetFrameIntervalX` (78), `ScheduleFrameEventX` (79) and `WaitFrameEventX` (80) before it indexes the PC vtable. Queue and list creation use the existing create paths. The frame-event calls are a per-device token counter: `WaitFrameEventX` returns at once with a fresh non-zero token, because there is no vblank source before a presenter exists. The five indices can be moved with `LEONIDA_XS_QUEUE_X_SLOT`, `LEONIDA_XS_LIST_X_SLOT`, `LEONIDA_XS_FRAME_INTERVAL_SLOT`, `LEONIDA_XS_SCHEDULE_EVENT_SLOT` and `LEONIDA_XS_WAIT_EVENT_SLOT` (3..127; a value that names a slot the proxy already handles keeps that slot's handler); a trace that disagrees wins. |
| VRS | Public `D3D12_SHADING_RATE` / `D3D12_AXIS_SHADING_RATE` | `VK_KHR_fragment_shading_rate`; `RSSetShadingRate`/image-based rate map to `vkCmdSetFragmentShadingRateKHR`. |
| Mesh shader tier | Public `D3D12_MESH_SHADER_TIER` | `VK_EXT_mesh_shader`; the 9,282 translated mesh shaders make this a hard requirement for any content that uses them. |
| Wave MMA | Public `D3D12_WAVE_MMA_TIER` | Not used: no shader sets `WAVE_MMA` in SFI0 and none contains `v_mfma`/`v_wmma` (all 26,503, `tools/feature_scan.py`). Answer the tier query truthfully; no cooperative-matrix mapping is needed. |
| Enhanced barriers | Public Xbox Series `D3D12_BARRIER_GROUP` | `VK_KHR_synchronization2`; implement directly. |
| XDXR batched/streaming BVH | `XboxBvhBuilder.dll` exports | See 4.8. |

### 4.10 Debug markers and private data

Games and middleware call these every frame (PIX events), so they must be cheap and never fail. Implemented
(D3D-24) in `runtime/shims/d3d12_com.cpp`, `runtime/shims/d3d12_com_object.h`, `runtime/src/command_list.h`,
`runtime/src/vulkan_replay.h` and `runtime/src/vulkan_backend.h`; tested by `runtime/test/d3d12_debug_markers_test.cpp`.

- **Labels, only with `VK_EXT_debug_utils`.** It is an instance extension, so `VulkanDeviceInfo::debug_utils` says
  whether the instance enabled it; `create_vulkan_device_standalone` enables it when the loader lists it. Without
  it every marker call is a no-op (nothing logged, nothing recorded on the queue).
- **Command lists.** `SetMarker`/`BeginEvent`/`EndEvent` are recorded in the IR (`DebugMarker`) in order with the
  other commands and replayed as `vkCmdInsertDebugUtilsLabelEXT`/`vkCmdBeginDebugUtilsLabelEXT`/
  `vkCmdEndDebugUtilsLabelEXT` into the submission's command buffer.
- **Queues.** The same three methods on `ID3D12CommandQueue` call `vkQueueInsert/Begin/EndDebugUtilsLabelEXT` under
  the device's submit lock, so they fall between the submissions issued before and after them.
- **Balancing.** D3D12 does not make a list balance its events; Vulkan requires every end to have a begin.
  One `ExecuteCommandLists` is one command buffer, so a begin in one list may be ended by a later list of the
  same call. An end with nothing open is dropped, and labels still open when the last list ends are ended
  before the command buffer is ended. Queue labels are balanced per queue the same way (an unmatched end is
  dropped; open ones are ended when the queue is destroyed). Command-buffer labels never span two
  `ExecuteCommandLists` calls.
- **Labels.** The `Metadata` argument says what the data is, following WinPixEventRuntime's `pix3_win.h`:
  0 (`PIX_EVENT_UNICODE_VERSION`) a UTF-16 string, 1 (`PIX_EVENT_ANSI_VERSION`) an ANSI string, both cut at the
  terminator or at `Size` bytes and converted to UTF-8; 2 (`PIX_EVENT_PIX3BLOB_VERSION`) PIX's own encoded event,
  which is not decoded and gets the fixed label `PIX event` so the nesting is still visible (question 21). Other
  values get the same label and log once.
- **Private data, on every object.** `SetPrivateData`, `GetPrivateData`, `SetPrivateDataInterface` and `SetName`
  live in the shared `Object` base, so the device, queues, allocators, lists, fences, descriptor heaps and
  resources all have them. Every answer is the host runtime's: `d3d12_private_data_test` runs the same 672 steps
  on the device, a fence, a buffer, an allocator and a descriptor heap of both devices and compares each HRESULT,
  returned size, byte and reference count (RTX 3060, driver 610.88, no debug layer):
  - `SetPrivateDataInterface` holds a reference until the entry is replaced (by bytes or another interface),
    removed or the object is destroyed; `GetPrivateData` returns the pointer with a reference added. A null
    interface releases the one held and stays as a pointer-sized entry that reads back as null.
  - A buffer too small for the entry gets `DXGI_ERROR_MORE_DATA`, nothing copied and no reference added, and
    `*pDataSize` set to the size needed; a missing entry is `DXGI_ERROR_NOT_FOUND` with `*pDataSize` 0; a null
    `pDataSize` is `E_INVALIDARG`.
  - `SetPrivateData` with null data removes the entry whatever the size: `S_OK` when there was one, `S_FALSE` when
    there wasn't. Zero bytes with data keep an empty entry (`S_OK`, size 0).
  - `SetName` is `SetPrivateData` with `WKPDID_D3DDebugObjectNameW` (`{4cca5fd8-921f-42c8-8566-70caf2a9b741}`) and
    the name with its terminator, as D3D12 documents: `GetPrivateData` reads it back, an empty name is a 2-byte
    entry, and the narrow `WKPDID_D3DDebugObjectName` is not touched. A null name, which the host runtime faults on,
    changes nothing (`E_INVALIDARG`). Before #2533 the layer stored the name under a mistyped GUID.
  Names are not forwarded to `vkSetDebugUtilsObjectNameEXT` yet: one D3D12 object may own several Vulkan objects.

## 5. Producing the shader contract

The table below is the contract from [design.md](design.md#translator-status-and-runtime-contract) and where the
runtime gets each piece. The shader side is already proven by `runtime/test/vk_texture_test.cpp` and the compute,
store, load, mesh, geometry and BVH tests; this layer is the other half.

| Contract | Runtime source |
|---|---|
| `s0..s31` user data | Root signature state at draw/dispatch: root constants copied, root CBV/SRV/UAV addresses and descriptor-table base addresses as 64-bit console VAs. Mapping per section 3.5; absent parameters zero. |
| `heap_base` (dwords 32–33) | The arena's `vkGetBufferDeviceAddress`. One value per command buffer; updated only if the arena is recreated. |
| `code_base` (dwords 34–35) | The console VA of the current shader's FSSM blob in the arena. Per-stage conflict: see below. |
| Descriptor hash table (set 0, binding 2) | `uint mask` + entries of `key[8], index, pad`; FNV-1a over the 8 key dwords; linear probe up to 8; miss = index 0. Implemented in `runtime/src/descriptor_table.h` (`DescriptorTable`). The layer inserts every T#/S# it sees and keeps a fallback image/sampler at index 0. |
| Bindless images (bindings 0, 3, 5, 6) | 2D, 2D-array/cube, 1D, 3D sampled-image arrays; one `VkImageView` per distinct T#. Cube textures are 6-layer 2D arrays and cube arrays 6N-layer ones, as the translator expects. |
| Samplers (binding 1) | One `VkSampler` per distinct S#; literal S# values from shader constants are pre-registered. |
| Storage images (bindings 4, 7, 8–19) | r32ui/float/sint views by dimension and numeric class; the layer binds each storage image into the float/uint/sint arrays of its dimension (the translator picks by the T# format at run time), with all three arrays as the safe default. |
| Buffer/image binding | Descriptors contain console VAs, never host BDAs. Buffers are in the arena at their VA; images are keyed by T# through the hash table. |
| Vertex fetch | PSO input layout + ISG1 signature; location = D3D input register; format from `DXGI_FORMAT`; instance step rate from the layout. The fetch subroutine is already replaced by Vulkan inputs. |
| PS system inputs | `FragCoord` (xy/z/w), `FrontFacing`, barycentrics; enable `fragmentShaderBarycentric` and `fragmentShaderPixelInterlock` when the index entry asks. |
| Mesh/geometry stages | `VK_EXT_mesh_shader`; the index's max vertices/primitives size the SPIR-V output arrays and are checked against the device's mesh properties; geometry stage for the 195 voxelize shaders; `DispatchMesh` → `vkCmdDrawMeshTasksEXT`. |
| POPS collision | The runtime puts 0 (no collision, bit 31 clear) in the POPS collision SGPR; the translator uses fragment-shader interlock for ordering. **Open:** the exact SGPR location is a translator/driver detail not named in the current contract. |
| Wave size | The PSO asks for the size the shader was translated for (`requiredSubgroupSize` 64 or 32, section 3.2 "Subgroup size") where the stage is in `requiredSubgroupSizeStages`; a shader translated with `--wave64-emulate` asks for nothing. A size the device cannot give is named in the log. |
| Code-embedded data | The whole FSSM blob is in the arena; `s_getpc_b64` becomes `code_base + next-instruction offset`. |

### 5.1 Descriptor hash table details

The exact layout the shader reads (from `tools/rdna2spirv.py`, `desc_lookup`):

```
uint32 mask;                 // slot count - 1
struct entry {               // 10 uints = 40 bytes
    uint32 key[8];           // T#: 8 dwords; S#: 4 dwords then 0xFFFFFFFF x4
    uint32 index;            // bindless index
    uint32 pad;              // unused by the shader
} entries[mask + 1];
```

- Hash: FNV-1a over the eight key dwords as dwords (not bytes): `h = 2166136261; h = (h ^ key[i]) * 16777619`.
- Probe: `slot = (h + k) & mask` for `k = 0..7`; the first entry whose eight key dwords match yields its
  `index`; no match yields 0.
- The layer must keep the table sparse enough that no live key needs more than eight probes. When all eight slots are occupied,
  the runtime tries to move one resident to the next empty slot within that resident's own probe window.
  A relocation requires an idle flush because it overwrites a live slot. Grow when this fails or when the next
  insertion would exceed 50% load. Update binding 2 after
  the rebuild, at a point where no in-flight command buffer reads the old table.
- Insert on `CreateShaderResourceView`, `CreateUnorderedAccessView`, `CreateSampler`, `CopyDescriptors` and heap
  scans. A duplicate key updates the index in place.
- Index 0 is the fallback: a 1×1 texture and a default sampler (and a valid dummy storage image for the storage
  arrays). The translator returns 0 on a miss, so index 0 must always be valid.
- The table buffer is a normal storage buffer, not part of the arena. It can be host-visible and updated directly
  (host-coherent, or flushed before submit); if the game is sensitive to PCIe traffic, stage it like any other
  upload. The shader only reads it.
- Inserts can come from several threads (view creation is often parallel). Serialise them with a mutex, or stage
  per-thread and merge at submit. The shader only reads the table, and D3D12 already forbids modifying a descriptor
  that an in-flight command list is using, so the same rule applies.

**Measured (D3D-56, `runtime/test/vk_descriptor_bench.cpp`).** Measured on an NVIDIA GeForce RTX 5080 (discrete, `0x10de:0x2c02`, subgroup 32, driver `NVIDIA 621.24`, Vulkan 1.4.357), Windows 11 Pro 26H2 (build 26300), revision `ec8ec4c9`, using 64-bit Vulkan GPU timestamps (`1.000 ns` period) against `leonida::DescriptorTable` (90% T# / 10% S#) with the exact `tools/rdna2spirv.py` `desc_probe` loop. Unless noted, dispatches run `4096 x 64` invocations x `64` iterations (`16.78M` single / `33.55M` T#+S# pair lookups); wave-uniform sweeps use `group_stride = iters` (`64`), `lane_stride = 0` so all distinct entries are visited (at 100k: `90,000` T#, `10,000` S#, `100,000` T#+S# pairs, `131,072` miss keys; `uniform_hot_tsharp` repeats `1` key and measures `2.6–4.4 us/1M` in `device_local`). This run includes `bc8e604`'s bounded slot relocation, so its 100k row is at the same `524,288`-slot geometry as the AMD RDNA3/RDNA4 tables below and directly comparable to them, and it refreshes an earlier pre-relocation RTX 5080 table. `device_local` and `host_device_local` are the median of 4 processes, `host_visible` one; every row is `timing_status=ok` with no dispatch auto-scaling:

| Entries (cap / load / bytes / mean probes) | Placement | `uniform_tsharp_hit` (us/1M, Glk/s) | `uniform_pair_ts` (us/1M, Glk/s) | `divergent_tsharp_hit` (us/1M, Glk/s) | `divergent_pair_ts` (us/1M, Glk/s) | `divergent_miss` (us/1M, Glk/s) |
|---|---|---|---|---|---|---|
| 1k (4,096 / 24.4% / 160 KiB / 1.316; 900 T#, 1k pair, 1,024 miss) | `device_local` | 3.322 (301.06) | 2.917 (342.84) | 24.008 (41.65) | 23.759 (42.09) | 96.175 (10.40) |
| 1k | `host_device_local` `host_visible` | 3.317 (301.49) 4.890 (204.48) | 2.918 (342.73) 3.583 (279.10) | 23.973 (41.71) 27.582 (36.26) | 23.732 (42.14) 26.426 (37.84) | 96.148 (10.40) 395.433 (2.53) |
| 10k (65,536 / 15.3% / 2.50 MiB / 1.310; 9k T#, 10k pair, 16,384 miss) | `device_local` | 12.842 (77.87) | 5.142 (194.49) | 38.202 (26.18) | 39.618 (25.24) | 206.654 (4.84) |
| 10k | `host_device_local` `host_visible` | 12.838 (77.89) 211.889 (4.72) | 5.166 (193.59) 16.257 (61.51) | 38.223 (26.16) 10261.465 (0.10) | 39.468 (25.34) 3334.286 (0.30) | 209.528 (4.77) 181717.455 (0.01) |
| 100k (524,288 / 19.1% / 20.00 MiB / 1.416; 90k T#, 100k pair, 131,072 miss) | `device_local` | 6.007 (166.47) | 7.159 (139.69) | 71.939 (13.90) | 47.306 (21.14) | 483.346 (2.07) |
| 100k | `host_device_local` `host_visible` | 6.025 (165.97) 213.404 (4.69) | 7.161 (139.64) 183.383 (5.45) | 71.895 (13.91) 15149.069 (0.07) | 47.277 (21.15) 16289.665 (0.06) | 482.358 (2.07) 205561.481 (0.00) |

Three consequences for the runtime and table layout (additional GPU measurements follow, including AMD RDNA3):

1. **In `DEVICE_LOCAL` VRAM, the hash probe itself is cheap at all three sizes:** wave-uniform full-table sweeps (`uniform_tsharp_hit`) run at `3.3–12.8 us / 1M`, divergent lookups at `24.0–71.9 us / 1M`, and the 8-probe miss at `96–483 us / 1M`.
2. **`BindlessSet::create()` currently allocates PCIe `HOST_VISIBLE | HOST_COHERENT` system memory (`runtime/src/vulkan_bindless.h`):** once working sets exceed cache at 10k–100k entries, `host_visible` slows `uniform_tsharp_hit` by **16.5x–35.5x**, divergent T# hits by **210x–269x**, and 8-probe misses by **425x–879x** (`181,717 us / 1M` at 10k, `205,561` at 100k). `host_device_local` (ReBAR `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`) matches `device_local` within **1.4%** on every row here (largest delta 10k `divergent_miss`, `+1.4%`). Staging binding 2 to `DEVICE_LOCAL` (or preferring `DEVICE_LOCAL | HOST_VISIBLE` when small) avoids this cliff without touching the shader contract.
3. **`bc8e604`'s bounded slot relocation keeps the table dense:** 100k settles at `524,288` slots (`19.07%` load, `20.00 MiB`, `max_probes=8`) instead of doubling to `2,097,152` on a 9-probe collision, so the large-table cost is bounded by the 8-probe window; the dominant divergent cost is the 8-probe miss (`483 us / 1M` at 100k).

Raw CSV rows (297, 9 processes): [D3D-56 RTX 5080 report](https://github.com/ItsNotRexed/Leonida/issues/313#issuecomment-7084711283).

**Measured (D3D-56, AMD RDNA3).** The same benchmark and 64-bit timestamps on an AMD Radeon RX 7900 XTX (discrete, `0x1002:0x744c`, subgroup 64, driver `AMD proprietary 26.10.1 (LLPC)`, Vulkan 1.4.359), Windows 11 build 26200, revision `2aed144`. Dispatches, iterations and reps match the NVIDIA run above, but the table code does not: this revision includes `bc8e604`'s bounded slot relocation, so at 100k it settles at `524,288` slots (`20.00 MiB`, `19.07%` load, `max_probes=8`), the same geometry as the refreshed RTX 5080 table above. The two tables now describe the same `DescriptorTable` code and their 100k rows are comparable; the remaining caveat is subgroup size (AMD `64` vs NVIDIA `32`).

| Entries (cap / load / bytes / mean probes) | Placement | `uniform_tsharp_hit` (us/1M, Glk/s) | `uniform_pair_ts` (us/1M, Glk/s) | `divergent_tsharp_hit` (us/1M, Glk/s) | `divergent_pair_ts` (us/1M, Glk/s) | `divergent_miss` (us/1M, Glk/s) |
|---|---|---|---|---|---|---|
| 1k (4,096 / 24.4% / 160 KiB / 1.316) | `device_local` | 6.220 (160.76) | 5.705 (175.27) | 15.285 (65.42) | 14.051 (71.17) | 50.461 (19.82) |
| 1k | `host_device_local` `host_visible` | 8.783 (113.85) 9.508 (105.17) | 5.360 (186.58) 5.783 (172.93) | 15.380 (65.02) 15.869 (63.02) | 13.974 (71.56) 13.465 (74.27) | 49.765 (20.09) 51.978 (19.24) |
| 10k (65,536 / 15.3% / 2.50 MiB / 1.310) | `device_local` | 5.152 (194.09) | 4.880 (204.90) | 27.998 (35.72) | 20.506 (48.77) | 146.217 (6.84) |
| 10k | `host_device_local` `host_visible` | 5.066 (197.38) 7.536 (132.69) | 4.841 (206.57) 6.236 (160.36) | 28.322 (35.31) 72.932 (13.71) | 20.641 (48.45) 54.183 (18.46) | 144.532 (6.92) 334.096 (2.99) |
| 100k (524,288 / 19.1% / 20.00 MiB / 1.416) | `device_local` | 7.618 (131.28) | 6.005 (166.54) | 53.558 (18.67) | 54.927 (18.21) | 290.279 (3.45) |
| 100k | `host_device_local` `host_visible` | 7.660 (130.54) 47.905 (20.87) | 6.040 (165.55) 29.887 (33.46) | 53.983 (18.52) 3492.770 (0.29) | 54.804 (18.25) 2240.356 (0.45) | 292.561 (3.42) 32108.853 (0.03) |

On this GPU:

1. **`device_local` is stable and cheap at every size.** Wave-uniform sweeps run `4.9–7.6 us / 1M` (the 100k `uniform_tsharp_hit` is `7.618`), divergent lookups `14.1–54.9 us / 1M`, and the 8-probe miss `50.5–290.3 us / 1M`. Repeating the 10k measurement keeps `divergent_miss` at `141–143 us / 1M` net, so this column is not noise.
2. **`host_visible` falls off a cliff at 100k, and is unreliable below it.** At 100k it is `5.0x–6.3x` slower than `device_local` on wave-uniform sweeps, `41x–65x` on divergent hits and pairs, and `111x` on 8-probe misses. At 10k the same rows varied by up to 6x between runs (`divergent_miss` net `147–845 us / 1M` over five runs), so the 10k host row above is one representative run; the uniform rows moved much less (`6.1–8.7 us / 1M`). The conclusion does not change: `device_local` is stable, `host_visible` is not.
3. **`host_device_local` (ReBAR) is a safe fallback on AMD.** It is within 1% of `device_local` at 100k and within 2% at 10k; at 1k it is within 1.5% on the divergent and miss rows, `uniform_pair_ts` is 6% faster (`5.36` vs `5.71 us / 1M`) and `uniform_tsharp_hit` is 41% slower (`8.78` vs `6.22 us / 1M`). The `DEVICE_LOCAL | HOST_VISIBLE` allocation succeeds and runs here; it is not guaranteed on every Windows driver.
4. **At the shared `524,288`/8-probe geometry, the 100k rows are comparable across the tables.** The RX 7900 XTX `divergent_miss` (`290 us / 1M`) is below the refreshed RTX 5080 (`483`), with the RX 9070 XT (RDNA4) between at `422` ([#313](https://github.com/ItsNotRexed/Leonida/issues/313)); the RX 7900 XTX is also faster on 100k divergent T# hits (`54` vs `72`) but slower on wave-uniform sweeps (`7.6` vs `6.0 us / 1M`). The subgroup size differs (AMD `64`, NVIDIA `32`), so this is a device-plus-subgroup comparison, not a pure microarchitecture one.

Raw CSV rows: [D3D-56 report](https://github.com/ItsNotRexed/Leonida/issues/313#issuecomment-7074181793), addendum with the repeatability runs: [same issue](https://github.com/ItsNotRexed/Leonida/issues/313#issuecomment-7074220711).

**Measured (D3D-56, Intel integrated + RTX 5070 Ti Laptop).** These synthetic runs use `vk_descriptor_bench` from revision `f85582e`, Windows 11 Pro Insider Preview `10.0.29704`, MSVC 19.51, Vulkan SDK 1.4.363.0, and the exact `4096 x 64` invocations x `64` iterations (`16.78M` single / `33.55M` T#+S# pair lookups, median of 25 reps). NVIDIA runs used `--sizes 1000,1000,10000,100000 --warmup 200`; the Intel run additionally duplicated 100k (`--sizes 1000,1000,10000,100000,100000 --warmup 200`). The first 1k block is excluded because its `baseline_uniform` timing differed from the duplicate (Intel device-local: `0.771 ms` vs `0.594 ms`; NVIDIA device-local: `0.0555 ms` vs `0.0765 ms`). For Intel 100k, the second block is retained after the duplicate confirmed stable timing per allocation. Retained rows were not auto-scaled. Intel's device-local and host-visible placements ran together on the integrated GPU. NVIDIA device-local and ReBAR were measured in separate processes. The RTX 5070 Ti Laptop is `0x10de:0x2f58` (driver `621.01`, Vulkan `1.4.357`); Intel Graphics is `0x8086:0x7d67` (driver `32.0.101.8132`, Vulkan `1.4.325`).

| GPU / entries (cap / load / bytes / mean probes) | Placement | `uniform_tsharp_hit` (us/1M, Glk/s) | `uniform_pair_ts` (us/1M, Glk/s) | `divergent_tsharp_hit` (us/1M, Glk/s) | `divergent_pair_ts` (us/1M, Glk/s) | `divergent_miss` (us/1M, Glk/s) |
|---|---|---|---|---|---|---|
| Intel iGPU, 1k (4,096 / 24.4% / 160 KiB / 1.316) | `device_local` | 85.669 (11.67) | 76.483 (13.07) | 468.524 (2.13) | 431.528 (2.32) | 1,735.402 (0.58) |
| Intel iGPU, 1k | `host_visible` | 85.356 (11.72) | 76.500 (13.07) | 468.750 (2.13) | 425.419 (2.35) | 1,741.669 (0.57) |
| Intel iGPU, 10k (65,536 / 15.3% / 2.50 MiB / 1.310) | `device_local` | 86.806 (11.52) | 76.642 (13.05) | 597.170 (1.67) | 507.250 (1.97) | 2,495.724 (0.40) |
| Intel iGPU, 10k | `host_visible` | 87.665 (11.41) | 76.415 (13.09) | 597.099 (1.67) | 506.874 (1.97) | 2,496.882 (0.40) |
| Intel iGPU, 100k (524,288 / 19.1% / 20.00 MiB / 1.416) | `device_local` | 101.132 (9.89) | 84.286 (11.86) | 672.021 (1.49) | 964.268 (1.04) | 5,580.289 (0.18) |
| Intel iGPU, 100k | `host_visible` | 100.269 (9.97) | 84.437 (11.84) | 668.788 (1.50) | 975.549 (1.02) | 5,581.019 (0.18) |
| RTX 5070 Ti Laptop, 1k (4,096 / 24.4% / 160 KiB / 1.316) | `device_local` | 11.028 (90.68) | 9.652 (103.60) | 48.388 (20.67) | 48.201 (20.75) | 195.543 (5.11) |
| RTX 5070 Ti Laptop, 1k | `host_device_local` (ReBAR) | 11.072 (90.32) | 9.682 (103.29) | 51.245 (19.51) | 48.427 (20.65) | 194.840 (5.13) |
| RTX 5070 Ti Laptop, 10k (65,536 / 15.3% / 2.50 MiB / 1.310) | `device_local` | 11.562 (86.49) | 10.164 (98.38) | 63.725 (15.69) | 66.871 (14.95) | 367.062 (2.72) |
| RTX 5070 Ti Laptop, 10k | `host_device_local` (ReBAR) | 11.494 (87.00) | 10.184 (98.19) | 63.858 (15.66) | 66.909 (14.95) | 297.104 (3.37) |
| RTX 5070 Ti Laptop, 100k (524,288 / 19.1% / 20.00 MiB / 1.416) | `device_local` | 11.993 (83.38) | 46.172 (21.66) | 116.020 (8.62) | 82.513 (12.12) | 788.086 (1.27) |
| RTX 5070 Ti Laptop, 100k | `host_device_local` (ReBAR) | 9.186 (108.86) | 46.162 (21.66) | 99.737 (10.03) | 141.439 (7.07) | 632.504 (1.58) |

On these devices:

1. **The Intel placements are close on the lookup workloads.** Gross dispatch times for `device_local` and `host_visible` track within 1.5%; after subtracting `baseline_uniform`, differences are within 1.7% through 10k and within 11.2% on 100k uniform hits. At 100k, the baseline itself is repeatable per allocation but differs between them (`0.730 ms` device-local vs `0.608 ms` host-visible), so the net uniform rows carry more uncertainty. Both allocations use the iGPU's shared system memory; this does not predict discrete-GPU PCIe behavior.
2. **NVIDIA ReBAR is close to device-local for uniform sweeps at 1k–10k and at 100k T#+S# pairs.** The 100k divergent and miss cases differ by workload (ReBAR is not uniformly faster or slower), so this run does not support a blanket placement ranking.
3. **The NVIDIA host-visible run is omitted.** A separate full run returned `vkWaitForFences -> -4` while measuring the 10k host-visible cases; those partial timings are not reported. A safer capped profile is needed before drawing conclusions about PCIe system-memory lookup on this laptop.

The run is synthetic and contains no game data. Raw retained rows and the corrected methodology are recorded in [the D3D-56 report on issue #313](https://github.com/ItsNotRexed/Leonida/issues/313#issuecomment-7078191651).

### 5.2 Per-stage code base

The current 144-byte block has one 64-bit code base at dwords 34–35. A graphics pipeline has at least a vertex and
a pixel shader, and each may use `s_getpc_b64` for its own constant table. One block cannot hold two different
code bases at the same offset. The pipeline layout declares the block for every stage that may read it (or for all
graphics stages), and `vkCmdPushConstants` is called with those stage bits. This is a real gap in the current
contract, not a runtime bug:

- Only 155 shaders use code-embedded data, and a graphics pipeline has at least two stages (typically VS+PS, or
  MS+PS). If only one stage in a pipeline uses it, the layer can set that stage's code base; the other stage never
  reads it. This probably covers the first boot. **ASSUMPTION.**
- If both stages use it, the layer cannot satisfy both with the current contract. The clean fix is a translator
  change: give each stage its own code-base slot (for example dwords 34–35 VS, 36–37 PS, 38–39 CS, 40–41 GS/MS,
  extending the block to 168 bytes) and have each compiled shader read its stage's slot. A 168-byte push-constant
  block is still under the 256-byte limit of common desktop GPUs, but above Vulkan's 128-byte minimum; the device
  requirement must be checked.
- A second option is a per-stage uniform/storage buffer for the code base, patched in at pipeline creation. That
  adds a descriptor binding and a load per `s_getpc`, but avoids growing push constants. Decide after measuring
  how many real PSOs have two code-base stages.

Measured per stage (all 26,503, `tools/feature_scan.py`): `s_getpc_b64` is kept to read code-embedded data in 54
VS, 93 PS, 266 MS and 263 CS (0 GS); the 147 VS/PS match the translator's own note. Both stages of a VS+PS or
MS+PS pipeline can therefore need a code base.

**Measured:** across the GPU test fleet (Issue #166), desktop NVIDIA (Turing, Ampere, Ada, Blackwell: 256 bytes), desktop AMD (RDNA2: 256 bytes) and Apple Silicon (MoltenVK: 4096 bytes) all satisfy the 144-byte contract natively (`maxPushConstantsSize >= 144`). Intel Arc is not uniform: the earlier fleet entries report the 128-byte specification minimum (`maxPushConstantsSize == 128`) for Arc, but an Arc B580 (Battlemage) on Windows 11, driver 32.0.101.9175, reports 256 bytes (#166, @rtrexmass), so there D3D-14's fallback never engages. The 128 figure may be specific to another OS or driver (Mesa ANV on Linux is the guess; no Arc run on Linux that records the OS is cited here, and an ANV result would settle it). So does an integrated AMD RDNA2 GPU, a Radeon 680M on the Windows driver 22.40.84.06, so the minimum is not Intel-only. That figure follows the driver as well: a Radeon 610M reports 128 on 24.10.16.14 and 256 on 26.11.2 (#166). 128 bytes cannot fit the 16-byte tail (dwords 32–35: heap base and code base) inside push constants. The runtime detects `maxPushConstantsSize < 144` at device creation (`d3d12_vulkan_device.cpp`) and flags `push_constants_need_fallback` for the uniform-buffer fallback.
**Open:** the exact count of such PSOs (the pairing is in the executable), and the minimum `maxPushConstantsSize` across target GPUs. If a GPU only
supports 128 bytes, the current 144-byte contract already fails; the layer needs a fallback (uniform buffer for the
tail) regardless.

At run time, `CreateGraphicsPipelineState` refuses a pair where both stages read their code base, with a log, so
the first boot measures the pair count. A device whose `maxPushConstantsSize` is below 144 has every PSO refused,
with the limit in the log, until the uniform-buffer tail exists.

### 5.3 Descriptor T#/S# parsing

The runtime needs a C++ decoder for the structures the shaders read. The fields it needs:

- **T# (image SRV):** base console VA, format (GFX10 format code), width/height/depth, mip levels, array layers,
  sample count, destination swizzle, dimension. The format table is `tools/gfx10_formats.json` (Mesa's
  `gfx10-rsrc.json`), the same one the translator uses.
- **S# (sampler):** filter modes, address modes (U/V/W), border color, min/max LOD, LOD bias, compare function,
  anisotropy, coordinate normalization.
- **V# (buffer):** base console VA, stride, record count, format, destination swizzle. Buffer descriptors are
  dereferenced directly by the shader; the layer only needs to validate the VA and the format for typed loads.

If the game calls `CreateShaderResourceView`/`CreateSampler`, the layer can encode these structures itself from
the D3D12 descriptors. If the game writes them, the layer decodes them. Both paths meet in the same parser. The
GPU texture test already proves the shader-side contract on a synthetic T#/S#; the runtime parser is new code.

## 6. Boot milestones

Each milestone is a capability gate with a synthetic test; the game is the integration test once it is available.
M0–M2 do not need the executable.

| # | Scope | What it proves | Done when |
|---|---|---|---|
| **M0** | `D3D12XboxCreateDevice` and the measured device overrides (heap start handles in RAX), one direct queue, allocator/list, fence, fake DXGI factory/adapter/output, `ALLOW_DISPLAY` back buffers, frame events + `PresentX` (DXGI swapchain only if the game creates one), clear + present | COM interception, the known Xbox device entry point and ABI overrides, presentation (`PresentX` + frame events, or swapchain), frame loop, clean shutdown | A window shows a stable clear colour for 60 frames, no Vulkan validation errors, every D3D12 call traced. **Synthetic part met** by `d3d12_present_test`: 60 frames through the proxy on Vulkan, validation clean, colour checked in the presented images and on screen where GDI can read it. Still open: the game's present front end (frame events + `PresentX`, or a DXGI swapchain) and the per-call trace |
| **M1** | Root signature with root constants (mask the measured `0x03000000` flag byte on a copy), one translated VS+PS, PSO, vertex buffer in the arena, draw, push constants `s0..s31` + heap base | PSO translation, shader hash lookup, vertex input, root-constant packing, heap base, present | The triangle matches a CPU/GPU reference; the existing texture-test style comparison passes with error 0. **Met with stand-in shaders** by `d3d12_draw_test`: every pixel of two indexed draws and a culled triangle matches, through the proxy, validation clean. With translated game shaders most of them declare bindings, which M2 (pipelines bind set 0) lets through. **M1.5 (D3D-53) met with translator output** by `d3d12_translated_draw_test`: see below |
| **M1.5** | The same, with real translator output: hand-written RDNA2 VS+PS, PSO found by FSSM hash, vertex fetch through a V# loaded from the user-data table, a root constant, a table in the arena, the code base | The push-constant, vertex-fetch and code-base contracts on translated code | Met by `d3d12_translated_draw_test`: three draws, every pixel off the triangle's edge exact in a float target, validation clean on the AMD Radeon 880M and the RTX 5070 Laptop. An upload buffer behind a V# is not supported (section 3.3) |
| **M2** | Descriptor heap (heap slots 9/10 return in RAX), `CreateShaderResourceView`, sampler, root descriptor table, hash-table insert, bindless sampling | The full D3D12 descriptor path and the shader's T#/S# contract | A textured quad matches the synthetic 8×8 gradient or a known texture with error 0; the hash table has the expected entries. **Synthetic part met** by `d3d12_m2_textured_test`: through the proxy, D3D12 API only, a shader-visible CBV/SRV/UAV heap and a sampler heap (start handles from slots 9/10), two 8×8 textures uploaded with `Map` + `CopyTextureRegion`, `CreateShaderResourceView` and a POINT `CreateSampler`, a root signature of two descriptor tables, `SetDescriptorHeaps`, `SetGraphicsRootDescriptorTable` and `DrawIndexedInstanced` of a quad whose PS is translator output (`ps_m2.asm`: `s_load` of the T# and the S# from the tables, `image_sample`). Every one of the 4096 pixels matches the CPU-computed gradient (left half) and its point reflection (right half, second table) with error 0, and the hash table holds a distinct non-zero index for each of the two T#s and the S# and none for bytes nothing created; validation clean on the AMD Radeon 880M and the RTX 5070 Laptop. The layer change it needed: pipelines whose shader declares set-0 bindings (graphics and compute) were refused; their layout now holds the device's bindless set 0 and replay binds it at every draw and dispatch. Still open: shaders of the game itself (direct heap indexing, unbounded tables, static samplers, descriptors the game writes itself), no host D3D12/WARP byte comparison (the reference is computed on the CPU). Formats, mips and arrays through a PSO: `d3d12_m2_formats_test` samples an 8×8 BC1 and an 8×8 BC7 texture with 4 mips each and an R8G8B8A8 2D array of 3 slices through the same path (every subresource uploaded at its `GetCopyableFootprints` layout, POINT samplers with `MinLOD = MaxLOD` pinning each level, a view from mip 1, two array views read by `ps_m2_array.asm`); all 12 cases match the CPU with error 0, validation clean on the RTX 4070 Ti SUPER, with no layer change needed |
| **M3** | Depth/stencil, multiple render targets, resource barriers, render passes, occlusion queries, fences, 2–3 frames in flight; run the game's first frames | Real frame structure and the first game code | The game reaches its first rendered frame (loading screen/menu or whatever the executable does) or stops on a single unsupported call with a clean report. **Synthetic part met** by `d3d12_m3_frame_test`: through the proxy, D3D12 API only, six frames through a ring of three slots (three frames in flight; each slot has its own allocator, list, targets, depth buffer, query and fence, and a frame waits with `SetEventOnCompletion` + an event for the fence its slot signalled three frames earlier before reading back and recording the slot again). Pass 1 draws three rectangles from root constants with translator-output shaders (`vs_m3.asm`, `ps_m3.asm`, hand-written RDNA2) through `OMSetRenderTargets` into two render targets (R8G8B8A8_UNORM and R32_FLOAT) and a `D24_UNORM_S8_UINT` depth/stencil buffer (the layer substitutes D32_SFLOAT_S8_UINT on the Radeon; the RTX has it natively): depth LESS with write, stencil REPLACE with references 1 and 2, then a draw with no depth test whose stencil test is EQUAL; before the far rectangle, a single-output probe (`ps_m3_query.asm`) uses the same depth/stencil tests with color, depth and stencil writes disabled, inside `BeginQuery`/`EndQuery(OCCLUSION)`, and `ResolveQueryData` writes its sample count to a readback buffer. The actual far-rectangle draw still writes both render targets with depth/stencil writes enabled. Resource barriers move the first target RENDER_TARGET → PIXEL_SHADER_RESOURCE\|COPY_SOURCE and the second to COPY_SOURCE (and back at the next frame, from resources created in those states). Pass 2 samples the first target through an SRV in a descriptor table (the M2 path) with `ps_m3_post.asm` into a third target. Every pixel of all three targets (the float one bit for bit) matches the CPU reference with error 0 in each of the six frames, and the single-output probe's occlusion count equals the exact visible area of the far rectangle in each frame (3555 samples over the six frames). The original version was validation clean (including synchronization validation) on the AMD Radeon 880M and the RTX 5070 Laptop; current probe/isolation results are recorded below. @whattheflop's run on #1584 (GeForce RTX 3080 and a Ryzen 7 9800X3D Radeon iGPU) sampled the fences with nonblocking reads after each submit and saw 1, 2, 3, 1, 2, 3 slots still incomplete on NVIDIA and 1, 1, 1, 1, 1, 1 on AMD: up to three slots in flight on NVIDIA, one on AMD (sampled fence completion; GPU execution overlap was not measured). **Still open for M3:** render passes (`BeginRenderPass`, D3D-27) are not exercised, nor `BINARY_OCCLUSION`, occlusion during pipeline statistics (D3D-162), MSAA or depth/stencil copies and read-only DSV states in this test; the game's own first frames need the executable |
| **M4** | Compute PSOs, UAVs, storage images, indirect dispatch, copy queues, async streaming | Asset loading and GPU compute | The world's streaming path runs without device loss; compute results match a CPU/reference check. **Synthetic part met** by `d3d12_m4_compute_test`: through the proxy, D3D12 API only, with translator-output compute shaders (`cs_m4_fill.asm` and `cs_m4_indirect.asm`, hand-written RDNA2). A COPY queue streams an input buffer in (`CopyBufferRegion` from an UPLOAD buffer into a DEFAULT buffer created in COMMON), `Signal`; a COMPUTE queue `Wait`s on it and runs one list: a compute root signature (four root constants and a descriptor table of five UAVs: a structured buffer, three raw buffers and an `R32_UINT` `RWTexture2D`), a compute PSO and `Dispatch` (256 threads) that fills the structured buffer and the image from the input and writes the arguments of the next dispatch into a raw buffer; barriers (the argument buffer to `INDIRECT_ARGUMENT`, UAV barriers on the buffer and the image); `ExecuteIndirect` with a dispatch command signature running a second PSO for the 3 groups the first dispatch asked for, which reads the buffer and the image back (`image_load`) and writes a raw output; then everything to COMMON and `Signal`. A COPY queue `Wait`s on that fence, copies the five results (the image through `CopyTextureRegion`) to a readback buffer, and the CPU waits with `SetEventOnCompletion`. Each D3D12 queue has its own `VkQueue` where the family has two (both test GPUs); all 1,924 result words match the CPU reference with error 0; validation (with synchronization validation) clean on the AMD Radeon 880M and the RTX 5070 Laptop. Layer gaps this test found, closed with it: buffer UAVs (structured and raw) were not written (`CreateUnorderedAccessView` stubbed them, now a V# in the slot); `ExecuteIndirect` of a dispatch ran whatever pipeline and root arguments the command buffer had bound instead of the list's (now recorded with the command, as `Dispatch` does); an image created in `UNORDERED_ACCESS` and first used by a shader was zero-filled by the first barrier or copy that named it, losing the shader's writes (now moved to GENERAL, zeroed, by the first submission after its UAV view is created that is not on a copy queue and dispatches or draws; every other queue's next submission waits for that one); and a queue submitted before the queue whose fence it waits on, when a resource's layout is handed over between them (the layout tracker followed submission order, so a copy list was replayed against the layout from before the compute list ran and the run stopped on a validation error): such a batch is now held on the CPU until a Signal that satisfies its Wait has been submitted, and recorded in dependency order (section 4.4, "Wait before Signal across queues"; `d3d12_queue_handover_test`, every queue-type and submission order, exact against the host runtime). **Still open for M4:** the storage-image initialization is tied to the first shader-running submission, not to the list that uses the image, so a list that only reaches an image through a barrier or copy before any dispatch or draw is submitted still zero-fills it first (as before), and an image used by a shader only on a queue that never runs one of its own dispatches or draws first relies on another queue's submission; buffer SRVs and CBVs as V#s (typed buffer UAVs and UAV counters are done: `d3d12_buffer_uav_test`, section 3.4 "Buffer UAVs"), `ExecuteIndirect` signatures that change root arguments or use count buffers for dispatches, and async streaming itself and the world's streaming path, which need the game |
| **M5** | Mesh shaders, `ExecuteIndirect`, bindless-table pressure, multi-threaded command lists, pipeline cache | World rendering | A stable open-world frame at any frame rate; then profile and optimise |
| **M6** | Ray tracing/XDXR: standard AS builds or the shipped `XboxBvhBuilder` compute path; BVH node format; the builder's 52 console `FSSM` programs (51 pass `spirv-val`, one rule left: [embedded_shaders.md](embedded_shaders.md)) | RT shadows and any RT effects | RT output matches a reference; the builder's shaders run through the `FSSM` translator or a substitute; no unknown XDXR call remains |
| **M7** | HDR, VRS, `PresentX` planes (2-plane HDR + GameDVR compose, `ImmediateThresholdPercent`), resize, alt-tab, `SuspendX`/`ResumeX`, frame pacing and `GetFrameStatisticsX` | Presentation polish | HDR and resize work on the target display; no validation errors over a long run. **Partly met:** `ImmediateThresholdPercent` (late frame flipped at once through `VK_KHR/EXT_swapchain_maintenance1` IMMEDIATE/FIFO_RELAXED, section 4.5; checked on the fake-clock decision and on the Vulkan present mode, 0 VUIDs in our runs with `LEONIDA_VK_VALIDATION=1` on two GPUs, no tearing observed on a display; the console's real semantics are unmeasured). **Plane compose** (section 4.5): 1-2 planes scaled from a source rectangle to a destination placement, point or bilinear, plane 1 alpha-blended (premultiplied or straight), filtered and blended in linear light, SDR (sRGB or gamma 2.2) and HDR10 (PQ) planes converted to the swapchain, SDR white from Windows' setting when HDR10 is active (else 203 nits), checked against a CPU model on two GPUs with 0 VUIDs (exact for point filtering, within 1-2 levels for bilinear, alpha, tone map and PQ conversions); the layouts, flag values and alpha convention are ours, the console's plane semantics are unmeasured, and the public HDR sample's second plane is the GameDVR copy, not an overlay. HDR10 swapchain (ST.2084 surface): verified on an RTX 5080 with an HDR10 display by @JavierBlanco23 in #2069, not on AMD (our test displays list no ST.2084 surface); HDR10 on an SDR swapchain is tone mapped (identity below 0.8 of SDR white, Reinhard shoulder above, then clipped per channel). `SuspendX`/`ResumeX`, frame pacing and a minimal `GetFrameStatisticsX` exist; VRS, alt-tab, an AMD HDR10 display run are open |

**M3 occlusion portability (D3D-209).** Vulkan's [occlusion query specification](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#queries-occlusion)
permits a count multiplied by the number of written color outputs, even with `VK_QUERY_CONTROL_PRECISE_BIT`.
The original two-output query returned twice the CPU area on Intel UHD (Windows driver 31.0.101.4146),
while NVIDIA RTX 3050 6GB Laptop (617.14) returned the area itself. Synthetic direct Vulkan and native D3D12
controls also returned 8192 on Intel for a 64×64 quad with two outputs, versus 4096 with one;
NVIDIA and D3D12 WARP returned 4096 in both cases. This does not establish duplicated replay commands.
The test now measures a single-output, write-free probe, keeping its exact CPU oracle and the original
MRT rendering and pixel checks. It does not divide application query results or change the runtime backend.

`d3d12_m3_first_pass` runs the same executable with `--first-pass-only`: all six frames, MRT pixels,
depth/stencil, barriers, queries and slot fences are still checked, and the third target must stay clear.
It isolates post-pass pipeline creation and execution: it skips creating the post-process PSO and recording
the sampled draw. All five FSSM files and their shader-library entries, including `vs_m2` and `ps_m3_post`,
are still required at startup. The second-pass root signature and descriptor heaps are still created,
so missing post-pass shaders or failed descriptor setup still fail this mode. `d3d12_m3_frame` remains
the full test. The first-pass mode passes with validation on both GPUs above (3555 samples). The full test passes on NVIDIA;
on this older Intel driver it still fails validation because `ps_m3_post` declares `Int64` without
`shaderInt64` support. The full test continues to report that separate capability issue.

### 6.1 Minimum API set for the first frame

M0–M3's synthetic tests need only this subset. The game itself may need more as soon as its first frame does;
everything else can be absent or a clean failure:

- `D3D12XboxCreateDevice` and `CheckFeatureSupport`; the measured device overrides (heap start handles in RAX;
  slot 76 texture creation if the game uses it; slot 62, compute-PSO creation in the builder's ABI, is not needed
  for the first frame); one direct queue, one allocator, one command list, one fence;
- DXGI factory/adapter/output enumeration; frame events and `PresentX` (or `GetBuffer`/`Present` if the game creates a swapchain);
- a root signature with root constants and one descriptor table, masking `0x03000000` on a copy; one VS+PS PSO and
  an input layout;
- a shader-visible CBV/SRV/UAV heap and a sampler heap; `CreateShaderResourceView`, `CreateSampler`,
  `SetDescriptorHeaps`, `SetGraphicsRootDescriptorTable`;
- one committed vertex/index buffer, one texture, upload through `Map`/`Unmap` or `WriteToSubresource`,
  `CopyBufferToImage`;
- resource barriers, RTV/DSV, `OMSetRenderTargets`, `Clear*View`, `DrawIndexedInstanced`, `PresentX`/`Present`;
- the runtime side: descriptor hash table, bindless set 0, push constants (`s0..s31`, heap base, code base),
  vertex input from the PSO layout.

Not needed for the first frame: compute, mesh, geometry, ray tracing, VRS, PresentX plane compose, reserved/tiled resources,
heap aliasing, indirect draws, multiple queues, pipeline libraries. (`PresentX` itself is needed; only its plane compose, HDR and statistics are not.)

**Why this order.** M0–M2 isolate the layer from the game: they exercise every contract piece with known data.
M3 is the first point where the game's own root signatures, PSOs and descriptors appear; expect the trace to
rewrite parts of section 3.5. M4/M5 are ordered by dependency: assets must stream before the world can draw, and
the corpus's 9,282 mesh shaders (and, very likely, indirect draws) are world-rendering features. M6 is last
because the game ships its own BVH builder; the layer may only need to run its compute work, not implement a full
DXR path.

**First-boot minimum.** Strictly, the first boot needs M0–M3 plus whatever the game does before its first frame.
If the first frame is a loading screen with a simple UI draw, M3 may be enough; if it is the world, M5 is the real
minimum. The trace decides, not the milestone labels.

## 7. Prior art and measured Xbox ABI

A GPL-2.0-only fork of AnyPS5 (`americas-recomp`, branch `xbox-series`) recompiles the same GTA VI Xbox Series
shaders and has run the package's DLLs against a PC D3D12 device behind an Xbox-ABI proxy. Its measurements are
prior art for this design. The findings below are paraphrased; no code or long passages are copied. They were
verified there by running the package DLLs (`XS_TRACE=1`, per-slot hit counters) or by static import analysis, not
independently re-measured here. **Basis:** *static imports* means PE import-table analysis; *run* means the package
DLLs were executed against a PC D3D12 device through the Xbox-ABI proxy.

Other Xbox Series compatibility-layer efforts have since measured overlapping parts of the same surface — running
the package's own console DLLs and shader pack, retail titles, or a console-compiler output. Their findings are
folded into the table below with the measurement basis noted per row; nothing is re-derived from their code.

| Finding | Basis and link | Consequence for this design |
|---|---|---|
| Across the 16 package DLLs there are only 6 Xbox-specific static imports: `xmem.dll` → `XMemAlloc`, `XMemFree`, `XMemGetWorkingSetStatistics`, `XMemVirtualQuery`; `d3d12_xs.dll` → `D3D12SerializeVersionedRootSignature`, `D3D12CreateRootSignatureDeserializer`. Both `d3d12_xs` names exist with the same names in the PC `d3d12.dll`. The other 492 functions are standard Windows and 113 are shipped in the package. Argument shapes for the less common `xmem` calls come from caller disassembly; some attribute bits are inferred. | *Static imports* — [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Confirms the middleware surface is tiny. The repo's `xmem` shim already covers all four; `d3d12_xs` still forwards only the two root-signature functions. The device/queue/command-list surface belongs to the executable. |
| `D3D12XboxCreateDevice(adapter, params, riid, out)` returns a PC `ID3D12Device` behind the Xbox-ABI proxy. The params struct is 24 words and is only logged; the signature comes from public documentation, not a header. | *Run* — [platform README](https://github.com/realdody/americas-recomp/blob/xbox-series/tools/xbox/platform/README.md), [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Corrects "device entry point unknown": the entry point is named. The params/flag layout remains **Open**. Do not assume `D3D12CreateDevice`. |
| Xbox texture creation is device slot 76 (heap properties / heap flags / description / initial state / clear value / iid / out), not `CreateCommittedResource`. The description is 56 bytes plus extension words; its layout field is `0x101` (tiled) or `0x1` (row-major buffer), and the PC proxy forwards it to `CreateCommittedResource` slot 27 with a driver-chosen layout. The FidelityFX upscaler creates 10 default-heap resources this way. | *Run* — [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Adds a required Xbox-specific device method and a non-public layout value. `0x101` must not be passed to the public layout enum; the tiled data layout/detile path remains **Open**. See section 4.2. |
| Device slot 14 is the public `CreateDescriptorHeap`; the returned heap is wrapped because `GetCPUDescriptorHandleForHeapStart` / `GetGPUDescriptorHandleForHeapStart` (heap slots 9/10) return the 8-byte handle in RAX on the Xbox build. The PC call returns it through a hidden pointer in RDX, so the proxy special-cases the return. | *Run* — [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Adds a return-convention quirk to the descriptor-heap ABI. A generic IDL/sret thunk will corrupt the handle. See section 3.4. |
| Xbox root-signature descriptors carry flags `0x03000000`. The PC serializer rejects them with `E_INVALIDARG`; the proxy clears the high byte on a copy. The v1.0 `D3D12SerializeRootSignature` path keeps flags at +32, and `0x3000001` also fails on the PC runtime. | *Run* — [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Narrows the root-signature reflection risk. Mask the high byte on a private copy; the meaning of the bits remains **Open**. See section 3.5. |
| The BVH builder's constructor creates 53 compute pipelines through device slot 62, called with a compute-PSO shape (a plain `D3D12_COMPUTE_PIPELINE_STATE_DESC`, two zero words, iid, out). On the PC interface the prior art forwards to, slot 62 is `CreateStateObject` (wrong arity), so the proxy overrides it. | *Run* — [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Corrects the assumption that the whole device vtable is public order. Slot 62 must be compute-PSO creation for the builder, not a state object. Slots 1/15/18/19/27/38 did match the desktop order. |
| A reusable `xbox_device_proxy.h` forwards every slot to a PC object, counts hits per slot and logs first calls with `XS_TRACE=1`; the observed device slots include 1, 14, 15, 16, 18, 19, 27, 38, 62 and 76. Their `xgr` stand-in covers 48 of 398 methods; Xodus (LGPL-2.1) covers 152 of 396 via IDL-generated stubs. | *Run* — [platform README](https://github.com/realdody/americas-recomp/blob/xbox-series/tools/xbox/platform/README.md), [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Confirms the trace-first approach and shows the long tail. We adopt the *method* (per-slot counters, first-call logging), not the code. |
| The BVH builder's 53 compute shaders are Xbox DXIL with `@xbox.*` intrinsics; 44 of 53 fail the PC D3D12 shader compiler, 22 have been lowered to standard DXIL and accepted as pipelines, and 31 use raw 64-bit GPU addresses. | *Run* — [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | A separate translation path from the FSSM RDNA2 corpus when measured on the **bitcode**. See section 4.8 and the paragraph below. Measured later on the console `XBCH` record of the same containers: 52 programs, 51 pass `spirv-val` through the FSSM translator ([embedded_shaders.md](embedded_shaders.md)). 52 programs, 42 pass `spirv-val` through the FSSM translator ([embedded_shaders.md](embedded_shaders.md)). |
| On the console's own shader compiler, compute user parameters go into user SGPRs `s4`… in root-signature order while each fits whole (root constant = 1 dword, root descriptor = 2, descriptor table = 1 dword holding the 32-bit address of its first descriptor); `s0`..`s3` are the driver's. From the first parameter that does not fit, it and all after it go into a packed array addressed by a 40-bit pointer in `s[0:1]` (constant 4 B, descriptor 8 B, no alignment). Thread-group ids follow the user SGPRs. | *Console-compiler output, sibling layer* | Narrows question 3 for the compute stage: the inline-from-`s4` model matches this repo's census (lowest CS pointer SGPR `s4`), and the spill target is named (the array at `s[0:1]`), not guessed. VS/PS/MS bases were not measured there; verify against the game's own signatures. |
| The command list's inline records: a `Dispatch` is `0x4002C400` followed by x, y, z (16 B); a 20-byte record `0x4003CC00` stands in for console slot 80; the pipeline record is a 28-byte copy of the PSO object's +0x10..+0x2B; a root parameter starts from the header word at list +0x344+4·i (the high half carries n−1 for a multi-constant write), then a descriptor is 12 B, one constant 8 B and n constants 4+4n B. | *Console-DLL run, sibling layer* | Supplies the record opcode for the "five-dword packet" this repo inferred at slot 80 and the header encoding for parameter groups; consistent with our own measurements of +0x18/+0x28/+0x30/+0x140/+0x344/+0x544. |
| Mesh (NGG) inputs measured across the corpus: `s3[27:24]` = wave index in the group, `v5` = group·128 + thread, `s16`–`s24` = user data. | *Corpus measurement, sibling layer* | Confirms the `s16` mesh base of §3.5's census and the `v5` group numbering already used for the mesh stage. |
| Console unified memory can be mapped GPU-side without copying: a region of at most 4,095 MB at a 4 GiB-aligned address, committed up front, opened through `ID3D12Device3::OpenExistingHeapFromAddress` (a partial commit and a full 4 GiB are rejected). The low 32 bits of a console address then index the buffer directly. | *Measured on PC D3D12, sibling layer* | A concrete recipe for §3.3's single address space: game addresses as offsets into one buffer that shaders reach through `PhysicalStorageBuffer64`. |
| The title physical-page calls use 64 KiB pages; mapping a run of consecutive pages returns a contiguous range inside the unified region, and scattered or explicitly addressed mappings are refused. | *Console-DLL run, sibling layer* | Sizes the page granularity the `xmem` shim should answer before the executable's real calls are known. |
| On a retail UE5 title the graphics command list takes console slot 83 for `CopyTextureRegion`, 84 for `CopyResource`, 26 for barriers, 127 for direct mesh dispatch, 59 for a second indirect/mesh dispatch path, and 96 for an async-compute budget call that accepts six zero words. | *Retail-title run, sibling layer* | Narrows question 2: mesh dispatch is not at the PC position (GCL6 slot 79) on the console, and the copy/barrier/budget slots add to the shifted groups already measured here. |
| The console `d3d12_xs.dll` also exports `D3D12XboxCreateSyntheticDevice` and `D3D12XboxSetProcessDebugFlags`; sibling console modules include `xg_xs.dll` (`XGCreateTexture2DComputer`, `XGComputeOptimalSwizzleMode`), `AcpHal.dll` (`ApuAlloc`, `ApuFree`, `ApuMapVirtualAddress`, `ApuIsVirtualAddressValid`, `AcpHalCreate`, `AcpHalAllocateShapeContexts`, `AcpHalReleaseShapeContexts`) and `dstorage_xs.dll` (console DirectStorage); `XAudio2_9.dll` ordinal 1 is `XAudio2Create` and `GameInput.dll` ordinal 1 is `GameInputCreate`. | *Export inventory, sibling layer* | Names the console-only modules the loader's shim registry should expect beyond `xgameruntime`/`d3d12_xs`/`xmem`/`gameinput`. |
| Console `xaudio2_9.dll` voices accept the console formats: XMA/XMA2 and an Opus voice format `0x704F`; a `CO_E_NOTINITIALIZED` from the system XAudio2 is retried after `CoInitializeEx` on the calling thread. | *Console-DLL run, sibling layer* | Forwarding console `XAudio2_9.dll` to the system XAudio2 2.9 works; the retry rule and the format remapping are the quirks to reproduce. |
| A retail title boots to its menu and lobby through the same shim-layer route (native x64 code, console-library stand-ins); its console-only library set adds `chad` (`XDsp*`), `xapu` (`XApu*`), `xcrdapi` and `*legacy` DLL variants to the list. | *Retail-title run, sibling layer* | Confirms the architecture end to end on retail code and extends the console-DLL checklist. |

### 7.1 What this confirms, corrects and narrows

**Confirms.**

- The 16 middleware DLLs need only four `xmem` functions and the two `d3d12_xs` root-signature functions. Our
  earlier statement that the bundled DLLs only import the two root-signature calls was right but incomplete; the
  four `xmem` imports complete the static surface. The repo's `xmem` shim already implements all four.
- Device slots 1 (AddRef), 15 (`GetDescriptorHandleIncrementSize`), 18 (`CreateShaderResourceView`),
  19 (`CreateUnorderedAccessView`), 27 (`CreateCommittedResource`) and 38 (`GetCopyableFootprints`) match the
  desktop order. Device slot 14 is the public `CreateDescriptorHeap`; only the returned heap's start-handle return
  is wrapped. The public chain is a usable base for those methods, not a guess.
- The BVH builder receives the device and command lists from the caller; prior art runs it exactly that way.

**Corrects.**

- **Device creation.** `D3D12XboxCreateDevice` is the entry point. The 24-word params struct is only logged, so
  the flag/layout meaning is still unknown; the name and call shape are no longer unknown. Prior art returns a PC
  D3D12 device behind the proxy; this design returns the Vulkan-backed `Device` behind the same call shape.
- **Vtable.** The Xbox device is not uniformly the public order. Slot 76 is an extra texture-creation method,
  slot 62 is called with a compute-PSO signature where the PC interface has a state-object method, and the
  descriptor-heap start handles use a different return convention. Implement the public chain plus explicit
  measured overrides; do not trust a single generic thunk for every slot.
- **Texture layout.** Xbox `0x101` tiled is not a public `D3D12_TEXTURE_LAYOUT` value, and slot 76 is the texture
  path. Our earlier `D3D12_RESOURCE_DESC::Layout` discussion covered only the public enum; it now includes the
  Xbox value and slot 76.
- **Root signatures.** The `0x03000000` flags must be masked on a copy before the host serializer/deserializer.
  Our earlier "forward the blob to the host" was too simple; the PC serializer rejects the Xbox bits.
- **BVH builder.** Its console programs are `FSSM`; the bitcode in the same container is the representation the
  PC compiler consumed. "Run the builder's dispatches" therefore takes the same FSSM → SPIR-V path as the game,
  with one rule still missing ([embedded_shaders.md](embedded_shaders.md)).

**Narrows.**

- Presentation slots and struct layouts (`PresentX` and plane parameters; call shapes are known from public samples, §4.5), command-list extensions, and queue/fence creation slots remain
  unknown; the prior art explicitly stops before them.
- Descriptor start-handle ABI is known; descriptor bytes and heap increments are still **Open**. Slot 15 is
  forwarded unchanged, but the actual increment values were not measured.
- Texture creation ABI and the `0x101` value are known; the tiled data layout, detile path and DCC remain
  **Open**.
- The device params (command-queue ring size, scratch memory, debug flags) are unused and their meanings are
  still unknown.
- The `xgr` 48/398 and Xodus 152/396 coverage numbers show how large the long tail is. They do not change the
  plan: implement what the game calls, in the order it calls it.

### 7.2 The BVH builder's shader path

The builder's 52 compute shaders are a second *corpus*, not a second shader front end. Each container carries an
LLVM bitcode blob and an `XDAT`/`XBCH` record; the bitcode is what the PC D3D12 compiler experiment consumed
(44 of 53 rejected, 22 lowered), while the console runtime executes the `XBCH` record. That record decompresses
to `FSSM`, the same container the game's own compute shaders use: 52/52 unpack, 51 translate and all 51 pass
`spirv-val` with `tools/rdna2spirv.py`. The one left, `Box16Rehydration`, stops at `buffer_store_short`;
[embedded_shaders.md](embedded_shaders.md) lists the rules that closed the other nine. The raw 64-bit addresses are the translator's existing raw-address path, so no address-to-resource
table is needed on top. A port that runs `XboxBvhBuilder` therefore needs the FSSM translator plus one rule,
not a DXIL front end (a substitute builder remains the alternative).
`spirv-val` with `tools/rdna2spirv.py`. The single program left (`Box16Rehydration`) stops at `buffer_store_short`
(#1864); [embedded_shaders.md](embedded_shaders.md) lists it. The raw 64-bit addresses are the translator's
existing raw-address path, so no address-to-resource table is needed on top. A port that runs `XboxBvhBuilder`
therefore needs the FSSM translator plus that rule, not a DXIL front end (a substitute builder remains the
alternative).
Correctness is still unverified on a GPU and the call sites need the executable (RT-5).

### 7.3 Licence

`americas-recomp` / AnyPS5 is **GPL-2.0-only**. GPL-2.0-only is not compatible with this repository's
GPL-3.0-or-later, so no code, headers, generated stubs or other implementation text from that tree may be copied
into Leonida. This section records ABI/format facts and links only; the designs in this document are independent.
Xodus is LGPL-2.1 and is cited only for its coverage number, not as a code source. The same line holds for the
other compatibility-layer efforts whose measurements are folded in above: facts and numbers only, paraphrased —
nothing is copied from their trees.

## 8. Known vs unknown

### 8.1 Known

- The public D3D12 ABI and method order (Windows SDK `d3d12.h` 10.0.26100.0); the Xbox public reference lists the
  same base interfaces and method names under `d3d12_xs.h`, but the measured Xbox device ABI is only mostly the
  public order: slots 1/15/18/19/27/38 matched, while slots 62 and 76 and the descriptor-heap start-handle return
  did not (section 7).
- The translator/runtime shader contract, and that it works on a real GPU for the tested paths.
- The package evidence in section 2 and Appendix A: the BVH builder's Device11/CommandList4/6/8 signatures and
  XDXR types, the FSR IIDs, the other DLLs' lack of `d3d12_xs` imports, `gameos.xvd`'s encrypted payload, and the
  absent executable.
- The prior art's measured Xbox ABI (section 7): 6 Xbox-specific static imports across the 16 DLLs,
  `D3D12XboxCreateDevice`, device slots 62/76, descriptor-heap start handles in RAX, root-signature flags
  `0x03000000`, and the BVH builder's container format (a DXIL bitcode blob plus the console `XBCH` record).

### 8.2 Unknown / open questions

| # | Question | How to resolve |
|---|---|---|
| 1 | The device entry point is now named `D3D12XboxCreateDevice(adapter, params, riid, out)`. What do its 24-word params and flag bits mean? | Public documentation and the executable's call sites. Log the words and accept any flag bits; do not treat a wrong params layout as fatal until the fields are understood. |
| 2 | Which Xbox device/heap/queue/command-list slots differ? Prior art measured device slots 62 and 76 and descriptor-heap slots 9/10; slot 14 is the public `CreateDescriptorHeap`, with only the returned heap's start-handle return wrapped. **Partly measured** with the FSR DLL ([d3d12_command_objects.md §3.5a](d3d12_command_objects.md)): device slots 14, 15, 16, 18, 19, 27, 38, 62 and 76 at their expected shapes (24 in the code, not called), slot 67 a console-only pipeline call; command-list slots 14, 15, 25 and 26 at their PC positions, then shifted (+4 from `SetDescriptorHeaps` at 32, +2 at `ExecuteIndirect` at 61), with console-only methods at 85 and 141. The BVH builder's call sites add the rest of the +4 root group (37, 39, 43, 45) and console slots 71, 73, 80, 146 (the GCL7 `Barrier`) and 148 (an upload allocator whose pointer is also the GPU address) ([d3d12_command_objects.md §3.3, §3.3a](d3d12_command_objects.md)). A second, retail-title measurement adds console slots 83 (`CopyTextureRegion`), 84 (`CopyResource`), 26 (barriers), 127 (direct mesh dispatch — not the PC position, GCL6 slot 79), 59 (a second indirect/mesh dispatch path) and 96 (an async-compute budget call) (section 7). Presentation, queue and fence slots remain unknown. | Trace every `QueryInterface` and every method slot with per-slot hit counters. A wrong slot shows as an immediate crash or wrong argument; compare against the SDK vtable. |
| 3 | How does the console map a root signature to `s0..s31` (order, per-stage visibility, spills above 32 dwords, static samplers, direct-heap base)? | **Compute partly measured** by a sibling project's console-compiler output (section 7): parameters inline in user SGPRs from `s4` in root order while each fits whole; the first that does not fit, and all after it, go into a packed array addressed by a 40-bit pointer at `s[0:1]`; thread-group ids follow the user SGPRs. That matches this repo's CS census (lowest CS pointer SGPR `s4`). VS/PS/MS bases remain open. Static samplers: no sampling instruction was found reading one from a driver table (SHADER-735, section 3.5; 0.66% undetermined); which S# is a static sampler remains open. Dump the game's root-signature blobs with the host deserializer; mask the `0x03000000` flag byte on a copy first; match against the per-shader user-SGPR read sets. If a signature exceeds 32 dwords, reverse the spill pointer from the shader. |
| 4 | Are Xbox descriptor bytes GFX10 T#/S#, and what are the heap increments? Does the game use `Create*View` or build descriptors itself? | Dump a shader-visible heap after the first `CreateShaderResourceView` and after the first draw; compare the bytes the shader loads with the T# parser. The start-handle return ABI is known (RAX); the increment values still need measuring. |
| 5 | What is the buffer/descriptor/code VA range, and what arena capacity is needed? | Log every `GetGPUVirtualAddress` and every descriptor base; track the high-water mark. Choose a capacity; grow at a frame boundary if needed. |
| 6 | Which swapchain/present path is used: DXGI, `PresentX`, or both? | **Partly answered from public samples (§4.5):** GDKX samples use `PresentX` + frame events and no DXGI swapchain (DXGI only for `IDXGIDevice1`→adapter→output). Hypothesis for the game, not observed: same. Still open: vtable slots, `D3D12XBOX_*` layouts, and whether the game also creates a swapchain. Check imports and the first present call. Implement the `PresentX` + frame-event presenter first; keep DXGI swapchain as fallback. |
| 7 | Does the game build acceleration structures through the standard D3D12 calls or through `XboxBvhBuilder`'s own compute path? | Log `BuildRaytracingAccelerationStructure`, `CopyRaytracingAccelerationStructure`, `Emit...PostbuildInfo`, and the builder's command-list method calls on the first RT frame. The builder's container carries a DXIL bitcode blob and a console `XBCH` record; the record is `FSSM` and 51 of 52 programs pass `spirv-val` ([embedded_shaders.md](embedded_shaders.md)). If the builder path is used, translate the `XBCH` record rather than the bitcode. |
| 8 | What are the layouts of `D3D12XBOX_XDXR_STREAM_DESC`, the batched build desc, and the offline statistics desc? | **Measured** from `XboxBvhBuilder.dll`'s reads ([xdxr.md](xdxr.md)): `BATCHED_DESC2` (`0x50`) and `XDXR_STREAM_DESC` (`0x10`) with every field's offset, and the stream records' offsets, types and defaults. Still open: the record names, and the offline statistics layout, which the DLL never writes (it rejects that postbuild type). |
| 9 | Xbox texture creation is device slot 76 with layout `0x101` (tiled) or `0x1` (row-major). What is the tiled data layout, and does the game upload console-tiled or DCC data? | Trace slot 76, `CopyTextureRegion` and the resource descs. If the game uploads console-tiled data, add a detile step or a matching layout. |
| 10 | Does the game use binning, DCC, VRS, wave MMA, stream output, tessellation, or amplification shaders? | **Shaders: answered** by `tools/feature_scan.py` over all 26,503 (1.0.2.1): no hull, domain or amplification shader (DXIL program kinds: 2,236 VS, 12,101 PS, 195 GS, 2,689 CS, 9,282 MS); no `WAVE_MMA` or `SHADING_RATE` in any SFI0, and no `v_mfma`/`v_wmma`/`v_smfmac` in the disassembly; no output-signature element on a stream other than 0 (all 195 GS write stream 0 only) and no GDS / `ds_ordered_count` / GWS instruction, so no stream-output path. Also from SFI0: 10 PS need ROVs (pixel interlock), 30 shaders `TILED_RESOURCES` (see #18), 685 `ATOMIC_INT64_ON_TYPED_RESOURCE`. Binning and DCC are not shader-visible; they stay open for the PSO/stream subobjects. |
| 11 | What should `CheckFeatureSupport` answer for UMA, resource binding tier, mesh tier, RT tier? | **Implemented** (D3D-11). Reports truthful host architecture (a discrete GPU: non-UMA; an integrated GPU, `VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`: UMA, not cache-coherent, as the host runtime answers on a Radeon 780M), ResourceBindingTier 3 (descriptor indexing / bindless support), 48-bit GPU virtual address support matching the arena, and maps `D3D12_FEATURE_FORMAT_SUPPORT` to Vulkan format properties plus multisample render-target/resolve support and one quality level per supported sample count. |
| 12 | Does the game rely on async compute or copy queues for correctness? | Trace queue creation and cross-queue fences. If only performance, serialize on one queue for boot. |
| 13 | Is 144 bytes of push constants available on all target GPUs? | **Resolved / Measured** (D3D-14). Checked at device creation in `runtime/shims/d3d12_vulkan_device.cpp`. Measured across the test fleet (#166): NVIDIA (256 B), desktop AMD (256 B), and Apple M1/MoltenVK (4096 B) satisfy 144 B natively. A Radeon 680M on the Windows driver 22.40.84.06 and the earlier Intel Arc entries expose only the core minimum 128 B (`maxPushConstantsSize == 128`), but an Arc B580 on Windows 11 (driver 32.0.101.9175) reports 256 B (#166), so the 128 B figure for Arc depends on the OS or driver; an Arc result on Linux (Mesa ANV) would settle which side it is. `d3d12_vulkan_device_test` accepts either answer and checks that the fallback flag matches it. Below 144 B, the device logs a warning and marks `push_constants_need_fallback`, routing the 16-byte tail (dwords 32..35: heap base and code base) to a uniform-buffer fallback. |
| 14 | Are there PSOs with two code-base stages? | Count from the PSO/shader hash table and the index metadata. If yes, implement per-stage code-base slots (section 5.2). **Per stage, measured** (`tools/feature_scan.py`, `s_getpc_b64` kept for data rather than a far branch): 54 VS, 93 PS, 266 MS, 263 CS, 0 GS; the translator's `uses_code_base` gives the same counts (with LLVM 23 that needs #223, without which these 1,471 shaders are rejected). So VS+PS and MS+PS pipelines with two code-base stages are possible; whether any exists depends on which shaders a PSO pairs, which is in the executable, not in `shaders.rpf`. |
| 15 | Is `ExecuteIndirect` used with signatures Vulkan cannot express directly? | Trace `CreateCommandSignature` and `ExecuteIndirect`; implement simple signatures first, expand the complex ones with compute. The simple signatures are D3D-17 (#313). |
| 16 | Does the game use append/consume UAV counters or typed buffer formats the translator does not model? | **Shaders: answered** by `tools/feature_scan.py` over all 26,503 (1.0.2.1, disassembled with LLVM 21.1.8 for gfx1013; SHADER-65). **No append/consume counters:** none of the 445,394 PSV0 binding records is a structured UAV with a counter, and there is no `ds_append`/`ds_consume`/GDS instruction. Typed buffers are declared in 5,359 shaders as SRVs (17,328 records) and in 2,177 as UAVs (4,005). **MTBUF** (format in the instruction): 52,890 loads in 10,278 shaders, all in five formats the translator decodes (`16_16_SNORM` 24,881, `8_8_8_8_UNORM` 19,835, `8_8_8_8_UINT` 5,099, `2_10_10_10_UNORM` 2,648, `16_16_16_16_UINT` 427), and no MTBUF store. **MUBUF** `buffer_*_format_*` (format from V# word 3): 90,316 accesses (84,908 loads, 5,408 stores) in 11,591 shaders; word 3 comes from a scalar load in 88,924 of them, is computed in 1,390, and is a literal in 2 (FORMAT 0, INVALID, which reads 0 on the hardware and in the translator). So the shaders fix no MUBUF format: the descriptors the executable writes decide it. Still open: which formats those descriptors carry; the runtime can log a V# format outside `Gen.FORMATS` (those read 0 today) when the descriptor is created or copied. Trace UAV descs and shader `buffer_*` ops; extend the translator/runtime together. **Runtime side done, console side open:** the layer writes typed buffer V#s and a counter V# (a hypothesis for where the shader finds it, section 3.4 "Buffer UAVs"); whether the game uses either is for the trace. |
| 17 | Do upload/readback resources ever reach shaders (as vertex/index buffers or descriptors), and does the `-heap_base` bias work for them? | Trace `CreateCommittedResource` heap types and every VB/IB/descriptor that points at an upload resource; test the bias with a synthetic draw. |
| 18 | Does the game alias buffers/images in a heap, or use reserved/tiled resources? | Trace `CreatePlacedResource` ranges, aliasing barriers, `CreateReservedResource` and tile mappings. If yes, implement shared memory per heap range or fall back to linear allocations with a clear log. Shaders: 30 declare `TILED_RESOURCES` in SFI0 (`tools/feature_scan.py`), so reserved/tiled resources are likely used. Placed resources are D3D-16 and reserved resources D3D-20 (#313). |
| 19 | What do the `0x03000000` root-signature flag bits mean? | Masking them on a copy works around the PC serializer; the semantics remain unknown. **Origin measured** (section "Root-signature flag quirk"): in the package only `XboxBvhBuilder.dll` sets them, in its 52 compiled `RTS0` parts and unconditionally at `0xc30e` before serialising; the FSR DLL's 29 `RTS0` parts carry 0. Compare against the public flag enum and observe the game's behaviour with and without the mask; check whether the executable's root signatures carry them. |
| 20 | Which of the BVH builder's 52 `FSSM` programs does the game actually use, and do the translated programs match a reference? | Dump the slot-62 compute PSOs on the first RT frame and map each to its container by the `ILDN` name; 51/52 pass `spirv-val` today and the one missing rule is named in [embedded_shaders.md](embedded_shaders.md); compare against the builder's CPU validation (`CpuValidateResultX`) or a captured reference. 42/52 pass `spirv-val` today and the four missing rules are named in [embedded_shaders.md](embedded_shaders.md); compare against the builder's CPU validation (`CpuValidateResultX`) or a captured reference. |
| 21 | Which `Metadata` values and blob format does the console's PIX use for `SetMarker`/`BeginEvent`? The layer reads 0 and 1 as strings and labels 2 (the PC PIX3 blob) `PIX event` (section 4.10). | Log the `Metadata` values the game passes; if it is 2, decode the console blob from the executable's PIX calls rather than assuming the PC WinPixEventRuntime layout. |
| 22 | Does the game ask `CheckFeatureSupport(D3D12_OPTIONS5)` for `RaytracingTier` and switch ray tracing off when it reads `NOT_SUPPORTED`? The layer answers `NOT_SUPPORTED` because full DXR (`CreateStateObject`, `DispatchRays`) is not implemented, even though the XDXR acceleration-structure path the game's own traversal shaders use is ([bvh.md](bvh.md) §7a). On the console, ray tracing is always present, so the game may never ask. | The first-boot trace (`--trace-d3d12`) logs every `CheckFeatureSupport` call and its answer. If the game asks and gates its RT on the tier, decide with the maintainer whether to report `TIER_1_0`/`1_1` for this game while `DispatchRays` stays unimplemented; until then, keep the conservative answer. |

## 9. Risks and non-goals

**Risks.**

- **Xbox ABI deviations.** The device vtable is mostly public but not uniformly: slot 76 adds texture creation,
  slot 62 is compute-PSO creation for the BVH builder, descriptor-heap slots 9/10 return in RAX, and root signatures
  carry `0x03000000`. Use a per-slot proxy with counters; presentation, command-list, queue and fence slots remain
  unknown. A wrong return convention or slot mapping corrupts state rather than failing cleanly.
- **Root-signature mapping.** A wrong `s0..s31` packing produces silently wrong rendering, not a crash. The
  per-shader read-set metadata and a strict mismatch check are the mitigation. The `0x03000000` flag mask is a
  workaround, not an explanation.
- **The arena.** A single BDA arena is required by the shader contract for buffers, descriptor tables and code; the
  game's buffer working set may exceed `maxBufferSize` or PC VRAM. Host-visible arenas are slow; device-local
  arenas need staging and careful upload/readback handling. This is the main performance risk.
- **Hash-table cost.** Every image/sampler access hashes a T#/S# and probes a storage buffer. **Measured (D3D-56, section 5.1)** on an RTX 5080 (driver 621.24): in `DEVICE_LOCAL` VRAM the lookup itself stays fast up to 100k entries (wave-uniform full sweeps `3.3–12.8 us/1M`, divergent hits `24–72 us/1M`), whereas `HOST_VISIBLE | HOST_COHERENT` system memory slows uniform full sweeps 16x–36x and divergent lookups 210x–879x at 10k–100k entries; `bc8e604`'s bounded slot relocation keeps the 100k table at `20.00 MiB` (`19.07%` load, `max_probes=8`) instead of doubling. **The runtime now keeps that table in `DEVICE_LOCAL` memory** and uploads only the slots that changed, from a mapped staging copy, with the copy and its barrier in the command buffer that will read it (D3D-76; 46.2x on a divergent miss at 100k on an RTX 2060 SUPER). Improving slot placement within the 8-probe window is the open optimisation (SHADER-100); Intel, RTX 5070 Ti Laptop and AMD RDNA3 are measured in section 5.1; Apple numbers remain open via `vk_descriptor_bench`.
- **Wave64.** A native wave64 shader needs a 64-wide subgroup in its stage: AMD (Windows driver, RADV on GFX10+) can require
  it for compute, mesh and pixel shaders, NVIDIA cannot (32 only). There, compute and mesh shaders translated with
  `--wave64-emulate` are exact; wave64 pixel, vertex and geometry shaders have no emulation and keep the
  translator's 32-wide approximation (cross-half ops are approximate). This is a known fidelity gap, not specific to
  the layer.
- **Xbox tiling/DCC.** Slot 76 and layout `0x101` are known, but the tiled data layout, detile path and DCC remain
  unknown. If the game assumes console tiling or compression metadata, the layer needs a detile/repack step.
- **BVH builder shaders.** The container's DXIL bitcode is Xbox DXIL that PC drivers reject; its console `XBCH`
  record is `FSSM`, and 51 of the 52 programs already pass `spirv-val` through the same translator as the game
  ([embedded_shaders.md](embedded_shaders.md)). The one left needs `buffer_store_short` (#1864). Translation is not execution:
  there is no GPU reference for any of the 52 yet.
- **Ray tracing.** The game's RT work includes 64 compute shaders that traverse AMD-format BVH nodes, and it ships
  a BVH builder; a standard D3D12 AS path may also exist. If it is used, the layer must build AMD-format nodes, not
  a Vulkan AS.

**Non-goals.**

- Xbox One ESRAM or `d3d12_x` (the Xbox One flavour); this is a Series build.
- D3D11, D3D12 video, protected content, cross-process sharing, meta commands, DRED tooling.
- A general D3D12 compatibility layer for arbitrary applications.
- Any use of game data, keys or decryption.
- Copying code, headers or generated stubs from the GPL-2.0-only prior art; its ABI/format facts are cited with
  links only.
- Replacing the translator's shader contract. The layer adapts to it; changes to it (per-stage code base, extra
  metadata) are coordinated with `tools/rdna2spirv.py`.

## 10. References

Prior art (facts and links only; no code copied; GPL-2.0-only, see section 7.3):

- [americas-recomp, branch `xbox-series`](https://github.com/realdody/americas-recomp/tree/xbox-series)
- [PLATFORM_SURFACE.md: package import surface and measured device ABI](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md)
- [platform tools README: proxy, `XS_TRACE`, xgr, first-run workflow](https://github.com/realdody/americas-recomp/blob/xbox-series/tools/xbox/platform/README.md)
- [GPL-2.0 license text](https://github.com/realdody/americas-recomp/blob/xbox-series/LICENSE)

Public D3D12:

- [D3D12 programming guide](https://learn.microsoft.com/en-us/windows/win32/direct3d12/directx-12-programming-guide)
- [Root signatures](https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signatures)
- [`D3D12_ROOT_SIGNATURE_FLAGS`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_root_signature_flags)
- [Resource binding in HLSL](https://learn.microsoft.com/en-us/windows/win32/direct3d12/resource-binding-in-hlsl)
- [Descriptor heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptor-heaps)
- [Recording command lists and bundles](https://learn.microsoft.com/en-us/windows/win32/direct3d12/recording-command-lists-and-bundles)
- [Using resource barriers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12)
- [Enhanced barriers (DirectX spec)](https://microsoft.github.io/DirectX-Specs/d3d/D3D12EnhancedBarriers.html)
- [Executing and synchronizing command lists](https://learn.microsoft.com/en-us/windows/win32/direct3d12/executing-and-synchronizing-command-lists)
- [Fence-based resource management](https://learn.microsoft.com/en-us/windows/win32/direct3d12/fence-based-resource-management)
- [Queries](https://learn.microsoft.com/en-us/windows/win32/direct3d12/queries)
- [Mesh shader (DirectX spec)](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html)
- [Direct3D 12 raytracing](https://learn.microsoft.com/en-us/windows/win32/direct3d12/direct3d-12-raytracing)
- [`ID3D12Device10`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nn-d3d12-id3d12device10),
  [`ID3D12GraphicsCommandList7`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nn-d3d12-id3d12graphicscommandlist7)
  (the Win32 reference does not yet have pages for `Device11`/`CommandList8`; the Windows SDK header defines them)

Xbox public documentation:

- [Xbox graphics API reference](https://devdocs.xbox.com/reference/graphics/d3d12)
- [Xbox `ID3D12Device::CreateCommittedResource`](https://devdocs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device/methods/id3d12device_createcommittedresource_public)
- [Xbox-GDK-Samples (MIT): SimpleDeviceAndSwapChain, FramePacing](https://github.com/microsoft/Xbox-GDK-Samples)
- [DirectXTK12 wiki: DeviceResources](https://github.com/Microsoft/DirectXTK12/wiki/DeviceResources); [directxtk12modelviewer](https://github.com/walbourn/directxtk12modelviewer) (MIT); SDL2 `SDL_render_d3d12_xbox.cpp` (zlib)
- [`PresentX` mentioned in `XGameStreamingSetResolution`](https://docs.xbox.com/reference/system/xgamestreaming/functions/xgamestreamingsetresolution)
- [GDK release notes mentioning `D3D12XBOX_RESOURCE_FLAG_FORCE_TEXTURE_COMPATIBILITY`](https://docs.xbox.com/announcements/gdk/2510/release-notes-2510)

Vulkan:

- [Descriptor indexing](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_descriptor_indexing.html)
- [Buffer device address](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_buffer_device_address.html)
- [Synchronization2](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_synchronization2.html)
- [Dynamic rendering](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_dynamic_rendering.html)
- [Timeline semaphores](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_timeline_semaphore.html)
- [Mesh shader](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_mesh_shader.html)
- [Fragment shader barycentric](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_fragment_shader_barycentric.html)
- [Fragment shader interlock](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_fragment_shader_interlock.html)
- [Subgroup size control](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_subgroup_size_control.html)
- [Maintenance4](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_maintenance4.html)

## Appendix A: evidence and reproduction

REA was used for the PE import/export and string inventory. The XVD header and entropy scan used local .NET,
because `gameos.xvd` is an XVD container rather than a PE. Nothing was decrypted or executed.

### A.1 `gameos.xvd`

- Size 348,069,888 bytes. `msft-xvd` magic at file offset **0x200**; the 512 bytes before it are a high-entropy
  block, consistent with a signature. Using the standard XVD header layout, the first two dwords after the magic
  are `volume_flags = 1` and `format_version = 3`.
- 64 KiB Shannon entropy sampled at offsets 0, 0x1000, 0x10000, 1 MiB, 16 MiB, 64 MiB, 128 MiB, 256 MiB:
  6.40, 6.69, 7.68, 7.68, 7.997, 7.998, 7.997, 7.997 bits/byte.
- A chunked ASCII scan (4 MiB chunks) of the whole file found no `d3d12_xs.dll`, `d3d12.dll`,
  `D3D12CreateDevice`, `ID3D12Device`, `gameos`, or `xboxgip` strings. A token spanning a chunk boundary could be
  missed, but the entropy result does not depend on the scan.
- Conclusion: the payload is encrypted or compressed. It cannot supply the D3D12 API surface without keys, which
  are out of scope. No key search or decryption was attempted.

Reproduce: read the first 4 KiB, locate `msft-xvd`; compute per-block entropy; scan for the ASCII tokens above.
The file is an XVD container, not a PE; PE-oriented import tools do not apply.

### A.2 `XboxBvhBuilder.dll`

- Size 884,224 bytes; PE32+ x86-64; export directory RVA 861,536, 12 named exports.
- The exported C++ signatures name:
  - `ID3D12Device11` (constructor, rehydration),
  - `ID3D12GraphicsCommandList4` (standard build, copy, CPU validation),
  - `ID3D12GraphicsCommandList6` (batched build),
  - `ID3D12GraphicsCommandList8` (batched rehydration),
  - `D3D12XBOX_XDXR_STREAM_DESC` and `D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2`,
  - `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC` and the standard build/copy/prebuild types.
- Imports: `d3d12_xs.dll!D3D12SerializeVersionedRootSignature` (ILT RVA 864,368, IAT RVA 148,048) and
  `xmem.dll!XMemVirtualQuery` (ILT RVA 864,384, IAT RVA 148,064). No other console graphics imports.
- GUID byte scan against the Windows SDK `d3d12.h` found `IID_ID3D12PipelineState` at file offset 0x0c9ed8 and
  `IID_ID3D12RootSignature` at 0x0c9ee8. Presence in a read-only data table is evidence of the interface version
  the DLL was compiled against, not proof that every method is called.

Reproduce: `rea.inspect_native_exports` and `rea.inspect_pe_imports` on the DLL, then a byte scan for the
little-endian GUIDs from `d3d12.h`.

Constructed by `runtime/test/bvh_builder_test.cpp` (1.0.2.1: `rt::Loader`, `Loader::initialise()` for TLS and
`DllMain`, `.pdata` registered by the loader (#176), then `XboxBvhBuilder(ID3D12Device11*, bool)`
on a PC device behind the Xbox-ABI proxy). Measured:

- 0 unimplemented imports; `D3D12SerializeVersionedRootSignature` and `XMemVirtualQuery` bind to shims. The constructor writes 24 bytes of the object (`sizeof(XboxBvhBuilder)` >= 24;
  the rest of its state is elsewhere). It prints nothing through `OutputDebugString` and never calls
  `TerminateProcess`. On failure it throws the failing `HRESULT` as a C++ `long`.
- **Device slot 62 is called as `(desc, 0, 0, &IID_ID3D12PipelineState, out)`**, with a plain
  `D3D12_COMPUTE_PIPELINE_STATE_DESC` (root signature set, node mask 0, flags 0). The proxy used to read
  `(desc, riid, out)`, so the first call returned `E_POINTER` and the constructor threw it.
- With slot 62 read that way, on **WARP** the constructor returns after 62 ms: 17 root signatures (slot 16) and 53
  compute PSOs (slot 62, 53 of 53 `S_OK`), CS bytecode 4–75 KB each. `~XboxBvhBuilder` returns and makes no device
  call. Created is not the same as correct: the container's DXIL bitcode is Xbox DXIL with `@xbox.*` intrinsics
  (section 7.2), its console `XBCH` record is `FSSM`, and nothing has been dispatched yet.
- **NVIDIA (RT-54).** On an RTX 3060 (616.64), an RTX 4090 (621.14) and an RTX 5070 Ti (621.24), all on Windows 11,
  the first 48 PSOs return `S_OK`, and the 49th (38,248 bytes) never does. Some runs return `E_OUTOFMEMORY` for it,
  which the constructor throws. Other runs end the process inside `nvwgf2umx.dll` with `0xC0000409`, after the
  driver's LLVM 7.0.1 bitcode reader prints `Invalid record` or `Never resolved value found in function`. That
  can't be caught. The D3D12 debug layer prints nothing first.
  - **Cause: the input, not Leonida.** The proxy passes the bytecode through unchanged, so nothing is wrong in what
    it forwards. The input is the problem: `LEONIDA_BVH_RECORD62=1 LEONIDA_BVH_DXIL=1` shows that every one of the
    53 PSO calls (52 distinct containers) is signed, has the same 9 parts, and is shader model 6.5 / DXIL 1.5. But dxil.dll (Windows SDK
    10.0.26100) rejects all 53 (`0x80aa0009`): the target triple is `dxil-ms-xb2` and the module carries
    `xbox.compiler.options` metadata. Most also have Xbox-only externals (44 of 53), `extractvalue` on non-DXIL types
    (16) or other disallowed instruction types (5). No PC driver has to accept this, and WARP happens to.
  - **What sets the 49th apart: one error kind.** It is the only shader of the 53 that the validator reports for a
    resource in a function signature (a non-entry function that takes a resource). Its other counts are 265
    `extractvalue`, 6 Xbox externals and 514 `i64` mentions; the 36th shader has the same 265 and 468 and is created.
  - **Workaround: the test uses WARP on every hardware vendor.** AMD doesn't take these shaders either: on an RX 7900
    XTX (32.0.32029.2010) the constructor faults with `0xC0000005` right after the first slot 62 call (RT-89, #166).
    So `bvh_builder_test` uses WARP whenever there is a hardware adapter, unless `LEONIDA_BVH_HARDWARE=1`, and it
    prints the reason with the adapter's PCI vendor id. WARP creates all 53 PSOs, and the constructor and destructor
    return.
  - **The runtime itself still shouldn't hand Xbox DXIL to a PC driver.** The same container carries the console
    `XBCH` record, which decompresses to `FSSM`; 51 of the 52 programs translate and pass `spirv-val` today
    ([embedded_shaders.md](embedded_shaders.md)). Translating that record is the path that avoids a DXIL lowering
    front end (section 7.2, M6); the bitcode is a second representation.
- **On the Vulkan device (D3D-557).** With `LEONIDA_BVH_DEVICE=vulkan` the constructor runs on the Vulkan-backed
  device, where slot 62 needs the translated module of each container's `FSSM` program. `make_game.py` translates
  the DLL's programs into `<shaders>/bvh/` and the shader library reads that index
  ([embedded_shaders.md](embedded_shaders.md) section 7). Measured on 1.0.2.1: before that step creation #1 failed
  (`E_FAIL` with no library, `E_INVALIDARG` with the game's corpus loaded). With it, on an RTX 3070 (NVIDIA 610.74)
  and on a Ryzen 7 9700X's integrated Radeon (AMD 32.0.21042.62), both with the `--wave64-emulate` translation, 49 of
  the first 50 creations return `S_OK` and the constructor throws `E_INVALIDARG` at #50, `Box16Rehydration`, the one
  program the translator has no rule for yet (#1864). With that rule applied locally all 53 are created and the
  constructor and destructor return. Nothing has been dispatched.

### A.3 `amd_fidelityfx_upscaler_xbox.dll`

- Size 786,432 bytes; PE32+ x86-64.
- Import: `d3d12_xs.dll!D3D12CreateRootSignatureDeserializer` (IAT RVA 164,672).
- GUID byte scan found `IID_ID3D12Device` (0x0472f0), `IID_ID3D12Resource` (0x047310),
  `IID_ID3D12RootSignatureDeserializer` (0x047320), `IID_ID3D12PipelineState` (0x047330),
  `IID_ID3D12Heap` (0x047340), `IID_ID3D12RootSignature` (0x047350), `IID_ID3D12DescriptorHeap` (0x0472e0) and
  `IID_ID3D12CommandSignature` (0x047300). Again, presence ≠ use.
- Run through the public FidelityFX API by `runtime/test/fsr_dispatch_test.cpp` (1.0.2.1: `rt::Loader` and
  `Loader::initialise()`, with #93's static TLS, which the DLL's thread-safe statics need, and `.pdata` registered by
  the loader (#176); PC D3D12 device on an RTX 4090 behind the Xbox-ABI proxy). Measured:
  - It maps with 0 unimplemented imports: `D3D12CreateRootSignatureDeserializer` and `XMemAlloc` /
    `XMemFree` / `XMemGetWorkingSetStatistics` bind to the shims, 82 imports to host DLLs. Its TLS template is
    8 bytes with no callbacks. `DllMain(DLL_PROCESS_ATTACH)` returns TRUE.
  - `ffxConfigure(GLOBALDEBUG)` and the CPU-only upscale queries accept the **PC struct type ids**: the
    Performance ratio is 2.0 and 3840x2160 Quality renders at 2560x1440. `GET_VERSIONS` for the upscaler lists one
    provider, `0xf5a5ca1e01001001` "4.1.1".
  - `ffxCreateContext` reports, through the message callback, that only one configuration is built in:
    `maxUpscaleSize` 1920x1080 ("only the fixed-resolution Quality shader tables are included", so a 1280x720
    render size), with `HIGH_DYNAMIC_RANGE`, `DEPTH_INVERTED` and `AUTO_EXPOSURE` all required. It also requires an
    `ffxCreateContextDescUpscaleVersion` in the chain; 4.1.2 is rejected as invalid, 4.1.1 and 4.0.0 are accepted.
  - With all of that, `ffxCreateContext` returns `FFX_API_RETURN_ERROR` with no message, no XMem allocation and no
    call on the device proxy, for the PC DX12 backend descriptor (`0x2`), for the same `{header, device}` layout
    under `0x02000000`, `0x02000001` and `0x02000002`, and with no backend descriptor at all.
- **The Xbox backend descriptor, measured from the DLL** (Ghidra 12.2 through REA; RVAs, image base `0x180000000`):
  - `ffxCreateContext` (`0x11d0`) picks its one provider (`ffxProvider_FSR4_Int8` by RTTI) by finding a device in
    the chain: type 4 (`GET_VERSIONS`) at `+0x18`, types `0x10009` and **`0x01000002`** at `+0x10`.
  - The provider's create (`0xc9b0`) wants `CreateContextDescUpscale` (`0x10000`) first, checks the version
    descriptor (`0x1000b`, 4.0.0 to 4.1.1), then walks `pNext` for **type `0x01000002`: `{header, device}`, the device
    at `+0x10`**, the same layout as the PC DX12 descriptor. Two of them, or none, is `FFX_API_RETURN_ERROR`
    without a message: that is what the ids above got. An optional type `0x01000003` sets function pointers: when the
    one at `+0x20` is set (with `+0x28`), the backend calls it instead of `CreateHeap` (device slot 28, `0x7390`); what
    its other fields do is open.
  - The HRESULT checks at the calls seen here (`0x33d4`, `0x65b3`, `0x74a8`) go through one helper (`0x1c60`):
    `FormatMessageW` of the code, then `throw 1`.
- **Run with that descriptor** (`fsr_dispatch_test`, PC device on an RTX 2000 Ada, driver 597.06, behind the
  Xbox-ABI proxy): the DLL creates its backend through the proxy. Device slots called: 14 `CreateDescriptorHeap` (5),
  15 `GetDescriptorHandleIncrementSize`, 18 `CreateShaderResourceView`, 19 `CreateUnorderedAccessView` and **76** (4),
  plus heap slot **9** `GetCPUDescriptorHandleForHeapStart` (Xbox RAX form). No earlier test in this repository
  reached slot 76 or heap slot 9 with a real client; their measured shapes hold. Slot 76 is called with the 7
  arguments of `CreateCommittedResource`
  (heap properties, heap flags, resource desc, state, clear value, IID, out; call at `0x33cc`), the desc being the
  56-byte `D3D12_RESOURCE_DESC` with layout `0x101` and the word 3 at `+0x34`.
  - On that GPU it stops at its 4th slot-76 texture: 1920x1080 `DXGI_FORMAT_R9G9B9E5_SHAREDEXP` with
    `ALLOW_UNORDERED_ACCESS`, which the NVIDIA driver refuses (`E_INVALIDARG`); WARP creates it. The console build
    creates it, so the console takes that format as a UAV. On the same GPU Vulkan has no `STORAGE_IMAGE` support for
    `VK_FORMAT_E5B9G9R9_UFLOAT_PACK32` either (vulkaninfo, driver 597.06), so the layer needs an emulation (32-bit
    storage plus a pack/unpack) where a GPU lacks it. @Vonsek measured the same texture on more devices (#864):
    refused on an RTX 5080 (616.92), created on an AMD Radeon and on WARP, where typed UAV writes round-trip; and on
    both GPUs one allocation can serve both uses, a mutable `R32_UINT` Vulkan image with `E5B9G9R9_UFLOAT_PACK32` in
    its format list, written through an `R32_UINT` storage view and sampled through an `E5B9G9R9` view.
    **Implemented** in `d3d12_vulkan_device.cpp` and `src/uav_clear.h` (`d3d12_r9g9b9e5_uav_test`): on an RTX 3060 Ti
    (driver 621.01, `LEONIDA_FSR_DEVICE=vulkan LEONIDA_FSR_NO_STANDIN=1`, `LEONIDA_VK_VALIDATION=1`),
    `ffxCreateContext` and `ffxDestroyContext` both return OK without `standin_slot76` and with 0 validation errors.
    Open: a float store to that format rounds differently on PC devices (@Vonsek, #864: AMD Radeon truncates the
    mantissas, WARP rounds to nearest even; 609 of 794 test triples differ). Which rule the console FSR needs, and
    whether its shaders store floats to that texture at all (requiring shader-side packing into the `R32_UINT` storage
    view), waits on decoding its `FSPX` shaders.
  - With that one texture created as `R32_UINT` instead (the test's PC stand-in, `standin_slot76`), or as is on
    WARP, `ffxCreateContext` and `ffxDestroyContext` both return OK. Device slots called: 14 (5), 15 (26), 18 (10),
    19 (16), **27
    `CreateCommittedResource` (4)**, **38 `GetCopyableFootprints` (4)**, 76 (10); heap slot 9 (26). So the console
    device also serves the PC `CreateCommittedResource` at slot 27, and slots 14, 15, 18, 19, 27 and 38 keep their PC
    positions for this client. No compute pipeline or root signature is created at context creation.
  - On that context, one `ffxDispatch` recorded into a logging command list measures the **console command list**:
    slots 14, 15, 25 and 26 are at their PC positions, then `SetDescriptorHeaps` is at 32 (PC 28), the compute root
    setters four slots later (33, 35, 41; two of them inferred from their shapes), `ExecuteIndirect` two slots later
    (61), and there are console-only methods at 85 and 141. The list object also carries a write cursor and limits
    (`+0x18`, `+0x28`, `+0x30`) the DLL writes commands through while there is room. Table and evidence:
    [d3d12_command_objects.md §3.5a](d3d12_command_objects.md).
    Device slot 67 is a console-only compute-pipeline call, `DeserializeComputePipelineStateX` in the DLL's own
    messages, not the PC method at that index; the proxy routes it to `CreateComputePipelineState`. The dispatch stops where the PC runtime
    refuses the DLL's compute code at slot 62: the DLL's shaders carry an `ACCC` part instead of DXIL, an `FSPX`
    payload with an RDNA2 program ([fspx.md](fspx.md)). The Vulkan device takes both forms at
    `CreateComputePipelineState`: the `ACCC` container slot 62 gets, and the `FSPX` payload the proxy builds from the
    slot-67 description's four pairs. It finds the translated module by `fspx_code_sha256` and `fspx_state_sha256` in
    `<shaders>/fspx/index.json`, which `make_game.py` writes ([fspx.md](fspx.md), "Build and run-time lookup";
    `d3d12_fspx_compute_test`). The pipeline answers the console slot 11 the DLL calls afterwards with 0, so no
    compute scratch buffer is made ([d3d12_command_objects.md](d3d12_command_objects.md) §3.5a). Not run here: the
    DLL's own dispatch on this device (`fsr_dispatch_test` needs MSVC structured exception handling).
  - Its `throw 1` reaches the caller. Before #965 the throw carried an image base of 0, because `_CxxThrowException`
    did not know manually mapped images; with #965 it carries the DLL's base, and the DLL still does not catch it on
    that path (measured with `fsr_dispatch_test` and `LEONIDA_FSR_NO_STANDIN=1`).
  - `xmem.dll!XMemGetWorkingSetStatistics` takes two arguments, `(0x20000001, stats*)`, at all seven call sites;
    the one-argument shim wrote through the constant. Fixed in `runtime/shims/xmem.cpp`.

### A.4 Other DLLs and the executable

- `amd_fidelityfx_loader_xbox.dll`, `Bink2Scarlett.dll`, `libHttpClient.GDK.dll`, `XCurl.dll`: no `d3d12_xs`
  imports.
- The package layout XML, chunk 1001, lists `game_scarlett_master.exe`; the file is absent from the folder. No
  claim in this document depends on the executable's contents.
- The repository's existing import survey (`docs/design.md`, "Baseline already recorded") records that the 16
  bundled DLLs need only `xmem` and the two `d3d12_xs` root-signature calls; our scan of the six non-CRT DLLs is
  consistent.

## Appendix B: interface versions seen in shipped binaries

| Interface | Evidence | Public methods beyond the previous version |
|---|---|---|
| `ID3D12Device11` | `XboxBvhBuilder.dll` constructor and rehydration signatures | `CreateSampler2` (public SDK) |
| `ID3D12GraphicsCommandList4` | standard BVH build/copy/validation signatures | `BeginRenderPass`/`EndRenderPass`, `BuildRaytracingAccelerationStructure`, `EmitRaytracingAccelerationStructurePostbuildInfo`, `CopyRaytracingAccelerationStructure`, `DispatchRays`, `SetPipelineState1` |
| `ID3D12GraphicsCommandList6` | batched BVH build signature | `DispatchMesh` |
| `ID3D12GraphicsCommandList8` | batched rehydration signatures | `OMSetFrontAndBackStencilRef` |

The public Xbox graphics index retrieved for this document lists interfaces up to `ID3D12Device10` and
`ID3D12GraphicsCommandList7`; the shipped DLL names `Device11` and `CommandList8`, which the index does not cover.
This is why the layer targets the Windows SDK's newer public chain rather than assuming the Xbox public index is
complete.
