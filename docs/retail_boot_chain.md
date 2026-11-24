# The retail boot chain: what stops `gta6rt --run`, in order

**Status: measured on one machine, 2026-11-23. A measurement for RT-5, not a claim; RT-5 stays with its owners
on the coordination issue.** Each row is the first failure with every fix above it in place. Nothing here quotes
the executable: no addresses, offsets, IID values or bytes; only export names, public SDK types, vtable slot
numbers, call shapes and counts. [boot.md](boot.md) has the runbook that produced the traces; #6715 maps the same
executable under `--dry-run`.

Setup: `main` `1cc6e3c12` with #6709, #4531, #6582, #4313, #4348 and #2490 merged locally, plus the local
stand-ins named below (not pull requests yet). Own 1.0.2.1 copy, Windows 11, RTX 4090 (driver 617.42),
`LEONIDA_D3D12=vulkan`, MSVC 2022 `RelWithDebInfo` build.

## 1. The chain

| # | Stop | What clears it |
|---|---|---|
| 1 | The first `XMemVirtualAlloc` calls: two 64 KiB reserve forms, then 2 MiB pages | #6709, #4531 |
| 2 | The 2 MiB reserve+commit calls carry attribute words the shim refuses, and `PAGE_GRAPHICS_*` bits (0x800..0x40000) on top of `PAGE_READWRITE` | Local: accept the attribute word, drop the graphics bits and keep the CPU protection. All seven startup allocations then succeed. #6843 measures the attribute words; no PR changes the check yet |
| 3 | `xgameruntime.lib`'s thunk treats a missing `XErrorReport` as fatal | #6582 |
| 4 | Access violation in the title's allocator: its `/GA` code reads its TLS from module index 0 | #4313 |
| 5 | Fatal assert when `DStorageGetFactory` returns null | #4348 |
| 6 | `XMemAllocatePhysicalPages` unimplemented; the title asserts on FALSE | Local stand-in, §2 |
| 7 | `D3D12XboxCreateDevice` succeeds; device slot 26 is called with its arguments swapped | Local, §3.1 |
| 8 | Device slot 48 is forwarded to the PC slot 48 and fails | Local, §3.2 |
| 9 | Device slot 73 is forwarded to the PC slot 73; the title then reads through a null list | #2490's list proxy plus local routing, §3.3 |
| 10 | An `XMemVirtualAlloc` reserve of about 185 MiB with type `0x42000`, attribute 7 and `PAGE_NOACCESS` is refused with `ERROR_INVALID_PARAMETER`. `0x40000` is `MEM_RESERVE_PLACEHOLDER` on the PC, which `VirtualAlloc` refuses and only `VirtualAlloc2` accepts; stop 11 supports the same reading on the console | Local: reserve it with `VirtualAlloc2(MEM_RESERVE \| MEM_RESERVE_PLACEHOLDER)` |
| 11 | The title carves five sub-ranges out of that placeholder with `MEM_RESERVE \| MEM_REPLACE_PLACEHOLDER \| MEM_PHYSICAL \| MEM_LARGE_PAGES` (`0x20406000`) at addresses inside it, then allocates a second batch of physical pages | Local: a physical reserve inside a known placeholder stays a placeholder (the page maps split it); other `MEM_REPLACE_PLACEHOLDER` calls split the range and allocate it in place |
| 12 | Device slot 133 is called; the proxy table had 128 slots, so the call went through a null pointer | Local: a 192-slot table, and a slot 133 stand-in, §3.4 |
| 13 | **Current stop**: right after the first `CreateDescriptorHeap`, the title calls the heap's slot 9 (`GetCPUDescriptorHandleForHeapStart` on the PC), takes the handle from `rax` and asserts that it equals a value it already holds. Hypothesis: a console descriptor heap starts at an address the title chose or can predict, which the PC handle doesn't match | Open: a question for the descriptor model ([boot.md](boot.md) Q4) |

At stop 13 the D3D12 trace holds 1,185 calls: the device, 5 command queues, 5 fences, 7 command allocators,
581 command lists created and closed, a shader cache session, `XboxIDXGIDevice::GetAdapter`, one failed
`ID3D12Device10::CreateReservedResource2`, the first `CreateSampler2`, one committed resource (the slot 133
stand-in) and the first descriptor heap.

Without a debugger, the unhandled `int3` of a fatal assert ends the process with `C0000374` (heap corruption) or
`C0000005` during teardown, not `80000003`. An exit code of `C0000374` from `--run` can be such an assert, not a
heap bug.

## 2. Physical pages (`xmem.dll`)

The shapes, from the call sites and the checks that follow them:

| Export | Shape |
|---|---|
| `XMemAllocatePhysicalPages` | `BOOL (ULONG64 attributes, ULONG_PTR* page_count /*in/out, 64 KiB pages*/, ULONG_PTR* page_array)`; the title checks both TRUE and the count written back. Attributes seen: `0x280`, `0x380` |
| `XMemVirtualAlloc` | the range for those pages is reserved with `MEM_RESERVE \| MEM_PHYSICAL \| MEM_LARGE_PAGES` (`0x20402000`) and no `MEM_COMMIT` |
| `XMemMapPhysicalPages` | `(va, page_count, page_array)`; a null `page_array` unmaps; non-null return on success |
| `XMemMapPhysicalPagesScatter` | `(va_array, page_count, page_array)`, the same with one VA per page |
| `XMemFreePhysicalPages` | `(page_count, page_array)` |

The local stand-in that gets past stop 6 is CPU-only: one pagefile section per 64 KiB page
(`CreateFileMappingW(INVALID_HANDLE_VALUE, …, SEC_COMMIT)`), page numbers handed out as indexes into that table,
the physical reserve made with `VirtualAlloc2(MEM_RESERVE | MEM_RESERVE_PLACEHOLDER)`, each map a
`VirtualFree(MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)` split followed by `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)`,
and each unmap `UnmapViewOfFile2(MEM_PRESERVE_PLACEHOLDER)`. #5381 has the CPU half of physical pages with dirty
tracking and #5378 a Vulkan sparse-binding mapper; neither is wired to these exports.

## 3. Device calls whose Xbox shape differs from the PC

### 3.1 Struct-returning methods take the return buffer first

Device slot 26 (`GetCustomHeapProperties`, which returns a `D3D12_HEAP_PROPERTIES`) is called as
`(return buffer, this, node_mask, heap_type)`: the first register holds the return buffer, the second the device
proxy. The PC MSVC method expects `(this, return buffer, …)`. A proxy thunk can tell the two apart: in the swapped
call the `this` register does not point at the proxy, and the second argument does. Calling the real method with
the two exchanged and returning the buffer pointer gets past it. Other struct-returning slots probably follow the
same rule; only slot 26 was reached.

### 3.2 Device slot 48 creates a command queue

Xbox device slot 48 is `(desc, riid, out)` with `riid = IID_ID3D12CommandQueue`. `desc` starts with a PC
`D3D12_COMMAND_QUEUE_DESC` (Type, Priority, Flags, NodeMask) followed by Xbox words. On the PC, slot 48 is
`OpenExistingHeapFromAddress`, so forwarding by number fails. Creating the queue from the 16-byte PC prefix, with
`Flags` limited to `D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT`, through the existing queue proxy works: four
queues come from this slot, the first of them a compute queue. The first queue comes from slot 8, as on the PC.

### 3.3 Device slot 73 creates a command list

Xbox device slot 73 is `(desc, allocator, pso, riid, out)`. `desc` is 16 bytes and its first word reads as the
`D3D12_COMMAND_LIST_TYPE`. `riid` is a console command-list IID that is not in the PC SDK (#3520 records that the
package doesn't export it). The shim's unknown-IID log names it on the player's machine. Creating an
`ID3D12GraphicsCommandList` of that type and handing out #2490's console list
proxy gets all 581 lists created and closed.

### 3.4 Device slot 133 places a resource at a virtual address

Xbox device slot 133 is `(va, const D3D12_RESOURCE_DESC*, state, clear, riid, out)` with `riid = IID_ID3D12Resource`
and `state = 0xFFFFFFFF`, which isn't a valid PC state. At the first call, `va` is the start of a physical-page
range the title had just reserved, and `desc` is a buffer of that range's size with Xbox bits in `Flags` above
the PC flags. The shape matches the console `CreatePlacedResourceX`. The local stand-in creates a committed
DEFAULT-heap buffer from the desc, with `Flags` limited to the PC bits and `D3D12_RESOURCE_STATE_COMMON`. It is
not tied to the VA, so what the title writes there doesn't reach the GPU. A real implementation needs the page
mapper (#5378) or an imported host allocation.

The proxy's slot table must also be longer than the title's highest slot. A shorter table leaves a null pointer,
not an `E_NOTIMPL`.

## 4. What this does not show

These are one run's first failures, not a list of what is missing. Every stand-in above is local and untested
beyond "the title goes on to the next call". Nothing has reached a draw or a present yet.
