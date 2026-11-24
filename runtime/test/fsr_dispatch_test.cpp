// Optional Windows test: map the package's amd_fidelityfx_upscaler_xbox.dll (FSR) through rt::Loader, bring it up
// (Loader::initialise: TLS and DllMain), and drive it through the public FidelityFX API on a PC D3D12
// device wrapped in the Xbox-ABI device proxy. It records what the DLL answers and which device slots it calls.
//
//   LEONIDA_GAME_DIR=<package root>   (contains amd_fidelityfx_upscaler_xbox.dll)
//   LEONIDA_FSR_WARP=1                (optional: use the WARP adapter instead of the default hardware adapter)
//   LEONIDA_FSR_DEVICE=vulkan         (optional: use the Vulkan-backed ID3D12Device instead of a PC device)
//
// Without LEONIDA_GAME_DIR it prints "SKIP (no package)" and exits 77 (CTest skip). Nothing from the DLL is written to disk.
//
// Steps, each one only if the previous worked:
//   1. map: the console imports bind to the shims and nothing is unimplemented
//   2. Loader::initialise(): TLS and DllMain(DLL_PROCESS_ATTACH)
//   3. ffxConfigure(global debug) with a message callback, so the DLL's own messages are printed
//   4. CPU-only queries (upscale ratio, render resolution): do the PC struct type ids work on the Xbox build?
//   5. ffxQuery(GET_VERSIONS) for the upscaler on the proxied device
//   6. ffxCreateContext(upscale + version + backend) on the proxied device; the device/heap slots it hit are listed
//   7. if the driver refused a texture in step 6, the same with one PC stand-in (see standin_slot76); then, on the
//      context, one ffxDispatch recorded into a logging command list (ListProxy, never executed): the console list
//      slots it calls, in order, with their first argument words (docs/d3d12_command_objects.md section 3.5a)
// Calls into the DLL run under SEH, so a fault inside it is reported as a result, not a crash: where it happened, and
// for the DLL's own HRESULT check (it formats the code with FormatMessageW, then throws an int) the HRESULT and the
// DLL frames that made the failing call.
//
// On 1.0.2.1 steps 1-5 work. In step 6 the measured Xbox backend descriptor (type 0x01000002, device at +0x10, see
// docs/d3d12_vulkan.md Appendix A.3) gets the DLL into its D3D12 backend: it creates descriptor heaps, views and
// resources through the proxy. On an RTX 2000 Ada (driver 597.06) it then stops at its HRESULT check, because the
// driver refuses one of its textures (DXGI_FORMAT_R9G9B9E5_SHAREDEXP with ALLOW_UNORDERED_ACCESS); on WARP the
// context is created as is. The other backend ids are controls: they end in FFX_API_RETURN_ERROR before the device
// is touched. In step 7 the dispatch is recorded up to the DLL's compute pipeline, which a PC device cannot create
// (slot 67 and 62 codes are not PC shaders). The INFO lines are the measurement; the
// PASS/FAIL checks only cover what is known to work.
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>

#include "../shims/d3d12_xs_device.h"
#ifdef LEONIDA_VULKAN_DEVICE
#include "../shims/d3d12_com.h"
#endif
#include "../shims/shims.h"
#include "../src/pe_loader.h"
#include "ffx_api_min.h"

namespace {

int g_failures = 0;
int g_messages = 0;

void check(bool condition, const std::string& what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what.c_str());
    if (!condition) ++g_failures;
}

void note(const std::string& what) { std::printf("INFO %s\n", what.c_str()); }

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

std::string hex(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof(b), "0x%llx", static_cast<unsigned long long>(v));
    return b;
}

void on_message(uint32_t type, const wchar_t* message) {
    ++g_messages;
    std::printf("  ffx %s: %ls\n", type == 0 ? "error" : "warning", message ? message : L"(null)");
}

// SEH wrappers: no C++ objects with destructors in these frames.
struct Call {
    bool faulted = false;
    DWORD code = 0;
    ffx::ReturnCode rc = 0;
    const void* at = nullptr;     // faulting instruction
    ULONG_PTR access = 0;         // access violation: 0 read, 1 write, 8 execute
    ULONG_PTR address = 0;        // access violation: the address touched
    ULONG_PTR throw_info = 0;     // MSVC C++ exception: the ThrowInfo of the thrown type
    ULONG_PTR throw_image = 0;    // MSVC C++ exception: the image base _CxxThrowException found for the thrower
    bool has_thrown_int = false;  // the first 4 bytes of the thrown object
    int32_t thrown_int = 0;
    void* stack[48] = {};         // C++ exception: return addresses unwound from the throwing context
    USHORT frames = 0;
};

constexpr DWORD kMsvcCppException = 0xe06d7363;

// A word at an address that may not be mapped; ok says whether it could be read.
uint64_t read_word(uint64_t address, bool& ok) {
    ok = false;
    if (address < 0x10000) return 0;
    uint64_t v = 0;
    __try {
        v = *reinterpret_cast<const volatile uint64_t*>(address);
        ok = true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    return v;
}

// Return addresses from a context, unwound with the registered unwind tables. RtlCaptureStackBackTrace stops at the
// first frame of a manually mapped image; RtlLookupFunctionEntry also finds the tables the loader registered.
USHORT walk_stack(CONTEXT context, void** out, USHORT max) {
    USHORT n = 0;
    while (n < max && context.Rip) {
        DWORD64 image_base = 0;
        PRUNTIME_FUNCTION entry = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
        if (entry) {
            void* handler_data = nullptr;
            DWORD64 establisher = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, entry, &context, &handler_data,
                             &establisher, nullptr);
        } else {  // a leaf function: the return address is on top of the stack
            bool ok = false;
            context.Rip = read_word(context.Rsp, ok);
            if (!ok) break;
            context.Rsp += 8;
        }
        if (context.Rip) out[n++] = reinterpret_cast<void*>(context.Rip);
    }
    return n;
}

USHORT walk_this_stack(void** out, USHORT max) {
    CONTEXT context{};
    RtlCaptureContext(&context);
    return walk_stack(context, out, max);
}

int capture_fault(const EXCEPTION_POINTERS* ep, Call& c) {
    c.code = ep->ExceptionRecord->ExceptionCode;
    c.at = ep->ExceptionRecord->ExceptionAddress;
    if (c.code == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2) {
        c.access = ep->ExceptionRecord->ExceptionInformation[0];
        c.address = ep->ExceptionRecord->ExceptionInformation[1];
    }
    if (c.code == kMsvcCppException && ep->ExceptionRecord->NumberParameters >= 4) {
        c.throw_info = ep->ExceptionRecord->ExceptionInformation[2];
        c.throw_image = ep->ExceptionRecord->ExceptionInformation[3];
        // The thrown object lives until the handler runs, so it can be read from the filter.
        if (const auto* object = reinterpret_cast<const int32_t*>(ep->ExceptionRecord->ExceptionInformation[1])) {
            c.thrown_int = *object;
            c.has_thrown_int = true;
        }
        c.frames = walk_stack(*ep->ContextRecord, c.stack, static_cast<USHORT>(std::size(c.stack)));
    }
    return EXCEPTION_EXECUTE_HANDLER;
}

// Each guard returns a copy of `c`, not `c` itself: with `return c;` the local becomes the named return value, and
// clang-cl 19 and 20 crash (exit 139, in recoverAddrOfEscapedLocal) generating the __except filter that writes it,
// which stops the Linux cross-build (runtime/cmake/clang-cl-xwin.cmake). MSVC is fine either way.
Call guarded_configure(ffx::PfnConfigure fn, ffx::Context* ctx, const ffx::Header* desc) {
    Call c;
    __try {
        c.rc = fn(ctx, desc);
    } __except (capture_fault(GetExceptionInformation(), c)) {
        c.faulted = true;
    }
    return Call(c);
}

Call guarded_query(ffx::PfnQuery fn, ffx::Context* ctx, ffx::Header* desc) {
    Call c;
    __try {
        c.rc = fn(ctx, desc);
    } __except (capture_fault(GetExceptionInformation(), c)) {
        c.faulted = true;
    }
    return Call(c);
}

Call guarded_create(ffx::PfnCreateContext fn, ffx::Context* ctx, ffx::Header* desc) {
    Call c;
    __try {
        c.rc = fn(ctx, desc, nullptr);
    } __except (capture_fault(GetExceptionInformation(), c)) {
        c.faulted = true;
    }
    return Call(c);
}

Call guarded_destroy(ffx::PfnDestroyContext fn, ffx::Context* ctx) {
    Call c;
    __try {
        c.rc = fn(ctx, nullptr);
    } __except (capture_fault(GetExceptionInformation(), c)) {
        c.faulted = true;
    }
    return Call(c);
}

Call guarded_dispatch(ffx::PfnDispatch fn, ffx::Context* ctx, const ffx::Header* desc) {
    Call c;
    __try {
        c.rc = fn(ctx, desc);
    } __except (capture_fault(GetExceptionInformation(), c)) {
        c.faulted = true;
    }
    return Call(c);
}

// PC stand-in for one console capability. The console build creates a DXGI_FORMAT_R9G9B9E5_SHAREDEXP texture with
// ALLOW_UNORDERED_ACCESS; some PC drivers refuse it (E_INVALIDARG on an RTX 2000 Ada; WARP takes it), which stops
// the DLL's context creation. While
// installed, device slot 76 creates such a texture as R32_UINT (the same 32 bits per texel), so the test can measure
// what the DLL does after it. Installed by swapping the proxy's vtable pointer, and removed again before the proxy's
// own helpers run, because they recognise the proxy by that pointer.
using Slot76Fn = HRESULT(WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC*,
                                  D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*, const IID*, void**);
Slot76Fn g_slot76 = nullptr;
int g_standins = 0;
void* g_standin_vtable[128];
void** g_proxy_vtable = nullptr;

HRESULT WINAPI standin_slot76(void* self, const D3D12_HEAP_PROPERTIES* heap, D3D12_HEAP_FLAGS heap_flags,
                              const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES state,
                              const D3D12_CLEAR_VALUE* clear, const IID* riid, void** out) {
    if (desc && desc->Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP &&
        (desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
        D3D12_RESOURCE_DESC copy = *desc;  // the proxy itself reads these 56 bytes
        copy.Format = DXGI_FORMAT_R32_UINT;
        ++g_standins;
        return g_slot76(self, heap, heap_flags, &copy, state, clear, riid, out);
    }
    return g_slot76(self, heap, heap_flags, desc, state, clear, riid, out);
}

// Device slots 67 and 62 are the DLL's two compute-pipeline paths (calls at RVAs 0x56d9 and 0x5825): slot 67 from
// native code, which the proxy refuses, then slot 62 with a D3D12_COMPUTE_PIPELINE_STATE_DESC. This wrapper records
// slot 62's HRESULT: on failure the DLL's pipeline function returns 0x8000000d (RVA 0x4b85), but the dispatch goes
// on and records the null pipeline.
using Slot62Fn = HRESULT(WINAPI*)(void*, const void*, uint64_t, uint64_t, const IID*, void**);
Slot62Fn g_slot62 = nullptr;
int g_slot62_calls = 0;
HRESULT g_slot62_hr = S_OK;

HRESULT WINAPI standin_slot62(void* self, const void* desc, uint64_t w2, uint64_t w3, const IID* riid, void** out) {
    ++g_slot62_calls;
    g_slot62_hr = g_slot62(self, desc, w2, w3, riid, out);
    return g_slot62_hr;
}

// The slot-62 recorder always; the slot-76 stand-in only when the driver refused the texture.
void install_standins(void* proxy, unsigned slots, bool substitute_texture) {
    g_proxy_vtable = *static_cast<void***>(proxy);
    for (unsigned i = 0; i < slots && i < std::size(g_standin_vtable); ++i) g_standin_vtable[i] = g_proxy_vtable[i];
    g_slot76 = reinterpret_cast<Slot76Fn>(g_proxy_vtable[76]);
    if (substitute_texture) g_standin_vtable[76] = reinterpret_cast<void*>(&standin_slot76);
    g_slot62 = reinterpret_cast<Slot62Fn>(g_proxy_vtable[62]);
    g_standin_vtable[62] = reinterpret_cast<void*>(&standin_slot62);
    *static_cast<void***>(proxy) = g_standin_vtable;
}

void remove_standins(void* proxy) {
    if (g_proxy_vtable) *static_cast<void***>(proxy) = g_proxy_vtable;
    g_proxy_vtable = nullptr;
}

// A logging command list: the DLL was compiled against the console's command list, so each method it calls is
// counted by vtable slot and recorded with its first argument words. Only slots whose console meaning is measured
// are forwarded to the PC list (a wrong slot is worse than a clean refusal; the list is never executed):
//   - IUnknown (0..2), and 14 Dispatch, 15 CopyBufferRegion, 25 SetPipelineState and 26 ResourceBarrier at the same
//     index (this DLL's run and recorder, and the BVH builder's 14, 25 and 26);
//   - 32 as SetDescriptorHeaps (PC slot 28), measured here: it gets (1, &heap) with a device-proxy heap, whose PC heap
//     is passed on.
//   - the console slots from this DLL's recorder (RVA 0x59a0) and trace, at their PC slot: 33
//     SetComputeRootSignature (29, inferred: one pointer argument), 35 SetComputeRootDescriptorTable (31),
//     41 SetComputeRootConstantBufferView (37, inferred: an index and a 64-bit GPU VA from the constant allocator),
//     53 ClearUnorderedAccessViewUint (49), 61 ExecuteIndirect (59).
// Everything else returns 0 without a PC call. Forwarded calls keep integer/pointer registers, the hidden
// struct-return pointer and stack words, and do not touch XMM registers, as in the device proxy.
//
// The console list is also wider than its vtable: before a call the DLL checks a write cursor at +0x18 against a
// limit (+0x30 for root arguments, +0x28 for a dispatch) and, while there is room, writes the command into that
// buffer itself (headers from the per-root-parameter table at +0x344) instead of calling. With the cursor at the
// limit every command takes the vtable, which is what this proxy measures. The DLL also stores the pipeline's
// 32 bytes at +0x544 and flags at +0x140..+0x143, so the object reserves that room.
constexpr unsigned kListSlots = 160;
constexpr unsigned kPcListSlots = 82;  // ID3D12GraphicsCommandList8: slot 81 is OMSetFrontAndBackStencilRef
constexpr unsigned kConsoleSetDescriptorHeaps = 32;
constexpr unsigned kNotForwarded = ~0u;

constexpr unsigned console_to_pc_list_slot(unsigned slot) {
    switch (slot) {
        case 0: case 1: case 2: case 14: case 15: case 25: case 26: return slot;
        case 33: return 29;  // SetComputeRootSignature
        case 35: return 31;  // SetComputeRootDescriptorTable
        case 41: return 37;  // SetComputeRootConstantBufferView
        case 53: return 49;  // ClearUnorderedAccessViewUint
        case 61: return 59;  // ExecuteIndirect
        default: return kNotForwarded;
    }
}
std::atomic<uint64_t> g_list_hits[kListSlots];

// The first calls in order, with their first three argument words: the order says which call a driver fault follows.
struct ListCall {
    unsigned slot;
    uint64_t a1, a2, a3;
    uint64_t at_a2;          // the first word a2 points at, when a2 is a readable pointer
    bool at_a2_is_heap;      // ...and that word is one of the device proxy's descriptor heaps
};

ListCall g_list_trace[256];
std::atomic<unsigned> g_list_trace_count{0};

struct ListProxy {
    void** vtable;
    ID3D12GraphicsCommandList* real;
    alignas(8) uint8_t console_fields[0x600];  // +0x10..: the fields the DLL reads and writes past the vtable

    ListProxy(void** v, ID3D12GraphicsCommandList* r) : vtable(v), real(r), console_fields{} {
        // Cursor (+0x18) == both limits (+0x28, +0x30): no room, so every command goes through the vtable.
        const uint64_t at_limit = reinterpret_cast<uint64_t>(&console_fields[0x500]);
        for (const size_t offset : {0x18, 0x28, 0x30})
            std::memcpy(reinterpret_cast<uint8_t*>(this) + offset, &at_limit, sizeof(at_limit));
    }
};
static_assert(offsetof(ListProxy, console_fields) == 0x10);

template <unsigned Slot>
uint64_t WINAPI list_slot(ListProxy* self, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5,
                          uint64_t a6, uint64_t a7, uint64_t a8, uint64_t a9, uint64_t a10, uint64_t a11,
                          uint64_t a12, uint64_t a13, uint64_t a14) {
    g_list_hits[Slot].fetch_add(1, std::memory_order_relaxed);
    if (const unsigned i = g_list_trace_count.fetch_add(1, std::memory_order_relaxed); i < std::size(g_list_trace)) {
        bool readable = false, word_is_pointer = false;
        const uint64_t word = read_word(a2, readable);
        if (readable) read_word(word, word_is_pointer);  // is_heap_proxy reads through the word
        g_list_trace[i] = {Slot, a1, a2, a3, readable ? word : 0,
                           word_is_pointer && rt::d3d12_xs::is_heap_proxy(reinterpret_cast<const void*>(word))};
    }
    if constexpr (Slot == 0) {  // QueryInterface: stay behind the proxy for every interface the PC list has
        if (!a1 || !a2) return static_cast<uint64_t>(static_cast<uint32_t>(E_POINTER));
        void* found = nullptr;
        const HRESULT hr = self->real->QueryInterface(*reinterpret_cast<const IID*>(a1), &found);
        *reinterpret_cast<void**>(a2) = nullptr;
        if (SUCCEEDED(hr) && found) {
            static_cast<IUnknown*>(found)->Release();
            self->real->AddRef();
            *reinterpret_cast<void**>(a2) = self;
        }
        return static_cast<uint64_t>(static_cast<uint32_t>(hr));
    } else if constexpr (Slot == kConsoleSetDescriptorHeaps) {  // SetDescriptorHeaps(count, heaps)
        const auto count = static_cast<UINT>(a1);
        auto* const* heaps = reinterpret_cast<ID3D12DescriptorHeap* const*>(a2);
        ID3D12DescriptorHeap* real_heaps[8] = {};
        for (UINT i = 0; i < count && i < 8; ++i)
            real_heaps[i] = rt::d3d12_xs::is_heap_proxy(heaps[i]) ? rt::d3d12_xs::heap_proxy_real_heap(heaps[i])
                                                                   : heaps[i];
        self->real->SetDescriptorHeaps(count < 8 ? count : 8, real_heaps);
        return 0;
    } else if constexpr (console_to_pc_list_slot(Slot) == kNotForwarded) {  // not measured: recorded only
        return 0;
    } else {
        using ForwardFn = uint64_t(WINAPI*)(void*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                            uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                            uint64_t);
        auto fn = reinterpret_cast<ForwardFn>(
            (*reinterpret_cast<void***>(self->real))[console_to_pc_list_slot(Slot)]);
        return fn(self->real, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14);
    }
}

template <std::size_t... I>
void fill_list_vtable(void** vtable, std::index_sequence<I...>) {
    ((vtable[I] = reinterpret_cast<void*>(&list_slot<static_cast<unsigned>(I)>)), ...);
}

void** list_vtable() {
    static void* storage[kListSlots];
    static const bool filled = (fill_list_vtable(storage, std::make_index_sequence<kListSlots>{}), true);
    (void)filled;
    return storage;
}

// The PC ID3D12GraphicsCommandList method at a slot (d3d12.h order), for reading the hit list.
const char* list_slot_name(unsigned slot) {
    static const char* const names[] = {
        "QueryInterface", "AddRef", "Release", "GetPrivateData", "SetPrivateData", "SetPrivateDataInterface",
        "SetName", "GetDevice", "GetType", "Close", "Reset", "ClearState", "DrawInstanced", "DrawIndexedInstanced",
        "Dispatch", "CopyBufferRegion", "CopyTextureRegion", "CopyResource", "CopyTiles", "ResolveSubresource",
        "IASetPrimitiveTopology", "RSSetViewports", "RSSetScissorRects", "OMSetBlendFactor", "OMSetStencilRef",
        "SetPipelineState", "ResourceBarrier", "ExecuteBundle", "SetDescriptorHeaps", "SetComputeRootSignature",
        "SetGraphicsRootSignature", "SetComputeRootDescriptorTable", "SetGraphicsRootDescriptorTable",
        "SetComputeRoot32BitConstant", "SetGraphicsRoot32BitConstant", "SetComputeRoot32BitConstants",
        "SetGraphicsRoot32BitConstants", "SetComputeRootConstantBufferView", "SetGraphicsRootConstantBufferView",
        "SetComputeRootShaderResourceView", "SetGraphicsRootShaderResourceView",
        "SetComputeRootUnorderedAccessView", "SetGraphicsRootUnorderedAccessView", "IASetIndexBuffer",
        "IASetVertexBuffers", "SOSetTargets", "OMSetRenderTargets", "ClearDepthStencilView", "ClearRenderTargetView",
        "ClearUnorderedAccessViewUint", "ClearUnorderedAccessViewFloat", "DiscardResource", "BeginQuery", "EndQuery",
        "ResolveQueryData", "SetPredication", "SetMarker", "BeginEvent", "EndEvent", "ExecuteIndirect",
    };
    if (slot < std::size(names)) return names[slot];
    return slot < kPcListSlots ? "GraphicsCommandList1-8" : "past the PC list";
}

// What a console list slot is, by the measured map: the PC method it forwards to, or that it is not measured.
std::string console_list_slot(unsigned slot) {
    if (slot == kConsoleSetDescriptorHeaps) return "SetDescriptorHeaps, PC 28";
    const unsigned pc = console_to_pc_list_slot(slot);
    if (pc == kNotForwarded) return "not measured; not forwarded";
    return std::string(list_slot_name(pc)) + ", PC " + std::to_string(pc);
}

void print_list_hits() {
    std::string hits;
    for (unsigned s = 0; s < kListSlots; ++s)
        if (const uint64_t n = g_list_hits[s].load(std::memory_order_relaxed))
            hits += "\n    " + std::to_string(s) + " (+" + hex(s * 8ull) + ": " + console_list_slot(s) + ") x" +
                    std::to_string(n);
    note("console command-list slots hit (slot, vtable offset, meaning by the measured map, count):" +
         (hits.empty() ? std::string(" none") : hits));
    std::string trace;
    const unsigned n = g_list_trace_count.load(std::memory_order_relaxed);
    for (unsigned i = 0; i < n && i < std::size(g_list_trace); ++i)
        trace += "\n    " + std::to_string(i) + ": slot " + std::to_string(g_list_trace[i].slot) + " (" +
                 console_list_slot(g_list_trace[i].slot) + ") " + hex(g_list_trace[i].a1) + " " +
                 hex(g_list_trace[i].a2) + " " + hex(g_list_trace[i].a3) +
                 (g_list_trace[i].at_a2_is_heap ? "  [a2 points at a device-proxy descriptor heap]" : "");
    note("command-list calls in order (first three argument words):" + (trace.empty() ? std::string(" none") : trace));
}

// Counting wrappers around the xmem shims, patched into this DLL's IAT only, so the test sees how far a call gets.
using XMemAllocFn = void*(__stdcall*)(size_t, uint64_t);
using XMemFreeFn = void(__stdcall*)(void*, uint64_t);
XMemAllocFn g_xmem_alloc = nullptr;
XMemFreeFn g_xmem_free = nullptr;
uint64_t g_alloc_calls = 0, g_alloc_bytes = 0, g_free_calls = 0;

void* __stdcall counting_alloc(size_t size, uint64_t attributes) {
    ++g_alloc_calls;
    g_alloc_bytes += size;
    return g_xmem_alloc(size, attributes);
}
void __stdcall counting_free(void* p, uint64_t attributes) {
    ++g_free_calls;
    g_xmem_free(p, attributes);
}

// The DLL's HRESULT check (a FormatMessageW of the code, then `throw 1`) tells nothing outside the DLL, so the
// FormatMessageW import is wrapped too: the first code it formats is the failing HRESULT, and the stack at that
// point names the DLL function that made the failing call.
using FormatMessageWFn = DWORD(WINAPI*)(DWORD, LPCVOID, DWORD, DWORD, LPWSTR, DWORD, va_list*);
FormatMessageWFn g_format_message = nullptr;
int g_formatted = 0;
DWORD g_first_formatted = 0;
void* g_first_formatted_stack[24] = {};
USHORT g_first_formatted_frames = 0;

DWORD WINAPI recording_format_message(DWORD flags, LPCVOID source, DWORD id, DWORD language, LPWSTR buffer,
                                      DWORD size, va_list* arguments) {
    if (!g_formatted++) {
        g_first_formatted = id;
        g_first_formatted_frames =
            walk_this_stack(g_first_formatted_stack, static_cast<USHORT>(std::size(g_first_formatted_stack)));
    }
    return g_format_message(flags, source, id, language, buffer, size, arguments);
}

void** iat_slot(rt::Module& m, const char* want_dll, const char* want_name) {
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(m.base);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(m.base + dos->e_lfanew);
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    for (auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(m.base + dir.VirtualAddress); d->Name; ++d) {
        if (rt::lower(reinterpret_cast<const char*>(m.base + d->Name)) != want_dll) continue;
        auto* lookup = reinterpret_cast<IMAGE_THUNK_DATA64*>(m.base + (d->OriginalFirstThunk ? d->OriginalFirstThunk : d->FirstThunk));
        auto* iat = reinterpret_cast<IMAGE_THUNK_DATA64*>(m.base + d->FirstThunk);
        for (; lookup->u1.AddressOfData; ++lookup, ++iat) {
            if (IMAGE_SNAP_BY_ORDINAL64(lookup->u1.Ordinal)) continue;
            auto* n = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(m.base + static_cast<uint32_t>(lookup->u1.AddressOfData));
            if (std::string(want_name) == n->Name) return reinterpret_cast<void**>(&iat->u1.Function);
        }
    }
    return nullptr;
}

bool patch(void** slot, void* fn, void** old) {
    DWORD prot = 0;
    if (!slot || !VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &prot)) return false;
    *old = *slot;
    *slot = fn;
    VirtualProtect(slot, sizeof(void*), prot, &prot);
    return true;
}

// main's loader (#95) maps, sets up TLS and runs DllMain, but doesn't register the image's x64 unwind tables yet
// (#176 adds that). Both DLLs throw and catch C++ exceptions inside, which ends the process without them. Register
// .pdata here only when the loader hasn't, so the test keeps working once the loader does.
bool ensure_unwind_registered(rt::Module& m) {
    DWORD64 base = 0;
    if (RtlLookupFunctionEntry(reinterpret_cast<DWORD64>(m.base + m.entry_rva), &base, nullptr)) return true;
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(m.base + reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    const auto& ex = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
    return ex.Size && RtlAddFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(m.base + ex.VirtualAddress),
                                          ex.Size / sizeof(RUNTIME_FUNCTION), reinterpret_cast<DWORD64>(m.base));
}

// Loader::initialise() runs the module's TLS setup and DllMain (#95). SEH faults inside the DLL are reported here;
// C++ exceptions (a DllMain returning FALSE, a corrupt TLS directory) go on to the caller's catch.
DWORD guarded_initialise(rt::Loader* loader) {
    DWORD code = 0;
    __try {
        loader->initialise();
    } __except ((code = GetExceptionCode()) == 0xE06D7363 ? EXCEPTION_CONTINUE_SEARCH : EXCEPTION_EXECUTE_HANDLER) {
    }
    return code == 0xE06D7363 ? 0 : code;
}

// Where a fault happened, as an RVA in the mapped DLL when it is inside it (set once the DLL is mapped).
const uint8_t* g_dll_base = nullptr;
uint32_t g_dll_size = 0;

std::string where(const void* p) {
    const auto* b = static_cast<const uint8_t*>(p);
    if (g_dll_base && b >= g_dll_base && b < g_dll_base + g_dll_size)
        return "DLL RVA " + hex(static_cast<uint64_t>(b - g_dll_base));
    HMODULE module = nullptr;  // the test itself, the runtime or a host DLL: name it so a symbolizer can take the RVA
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(p), &module) &&
        GetModuleFileNameA(module, name, MAX_PATH)) {
        const char* base_name = std::strrchr(name, '\\');
        return std::string(base_name ? base_name + 1 : name) + "+" +
               hex(static_cast<uint64_t>(b - reinterpret_cast<const uint8_t*>(module)));
    }
    return "address " + hex(reinterpret_cast<uint64_t>(p));
}

std::string describe(const Call& c) {
    if (c.faulted) {
        std::string s = "faulted with exception " + hex(c.code) + " at " + where(c.at);
        if (c.code == EXCEPTION_ACCESS_VIOLATION)
            s += std::string(c.access == 1 ? ", writing " : c.access == 8 ? ", executing " : ", reading ") +
                 where(reinterpret_cast<const void*>(c.address));
        if (c.code == kMsvcCppException) {
            // On x64 the ThrowInfo is an RVA from the image base _CxxThrowException found (0 if it found none).
            s += ", C++ throw: image base " + hex(c.throw_image) + ", ThrowInfo " +
                 (c.throw_image ? where(reinterpret_cast<const void*>(c.throw_image + c.throw_info))
                                : "RVA " + hex(c.throw_info));
            if (c.has_thrown_int) s += ", first word of the thrown object " + hex(static_cast<uint32_t>(c.thrown_int));
            // The DLL frames on the throwing stack, innermost first: where in the DLL the throw comes from.
            std::string dll_frames;
            for (USHORT i = 0; i < c.frames; ++i) {
                const auto* f = static_cast<const uint8_t*>(c.stack[i]);
                if (g_dll_base && f >= g_dll_base && f < g_dll_base + g_dll_size)
                    dll_frames += " " + hex(static_cast<uint64_t>(f - g_dll_base));
            }
            if (!dll_frames.empty()) s += ", DLL return addresses (RVA, innermost first):" + dll_frames;
        }
        return s;
    }
    return std::string(ffx::return_code_name(c.rc)) + " (" + std::to_string(c.rc) + ")";
}

void* symbol(rt::Module& m, const char* name) {
    auto it = m.exports.find(name);
    return it == m.exports.end() ? nullptr : it->second;
}

// The PC device to wrap: the first hardware adapter, or WARP.
ID3D12Device* create_pc_device(bool warp) {
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return nullptr;
    IDXGIAdapter1* adapter = nullptr;
    if (warp) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
    } else {
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d{};
            adapter->GetDesc1(&d);
            if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) break;
            adapter->Release();
            adapter = nullptr;
        }
    }
    ID3D12Device* device = nullptr;
    if (adapter) {
        DXGI_ADAPTER_DESC1 d{};
        adapter->GetDesc1(&d);
        std::printf("adapter: %ls\n", d.Description);
        D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
        adapter->Release();
    }
    factory->Release();
    return device;
}

void print_slot_hits(const void* proxy) {
    // The proxy forwards the PC device's own slots and logs-and-fails beyond them; a margin covers those too.
    const unsigned n = rt::d3d12_xs::device_proxy_slot_count(proxy) + 32;
    std::string dev, heap;
    for (unsigned s = 0; s < n; ++s)
        if (auto h = rt::d3d12_xs::device_slot_hits(s)) dev += " " + std::to_string(s) + "x" + std::to_string(h);
    for (unsigned s = 0; s < 16; ++s)
        if (auto h = rt::d3d12_xs::heap_slot_hits(s)) heap += " " + std::to_string(s) + "x" + std::to_string(h);
    note("device slots hit (slot x count):" + (dev.empty() ? std::string(" none") : dev));
    note("descriptor-heap slots hit:" + (heap.empty() ? std::string(" none") : heap));
}

// The module's TLS index, whichever loader this builds against: #95 records a slot (has_tls_slot/tls_slot), #93
// a module TLS index (tls_index, kNoTlsIndex when there is none). ~0u means no TLS.
template <class M>
auto module_tls_index(const M& m, int) -> decltype(m.tls_index, uint32_t()) { return m.tls_index; }
template <class M>
auto module_tls_index(const M& m, long) -> decltype(m.tls_slot, uint32_t()) { return m.has_tls_slot ? m.tls_slot : ~0u; }

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep every line if the DLL takes the process down
    const std::string game_dir_arg = env("LEONIDA_GAME_DIR");
    if (game_dir_arg.empty()) {
        std::printf("SKIP (no package)\n");
        return 77;
    }
    const std::filesystem::path game_dir = game_dir_arg;
    const std::filesystem::path dll = game_dir / "amd_fidelityfx_upscaler_xbox.dll";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(dll, ec)) {
        std::printf("FAIL LEONIDA_GAME_DIR does not contain amd_fidelityfx_upscaler_xbox.dll\n");
        return 1;
    }

    try {
        // 1. Map.
        rt::Loader loader(game_dir);
        rt::Module& fsr = loader.load(dll);
        g_dll_base = fsr.base;
        g_dll_size = fsr.size;
        std::printf("mapped %s at %p, size %u, entry +%#x%s\n", fsr.name.c_str(), fsr.base, fsr.size,
                    fsr.entry_rva, fsr.relocated ? " (relocated)" : "");
        int shim = 0, host = 0, unimplemented = 0;
        for (auto& r : fsr.imports) {
            if (r.binding == rt::Binding::Shim) ++shim;
            if (r.binding == rt::Binding::Host) ++host;
            if (r.binding == rt::Binding::Unimplemented) {
                ++unimplemented;
                std::printf("  UNIMPLEMENTED %s!%s\n", r.dll.c_str(), r.name.c_str());
            }
            const std::string d = rt::lower(r.dll);
            if (d == "xmem.dll" || d == "d3d12_xs.dll")
                check(r.binding == rt::Binding::Shim, r.dll + "!" + r.name + " binds to the shim");
        }
        std::printf("imports: shim=%d host=%d unimplemented=%d\n", shim, host, unimplemented);
        check(unimplemented == 0, "no unimplemented imports");

        auto configure = reinterpret_cast<ffx::PfnConfigure>(symbol(fsr, "ffxConfigure"));
        auto query = reinterpret_cast<ffx::PfnQuery>(symbol(fsr, "ffxQuery"));
        auto create = reinterpret_cast<ffx::PfnCreateContext>(symbol(fsr, "ffxCreateContext"));
        auto destroy = reinterpret_cast<ffx::PfnDestroyContext>(symbol(fsr, "ffxDestroyContext"));
        check(configure && query && create && destroy && symbol(fsr, "ffxDispatch"), "the five ffx* entry points are exported");
        if (g_failures) return 1;
        check(patch(iat_slot(fsr, "xmem.dll", "XMemAlloc"), reinterpret_cast<void*>(&counting_alloc),
                    reinterpret_cast<void**>(&g_xmem_alloc)) &&
                  patch(iat_slot(fsr, "xmem.dll", "XMemFree"), reinterpret_cast<void*>(&counting_free),
                        reinterpret_cast<void**>(&g_xmem_free)),
              "XMemAlloc/XMemFree IAT slots wrapped with counters");
        check(patch(iat_slot(fsr, "api-ms-win-core-localization-l1-2-0.dll", "FormatMessageW"),
                    reinterpret_cast<void*>(&recording_format_message), reinterpret_cast<void**>(&g_format_message)),
              "FormatMessageW IAT slot wrapped (records the HRESULT the DLL's error check formats)");

        // 2. TLS and DllMain.
        check(ensure_unwind_registered(fsr), "x64 unwind tables registered (by the loader or here)");
        const DWORD init = guarded_initialise(&loader);
        check(init == 0, "Loader::initialise (TLS, DllMain) succeeds" + (init ? " (exception " + hex(init) + ")" : ""));
        if (init) return 1;
        const uint32_t tls = module_tls_index(fsr, 0);
        std::printf("TLS: %s index %u\n", tls != ~0u ? "loader" : "no", tls);
        if (tls != ~0u) {
            // __declspec(thread) code reads TEB->ThreadLocalStoragePointer[_tls_index] (gs:[58h]), not TlsGetValue.
            auto** tls_array = reinterpret_cast<void**>(__readgsqword(0x58));
            void* seen = tls_array ? tls_array[tls] : nullptr;
            note("TLS check: ThreadLocalStoragePointer[" + std::to_string(tls) + "] = " +
                 hex(reinterpret_cast<uint64_t>(seen)) + ", loader block = " + hex(reinterpret_cast<uint64_t>(fsr.tls_block)));
            // A TlsAlloc index names TEB->TlsSlots, a different array: the image would read another module's TLS
            // block, and its thread-safe statics then fault or not depending on what is there (#93 fixes this).
            check(seen == fsr.tls_block, "the image's __declspec(thread) block is the one the loader set up");
            if (seen != fsr.tls_block) return 1;
        }

        // 3. Debug messages.
        ffx::ConfigureDescGlobalDebug debug{};
        debug.header.type = ffx::CONFIGURE_DESC_TYPE_GLOBALDEBUG;
        debug.fpMessage = on_message;
        debug.debugLevel = ffx::GLOBALDEBUG_LEVEL_VERBOSE;
        Call c = guarded_configure(configure, nullptr, &debug.header);
        note("ffxConfigure(null, GLOBALDEBUG) -> " + describe(c));
        check(!c.faulted, "ffxConfigure(GLOBALDEBUG) returns");

        // 4. CPU-only queries with the PC struct type ids.
        float ratio = 0.0f;
        ffx::QueryDescUpscaleGetUpscaleRatioFromQualityMode qr{};
        qr.header.type = ffx::QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE;
        qr.qualityMode = ffx::QUALITY_MODE_PERFORMANCE;
        qr.pOutUpscaleRatio = &ratio;
        c = guarded_query(query, nullptr, &qr.header);
        note("ffxQuery(null, GETUPSCALERATIOFROMQUALITYMODE, PERFORMANCE) -> " + describe(c) + ", ratio " +
             std::to_string(ratio));
        check(!c.faulted && c.rc == ffx::RETURN_OK && std::fabs(ratio - 2.0f) < 1e-6f,
              "upscale ratio for PERFORMANCE is 2.0 (PC struct type id accepted)");

        uint32_t rw = 0, rh = 0;
        ffx::QueryDescUpscaleGetRenderResolutionFromQualityMode qres{};
        qres.header.type = ffx::QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE;
        qres.displayWidth = 3840;
        qres.displayHeight = 2160;
        qres.qualityMode = ffx::QUALITY_MODE_QUALITY;
        qres.pOutRenderWidth = &rw;
        qres.pOutRenderHeight = &rh;
        c = guarded_query(query, nullptr, &qres.header);
        note("ffxQuery(null, GETRENDERRESOLUTIONFROMQUALITYMODE, 3840x2160 QUALITY) -> " + describe(c) + ", " +
             std::to_string(rw) + "x" + std::to_string(rh));
        check(!c.faulted && c.rc == ffx::RETURN_OK && rw == 2560 && rh == 1440, "QUALITY at 3840x2160 renders at 2560x1440");

        // 5/6 need a device.
        ID3D12Device* pc = nullptr;
        const bool on_vulkan = env("LEONIDA_FSR_DEVICE") == "vulkan";
        if (on_vulkan) {
#ifdef LEONIDA_VULKAN_DEVICE
            const HRESULT vhr = rt::d3d12::create_vulkan_device_standalone(__uuidof(ID3D12Device),
                                                                           reinterpret_cast<void**>(&pc));
            note("LEONIDA_FSR_DEVICE=vulkan -> " + hex(static_cast<uint32_t>(vhr)));
#else
            check(false, "LEONIDA_FSR_DEVICE=vulkan needs a build with the Vulkan SDK (gta6rt_d3d12_com)");
#endif
        } else {
            pc = create_pc_device(env("LEONIDA_FSR_WARP") == "1");
        }
        check(pc != nullptr, on_vulkan ? "Vulkan-backed D3D12 device created" : "PC D3D12 device created");
        if (!pc) return 1;
        void* proxy = nullptr;
        const HRESULT hr = rt::d3d12_xs::create_device_proxy(pc, __uuidof(ID3D12Device), &proxy);
        check(SUCCEEDED(hr) && proxy, "Xbox-ABI device proxy created");
        if (!proxy) return 1;
        rt::d3d12_xs::reset_device_proxy_stats();

        uint64_t count = 0;
        ffx::QueryDescGetVersions qv{};
        qv.header.type = ffx::QUERY_DESC_TYPE_GET_VERSIONS;
        qv.createDescType = ffx::CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        qv.device = proxy;
        qv.outputCount = &count;
        c = guarded_query(query, nullptr, &qv.header);
        note("ffxQuery(null, GET_VERSIONS, upscale) -> " + describe(c) + ", " + std::to_string(count) + " version(s)");
        if (!c.faulted && c.rc == ffx::RETURN_OK && count > 0 && count < 64) {
            uint64_t ids[64]{};
            const char* names[64]{};
            qv.versionIds = ids;
            qv.versionNames = names;
            c = guarded_query(query, nullptr, &qv.header);
            for (uint64_t i = 0; i < count && !c.faulted; ++i)
                note("  version " + hex(ids[i]) + " " + (names[i] ? names[i] : "(no name)"));
        }

        // 6. Create a context. The shipping build accepts one configuration only, and says so through the message
        //    callback when asked for another (measured on 1.0.2.1): 1280x720 -> 1920x1080 Quality, HDR input,
        //    inverted depth, auto exposure.
        ffx::CreateBackendDX12Desc backend{};
        backend.header.type = ffx::CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
        backend.device = proxy;
        // The DLL insists on a version descriptor; it reports provider version 4.1.1 (GET_VERSIONS above).
        ffx::CreateContextDescUpscaleVersion version{};
        version.header.type = ffx::CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
        version.header.pNext = &backend.header;
        version.version = ffx::upscaler_version(4, 1, 1);
        ffx::CreateContextDescUpscale up{};
        up.header.type = ffx::CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        up.header.pNext = &version.header;
        up.flags = ffx::UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | ffx::UPSCALE_ENABLE_DEPTH_INVERTED |
                   ffx::UPSCALE_ENABLE_AUTO_EXPOSURE | ffx::UPSCALE_ENABLE_DEBUG_CHECKING;
        up.maxRenderSize = {1280, 720};
        up.maxUpscaleSize = {1920, 1080};
        up.fpMessage = on_message;
        ffx::Context ctx = nullptr;
        // Which backend descriptor the Xbox build takes is not public. Measured in the DLL (docs/d3d12_vulkan.md,
        // Appendix A.3): the provider's create walks the chain for type 0x01000002 and reads the device at +0x10, the
        // same {header, device} layout as the PC DX12 descriptor; any other type ends in RETURN_ERROR before the
        // device is touched. The other ids stay as controls.
        constexpr ffx::StructType kCreateContextDescTypeBackendXbox = 0x01000002u;
        const ffx::StructType backend_types[] = {
            kCreateContextDescTypeBackendXbox,
            ffx::CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12,
            ffx::backend_sub_id(ffx::BACKEND_ID_XBOX, 0x02),
            ffx::backend_sub_id(ffx::BACKEND_ID_XBOX, 0x01),
            ffx::backend_sub_id(ffx::BACKEND_ID_XBOX, 0x00),
        };
        const uint32_t versions[] = {ffx::upscaler_version(4, 1, 1), ffx::upscaler_version(4, 1, 2),
                                     ffx::upscaler_version(4, 0, 0)};
        const uint32_t base_flags = ffx::UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | ffx::UPSCALE_ENABLE_DEPTH_INVERTED |
                                    ffx::UPSCALE_ENABLE_AUTO_EXPOSURE;
        bool created = false;
        for (ffx::StructType type : backend_types) {
            for (uint32_t v : versions) {
                for (uint32_t extra : {0u, ffx::UPSCALE_ENABLE_DEBUG_CHECKING}) {
                    backend.header.type = type;
                    version.version = v;
                    up.flags = base_flags | extra;
                    rt::d3d12_xs::reset_device_proxy_stats();
                    g_alloc_calls = g_alloc_bytes = g_free_calls = 0;
                    g_formatted = 0;
                    const int messages_before = g_messages;
                    c = guarded_create(create, &ctx, &up.header);
                    char line[1024];  // a fault description can carry a stack
                    std::snprintf(line, sizeof(line),
                                  "ffxCreateContext(backend %#llx, version %u.%u.%u, flags %#x) -> %s; XMemAlloc %llu "
                                  "(%llu bytes), XMemFree %llu, messages %d",
                                  static_cast<unsigned long long>(type), v >> 22, (v >> 12) & 0x3ff, v & 0xfff, up.flags,
                                  describe(c).c_str(), static_cast<unsigned long long>(g_alloc_calls),
                                  static_cast<unsigned long long>(g_alloc_bytes),
                                  static_cast<unsigned long long>(g_free_calls), g_messages - messages_before);
                    note(line);
                    if (g_formatted) {
                        std::string frames;
                        for (USHORT i = 0; i < g_first_formatted_frames; ++i) {
                            const auto* f = static_cast<const uint8_t*>(g_first_formatted_stack[i]);
                            if (f >= g_dll_base && f < g_dll_base + g_dll_size)
                                frames += " " + hex(static_cast<uint64_t>(f - g_dll_base));
                        }
                        note("  the DLL formatted HRESULT " + hex(g_first_formatted) + " (" +
                             std::to_string(g_formatted) +
                             " formatted in all); DLL return addresses, innermost first:" +
                             (frames.empty() ? std::string(" none") : frames));
                    }
                    created = c.faulted || (c.rc == ffx::RETURN_OK && ctx);
                    if (created) break;
                }
                if (created) break;
            }
            if (created) break;
        }
        if (!created) {
            // Control: the same chain with no backend descriptor at all. If this fails the same way, the DLL does
            // not recognise any of the backend descriptors above.
            version.header.pNext = nullptr;
            version.version = ffx::upscaler_version(4, 1, 1);
            up.flags = base_flags | ffx::UPSCALE_ENABLE_DEBUG_CHECKING;
            c = guarded_create(create, &ctx, &up.header);
            note("ffxCreateContext(no backend descriptor, 4.1.1) -> " + describe(c));
            version.header.pNext = &backend.header;
        }
        print_slot_hits(proxy);
        // The DLL's HRESULT check throws an int. With the image mapped by rt::Loader the CRT's throw finds no image
        // base, so nothing inside the DLL can catch it and it reaches the guard: a reported stop (the HRESULT and the
        // DLL frames are printed above), not a crash.
        const bool hresult_stop = c.faulted && c.code == kMsvcCppException && g_formatted > 0;
        if (hresult_stop)
            note("ffxCreateContext stopped at the DLL's HRESULT check (" + hex(g_first_formatted) + "), not a crash");
        check(!c.faulted || hresult_stop, "ffxCreateContext returns, or stops at the DLL's HRESULT check");

        // 7. If the driver refused the texture, the same create with the PC stand-in (standin_slot76;
        //    LEONIDA_FSR_NO_STANDIN=1 skips it). Some drivers take that texture (WARP does), so the context may exist
        //    already. Either way, one ffxDispatch on the context, recorded into the logging command list (not
        //    executed).
        bool pipeline_stop = false;
        const bool want_standin = hresult_stop && env("LEONIDA_FSR_NO_STANDIN") != "1";
        if (want_standin || ctx) {
            const unsigned proxy_slots = rt::d3d12_xs::device_proxy_slot_count(proxy) + 32;
            install_standins(proxy, proxy_slots, want_standin);
            if (want_standin) {
                backend.header.type = kCreateContextDescTypeBackendXbox;
                version.header.pNext = &backend.header;
                version.version = ffx::upscaler_version(4, 1, 1);
                up.flags = base_flags;
                rt::d3d12_xs::reset_device_proxy_stats();
                g_formatted = 0;
                g_standins = 0;
                c = guarded_create(create, &ctx, &up.header);
                note("with the PC stand-in: ffxCreateContext(backend 0x1000002, 4.1.1) -> " + describe(c) + "; " +
                     std::to_string(g_standins) + " R9G9B9E5_SHAREDEXP UAV texture(s) created as R32_UINT");
                if (g_formatted) note("  the DLL formatted HRESULT " + hex(g_first_formatted));
                check(!c.faulted && c.rc == ffx::RETURN_OK && ctx, "ffxCreateContext succeeds with the PC stand-in");
            }

            if (ctx) {
                auto texture = [&](UINT w, UINT h, DXGI_FORMAT format, bool uav, D3D12_RESOURCE_STATES state) {
                    D3D12_HEAP_PROPERTIES heap{};
                    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
                    D3D12_RESOURCE_DESC d{};
                    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                    d.Width = w;
                    d.Height = h;
                    d.DepthOrArraySize = 1;
                    d.MipLevels = 1;
                    d.Format = format;
                    d.SampleDesc.Count = 1;
                    d.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
                    ID3D12Resource* r = nullptr;
                    if (FAILED(pc->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr,
                                                           IID_PPV_ARGS(&r))))
                        r = nullptr;
                    return r;
                };
                auto ffx_texture = [](ID3D12Resource* r, uint32_t format, UINT w, UINT h, uint32_t usage,
                                      uint32_t state) {
                    ffx::Resource x{};
                    x.resource = r;
                    x.description = {ffx::RESOURCE_TYPE_TEXTURE2D, format, w, h, 1, 1, 0, usage};
                    x.state = state;
                    return x;
                };
                constexpr auto kRead = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                ID3D12Resource* color = texture(1280, 720, DXGI_FORMAT_R16G16B16A16_FLOAT, false, kRead);
                ID3D12Resource* depth = texture(1280, 720, DXGI_FORMAT_R32_FLOAT, false, kRead);
                ID3D12Resource* motion = texture(1280, 720, DXGI_FORMAT_R16G16_FLOAT, false, kRead);
                ID3D12Resource* output = texture(1920, 1080, DXGI_FORMAT_R16G16B16A16_FLOAT, true,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                ID3D12CommandAllocator* allocator = nullptr;
                ID3D12GraphicsCommandList* list = nullptr;
                const bool made = color && depth && motion && output &&
                                  SUCCEEDED(pc->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                                       IID_PPV_ARGS(&allocator))) &&
                                  SUCCEEDED(pc->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator,
                                                                  nullptr, IID_PPV_ARGS(&list)));
                check(made, "PC inputs, output and command list for ffxDispatch created");
                auto dispatch = reinterpret_cast<ffx::PfnDispatch>(symbol(fsr, "ffxDispatch"));
                if (made && dispatch) {
                    ListProxy logging{list_vtable(), list};
                    for (auto& h : g_list_hits) h.store(0, std::memory_order_relaxed);
                    g_list_trace_count.store(0, std::memory_order_relaxed);
                    ffx::DispatchDescUpscale dd{};
                    dd.header.type = ffx::DISPATCH_DESC_TYPE_UPSCALE;
                    dd.commandList = &logging;
                    dd.color = ffx_texture(color, ffx::SURFACE_FORMAT_R16G16B16A16_FLOAT, 1280, 720,
                                           ffx::RESOURCE_USAGE_READ_ONLY, ffx::RESOURCE_STATE_COMPUTE_READ);
                    dd.depth = ffx_texture(depth, ffx::SURFACE_FORMAT_R32_FLOAT, 1280, 720,
                                           ffx::RESOURCE_USAGE_READ_ONLY, ffx::RESOURCE_STATE_COMPUTE_READ);
                    dd.motionVectors = ffx_texture(motion, ffx::SURFACE_FORMAT_R16G16_FLOAT, 1280, 720,
                                                   ffx::RESOURCE_USAGE_READ_ONLY, ffx::RESOURCE_STATE_COMPUTE_READ);
                    dd.output = ffx_texture(output, ffx::SURFACE_FORMAT_R16G16B16A16_FLOAT, 1920, 1080,
                                            ffx::RESOURCE_USAGE_UAV, ffx::RESOURCE_STATE_UNORDERED_ACCESS);
                    dd.motionVectorScale = {1.0f, 1.0f};
                    dd.renderSize = {1280, 720};
                    dd.upscaleSize = {1920, 1080};
                    dd.frameTimeDelta = 16.7f;
                    dd.preExposure = 1.0f;
                    dd.reset = true;
                    dd.cameraNear = 1000.0f;  // inverted depth: near and far swap
                    dd.cameraFar = 0.1f;
                    dd.cameraFovAngleVertical = 1.0f;
                    dd.viewSpaceToMetersFactor = 1.0f;
                    rt::d3d12_xs::reset_device_proxy_stats();
                    g_formatted = 0;
                    c = guarded_dispatch(dispatch, &ctx, &dd.header);
                    note("ffxDispatch(upscale 1280x720 -> 1920x1080) into the logging list -> " + describe(c));
                    note("  device slot 62 (compute pipeline from a D3D12_COMPUTE_PIPELINE_STATE_DESC) called " +
                         std::to_string(g_slot62_calls) + "x, last HRESULT " + hex(static_cast<uint32_t>(g_slot62_hr)));
                    if (g_formatted) note("  the DLL formatted HRESULT " + hex(g_first_formatted));
                    // On a PC device both pipeline paths fail (neither slot 67's nor 62's code is a PC shader); on the
                    // Vulkan device CreateRootSignature (slot 16) is not implemented yet. Either way the dispatch goes
                    // on with a null pipeline and faults when it records it (RVA 0x65ff reads the pipeline's +0x10,
                    // the same pointer it passes to SetPipelineState): a documented stop, not a test failure.
                    pipeline_stop = c.faulted && c.code == EXCEPTION_ACCESS_VIOLATION && g_slot62_calls > 0 &&
                                    FAILED(g_slot62_hr);
                    const bool vulkan_stop = on_vulkan && c.faulted && c.code == EXCEPTION_ACCESS_VIOLATION;
                    if (pipeline_stop)
                        note("ffxDispatch stopped where the PC refused the DLL's compute pipeline, not a crash");
                    else if (vulkan_stop)
                        note("ffxDispatch stopped where the Vulkan device returned E_NOTIMPL for CreateRootSignature");
                    check(!c.faulted || pipeline_stop || vulkan_stop ||
                              (c.code == kMsvcCppException && g_formatted > 0),
                          "ffxDispatch returns, or stops at a documented limit");
                    note("ffxDispatch: device slots below are the ones it called");
                    print_list_hits();
                    note("command list Close -> " + hex(static_cast<uint32_t>(list->Close())));
                }
                for (IUnknown* u : std::initializer_list<IUnknown*>{list, allocator, output, motion, depth, color})
                    if (u) u->Release();
            }
            remove_standins(proxy);
            print_slot_hits(proxy);
        }

        if (ctx && pipeline_stop) {
            // The fault left the context half-recorded (destroying it frees state the fault never set up).
            note("ffxDestroyContext skipped: the dispatch stopped inside the DLL");
        } else if (ctx) {  // ffxCreateContext clears it first and only sets it on success
            c = guarded_destroy(destroy, &ctx);
            note("ffxDestroyContext -> " + describe(c));
            check(!c.faulted && c.rc == ffx::RETURN_OK, "ffxDestroyContext succeeds");
        }
        note("messages from the DLL: " + std::to_string(g_messages));
        reinterpret_cast<IUnknown*>(proxy)->Release();
    } catch (const std::exception& e) {
        std::printf("FAIL %s\n", e.what());
        return 1;
    }

    std::printf("fsr_dispatch_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
