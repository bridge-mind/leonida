// Xbox-ABI queue and fence proxies (d3d12_xs_device.h), behind the device proxy, on the host PC runtime (WARP, and
// the first hardware adapter when there is one) and on the Vulkan-backed device:
//   - device slots 8, 75, 36 and 32 wrap what they create, and every public queue and fence slot is forwarded to
//     the slot it names (counted per slot) with the effect the method has;
//   - error and edge cases give the same HRESULT through the proxy as on the real object (a differential check);
//     a case the real runtime itself faults on is reported and not compared;
//   - QueryInterface identity, AddRef/Release across interfaces and release orders;
//   - heap, queue and fence proxies are unwrapped where the real runtime receives them (Signal/Wait,
//     SetEventOnMultipleFenceCompletion, MakeResident/Evict/SetResidencyPriority/EnqueueMakeResident,
//     CreateSharedHandle), and OpenSharedHandle wraps a shared fence;
//   - slots past the public chain are refused (E_NOTIMPL) up to the end of the table, and a slot past the table
//     faults on the vtable read in the guard region and is counted;
//   - calls from eight threads at once, and repeated device, queue and fence creation without leaks;
//   - the child-process runner of the differential check itself: large output, hangs, crashes, failed starts.
// No shaders, game data or Xbox DLLs. LEONIDA_VK_VALIDATION=1 runs the Vulkan device under
// VK_LAYER_KHRONOS_validation (with synchronization validation); any validation error fails the test. Under Wine only
// the Vulkan part runs. --trace-d3d12 also records every proxy call on the trace channel. Exits 77 when neither a
// host D3D12 device nor a Vulkan device can be made.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_trace.h"
#include "../shims/d3d12_xs_device.h"

#include <d3d12.h>
#include <dxgi1_4.h>

#include <crtdbg.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <iterator>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", label, __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace xs = rt::d3d12_xs;

template<class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

struct Event {
    HANDLE h = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { CloseHandle(h); }
    bool set_within(DWORD ms) const { return WaitForSingleObject(h, ms) == WAIT_OBJECT_0; }
};

// {6f1f2d3e-8a4b-4c5d-9e0f-1a2b3c4d5e6f}, {7a2b3c4d-...}: private-data keys of this test.
constexpr GUID kDataKey{0x6f1f2d3e, 0x8a4b, 0x4c5d, {0x9e, 0x0f, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f}};
constexpr GUID kInterfaceKey{0x7a2b3c4d, 0x9b5c, 0x4d6e, {0xaf, 0x10, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70}};
constexpr GUID kMissingKey{0x8b3c4d5e, 0xac6d, 0x4e7f, {0xb0, 0x21, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81}};

using SlotFn = std::uint64_t(WINAPI*)(void*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                      std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                      std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);

// A raw call through slot `slot` of an object's vtable with all words zero, the way a title calls a console method
// the PC interface does not declare.
std::uint64_t call_slot(void* object, unsigned slot) {
    void** vtable = *reinterpret_cast<void***>(object);
    return reinterpret_cast<SlotFn>(vtable[slot])(object, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

HRESULT as_hr(std::uint64_t value) { return static_cast<HRESULT>(static_cast<std::uint32_t>(value)); }

// A raw call through slot `slot` with the given words (the rest zero).
std::uint64_t raw_call(void* object, unsigned slot, std::uint64_t a1 = 0, std::uint64_t a2 = 0,
                       std::uint64_t a3 = 0, std::uint64_t a4 = 0) {
    void** vtable = *reinterpret_cast<void***>(object);
    return reinterpret_cast<SlotFn>(vtable[slot])(object, a1, a2, a3, a4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

// ---- Structured-exception guards (no C++ objects in these frames) ----

ULONG_PTR g_fault_address = 0;
// The code is taken in the filter, from the exception record: GetExceptionCode() in the __except body returns 0
// under Wine 9.0 (clang-cl reads it from what the unwind hands the handler, which that Wine does not supply).
DWORD g_fault_code = 0;
bool g_shared_checked = false;    // check_shared ran OpenSharedHandle (the backend has shared fences)
bool g_shared_created = false;    // check_shared made a shared fence and called CreateSharedHandle (slot 31)
bool g_reserved_checked = false;  // check_slots had reserved resources for UpdateTileMappings/CopyTileMappings

int record_fault(EXCEPTION_POINTERS* info) {
    g_fault_code = info->ExceptionRecord->ExceptionCode;
    g_fault_address = info->ExceptionRecord->NumberParameters >= 2 ? info->ExceptionRecord->ExceptionInformation[1] : 0;
    return EXCEPTION_EXECUTE_HANDLER;
}

// The read itself, in a function of its own: clang-cl's __try only catches a fault raised inside a call, not one
// raised by a load in the __try body itself (MSVC catches both).
__declspec(noinline) void* read_entry(void* const* vtable, unsigned slot) {
    return *static_cast<void* const volatile*>(&vtable[slot]);
}

// Reads vtable[slot] the way a call through it does. Returns the exception code, or 0 when the read worked.
DWORD read_slot_guarded(void* const* vtable, unsigned slot) {
    g_fault_address = 0;
    g_fault_code = 0;
    __try {
        void* volatile entry = read_entry(vtable, slot);
        (void)entry;
        return 0;
    } __except (record_fault(GetExceptionInformation())) {
        return g_fault_code;
    }
}

using Case = HRESULT (*)(void* context);

// Runs a case; false when it raised an access violation (the real runtime dereferencing a null argument). Any other
// exception is not this test's to handle and takes the child process down, which the parent reports as a failure.
bool run_guarded(Case fn, void* context, HRESULT* result) {
    __try {
        *result = fn(context);
        return true;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER
                                                                  : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

int wait_for(const char* label, ID3D12Fence* fence, UINT64 value) {
    Event event;
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(value, event.h)));
    EXPECT(event.set_within(5000));
    EXPECT(fence->GetCompletedValue() >= value);
    return 0;
}

struct Backend {
    const char* label;
    ID3D12Device* device;  // the device proxy
    ID3D12Device* real;    // the device behind it (PC runtime or Vulkan layer)
    bool vulkan;
    const char* kind;      // "warp", "hw" or "vulkan": what a --diff child creates
};

// ---- 1. Creation, every public slot, and the refused tail ----

int check_slots(const Backend& b) {
    const char* label = b.label;
    ID3D12Device* device = b.device;
    EXPECT(xs::is_device_proxy(device));
    xs::reset_device_proxy_stats();
    const unsigned long device_refs = xs::device_proxy_ref_count(device);
    {
        D3D12_COMMAND_QUEUE_DESC direct_desc{};
        direct_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&direct_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(xs::device_slot_hits(8) == 1);
        EXPECT(xs::is_queue_proxy(queue.p));
        ID3D12CommandQueue* real_queue = xs::queue_proxy_real_queue(queue.p);
        EXPECT(real_queue && static_cast<void*>(real_queue) != static_cast<void*>(queue.p));
        EXPECT(!xs::is_queue_proxy(real_queue) && !xs::is_fence_proxy(queue.p));
        EXPECT(xs::queue_proxy_slot_count(queue.p) == 19 && xs::queue_proxy_table_size() == 128);
        EXPECT(xs::command_object_proxy_ref_count(queue.p) == 1);
        // The queue keeps its device alive; on the Vulkan layer the real queue also holds the device proxy (its
        // identity), so the count may rise by two. The end of this function checks that all are given back.
        EXPECT(xs::device_proxy_ref_count(device) >= device_refs + 1);

        Ref<ID3D12Fence> fence, second;
        EXPECT(SUCCEEDED(device->CreateFence(5, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&second.p))));
        EXPECT(xs::device_slot_hits(36) == 2);
        EXPECT(xs::is_fence_proxy(fence.p) && xs::is_fence_proxy(second.p) && !xs::is_queue_proxy(fence.p));
        ID3D12Fence* real_fence = xs::fence_proxy_real_fence(fence.p);
        EXPECT(real_fence && static_cast<void*>(real_fence) != static_cast<void*>(fence.p));
        const unsigned fence_slots = xs::fence_proxy_slot_count(fence.p);
        EXPECT(fence_slots == 11 || fence_slots == 12);
        EXPECT(xs::fence_proxy_table_size() == 64);

        // Every public queue slot, each counted exactly once per call, with its effect.
        const auto qhits = [](unsigned s) { return xs::queue_slot_hits(s); };
        const auto fhits = [](unsigned s) { return xs::fence_slot_hits(s); };
        std::uint64_t before = qhits(4);
        const UINT value = 0x1234abcd;
        EXPECT(SUCCEEDED(queue->SetPrivateData(kDataKey, sizeof(value), &value)));  // 4
        EXPECT(qhits(4) == before + 1);
        UINT read = 0, size = sizeof(read);
        before = qhits(3);
        EXPECT(SUCCEEDED(queue->GetPrivateData(kDataKey, &size, &read)) && read == value && size == sizeof(read));  // 3
        EXPECT(qhits(3) == before + 1);
        before = qhits(5);
        EXPECT(SUCCEEDED(queue->SetPrivateDataInterface(kInterfaceKey, second.p)));  // 5
        EXPECT(qhits(5) == before + 1);
        {
            IUnknown* stored = nullptr;
            UINT stored_size = sizeof(stored);
            EXPECT(SUCCEEDED(queue->GetPrivateData(kInterfaceKey, &stored_size, &stored)));
            EXPECT(static_cast<void*>(stored) == static_cast<void*>(second.p));  // the proxy, not the real fence
            if (stored) stored->Release();
        }
        EXPECT(SUCCEEDED(queue->SetPrivateDataInterface(kInterfaceKey, nullptr)));
        before = qhits(6);
        EXPECT(SUCCEEDED(queue->SetName(L"queue_proxy_test queue")));  // 6
        EXPECT(qhits(6) == before + 1);
        {
            Ref<ID3D12Device> owner;
            before = qhits(7);
            EXPECT(SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&owner.p))) && owner.p == device);  // 7
            EXPECT(qhits(7) == before + 1);
        }

        // 8/9: tile mappings need a reserved resource; skipped (and said so) where the device has none.
        {
            D3D12_RESOURCE_DESC reserved_desc{};
            reserved_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            reserved_desc.Width = 2 * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
            reserved_desc.Height = reserved_desc.DepthOrArraySize = reserved_desc.MipLevels = 1;
            reserved_desc.SampleDesc.Count = 1;
            reserved_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            Ref<ID3D12Resource> a, c;
            const bool reserved =
                SUCCEEDED(device->CreateReservedResource(&reserved_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&a.p))) &&
                SUCCEEDED(device->CreateReservedResource(&reserved_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&c.p)));
            if (reserved) {
                g_reserved_checked = true;
                const D3D12_TILED_RESOURCE_COORDINATE origin{};
                D3D12_TILE_REGION_SIZE region{};
                region.NumTiles = 1;
                const D3D12_TILE_RANGE_FLAGS null_range = D3D12_TILE_RANGE_FLAG_NULL;
                const UINT one = 1;
                before = qhits(8);
                queue->UpdateTileMappings(a.p, 1, &origin, &region, nullptr, 1, &null_range, nullptr, &one,
                                          D3D12_TILE_MAPPING_FLAG_NONE);  // 8
                EXPECT(qhits(8) == before + 1);
                before = qhits(9);
                queue->CopyTileMappings(c.p, &origin, a.p, &origin, &region, D3D12_TILE_MAPPING_FLAG_NONE);  // 9
                EXPECT(qhits(9) == before + 1);
            } else {
                std::printf("%s: no reserved resources, UpdateTileMappings/CopyTileMappings not called\n", label);
            }
        }

        // 10: ExecuteCommandLists passes the (real, not wrapped) command lists through.
        {
            Ref<ID3D12CommandAllocator> allocator;
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                            IID_PPV_ARGS(&allocator.p))));
            Ref<ID3D12GraphicsCommandList> list;
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
            EXPECT(SUCCEEDED(list->Close()));
            ID3D12CommandList* lists[] = {list.p};
            before = qhits(10);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(qhits(10) == before + 1);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 6)));
            if (wait_for(label, fence.p, 6)) return 1;
            if (b.vulkan) {
                // The layer's own helpers find their queue behind the proxy.
                EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
                (void)rt::d3d12::queue_uses_calibrated_timestamps(queue.p);
            }
        }

        // 11-13: markers.
        static const char marker[] = "queue_proxy_test";
        before = qhits(11);
        queue->SetMarker(1, marker, sizeof(marker));
        EXPECT(qhits(11) == before + 1);
        before = qhits(12);
        queue->BeginEvent(1, marker, sizeof(marker));
        EXPECT(qhits(12) == before + 1);
        before = qhits(13);
        queue->EndEvent();
        EXPECT(qhits(13) == before + 1);

        // 14/15: Signal and Wait take fence proxies; the real queue gets the real fences.
        before = qhits(14);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 7)));
        EXPECT(qhits(14) == before + 1);
        if (wait_for(label, fence.p, 7)) return 1;
        before = qhits(15);
        EXPECT(SUCCEEDED(queue->Wait(fence.p, 8)));  // the queue waits until the CPU signals 8
        EXPECT(qhits(15) == before + 1);
        EXPECT(SUCCEEDED(queue->Signal(second.p, 1)));
        Sleep(50);
        EXPECT(second->GetCompletedValue() == 0);  // still held by the Wait
        before = fhits(10);
        EXPECT(SUCCEEDED(fence->Signal(8)));  // fence 10, from the CPU
        EXPECT(fhits(10) == before + 1);
        if (wait_for(label, second.p, 1)) return 1;

        // A real fence (made on the real device) passes through a proxy queue unchanged.
        {
            Ref<ID3D12Fence> raw;
            EXPECT(SUCCEEDED(b.real->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&raw.p))));
            EXPECT(!xs::is_fence_proxy(raw.p));
            EXPECT(SUCCEEDED(queue->Signal(raw.p, 3)));
            if (wait_for(label, raw.p, 3)) return 1;
        }

        // 16-18.
        UINT64 frequency = 0;
        before = qhits(16);
        EXPECT(SUCCEEDED(queue->GetTimestampFrequency(&frequency)) && frequency > 0);
        EXPECT(qhits(16) == before + 1);
        UINT64 gpu = 0, cpu = 0;
        before = qhits(17);
        EXPECT(SUCCEEDED(queue->GetClockCalibration(&gpu, &cpu)) && cpu > 0);
        EXPECT(qhits(17) == before + 1);
        before = qhits(18);
        EXPECT(queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT);  // struct through the hidden pointer
        EXPECT(qhits(18) == before + 1);
        D3D12_COMMAND_QUEUE_DESC compute_desc{};
        compute_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        compute_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
        Ref<ID3D12CommandQueue> compute;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&compute_desc, IID_PPV_ARGS(&compute.p))));
        const D3D12_COMMAND_QUEUE_DESC compute_back = compute->GetDesc();
        EXPECT(xs::is_queue_proxy(compute.p) && compute_back.Type == D3D12_COMMAND_LIST_TYPE_COMPUTE);
        EXPECT(compute_back.Priority == D3D12_COMMAND_QUEUE_PRIORITY_HIGH);

        // Every public fence slot.
        before = fhits(4);
        EXPECT(SUCCEEDED(fence->SetPrivateData(kDataKey, sizeof(value), &value)));
        EXPECT(fhits(4) == before + 1);
        read = 0;
        size = sizeof(read);
        before = fhits(3);
        EXPECT(SUCCEEDED(fence->GetPrivateData(kDataKey, &size, &read)) && read == value);
        EXPECT(fhits(3) == before + 1);
        before = fhits(5);
        EXPECT(SUCCEEDED(fence->SetPrivateDataInterface(kInterfaceKey, nullptr)));
        EXPECT(fhits(5) == before + 1);
        before = fhits(6);
        EXPECT(SUCCEEDED(fence->SetName(L"queue_proxy_test fence")));
        EXPECT(fhits(6) == before + 1);
        {
            Ref<ID3D12Device> owner;
            before = fhits(7);
            EXPECT(SUCCEEDED(fence->GetDevice(IID_PPV_ARGS(&owner.p))) && owner.p == device);
            EXPECT(fhits(7) == before + 1);
        }
        before = fhits(8);
        EXPECT(fence->GetCompletedValue() == 8);
        EXPECT(fhits(8) == before + 1);
        {
            Event done;
            before = fhits(9);
            EXPECT(SUCCEEDED(fence->SetEventOnCompletion(8, done.h)) && done.set_within(5000));
            EXPECT(fhits(9) == before + 1);
        }
        if (fence_slots == 12) {
            Ref<ID3D12Fence1> fence1;
            EXPECT(SUCCEEDED(fence->QueryInterface(IID_PPV_ARGS(&fence1.p))));
            before = fhits(11);
            EXPECT(fence1->GetCreationFlags() == D3D12_FENCE_FLAG_NONE);
            EXPECT(fhits(11) == before + 1);
        }

        // Device slot 75 (ID3D12Device9::CreateCommandQueue1) also wraps its queue.
        {
            Ref<ID3D12Device9> device9;
            if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device9.p)))) {
                Ref<ID3D12CommandQueue> queue1;
                const HRESULT hr = device9->CreateCommandQueue1(&direct_desc, __uuidof(ID3D12Device),
                                                                IID_PPV_ARGS(&queue1.p));
                EXPECT(xs::device_slot_hits(75) == 1);
                EXPECT(SUCCEEDED(hr) && xs::is_queue_proxy(queue1.p));
            } else {
                std::printf("%s: no ID3D12Device9, CreateCommandQueue1 not checked\n", label);
            }
        }

        // The refused tail: every slot from the end of the public chain to the end of the table.
        for (unsigned slot = 19; slot < xs::queue_proxy_table_size(); ++slot) {
            before = qhits(slot);
            EXPECT(as_hr(call_slot(queue.p, slot)) == E_NOTIMPL);
            EXPECT(qhits(slot) == before + 1 && xs::queue_slot_first_call_seen(slot));
        }
        for (unsigned slot = fence_slots; slot < xs::fence_proxy_table_size(); ++slot) {
            before = fhits(slot);
            EXPECT(as_hr(call_slot(fence.p, slot)) == E_NOTIMPL);
            EXPECT(fhits(slot) == before + 1 && xs::fence_slot_first_call_seen(slot));
        }
        // The objects still work after refused calls.
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 10)));
        if (wait_for(label, fence.p, 10)) return 1;
        EXPECT(xs::command_object_proxy_ref_count(queue.p) == 1 && xs::command_object_proxy_ref_count(fence.p) == 1);
    }
    EXPECT(xs::device_proxy_ref_count(device) == device_refs);  // every queue and fence gave its reference back
    std::printf("%s: every public queue and fence slot forwarded and counted; tail refused\n", label);
    return 0;
}

// ---- 2. The guard region past the tables ----

int check_guard(const Backend& b) {
    const char* label = b.label;
    EXPECT(xs::command_object_tables_guarded());
    D3D12_COMMAND_QUEUE_DESC desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(b.device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(b.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    struct Probe {
        void* object;
        unsigned table;
        std::uint64_t (*overruns)();
    } probes[] = {{queue.p, xs::queue_proxy_table_size(), &xs::queue_table_overruns},
                  {fence.p, xs::fence_proxy_table_size(), &xs::fence_table_overruns}};
    for (const Probe& probe : probes) {
        void* const* vtable = *reinterpret_cast<void* const* const*>(probe.object);
        EXPECT(read_slot_guarded(vtable, probe.table - 1) == 0);  // the last table slot is readable
        // The first slot past the table, one a page further, and one near the end of the reservation.
        for (unsigned slot : {probe.table, probe.table + 512, probe.table + 7000}) {
            const std::uint64_t before = probe.overruns();
            EXPECT(read_slot_guarded(vtable, slot) == EXCEPTION_ACCESS_VIOLATION);
            EXPECT(g_fault_address == reinterpret_cast<ULONG_PTR>(&vtable[slot]));
            EXPECT(probe.overruns() == before + 1);
        }
    }
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
    if (wait_for(label, fence.p, 1)) return 1;
    std::printf("%s: a slot past either table faults in the guard region and is counted\n", label);
    return 0;
}

// ---- 3. Same HRESULT through the proxy as on the real object ----

// What a case returns when the device lacks the interface the case needs. Not S_FALSE: CreateCommandQueue1 with a
// null output returns S_FALSE for a valid description, so a proxy wrongly refusing ID3D12Device9 would match it.
constexpr HRESULT kNoInterface = HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);

struct DiffContext {
    ID3D12Device* device;      // proxy or real
    ID3D12CommandQueue* queue;
    ID3D12Fence* fence;
    ID3D12Fence* second;
    HANDLE event;
};

struct DiffCase {
    const char* name;
    Case run;
};

// The cases pass null and invalid arguments on purpose: that is what they compare.
#pragma warning(push)
#pragma warning(disable : 6387)

HRESULT create_queue_no_output(void* c) {
    D3D12_COMMAND_QUEUE_DESC desc{};
    return static_cast<DiffContext*>(c)->device->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue), nullptr);
}
HRESULT create_queue_bad_type(void* c) {
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = static_cast<D3D12_COMMAND_LIST_TYPE>(99);
    void* out = nullptr;
    const HRESULT hr = static_cast<DiffContext*>(c)->device->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue),
                                                                                 &out);
    if (out) static_cast<IUnknown*>(out)->Release();
    return hr;
}
HRESULT create_queue_bad_priority(void* c) {
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Priority = 7;
    void* out = nullptr;
    const HRESULT hr = static_cast<DiffContext*>(c)->device->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue),
                                                                                 &out);
    if (out) static_cast<IUnknown*>(out)->Release();
    return hr;
}
HRESULT create_queue_as_fence(void* c) {
    D3D12_COMMAND_QUEUE_DESC desc{};
    void* out = reinterpret_cast<void*>(1);
    const HRESULT hr = static_cast<DiffContext*>(c)->device->CreateCommandQueue(&desc, __uuidof(ID3D12Fence), &out);
    if (SUCCEEDED(hr) && out) static_cast<IUnknown*>(out)->Release();
    return SUCCEEDED(hr) || out == nullptr ? hr : E_UNEXPECTED;  // a failed create must clear the output
}
HRESULT create_fence_no_output(void* c) {
    return static_cast<DiffContext*>(c)->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), nullptr);
}
HRESULT create_fence_bad_flags(void* c) {
    void* out = nullptr;
    const HRESULT hr = static_cast<DiffContext*>(c)->device->CreateFence(0, static_cast<D3D12_FENCE_FLAGS>(0x80),
                                                                          __uuidof(ID3D12Fence), &out);
    if (out) static_cast<IUnknown*>(out)->Release();
    return hr;
}
HRESULT create_fence_as_queue(void* c) {
    void* out = reinterpret_cast<void*>(1);
    const HRESULT hr = static_cast<DiffContext*>(c)->device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                                                          __uuidof(ID3D12CommandQueue), &out);
    if (SUCCEEDED(hr) && out) static_cast<IUnknown*>(out)->Release();
    return SUCCEEDED(hr) || out == nullptr ? hr : E_UNEXPECTED;
}
HRESULT create_queue1_no_output(void* c) {
    ID3D12Device9* device9 = nullptr;
    if (FAILED(static_cast<DiffContext*>(c)->device->QueryInterface(IID_PPV_ARGS(&device9)))) return kNoInterface;
    D3D12_COMMAND_QUEUE_DESC desc{};
    const HRESULT hr = device9->CreateCommandQueue1(&desc, __uuidof(ID3D12Device), __uuidof(ID3D12CommandQueue),
                                                    nullptr);
    device9->Release();
    return hr;
}
HRESULT queue_get_device_wrong_iid(void* c) {
    void* out = reinterpret_cast<void*>(1);
    const HRESULT hr = static_cast<DiffContext*>(c)->queue->GetDevice(__uuidof(ID3D12Fence), &out);
    if (SUCCEEDED(hr) && out) static_cast<IUnknown*>(out)->Release();
    return hr;
}
HRESULT queue_get_device_no_output(void* c) {
    return static_cast<DiffContext*>(c)->queue->GetDevice(__uuidof(ID3D12Device), nullptr);
}
HRESULT queue_query_no_output(void* c) {
    return static_cast<DiffContext*>(c)->queue->QueryInterface(__uuidof(ID3D12CommandQueue), nullptr);
}
HRESULT queue_query_wrong_iid(void* c) {
    void* out = reinterpret_cast<void*>(1);
    const HRESULT hr = static_cast<DiffContext*>(c)->queue->QueryInterface(__uuidof(ID3D12Device), &out);
    if (SUCCEEDED(hr) && out) static_cast<IUnknown*>(out)->Release();
    return SUCCEEDED(hr) || out == nullptr ? hr : E_UNEXPECTED;
}
HRESULT queue_private_data_missing(void* c) {
    UINT size = 4;
    UINT data = 0;
    return static_cast<DiffContext*>(c)->queue->GetPrivateData(kMissingKey, &size, &data);
}
HRESULT queue_private_data_small(void* c) {
    const UINT64 value = 42;
    ID3D12CommandQueue* queue = static_cast<DiffContext*>(c)->queue;
    if (FAILED(queue->SetPrivateData(kDataKey, sizeof(value), &value))) return E_FAIL;
    UINT size = 1;
    UINT64 data = 0;
    const HRESULT hr = queue->GetPrivateData(kDataKey, &size, &data);
    return hr == S_OK && size != 1 ? E_UNEXPECTED : hr;
}
HRESULT queue_private_data_size_query(void* c) {
    const UINT64 value = 42;
    ID3D12CommandQueue* queue = static_cast<DiffContext*>(c)->queue;
    if (FAILED(queue->SetPrivateData(kDataKey, sizeof(value), &value))) return E_FAIL;
    UINT size = 0;
    const HRESULT hr = queue->GetPrivateData(kDataKey, &size, nullptr);
    return SUCCEEDED(hr) && size != sizeof(value) ? E_UNEXPECTED : hr;
}
HRESULT queue_signal_null(void* c) { return static_cast<DiffContext*>(c)->queue->Signal(nullptr, 1); }
HRESULT queue_wait_null(void* c) { return static_cast<DiffContext*>(c)->queue->Wait(nullptr, 1); }
HRESULT queue_frequency_null(void* c) { return static_cast<DiffContext*>(c)->queue->GetTimestampFrequency(nullptr); }
HRESULT queue_calibration_null(void* c) {
    return static_cast<DiffContext*>(c)->queue->GetClockCalibration(nullptr, nullptr);
}
HRESULT fence_private_data_missing(void* c) {
    UINT size = 4;
    UINT data = 0;
    return static_cast<DiffContext*>(c)->fence->GetPrivateData(kMissingKey, &size, &data);
}
HRESULT fence_event_reached_null_handle(void* c) {
    return static_cast<DiffContext*>(c)->fence->SetEventOnCompletion(0, nullptr);  // reached: returns at once
}
HRESULT multiple_fences_count0(void* c) {
    auto* d = static_cast<DiffContext*>(c);
    ID3D12Device1* device1 = nullptr;
    if (FAILED(d->device->QueryInterface(IID_PPV_ARGS(&device1)))) return kNoInterface;
    ID3D12Fence* both[] = {d->fence, d->second};
    const UINT64 values[] = {0, 0};
    const HRESULT hr = device1->SetEventOnMultipleFenceCompletion(both, values, 0, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL,
                                                                  d->event);
    device1->Release();
    return hr;
}
HRESULT multiple_fences_bad_flags(void* c) {
    auto* d = static_cast<DiffContext*>(c);
    ID3D12Device1* device1 = nullptr;
    if (FAILED(d->device->QueryInterface(IID_PPV_ARGS(&device1)))) return kNoInterface;
    ID3D12Fence* both[] = {d->fence, d->second};
    const UINT64 values[] = {0, 0};
    const HRESULT hr = device1->SetEventOnMultipleFenceCompletion(
        both, values, 2, static_cast<D3D12_MULTIPLE_FENCE_WAIT_FLAGS>(7), d->event);
    device1->Release();
    return hr;
}
HRESULT multiple_fences_null_entry(void* c) {
    auto* d = static_cast<DiffContext*>(c);
    ID3D12Device1* device1 = nullptr;
    if (FAILED(d->device->QueryInterface(IID_PPV_ARGS(&device1)))) return kNoInterface;
    ID3D12Fence* both[] = {d->fence, nullptr};
    const UINT64 values[] = {0, 0};
    const HRESULT hr = device1->SetEventOnMultipleFenceCompletion(both, values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL,
                                                                  d->event);
    device1->Release();
    return hr;
}
HRESULT make_resident_fence_and_queue(void* c) {
    auto* d = static_cast<DiffContext*>(c);
    ID3D12Pageable* objects[] = {d->fence, d->queue};
    return d->device->MakeResident(2, objects);
}
HRESULT evict_fence_and_queue(void* c) {
    auto* d = static_cast<DiffContext*>(c);
    ID3D12Pageable* objects[] = {d->fence, d->queue};
    const HRESULT hr = d->device->Evict(2, objects);
    d->device->MakeResident(2, objects);
    return hr;
}
HRESULT residency_priority(void* c) {
    auto* d = static_cast<DiffContext*>(c);
    ID3D12Device1* device1 = nullptr;
    if (FAILED(d->device->QueryInterface(IID_PPV_ARGS(&device1)))) return kNoInterface;
    ID3D12Pageable* objects[] = {d->fence, d->queue};
    const D3D12_RESIDENCY_PRIORITY priorities[] = {D3D12_RESIDENCY_PRIORITY_NORMAL, D3D12_RESIDENCY_PRIORITY_HIGH};
    const HRESULT hr = device1->SetResidencyPriority(2, objects, priorities);
    device1->Release();
    return hr;
}
HRESULT enqueue_make_resident(void* c) {
    auto* d = static_cast<DiffContext*>(c);
    ID3D12Device3* device3 = nullptr;
    if (FAILED(d->device->QueryInterface(IID_PPV_ARGS(&device3)))) return kNoInterface;
    ID3D12Pageable* objects[] = {d->fence, d->queue};
    const UINT64 target = d->second->GetCompletedValue() + 1;
    HRESULT hr = device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 2, objects, d->second, target);
    device3->Release();
    if (SUCCEEDED(hr)) {  // the fence must reach the value (the call signals it when the objects are resident)
        if (FAILED(d->second->SetEventOnCompletion(target, d->event)) ||
            WaitForSingleObject(d->event, 5000) != WAIT_OBJECT_0)
            return E_UNEXPECTED;
    }
    return hr;
}
HRESULT shared_handle_unshared_fence(void* c) {
    auto* d = static_cast<DiffContext*>(c);
    HANDLE handle = nullptr;
    const HRESULT hr = d->device->CreateSharedHandle(d->fence, nullptr, GENERIC_ALL, nullptr, &handle);
    if (handle) CloseHandle(handle);
    return hr;
}
HRESULT shared_handle_queue(void* c) {
    auto* d = static_cast<DiffContext*>(c);
    HANDLE handle = nullptr;
    const HRESULT hr = d->device->CreateSharedHandle(d->queue, nullptr, GENERIC_ALL, nullptr, &handle);
    if (handle) CloseHandle(handle);
    return hr;
}
// A descriptor heap from device slot 14 (a heap proxy behind the device proxy) is a pageable as well.
HRESULT residency_descriptor_heap(void* c, bool evict) {
    auto* d = static_cast<DiffContext*>(c);
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 4;
    ID3D12DescriptorHeap* heap = nullptr;
    if (FAILED(d->device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)))) return E_FAIL;
    ID3D12Pageable* objects[] = {heap, d->fence};
    HRESULT hr = evict ? d->device->Evict(2, objects) : d->device->MakeResident(2, objects);
    if (evict && SUCCEEDED(hr)) hr = d->device->MakeResident(2, objects);
    heap->Release();
    return hr;
}
HRESULT make_resident_descriptor_heap(void* c) { return residency_descriptor_heap(c, false); }
HRESULT evict_descriptor_heap(void* c) { return residency_descriptor_heap(c, true); }

#pragma warning(pop)

const DiffCase kDiffCases[] = {
    {"CreateCommandQueue, no output", &create_queue_no_output},
    {"MakeResident(descriptor heap, fence)", &make_resident_descriptor_heap},
    {"Evict + MakeResident(descriptor heap, fence)", &evict_descriptor_heap},
    {"CreateCommandQueue, type 99", &create_queue_bad_type},
    {"CreateCommandQueue, priority 7", &create_queue_bad_priority},
    {"CreateCommandQueue as IID_ID3D12Fence", &create_queue_as_fence},
    {"CreateFence, no output", &create_fence_no_output},
    {"CreateFence, flags 0x80", &create_fence_bad_flags},
    {"CreateFence as IID_ID3D12CommandQueue", &create_fence_as_queue},
    {"CreateCommandQueue1, no output", &create_queue1_no_output},
    {"queue GetDevice as IID_ID3D12Fence", &queue_get_device_wrong_iid},
    {"queue GetDevice, no output", &queue_get_device_no_output},
    {"queue QueryInterface, no output", &queue_query_no_output},
    {"queue QueryInterface as IID_ID3D12Device", &queue_query_wrong_iid},
    {"queue GetPrivateData, missing key", &queue_private_data_missing},
    {"queue GetPrivateData, buffer too small", &queue_private_data_small},
    {"queue GetPrivateData, size query", &queue_private_data_size_query},
    {"queue Signal(null)", &queue_signal_null},
    {"queue Wait(null)", &queue_wait_null},
    {"queue GetTimestampFrequency(null)", &queue_frequency_null},
    {"queue GetClockCalibration(null, null)", &queue_calibration_null},
    {"fence GetPrivateData, missing key", &fence_private_data_missing},
    {"fence SetEventOnCompletion(reached, null)", &fence_event_reached_null_handle},
    {"SetEventOnMultipleFenceCompletion, count 0", &multiple_fences_count0},
    {"SetEventOnMultipleFenceCompletion, flags 7", &multiple_fences_bad_flags},
    {"SetEventOnMultipleFenceCompletion, null entry", &multiple_fences_null_entry},
    {"MakeResident(fence, queue)", &make_resident_fence_and_queue},
    {"Evict(fence, queue)", &evict_fence_and_queue},
    {"SetResidencyPriority(fence, queue)", &residency_priority},
    {"EnqueueMakeResident(fence, queue) -> second fence", &enqueue_make_resident},
    {"CreateSharedHandle(unshared fence)", &shared_handle_unshared_fence},
    {"CreateSharedHandle(queue)", &shared_handle_queue},
};

int make_context(const char* label, ID3D12Device* device, DiffContext& c, Ref<ID3D12CommandQueue>& queue,
                 Ref<ID3D12Fence>& fence, Ref<ID3D12Fence>& second) {
    D3D12_COMMAND_QUEUE_DESC desc{};
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&second.p))));
    c.device = device;
    c.queue = queue.p;
    c.fence = fence.p;
    c.second = second.p;
    return 0;
}

// The real device for a backend kind: "warp", "hw" (the first hardware adapter) or "vulkan" (on `vk`). One
// reference, or null.
ID3D12Device* create_backend_device(const char* kind, vktest::Device& vk) {
    ID3D12Device* device = nullptr;
    if (!std::strcmp(kind, "vulkan")) {
        if (!vk.create()) return nullptr;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        return SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device))) ? device : nullptr;
    }
    Ref<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return nullptr;
    Ref<IDXGIAdapter1> adapter;
    if (!std::strcmp(kind, "warp")) {
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) return nullptr;
    } else {
        for (UINT i = 0; !adapter.p; ++i) {
            if (factory->EnumAdapters1(i, &adapter.p) == DXGI_ERROR_NOT_FOUND) return nullptr;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) adapter.reset();
        }
    }
    return SUCCEEDED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))) ? device : nullptr;
}

// Child mode (--diff <kind> <case> real|proxy): one case on its own device, in its own process, because the PC
// runtime's device is one object per adapter and process, and some error cases make it remove that device (it does
// for SetResidencyPriority on a fence) or fault inside it. Prints "DIFF <ran> <hr> <removed> <validation errors>".
int run_diff_child(const char* kind, int index, bool through_proxy) {
    if (index < 0 || index >= static_cast<int>(std::size(kDiffCases))) return 2;
    vktest::Device vk;
    ID3D12Device* real = create_backend_device(kind, vk);
    if (!real) return 3;
    Ref<ID3D12Device> device;
    if (through_proxy) {
        if (FAILED(xs::create_device_proxy(real, __uuidof(ID3D12Device), reinterpret_cast<void**>(&device.p)))) {
            real->Release();
            return 3;
        }
        if (!std::strcmp(kind, "vulkan")) rt::d3d12::set_device_outer(real, device.p);
    } else {
        device.p = real;
    }
    const char* label = kind;
    Event event;
    DiffContext context{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence, second;
    if (make_context(label, device.p, context, queue, fence, second)) return 3;
    EXPECT(xs::is_queue_proxy(context.queue) == through_proxy && xs::is_fence_proxy(context.fence) == through_proxy);
    context.event = event.h;
    HRESULT result = S_OK;
    const bool ran = run_guarded(kDiffCases[index].run, &context, &result);
    const HRESULT removed = ran ? device->GetDeviceRemovedReason() : S_OK;
    std::printf("DIFF %d 0x%08lx 0x%08lx %u\n", ran ? 1 : 0, static_cast<unsigned long>(result),
                static_cast<unsigned long>(removed), vk.validation_errors());
    std::fflush(stdout);
    // A faulted case may have left the runtime in any state: leave without releasing anything.
    if (!ran) TerminateProcess(GetCurrentProcess(), 0);
    return 0;
}

struct ChildResult {
    bool ok = false;  // the child ran and reported
    int ran = 0;
    unsigned long hr = 0, removed = 0, validation = 0;
};

// How a child of this executable ended: it exited by itself, it outlived the limit and was ended, or it could not be
// started or watched (the reason is printed on stderr).
enum class ChildEnd { exited, timed_out, failed };

constexpr DWORD kChildLimitMs = 60000;              // one differential case, including device creation
constexpr std::size_t kChildOutputKeep = 1u << 20;  // the most of a child's stdout kept: its last 1 MiB

// Failures the runner's self-test injects: the pipe, or the thread that drains it, cannot be made.
enum class RunnerFault { none, pipe, thread };

struct ChildRun {
    ChildEnd end = ChildEnd::failed;
    DWORD exit_code = ~0ul;          // the child's exit code (1 when the runner ended it), ~0 when it never ran
    unsigned long long printed = 0;  // bytes it printed on stdout in all
    std::string tail;                // the last kChildOutputKeep of them, at most
};

void report_runner_error(const char* what) {
    std::fprintf(stderr, "child runner: %s failed (error %lu)\n", what, GetLastError());
}

// Runs `exe` with `arguments` and collects its stdout. A thread drains the pipe while the child runs, so a child that
// prints more than the pipe buffers keeps running instead of blocking in WriteFile until the limit ends it and it is
// taken for a hung one. Only the last kChildOutputKeep bytes are kept. A child still running after `limit_ms` is
// ended with exit code 1, and what it printed until then is still returned.
ChildRun run_process(const wchar_t* exe, const wchar_t* arguments, DWORD limit_ms,
                     RunnerFault fault = RunnerFault::none) {
    ChildRun run;
    std::wstring command = L"\"" + std::wstring(exe) + L"\" " + arguments;
    // Both ends are made uninheritable. The write end becomes inheritable only while the start lock is held, just for
    // its own child's CreateProcessW: children are started from several threads at once (check_differential), and a
    // write end inheritable at any other time would also go to a sibling started meanwhile, which would then keep this
    // pipe open after this child ended.
    HANDLE read_end = nullptr, write_end = nullptr;
    if (fault == RunnerFault::pipe) SetLastError(ERROR_NOT_ENOUGH_MEMORY);
    if (fault == RunnerFault::pipe || !CreatePipe(&read_end, &write_end, nullptr, 0)) {
        report_runner_error("CreatePipe");
        return run;
    }
    STARTUPINFOW startup{sizeof(startup)};
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = write_end;
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION process{};
    BOOL started = FALSE;
    DWORD start_error = 0;
    {
        static std::mutex start_lock;
        const std::lock_guard<std::mutex> hold(start_lock);
        if (!SetHandleInformation(write_end, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT)) {
            report_runner_error("SetHandleInformation");
            CloseHandle(read_end);
            CloseHandle(write_end);
            return run;
        }
        started = CreateProcessW(exe, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &process);
        start_error = GetLastError();
        CloseHandle(write_end);  // the child holds the only write end now: the reader sees the pipe end when it is gone
    }
    if (!started) {
        std::fprintf(stderr, "child runner: CreateProcessW \"%ls\" failed (error %lu)\n", exe, start_error);
        CloseHandle(read_end);
        return run;
    }
    CloseHandle(process.hThread);
    const auto end_child = [&] {
        if (!TerminateProcess(process.hProcess, 1) && WaitForSingleObject(process.hProcess, 0) != WAIT_OBJECT_0)
            report_runner_error("TerminateProcess");
        if (WaitForSingleObject(process.hProcess, 5000) != WAIT_OBJECT_0)
            std::fprintf(stderr, "child runner: the ended child is still there after 5 s\n");
    };

    std::atomic<bool> stop{false};
    std::thread reader;
    try {
        if (fault == RunnerFault::thread)
            throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
        reader = std::thread([read_end, &run, &stop] {
            char buffer[4096];
            DWORD got = 0;
            while (!stop.load() && ReadFile(read_end, buffer, sizeof(buffer), &got, nullptr) && got) {
                run.printed += got;
                run.tail.append(buffer, got);
                if (run.tail.size() > 2 * kChildOutputKeep) run.tail.erase(0, run.tail.size() - kChildOutputKeep);
            }
        });
    } catch (const std::system_error& e) {
        std::fprintf(stderr, "child runner: no thread to read the child's output (%s); the child is ended\n", e.what());
        end_child();
        if (!GetExitCodeProcess(process.hProcess, &run.exit_code)) run.exit_code = ~0ul;
        CloseHandle(process.hProcess);
        CloseHandle(read_end);
        return run;
    }

    const DWORD waited = WaitForSingleObject(process.hProcess, limit_ms);
    if (waited == WAIT_OBJECT_0) {
        run.end = ChildEnd::exited;
    } else {
        if (waited == WAIT_FAILED) report_runner_error("WaitForSingleObject");
        end_child();
        run.end = waited == WAIT_TIMEOUT ? ChildEnd::timed_out : ChildEnd::failed;
    }
    // The child and its write end are gone, so the reader stops at the end of the pipe. Should something else still
    // hold the write end (a process the child started), its blocked read is cancelled rather than waited on.
    const HANDLE thread = reader.native_handle();
    if (WaitForSingleObject(thread, 5000) != WAIT_OBJECT_0) {
        std::fprintf(stderr, "child runner: the pipe is still open after the child ended; the read is cancelled\n");
        stop.store(true);
        DWORD waited_reader = WAIT_TIMEOUT;
        for (int tries = 0; tries < 50 && waited_reader == WAIT_TIMEOUT; ++tries) {  // 5 s at most
            CancelSynchronousIo(thread);
            waited_reader = WaitForSingleObject(thread, 100);
        }
        if (waited_reader != WAIT_OBJECT_0) {
            // The reader can neither be joined nor left behind (it writes into `run`): fail now rather than hang.
            std::fprintf(stderr, "child runner: the reader thread cannot be stopped (wait 0x%lx, error %lu)\n",
                         waited_reader, GetLastError());
            std::fflush(stderr);
            TerminateProcess(GetCurrentProcess(), 1);
        }
    }
    reader.join();
    if (run.tail.size() > kChildOutputKeep) run.tail.erase(0, run.tail.size() - kChildOutputKeep);
    if (!GetExitCodeProcess(process.hProcess, &run.exit_code)) {
        report_runner_error("GetExitCodeProcess");
        run.exit_code = ~0ul;
    }
    CloseHandle(process.hProcess);
    CloseHandle(read_end);
    return run;
}

// This executable's own path, for its children; empty (and reported) when it cannot be had in full.
std::wstring self_path() {
    wchar_t exe[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameW(nullptr, exe, MAX_PATH);
    if (!length || length >= MAX_PATH) {
        report_runner_error("GetModuleFileNameW");
        return {};
    }
    return exe;
}

ChildRun run_self(const wchar_t* arguments, DWORD limit_ms, RunnerFault fault = RunnerFault::none) {
    const std::wstring exe = self_path();
    return exe.empty() ? ChildRun{} : run_process(exe.c_str(), arguments, limit_ms, fault);
}

// The last "DIFF ..." line a child printed.
bool parse_diff_report(const std::string& output, ChildResult& r) {
    const std::size_t at = output.rfind("DIFF ");
    return at != std::string::npos &&
           sscanf_s(output.c_str() + at, "DIFF %d %lx %lx %lu", &r.ran, &r.hr, &r.removed, &r.validation) == 4;
}

ChildResult run_child(const char* kind, int index, bool through_proxy) {
    ChildResult r;
    wchar_t arguments[64];
    std::swprintf(arguments, std::size(arguments), L"--diff %hs %d %hs", kind, index,
                  through_proxy ? "proxy" : "real");
    const ChildRun run = run_self(arguments, kChildLimitMs);
    const char* side = through_proxy ? "proxy" : "real";
    if (run.end == ChildEnd::timed_out)
        std::fprintf(stderr, "%s case %d (%s): the child did not finish within %lu s and was ended\n", kind, index,
                     side, kChildLimitMs / 1000);
    if (run.end != ChildEnd::exited) return r;
    r.ok = parse_diff_report(run.tail, r);
    if (!r.ok)
        std::fprintf(stderr, "%s case %d (%s): the child exited with 0x%08lx without a report\n", kind, index, side,
                     run.exit_code);
    return r;
}

constexpr DWORD kCrashCode = 0xE04C4E44;  // the self-test child's own exception code, raised and not handled

// Self-test child modes of the runner, no device: print `bytes` bytes, then exit with a report (--child-flood),
// never return (--child-hang), or end on an unhandled exception (--child-crash).
int run_runner_child(const char* mode, unsigned long bytes) {
    const std::string chunk(4096, 'x');
    for (unsigned long left = bytes; left;) {
        const unsigned long n = left < chunk.size() ? left : static_cast<unsigned long>(chunk.size());
        if (std::fwrite(chunk.data(), 1, n, stdout) != n) return 3;
        left -= n;
    }
    if (std::strcmp(mode, "--child-hang") == 0) Sleep(INFINITE);
    if (std::strcmp(mode, "--child-crash") == 0) {
        SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);  // no error dialog to wait on
        RaiseException(kCrashCode, EXCEPTION_NONCONTINUABLE, 0, nullptr);
    }
    std::printf("\nDIFF 1 0x00000000 0x00000000 0\n");
    return 0;
}

// The runner itself, on every machine (it needs no device):
//   - a child that exits normally is read and its report found;
//   - one printing 8 MiB (128 times a 64 KiB pipe buffer) finishes, all of it is counted, and only the tail is kept;
//   - hung children, silent or after printing 256 KiB, are ended at the limit and their output still read;
//   - a child that crashes, or cannot be started, is reported, not taken for a report;
//   - an injected pipe or reader-thread failure is reported, and the child of the latter is ended, not waited on;
//   - no handle is left behind by any of it.
int check_child_runner() {
    const char* label = "child runner";
    ChildResult report;
    ChildRun run;
    // The first two successful starts and the first failed start make Windows load and cache their persistent process
    // handles (on Windows 11 26200, a second CreateProcessW caches the HKCU root key and a failed CreateProcessW keeps
    // 41-42 AppCompat/PCA handles; later starts add none). Warm up both before reading handles_before.
    for (int warmup = 0; warmup < 2; ++warmup) {
        run = run_self(L"--child-flood 0", 30000);
        EXPECT(run.end == ChildEnd::exited && run.exit_code == 0 && parse_diff_report(run.tail, report));
        EXPECT(report.ran == 1 && report.hr == 0 && report.removed == 0 && report.validation == 0);
    }
    std::fprintf(stderr, "child runner: the next four errors are expected\n");
    const std::wstring missing = self_path() + L".missing.exe";
    run = run_process(missing.c_str(), L"", 5000);
    EXPECT(run.end == ChildEnd::failed && run.exit_code == ~0ul && run.printed == 0);
    DWORD handles_before = 0, handles_after = 0;
    EXPECT(GetProcessHandleCount(GetCurrentProcess(), &handles_before));

    constexpr unsigned long long kFlood = 8ull << 20, kBeforeHang = 256ull << 10;
    run = run_self(L"--child-flood 8388608", 30000);
    EXPECT(run.end == ChildEnd::exited && run.exit_code == 0);
    EXPECT(run.printed > kFlood && run.tail.size() <= kChildOutputKeep && parse_diff_report(run.tail, report));

    ULONGLONG start = GetTickCount64();
    run = run_self(L"--child-hang 262144", 2000);
    ULONGLONG took = GetTickCount64() - start;
    EXPECT(run.end == ChildEnd::timed_out && run.exit_code == 1 && run.printed == kBeforeHang);
    EXPECT(took >= 1900 && took < 15000);

    start = GetTickCount64();
    run = run_self(L"--child-hang 0", 1000);
    took = GetTickCount64() - start;
    EXPECT(run.end == ChildEnd::timed_out && run.exit_code == 1 && run.printed == 0 && run.tail.empty());
    EXPECT(took >= 900 && took < 15000);

    run = run_self(L"--child-crash 100", 30000);
    EXPECT(run.end == ChildEnd::exited && run.exit_code != 0 && run.printed == 100);
    EXPECT(!parse_diff_report(run.tail, report));
    const DWORD crash_code = run.exit_code;

    run = run_process(missing.c_str(), L"", 5000);
    EXPECT(run.end == ChildEnd::failed && run.exit_code == ~0ul && run.printed == 0);
    run = run_self(L"--child-flood 16", 5000, RunnerFault::pipe);
    EXPECT(run.end == ChildEnd::failed && run.exit_code == ~0ul);
    start = GetTickCount64();
    run = run_self(L"--child-hang 0", kChildLimitMs, RunnerFault::thread);
    took = GetTickCount64() - start;
    EXPECT(run.end == ChildEnd::failed && run.exit_code == 1 && took < 15000);

    EXPECT(GetProcessHandleCount(GetCurrentProcess(), &handles_after));
    if (handles_after != handles_before)
        std::fprintf(stderr, "%s: %lu handles before, %lu after\n", label, handles_before, handles_after);
    EXPECT(handles_after == handles_before);
    std::printf("%s: normal exit, 8 MiB of output (tail kept), hung children (silent and after 256 KiB), a crash "
                "(0x%08lx), a missing executable and injected pipe and thread failures behave, no handle left\n",
                label, crash_code);
    return 0;
}

// How many differential children run at once: LEONIDA_QUEUE_PROXY_JOBS when set (1 = one after another, as before),
// else half the logical processors, from 1 to 8. Each child pays its own device start-up (about 5 s each on an AMD
// RDNA2 iGPU, TEST-242), and that waiting is what running them side by side hides.
unsigned differential_jobs() {
    char value[16] = {};
    const DWORD length = GetEnvironmentVariableA("LEONIDA_QUEUE_PROXY_JOBS", value, sizeof(value));
    if (length > 0 && length < sizeof(value)) {  // set, and short enough to be a number
        const unsigned long jobs = std::strtoul(value, nullptr, 10);
        if (jobs >= 1 && jobs <= 64) return static_cast<unsigned>(jobs);
    }
    const unsigned half = std::thread::hardware_concurrency() / 2;
    return half < 1 ? 1 : half > 8 ? 8 : half;
}

// Cases that fault in the PC runtime (17..20, 25), modify device residency (1, 2, 26..29), or touch the shared
// handle subsystem (30, 31) are isolated in child processes so the host device remains completely clean and intact.
// Non-destructive argument-validation and proxy-wrapping cases run in-process for speed (<1 ms each).
bool is_isolated_diff_case(int index) {
    if (index == 1 || index == 2) return true;         // residency (descriptor heap)
    if (index >= 17 && index <= 20) return true;       // faults on PC runtime (queue null args)
    if (index == 25) return true;                      // faults on PC runtime (null fence entry)
    if (index >= 26 && index <= 29) return true;       // residency (MakeResident/Evict/Priority/Enqueue)
    if (index == 30 || index == 31) return true;       // shared handle creation
    return false;
}

ChildResult run_in_process(const char* label, ID3D12Device* target_device, int index, bool through_proxy,
                           unsigned long validation_errors = 0) {
    ChildResult r{};
    Event event;
    DiffContext context{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence, second;
    if (make_context(label, target_device, context, queue, fence, second)) return r;
    if (xs::is_queue_proxy(context.queue) != through_proxy || xs::is_fence_proxy(context.fence) != through_proxy) {
        return r;
    }
    context.event = event.h;
    HRESULT result = S_OK;
    const bool ran = run_guarded(kDiffCases[index].run, &context, &result);
    const HRESULT removed = ran ? target_device->GetDeviceRemovedReason() : S_OK;
    r.ok = true;
    r.ran = ran ? 1 : 0;
    r.hr = static_cast<unsigned long>(result);
    r.removed = static_cast<unsigned long>(removed);
    r.validation = validation_errors;
    return r;
}

// Each case, in a child process for the real object and in another for the proxy: the same HRESULT, the device
// removed (or not) alike, and no validation errors. A case the real runtime faults on is reported, not compared.
// Non-destructive argument-validation and proxy cases run in-process (<1 ms each instead of ~5 s per device creation).
// Faulting, residency, or shared cases run in isolated child processes using differential_jobs() workers.
int check_differential(const Backend& b) {
    const char* label = b.label;
    int compared = 0, faulting = 0;
    constexpr int kCases = static_cast<int>(std::size(kDiffCases));
    std::vector<ChildResult> results(2 * kCases);  // [2 * case] real, [2 * case + 1] proxy

    const bool force_isolate_all = GetEnvironmentVariableA("LEONIDA_QUEUE_PROXY_ISOLATE_ALL", nullptr, 0) > 0;

    std::vector<int> child_jobs;
    for (int i = 0; i < kCases; ++i) {
        if (force_isolate_all || is_isolated_diff_case(i)) {
            child_jobs.push_back(2 * i);      // real
            child_jobs.push_back(2 * i + 1);  // proxy
        } else {
            results[2 * i] = run_in_process(label, b.real, i, false);
            results[2 * i + 1] = run_in_process(label, b.device, i, true);
        }
    }

    if (!child_jobs.empty()) {
        std::atomic<size_t> next{0};
        const auto work = [&] {
            for (size_t idx; (idx = next.fetch_add(1)) < child_jobs.size();) {
                const int job = child_jobs[idx];
                results[job] = run_child(b.kind, job / 2, job % 2 == 1);
            }
        };
        const unsigned jobs = differential_jobs();
        std::vector<std::thread> pool;
        try {
            for (unsigned t = 1; t < jobs && t < child_jobs.size(); ++t) pool.emplace_back(work);
        } catch (const std::system_error& e) {
            std::fprintf(stderr, "%s: %zu of %u differential runner threads (%s)\n", label, pool.size() + 1, jobs,
                         e.what());
        }
        work();
        for (std::thread& t : pool) t.join();
    }
    for (int i = 0; i < kCases; ++i) {
        const char* name = kDiffCases[i].name;
        const ChildResult& want = results[2 * i];
        if (!want.ok) {
            std::fprintf(stderr, "%s: %s: the real-object child did not report\n", label, name);
            return 1;
        }
        if (!want.ran) {
            std::printf("%s: %s: the real object faults, not compared\n", label, name);
            ++faulting;
            continue;
        }
        const ChildResult& got = results[2 * i + 1];
        if (!got.ok || !got.ran || got.hr != want.hr || got.removed != want.removed || got.validation ||
            want.validation) {
            std::fprintf(stderr,
                         "%s: %s: proxy %s 0x%08lx (removed 0x%08lx, %lu validation errors), real 0x%08lx (removed "
                         "0x%08lx, %lu validation errors)\n",
                         label, name, !got.ok ? "did not report" : got.ran ? "returned" : "faulted", got.hr,
                         got.removed, got.validation, want.hr, want.removed, want.validation);
            return 1;
        }
        ++compared;
    }
    std::printf("%s: %d error and edge cases give the real object's HRESULT and removal state through the proxy, "
                "each in its own process (%d not comparable)\n", label, compared, faulting);
    return 0;
}

// ---- 4. Identity, references and release orders ----

int check_identity(const Backend& b) {
    const char* label = b.label;
    ID3D12Device* device = b.device;
    const unsigned long device_refs = xs::device_proxy_ref_count(device);
    {
        Ref<ID3D12Fence> fence;
        const HRESULT made = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p));
        if (FAILED(made))
            std::fprintf(stderr, "%s: CreateFence 0x%08lx, removed reason 0x%08lx\n", label,
                         static_cast<unsigned long>(made),
                         static_cast<unsigned long>(device->GetDeviceRemovedReason()));
        EXPECT(SUCCEEDED(made));
        const bool has1 = xs::fence_proxy_slot_count(fence.p) == 12;
        // Every modelled interface is the same pointer, and QueryInterface is reflexive, symmetric, transitive.
        Ref<IUnknown> unknown;
        Ref<ID3D12Object> object;
        Ref<ID3D12DeviceChild> child;
        Ref<ID3D12Pageable> pageable;
        Ref<ID3D12Fence> again;
        EXPECT(SUCCEEDED(fence->QueryInterface(IID_PPV_ARGS(&unknown.p))));
        EXPECT(SUCCEEDED(unknown->QueryInterface(IID_PPV_ARGS(&object.p))));
        EXPECT(SUCCEEDED(object->QueryInterface(IID_PPV_ARGS(&child.p))));
        EXPECT(SUCCEEDED(child->QueryInterface(IID_PPV_ARGS(&pageable.p))));
        EXPECT(SUCCEEDED(pageable->QueryInterface(IID_PPV_ARGS(&again.p))));
        void* const p = fence.p;
        EXPECT(unknown.p == p && object.p == p && child.p == p && pageable.p == p && again.p == p);
        Ref<ID3D12Fence1> fence1;
        const HRESULT hr1 = again->QueryInterface(IID_PPV_ARGS(&fence1.p));
        EXPECT(has1 ? SUCCEEDED(hr1) && static_cast<void*>(fence1.p) == p : hr1 == E_NOINTERFACE);
        EXPECT(xs::command_object_proxy_ref_count(p) == (has1 ? 7u : 6u));
        // Interfaces the proxy does not model are refused, also those the real object has.
        Ref<IUnknown> other;
        EXPECT(fence->QueryInterface(__uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&other.p)) ==
               E_NOINTERFACE && !other.p);
        EXPECT(fence->QueryInterface(__uuidof(ID3D12Device), reinterpret_cast<void**>(&other.p)) == E_NOINTERFACE);
        // AddRef through one interface, Release through another.
        EXPECT(child->AddRef() == (has1 ? 8u : 7u));
        EXPECT(pageable->Release() == (has1 ? 7u : 6u));
        // Release in an order unlike the acquisition.
        child.reset();
        unknown.reset();
        fence1.reset();
        object.reset();
        again.reset();
        pageable.reset();
        EXPECT(xs::command_object_proxy_ref_count(p) == 1);
    }
    {
        // The queue the same way, and its device: GetDevice and QueryInterface of the device give one object.
        D3D12_COMMAND_QUEUE_DESC desc{};
        Ref<IUnknown> queue;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&desc, __uuidof(IUnknown), reinterpret_cast<void**>(&queue.p))));
        EXPECT(xs::is_queue_proxy(queue.p));
        Ref<ID3D12CommandQueue> typed;
        EXPECT(SUCCEEDED(queue->QueryInterface(IID_PPV_ARGS(&typed.p))) && static_cast<void*>(typed.p) == queue.p);
        Ref<ID3D12Device> owner;
        Ref<ID3D12Device1> owner1;
        EXPECT(SUCCEEDED(typed->GetDevice(IID_PPV_ARGS(&owner.p))));
        EXPECT(SUCCEEDED(typed->GetDevice(IID_PPV_ARGS(&owner1.p))));
        EXPECT(owner.p == device && static_cast<void*>(owner1.p) == static_cast<void*>(device));
        Ref<ID3D12Fence> not_a_device;
        EXPECT(typed->GetDevice(IID_PPV_ARGS(&not_a_device.p)) == E_NOINTERFACE && !not_a_device.p);

        // A null IID: the PC runtime dereferences it (no answer to compare with); the proxies refuse it with
        // E_POINTER and leave the output null, on every slot that takes one.
        const D3D12_COMMAND_QUEUE_DESC desc0{};
        void* out = reinterpret_cast<void*>(1);
        EXPECT(as_hr(raw_call(device, 8, reinterpret_cast<std::uint64_t>(&desc0), 0,
                              reinterpret_cast<std::uint64_t>(&out))) == E_POINTER);
        EXPECT(as_hr(raw_call(device, 36, 0, 0, 0, reinterpret_cast<std::uint64_t>(&out))) == E_POINTER);
        EXPECT(as_hr(raw_call(device, 32, 0, 0, reinterpret_cast<std::uint64_t>(&out))) == E_POINTER);
        if (xs::device_proxy_slot_count(device) > 75)
            EXPECT(as_hr(raw_call(device, 75, reinterpret_cast<std::uint64_t>(&desc0), 0, 0,
                                  reinterpret_cast<std::uint64_t>(&out))) == E_POINTER);
        out = reinterpret_cast<void*>(1);
        EXPECT(as_hr(raw_call(typed.p, 0, 0, reinterpret_cast<std::uint64_t>(&out))) == E_POINTER && !out);
        EXPECT(as_hr(raw_call(typed.p, 7, 0, reinterpret_cast<std::uint64_t>(&out))) == E_POINTER);
        EXPECT(xs::command_object_proxy_ref_count(typed.p) == 2);  // queue + typed: nothing leaked or added
    }
    EXPECT(xs::device_proxy_ref_count(device) == device_refs);
    std::printf("%s: QueryInterface identity, references across interfaces and release orders hold\n", label);
    return 0;
}

// ---- 5. Shared fences ----

int check_shared(const Backend& b) {
    const char* label = b.label;
    Ref<ID3D12Fence> fence;
    const HRESULT created = b.device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence.p));
    {
        Ref<ID3D12Fence> probe;
        EXPECT(created == b.real->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&probe.p)));
    }
    if (FAILED(created)) {
        std::printf("%s: no shared fences (0x%08lx, as on the real device), not checked\n", label,
                    static_cast<unsigned long>(created));
        return 0;
    }
    HANDLE handle = nullptr;
    g_shared_created = true;
    const HRESULT shared = b.device->CreateSharedHandle(fence.p, nullptr, GENERIC_ALL, nullptr, &handle);
    Ref<ID3D12Fence> raw;
    HANDLE raw_handle = nullptr;
    EXPECT(SUCCEEDED(b.real->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&raw.p))));
    const HRESULT raw_shared = b.real->CreateSharedHandle(raw.p, nullptr, GENERIC_ALL, nullptr, &raw_handle);
    if (raw_handle) CloseHandle(raw_handle);
    EXPECT(shared == raw_shared);  // the proxy unwrapped the fence for the real device
    if (FAILED(shared)) {
        std::printf("%s: shared fences not supported (0x%08lx), OpenSharedHandle not checked\n", label,
                    static_cast<unsigned long>(shared));
        return 0;
    }
    Ref<ID3D12Fence> opened;
    Ref<IUnknown> opened_unknown;
    EXPECT(SUCCEEDED(b.device->OpenSharedHandle(handle, IID_PPV_ARGS(&opened.p))));
    EXPECT(SUCCEEDED(b.device->OpenSharedHandle(handle, __uuidof(IUnknown),
                                                reinterpret_cast<void**>(&opened_unknown.p))));
    CloseHandle(handle);
    EXPECT(xs::is_fence_proxy(opened.p) && xs::is_fence_proxy(opened_unknown.p));  // also when asked as IUnknown
    EXPECT(xs::command_object_proxy_ref_count(opened.p) == 1);
    EXPECT(xs::command_object_proxy_ref_count(opened_unknown.p) == 1);
    EXPECT(xs::device_slot_hits(32) >= 2);
    EXPECT(SUCCEEDED(fence->Signal(4)));
    EXPECT(opened->GetCompletedValue() == 4);  // one fence, seen through both proxies
    g_shared_checked = true;

    // Not a fence: a shared buffer goes through CreateSharedHandle (31) as it is and comes back from
    // OpenSharedHandle (32) as the real resource, not a proxy.
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = 65536;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> resource;
    if (SUCCEEDED(b.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &buffer,
                                                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                    IID_PPV_ARGS(&resource.p)))) {
        HANDLE resource_handle = nullptr;
        EXPECT(SUCCEEDED(b.device->CreateSharedHandle(resource.p, nullptr, GENERIC_ALL, nullptr, &resource_handle)));
        Ref<ID3D12Resource> reopened;
        const HRESULT reopen = b.device->OpenSharedHandle(resource_handle, IID_PPV_ARGS(&reopened.p));
        CloseHandle(resource_handle);
        EXPECT(SUCCEEDED(reopen) && reopened.p && !xs::is_fence_proxy(reopened.p) && !xs::is_queue_proxy(reopened.p));
        EXPECT(reopened->GetDesc().Width == buffer.Width);
        // An invalid (null) handle: the real runtime's own answer, and no object. Not the closed handle above: its
        // value may already name another object of this process.
        Ref<ID3D12Fence> none, none_real;
        none.p = reinterpret_cast<ID3D12Fence*>(1);
#pragma warning(suppress : 6387)  // the null handle is the invalid input under test
        const HRESULT invalid = b.device->OpenSharedHandle(nullptr, IID_PPV_ARGS(&none.p));
        const bool cleared = !none.p;
        none.p = nullptr;
#pragma warning(suppress : 6387)  // the same input on the real device
        const HRESULT invalid_real = b.real->OpenSharedHandle(nullptr, IID_PPV_ARGS(&none_real.p));
        EXPECT(FAILED(invalid) && cleared && invalid == invalid_real && !none_real.p);
    } else {
        std::printf("%s: no shared committed buffer, the non-fence OpenSharedHandle not checked\n", label);
    }
    std::printf("%s: CreateSharedHandle unwraps the fence and OpenSharedHandle wraps it; a shared buffer and an "
                "invalid handle pass through unchanged\n", label);
    return 0;
}

// ---- 8. A console-built client, end to end ----
//
// The way a library compiled against the console device calls it: raw vtable slots, the measured Xbox shapes where
// they differ (device slot 14 hands out a heap proxy whose slot 9 returns the handle in RAX; slot 76 creates a
// resource from an Xbox description with the layout word 0x1), and the queue and fence slots for its GPU work. The
// shipped package has no queue or fence client (docs/d3d12_command_objects.md §3.5), so this is the synthetic stand-in
// for what bvh_builder_test and fsr_dispatch_test cannot show: real copies executed through the queue proxy, finished
// through the fence proxy and read back byte for byte, and heap, queue and fence proxies in the residency calls.

std::uint64_t raw(void* object, unsigned slot, std::uint64_t a1 = 0, std::uint64_t a2 = 0, std::uint64_t a3 = 0,
                  std::uint64_t a4 = 0, std::uint64_t a5 = 0, std::uint64_t a6 = 0, std::uint64_t a7 = 0) {
    void** vtable = *reinterpret_cast<void***>(object);
    return reinterpret_cast<SlotFn>(vtable[slot])(object, a1, a2, a3, a4, a5, a6, a7, 0, 0, 0, 0, 0, 0, 0);
}

template<class T>
std::uint64_t word(T* pointer) { return reinterpret_cast<std::uint64_t>(pointer); }

int check_xbox_client(const Backend& b) {
    const char* label = b.label;
    void* device = b.device;
    constexpr UINT64 kBytes = 256;

    // Device slot 76: buffers from an Xbox description (layout word 0x1, row-major buffer).
    const auto create_buffer = [&](D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state, ID3D12Resource** out) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = type;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = kBytes;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x1);
        const IID iid = __uuidof(ID3D12Resource);
        return as_hr(raw(device, 76, word(&heap), D3D12_HEAP_FLAG_NONE, word(&desc), state, 0, word(&iid), word(out)));
    };
    Ref<ID3D12Resource> upload, gpu, readback;
    EXPECT(SUCCEEDED(create_buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, &upload.p)));
    EXPECT(SUCCEEDED(create_buffer(D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, &gpu.p)));
    EXPECT(SUCCEEDED(create_buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, &readback.p)));
    {
        std::uint8_t* bytes = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&bytes))));
        for (UINT64 i = 0; i < kBytes; ++i) bytes[i] = static_cast<std::uint8_t>(i * 7 + 3);
        upload->Unmap(0, nullptr);
    }

    // Device slot 14 and heap slot 9 in the Xbox form (one argument, handle in RAX).
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 8;
    const IID iid_heap = __uuidof(ID3D12DescriptorHeap);
    Ref<ID3D12DescriptorHeap> heap;
    EXPECT(SUCCEEDED(as_hr(raw(device, 14, word(&heap_desc), word(&iid_heap), word(&heap.p)))));
    EXPECT(xs::is_heap_proxy(heap.p));
    EXPECT(raw(heap.p, 9) == xs::heap_proxy_real_heap(heap.p)->GetCPUDescriptorHandleForHeapStart().ptr);

    // Device slots 8 and 36.
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const IID iid_queue = __uuidof(ID3D12CommandQueue), iid_fence = __uuidof(ID3D12Fence);
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence, second;
    EXPECT(SUCCEEDED(as_hr(raw(device, 8, word(&queue_desc), word(&iid_queue), word(&queue.p)))));
    EXPECT(SUCCEEDED(as_hr(raw(device, 36, 0, D3D12_FENCE_FLAG_NONE, word(&iid_fence), word(&fence.p)))));
    EXPECT(SUCCEEDED(as_hr(raw(device, 36, 0, D3D12_FENCE_FLAG_NONE, word(&iid_fence), word(&second.p)))));
    EXPECT(xs::is_queue_proxy(queue.p) && xs::is_fence_proxy(fence.p) && xs::is_fence_proxy(second.p));

    // Residency with heap, queue and fence proxies among the objects (device slots 34, 35, 46).
    ID3D12Pageable* pageables[] = {heap.p, gpu.p, fence.p, queue.p};
    EXPECT(SUCCEEDED(as_hr(raw(device, 34, 4, word(pageables)))));
    EXPECT(SUCCEEDED(as_hr(raw(device, 35, 4, word(pageables)))));
    EXPECT(SUCCEEDED(as_hr(raw(device, 34, 4, word(pageables)))));
    ID3D12Pageable* resources[] = {gpu.p};
    const D3D12_RESIDENCY_PRIORITY priority[] = {D3D12_RESIDENCY_PRIORITY_HIGH};
    EXPECT(SUCCEEDED(as_hr(raw(device, 46, 1, word(resources), word(priority)))));

    // The copies: upload -> GPU buffer -> readback, executed through queue slot 10 and finished through slot 14
    // and fence slot 9.
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(b.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(b.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                 IID_PPV_ARGS(&list.p))));
    list->CopyBufferRegion(gpu.p, 0, upload.p, 0, kBytes);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = gpu.p;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(readback.p, 0, gpu.p, 0, kBytes);
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    raw(queue.p, 10, 1, word(lists));
    EXPECT(SUCCEEDED(as_hr(raw(queue.p, 14, word(fence.p), 1))));
    Event done;
    EXPECT(SUCCEEDED(as_hr(raw(fence.p, 9, 1, word(done.h)))) && done.set_within(5000));
    EXPECT(raw(fence.p, 8) >= 1);
    {
        std::uint8_t* bytes = nullptr;
        const D3D12_RANGE read_range{0, kBytes};
        EXPECT(SUCCEEDED(readback->Map(0, &read_range, reinterpret_cast<void**>(&bytes))));
        bool same = true;
        for (UINT64 i = 0; i < kBytes; ++i) same = same && bytes[i] == static_cast<std::uint8_t>(i * 7 + 3);
        const D3D12_RANGE nothing{0, 0};
        readback->Unmap(0, &nothing);
        EXPECT(same);  // the GPU really ran the list the queue proxy submitted
    }

    // Device slot 50: EnqueueMakeResident with the objects and a fence proxy to signal.
    EXPECT(SUCCEEDED(as_hr(raw(device, 50, D3D12_RESIDENCY_FLAG_NONE, 4, word(pageables), word(fence.p), 2))));
    EXPECT(SUCCEEDED(as_hr(raw(fence.p, 9, 2, word(done.h)))) && done.set_within(5000));

    // Queue slot 15 holds the queue until the CPU (fence slot 10) signals; device slot 45 then sees both fences.
    EXPECT(SUCCEEDED(as_hr(raw(queue.p, 15, word(fence.p), 3))));
    EXPECT(SUCCEEDED(as_hr(raw(queue.p, 14, word(second.p), 1))));
    Sleep(50);
    EXPECT(raw(second.p, 8) == 0);
    EXPECT(SUCCEEDED(as_hr(raw(fence.p, 10, 3))));
    ID3D12Fence* both[] = {fence.p, second.p};
    const UINT64 values[] = {3, 1};
    EXPECT(SUCCEEDED(as_hr(raw(device, 45, word(both), word(values), 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL,
                               word(done.h)))) &&
           done.set_within(5000));
    EXPECT(raw(second.p, 8) == 1);

    // Release the way such a client does: through slot 2, the queue and fences before the resources.
    EXPECT(raw(second.p, 2) == 0);
    second.p = nullptr;
    EXPECT(raw(fence.p, 2) == 0);
    fence.p = nullptr;
    EXPECT(raw(queue.p, 2) == 0);
    queue.p = nullptr;
    std::printf("%s: console-style client: Xbox heap/resource slots, copies through the queue proxy read back "
                "intact, residency with heap, queue and fence proxies\n", label);
    return 0;
}

// ---- 9. Coverage gate ----
//
// Every changed device slot, every queue slot and every fence slot of the tables must have been called in this
// process by the checks above (the differential cases run in child processes and do not count here), and the guard
// region must have caught the overruns check_guard provoked.
int check_coverage(const Backend& b) {
    const char* label = b.label;
    int missing = 0;
    const auto need = [&](const char* what, unsigned slot, std::uint64_t hits) {
        if (hits) return;
        std::fprintf(stderr, "%s: %s slot %u never called\n", label, what, slot);
        ++missing;
    };
    // Slots an earlier check skips on purpose on this backend (and says so) are not asked for: 75 without
    // ID3D12Device9, queue 8/9 without reserved resources, 31 without shared fences, 32 when they cannot be opened.
    const bool device9 = xs::device_proxy_slot_count(b.device) > 75;
    for (unsigned slot : {8u, 34u, 35u, 36u, 45u, 46u, 50u}) need("device", slot, xs::device_slot_hits(slot));
    if (device9) need("device", 75, xs::device_slot_hits(75));
    if (g_shared_created) need("device", 31, xs::device_slot_hits(31));
    if (g_shared_checked) need("device", 32, xs::device_slot_hits(32));
    for (unsigned slot = 0; slot < xs::queue_proxy_table_size(); ++slot) {
        if ((slot == 8 || slot == 9) && !g_reserved_checked) continue;
        need("queue", slot, xs::queue_slot_hits(slot));
    }
    for (unsigned slot = 0; slot < xs::fence_proxy_table_size(); ++slot) need("fence", slot, xs::fence_slot_hits(slot));
    EXPECT(missing == 0);
    EXPECT(xs::queue_table_overruns() >= 3 && xs::fence_table_overruns() >= 3);
    std::printf("%s: coverage: device slots 8/34/35/36/45/46/50%s%s%s, %s %u queue and all %u fence slots, both "
                "guard regions\n", label, device9 ? "/75" : "", g_shared_created ? "/31" : "",
                g_shared_checked ? "/32" : "", g_reserved_checked ? "all" : "all but 8/9 of the",
                xs::queue_proxy_table_size(), xs::fence_proxy_table_size());
    return 0;
}

// ---- 6. Eight threads at once ----

int check_threads(const Backend& b) {
    const char* label = b.label;
    ID3D12Device* device = b.device;
    const unsigned long device_refs = xs::device_proxy_ref_count(device);
    D3D12_COMMAND_QUEUE_DESC desc{};
    Ref<ID3D12CommandQueue> shared_queue;
    Ref<ID3D12Fence> shared_fence;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&shared_queue.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&shared_fence.p))));
    constexpr int kThreads = 8, kRounds = 300, kSignals = 20;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            const auto fail = [&](int line) {
                std::fprintf(stderr, "%s thread %d line %d\n", label, t, line);
                failures.fetch_add(1);
            };
            for (int i = 0; i < kRounds; ++i) {
                shared_queue->AddRef();
                ID3D12Fence* f = nullptr;
                if (FAILED(shared_fence->QueryInterface(IID_PPV_ARGS(&f))) || f != shared_fence.p) fail(__LINE__);
                if (f) f->Release();
                ID3D12Device* owner = nullptr;
                if (FAILED(shared_queue->GetDevice(IID_PPV_ARGS(&owner))) || owner != device) fail(__LINE__);
                if (owner) owner->Release();
                (void)shared_fence->GetCompletedValue();
                ID3D12Fence* mine = nullptr;  // device slot 36 from every thread
                if (FAILED(device->CreateFence(i, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mine))) ||
                    !xs::is_fence_proxy(mine) || mine->GetCompletedValue() != static_cast<UINT64>(i))
                    fail(__LINE__);
                if (mine) mine->Release();
                shared_queue->Release();
            }
            // GPU work from every thread: its own queue and fence, signalled and waited on.
            // The shared queue signals a fence of its own per thread: were it to signal `fence`, it could land after
            // the thread's queue had moved `fence` to v + 1 and set it back to v, which the PC runtime allows (the
            // Vulkan layer drops such a signal), and the wait for v + 1 would never end.
            ID3D12CommandQueue* queue = nullptr;
            ID3D12Fence* fence = nullptr;
            ID3D12Fence* mark = nullptr;
            HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!event || FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue))) ||
                FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) ||
                FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mark))))
                fail(__LINE__);
            for (UINT64 v = 1; event && queue && fence && mark && v <= kSignals; ++v) {
                if (FAILED(queue->Signal(fence, v)) || FAILED(fence->SetEventOnCompletion(v, event)) ||
                    WaitForSingleObject(event, 5000) != WAIT_OBJECT_0)
                    fail(__LINE__);
                if (FAILED(shared_queue->Signal(mark, v))) fail(__LINE__);  // one queue, many threads
            }
            // The shared queue's signals land in order: the last one means all of this thread's have.
            if (event && mark && (FAILED(mark->SetEventOnCompletion(kSignals, event)) ||
                                  WaitForSingleObject(event, 5000) != WAIT_OBJECT_0))
                fail(__LINE__);
            if (event) CloseHandle(event);
            if (mark) mark->Release();
            if (fence) fence->Release();
            if (queue) queue->Release();
        });
    }
    for (auto& thread : threads) thread.join();
    EXPECT(failures.load() == 0);
    // The shared queue's last Signals must also have landed.
    Ref<ID3D12Fence> tail;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&tail.p))));
    EXPECT(SUCCEEDED(shared_queue->Signal(tail.p, 1)));
    if (wait_for(label, tail.p, 1)) return 1;
    tail.reset();
    EXPECT(xs::command_object_proxy_ref_count(shared_queue.p) == 1);
    EXPECT(xs::command_object_proxy_ref_count(shared_fence.p) == 1);
    shared_queue.reset();
    shared_fence.reset();
    EXPECT(xs::device_proxy_ref_count(device) == device_refs);
    std::printf("%s: %d threads x %d rounds of AddRef/QueryInterface/GetDevice/CreateFence/Release, "
                "%d GPU signals each, consistent\n", label, kThreads, kRounds, kSignals);
    return 0;
}

// ---- 7. Repeated creation and release orders, without leaks ----

// One round on a fresh device proxy over `real`, which the caller keeps one reference to; the proxy takes another.
// `children_last` releases the device proxy before its queues and fences.
int one_round(const char* label, ID3D12Device* real, bool vulkan, bool children_last) {
    real->AddRef();
    ID3D12Device* device = nullptr;
    EXPECT(SUCCEEDED(xs::create_device_proxy(real, __uuidof(ID3D12Device), reinterpret_cast<void**>(&device))));
    if (vulkan) rt::d3d12::set_device_outer(real, device);
    D3D12_COMMAND_QUEUE_DESC desc{};
    ID3D12CommandQueue* queues[3] = {};
    ID3D12Fence* fences[3] = {};
    for (int i = 0; i < 3; ++i) {
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queues[i]))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fences[i]))));
        EXPECT(SUCCEEDED(queues[i]->Signal(fences[i], 1)));
    }
    for (int i = 0; i < 3; ++i)
        if (wait_for(label, fences[i], 1)) return 1;
    if (children_last) {
        // Release the title's device reference first: the children keep the proxy alive and still work.
        device->Release();
        EXPECT(SUCCEEDED(queues[0]->Signal(fences[1], 2)));
        if (wait_for(label, fences[1], 2)) return 1;
        ID3D12Device* owner = nullptr;
        EXPECT(SUCCEEDED(fences[2]->GetDevice(IID_PPV_ARGS(&owner))) && owner == device);
        owner->Release();
        for (int i = 2; i >= 0; --i) {
            fences[i]->Release();
            queues[i]->Release();
        }
    } else {
        for (int i = 0; i < 3; ++i) {
            queues[i]->Release();
            fences[i]->Release();
        }
        EXPECT(xs::device_proxy_ref_count(device) == 1);
        device->Release();
    }
    return 0;
}

int check_repeated(const Backend& b) {
    const char* label = b.label;
    constexpr int kRounds = 12;
    b.real->AddRef();
    const ULONG refs_before = b.real->Release();
    int failed = 0;
    for (int round = 0; round < kRounds && !failed; ++round)
        failed = one_round(label, b.real, b.vulkan, round % 2 == 1);
    // The layer's device names its outer device for the children it creates; give it back the backend's proxy, as
    // the round's proxy is gone.
    if (b.vulkan) rt::d3d12::set_device_outer(b.real, b.device);
    if (failed) return 1;
    // Every round's proxy, queue and fence is gone: the real device has the references it had before.
    b.real->AddRef();
    EXPECT(b.real->Release() == refs_before);
    std::printf("%s: %d rounds of device proxy, 3 queues and 3 fences in both release orders, no leak\n", label,
                kRounds);
    return 0;
}

int run_all(const Backend& b) {
    g_shared_checked = g_shared_created = g_reserved_checked = false;
    return check_slots(b) || check_guard(b) || check_differential(b) || check_identity(b) || check_shared(b) ||
           check_threads(b) || check_repeated(b) || check_xbox_client(b) || check_coverage(b);
}

// Host PC runtime on `adapter`. 77 when no device can be made on it.
int run_host(IDXGIAdapter1* adapter, const char* label, const char* kind) {
    ID3D12Device* real = nullptr;
    if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&real)))) {
        std::printf("%s: no D3D12 device, skipped\n", label);
        return 77;
    }
    real->AddRef();  // the backend's own reference; the proxy takes the other
    Ref<ID3D12Device> device;
    if (FAILED(xs::create_device_proxy(real, __uuidof(ID3D12Device), reinterpret_cast<void**>(&device.p)))) {
        real->Release();
        real->Release();
        std::fprintf(stderr, "%s: device proxy creation failed\n", label);
        return 1;
    }
    const int result = run_all(Backend{label, device.p, real, false, kind});
    device.reset();
    if (result) return 1;
    if (real->Release() != 0) {  // nothing the proxies made kept the PC device alive
        std::fprintf(stderr, "%s: the PC device outlives its proxies\n", label);
        return 1;
    }
    return 0;
}

}  // namespace

// A failure ends the process with a message on stderr, never in a dialog a test run would wait on: abort() (an
// exception that escapes, for one), the debug CRT's reports, and Windows' fault and critical-error boxes.
static void no_dialogs() {
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#ifdef _DEBUG  // the report functions exist only in the debug CRT (empty macros otherwise)
    for (const int type : {_CRT_WARN, _CRT_ERROR, _CRT_ASSERT}) {
        _CrtSetReportMode(type, _CRTDBG_MODE_FILE);
        _CrtSetReportFile(type, _CRTDBG_FILE_STDERR);
    }
#endif
    SetErrorMode(GetErrorMode() | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
}

int main(int argc, char** argv) {
    no_dialogs();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc == 5 && std::strcmp(argv[1], "--diff") == 0)
        return run_diff_child(argv[2], std::atoi(argv[3]), std::strcmp(argv[4], "proxy") == 0);
    if (argc == 3 && (std::strcmp(argv[1], "--child-flood") == 0 || std::strcmp(argv[1], "--child-hang") == 0 ||
                      std::strcmp(argv[1], "--child-crash") == 0))
        return run_runner_child(argv[1], std::strtoul(argv[2], nullptr, 10));
    // --trace-d3d12, as gta6rt takes it: every proxy call is recorded on the trace channel as well.
    if (argc == 2 && std::strcmp(argv[1], "--trace-d3d12") == 0) rt::d3d12_trace::set_enabled(true);
    else if (argc != 1) return 2;
    // The differential check depends on the child runner, which needs no device: checked first, on every machine.
    if (check_child_runner()) return 1;
    int ran = 0;

    // Host PC runtime: WARP, then the first hardware adapter. Under Wine the host d3d12.dll is Wine's own, not the
    // PC runtime the proxy is written against (as d3d12_copy_compare.h's host_is_wine), so only Vulkan runs there.
    const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
    Ref<IDXGIFactory4> factory;
    if (wine) std::printf("host D3D12: under Wine, skipped\n");
    if (!wine && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
        Ref<IDXGIAdapter1> warp;
        if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p)))) {
            const int r = run_host(warp.p, "host WARP", "warp");
            if (r == 1) return 1;
            ran += r == 0;
        }
        for (UINT i = 0;; ++i) {
            Ref<IDXGIAdapter1> adapter;
            if (factory->EnumAdapters1(i, &adapter.p) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
            char name[128] = {};
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
            char host_label[160];
            std::snprintf(host_label, sizeof(host_label), "host %s", name);
            const int r = run_host(adapter.p, host_label, "hw");
            if (r == 1) return 1;
            ran += r == 0;
            break;
        }
    }

    // The Vulkan-backed device, the way D3D12XboxCreateDevice hands it out with LEONIDA_D3D12=vulkan.
    vktest::Device d;
    if (d.create()) {
        const char* label = "Vulkan";
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        inner->AddRef();  // the backend's own reference; the proxy takes the other
        ID3D12Device* device = nullptr;
        EXPECT(SUCCEEDED(xs::create_device_proxy(inner, __uuidof(ID3D12Device), reinterpret_cast<void**>(&device))));
        rt::d3d12::set_device_outer(inner, device);
        const int result = run_all(Backend{label, device, inner, true, "vulkan"});
        device->Release();
        if (result) return 1;
        EXPECT(inner->Release() == 0);  // nothing the proxies made kept the layer's device alive
        ++ran;
        if (d.validation_errors()) {
            std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
            return 1;
        }
    } else {
        std::printf("Vulkan: no device, skipped\n");
    }

    if (!ran) {
        std::printf("SKIP queue_proxy_test: neither a host D3D12 device nor a Vulkan device\n");
        return 77;
    }
    std::printf("queue_proxy_test: %d backend(s) passed\n", ran);
    return 0;
}
