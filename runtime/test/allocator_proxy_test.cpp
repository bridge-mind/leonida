// Xbox-ABI command allocator proxies (d3d12_xs_device.h, device slot 9) behind the device proxy, on the Vulkan-backed
// device and on the host PC runtime (WARP, and the first hardware adapter when there is one):
//   - Vulkan device: slot 9 wraps the allocator; the public chain (9 slots) is forwarded with the real allocator's
//     answers, every slot past it up to the end of the table is refused (E_NOTIMPL), and a read past the table faults
//     in the guard region and is counted;
//   - QueryInterface identity, GetDevice -> the device proxy, references and release orders;
//   - a command list created on the proxy and Reset on it (the Vulkan layer resolves the proxy), recording a buffer
//     copy that runs through the queue and fence proxies and is read back, twice, with the allocator Reset between;
//     and D3D12's allocator rules through the proxy (Reset while a list records: E_FAIL; a second list on a busy
//     allocator: E_INVALIDARG);
//   - MakeResident/Evict with an allocator proxy give the real allocator's HRESULT;
//   - host runtime: slot 9 hands out the PC allocator unwrapped (its lists could not resolve a proxy), and the same
//     copy works.
// LEONIDA_VK_VALIDATION=1 runs the Vulkan device under VK_LAYER_KHRONOS_validation with synchronization validation;
// any validation error fails the test. Exits 77 when neither a host D3D12 device nor a Vulkan device can be made.
// No shaders, game data or Xbox DLLs.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"

#include <d3d12.h>
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", label, __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace xs = rt::d3d12_xs;

template <class T>
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

struct Backend {
    const char* label;
    ID3D12Device* device;  // the device proxy
    ID3D12Device* real;    // the device behind it (PC runtime or Vulkan layer)
    bool vulkan;
};

using SlotFn = std::uint64_t(WINAPI*)(void*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                      std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                      std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);

// A call through slot `slot` with all words zero, the way a title calls a console method the PC interface lacks.
HRESULT call_slot(void* object, unsigned slot) {
    void** vtable = *reinterpret_cast<void***>(object);
    return static_cast<HRESULT>(static_cast<std::uint32_t>(
        reinterpret_cast<SlotFn>(vtable[slot])(object, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
}

ULONG_PTR g_fault_address = 0;
// The code is taken in the filter, from the exception record: GetExceptionCode() in the __except body returns 0
// under Wine 9.0 (clang-cl reads it from what the unwind hands the handler, which that Wine does not supply).
DWORD g_fault_code = 0;

int record_fault(EXCEPTION_POINTERS* info) {
    g_fault_code = info->ExceptionRecord->ExceptionCode;
    g_fault_address = info->ExceptionRecord->NumberParameters >= 2 ? info->ExceptionRecord->ExceptionInformation[1] : 0;
    return EXCEPTION_EXECUTE_HANDLER;
}

// The read in a function of its own: clang-cl's __try only catches a fault raised inside a call.
__declspec(noinline) void* read_entry(void* const* vtable, unsigned slot) {
    return *static_cast<void* const volatile*>(&vtable[slot]);
}

// Reads vtable[slot] the way a call through it does. The exception code, or 0 when the read worked.
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

int wait_for(const char* label, ID3D12Fence* fence, UINT64 value) {
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(event != nullptr);
    const bool set = SUCCEEDED(fence->SetEventOnCompletion(value, event)) && WaitForSingleObject(event, 5000) == 0;
    CloseHandle(event);
    EXPECT(set && fence->GetCompletedValue() >= value);
    return 0;
}

int make_buffer(const char* label, ID3D12Device* device, D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_STATES state,
                Ref<ID3D12Resource>& out) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = heap_type;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 256;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                     IID_PPV_ARGS(&out.p))));
    return 0;
}

// One recording on `list` (open, recording into the allocator): 256 bytes of `byte` copied from an upload buffer to a
// readback buffer, executed through the queue proxy, finished through the fence proxy at `value`, then compared.
int copy_round(const char* label, ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Fence* fence,
               ID3D12GraphicsCommandList* list, std::uint8_t byte, UINT64 value) {
    Ref<ID3D12Resource> upload, readback;
    if (make_buffer(label, device, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, upload)) return 1;
    if (make_buffer(label, device, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, readback)) return 1;
    void* mapped = nullptr;
    EXPECT(SUCCEEDED(upload->Map(0, nullptr, &mapped)) && mapped);
    std::memset(mapped, byte, 256);
    upload->Unmap(0, nullptr);
    list->CopyBufferRegion(readback.p, 0, upload.p, 0, 256);
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(SUCCEEDED(queue->Signal(fence, value)));
    if (wait_for(label, fence, value)) return 1;
    const D3D12_RANGE read{0, 256};
    EXPECT(SUCCEEDED(readback->Map(0, &read, &mapped)) && mapped);
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    int wrong = 0;
    for (int i = 0; i < 256; ++i) wrong += bytes[i] != byte;
    const D3D12_RANGE none{0, 0};
    readback->Unmap(0, &none);
    EXPECT(wrong == 0);
    return 0;
}

// ---- 1. The proxy itself: slots, identity, guard (Vulkan device) ----

int check_proxy(const Backend& b, ID3D12CommandAllocator* allocator) {
    const char* label = b.label;
    ID3D12CommandAllocator* real = xs::allocator_proxy_real_allocator(allocator);
    EXPECT(real && static_cast<void*>(real) != static_cast<void*>(allocator));
    EXPECT(xs::allocator_proxy_slot_count(allocator) == 9);
    EXPECT(xs::command_object_proxy_ref_count(allocator) == 1);

    // QueryInterface: the base chain and the allocator itself answer with the proxy; anything else is refused.
    const IID modelled[] = {__uuidof(IUnknown), __uuidof(ID3D12Object), __uuidof(ID3D12DeviceChild),
                            __uuidof(ID3D12Pageable), __uuidof(ID3D12CommandAllocator)};
    for (const IID& iid : modelled) {
        void* same = nullptr;
        EXPECT(SUCCEEDED(allocator->QueryInterface(iid, &same)) && same == allocator);
        static_cast<IUnknown*>(same)->Release();
    }
    void* other = reinterpret_cast<void*>(1);
    EXPECT(allocator->QueryInterface(__uuidof(ID3D12Fence), &other) == E_NOINTERFACE && !other);
    other = reinterpret_cast<void*>(1);  // the Vulkan layer's private tag: not modelled, the real object stays hidden
    EXPECT(allocator->QueryInterface(xs::kLeonidaCommandAllocatorTag, &other) == E_NOINTERFACE && !other);
    EXPECT(xs::command_object_proxy_ref_count(allocator) == 1);

    ID3D12Device* owner = nullptr;
    EXPECT(SUCCEEDED(allocator->GetDevice(IID_PPV_ARGS(&owner))) && owner == b.device);
    owner->Release();
    // Private data and SetName go to the real allocator; the proxy pointer round-trips as an interface.
    EXPECT(SUCCEEDED(allocator->SetName(L"leonida allocator proxy")));
    constexpr GUID key{0x2a6b8c41, 0x77d2, 0x4e19, {0x9c, 0x3a, 0x51, 0xe0, 0x8f, 0x24, 0x6d, 0x0b}};
    EXPECT(SUCCEEDED(allocator->SetPrivateDataInterface(key, allocator)));
    IUnknown* back = nullptr;
    UINT size = sizeof(back);
    EXPECT(SUCCEEDED(allocator->GetPrivateData(key, &size, &back)) && back == allocator);
    back->Release();
    EXPECT(SUCCEEDED(allocator->SetPrivateDataInterface(key, nullptr)));

    // The refused tail: every slot past the public chain, up to the table end, is E_NOTIMPL and counted.
    const unsigned table = xs::allocator_proxy_table_size();
    EXPECT(table == 64);
    for (unsigned slot = 9; slot < table; ++slot) {
        const std::uint64_t before = xs::allocator_slot_hits(slot);
        EXPECT(call_slot(allocator, slot) == E_NOTIMPL);
        EXPECT(xs::allocator_slot_hits(slot) == before + 1 && xs::allocator_slot_first_call_seen(slot));
    }

    // Past the table: the vtable read itself faults in the guard region, at the slot's address, and is counted.
    EXPECT(xs::command_object_tables_guarded());
    void* const* vtable = *reinterpret_cast<void* const* const*>(allocator);
    EXPECT(read_slot_guarded(vtable, table - 1) == 0);
    for (unsigned slot : {table, table + 512, table + 7000}) {
        const std::uint64_t before = xs::allocator_table_overruns();
        EXPECT(read_slot_guarded(vtable, slot) == EXCEPTION_ACCESS_VIOLATION);
        EXPECT(g_fault_address == reinterpret_cast<ULONG_PTR>(&vtable[slot]));
        EXPECT(xs::allocator_table_overruns() == before + 1);
    }
    std::printf("%s: allocator proxy: 9 public slots forwarded, slots 9-%u refused, the guard region names a read "
                "past the table; QueryInterface/GetDevice/private data keep the proxy identity\n", label, table - 1);
    return 0;
}

// ---- 2. Lists on the allocator, GPU work, D3D12's allocator rules, residency ----

int check_backend(const Backend& b) {
    const char* label = b.label;
    const unsigned long device_refs = xs::device_proxy_ref_count(b.device);
    {
        Ref<ID3D12CommandAllocator> allocator;
        EXPECT(SUCCEEDED(b.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                          IID_PPV_ARGS(&allocator.p))));
        // Vulkan layer: wrapped. Host runtime: its lists could not resolve a proxy, so the PC allocator comes back.
        EXPECT(xs::is_allocator_proxy(allocator.p) == b.vulkan);
        if (b.vulkan && check_proxy(b, allocator.p)) return 1;
        // As IUnknown, and without an output (the device's own answer), as slot 8 does for queues.
        Ref<IUnknown> as_unknown;
        EXPECT(SUCCEEDED(b.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(IUnknown),
                                                          reinterpret_cast<void**>(&as_unknown.p))));
        EXPECT(xs::is_allocator_proxy(as_unknown.p) == b.vulkan);
#pragma warning(suppress : 6387)  // the null output on purpose
        EXPECT(b.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator),
                                                nullptr) ==
               b.real->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator),
                                              nullptr));
        Ref<ID3D12Fence> not_an_allocator;  // asked for an interface the allocator does not have
        not_an_allocator.p = reinterpret_cast<ID3D12Fence*>(1);
        const HRESULT wrong_iid = b.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                                   IID_PPV_ARGS(&not_an_allocator.p));
        const bool cleared = !not_an_allocator.p;
        not_an_allocator.p = nullptr;
        EXPECT(FAILED(wrong_iid) && cleared);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(b.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(b.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        EXPECT(xs::is_queue_proxy(queue.p) && xs::is_fence_proxy(fence.p));

        // A list on the allocator: the Vulkan layer resolves the proxy in CreateCommandList.
        Ref<ID3D12GraphicsCommandList> list;
        EXPECT(SUCCEEDED(b.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                     IID_PPV_ARGS(&list.p))));
        // D3D12's rules, through the proxy: a recording list holds the allocator.
        const std::uint64_t resets_before = xs::allocator_slot_hits(8);
        EXPECT(allocator->Reset() == E_FAIL);
        Ref<ID3D12GraphicsCommandList> second;
        EXPECT(b.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                           IID_PPV_ARGS(&second.p)) == E_INVALIDARG && !second.p);
        if (copy_round(label, b.device, queue.p, fence.p, list.p, 0x5a, 1)) return 1;

        // Done on the GPU: the allocator resets through the proxy, and the list resets onto it (Reset resolves it).
        EXPECT(SUCCEEDED(allocator->Reset()));
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        if (copy_round(label, b.device, queue.p, fence.p, list.p, 0xa5, 2)) return 1;
        // The two allocator Reset calls above went through slot 8 (list->Reset is the list's method, not this one).
        if (b.vulkan) EXPECT(xs::allocator_slot_hits(8) == resets_before + 2);

        // Residency: the real runtime gets the real allocator (an allocator is a pageable).
        ID3D12Pageable* pageable = allocator.p;
        ID3D12CommandAllocator* real_allocator =
            b.vulkan ? xs::allocator_proxy_real_allocator(allocator.p) : allocator.p;
        ID3D12Pageable* real_pageable = real_allocator;
        EXPECT(b.device->MakeResident(1, &pageable) == b.real->MakeResident(1, &real_pageable));
        EXPECT(b.device->Evict(1, &pageable) == b.real->Evict(1, &real_pageable));

        // Release orders: the allocator first while its (closed) list still exists, then the list.
        if (b.vulkan) EXPECT(xs::command_object_proxy_ref_count(allocator.p) == 1);
        allocator.reset();
        list.reset();
    }
    EXPECT(xs::device_proxy_ref_count(b.device) == device_refs);  // nothing the proxies made kept the device
    std::printf("%s: CreateCommandAllocator %s; a list created and Reset on it, two GPU copies through the queue and "
                "fence proxies read back exactly; allocator rules and residency as on the real allocator\n", label,
                b.vulkan ? "returns an allocator proxy" : "returns the PC allocator unwrapped");
    return 0;
}

int run_host(IDXGIAdapter1* adapter, const char* label) {
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
    const int result = check_backend(Backend{label, device.p, real, false});
    device.reset();
    if (result) return 1;
    if (real->Release() != 0) {
        std::fprintf(stderr, "%s: the PC device outlives its proxies\n", label);
        return 1;
    }
    return 0;
}

}  // namespace

// A failure ends the process with a message on stderr, never in a dialog a test run would wait on.
static void no_dialogs() {
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    SetErrorMode(GetErrorMode() | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
}

int main() {
    no_dialogs();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* label = "allocator_proxy_test";
    int ran = 0;

    // Host PC runtime: WARP, then the first hardware adapter. Under Wine the host d3d12.dll is Wine's own, not the
    // PC runtime the proxy is written against, so only Vulkan runs there.
    const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
    Ref<IDXGIFactory4> factory;
    if (wine) std::printf("host D3D12: under Wine, skipped\n");
    if (!wine && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
        Ref<IDXGIAdapter1> warp;
        if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p)))) {
            const int r = run_host(warp.p, "host WARP");
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
            const int r = run_host(adapter.p, host_label);
            if (r == 1) return 1;
            ran += r == 0;
            break;
        }
    }

    // The Vulkan-backed device, the way D3D12XboxCreateDevice hands it out with LEONIDA_D3D12=vulkan.
    vktest::Device d;
    if (d.create()) {
        label = "Vulkan";
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
        const int result = check_backend(Backend{label, device, inner, true});
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
        std::printf("SKIP allocator_proxy_test: neither a host D3D12 device nor a Vulkan device\n");
        return 77;
    }
    std::printf("allocator_proxy_test: %d backend(s) passed\n", ran);
    return 0;
}
