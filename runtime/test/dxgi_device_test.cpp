// IDXGIDevice1 on the Xbox-ABI device proxy (d3d12_xs_device.h): the step a GDKX title takes to find its output
// (docs/d3d12_vulkan.md section 4.5: QueryInterface the device for IDXGIDevice1, GetAdapter, EnumOutputs(0)). Walked
// on the host PC runtime (WARP, and the first hardware adapter when there is one) and on the Vulkan-backed device:
//   - the PC device behind the proxy has no IDXGIDevice of its own (host runtime), so the proxy's answer is its own;
//   - QueryInterface for IDXGIDevice, IDXGIDevice1 and IDXGIObject gives one pointer, which is not the device's but
//     is the same COM object: one IUnknown, QueryInterface back to ID3D12Device gives the proxy, one reference count;
//   - IDXGIDevice2 and later are refused;
//   - GetAdapter and GetParent(IDXGIAdapter) give the adapter of the device behind the proxy: its LUID is the
//     device's GetAdapterLuid. On the host runtime it is the host's adapter; on the Vulkan device it is the layer's
//     own (rt::dxgi, which stops at IDXGIAdapter3 and whose parent is the layer's factory);
//   - EnumOutputs(0) on that adapter answers S_OK with an output or DXGI_ERROR_NOT_FOUND, never anything else;
//   - null arguments, the frame latency and the GPU thread priority, and private data shared with the device;
//   - a device created as IDXGIDevice1 directly (a title may pass that IID to D3D12XboxCreateDevice): the pointer
//     that comes back is the DXGI one, the proxy behind it is the device's identity (a fence's GetDevice gives it),
//     and the adapter provider still reaches it, so the Vulkan device hands out the layer's adapter there too.
// Exits 77 when neither a host D3D12 device nor a Vulkan device can be made. No shaders, game data or Xbox DLLs.
#include "vk_test_device.h"
#include "window_station.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"

#include <d3d12.h>
#include <dxgi1_6.h>

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

// {8f3a1c52-6e0b-4d97-b1c4-2a7e9d5f0c36}: a private-data key of this test.
constexpr GUID kTestData{0x8f3a1c52, 0x6e0b, 0x4d97, {0xb1, 0xc4, 0x2a, 0x7e, 0x9d, 0x5f, 0x0c, 0x36}};

bool same_luid(const LUID& a, const LUID& b) { return a.LowPart == b.LowPart && a.HighPart == b.HighPart; }

// `device` is the device proxy, `real` the device behind it.
int check_backend(const char* label, ID3D12Device* device, ID3D12Device* real, bool vulkan) {
    const unsigned long device_refs = xs::device_proxy_ref_count(device);

    // The device behind the proxy: the PC runtime's has no IDXGIDevice, which is why the proxy answers it itself.
    if (!vulkan) {
        void* none = reinterpret_cast<void*>(1);
        EXPECT(real->QueryInterface(__uuidof(IDXGIDevice1), &none) == E_NOINTERFACE && !none);
    }

    Ref<IDXGIDevice1> dxgi;
    EXPECT(device->QueryInterface(__uuidof(IDXGIDevice1), reinterpret_cast<void**>(&dxgi.p)) == S_OK && dxgi.p);
    EXPECT(static_cast<void*>(dxgi.p) != static_cast<void*>(device));
    EXPECT(xs::device_proxy_from_dxgi_device(dxgi.p) == device && !xs::device_proxy_from_dxgi_device(device));
    EXPECT(xs::device_proxy_ref_count(device) == device_refs + 1);
    {
        // One COM object: every DXGI interface is the same pointer, IUnknown is the device's, and QueryInterface
        // leads back to the device proxy from each of them.
        Ref<IDXGIDevice> dxgi0;
        Ref<IDXGIObject> object;
        Ref<IUnknown> unknown_a, unknown_b;
        Ref<ID3D12Device> back;
        EXPECT(dxgi->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgi0.p)) == S_OK);
        EXPECT(device->QueryInterface(__uuidof(IDXGIObject), reinterpret_cast<void**>(&object.p)) == S_OK);
        EXPECT(static_cast<void*>(dxgi0.p) == static_cast<void*>(dxgi.p) &&
               static_cast<void*>(object.p) == static_cast<void*>(dxgi.p));
        EXPECT(dxgi->QueryInterface(__uuidof(IUnknown), reinterpret_cast<void**>(&unknown_a.p)) == S_OK);
        EXPECT(device->QueryInterface(__uuidof(IUnknown), reinterpret_cast<void**>(&unknown_b.p)) == S_OK);
        EXPECT(unknown_a.p == unknown_b.p && static_cast<void*>(unknown_a.p) == static_cast<void*>(device));
        EXPECT(dxgi->QueryInterface(__uuidof(ID3D12Device), reinterpret_cast<void**>(&back.p)) == S_OK);
        EXPECT(back.p == device);
        EXPECT(xs::device_proxy_ref_count(device) == device_refs + 6);
        // Later DXGI device interfaces, and a null output or IID.
        void* none = reinterpret_cast<void*>(1);
        EXPECT(device->QueryInterface(__uuidof(IDXGIDevice2), &none) == E_NOINTERFACE && !none);
        none = reinterpret_cast<void*>(1);
        EXPECT(dxgi->QueryInterface(__uuidof(IDXGIDevice2), &none) == E_NOINTERFACE && !none);
        EXPECT(dxgi->QueryInterface(__uuidof(IDXGIDevice1), nullptr) == E_POINTER);
    }
    EXPECT(xs::device_proxy_ref_count(device) == device_refs + 1);
    // AddRef and Release on the DXGI pointer count on the device.
    EXPECT(dxgi->AddRef() == device_refs + 2 && dxgi->Release() == device_refs + 1);

    // GetAdapter: the adapter of the device behind the proxy.
    const LUID device_luid = device->GetAdapterLuid();
    EXPECT(same_luid(device_luid, real->GetAdapterLuid()));
    Ref<IDXGIAdapter> adapter;
    EXPECT(dxgi->GetAdapter(&adapter.p) == S_OK && adapter.p);
    DXGI_ADAPTER_DESC desc{};
    EXPECT(adapter->GetDesc(&desc) == S_OK && same_luid(desc.AdapterLuid, device_luid));
    char name[128] = {};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
    EXPECT(dxgi->GetAdapter(nullptr) == E_POINTER);
    {
        // GetParent is the same adapter by another road, as any interface it has; the device is not its own parent.
        Ref<IDXGIAdapter> parent;
        EXPECT(dxgi->GetParent(__uuidof(IDXGIAdapter), reinterpret_cast<void**>(&parent.p)) == S_OK && parent.p);
        DXGI_ADAPTER_DESC parent_desc{};
        EXPECT(parent->GetDesc(&parent_desc) == S_OK && same_luid(parent_desc.AdapterLuid, device_luid));
        void* none = reinterpret_cast<void*>(1);
        EXPECT(FAILED(dxgi->GetParent(__uuidof(ID3D12Device), &none)) && !none);
        EXPECT(dxgi->GetParent(__uuidof(IDXGIAdapter), nullptr) == E_POINTER);
    }
    {
        // Whose adapter it is. The layer's own stops at IDXGIAdapter3 and its parent is the layer's factory, which
        // enumerates exactly that one adapter; the host's is the host DXGI's (IDXGIAdapter4 since Windows 10 1803).
        Ref<IDXGIAdapter4> adapter4;
        const HRESULT hr4 = adapter->QueryInterface(__uuidof(IDXGIAdapter4), reinterpret_cast<void**>(&adapter4.p));
        Ref<IDXGIFactory1> factory;
        EXPECT(adapter->GetParent(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory.p)) == S_OK);
        Ref<IDXGIAdapter> second;
        const HRESULT second_hr = factory->EnumAdapters(1, &second.p);
        if (vulkan) {
            EXPECT(hr4 == E_NOINTERFACE && !adapter4.p);
            EXPECT(second_hr == DXGI_ERROR_NOT_FOUND && !second.p);
        }
    }
    // The next step of the sequence. Which answer comes back is the adapter's business (a software adapter has no
    // output; the layer's adapter has none before D3D-313): it is one of the two, and an output only with S_OK.
    // Without an interactive window station (SSH, a service) the host has no outputs to give and answers
    // DXGI_ERROR_NOT_CURRENTLY_AVAILABLE.
    Ref<IDXGIOutput> output;
    const HRESULT output_hr = adapter->EnumOutputs(0, &output.p);
    EXPECT((output_hr == S_OK && output.p) || (output_hr == DXGI_ERROR_NOT_FOUND && !output.p) ||
           (output_hr == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE && !output.p && !has_interactive_window_station()));

    // IDXGIDevice1's own state: kept and read back.
    UINT latency = 0;
    EXPECT(dxgi->GetMaximumFrameLatency(&latency) == S_OK && latency == 3);
    EXPECT(dxgi->SetMaximumFrameLatency(1) == S_OK && dxgi->GetMaximumFrameLatency(&latency) == S_OK && latency == 1);
    EXPECT(dxgi->SetMaximumFrameLatency(17) == DXGI_ERROR_INVALID_CALL);
    EXPECT(dxgi->GetMaximumFrameLatency(&latency) == S_OK && latency == 1);
    EXPECT(dxgi->SetMaximumFrameLatency(0) == S_OK && dxgi->GetMaximumFrameLatency(&latency) == S_OK && latency == 3);
    EXPECT(dxgi->GetMaximumFrameLatency(nullptr) == E_POINTER);
    INT priority = 99;
    EXPECT(dxgi->GetGPUThreadPriority(&priority) == S_OK && priority == 0);
    EXPECT(dxgi->SetGPUThreadPriority(7) == S_OK && dxgi->GetGPUThreadPriority(&priority) == S_OK && priority == 7);
    EXPECT(dxgi->SetGPUThreadPriority(8) == E_INVALIDARG && dxgi->SetGPUThreadPriority(-8) == E_INVALIDARG);
    EXPECT(dxgi->GetGPUThreadPriority(&priority) == S_OK && priority == 7);
    EXPECT(dxgi->GetGPUThreadPriority(nullptr) == E_POINTER);
    IDXGISurface* surface = reinterpret_cast<IDXGISurface*>(std::uintptr_t{1});
    EXPECT(dxgi->CreateSurface(nullptr, 0, 0, nullptr, &surface) == E_NOTIMPL && !surface);
    EXPECT(dxgi->QueryResourceResidency(nullptr, nullptr, 0) == E_NOTIMPL);

    // Private data set through the DXGI interface is the device's, and the other way round.
    const std::uint32_t written = 0x12345678;
    std::uint32_t read = 0;
    UINT size = sizeof(read);
    EXPECT(dxgi->SetPrivateData(kTestData, sizeof(written), &written) == S_OK);
    EXPECT(device->GetPrivateData(kTestData, &size, &read) == S_OK && size == sizeof(read) && read == written);
    EXPECT(device->SetPrivateData(kTestData, 0, nullptr) == S_OK);
    size = sizeof(read);
    EXPECT(FAILED(dxgi->GetPrivateData(kTestData, &size, &read)));

    // The adapter does not keep the device alive, and the device outlives its DXGI pointer's last use.
    adapter.reset();
    output.reset();
    dxgi.reset();
    EXPECT(xs::device_proxy_ref_count(device) == device_refs);
    std::printf("%s: IDXGIDevice1 on the device proxy is one COM object with it; GetAdapter and GetParent give \"%s\" "
                "(%s adapter, the device's LUID); EnumOutputs(0) %s\n",
                label, name, vulkan ? "the layer's own" : "the host's",
                output_hr == S_OK ? "gives an output" : "is DXGI_ERROR_NOT_FOUND");
    return 0;
}

// The device created as IDXGIDevice1, the way D3D12XboxCreateDevice creates it when a title passes that IID: the
// proxy over `real` (which takes one reference to it), then set_device_outer with what came back for the Vulkan
// device. `real` is released by the proxy's last reference.
int check_direct(const char* label, ID3D12Device* real, bool vulkan) {
    IDXGIDevice1* dxgi = nullptr;
    EXPECT(xs::create_device_proxy(real, __uuidof(IDXGIDevice1), reinterpret_cast<void**>(&dxgi)) == S_OK && dxgi);
    EXPECT(!xs::is_device_proxy(dxgi));  // the DXGI pointer, not the proxy
    void* const proxy = xs::device_proxy_from_dxgi_device(dxgi);
    EXPECT(proxy && xs::is_device_proxy(proxy) && xs::device_proxy_primary(dxgi) == proxy &&
           xs::device_proxy_primary(proxy) == proxy);
    EXPECT(xs::device_proxy_ref_count(proxy) == 1);  // the caller's one reference, held through the DXGI pointer
    if (vulkan) rt::d3d12::set_device_outer(real, dxgi);
    {
        // QueryInterface back to the device gives the proxy, and the device's children name the proxy as their
        // device: the outer identity is the proxy, not the DXGI pointer that creation returned.
        Ref<ID3D12Device> device;
        EXPECT(dxgi->QueryInterface(__uuidof(ID3D12Device), reinterpret_cast<void**>(&device.p)) == S_OK);
        EXPECT(static_cast<void*>(device.p) == proxy && xs::device_proxy_ref_count(proxy) == 2);
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        Ref<ID3D12Device> owner;
        EXPECT(SUCCEEDED(fence->GetDevice(IID_PPV_ARGS(&owner.p))) && static_cast<void*>(owner.p) == proxy);
        // The adapter: the device's LUID, and on the Vulkan device the layer's own, as on the query path.
        Ref<IDXGIAdapter> adapter;
        EXPECT(dxgi->GetAdapter(&adapter.p) == S_OK && adapter.p);
        DXGI_ADAPTER_DESC desc{};
        EXPECT(adapter->GetDesc(&desc) == S_OK && same_luid(desc.AdapterLuid, real->GetAdapterLuid()));
        Ref<IDXGIAdapter4> adapter4;
        const HRESULT hr4 = adapter->QueryInterface(__uuidof(IDXGIAdapter4), reinterpret_cast<void**>(&adapter4.p));
        Ref<IDXGIFactory1> factory;
        EXPECT(adapter->GetParent(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory.p)) == S_OK);
        Ref<IDXGIAdapter> second;
        const HRESULT second_hr = factory->EnumAdapters(1, &second.p);
        if (vulkan) {
            EXPECT(hr4 == E_NOINTERFACE && !adapter4.p);
            EXPECT(second_hr == DXGI_ERROR_NOT_FOUND && !second.p);
        }
    }
    EXPECT(xs::device_proxy_ref_count(proxy) == 1);
    {
        // A provider set through the DXGI pointer is the one GetAdapter asks, and one handed something that is not
        // a proxy is refused (it says so on stderr) instead of leaving the default in place unnoticed.
        const xs::DxgiAdapterProvider refuse = [](ID3D12Device*, REFIID, void** out) {
            *out = reinterpret_cast<void*>(std::uintptr_t{1});  // a failing provider's output is not handed on
            return E_ABORT;
        };
        EXPECT(xs::set_device_adapter_provider(dxgi, refuse));
        IDXGIAdapter* none = reinterpret_cast<IDXGIAdapter*>(std::uintptr_t{1});
        EXPECT(dxgi->GetAdapter(&none) == E_ABORT && !none);
        EXPECT(xs::set_device_adapter_provider(proxy, nullptr));  // back to the default (the host's adapter)
        Ref<IDXGIAdapter> host;
        const HRESULT default_hr = dxgi->GetAdapter(&host.p);
        // The default asks the host's DXGI for the adapter with the device's LUID. Under Wine 9 the Vulkan device's
        // LUID is not one Wine's DXGI enumerates, so the host answers DXGI_ERROR_NOT_FOUND: a host difference (the
        // Wine 11.22 run in docs/testing.md doesn't list this test as failing), checked as a failure that hands nothing
        // out.
        static const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
        if (vulkan && wine && default_hr == DXGI_ERROR_NOT_FOUND) {
            std::printf("%s: the default adapter provider: Wine's DXGI has no adapter with the Vulkan device's LUID "
                        "(DXGI_ERROR_NOT_FOUND), not compared\n", label);
            EXPECT(!host.p);
        } else {
            EXPECT(default_hr == S_OK && host.p);
        }
        void* not_a_proxy = nullptr;  // pointer-sized: the helpers read a vtable pointer from what they are given
        EXPECT(!xs::set_device_adapter_provider(&not_a_proxy, refuse) && !xs::set_device_adapter_provider(real, refuse));
        EXPECT(!xs::device_proxy_primary(&not_a_proxy) && !xs::device_proxy_primary(nullptr));
    }
    EXPECT(dxgi->Release() == 0);  // the last reference: the proxy goes, and with it its reference to `real`
    std::printf("%s: created as IDXGIDevice1: the proxy is the device's identity and the adapter is %s\n", label,
                vulkan ? "the layer's own" : "the host's");
    return 0;
}

int run_host(IDXGIAdapter1* adapter, const char* label) {
    ID3D12Device* real = nullptr;
    if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&real)))) {
        std::printf("%s: no D3D12 device, skipped\n", label);
        return 77;
    }
    real->AddRef();  // for the second proxy below: each proxy takes one reference
    if (check_direct(label, real, false)) {
        real->Release();
        return 1;
    }
    real->AddRef();  // the backend's own reference; the proxy takes the other
    Ref<ID3D12Device> device;
    if (FAILED(xs::create_device_proxy(real, __uuidof(ID3D12Device), reinterpret_cast<void**>(&device.p)))) {
        real->Release();
        real->Release();
        std::fprintf(stderr, "%s: device proxy creation failed\n", label);
        return 1;
    }
    const int result = check_backend(label, device.p, real, false);
    device.reset();
    if (result) return 1;
    if (real->Release() != 0) {
        std::fprintf(stderr, "%s: the PC device outlives its proxy\n", label);
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
    const char* label = "dxgi_device_test";
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
            const int r = run_host(adapter.p, "host hardware adapter");
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
        const int result = check_backend(label, device, inner, true);
        device->Release();
        if (result) return 1;
        EXPECT(inner->Release() == 0);  // nothing the adapter or the factory made kept the layer's device alive
        // A second device on the same Vulkan device, created as IDXGIDevice1.
        ID3D12Device* direct = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&direct))));
        direct->AddRef();  // the backend's own reference; the proxy takes the other
        if (check_direct(label, direct, true)) return 1;
        EXPECT(direct->Release() == 0);
        ++ran;
        if (d.validation_errors()) {
            std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
            return 1;
        }
    } else {
        std::printf("Vulkan: no device, skipped\n");
    }

    if (!ran) {
        std::printf("SKIP dxgi_device_test: neither a host D3D12 device nor a Vulkan device\n");
        return 77;
    }
    std::printf("dxgi_device_test: %d backend(s) passed\n", ran);
    return 0;
}
