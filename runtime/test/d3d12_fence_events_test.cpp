// Fence events the host D3D12 runtime (RTX 5070 and WARP) sets or refuses, checked on the Vulkan-backed device and,
// when present, on the host devices too: releasing a fence completes the events still waiting on it (ALL counts it
// as one fence reached, ANY is satisfied), and SetEventOnMultipleFenceCompletion with no fences is E_INVALIDARG and
// sets nothing. No shaders or game data. LEONIDA_VK_VALIDATION=1 fails the test on any validation error.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", who, __LINE__, #condition); return 1; } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

struct Event {
    HANDLE h = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { CloseHandle(h); }
    bool set_within(DWORD ms) const { return WaitForSingleObject(h, ms) == WAIT_OBJECT_0; }
};

int check(ID3D12Device* device, const char* who) {
    Ref<ID3D12Device1> device1;
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device1.p))));
    constexpr auto ALL = D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, ANY = D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY;
    {  // A fence released before its value is reached sets the event.
        ID3D12Fence* fence = nullptr;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
        Event event;
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(5, event.h)));
        fence->Release();
        EXPECT(event.set_within(2000));
    }
    for (const auto flags : {ALL, ANY}) {  // Released with a multi-fence wait pending: one fence counted as reached.
        ID3D12Fence* released = nullptr;
        Ref<ID3D12Fence> other;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&released))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&other.p))));
        ID3D12Fence* fences[2] = {released, other.p};
        const UINT64 values[2] = {5, 1};
        Event event;
        EXPECT(SUCCEEDED(device1->SetEventOnMultipleFenceCompletion(fences, values, 2, flags, event.h)));
        released->Release();
        if (flags == ANY) {
            EXPECT(event.set_within(2000));
        } else {
            EXPECT(!event.set_within(100));  // still waiting for the other fence
            EXPECT(SUCCEEDED(other->Signal(1)));
            EXPECT(event.set_within(2000));
        }
    }
    {  // No fences: refused, with or without an event, and the event is left alone.
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        ID3D12Fence* fences[1] = {fence.p};
        const UINT64 values[1] = {1};
        Event event;
        for (const auto flags : {ALL, ANY}) {
            EXPECT(device1->SetEventOnMultipleFenceCompletion(fences, values, 0, flags, event.h) == E_INVALIDARG);
            EXPECT(device1->SetEventOnMultipleFenceCompletion(nullptr, nullptr, 0, flags, event.h) == E_INVALIDARG);
            EXPECT(device1->SetEventOnMultipleFenceCompletion(fences, values, 0, flags, nullptr) == E_INVALIDARG);
        }
        EXPECT(!event.set_within(0));
    }
    std::printf("%s: fence events match the host runtime\n", who);
    return 0;
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    int failures = 0;
    {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        failures += check(ours.p, "vulkan");
        Ref<IDXGIFactory4> factory;
        if (!hostwine::host_is_wine() && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            for (const bool warp : {false, true}) {  // the reference the expectations were measured on
                Ref<IDXGIAdapter> adapter;
                Ref<ID3D12Device> host;
                std::string adapter_name;
                if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
                if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
                failures += check(host.p, ("host " + adapter_name).c_str());
            }
        }
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}
