// D3D12 fence device-loss reporting when vkGetSemaphoreCounterValue first discovers a lost device.
// The injected VkResult is synthetic and never forwarded to the physical Vulkan driver. No game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
    bool ready(DWORD ms) const { return handle && WaitForSingleObject(handle, ms) == WAIT_OBJECT_0; }
};

namespace driver {
PFN_vkGetInstanceProcAddr real_instance = nullptr;
PFN_vkGetDeviceProcAddr real_device = nullptr;
PFN_vkGetSemaphoreCounterValue real_counter = nullptr;
PFN_vkWaitSemaphores real_wait = nullptr;
std::atomic<VkResult> fail_next_counter{VK_SUCCESS};
std::atomic<VkResult> fail_next_wait{VK_SUCCESS};
std::atomic<unsigned> failed_count{0};
std::atomic<unsigned> failed_wait_count{0};
DWORD test_thread = 0;

VKAPI_ATTR VkResult VKAPI_CALL get_counter(VkDevice device, VkSemaphore semaphore, std::uint64_t* value) {
    // A pending fence may also be polled by the dispatcher thread. Only the explicitly armed
    // synchronous GetCompletedValue call gets the test failure, so it is deterministic.
    if (GetCurrentThreadId() == test_thread) {
        const VkResult injected = fail_next_counter.exchange(VK_SUCCESS);
        if (injected != VK_SUCCESS) {
            // Even a bogus output counter from a failed query may not be treated as completion.
            *value = 999;
            failed_count.fetch_add(1);
            return injected;
        }
    }
    return real_counter(device, semaphore, value);
}

VKAPI_ATTR VkResult VKAPI_CALL wait_semaphores(VkDevice device, const VkSemaphoreWaitInfo* info,
                                               std::uint64_t timeout) {
    if (GetCurrentThreadId() == test_thread) {
        const VkResult injected = fail_next_wait.exchange(VK_SUCCESS);
        if (injected != VK_SUCCESS) {
            failed_wait_count.fetch_add(1);
            return injected;
        }
    }
    return real_wait(device, info, timeout);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkGetSemaphoreCounterValue") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(get_counter);
    if (std::strcmp(name, "vkWaitSemaphores") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(wait_semaphores);
    return real_device(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance(VkInstance instance, const char* name) {
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(get_device);
    return real_instance(instance, name);
}
}  // namespace driver

bool native_warp_control() {
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> adapter;
    Ref<ID3D12Device> device;
    Ref<ID3D12Fence> fence;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) ||
        FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;
    Event event;
    if (!event.handle || fence->GetCompletedValue() != 0 ||
        FAILED(fence->SetEventOnCompletion(2, event.handle)) || event.ready(0) ||
        FAILED(fence->Signal(2)) || !event.ready(2000) ||
        fence->GetCompletedValue() != 2 || device->GetDeviceRemovedReason() != S_OK)
        return false;
    Ref<ID3D12Device1> device1;
    Ref<ID3D12Fence> second;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device1.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&second.p))))
        return false;
    ID3D12Fence* const fences[] = {fence.p, second.p};
    const UINT64 values[] = {2, 3};
    if (FAILED(second->Signal(3)) ||
        FAILED(device1->SetEventOnMultipleFenceCompletion(fences, values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, nullptr)) ||
        FAILED(device1->SetEventOnMultipleFenceCompletion(fences, values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY, nullptr)))
        return false;
    std::printf("native WARP: normal fence query/signal/event completed value 2 and two-fence ALL/ANY blocking waits\n");
    return true;
}
}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (!native_warp_control()) {
        std::fprintf(stderr, "SKIP: native WARP unavailable or failed normal fence control\n");
        return 77;
    }
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    driver::real_instance = vk.instance_proc;
    driver::real_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        vk.instance_proc(vk.instance, "vkGetDeviceProcAddr"));
    if (!driver::real_device) return 1;
    driver::real_counter = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
        driver::real_device(vk.device, "vkGetSemaphoreCounterValue"));
    if (!driver::real_counter) return 1;
    driver::real_wait = reinterpret_cast<PFN_vkWaitSemaphores>(
        driver::real_device(vk.device, "vkWaitSemaphores"));
    if (!driver::real_wait) return 1;
    driver::test_thread = GetCurrentThreadId();

    Event pending;
    if (!pending.handle) return 1;
    int failures = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)))) return 1;
        Ref<ID3D12Fence> fence;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))) ||
            fence->GetCompletedValue() != 0 || device->GetDeviceRemovedReason() != S_OK)
            return 1;
        if (FAILED(fence->SetEventOnCompletion(7, pending.handle)) || pending.ready(0)) return 1;

        driver::fail_next_counter.store(VK_ERROR_OUT_OF_HOST_MEMORY);
        const UINT64 ignored = fence->GetCompletedValue();
        const HRESULT still_alive = device->GetDeviceRemovedReason();
        const bool false_completion = pending.ready(0);
        std::printf("Leonida Vulkan: injected VK_ERROR_OUT_OF_HOST_MEMORY with bogus counter 999: "
                    "value=%016llx, device_removed=%08x, pending_event=%d\n",
                    static_cast<unsigned long long>(ignored), static_cast<unsigned>(still_alive),
                    int(false_completion));
        if (ignored != 0 || still_alive != S_OK || false_completion || driver::failed_count.load() != 1)
            ++failures;

        driver::fail_next_counter.store(VK_ERROR_DEVICE_LOST);
        const UINT64 first = fence->GetCompletedValue();
        const HRESULT removed = device->GetDeviceRemovedReason();
        const bool waiter_released = pending.ready(1500);
        const unsigned faults = driver::failed_count.load();
        std::printf("Leonida Vulkan: injected VK_ERROR_DEVICE_LOST on first GetCompletedValue: "
                    "value=%016llx, device_removed=%08x, pending_event=%d, interceptions=%u\n",
                    static_cast<unsigned long long>(first), static_cast<unsigned>(removed),
                    int(waiter_released), faults);
        if (first != UINT64_MAX || removed != DXGI_ERROR_DEVICE_REMOVED ||
            !waiter_released || faults != 2 || driver::fail_next_counter.load() != VK_SUCCESS)
            ++failures;
        // COM releases the fence before its externally owned VkDevice; on failure the cancel
        // path still drains the pending event rather than leaving a dispatcher thread blocked.
    }
    // A blocking host wait can discover device loss after a successful counter query.
    // Unlike an asynchronous wait registered with a HANDLE, it directly invokes vkWaitSemaphores.
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)))) return 1;
        Ref<ID3D12Fence> fence;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))) ||
            fence->GetCompletedValue() != 0 || device->GetDeviceRemovedReason() != S_OK)
            return 1;
        driver::fail_next_wait.store(VK_ERROR_DEVICE_LOST);
        const HRESULT blocking = fence->SetEventOnCompletion(7, nullptr);
        const HRESULT removed = device->GetDeviceRemovedReason();
        std::printf("Leonida Vulkan: injected VK_ERROR_DEVICE_LOST on blocking fence wait: "
                    "wait=%08x, device_removed=%08x, interceptions=%u\n",
                    unsigned(blocking), unsigned(removed), driver::failed_wait_count.load());
        if (blocking == S_OK || removed != DXGI_ERROR_DEVICE_REMOVED ||
            driver::failed_wait_count.load() != 1 || driver::fail_next_wait.load() != VK_SUCCESS)
            ++failures;
    }
    // ID3D12Device1::SetEventOnMultipleFenceCompletion has a separate direct blocking Vulkan wait,
    // not the single-fence VulkanFence::wait path. Both ALL and ANY must report device loss to the
    // shared context; a transient OOM error must return failure without falsely removing the device.
    for (const auto flags : {D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY}) {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        Ref<ID3D12Device> device;
        Ref<ID3D12Device1> device1;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))) ||
            FAILED(device->QueryInterface(IID_PPV_ARGS(&device1.p)))) return 1;
        Ref<ID3D12Fence> fence_a, fence_b;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_a.p))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_b.p)))) return 1;
        ID3D12Fence* const fences[] = {fence_a.p, fence_b.p};
        const UINT64 values[] = {7, 9};
        const unsigned previous_wait_faults = driver::failed_wait_count.load();
        driver::fail_next_wait.store(VK_ERROR_OUT_OF_HOST_MEMORY);
        const HRESULT oom = device1->SetEventOnMultipleFenceCompletion(fences, values, 2, flags, nullptr);
        const HRESULT still_alive = device->GetDeviceRemovedReason();
        driver::fail_next_wait.store(VK_ERROR_DEVICE_LOST);
        const HRESULT lost = device1->SetEventOnMultipleFenceCompletion(fences, values, 2, flags, nullptr);
        const HRESULT removed = device->GetDeviceRemovedReason();
        const unsigned delta = driver::failed_wait_count.load() - previous_wait_faults;
        std::printf("Leonida Vulkan: multi-fence blocking %s: transient=%08x, nonfatal_reason=%08x, "
                    "lost_wait=%08x, removed=%08x, interceptions=%u\n",
                    flags == D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY ? "ANY" : "ALL",
                    unsigned(oom), unsigned(still_alive), unsigned(lost), unsigned(removed), delta);
        if (oom == S_OK || still_alive != S_OK || lost == S_OK ||
            removed != DXGI_ERROR_DEVICE_REMOVED || delta != 2 ||
            driver::fail_next_wait.load() != VK_SUCCESS)
            ++failures;
    }
    const unsigned errors = vk.validation_errors();
    std::printf("Vulkan validation errors: %u\n", errors);
    return failures || errors ? 1 : 0;
}
