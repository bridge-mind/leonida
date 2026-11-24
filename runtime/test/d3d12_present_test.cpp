// Milestone M0 (docs/d3d12_vulkan.md section 6): a window shows a stable clear colour for 60 frames, through the
// Vulkan-backed device behind the Xbox-ABI proxy, with no Vulkan validation errors. Each frame resets the list,
// moves the back buffer PRESENT -> RENDER_TARGET, clears it, moves it back, executes, presents and signals a fence.
// The window is then resized, which recreates the swapchain and scales the blit. The presented images are copied
// back (the presenter's capture) and must hold the clear colour at the swapchain's size. Last, the colour on screen
// is read back with GDI on native Windows when the window is in front with no other window over its centre, the
// display is in SDR and GDI can read the screen. Under Wine, rootless XWayland can return black for desktop reads,
// so the swapchain checks confirm the colour there. No shaders or game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails the test.
#include "vk_test_device.h"
#include "window_station.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"

#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan_win32.h>

#include <dxgi1_6.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string_view>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: 8-byte handle in RAX

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

// Returns true when the monitor containing the window is running with Windows HDR enabled
// (DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020). On an HDR desktop, DWM scales SDR GDI reads
// by the user's SDR white level, so fixed-tolerance on-screen colour checks do not hold (TEST-209).
bool is_hdr_monitor(HWND window) {
    const HMONITOR target = MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST);
    if (!target) return false;
    Ref<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return false;
    for (UINT i = 0;; ++i) {
        Ref<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(i, &adapter.p) == DXGI_ERROR_NOT_FOUND) break;
        if (!adapter.p) continue;
        for (UINT j = 0;; ++j) {
            Ref<IDXGIOutput> output;
            if (adapter->EnumOutputs(j, &output.p) == DXGI_ERROR_NOT_FOUND) break;
            if (!output.p) continue;
            DXGI_OUTPUT_DESC desc{};
            if (SUCCEEDED(output->GetDesc(&desc)) && desc.Monitor == target) {
                Ref<IDXGIOutput6> output6;
                if (SUCCEEDED(output->QueryInterface(IID_PPV_ARGS(&output6.p)))) {
                    DXGI_OUTPUT_DESC1 desc1{};
                    if (SUCCEEDED(output6->GetDesc1(&desc1)) &&
                        desc1.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020)
                        return true;
                }
            }
        }
    }
    return false;
}

// The LUID of the DXGI adapter whose output shows the window's monitor, or false when it cannot be told. A Vulkan
// swapchain made on another adapter (a render-only GPU, such as an Intel iGPU with no monitor attached while the
// desktop runs on a discrete GPU) is not shown by the desktop compositor at all: a bare vkCreateSwapchainKHR + clear +
// vkQueuePresentKHR program on the UHD 770 leaves the window's own white background on screen, and GDI, BitBlt with
// CAPTUREBLT and PrintWindow all read that white, while the same program on the display GPU reads the clear colour.
bool display_adapter_luid(HWND window, LUID* luid) {
    const HMONITOR target = MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST);
    if (!target) return false;
    Ref<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return false;
    for (UINT i = 0;; ++i) {
        Ref<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(i, &adapter.p) == DXGI_ERROR_NOT_FOUND) break;
        if (!adapter.p) continue;
        for (UINT j = 0;; ++j) {
            Ref<IDXGIOutput> output;
            if (adapter->EnumOutputs(j, &output.p) == DXGI_ERROR_NOT_FOUND) break;
            if (!output.p) continue;
            DXGI_OUTPUT_DESC desc{};
            DXGI_ADAPTER_DESC1 adapter_desc{};
            if (SUCCEEDED(output->GetDesc(&desc)) && desc.Monitor == target &&
                SUCCEEDED(adapter->GetDesc1(&adapter_desc))) {
                *luid = adapter_desc.AdapterLuid;
                return true;
            }
        }
    }
    return false;
}

// True when the Vulkan device is known to be a different adapter from the one driving the window's monitor.
bool device_is_not_display_adapter(const vktest::Device& d, HWND window) {
    LUID display{};
    if (!display_adapter_luid(window, &display)) return false;
    const auto properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        d.instance_proc(d.instance, "vkGetPhysicalDeviceProperties2"));
    if (!properties2) return false;
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    properties2(d.physical, &properties);
    if (!id.deviceLUIDValid) return false;
    return std::memcmp(id.deviceLUID, &display, sizeof(LUID)) != 0;
}

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM w, LPARAM l) {
    return DefWindowProcW(window, message, w, l);
}

void pump() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

// True when the window manager reports this window as the top-level window at a screen point. Being the foreground
// window does not make it so: an always-on-top window, or one another process showed without taking the foreground,
// can lie over it, and a GDI read of the screen then returns that window's pixel. WindowFromPoint passes over hidden
// and disabled windows and over windows that are transparent to hit testing (a click-through overlay,
// WS_EX_LAYERED | WS_EX_TRANSPARENT), so an overlay of that kind drawn over the point is not noticed, and neither
// is anything the compositor draws there itself.
bool is_top_window_at(HWND window, POINT screen_point) {
    const HWND hit = WindowFromPoint(screen_point);
    return hit && GetAncestor(hit, GA_ROOT) == window;
}

void set_client_size(HWND window, int width, int height) {
    RECT rect{0, 0, width, height};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    SetWindowPos(window, HWND_TOP, 0, 0, rect.right - rect.left, rect.bottom - rect.top,
                 SWP_NOMOVE | SWP_NOZORDER | SWP_SHOWWINDOW);
}

// A driver double around the real driver: the device and the presenter get their functions through it
// (VulkanDeviceInfo::get_instance_proc). It forwards every call, and when armed makes one queue submission fail
// without submitting, reports one present that succeeded as another status, or fails one semaphore allocation. It
// also counts the presents the real driver reported out of date, and the binary semaphores alive.
namespace driver {
PFN_vkGetInstanceProcAddr real_get_instance = nullptr;
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;
PFN_vkQueuePresentKHR real_present = nullptr;
PFN_vkCreateSemaphore real_create_semaphore = nullptr;
PFN_vkDestroySemaphore real_destroy_semaphore = nullptr;
PFN_vkGetPhysicalDeviceSurfaceFormatsKHR real_surface_formats = nullptr;
PFN_vkCreateSwapchainKHR real_create_swapchain = nullptr;
PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR real_get_caps = nullptr;
PFN_vkCreateWin32SurfaceKHR real_create_surface = nullptr;
VkResult fail_next_submit = VK_SUCCESS;  // armed: the next vkQueueSubmit returns this and submits nothing
VkResult next_present = VK_SUCCESS;      // armed: the next present that succeeds reports this instead
int fail_semaphore_in = 0;               // armed: the n-th vkCreateSemaphore from now fails
bool fail_surface_formats = false;       // armed: the next surface-format query fails before creation
bool fail_surface_create = false;        // armed: the next vkCreateWin32SurfaceKHR fails (recreation after a loss)
unsigned surface_creations = 0;          // vkCreateWin32SurfaceKHR successes through the double
bool observe_next_swapchain = false;
bool report_undefined_extent = false;    // armed: surface capabilities reports undefined extent (0xFFFFFFFF)
bool undefined_extent_queried = false;
VkExtent2D last_swapchain_extent{};
VkSwapchainKHR last_created_swapchain = VK_NULL_HANDLE;
VkSwapchainKHR observed_old_swapchain = VK_NULL_HANDLE;
unsigned out_of_date = 0;                // presents the real driver reported out of date
std::vector<VkSemaphore> binary;         // binary semaphores alive (the device's fences are timeline ones)

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    if (fail_next_submit != VK_SUCCESS) {
        const VkResult failure = fail_next_submit;
        fail_next_submit = VK_SUCCESS;
        return failure;
    }
    return real_submit(queue, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL queue_present(VkQueue queue, const VkPresentInfoKHR* info) {
    VkResult result = real_present(queue, info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) ++out_of_date;
    if (result == VK_SUCCESS && next_present != VK_SUCCESS) {
        result = next_present;
        next_present = VK_SUCCESS;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL create_semaphore(VkDevice device, const VkSemaphoreCreateInfo* info,
                                                const VkAllocationCallbacks* allocator, VkSemaphore* out) {
    if (fail_semaphore_in && --fail_semaphore_in == 0) return VK_ERROR_OUT_OF_HOST_MEMORY;
    const VkResult result = real_create_semaphore(device, info, allocator, out);
    bool timeline = false;
    for (auto* next = static_cast<const VkBaseInStructure*>(info->pNext); next; next = next->pNext)
        if (next->sType == VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO)
            timeline = reinterpret_cast<const VkSemaphoreTypeCreateInfo*>(next)->semaphoreType ==
                       VK_SEMAPHORE_TYPE_TIMELINE;
    if (result == VK_SUCCESS && !timeline) binary.push_back(*out);
    return result;
}

VKAPI_ATTR void VKAPI_CALL destroy_semaphore(VkDevice device, VkSemaphore semaphore,
                                            const VkAllocationCallbacks* allocator) {
    std::erase(binary, semaphore);
    real_destroy_semaphore(device, semaphore, allocator);
}

VKAPI_ATTR VkResult VKAPI_CALL surface_formats(VkPhysicalDevice physical, VkSurfaceKHR surface,
                                               std::uint32_t* count, VkSurfaceFormatKHR* formats) {
    if (fail_surface_formats) {
        fail_surface_formats = false;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return real_surface_formats(physical, surface, count, formats);
}

VKAPI_ATTR VkResult VKAPI_CALL create_swapchain(VkDevice device, const VkSwapchainCreateInfoKHR* info,
                                                const VkAllocationCallbacks* allocator, VkSwapchainKHR* out) {
    if (info) last_swapchain_extent = info->imageExtent;
    if (observe_next_swapchain) {
        observed_old_swapchain = info->oldSwapchain;
        observe_next_swapchain = false;
    }
    const VkResult result = real_create_swapchain(device, info, allocator, out);
    if (result == VK_SUCCESS) last_created_swapchain = *out;
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL get_surface_capabilities(VkPhysicalDevice physical, VkSurfaceKHR surface,
                                                        VkSurfaceCapabilitiesKHR* caps) {
    const VkResult result = real_get_caps(physical, surface, caps);
    if (result == VK_SUCCESS && report_undefined_extent && caps) {
        caps->currentExtent.width = 0xFFFFFFFF;
        caps->currentExtent.height = 0xFFFFFFFF;
        undefined_extent_queried = true;
    }
    return result;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    if (n == "vkQueuePresentKHR") return reinterpret_cast<PFN_vkVoidFunction>(queue_present);
    if (n == "vkCreateSemaphore") return reinterpret_cast<PFN_vkVoidFunction>(create_semaphore);
    if (n == "vkDestroySemaphore") return reinterpret_cast<PFN_vkVoidFunction>(destroy_semaphore);
    if (n == "vkCreateSwapchainKHR") return reinterpret_cast<PFN_vkVoidFunction>(create_swapchain);
    return real_get_device(device, name);
}

VKAPI_ATTR VkResult VKAPI_CALL create_surface(VkInstance instance, const VkWin32SurfaceCreateInfoKHR* info,
                                              const VkAllocationCallbacks* allocator, VkSurfaceKHR* out) {
    if (fail_surface_create) {
        fail_surface_create = false;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    const VkResult result = real_create_surface(instance, info, allocator, out);
    if (result == VK_SUCCESS) ++surface_creations;
    return result;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance(VkInstance instance, const char* name) {
    const std::string_view n = name;
    if (n == "vkGetDeviceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(get_device);
    if (n == "vkGetPhysicalDeviceSurfaceFormatsKHR")
        return reinterpret_cast<PFN_vkVoidFunction>(surface_formats);
    if (n == "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")
        return reinterpret_cast<PFN_vkVoidFunction>(get_surface_capabilities);
    if (n == "vkCreateWin32SurfaceKHR") return reinterpret_cast<PFN_vkVoidFunction>(create_surface);
    return real_get_instance(instance, name);
}

void install(PFN_vkGetInstanceProcAddr instance_proc, VkInstance instance, VkDevice device) {
    real_get_instance = instance_proc;
    real_get_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(instance_proc(instance, "vkGetDeviceProcAddr"));
    real_submit = reinterpret_cast<PFN_vkQueueSubmit>(real_get_device(device, "vkQueueSubmit"));
    real_present = reinterpret_cast<PFN_vkQueuePresentKHR>(real_get_device(device, "vkQueuePresentKHR"));
    real_create_semaphore = reinterpret_cast<PFN_vkCreateSemaphore>(real_get_device(device, "vkCreateSemaphore"));
    real_destroy_semaphore = reinterpret_cast<PFN_vkDestroySemaphore>(real_get_device(device, "vkDestroySemaphore"));
    real_surface_formats = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
        instance_proc(instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"));
    real_create_swapchain = reinterpret_cast<PFN_vkCreateSwapchainKHR>(real_get_device(device, "vkCreateSwapchainKHR"));
    real_get_caps = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
        instance_proc(instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
    real_create_surface = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
        instance_proc(instance, "vkCreateWin32SurfaceKHR"));
}
}  // namespace driver

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize(true);
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan presentation device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    WNDCLASSW window_class{};
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.lpszClassName = L"LeonidaPresentTest";
    window_class.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
    if (!has_interactive_window_station()) {
        std::fprintf(stderr, "SKIP: the process has no interactive window station\n");
        return 77;
    }
    if (!RegisterClassW(&window_class)) {
        std::fprintf(stderr, "FAIL: cannot register a window class (Win32 error %lu)\n", GetLastError());
        return 1;
    }
    constexpr int width = 320, height = 240;
    RECT frame_rect{0, 0, width, height};
    AdjustWindowRect(&frame_rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(0, window_class.lpszClassName, L"Leonida M0: D3D12 clear on Vulkan",
                                  WS_OVERLAPPEDWINDOW, 100, 100, frame_rect.right - frame_rect.left,
                                  frame_rect.bottom - frame_rect.top, nullptr, nullptr, window_class.hInstance,
                                  nullptr);
    if (!window) {
        std::fprintf(stderr, "FAIL: cannot create the presentation window (Win32 error %lu)\n", GetLastError());
        return 1;
    }
    ShowWindow(window, SW_SHOWNOACTIVATE);
    UpdateWindow(window);
    pump();

    const FLOAT colour[4] = {0.2f, 0.6f, 0.9f, 1.0f};
    bool screen_checked = false;
    {
        driver::install(d.instance_proc, d.instance, d.device);
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.presentation = true;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        auto presenter = rt::d3d12::create_window_presenter(inner, window);
        EXPECT(presenter);
        EXPECT(presenter->enable_capture());  // every presented frame is copied back, for the colour checks

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        Ref<ID3D12DescriptorHeap> rtv_heap;
        Ref<ID3D12Resource> back_buffer;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(list->Close()));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{SIZE_T(reinterpret_cast<XboxHeapHandleFn>(
            (*reinterpret_cast<void***>(rtv_heap.p))[9])(rtv_heap.p))};
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                         D3D12_RESOURCE_STATE_PRESENT, nullptr,
                                                         IID_PPV_ARGS(&back_buffer.p))));
        device->CreateRenderTargetView(back_buffer.p, nullptr, rtv);

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = back_buffer.p;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        ID3D12CommandList* lists[] = {list.p};
        UINT64 signalled = 0;
        // The last presented frame as the swapchain holds it: the whole image is the cleared colour at the size the
        // presenter reports. This reads the image the presentation engine was given, so it holds wherever the screen
        // itself cannot be read back.
        const auto presented_image_is_clear_colour = [&]() -> bool {
            std::vector<std::uint8_t> pixels;
            VkExtent2D size{};
            VkFormat format = VK_FORMAT_UNDEFINED;
            if (!presenter->read_back(pixels, &size, &format)) return false;
            const VkExtent2D current = presenter->extent();
            if (size.width != current.width || size.height != current.height) return false;
            const bool bgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
            for (std::size_t i = 0; i < pixels.size(); i += 4) {
                const int r = pixels[i + (bgra ? 2 : 0)], g = pixels[i + 1], b = pixels[i + (bgra ? 0 : 2)];
                if (std::abs(r - 51) > 1 || std::abs(g - 153) > 1 || std::abs(b - 230) > 1) {
                    const std::size_t texel = i / 4;
                    std::fprintf(stderr, "presented image %ux%u: %d,%d,%d at %zu,%zu (cleared to 51,153,230)\n",
                                 size.width, size.height, r, g, b, texel % size.width, texel / size.width);
                    return false;
                }
            }
            return true;
        };
        // `arm` runs just before the present, so an injection it sets up hits the presenter, not the list's work.
        const auto frame = [&](UINT sync_interval, const std::function<void()>& arm = {}) -> bool {
            pump();
            if (FAILED(fence->SetEventOnCompletion(signalled, nullptr))) return false;  // the list's last run is done
            if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return false;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            list->ResourceBarrier(1, &barrier);
            list->ClearRenderTargetView(rtv, colour, 0, nullptr);
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
            list->ResourceBarrier(1, &barrier);
            if (FAILED(list->Close())) return false;
            queue->ExecuteCommandLists(1, lists);
            if (rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) return false;
            if (arm) arm();
            const HRESULT presented = presenter->present(back_buffer.p, sync_interval);
            driver::fail_next_submit = VK_SUCCESS;  // whatever the present did not use
            driver::next_present = VK_SUCCESS;
            driver::fail_semaphore_in = 0;
            driver::fail_surface_formats = false;
            driver::observe_next_swapchain = false;
            if (FAILED(presented)) return false;
            return SUCCEEDED(queue->Signal(fence.p, ++signalled));
        };

        // Every present is counted unless the surface reported it out of date, which the driver may do around a
        // resize; the double counts those.
        for (int i = 0; i < 60; ++i) EXPECT(frame(1));
        EXPECT(presenter->presented() == 60 - driver::out_of_date);
        const VkExtent2D first = presenter->extent();
        EXPECT(first.width == width && first.height == height);
        EXPECT(presented_image_is_clear_colour());

        // A resize recreates the swapchain; the 320x240 back buffer is scaled to the new size. Unsynchronised
        // presents (sync interval 0) switch the present mode on the way.
        set_client_size(window, 400, 300);
        for (int i = 0; i < 10; ++i) EXPECT(frame(i < 5 ? 1 : 0));
        EXPECT(presenter->extent().width == 400 && presenter->extent().height == 300);
        EXPECT(presenter->presented() == 70 - driver::out_of_date);
        EXPECT(presented_image_is_clear_colour());  // the 320x240 back buffer scaled to the whole 400x300 image
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(signalled, nullptr)));
        const unsigned real_out_of_date = driver::out_of_date;

        // Three of the failures below make the runtime log "[rt] d3d12: present failed"; say so first, so the log is
        // not read as a fault.
        std::fprintf(stderr, "injecting presenter failures: the next three \"present failed\" lines (VkResult "
                             "-1000001004 twice, then -1) are expected\n");

        // Failures injected by the driver double (VictorJulianiR's review of the presenter):
        // 1. Present statuses: OUT_OF_DATE was not shown, so it is not counted; SUBOPTIMAL was, so it is.
        std::uint64_t before = presenter->presented();
        EXPECT(frame(0, [] { driver::next_present = VK_ERROR_OUT_OF_DATE_KHR; }));
        EXPECT(presenter->presented() == before);
        EXPECT(frame(0));  // recreates the swapchain the out-of-date present asked for
        before = presenter->presented();
        EXPECT(frame(0, [] { driver::next_present = VK_SUBOPTIMAL_KHR; }));
        EXPECT(presenter->presented() == before + 1);
        EXPECT(frame(0));

        // A transient surface-format query failure happens before vkCreateSwapchainKHR can retire the old one.
        // A retry can still use the old swapchain for replacement; before this, its handle was lost on failure.
        before = presenter->presented();
        const VkSwapchainKHR old_swapchain = driver::last_created_swapchain;
        EXPECT(old_swapchain);
        EXPECT(!frame(1, [] { driver::fail_surface_formats = true; }));
        EXPECT(presenter->presented() == before);
        EXPECT(frame(0, [] { driver::observe_next_swapchain = true; }));
        EXPECT(driver::observed_old_swapchain == old_swapchain);

        // 2. A swapchain recreation (here for a vsync change) whose second per-image semaphore cannot be made fails
        //    whole: the half-made semaphores and swapchain go, and a retry at the same vsync makes a complete one
        //    instead of acquiring an image without its semaphore (validation would report a null semaphore).
        //    The device makes only timeline semaphores, so the binary ones are the presenter's: one acquire
        //    semaphore per frame in flight (2) and one per swapchain image.
        EXPECT(driver::binary.size() > 2);
        EXPECT(!frame(1, [] { driver::fail_semaphore_in = 2; }));
        EXPECT(driver::binary.size() == 2);  // the old swapchain's went, and nothing of the new set is left
        before = presenter->presented();
        EXPECT(frame(1));
        EXPECT(presenter->presented() == before + 1 && driver::binary.size() > 2);

        // 3. A submit that fails after an image was acquired: present reports it, and the next present neither
        //    waits forever on the frame's fence nor reuses a semaphore that is still pending.
        before = presenter->presented();
        EXPECT(!frame(1, [] { driver::fail_next_submit = VK_ERROR_OUT_OF_HOST_MEMORY; }));
        EXPECT(presenter->presented() == before);
        const unsigned out_of_date_before = driver::out_of_date;
        for (int i = 0; i < 4; ++i) EXPECT(frame(1));  // both frames in flight come round again
        EXPECT(presenter->presented() == before + 4 - (driver::out_of_date - out_of_date_before));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(signalled, nullptr)));

        // 4. Undefined surface extent (0xFFFFFFFF, reported by Intel UHD graphics / Win32 WSI; D3D-165):
        //    switch vsync to 0 to force swapchain recreation, verifying that vkGetPhysicalDeviceSurfaceCapabilitiesKHR
        //    is reached while armed and that vkCreateSwapchainKHR receives the clamped client extent (400x300)
        //    instead of 0xFFFFFFFF.
        driver::report_undefined_extent = true;
        driver::undefined_extent_queried = false;
        EXPECT(frame(0));
        EXPECT(driver::undefined_extent_queried);
        EXPECT(driver::last_swapchain_extent.width == 400 && driver::last_swapchain_extent.height == 300);
        EXPECT(presenter->extent().width == 400 && presenter->extent().height == 300);
        driver::report_undefined_extent = false;
        EXPECT(presented_image_is_clear_colour());

        // 5. A surface lost under a presented swapchain (the window's display went away): the frame is not shown,
        //    the sticky E_FAIL of before #5689's present handling is avoided by remaking the surface for the same
        //    window at the next present and rebuilding the swapchain on it. Until the surface can be remade, the
        //    vkd3d precedent keeps rendering to nothing instead of failing every present.
        unsigned surfaces_before = driver::surface_creations;
        before = presenter->presented();
        EXPECT(frame(0, [] { driver::next_present = VK_ERROR_SURFACE_LOST_KHR; }));
        EXPECT(presenter->presented() == before);                      // nothing was shown
        EXPECT(!presented_image_is_clear_colour());                    // and read_back does not hand out the frame
        EXPECT(driver::surface_creations == surfaces_before);          // the loss is noted, not yet recreated
        EXPECT(frame(0));                                              // the next present remakes the surface...
        EXPECT(driver::surface_creations == surfaces_before + 1);      // ...through the factory callback...
        EXPECT(presenter->presented() == before + 1);                  // ...and shows again
        EXPECT(presented_image_is_clear_colour());
        before = presenter->presented();
        EXPECT(frame(0, [] { driver::next_present = VK_ERROR_SURFACE_LOST_KHR; }));  // lost again, and the remake
        EXPECT(frame(0, [] { driver::fail_surface_create = true; }));  // also fails: keep rendering to nothing
        EXPECT(presenter->presented() == before);                      // the two frames were not shown
        EXPECT(!presented_image_is_clear_colour());
        driver::fail_surface_create = false;
        EXPECT(frame(0));                                              // surfaceable again: recover
        EXPECT(presenter->presented() == before + 1);

        std::printf("injected: out-of-date and suboptimal presents counted apart, a failed swapchain recreation "
                    "released and retried, a failed blit submission recovered, undefined surface extent resolved "
                    "(%u real out-of-date presents)\n",
                    real_out_of_date);

        // Wine's desktop GDI read can return black even when the compositor presents the right colour.
        // The swapchain checks above also cover Wine drivers that cannot read the desktop at all.
        const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
        if (!wine) {
            Sleep(200);
            pump();
        }
        if (wine) {
            std::printf("Wine: on-screen colour check skipped after swapchain readback\n");
        } else if (GetForegroundWindow() == window && !IsIconic(window)) {
            if (is_hdr_monitor(window)) {
                std::printf("the monitor is in HDR mode: on-screen colour check skipped\n");
            } else {
                RECT client{};
                GetClientRect(window, &client);
                POINT centre{(client.right - client.left) / 2, (client.bottom - client.top) / 2};
                ClientToScreen(window, &centre);
                const auto is_clear_colour = [](COLORREF c) {
                    return std::abs(GetRValue(c) - 51) <= 3 && std::abs(GetGValue(c) - 153) <= 3 &&
                           std::abs(GetBValue(c) - 230) <= 3;
                };
                // The foreground window can still be covered at its centre, so a read counts only when this window
                // was the top one there both before and after it. How long the compositor takes to show the last
                // frame, or to stop showing a window that has just gone, is not known (the 200 ms above is a guess),
                // so a pixel that is not the clear colour is read again for up to a second. Nothing is presented
                // meanwhile: a late frame passes, a wrong colour that stays fails.
                COLORREF pixel = CLR_INVALID, first_pixel = CLR_INVALID;
                bool covered = false;
                int reads = 0;
                for (;;) {
                    covered = !is_top_window_at(window, centre);
                    if (covered) break;
                    HDC screen = GetDC(nullptr);
                    pixel = GetPixel(screen, centre.x, centre.y);
                    ReleaseDC(nullptr, screen);
                    covered = !is_top_window_at(window, centre);
                    if (covered) break;
                    if (++reads == 1) first_pixel = pixel;
                    if (pixel == CLR_INVALID || is_clear_colour(pixel) || reads > 20) break;
                    Sleep(50);
                    pump();
                }
                if (covered) {
                    std::printf("another window covers the window centre: on-screen colour check skipped, the "
                                "colour was checked in the presented images\n");
                } else if (pixel == CLR_INVALID) {
                    std::printf("GDI cannot read the screen here (GetPixel on the screen DC returns CLR_INVALID): "
                                "on-screen colour check skipped, the colour was checked in the presented images\n");
                } else {
                    const int r = GetRValue(pixel), g = GetGValue(pixel), b = GetBValue(pixel);
                    std::printf("screen pixel at the window centre: %d,%d,%d (cleared to 51,153,230)\n", r, g, b);
                    if (reads > 1)
                        std::printf("read %d times, 50 ms apart; the first read was %d,%d,%d\n", reads,
                                    GetRValue(first_pixel), GetGValue(first_pixel), GetBValue(first_pixel));
                    if (device_is_not_display_adapter(d, window)) {
                        // The pixel is what the desktop shows, and the desktop does not show this GPU's swapchain.
                        std::printf("the Vulkan device is not the adapter that drives this monitor: the desktop does "
                                    "not show its swapchain, on-screen colour check skipped, the colour was checked "
                                    "in the presented images\n");
                    } else {
                        EXPECT(is_clear_colour(pixel));
                        screen_checked = true;
                    }
                }
            }
        } else {
            std::printf("the window is not in front: on-screen colour check skipped\n");
        }

        presenter.reset();  // before the device's queue and resources go
    }
    DestroyWindow(window);
    UnregisterClassW(window_class.lpszClassName, window_class.hInstance);

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("M0: 70 frames cleared and presented through the Xbox-ABI proxy on Vulkan, swapchain recreated on "
                "resize, colour confirmed in the presented images%s\n", screen_checked ? " and on screen" : "");
    return 0;
}
