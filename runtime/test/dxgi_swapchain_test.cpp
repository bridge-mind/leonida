// Real Win32 HWND GPU test for the Vulkan-backed IDXGIFactory7 / IDXGISwapChain3 COM shim
// (runtime/shims/dxgi_swapchain.h, runtime/shims/dxgi_com.{h,cpp}).
// No shaders or game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation
// (Core + Synchronization Validation); any validation error fails the test.
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "../shims/d3d12_com.h"
#include "../shims/dxgi_com.h"
#include "vulkan_loader.h"
#include "window_station.h"
#include "../src/vulkan_device_filter.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <vector>

namespace {

int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

unsigned& validation_errors() {
    static unsigned count = 0;
    return count;
}

VKAPI_ATTR VkBool32 VKAPI_CALL on_validation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT types,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) &&
        (types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT))
        ++validation_errors();
    std::fprintf(stderr, "validation: %s\n", data && data->pMessage ? data->pMessage : "(no message)");
    return VK_FALSE;
}

D3D12_RESOURCE_BARRIER make_transition(ID3D12Resource* res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

bool wait_fence(ID3D12Fence* fence, UINT64 value) {
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ev) return false;
    const HRESULT hr = fence->SetEventOnCompletion(value, ev);
    if (FAILED(hr)) {
        CloseHandle(ev);
        return false;
    }
    const DWORD w = WaitForSingleObject(ev, 5000);
    CloseHandle(ev);
    return w == WAIT_OBJECT_0;
}

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) { return DefWindowProcW(hwnd, msg, wp, lp); }

HWND create_test_window(int width, int height) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"LeonidaDxgiSwapchainTestWnd";
    RegisterClassW(&wc);
    RECT rc{0, 0, width, height};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    return CreateWindowExW(0, wc.lpszClassName, L"Leonida DXGI Test", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                           CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance,
                           nullptr);
}

// Native DXGI on NVIDIA and WARP reports DISJOINT once after creation or ResizeBuffers, including when the
// first query follows a Present. A null output must not consume that notification (D3D-211).
bool check_statistics_disjoint(IDXGISwapChain* swapchain) {
    if (swapchain->GetFrameStatistics(nullptr) != DXGI_ERROR_INVALID_CALL) return false;
    DXGI_FRAME_STATISTICS stats;
    std::memset(&stats, 0xcd, sizeof(stats));
    if (swapchain->GetFrameStatistics(&stats) != DXGI_ERROR_FRAME_STATISTICS_DISJOINT) return false;
    if (stats.PresentCount || stats.PresentRefreshCount || stats.SyncRefreshCount ||
        stats.SyncQPCTime.QuadPart || stats.SyncGPUTime.QuadPart)
        return false;
    return swapchain->GetFrameStatistics(&stats) == S_OK;
}

// Counts references: AddRef/Release of the private-data interface are observable through refs.
struct CountedUnknown final : IUnknown {
    ULONG refs = 1;
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void** out) override {
        if (out) *out = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
    ULONG STDMETHODCALLTYPE Release() override { return --refs; }
};

// IDXGIObject private data (Microsoft: IDXGIObject::SetPrivateData / SetPrivateDataInterface / GetPrivateData):
// the bytes are copied, an interface is held with a reference until it is replaced or removed, and the same GUID
// replaces the entry. Returns null on success or the failing step.
const char* check_private_data(IDXGIObject* object) {
    static constexpr GUID key{0x6c2c5b0e, 0x4d9a, 0x4f43, {0x9e, 0x1b, 0x3a, 0x57, 0x0c, 0x91, 0x2d, 0x44}};
    UINT size = 99;
    if (object->GetPrivateData(key, &size, nullptr) != DXGI_ERROR_NOT_FOUND) return "GetPrivateData before any Set";
    std::uint32_t value = 0x12345678u;
    if (object->SetPrivateData(key, sizeof(value), &value) != S_OK) return "SetPrivateData";
    value = 0;  // SetPrivateData copies
    size = 0;
    if (object->GetPrivateData(key, &size, nullptr) != S_OK || size != sizeof(value)) return "GetPrivateData size query";
    std::uint32_t read = 0;
    size = sizeof(read);
    if (object->GetPrivateData(key, &size, &read) != S_OK || size != sizeof(read) || read != 0x12345678u)
        return "GetPrivateData did not return the stored bytes";
    std::uint16_t too_small = 0;
    size = sizeof(too_small);
    if (object->GetPrivateData(key, &size, &too_small) != DXGI_ERROR_MORE_DATA || size != sizeof(read))
        return "GetPrivateData with a small buffer should be DXGI_ERROR_MORE_DATA and report the size";
    CountedUnknown counted;
    if (object->SetPrivateDataInterface(key, &counted) != S_OK || counted.refs != 2) return "SetPrivateDataInterface";
    IUnknown* got = nullptr;
    size = sizeof(got);
    if (object->GetPrivateData(key, &size, &got) != S_OK || got != &counted || counted.refs != 3)
        return "GetPrivateData did not return the interface with a reference added";
    got->Release();
    if (object->SetPrivateData(key, sizeof(read), &read) != S_OK || counted.refs != 1)
        return "SetPrivateData with the same GUID should release the held interface";
    if (object->SetPrivateDataInterface(key, &counted) != S_OK || counted.refs != 2) return "SetPrivateDataInterface again";
    if (object->SetPrivateDataInterface(key, nullptr) != S_OK || counted.refs != 1)
        return "SetPrivateDataInterface(null) should remove the entry and release the interface";
    size = 99;
    if (object->GetPrivateData(key, &size, nullptr) != DXGI_ERROR_NOT_FOUND) return "GetPrivateData after removal";
    return nullptr;
}

}  // namespace


// CreateSwapChainForHwnd on a guarded call: a host runtime that faults on a case reports it instead of ending the test.
HRESULT create_guarded(IDXGIFactory4* factory, IUnknown* queue, HWND hwnd, const DXGI_SWAP_CHAIN_DESC1* desc,
                       IDXGISwapChain1** out) {
    __try {
        return factory->CreateSwapChainForHwnd(queue, hwnd, desc, nullptr, nullptr, out);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return HRESULT(0xC0000005);
    }
}

// CreateSwapChainForHwnd with descriptions D3D12 refuses (and the valid ones next to them), against the host runtime's
// DXGI: each case gets its own window on each side, and the HRESULTs must match. Returns the number of mismatches,
// or -1 without a host D3D12 device.
int compare_swap_chain_arguments(IDXGIFactory4* ours, ID3D12Device* our_device, ID3D12CommandQueue* our_queue) {
    using CreateDevice = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
    using CreateFactory2 = HRESULT(WINAPI*)(UINT, REFIID, void**);
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const HMODULE dxgi = LoadLibraryExA("dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create_device = d3d12 ? reinterpret_cast<CreateDevice>(
                                           reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                                     : nullptr;
    const auto create_factory = dxgi ? reinterpret_cast<CreateFactory2>(
                                           reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory2")))
                                     : nullptr;
    ID3D12Device* host_device = nullptr;
    IDXGIFactory4* host_factory = nullptr;
    ID3D12CommandQueue* host_queue = nullptr;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    if (!create_device || !create_factory ||
        FAILED(create_device(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host_device))) ||
        FAILED(create_factory(0, IID_PPV_ARGS(&host_factory))) ||
        FAILED(host_device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&host_queue)))) {
        if (host_factory) host_factory->Release();
        if (host_device) host_device->Release();
        return -1;
    }
    DXGI_SWAP_CHAIN_DESC1 base{};
    base.Width = 64;
    base.Height = 64;
    base.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    base.SampleDesc.Count = 1;
    base.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    base.BufferCount = 2;
    base.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    enum class Pass { queue, null_queue, device, null_window, null_desc, null_out };
    struct Case {
        const char* name;
        DXGI_SWAP_CHAIN_DESC1 desc;
        Pass pass = Pass::queue;
    };
    const auto with = [&](void (*edit)(DXGI_SWAP_CHAIN_DESC1&)) {
        DXGI_SWAP_CHAIN_DESC1 d = base;
        edit(d);
        return d;
    };
    const Case cases[] = {
        {"valid", base},
        {"flip sequential, 3 buffers", with([](DXGI_SWAP_CHAIN_DESC1& d) {
             d.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
             d.BufferCount = 3;
         })},
        {"effect DISCARD", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.SwapEffect = DXGI_SWAP_EFFECT_DISCARD; })},
        {"effect SEQUENTIAL", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.SwapEffect = DXGI_SWAP_EFFECT_SEQUENTIAL; })},
        {"effect 99", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.SwapEffect = DXGI_SWAP_EFFECT(99); })},
        {"scaling NONE", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Scaling = DXGI_SCALING_NONE; })},
        {"scaling ASPECT_RATIO_STRETCH",
         with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Scaling = DXGI_SCALING_ASPECT_RATIO_STRETCH; })},
        {"scaling 7", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Scaling = DXGI_SCALING(7); })},
        {"alpha IGNORE", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.AlphaMode = DXGI_ALPHA_MODE_IGNORE; })},
        {"alpha PREMULTIPLIED", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED; })},
        {"alpha STRAIGHT", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.AlphaMode = DXGI_ALPHA_MODE_STRAIGHT; })},
        {"alpha 9", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.AlphaMode = DXGI_ALPHA_MODE(9); })},
        {"flag NONPREROTATED", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Flags = DXGI_SWAP_CHAIN_FLAG_NONPREROTATED; })},
        {"flag ALLOW_MODE_SWITCH",
         with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH; })},
        {"flag GDI_COMPATIBLE", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Flags = DXGI_SWAP_CHAIN_FLAG_GDI_COMPATIBLE; })},
        {"flag DISPLAY_ONLY", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Flags = DXGI_SWAP_CHAIN_FLAG_DISPLAY_ONLY; })},
        {"flag FRAME_LATENCY_WAITABLE_OBJECT", with([](DXGI_SWAP_CHAIN_DESC1& d) {
             d.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
         })},
        {"flag FOREGROUND_LAYER",
         with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Flags = DXGI_SWAP_CHAIN_FLAG_FOREGROUND_LAYER; })},
        {"flag HW_PROTECTED", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Flags = DXGI_SWAP_CHAIN_FLAG_HW_PROTECTED; })},
        {"flag ALLOW_TEARING", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING; })},
        {"flag 0x1000", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Flags = 0x1000; })},
        {"flag 0x2000", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Flags = 0x2000; })},
        {"flag 0x4000", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Flags = 0x4000; })},
        {"flag 0x80000000", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Flags = 0x80000000u; })},
        {"usage 0", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.BufferUsage = 0; })},
        {"usage SHADER_INPUT", with([](DXGI_SWAP_CHAIN_DESC1& d) {
             d.BufferUsage = DXGI_USAGE_SHADER_INPUT | DXGI_USAGE_RENDER_TARGET_OUTPUT;
         })},
        {"usage SHARED", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.BufferUsage = DXGI_USAGE_SHARED; })},
        {"usage READ_ONLY", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.BufferUsage = DXGI_USAGE_READ_ONLY; })},
        {"usage DISCARD_ON_PRESENT",
         with([](DXGI_SWAP_CHAIN_DESC1& d) { d.BufferUsage = DXGI_USAGE_DISCARD_ON_PRESENT; })},
        {"usage UNORDERED_ACCESS", with([](DXGI_SWAP_CHAIN_DESC1& d) {
             d.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_UNORDERED_ACCESS;
         })},
        {"16384 x 16", with([](DXGI_SWAP_CHAIN_DESC1& d) {
             d.Width = 16384;
             d.Height = 16;
         })},
        {"width 16385", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Width = 16385; })},
        {"height 16385", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.Height = 16385; })},
        {"one buffer", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.BufferCount = 1; })},
        {"MSAA 4", with([](DXGI_SWAP_CHAIN_DESC1& d) { d.SampleDesc.Count = 4; })},
        {"width 16385 and MSAA 4", with([](DXGI_SWAP_CHAIN_DESC1& d) {
             d.Width = 16385;
             d.SampleDesc.Count = 4;
         })},
        {"width 16385 and one buffer", with([](DXGI_SWAP_CHAIN_DESC1& d) {
             d.Width = 16385;
             d.BufferCount = 1;
         })},
        {"width 16385 and format UNKNOWN", with([](DXGI_SWAP_CHAIN_DESC1& d) {
             d.Width = 16385;
             d.Format = DXGI_FORMAT_UNKNOWN;
         })},
        {"height 16385 and format BC1", with([](DXGI_SWAP_CHAIN_DESC1& d) {
             d.Height = 16385;
             d.Format = DXGI_FORMAT_BC1_UNORM;
         })},
        {"width 16385 and effect DISCARD", with([](DXGI_SWAP_CHAIN_DESC1& d) {
             d.Width = 16385;
             d.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
         })},
        {"null queue", base, Pass::null_queue},
        {"a device, not a queue", base, Pass::device},
        {"null window", base, Pass::null_window},
        {"null description", base, Pass::null_desc},
        {"null output", base, Pass::null_out},
    };
    int mismatches = 0;
    for (const Case& c : cases) {
        HRESULT result[2];
        for (int side = 0; side < 2; ++side) {
            IDXGIFactory4* factory = side ? host_factory : ours;
            IUnknown* queue = side ? static_cast<IUnknown*>(host_queue) : static_cast<IUnknown*>(our_queue);
            if (c.pass == Pass::null_queue) queue = nullptr;
            if (c.pass == Pass::device) queue = side ? static_cast<IUnknown*>(host_device) : our_device;
            HWND hwnd = create_test_window(64, 64);
            IDXGISwapChain1* chain = nullptr;
            result[side] = create_guarded(factory, queue, c.pass == Pass::null_window ? nullptr : hwnd,
                                          c.pass == Pass::null_desc ? nullptr : &c.desc,
                                          c.pass == Pass::null_out ? nullptr : &chain);
            if (chain) chain->Release();
            if (hwnd) DestroyWindow(hwnd);
        }
        if (result[0] != result[1]) {
            ++mismatches;
            std::fprintf(stderr, "CreateSwapChainForHwnd %s: ours 0x%08lx, host 0x%08lx\n", c.name,
                         static_cast<unsigned long>(result[0]), static_cast<unsigned long>(result[1]));
        }
    }
    std::printf("CreateSwapChainForHwnd: %zu descriptions compared with the host runtime, %d mismatches\n",
                std::size(cases), mismatches);
    host_queue->Release();
    host_factory->Release();
    host_device->Release();
    return mismatches;
}

int main() {
    if (!has_interactive_window_station()) {
        // The host answers DXGI_ERROR_NOT_CURRENTLY_AVAILABLE to every CreateSwapChainForHwnd there: nothing to compare.
        std::printf("SKIP: the process has no interactive window station\n");
        return 77;  // ctest skips
    }
    const PFN_vkGetInstanceProcAddr gipa = load_vulkan();
    if (!gipa) {
        std::printf("SKIP: vulkan-1.dll not available\n");
        return 77;  // ctest skips
    }
    const auto create_inst = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
    const auto enum_layers =
        reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(gipa(nullptr, "vkEnumerateInstanceLayerProperties"));
    if (!create_inst) return fail("vkCreateInstance missing");

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const char* env_val = std::getenv("LEONIDA_VK_VALIDATION");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    const bool want_val = env_val && std::strcmp(env_val, "0") != 0;
    const char* val_layer = "VK_LAYER_KHRONOS_validation";
    bool val_active = false;
    if (want_val && enum_layers) {
        std::uint32_t cnt = 0;
        enum_layers(&cnt, nullptr);
        std::vector<VkLayerProperties> layers(cnt);
        enum_layers(&cnt, layers.data());
        for (const auto& l : layers) {
            if (!std::strcmp(l.layerName, val_layer)) val_active = true;
        }
        if (!val_active) return fail("LEONIDA_VK_VALIDATION is set but VK_LAYER_KHRONOS_validation is not installed");
    }

    std::vector<const char*> inst_exts = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    if (val_active) inst_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkDebugUtilsMessengerCreateInfoEXT dbg{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    dbg.pfnUserCallback = on_validation;

    const VkValidationFeatureEnableEXT sync_val = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT val_feats{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT, &dbg};
    val_feats.enabledValidationFeatureCount = 1;
    val_feats.pEnabledValidationFeatures = &sync_val;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = static_cast<std::uint32_t>(inst_exts.size());
    ici.ppEnabledExtensionNames = inst_exts.data();
    if (val_active) {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &val_layer;
        ici.pNext = &val_feats;
    }

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult created = create_inst(&ici, nullptr, &instance);
    if (created == VK_ERROR_INCOMPATIBLE_DRIVER) {  // no Vulkan driver at all, as on a GPU-less runner
        std::printf("SKIP: no Vulkan driver\n");
        return 77;
    }
    if (created != VK_SUCCESS) return fail("vkCreateInstance");

    const auto destroy_inst = reinterpret_cast<PFN_vkDestroyInstance>(gipa(instance, "vkDestroyInstance"));
    const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance, "vkGetDeviceProcAddr"));
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_dbg = nullptr;
    if (val_active) {
        const auto create_dbg =
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(gipa(instance, "vkCreateDebugUtilsMessengerEXT"));
        destroy_dbg =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(gipa(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (create_dbg) create_dbg(instance, &dbg, nullptr, &messenger);
    }

    const auto enum_devs = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(gipa(instance, "vkEnumeratePhysicalDevices"));
    const auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(gipa(instance, "vkGetPhysicalDeviceProperties"));
    const auto get_qf = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        gipa(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
    const auto create_dev = reinterpret_cast<PFN_vkCreateDevice>(gipa(instance, "vkCreateDevice"));

    std::uint32_t pcnt = 0;
    enum_devs(instance, &pcnt, nullptr);
    std::vector<VkPhysicalDevice> physicals(pcnt);
    enum_devs(instance, &pcnt, physicals.data());

    VkPhysicalDevice physical = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkPhysicalDeviceProperties chosen{};
    int best_rank = -1;
    for (auto cand : physicals) {
        VkPhysicalDeviceProperties props{};
        get_props(cand, &props);
        if (props.apiVersion < VK_API_VERSION_1_3 || !leonida::vulkan_device_wanted(props.deviceName)) continue;
        std::uint32_t fcnt = 0;
        get_qf(cand, &fcnt, nullptr);
        std::vector<VkQueueFamilyProperties> fams(fcnt);
        get_qf(cand, &fcnt, fams.data());
        for (std::uint32_t i = 0; i < fcnt; ++i) {
            if (!(fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            // the runtime's pick: the first usable device of the highest rank (native driver over Dozen, #915)
            if (const int rank = leonida::vulkan_device_rank(static_cast<int>(props.deviceType), props.vendorID);
                rank > best_rank) {
                best_rank = rank;
                physical = cand;
                family = i;
                chosen = props;
            }
            break;
        }
    }
    if (!physical) {
        const std::string filter = leonida::vulkan_device_filter();
        if (filter.empty()) std::printf("SKIP: no Vulkan 1.3 GPU found\n");
        else std::fprintf(stderr, "LEONIDA_VK_DEVICE=%s: no Vulkan 1.3 GPU matches\n", filter.c_str());
        if (messenger && destroy_dbg) destroy_dbg(instance, messenger, nullptr);
        destroy_inst(instance, nullptr);
        if (!filter.empty()) return 1;  // a filter matching nothing fails, never skips
        return 77;  // ctest skips
    }

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.timelineSemaphore = VK_TRUE;
    f12.separateDepthStencilLayouts = VK_TRUE;

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    const char* dev_ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f12};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = &dev_ext;

    VkDevice vk_dev = VK_NULL_HANDLE;
    if (create_dev(physical, &dci, nullptr, &vk_dev) != VK_SUCCESS) return fail("vkCreateDevice");
    const auto destroy_dev = reinterpret_cast<PFN_vkDestroyDevice>(gdpa(vk_dev, "vkDestroyDevice"));
    const auto wait_idle = reinterpret_cast<PFN_vkDeviceWaitIdle>(gdpa(vk_dev, "vkDeviceWaitIdle"));
    VkQueue vk_queue = VK_NULL_HANDLE;
    reinterpret_cast<PFN_vkGetDeviceQueue>(gdpa(vk_dev, "vkGetDeviceQueue"))(vk_dev, family, 0, &vk_queue);

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = gipa;
    info.instance = instance;
    info.physical = physical;
    info.device = vk_dev;
    info.queue = vk_queue;
    info.queue_family = family;
    info.separate_depth_stencil_layouts = true;

    ID3D12Device* dev = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev))) || !dev) return fail("create_vulkan_device");

    HWND hwnd = create_test_window(128, 128);
    if (!hwnd) return fail("CreateWindowExW");

    IDXGIFactory4* factory = nullptr;
    if (FAILED(rt::d3d12::create_dxgi_factory(dev, IID_PPV_ARGS(&factory))) || !factory)
        return fail("create_dxgi_factory");

    // Host-measured on Windows 11: MakeWindowAssociation takes DXGI_MWA_NO_WINDOW_CHANGES, DXGI_MWA_NO_ALT_ENTER
    // and DXGI_MWA_NO_PRINT_SCREEN together or alone; any other bit is DXGI_ERROR_INVALID_CALL.
    {
        constexpr UINT known = DXGI_MWA_NO_WINDOW_CHANGES | DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_PRINT_SCREEN;
        if (FAILED(factory->MakeWindowAssociation(hwnd, known))) return fail("MakeWindowAssociation(known flags)");
        for (const UINT bad : {0x8u, 0x40u, known | 0x8u}) {
            if (factory->MakeWindowAssociation(hwnd, bad) != DXGI_ERROR_INVALID_CALL)
                return fail("MakeWindowAssociation accepted an unknown flag bit");
        }
    }

    D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* cl = nullptr;
    ID3D12Fence* fence = nullptr;
    if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) ||
        FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
        FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&cl))) ||
        FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
        return fail("queue/alloc/cl/fence");
    cl->Close();

    // Before the test's own swap chain: the host comparison creates and releases its own windows and chains.
    const int argument_mismatches = compare_swap_chain_arguments(factory, dev, queue);
    if (argument_mismatches < 0) std::printf("CreateSwapChainForHwnd: no host D3D12 device to compare with, skipped\n");
    if (argument_mismatches > 0) return fail("CreateSwapChainForHwnd answers differ from the host runtime");

    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.Width = 128;
    scd.Height = 128;
    scd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount = 2;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    for (const UINT count : {0u, 1u, DXGI_MAX_SWAP_CHAIN_BUFFERS + 1u}) {
        DXGI_SWAP_CHAIN_DESC1 invalid = scd;
        invalid.BufferCount = count;
        IDXGISwapChain1* rejected = nullptr;
        const HRESULT hr = factory->CreateSwapChainForHwnd(queue, hwnd, &invalid, nullptr, nullptr, &rejected);
        if (rejected) {
            rejected->Release();
            return fail("CreateSwapChainForHwnd returned a swapchain for an invalid buffer count");
        }
        if (hr != DXGI_ERROR_INVALID_CALL) return fail("CreateSwapChainForHwnd invalid buffer count HRESULT");
    }

    constexpr DXGI_SAMPLE_DESC invalid_samples[] = {{0, 0}, {2, 0}, {4, 0}, {1, 1}};
    for (const auto& samples : invalid_samples) {
        DXGI_SWAP_CHAIN_DESC1 invalid = scd;
        invalid.SampleDesc = samples;
        IDXGISwapChain1* rejected = nullptr;
        const HRESULT hr = factory->CreateSwapChainForHwnd(queue, hwnd, &invalid, nullptr, nullptr, &rejected);
        if (rejected) {
            rejected->Release();
            return fail("CreateSwapChainForHwnd returned a swapchain for an invalid sample description");
        }
        if (hr != DXGI_ERROR_INVALID_CALL) return fail("CreateSwapChainForHwnd invalid sample description HRESULT");
    }

    IDXGISwapChain1* sc1 = nullptr;
    if (FAILED(factory->CreateSwapChainForHwnd(queue, hwnd, &scd, nullptr, nullptr, &sc1)) || !sc1)
        return fail("CreateSwapChainForHwnd");
    IDXGISwapChain3* sc = nullptr;
    if (FAILED(sc1->QueryInterface(IID_PPV_ARGS(&sc))) || !sc) return fail("QI IDXGISwapChain3");
    sc1->Release();

    // DXGI reports the BufferCount the app created with, never a larger native/Vulkan image count (D3D-389).
    {
        DXGI_SWAP_CHAIN_DESC1 created_desc{};
        DXGI_SWAP_CHAIN_DESC created_desc0{};
        if (FAILED(sc->GetDesc1(&created_desc)) || created_desc.BufferCount != scd.BufferCount)
            return fail("GetDesc1 BufferCount is not the requested count");
        if (FAILED(sc->GetDesc(&created_desc0)) || created_desc0.BufferCount != scd.BufferCount)
            return fail("GetDesc BufferCount is not the requested count");
    }

    if (!check_statistics_disjoint(sc)) return fail("initial frame-statistics disjoint notification");

    ID3D12Device* got_device = nullptr;
    if (FAILED(sc->GetDevice(IID_PPV_ARGS(&got_device))) || got_device != dev) {
        if (got_device) got_device->Release();
        return fail("GetDevice did not return the command queue's parent device");
    }
    got_device->Release();

    IUnknown* got_unknown = nullptr;
    IUnknown* expected_device_unknown = nullptr;
    IUnknown* queue_unknown = nullptr;
    const HRESULT got_unknown_hr = sc->GetDevice(IID_PPV_ARGS(&got_unknown));
    if (FAILED(got_unknown_hr) || !got_unknown) {
        if (got_unknown) got_unknown->Release();
        return fail("GetDevice(IUnknown) failed");
    }
    if (FAILED(dev->QueryInterface(IID_PPV_ARGS(&expected_device_unknown))) || !expected_device_unknown) {
        got_unknown->Release();
        return fail("QueryInterface(IUnknown) on device failed");
    }
    if (FAILED(queue->QueryInterface(IID_PPV_ARGS(&queue_unknown))) || !queue_unknown) {
        expected_device_unknown->Release();
        got_unknown->Release();
        return fail("QueryInterface(IUnknown) on queue failed");
    }
    const bool has_device_identity = got_unknown == expected_device_unknown && got_unknown != queue_unknown;
    queue_unknown->Release();
    expected_device_unknown->Release();
    got_unknown->Release();
    if (!has_device_identity)
        return fail("GetDevice(IUnknown) returned the queue identity instead of its parent device");
    DXGI_SWAP_CHAIN_FULLSCREEN_DESC fullscreen{};
    if (FAILED(sc->GetFullscreenDesc(&fullscreen)) || !fullscreen.Windowed ||
        fullscreen.Scaling != DXGI_MODE_SCALING_UNSPECIFIED ||
        fullscreen.ScanlineOrdering != DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED)
        return fail("GetFullscreenDesc");

    // Factory::GetParent: root DXGI object has no parent; native DXGI returns E_NOINTERFACE
    void* parent_out = reinterpret_cast<void*>(1);
    if (factory->GetParent(IID_IUnknown, nullptr) != E_POINTER)
        return fail("Factory::GetParent with null out should return E_POINTER");
    if (factory->GetParent(IID_IUnknown, &parent_out) != E_NOINTERFACE || parent_out != nullptr)
        return fail("Factory::GetParent should return E_NOINTERFACE and set null out");

    // SwapChain::GetCoreWindow: HWND swapchain is not associated with a CoreWindow
    void* cw_out = reinterpret_cast<void*>(1);
    if (sc->GetCoreWindow(IID_IUnknown, nullptr) != DXGI_ERROR_INVALID_CALL)
        return fail("GetCoreWindow with null out should return DXGI_ERROR_INVALID_CALL");
    if (sc->GetCoreWindow(IID_IUnknown, &cw_out) != DXGI_ERROR_INVALID_CALL || cw_out != nullptr)
        return fail("GetCoreWindow on HWND swapchain should return DXGI_ERROR_INVALID_CALL and set null out");

    // SwapChain::GetRestrictToOutput: unrestricted swapchain returns DXGI_ERROR_NOT_FOUND
    IDXGIOutput* restrict_out = reinterpret_cast<IDXGIOutput*>(1);
    if (sc->GetRestrictToOutput(nullptr) != E_INVALIDARG)
        return fail("GetRestrictToOutput with null out should return E_INVALIDARG");
    if (sc->GetRestrictToOutput(&restrict_out) != DXGI_ERROR_NOT_FOUND || restrict_out != nullptr)
        return fail("GetRestrictToOutput on unrestricted swapchain should return DXGI_ERROR_NOT_FOUND and set null out");

    D3D12_DESCRIPTOR_HEAP_DESC rtv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ID3D12DescriptorHeap* rtv_heap = nullptr;
    if (FAILED(dev->CreateDescriptorHeap(&rtv_hd, IID_PPV_ARGS(&rtv_heap)))) return fail("CreateDescriptorHeap");
    const D3D12_CPU_DESCRIPTOR_HANDLE h_rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();

    UINT64 fence_val = 0;
    auto run_clear_readback_present = [&](UINT exp_w, UINT exp_h, const float rgba[4], std::uint8_t exp_r,
                                          std::uint8_t exp_g, std::uint8_t exp_b) -> int {
        DXGI_SWAP_CHAIN_DESC1 cur_desc{};
        if (FAILED(sc->GetDesc1(&cur_desc)) || cur_desc.Width != exp_w || cur_desc.Height != exp_h) {
            std::fprintf(stderr, "FAIL: GetDesc1 (%ux%u) != expected (%ux%u)\n", cur_desc.Width, cur_desc.Height,
                         exp_w, exp_h);
            return 1;
        }
        const UINT idx = sc->GetCurrentBackBufferIndex();
        if (idx >= cur_desc.BufferCount) return fail("GetCurrentBackBufferIndex >= BufferCount");
        ID3D12Resource* bb = nullptr;
        if (FAILED(sc->GetBuffer(idx, IID_PPV_ARGS(&bb))) || !bb) return fail("GetBuffer");

        const D3D12_RESOURCE_DESC bb_desc = bb->GetDesc();
        if (bb_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || bb_desc.Width != exp_w ||
            bb_desc.Height != exp_h) {
            std::fprintf(stderr, "FAIL: BackBuffer GetDesc (%llux%u) != expected (%ux%u)\n",
                         static_cast<unsigned long long>(bb_desc.Width), bb_desc.Height, exp_w, exp_h);
            bb->Release();
            return 1;
        }
        D3D12_HEAP_PROPERTIES hp_out{};
        D3D12_HEAP_FLAGS hf_out{};
        if (FAILED(bb->GetHeapProperties(&hp_out, &hf_out)) || hp_out.Type != D3D12_HEAP_TYPE_DEFAULT) {
            bb->Release();
            return fail("BackBuffer GetHeapProperties invalid");
        }

        dev->CreateRenderTargetView(bb, nullptr, h_rtv);

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT64 total_bytes = 0;
        dev->GetCopyableFootprints(&bb_desc, 0, 1, 0, &fp, nullptr, nullptr, &total_bytes);

        D3D12_HEAP_PROPERTIES rb_hp{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC rb_desc{};
        rb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rb_desc.Width = total_bytes;
        rb_desc.Height = 1;
        rb_desc.DepthOrArraySize = 1;
        rb_desc.MipLevels = 1;
        rb_desc.SampleDesc.Count = 1;
        rb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* rb = nullptr;
        if (FAILED(dev->CreateCommittedResource(&rb_hp, D3D12_HEAP_FLAG_NONE, &rb_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                nullptr, IID_PPV_ARGS(&rb)))) {
            bb->Release();
            return fail("CreateCommittedResource(readback)");
        }

        alloc->Reset();
        cl->Reset(alloc, nullptr);
        auto b_rt = make_transition(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
        cl->ResourceBarrier(1, &b_rt);
        cl->ClearRenderTargetView(h_rtv, rgba, 0, nullptr);
        auto b_src = make_transition(bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cl->ResourceBarrier(1, &b_src);
        D3D12_TEXTURE_COPY_LOCATION src_loc{bb, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
        D3D12_TEXTURE_COPY_LOCATION dst_loc{};
        dst_loc.pResource = rb;
        dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst_loc.PlacedFootprint = fp;
        cl->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
        auto b_pres = make_transition(bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
        cl->ResourceBarrier(1, &b_pres);
        cl->Close();

        ID3D12CommandList* lists[1] = {cl};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, ++fence_val);
        if (!wait_fence(fence, fence_val)) {
            rb->Release();
            bb->Release();
            return fail("wait_fence");
        }

        void* mapped = nullptr;
        if (FAILED(rb->Map(0, nullptr, &mapped)) || !mapped) {
            rb->Release();
            bb->Release();
            return fail("readback Map");
        }
        const auto* base = static_cast<const std::uint8_t*>(mapped);
        const bool is_bgra = (bb_desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM ||
                              bb_desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB);
        bool mismatch = false;
        std::uint8_t bad_r = 0, bad_g = 0, bad_b = 0, bad_a = 0;
        for (UINT y = 0; y < exp_h && !mismatch; ++y) {
            const std::uint8_t* row = base + static_cast<std::size_t>(y) * fp.Footprint.RowPitch;
            for (UINT x = 0; x < exp_w; ++x) {
                const std::uint8_t* px = row + static_cast<std::size_t>(x) * 4;
                const std::uint8_t r = is_bgra ? px[2] : px[0];
                const std::uint8_t g = px[1];
                const std::uint8_t b = is_bgra ? px[0] : px[2];
                const std::uint8_t a = px[3];
                if (std::abs(int(r) - int(exp_r)) > 1 || std::abs(int(g) - int(exp_g)) > 1 ||
                    std::abs(int(b) - int(exp_b)) > 1 || a != 255) {
                    mismatch = true;
                    bad_r = r;
                    bad_g = g;
                    bad_b = b;
                    bad_a = a;
                    break;
                }
            }
        }
        rb->Unmap(0, nullptr);
        rb->Release();
        bb->Release();

        if (mismatch) {
            std::fprintf(stderr, "FAIL: backbuffer pixel (%u,%u,%u,%u) != expected (%u,%u,%u,255)\n", bad_r, bad_g,
                         bad_b, bad_a, exp_r, exp_g, exp_b);
            return 1;
        }
        if (FAILED(sc->Present(0, 0))) return fail("Present");
        return 0;
    };

    {
        DXGI_SWAP_CHAIN_DESC1 before{}, after{};
        if (FAILED(sc->GetDesc1(&before)) || before.BufferCount < 2)
            return fail("GetDesc1 before ResizeBuffers1 argument tests");
        std::vector<UINT> masks(before.BufferCount, 1);
        std::vector<IUnknown*> queues(before.BufferCount, queue);
        const auto resize = [&](UINT count) {
            return sc->ResizeBuffers1(count, before.Width + 1, before.Height, DXGI_FORMAT_UNKNOWN, 0,
                                      masks.data(), queues.data());
        };
        for (UINT invalid_mask : {2u, 3u}) {
            masks.back() = invalid_mask;
            if (resize(0) != DXGI_ERROR_INVALID_CALL) return fail("ResizeBuffers1 invalid node mask with count zero");
        }
        masks.back() = 1;
        queues.back() = dev;
        if (resize(before.BufferCount) != DXGI_ERROR_INVALID_CALL)
            return fail("ResizeBuffers1 should reject a non-queue object");

        D3D12_COMMAND_QUEUE_DESC copy_desc{};
        copy_desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        ID3D12CommandQueue* copy = nullptr;
        if (FAILED(dev->CreateCommandQueue(&copy_desc, IID_PPV_ARGS(&copy))) || !copy)
            return fail("CreateCommandQueue for ResizeBuffers1 copy queue test");
        queues.back() = copy;
        const HRESULT copy_hr = resize(0);
        copy->Release();
        if (copy_hr != DXGI_ERROR_INVALID_CALL) return fail("ResizeBuffers1 should reject a copy queue");

        if (FAILED(sc->GetDesc1(&after)) || after.Width != before.Width || after.Height != before.Height ||
            after.BufferCount != before.BufferCount)
            return fail("Invalid ResizeBuffers1 arguments changed the swapchain");
        queues.back() = queue;
        masks.front() = 0;
        if (FAILED(sc->ResizeBuffers1(0, before.Width, before.Height, DXGI_FORMAT_UNKNOWN, 0, masks.data(), queues.data())))
            return fail("ResizeBuffers1 should accept node masks zero and one with direct queues");
    }

    // Invalid counts must leave the initial back buffers usable for the first frame.
    {
        DXGI_SWAP_CHAIN_DESC1 before_resize{};
        if (FAILED(sc->GetDesc1(&before_resize))) return fail("GetDesc1 before invalid resizes");
        const std::vector<UINT> node_masks(DXGI_MAX_SWAP_CHAIN_BUFFERS + 1, 1);
        const std::vector<IUnknown*> present_queues(DXGI_MAX_SWAP_CHAIN_BUFFERS + 1, queue);
        for (const UINT count : {1u, DXGI_MAX_SWAP_CHAIN_BUFFERS + 1u}) {
            if (sc->ResizeBuffers(count, 64, 64, DXGI_FORMAT_UNKNOWN, 0) != E_INVALIDARG)
                return fail("ResizeBuffers invalid buffer count HRESULT");
            if (sc->ResizeBuffers1(count, 64, 64, DXGI_FORMAT_UNKNOWN, 0, node_masks.data(), present_queues.data()) !=
                E_INVALIDARG)
                return fail("ResizeBuffers1 invalid buffer count HRESULT");
        }
        DXGI_SWAP_CHAIN_DESC1 after_resize{};
        if (FAILED(sc->GetDesc1(&after_resize)) || after_resize.BufferCount != before_resize.BufferCount ||
            after_resize.Width != before_resize.Width || after_resize.Height != before_resize.Height)
            return fail("Invalid resize changed swapchain buffers");
    }

    {
        DXGI_SWAP_CHAIN_DESC1 buffer_desc{};
        if (FAILED(sc->GetDesc1(&buffer_desc)) || buffer_desc.BufferCount == 0)
            return fail("GetDesc1 before testing GetBuffer indices");
        ID3D12Resource* last = nullptr;
        if (FAILED(sc->GetBuffer(buffer_desc.BufferCount - 1, IID_PPV_ARGS(&last))) || !last)
            return fail("GetBuffer should return the last back buffer");
        last->Release();

        for (UINT index : {buffer_desc.BufferCount, buffer_desc.BufferCount + 1, ~0u}) {
            void* rejected = reinterpret_cast<void*>(1);
            const HRESULT hr = sc->GetBuffer(index, __uuidof(ID3D12Resource), &rejected);
            if (hr != DXGI_ERROR_INVALID_CALL || rejected != nullptr) {
                std::fprintf(stderr, "FAIL: GetBuffer(%u) returned 0x%08lx, expected DXGI_ERROR_INVALID_CALL and null\n",
                             index, static_cast<unsigned long>(hr));
                return 1;
            }
        }
    }

    // Frame 1: 128x128 clear to (64, 128, 255, 255), read back all 16,384 pixels, Present.
    const float blue_ish[4] = {64.0f / 255.0f, 128.0f / 255.0f, 1.0f, 1.0f};
    if (run_clear_readback_present(128, 128, blue_ish, 64, 128, 255) != 0) return 1;

    // Frame 2: ResizeBuffers to explicit 37x19 on the unchanged 128x128 HWND, clear to green, read back all 703 pixels, Present.
    if (FAILED(sc->ResizeBuffers(2, 37, 19, DXGI_FORMAT_UNKNOWN, 0))) return fail("ResizeBuffers");
    DXGI_SWAP_CHAIN_DESC1 preserved_format{};
    if (FAILED(sc->GetDesc1(&preserved_format)) || preserved_format.Format != scd.Format)
        return fail("ResizeBuffers with DXGI_FORMAT_UNKNOWN changed the format");
    const float green[4] = {0.0f, 1.0f, 0.0f, 1.0f};
    if (run_clear_readback_present(37, 19, green, 0, 255, 0) != 0) return 1;

    // Frame 3: ResizeBuffers to 37x19 DXGI_FORMAT_R10G10B10A2_UNORM on the unchanged 128x128 HWND.
    // Preserves 10-bit format in GetDesc1, GetBuffer()->GetDesc(), and GPU readback before blitting on Present.
    if (FAILED(sc->ResizeBuffers(2, 37, 19, DXGI_FORMAT_R10G10B10A2_UNORM, 0)))
        return fail("ResizeBuffers R10G10B10A2_UNORM");
    {
        DXGI_SWAP_CHAIN_DESC1 d10{};
        if (FAILED(sc->GetDesc1(&d10))) return fail("GetDesc1 (10-bit)");
        if (d10.Width != 37 || d10.Height != 19 || d10.Format != DXGI_FORMAT_R10G10B10A2_UNORM)
            return fail("GetDesc1 did not preserve 37x19 R10G10B10A2_UNORM");

        const UINT idx = sc->GetCurrentBackBufferIndex();
        if (idx >= d10.BufferCount) return fail("GetCurrentBackBufferIndex >= BufferCount (10-bit)");
        ID3D12Resource* bb = nullptr;
        if (FAILED(sc->GetBuffer(idx, IID_PPV_ARGS(&bb))) || !bb) return fail("GetBuffer (10-bit)");
        const D3D12_RESOURCE_DESC bb_desc = bb->GetDesc();
        if (bb_desc.Width != 37 || bb_desc.Height != 19 || bb_desc.Format != DXGI_FORMAT_R10G10B10A2_UNORM) {
            bb->Release();
            return fail("backbuffer GetDesc did not preserve 37x19 R10G10B10A2_UNORM");
        }

        const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        dev->CreateRenderTargetView(bb, nullptr, rtv);

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT64 total_bytes = 0;
        dev->GetCopyableFootprints(&bb_desc, 0, 1, 0, &fp, nullptr, nullptr, &total_bytes);

        D3D12_HEAP_PROPERTIES rb_hp{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC rb_desc{};
        rb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rb_desc.Width = total_bytes;
        rb_desc.Height = 1;
        rb_desc.DepthOrArraySize = 1;
        rb_desc.MipLevels = 1;
        rb_desc.SampleDesc.Count = 1;
        rb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* rb = nullptr;
        if (FAILED(dev->CreateCommittedResource(&rb_hp, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb))) ||
            !rb) {
            bb->Release();
            return fail("CreateCommittedResource readback (10-bit)");
        }

        alloc->Reset();
        cl->Reset(alloc, nullptr);
        D3D12_RESOURCE_BARRIER b0{};
        b0.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b0.Transition = {bb, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_PRESENT,
                         D3D12_RESOURCE_STATE_RENDER_TARGET};
        cl->ResourceBarrier(1, &b0);

        const float band0[4] = {0.0f, 0.75f, 0.25f, 1.0f};
        const float band1[4] = {1.0f / 1024.0f, 0.75f, 0.25f, 1.0f};
        const float band2[4] = {1.0f / 128.0f, 0.75f, 0.25f, 1.0f};
        const D3D12_RECT r0{0, 0, 12, 19};
        const D3D12_RECT r1{12, 0, 24, 19};
        const D3D12_RECT r2{24, 0, 37, 19};
        cl->ClearRenderTargetView(rtv, band0, 1, &r0);
        cl->ClearRenderTargetView(rtv, band1, 1, &r1);
        cl->ClearRenderTargetView(rtv, band2, 1, &r2);

        D3D12_RESOURCE_BARRIER b1{};
        b1.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b1.Transition = {bb, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET,
                         D3D12_RESOURCE_STATE_COPY_SOURCE};
        cl->ResourceBarrier(1, &b1);

        D3D12_TEXTURE_COPY_LOCATION src_loc{bb, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
        D3D12_TEXTURE_COPY_LOCATION dst_loc{rb, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst_loc.PlacedFootprint = fp;
        cl->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);

        D3D12_RESOURCE_BARRIER b2{};
        b2.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b2.Transition = {bb, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_SOURCE,
                         D3D12_RESOURCE_STATE_PRESENT};
        cl->ResourceBarrier(1, &b2);
        cl->Close();

        ID3D12CommandList* lists[1] = {cl};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, ++fence_val);
        if (!wait_fence(fence, fence_val)) {
            rb->Release();
            bb->Release();
            return fail("wait_fence (10-bit)");
        }

        void* mapped = nullptr;
        if (FAILED(rb->Map(0, nullptr, &mapped)) || !mapped) {
            rb->Release();
            bb->Release();
            return fail("readback Map (10-bit)");
        }
        const auto* base = static_cast<const std::uint8_t*>(mapped);
        bool mismatch = false;
        std::uint32_t bad_word = 0, want_word = 0;
        for (UINT y = 0; y < 19 && !mismatch; ++y) {
            const auto* row = reinterpret_cast<const std::uint32_t*>(base + static_cast<std::size_t>(y) * fp.Footprint.RowPitch);
            for (UINT x = 0; x < 37; ++x) {
                const std::uint32_t want = (x < 12) ? 0xd00bfc00u : ((x < 24) ? 0xd00bfc01u : 0xd00bfc08u);
                if (row[x] != want) {
                    mismatch = true;
                    bad_word = row[x];
                    want_word = want;
                    break;
                }
            }
        }
        rb->Unmap(0, nullptr);
        rb->Release();
        bb->Release();
        if (mismatch) {
            std::fprintf(stderr, "FAIL: 10-bit backbuffer word 0x%08x != expected 0x%08x\n", bad_word, want_word);
            return 1;
        }
        if (FAILED(sc->Present(0, 0))) return fail("Present (10-bit)");
    }

    // IDXGISwapChain1::GetBackgroundColor must return what its setter
    // stored, with transparent black as the default.
    {
        DXGI_RGBA background{};
        if (FAILED(sc->GetBackgroundColor(&background)) || background.r != 0 || background.g != 0 ||
            background.b != 0 || background.a != 0)
            return fail("GetBackgroundColor default should be transparent black");
        const DXGI_RGBA wanted{0.25f, 0.5f, 0.75f, 1.0f};
        if (FAILED(sc->SetBackgroundColor(&wanted)) || FAILED(sc->GetBackgroundColor(&background)) ||
            background.r != wanted.r || background.g != wanted.g || background.b != wanted.b ||
            background.a != wanted.a)
            return fail("SetBackgroundColor/GetBackgroundColor did not round-trip");
        if (sc->SetBackgroundColor(nullptr) != E_INVALIDARG || sc->GetBackgroundColor(nullptr) != E_INVALIDARG)
            return fail("Get/SetBackgroundColor(nullptr) should return E_INVALIDARG");

        // A matrix transform is for composition swapchains only: host DXGI answers DXGI_ERROR_INVALID_CALL to both
        // methods on this HWND swapchain, with a matrix and with null (Windows 11, RTX 5060 Ti, driver 621.14).
        DXGI_MATRIX_3X2_F matrix{};
        const DXGI_MATRIX_3X2_F wanted_matrix{2, 0, 0, 3, 4, 5};
        if (sc->SetMatrixTransform(&wanted_matrix) != DXGI_ERROR_INVALID_CALL ||
            sc->GetMatrixTransform(&matrix) != DXGI_ERROR_INVALID_CALL ||
            sc->SetMatrixTransform(nullptr) != DXGI_ERROR_INVALID_CALL ||
            sc->GetMatrixTransform(nullptr) != DXGI_ERROR_INVALID_CALL)
            return fail("Set/GetMatrixTransform on an HWND swapchain should be DXGI_ERROR_INVALID_CALL, as on the host");

        DXGI_MODE_ROTATION rotation = DXGI_MODE_ROTATION_ROTATE180;
        if (FAILED(sc->GetRotation(&rotation)) || rotation != DXGI_MODE_ROTATION_IDENTITY)
            return fail("GetRotation default should be identity");
        if (FAILED(sc->SetRotation(DXGI_MODE_ROTATION_ROTATE90)) || FAILED(sc->GetRotation(&rotation)) ||
            rotation != DXGI_MODE_ROTATION_ROTATE90)
            return fail("SetRotation/GetRotation did not round-trip");
        if (sc->GetRotation(nullptr) != E_INVALIDARG)
            return fail("GetRotation(nullptr) should return E_INVALIDARG");
    }

    constexpr DXGI_FORMAT invalid_formats[] = {
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R32_UINT,
    };
    {
        DXGI_SWAP_CHAIN_DESC1 before{}, after{};
        if (FAILED(sc->GetDesc1(&before))) return fail("GetDesc1 before invalid formats");
        const UINT node_masks[2] = {1, 1};
        IUnknown* const present_queues[2] = {queue, queue};
        for (DXGI_FORMAT format : invalid_formats) {
            if (sc->ResizeBuffers(2, 64, 64, format, 0) != E_INVALIDARG ||
                sc->ResizeBuffers1(2, 64, 64, format, 0, node_masks, present_queues) != E_INVALIDARG)
                return fail("ResizeBuffers and ResizeBuffers1 should reject invalid flip-model formats");
        }
        if (FAILED(sc->GetDesc1(&after)) || after.Format != before.Format || after.BufferCount != before.BufferCount ||
            after.Width != before.Width || after.Height != before.Height)
            return fail("Invalid format changed the existing swapchain");
        ID3D12Resource* buffer = nullptr;
        if (FAILED(sc->GetBuffer(sc->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&buffer))) || !buffer)
            return fail("GetBuffer after invalid swapchain formats");
        buffer->Release();
    }

    // Measured on host DXGI (RTX 5070 and WARP): DXGI_PRESENT_TEST returns S_OK without presenting, so the back
    // buffer index and last present count stay; a sync interval above 4 is DXGI_ERROR_INVALID_CALL and presents nothing.
    {
        const UINT before = sc->GetCurrentBackBufferIndex();
        UINT count_before = 0;
        if (FAILED(sc->GetLastPresentCount(&count_before)) || count_before == 0)
            return fail("GetLastPresentCount should be non-zero after earlier Presents");
        if (sc->Present(0, DXGI_PRESENT_TEST) != S_OK || sc->GetCurrentBackBufferIndex() != before)
            return fail("Present(0, DXGI_PRESENT_TEST) presented a frame");
        UINT count_after_test = 0;
        if (FAILED(sc->GetLastPresentCount(&count_after_test)) || count_after_test != count_before)
            return fail("DXGI_PRESENT_TEST should not increment GetLastPresentCount");
        if (sc->Present(5, 0) != DXGI_ERROR_INVALID_CALL || sc->GetCurrentBackBufferIndex() != before)
            return fail("Present(5, 0) should be DXGI_ERROR_INVALID_CALL and present nothing");
        if (sc->GetLastPresentCount(nullptr) != E_INVALIDARG)
            return fail("GetLastPresentCount(nullptr) should return E_INVALIDARG");

        if (!check_statistics_disjoint(sc)) return fail("post-resize frame-statistics disjoint notification");
        DXGI_FRAME_STATISTICS stats{};
        if (FAILED(sc->GetFrameStatistics(&stats)) || stats.PresentCount != count_before ||
            stats.PresentRefreshCount != count_before || stats.SyncRefreshCount != count_before ||
            stats.SyncQPCTime.QuadPart == 0)
            return fail("GetFrameStatistics values");
        if (sc->GetFrameStatistics(nullptr) != DXGI_ERROR_INVALID_CALL)
            return fail("GetFrameStatistics(nullptr) should return DXGI_ERROR_INVALID_CALL");
    }

    // SwapChain::GetParent and non-waitable SetMaximumFrameLatency/GetMaximumFrameLatency (host DXGI returns
    // DXGI_ERROR_INVALID_CALL and leaves the output untouched when DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT
    // was not set).
    {
        IDXGIFactory4* parent = nullptr;
        if (FAILED(sc->GetParent(IID_PPV_ARGS(&parent))) || parent != factory)
            return fail("SwapChain::GetParent did not return the parent factory");
        parent->Release();
        UINT sentinel = 0xdeadbeefu;
        if (sc->GetMaximumFrameLatency(&sentinel) != DXGI_ERROR_INVALID_CALL || sentinel != 0xdeadbeefu)
            return fail("Non-waitable GetMaximumFrameLatency should return DXGI_ERROR_INVALID_CALL without writing output");
        if (sc->SetMaximumFrameLatency(3) != DXGI_ERROR_INVALID_CALL ||
            sc->SetMaximumFrameLatency(1) != DXGI_ERROR_INVALID_CALL)
            return fail("Non-waitable SetMaximumFrameLatency should return DXGI_ERROR_INVALID_CALL");
    }

    // IDXGIObject private data on the swapchain, the factory and an adapter.
    {
        if (const char* step = check_private_data(sc)) return fail(step);
        if (const char* step = check_private_data(factory)) return fail(step);
        IDXGIAdapter1* pd_adapter = nullptr;
        if (FAILED(factory->EnumAdapters1(0, &pd_adapter)) || !pd_adapter) return fail("EnumAdapters1(0) for private data");
        if (const char* step = check_private_data(pd_adapter)) return fail(step);
        pd_adapter->Release();
    }

    // IDXGIFactory4 adapter enumeration, IDXGIAdapter3 descriptors, EnumAdapterByLuid, and QueryVideoMemoryInfo.
    {
        IDXGIAdapter1* adapter1 = nullptr;
        if (FAILED(factory->EnumAdapters1(0, &adapter1)) || !adapter1)
            return fail("EnumAdapters1(0) failed");
        IDXGIAdapter1* missing = nullptr;
        if (factory->EnumAdapters1(1, &missing) != DXGI_ERROR_NOT_FOUND || missing != nullptr)
            return fail("EnumAdapters1(1) should return DXGI_ERROR_NOT_FOUND");

        IDXGIAdapter* adapter0 = nullptr;
        if (FAILED(factory->EnumAdapters(0, &adapter0)) || !adapter0)
            return fail("EnumAdapters(0) failed");
        adapter0->Release();

        IDXGIFactory4* adapter_parent = nullptr;
        if (FAILED(adapter1->GetParent(IID_PPV_ARGS(&adapter_parent))) || adapter_parent != factory)
            return fail("Adapter::GetParent did not return the parent factory");
        adapter_parent->Release();

        IDXGIAdapter3* adapter3 = nullptr;
        if (FAILED(adapter1->QueryInterface(IID_PPV_ARGS(&adapter3))) || !adapter3)
            return fail("QueryInterface(IDXGIAdapter3) failed");

        DXGI_ADAPTER_DESC desc0{};
        DXGI_ADAPTER_DESC1 desc1{};
        DXGI_ADAPTER_DESC2 desc2{};
        if (FAILED(adapter3->GetDesc(&desc0)) || FAILED(adapter3->GetDesc1(&desc1)) || FAILED(adapter3->GetDesc2(&desc2)))
            return fail("Adapter GetDesc/GetDesc1/GetDesc2 failed");
        if (desc1.Description[0] == L'\0' || desc1.VendorId != chosen.vendorID || desc1.DeviceId != chosen.deviceID)
            return fail("Adapter desc did not match Vulkan physical device");
        if (desc1.DedicatedVideoMemory + desc1.SharedSystemMemory == 0)
            return fail("Adapter reported zero total memory");
        if (const auto get_mem = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
                gipa(instance, "vkGetPhysicalDeviceMemoryProperties"))) {
            VkPhysicalDeviceMemoryProperties mem{};
            get_mem(physical, &mem);
            VkDeviceSize max_local = 0;
            for (std::uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
                if ((mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && mem.memoryHeaps[i].size > max_local)
                    max_local = mem.memoryHeaps[i].size;
            }
            if (desc1.DedicatedVideoMemory != static_cast<SIZE_T>(max_local))
                return fail("DedicatedVideoMemory should match the primary DEVICE_LOCAL heap (without double-counting BAR1)");
        }
        if (desc0.VendorId != desc1.VendorId || desc2.DeviceId != desc1.DeviceId)
            return fail("Adapter desc mismatch between GetDesc/GetDesc1/GetDesc2");

        // CheckInterfaceSupport(IDXGIDevice) reports the adapter's user-mode driver version, as host DXGI does for
        // the adapter with the same LUID (RTX 5060 Ti / 621.14: 0x0020000000100842), not 0.
        {
            LARGE_INTEGER ours{};
            if (FAILED(adapter3->CheckInterfaceSupport(__uuidof(IDXGIDevice), &ours)))
                return fail("Adapter CheckInterfaceSupport(IDXGIDevice) failed");
            IDXGIFactory4* host_factory = nullptr;
            IDXGIAdapter1* host_adapter = nullptr;
            // The host dxgi.dll, loaded at run time: this test doesn't link it, and a Wine run without it skips.
            HMODULE host_dxgi = LoadLibraryW(L"dxgi.dll");
            const auto create_factory = host_dxgi ? reinterpret_cast<decltype(&CreateDXGIFactory1)>(
                GetProcAddress(host_dxgi, "CreateDXGIFactory1")) : nullptr;
            if (create_factory && SUCCEEDED(create_factory(IID_PPV_ARGS(&host_factory))) && host_factory &&
                SUCCEEDED(host_factory->EnumAdapterByLuid(desc1.AdapterLuid, IID_PPV_ARGS(&host_adapter))) &&
                host_adapter) {
                LARGE_INTEGER host{};
                const HRESULT hr = host_adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &host);
                host_adapter->Release();
                host_factory->Release();
                if (SUCCEEDED(hr) && host.QuadPart != ours.QuadPart) {
                    std::fprintf(stderr, "FAIL: UMD version 0x%016llx, host DXGI 0x%016llx\n",
                                 static_cast<unsigned long long>(ours.QuadPart),
                                 static_cast<unsigned long long>(host.QuadPart));
                    return 1;
                }
            } else if (host_factory) {
                host_factory->Release();
            }
        }

        // EnumAdapterByLuid with matching LUID succeeds; bogus LUID returns DXGI_ERROR_NOT_FOUND.
        IDXGIAdapter3* by_luid = nullptr;
        if (FAILED(factory->EnumAdapterByLuid(desc1.AdapterLuid, IID_PPV_ARGS(&by_luid))) || !by_luid)
            return fail("EnumAdapterByLuid(matching) failed");
        by_luid->Release();
        LUID bogus = desc1.AdapterLuid;
        bogus.LowPart ^= 0x5a5a5a5au;
        bogus.HighPart ^= 0x13572468;
        if (factory->EnumAdapterByLuid(bogus, IID_PPV_ARGS(&by_luid)) != DXGI_ERROR_NOT_FOUND || by_luid != nullptr)
            return fail("EnumAdapterByLuid(bogus) should return DXGI_ERROR_NOT_FOUND");

        // QueryVideoMemoryInfo + SetVideoMemoryReservation (shared across distinct handles for the same LUID).
        IDXGIAdapter3* by_luid_b = nullptr;
        if (FAILED(factory->EnumAdapterByLuid(desc1.AdapterLuid, IID_PPV_ARGS(&by_luid_b))) || !by_luid_b)
            return fail("EnumAdapterByLuid(second handle) failed");
        DXGI_QUERY_VIDEO_MEMORY_INFO local_info{};
        if (FAILED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info)) ||
            local_info.Budget == 0)
            return fail("QueryVideoMemoryInfo(LOCAL) failed or reported 0 budget");
        if (FAILED(adapter3->SetVideoMemoryReservation(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, 64u * 1024u * 1024u)))
            return fail("SetVideoMemoryReservation(LOCAL) failed");
        if (FAILED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info)) ||
            local_info.CurrentReservation != 64u * 1024u * 1024u)
            return fail("SetVideoMemoryReservation did not reflect on handle A");
        DXGI_QUERY_VIDEO_MEMORY_INFO local_info_b{};
        if (FAILED(by_luid_b->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info_b)) ||
            local_info_b.CurrentReservation != 64u * 1024u * 1024u)
            return fail("SetVideoMemoryReservation(LOCAL) did not reflect across second IDXGIAdapter3 handle");

        DXGI_QUERY_VIDEO_MEMORY_INFO nonlocal_info{};
        if (FAILED(adapter3->SetVideoMemoryReservation(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, 32u * 1024u * 1024u)) ||
            FAILED(by_luid_b->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal_info)) ||
            nonlocal_info.CurrentReservation != 32u * 1024u * 1024u)
            return fail("SetVideoMemoryReservation(NON_LOCAL) did not reflect across second IDXGIAdapter3 handle");
        // Restore reservations to 0 and verify both handles see 0.
        if (FAILED(adapter3->SetVideoMemoryReservation(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, 0)) ||
            FAILED(adapter3->SetVideoMemoryReservation(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, 0)) ||
            FAILED(by_luid_b->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info_b)) ||
            local_info_b.CurrentReservation != 0)
            return fail("Restoring SetVideoMemoryReservation to 0 failed");
        by_luid_b->Release();

        if (adapter3->QueryVideoMemoryInfo(1, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info) != E_INVALIDARG ||
            adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, nullptr) != E_INVALIDARG)
            return fail("QueryVideoMemoryInfo invalid args should return E_INVALIDARG");

        // Budget-change notification (Microsoft: IDXGIAdapter3::RegisterVideoMemoryBudgetChangeNotificationEvent
        // returns a success code and a cookie for UnregisterVideoMemoryBudgetChangeNotification).
        HANDLE budget_ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        DWORD budget_cookie = 0, budget_cookie2 = 0;
        if (adapter3->RegisterVideoMemoryBudgetChangeNotificationEvent(budget_ev, &budget_cookie) != S_OK ||
            budget_cookie == 0)
            return fail("RegisterVideoMemoryBudgetChangeNotificationEvent should succeed with a cookie");
        if (adapter3->RegisterVideoMemoryBudgetChangeNotificationEvent(budget_ev, &budget_cookie2) != S_OK ||
            budget_cookie2 == 0 || budget_cookie2 == budget_cookie)
            return fail("a second budget-change registration should get its own cookie");
        adapter3->UnregisterVideoMemoryBudgetChangeNotification(budget_cookie);
        adapter3->UnregisterVideoMemoryBudgetChangeNotification(budget_cookie2);
        CloseHandle(budget_ev);

        // IDXGIAdapter::EnumOutputs, IDXGIOutput..IDXGIOutput6, and host DXGI comparison (D3D-313).
        if (adapter1->EnumOutputs(0, nullptr) != E_INVALIDARG)
            return fail("EnumOutputs(0, nullptr) should return E_INVALIDARG");

        UINT our_output_count = 0;
        for (UINT oi = 0;; ++oi) {
            IDXGIOutput* out_i = nullptr;
            const HRESULT ehr = adapter1->EnumOutputs(oi, &out_i);
            if (ehr == DXGI_ERROR_NOT_FOUND) {
                if (out_i != nullptr) return fail("EnumOutputs out-of-range left non-null pointer");
                break;
            }
            if (FAILED(ehr) || !out_i) return fail("EnumOutputs failed unexpectedly");
            ++our_output_count;
            out_i->Release();
        }
        if (our_output_count == 0) return fail("EnumOutputs(0) should expose at least one output");

        IDXGIOutput* out0 = nullptr;
        if (FAILED(adapter1->EnumOutputs(0, &out0)) || !out0) return fail("EnumOutputs(0) failed");

        IDXGIAdapter* out_parent = nullptr;
        if (FAILED(out0->GetParent(IID_PPV_ARGS(&out_parent))) || out_parent != static_cast<IDXGIAdapter*>(adapter1))
            return fail("Output::GetParent did not return parent adapter");
        out_parent->Release();

        if (out0->GetDesc(nullptr) != E_INVALIDARG) return fail("Output::GetDesc(nullptr) should return E_INVALIDARG");
        DXGI_OUTPUT_DESC od{};
        if (FAILED(out0->GetDesc(&od)) || od.DeviceName[0] == L'\0' || !od.AttachedToDesktop ||
            od.DesktopCoordinates.right <= od.DesktopCoordinates.left ||
            od.DesktopCoordinates.bottom <= od.DesktopCoordinates.top)
            return fail("Output::GetDesc reported invalid fields");

        IDXGIOutput6* out6 = nullptr;
        if (FAILED(out0->QueryInterface(IID_PPV_ARGS(&out6))) || !out6)
            return fail("QueryInterface(IDXGIOutput6) failed");
        if (out6->GetDesc1(nullptr) != E_INVALIDARG) return fail("Output6::GetDesc1(nullptr) should return E_INVALIDARG");
        DXGI_OUTPUT_DESC1 od1{};
        if (FAILED(out6->GetDesc1(&od1)) || std::wcscmp(od1.DeviceName, od.DeviceName) != 0 ||
            od1.BitsPerColor == 0 || od1.Monitor != od.Monitor)
            return fail("Output6::GetDesc1 mismatch");

        // Display mode enumeration (2-pass + MORE_DATA + unsupported/unknown formats returning count 0).
        if (out0->GetDisplayModeList(DXGI_FORMAT_B8G8R8A8_UNORM, 0, nullptr, nullptr) != DXGI_ERROR_INVALID_CALL)
            return fail("GetDisplayModeList(nullptr count) should return DXGI_ERROR_INVALID_CALL");
        UINT unk_modes = 99;
        if (FAILED(out0->GetDisplayModeList(DXGI_FORMAT_UNKNOWN, 0, &unk_modes, nullptr)) || unk_modes != 0)
            return fail("GetDisplayModeList(UNKNOWN) should succeed with 0 modes");
        UINT r32_modes = 99;
        if (FAILED(out0->GetDisplayModeList(DXGI_FORMAT_R32_UINT, 0, &r32_modes, nullptr)) || r32_modes != 0)
            return fail("GetDisplayModeList(R32_UINT) should succeed with 0 modes");

        UINT mode_count = 0;
        if (FAILED(out0->GetDisplayModeList(DXGI_FORMAT_B8G8R8A8_UNORM, 0, &mode_count, nullptr)) || mode_count == 0)
            return fail("GetDisplayModeList(BGRA8) count failed");
        std::vector<DXGI_MODE_DESC> modes(mode_count);
        UINT filled = mode_count;
        if (FAILED(out0->GetDisplayModeList(DXGI_FORMAT_B8G8R8A8_UNORM, 0, &filled, modes.data())) ||
            filled != mode_count)
            return fail("GetDisplayModeList(BGRA8) fill failed");
        if (mode_count > 1) {
            UINT small_count = 1;
            DXGI_MODE_DESC one{};
            if (out0->GetDisplayModeList(DXGI_FORMAT_B8G8R8A8_UNORM, 0, &small_count, &one) != DXGI_ERROR_MORE_DATA ||
                small_count != 1)
                return fail("GetDisplayModeList short buffer should return DXGI_ERROR_MORE_DATA and keep copied count");
            UINT zero_count = 0;
            if (out0->GetDisplayModeList(DXGI_FORMAT_B8G8R8A8_UNORM, 0, &zero_count, &one) != DXGI_ERROR_MORE_DATA ||
                zero_count != 0)
                return fail("GetDisplayModeList zero buffer should return DXGI_ERROR_MORE_DATA and keep 0");
        }

        UINT mode1_count = 0;
        if (FAILED(out6->GetDisplayModeList1(DXGI_FORMAT_R8G8B8A8_UNORM, 0, &mode1_count, nullptr)) ||
            mode1_count != mode_count)
            return fail("GetDisplayModeList1(RGBA8) count mismatch");
        std::vector<DXGI_MODE_DESC1> modes1(mode1_count);
        if (FAILED(out6->GetDisplayModeList1(DXGI_FORMAT_R8G8B8A8_UNORM, 0, &mode1_count, modes1.data())))
            return fail("GetDisplayModeList1(RGBA8) fill failed");

        // FindClosestMatchingMode / FindClosestMatchingMode1 error & match paths.
        DXGI_MODE_DESC match_req{}, matched{};
        if (out0->FindClosestMatchingMode(nullptr, &matched, nullptr) != DXGI_ERROR_INVALID_CALL ||
            out0->FindClosestMatchingMode(&match_req, nullptr, nullptr) != DXGI_ERROR_INVALID_CALL ||
            out0->FindClosestMatchingMode(&match_req, &matched, nullptr) != DXGI_ERROR_INVALID_CALL)
            return fail("FindClosestMatchingMode invalid arg handling mismatch");
        match_req.Format = DXGI_FORMAT_R32_UINT;
        match_req.Width = 1920;
        match_req.Height = 1080;
        if (out0->FindClosestMatchingMode(&match_req, &matched, nullptr) != DXGI_ERROR_NOT_FOUND)
            return fail("FindClosestMatchingMode(R32_UINT) should return DXGI_ERROR_NOT_FOUND");
        match_req.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        match_req.Width = 1280;
        match_req.Height = 720;
        if (FAILED(out0->FindClosestMatchingMode(&match_req, &matched, nullptr)) || matched.Width == 0 ||
            matched.Height == 0 || matched.RefreshRate.Numerator == 0 || matched.Format != DXGI_FORMAT_R8G8B8A8_UNORM)
            return fail("FindClosestMatchingMode(RGBA8 1280x720) failed");

        DXGI_MODE_DESC1 req1{}, matched1{};
        req1.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        if (FAILED(out6->FindClosestMatchingMode1(&req1, &matched1, nullptr)) || matched1.Width == 0 ||
            matched1.Height == 0)
            return fail("FindClosestMatchingMode1 wildcard size failed");

        // VBlank, frame statistics, overlay/hw-composition, and duplication checks.
        if (FAILED(out0->WaitForVBlank())) return fail("WaitForVBlank failed");
        DXGI_FRAME_STATISTICS out_stats{};
        if (out0->GetFrameStatistics(nullptr) != DXGI_ERROR_INVALID_CALL ||
            FAILED(out0->GetFrameStatistics(&out_stats)) || out_stats.SyncQPCTime.QuadPart == 0)
            return fail("Output::GetFrameStatistics failed");
        UINT comp_flags = 99;
        if (out6->CheckHardwareCompositionSupport(nullptr) != E_INVALIDARG ||
            FAILED(out6->CheckHardwareCompositionSupport(&comp_flags)) ||
            out6->CheckOverlaySupport(DXGI_FORMAT_R8G8B8A8_UNORM, nullptr, &comp_flags) != E_INVALIDARG ||
            out6->CheckOverlaySupport(DXGI_FORMAT_R8G8B8A8_UNORM, dev, nullptr) != E_INVALIDARG ||
            FAILED(out6->CheckOverlaySupport(DXGI_FORMAT_R8G8B8A8_UNORM, dev, &comp_flags)) ||
            out6->SupportsOverlays() != FALSE)
            return fail("Output overlay/hardware composition checks failed");
        IDXGIOutputDuplication* dup = nullptr;
        if (out6->DuplicateOutput(nullptr, &dup) != E_INVALIDARG ||
            out6->DuplicateOutput(dev, nullptr) != E_INVALIDARG ||
            out6->DuplicateOutput(dev, &dup) != DXGI_ERROR_UNSUPPORTED || dup != nullptr)
            return fail("Output::DuplicateOutput checks failed");

        // Compare directly against Windows host dxgi.dll when a host adapter has attached outputs.
        if (const HMODULE host_dxgi = LoadLibraryW(L"dxgi.dll")) {
            using CreateFactory1Fn = HRESULT(WINAPI*)(REFIID, void**);
            if (const auto create_host =
                    reinterpret_cast<CreateFactory1Fn>(GetProcAddress(host_dxgi, "CreateDXGIFactory1"))) {
                IDXGIFactory4* hfac = nullptr;
                if (SUCCEEDED(create_host(IID_PPV_ARGS(&hfac))) && hfac) {
                    IDXGIAdapter1* hadp = nullptr;
                    if (SUCCEEDED(hfac->EnumAdapterByLuid(desc1.AdapterLuid, IID_PPV_ARGS(&hadp))) && hadp) {
                        if (hadp->EnumOutputs(0, nullptr) != adapter1->EnumOutputs(0, nullptr)) {
                            hadp->Release();
                            hfac->Release();
                            FreeLibrary(host_dxgi);
                            return fail("EnumOutputs(0, nullptr) did not match host dxgi.dll");
                        }
                        IDXGIOutput* hout0 = nullptr;
                        if (SUCCEEDED(hadp->EnumOutputs(0, &hout0)) && hout0) {
                            DXGI_OUTPUT_DESC hod{};
                            if (SUCCEEDED(hout0->GetDesc(&hod))) {
                                // Match against any enumerated output (hybrid laptops can place output 0 on another GPU).
                                bool found_name = false;
                                for (UINT oi = 0; oi < our_output_count && !found_name; ++oi) {
                                    IDXGIOutput* cur = nullptr;
                                    if (SUCCEEDED(adapter1->EnumOutputs(oi, &cur)) && cur) {
                                        DXGI_OUTPUT_DESC cur_desc{};
                                        if (SUCCEEDED(cur->GetDesc(&cur_desc)) &&
                                            std::wcscmp(hod.DeviceName, cur_desc.DeviceName) == 0)
                                            found_name = true;
                                        cur->Release();
                                    }
                                }
                                if (!found_name) {
                                    hout0->Release();
                                    hadp->Release();
                                    hfac->Release();
                                    FreeLibrary(host_dxgi);
                                    return fail("Host output 0 DeviceName not found in EnumOutputs");
                                }
                            }
                            UINT host_modes = 0;
                            if (SUCCEEDED(hout0->GetDisplayModeList(DXGI_FORMAT_UNKNOWN, 0, &host_modes, nullptr)) &&
                                host_modes != unk_modes) {
                                hout0->Release();
                                hadp->Release();
                                hfac->Release();
                                FreeLibrary(host_dxgi);
                                return fail("GetDisplayModeList(UNKNOWN) count did not match host dxgi.dll");
                            }
                            DXGI_MODE_DESC host_req{}, host_matched{};
                            if (hout0->FindClosestMatchingMode(&host_req, &host_matched, nullptr) !=
                                out0->FindClosestMatchingMode(&host_req, &matched, nullptr)) {
                                hout0->Release();
                                hadp->Release();
                                hfac->Release();
                                FreeLibrary(host_dxgi);
                                return fail("FindClosestMatchingMode(empty) did not match host dxgi.dll");
                            }
                            hout0->Release();
                        }
                        hadp->Release();
                    }
                    hfac->Release();
                }
            }
            FreeLibrary(host_dxgi);
        }

        // SwapChain::GetContainingOutput validation and monitor lookup.
        if (sc->GetContainingOutput(nullptr) != DXGI_ERROR_INVALID_CALL)
            return fail("SwapChain::GetContainingOutput(nullptr) should return DXGI_ERROR_INVALID_CALL");
        IDXGIOutput* containing = nullptr;
        if (FAILED(sc->GetContainingOutput(&containing)) || !containing)
            return fail("SwapChain::GetContainingOutput failed");
        DXGI_OUTPUT_DESC cod{};
        if (FAILED(containing->GetDesc(&cod)) || cod.DeviceName[0] == L'\0')
            return fail("SwapChain::GetContainingOutput returned invalid output desc");
        containing->Release();

        out6->Release();
        out0->Release();
        adapter3->Release();
        adapter1->Release();
    }

    // Host DXGI refuses ResizeBuffers with DXGI_ERROR_INVALID_CALL while the app still holds a back buffer,
    // and keeps the old buffers.
    {
        ID3D12Resource* held = nullptr;
        if (FAILED(sc->GetBuffer(0, IID_PPV_ARGS(&held))) || !held) return fail("GetBuffer (held)");
        const HRESULT hr = sc->ResizeBuffers(2, 64, 64, DXGI_FORMAT_UNKNOWN, 0);
        DXGI_SWAP_CHAIN_DESC1 d{};
        sc->GetDesc1(&d);
        held->Release();
        if (hr != DXGI_ERROR_INVALID_CALL || d.Width != 37 || d.Height != 19)
            return fail("ResizeBuffers with a held back buffer should be DXGI_ERROR_INVALID_CALL and keep 37x19");
    }

    // Zero width or height means the window's client area, in ResizeBuffers (the usual WM_SIZE call) and at
    // creation; host DXGI: ResizeBuffers(0, 0, 0) after resizing the window gives the new client size.
    {
        DXGI_SWAP_CHAIN_DESC1 before_zero{};
        if (FAILED(sc->GetDesc1(&before_zero))) return fail("GetDesc1 before zero-count resize");
        RECT wr{0, 0, 300, 200};
        AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
        SetWindowPos(hwnd, nullptr, 0, 0, wr.right - wr.left, wr.bottom - wr.top, SWP_NOMOVE | SWP_NOZORDER);
        RECT cr{};
        GetClientRect(hwnd, &cr);
        if (FAILED(sc->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, 0))) return fail("ResizeBuffers(0, 0, 0)");
        DXGI_SWAP_CHAIN_DESC1 d{};
        if (FAILED(sc->GetDesc1(&d)) || d.BufferCount != before_zero.BufferCount)
            return fail("ResizeBuffers(0, 0, 0) changed the buffer count");
        if (d.Width != UINT(cr.right) || d.Height != UINT(cr.bottom)) {
            std::fprintf(stderr, "FAIL: ResizeBuffers(0, 0, 0) gave %ux%u, client area is %ldx%ld\n", d.Width, d.Height,
                         cr.right, cr.bottom);
            return 1;
        }

        HWND hwnd2 = create_test_window(200, 150);
        if (!hwnd2) return fail("CreateWindowExW (second)");
        GetClientRect(hwnd2, &cr);
        const auto rejects_creation_format = [&](DXGI_FORMAT format) {
            DXGI_SWAP_CHAIN_DESC1 invalid = scd;
            invalid.Format = format;
            IDXGISwapChain1* rejected = nullptr;
            const HRESULT hr = factory->CreateSwapChainForHwnd(queue, hwnd2, &invalid, nullptr, nullptr, &rejected);
            if (rejected) {
                rejected->Release();
                return false;
            }
            return hr == DXGI_ERROR_INVALID_CALL;
        };
        if (!rejects_creation_format(DXGI_FORMAT_UNKNOWN))
            return fail("CreateSwapChainForHwnd accepted DXGI_FORMAT_UNKNOWN");
        for (DXGI_FORMAT format : invalid_formats)
            if (!rejects_creation_format(format)) return fail("CreateSwapChainForHwnd invalid flip-model format");
        DXGI_SWAP_CHAIN_DESC1 zd = scd;
        zd.Width = 0;
        zd.Height = 0;
        IDXGISwapChain1* zsc = nullptr;
        if (FAILED(factory->CreateSwapChainForHwnd(queue, hwnd2, &zd, nullptr, nullptr, &zsc)) || !zsc)
            return fail("CreateSwapChainForHwnd 0x0");
        zsc->GetDesc1(&d);
        zsc->Release();
        DestroyWindow(hwnd2);
        if (d.Width != UINT(cr.right) || d.Height != UINT(cr.bottom)) {
            std::fprintf(stderr, "FAIL: 0x0 swapchain is %ux%u, client area is %ldx%ld\n", d.Width, d.Height, cr.right,
                         cr.bottom);
            return 1;
        }
    }

    // ResizeTarget on a windowed swapchain resizes the target's client area to the mode; the back buffers keep
    // their size until ResizeBuffers(0, 0, 0) takes the new client area.
    {
        // Host-measured on Windows 11 / RX 7900 XTX: a null target is DXGI_ERROR_INVALID_CALL, and 0x0 is accepted
        // (the window's minimum tracking size decides the resulting client area).
        if (sc->ResizeTarget(nullptr) != DXGI_ERROR_INVALID_CALL)
            return fail("ResizeTarget(nullptr) should be DXGI_ERROR_INVALID_CALL");
        DXGI_MODE_DESC mode{};
        mode.Width = 234;
        mode.Height = 156;
        mode.RefreshRate.Numerator = 60;
        mode.RefreshRate.Denominator = 1;
        mode.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        if (FAILED(sc->ResizeTarget(&mode))) return fail("ResizeTarget");
        RECT cr{};
        GetClientRect(hwnd, &cr);
        if (cr.right != 234 || cr.bottom != 156) {
            std::fprintf(stderr, "FAIL: ResizeTarget gave a %ldx%ld client area, want 234x156\n", cr.right, cr.bottom);
            return 1;
        }
        DXGI_SWAP_CHAIN_DESC1 d{};
        if (FAILED(sc->GetDesc1(&d))) return fail("GetDesc1 after ResizeTarget");
        if (d.Width != 300 || d.Height != 200)
            return fail("ResizeTarget should leave the back buffers for ResizeBuffers");
        if (FAILED(sc->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, 0))) return fail("ResizeBuffers after ResizeTarget");
        if (FAILED(sc->GetDesc1(&d)) || d.Width != 234 || d.Height != 156) {
            std::fprintf(stderr, "FAIL: ResizeBuffers after ResizeTarget gave %ux%u, want 234x156\n", d.Width, d.Height);
            return 1;
        }
    }

    // SetSourceSize stores the source region (GetSourceSize returns it), refuses one larger than the back buffers,
    // needs both pointers on read, and ResizeBuffers resets it. All four are host-measured on Windows 11.
    {
        // The ResizeTarget block above leaves the back buffers at 234x156: start again from 300x200.
        if (FAILED(sc->ResizeBuffers(0, 300, 200, DXGI_FORMAT_UNKNOWN, 0))) return fail("ResizeBuffers to 300x200");
        UINT sw = 0, sh = 0;
        if (FAILED(sc->GetSourceSize(&sw, &sh)) || sw != 300 || sh != 200)
            return fail("GetSourceSize should start at the back-buffer size");
        if (sc->GetSourceSize(nullptr, nullptr) != DXGI_ERROR_INVALID_CALL ||
            sc->GetSourceSize(&sw, nullptr) != DXGI_ERROR_INVALID_CALL ||
            sc->GetSourceSize(nullptr, &sh) != DXGI_ERROR_INVALID_CALL)
            return fail("GetSourceSize needs both pointers");
        if (FAILED(sc->SetSourceSize(64, 32))) return fail("SetSourceSize");
        if (FAILED(sc->GetSourceSize(&sw, &sh)) || sw != 64 || sh != 32) return fail("SetSourceSize round-trip");
        if (sc->SetSourceSize(301, 32) != DXGI_ERROR_INVALID_CALL) return fail("SetSourceSize accepted a wider region");
        if (sc->SetSourceSize(64, 201) != DXGI_ERROR_INVALID_CALL) return fail("SetSourceSize accepted a taller region");
        if (FAILED(sc->GetSourceSize(&sw, &sh)) || sw != 64 || sh != 32)
            return fail("a rejected SetSourceSize should keep the stored source size");
        if (FAILED(sc->SetSourceSize(0, 0)) || FAILED(sc->GetSourceSize(&sw, &sh)) || sw != 0 || sh != 0)
            return fail("SetSourceSize(0, 0) should be accepted and stored");
        if (FAILED(sc->ResizeBuffers(2, 96, 64, DXGI_FORMAT_UNKNOWN, 0))) return fail("ResizeBuffers for source size");
        if (FAILED(sc->GetSourceSize(&sw, &sh)) || sw != 96 || sh != 64)
            return fail("ResizeBuffers should reset the source size to the new back buffers");
    }

    // IDXGIFactory5::CheckFeatureSupport, DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING, and waitable frame-latency swapchains.
    {
        IDXGIFactory5* factory5 = nullptr;
        if (FAILED(factory->QueryInterface(IID_PPV_ARGS(&factory5))) || !factory5)
            return fail("QueryInterface(IDXGIFactory5) failed");
        BOOL allow_tearing = FALSE;
        if (FAILED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow_tearing,
                                                 sizeof(allow_tearing))))
            return fail("IDXGIFactory5::CheckFeatureSupport(ALLOW_TEARING) failed");
        if (factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, nullptr, sizeof(BOOL)) != E_INVALIDARG ||
            factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow_tearing, 2) != E_INVALIDARG)
            return fail("CheckFeatureSupport should reject null data or wrong size with E_INVALIDARG");
        factory5->Release();

        // IDXGIFactory6/7 (D3D-2). Every answer below is the host dxgi.dll's (Windows 11, RTX 3070).
        IDXGIFactory6* factory6 = nullptr;
        IDXGIFactory7* factory7 = nullptr;
        if (FAILED(factory->QueryInterface(IID_PPV_ARGS(&factory6))) || !factory6 ||
            FAILED(factory->QueryInterface(IID_PPV_ARGS(&factory7))) || !factory7)
            return fail("QueryInterface(IDXGIFactory6/7) failed");
        IDXGIFactory7* created7 = nullptr;
        if (FAILED(rt::d3d12::create_dxgi_factory(dev, IID_PPV_ARGS(&created7))) || !created7)
            return fail("create_dxgi_factory(IID_IDXGIFactory7) failed");
        created7->Release();

        DXGI_ADAPTER_DESC1 enum_desc{};
        {
            IDXGIAdapter1* first = nullptr;
            if (FAILED(factory->EnumAdapters1(0, &first)) || !first || FAILED(first->GetDesc1(&enum_desc)))
                return fail("EnumAdapters1(0) for the preference comparison");
            first->Release();
        }
        // The host accepts a preference outside the enum, so -1 and 3 are checked as well.
        for (const int preference : {0, 1, 2, 3, -1}) {
            const auto pref = static_cast<DXGI_GPU_PREFERENCE>(preference);
            IDXGIAdapter1* by_pref = nullptr;
            DXGI_ADAPTER_DESC1 pref_desc{};
            if (FAILED(factory6->EnumAdapterByGpuPreference(0, pref, IID_PPV_ARGS(&by_pref))) || !by_pref ||
                FAILED(by_pref->GetDesc1(&pref_desc)))
                return fail("EnumAdapterByGpuPreference(0) failed");
            by_pref->Release();
            if (pref_desc.AdapterLuid.LowPart != enum_desc.AdapterLuid.LowPart ||
                pref_desc.AdapterLuid.HighPart != enum_desc.AdapterLuid.HighPart)
                return fail("EnumAdapterByGpuPreference(0) is not EnumAdapters1(0)'s adapter");
            void* missing = reinterpret_cast<void*>(1);
            if (factory6->EnumAdapterByGpuPreference(1, pref, __uuidof(IDXGIAdapter1), &missing) !=
                    DXGI_ERROR_NOT_FOUND || missing != nullptr)
                return fail("EnumAdapterByGpuPreference past the last adapter should be DXGI_ERROR_NOT_FOUND");
        }
        if (factory6->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_UNSPECIFIED, __uuidof(IDXGIAdapter1),
                                                 nullptr) != DXGI_ERROR_INVALID_CALL)
            return fail("EnumAdapterByGpuPreference(out = null) should be DXGI_ERROR_INVALID_CALL");
        {
            void* wrong = reinterpret_cast<void*>(1);
            if (factory6->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_UNSPECIFIED, __uuidof(IDXGIFactory),
                                                     &wrong) != E_NOINTERFACE || wrong != nullptr)
                return fail("EnumAdapterByGpuPreference with a non-adapter riid should be E_NOINTERFACE");
        }

        HANDLE changed = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        HANDLE changed2 = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!changed || !changed2) return fail("CreateEventW (adapters changed)");
        DWORD c_null = 0xDEAD, c1 = 0, c2 = 0, c3 = 0;
        if (factory7->RegisterAdaptersChangedEvent(nullptr, &c_null) != DXGI_ERROR_INVALID_CALL || c_null != 0xDEAD)
            return fail("RegisterAdaptersChangedEvent(null event) should be DXGI_ERROR_INVALID_CALL");
        if (factory7->RegisterAdaptersChangedEvent(changed, nullptr) != DXGI_ERROR_INVALID_CALL)
            return fail("RegisterAdaptersChangedEvent(null cookie) should be DXGI_ERROR_INVALID_CALL");
        if (FAILED(factory7->RegisterAdaptersChangedEvent(changed, &c1)) || c1 != 1)
            return fail("RegisterAdaptersChangedEvent should succeed with cookie 1");
        if (factory7->RegisterAdaptersChangedEvent(changed, &c2) != DXGI_ERROR_INVALID_CALL)
            return fail("RegisterAdaptersChangedEvent with an event already registered should be DXGI_ERROR_INVALID_CALL");
        if (FAILED(factory7->RegisterAdaptersChangedEvent(changed2, &c3)) || c3 == c1)
            return fail("RegisterAdaptersChangedEvent with a second event should get its own cookie");
        if (WaitForSingleObject(changed, 0) != WAIT_TIMEOUT)
            return fail("registering must not signal the adapters-changed event");
        DWORD occlusion = 0;
        if (FAILED(factory7->RegisterOcclusionStatusEvent(changed, &occlusion)))
            return fail("RegisterOcclusionStatusEvent (cookie space check)");
        if (occlusion != c1 && occlusion != c3 &&
            factory7->UnregisterAdaptersChangedEvent(occlusion) != DXGI_ERROR_NOT_FOUND)
            return fail("an occlusion cookie should not unregister an adapters-changed event");
        factory7->UnregisterOcclusionStatus(occlusion);
        if (FAILED(factory7->UnregisterAdaptersChangedEvent(c1)) ||
            factory7->UnregisterAdaptersChangedEvent(c1) != DXGI_ERROR_NOT_FOUND ||
            factory7->UnregisterAdaptersChangedEvent(0) != DXGI_ERROR_NOT_FOUND ||
            factory7->UnregisterAdaptersChangedEvent(12345) != DXGI_ERROR_NOT_FOUND)
            return fail("UnregisterAdaptersChangedEvent: a cookie unregisters once, an unknown one is NOT_FOUND");
        // After unregistering, the same event can be registered again.
        DWORD c4 = 0;
        if (FAILED(factory7->RegisterAdaptersChangedEvent(changed, &c4)) ||
            FAILED(factory7->UnregisterAdaptersChangedEvent(c4)) ||
            FAILED(factory7->UnregisterAdaptersChangedEvent(c3)))
            return fail("re-registering an unregistered event");
        CloseHandle(changed);
        CloseHandle(changed2);
        factory7->Release();
        factory6->Release();

        constexpr UINT tearing_flag = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        constexpr UINT waitable_flag = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        constexpr UINT both_flags = tearing_flag | waitable_flag;
        const UINT node_masks[2] = {1, 1};
        IUnknown* const present_queues[2] = {queue, queue};
        const auto rejects_flag_change = [&](IDXGISwapChain3* chain, UINT flags) {
            DXGI_SWAP_CHAIN_DESC1 before{}, after{};
            if (FAILED(chain->GetDesc1(&before))) return false;
            const UINT width = before.Width + 1;
            return chain->ResizeBuffers(0, width, before.Height, DXGI_FORMAT_UNKNOWN, flags) == E_INVALIDARG &&
                   chain->ResizeBuffers1(2, width, before.Height, DXGI_FORMAT_UNKNOWN, flags,
                                         node_masks, present_queues) == E_INVALIDARG &&
                   SUCCEEDED(chain->GetDesc1(&after)) && after.Flags == before.Flags &&
                   after.BufferCount == before.BufferCount && after.Width == before.Width &&
                   after.Height == before.Height;
        };

        // Non-waitable, non-tearing swapchain: GetFrameLatencyWaitableObject is null and ALLOW_TEARING is rejected.
        for (UINT flags : {waitable_flag, tearing_flag, both_flags})
            if (!rejects_flag_change(sc, flags)) return fail("Resize accepted adding a creation-only flag");
        if (sc->GetFrameLatencyWaitableObject() != nullptr)
            return fail("Non-waitable swapchain should return nullptr from GetFrameLatencyWaitableObject");
        if (sc->Present(0, DXGI_PRESENT_ALLOW_TEARING) != DXGI_ERROR_INVALID_CALL)
            return fail("Present(ALLOW_TEARING) on non-tearing swapchain should return DXGI_ERROR_INVALID_CALL");
        if (FAILED(sc->Present(0, 0))) return fail("Present after rejected resize on a plain swapchain");

        HWND hwnd3 = create_test_window(128, 96);
        if (!hwnd3) return fail("CreateWindowExW (third)");
        DXGI_SWAP_CHAIN_DESC1 wdesc = scd;
        wdesc.Width = 128;
        wdesc.Height = 96;
        wdesc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;  // also tests presenting an untouched offscreen backbuffer
        wdesc.Flags = both_flags;
        IDXGISwapChain1* wsc1 = nullptr;
        if (FAILED(factory->CreateSwapChainForHwnd(queue, hwnd3, &wdesc, nullptr, nullptr, &wsc1)) || !wsc1)
            return fail("CreateSwapChainForHwnd (tearing + waitable)");
        IDXGISwapChain3* wsc = nullptr;
        if (FAILED(wsc1->QueryInterface(IID_PPV_ARGS(&wsc))) || !wsc) return fail("QI IDXGISwapChain3 (waitable)");
        wsc1->Release();

        for (UINT flags : {0u, waitable_flag, tearing_flag})
            if (!rejects_flag_change(wsc, flags)) return fail("Resize accepted removing a creation-only flag");
        DXGI_SWAP_CHAIN_DESC1 got_desc{};
        if (FAILED(wsc->GetDesc1(&got_desc)) || got_desc.Flags != wdesc.Flags)
            return fail("Waitable swapchain GetDesc1 did not preserve Flags");
        UINT lat = 0;
        if (FAILED(wsc->GetMaximumFrameLatency(&lat)) || lat != 1)
            return fail("Waitable swapchain default GetMaximumFrameLatency should be 1");
        if (wsc->SetMaximumFrameLatency(0) != DXGI_ERROR_INVALID_CALL ||
            FAILED(wsc->GetMaximumFrameLatency(&lat)) || lat != 1)
            return fail("Waitable SetMaximumFrameLatency(0) should return INVALID_CALL and preserve latency");
        if (FAILED(wsc->SetMaximumFrameLatency(2)) || FAILED(wsc->GetMaximumFrameLatency(&lat)) || lat != 2)
            return fail("Waitable SetMaximumFrameLatency(2) round-trip failed");
        HANDLE waitable = wsc->GetFrameLatencyWaitableObject();
        if (!waitable) return fail("Waitable swapchain returned null handle");
        if (WaitForSingleObject(waitable, 5000) != WAIT_OBJECT_0 ||
            WaitForSingleObject(waitable, 5000) != WAIT_OBJECT_0)
            return fail("Initial WaitForSingleObject on latency-2 waitable swapchain failed");
        if (wsc->Present(1, DXGI_PRESENT_ALLOW_TEARING) != DXGI_ERROR_INVALID_CALL)
            return fail("Present(1, ALLOW_TEARING) should return DXGI_ERROR_INVALID_CALL");
        if (FAILED(wsc->Present(0, DXGI_PRESENT_ALLOW_TEARING)))
            return fail("Present(0, ALLOW_TEARING) on untouched 10-bit offscreen swapchain failed");
        if (!check_statistics_disjoint(wsc)) return fail("first statistics query after Present should be disjoint");
        if (WaitForSingleObject(waitable, 5000) != WAIT_OBJECT_0)
            return fail("Post-Present WaitForSingleObject on waitable swapchain failed");
        if (FAILED(wsc->ResizeBuffers(0, wdesc.Width + 1, wdesc.Height, DXGI_FORMAT_UNKNOWN, both_flags)) ||
            FAILED(wsc->GetDesc1(&got_desc)) || got_desc.Width != wdesc.Width + 1 || got_desc.Flags != both_flags)
            return fail("ResizeBuffers with matching creation flags failed");
        if (FAILED(wsc->Present(0, DXGI_PRESENT_ALLOW_TEARING)) ||
            WaitForSingleObject(waitable, 5000) != WAIT_OBJECT_0)
            return fail("Present or wait after resizing the waitable swapchain failed");
        if (FAILED(wsc->SetMaximumFrameLatency(40)) || FAILED(wsc->GetMaximumFrameLatency(&lat)) || lat != 40)
            return fail("Waitable SetMaximumFrameLatency(40) round-trip failed");
        if (WaitForSingleObject(waitable, 0) != WAIT_OBJECT_0)
            return fail("Waitable handle was not signaled after increasing latency from 2 to 40");
        if (wsc->SetMaximumFrameLatency(41) != DXGI_ERROR_INVALID_CALL ||
            FAILED(wsc->GetMaximumFrameLatency(&lat)) || lat != 40)
            return fail("Waitable SetMaximumFrameLatency(41) should return INVALID_CALL and preserve latency");
        CloseHandle(waitable);
        wsc->Release();
        DestroyWindow(hwnd3);
    }

    // CreateSwapChainForCoreWindow & Occlusion status registration
    {
        // 1. Parameter validation
        IDXGISwapChain1* sc_dummy = nullptr;
        if (factory->CreateSwapChainForCoreWindow(queue, nullptr, &scd, nullptr, &sc_dummy) != E_INVALIDARG)
            return fail("CreateSwapChainForCoreWindow(null window) should return E_INVALIDARG");
        if (factory->CreateSwapChainForCoreWindow(nullptr, queue, &scd, nullptr, &sc_dummy) != E_INVALIDARG)
            return fail("CreateSwapChainForCoreWindow(null device) should return E_INVALIDARG");
        if (factory->CreateSwapChainForCoreWindow(queue, queue, nullptr, nullptr, &sc_dummy) != E_INVALIDARG)
            return fail("CreateSwapChainForCoreWindow(null desc) should return E_INVALIDARG");
        if (factory->CreateSwapChainForCoreWindow(queue, queue, &scd, nullptr, nullptr) != E_POINTER)
            return fail("CreateSwapChainForCoreWindow(null out) should return E_POINTER");

        // Non-CoreWindow IUnknown
        if (factory->CreateSwapChainForCoreWindow(queue, queue, &scd, nullptr, &sc_dummy) != DXGI_ERROR_INVALID_CALL)
            return fail("CreateSwapChainForCoreWindow with non-CoreWindow should return DXGI_ERROR_INVALID_CALL");

        // 2. Mock ICoreWindow with ICoreWindowInterop
        static constexpr GUID kIID_ICoreWindowInterop = {
            0x45D64A29, 0xA63E, 0x4CB6, {0xB4, 0x98, 0x57, 0x81, 0xD2, 0x98, 0xCB, 0x4F}};
        struct MockCoreWindow final : public IUnknown {
            HWND window_handle = nullptr;
            std::atomic<ULONG> refs{1};

            HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
                if (!out) return E_POINTER;
                if (riid == IID_IUnknown || riid == kIID_ICoreWindowInterop) {
                    *out = this;
                    AddRef();
                    return S_OK;
                }
                *out = nullptr;
                return E_NOINTERFACE;
            }
            ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
            ULONG STDMETHODCALLTYPE Release() override {
                ULONG r = --refs;
                if (!r) delete this;
                return r;
            }
            virtual HRESULT STDMETHODCALLTYPE get_WindowHandle(HWND* hwnd) {
                if (!hwnd) return E_POINTER;
                *hwnd = window_handle;
                return S_OK;
            }
            virtual HRESULT STDMETHODCALLTYPE put_MessageHandled(unsigned char) { return S_OK; }
        };

        auto* mock_cw = new MockCoreWindow();
        // Null HWND in interop
        if (factory->CreateSwapChainForCoreWindow(queue, mock_cw, &scd, nullptr, &sc_dummy) != DXGI_ERROR_INVALID_CALL) {
            mock_cw->Release();
            return fail("CreateSwapChainForCoreWindow with null HWND should return DXGI_ERROR_INVALID_CALL");
        }

        // Valid HWND in mock CoreWindow
        HWND hwnd4 = create_test_window(160, 120);
        if (!hwnd4) {
            mock_cw->Release();
            return fail("CreateWindowExW (fourth)");
        }
        mock_cw->window_handle = hwnd4;
        for (const auto& samples : invalid_samples) {
            DXGI_SWAP_CHAIN_DESC1 invalid = scd;
            invalid.SampleDesc = samples;
            IDXGISwapChain1* rejected = nullptr;
            const HRESULT hr = factory->CreateSwapChainForCoreWindow(queue, mock_cw, &invalid, nullptr, &rejected);
            if (rejected || hr != DXGI_ERROR_INVALID_CALL) {
                if (rejected) rejected->Release();
                mock_cw->Release();
                DestroyWindow(hwnd4);
                return fail("CreateSwapChainForCoreWindow should reject invalid sample descriptions");
            }
        }
        IDXGISwapChain1* cw_sc1 = nullptr;
        if (FAILED(factory->CreateSwapChainForCoreWindow(queue, mock_cw, &scd, nullptr, &cw_sc1)) || !cw_sc1) {
            mock_cw->Release();
            DestroyWindow(hwnd4);
            return fail("CreateSwapChainForCoreWindow with valid MockCoreWindow failed");
        }
        IDXGISwapChain3* cw_sc = nullptr;
        if (FAILED(cw_sc1->QueryInterface(IID_PPV_ARGS(&cw_sc))) || !cw_sc) {
            cw_sc1->Release();
            mock_cw->Release();
            DestroyWindow(hwnd4);
            return fail("QI IDXGISwapChain3 on CoreWindow swapchain failed");
        }
        cw_sc1->Release();
        DXGI_SWAP_CHAIN_DESC1 cw_desc{};
        if (FAILED(cw_sc->GetDesc1(&cw_desc)) || cw_desc.Width != scd.Width || cw_desc.Height != scd.Height ||
            cw_desc.SampleDesc.Count != 1 || cw_desc.SampleDesc.Quality != 0) {
            cw_sc->Release();
            mock_cw->Release();
            DestroyWindow(hwnd4);
            return fail("GetDesc1 on CoreWindow swapchain mismatch");
        }
        if (FAILED(cw_sc->Present(0, 0))) {
            cw_sc->Release();
            mock_cw->Release();
            DestroyWindow(hwnd4);
            return fail("Present on CoreWindow swapchain failed");
        }
        cw_sc->Release();
        mock_cw->Release();
        DestroyWindow(hwnd4);

        // 3. Occlusion status registration
        DWORD cookie1 = 0, cookie2 = 0;
        if (factory->RegisterOcclusionStatusWindow(nullptr, WM_USER, &cookie1) != DXGI_ERROR_INVALID_CALL)
            return fail("RegisterOcclusionStatusWindow(null hwnd) should return DXGI_ERROR_INVALID_CALL");
        if (factory->RegisterOcclusionStatusWindow(hwnd, WM_USER, nullptr) != DXGI_ERROR_INVALID_CALL)
            return fail("RegisterOcclusionStatusWindow(null cookie) should return DXGI_ERROR_INVALID_CALL");
        if (FAILED(factory->RegisterOcclusionStatusWindow(hwnd, WM_USER, &cookie1)) || cookie1 == 0)
            return fail("RegisterOcclusionStatusWindow failed");

        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (factory->RegisterOcclusionStatusEvent(nullptr, &cookie2) != DXGI_ERROR_INVALID_CALL)
            return fail("RegisterOcclusionStatusEvent(null event) should return DXGI_ERROR_INVALID_CALL");
        if (factory->RegisterOcclusionStatusEvent(ev, nullptr) != DXGI_ERROR_INVALID_CALL)
            return fail("RegisterOcclusionStatusEvent(null cookie) should return DXGI_ERROR_INVALID_CALL");
        if (FAILED(factory->RegisterOcclusionStatusEvent(ev, &cookie2)) || cookie2 == 0 || cookie2 == cookie1)
            return fail("RegisterOcclusionStatusEvent failed or duplicated cookie");
        factory->UnregisterOcclusionStatus(cookie1);
        factory->UnregisterOcclusionStatus(cookie2);
        CloseHandle(ev);

        // 4. Stereo status registration (Microsoft: IDXGIFactory2::RegisterStereoStatusWindow / RegisterStereoStatusEvent
        // return S_OK with a cookie for UnregisterStereoStatus; stereo is never enabled, so nothing is notified).
        DWORD stereo_window = 0, stereo_event = 0;
        HANDLE stereo_ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (factory->RegisterStereoStatusWindow(hwnd, WM_USER, &stereo_window) != S_OK || stereo_window == 0)
            return fail("RegisterStereoStatusWindow should return S_OK with a cookie");
        if (factory->RegisterStereoStatusEvent(stereo_ev, &stereo_event) != S_OK || stereo_event == 0 ||
            stereo_event == stereo_window)
            return fail("RegisterStereoStatusEvent should return S_OK with a cookie of its own");
        factory->UnregisterStereoStatus(stereo_window);
        factory->UnregisterStereoStatus(stereo_event);
        CloseHandle(stereo_ev);
    }

    // Classic IDXGIFactory::CreateSwapChain delegation
    {
        if (factory->CreateSwapChain(nullptr, nullptr, nullptr) != DXGI_ERROR_INVALID_CALL)
            return fail("CreateSwapChain(null out) should return DXGI_ERROR_INVALID_CALL");
        IDXGISwapChain* legacy_sc = nullptr;
        if (factory->CreateSwapChain(nullptr, nullptr, &legacy_sc) != DXGI_ERROR_INVALID_CALL)
            return fail("CreateSwapChain(null device, null desc) should return DXGI_ERROR_INVALID_CALL");
        if (factory->CreateSwapChain(queue, nullptr, &legacy_sc) != DXGI_ERROR_INVALID_CALL)
            return fail("CreateSwapChain(valid device, null desc) should return DXGI_ERROR_INVALID_CALL");

        HWND hwnd_legacy = create_test_window(160, 120);
        if (!hwnd_legacy) return fail("CreateWindowExW (legacy)");

        DXGI_SWAP_CHAIN_DESC leg_desc{};
        leg_desc.BufferDesc.Width = 160;
        leg_desc.BufferDesc.Height = 120;
        leg_desc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        leg_desc.SampleDesc.Count = 1;
        leg_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        leg_desc.BufferCount = 2;
        leg_desc.OutputWindow = hwnd_legacy;
        leg_desc.Windowed = TRUE;
        leg_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

        if (FAILED(factory->CreateSwapChain(queue, &leg_desc, &legacy_sc)) || !legacy_sc) {
            DestroyWindow(hwnd_legacy);
            return fail("CreateSwapChain failed");
        }

        DXGI_SWAP_CHAIN_DESC got_leg_desc{};
        if (FAILED(legacy_sc->GetDesc(&got_leg_desc)) || got_leg_desc.BufferDesc.Width != 160 ||
            got_leg_desc.BufferDesc.Height != 120) {
            legacy_sc->Release();
            DestroyWindow(hwnd_legacy);
            return fail("GetDesc on legacy swapchain mismatch");
        }

        // Fullscreen request fallback: when exclusive fullscreen is unavailable,
        // CreateSwapChain falls back to windowed chain and returns DXGI_STATUS_OCCLUDED.
        HWND hwnd_fs = create_test_window(160, 120);
        if (!hwnd_fs) return fail("CreateWindowExW (fullscreen fallback)");
        DXGI_SWAP_CHAIN_DESC fs_req_desc = leg_desc;
        fs_req_desc.OutputWindow = hwnd_fs;
        fs_req_desc.Windowed = FALSE;
        IDXGISwapChain* fs_sc = nullptr;
        const HRESULT fs_hr = factory->CreateSwapChain(queue, &fs_req_desc, &fs_sc);
        if (fs_hr != DXGI_STATUS_OCCLUDED || !fs_sc) {
            if (fs_sc) fs_sc->Release();
            DestroyWindow(hwnd_fs);
            return fail("CreateSwapChain(Windowed == FALSE) should return DXGI_STATUS_OCCLUDED fallback");
        }
        BOOL is_fs = TRUE;
        fs_sc->GetFullscreenState(&is_fs, nullptr);
        if (is_fs != FALSE) {
            fs_sc->Release();
            DestroyWindow(hwnd_fs);
            return fail("GetFullscreenState should remain FALSE on fallback windowed swapchain");
        }
        fs_sc->Release();
        DestroyWindow(hwnd_fs);

        if (FAILED(legacy_sc->Present(0, 0))) {
            legacy_sc->Release();
            DestroyWindow(hwnd_legacy);
            return fail("Present on legacy swapchain failed");
        }

        legacy_sc->Release();
        DestroyWindow(hwnd_legacy);
    }

    rtv_heap->Release();
    sc->Release();
    fence->Release();
    cl->Release();
    alloc->Release();
    queue->Release();
    factory->Release();
    dev->Release();
    DestroyWindow(hwnd);
    wait_idle(vk_dev);
    destroy_dev(vk_dev, nullptr);
    if (messenger && destroy_dbg) destroy_dbg(instance, messenger, nullptr);
    destroy_inst(instance, nullptr);

    if (validation_errors() != 0) {
        std::fprintf(stderr, "FAIL: %u Vulkan validation errors\n", validation_errors());
        return 1;
    }
    std::printf("device: %s (validation %s)\n", chosen.deviceName, val_active ? "on" : "off");
    std::printf("dxgi_swapchain_test: PASS (CreateSwapChain, CreateSwapChainForHwnd, CreateSwapChainForCoreWindow, occlusion status, "
                "GetBuffer+GetDesc, clear+readback, Present, ResizeBuffers; 0 validation errors)\n");
    return 0;
}
