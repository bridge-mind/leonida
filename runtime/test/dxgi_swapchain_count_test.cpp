// Synthetic regression test for the DXGI swapchain buffer-count contract (D3D-389), with no GPU and no
// window: VulkanSwapchain (runtime/shims/dxgi_swapchain.h) runs on a fake Vulkan function table, so
// vkGetSwapchainImagesKHR can report more images than the app's requested BufferCount.
//
// DXGI rule: GetDesc/GetDesc1 report the BufferCount the app created with, and GetCurrentBackBufferIndex is
// always below it, one stable resource per index. The shim must not expose images the driver added above
// minImageCount, and must hide a caps.minImageCount above the request. When the native count differs, the
// exposed buffers form an offscreen ring; each present blits the current slot into the acquired native image.
#include "../shims/dxgi_swapchain.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

constexpr std::uintptr_t kSurface = 0x5100;
constexpr std::uintptr_t kSwapchain = 0x5101;
constexpr std::uintptr_t kFence = 0x5102;
constexpr std::uintptr_t kPool = 0x5103;
constexpr std::uintptr_t kCmd = 0x5104;
constexpr std::uintptr_t kNativeBase = 0x6000;
constexpr std::uintptr_t kOffscreenBase = 0x7000;

struct Fake {
    std::uint32_t min_image_count = 2;
    std::uint32_t max_image_count = 0;
    std::uint32_t returned_images = 2;
    std::uint32_t create_min_image_count = 0;
    std::uint32_t acquire_cursor = 0;
    std::uint32_t last_acquired = 0;
    std::uint32_t presented_image = UINT32_MAX;
    VkImage blit_src = VK_NULL_HANDLE;
    VkImage blit_dst = VK_NULL_HANDLE;
    unsigned presents = 0;
    unsigned created_offscreen = 0;
};
Fake g;

VkImage native_image(std::uint32_t i) {
    return reinterpret_cast<VkImage>(static_cast<std::uintptr_t>(kNativeBase + i));
}

template <typename T>
PFN_vkVoidFunction as_function(T fn) {
    return reinterpret_cast<PFN_vkVoidFunction>(fn);
}

VKAPI_ATTR VkResult VKAPI_CALL create_win32_surface(VkInstance, const VkWin32SurfaceCreateInfoKHR*,
                                                    const VkAllocationCallbacks*, VkSurfaceKHR* out) {
    if (out) *out = reinterpret_cast<VkSurfaceKHR>(kSurface);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_surface(VkInstance, VkSurfaceKHR, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL surface_support(VkPhysicalDevice, std::uint32_t, VkSurfaceKHR, VkBool32* out) {
    if (out) *out = VK_TRUE;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL surface_caps(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR* caps) {
    if (!caps) return VK_ERROR_INITIALIZATION_FAILED;
    std::memset(caps, 0, sizeof(*caps));
    caps->minImageCount = g.min_image_count;
    caps->maxImageCount = g.max_image_count;
    caps->currentExtent = {64, 64};
    caps->minImageExtent = {1, 1};
    caps->maxImageExtent = {4096, 4096};
    caps->supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    caps->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL surface_formats(VkPhysicalDevice, VkSurfaceKHR, std::uint32_t* count,
                                               VkSurfaceFormatKHR* formats) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (formats && *count) {
        formats[0].format = VK_FORMAT_B8G8R8A8_UNORM;
        formats[0].colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    }
    *count = 1;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL surface_present_modes(VkPhysicalDevice, VkSurfaceKHR, std::uint32_t* count,
                                                     VkPresentModeKHR* modes) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (modes && *count) modes[0] = VK_PRESENT_MODE_FIFO_KHR;
    *count = 1;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL memory_properties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* props) {
    std::memset(props, 0, sizeof(*props));
    props->memoryTypeCount = 1;
    props->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    props->memoryHeapCount = 1;
    props->memoryHeaps[0].size = 1ull << 30;
    props->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
}
VKAPI_ATTR VkResult VKAPI_CALL create_swapchain(VkDevice, const VkSwapchainCreateInfoKHR* info,
                                                const VkAllocationCallbacks*, VkSwapchainKHR* out) {
    if (!info || !out) return VK_ERROR_INITIALIZATION_FAILED;
    g.create_min_image_count = info->minImageCount;
    *out = reinterpret_cast<VkSwapchainKHR>(kSwapchain);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_swapchain(VkDevice, VkSwapchainKHR, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL get_images(VkDevice, VkSwapchainKHR, std::uint32_t* count, VkImage* images) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (images) {
        for (std::uint32_t i = 0; i < g.returned_images && i < *count; ++i) images[i] = native_image(i);
    }
    *count = g.returned_images;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL acquire_next_image(VkDevice, VkSwapchainKHR, std::uint64_t, VkSemaphore, VkFence,
                                                  std::uint32_t* out) {
    if (!out || g.returned_images == 0) return VK_ERROR_INITIALIZATION_FAILED;
    g.last_acquired = g.acquire_cursor % g.returned_images;
    ++g.acquire_cursor;
    *out = g.last_acquired;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL queue_present(VkQueue, const VkPresentInfoKHR* info) {
    if (!info || info->swapchainCount != 1 || !info->pImageIndices) return VK_ERROR_INITIALIZATION_FAILED;
    g.presented_image = info->pImageIndices[0];
    ++g.presents;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_fence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*,
                                            VkFence* out) {
    if (out) *out = reinterpret_cast<VkFence>(kFence);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_fence(VkDevice, VkFence, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL wait_for_fences(VkDevice, std::uint32_t, const VkFence*, VkBool32, std::uint64_t) {
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL reset_fences(VkDevice, std::uint32_t, const VkFence*) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL device_wait_idle(VkDevice) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice, const VkImageCreateInfo*, const VkAllocationCallbacks*,
                                            VkImage* out) {
    if (!out) return VK_ERROR_INITIALIZATION_FAILED;
    *out = reinterpret_cast<VkImage>(static_cast<std::uintptr_t>(kOffscreenBase + g.created_offscreen++));
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_image(VkDevice, VkImage, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL image_mem_req(VkDevice, VkImage, VkMemoryRequirements* req) {
    std::memset(req, 0, sizeof(*req));
    req->size = 4096;
    req->alignment = 1;
    req->memoryTypeBits = 1;
}
VKAPI_ATTR VkResult VKAPI_CALL alloc_mem(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*,
                                         VkDeviceMemory* out) {
    if (out) *out = reinterpret_cast<VkDeviceMemory>(kOffscreenBase);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL free_mem(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL bind_image_mem(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_cmd_pool(VkDevice, const VkCommandPoolCreateInfo*,
                                               const VkAllocationCallbacks*, VkCommandPool* out) {
    if (out) *out = reinterpret_cast<VkCommandPool>(kPool);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_cmd_pool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL alloc_cmd_bufs(VkDevice, const VkCommandBufferAllocateInfo*,
                                              VkCommandBuffer* out) {
    if (out) *out = reinterpret_cast<VkCommandBuffer>(kCmd);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL reset_cmd_buf(VkCommandBuffer, VkCommandBufferResetFlags) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL begin_cmd_buf(VkCommandBuffer, const VkCommandBufferBeginInfo*) {
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL end_cmd_buf(VkCommandBuffer) { return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL cmd_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, std::uint32_t,
                                       const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                       std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL cmd_blit(VkCommandBuffer, VkImage src, VkImageLayout, VkImage dst, VkImageLayout,
                                    std::uint32_t, const VkImageBlit*, VkFilter) {
    g.blit_src = src;
    g.blit_dst = dst;
}
VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue, std::uint32_t, const VkSubmitInfo*, VkFence) {
    return VK_SUCCESS;
}

struct Entry {
    const char* name;
    PFN_vkVoidFunction fn;
};
const Entry kEntries[] = {
    {"vkCreateWin32SurfaceKHR", as_function(create_win32_surface)},
    {"vkDestroySurfaceKHR", as_function(destroy_surface)},
    {"vkGetPhysicalDeviceSurfaceSupportKHR", as_function(surface_support)},
    {"vkGetPhysicalDeviceSurfaceCapabilitiesKHR", as_function(surface_caps)},
    {"vkGetPhysicalDeviceSurfaceFormatsKHR", as_function(surface_formats)},
    {"vkGetPhysicalDeviceSurfacePresentModesKHR", as_function(surface_present_modes)},
    {"vkGetPhysicalDeviceMemoryProperties", as_function(memory_properties)},
    {"vkCreateSwapchainKHR", as_function(create_swapchain)},
    {"vkDestroySwapchainKHR", as_function(destroy_swapchain)},
    {"vkGetSwapchainImagesKHR", as_function(get_images)},
    {"vkAcquireNextImageKHR", as_function(acquire_next_image)},
    {"vkQueuePresentKHR", as_function(queue_present)},
    {"vkCreateFence", as_function(create_fence)},
    {"vkDestroyFence", as_function(destroy_fence)},
    {"vkWaitForFences", as_function(wait_for_fences)},
    {"vkResetFences", as_function(reset_fences)},
    {"vkDeviceWaitIdle", as_function(device_wait_idle)},
    {"vkCreateImage", as_function(create_image)},
    {"vkDestroyImage", as_function(destroy_image)},
    {"vkGetImageMemoryRequirements", as_function(image_mem_req)},
    {"vkAllocateMemory", as_function(alloc_mem)},
    {"vkFreeMemory", as_function(free_mem)},
    {"vkBindImageMemory", as_function(bind_image_mem)},
    {"vkCreateCommandPool", as_function(create_cmd_pool)},
    {"vkDestroyCommandPool", as_function(destroy_cmd_pool)},
    {"vkAllocateCommandBuffers", as_function(alloc_cmd_bufs)},
    {"vkResetCommandBuffer", as_function(reset_cmd_buf)},
    {"vkBeginCommandBuffer", as_function(begin_cmd_buf)},
    {"vkEndCommandBuffer", as_function(end_cmd_buf)},
    {"vkCmdPipelineBarrier", as_function(cmd_barrier)},
    {"vkCmdBlitImage", as_function(cmd_blit)},
    {"vkQueueSubmit", as_function(queue_submit)},
};

PFN_vkVoidFunction lookup(const char* name) {
    for (const Entry& entry : kEntries)
        if (name && std::strcmp(name, entry.name) == 0) return entry.fn;
    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance_proc(VkInstance, const char* name) { return lookup(name); }
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device_proc(VkDevice, const char* name) { return lookup(name); }

int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

// Requests `requested` buffers while faking `native` images and a surface minimum of `minimum`.
int run_case(const char* label, std::uint32_t requested, std::uint32_t native, std::uint32_t minimum,
             bool expect_offscreen) {
    g = Fake{};
    g.min_image_count = minimum;
    g.returned_images = native;
    std::mutex mutex;
    leonida::VulkanSwapchain swapchain(get_instance_proc, reinterpret_cast<VkInstance>(1),
                                      reinterpret_cast<VkPhysicalDevice>(2), get_device_proc,
                                      reinterpret_cast<VkDevice>(3), reinterpret_cast<VkQueue>(4), 0, mutex);
    if (!swapchain.create(reinterpret_cast<HWND>(static_cast<std::uintptr_t>(1)), 64, 64,
                          VK_FORMAT_B8G8R8A8_UNORM, requested))
        return fail("VulkanSwapchain::create");

    // GetDesc/GetDesc1 read image_count(): the app's request, never the native count.
    if (swapchain.image_count() != requested) {
        std::fprintf(stderr, "FAIL: %s: exposed BufferCount %u, requested %u (native %u)\n", label,
                     swapchain.image_count(), requested, native);
        return 1;
    }
    const std::uint32_t expected_min = minimum > requested ? minimum : requested;
    if (g.create_min_image_count != expected_min) {
        std::fprintf(stderr, "FAIL: %s: vkCreateSwapchainKHR minImageCount %u, expected %u\n", label,
                     g.create_min_image_count, expected_min);
        return 1;
    }
    if (swapchain.is_offscreen() != expect_offscreen)
        return fail(expect_offscreen ? "expected the offscreen ring" : "expected direct native images");

    if (expect_offscreen) {
        if (g.created_offscreen != requested) return fail("offscreen ring size != requested BufferCount");
        std::vector<VkImage> exposed;
        for (std::uint32_t i = 0; i < requested; ++i) {
            const leonida::ResourceBinding binding = swapchain.resource_binding(i);
            if (!binding.image || binding.presentable) return fail("offscreen backbuffer binding is not offscreen");
            if (binding.image != swapchain.image(i)) return fail("binding/image disagree");
            for (VkImage other : exposed)
                if (other == binding.image) return fail("two exposed backbuffers share one image");
            exposed.push_back(binding.image);
        }
        if (swapchain.current_back_buffer_index() >= requested)
            return fail("initial current index out of range");

        std::uint32_t index = 0;
        if (!swapchain.acquire_next_image(index)) return fail("first acquire");
        constexpr unsigned kFrames = 20;
        for (unsigned frame = 0; frame < kFrames; ++frame) {
            const std::uint32_t current = swapchain.current_back_buffer_index();
            if (current >= requested) return fail("GetCurrentBackBufferIndex >= BufferCount");
            if (current != frame % requested) return fail("current index does not cycle over the exposed ring");
            const std::uint32_t acquired = g.last_acquired;
            g.blit_src = g.blit_dst = VK_NULL_HANDLE;
            g.presented_image = UINT32_MAX;
            if (!swapchain.present(current)) return fail("present");
            if (g.blit_src != exposed[current]) return fail("present did not blit the current backbuffer");
            if (g.blit_dst != native_image(acquired)) return fail("present blitted to the wrong acquired image");
            if (g.presented_image != acquired) return fail("presented an image other than the acquired one");
            if (frame + 1 < kFrames && !swapchain.acquire_next_image(index)) return fail("acquire");
        }
        // ResizeBuffers(count=0) keeps the exposed BufferCount, not the native image count.
        if (!swapchain.resize(64, 64)) return fail("resize");
        if (swapchain.image_count() != requested) return fail("resize(count=0) changed the exposed BufferCount");
        if (!swapchain.is_offscreen()) return fail("resize dropped the offscreen ring");
        return 0;
    }

    // Direct path: exposed index and native image index are the same, and nothing is blitted.
    if (swapchain.current_back_buffer_index() != 0) return fail("direct initial index");
    std::uint32_t index = 0;
    if (!swapchain.acquire_next_image(index)) return fail("direct first acquire");
    for (unsigned frame = 0; frame < 9; ++frame) {
        const std::uint32_t current = swapchain.current_back_buffer_index();
        if (current >= requested) return fail("direct GetCurrentBackBufferIndex >= BufferCount");
        if (current != g.last_acquired) return fail("direct current index is not the acquired image");
        if (swapchain.resource_binding(current).image != native_image(current))
            return fail("direct binding is not the native image");
        g.blit_src = g.blit_dst = VK_NULL_HANDLE;
        g.presented_image = UINT32_MAX;
        if (!swapchain.present(current)) return fail("direct present");
        if (g.blit_src || g.blit_dst) return fail("direct present took the blit path");
        if (g.presented_image != current) return fail("direct present index");
        if (frame + 1 < 9 && !swapchain.acquire_next_image(index)) return fail("direct acquire");
    }
    return 0;
}

struct Case {
    const char* label;
    std::uint32_t requested;
    std::uint32_t native;
    std::uint32_t minimum;
    bool offscreen;
};

}  // namespace

int main() {
    const Case cases[] = {
        // Vulkan returns more images than minImageCount: the app asked for 2 and must see 2.
        {"driver returns 4 for BufferCount 2", 2, 4, 2, true},
        // The surface minimum exceeds the request (2): still expose the app's 2.
        {"minImageCount 3 for BufferCount 2", 2, 3, 3, true},
        // Control: the app asks for more than minImageCount; native and exposed counts agree, no blit.
        {"BufferCount 3 above minImageCount 2", 3, 3, 2, false},
    };
    for (const Case& c : cases) {
        if (const int result = run_case(c.label, c.requested, c.native, c.minimum, c.offscreen)) return result;
    }
    std::printf("dxgi_swapchain_count_test: PASS (3 cases: extra native images, min above request, "
                "request above min)\n");
    return 0;
}
