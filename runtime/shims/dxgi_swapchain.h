#pragma once

// Vulkan swapchain backend for the dxgi.dll shim (docs/d3d12_vulkan.md section 3.1 & 4):
// Wraps VK_KHR_win32_surface + VkSwapchainKHR over a Win32 HWND, exposing acquire/present/resize
// and ResourceBindings for the swapchain images (with presentable = true so D3D12_RESOURCE_STATE_PRESENT
// maps to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR).

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <windows.h>
#include <vulkan/vulkan.h>

#include "../src/vulkan_backend.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

namespace leonida {

struct SwapchainFunctions {
    PFN_vkCreateWin32SurfaceKHR create_win32_surface = nullptr;
    PFN_vkDestroySurfaceKHR destroy_surface = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR surface_support = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR surface_caps = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR surface_formats = nullptr;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR surface_present_modes = nullptr;
    PFN_vkCreateSwapchainKHR create_swapchain = nullptr;
    PFN_vkDestroySwapchainKHR destroy_swapchain = nullptr;
    PFN_vkGetSwapchainImagesKHR get_images = nullptr;
    PFN_vkAcquireNextImageKHR acquire_next_image = nullptr;
    PFN_vkQueuePresentKHR queue_present = nullptr;
    PFN_vkCreateFence create_fence = nullptr;
    PFN_vkDestroyFence destroy_fence = nullptr;
    PFN_vkWaitForFences wait_for_fences = nullptr;
    PFN_vkResetFences reset_fences = nullptr;
    PFN_vkDeviceWaitIdle device_wait_idle = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties mem_props = nullptr;
    PFN_vkCreateImage create_image = nullptr;
    PFN_vkDestroyImage destroy_image = nullptr;
    PFN_vkGetImageMemoryRequirements image_mem_req = nullptr;
    PFN_vkAllocateMemory alloc_mem = nullptr;
    PFN_vkFreeMemory free_mem = nullptr;
    PFN_vkBindImageMemory bind_image_mem = nullptr;
    PFN_vkCreateCommandPool create_cmd_pool = nullptr;
    PFN_vkDestroyCommandPool destroy_cmd_pool = nullptr;
    PFN_vkAllocateCommandBuffers alloc_cmd_bufs = nullptr;
    PFN_vkResetCommandBuffer reset_cmd_buf = nullptr;
    PFN_vkBeginCommandBuffer begin_cmd_buf = nullptr;
    PFN_vkEndCommandBuffer end_cmd_buf = nullptr;
    PFN_vkCmdPipelineBarrier cmd_barrier = nullptr;
    PFN_vkCmdBlitImage cmd_blit = nullptr;
    PFN_vkQueueSubmit queue_submit = nullptr;

    bool load(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, PFN_vkGetDeviceProcAddr gdpa, VkDevice device) {
#define LOAD_I(member, name) member = reinterpret_cast<PFN_##name>(gipa(instance, #name))
#define LOAD_D(member, name) member = reinterpret_cast<PFN_##name>(gdpa(device, #name))
        LOAD_I(create_win32_surface, vkCreateWin32SurfaceKHR);
        LOAD_I(destroy_surface, vkDestroySurfaceKHR);
        LOAD_I(surface_support, vkGetPhysicalDeviceSurfaceSupportKHR);
        LOAD_I(surface_caps, vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
        LOAD_I(surface_formats, vkGetPhysicalDeviceSurfaceFormatsKHR);
        LOAD_I(surface_present_modes, vkGetPhysicalDeviceSurfacePresentModesKHR);
        LOAD_I(mem_props, vkGetPhysicalDeviceMemoryProperties);
        LOAD_D(create_swapchain, vkCreateSwapchainKHR);
        LOAD_D(destroy_swapchain, vkDestroySwapchainKHR);
        LOAD_D(get_images, vkGetSwapchainImagesKHR);
        LOAD_D(acquire_next_image, vkAcquireNextImageKHR);
        LOAD_D(queue_present, vkQueuePresentKHR);
        LOAD_D(create_fence, vkCreateFence);
        LOAD_D(destroy_fence, vkDestroyFence);
        LOAD_D(wait_for_fences, vkWaitForFences);
        LOAD_D(reset_fences, vkResetFences);
        LOAD_D(device_wait_idle, vkDeviceWaitIdle);
        LOAD_D(create_image, vkCreateImage);
        LOAD_D(destroy_image, vkDestroyImage);
        LOAD_D(image_mem_req, vkGetImageMemoryRequirements);
        LOAD_D(alloc_mem, vkAllocateMemory);
        LOAD_D(free_mem, vkFreeMemory);
        LOAD_D(bind_image_mem, vkBindImageMemory);
        LOAD_D(create_cmd_pool, vkCreateCommandPool);
        LOAD_D(destroy_cmd_pool, vkDestroyCommandPool);
        LOAD_D(alloc_cmd_bufs, vkAllocateCommandBuffers);
        LOAD_D(reset_cmd_buf, vkResetCommandBuffer);
        LOAD_D(begin_cmd_buf, vkBeginCommandBuffer);
        LOAD_D(end_cmd_buf, vkEndCommandBuffer);
        LOAD_D(cmd_barrier, vkCmdPipelineBarrier);
        LOAD_D(cmd_blit, vkCmdBlitImage);
        LOAD_D(queue_submit, vkQueueSubmit);
#undef LOAD_I
#undef LOAD_D
        return create_win32_surface && destroy_surface && surface_support && surface_caps && surface_formats &&
               mem_props && create_swapchain && destroy_swapchain && get_images && acquire_next_image &&
               queue_present && create_fence && destroy_fence && wait_for_fences && reset_fences && device_wait_idle &&
               create_image && destroy_image && image_mem_req && alloc_mem && free_mem && bind_image_mem &&
               create_cmd_pool && destroy_cmd_pool && alloc_cmd_bufs && reset_cmd_buf && begin_cmd_buf &&
               end_cmd_buf && cmd_barrier && cmd_blit && queue_submit;
    }
};

class VulkanSwapchain {
public:
    VulkanSwapchain(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkPhysicalDevice physical,
                    PFN_vkGetDeviceProcAddr gdpa, VkDevice device, VkQueue queue, std::uint32_t queue_family,
                    std::mutex& submit_mutex)
        : instance_(instance), physical_(physical), device_(device), queue_(queue), queue_family_(queue_family),
          submit_mutex_(submit_mutex) {
        loaded_ = fn_.load(gipa, instance, gdpa, device);
        if (loaded_) {
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            if (fn_.create_fence(device_, &fi, nullptr, &acquire_fence_) != VK_SUCCESS) acquire_fence_ = VK_NULL_HANDLE;
            if (fn_.create_fence(device_, &fi, nullptr, &blit_fence_) != VK_SUCCESS) blit_fence_ = VK_NULL_HANDLE;
            VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pi.queueFamilyIndex = queue_family_;
            if (fn_.create_cmd_pool(device_, &pi, nullptr, &blit_pool_) == VK_SUCCESS) {
                VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                ai.commandPool = blit_pool_;
                ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                ai.commandBufferCount = 1;
                if (fn_.alloc_cmd_bufs(device_, &ai, &blit_cmd_) != VK_SUCCESS) blit_cmd_ = VK_NULL_HANDLE;
            }
        }
    }

    ~VulkanSwapchain() {
        destroy_swapchain();
        if (blit_pool_) fn_.destroy_cmd_pool(device_, blit_pool_, nullptr);
        if (surface_) fn_.destroy_surface(instance_, surface_, nullptr);
        if (acquire_fence_) fn_.destroy_fence(device_, acquire_fence_, nullptr);
        if (blit_fence_) fn_.destroy_fence(device_, blit_fence_, nullptr);
    }

    VulkanSwapchain(const VulkanSwapchain&) = delete;
    VulkanSwapchain& operator=(const VulkanSwapchain&) = delete;

    bool create(HWND hwnd, std::uint32_t width, std::uint32_t height,
                VkFormat preferred_format = VK_FORMAT_B8G8R8A8_UNORM, std::uint32_t buffer_count = 2,
                bool allow_tearing = false) {
        if (!loaded_ || !acquire_fence_ || !hwnd) return false;
        if (!surface_) {
            VkWin32SurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
            sci.hinstance = GetModuleHandleW(nullptr);
            sci.hwnd = hwnd;
            if (fn_.create_win32_surface(instance_, &sci, nullptr, &surface_) != VK_SUCCESS) return false;
            VkBool32 supported = VK_FALSE;
            if (fn_.surface_support(physical_, queue_family_, surface_, &supported) != VK_SUCCESS || !supported)
                return false;
        }
        allow_tearing_ = allow_tearing;
        return rebuild(width, height, preferred_format, buffer_count);
    }

    bool resize(std::uint32_t width, std::uint32_t height, VkFormat preferred_format = VK_FORMAT_UNDEFINED,
                std::uint32_t buffer_count = 0, bool allow_tearing = false) {
        if (!loaded_ || !surface_) return false;
        {
            std::lock_guard lock(submit_mutex_);
            fn_.device_wait_idle(device_);
        }
        allow_tearing_ = allow_tearing;
        const VkFormat fmt = (preferred_format != VK_FORMAT_UNDEFINED) ? preferred_format : format_;
        const std::uint32_t count = buffer_count ? buffer_count : exposed_count_;
        return rebuild(width, height, fmt, count ? count : 2u);
    }

    bool acquire_next_image(std::uint32_t& out_index, std::uint64_t timeout_ns = UINT64_MAX) {
        if (!swapchain_ || !acquire_fence_) return false;
        if (fn_.reset_fences(device_, 1, &acquire_fence_) != VK_SUCCESS) return false;
        const VkResult r = fn_.acquire_next_image(device_, swapchain_, timeout_ns, VK_NULL_HANDLE, acquire_fence_,
                                                  &out_index);
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) return false;
        if (fn_.wait_for_fences(device_, 1, &acquire_fence_, VK_TRUE, timeout_ns) != VK_SUCCESS) return false;
        acquired_index_ = out_index;
        if (is_offscreen()) {
            // The native image count may differ from the DXGI BufferCount (Vulkan is free to return more
            // images than minImageCount, and caps.minImageCount can exceed the app's request). The app's
            // backbuffers are the stable offscreen ring; each present blits the current slot into whichever
            // native image was acquired, so the exposed index never follows the native one.
            current_index_ = next_back_buffer_;
            next_back_buffer_ = (next_back_buffer_ + 1) % exposed_count_;
        } else {
            current_index_ = out_index;
        }
        // D3D12 swapchain backbuffers begin in D3D12_RESOURCE_STATE_PRESENT; transition each newly acquired
        // native swapchain image once from UNDEFINED -> PRESENT_SRC_KHR so presenting an untouched backbuffer
        // or transitioning from PRESENT matches the tracked layout.
        if (!is_offscreen() && out_index < image_initialized_.size() && !image_initialized_[out_index]) {
            if (!transition_image_layout(images_[out_index], VK_IMAGE_LAYOUT_UNDEFINED,
                                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR))
                return false;
            image_initialized_[out_index] = true;
        }
        return true;
    }

    bool present(std::uint32_t back_buffer_index) {
        if (!swapchain_ || back_buffer_index >= exposed_count_ || acquired_index_ >= images_.size()) return false;
        std::uint32_t image_index = back_buffer_index;
        if (is_offscreen()) {
            if (!blit_to_swapchain(back_buffer_index)) return false;
            image_index = acquired_index_;
        }
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain_;
        pi.pImageIndices = &image_index;
        std::lock_guard lock(submit_mutex_);
        const VkResult r = fn_.queue_present(queue_, &pi);
        return r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR;
    }

    // DXGI exposes exactly the BufferCount the app asked for (GetDesc/GetDesc1 and GetCurrentBackBufferIndex
    // stay inside it). When Vulkan supplies a different number of native images the extra ones are hidden
    // behind a stable offscreen ring (is_offscreen()).
    std::uint32_t image_count() const { return exposed_count_; }
    std::uint32_t current_back_buffer_index() const { return current_index_; }
    VkFormat format() const { return format_; }
    VkExtent2D extent() const { return extent_; }
    bool is_offscreen() const { return !offscreen_images_.empty(); }
    VkPresentModeKHR present_mode() const { return present_mode_; }
    VkImage image(std::uint32_t i) const {
        if (i >= exposed_count_) return VK_NULL_HANDLE;
        return is_offscreen() ? offscreen_images_[i] : images_[i];
    }

    ResourceBinding resource_binding(std::uint32_t i) const {
        ResourceBinding b{};
        if (i >= exposed_count_) return b;
        const bool offscreen = is_offscreen();
        b.image = offscreen ? offscreen_images_[i] : images_[i];
        b.format = format_;
        b.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        b.width = extent_.width;
        b.height = extent_.height;
        b.mip_levels = 1;
        b.array_layers = 1;
        b.presentable = !offscreen;
        return b;
    }

private:
    bool transition_image_layout(VkImage img, VkImageLayout old_layout, VkImageLayout new_layout) {
        if (!blit_cmd_ || !blit_fence_ || !img) return false;
        std::lock_guard lock(submit_mutex_);
        if (fn_.reset_fences(device_, 1, &blit_fence_) != VK_SUCCESS ||
            fn_.reset_cmd_buf(blit_cmd_, 0) != VK_SUCCESS)
            return false;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (fn_.begin_cmd_buf(blit_cmd_, &bi) != VK_SUCCESS) return false;
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask =
            (new_layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) ? 0 : (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = img;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        fn_.cmd_barrier(blit_cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                        nullptr, 0, nullptr, 1, &barrier);
        if (fn_.end_cmd_buf(blit_cmd_) != VK_SUCCESS) return false;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &blit_cmd_;
        return fn_.queue_submit(queue_, 1, &si, blit_fence_) == VK_SUCCESS &&
               fn_.wait_for_fences(device_, 1, &blit_fence_, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    }

    void destroy_offscreen() {
        for (VkImage img : offscreen_images_)
            if (img) fn_.destroy_image(device_, img, nullptr);
        for (VkDeviceMemory mem : offscreen_memory_)
            if (mem) fn_.free_mem(device_, mem, nullptr);
        offscreen_images_.clear();
        offscreen_memory_.clear();
    }

    void destroy_swapchain() {
        if (swapchain_ || !offscreen_images_.empty()) {
            std::lock_guard lock(submit_mutex_);
            fn_.device_wait_idle(device_);
            destroy_offscreen();
            if (swapchain_) {
                fn_.destroy_swapchain(device_, swapchain_, nullptr);
                swapchain_ = VK_NULL_HANDLE;
            }
        }
        images_.clear();
        exposed_count_ = 0;
    }

    bool create_offscreen_backbuffers(std::uint32_t count) {
        VkPhysicalDeviceMemoryProperties mp{};
        fn_.mem_props(physical_, &mp);
        offscreen_images_.reserve(count);
        offscreen_memory_.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = format_;
            ici.extent = {extent_.width, extent_.height, 1};
            ici.mipLevels = 1;
            ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImage img = VK_NULL_HANDLE;
            if (fn_.create_image(device_, &ici, nullptr, &img) != VK_SUCCESS) {
                destroy_offscreen();
                return false;
            }
            VkMemoryRequirements req{};
            fn_.image_mem_req(device_, img, &req);
            std::uint32_t mem_idx = UINT32_MAX;
            for (std::uint32_t m = 0; m < mp.memoryTypeCount; ++m) {
                if ((req.memoryTypeBits & (1u << m)) &&
                    (mp.memoryTypes[m].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                    mem_idx = m;
                    break;
                }
            }
            if (mem_idx == UINT32_MAX) {
                for (std::uint32_t m = 0; m < mp.memoryTypeCount; ++m) {
                    if (req.memoryTypeBits & (1u << m)) {
                        mem_idx = m;
                        break;
                    }
                }
            }
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = mem_idx;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            if (mem_idx == UINT32_MAX || fn_.alloc_mem(device_, &mai, nullptr, &mem) != VK_SUCCESS ||
                fn_.bind_image_mem(device_, img, mem, 0) != VK_SUCCESS) {
                if (mem) fn_.free_mem(device_, mem, nullptr);
                fn_.destroy_image(device_, img, nullptr);
                destroy_offscreen();
                return false;
            }
            offscreen_images_.push_back(img);
            offscreen_memory_.push_back(mem);
        }
        // Transition newly created offscreen backbuffers to GENERAL so presenting an untouched buffer
        // never violates blit_to_swapchain's GENERAL -> TRANSFER_SRC_OPTIMAL transition.
        if (blit_cmd_ && blit_fence_ && !offscreen_images_.empty()) {
            std::lock_guard lock(submit_mutex_);
            if (fn_.reset_fences(device_, 1, &blit_fence_) != VK_SUCCESS ||
                fn_.reset_cmd_buf(blit_cmd_, 0) != VK_SUCCESS) {
                destroy_offscreen();
                return false;
            }
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (fn_.begin_cmd_buf(blit_cmd_, &bi) != VK_SUCCESS) {
                destroy_offscreen();
                return false;
            }
            std::vector<VkImageMemoryBarrier> barriers(offscreen_images_.size());
            for (std::size_t i = 0; i < offscreen_images_.size(); ++i) {
                barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barriers[i].srcAccessMask = 0;
                barriers[i].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[i].image = offscreen_images_[i];
                barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            }
            fn_.cmd_barrier(blit_cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                            nullptr, 0, nullptr, static_cast<std::uint32_t>(barriers.size()), barriers.data());
            if (fn_.end_cmd_buf(blit_cmd_) != VK_SUCCESS) {
                destroy_offscreen();
                return false;
            }
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &blit_cmd_;
            if (fn_.queue_submit(queue_, 1, &si, blit_fence_) != VK_SUCCESS ||
                fn_.wait_for_fences(device_, 1, &blit_fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
                destroy_offscreen();
                return false;
            }
        }
        return true;
    }

    bool blit_to_swapchain(std::uint32_t image_index) {
        if (!blit_cmd_ || !blit_fence_ || image_index >= offscreen_images_.size() ||
            acquired_index_ >= images_.size())
            return false;
        const VkImage dst_image = images_[acquired_index_];
        std::lock_guard lock(submit_mutex_);
        if (fn_.reset_fences(device_, 1, &blit_fence_) != VK_SUCCESS) return false;
        if (fn_.reset_cmd_buf(blit_cmd_, 0) != VK_SUCCESS) return false;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (fn_.begin_cmd_buf(blit_cmd_, &bi) != VK_SUCCESS) return false;

        VkImageMemoryBarrier pre[2]{};
        pre[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        pre[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        pre[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        pre[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        pre[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        pre[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre[0].image = offscreen_images_[image_index];
        pre[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        pre[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        pre[1].srcAccessMask = 0;
        pre[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        pre[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        pre[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        pre[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre[1].image = dst_image;
        pre[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        fn_.cmd_barrier(blit_cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                        0, nullptr, 2, pre);

        VkImageBlit region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.srcOffsets[1] = {static_cast<std::int32_t>(extent_.width), static_cast<std::int32_t>(extent_.height), 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstOffsets[1] = {static_cast<std::int32_t>(surface_extent_.width),
                                static_cast<std::int32_t>(surface_extent_.height), 1};
        fn_.cmd_blit(blit_cmd_, offscreen_images_[image_index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);

        VkImageMemoryBarrier post[2]{};
        post[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        post[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        post[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        post[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        post[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        post[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        post[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        post[0].image = offscreen_images_[image_index];
        post[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        post[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        post[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        post[1].dstAccessMask = 0;
        post[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        post[1].newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        post[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        post[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        post[1].image = dst_image;
        post[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        fn_.cmd_barrier(blit_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                        0, nullptr, 2, post);

        if (fn_.end_cmd_buf(blit_cmd_) != VK_SUCCESS) return false;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &blit_cmd_;
        if (fn_.queue_submit(queue_, 1, &si, blit_fence_) != VK_SUCCESS) return false;
        return fn_.wait_for_fences(device_, 1, &blit_fence_, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    }

    bool rebuild(std::uint32_t width, std::uint32_t height, VkFormat preferred_format, std::uint32_t buffer_count) {
        VkSurfaceCapabilitiesKHR caps{};
        if (fn_.surface_caps(physical_, surface_, &caps) != VK_SUCCESS) return false;

        std::uint32_t fmt_count = 0;
        fn_.surface_formats(physical_, surface_, &fmt_count, nullptr);
        if (fmt_count == 0) return false;
        std::vector<VkSurfaceFormatKHR> formats(fmt_count);
        fn_.surface_formats(physical_, surface_, &fmt_count, formats.data());

        VkSurfaceFormatKHR chosen = formats[0];
        for (const auto& sf : formats) {
            if (sf.format == preferred_format && sf.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosen = sf;
                break;
            }
        }

        VkExtent2D ext{};
        if (caps.currentExtent.width != UINT32_MAX && caps.currentExtent.width > 0 && caps.currentExtent.height > 0) {
            ext = caps.currentExtent;
        } else {
            ext.width = std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
            ext.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
        }

        std::uint32_t count = std::max(buffer_count, caps.minImageCount);
        if (caps.maxImageCount > 0) count = std::min(count, caps.maxImageCount);

        VkPresentModeKHR chosen_present_mode = VK_PRESENT_MODE_FIFO_KHR;
        if (allow_tearing_ && fn_.surface_present_modes) {
            std::uint32_t mode_count = 0;
            if (fn_.surface_present_modes(physical_, surface_, &mode_count, nullptr) == VK_SUCCESS && mode_count > 0) {
                std::vector<VkPresentModeKHR> modes(mode_count);
                if (fn_.surface_present_modes(physical_, surface_, &mode_count, modes.data()) == VK_SUCCESS) {
                    bool has_immediate = false;
                    bool has_mailbox = false;
                    for (VkPresentModeKHR m : modes) {
                        if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) has_immediate = true;
                        if (m == VK_PRESENT_MODE_MAILBOX_KHR) has_mailbox = true;
                    }
                    if (has_immediate) chosen_present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                    else if (has_mailbox) chosen_present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
                }
            }
        }

        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface = surface_;
        ci.minImageCount = count;
        ci.imageFormat = chosen.format;
        ci.imageColorSpace = chosen.colorSpace;
        ci.imageExtent = ext;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
            ci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = caps.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = chosen_present_mode;
        ci.clipped = VK_TRUE;
        ci.oldSwapchain = swapchain_;

        VkSwapchainKHR next = VK_NULL_HANDLE;
        if (fn_.create_swapchain(device_, &ci, nullptr, &next) != VK_SUCCESS) return false;
        if (swapchain_) fn_.destroy_swapchain(device_, swapchain_, nullptr);
        destroy_offscreen();
        swapchain_ = next;
        present_mode_ = chosen_present_mode;
        format_ = (preferred_format != VK_FORMAT_UNDEFINED) ? preferred_format : chosen.format;
        surface_extent_ = ext;
        extent_ = (width > 0 && height > 0) ? VkExtent2D{width, height} : ext;

        std::uint32_t img_count = 0;
        fn_.get_images(device_, swapchain_, &img_count, nullptr);
        if (img_count == 0) return false;
        images_.resize(img_count);
        fn_.get_images(device_, swapchain_, &img_count, images_.data());
        image_initialized_.assign(img_count, false);
        const std::uint32_t requested = buffer_count ? buffer_count : 2u;
        const bool blit_needed = extent_.width != surface_extent_.width || extent_.height != surface_extent_.height ||
                                 format_ != chosen.format;
        if (blit_needed || img_count != requested) {
            // A varying native count (Vulkan may return more images than minImageCount, and minImageCount may
            // exceed the app's request) cannot be mapped one-to-one onto DXGI backbuffers: the app owns one
            // persistent resource per BufferCount index. Expose the requested count as offscreen images and
            // blit the presented slot into the acquired native image instead.
            if (!create_offscreen_backbuffers(requested)) return false;
            exposed_count_ = requested;
        } else {
            exposed_count_ = img_count;
        }
        current_index_ = 0;
        next_back_buffer_ = 0;
        acquired_index_ = 0;
        return true;
    }

    SwapchainFunctions fn_{};
    bool loaded_ = false;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    std::mutex& submit_mutex_;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFence acquire_fence_ = VK_NULL_HANDLE;
    VkFence blit_fence_ = VK_NULL_HANDLE;
    VkCommandPool blit_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer blit_cmd_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    bool allow_tearing_ = false;
    VkExtent2D surface_extent_{0, 0};
    VkExtent2D extent_{0, 0};
    std::vector<VkImage> images_;
    std::vector<bool> image_initialized_;
    std::vector<VkImage> offscreen_images_;
    std::vector<VkDeviceMemory> offscreen_memory_;
    std::uint32_t exposed_count_ = 0;
    std::uint32_t current_index_ = 0;
    std::uint32_t acquired_index_ = 0;
    std::uint32_t next_back_buffer_ = 0;
};

}  // namespace leonida
