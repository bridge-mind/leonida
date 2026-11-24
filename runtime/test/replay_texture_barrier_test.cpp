// emit of an enhanced texture barrier (TextureBarrier): one VkImageMemoryBarrier2 for each layout piece, in one
// vkCmdPipelineBarrier2 call.
//   - A range in one layout is one piece: emit makes no heap allocation for it.
//   - A range whose mips are in different layouts is one piece for each mip: all of them are in the same call,
//     each with its own old layout.
// vkCmdPipelineBarrier2 is a function that records its barriers. No Vulkan loader, GPU or game data.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <malloc.h>
#endif
#include <new>
#include <vector>

static std::size_t g_allocations = 0;

void* operator new(std::size_t size) {
    ++g_allocations;
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
// The aligned forms, so that an allocation of an over-aligned type is counted also.
void* operator new(std::size_t size, std::align_val_t alignment) {
    ++g_allocations;
    const std::size_t align = static_cast<std::size_t>(alignment);
#ifdef _WIN32
    if (void* p = _aligned_malloc(size ? size : 1, align)) return p;
#else
    if (void* p = std::aligned_alloc(align, (size + align - 1) / align * align + (size ? 0 : align))) return p;
#endif
    throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t alignment) { return operator new(size, alignment); }
void operator delete(void* p, std::align_val_t) noexcept {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
void operator delete[](void* p, std::align_val_t alignment) noexcept { operator delete(p, alignment); }
void operator delete(void* p, std::size_t, std::align_val_t alignment) noexcept { operator delete(p, alignment); }
void operator delete[](void* p, std::size_t, std::align_val_t alignment) noexcept { operator delete(p, alignment); }

namespace {

using namespace leonida;

// The barriers of every vkCmdPipelineBarrier2 call, and the number of calls. The storage is reserved before the
// measured emit, so that the record itself does not allocate there.
std::vector<VkImageMemoryBarrier2> g_barriers;
std::uint32_t g_calls = 0;

void VKAPI_CALL record_barrier2(VkCommandBuffer, const VkDependencyInfo* info) {
    ++g_calls;
    for (std::uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i)
        g_barriers.push_back(info->pImageMemoryBarriers[i]);
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    namespace y = d3d12_sync;
    namespace a = d3d12_access;
    namespace l = d3d12_layout;
    constexpr ResourceId id = 1;
    constexpr std::uint32_t mips = 3;
    constexpr VkImageLayout shader_read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    constexpr VkImageLayout transfer_src = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    constexpr VkImageLayout transfer_dst = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    constexpr VkImageLayout attachment = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    ResourceBinding image{};
    image.image = reinterpret_cast<VkImage>(std::uintptr_t(0x1000));
    image.format = VK_FORMAT_R8G8B8A8_UNORM;
    image.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    image.width = image.height = 64;
    image.mip_levels = mips;
    const auto resolve = [&](ResourceId r) { return r == id ? &image : nullptr; };

    ReplayFunctions vk{};
    vk.pipeline_barrier2 = record_barrier2;
    LayoutTracker tracker;
    tracker.track(id, image, shader_read);
    const VkCommandBuffer cb = reinterpret_cast<VkCommandBuffer>(std::uintptr_t(0x10));
    const BarrierSubresourceRange all{all_subresources, 0, 0, 0, 0, 0};
    g_barriers.reserve(16);

    // 1. All mips in one layout: one barrier, and emit allocates nothing.
    {
        CommandList list;
        expect(list.barrier(TextureBarrier{y::pixel_shading, y::copy, a::shader_resource, a::copy_dest,
                                           l::shader_resource, l::copy_dest, id, all, 0}) &&
                   list.close(),
               "record a barrier over one layout");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare a barrier over one layout");
        const std::size_t before = g_allocations;
        emit(prepared, cb, vk, &tracker);
        const std::size_t allocations = g_allocations - before;
        if (allocations)
            std::fprintf(stderr, "emit made %zu heap allocations for a barrier with one piece\n", allocations);
        expect(allocations == 0, "emit allocates nothing for a barrier with one piece");
        expect(g_calls == 1 && g_barriers.size() == 1, "one call with one image barrier");
        if (g_barriers.size() == 1) {
            const VkImageMemoryBarrier2& b = g_barriers[0];
            expect(b.sType == VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 && b.pNext == nullptr, "barrier header");
            expect(b.image == image.image && b.oldLayout == shader_read && b.newLayout == transfer_dst,
                   "image and layouts of the one barrier");
            expect(b.subresourceRange.baseMipLevel == 0 && b.subresourceRange.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT,
                   "range of the one barrier");
            expect(b.srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED && b.dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED,
                   "no queue family transfer");
            expect(b.srcStageMask != 0 && b.dstStageMask != 0 && b.srcAccessMask != 0 && b.dstAccessMask != 0,
                   "stages and access of the one barrier");
        }
        for (std::uint32_t mip = 0; mip < mips; ++mip)
            expect(tracker.layout(id, mip) == transfer_dst, "the tracker has the new layout");
    }

    // 2. Three mips in three layouts: three barriers in one call, each from the layout of its mip.
    {
        const VkImageLayout old_layouts[mips] = {shader_read, transfer_src, transfer_dst};
        for (std::uint32_t mip = 0; mip < mips; ++mip)
            tracker.set(id, {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1}, old_layouts[mip]);
        g_barriers.clear();
        g_calls = 0;
        CommandList list;
        expect(list.barrier(TextureBarrier{y::pixel_shading, y::render_target, a::shader_resource, a::render_target,
                                           l::shader_resource, l::render_target, id, all, 0}) &&
                   list.close(),
               "record a barrier over three layouts");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare a barrier over three layouts");
        emit(prepared, cb, vk, &tracker);
        expect(g_calls == 1 && g_barriers.size() == mips, "one call with three image barriers");
        for (std::uint32_t mip = 0; mip < mips && g_barriers.size() == mips; ++mip) {
            const VkImageMemoryBarrier2& b = g_barriers[mip];
            expect(b.sType == VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 && b.pNext == nullptr &&
                       b.image == image.image,
                   "header and image of each barrier");
            expect(b.oldLayout == old_layouts[mip] && b.newLayout == attachment, "layouts of each barrier");
            expect(b.subresourceRange.baseMipLevel == mip && b.subresourceRange.levelCount == 1 &&
                       b.subresourceRange.baseArrayLayer == 0 && b.subresourceRange.layerCount == 1,
                   "range of each barrier");
            expect(b.srcStageMask == g_barriers[0].srcStageMask && b.dstStageMask == g_barriers[0].dstStageMask &&
                       b.srcAccessMask == g_barriers[0].srcAccessMask &&
                       b.dstAccessMask == g_barriers[0].dstAccessMask && b.srcStageMask != 0 && b.dstStageMask != 0,
                   "stages and access of each barrier");
            expect(tracker.layout(id, mip) == attachment, "the tracker has the new layout for each mip");
        }
    }

    if (failures) return 1;
    std::printf("replay_texture_barrier_test: OK\n");
    return 0;
}
