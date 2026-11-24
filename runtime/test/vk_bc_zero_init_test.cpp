// Committed block-compressed (BC*) texture zero-initialization regression (D3D-220, #1725):
// D3D12 committed textures start zeroed unless D3D12_HEAP_FLAG_CREATE_NOT_ZEROED is set. Vulkan forbids
// vkCmdClearColorImage on BC formats, so detail::initialize() must zero-fill them via vkCmdCopyBufferToImage
// from either the device zero page (vk.zero_buffer) or a zeroed transfer-scratch chunk (vk.copy_scratch).
//
// Cases (all on explicitly poisoned 0x5A backing memory when host-visible, plus default device memory):
//   1. Path A (vk.zero_buffer): CopyTextureRegion R32G32_UINT 2x2 -> BC1 16x16 at (8, 4) — untouched blocks are 0x00.
//   2. Path A (vk.zero_buffer): CopyResource R32G32_UINT 3x2 -> BC1 12x8, 2 mips — mip 1 untouched block (bytes 56..63) is 0x00.
//   3. Path A (tiny 32-byte vk.zero_buffer): 32x32 BC7 image tiled in multiple row/column chunks — all 1024 bytes are 0x00.
//   4. Path B (enable_copy_scratch, no zero_buffer — matches d3d12_copy_xfmt): both D3D-220 cases match bit for bit.
//   5. not_zeroed = true (CREATE_NOT_ZEROED): zero-fill is skipped.
// Skips with exit code 77 when no Vulkan 1.3 device with textureCompressionBC is available. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

struct OwnedImage {
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkDestroyImage destroy_image = nullptr;
    PFN_vkFreeMemory free_memory = nullptr;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    bool poisoned = false;

    ~OwnedImage() {
        if (image && destroy_image) destroy_image(device, image, nullptr);
        if (memory && free_memory) free_memory(device, memory, nullptr);
    }
    OwnedImage(const OwnedImage&) = delete;
    OwnedImage& operator=(const OwnedImage&) = delete;
    OwnedImage() = default;
};

bool create_image(const vktest::Device& d, VkFormat format, std::uint32_t width, std::uint32_t height,
                  std::uint32_t mips, VkBuffer poison_buf, OwnedImage& out) {
    out.device = d.device;
    out.destroy_image = d.fn<PFN_vkDestroyImage>("vkDestroyImage");
    out.free_memory = d.fn<PFN_vkFreeMemory>("vkFreeMemory");

    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = mips;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (d.fn<PFN_vkCreateImage>("vkCreateImage")(d.device, &info, nullptr, &out.image) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    d.fn<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(d.device, out.image, &req);
    VkPhysicalDeviceMemoryProperties mp{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &mp);

    std::uint32_t type = UINT32_MAX;
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(req.memoryTypeBits & (1u << i))) continue;
        if (type == UINT32_MAX) type = i;
        if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            type = i;
            break;
        }
    }
    if (type == UINT32_MAX) return false;

    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    if (d.fn<PFN_vkAllocateMemory>("vkAllocateMemory")(d.device, &alloc, nullptr, &out.memory) != VK_SUCCESS)
        return false;
    if (d.fn<PFN_vkBindImageMemory>("vkBindImageMemory")(d.device, out.image, out.memory, 0) != VK_SUCCESS)
        return false;

    // Pre-fill every mip on the GPU with 0x5A so stale recycled VRAM is modeled deterministically on every GPU
    // (discrete and integrated alike), before registering the image in LayoutTracker as UNDEFINED.
    if (poison_buf) {
        const bool ok = const_cast<vktest::Device&>(d).run([&](VkCommandBuffer cb) {
            VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_dst.srcQueueFamilyIndex = to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_dst.image = out.image;
            to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
            d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                                                                   nullptr, 1, &to_dst);
            std::vector<VkBufferImageCopy> regions;
            for (std::uint32_t m = 0; m < mips; ++m) {
                VkBufferImageCopy r{};
                r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1};
                r.imageExtent = {std::max(1u, width >> m), std::max(1u, height >> m), 1};
                regions.push_back(r);
            }
            d.fn<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(
                cb, poison_buf, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, std::uint32_t(regions.size()),
                regions.data());
            return true;
        });
        if (!ok) return false;
        out.poisoned = true;
    }
    return true;
}

VkDeviceMemory allocate_device_local(const vktest::Device& d, const VkMemoryRequirements& req) {
    VkPhysicalDeviceMemoryProperties mp{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &mp);
    std::uint32_t chosen = UINT32_MAX;
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(req.memoryTypeBits & (1u << i))) continue;
        if (chosen == UINT32_MAX) chosen = i;
        if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            chosen = i;
            break;
        }
    }
    if (chosen == UINT32_MAX) return VK_NULL_HANDLE;
    VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    info.allocationSize = req.size;
    info.memoryTypeIndex = chosen;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (d.fn<PFN_vkAllocateMemory>("vkAllocateMemory")(d.device, &info, nullptr, &mem) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return mem;
}

// Runs the two D3D-220 partial uncompressed -> BC1 copies on a given context/queue:
//   (a) CopyTextureRegion R32G32_UINT 2x2 -> BC1 16x16 at (8, 4)
//   (b) CopyResource R32G32_UINT 3x2 -> BC1 12x8, 2 mips
int run_d3d220_cases(const vktest::Device& d, leonida::VulkanContext& ctx, leonida::VulkanQueue& q,
                     VkBuffer poison_buf, VkBuffer up_buf, void* up_ptr, VkBuffer rb_buf, void* rb_ptr,
                     const char* label) {
    using namespace leonida;

    auto* up = static_cast<std::uint8_t*>(up_ptr);
    auto* rb = static_cast<std::uint8_t*>(rb_ptr);
    for (std::size_t i = 0; i < 4096; ++i) up[i] = static_cast<std::uint8_t>(i * 7 + 3);

    // Case (a): CopyTextureRegion R32G32_UINT 2x2 -> BC1 16x16 at (8, 4).
    // Destination has 4x4 BC1 blocks (8 bytes/block). Only blocks (2,1), (3,1), (2,2), (3,2) are written;
    // the other 12 blocks (including block (0,0) at byte 0) must be 0x00.
    {
        OwnedImage src_img, dst_img;
        EXPECT(create_image(d, VK_FORMAT_R32G32_UINT, 2, 2, 1, VK_NULL_HANDLE, src_img));
        EXPECT(create_image(d, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 16, 16, 1, poison_buf, dst_img));

        ResourceBinding src_rb{};
        src_rb.image = src_img.image;
        src_rb.format = VK_FORMAT_R32G32_UINT;
        src_rb.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        src_rb.width = 2; src_rb.height = 2; src_rb.mip_levels = 1; src_rb.array_layers = 1; src_rb.depth = 1; src_rb.samples = 1;

        ResourceBinding dst_rb{};
        dst_rb.image = dst_img.image;
        dst_rb.format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        dst_rb.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        dst_rb.width = 16; dst_rb.height = 16; dst_rb.mip_levels = 1; dst_rb.array_layers = 1; dst_rb.depth = 1; dst_rb.samples = 1;

        ctx.add_resource(10, src_rb, VK_IMAGE_LAYOUT_UNDEFINED);
        ctx.add_resource(11, dst_rb, VK_IMAGE_LAYOUT_UNDEFINED);
        std::memset(rb, 0xCD, 4096);

        CommandList list;
        CopyLocation up_loc{1, true, 0, 0, 2, 2, 1, 256};
        CopyLocation src_loc{10, false, 0, 0, 0, 0, 0, 0};
        CopyLocation dst_loc{11, false, 0, 0, 0, 0, 0, 0};
        CopyLocation rb_loc{2, true, 0, 0, 16, 16, 1, 256};
        EXPECT(list.copy_texture({src_loc, 0, 0, 0, up_loc, false, {}}));
        EXPECT(list.transition({10, all_subresources, d3d12_state::copy_dest, d3d12_state::copy_source}));
        EXPECT(list.copy_texture({dst_loc, 8, 4, 0, src_loc, false, {}}));
        EXPECT(list.transition({11, all_subresources, d3d12_state::copy_dest, d3d12_state::copy_source}));
        EXPECT(list.copy_texture({rb_loc, 0, 0, 0, dst_loc, false, {}}));
        EXPECT(list.close());

        const CommandList* ptr = &list;
        EXPECT(q.execute({&ptr, 1}) == ReplayResult::success);
        EXPECT(q.idle());

        // Check every 8-byte block across the 4x4 block grid (row pitch 256).
        for (std::uint32_t by = 0; by < 4; ++by) {
            for (std::uint32_t bx = 0; bx < 4; ++bx) {
                const std::uint8_t* got = rb + std::size_t(by) * 256 + std::size_t(bx) * 8;
                if (bx >= 2 && bx < 4 && by >= 1 && by < 3) {
                    const std::uint8_t* want = up + std::size_t(by - 1) * 256 + std::size_t(bx - 2) * 8;
                    EXPECT(std::memcmp(got, want, 8) == 0);
                } else {
                    for (int k = 0; k < 8; ++k) EXPECT(got[k] == 0x00);
                }
            }
        }
        ctx.remove_resource(10);
        ctx.remove_resource(11);
    }

    // Case (b): CopyResource R32G32_UINT 3x2 -> BC1 12x8, 2 mips.
    // Mip 0: 3x2 -> 12x8 (3x2 = 6 blocks written).
    // Mip 1: 1x1 -> 6x4 (2x1 = 2 blocks; block 0 is written, block 1 at bytes 8..15 of mip 1 is untouched -> 0x00).
    {
        OwnedImage src_img, dst_img;
        EXPECT(create_image(d, VK_FORMAT_R32G32_UINT, 3, 2, 2, VK_NULL_HANDLE, src_img));
        EXPECT(create_image(d, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 12, 8, 2, poison_buf, dst_img));

        ResourceBinding src_rb{};
        src_rb.image = src_img.image;
        src_rb.format = VK_FORMAT_R32G32_UINT;
        src_rb.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        src_rb.width = 3; src_rb.height = 2; src_rb.mip_levels = 2; src_rb.array_layers = 1; src_rb.depth = 1; src_rb.samples = 1;

        ResourceBinding dst_rb{};
        dst_rb.image = dst_img.image;
        dst_rb.format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        dst_rb.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        dst_rb.width = 12; dst_rb.height = 8; dst_rb.mip_levels = 2; dst_rb.array_layers = 1; dst_rb.depth = 1; dst_rb.samples = 1;

        ctx.add_resource(12, src_rb, VK_IMAGE_LAYOUT_UNDEFINED);
        ctx.add_resource(13, dst_rb, VK_IMAGE_LAYOUT_UNDEFINED);
        std::memset(rb, 0xCD, 4096);

        // Upload mip 0 at offset 0 (2 rows * 256 = 512 bytes) and mip 1 at offset 512 (1 row * 256).
        CommandList list;
        CopyLocation up0{1, true, 0, 0, 3, 2, 1, 256};
        CopyLocation up1{1, true, 0, 512, 1, 1, 1, 256};
        CopyLocation src0{12, false, 0, 0, 0, 0, 0, 0};
        CopyLocation src1{12, false, 1, 0, 0, 0, 0, 0};
        CopyLocation dst0{13, false, 0, 0, 0, 0, 0, 0};
        CopyLocation dst1{13, false, 1, 0, 0, 0, 0, 0};
        CopyLocation rb0{2, true, 0, 0, 12, 8, 1, 256};
        CopyLocation rb1{2, true, 0, 512, 6, 4, 1, 256};
        EXPECT(list.copy_texture({src0, 0, 0, 0, up0, false, {}}));
        EXPECT(list.copy_texture({src1, 0, 0, 0, up1, false, {}}));
        EXPECT(list.transition({12, all_subresources, d3d12_state::copy_dest, d3d12_state::copy_source}));
        EXPECT(list.copy_resource({13, 12}));
        EXPECT(list.transition({13, all_subresources, d3d12_state::copy_dest, d3d12_state::copy_source}));
        EXPECT(list.copy_texture({rb0, 0, 0, 0, dst0, false, {}}));
        EXPECT(list.copy_texture({rb1, 0, 0, 0, dst1, false, {}}));
        EXPECT(list.close());

        const CommandList* ptr = &list;
        EXPECT(q.execute({&ptr, 1}) == ReplayResult::success);
        EXPECT(q.idle());

        // Mip 0: 2 rows of 3 blocks (24 bytes per row).
        EXPECT(std::memcmp(rb + 0, up + 0, 24) == 0);
        EXPECT(std::memcmp(rb + 256, up + 256, 24) == 0);
        // Mip 1: block 0 (bytes 0..7) matches up + 512; untouched block 1 (bytes 8..15) must be 0x00.
        EXPECT(std::memcmp(rb + 512, up + 512, 8) == 0);
        for (int k = 8; k < 16; ++k) EXPECT(rb[512 + k] == 0x00);

        ctx.remove_resource(12);
        ctx.remove_resource(13);
    }

    (void)up_buf;
    (void)rb_buf;
    std::printf("bc zero-init (%s): partial CopyTextureRegion and 2-mip CopyResource ok\n", label);
    return 0;
}

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false) || !d.bc) {
        std::fprintf(stderr, "vk_bc_zero_init_test: no Vulkan 1.3 device with BC support, skipped\n");
        return 77;
    }

    void* up_ptr = nullptr;
    void* rb_ptr = nullptr;
    void* zero_ptr = nullptr;
    void* poison_ptr = nullptr;
    const VkBuffer up_buf = d.buffer(4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &up_ptr);
    const VkBuffer rb_buf = d.buffer(8192, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb_ptr);
    const VkBuffer zero_buf = d.buffer(65536, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &zero_ptr);
    const VkBuffer poison_buf = d.buffer(4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &poison_ptr);
    EXPECT(up_buf && rb_buf && zero_buf && poison_buf);
    std::memset(zero_ptr, 0, 65536);
    std::memset(poison_ptr, 0x5A, 4096);

    ResourceBinding up_rb{};
    up_rb.buffer = up_buf;
    up_rb.size = 4096;
    ResourceBinding rb_rb{};
    rb_rb.buffer = rb_buf;
    rb_rb.size = 8192;
    rb_rb.host_readable = true;

    // Path A: device arena zero page (vk.zero_buffer).
    {
        VulkanContext ctx(d.device_proc, d.device, true);
        EXPECT(ctx.loaded());
        ctx.set_zero_page(zero_buf, 0, 65536);
        ctx.add_resource(1, up_rb);
        ctx.add_resource(2, rb_rb);
        VulkanQueue q(ctx, d.queue, d.family);
        EXPECT(q.ok());
        EXPECT(run_d3d220_cases(d, ctx, q, poison_buf, up_buf, up_ptr, rb_buf, rb_ptr, "zero_buffer") == 0);

        // Multi-tile chunking: shrink zero_page to 32 bytes (2 BC7 blocks) and zero-init a 32x32 BC7 image (64 blocks).
        ctx.set_zero_page(zero_buf, 16, 32);
        OwnedImage bc7_img;
        EXPECT(create_image(d, VK_FORMAT_BC7_UNORM_BLOCK, 32, 32, 1, poison_buf, bc7_img));
        ResourceBinding bc7_rb{};
        bc7_rb.image = bc7_img.image;
        bc7_rb.format = VK_FORMAT_BC7_UNORM_BLOCK;
        bc7_rb.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        bc7_rb.width = 32; bc7_rb.height = 32; bc7_rb.mip_levels = 1; bc7_rb.array_layers = 1; bc7_rb.depth = 1; bc7_rb.samples = 1;
        ctx.add_resource(20, bc7_rb, VK_IMAGE_LAYOUT_UNDEFINED);
        std::memset(rb_ptr, 0xCD, 8192);

        CommandList list;
        EXPECT(list.transition({20, all_subresources, d3d12_state::copy_dest, d3d12_state::copy_source}));
        CopyLocation bc7_loc{20, false, 0, 0, 0, 0, 0, 0};
        CopyLocation rb_loc{2, true, 0, 0, 32, 32, 1, 256};
        EXPECT(list.copy_texture({rb_loc, 0, 0, 0, bc7_loc, false, {}}));
        EXPECT(list.close());
        const CommandList* ptr = &list;
        EXPECT(q.execute({&ptr, 1}) == ReplayResult::success);
        EXPECT(q.idle());

        auto* rb = static_cast<std::uint8_t*>(rb_ptr);
        for (std::uint32_t row = 0; row < 8; ++row)
            for (std::uint32_t x = 0; x < 8 * 16; ++x)
                EXPECT(rb[row * 256 + x] == 0x00);
        ctx.remove_resource(20);
        std::printf("bc zero-init (tiny 32-byte zero_buffer chunking on 32x32 BC7): ok\n");

        // CREATE_NOT_ZEROED (not_zeroed = true) must skip zero-fill.
        ctx.set_zero_page(zero_buf, 0, 65536);
        OwnedImage nz_img;
        EXPECT(create_image(d, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 16, 16, 1, poison_buf, nz_img));
        EXPECT(nz_img.poisoned);
        ResourceBinding nz_rb{};
        nz_rb.image = nz_img.image;
        nz_rb.format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        nz_rb.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        nz_rb.width = 16; nz_rb.height = 16; nz_rb.mip_levels = 1; nz_rb.array_layers = 1; nz_rb.depth = 1; nz_rb.samples = 1;
        nz_rb.not_zeroed = true;
        ctx.add_resource(21, nz_rb, VK_IMAGE_LAYOUT_UNDEFINED);
        std::memset(rb_ptr, 0xCD, 8192);

        CommandList nz_list;
        EXPECT(nz_list.transition({21, all_subresources, d3d12_state::copy_dest, d3d12_state::copy_source}));
        CopyLocation nz_loc{21, false, 0, 0, 0, 0, 0, 0};
        CopyLocation nz_rb_loc{2, true, 0, 0, 16, 16, 1, 256};
        EXPECT(nz_list.copy_texture({nz_rb_loc, 0, 0, 0, nz_loc, false, {}}));
        EXPECT(nz_list.close());
        const CommandList* nz_ptr = &nz_list;
        EXPECT(q.execute({&nz_ptr, 1}) == ReplayResult::success);
        EXPECT(q.idle());
        EXPECT(rb[0] == 0x5A);
        ctx.remove_resource(21);
        std::printf("bc zero-init (not_zeroed = true skips zero-fill): ok\n");
    }

    // Path B: copy_scratch + vkCmdFillBuffer fallback (no zero_buffer, matching d3d12_copy_xfmt without BDA).
    {
        VulkanContext ctx(d.device_proc, d.device, true);
        EXPECT(ctx.loaded());
        ctx.enable_copy_scratch(d.device_proc, [&](const VkMemoryRequirements& req) {
            return allocate_device_local(d, req);
        });
        ctx.add_resource(1, up_rb);
        ctx.add_resource(2, rb_rb);
        VulkanQueue q(ctx, d.queue, d.family);
        EXPECT(q.ok());
        EXPECT(run_d3d220_cases(d, ctx, q, poison_buf, up_buf, up_ptr, rb_buf, rb_ptr, "copy_scratch") == 0);
    }

    EXPECT(d.validation_errors() == 0);
    std::printf("vk_bc_zero_init_test: ok\n");
    return 0;
}
