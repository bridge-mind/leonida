// PreparedList::wrote_readback and readback_src_stages follow the actual Vulkan stage writing a host_readable
// buffer: COMPUTE_SHADER_BIT for compute-converted CopyTexture readbacks (depth24_as_float depth plane, rgb96),
// COPY_BIT for ordinary texture/stencil/buffer copies, and TRANSFER_BIT for buffer UAV clears (CUSTOM write-back).
// No Vulkan loader, GPU or game data.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

template<typename T>
T available() {
    return reinterpret_cast<T>(+[]() {});
}

template<typename T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
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
    using namespace leonida;

    bool allow_depth24 = true;
    ReplayFunctions vk{};
    vk.pipeline_barrier = available<PFN_vkCmdPipelineBarrier>();
    vk.copy_buffer = available<PFN_vkCmdCopyBuffer>();
    vk.copy_buffer_to_image = available<PFN_vkCmdCopyBufferToImage>();
    vk.copy_image_to_buffer = available<PFN_vkCmdCopyImageToBuffer>();
    vk.copy_image = available<PFN_vkCmdCopyImage>();
    vk.fill_buffer = available<PFN_vkCmdFillBuffer>();
    vk.update_buffer = available<PFN_vkCmdUpdateBuffer>();
    vk.depth24_prepare = [&](Depth24Job& job, std::vector<Depth24Job>& parts) {
        if (!allow_depth24) return false;
        job.scratch = handle<VkBuffer>(0x9000);
        job.scratch_offset = 0;
        parts.assign(1, job);
        return true;
    };
    vk.depth24_run = [](VkCommandBuffer, const Depth24Job&) {};

    ResourceBinding upload_buf{};
    upload_buf.buffer = handle<VkBuffer>(1);
    upload_buf.size = 4096;

    ResourceBinding readback_buf{};
    readback_buf.buffer = handle<VkBuffer>(2);
    readback_buf.size = 4096;
    readback_buf.host_readable = true;

    ResourceBinding default_buf{};
    default_buf.buffer = handle<VkBuffer>(3);
    default_buf.size = 4096;

    ResourceBinding rgba_img{};
    rgba_img.image = handle<VkImage>(4);
    rgba_img.format = VK_FORMAT_R8G8B8A8_UNORM;
    rgba_img.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    rgba_img.width = 8;
    rgba_img.height = 4;
    rgba_img.mip_levels = 1;
    rgba_img.array_layers = 1;
    rgba_img.depth = 1;
    rgba_img.samples = 1;

    ResourceBinding d24_img{};
    d24_img.image = handle<VkImage>(5);
    d24_img.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    d24_img.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    d24_img.width = 8;
    d24_img.height = 4;
    d24_img.mip_levels = 1;
    d24_img.array_layers = 1;
    d24_img.depth = 1;
    d24_img.samples = 1;
    d24_img.depth24 = true;
    d24_img.depth24_as_float = true;

    ResourceBinding rgb96_img{};
    rgb96_img.image = handle<VkImage>(6);
    rgb96_img.format = VK_FORMAT_R32G32B32A32_UINT;
    rgb96_img.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    rgb96_img.width = 8;
    rgb96_img.height = 4;
    rgb96_img.mip_levels = 1;
    rgb96_img.array_layers = 1;
    rgb96_img.depth = 1;
    rgb96_img.samples = 1;
    rgb96_img.rgb96 = true;

    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        switch (id) {
        case 1: return &upload_buf;
        case 2: return &readback_buf;
        case 3: return &default_buf;
        case 4: return &rgba_img;
        case 5: return &d24_img;
        case 6: return &rgb96_img;
        default: return nullptr;
        }
    };

    const CopyLocation fp_up{1, true, 0, 0, 8, 4, 1, 256};
    const CopyLocation fp_rb{2, true, 0, 0, 8, 4, 1, 256};
    const CopyLocation fp_def{3, true, 0, 0, 8, 4, 1, 256};
    const CopyLocation loc_rgba{4, false, 0, 0, 0, 0, 0, 0};
    const CopyLocation loc_d24_depth{5, false, 0, 0, 0, 0, 0, 0};
    const CopyLocation loc_d24_stencil{5, false, 1, 0, 0, 0, 0, 0};
    const CopyLocation loc_rgb96{6, false, 0, 0, 0, 0, 0, 0};

    PreparedList out;

    // 1. depth24_as_float depth-plane readback writes to the readback buffer via the compute pass.
    {
        CommandList list;
        expect(list.copy_texture({fp_rb, 0, 0, 0, loc_d24_depth, false, {}}) && list.close(),
               "record depth24 depth readback");
        expect(prepare(list, resolve, vk, out) == ReplayResult::success, "prepare depth24 depth readback");
        expect(out.wrote_readback && out.readback_src_stages == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               "depth24 depth readback reports COMPUTE_SHADER_BIT");
    }

    // 2. depth24_as_float stencil-plane readback uses plain vkCmdCopyImageToBuffer (COPY_BIT).
    {
        CommandList list;
        expect(list.copy_texture({fp_rb, 0, 0, 0, loc_d24_stencil, false, {}}) && list.close(),
               "record depth24 stencil readback");
        expect(prepare(list, resolve, vk, out) == ReplayResult::success, "prepare depth24 stencil readback");
        expect(out.wrote_readback && out.readback_src_stages == VK_PIPELINE_STAGE_2_COPY_BIT,
               "depth24 stencil readback reports COPY_BIT");
    }

    // 3. rgb96 readback writes to the readback buffer via the compute pass.
    {
        CommandList list;
        expect(list.copy_texture({fp_rb, 0, 0, 0, loc_rgb96, false, {}}) && list.close(), "record rgb96 readback");
        expect(prepare(list, resolve, vk, out) == ReplayResult::success, "prepare rgb96 readback");
        expect(out.wrote_readback && out.readback_src_stages == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               "rgb96 readback reports COMPUTE_SHADER_BIT");
    }

    // 4. Ordinary RGBA8 readback and CopyResource buffer-to-buffer keep COPY_BIT.
    {
        CommandList rgba_rb;
        expect(rgba_rb.copy_texture({fp_rb, 0, 0, 0, loc_rgba, false, {}}) && rgba_rb.close(),
               "record RGBA8 readback");
        expect(prepare(rgba_rb, resolve, vk, out) == ReplayResult::success, "prepare RGBA8 readback");
        expect(out.wrote_readback && out.readback_src_stages == VK_PIPELINE_STAGE_2_COPY_BIT,
               "RGBA8 readback reports COPY_BIT");

        CommandList buf_copy;
        expect(buf_copy.copy_resource({2, 1}) && buf_copy.close(), "record CopyResource to readback");
        expect(prepare(buf_copy, resolve, vk, out) == ReplayResult::success, "prepare CopyResource to readback");
        expect(out.wrote_readback && out.readback_src_stages == VK_PIPELINE_STAGE_2_COPY_BIT,
               "CopyResource to readback reports COPY_BIT");
    }

    // 5. Mixed list combines COMPUTE_SHADER_BIT and COPY_BIT.
    {
        CommandList mixed;
        expect(mixed.copy_texture({fp_rb, 0, 0, 0, loc_d24_depth, false, {}}), "record mixed depth24");
        expect(mixed.copy_texture({fp_rb, 0, 0, 0, loc_rgba, false, {}}) && mixed.close(), "record mixed rgba");
        expect(prepare(mixed, resolve, vk, out) == ReplayResult::success, "prepare mixed readback");
        expect(out.wrote_readback &&
                   out.readback_src_stages == (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT),
               "mixed depth24 + RGBA8 readback reports COMPUTE_SHADER_BIT | COPY_BIT");
    }

    // 6. Uploads and readbacks into non-host_readable buffers do not request a host barrier.
    {
        CommandList up;
        expect(up.copy_texture({loc_d24_depth, 0, 0, 0, fp_up, false, {}}), "record depth24 upload");
        expect(up.copy_texture({loc_rgb96, 0, 0, 0, fp_up, false, {}}) && up.close(), "record rgb96 upload");
        expect(prepare(up, resolve, vk, out) == ReplayResult::success, "prepare uploads");
        expect(!out.wrote_readback && out.readback_src_stages == 0, "uploads leave wrote_readback false");

        CommandList non_rb;
        expect(non_rb.copy_texture({fp_def, 0, 0, 0, loc_d24_depth, false, {}}), "record non-readback depth24");
        expect(non_rb.copy_texture({fp_def, 0, 0, 0, loc_rgb96, false, {}}) && non_rb.close(),
               "record non-readback rgb96");
        expect(prepare(non_rb, resolve, vk, out) == ReplayResult::success, "prepare non-readback copies");
        expect(!out.wrote_readback && out.readback_src_stages == 0,
               "copies into non-host_readable buffers leave wrote_readback false");
    }

    // 7. Failed copy preparation does not mark wrote_readback or readback_src_stages.
    {
        allow_depth24 = false;
        CommandList failed;
        expect(failed.copy_texture({fp_rb, 0, 0, 0, loc_rgb96, false, {}}) && failed.close(),
               "record failing rgb96 readback");
        expect(prepare(failed, resolve, vk, out) == ReplayResult::unsupported, "failing rgb96 returns unsupported");
        expect(!out.wrote_readback && out.readback_src_stages == 0,
               "failed copy preparation leaves wrote_readback false");
        allow_depth24 = true;
    }

    // 8. ClearUnorderedAccess on a host_readable buffer (CUSTOM write-back heap) flushes TRANSFER_BIT to HOST.
    {
        CommandList clear_rb;
        expect(clear_rb.clear_unordered_access({2, true, 0, 0, 256, 4, 0, 0, 1}, false, {1, 2, 3, 4}, {}) &&
                   clear_rb.close(),
               "record ClearUnorderedAccess on host_readable buffer");
        expect(prepare(clear_rb, resolve, vk, out) == ReplayResult::success, "prepare host_readable buffer UAV clear");
        expect(out.wrote_readback && out.readback_src_stages == VK_PIPELINE_STAGE_2_TRANSFER_BIT,
               "host_readable buffer UAV clear reports TRANSFER_BIT");

        // Clipped-away rect clear writes nothing, so no host barrier is needed.
        const Rect off_row{0, 1, 4, 2};
        CommandList skipped_rb;
        expect(skipped_rb.clear_unordered_access({2, true, 0, 0, 256, 4, 0, 0, 1}, false, {1, 2, 3, 4},
                                                 std::span<const Rect>(&off_row, 1)) &&
                   skipped_rb.close(),
               "record skipped ClearUnorderedAccess");
        expect(prepare(skipped_rb, resolve, vk, out) == ReplayResult::success, "prepare skipped buffer UAV clear");
        expect(!out.wrote_readback && out.readback_src_stages == 0,
               "skipped buffer UAV clear leaves wrote_readback false");

        // Non-host_readable buffer UAV clear leaves wrote_readback false.
        CommandList clear_def;
        expect(clear_def.clear_unordered_access({3, true, 0, 0, 256, 4, 0, 0, 1}, false, {1, 2, 3, 4}, {}) &&
                   clear_def.close(),
               "record ClearUnorderedAccess on default buffer");
        expect(prepare(clear_def, resolve, vk, out) == ReplayResult::success, "prepare default buffer UAV clear");
        expect(!out.wrote_readback && out.readback_src_stages == 0,
               "non-host_readable buffer UAV clear leaves wrote_readback false");
    }

    if (failures) return 1;
    std::printf("replay_readback_stages: depth24, rgb96, copy and UAV-clear readback stages exact\n");
    return 0;
}
