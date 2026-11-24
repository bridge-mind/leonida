// CopyResource (buffer-to-buffer) and CopyTexture (buffer <-> image, including depth24_as_float) apply
// ResourceBinding::buffer_offset so suballocated arena buffers read and write their own range of the backing VkBuffer.
// No Vulkan loader, GPU or game data: fake Vulkan callbacks capture the emitted copy parameters.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

std::vector<VkBufferCopy> buffer_copies;
std::vector<VkBufferImageCopy> b2i_copies;
std::vector<VkBufferImageCopy> i2b_copies;
std::vector<leonida::Depth24Job> depth24_jobs;

VKAPI_ATTR void VKAPI_CALL noop_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                        std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                        const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}

VKAPI_ATTR void VKAPI_CALL rec_copy_buffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t count,
                                           const VkBufferCopy* regions) {
    for (std::uint32_t i = 0; i < count; ++i) buffer_copies.push_back(regions[i]);
}

VKAPI_ATTR void VKAPI_CALL rec_copy_b2i(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t count,
                                        const VkBufferImageCopy* regions) {
    for (std::uint32_t i = 0; i < count; ++i) b2i_copies.push_back(regions[i]);
}

VKAPI_ATTR void VKAPI_CALL rec_copy_i2b(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t count,
                                        const VkBufferImageCopy* regions) {
    for (std::uint32_t i = 0; i < count; ++i) i2b_copies.push_back(regions[i]);
}

VKAPI_ATTR void VKAPI_CALL noop_copy_image(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                                           std::uint32_t, const VkImageCopy*) {}

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

    ReplayFunctions vk{};
    vk.pipeline_barrier = noop_barrier;
    vk.copy_buffer = rec_copy_buffer;
    vk.copy_buffer_to_image = rec_copy_b2i;
    vk.copy_image_to_buffer = rec_copy_i2b;
    vk.copy_image = noop_copy_image;
    vk.depth24_prepare = [](Depth24Job& job, std::vector<Depth24Job>& parts) {
        depth24_jobs.push_back(job);
        const std::uint64_t aw = std::uint64_t(job.width) * job.app_texel_words();
        const std::uint64_t span =
            ((std::uint64_t(job.slices) - 1) * job.app_slice_words +
             (std::uint64_t(job.height) - 1) * job.app_row_words + aw) * 4;
        if (job.app_offset % 4 || job.app_offset > job.app_size || span > job.app_size - job.app_offset) return false;
        job.scratch = handle<VkBuffer>(0x9000);
        job.scratch_offset = 0;
        parts.assign(1, job);
        return true;
    };
    vk.depth24_run = [](VkCommandBuffer, const Depth24Job&) {};

    ResourceBinding src_buf{};
    src_buf.buffer = handle<VkBuffer>(1);
    src_buf.size = 4096;
    src_buf.buffer_offset = (UINT64_C(1) << 33) + 512;

    ResourceBinding dst_buf{};
    dst_buf.buffer = handle<VkBuffer>(2);
    dst_buf.size = 4096;
    dst_buf.buffer_offset = (UINT64_C(1) << 34) + 1024;
    dst_buf.host_readable = true;

    ResourceBinding rgba_img{};
    rgba_img.image = handle<VkImage>(3);
    rgba_img.format = VK_FORMAT_R8G8B8A8_UNORM;
    rgba_img.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    rgba_img.width = 16;
    rgba_img.height = 16;
    rgba_img.mip_levels = 1;
    rgba_img.array_layers = 1;
    rgba_img.depth = 1;
    rgba_img.samples = 1;

    ResourceBinding d24_img{};
    d24_img.image = handle<VkImage>(4);
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
    rgb96_img.image = handle<VkImage>(5);
    rgb96_img.format = VK_FORMAT_R32G32B32A32_UINT;
    rgb96_img.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    rgb96_img.width = 17;
    rgb96_img.height = 3;
    rgb96_img.mip_levels = 1;
    rgb96_img.array_layers = 1;
    rgb96_img.depth = 1;
    rgb96_img.samples = 1;
    rgb96_img.rgb96 = true;

    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        switch (id) {
        case 1: return &src_buf;
        case 2: return &dst_buf;
        case 3: return &rgba_img;
        case 4: return &d24_img;
        case 5: return &rgb96_img;
        default: return nullptr;
        }
    };

    const auto run = [&](const CommandList& list) {
        buffer_copies.clear();
        b2i_copies.clear();
        i2b_copies.clear();
        depth24_jobs.clear();
        return replay(list, VK_NULL_HANDLE, resolve, vk);
    };

    // 1. CopyResource between distinct backing VkBuffers with 64-bit offsets.
    {
        CommandList list;
        expect(list.copy_resource({2, 1}) && list.close(), "record CopyResource");
        expect(run(list) == ReplayResult::success, "CopyResource distinct buffers replays");
        expect(buffer_copies.size() == 1 && buffer_copies[0].srcOffset == src_buf.buffer_offset &&
                   buffer_copies[0].dstOffset == dst_buf.buffer_offset && buffer_copies[0].size == src_buf.size,
               "CopyResource applies src and dst buffer_offset");
    }

    // 2. CopyResource within a single shared backing VkBuffer (arena): disjoint & adjacent succeed, overlap fails.
    {
        const auto saved_src = src_buf, saved_dst = dst_buf;
        dst_buf.buffer = src_buf.buffer;
        src_buf.size = dst_buf.size = 256;
        src_buf.buffer_offset = 512;
        dst_buf.buffer_offset = 1024;
        CommandList list;
        expect(list.copy_resource({2, 1}) && list.close(), "record arena CopyResource");
        expect(run(list) == ReplayResult::success, "CopyResource disjoint arena slices replays");
        expect(buffer_copies.size() == 1 && buffer_copies[0].srcOffset == 512 && buffer_copies[0].dstOffset == 1024 &&
                   buffer_copies[0].size == 256,
               "CopyResource arena uses slice offsets");

        dst_buf.buffer_offset = 768;  // immediately adjacent [512, 768) and [768, 1024)
        expect(run(list) == ReplayResult::success, "CopyResource adjacent arena slices replays");
        expect(buffer_copies.size() == 1 && buffer_copies[0].srcOffset == 512 && buffer_copies[0].dstOffset == 768,
               "CopyResource adjacent arena slices keep exact offsets");

        dst_buf.buffer_offset = 640;  // overlaps [512, 768)
        expect(run(list) == ReplayResult::invalid_copy, "CopyResource rejects overlapping arena slices");
        dst_buf.buffer_offset = 512;  // identical offset in same VkBuffer
        expect(run(list) == ReplayResult::invalid_copy, "CopyResource rejects aliased arena slices");

        // Zero-size buffer copy and buffer_offset + size overflow must be rejected cleanly.
        src_buf.size = dst_buf.size = 0;
        dst_buf.buffer = handle<VkBuffer>(2);
        expect(run(list) == ReplayResult::invalid_copy, "CopyResource rejects zero-size buffers");

        src_buf.size = dst_buf.size = 64;
        src_buf.buffer_offset = std::numeric_limits<VkDeviceSize>::max() - 32;
        dst_buf.buffer_offset = 0;
        expect(run(list) == ReplayResult::invalid_copy, "CopyResource rejects src buffer_offset overflow");

        src_buf.buffer_offset = 0;
        dst_buf.buffer_offset = std::numeric_limits<VkDeviceSize>::max() - 32;
        expect(run(list) == ReplayResult::invalid_copy, "CopyResource rejects dst buffer_offset overflow");
        src_buf = saved_src;
        dst_buf = saved_dst;
    }

    // 3. CopyTexture (ordinary RGBA8 upload and readback) adds buffer_offset to region.bufferOffset.
    {
        CommandList up;
        const CopyLocation fp_up{1, true, 0, 512, 8, 4, 1, 256};
        const CopyLocation tex{3, false, 0, 0, 0, 0, 0, 0};
        const CopyBox box{2, 1, 0, 6, 3, 1};
        expect(up.copy_texture({tex, 1, 2, 0, fp_up, true, box}) && up.close(), "record RGBA8 upload");
        expect(run(up) == ReplayResult::success, "RGBA8 upload with buffer_offset replays");
        const VkDeviceSize expected_up = src_buf.buffer_offset + 512 + 1 * 256 + 2 * 4;
        expect(b2i_copies.size() == 1 && b2i_copies[0].bufferOffset == expected_up,
               "RGBA8 upload applies src buffer_offset");

        CommandList rb;
        const CopyLocation fp_rb{2, true, 0, 1024, 8, 4, 1, 256};
        const CopyBox rb_box{0, 0, 0, 4, 2, 1};
        expect(rb.copy_texture({fp_rb, 1, 2, 0, tex, true, rb_box}) && rb.close(), "record RGBA8 readback");
        expect(run(rb) == ReplayResult::success, "RGBA8 readback with buffer_offset replays");
        const VkDeviceSize expected_rb = dst_buf.buffer_offset + 1024 + 2 * 256 + 1 * 4;
        expect(i2b_copies.size() == 1 && i2b_copies[0].bufferOffset == expected_rb,
               "RGBA8 readback applies dst buffer_offset");

        // Misaligned backing buffer_offset (2 % 4 != 0) and buffer_offset + size overflow must fail.
        const auto saved_src = src_buf;
        src_buf.buffer_offset = 2;
        CommandList misaligned_backing;
        const CopyLocation fp_zero{1, true, 0, 0, 4, 2, 1, 256};
        expect(misaligned_backing.copy_texture({tex, 0, 0, 0, fp_zero, false, {}}) && misaligned_backing.close(),
               "record misaligned backing upload");
        expect(run(misaligned_backing) == ReplayResult::unsupported,
               "CopyTexture rejects misaligned backing buffer_offset");

        // Resource-relative stencil offset misaligned (1 % 4 != 0) even when buffer_offset + offset is 4-byte aligned.
        src_buf.buffer_offset = 3;
        CommandList misaligned_rel;
        const CopyLocation stencil_tex{4, false, 1, 0, 0, 0, 0, 0};  // plane 1 = stencil (1 byte/texel, align 4)
        const CopyLocation fp_stencil{1, true, 0, 0, 8, 2, 1, 256};
        const CopyBox stencil_box{1, 0, 0, 5, 2, 1};
        expect(misaligned_rel.copy_texture({stencil_tex, 0, 0, 0, fp_stencil, true, stencil_box}) &&
                   misaligned_rel.close(),
               "record misaligned relative stencil upload");
        expect(run(misaligned_rel) == ReplayResult::unsupported,
               "CopyTexture rejects misaligned relative footprint offset");

        src_buf.buffer_offset = std::numeric_limits<VkDeviceSize>::max() - 128;
        expect(run(misaligned_backing) == ReplayResult::invalid_copy,
               "CopyTexture rejects buffer_offset + size overflow");
        src_buf = saved_src;
    }

    // 4. CopyTexture on depth24_as_float passes buffer_offset + offset and buffer_offset + size to Depth24Converter.
    {
        const auto saved_src = src_buf, saved_dst = dst_buf;
        src_buf.buffer_offset = 65536;
        src_buf.size = 1536;
        dst_buf.buffer_offset = 131072 + 4;  // 4-byte aligned, non-256-aligned
        dst_buf.size = 1536;

        const CopyLocation d24_tex{4, false, 0, 0, 0, 0, 0, 0};
        const CopyLocation fp_up{1, true, 0, 512, 8, 2, 1, 256};
        CommandList up;
        expect(up.copy_texture({d24_tex, 0, 0, 0, fp_up, false, {}}) && up.close(), "record depth24 upload");
        expect(run(up) == ReplayResult::success, "depth24 upload with buffer_offset replays");
        expect(depth24_jobs.size() == 1 && depth24_jobs[0].to_float &&
                   depth24_jobs[0].app_offset == src_buf.buffer_offset + 512 &&
                   depth24_jobs[0].app_size == src_buf.buffer_offset + src_buf.size,
               "depth24 upload passes backing offset and backing end size");

        const CopyLocation fp_rb{2, true, 0, 512, 8, 2, 1, 256};
        const CopyBox rb_box{0, 0, 0, 8, 2, 1};
        CommandList rb;
        expect(rb.copy_texture({fp_rb, 0, 0, 0, d24_tex, true, rb_box}) && rb.close(), "record depth24 readback");
        expect(run(rb) == ReplayResult::success, "depth24 readback with buffer_offset replays");
        expect(depth24_jobs.size() == 1 && !depth24_jobs[0].to_float &&
                   depth24_jobs[0].app_offset == dst_buf.buffer_offset + 512 &&
                   depth24_jobs[0].app_size == dst_buf.buffer_offset + dst_buf.size,
               "depth24 readback passes backing offset and backing end size");

        src_buf.buffer_offset = 2;  // violates 4-byte depth alignment
        expect(run(up) == ReplayResult::unsupported, "depth24 upload rejects misaligned backing buffer_offset");
        src_buf = saved_src;
        dst_buf = saved_dst;
    }

    // 5. CopyTexture on rgb96 passes buffer_offset + offset and buffer_offset + size to Depth24Converter.
    {
        const auto saved_src = src_buf, saved_dst = dst_buf;
        src_buf.buffer_offset = 2048;
        src_buf.size = 1536;
        dst_buf.buffer_offset = 4096 + 4;  // 4-byte aligned, non-256-aligned
        dst_buf.size = 1536;

        const CopyLocation rgb96_tex{5, false, 0, 0, 0, 0, 0, 0};
        const CopyLocation fp_up{1, true, 0, 512, 17, 3, 1, 256};
        CommandList up;
        expect(up.copy_texture({rgb96_tex, 0, 0, 0, fp_up, false, {}}) && up.close(), "record rgb96 upload");
        expect(run(up) == ReplayResult::success, "rgb96 upload with buffer_offset replays");
        expect(depth24_jobs.size() == 1 && depth24_jobs[0].rgb96 && depth24_jobs[0].to_float &&
                   depth24_jobs[0].app_offset == src_buf.buffer_offset + 512 &&
                   depth24_jobs[0].app_size == src_buf.buffer_offset + src_buf.size &&
                   depth24_jobs[0].app_row_words == 64,
               "rgb96 upload passes backing offset and backing end size");

        const CopyLocation fp_rb{2, true, 0, 512, 17, 3, 1, 256};
        CommandList rb;
        expect(rb.copy_texture({fp_rb, 0, 0, 0, rgb96_tex, false, {}}) && rb.close(), "record rgb96 readback");
        expect(run(rb) == ReplayResult::success, "rgb96 readback with buffer_offset replays");
        expect(depth24_jobs.size() == 1 && depth24_jobs[0].rgb96 && !depth24_jobs[0].to_float &&
                   depth24_jobs[0].app_offset == dst_buf.buffer_offset + 512 &&
                   depth24_jobs[0].app_size == dst_buf.buffer_offset + dst_buf.size &&
                   depth24_jobs[0].app_row_words == 64,
               "rgb96 readback passes backing offset and backing end size");

        src_buf.buffer_offset = 2;  // violates 4-byte rgb96 alignment
        expect(run(up) == ReplayResult::unsupported, "rgb96 upload rejects misaligned backing buffer_offset");
        src_buf = saved_src;
        dst_buf = saved_dst;
    }

    if (failures) return 1;
    std::printf("replay_copy_offset: CopyResource and CopyTexture apply ResourceBinding::buffer_offset\n");
    return 0;
}
