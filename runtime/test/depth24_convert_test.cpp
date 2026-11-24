// The compute pass behind D24_UNORM_S8_UINT on a GPU without D24 (src/depth24_convert.comp, vulkan_depth24.h), checked
// on the GPU against the C++ reference of depth24.h over the whole 24-bit range:
//   1. every one of the 16777216 words, upload direction: the float must be the one nearest to d / 16777215
//      (the top byte of the word is junk and must be ignored);
//   2. the same floats back through the readback direction: every word must return unchanged, with a zero top byte;
//   3. 16777216 float bit patterns (every 256th pattern of the 32-bit range, offset by a rotating low byte, plus the
//      specials: zeros, denormals, NaNs, infinities, negatives, 1.0 and its neighbours), readback direction, against
//      depth24_quantize;
//   4. a pitched copy (row and slice strides larger than the extent, an offset into the buffer): only the texels of
//      the region change, to the right values.
// The pass is tested on the device the Vulkan loader picks (VK_LOADER_DRIVERS_SELECT='*amd*' or '*nv*'); it needs no
// GPU without D24. Steps 3 and 4 use converters with a 1 MiB and a 1 KiB storage range, so copies split into dispatches.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails the test.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"
#include "../src/vulkan_depth24.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

constexpr std::uint32_t words = 1u << 24;

float as_float(std::uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
std::uint32_t as_bits(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return bits;
}

PFN_vkCmdPipelineBarrier2 real_barrier2 = nullptr;
std::uint32_t host_barriers = 0;
VkPipelineStageFlags2 host_src_stages = 0;
VkAccessFlags2 host_src_access = 0;
VKAPI_ATTR void VKAPI_CALL observe_host(VkCommandBuffer cb, const VkDependencyInfo* dep) {
    if (dep) {
        for (std::uint32_t i = 0; i < dep->memoryBarrierCount; ++i) {
            const auto& b = dep->pMemoryBarriers[i];
            if ((b.dstStageMask & VK_PIPELINE_STAGE_2_HOST_BIT) &&
                (b.dstAccessMask & VK_ACCESS_2_HOST_READ_BIT)) {
                ++host_barriers;
                host_src_stages |= b.srcStageMask;
                host_src_access |= b.srcAccessMask;
            }
        }
    }
    real_barrier2(cb, dep);
}

}  // namespace

int main() {
    vktest::Device gpu;
    const auto status = gpu.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("depth24_convert_test: no Vulkan device, skipped\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;

    // The converter allocates nothing here (every job brings its scratch buffer), but the callback must exist.
    Depth24Converter converter(gpu.device_proc, gpu.device, [](const VkMemoryRequirements&) { return VK_NULL_HANDLE; });
    EXPECT(converter.loaded());
    // The same pass with a 1 MiB storage range: a copy must split into slices or rows that each fit it.
    Depth24Converter small(gpu.device_proc, gpu.device, [](const VkMemoryRequirements&) { return VK_NULL_HANDLE; }, 1u << 20);
    Depth24Converter tiny(gpu.device_proc, gpu.device, [](const VkMemoryRequirements&) { return VK_NULL_HANDLE; }, 1024);
    std::size_t last_parts = 0;

    constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    std::uint32_t *app_words = nullptr, *scratch_words = nullptr;
    const VkBuffer app = gpu.buffer(std::uint64_t(words) * 4 + 4096, usage, reinterpret_cast<void**>(&app_words));
    const VkBuffer scratch = gpu.buffer(std::uint64_t(words) * 4, usage, reinterpret_cast<void**>(&scratch_words));
    EXPECT(app && scratch && app_words && scratch_words);

    const auto convert = [&](bool to_float, std::uint32_t width, std::uint32_t height, std::uint32_t slices,
                             std::uint32_t row_words, std::uint32_t slice_words, std::uint64_t offset, int limited = 0,
                             bool rgb96 = false) {
        Depth24Job job;
        job.to_float = to_float;
        job.rgb96 = rgb96;
        job.app = app;
        job.app_offset = offset;
        job.app_size = std::uint64_t(words) * 4 + 4096;
        job.width = width;
        job.height = height;
        job.slices = slices;
        job.app_row_words = row_words;
        job.app_slice_words = slice_words;
        job.scratch = scratch;
        std::vector<Depth24Job> parts;
        if (!(limited == 2 ? tiny : limited ? small : converter).prepare(job, parts)) return false;
        last_parts = parts.size();
        return gpu.run([&](VkCommandBuffer command_buffer) {
            for (const auto& part : parts) converter.run(command_buffer, part);
            VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            gpu.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                                     VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr,
                                                                     0, nullptr);
            return true;
        });
    };

    // 1. d -> float, for every d, with junk in the top byte.
    for (std::uint32_t d = 0; d < words; ++d) app_words[d] = d | (0xa5u << 24);
    std::memset(scratch_words, 0xcd, std::size_t(words) * 4);
    EXPECT(convert(true, 4096, 4096, 1, 4096, 4096u * 4096u, 0));
    std::uint64_t wrong = 0;
    for (std::uint32_t d = 0; d < words; ++d) {
        const float want = depth24_to_float(d);
        if (as_bits(want) != scratch_words[d] && wrong++ < 8)
            std::fprintf(stderr, "to float: d=%u got %08x want %08x\n", d, scratch_words[d], as_bits(want));
    }
    EXPECT(wrong == 0);
    std::printf("depth24: all %u words to float match the nearest float of d / 16777215\n", words);

    // 2. the same floats back: every word returns, the top byte becomes 0.
    std::memset(app_words, 0xcd, std::size_t(words) * 4);
    EXPECT(convert(false, 4096, 4096, 1, 4096, 4096u * 4096u, 0));
    wrong = 0;
    for (std::uint32_t d = 0; d < words; ++d)
        if (app_words[d] != d && wrong++ < 8) std::fprintf(stderr, "round trip: d=%u got %08x\n", d, app_words[d]);
    EXPECT(wrong == 0);
    std::printf("depth24: all %u words survive float and back\n", words);

    // All payloads and both signs with an all-ones exponent. Only positive infinity saturates.
    for (std::uint32_t i = 0; i < words; ++i)
        scratch_words[i] = 0x7f800000u | ((i & 0x800000u) << 8) | (i & 0x7fffffu);
    std::memset(app_words, 0xcd, std::size_t(words) * 4);
    EXPECT(convert(false, 4096, 4096, 1, 4096, words, 0));
    wrong = 0;
    for (std::uint32_t i = 0; i < words; ++i) {
        const std::uint32_t want = i == 0 ? 0xffffffu : 0u;
        if (app_words[i] != want && wrong++ < 8)
            std::fprintf(stderr, "special: f=%08x got %u want %u\n", scratch_words[i], app_words[i], want);
    }
    if (wrong) std::fprintf(stderr, "special: %llu mismatches\n", static_cast<unsigned long long>(wrong));
    EXPECT(wrong == 0);
    std::printf("depth24: both infinities and all NaN payloads quantise correctly\n");

    // 3. float patterns to words, against the reference.
    {
        std::uint32_t x = 7;
        for (std::uint32_t i = 0; i < words; ++i) {
            x = x * 1664525u + 1013904223u;
            scratch_words[i] = i * 256u + (x >> 24);  // every 256th pattern of the 32-bit range, jittered inside it
        }
        const std::uint32_t specials[] = {0x00000000, 0x80000000, 0x00000001, 0x007fffff, 0x00800000, 0x33000000,
                                          0x33800000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7fa00000,
                                          0x3f800000, 0x3f7fffff, 0x3f800001, 0x3effffff, 0x3f000000, 0x3f000001,
                                          0xbf800000, 0x40000000, 0x7f7fffff, 0xff7fffff, 0x3f7ffffe, 0x34000000,
                                          0x34000001, 0x33ffffff, 0x33c00000, 0x33c00001, 0x33bfffff};
        for (std::size_t i = 0; i < sizeof(specials) / 4; ++i) scratch_words[i] = specials[i];
        // Exactly the floats around the half steps, where the tie rule shows.
        std::uint32_t y = 99;
        for (std::uint32_t i = 64; i < 64 + 400000; i += 3) {
            y = y * 1664525u + 1013904223u;
            const float half = float((double(y >> 8) + 0.5) / 16777215.0);
            scratch_words[i] = as_bits(half);
            scratch_words[i + 1] = as_bits(std::nextafter(half, 0.0f));
            scratch_words[i + 2] = as_bits(std::nextafter(half, 1.0f));
        }
        std::memset(app_words, 0xcd, std::size_t(words) * 4);
        std::vector<std::uint32_t> input(scratch_words, scratch_words + words);
        EXPECT(convert(false, 4096, 4096, 1, 4096, 4096u * 4096u, 0, 1));
        EXPECT(last_parts > 32);
        wrong = 0;
        for (std::uint32_t i = 0; i < words; ++i) {
            const std::uint32_t want = depth24_quantize(as_float(input[i]));
            if (app_words[i] != want && wrong++ < 8)
                std::fprintf(stderr, "to words: f=%08x got %u want %u\n", input[i], app_words[i], want);
        }
        EXPECT(wrong == 0);
        std::printf("depth24: %u float patterns (specials, ties and their neighbours included) quantise like the reference\n",
                    words);
    }

    // 4. a pitched copy into the middle of a buffer: words outside the region stay, inside they are converted.
    {
        constexpr std::uint32_t w = 37, h = 5, s = 3, row = 64, slice = 400;
        constexpr std::uint64_t offset = 1100;  // bytes: not a multiple of the 256-byte descriptor window
        std::memset(app_words, 0xcd, 4096 * 4);
        for (std::uint32_t z = 0; z < s; ++z)
            for (std::uint32_t y = 0; y < h; ++y)
                for (std::uint32_t x = 0; x < w; ++x)
                    scratch_words[(z * h + y) * w + x] = as_bits(depth24_to_float((z * 1000 + y * 100 + x) * 5000u + 1));
        EXPECT(convert(false, w, h, s, row, slice, offset, 2));
        EXPECT(last_parts == 6);  // 1 KiB holds three rows of a slice: two parts a slice, three slices
        for (std::uint32_t i = 0; i < 4096; ++i) {
            const std::uint32_t rel = i - std::uint32_t(offset / 4);
            const bool inside = i >= offset / 4 && rel / slice < s && (rel % slice) / row < h && (rel % slice) % row < w;
            const std::uint32_t want = inside ? ((rel / slice * 1000 + (rel % slice) / row * 100 + (rel % slice) % row) * 5000u + 1)
                                              : 0xcdcdcdcdu;
            if (app_words[i] != want) {
                std::fprintf(stderr, "pitched: word %u got %08x want %08x\n", i, app_words[i], want);
                return 1;
            }
        }
        // And upload from the same layout: the words come out in the scratch, tightly packed.
        std::memset(scratch_words, 0xcd, 4096 * 4);
        EXPECT(convert(true, w, h, s, row, slice, offset, 2));
        for (std::uint32_t z = 0; z < s; ++z)
            for (std::uint32_t y = 0; y < h; ++y)
                for (std::uint32_t x = 0; x < w; ++x) {
                    const std::uint32_t d = (z * 1000 + y * 100 + x) * 5000u + 1;
                    EXPECT(scratch_words[(z * h + y) * w + x] == as_bits(depth24_to_float(d)));
                }
        std::printf("depth24: a pitched copy at an offset touches only its texels\n");
    }

    // 5. Concurrent VulkanQueue::execute() with enable_depth24 active: one queue runs depth24_as_float
    //    CopyTexture roundtrips (lazy tracker_mutex_ acquisition + collect/retire), while a second queue
    //    runs non-depth24 CopyBuffer batches concurrently (parallel prepare()), plus an early-failure batch
    //    that triggers Depth24Guard retirement.
    {
        VkPhysicalDeviceMemoryProperties mem_props{};
        gpu.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(
            gpu.physical, &mem_props);
        const auto alloc_device_local = [&](const VkMemoryRequirements& req) -> VkDeviceMemory {
            for (std::uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
                if (!(req.memoryTypeBits & (1u << i))) continue;
                if ((mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) !=
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                    continue;
                VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                info.allocationSize = req.size;
                info.memoryTypeIndex = i;
                VkDeviceMemory mem = VK_NULL_HANDLE;
                if (gpu.fn<PFN_vkAllocateMemory>("vkAllocateMemory")(gpu.device, &info, nullptr, &mem) == VK_SUCCESS)
                    return mem;
                return VK_NULL_HANDLE;
            }
            return VK_NULL_HANDLE;
        };

        VulkanContext context(gpu.device_proc, gpu.device, true);
        EXPECT(context.loaded());
        context.enable_depth24(gpu.device_proc, alloc_device_local);
        context.set_occlusion_resolve_memory(mem_props);

        constexpr int kIters = 16;
        constexpr std::uint32_t tex_w = 64, tex_h = 16;
        constexpr std::uint32_t tex_words = tex_w * tex_h;
        constexpr std::uint64_t tex_bytes = std::uint64_t(tex_words) * 4;
        const VkImage depth_img =
            gpu.image(VK_FORMAT_D32_SFLOAT_S8_UINT, tex_w, tex_h, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        EXPECT(depth_img != VK_NULL_HANDLE);

        std::uint32_t *d24_up_ptr = nullptr, *d24_rb_ptr = nullptr;
        std::uint32_t *buf_src_ptr = nullptr, *buf_dst_ptr = nullptr;
        const VkBuffer d24_up = gpu.buffer(tex_bytes, usage, reinterpret_cast<void**>(&d24_up_ptr));
        const VkBuffer d24_rb = gpu.buffer(tex_bytes * kIters, usage, reinterpret_cast<void**>(&d24_rb_ptr));
        const VkBuffer buf_src = gpu.buffer(tex_bytes, usage, reinterpret_cast<void**>(&buf_src_ptr));
        const VkBuffer buf_dst = gpu.buffer(tex_bytes * kIters, usage, reinterpret_cast<void**>(&buf_dst_ptr));
        EXPECT(d24_up && d24_rb && buf_src && buf_dst);

        for (std::uint32_t i = 0; i < tex_words; ++i) {
            d24_up_ptr[i] = ((i * 15973u + 13u) & depth24_max) | (0x7eu << 24);
            buf_src_ptr[i] = 0x12340000u + i;
        }

        constexpr ResourceId id_depth = 1, id_d24_up = 2, id_d24_rb = 3, id_buf_src = 4, id_buf_dst = 5;
        ResourceBinding img_b{};
        img_b.image = depth_img;
        img_b.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
        img_b.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        img_b.width = tex_w;
        img_b.height = tex_h;
        img_b.mip_levels = 1;
        img_b.array_layers = 1;
        img_b.depth = 1;
        img_b.samples = 1;
        img_b.depth24 = true;
        img_b.depth24_as_float = true;
        context.add_resource(id_depth, img_b, VK_IMAGE_LAYOUT_UNDEFINED);

        const auto reg_buf = [&](ResourceId id, VkBuffer buf, VkDeviceSize sz, bool readback) {
            ResourceBinding b{};
            b.buffer = buf;
            b.size = sz;
            b.host_readable = readback;
            context.add_resource(id, b);
        };
        reg_buf(id_d24_up, d24_up, tex_bytes, false);
        reg_buf(id_d24_rb, d24_rb, tex_bytes * kIters, true);
        reg_buf(id_buf_src, buf_src, tex_bytes, false);
        reg_buf(id_buf_dst, buf_dst, tex_bytes * kIters, true);

        VulkanQueue queue_a(context, gpu.queue, gpu.family);
        VulkanQueue queue_b(context, gpu.queue, gpu.family);
        EXPECT(queue_a.ok() && queue_b.ok());

        std::vector<CommandList> d24_lists(kIters);
        std::vector<CommandList> buf_lists(kIters);
        for (int iter = 0; iter < kIters; ++iter) {
            const std::uint64_t off = std::uint64_t(iter) * tex_bytes;
            CopyLocation fp_up{id_d24_up, true, 0, 0, tex_w, tex_h, 1, tex_w * 4};
            CopyLocation fp_rb{id_d24_rb, true, 0, off, tex_w, tex_h, 1, tex_w * 4};
            CopyLocation tex_loc{id_depth, false, 0, 0, 0, 0, 0, 0};
            EXPECT(d24_lists[iter].copy_texture(CopyTexture{tex_loc, 0, 0, 0, fp_up, false, {}}));
            EXPECT(d24_lists[iter].copy_texture(CopyTexture{fp_rb, 0, 0, 0, tex_loc, false, {}}));
            EXPECT(d24_lists[iter].close());

            EXPECT(buf_lists[iter].copy_buffer(CopyBuffer{id_buf_dst, off, id_buf_src, 0, tex_bytes}));
            EXPECT(buf_lists[iter].close());
        }

        // Early-failure batch: valid depth24_as_float copy followed by an unknown resource copy must fail cleanly
        // and retire the uncommitted depth24 scratch via Depth24Guard without corrupting subsequent executions.
        {
            CommandList failing_list;
            CopyLocation fp_up{id_d24_up, true, 0, 0, tex_w, tex_h, 1, tex_w * 4};
            CopyLocation tex_loc{id_depth, false, 0, 0, 0, 0, 0, 0};
            EXPECT(failing_list.copy_texture(CopyTexture{tex_loc, 0, 0, 0, fp_up, false, {}}));
            EXPECT(failing_list.copy_buffer(CopyBuffer{99999, 0, id_buf_src, 0, 4}));
            EXPECT(failing_list.close());
            const CommandList* ptrs[] = {&failing_list};
            EXPECT(queue_a.execute(ptrs) == ReplayResult::unresolved_resource);
        }

        std::atomic<bool> failed{false};
        std::thread ta([&] {
            for (int iter = 0; iter < kIters; ++iter) {
                const CommandList* ptrs[] = {&d24_lists[iter]};
                if (queue_a.execute(ptrs) != ReplayResult::success) {
                    failed.store(true);
                    break;
                }
            }
        });
        std::thread tb([&] {
            for (int iter = 0; iter < kIters; ++iter) {
                const CommandList* ptrs[] = {&buf_lists[iter]};
                if (queue_b.execute(ptrs) != ReplayResult::success) {
                    failed.store(true);
                    break;
                }
            }
        });
        ta.join();
        tb.join();
        EXPECT(!failed.load());
        EXPECT(gpu.fn<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(gpu.device) == VK_SUCCESS);

        for (int iter = 0; iter < kIters; ++iter) {
            const std::size_t base = std::size_t(iter) * tex_words;
            for (std::uint32_t i = 0; i < tex_words; ++i) {
                EXPECT(d24_rb_ptr[base + i] == (d24_up_ptr[i] & depth24_max));
                EXPECT(buf_dst_ptr[base + i] == buf_src_ptr[i]);
            }
        }
        std::printf("depth24: concurrent VulkanQueue::execute (lazy depth24 lock + parallel non-d24 prepare) ok\n");

        // Suballocated arena slices (nonzero ResourceBinding::buffer_offset) in a single shared VkBuffer:
        // CopyResource, ordinary RGBA8 CopyTexture, depth24_as_float CopyTexture, and rgb96 CopyTexture (including
        // non-256-aligned buffer_offsets) must read/write only their own slices and leave guard sentinels untouched.
        {
            constexpr std::uint32_t w = 64, h = 4;
            constexpr std::uint32_t slice_words = w * h;
            constexpr VkDeviceSize slice_bytes = VkDeviceSize(slice_words) * 4;  // 1024 bytes
            constexpr VkDeviceSize src_off = 512;
            constexpr VkDeviceSize copy_off = 2048;
            constexpr VkDeviceSize rgba_rb_off = 4096;
            constexpr VkDeviceSize d24_rb_off = 6148;  // 4-byte aligned, non-256-aligned (exercises app_base_words != 0)
            constexpr std::uint32_t rgb96_w = 17, rgb96_h = 3, rgb96_pitch = 256, rgb96_fp_off = 512;
            constexpr VkDeviceSize rgb96_slice_bytes = 1536;  // 512 prefix + 3 * 256 rows = 1280 <= 1536
            constexpr VkDeviceSize rgb96_src_off = 8192;
            constexpr VkDeviceSize rgb96_rb_off = 10244;  // 4-byte aligned, non-256-aligned
            constexpr VkDeviceSize arena_bytes = 12288;
            constexpr std::uint32_t guard_word = 0xA5A5A5A5u;
            constexpr std::uint32_t poison_word = 0xCDCDCDCDu;

            std::uint32_t* arena_ptr = nullptr;
            const VkBuffer arena = gpu.buffer(arena_bytes, usage, reinterpret_cast<void**>(&arena_ptr));
            EXPECT(arena && arena_ptr);
            for (std::size_t i = 0; i < arena_bytes / 4; ++i) arena_ptr[i] = guard_word;
            for (std::uint32_t i = 0; i < slice_words; ++i) {
                arena_ptr[src_off / 4 + i] = ((i * 65521u + 17u) & depth24_max) | ((0x40u + (i & 0x3fu)) << 24);
                arena_ptr[copy_off / 4 + i] = poison_word;
                arena_ptr[rgba_rb_off / 4 + i] = poison_word;
                arena_ptr[d24_rb_off / 4 + i] = poison_word;
            }
            const auto rgb96_word = [](std::uint32_t y, std::uint32_t x, std::uint32_t c) {
                constexpr std::uint32_t special[] = {0x7fa00001u, 0xffc12345u, 0x00000001u, 0x80000000u, 0xffffffffu};
                const std::uint32_t i = (y * rgb96_w + x) * 3u + c;
                return i % 7u == 0u ? special[(i / 7u) % 5u] : i * 2654435761u + 0x13579bdfu;
            };
            for (std::uint32_t y = 0; y < rgb96_h; ++y) {
                for (std::uint32_t x = 0; x < rgb96_w; ++x) {
                    for (std::uint32_t c = 0; c < 3; ++c) {
                        const std::size_t rel_w = (rgb96_fp_off + std::size_t(y) * rgb96_pitch) / 4 + x * 3 + c;
                        arena_ptr[rgb96_src_off / 4 + rel_w] = rgb96_word(y, x, c);
                        arena_ptr[rgb96_rb_off / 4 + rel_w] = poison_word;
                    }
                }
            }

            const VkImage rgba_img = gpu.image(VK_FORMAT_R8G8B8A8_UNORM, w, h, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            const VkImage d24_slice_img =
                gpu.image(VK_FORMAT_D32_SFLOAT_S8_UINT, w, h, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
            const VkImage rgb96_slice_img =
                gpu.image(VK_FORMAT_R32G32B32A32_UINT, rgb96_w, rgb96_h, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            EXPECT(rgba_img != VK_NULL_HANDLE && d24_slice_img != VK_NULL_HANDLE && rgb96_slice_img != VK_NULL_HANDLE);

            constexpr ResourceId id_src = 10, id_copy = 11, id_rgba_rb = 12, id_d24_rb = 13;
            constexpr ResourceId id_rgba_img = 14, id_d24_img = 15;
            constexpr ResourceId id_rgb96_src = 16, id_rgb96_rb = 17, id_rgb96_img = 18;
            const auto reg_slice = [&](ResourceId id, VkDeviceSize off, VkDeviceSize sz, bool readback) {
                ResourceBinding b{};
                b.buffer = arena;
                b.buffer_offset = off;
                b.size = sz;
                b.host_readable = readback;
                context.add_resource(id, b);
            };
            reg_slice(id_src, src_off, slice_bytes, false);
            reg_slice(id_copy, copy_off, slice_bytes, true);
            reg_slice(id_rgba_rb, rgba_rb_off, slice_bytes, true);
            reg_slice(id_d24_rb, d24_rb_off, slice_bytes, true);
            reg_slice(id_rgb96_src, rgb96_src_off, rgb96_slice_bytes, false);
            reg_slice(id_rgb96_rb, rgb96_rb_off, rgb96_slice_bytes, true);

            ResourceBinding rgba_b{};
            rgba_b.image = rgba_img;
            rgba_b.format = VK_FORMAT_R8G8B8A8_UNORM;
            rgba_b.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
            rgba_b.width = w;
            rgba_b.height = h;
            rgba_b.mip_levels = 1;
            rgba_b.array_layers = 1;
            rgba_b.depth = 1;
            rgba_b.samples = 1;
            context.add_resource(id_rgba_img, rgba_b, VK_IMAGE_LAYOUT_UNDEFINED);

            ResourceBinding d24_b{};
            d24_b.image = d24_slice_img;
            d24_b.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
            d24_b.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
            d24_b.width = w;
            d24_b.height = h;
            d24_b.mip_levels = 1;
            d24_b.array_layers = 1;
            d24_b.depth = 1;
            d24_b.samples = 1;
            d24_b.depth24 = true;
            d24_b.depth24_as_float = true;
            context.add_resource(id_d24_img, d24_b, VK_IMAGE_LAYOUT_UNDEFINED);

            ResourceBinding rgb96_b{};
            rgb96_b.image = rgb96_slice_img;
            rgb96_b.format = VK_FORMAT_R32G32B32A32_UINT;
            rgb96_b.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
            rgb96_b.width = rgb96_w;
            rgb96_b.height = rgb96_h;
            rgb96_b.mip_levels = 1;
            rgb96_b.array_layers = 1;
            rgb96_b.depth = 1;
            rgb96_b.samples = 1;
            rgb96_b.rgb96 = true;
            context.add_resource(id_rgb96_img, rgb96_b, VK_IMAGE_LAYOUT_UNDEFINED);

            // D3D-162: both conversion directions and formats must leave all 11 application counters zero.
            VkQueryPool statistics = VK_NULL_HANDLE;
            void* statistics_data = nullptr;
            if (gpu.pipeline_statistics_query) {
                VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
                info.queryCount = 1; info.pipelineStatistics = 0x7ff;
                EXPECT(gpu.fn<PFN_vkCreateQueryPool>("vkCreateQueryPool")(gpu.device, &info, nullptr, &statistics) == VK_SUCCESS);
                context.add_query_pool(100, {statistics, d3d12_query_heap::pipeline_statistics, 1, false, {false}, {}});
                ResourceBinding counters{}; counters.size = 88; counters.host_readable = true;
                counters.buffer = gpu.buffer(88, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &statistics_data);
                EXPECT(counters.buffer && statistics_data);
                std::memset(statistics_data, 0xcd, 88);
                context.add_resource(100, counters);
            }
            CommandList slice_list;
            if (statistics) EXPECT(slice_list.begin_query({100, d3d12_query::pipeline_statistics, 0}));
            EXPECT(slice_list.copy_resource({id_copy, id_src}));
            const CopyLocation fp_src{id_src, true, 0, 0, w, h, 1, w * 4};
            const CopyLocation fp_rgba_rb{id_rgba_rb, true, 0, 0, w, h, 1, w * 4};
            const CopyLocation fp_d24_rb{id_d24_rb, true, 0, 0, w, h, 1, w * 4};
            const CopyLocation loc_rgba{id_rgba_img, false, 0, 0, 0, 0, 0, 0};
            const CopyLocation loc_d24{id_d24_img, false, 0, 0, 0, 0, 0, 0};
            const CopyLocation fp_rgb96_src{id_rgb96_src, true, 0, rgb96_fp_off, rgb96_w, rgb96_h, 1, rgb96_pitch};
            const CopyLocation fp_rgb96_rb{id_rgb96_rb, true, 0, rgb96_fp_off, rgb96_w, rgb96_h, 1, rgb96_pitch};
            const CopyLocation loc_rgb96{id_rgb96_img, false, 0, 0, 0, 0, 0, 0};
            EXPECT(slice_list.copy_texture({loc_rgba, 0, 0, 0, fp_src, false, {}}));
            EXPECT(slice_list.copy_texture({fp_rgba_rb, 0, 0, 0, loc_rgba, false, {}}));
            EXPECT(slice_list.copy_texture({loc_d24, 0, 0, 0, fp_src, false, {}}));
            EXPECT(slice_list.copy_texture({fp_d24_rb, 0, 0, 0, loc_d24, false, {}}));
            EXPECT(slice_list.copy_texture({loc_rgb96, 0, 0, 0, fp_rgb96_src, false, {}}));
            EXPECT(slice_list.copy_texture({fp_rgb96_rb, 0, 0, 0, loc_rgb96, false, {}}));
            if (statistics) {
                EXPECT(slice_list.end_query({100, d3d12_query::pipeline_statistics, 0}));
                EXPECT(slice_list.resolve_queries({100, d3d12_query::pipeline_statistics, 0, 1, 100, 0}));
            }
            EXPECT(slice_list.close());

            const CommandList* ptr = &slice_list;
            EXPECT(queue_a.execute({&ptr, 1}) == ReplayResult::success);
            EXPECT(queue_a.idle());

            for (std::uint32_t i = 0; i < slice_words; ++i) {
                const std::uint32_t src_word = arena_ptr[src_off / 4 + i];
                EXPECT(arena_ptr[copy_off / 4 + i] == src_word);
                EXPECT(arena_ptr[rgba_rb_off / 4 + i] == src_word);
                EXPECT(arena_ptr[d24_rb_off / 4 + i] == (src_word & depth24_max));
            }
            for (std::uint32_t y = 0; y < rgb96_h; ++y) {
                for (std::uint32_t x = 0; x < rgb96_w; ++x) {
                    for (std::uint32_t c = 0; c < 3; ++c) {
                        const std::size_t rel_w = (rgb96_fp_off + std::size_t(y) * rgb96_pitch) / 4 + x * 3 + c;
                        EXPECT(arena_ptr[rgb96_rb_off / 4 + rel_w] == rgb96_word(y, x, c));
                    }
                }
            }
            const auto in_written = [&](std::size_t word_idx) {
                const VkDeviceSize byte_off = VkDeviceSize(word_idx) * 4;
                const auto within_slot = [&](VkDeviceSize start, VkDeviceSize sz) {
                    return byte_off >= start && byte_off < start + sz;
                };
                if (within_slot(src_off, slice_bytes) || within_slot(copy_off, slice_bytes) ||
                    within_slot(rgba_rb_off, slice_bytes) || within_slot(d24_rb_off, slice_bytes))
                    return true;
                const auto in_rgb96_texels = [&](VkDeviceSize base) {
                    if (byte_off < base + rgb96_fp_off) return false;
                    const VkDeviceSize rel = byte_off - (base + rgb96_fp_off);
                    const VkDeviceSize row = rel / rgb96_pitch;
                    const VkDeviceSize col = rel % rgb96_pitch;
                    return row < rgb96_h && col < VkDeviceSize(rgb96_w) * 12;
                };
                return in_rgb96_texels(rgb96_src_off) || in_rgb96_texels(rgb96_rb_off);
            };
            for (std::size_t i = 0; i < arena_bytes / 4; ++i)
                if (!in_written(i)) EXPECT(arena_ptr[i] == guard_word);

            if (statistics) {
                for (unsigned word = 0; word < 11; ++word) {
                    std::uint64_t value = UINT64_MAX;
                    std::memcpy(&value, static_cast<const char*>(statistics_data) + 8 * word, 8);
                    EXPECT(value == 0);
                }
                gpu.fn<PFN_vkDestroyQueryPool>("vkDestroyQueryPool")(gpu.device, statistics, nullptr);
            }
            std::printf("depth24: shared-buffer offsets (CopyResource, RGBA8, depth24 and rgb96 CopyTexture) ok\n");

            // Host-readback barrier observation across logical queue kinds: compute-converted depth24 and rgb96
            // readbacks must emit COMPUTE_SHADER_BIT / MEMORY_WRITE_BIT -> HOST_BIT / HOST_READ_BIT on direct and
            // compute queues (and stay within queue_supported_stages on copy queues), and a host_readable buffer UAV
            // clear must emit a TRANSFER_BIT / TRANSFER_WRITE_BIT -> HOST_BIT / HOST_READ_BIT barrier.
            real_barrier2 = context.vk.vkCmdPipelineBarrier2;
            EXPECT(real_barrier2 != nullptr);
            context.vk.vkCmdPipelineBarrier2 = observe_host;

            CommandList d24_rb_only, rgb96_rb_only, uav_clear_rb;
            EXPECT(d24_rb_only.copy_texture({fp_d24_rb, 0, 0, 0, loc_d24, false, {}}) && d24_rb_only.close());
            EXPECT(rgb96_rb_only.copy_texture({fp_rgb96_rb, 0, 0, 0, loc_rgb96, false, {}}) && rgb96_rb_only.close());
            EXPECT(uav_clear_rb.clear_unordered_access({id_copy, true, 0, 0, slice_bytes, 4, 0, 0, 1}, false,
                                                       {0x13572468u, 0, 0, 0}, {}) &&
                   uav_clear_rb.close());

            const CommandList* d24_ptr = &d24_rb_only;
            const CommandList* rgb96_ptr = &rgb96_rb_only;
            const CommandList* uav_ptr = &uav_clear_rb;
            for (const auto kind : {VulkanQueue::QueueKind::direct, VulkanQueue::QueueKind::compute,
                                    VulkanQueue::QueueKind::copy}) {
                VulkanQueue typed(context, gpu.queue, gpu.family, kind);
                EXPECT(typed.ok());

                // depth24_as_float readback
                for (std::uint32_t i = 0; i < slice_words; ++i) arena_ptr[d24_rb_off / 4 + i] = poison_word;
                host_barriers = 0;
                host_src_stages = 0;
                host_src_access = 0;
                EXPECT(typed.execute({&d24_ptr, 1}) == ReplayResult::success);
                EXPECT(typed.idle() && host_barriers == 1);
                EXPECT((host_src_stages & ~VulkanQueue::queue_supported_stages(kind)) == 0);
                if (kind == VulkanQueue::QueueKind::copy) {
                    EXPECT(host_src_stages == VK_PIPELINE_STAGE_2_TRANSFER_BIT);
                    EXPECT(host_src_access == VK_ACCESS_2_TRANSFER_WRITE_BIT);
                } else {
                    EXPECT(host_src_stages == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                    EXPECT(host_src_access == VK_ACCESS_2_MEMORY_WRITE_BIT);
                }
                for (std::uint32_t i = 0; i < slice_words; ++i)
                    EXPECT(arena_ptr[d24_rb_off / 4 + i] == (arena_ptr[src_off / 4 + i] & depth24_max));

                // rgb96 readback
                for (std::uint32_t y = 0; y < rgb96_h; ++y)
                    for (std::uint32_t x = 0; x < rgb96_w; ++x)
                        for (std::uint32_t c = 0; c < 3; ++c)
                            arena_ptr[rgb96_rb_off / 4 + (rgb96_fp_off + std::size_t(y) * rgb96_pitch) / 4 + x * 3 + c] =
                                poison_word;
                host_barriers = 0;
                host_src_stages = 0;
                host_src_access = 0;
                EXPECT(typed.execute({&rgb96_ptr, 1}) == ReplayResult::success);
                EXPECT(typed.idle() && host_barriers == 1);
                EXPECT((host_src_stages & ~VulkanQueue::queue_supported_stages(kind)) == 0);
                if (kind == VulkanQueue::QueueKind::copy) {
                    EXPECT(host_src_stages == VK_PIPELINE_STAGE_2_TRANSFER_BIT);
                    EXPECT(host_src_access == VK_ACCESS_2_TRANSFER_WRITE_BIT);
                } else {
                    EXPECT(host_src_stages == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                    EXPECT(host_src_access == VK_ACCESS_2_MEMORY_WRITE_BIT);
                }
                for (std::uint32_t y = 0; y < rgb96_h; ++y)
                    for (std::uint32_t x = 0; x < rgb96_w; ++x)
                        for (std::uint32_t c = 0; c < 3; ++c)
                            EXPECT(arena_ptr[rgb96_rb_off / 4 + (rgb96_fp_off + std::size_t(y) * rgb96_pitch) / 4 +
                                             x * 3 + c] == rgb96_word(y, x, c));

                // ClearUnorderedAccess on host_readable buffer
                for (std::uint32_t i = 0; i < slice_words; ++i) arena_ptr[copy_off / 4 + i] = poison_word;
                host_barriers = 0;
                host_src_stages = 0;
                host_src_access = 0;
                EXPECT(typed.execute({&uav_ptr, 1}) == ReplayResult::success);
                EXPECT(typed.idle() && host_barriers == 1);
                EXPECT(host_src_stages == VK_PIPELINE_STAGE_2_TRANSFER_BIT);
                EXPECT(host_src_access == VK_ACCESS_2_TRANSFER_WRITE_BIT);
                for (std::uint32_t i = 0; i < slice_words; ++i) EXPECT(arena_ptr[copy_off / 4 + i] == 0x13572468u);
            }
            context.vk.vkCmdPipelineBarrier2 = real_barrier2;
            std::printf("depth24: host-readback visibility barriers (depth24, rgb96, buffer UAV clear) ok\n");
        }
    }

    // 6. 96-bit texels (Depth24Job::rgb96, D3D-62): three words a texel in the footprint, four in the scratch, the
    // words copied as they are. A pitched copy at an offset, split into rows by the 1 KiB range.
    {
        constexpr std::uint32_t w = 37, h = 5, s = 3, row = 128, slice = 700;  // a row of texels is 111 words
        constexpr std::uint64_t offset = 1100;
        const auto word = [](std::uint32_t z, std::uint32_t y, std::uint32_t x, std::uint32_t c) {
            // Bit patterns a float conversion would not keep, among others: NaN payloads, a denormal, -0.
            constexpr std::uint32_t special[] = {0x7fa00001, 0xffc12345, 0x00000001, 0x80000000, 0xffffffff};
            const std::uint32_t i = ((z * h + y) * w + x) * 3 + c;
            return i % 7 == 0 ? special[i / 7 % 5] : i * 2654435761u + 12345u;
        };
        std::memset(app_words, 0xcd, 4096 * 4);
        std::memset(scratch_words, 0xee, 4096 * 4);
        for (std::uint32_t z = 0; z < s; ++z)
            for (std::uint32_t y = 0; y < h; ++y)
                for (std::uint32_t x = 0; x < w; ++x)
                    for (std::uint32_t c = 0; c < 3; ++c) scratch_words[((z * h + y) * w + x) * 4 + c] = word(z, y, x, c);
        EXPECT(convert(false, w, h, s, row, slice, offset, 2, true));
        EXPECT(last_parts == 15);  // 1 KiB holds one row of 37 four-word texels: a part a row
        for (std::uint32_t i = 0; i < 4096; ++i) {
            const std::uint32_t rel = i - std::uint32_t(offset / 4);
            const std::uint32_t z = rel / slice, y = (rel % slice) / row, in_row = (rel % slice) % row;
            const bool inside = i >= offset / 4 && z < s && y < h && in_row < w * 3;
            const std::uint32_t want = inside ? word(z, y, in_row / 3, in_row % 3) : 0xcdcdcdcdu;
            if (app_words[i] != want) {
                std::fprintf(stderr, "rgb96 readback: word %u got %08x want %08x\n", i, app_words[i], want);
                return 1;
            }
        }
        // The upload from the same layout: three words of each texel in the scratch, the fourth written as 0, and
        // nothing past the last texel.
        std::memset(scratch_words, 0xee, 4096 * 4);
        EXPECT(convert(true, w, h, s, row, slice, offset, 2, true));
        for (std::uint32_t z = 0; z < s; ++z)
            for (std::uint32_t y = 0; y < h; ++y)
                for (std::uint32_t x = 0; x < w; ++x)
                    for (std::uint32_t c = 0; c < 4; ++c)
                        EXPECT(scratch_words[((z * h + y) * w + x) * 4 + c] == (c < 3 ? word(z, y, x, c) : 0u));
        EXPECT(scratch_words[w * h * s * 4] == 0xeeeeeeeeu);

        // Slices too large to share a 1 MiB range: a part a slice (256 x 128 texels are 512 KiB of scratch).
        constexpr std::uint32_t bw = 256, bh = 128, bs = 6, brow = 800, bslice = brow * bh + 64;
        const std::uint32_t texels = bw * bh * bs;
        for (std::uint32_t z = 0; z < bs; ++z)
            for (std::uint32_t y = 0; y < bh; ++y)
                for (std::uint32_t x = 0; x < bw * 3; ++x)
                    app_words[z * bslice + y * brow + x] = (z * bh + y) * bw * 3 + x + 0x9e3779b9u;
        std::memset(scratch_words, 0xee, std::size_t(texels) * 16);
        EXPECT(convert(true, bw, bh, bs, brow, bslice, 0, 1, true));
        EXPECT(last_parts == bs);
        wrong = 0;
        for (std::uint32_t t = 0; t < texels; ++t)
            for (std::uint32_t c = 0; c < 4; ++c)
                if (scratch_words[t * 4 + c] != (c < 3 ? t * 3 + c + 0x9e3779b9u : 0u)) ++wrong;
        EXPECT(wrong == 0);
        std::printf("depth24: 96-bit texels repack both ways, split into rows and into slices\n");
    }

    // 7. ScratchBuffers::collect() must check VkResult from vkGetSemaphoreCounterValue() (leaving in-flight
    //    chunks retired if the query fails even when *pValue is overwritten with UINT64_MAX) and reclaim
    //    ticket-0 batches without querying the timeline semaphore.
    {
        struct MockVk {
            std::uint64_t next_handle = 1;
            VkResult sem_result = VK_ERROR_DEVICE_LOST;
            std::uint64_t sem_value = UINT64_MAX;
            unsigned sem_queries = 0;
            unsigned destroyed_buffers = 0;

            static MockVk*& state() {
                static MockVk* s = nullptr;
                return s;
            }
            static VKAPI_ATTR VkResult VKAPI_CALL create_buf(VkDevice, const VkBufferCreateInfo*,
                                                             const VkAllocationCallbacks*, VkBuffer* out) {
                *out = reinterpret_cast<VkBuffer>(state()->next_handle++);
                return VK_SUCCESS;
            }
            static VKAPI_ATTR void VKAPI_CALL destroy_buf(VkDevice, VkBuffer, const VkAllocationCallbacks*) {
                ++state()->destroyed_buffers;
            }
            static VKAPI_ATTR void VKAPI_CALL mem_req(VkDevice, VkBuffer, VkMemoryRequirements* req) {
                *req = {};
                req->size = 4096;
                req->alignment = 256;
                req->memoryTypeBits = 1;
            }
            static VKAPI_ATTR VkResult VKAPI_CALL bind_buf(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
                return VK_SUCCESS;
            }
            static VKAPI_ATTR void VKAPI_CALL free_mem(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {}
            static VKAPI_ATTR VkResult VKAPI_CALL get_sem(VkDevice, VkSemaphore, std::uint64_t* out) {
                ++state()->sem_queries;
                if (out) *out = state()->sem_value;
                return state()->sem_result;
            }
            static VKAPI_ATTR VkResult VKAPI_CALL wait_idle(VkDevice) { return VK_SUCCESS; }
            static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL gdpa(VkDevice, const char* name) {
                if (std::strcmp(name, "vkCreateBuffer") == 0) return reinterpret_cast<PFN_vkVoidFunction>(create_buf);
                if (std::strcmp(name, "vkDestroyBuffer") == 0)
                    return reinterpret_cast<PFN_vkVoidFunction>(destroy_buf);
                if (std::strcmp(name, "vkGetBufferMemoryRequirements") == 0)
                    return reinterpret_cast<PFN_vkVoidFunction>(mem_req);
                if (std::strcmp(name, "vkBindBufferMemory") == 0)
                    return reinterpret_cast<PFN_vkVoidFunction>(bind_buf);
                if (std::strcmp(name, "vkFreeMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(free_mem);
                if (std::strcmp(name, "vkGetSemaphoreCounterValue") == 0)
                    return reinterpret_cast<PFN_vkVoidFunction>(get_sem);
                if (std::strcmp(name, "vkDeviceWaitIdle") == 0) return reinterpret_cast<PFN_vkVoidFunction>(wait_idle);
                return nullptr;
            }
        };
        MockVk mock{};
        MockVk::state() = &mock;
        const VkSemaphore fake_sem = reinterpret_cast<VkSemaphore>(std::uintptr_t(0x1234));
        {
            ScratchBuffers scratch(&MockVk::gdpa, reinterpret_cast<VkDevice>(std::uintptr_t(1)),
                                   [&](const VkMemoryRequirements&) {
                                       return reinterpret_cast<VkDeviceMemory>(mock.next_handle++);
                                   });
            EXPECT(scratch.loaded());
            VkBuffer b1 = VK_NULL_HANDLE, b2 = VK_NULL_HANDLE;
            VkDeviceSize off = 0;
            EXPECT(scratch.get(1024, b1, off));
            scratch.retire(fake_sem, 1);
            // Query fails with VK_ERROR_DEVICE_LOST while writing UINT64_MAX to *pValue: chunk must NOT be recycled!
            scratch.collect();
            EXPECT(scratch.get(1024, b2, off));
            EXPECT(b2 != b1);  // b1 is still in flight; must not have been reclaimed into spare_
            // Unsubmitted batch (ticket == 0): must reclaim without calling vkGetSemaphoreCounterValue.
            const unsigned queries_before = mock.sem_queries;
            scratch.retire(fake_sem, 0);
            mock.sem_result = VK_SUCCESS;
            mock.sem_value = 0;
            scratch.collect();
            EXPECT(mock.sem_queries == queries_before + 1);  // only queried for b1 (ticket 1), not b2 (ticket 0)
            VkBuffer b3 = VK_NULL_HANDLE;
            EXPECT(scratch.get(1024, b3, off));
            EXPECT(b3 == b2);  // b2 was reclaimed cleanly from ticket 0
            // Now b1's ticket completes cleanly.
            mock.sem_value = 1;
            scratch.collect();
            scratch.retire(fake_sem, 0);
        }
        EXPECT(mock.destroyed_buffers == 2);
        MockVk::state() = nullptr;
        std::printf("depth24: ScratchBuffers::collect() checks vkGetSemaphoreCounterValue VkResult and ticket 0 ok\n");
    }

    EXPECT(gpu.validation_errors() == 0);
    std::printf("depth24_convert_test: ok\n");
    return 0;
}
