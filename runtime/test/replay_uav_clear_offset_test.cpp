// A buffer UAV clear on a resource that is a range of a larger VkBuffer (ResourceBinding::buffer_offset, as the GPU
// address arena places every buffer) writes that range, not the start of the VkBuffer. UavView::offset is relative to
// the resource, as copies and atomic copies already treat it. No Vulkan loader, GPU or game data: the transfer
// functions are fakes that record the ranges they are given.
#include "../src/vulkan_replay.h"
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
struct Write {
    char kind;  // 'f' fill, 'u' update, 'c' copy (dst; src in from)
    VkDeviceSize offset, size, from;
    VkBuffer buffer = VK_NULL_HANDLE, source = VK_NULL_HANDLE;
};
std::vector<Write> writes;

unsigned image_clears = 0, attachment_clears = 0;
VkFormat requested_view_format = VK_FORMAT_UNDEFINED;
VkImageSubresourceRange requested_view_range{};
VkClearColorValue captured_color{};
VkClearRect captured_rect{};
VkRect2D captured_render_area{};
std::uint32_t captured_render_layers = 0;

void reset_texture_capture() {
    image_clears = attachment_clears = 0;
    requested_view_format = VK_FORMAT_UNDEFINED;
    requested_view_range = {};
    captured_color = {};
    captured_rect = {};
    captured_render_area = {};
    captured_render_layers = 0;
}

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                   std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                   std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL fill(VkCommandBuffer, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, std::uint32_t) {
    writes.push_back({'f', offset, size, 0, buffer});
}
VKAPI_ATTR void VKAPI_CALL update(VkCommandBuffer, VkBuffer, VkDeviceSize offset, VkDeviceSize size, const void*) {
    writes.push_back({'u', offset, size, 0});
}
VKAPI_ATTR void VKAPI_CALL copy(VkCommandBuffer, VkBuffer source, VkBuffer buffer, std::uint32_t count, const VkBufferCopy* r) {
    for (std::uint32_t i = 0; i < count; ++i) writes.push_back({'c', r[i].dstOffset, r[i].size, r[i].srcOffset, buffer, source});
}
VKAPI_ATTR void VKAPI_CALL clear_image(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue* value,
                                       std::uint32_t, const VkImageSubresourceRange*) {
    captured_color = *value;
    ++image_clears;
}
VKAPI_ATTR void VKAPI_CALL begin_rendering(VkCommandBuffer, const VkRenderingInfo* info) {
    captured_render_area = info->renderArea;
    captured_render_layers = info->layerCount;
}
VKAPI_ATTR void VKAPI_CALL end_rendering(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL clear_attachments(VkCommandBuffer, std::uint32_t count, const VkClearAttachment* values,
                                             std::uint32_t rect_count, const VkClearRect* rects) {
    if (count == 1 && rect_count >= 1) {
        captured_color = values[0].clearValue.color;
        captured_rect = rects[0];
        ++attachment_clears;
    }
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// Every byte written (and read by a copy) lies in [begin, end) of the VkBuffer.
bool inside(VkDeviceSize begin, VkDeviceSize end) {
    if (writes.empty()) return false;
    for (const Write& w : writes) {
        if (w.offset < begin || w.offset + w.size > end) return false;
        if (w.kind == 'c' && (w.from < begin || w.from + w.size > end)) return false;
    }
    return true;
}
}  // namespace

int main() {
    using namespace leonida;
    constexpr VkDeviceSize at = 65536, bytes = 256;
    ResourceBinding range{};
    range.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{0x1000});
    range.buffer_offset = at;  // a range at 64 KiB of a shared buffer
    range.size = bytes;
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* { return id == 1 ? &range : nullptr; };
    ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.fill_buffer = fill;
    vk.update_buffer = update;
    vk.copy_buffer = copy;

    const auto run = [&](const UavView& view, std::span<const Rect> rects = {}) {
        writes.clear();
        CommandList list;
        list.clear_unordered_access(view, false, {7, 7, 7, 7}, rects);
        list.close();
        return replay(list, VK_NULL_HANDLE, resolve, vk);
    };

    // A raw view of the whole resource: one fill of [at, at + 256).
    expect(run({1, true, 0, 0, bytes, 4, 0, 0, 1}) == ReplayResult::success, "raw whole-view clear replays");
    expect(writes.size() == 1 && writes[0].kind == 'f' && writes[0].offset == at && writes[0].size == bytes,
           "raw whole-view clear fills the resource's range of the buffer");
    // A view starting 16 bytes into the resource.
    expect(run({1, true, 0, 16, 64, 4, 0, 0, 1}) == ReplayResult::success, "raw view at 16 replays");
    expect(inside(at + 16, at + 80), "raw view at 16 stays in [at + 16, at + 80)");
    // R8_UINT elements from byte 1: unaligned ends are copied from the aligned middle, all inside the view.
    expect(run({1, true, VK_FORMAT_R8_UINT, 1, 37, 1, 0, 0, 1}) == ReplayResult::success, "R8 view at 1 replays");
    expect(inside(at + 1, at + 38), "R8 view at 1 stays in [at + 1, at + 38)");
    // R32G32B32A32_UINT (16-byte elements) is written with vkCmdUpdateBuffer and doubled with copies.
    expect(run({1, true, VK_FORMAT_R32G32B32A32_UINT, 32, 128, 16, 0, 0, 1}) == ReplayResult::success,
           "RGBA32 view at 32 replays");
    expect(inside(at + 32, at + 160), "RGBA32 view at 32 stays in [at + 32, at + 160)");
    // Rectangles count elements from the view's first byte.
    const Rect rect{2, 0, 5, 1};
    expect(run({1, true, 0, 8, 64, 4, 0, 0, 1}, std::span<const Rect>(&rect, 1)) == ReplayResult::success,
           "rectangle clear replays");
    expect(inside(at + 16, at + 28), "elements 2..5 of a view at 8 are [at + 16, at + 28)");

    const auto scratch = reinterpret_cast<VkBuffer>(std::uintptr_t{0x3000});
    constexpr VkDeviceSize scratch_at = 512;
    unsigned allocations = 0;
    expect(run({1, true, VK_FORMAT_R8_UINT, 1, 1, 1, 0, 0, 1}) == ReplayResult::unsupported && writes.empty(),
           "short clear without scratch is rejected before any command is emitted");
    vk.copy_scratch = [&](VkDeviceSize size, VkBuffer& buffer, VkDeviceSize& offset) {
        expect(size == 8, "short clear requests eight scratch bytes");
        ++allocations;
        buffer = scratch;
        offset = scratch_at;
        return true;
    };
    for (VkDeviceSize offset = 0; offset < 8; ++offset) {
        for (VkDeviceSize size = 1; size <= 8; ++size) {
            allocations = 0;
            expect(run({1, true, VK_FORMAT_R8_UINT, offset, size, 1, 0, 0, 1}) == ReplayResult::success,
                   "small R8 range replays at every word phase");
            const bool short_range = size < (4 - offset % 4) % 4 + 4;
            expect(allocations == unsigned(short_range), "only ranges lacking an aligned word allocate scratch");
            if (short_range) {
                expect(writes.size() == 2 && writes[0].kind == 'f' && writes[0].buffer == scratch &&
                       writes[0].offset == scratch_at && writes[0].size == 8 &&
                       writes[1].kind == 'c' && writes[1].source == scratch && writes[1].buffer == range.buffer &&
                       writes[1].from == scratch_at && writes[1].offset == at + offset && writes[1].size == size,
                       "short clear writes exactly the requested bytes from scratch, not neighbouring resource bytes");
            } else expect(inside(at + offset, at + offset + size), "fast path stays within the range");
        }
    }
    allocations = 0;
    const std::array<Rect, 3> small_rects{{{0, 0, 1, 1}, {0, 0, 2, 1}, {3, 0, 4, 1}}};
    expect(run({1, true, VK_FORMAT_R16_UINT, 2, 8, 2, 0, 0, 1}, small_rects) == ReplayResult::success && allocations == 1,
           "overlapping short R16 rectangles share one scratch allocation per clear");
    vk.copy_scratch = [](VkDeviceSize, VkBuffer&, VkDeviceSize&) { return false; };
    expect(run({1, true, VK_FORMAT_R8_UINT, 255, 1, 1, 0, 0, 1}) == ReplayResult::unsupported && writes.empty(),
           "scratch allocation failure rejects the list without emitting commands");
    vk.copy_scratch = [&](VkDeviceSize, VkBuffer& buffer, VkDeviceSize& offset) {
        buffer = scratch; offset = 1; return true;
    };
    expect(run({1, true, VK_FORMAT_R8_UINT, 255, 1, 1, 0, 0, 1}) == ReplayResult::unsupported && writes.empty(),
           "misaligned scratch is rejected before vkCmdFillBuffer");
    const std::array<Rect, 2> empty_rects{{{-3, 0, 0, 1}, {0, 1, 1, 2}}};
    expect(run({1, true, VK_FORMAT_R8_UINT, 255, 1, 1, 0, 0, 1}, empty_rects) == ReplayResult::success && writes.empty(),
           "empty clipped rectangles need no scratch");
    range.buffer_offset = UINT64_MAX - 2;
    expect(run({1, true, VK_FORMAT_R8_UINT, 1, 2, 1, 0, 0, 1}) == ReplayResult::invalid_clear && writes.empty(),
           "backing range overflow is rejected before scratch allocation or emission");
    range.buffer_offset = at;
    vk.copy_scratch = {};

    // Typeless / mutable texture UAV clears where view_format != binding.format must clear through an attachment view
    // of view_format with clear_color decoded in view_format, avoiding SFLOAT NaN round-trips (issue #1902).
    ReplayFunctions tex_vk = vk;
    tex_vk.clear_color_image = clear_image;
    tex_vk.begin_rendering = begin_rendering;
    tex_vk.end_rendering = end_rendering;
    tex_vk.clear_attachments = clear_attachments;
    tex_vk.image_view = [](const ResourceBinding& b, const VkImageSubresourceRange& r) {
        requested_view_format = b.format;
        requested_view_range = r;
        return reinterpret_cast<VkImageView>(std::uintptr_t{0x2000});
    };

    const auto run_tex = [&](const ResourceBinding& binding, const UavView& view, bool is_float,
                             std::array<std::uint32_t, 4> values, std::span<const Rect> rects = {},
                             const ReplayFunctions* custom_vk = nullptr) {
        reset_texture_capture();
        CommandList list;
        list.clear_unordered_access(view, is_float, values, rects);
        list.close();
        return replay(list, VK_NULL_HANDLE, [&](ResourceId id) { return id == 2 ? &binding : nullptr; },
                      custom_vk ? *custom_vk : tex_vk);
    };

    ResourceBinding tex{};
    tex.image = reinterpret_cast<VkImage>(std::uintptr_t{0x3000});
    tex.format = VK_FORMAT_R16G16_SFLOAT;
    tex.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    tex.width = 8;
    tex.height = 6;
    tex.mip_levels = 2;
    tex.array_layers = 4;
    tex.color_attachment = true;

    const auto fbits = [](float f) { return std::bit_cast<std::uint32_t>(f); };
    // R16G16_TYPELESS (SFLOAT) cleared as R16G16_UNORM Float(1.0, 1.0): uses R16G16_UNORM view and float32 {1, 1}.
    expect(run_tex(tex, {2, false, VK_FORMAT_R16G16_UNORM, 0, 0, 0, 0, 1, 2}, true, {fbits(1.0f), fbits(1.0f), 0, 0}) ==
               ReplayResult::success,
           "R16G16_TYPELESS -> R16G16_UNORM whole-view clear replays");
    expect(image_clears == 0 && attachment_clears == 1,
           "R16G16_TYPELESS -> R16G16_UNORM uses attachment clear, not vkCmdClearColorImage");
    expect(requested_view_format == VK_FORMAT_R16G16_UNORM,
           "R16G16_TYPELESS -> R16G16_UNORM creates image view with view_format");
    expect(requested_view_range.baseMipLevel == 0 && requested_view_range.baseArrayLayer == 1 &&
               requested_view_range.layerCount == 2 && captured_render_layers == 2,
           "reinterpreted UAV clear preserves mip and layer range");
    expect(captured_color.float32[0] == 1.0f && captured_color.float32[1] == 1.0f,
           "R16G16_UNORM clear value is decoded in view_format (not half-float NaN)");
    expect(captured_rect.rect.offset.x == 0 && captured_rect.rect.offset.y == 0 &&
               captured_rect.rect.extent.width == 8 && captured_rect.rect.extent.height == 6 &&
               captured_rect.layerCount == 2,
           "reinterpreted whole-view clear covers full mip extent and layers");

    // R16G16_TYPELESS (SFLOAT) cleared as R16G16_UINT with half-float NaN bit patterns 0xffff and 0x7fff.
    const Rect sub_rect{1, 1, 4, 3};
    expect(run_tex(tex, {2, false, VK_FORMAT_R16G16_UINT, 0, 0, 0, 1, 0, 1}, false, {0xffff, 0x7fff, 0, 0},
                   std::span<const Rect>(&sub_rect, 1)) == ReplayResult::success,
           "R16G16_TYPELESS -> R16G16_UINT rect clear replays");
    expect(image_clears == 0 && attachment_clears == 1 && requested_view_format == VK_FORMAT_R16G16_UINT,
           "R16G16_UINT rect clear uses R16G16_UINT view");
    expect(captured_color.uint32[0] == 0xffff && captured_color.uint32[1] == 0x7fff,
           "R16G16_UINT clear passes exact uint32 values");
    expect(captured_render_area.extent.width == 4 && captured_render_area.extent.height == 3 &&
               captured_rect.rect.extent.width == 3 && captured_rect.rect.extent.height == 2,
           "R16G16_UINT rect clear clips to mip 1 extent");

    // Same-format whole-view unpredicated clear stays on vkCmdClearColorImage.
    expect(run_tex(tex, {2, false, VK_FORMAT_R16G16_SFLOAT, 0, 0, 0, 0, 0, 1}, true, {fbits(0.5f), fbits(-2.0f), 0, 0}) ==
               ReplayResult::success,
           "same-format R16G16_SFLOAT whole-view clear replays");
    expect(image_clears == 1 && attachment_clears == 0, "same-format whole-view clear uses vkCmdClearColorImage");

    // R32_TYPELESS (SFLOAT) cleared as R32_UINT with sNaN bit pattern 0x7f800001.
    ResourceBinding r32 = tex;
    r32.format = VK_FORMAT_R32_SFLOAT;
    expect(run_tex(r32, {2, false, VK_FORMAT_R32_UINT, 0, 0, 0, 0, 0, 1}, false, {0x7f800001u, 0, 0, 0}) ==
               ReplayResult::success,
           "R32_TYPELESS -> R32_UINT sNaN clear replays");
    expect(image_clears == 0 && attachment_clears == 1 && requested_view_format == VK_FORMAT_R32_UINT &&
               captured_color.uint32[0] == 0x7f800001u,
           "R32_UINT clear uses R32_UINT attachment view with exact sNaN bits");

    // R8G8B8A8_TYPELESS (UNORM) cleared as R8G8B8A8_UINT and R8G8B8A8_SINT uses view_format directly.
    ResourceBinding rgba8 = tex;
    rgba8.format = VK_FORMAT_R8G8B8A8_UNORM;
    expect(run_tex(rgba8, {2, false, VK_FORMAT_R8G8B8A8_UINT, 0, 0, 0, 0, 0, 1}, false, {0xff, 0x80, 0x7f, 1}) ==
               ReplayResult::success,
           "R8G8B8A8_TYPELESS -> R8G8B8A8_UINT clear replays");
    expect(image_clears == 0 && attachment_clears == 1 && requested_view_format == VK_FORMAT_R8G8B8A8_UINT &&
               captured_color.uint32[0] == 0xff && captured_color.uint32[1] == 0x80 &&
               captured_color.uint32[2] == 0x7f && captured_color.uint32[3] == 1,
           "R8G8B8A8_UINT clear uses R8G8B8A8_UINT attachment view");

    // Emulated R9G9B9E5_SHAREDEXP (backed by R32_UINT) must NOT request an E5B9G9R9 color attachment view.
    ResourceBinding r9e5 = tex;
    r9e5.format = VK_FORMAT_R32_UINT;
    expect(run_tex(r9e5, {2, false, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 0, 0, 0, 0, 0, 1}, true,
                   {fbits(1.0f), fbits(2.0f), fbits(4.0f), 0}) == ReplayResult::success,
           "emulated R9G9B9E5 whole-view clear replays");
    expect(image_clears == 1 && attachment_clears == 0 && captured_color.uint32[0] == 0x94010040u,
           "emulated R9G9B9E5 whole-view clear stays on R32_UINT vkCmdClearColorImage");
    expect(run_tex(r9e5, {2, false, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 0, 0, 0, 0, 0, 1}, true,
                   {fbits(1.0f), fbits(2.0f), fbits(4.0f), 0}, std::span<const Rect>(&sub_rect, 1)) ==
               ReplayResult::success,
           "emulated R9G9B9E5 rect clear replays");
    expect(image_clears == 0 && attachment_clears == 1 && requested_view_format == VK_FORMAT_R32_UINT &&
               captured_color.uint32[0] == 0x94010040u,
           "emulated R9G9B9E5 rect clear uses R32_UINT attachment view");

    // When dynamic rendering is unavailable, reinterpretation refuses instead of corrupting via vkCmdClearColorImage.
    ReplayFunctions no_render = tex_vk;
    no_render.begin_rendering = nullptr;
    expect(run_tex(tex, {2, false, VK_FORMAT_R16G16_UNORM, 0, 0, 0, 0, 0, 1}, true, {fbits(1.0f), fbits(1.0f), 0, 0},
                   {}, &no_render) == ReplayResult::unsupported,
           "reinterpreted UAV clear without dynamic rendering returns unsupported");

    // When color_attachment_format reports a view format unsupported as an attachment, keep the backing format path.
    ReplayFunctions no_unorm_rt = tex_vk;
    no_unorm_rt.color_attachment_format = [](VkFormat f) { return f != VK_FORMAT_R16G16_UNORM; };
    expect(run_tex(tex, {2, false, VK_FORMAT_R16G16_UNORM, 0, 0, 0, 0, 0, 1}, true, {fbits(0.0f), fbits(0.0f), 0, 0},
                   {}, &no_unorm_rt) == ReplayResult::success,
           "non-attachable view format falls back to backing format");
    expect(image_clears == 1 && attachment_clears == 0,
           "non-attachable view format whole-view clear uses vkCmdClearColorImage");

    if (failures) return 1;
    std::printf("replay_uav_clear_offset: buffer offsets and typeless texture view-format clears all passed\n");
    return 0;
}
