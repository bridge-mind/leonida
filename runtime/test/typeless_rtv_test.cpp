// CPU-only replay tests for typed RGBA8 views. No Vulkan loader, GPU or game data.
#include "../src/vulkan_replay.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <tuple>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {
unsigned image_clears = 0, attachment_clears = 0;
VkFormat requested_format = VK_FORMAT_UNDEFINED;
VkClearColorValue captured{};
VkClearRect captured_rect{};
VkRect2D render_area{};
VkImageSubresourceRange requested_range{};
unsigned rendering_layers = 0;
VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
    std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL clear(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue* value,
    std::uint32_t, const VkImageSubresourceRange*) { captured = *value; ++image_clears; }
VKAPI_ATTR void VKAPI_CALL begin(VkCommandBuffer, const VkRenderingInfo* info) {
    render_area = info->renderArea;
    rendering_layers = info->layerCount;
}
VKAPI_ATTR void VKAPI_CALL end(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL attachments(VkCommandBuffer, std::uint32_t n, const VkClearAttachment* values,
    std::uint32_t count, const VkClearRect* rects) {
    if (n == 1 && count == 1) { captured = values[0].clearValue.color; captured_rect = rects[0]; ++attachment_clears; }
}
leonida::ReplayFunctions functions() {
    leonida::ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.clear_color_image = clear;
    vk.begin_rendering = begin;
    vk.end_rendering = end;
    vk.clear_attachments = attachments;
    vk.image_view = [](const auto& binding, const auto& range) {
        requested_range = range;
        requested_format = binding.format;
        return reinterpret_cast<VkImageView>(std::uintptr_t{1});
    };
    return vk;
}
leonida::ResourceBinding resource() {
    leonida::ResourceBinding r{};
    r.image = reinterpret_cast<VkImage>(std::uintptr_t{1});
    r.format = VK_FORMAT_R8G8B8A8_UNORM;
    r.width = 5; r.height = 3;
    r.typeless = true;
    return r;
}
void reset_counters() {
    image_clears = attachment_clears = 0;
    requested_format = VK_FORMAT_UNDEFINED;
    captured = {}; captured_rect = {}; render_area = {};
    requested_range = {}; rendering_layers = 0;
}
leonida::ReplayResult run(const leonida::CommandList& list, const leonida::ResourceBinding& resource,
                          const leonida::ReplayFunctions& vk) {
    reset_counters();
    return leonida::replay(list, VK_NULL_HANDLE, [&](auto) { return &resource; }, vk);
}
} // namespace

int main() {
    using namespace leonida;
    const auto vk = functions();
    const auto binding = resource();
    const ClearColor color{1.5f, 2.5f, 256, -1};
    const Rect rectangle{1, 1, 4, 3};
    const std::pair<ColorViewFormat, VkFormat> formats[] = {
        {ColorViewFormat::rgba8_unorm, VK_FORMAT_R8G8B8A8_UNORM},
        {ColorViewFormat::rgba8_srgb, VK_FORMAT_R8G8B8A8_SRGB},
        {ColorViewFormat::rgba8_uint, VK_FORMAT_R8G8B8A8_UINT},
        {ColorViewFormat::rgba8_sint, VK_FORMAT_R8G8B8A8_SINT},
        {ColorViewFormat::rgba8_snorm, VK_FORMAT_R8G8B8A8_SNORM},
    };
    unsigned checks = 0;
    for (const auto& [format, expected] : formats) {
        for (const bool rect : {false, true}) {
            CommandList list;
            TargetView target{1, 0, 0, 1, format};
            EXPECT(list.clear_render_target(target, color, rect ? std::span<const Rect>(&rectangle, 1) : std::span<const Rect>{}));
            target.format = ColorViewFormat::resource; // Mutating the descriptor must not change the recorded format.
            EXPECT(list.close());
            EXPECT(run(list, binding, vk) == ReplayResult::success);
            const bool attachment = rect || format != ColorViewFormat::rgba8_unorm;
            EXPECT(image_clears == unsigned(!attachment) && attachment_clears == unsigned(attachment));
            if (attachment) {
                EXPECT(requested_format == expected);
                EXPECT(render_area.extent.width == 5 && render_area.extent.height == 3);
                EXPECT(captured_rect.rect.offset.x == (rect ? 1 : 0));
                EXPECT(captured_rect.rect.offset.y == (rect ? 1 : 0));
                EXPECT(captured_rect.rect.extent.width == (rect ? 3u : 5u));
                EXPECT(captured_rect.rect.extent.height == (rect ? 2u : 3u));
            }
            if (format == ColorViewFormat::rgba8_uint) {
                EXPECT(captured.uint32[0] == 1 && captured.uint32[1] == 2 && captured.uint32[2] == 255 && captured.uint32[3] == 0);
            } else if (format == ColorViewFormat::rgba8_sint) {
                EXPECT(captured.int32[0] == 1 && captured.int32[1] == 2 && captured.int32[2] == 127 && captured.int32[3] == -1);
            } else {
                // UNORM and sRGB views round and clamp like D3D (render_target_clear_color): 1.5 and 256 are 1.0, -1 is 0.
                const VkClearColorValue converted = leonida::detail::render_target_clear_color(expected, color);
                EXPECT(std::memcmp(captured.float32, converted.float32, sizeof(captured)) == 0);
            }
            ++checks;
        }
    }
    // Whole-view clears must keep main's predication attachment path, including same-format UNORM.
    auto conditional = vk;
    conditional.begin_conditional_rendering = +[](VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {};
    conditional.end_conditional_rendering = end;
    conditional.prepare_predicate = [](VkBuffer source, VkDeviceSize offset, PredicateCapture& out) {
        out = {source, offset, reinterpret_cast<VkBuffer>(std::uintptr_t{3}), 0, VK_NULL_HANDLE}; return true;
    };
    conditional.capture_predicate = [](VkCommandBuffer, const PredicateCapture&) {};
    ResourceBinding predicate{};
    predicate.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{2});
    predicate.size = 8;
    for (const auto& [format, expected] : formats) {
        CommandList predicated;
        EXPECT(predicated.set_predication({2, 0, d3d12_predication::equal_zero}));
        EXPECT(predicated.clear_render_target({1, 0, 0, 1, format}, color) && predicated.close());
        PreparedList prepared;
        reset_counters();
        const auto resolve = [&](ResourceId id) { return id == 2 ? &predicate : &binding; };
        EXPECT(prepare(predicated, resolve, conditional, prepared) == ReplayResult::success);
        EXPECT(prepared.commands[1].rects.size() == 1);
        EXPECT(prepared.commands[1].rects[0].rect.extent.width == binding.width);
        EXPECT(prepared.commands[1].rects[0].rect.extent.height == binding.height);
        EXPECT(requested_format == expected);
        ++checks;
    }
    // Eight bytes, not just the low word, must fit; missing/rejected preparation cannot emit any clears.
    {
        CommandList list;
        EXPECT(list.set_predication({2, 0, d3d12_predication::equal_zero}));
        EXPECT(list.clear_render_target({1, 0, 0, 1, ColorViewFormat::rgba8_uint}, color) && list.close());
        auto short_buffer = predicate; short_buffer.size = 4;
        PreparedList prepared;
        auto resolve = [&](ResourceId id) { return id == 2 ? &short_buffer : &binding; };
        EXPECT(prepare(list, resolve, conditional, prepared) == ReplayResult::invalid_predication);
        short_buffer.size = 8;
        EXPECT(prepare(list, resolve, conditional, prepared) == ReplayResult::success);
        auto unavailable = conditional; unavailable.prepare_predicate = {};
        EXPECT(prepare(list, resolve, unavailable, prepared) == ReplayResult::unsupported);
        unavailable = conditional;
        unavailable.prepare_predicate = [](VkBuffer, VkDeviceSize, PredicateCapture&) { return false; };
        EXPECT(prepare(list, resolve, unavailable, prepared) == ReplayResult::unsupported);
        checks += 4;
    }
    {
        QueryPool statistics{};
        statistics.pool = reinterpret_cast<VkQueryPool>(std::uintptr_t{4});
        statistics.heap_type = d3d12_query_heap::pipeline_statistics; statistics.count = 1;
        auto queries = conditional;
        queries.reset_query_pool = +[](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {};
        queries.begin_query = +[](VkCommandBuffer, VkQueryPool, std::uint32_t, VkQueryControlFlags) {};
        queries.end_query = +[](VkCommandBuffer, VkQueryPool, std::uint32_t) {};
        queries.write_timestamp = +[](VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, std::uint32_t) {};
        queries.copy_query_pool_results = +[](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t,
                                              VkBuffer, VkDeviceSize, VkDeviceSize, VkQueryResultFlags) {};
        queries.fill_buffer = +[](VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, std::uint32_t) {};
        queries.query_pool = [&](QueryHeapId) { return &statistics; };
        for (bool inside : {false, true}) {
            CommandList list;
            if (!inside) EXPECT(list.set_predication({2, 0, d3d12_predication::equal_zero}));
            EXPECT(list.begin_query({3, d3d12_query::pipeline_statistics, 0}));
            if (inside) EXPECT(list.set_predication({2, 0, d3d12_predication::equal_zero}));
            EXPECT(list.end_query({3, d3d12_query::pipeline_statistics, 0}) && list.close());
            PreparedList prepared;
            const auto resolve = [&](ResourceId id) { return id == 2 ? &predicate : &binding; };
            EXPECT(prepare(list, resolve, queries, prepared) ==
                   (inside ? ReplayResult::unsupported : ReplayResult::success));
            EXPECT(prepare(list, resolve, queries, prepared, true) == ReplayResult::success);
            EXPECT(prepared.statistics_segment_count == (inside ? 2u : 0u));
            ++checks;
        }
    }
    auto layered = binding;
    layered.mip_levels = 3; layered.array_layers = 4;
    for (const auto& [format, expected] : formats) {
        for (unsigned mip = 0; mip < 3; ++mip) {
            CommandList subresources;
            const Rect oversized{-2, -3, 20, 30};
            EXPECT(subresources.clear_render_target({1, mip, 1, 2, format}, color, {&oversized, 1}));
            EXPECT(subresources.close() && run(subresources, layered, vk) == ReplayResult::success);
            EXPECT(requested_format == expected);
            EXPECT(requested_range.baseMipLevel == mip && requested_range.levelCount == 1);
            EXPECT(requested_range.baseArrayLayer == 1 && requested_range.layerCount == 2);
            EXPECT(rendering_layers == 2 && captured_rect.baseArrayLayer == 0 && captured_rect.layerCount == 2);
            EXPECT(render_area.extent.width == std::max(1u, 5u >> mip));
            EXPECT(render_area.extent.height == std::max(1u, 3u >> mip));
            EXPECT(captured_rect.rect.extent.width == render_area.extent.width);
            EXPECT(captured_rect.rect.extent.height == render_area.extent.height);
            ++checks;
        }
    }
    // A predicate must preserve the selected mip/range and use attachments even for a whole UNORM view.
    for (const auto& [format, expected] : formats) {
        for (unsigned mip = 0; mip < 3; ++mip) {
            for (bool partial : {false, true}) {
                CommandList list;
                const Rect oversized{-1, -1, 8, 8};
                EXPECT(list.set_predication({2, 0, d3d12_predication::equal_zero}));
                EXPECT(list.clear_render_target({1, mip, 1, 2, format}, color,
                    partial ? std::span<const Rect>(&oversized, 1) : std::span<const Rect>{}));
                EXPECT(list.close());
                const auto resolve = [&](ResourceId id) { return id == 2 ? &predicate : &layered; };
                reset_counters();
                EXPECT(replay(list, VK_NULL_HANDLE, resolve, conditional) == ReplayResult::success);
                EXPECT(image_clears == 0 && attachment_clears == 1);
                EXPECT(requested_format == expected && requested_range.baseMipLevel == mip);
                EXPECT(requested_range.baseArrayLayer == 1 && requested_range.layerCount == 2);
                EXPECT(render_area.extent.width == std::max(1u, 5u >> mip));
                EXPECT(render_area.extent.height == std::max(1u, 3u >> mip));
                EXPECT(rendering_layers == 2 && captured_rect.baseArrayLayer == 0 && captured_rect.layerCount == 2);
                EXPECT(captured_rect.rect.offset.x == 0 && captured_rect.rect.offset.y == 0);
                EXPECT(captured_rect.rect.extent.width == render_area.extent.width);
                EXPECT(captured_rect.rect.extent.height == render_area.extent.height);
                ++checks;
            }
        }
    }
    const auto rejects = [&](ResourceBinding r, TargetView view, ReplayResult expected = ReplayResult::invalid_clear) {
        CommandList list;
        if (!list.clear_render_target(view, color) || !list.close()) return false;
        return run(list, r, vk) == expected && !image_clears && !attachment_clears;
    };
    EXPECT(rejects(binding, {1, 0, 0, 1})); // A typeless resource has no default typed RTV format.
    EXPECT(rejects(binding, {1, 0, 0, 1, static_cast<ColorViewFormat>(99)}));
    for (const TargetView view : {
             TargetView{1, 3, 0, 1, ColorViewFormat::rgba8_uint},
             TargetView{1, UINT32_MAX, 0, 1, ColorViewFormat::rgba8_uint},
             TargetView{1, 0, 4, 1, ColorViewFormat::rgba8_uint},
             TargetView{1, 0, 3, 2, ColorViewFormat::rgba8_uint},
             TargetView{1, 0, 1, UINT32_MAX, ColorViewFormat::rgba8_uint},
             TargetView{1, 0, UINT32_MAX, 2, ColorViewFormat::rgba8_uint}}) {
        EXPECT(rejects(layered, view, ReplayResult::invalid_subresource));
        ++checks;
    }
    CommandList empty_range;
    EXPECT(!empty_range.clear_render_target({1, 0, 0, 0, ColorViewFormat::rgba8_uint}, color));
    ++checks;
    // Both a range ending at the last layer and the complete layer range remain valid.
    for (const TargetView view : {TargetView{1, 2, 3, 1, ColorViewFormat::rgba8_uint},
                                  TargetView{1, 2, 0, 4, ColorViewFormat::rgba8_uint}}) {
        CommandList list;
        const Rect whole{0, 0, 1, 1};
        EXPECT(list.clear_render_target(view, color, {&whole, 1}) && list.close());
        EXPECT(run(list, layered, vk) == ReplayResult::success);
        EXPECT(requested_range.baseMipLevel == 2 && requested_range.baseArrayLayer == view.first_layer);
        EXPECT(requested_range.layerCount == view.layer_count && rendering_layers == view.layer_count);
        ++checks;
    }
    auto wrong = binding; wrong.typeless = false;
    EXPECT(rejects(wrong, {1, 0, 0, 1, ColorViewFormat::rgba8_srgb}));
    wrong = binding; wrong.format = VK_FORMAT_R32_SFLOAT;
    EXPECT(rejects(wrong, {1, 0, 0, 1, ColorViewFormat::rgba8_uint}));
    wrong = binding; wrong.mip_levels = 2;
    EXPECT(rejects(wrong, {1, 2, 0, 1, ColorViewFormat::rgba8_uint}, ReplayResult::invalid_subresource));
    wrong = binding; wrong.array_layers = 2;
    EXPECT(rejects(wrong, {1, 0, 1, 2, ColorViewFormat::rgba8_srgb}, ReplayResult::invalid_subresource));
    wrong = binding; wrong.volume = true;
    EXPECT(rejects(wrong, {1, 0, 0, 1, ColorViewFormat::rgba8_srgb}));
    // A castable RGBA8 or BGRA8 image clears through a view in the other channel order (the device checked that the
    // list names it); without a castable list, or for a UINT image, the other order stays refused.
    {
        const auto bgra8_unorm = static_cast<ColorViewFormat>(VK_FORMAT_B8G8R8A8_UNORM);
        const auto bgra8_srgb = static_cast<ColorViewFormat>(VK_FORMAT_B8G8R8A8_SRGB);
        auto castable = binding; castable.typeless = false; castable.castable_view_formats = true;
        auto castable_bgra = castable; castable_bgra.format = VK_FORMAT_B8G8R8A8_UNORM;
        const std::tuple<ResourceBinding, ColorViewFormat, VkFormat> crossings[] = {
            {castable, bgra8_unorm, VK_FORMAT_B8G8R8A8_UNORM},
            {castable, bgra8_srgb, VK_FORMAT_B8G8R8A8_SRGB},
            {castable_bgra, ColorViewFormat::rgba8_unorm, VK_FORMAT_R8G8B8A8_UNORM},
            {castable_bgra, ColorViewFormat::rgba8_srgb, VK_FORMAT_R8G8B8A8_SRGB},
        };
        for (const auto& [image, format, expected] : crossings) {
            CommandList cross;
            EXPECT(cross.clear_render_target({1, 0, 0, 1, format}, color) && cross.close());
            EXPECT(run(cross, image, vk) == ReplayResult::success);
            EXPECT(image_clears == 0 && attachment_clears == 1 && requested_format == expected);
            ++checks;
        }
        auto typed = castable; typed.castable_view_formats = false;
        EXPECT(rejects(typed, {1, 0, 0, 1, bgra8_unorm}));
        EXPECT(rejects(binding, {1, 0, 0, 1, bgra8_unorm}));  // typeless RGBA8: another typeless family
        auto uint_image = castable; uint_image.format = VK_FORMAT_R8G8B8A8_UINT;
        EXPECT(rejects(uint_image, {1, 0, 0, 1, bgra8_unorm}));
        checks += 3;
    }
    {
        auto msaa = binding; msaa.samples = 4;
        for (const auto [format, expected] : formats) {
            CommandList ms_list;
            EXPECT(ms_list.clear_render_target({1, 0, 0, 1, format}, color) && ms_list.close());
            EXPECT(run(ms_list, msaa, vk) == ReplayResult::success);
            const bool attachment = format != ColorViewFormat::rgba8_unorm;
            EXPECT(image_clears == (attachment ? 0u : 1u) && attachment_clears == (attachment ? 1u : 0u));
            EXPECT(!attachment || (requested_format == expected && rendering_layers == 1));
            ++checks;
        }
        auto msaa_layered = layered; msaa_layered.mip_levels = 1; msaa_layered.samples = 4;
        CommandList ms_layer_list;
        EXPECT(ms_layer_list.clear_render_target({1, 0, 1, 2, ColorViewFormat::rgba8_srgb}, color) &&
               ms_layer_list.close());
        EXPECT(run(ms_layer_list, msaa_layered, vk) == ReplayResult::success);
        EXPECT(attachment_clears == 1 && requested_format == VK_FORMAT_R8G8B8A8_SRGB);
        EXPECT(requested_range.baseArrayLayer == 1 && requested_range.layerCount == 2 && rendering_layers == 2);
        ++checks;
    }
    checks += 7;
    CommandList list;
    EXPECT(list.clear_render_target({1, 0, 0, 1, ColorViewFormat::rgba8_srgb}, color) && list.close());
    auto unavailable = vk; unavailable.begin_rendering = nullptr;
    EXPECT(run(list, binding, unavailable) == ReplayResult::unsupported && !image_clears && !attachment_clears);
    CommandList same_format;
    EXPECT(same_format.clear_render_target({1, 0, 0, 1, ColorViewFormat::rgba8_unorm}, color) && same_format.close());
    EXPECT(run(same_format, binding, unavailable) == ReplayResult::success && image_clears == 1 && !attachment_clears);
    ++checks;
    auto legacy = binding;
    legacy.typeless = false;
    legacy.format = VK_FORMAT_UNDEFINED; // Existing ownerless float-only replay clients omit this metadata.
    CommandList legacy_list;
    EXPECT(legacy_list.clear_render_target({1, 0, 0, 1}, color) && legacy_list.close());
    EXPECT(run(legacy_list, legacy, unavailable) == ReplayResult::success && image_clears == 1 && !attachment_clears);
    EXPECT(std::memcmp(captured.float32, color.data(), sizeof(captured)) == 0);
    ++checks;
    CommandList clipped;
    const Rect outside{20, 20, 30, 30};
    EXPECT(clipped.clear_render_target({1, 0, 0, 1, ColorViewFormat::rgba8_srgb}, color, {&outside, 1}) && clipped.close());
    EXPECT(run(clipped, binding, vk) == ReplayResult::success && !image_clears && !attachment_clears);
    EXPECT(requested_format == VK_FORMAT_UNDEFINED);
    CommandList invalid_batch;
    EXPECT(invalid_batch.clear_render_target({1, 0, 0, 1, ColorViewFormat::rgba8_srgb}, color));
    EXPECT(invalid_batch.clear_render_target({1, 0, 0, 1, static_cast<ColorViewFormat>(99)}, color) && invalid_batch.close());
    EXPECT(run(invalid_batch, binding, vk) == ReplayResult::invalid_clear && !image_clears && !attachment_clears);
    checks += 3;
    std::printf("typeless_rtv_test: %u format, snapshot, whole/rectangle and rejection scenarios passed\n", checks);
    return 0;
}
