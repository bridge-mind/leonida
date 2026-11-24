#include "predicated_copy_controls.h"
// Seed an occlusion heap with independent Vulkan draws, then resolve it through the D3D12 replay queue.
// This exercises the real GPU query copy and normalization without needing D3D12 draw commands.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <cstring>
#include <fstream>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;
constexpr std::uint32_t side = 8;
constexpr std::uint32_t first_visible = 2, visible_count = 1026;
constexpr std::uint32_t last_gap = first_visible + visible_count;
constexpr std::uint32_t ordinary_index = last_gap + 1, query_count = ordinary_index + 1;

VkShaderModule load_shader(vktest::Device& d, const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const std::streamoff bytes = file.tellg();
    if (bytes <= 0 || bytes % 4 != 0) return VK_NULL_HANDLE;
    std::vector<std::uint32_t> code(static_cast<std::size_t>(bytes) / 4);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(code.data()), bytes)) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = static_cast<std::size_t>(bytes);
    info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &info, nullptr, &module) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return module;
}

struct Fixture {
    vktest::Device& d;
    VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkQueryPool occlusion = VK_NULL_HANDLE, statistics = VK_NULL_HANDLE;

    ~Fixture() {
        const auto destroy_pool = d.fn<PFN_vkDestroyQueryPool>("vkDestroyQueryPool");
        const auto destroy_shader = d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
        if (statistics) destroy_pool(d.device, statistics, nullptr);
        if (occlusion) destroy_pool(d.device, occlusion, nullptr);
        if (pipeline) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(d.device, pipeline, nullptr);
        if (layout) d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(d.device, layout, nullptr);
        if (fs) destroy_shader(d.device, fs, nullptr);
        if (vs) destroy_shader(d.device, vs, nullptr);
    }

    bool create(const char* vertex, const char* fragment) {
        vs = load_shader(d, vertex);
        fs = load_shader(d, fragment);
        if (!vs || !fs) return false;
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        if (d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(d.device, &layout_info, nullptr, &layout) !=
            VK_SUCCESS)
            return false;
        const VkPipelineShaderStageCreateInfo stages[] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main"},
        };
        const VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        const VkViewport vp{0, 0, float(side), float(side), 0, 1};
        const VkRect2D scissor{{}, {side, side}};
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = viewport.scissorCount = 1;
        viewport.pViewports = &vp;
        viewport.pScissors = &scissor;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        const VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        const VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &vi;
        info.pInputAssemblyState = &ia;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &ms;
        info.pColorBlendState = &blend;
        info.layout = layout;
        if (d.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(
                d.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS)
            return false;
        const auto pool = d.fn<PFN_vkCreateQueryPool>("vkCreateQueryPool");
        VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query.queryType = VK_QUERY_TYPE_OCCLUSION;
        query.queryCount = query_count;
        if (pool(d.device, &query, nullptr, &occlusion) != VK_SUCCESS) return false;
        if (d.pipeline_statistics_query) {
            query.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            query.queryCount = 2;
            query.pipelineStatistics = 0x7ff;  // the same counters as the D3D12 heap
            if (pool(d.device, &query, nullptr, &statistics) != VK_SUCCESS) return false;
        }
        return true;
    }

    bool draw() {
        const auto begin = d.fn<PFN_vkCmdBeginQuery>("vkCmdBeginQuery");
        const auto end = d.fn<PFN_vkCmdEndQuery>("vkCmdEndQuery");
        const auto draw = d.fn<PFN_vkCmdDraw>("vkCmdDraw");
        return d.run([&](VkCommandBuffer cb) {
            d.fn<PFN_vkCmdResetQueryPool>("vkCmdResetQueryPool")(cb, occlusion, 0, query_count);
            VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
            rendering.renderArea.extent = {side, side};
            rendering.layerCount = 1;
            d.fn<PFN_vkCmdBeginRendering>("vkCmdBeginRendering")(cb, &rendering);
            d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            begin(cb, occlusion, 1, 0);
            end(cb, occlusion, 1);
            for (std::uint32_t index = first_visible; index < last_gap; ++index) {
                begin(cb, occlusion, index, 0);
                draw(cb, 3, 1, 0, 0);
                end(cb, occlusion, index);
            }
            begin(cb, occlusion, ordinary_index, d.occlusion_query_precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
            draw(cb, 3, 1, 0, 0);
            end(cb, occlusion, ordinary_index);
            d.fn<PFN_vkCmdEndRendering>("vkCmdEndRendering")(cb);
            return true;
        });
    }
};

int run(vktest::Device& d, const char* vertex, const char* fragment) {
    std::uint32_t family_count = 0;
    const auto families = d.instance_fn<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        "vkGetPhysicalDeviceQueueFamilyProperties");
    families(d.physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(family_count);
    families(d.physical, &family_count, queues.data());
    if (!(queues[d.family].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
        std::puts("the graphics queue does not support compute");
        return 77;
    }
    Fixture fixture{d};
    EXPECT(fixture.create(vertex, fragment));
    EXPECT(fixture.draw());

    VulkanContext context(d.device_proc, d.device, true, d.conditional_rendering);
    EXPECT(context.loaded());
    VkPhysicalDeviceMemoryProperties memory{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical,
                                                                                                &memory);
    context.set_occlusion_resolve_memory(memory);
    VkPhysicalDeviceProperties properties{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &properties);
    context.configure_predication(d.device_proc, memory, properties.limits.maxStorageBufferRange, d.family);
    PredicatedCopyControl copy_control;
    EXPECT(copy_control.create(d, context, properties.limits));
    VulkanQueue queue(context, d.queue, d.family);
    EXPECT(queue.ok());
    constexpr QueryHeapId query_id = 1, stats_id = 2;
    QueryPool heap{fixture.occlusion, d3d12_query_heap::occlusion, query_count, d.occlusion_query_precise,
                   std::vector<bool>(query_count, true)};
    heap.ended[0] = heap.ended[last_gap] = false;
    context.add_query_pool(query_id, heap);
    constexpr ResourceId destination_id = 3, predicate_id = 4, misaligned_id = 5;
    constexpr VkDeviceSize binding_offset = 24, range_offset = 16, result_size = sizeof(std::uint64_t);
    constexpr VkDeviceSize ordinary_offset = range_offset + (last_gap + 2) * result_size;
    constexpr VkDeviceSize ordinary_zero_offset = ordinary_offset + 2 * result_size;
    constexpr VkDeviceSize rerun_offset = ordinary_zero_offset + 2 * result_size;
    constexpr VkDeviceSize bytes = binding_offset + rerun_offset + 6 * result_size;
    void* mapped = nullptr;
    const VkBuffer dst = d.buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mapped);
    EXPECT(dst && mapped);
    std::memset(mapped, 0xcd, bytes);
    ResourceBinding destination{};
    destination.buffer = dst;
    destination.buffer_offset = binding_offset;
    destination.size = bytes - binding_offset;
    destination.host_readable = true;  // ResolveQueryData writes results that the CPU reads below.
    context.add_resource(destination_id, destination);

    ResourceBinding misaligned = destination;
    misaligned.buffer_offset += 4;
    misaligned.size -= 4;
    context.add_resource(misaligned_id, misaligned);
    CommandList invalid;
    EXPECT(invalid.resolve_queries({query_id, d3d12_query::binary_occlusion, first_visible, 1, misaligned_id, 0}));
    EXPECT(invalid.close());
    const CommandList* invalid_list = &invalid;
    EXPECT(queue.execute({&invalid_list, 1}) == ReplayResult::invalid_query);

    CommandList binary;
    if (d.conditional_rendering) {
        void* predicate_mapping = nullptr;
        ResourceBinding predicate{};
        predicate.size = 8;
        predicate.buffer = d.buffer(predicate.size, VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    &predicate_mapping);
        EXPECT(predicate.buffer && predicate_mapping);
        std::memset(predicate_mapping, 0, predicate.size);
        context.add_resource(predicate_id, predicate);
        // A zero predicate suppresses dispatch, but it must not suppress a query resolve.
        EXPECT(binary.set_predication({predicate_id, 0, d3d12_predication::equal_zero}));
    }
    EXPECT(binary.resolve_queries({query_id, d3d12_query::binary_occlusion, 0, last_gap + 1, destination_id,
                                   range_offset}));
    EXPECT(binary.resolve_queries({query_id, d3d12_query::occlusion, ordinary_index, 1, destination_id,
                                   ordinary_offset}));
    // An ordinary un-ended query zeroed via vkCmdFillBuffer must also run unconditionally under a false predicate.
    EXPECT(binary.resolve_queries({query_id, d3d12_query::occlusion, 0, 1, destination_id, ordinary_zero_offset}));
    EXPECT(copy_control.record(binary));
    EXPECT(binary.close());
    CommandList rerun;
    EXPECT(rerun.resolve_queries({query_id, d3d12_query::binary_occlusion, first_visible + 1, 3, destination_id,
                                  rerun_offset}));
    EXPECT(rerun.close());
    const CommandList* binary_list = &binary;
    const CommandList* rerun_list = &rerun;
    EXPECT(queue.execute({&binary_list, 1}) == ReplayResult::success);
    EXPECT(queue.execute({&rerun_list, 1}) == ReplayResult::success);  // same scratch, consecutive submissions
    EXPECT(queue.idle());

    std::uint64_t raw = 0;
    EXPECT(d.fn<PFN_vkGetQueryPoolResults>("vkGetQueryPoolResults")(
               d.device, fixture.occlusion, ordinary_index, 1, sizeof(raw), &raw, sizeof(raw),
               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS);
    EXPECT(raw > 0 && (!d.occlusion_query_precise || raw == side * side));
    for (VkDeviceSize offset = 0; offset < bytes; offset += result_size) {
        const VkDeviceSize range = binding_offset + range_offset;
        const VkDeviceSize rerun_begin = binding_offset + rerun_offset;
        std::uint64_t expected = 0xcdcdcdcdcdcdcdcd;
        if (offset >= range && offset < range + (last_gap + 1) * result_size) {
            const auto index = (offset - range) / result_size;
            expected = index >= first_visible && index < last_gap ? 1 : 0;
        } else if (offset == binding_offset + ordinary_offset) {
            expected = raw;
        } else if (offset == binding_offset + ordinary_zero_offset) {
            expected = 0;
        } else if (offset >= rerun_begin && offset < rerun_begin + 3 * result_size) {
            expected = 1;
        }
        std::uint64_t actual;
        std::memcpy(&actual, static_cast<const char*>(mapped) + offset, sizeof(actual));
        if (actual != expected) {
            std::fprintf(stderr, "offset %llu: got %llu, expected %llu\n",
                         static_cast<unsigned long long>(offset), static_cast<unsigned long long>(actual),
                         static_cast<unsigned long long>(expected));
            return 1;
        }
    }
    if (fixture.statistics) {
        // D3D-162: the normalisation dispatch of a binary resolve inside a pipeline-statistics query, and then the
        // statistics sum of that split query inside a second one. Neither helper dispatch may reach the
        // application's counters, and nothing else runs inside the queries, so every counter of both is 0.
        constexpr auto statistics = d3d12_query::pipeline_statistics;
        constexpr VkDeviceSize statistics_result = 11 * sizeof(std::uint64_t);
        context.add_query_pool(stats_id, {fixture.statistics, d3d12_query_heap::pipeline_statistics, 2, false,
                                          {false, false}});
        constexpr ResourceId statistics_id = 6;
        void* statistics_mapping = nullptr;
        ResourceBinding statistics_dst{};
        statistics_dst.size = 3 * statistics_result;
        statistics_dst.buffer = d.buffer(statistics_dst.size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &statistics_mapping);
        EXPECT(statistics_dst.buffer && statistics_mapping);
        std::memset(statistics_mapping, 0xcd, statistics_dst.size);
        context.add_resource(statistics_id, statistics_dst);
        CommandList overlap;
        EXPECT(overlap.begin_query({stats_id, statistics, 0}));
        EXPECT(overlap.resolve_queries({query_id, d3d12_query::binary_occlusion, first_visible, 1, destination_id, 0}));
        EXPECT(overlap.end_query({stats_id, statistics, 0}));
        EXPECT(overlap.close());
        CommandList nested;
        EXPECT(nested.begin_query({stats_id, statistics, 1}));
        EXPECT(nested.resolve_queries({stats_id, statistics, 0, 1, statistics_id, 0}));
        EXPECT(nested.end_query({stats_id, statistics, 1}));
        EXPECT(nested.resolve_queries({stats_id, statistics, 0, 2, statistics_id, statistics_result}));
        EXPECT(nested.close());
        const CommandList* overlap_list = &overlap;
        const CommandList* nested_list = &nested;
        EXPECT(queue.execute({&overlap_list, 1}) == ReplayResult::success);
        EXPECT(queue.execute({&nested_list, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        std::uint64_t visible = 0;
        std::memcpy(&visible, static_cast<const char*>(mapped) + binding_offset, sizeof(visible));
        EXPECT(visible == 1);
        for (VkDeviceSize offset = 0; offset < statistics_dst.size; offset += 8) {
            std::uint64_t counter = 0;
            std::memcpy(&counter, static_cast<const char*>(statistics_mapping) + offset, sizeof(counter));
            if (counter != 0) {
                std::fprintf(stderr, "statistics byte %llu: got %llu, expected 0\n",
                             static_cast<unsigned long long>(offset), static_cast<unsigned long long>(counter));
                return 1;
            }
        }
        std::puts("binary and statistics resolves inside pipeline-statistics queries: every counter 0");
    }
    std::printf("binary occlusion: %u visible, 3 zero, 3 repeated; ordinary result %llu; offsets and sentinels exact\n",
                visible_count, static_cast<unsigned long long>(raw));
    EXPECT(copy_control.check());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    EXPECT(argc == 3);
    vktest::Device device;
    const auto status = device.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    EXPECT(status == vktest::DeviceStatus::ready);
    const int result = run(device, argv[1], argv[2]);
    EXPECT(device.validation_errors() == 0);
    return result;
}
