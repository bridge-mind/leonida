#include "predicated_copy_controls.h"
// Record synthetic draws between the physical query boundaries emitted by the Leonida replay queue.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;
constexpr std::uint32_t side = 8;
constexpr std::uint32_t query_count = 10;
constexpr std::uint32_t statistics_count = 9;
constexpr std::uint32_t statistics_words = 11;  // D3D12_QUERY_DATA_PIPELINE_STATISTICS

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
    VkQueryPool occlusion = VK_NULL_HANDLE, second = VK_NULL_HANDLE, statistics = VK_NULL_HANDLE;

    ~Fixture() {
        const auto destroy_pool = d.fn<PFN_vkDestroyQueryPool>("vkDestroyQueryPool");
        const auto destroy_shader = d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
        if (statistics) destroy_pool(d.device, statistics, nullptr);
        if (second) destroy_pool(d.device, second, nullptr);
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
        const auto scissor_state = VK_DYNAMIC_STATE_SCISSOR;
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = 1;
        dynamic.pDynamicStates = &scissor_state;
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &vi;
        info.pInputAssemblyState = &ia;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &ms;
        info.pColorBlendState = &blend;
        info.pDynamicState = &dynamic;
        info.layout = layout;
        if (d.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(
                d.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS)
            return false;
        const auto pool = d.fn<PFN_vkCreateQueryPool>("vkCreateQueryPool");
        VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query.queryType = VK_QUERY_TYPE_OCCLUSION;
        query.queryCount = query_count;
        if (pool(d.device, &query, nullptr, &occlusion) != VK_SUCCESS ||
            pool(d.device, &query, nullptr, &second) != VK_SUCCESS) return false;
        if (d.pipeline_statistics_query) {
            query.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            query.queryCount = statistics_count;
            query.pipelineStatistics = 0x7ff;  // the same counters as the D3D12 heap
            if (pool(d.device, &query, nullptr, &statistics) != VK_SUCCESS) return false;
        }
        return true;
    }

    void draw(VkCommandBuffer cb, std::uint32_t extent) const {
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea.extent = {side, side};
        rendering.layerCount = 1;
        d.fn<PFN_vkCmdBeginRendering>("vkCmdBeginRendering")(cb, &rendering);
        d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        const VkRect2D scissor{{}, {extent, extent}};
        d.fn<PFN_vkCmdSetScissor>("vkCmdSetScissor")(cb, 0, 1, &scissor);
        d.fn<PFN_vkCmdDraw>("vkCmdDraw")(cb, 3, 1, 0, 0);
        d.fn<PFN_vkCmdEndRendering>("vkCmdEndRendering")(cb);
    }
};

// Main's replay has no D3D12 draws yet. Insert one native Vulkan draw while each actual query recorded
// by the queue is active, immediately before its end. Everything else goes to the real Vulkan driver.
struct Draws {
    Fixture& fixture;
    std::vector<std::uint32_t> extents;
    std::size_t next = 0;
    bool overflow = false;
    bool every = false;    // --bisect: one 1x1 draw before every end, no schedule
    bool no_draw = false;  // --bisect --no-draw: no draws at all, every query stays empty
};
Draws* draws = nullptr;

VKAPI_ATTR void VKAPI_CALL end_query_with_draw(VkCommandBuffer cb, VkQueryPool pool, std::uint32_t index) {
    if (draws->every) {
        if (!draws->no_draw) draws->fixture.draw(cb, 1);
    } else if (draws->next == draws->extents.size()) {
        draws->overflow = true;
    } else if (const auto extent = draws->extents[draws->next++]) {
        draws->fixture.draw(cb, extent);
    }
    draws->fixture.d.fn<PFN_vkCmdEndQuery>("vkCmdEndQuery")(cb, pool, index);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device_proc(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkCmdEndQuery") == 0) return reinterpret_cast<PFN_vkVoidFunction>(end_query_with_draw);
    return draws->fixture.d.device_proc(device, name);
}

// --bisect (D3D-189, #1540): the test's query patterns one by one, each in its own submission. Ungated, each
// submission is waited for before the next, so the last "running" line before a hang or a lost device names the
// pattern. --gated queues them all behind one semaphore the CPU signals afterwards, as the full test does. Named
// phases run only those, in the order below. Only completion is checked, not values.
// Two switches separate the draw from the resolve: --no-draw leaves every query empty (no rendering scope inside
// it), --no-resolve records the queries but never copies or sums their results.
struct Bisect {
    bool enabled = false;
    bool gated = false;
    bool no_draw = false;
    bool no_resolve = false;
    std::vector<std::string> phases;
    bool wants(const char* name) const {
        return phases.empty() || std::find(phases.begin(), phases.end(), name) != phases.end();
    }
};

int run(vktest::Device& d, const char* vertex, const char* fragment, const Bisect& bisect) {
    std::uint32_t family_count = 0;
    const auto families = d.instance_fn<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        "vkGetPhysicalDeviceQueueFamilyProperties");
    families(d.physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(family_count);
    families(d.physical, &family_count, queues.data());
    if (!d.occlusion_query_precise || !(queues[d.family].queueFlags & VK_QUEUE_COMPUTE_BIT)) return 77;

    Fixture fixture{d};
    EXPECT(fixture.create(vertex, fragment));
    constexpr std::uint32_t bursts = 130;
    Draws schedule{fixture, {8, 4, 2, 8, 4, 2, 3}};
    schedule.extents.insert(schedule.extents.end(), 2 * bursts + 1, 1);
    // The statistics lists below: a split query (two segments), two unsplit references and a second split query.
    if (d.pipeline_statistics_query) schedule.extents.insert(schedule.extents.end(), {3, 5, 3, 5, 3, 5});
    schedule.extents.insert(schedule.extents.end(), {2, 0, 0, 0, 1});
    // The D3D-252 lists: three crossing statistics segment ends, three statistics references with the same draws,
    // then the kind overlap's three segment ends and three occlusion references carrying the same draws.
    if (d.pipeline_statistics_query) schedule.extents.insert(schedule.extents.end(), {7, 9, 11, 7, 9, 11});
    schedule.extents.insert(schedule.extents.end(), {19, 21, 23, 19, 21, 23});
    schedule.every = bisect.enabled;
    schedule.no_draw = bisect.no_draw;
    draws = &schedule;
    VulkanContext context(get_device_proc, d.device, true, d.conditional_rendering);
    EXPECT(context.loaded());
    VkPhysicalDeviceMemoryProperties memory{};
    const auto get_memory = d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties");
    get_memory(d.physical, &memory);
    context.set_occlusion_resolve_memory(memory);
    VkPhysicalDeviceProperties properties{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &properties);
    context.configure_predication(get_device_proc, memory, properties.limits.maxStorageBufferRange, d.family);
    VulkanFence gate(context, 0);
    PredicatedCopyControl copy_control;
    EXPECT(copy_control.create(d, context, properties.limits));
    VulkanQueue queue(context, d.queue, d.family);
    EXPECT(queue.ok() && gate.semaphore());
    // The gate stalls the GPU while the queue replays (the point of this test). A Wait on a fence with no submitted
    // Signal is held on the CPU instead (queue_holds.h), so the gate is declared as having one: its Signal is
    // "submitted" and the CPU completes it later, as a GPU-side signal behind long work would be.
    context.holds().signaled(gate.semaphore(), 1);
    struct OpenGate { VulkanFence& fence; ~OpenGate() { fence.signal(1); } } open_gate{gate};
    constexpr QueryHeapId a = 1, b = 2, stats = 3;
    constexpr auto occ = d3d12_query::occlusion;
    constexpr auto binary = d3d12_query::binary_occlusion;
    for (const auto& [id, pool] : {std::pair{a, fixture.occlusion}, std::pair{b, fixture.second}})
        context.add_query_pool(id, {pool, d3d12_query_heap::occlusion, query_count, true,
                                    std::vector<bool>(query_count, false)});

    constexpr ResourceId dst_id = 4, predicate_id = 5;
    constexpr VkDeviceSize binding_offset = 24, size = 256;
    void* mapping = nullptr;
    ResourceBinding dst{};
    dst.buffer = d.buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mapping);
    dst.buffer_offset = binding_offset;
    dst.size = size - binding_offset;
    dst.host_readable = true;  // ResolveQueryData writes results that the CPU reads below.
    EXPECT(dst.buffer && mapping);
    std::memset(mapping, 0xcd, size);
    context.add_resource(dst_id, dst);
    if (d.conditional_rendering) {
        void* words = nullptr;
        ResourceBinding predicate{};
        predicate.size = 8;
        predicate.buffer = d.buffer(8, VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &words);
        EXPECT(predicate.buffer && words);
        std::memset(words, 0, 8);
        context.add_resource(predicate_id, predicate);
    }

    if (bisect.enabled) {
        std::vector<std::pair<const char*, CommandList>> lists;
        const auto add = [&](const char* name) -> CommandList* {
            if (!bisect.wants(name)) return nullptr;
            return &lists.emplace_back(name, CommandList{}).second;
        };
        const auto resolve = [&](CommandList* l, const ResolveQueries& r) {
            return bisect.no_resolve || l->resolve_queries(r);
        };
        if (auto* l = add("crossing")) {  // two queries of one heap overlap; resolve while one is active
            EXPECT(l->begin_query({a, occ, 0}) && l->begin_query({a, occ, 1}) && l->end_query({a, occ, 0}));
            EXPECT(resolve(l, {a, occ, 0, 1, dst_id, 8}) && l->end_query({a, occ, 1}));
            EXPECT(resolve(l, {a, occ, 0, 2, dst_id, 24}));
        }
        if (auto* l = add("active")) {  // a resolve inside one active query, without an overlap
            EXPECT(l->begin_query({a, occ, 0}) && l->end_query({a, occ, 0}) && l->begin_query({a, occ, 3}));
            EXPECT(resolve(l, {a, occ, 0, 1, dst_id, 128}) && l->end_query({a, occ, 3}));
        }
        if (auto* l = add("nested")) {  // the same index in two heaps, one inside the other
            EXPECT(l->begin_query({a, occ, 2}) && l->begin_query({b, occ, 2}));
            EXPECT(l->end_query({b, occ, 2}) && l->end_query({a, occ, 2}));
            EXPECT(resolve(l, {a, occ, 2, 1, dst_id, 40}) && resolve(l, {b, occ, 2, 1, dst_id, 48}));
        }
        if (auto* l = add("unwritten")) {  // a query never begun: zero fill
            EXPECT(resolve(l, {b, occ, 0, 1, dst_id, 56}));
        }
        if (auto* l = add("binary")) {  // the binary-occlusion compute resolve
            EXPECT(l->begin_query({a, binary, 3}) && l->end_query({a, binary, 3}));
            EXPECT(resolve(l, {a, binary, 3, 1, dst_id, 64}));
        }
        if (d.conditional_rendering) {
            if (auto* l = add("predicated")) {  // both compute resolves under a false predicate
                EXPECT(l->begin_query({a, binary, 3}) && l->end_query({a, binary, 3}));
                EXPECT(l->set_predication({predicate_id, 0, d3d12_predication::equal_zero}));
                EXPECT(resolve(l, {a, binary, 3, 1, dst_id, 64}));
                EXPECT(resolve(l, {a, occ, 0, 1, dst_id, 72}));
                EXPECT(resolve(l, {a, binary, 3, 1, dst_id, 80}));
                EXPECT(copy_control.record(*l));
            }
        }
        if (auto* l = add("chunked")) {  // 2 * bursts + 1 segments: the summing resolve takes two chunks
            EXPECT(l->begin_query({a, occ, 3}));
            for (std::uint32_t i = 0; i < bursts; ++i) EXPECT(l->begin_query({b, occ, 0}) && l->end_query({b, occ, 0}));
            EXPECT(l->end_query({a, occ, 3}));
            EXPECT(resolve(l, {a, occ, 3, 1, dst_id, 88}) && resolve(l, {b, occ, 0, 1, dst_id, 96}));
        }
        if (auto* l = add("replace")) {  // every heap query begun and ended again, then resolved
            for (const auto& [heap, index] : {std::pair{a, 0u}, std::pair{a, 1u}, std::pair{a, 2u}, std::pair{b, 2u}})
                EXPECT(l->begin_query({heap, occ, index}) && l->end_query({heap, occ, index}));
            EXPECT(resolve(l, {a, occ, 0, 2, dst_id, 112}));
        }
        EXPECT(!lists.empty());
        std::printf("bisect:%s%s%s\n", bisect.gated ? " gated" : " ungated", bisect.no_draw ? " no-draw" : " draws",
                    bisect.no_resolve ? " no-resolve" : " resolves");
        // After a TDR, AMD's Windows driver lets vkQueueWaitIdle succeed (#1540); every timeline semaphore then reads
        // UINT64_MAX, so the gate's value is checked as well as what the context has seen.
        const auto get_value = d.fn<PFN_vkGetSemaphoreCounterValue>("vkGetSemaphoreCounterValue");
        const auto lost = [&] {
            std::uint64_t value = 0;
            return context.is_device_lost() || get_value(d.device, gate.semaphore(), &value) != VK_SUCCESS ||
                   value == UINT64_MAX;
        };
        if (bisect.gated) queue.wait(gate.semaphore(), 1);
        for (auto& [name, list] : lists) {
            EXPECT(list.close());
            std::printf("running %s\n", name);
            std::fflush(stdout);
            const CommandList* submit[] = {&list};
            const ReplayResult result = queue.execute(submit);
            if (result != ReplayResult::success) {
                std::printf("%s: execute returned %d\n", name, int(result));
                return 1;
            }
            if (!bisect.gated && (!queue.idle() || lost())) {
                std::printf("%s: device lost\n", name);
                return 1;
            }
        }
        if (bisect.gated) {
            EXPECT(gate.signal(1));
            if (!queue.idle() || lost()) {
                std::printf("gated: device lost\n");
                return 1;
            }
        }
        std::printf("bisect: every phase completed%s\n", bisect.gated ? " (gated)" : "");
        return 0;
    }

    CommandList crossing;
    EXPECT(crossing.begin_query({a, occ, 0}));
    EXPECT(crossing.begin_query({a, occ, 1}));
    EXPECT(crossing.end_query({a, occ, 0}));
    EXPECT(crossing.resolve_queries({a, occ, 0, 1, dst_id, 8}));   // B is still active
    EXPECT(crossing.end_query({a, occ, 1}));
    EXPECT(crossing.close());
    CommandList nested;
    EXPECT(nested.begin_query({a, occ, 2}));
    EXPECT(nested.begin_query({b, occ, 2}));                       // same index, different logical heaps
    EXPECT(nested.end_query({b, occ, 2}));
    EXPECT(nested.end_query({a, occ, 2}));
    EXPECT(nested.close());
    const CommandList* first[] = {&crossing, &nested};
    queue.wait(gate.semaphore(), 1);
    EXPECT(queue.execute(first) == ReplayResult::success);

    CommandList later;
    EXPECT(later.resolve_queries({a, occ, 0, 3, dst_id, 24}));
    EXPECT(later.resolve_queries({b, occ, 2, 1, dst_id, 48}));
    EXPECT(later.resolve_queries({b, occ, 0, 1, dst_id, 56}));    // never begun
    EXPECT(later.begin_query({a, binary, 3}));
    EXPECT(later.end_query({a, binary, 3}));
    if (d.conditional_rendering) EXPECT(later.set_predication({predicate_id, 0, d3d12_predication::equal_zero}));
    // Alternate the two compute pipelines on the shared scratch buffer, all under the false predicate.
    EXPECT(later.resolve_queries({a, binary, 3, 1, dst_id, 64}));
    EXPECT(later.resolve_queries({a, occ, 0, 1, dst_id, 72}));
    EXPECT(later.resolve_queries({a, binary, 3, 1, dst_id, 80}));
    EXPECT(copy_control.record(later));
    EXPECT(later.close());
    const CommandList* second[] = {&later};
    EXPECT(queue.execute(second) == ReplayResult::success);

    CommandList chunked;
    EXPECT(chunked.begin_query({a, occ, 3}));
    for (std::uint32_t i = 0; i < bursts; ++i) {
        EXPECT(chunked.begin_query({b, occ, 0}));
        EXPECT(chunked.end_query({b, occ, 0}));
    }
    EXPECT(chunked.end_query({a, occ, 3}));
    EXPECT(chunked.resolve_queries({a, occ, 3, 1, dst_id, 88}));
    EXPECT(chunked.resolve_queries({b, occ, 0, 1, dst_id, 96}));
    EXPECT(chunked.close());
    const CommandList* third[] = {&chunked};
    EXPECT(queue.execute(third) == ReplayResult::success);

    CommandList bundle;
    EXPECT(bundle.begin_query({a, occ, 0}));
    EXPECT(bundle.end_query({a, occ, 0}));
    EXPECT(bundle.close());
    CommandList duplicate;
    EXPECT(duplicate.begin_query({a, occ, 0}));
    EXPECT(duplicate.execute_bundle(bundle));
    EXPECT(duplicate.end_query({a, occ, 0}));
    EXPECT(duplicate.close());
    const CommandList* invalid[] = {&duplicate};
    EXPECT(queue.execute(invalid) == ReplayResult::invalid_query);

    // D3D-162: an occlusion sum inside a pipeline-statistics query. Its dispatch must not reach the application's
    // counters, so the queue splits the statistics query around the resolve and sums the segments. Each segment
    // gets one native draw before it ends (extents 3 and 5); two unsplit queries with the same draws are the
    // reference, and CSInvocations stays 0 because no application compute runs.
    constexpr ResourceId stats_dst_id = 6;
    constexpr VkDeviceSize stats_result = statistics_words * sizeof(std::uint64_t), stats_slots = 10;
    void* stats_mapping = nullptr;
    ResourceBinding stats_dst{};
    if (fixture.statistics) {
        constexpr auto pipeline_statistics = d3d12_query::pipeline_statistics;
        context.add_query_pool(stats, {fixture.statistics, d3d12_query_heap::pipeline_statistics, statistics_count,
                                       false, std::vector<bool>(statistics_count, false)});
        stats_dst.size = stats_slots * stats_result;
        stats_dst.buffer = d.buffer(stats_dst.size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &stats_mapping);
        EXPECT(stats_dst.buffer && stats_mapping);
        std::memset(stats_mapping, 0xcd, stats_dst.size);
        context.add_resource(stats_dst_id, stats_dst);
        CommandList split;
        EXPECT(split.begin_query({stats, pipeline_statistics, 0}));
        EXPECT(split.resolve_queries({a, occ, 3, 1, dst_id, 0}));  // 261 segments: a sum dispatch
        EXPECT(split.end_query({stats, pipeline_statistics, 0}));
        EXPECT(split.resolve_queries({stats, pipeline_statistics, 0, 1, stats_dst_id, 0}));
        EXPECT(split.close());
        CommandList reference;
        for (std::uint32_t index : {1u, 2u}) {
            EXPECT(reference.begin_query({stats, pipeline_statistics, index}));
            EXPECT(reference.end_query({stats, pipeline_statistics, index}));
        }
        EXPECT(reference.resolve_queries({stats, pipeline_statistics, 1, 2, stats_dst_id, stats_result}));
        EXPECT(reference.close());
        // The statistics sum of query 0 is a helper dispatch too, inside query 3.
        CommandList nested_sum;
        EXPECT(nested_sum.begin_query({stats, pipeline_statistics, 3}));
        EXPECT(nested_sum.resolve_queries({stats, pipeline_statistics, 0, 1, stats_dst_id, 4 * stats_result}));
        EXPECT(nested_sum.end_query({stats, pipeline_statistics, 3}));
        EXPECT(nested_sum.resolve_queries({stats, pipeline_statistics, 3, 1, stats_dst_id, 3 * stats_result}));
        EXPECT(nested_sum.close());
        const CommandList* statistics_lists[] = {&split, &reference};
        EXPECT(queue.execute(statistics_lists) == ReplayResult::success);
        const CommandList* nested_lists[] = {&nested_sum};
        EXPECT(queue.execute(nested_lists) == ReplayResult::success);
    }

    // Overwrite every result from the first physical pool while the GPU is still gated. The first two
    // submissions still refer to that pool and must be able to execute once the gate opens.
    CommandList replace;
    for (const auto& [heap, index] : {std::pair{a, 0u}, std::pair{a, 1u}, std::pair{a, 2u}, std::pair{b, 2u}}) {
        EXPECT(replace.begin_query({heap, occ, index}));
        EXPECT(replace.end_query({heap, occ, index}));
    }
    EXPECT(replace.resolve_queries({a, occ, 0, 2, dst_id, 112}));
    EXPECT(replace.close());
    const CommandList* fourth[] = {&replace};
    EXPECT(queue.execute(fourth) == ReplayResult::success);

    // A resolve inside one active query, with no overlap.
    CommandList single;
    EXPECT(single.begin_query({a, occ, 3}));
    EXPECT(single.resolve_queries({a, occ, 0, 1, dst_id, 128}));
    EXPECT(single.end_query({a, occ, 3}));
    EXPECT(single.close());
    const CommandList* fifth[] = {&single};
    EXPECT(queue.execute(fifth) == ReplayResult::success);
    // D3D-252: overlapping pipeline-statistics queries and overlapping occlusion kinds. The hook's draw lands in
    // the segment that is about to close, so each crossing query's sum covers its own intervals plus the shared
    // intersection; the reference queries carry the same draws unsplit, and the sums must match them.
    CommandList crossing_stats, stats_references, kind_overlap, kind_references;
    if (d.pipeline_statistics_query) {
        EXPECT(crossing_stats.begin_query({stats, d3d12_query::pipeline_statistics, 4}));
        EXPECT(crossing_stats.begin_query({stats, d3d12_query::pipeline_statistics, 5}));
        EXPECT(crossing_stats.end_query({stats, d3d12_query::pipeline_statistics, 4}));
        EXPECT(crossing_stats.end_query({stats, d3d12_query::pipeline_statistics, 5}));
        EXPECT(crossing_stats.resolve_queries({stats, d3d12_query::pipeline_statistics, 4, 1, stats_dst_id,
                                               5 * stats_result}));
        EXPECT(crossing_stats.resolve_queries({stats, d3d12_query::pipeline_statistics, 5, 1, stats_dst_id,
                                               6 * stats_result}));
        EXPECT(crossing_stats.close());
        for (std::uint32_t index : {6u, 7u, 8u}) {
            EXPECT(stats_references.begin_query({stats, d3d12_query::pipeline_statistics, index}));
            EXPECT(stats_references.end_query({stats, d3d12_query::pipeline_statistics, index}));
        }
        EXPECT(stats_references.resolve_queries({stats, d3d12_query::pipeline_statistics, 6, 3, stats_dst_id,
                                                 7 * stats_result}));
        EXPECT(stats_references.close());
    }
    EXPECT(kind_overlap.begin_query({a, occ, 4}));
    EXPECT(kind_overlap.begin_query({a, binary, 5}));
    EXPECT(kind_overlap.end_query({a, binary, 5}));
    EXPECT(kind_overlap.end_query({a, occ, 4}));
    EXPECT(kind_overlap.resolve_queries({a, occ, 4, 1, dst_id, 168}));
    EXPECT(kind_overlap.resolve_queries({a, binary, 5, 1, dst_id, 176}));
    EXPECT(kind_overlap.close());
    for (std::uint32_t index : {6u, 7u, 8u}) {
        EXPECT(kind_references.begin_query({b, occ, index}));
        EXPECT(kind_references.end_query({b, occ, index}));
    }
    EXPECT(kind_references.resolve_queries({b, occ, 6, 3, dst_id, 184}));
    EXPECT(kind_references.close());
    // The two statistics lists are recorded and closed only where the device has the feature, and execute()
    // refuses a list that is not closed (list_open): MoltenVK has no pipelineStatisticsQuery.
    std::vector<const CommandList*> overlap_lists;
    if (d.pipeline_statistics_query) overlap_lists = {&crossing_stats, &stats_references};
    overlap_lists.insert(overlap_lists.end(), {&kind_overlap, &kind_references});
    EXPECT(queue.execute(overlap_lists) == ReplayResult::success);
    EXPECT(!schedule.overflow && schedule.next == schedule.extents.size());
    EXPECT(gate.signal(1));
    EXPECT(queue.idle());

    std::map<VkDeviceSize, std::uint64_t> expected = {
        {8, 80}, {24, 80}, {32, 20}, {40, 84}, {48, 16}, {56, 0}, {64, 1}, {72, 80}, {80, 1},
        {88, 2 * bursts + 1}, {96, 1}, {112, 4}, {120, 0}, {128, 4},
    };
    if (fixture.statistics) expected[0] = 2 * bursts + 1;  // the sum resolved inside the statistics query
    {   // D3D-252: the ordinary result covers three segments (draws 19, 21 and 23); the binary result is the
        // shared segment's count normalized to 1. The reference queries carry the same draws unsplit, and their
        // own results at 184, 192 and 200 are the expectation for the sums.
        const auto word_at = [&](VkDeviceSize at) {
            std::uint64_t value = 0;
            std::memcpy(&value, static_cast<const char*>(mapping) + binding_offset + at, sizeof(value));
            return value;
        };
        const std::uint64_t first = word_at(184), shared = word_at(192), last = word_at(200);
        expected[168] = first + shared + last;
        expected[176] = 1;
        expected[184] = first;
        expected[192] = shared;
        expected[200] = last;
    }
    for (VkDeviceSize offset = 0; offset < size; offset += 8) {
        const auto it = expected.find(offset - binding_offset);
        const auto want = it == expected.end() ? 0xcdcdcdcdcdcdcdcdULL : it->second;
        std::uint64_t actual = 0;
        std::memcpy(&actual, static_cast<const char*>(mapping) + offset, sizeof(actual));
        if (actual != want) {
            std::fprintf(stderr, "at byte %llu: got %llu, wanted %llu\n", static_cast<unsigned long long>(offset),
                         static_cast<unsigned long long>(actual), static_cast<unsigned long long>(want));
            return 1;
        }
    }
    if (fixture.statistics) {
        // Slots: 0 the split query, 1 and 2 the references, 3 the second split query, 4 query 0 summed inside it.
        const auto word = [&](VkDeviceSize slot, std::uint32_t counter) {
            std::uint64_t value = 0;
            std::memcpy(&value, static_cast<const char*>(stats_mapping) + slot * stats_result + counter * 8, 8);
            return value;
        };
        constexpr std::uint32_t cs_invocations = 10;
        for (std::uint32_t counter = 0; counter < statistics_words; ++counter) {
            const std::uint64_t want = word(1, counter) + word(2, counter);
            for (VkDeviceSize slot : {0, 3, 4}) {
                if (word(slot, counter) != want) {
                    std::fprintf(stderr, "statistics slot %llu counter %u: got %llu, wanted %llu\n",
                                 static_cast<unsigned long long>(slot), counter,
                                 static_cast<unsigned long long>(word(slot, counter)),
                                 static_cast<unsigned long long>(want));
                    return 1;
                }
            }
        }
        EXPECT(word(0, cs_invocations) == 0 && word(1, 0) != 0 && word(2, 0) != 0);
        // D3D-252: the crossing statistics queries (slots 5 and 6) each cover one private interval plus the shared
        // intersection; slots 7 to 9 are the references with the same draws unsplit.
        for (std::uint32_t counter = 0; counter < statistics_words; ++counter) {
            if (word(5, counter) != word(7, counter) + word(8, counter) ||
                word(6, counter) != word(8, counter) + word(9, counter)) {
                std::fprintf(stderr, "crossing statistics counter %u: got %llu/%llu, wanted %llu/%llu\n", counter,
                             static_cast<unsigned long long>(word(5, counter)),
                             static_cast<unsigned long long>(word(6, counter)),
                             static_cast<unsigned long long>(word(7, counter) + word(8, counter)),
                             static_cast<unsigned long long>(word(8, counter) + word(9, counter)));
                return 1;
            }
        }
        EXPECT(word(5, cs_invocations) == 0 && word(6, cs_invocations) == 0);
        std::printf("statistics split around a sum: IA vertices %llu, PS invocations %llu, CS invocations 0, "
                    "equal to two unsplit queries\n",
                    static_cast<unsigned long long>(word(0, 0)), static_cast<unsigned long long>(word(0, 7)));
        std::printf("crossing statistics: IA vertices %llu/%llu, CS invocations 0, equal to the unsplit references\n",
                    static_cast<unsigned long long>(word(5, 0)), static_cast<unsigned long long>(word(6, 0)));
    }
    std::printf("crossing 80/20; nested 84/16; %u segments summed; alternating binary results 1/1; "
                "resolve inside an active query 4; offsets and gated overwrites exact\n",
                2 * bursts + 1);
    EXPECT(copy_control.check());
    return 0;
}

} // namespace

// vk_overlapping_occlusion_test <vert.spv> <frag.spv> [--no-statistics]
//                               [--bisect [--gated] [--no-draw] [--no-resolve] [phase...]]
// --no-statistics runs the test as a device without pipelineStatisticsQuery does (MoltenVK): no statistics pool,
// no statistics lists. It makes that path testable on a device that has the feature.
int main(int argc, char** argv) {
    EXPECT(argc >= 3);
    Bisect bisect;
    bool no_statistics = false;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-statistics") == 0) no_statistics = true;
        else if (std::strcmp(argv[i], "--bisect") == 0) bisect.enabled = true;
        else if (std::strcmp(argv[i], "--gated") == 0) bisect.gated = true;
        else if (std::strcmp(argv[i], "--no-draw") == 0) bisect.no_draw = true;
        else if (std::strcmp(argv[i], "--no-resolve") == 0) bisect.no_resolve = true;
        else bisect.phases.emplace_back(argv[i]);
    }
    EXPECT(argc == 3 + (no_statistics ? 1 : 0) || bisect.enabled);
    vktest::Device device;
    const auto status = device.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    EXPECT(status == vktest::DeviceStatus::ready);
    if (no_statistics) device.pipeline_statistics_query = false;
    std::printf("pipeline statistics: %s\n", device.pipeline_statistics_query ? "checked" : "not checked");
    const int result = run(device, argv[1], argv[2], bisect);
    EXPECT(device.validation_errors() == 0);
    return result;
}
