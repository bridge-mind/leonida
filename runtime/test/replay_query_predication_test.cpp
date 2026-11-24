// ResolveQueryData is unconditional in D3D12, including un-ended slots zeroed via vkCmdFillBuffer. On Intel's
// Windows Vulkan driver, vkCmdFillBuffer after PredicateReducer::capture's compute dispatch is suppressed inside
// an active conditional rendering scope (D3D-148). Check that emit() suspends conditional rendering for the
// entire ResolveQueryData loop (not only before helper dispatches) and restores it afterward. Cases 5 and 6 check
// how resolves inside a pipeline-statistics query are replayed; cases 7 and 8 cover the other helper dispatches
// and their no-pool fallback (D3D-162). Cases 9 to 13 check overlapping queries: statistics over statistics, binary
// inside ordinary, the composite sum-and-normalize binary resolve, helper dispatches inside overlapping statistics
// queries, the single-list refusal (D3D-252) and, with a pool, its replay (D3D-399). Cases 15 and 16 check queries
// that hold mesh draws, split between the full and the mesh-compatible pool, alone and overlapping (D3D-291). No GPU or
// game data.
#include "../src/vulkan_replay.h"
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace leonida;

struct Trace {
    bool cond_active = false;
    unsigned cond_begins = 0;
    unsigned cond_ends = 0;
    unsigned fills = 0;
    unsigned fills_while_cond = 0;
    unsigned copies = 0;
    unsigned copies_while_cond = 0;
    unsigned binary_resolves = 0;
    unsigned binary_while_cond = 0;
    std::vector<std::string> events;
};

Trace* g_trace = nullptr;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

VKAPI_ATTR void VKAPI_CALL mock_begin_cond(VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {
    g_trace->cond_active = true;
    ++g_trace->cond_begins;
    g_trace->events.push_back("begin_cond");
}

VKAPI_ATTR void VKAPI_CALL mock_end_cond(VkCommandBuffer) {
    g_trace->cond_active = false;
    ++g_trace->cond_ends;
    g_trace->events.push_back("end_cond");
}

VKAPI_ATTR void VKAPI_CALL mock_fill_buffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, std::uint32_t) {
    ++g_trace->fills;
    if (g_trace->cond_active) ++g_trace->fills_while_cond;
    g_trace->events.push_back("fill");
}

VKAPI_ATTR void VKAPI_CALL mock_copy_query(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer,
                                           VkDeviceSize, VkDeviceSize, VkQueryResultFlags) {
    ++g_trace->copies;
    if (g_trace->cond_active) ++g_trace->copies_while_cond;
    g_trace->events.push_back("copy_query");
}

std::string query_event(const char* name, VkQueryPool pool, std::uint32_t index) {
    return std::string(name) + " " + std::to_string(reinterpret_cast<std::uintptr_t>(pool)) + ":" +
           std::to_string(index);
}

VKAPI_ATTR void VKAPI_CALL mock_reset_query(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL mock_begin_query(VkCommandBuffer, VkQueryPool pool, std::uint32_t index,
                                            VkQueryControlFlags) {
    g_trace->events.push_back(query_event("begin_query", pool, index));
}
VKAPI_ATTR void VKAPI_CALL mock_end_query(VkCommandBuffer, VkQueryPool pool, std::uint32_t index) {
    g_trace->events.push_back(query_event("end_query", pool, index));
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_pool(VkDevice, VkQueryPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL mock_write_timestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, std::uint32_t) {}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    ResourceBinding dst{};
    dst.buffer = handle<VkBuffer>(1);
    dst.buffer_offset = 24;
    dst.size = 128;

    ResourceBinding pred{};
    pred.buffer = handle<VkBuffer>(2);
    pred.size = 16;

    QueryPool heap{handle<VkQueryPool>(3), d3d12_query_heap::occlusion, 4, false, {false, true, false, true}, {}};
    QueryPool statistics_heap{handle<VkQueryPool>(7), d3d12_query_heap::pipeline_statistics, 2, false,
                              {false, false}, {}};

    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        if (id == 1) return &dst;
        if (id == 2) return &pred;
        return nullptr;
    };

    ReplayFunctions vk{};
    vk.begin_conditional_rendering = mock_begin_cond;
    vk.end_conditional_rendering = mock_end_cond;
    vk.prepare_predicate = [](VkBuffer source, VkDeviceSize offset, PredicateCapture& out) {
        out = {source, offset, handle<VkBuffer>(4), 0, handle<VkDescriptorSet>(5)};
        return true;
    };
    vk.capture_predicate = [](VkCommandBuffer, const PredicateCapture&) { g_trace->events.push_back("capture"); };
    vk.reset_query_pool = mock_reset_query;
    vk.begin_query = mock_begin_query;
    vk.end_query = mock_end_query;
    vk.write_timestamp = mock_write_timestamp;
    vk.copy_query_pool_results = mock_copy_query;
    vk.fill_buffer = mock_fill_buffer;
    vk.query_pool = [&](QueryHeapId id) -> QueryPool* {
        return id == 1 ? &heap : id == 6 ? &statistics_heap : nullptr;
    };
    vk.sum_statistics = [](VkCommandBuffer, VkQueryPool pool, VkQueryPool, const std::uint8_t*, std::uint32_t first,
                           std::uint32_t count, VkBuffer, VkDeviceSize) {
        g_trace->events.push_back(query_event("sum_statistics", pool, first) + "+" + std::to_string(count));
    };
    vk.resolve_binary_occlusion = [](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer,
                                     VkDeviceSize) {
        ++g_trace->binary_resolves;
        if (g_trace->cond_active) ++g_trace->binary_while_cond;
        g_trace->events.push_back("binary");
    };
    vk.sum_occlusion = [](VkCommandBuffer, VkQueryPool pool, std::uint32_t first, std::uint32_t count, VkBuffer,
                          VkDeviceSize) {
        g_trace->events.push_back(query_event("sum_occlusion", pool, first) + "+" + std::to_string(count));
    };
    vk.sum_binary_occlusion = [](VkCommandBuffer, VkQueryPool pool, std::uint32_t first, std::uint32_t count,
                                 VkBuffer, VkDeviceSize) {
        g_trace->events.push_back(query_event("sum_binary_occlusion", pool, first) + "+" + std::to_string(count));
    };

    // 1. Binary occlusion resolve with leading and trailing un-ended queries under a predicate:
    //    conditional rendering must end before the leading fill_buffer and resume after the trailing fill_buffer.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.set_predication({2, 0, d3d12_predication::equal_zero}), "record binary predication");
        expect(list.resolve_queries({1, d3d12_query::binary_occlusion, 0, 3, 1, 16}), "record binary resolve");
        expect(list.close(), "close binary list");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success, "replay binary resolve");
        expect(trace.fills == 2 && trace.fills_while_cond == 0,
               "binary resolve: leading and trailing fill_buffer run outside conditional rendering");
        expect(trace.binary_resolves == 1 && trace.binary_while_cond == 0,
               "binary resolve: normalization dispatch runs outside conditional rendering");
        expect(trace.cond_begins == 2 && trace.cond_ends == 2 && !trace.cond_active,
               "binary resolve: conditional rendering suspended once and restored");
        const std::vector<std::string> want{"capture", "begin_cond", "end_cond", "fill",
                                            "binary",  "fill",       "begin_cond", "end_cond"};
        expect(trace.events == want, "binary resolve: exact suspend/resume event order");
    }

    // 2. Ordinary occlusion resolve with a leading un-ended query under a predicate (no helper dispatch):
    //    fill_buffer and copy_query_pool_results must also run outside conditional rendering.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.set_predication({2, 0, d3d12_predication::equal_zero}), "record ordinary predication");
        expect(list.resolve_queries({1, d3d12_query::occlusion, 0, 2, 1, 0}), "record ordinary resolve");
        expect(list.close(), "close ordinary list");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success, "replay ordinary resolve");
        expect(trace.fills == 1 && trace.fills_while_cond == 0,
               "ordinary resolve: fill_buffer runs outside conditional rendering");
        expect(trace.copies == 1 && trace.copies_while_cond == 0,
               "ordinary resolve: copy_query_pool_results runs outside conditional rendering");
        expect(trace.cond_begins == 2 && trace.cond_ends == 2 && !trace.cond_active,
               "ordinary resolve: conditional rendering suspended once and restored");
    }

    // 3. Zero-count resolve under a predicate does not toggle conditional rendering unnecessarily.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.set_predication({2, 0, d3d12_predication::equal_zero}), "record empty resolve predication");
        expect(list.resolve_queries({1, d3d12_query::occlusion, 0, 0, 1, 0}), "record empty resolve");
        expect(list.close(), "close empty resolve list");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success, "replay empty resolve");
        expect(trace.fills == 0 && trace.copies == 0 && trace.cond_begins == 1 && trace.cond_ends == 1,
               "empty resolve: no redundant conditional rendering toggle");
    }

    // 4. Unpredicated resolve never touches conditional rendering.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.resolve_queries({1, d3d12_query::binary_occlusion, 0, 3, 1, 16}), "record unpredicated resolve");
        expect(list.close(), "close unpredicated list");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success, "replay unpredicated resolve");
        expect(trace.fills == 2 && trace.binary_resolves == 1 && trace.cond_begins == 0 && trace.cond_ends == 0,
               "unpredicated resolve: conditional rendering untouched");
    }

    // 5. D3D-162 without a statistics segment pool (the single-list replay()): a binary resolve that dispatches
    //    inside a pipeline-statistics query is still refused, while one that only writes zeros is replayed.
    {
        Trace trace;
        g_trace = &trace;
        CommandList dispatches, zeros;
        for (auto* list : {&dispatches, &zeros}) {
            const std::uint32_t index = list == &dispatches ? 1 : 0;  // query 1 was ended, query 0 was not
            expect(list->begin_query({6, d3d12_query::pipeline_statistics, 0}), "record statistics begin");
            expect(list->resolve_queries({1, d3d12_query::binary_occlusion, index, 1, 1, 0}), "record inner resolve");
            expect(list->end_query({6, d3d12_query::pipeline_statistics, 0}), "record statistics end");
            expect(list->close(), "close statistics list");
        }
        expect(replay(dispatches, VK_NULL_HANDLE, resolve, vk) == ReplayResult::unsupported,
               "no segment pool: a binary dispatch inside pipeline statistics is refused");
        expect(replay(zeros, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success,
               "no segment pool: a zero-fill binary resolve inside pipeline statistics is replayed");
        expect(trace.binary_resolves == 0 && trace.fills == 1, "no segment pool: only the zero fill was recorded");
    }

    // 6. D3D-162 with a statistics segment pool: the statistics query is split around the resolve inside it, so
    //    the normalisation dispatch runs between two segments, and resolving the query sums the segments.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "record split begin");
        expect(list.resolve_queries({1, d3d12_query::binary_occlusion, 1, 1, 1, 0}), "record split inner resolve");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "record split end");
        expect(list.resolve_queries({6, d3d12_query::pipeline_statistics, 0, 1, 1, 0}), "record split resolve");
        expect(list.close(), "close split list");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare split list");
        expect(prepared.statistics_segment_count == 2, "split list: two statistics segments");
        auto segments = std::make_shared<OcclusionSegmentPool>();
        segments->pool = handle<VkQueryPool>(9);
        segments->destroy = mock_destroy_pool;
        QueryReplay queries;
        expect(queries.plan(prepared, {}, 0, true, segments, 4) == ReplayResult::success, "plan split list");
        expect(queries.needs_binary && queries.needs_statistics_sum, "split list: binary and statistics helpers");
        emit(prepared, VK_NULL_HANDLE, vk);
        queries.commit();
        const std::vector<std::string> want{"begin_query 9:4", "end_query 9:4",  "binary",
                                            "begin_query 9:5", "end_query 9:5", "sum_statistics 9:4+2"};
        expect(trace.events == want, "split list: segments end before and begin after the helper dispatch");
        expect(statistics_heap.occlusion.count(0) && statistics_heap.occlusion[0].count == 2,
               "split list: the heap keeps the query's segments for later resolves");
    }

    // 7. All non-query helpers and a resolve share the segment numbering. Exercise both footprint
    // directions/formats and multiple conversion dispatches under an active predicate.
    for (bool rgb96 : {false, true}) for (bool upload : {false, true}) {
        Trace trace;
        g_trace = &trace;
        auto helpers = vk;
        unsigned allocations = 0;
        helpers.prepare_predicate = [&](VkBuffer source, VkDeviceSize offset, PredicateCapture& out) {
            ++allocations;
            out = {source, offset, handle<VkBuffer>(4), 0, handle<VkDescriptorSet>(5)};
            return true;
        };
        helpers.prepare_predicated_copy = [&](VkBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, VkDeviceSize,
                                               std::vector<PredicatedCopyDispatch>& jobs) {
            ++allocations;
            jobs.push_back({});
            return true;
        };
        helpers.emit_predicated_copy = [](VkCommandBuffer, const std::vector<PredicatedCopyDispatch>&) {
            expect(g_trace->cond_active, "copy helper remains predicated");
            g_trace->events.push_back("predicated_copy");
        };
        helpers.depth24_prepare = [&](Depth24Job& job, std::vector<Depth24Job>& parts) {
            ++allocations;
            job.scratch = handle<VkBuffer>(10);
            parts.assign(2, job);
            return true;
        };
        helpers.depth24_run = [](VkCommandBuffer, const Depth24Job&) {
            expect(!g_trace->cond_active, "footprint helper is unconditional");
            g_trace->events.push_back("footprint");
        };
        helpers.pipeline_barrier = +[](VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
                                       VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                       const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {};
        helpers.copy_buffer = +[](VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {};
        helpers.copy_image = +[](VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                                 std::uint32_t, const VkImageCopy*) {};
        helpers.copy_buffer_to_image = +[](VkCommandBuffer, VkBuffer, VkImage, VkImageLayout,
                                           std::uint32_t, const VkBufferImageCopy*) {
            g_trace->events.push_back("upload");
        };
        helpers.copy_image_to_buffer = +[](VkCommandBuffer, VkImage, VkImageLayout, VkBuffer,
                                           std::uint32_t, const VkBufferImageCopy*) {
            g_trace->events.push_back("readback");
        };
        ResourceBinding buffer = dst;
        buffer.buffer_offset = 0;
        buffer.size = 512;
        ResourceBinding image{};
        image.image = handle<VkImage>(11);
        image.format = rgb96 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_D32_SFLOAT_S8_UINT;
        image.aspects = rgb96 ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        image.width = image.height = image.depth = image.mip_levels = image.array_layers = image.samples = 1;
        image.rgb96 = rgb96;
        image.depth24 = image.depth24_as_float = !rgb96;
        const auto resources = [&](ResourceId id) -> const ResourceBinding* {
            return id == 1 ? &buffer : id == 2 ? &pred : id == 3 ? &image : nullptr;
        };
        const CopyLocation texture{3, false, 0, 0, 0, 0, 0, 0};
        const CopyLocation footprint{1, true, 0, 0, 1, 1, 1, 256};
        CommandList list;
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "mixed begin");
        expect(list.set_predication({2, 0, d3d12_predication::equal_zero}), "mixed capture");
        expect(list.resolve_queries({1, d3d12_query::binary_occlusion, 1, 1, 1, 0}), "mixed resolve");
        expect(list.copy_buffer({1, 0, 2, 0, 8}), "mixed predicated copy");
        expect(list.copy_texture({upload ? texture : footprint, 0, 0, 0,
                                  upload ? footprint : texture, false, {}}), "mixed footprint copy");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "mixed end");
        expect(list.resolve_queries({6, d3d12_query::pipeline_statistics, 0, 1, 1, 128}), "mixed sum");
        expect(list.close(), "close mixed list");
        PreparedList prepared;
        expect(prepare(list, resources, helpers, prepared, true) == ReplayResult::success, "prepare mixed helpers");
        expect(prepared.statistics_segment_count == 5 && allocations == 3, "five shared segments");
        QueryReplay no_pool;
        expect(no_pool.plan(prepared, {}, 0, true) == ReplayResult::unsupported,
               "plan refuses helper markers without statistics pool");
        expect(trace.events.empty(), "no-pool plan emits nothing");
        auto pool = std::make_shared<OcclusionSegmentPool>();
        pool->pool = handle<VkQueryPool>(9);
        pool->destroy = mock_destroy_pool;
        QueryReplay queries;
        expect(queries.plan(prepared, {}, 0, true, pool, 4) == ReplayResult::success, "plan mixed helpers");
        expect(queries.needs_binary && queries.needs_statistics_sum, "mixed helpers require statistics sum");
        emit(prepared, VK_NULL_HANDLE, helpers);
        std::vector<std::string> want{
            "begin_query 9:4", "end_query 9:4", "capture", "begin_query 9:5", "begin_cond",
            "end_query 9:5", "end_cond", "binary", "begin_cond", "begin_query 9:6",
            "end_query 9:6", "predicated_copy", "begin_query 9:7"};
        if (!upload) want.push_back("readback");
        const std::vector<std::string> conversion{
            "end_query 9:7", "end_cond", "footprint", "begin_cond", "end_cond", "footprint", "begin_cond",
            "begin_query 9:8"};
        want.insert(want.end(), conversion.begin(), conversion.end());
        if (upload) want.push_back("upload");
        const std::vector<std::string> tail{
            "end_query 9:8", "end_cond", "sum_statistics 9:4+5", "begin_cond", "end_cond"};
        want.insert(want.end(), tail.begin(), tail.end());
        expect(trace.events == want, "mixed helpers: exact segment and conditional scope order");

        // 8. Fallback checks each helper separately, before its scratch allocation and before emission.
        for (unsigned kind = 0; kind != 3; ++kind) {
            CommandList fallback;
            if (kind == 1) expect(fallback.set_predication({2, 0, 0}), "fallback pre-query predicate");
            expect(fallback.begin_query({6, d3d12_query::pipeline_statistics, 0}), "fallback begin");
            if (kind == 0) expect(fallback.set_predication({2, 0, 0}), "fallback capture");
            if (kind == 1) expect(fallback.copy_buffer({1, 0, 2, 0, 8}), "fallback copy");
            if (kind == 2) expect(fallback.copy_texture({upload ? texture : footprint, 0, 0, 0,
                                                       upload ? footprint : texture, false, {}}), "fallback footprint");
            expect(fallback.end_query({6, d3d12_query::pipeline_statistics, 0}) && fallback.close(), "fallback end");
            allocations = 0;
            trace.events.clear();
            expect(replay(fallback, VK_NULL_HANDLE, resources, helpers) == ReplayResult::unsupported,
                   "single-list fallback refuses helper");
            expect(trace.events.empty() && allocations == (kind == 1 ? 1u : 0u),
                   "fallback refuses before helper scratch and emits nothing");
            expect(prepare(fallback, resources, helpers, prepared, true) == ReplayResult::success,
                   "prepare fallback for explicit pool check");
            QueryReplay missing;
            expect(missing.plan(prepared) == ReplayResult::unsupported, "each helper needs a statistics pool");
        }
    }

    // 9. D3D-252, overlapping pipeline-statistics queries: the segments follow every Begin and End, so each logical
    //    query keeps a contiguous range — the intersection interval is shared — and each resolve sums its range.
    //    A resolve may only name an ended query, so the second resolve follows the second End.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "record overlap begin 0");
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 1}), "record overlap begin 1");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "record overlap end 0");
        expect(list.resolve_queries({6, d3d12_query::pipeline_statistics, 0, 1, 1, 0}), "record overlap resolve 0");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 1}), "record overlap end 1");
        expect(list.resolve_queries({6, d3d12_query::pipeline_statistics, 1, 1, 1, 0}), "record overlap resolve 1");
        expect(list.close(), "close overlap statistics list");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare overlap statistics list");
        expect(prepared.statistics_segment_count == 4, "overlap statistics: four physical segments");
        auto segments = std::make_shared<OcclusionSegmentPool>();
        segments->pool = handle<VkQueryPool>(9);
        segments->destroy = mock_destroy_pool;
        QueryReplay queries;
        expect(queries.plan(prepared, {}, 0, true, segments, 4) == ReplayResult::success,
               "plan overlap statistics list");
        expect(queries.needs_statistics_sum, "overlap statistics: the sum helper is needed");
        emit(prepared, VK_NULL_HANDLE, vk);
        queries.commit();
        const std::vector<std::string> want{"begin_query 9:4",      "end_query 9:4",       "begin_query 9:5",
                                            "end_query 9:5",        "begin_query 9:6",     "end_query 9:6",
                                            "sum_statistics 9:4+2", "begin_query 9:7",     "end_query 9:7",
                                            "sum_statistics 9:5+3"};
        expect(trace.events == want, "overlap statistics: one open segment per event, resolves sum the ranges");
        expect(statistics_heap.occlusion.count(0) && statistics_heap.occlusion[0].count == 2 &&
                   statistics_heap.occlusion.count(1) && statistics_heap.occlusion[1].count == 3,
               "overlap statistics: both heaps keep their segment ranges");
    }

    // 10. D3D-252, a binary occlusion query open inside an ordinary one: both are virtualized into the same physical
    //    pool. The ordinary resolve sums three segments; the binary resolve normalizes its one segment to 0/1.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.begin_query({1, d3d12_query::occlusion, 0}), "record kind overlap begin ordinary");
        expect(list.begin_query({1, d3d12_query::binary_occlusion, 1}), "record kind overlap begin binary");
        expect(list.end_query({1, d3d12_query::binary_occlusion, 1}), "record kind overlap end binary");
        expect(list.end_query({1, d3d12_query::occlusion, 0}), "record kind overlap end ordinary");
        expect(list.resolve_queries({1, d3d12_query::occlusion, 0, 1, 1, 0}), "record kind overlap resolve ordinary");
        expect(list.resolve_queries({1, d3d12_query::binary_occlusion, 1, 1, 1, 16}), "record kind overlap resolve binary");
        expect(list.close(), "close kind overlap list");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare kind overlap list");
        expect(prepared.segment_count == 3, "kind overlap: three physical segments");
        auto segments = std::make_shared<OcclusionSegmentPool>();
        segments->pool = handle<VkQueryPool>(9);
        segments->destroy = mock_destroy_pool;
        QueryReplay queries;
        expect(queries.plan(prepared, segments, 0, true, {}, 0) == ReplayResult::success,
               "plan kind overlap list");
        expect(queries.needs_sum && queries.needs_binary && !queries.needs_sum_binary,
               "kind overlap: sum for the ordinary resolve, normalize for the single-segment binary");
        emit(prepared, VK_NULL_HANDLE, vk);
        queries.commit();
        const std::vector<std::string> want{"begin_query 9:0",         "end_query 9:0",       "begin_query 9:1",
                                            "end_query 9:1",           "begin_query 9:2",     "end_query 9:2",
                                            "sum_occlusion 9:0+3",     "binary"};
        expect(trace.events == want, "kind overlap: shared physical segments, sum and normalize resolves");
        expect(heap.occlusion.count(0) && heap.occlusion[0].count == 3 && heap.occlusion.count(1) &&
                   heap.occlusion[1].count == 1,
               "kind overlap: both kinds keep their segment ranges in the heap");
    }

    // 11. D3D-252, a binary query overlapping the ordinary one with two segments: the resolve sums the segments and
    //    then normalizes the total (the composite helper), not one 0/1 per segment.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.begin_query({1, d3d12_query::occlusion, 0}), "record composite begin ordinary");
        expect(list.begin_query({1, d3d12_query::binary_occlusion, 1}), "record composite begin binary");
        expect(list.end_query({1, d3d12_query::occlusion, 0}), "record composite end ordinary");
        expect(list.end_query({1, d3d12_query::binary_occlusion, 1}), "record composite end binary");
        expect(list.resolve_queries({1, d3d12_query::binary_occlusion, 1, 1, 1, 16}), "record composite resolve binary");
        expect(list.close(), "close composite list");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare composite list");
        expect(prepared.segment_count == 3, "composite: three physical segments");
        auto segments = std::make_shared<OcclusionSegmentPool>();
        segments->pool = handle<VkQueryPool>(9);
        segments->destroy = mock_destroy_pool;
        QueryReplay queries;
        expect(queries.plan(prepared, segments, 0, true, {}, 0) == ReplayResult::success, "plan composite list");
        expect(queries.needs_sum_binary, "composite: the sum-and-normalize helper is needed");
        emit(prepared, VK_NULL_HANDLE, vk);
        queries.commit();
        const std::vector<std::string> want{"begin_query 9:0",           "end_query 9:0",       "begin_query 9:1",
                                            "end_query 9:1",             "begin_query 9:2",     "end_query 9:2",
                                            "sum_binary_occlusion 9:1+2"};
        expect(trace.events == want, "composite: the binary resolve sums its segments and normalizes once");
    }

    // 12. D3D-252, helper dispatches inside overlapping statistics queries: the helpers share the segment numbering
    //     here too — the shared physical segment closes before the dispatches and a new one opens after them, so a
    //     footprint conversion stays out of every query still running (D3D-162).
    {
        Trace trace;
        g_trace = &trace;
        auto helpers = vk;
        helpers.depth24_prepare = [&](Depth24Job& job, std::vector<Depth24Job>& parts) {
            job.scratch = handle<VkBuffer>(10);
            parts.assign(2, job);
            return true;
        };
        helpers.depth24_run = [](VkCommandBuffer, const Depth24Job&) {
            expect(!g_trace->cond_active, "footprint helper is unconditional");
            g_trace->events.push_back("footprint");
        };
        helpers.pipeline_barrier = +[](VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
                                       VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                       const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {};
        helpers.copy_buffer = +[](VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {};
        helpers.copy_buffer_to_image = +[](VkCommandBuffer, VkBuffer, VkImage, VkImageLayout,
                                           std::uint32_t, const VkBufferImageCopy*) {};
        helpers.copy_image = +[](VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                                 std::uint32_t, const VkImageCopy*) {};
        helpers.copy_image_to_buffer = +[](VkCommandBuffer, VkImage, VkImageLayout, VkBuffer,
                                           std::uint32_t, const VkBufferImageCopy*) {
            g_trace->events.push_back("readback");
        };
        ResourceBinding buffer = dst;
        buffer.buffer_offset = 0;
        buffer.size = 512;
        ResourceBinding image{};
        image.image = handle<VkImage>(11);
        image.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
        image.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        image.width = image.height = image.depth = image.mip_levels = image.array_layers = image.samples = 1;
        image.rgb96 = false;
        image.depth24 = image.depth24_as_float = true;
        const auto resources = [&](ResourceId id) -> const ResourceBinding* {
            return id == 1 ? &buffer : id == 3 ? &image : nullptr;
        };
        const CopyLocation texture{3, false, 0, 0, 0, 0, 0, 0};
        const CopyLocation footprint{1, true, 0, 0, 1, 1, 1, 256};
        CommandList list;
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "overlap helper begin 0");
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 1}), "overlap helper begin 1");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "overlap helper end 0");
        expect(list.copy_texture({footprint, 0, 0, 0, texture, false, {}}), "overlap helper footprint copy");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 1}), "overlap helper end 1");
        expect(list.resolve_queries({6, d3d12_query::pipeline_statistics, 0, 1, 1, 0}), "overlap helper resolve 0");
        expect(list.resolve_queries({6, d3d12_query::pipeline_statistics, 1, 1, 1, 0}), "overlap helper resolve 1");
        expect(list.close(), "close overlap helper list");
        PreparedList prepared;
        expect(prepare(list, resources, helpers, prepared, true) == ReplayResult::success,
               "prepare overlap helper list");
        expect(prepared.statistics_segment_count == 4, "the helper adds its two segments to the overlap");
        auto segments = std::make_shared<OcclusionSegmentPool>();
        segments->pool = handle<VkQueryPool>(9);
        segments->destroy = mock_destroy_pool;
        QueryReplay queries;
        expect(queries.plan(prepared, {}, 0, true, segments, 4) == ReplayResult::success,
               "plan overlap helper list");
        expect(queries.needs_statistics_sum, "overlap helpers need the statistics sum");
        emit(prepared, VK_NULL_HANDLE, helpers);
        queries.commit();
        const std::vector<std::string> want{
            "begin_query 9:4", "end_query 9:4", "begin_query 9:5", "end_query 9:5", "begin_query 9:6",
            "readback",        "end_query 9:6", "footprint",       "footprint",     "begin_query 9:7",
            "end_query 9:7",   "sum_statistics 9:4+2", "sum_statistics 9:5+3"};
        expect(trace.events == want, "overlap helpers: the dispatches run outside every open segment");
        expect(statistics_heap.occlusion.count(0) && statistics_heap.occlusion[0].count == 2 &&
                   statistics_heap.occlusion.count(1) && statistics_heap.occlusion[1].count == 3,
               "overlap helpers: the queries keep their segment ranges");
    }

    // 13. D3D-252, the single-list replay() helper: it has no segment pool, and running overlapping statistics
    //     queries whole would keep two Vulkan queries of one type active, so the list is refused outright.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "record helper begin 0");
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 1}), "record helper begin 1");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "record helper end 0");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 1}), "record helper end 1");
        expect(list.close(), "close helper list");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::unsupported,
               "no segment pool: overlapping statistics queries are refused instead of run whole");
        expect(trace.events.empty(), "no segment pool: nothing was emitted for the refused list");
    }

    // 14. D3D-399, the single-list replay() with a segment pool: the list case 13 refuses replays as on the batched
    //     path, and so does case 10's occlusion pair. The pools are created to the planned size and handed to `keep`.
    {
        Trace trace;
        g_trace = &trace;
        std::vector<std::pair<VkQueryType, std::uint32_t>> created;
        ReplayFunctions pooled = vk;
        pooled.create_segment_pool = [&](VkQueryType type, std::uint32_t count) {
            created.emplace_back(type, count);
            auto pool = std::make_shared<OcclusionSegmentPool>();
            pool->pool = handle<VkQueryPool>(type == VK_QUERY_TYPE_OCCLUSION ? 8 : 9);
            pool->destroy = mock_destroy_pool;
            return std::shared_ptr<const OcclusionSegmentPool>(pool);
        };
        CommandList statistics_list;  // case 13's list
        expect(statistics_list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "record pooled begin 0");
        expect(statistics_list.begin_query({6, d3d12_query::pipeline_statistics, 1}), "record pooled begin 1");
        expect(statistics_list.end_query({6, d3d12_query::pipeline_statistics, 0}), "record pooled end 0");
        expect(statistics_list.end_query({6, d3d12_query::pipeline_statistics, 1}), "record pooled end 1");
        expect(statistics_list.close(), "close pooled statistics list");
        CommandList occlusion_list;  // case 10's list
        expect(occlusion_list.begin_query({1, d3d12_query::occlusion, 0}), "record pooled occlusion begin");
        expect(occlusion_list.begin_query({1, d3d12_query::binary_occlusion, 1}), "record pooled binary begin");
        expect(occlusion_list.end_query({1, d3d12_query::binary_occlusion, 1}), "record pooled binary end");
        expect(occlusion_list.end_query({1, d3d12_query::occlusion, 0}), "record pooled occlusion end");
        expect(occlusion_list.resolve_queries({1, d3d12_query::occlusion, 0, 1, 1, 0}), "record pooled resolve");
        expect(occlusion_list.close(), "close pooled occlusion list");
        expect(replay(statistics_list, VK_NULL_HANDLE, resolve, pooled) == ReplayResult::unsupported &&
                   replay(occlusion_list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::unsupported &&
                   created.empty() && trace.events.empty(),
               "pooled: without `keep` or a pool both lists are still refused and no pool is made");
        std::vector<std::shared_ptr<const OcclusionSegmentPool>> keep;
        const auto r0 = replay(statistics_list, VK_NULL_HANDLE, resolve, pooled, nullptr, &keep);
        const auto r1 = replay(occlusion_list, VK_NULL_HANDLE, resolve, pooled, nullptr, &keep);
        expect(r0 == ReplayResult::success && r1 == ReplayResult::success,
               "pooled: overlapping statistics and occlusion queries replay on the single-list path");
        const std::vector<std::pair<VkQueryType, std::uint32_t>> sizes{{VK_QUERY_TYPE_PIPELINE_STATISTICS, 3},
                                                                        {VK_QUERY_TYPE_OCCLUSION, 3}};
        expect(created == sizes, "pooled: one pool of each type, sized to the planned segments");
        expect(keep.size() == 2, "pooled: both pools are handed to the caller");
        const std::vector<std::string> want{"begin_query 9:0", "end_query 9:0",  "begin_query 9:1",
                                            "end_query 9:1",   "begin_query 9:2", "end_query 9:2",
                                            "begin_query 8:0", "end_query 8:0",  "begin_query 8:1",
                                            "end_query 8:1",   "begin_query 8:2", "end_query 8:2",
                                            "sum_occlusion 8:0+3"};
        expect(trace.events == want, "pooled: the same segments and sums as the batched path");
    }

    // 15. D3D-291: a pipeline-statistics query around vertex, mesh, mesh and vertex draws is split where the kind
    //    changes, the mesh segment is marked for the mesh pool, and its resolve sums the three segments. A query that
    //    holds only a mesh draw is one mesh segment, still summed so its counters land in their D3D12 fields; one
    //    with only vertex draws is not split.
    {
        Trace trace;
        g_trace = &trace;
        const auto draw = [](bool mesh) {
            Draw d{};
            d.pipeline = 1;
            d.count = d.instances = d.first = 1;
            d.mesh = mesh;
            return d;
        };
        CommandList list;
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "record mixed begin");     // 0
        expect(list.draw(draw(false)), "record vertex draw");                                           // 1
        expect(list.draw(draw(true)), "record mesh draw");                                              // 2
        expect(list.draw(draw(true)), "record second mesh draw");                                       // 3
        expect(list.draw(draw(false)), "record second vertex draw");                                    // 4
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "record mixed end");          // 5
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 1}), "record mesh-only begin");  // 6
        expect(list.draw(draw(true)), "record lone mesh draw");                                         // 7
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 1}), "record mesh-only end");      // 8
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "record vertex-only begin"); // 9
        expect(list.draw(draw(false)), "record lone vertex draw");                                      // 10
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "record vertex-only end");    // 11
        expect(list.resolve_queries({6, d3d12_query::pipeline_statistics, 1, 1, 1, 0}), "record mesh resolve");
        expect(list.close(), "close mixed list");
        expect(list.commands().size() == 13, "mixed list: thirteen commands");

        // The draws need a pipeline and attachments to prepare, so the query plan is built by hand here; the GPU
        // test d3d12_mesh_test replays the same shape.
        PreparedList prepared;
        prepared.list = &list;
        prepared.commands.assign(list.commands().size(), {});
        for (std::size_t i : {0, 5, 6, 8, 9, 11, 12}) {
            const bool begin = std::holds_alternative<BeginQuery>(list.commands()[i]);
            prepared.queries.push_back({i, &statistics_heap, begin});  // as prepare() records statistics_active
        }
        plan_statistics_segments(prepared);
        expect(prepared.statistics_segment_count == 4, "mixed list: three segments and one for the mesh-only query");
        expect(prepared.statistics_mesh == std::vector<std::uint8_t>({0, 1, 0, 1}),
               "mixed list: vertex, mesh, vertex, then the mesh-only query's mesh segment");
        const auto& c = prepared.commands;
        constexpr auto none = UINT32_MAX;
        expect(c[1].statistics_begin == none && c[3].statistics_begin == none && c[7].statistics_begin == none &&
                   c[10].statistics_begin == none,
               "mixed list: no split before a draw of the segment's own kind");
        expect(c[2].statistics_end == 0 && c[2].statistics_begin == 1, "mixed list: the first mesh draw splits");
        expect(c[4].statistics_end == 1 && c[4].statistics_begin == 2, "mixed list: the vertex draw after it splits");
        const auto& q = prepared.queries;
        expect(q[0].statistics_begin == 0 && q[1].statistics_end == 2 && q[1].statistics_first == 0 &&
                   q[1].statistics_count == 3,
               "mixed list: the query ends in its third segment and sums all three");
        expect(q[2].statistics_begin == 3 && q[3].statistics_end == 3 && q[3].statistics_count == 1,
               "mixed list: the mesh-only query is one segment");
        expect(q[4].statistics_begin == detail::PreparedQuery::no_segment &&
                   q[5].statistics_end == detail::PreparedQuery::no_segment,
               "mixed list: the vertex-only query is not split");

        auto segments = std::make_shared<OcclusionSegmentPool>();
        segments->pool = handle<VkQueryPool>(9);
        segments->mesh_pool = handle<VkQueryPool>(10);
        segments->mesh = prepared.statistics_mesh;
        segments->destroy = mock_destroy_pool;
        QueryReplay queries;
        expect(queries.plan(prepared, {}, 0, true, segments, 0) == ReplayResult::success, "plan mixed list");
        expect(prepared.statistics_mesh_pool == handle<VkQueryPool>(10), "mixed list: the mesh pool is the list's");
        expect(q[6].resolves.size() == 1 && q[6].resolves[0].sum && q[6].resolves[0].first == 3 &&
                   q[6].resolves[0].count == 1 && q[6].resolves[0].mesh_pool == handle<VkQueryPool>(10) &&
                   q[6].resolves[0].mesh && q[6].resolves[0].mesh[3] == 1,
               "mixed list: the mesh-only query's resolve sums its one mesh segment");
        expect(queries.needs_statistics_sum, "mixed list: the statistics sum helper is needed");

        // Without the mesh-compatible pool, the query cannot run unsplit around a mesh draw.
        PreparedList again = prepared;
        again.statistics_pool = again.statistics_mesh_pool = VK_NULL_HANDLE;
        auto full_only = std::make_shared<OcclusionSegmentPool>();
        full_only->pool = handle<VkQueryPool>(9);
        full_only->destroy = mock_destroy_pool;
        QueryReplay refused;
        expect(refused.plan(again, {}, 0, true, full_only, 0) == ReplayResult::unsupported,
               "mixed list: refused without the mesh pool");
        QueryReplay unsplit;
        expect(unsplit.plan(again, {}, 0, true, {}, 0) == ReplayResult::unsupported,
               "mixed list: refused without a segment pool");
    }

    // 16. D3D-291 with overlapping statistics queries (D3D-252's planner): query 0 holds a vertex and a mesh draw,
    //     query 1 (begun inside it) a mesh draw, then a mesh and a vertex draw after query 0 ends. Every Begin and
    //     End opens a segment as before; the vertex draw after the mesh draws opens one more, so the four segments
    //     are vertex, mesh, mesh, vertex, query 0 sums segments 0-1 and query 1 segments 1-3.
    {
        const auto draw = [](bool mesh) {
            Draw d{};
            d.pipeline = 1;
            d.count = d.instances = d.first = 1;
            d.mesh = mesh;
            return d;
        };
        CommandList list;
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "record overlap-mesh begin 0");  // 0
        expect(list.draw(draw(false)), "record overlap-mesh vertex draw");                                   // 1
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 1}), "record overlap-mesh begin 1");  // 2
        expect(list.draw(draw(true)), "record overlap-mesh mesh draw");                                      // 3
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "record overlap-mesh end 0");      // 4
        expect(list.draw(draw(true)), "record overlap-mesh second mesh draw");                               // 5
        expect(list.draw(draw(false)), "record overlap-mesh second vertex draw");                            // 6
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 1}), "record overlap-mesh end 1");      // 7
        expect(list.close(), "close overlap-mesh list");
        PreparedList prepared;
        prepared.list = &list;
        prepared.commands.assign(list.commands().size(), {});
        for (std::size_t i : {0, 2, 4, 7}) {
            const bool begin = std::holds_alternative<BeginQuery>(list.commands()[i]);
            prepared.queries.push_back({i, &statistics_heap, begin});
        }
        expect(plan_overlapping_statistics_segments(prepared) == ReplayResult::success, "plan overlap-mesh list");
        expect(prepared.statistics_segment_count == 4, "overlap-mesh: four segments");
        expect(prepared.statistics_mesh == std::vector<std::uint8_t>({0, 1, 1, 0}),
               "overlap-mesh: vertex, mesh, mesh, vertex");
        const auto& c = prepared.commands;
        constexpr auto none = detail::PreparedQuery::no_segment;
        expect(c[1].statistics_begin == none && c[3].statistics_begin == none && c[5].statistics_begin == none,
               "overlap-mesh: a segment's first draw and a draw of its kind do not split");
        expect(c[6].statistics_end == 2 && c[6].statistics_begin == 3,
               "overlap-mesh: the vertex draw after the mesh draws splits");
        const auto& q = prepared.queries;
        // queries[] follows the commands: Begin 0, Begin 1, End 0, End 1.
        expect(q[2].statistics_first == 0 && q[2].statistics_count == 2, "overlap-mesh: query 0 sums segments 0-1");
        expect(q[3].statistics_first == 1 && q[3].statistics_count == 3, "overlap-mesh: query 1 sums segments 1-3");
    }

    if (failures) return 1;
    std::puts("replay_query_predication_test: query resolves suspend conditional rendering around fills, copies and dispatches");
    return 0;
}
