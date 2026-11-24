// Query heap boundaries, resolve helper chunking, arena offsets, query reuse across lists/submissions, and
// predication around copies. These edge cases are only reached by GPU tests today; this CPU mock lets CI
// without a GPU check replay planning, emitted commands, destination guards, and pool ownership on submit failure.
// Shader arithmetic and Vulkan entry points are mocked; no driver is loaded. No GPU or game data.
#include "../src/vulkan_backend.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace leonida;

// OcclusionResolve keeps these capacities private (runtime/src/vulkan_backend.h). Pin the expected boundaries
// here so a change to its chunking still has an independent oracle; derive public result sizes from the IR.
constexpr unsigned sum_capacity = 256;
constexpr unsigned binary_capacity = 1024;
constexpr unsigned statistics_capacity = 64;
constexpr unsigned binary_group_size = 64;
constexpr VkDeviceSize word_size = sizeof(std::uint64_t);
constexpr unsigned statistics_size = query_result_size(d3d12_query::pipeline_statistics);
constexpr unsigned statistics_counters = statistics_size / word_size;
constexpr VkDeviceSize scratch_size = std::max({sum_capacity + 1, binary_capacity,
    (statistics_capacity + 1) * statistics_counters}) * word_size;
constexpr unsigned char guard_byte = 0xa5;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

// Stop the current case on failure so a failed mock bounds check cannot turn into an out-of-bounds access.
void expect(bool ok, const char* what) {
    if (!ok) {
        throw std::runtime_error(what);
    }
}
struct Call {
    VkQueryPool pool;
    unsigned first;
    unsigned count;
    VkDeviceSize offset;
    VkDeviceSize stride;
};

struct MockReplay {
    static inline MockReplay* current;
    std::map<VkBuffer, std::vector<unsigned char>> buffers;
    std::map<VkQueryPool, std::vector<std::array<std::uint64_t, statistics_counters>>> values;
    std::vector<Call> copies;
    std::vector<std::pair<VkQueryPool, unsigned>> resets;
    std::set<VkQueryPool> statistics_pools;
    std::vector<unsigned> chunks;
    std::vector<unsigned> groups;
    std::vector<VkBufferCopy> transfers;
    std::vector<std::string> events;
    std::map<VkQueryPool, std::set<unsigned>> active;
    bool conditional = false;
    bool execute = true;
    unsigned captures = 0;
    unsigned pred_copies = 0;
    unsigned native_copies = 0;
    std::uint64_t snapshot = 0;
    std::uint64_t serial = 10;
    enum Mode {
        none,
        sum,
        binary,
        statistics
    };
    Mode mode = none;
    MockReplay() {
        current = this;
        // OcclusionResolve is exercised without initialize(); the null buffer stands in for its scratch storage.
        buffers[VK_NULL_HANDLE].resize(scratch_size);
    }

    auto& bytes(VkBuffer b, VkDeviceSize offset, VkDeviceSize size) {
        auto& v = buffers.at(b);
        expect(offset <= v.size() && size <= v.size() - offset, "mock buffer access out of bounds");
        return v;
    }

    std::uint64_t word(VkBuffer b, VkDeviceSize off) {
        std::uint64_t v;
        auto&a = bytes(b, off, word_size);
        std::memcpy(&v, a.data() + off, word_size);
        return v;
    }

    void word(VkBuffer b, VkDeviceSize off, std::uint64_t v) {
        auto&a = bytes(b, off, word_size);
        std::memcpy(a.data() + off, &v, word_size);
    }

    void query(VkQueryPool p, unsigned first, unsigned count, VkBuffer b, VkDeviceSize off, VkDeviceSize stride,
        VkQueryResultFlags flags) {
        constexpr auto result_flags = VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT;
        expect((flags & result_flags) == result_flags, "query result flags");
        copies.push_back({p, first, count, off, stride});
        auto& q = values.at(p);
        expect(first <= q.size() && count <= q.size() - first, "mock query access out of bounds");
        for (unsigned i = 0; i < count; ++i) {
            for (unsigned j = 0; j < (stride == statistics_size ? statistics_counters : 1u); ++j) {
                word(b, off + i * stride + word_size * j, q[first + i][j]);
            }
        }
        events.push_back("query_copy");
    }

    void fill(VkBuffer b, VkDeviceSize off, VkDeviceSize n, unsigned value) {
        auto& v = bytes(b, off, n);
        for (VkDeviceSize i = 0; i < n; i += sizeof(value)) {
            std::memcpy(v.data() + off + i, &value, sizeof(value));
        }
        events.push_back("fill");
    }

    void copy(VkBuffer a, VkBuffer b, const VkBufferCopy& c) {
        auto& av = bytes(a, c.srcOffset, c.size);
        auto& bv = bytes(b, c.dstOffset, c.size);
        std::memmove(bv.data() + c.dstOffset, av.data() + c.srcOffset, c.size);
        transfers.push_back(c);
    }

    ReplayFunctions replay_functions() {
        ReplayFunctions v{};
        v.reset_query_pool = +[](VkCommandBuffer, VkQueryPool p, unsigned first, unsigned count) {
            auto& c = *current;
            expect(first <= c.values.at(p).size() && count <= c.values.at(p).size() - first, "reset query bounds");
            for (unsigned i = 0; i < count; ++i) {
                expect(!c.active[p].count(first + i), "reset active query");
                c.values[p][first + i] = {};
                c.resets.emplace_back(p, first + i);
            }
        };
        v.begin_query = +[](VkCommandBuffer, VkQueryPool p, unsigned i, VkQueryControlFlags) {
            expect(current->active[p].empty(), "overlapping physical query");
            current->active[p].insert(i);
            current->events.push_back("begin_query");
        };
        v.end_query = +[](VkCommandBuffer, VkQueryPool p, unsigned i) {
            expect(current->active[p].erase(i) == 1, "end inactive query");
            for (unsigned j = 0; j < statistics_counters; ++j) {
                current->values.at(p).at(i)[j] = ++current->serial + j;
            }
            current->events.push_back("end_query");
        };
        v.write_timestamp = +[](VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool p, unsigned i) {
            current->values.at(p).at(i)[0] = ++current->serial;
        };
        v.copy_query_pool_results = +[](VkCommandBuffer, VkQueryPool p, unsigned first, unsigned count, VkBuffer b,
            VkDeviceSize off, VkDeviceSize stride, VkQueryResultFlags flags) {
            current->query(p, first, count, b, off, stride, flags);
        };
        v.fill_buffer = +[](VkCommandBuffer, VkBuffer b, VkDeviceSize off, VkDeviceSize n, unsigned value) {
            expect(!current->conditional, "resolve fill inside predicate");
            current->fill(b, off, n, value);
        };
        v.copy_buffer = +[](VkCommandBuffer, VkBuffer a, VkBuffer b, unsigned n, const VkBufferCopy* copies) {
            ++current->native_copies;
            for (unsigned i = 0; i < n; ++i) {
                current->copy(a, b, copies[i]);
            }
            current->events.push_back("native_copy");
        };
        // Capability entries are never called in buffer-only tests.
        v.copy_buffer_to_image = +[](VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, unsigned,
            const VkBufferImageCopy*) {
            expect(false, "unexpected image copy");
        };
        v.copy_image_to_buffer = +[](VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, unsigned,
            const VkBufferImageCopy*) {
            expect(false, "unexpected image copy");
        };
        v.copy_image = +[](VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, unsigned,
            const VkImageCopy*) {
            expect(false, "unexpected image copy");
        };
        v.begin_conditional_rendering = +[](VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT* info) {
            auto& c = *current;
            expect(!c.conditional, "nested predicate");
            c.conditional = true;
            c.execute = (c.snapshot != 0);
            if (info->flags & VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT) {
                c.execute = !c.execute;
            }
            c.events.push_back("begin_cond");
        };
        v.end_conditional_rendering = +[](VkCommandBuffer) {
            expect(current->conditional, "end missing predicate");
            current->conditional = false;
            current->events.push_back("end_cond");
        };
        v.prepare_predicate = [](VkBuffer b, VkDeviceSize off, PredicateCapture& out) {
            out = {b, off, handle<VkBuffer>(99), 0, handle<VkDescriptorSet>(99)};
            return true;
        };
        v.capture_predicate = [](VkCommandBuffer, const PredicateCapture& p) {
            expect(!current->conditional, "predicated capture");
            current->snapshot = current->word(p.source, p.source_offset);
            ++current->captures;
            current->events.push_back("capture");
        };
        auto helper = [](bool stats, bool binary, VkQueryPool p, unsigned first, unsigned count, VkBuffer dst,
            VkDeviceSize off) {
            auto& c = *current;
            expect(!c.conditional, "predicated resolve helper");
            c.events.push_back("helper");
            for (auto pool : c.statistics_pools) {
                expect(c.active[pool].empty(), "resolve helper pollutes active pipeline-statistics query");
            }
            if (binary) {
                for (unsigned i = 0; i < count; ++i) {
                    c.word(dst, off + word_size * i, c.values.at(p).at(first + i)[0] != 0);
                }
            } else {
                for (unsigned j = 0; j < (stats ? statistics_counters : 1u); ++j) {
                    std::uint64_t total = 0;
                    for (unsigned i = 0; i < count; ++i) {
                        total += c.values.at(p).at(first + i)[j];
                    }
                    c.word(dst, off + word_size * j, total);
                }
            }
        };
        v.sum_occlusion = [helper](VkCommandBuffer, VkQueryPool p, unsigned f, unsigned n, VkBuffer b,
            VkDeviceSize off) {
            helper(false, false, p, f, n, b, off);
        };
        v.sum_statistics = [helper](VkCommandBuffer, VkQueryPool p, VkQueryPool, const std::uint8_t*, unsigned f,
            unsigned n, VkBuffer b, VkDeviceSize off) {
            helper(true, false, p, f, n, b, off);
        };
        v.resolve_binary_occlusion = [helper](VkCommandBuffer, VkQueryPool p, unsigned f, unsigned n, VkBuffer b,
            VkDeviceSize off) {
            helper(false, true, p, f, n, b, off);
        };
        return v;
    }
};

inline std::shared_ptr<OcclusionSegmentPool> make_segment_pool(unsigned id) {
    auto p = std::make_shared<OcclusionSegmentPool>();
    p->pool = handle<VkQueryPool>(id);
    p->destroy = +[](VkDevice, VkQueryPool, const VkAllocationCallbacks*) {};
    return p;
}

void test_helper_chunks() {
    constexpr unsigned first_query = 7;
    constexpr VkDeviceSize arena_offset = 40;
    constexpr VkDeviceSize resolve_offset = 24;
    constexpr VkDeviceSize output_offset = arena_offset + resolve_offset;
    constexpr unsigned buffer_bytes = 10000;
    constexpr std::uint64_t reset_sum = 3;
    constexpr std::uint64_t reset_counter = 9;

    MockReplay c;
    VulkanDeviceFunctions vk{};
    vk.vkCmdFillBuffer = +[](VkCommandBuffer, VkBuffer b, VkDeviceSize off, VkDeviceSize n, unsigned value) {
        MockReplay::current->fill(b, off, n, value);
    };
    vk.vkCmdCopyQueryPoolResults = +[](VkCommandBuffer, VkQueryPool p, unsigned first, unsigned count, VkBuffer b,
        VkDeviceSize off, VkDeviceSize stride, VkQueryResultFlags flags) {
        MockReplay::current->query(p, first, count, b, off, stride, flags);
    };
    vk.vkCmdCopyBuffer = +[](VkCommandBuffer, VkBuffer a, VkBuffer b, unsigned n, const VkBufferCopy* copies) {
        for (unsigned i = 0; i < n; ++i) {
            MockReplay::current->copy(a, b, copies[i]);
        }
    };
    vk.vkCmdBindPipeline = +[](VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {};
    vk.vkCmdBindDescriptorSets = +[](VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, unsigned, unsigned,
        const VkDescriptorSet*, unsigned, const unsigned*) {};
    vk.vkCmdPushConstants = +[](VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, unsigned off, unsigned size,
        const void*p) {
        // The count, then statistics_sum.comp's mesh flag (D3D-291), which no segment here sets.
        expect(off == 0 && size == 2 * sizeof(unsigned), "chunk push constants");
        expect(static_cast<const unsigned *>(p)[1] == 0, "chunk push constants: no mesh segment");
        MockReplay::current->chunks.push_back(*static_cast<const unsigned *>(p));
    };
    vk.vkCmdDispatch = +[](VkCommandBuffer, unsigned x, unsigned y, unsigned z) {
        auto& c = *MockReplay::current;
        c.groups.push_back(x);
        expect(y == 1 && z == 1, "dispatch dimensions");
        auto n = c.chunks.back();
        if (c.mode == MockReplay::binary) {
            for (unsigned i = 0; i < n; ++i) {
                c.word(VK_NULL_HANDLE, word_size * i, c.word(VK_NULL_HANDLE, word_size * i) != 0);
            }
        } else {
            for (unsigned j = 0; j < (c.mode == MockReplay::statistics ? statistics_counters : 1u); ++j) {
                unsigned width = c.mode == MockReplay::statistics ? statistics_counters : 1;
                auto total = c.word(VK_NULL_HANDLE, word_size * j);
                for (unsigned i = 0; i < n; ++i) {
                    total += c.word(VK_NULL_HANDLE, word_size * (width * (i + 1) + j));
                }
                c.word(VK_NULL_HANDLE, word_size * j, total);
            }
        }
    };
    vk.vkCmdPipelineBarrier = +[](VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
        unsigned, const VkMemoryBarrier*, unsigned n, const VkBufferMemoryBarrier* b, unsigned,
        const VkImageMemoryBarrier*) {
        expect(n == 1 && b->offset == 0 && b->size == scratch_size, "scratch barrier covers the shared scratch buffer");
    };
    OcclusionResolve helper(vk, VK_NULL_HANDLE);
    auto qp = handle<VkQueryPool>(1);
    auto dst = handle<VkBuffer>(2);
    for (auto mode : {MockReplay::sum, MockReplay::binary, MockReplay::statistics}) {
        for (unsigned extra : {0u, 1u}) {
            c.mode = mode;
            const unsigned capacity = mode == MockReplay::sum ? sum_capacity
                                      : mode == MockReplay::binary ? binary_capacity
                                                                  : statistics_capacity;
            const unsigned n = capacity + extra;
            const unsigned first = first_query;
            const unsigned width = mode == MockReplay::statistics ? statistics_counters : 1;
            c.values[qp].resize(first + n);
            for (unsigned i = 0; i < first + n; ++i) {
                for (unsigned j = 0; j < statistics_counters; ++j) {
                    c.values[qp][i][j] = i % 3 == 0 ? 0 : (UINT64_C(1) << 32) + i * 17 + j;
                }
            }
            c.buffers[dst].assign(buffer_bytes, guard_byte);
            auto expected = c.buffers[dst];
            c.copies.clear();
            c.chunks.clear();
            c.groups.clear();
            c.transfers.clear();
            if (mode == MockReplay::binary) {
                helper.normalize({}, qp, first, n, dst, output_offset);
                for (unsigned i = 0; i < n; ++i) {
                    std::uint64_t v = c.values[qp][first + i][0] != 0;
                    std::memcpy(expected.data() + output_offset + word_size * i, &v, word_size);
                }
            } else {
                if (mode == MockReplay::statistics) {
                    helper.sum_statistics({}, qp, VK_NULL_HANDLE, nullptr, first, n, dst, output_offset);
                } else {
                    helper.sum({}, qp, first, n, dst, output_offset);
                }
                for (unsigned j = 0; j < width; ++j) {
                    std::uint64_t v = 0;
                    for (unsigned i = 0; i < n; ++i) {
                        v += c.values[qp][first + i][j];
                    }
                    std::memcpy(expected.data() + output_offset + word_size * j, &v, word_size);
                }
            }
            expect(c.buffers[dst] == expected, "numeric helper result and destination guard bytes");
            expect(c.chunks == (extra ? std::vector<unsigned>{capacity, 1}
            : std::vector<unsigned>{capacity}), "exact chunk sizes");
            expect(c.copies.size() == 1 + extra && c.copies[0].first == first && (!extra
                   || c.copies[1].first == first + capacity), "chunk query indices end exactly at heap boundary");
            std::vector<unsigned> expected_groups(1 + extra, 1);
            if (mode == MockReplay::binary) {
                expected_groups[0] = binary_capacity / binary_group_size;
            }
            expect(c.groups == expected_groups, "dispatch workgroup counts");
            if (mode == MockReplay::binary) {
                expect(c.transfers.size() == 1 + extra && (!extra
                       || c.transfers[1].dstOffset == output_offset + word_size * capacity),
                       "binary second output chunk offset");
            } else {
                expect(c.transfers.size() == 1 && c.transfers[0].size == word_size * width,
                       "sum copies only the final accumulator");
            }
        }
    }

    // A previous call must not leave stale accumulator state in the next call.
    c.mode = MockReplay::sum;
    c.values[qp][first_query][0] = reset_sum;
    helper.sum({}, qp, first_query, 1, dst, output_offset);
    expect(c.word(dst, output_offset) == reset_sum, "sum accumulator reset between calls");
    c.mode = MockReplay::statistics;
    c.values[qp][first_query].fill(reset_counter);
    helper.sum_statistics({}, qp, VK_NULL_HANDLE, nullptr, first_query, 1, dst, output_offset);
    for (unsigned j = 0; j < statistics_counters; ++j) {
        expect(c.word(dst, output_offset + word_size * j) == reset_counter, "statistics accumulator reset");
    }
}

void test_query_state() {
    constexpr unsigned heap_count = 4;
    constexpr unsigned overlap_segments = 3;
    constexpr unsigned split_statistics_segments = 5;
    constexpr unsigned mixed_statistics_segments = 4;
    constexpr VkDeviceSize arena_offset = 40;
    constexpr VkDeviceSize boundary_offset = 24;
    constexpr VkDeviceSize buffer_bytes = 4200;
    constexpr VkDeviceSize resource_bytes = 4096;
    constexpr VkDeviceSize reused_offset = 16;
    constexpr VkDeviceSize repeated_offset = 128;
    constexpr VkDeviceSize overwrite_offset = 144;
    constexpr VkDeviceSize zero_offset = 200;
    constexpr VkDeviceSize statistics_offset = 256;
    constexpr VkDeviceSize later_statistics_offset = 512;
    constexpr VkDeviceSize inner_statistics_offset = 768;
    constexpr VkDeviceSize binary_offset = 864;
    constexpr VkDeviceSize timestamp_offset = 880;
    constexpr VkDeviceSize sum_offset = 896;
    constexpr VkDeviceSize mixed_statistics_offset = 1024;
    constexpr std::uint64_t first_saved_count = 17;
    constexpr std::uint64_t second_saved_count = 23;

    MockReplay c;
    auto vk = c.replay_functions();
    ResourceBinding dst{};
    dst.buffer = handle<VkBuffer>(1);
    dst.buffer_offset = arena_offset;
    dst.size = resource_bytes;
    dst.host_readable = true;
    c.buffers[dst.buffer].assign(buffer_bytes, guard_byte);
    auto resource = [&](ResourceId id) -> const ResourceBinding* {
        return id == 1 ? &dst : nullptr;
    };
    QueryPool heap{handle<VkQueryPool>(2), d3d12_query_heap::occlusion, heap_count, true,
        std::vector<bool>(heap_count), {}};
    QueryPool stat{handle<VkQueryPool>(3), d3d12_query_heap::pipeline_statistics, heap_count, false,
        std::vector<bool>(heap_count), {}};
    QueryPool time{handle<VkQueryPool>(4), d3d12_query_heap::timestamp, heap_count, false,
        std::vector<bool>(heap_count), {}};
    for (auto p : {heap.pool, stat.pool, time.pool}) {
        c.values[p].resize(heap_count);
    }
    vk.query_pool = [&](QueryHeapId id) -> QueryPool* {
        return id == 1 ? &heap : id == 2 ? &stat : id == 3 ? &time : nullptr;
    };
    auto prep = [&](CommandList& list, PreparedList& out) {
        expect(list.close(), "close query list");
        expect(prepare(list, resource, vk, out) == ReplayResult::success, "prepare query list");
    };

    // Inclusive final slot / exclusive heap end, for every supported resolve result width.
    for (auto type : {d3d12_query::occlusion, d3d12_query::binary_occlusion, d3d12_query::timestamp,
        d3d12_query::pipeline_statistics}) {
        unsigned id = type == d3d12_query::timestamp ? 3 : type == d3d12_query::pipeline_statistics ? 2 : 1;
        auto* hp = vk.query_pool(id);
        hp->ended = {false, false, true, true};
        for (unsigned i = 2; i < heap_count; ++i) {
            for (unsigned j = 0; j < statistics_counters; ++j) {
                c.values[hp->pool][i][j] = (UINT64_C(1) << 32) + 100 * i + j;
            }
        }
        auto size = query_result_size(type);
        for (unsigned count : {1u, 2u}) {
            CommandList list;
            expect(list.resolve_queries({id, type, heap_count - count, count, 1, boundary_offset}),
                   "record boundary resolve");
            PreparedList out;
            dst.size = boundary_offset + size * count;
            c.buffers[dst.buffer].assign(buffer_bytes, guard_byte);
            auto expected = c.buffers[dst.buffer];
            prep(list, out);
            QueryReplay q;
            expect(q.plan(out, {}, 0, true) == ReplayResult::success, "plan boundary");
            emit(out, {}, vk);
            for (unsigned i = 0; i < count; ++i) {
                for (unsigned j = 0; j < size / word_size; ++j) {
                    const auto v = type == d3d12_query::binary_occlusion
                                       ? UINT64_C(1)
                                       : c.values[hp->pool][heap_count - count + i][j];
                    std::memcpy(expected.data() + arena_offset + boundary_offset + size * i + word_size * j, &v,
                        word_size);
                }
            }
            expect(c.buffers[dst.buffer] == expected, "heap-end resolve numeric values, arena offset and guards");
            expect(out.wrote_readback, "query resolve readback visibility");
        }
        CommandList bad;
        expect(bad.resolve_queries({id, type, heap_count - 1, 2, 1, 0}) && bad.close(), "record one-past heap end");
        PreparedList invalid;
        expect(prepare(bad, resource, vk, invalid) == ReplayResult::invalid_query, "reject one-past heap range");
        CommandList short_dst;
        expect(short_dst.resolve_queries({id, type, heap_count - 1, 1, 1, boundary_offset}) && short_dst.close(),
               "record short destination");
        dst.size = boundary_offset + size - 1;
        expect(prepare(short_dst, resource, vk, invalid) == ReplayResult::invalid_query,
               "reject destination one byte short");
    }
    dst.size = resource_bytes;
    heap.ended.assign(heap_count, false);
    heap.occlusion.clear();

    // Plan three lists as VulkanQueue::execute does, before any emission or commit.
    CommandList a, b, d;
    PreparedList pa, pb, pd;
    expect(a.begin_query({1, d3d12_query::occlusion, 0}) && a.begin_query({1, d3d12_query::occlusion, 1})
           && a.end_query({1, d3d12_query::occlusion, 0}) && a.end_query({1, d3d12_query::occlusion, 1}),
           "record overlap");
    expect(b.resolve_queries({1, d3d12_query::occlusion, 0, 2, 1, 0}) && b.begin_query({1, d3d12_query::occlusion,
           0}) && b.end_query({1, d3d12_query::occlusion, 0}) && b.resolve_queries({1, d3d12_query::occlusion, 0, 1,
           1, reused_offset}), "record reuse after resolve");
    expect(d.resolve_queries({1, d3d12_query::occlusion, 0, 2, 1, boundary_offset}), "record third list resolve");
    prep(a, pa);
    prep(b, pb);
    prep(d, pd);
    expect(pa.segment_count == overlap_segments && pb.segment_count == 0 && pd.segment_count == 0,
           "queue needs exactly three physical occlusion slots");
    auto physical = make_segment_pool(10);
    c.values[physical->pool].resize(overlap_segments);
    std::weak_ptr<const OcclusionSegmentPool> alive = physical;
    QueryReplay batch;
    unsigned base = 0;
    for (auto* p : {&pa, &pb, &pd}) {
        expect(batch.plan(*p, physical, base, true) == ReplayResult::success, "plan multi-list batch");
        base += p->segment_count;
    }
    expect(heap.occlusion.empty() && !heap.ended[0] && !heap.ended[1],
           "planning does not publish results before commit");
    emit(pa, {}, vk);
    auto old0 = c.values[physical->pool][0][0] + c.values[physical->pool][1][0];
    auto old1 = c.values[physical->pool][1][0] + c.values[physical->pool][2][0];
    emit(pb, {}, vk);
    auto new0 = c.values[heap.pool][0][0];
    emit(pd, {}, vk);
    expect(c.word(dst.buffer, arena_offset) == old0 && c.word(dst.buffer, arena_offset + word_size) == old1
           && c.word(dst.buffer, arena_offset + reused_offset) == new0 && c.word(dst.buffer,
           arena_offset + boundary_offset) == new0 && c.word(dst.buffer,
           arena_offset + boundary_offset + word_size) == old1,
           "pending results, resolve preservation and slot reuse across three lists");
    batch.commit();
    expect(heap.ended[0] && heap.ended[1] && !heap.occlusion.count(0) && heap.occlusion.at(1).count == 2,
           "commit replaces reused slot metadata only");
    physical.reset();
    batch.retained.clear();
    expect(!alive.expired(), "heap retains segment pool after submission owner releases it");
    for (unsigned submission = 0; submission < 3; ++submission) {
        CommandList list;
        PreparedList p;
        expect(list.resolve_queries({1, d3d12_query::occlusion, 0, 2, 1, repeated_offset}), "record later resolve");
        prep(list, p);
        QueryReplay q;
        expect(q.plan(p, {}, 0, true) == ReplayResult::success, "plan later submission");
        emit(p, {}, vk);
        q.commit();
        expect(c.word(dst.buffer, arena_offset + repeated_offset) == new0 && c.word(dst.buffer,
               arena_offset + repeated_offset + word_size) == old1, "repeat resolve across submissions");
    }
    CommandList overwrite;
    PreparedList po;
    expect(overwrite.begin_query({1, d3d12_query::occlusion, 1}) && overwrite.end_query({1, d3d12_query::occlusion,
           1}) && overwrite.resolve_queries({1, d3d12_query::occlusion, 1, 1, 1, overwrite_offset}),
           "record last segment overwrite");
    prep(overwrite, po);
    {
        QueryReplay q;
        expect(q.plan(po, {}, 0, true) == ReplayResult::success, "plan native replacement");
        emit(po, {}, vk);
        q.commit();
        expect(!heap.occlusion.count(1) && c.word(dst.buffer,
               arena_offset + overwrite_offset) == c.values[heap.pool][1][0],
               "native result replaces last virtual result");
    }
    expect(alive.expired(), "last heap overwrite releases old segment pool");

    // Statistics pool size/base accumulation across two split lists; resolve during statistics must not count helpers.
    heap.ended[3] = false;
    CommandList s0, s1;
    PreparedList ps0, ps1;
    for (unsigned index = 0; index < 2; ++index) {
        auto& list = index ? s1 : s0;
        expect(list.begin_query({2, d3d12_query::pipeline_statistics, index}), "statistics begin");
        for (unsigned i = 0; i <= index; ++i) {
            expect(list.resolve_queries({1, d3d12_query::occlusion, 3, 1, 1, zero_offset}), "inner zero-fill resolve");
        }
        expect(list.resolve_queries({1, d3d12_query::occlusion, heap_count, 0, 1, 0}), "zero-count resolve at heap end");
        expect(list.end_query({2, d3d12_query::pipeline_statistics, index}) && list.resolve_queries({2,
               d3d12_query::pipeline_statistics, index, 1, 1, statistics_offset + statistics_size * index}),
               "statistics end and resolve");
    }
    prep(s0, ps0);
    prep(s1, ps1);
    expect(ps0.statistics_segment_count == 2 && ps1.statistics_segment_count == 3,
           "zero-count resolves allocate no statistics segment");
    auto sp = make_segment_pool(11);
    c.values[sp->pool].resize(split_statistics_segments);
    c.statistics_pools.insert(sp->pool);
    QueryReplay stats;
    unsigned sb = 0;
    for (auto* p : {&ps0, &ps1}) {
        expect(stats.plan(*p, {}, 0, true, sp, sb) == ReplayResult::success, "plan split statistics lists");
        sb += p->statistics_segment_count;
    }
    emit(ps0, {}, vk);
    emit(ps1, {}, vk);
    stats.commit();
    expect(stat.occlusion.at(0).first == 0 && stat.occlusion.at(1).first == 2 && sb == split_statistics_segments,
           "queue statistics pool sizing and disjoint bases");
    for (unsigned i = 0; i < 2; ++i) {
        for (unsigned j = 0; j < statistics_counters; ++j) {
            std::uint64_t total = 0;
            auto r = stat.occlusion.at(i);
            for (unsigned k = 0; k < r.count; ++k) {
                total += c.values[sp->pool][r.first + k][j];
            }
            expect(c.word(dst.buffer,
                   arena_offset + statistics_offset + statistics_size * i + word_size * j) == total,
                   "all 11 split statistics counters");
        }
    }

    // No segment ownership is needed on the next submission: heap metadata carries it.
    CommandList later;
    PreparedList pl;
    expect(later.resolve_queries({2, d3d12_query::pipeline_statistics, 0, 2, 1, later_statistics_offset}),
           "later statistics resolve");
    prep(later, pl);
    QueryReplay later_batch;
    expect(later_batch.plan(pl, {}, 0, true) == ReplayResult::success && later_batch.needs_statistics_sum,
           "later statistics helper selection");
    emit(pl, {}, vk);
    for (unsigned i = 0; i < (2 * statistics_size); i += word_size) {
        expect(c.word(dst.buffer, arena_offset + later_statistics_offset + i) == c.word(dst.buffer,
               arena_offset + statistics_offset + i), "statistics persist after commit");
    }

    // Mixed helper types inside one statistics scope, including a timestamp copy that must not split it.
    auto saved = make_segment_pool(12);
    c.values[saved->pool].resize(2);
    c.values[saved->pool][0][0] = 17;
    c.values[saved->pool][1][0] = 23;
    heap.occlusion[2] = {saved, 0, 2};
    heap.ended[2] = true;
    CommandList mixed;
    PreparedList pm;
    expect(mixed.begin_query({2, d3d12_query::pipeline_statistics, 2}) && mixed.resolve_queries({2,
           d3d12_query::pipeline_statistics, 0, 1, 1, inner_statistics_offset}) && mixed.resolve_queries({1,
           d3d12_query::binary_occlusion, 0, 1, 1, binary_offset}) && mixed.resolve_queries({3,
           d3d12_query::timestamp, 3, 1, 1, timestamp_offset}) && mixed.resolve_queries({1, d3d12_query::occlusion,
           2, 1, 1, sum_offset}) && mixed.end_query({2, d3d12_query::pipeline_statistics, 2})
           && mixed.resolve_queries({2, d3d12_query::pipeline_statistics, 2, 1, 1, mixed_statistics_offset}),
           "mixed helpers inside statistics");
    prep(mixed, pm);
    expect(pm.statistics_segment_count == mixed_statistics_segments, "only three non-timestamp resolves split statistics");
    auto mixed_pool = make_segment_pool(13);
    c.values[mixed_pool->pool].resize(mixed_statistics_segments);
    c.statistics_pools.insert(mixed_pool->pool);
    QueryReplay mixed_batch;
    expect(mixed_batch.plan(pm, {}, 0, true, mixed_pool) == ReplayResult::success && mixed_batch.needs_sum
           && mixed_batch.needs_binary && mixed_batch.needs_statistics_sum, "mixed helper selection");
    emit(pm, {}, vk);
    mixed_batch.commit();
    expect(c.word(dst.buffer, arena_offset + sum_offset) == first_saved_count + second_saved_count
           && c.word(dst.buffer, arena_offset + binary_offset) == 1,
           "ordinary sum and binary results inside statistics");
    for (unsigned j = 0; j < statistics_counters; ++j) {
        std::uint64_t total = 0;
        for (unsigned k = 0; k < mixed_statistics_segments; ++k) {
            total += c.values[mixed_pool->pool][k][j];
        }
        expect(c.word(dst.buffer, arena_offset + mixed_statistics_offset + word_size * j) == total,
               "mixed statistics sum");
        expect(c.word(dst.buffer, arena_offset + inner_statistics_offset + word_size * j) == c.word(dst.buffer,
               arena_offset + statistics_offset + word_size * j), "previous statistics sum inside a new scope");
    }
}

void test_predicated_copies() {
    constexpr VkDeviceSize source_arena = 32;
    constexpr VkDeviceSize destination_arena = 40;
    constexpr VkDeviceSize predicate_arena = 24;
    constexpr VkDeviceSize predicate_offset = word_size;
    constexpr VkDeviceSize predicate_address = predicate_arena + predicate_offset;
    constexpr unsigned resource_bytes = 128;
    constexpr unsigned backing_bytes = 200;
    constexpr unsigned predicate_backing_bytes = 64;
    constexpr unsigned copy_source = 9;
    constexpr unsigned copy_bytes = 7;
    constexpr unsigned cleared_offset = 8;
    constexpr unsigned resolved_offset = 16;
    constexpr unsigned snapshot_offset = 24;
    constexpr unsigned skipped_offset = 32;
    constexpr unsigned inverted_offset = 40;
    constexpr unsigned second_cleared_offset = 48;
    constexpr unsigned next_list_offset = 56;
    constexpr unsigned high_word_offset = 65;
    constexpr unsigned high_word_source = 11;
    constexpr std::uint32_t invalid_operation = 999;

    MockReplay c;
    auto vk = c.replay_functions();
    ResourceBinding src{}, dst{}, pred{};
    src.buffer = handle<VkBuffer>(1);
    src.buffer_offset = source_arena;
    src.size = resource_bytes;
    dst.buffer = handle<VkBuffer>(2);
    dst.buffer_offset = destination_arena;
    dst.size = resource_bytes;
    dst.host_readable = true;
    pred.buffer = handle<VkBuffer>(3);
    pred.buffer_offset = predicate_arena;
    pred.size = 2 * word_size;
    c.buffers[src.buffer].resize(backing_bytes);
    for (unsigned i = 0; i < backing_bytes; ++i) {
        c.buffers[src.buffer][i] = static_cast<unsigned char>(i);
    }
    c.word(src.buffer, source_arena, 0);
    c.buffers[dst.buffer].assign(backing_bytes, guard_byte);
    auto expected = c.buffers[dst.buffer];
    c.buffers[pred.buffer].assign(predicate_backing_bytes, guard_byte);
    c.word(pred.buffer, predicate_address, 0);
    QueryPool heap{handle<VkQueryPool>(4), d3d12_query_heap::occlusion, 1, false, {true}, {}};
    c.values[heap.pool].resize(1);
    c.values[heap.pool][0][0] = UINT64_C(1) << 40;
    vk.query_pool = [&](QueryHeapId id) -> QueryPool* {
        return id == 1 ? &heap : nullptr;
    };
    auto resources = [&](ResourceId id) -> const ResourceBinding* {
        return id == 1 ? &src : id == 2 ? &dst : id == 3 ? &pred : nullptr;
    };
    struct Job {
        VkBuffer a, b;
        VkBufferCopy copy;
    };
    std::vector<Job> jobs;
    vk.prepare_predicated_copy = [&](VkBuffer a, VkBuffer b, VkDeviceSize so, VkDeviceSize d, VkDeviceSize n,
        std::vector<PredicatedCopyDispatch>& out) {
        jobs.push_back({a, b, {so, d, n}});
        out.push_back({handle<VkDescriptorSet>(jobs.size()), 0, 0, static_cast<unsigned>(n)});
        return true;
    };
    vk.emit_predicated_copy = [&](VkCommandBuffer, const std::vector<PredicatedCopyDispatch>& out) {
        expect(c.conditional, "predicated copy outside scope");
        ++c.pred_copies;
        c.events.push_back("pred_copy");
        for (auto& p : out) {
            auto& job = jobs.at(reinterpret_cast<std::uintptr_t>(p.set) - 1);
            if (c.execute) {
                c.copy(job.a, job.b, job.copy);
            }
        }
    };
    auto expect_copy = [&](unsigned dest, unsigned source, unsigned n) {
        std::memcpy(expected.data() + destination_arena + dest,
            c.buffers[src.buffer].data() + source_arena + source, n);
    };
    CommandList list;
    expect(list.set_predication({3, predicate_offset, d3d12_predication::equal_zero}) && list.copy_buffer({2, 0, 1,
           copy_source, copy_bytes}), "record skipped copy");
    expect(list.resolve_queries({1, d3d12_query::binary_occlusion, 0, 1, 3, predicate_offset})
           && list.copy_buffer({2, 0, 1, copy_source, copy_bytes}),
           "resolve into active predicate then copy without recapture");
    expect(list.set_predication({0, UINT64_MAX, UINT32_MAX}) && list.copy_buffer({2, cleared_offset, 1, copy_source,
           copy_bytes}), "clear predicate ignores invalid offset/op");
    expect_copy(cleared_offset, copy_source, copy_bytes);
    expect(list.set_predication({3, predicate_offset, d3d12_predication::equal_zero}) && list.copy_buffer({2,
           resolved_offset, 1, copy_source, copy_bytes}), "recapture resolved nonzero");
    expect_copy(resolved_offset, copy_source, copy_bytes);
    expect(list.copy_buffer({3, predicate_offset, 1, 0, word_size}) && list.copy_buffer({2, snapshot_offset, 1,
           copy_source, copy_bytes}), "overwrite predicate without recapture");
    expect_copy(snapshot_offset, copy_source, copy_bytes);
    expect(list.set_predication({3, predicate_offset, d3d12_predication::equal_zero}) && list.copy_buffer({2,
           skipped_offset, 1, copy_source, copy_bytes}), "recapture zero skips");
    expect(list.set_predication({3, predicate_offset, d3d12_predication::not_equal_zero}) && list.copy_buffer({2,
           inverted_offset, 1, copy_source, copy_bytes}), "inverted zero executes");
    expect_copy(inverted_offset, copy_source, copy_bytes);
    expect(list.set_predication({0, 7, invalid_operation}) && list.copy_buffer({2, second_cleared_offset, 1,
           copy_source, copy_bytes}), "clear then native copy");
    expect_copy(second_cleared_offset, copy_source, copy_bytes);
    expect(list.set_predication({3, predicate_offset, d3d12_predication::equal_zero}) && list.copy_buffer({2,
           resource_bytes, 1, resource_bytes, 0}) && list.close(), "zero-length copy at buffer end while predicated");
    PreparedList p;
    expect(prepare(list, resources, vk, p) == ReplayResult::success, "prepare mixed list");
    expect(p.unpredicated_copies == 0, "buffer copies never counted as unsupported predication fallback");
    expect(p.readback_src_stages == (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT),
           "mixed readback producers");
    QueryReplay q;
    expect(q.plan(p, {}, 0, true) == ReplayResult::success, "plan resolve into predicate");
    emit(p, {}, vk);
    q.commit();
    expect(c.buffers[dst.buffer] == expected, "all copied/skipped bytes and guards match independent oracle");
    expect(c.word(pred.buffer, predicate_address) == 0, "predicate overwrite executes");
    expect(c.captures == 5 && c.pred_copies == 7 && c.native_copies == 2 && !c.conditional,
           "capture/copy counts; no zero-byte dispatch; final scope balanced");
    auto expected_pred = std::vector<unsigned char>(predicate_backing_bytes, guard_byte);
    std::uint64_t zero = 0;
    std::memcpy(expected_pred.data() + predicate_address, &zero, word_size);
    expect(c.buffers[pred.buffer] == expected_pred, "predicate-buffer arena guards");
    CommandList next;
    expect(next.copy_buffer({2, next_list_offset, 1, copy_source, copy_bytes}) && next.close(),
           "next list starts unpredicated");
    expect(replay(next, {}, resources, vk) == ReplayResult::success, "replay next list");
    expect_copy(next_list_offset, copy_source, copy_bytes);
    expect(c.buffers[dst.buffer] == expected && c.native_copies == 3, "predication does not leak to next list");
    c.word(pred.buffer, predicate_address, UINT64_C(1) << 40);
    CommandList high;
    expect(high.set_predication({3, predicate_offset, d3d12_predication::equal_zero}) && high.copy_buffer({2,
           high_word_offset, 1, high_word_source, copy_bytes}) && high.close(), "high-word-only predicate");
    expect(replay(high, {}, resources, vk) == ReplayResult::success, "replay high-word predicate");
    expect_copy(high_word_offset, high_word_source, copy_bytes);
    expect(c.buffers[dst.buffer] == expected, "full 64-bit predicate and unaligned copy offsets");

    // Recording-time controls; false returns must not mutate active query state.
    CommandList validation;
    expect(!validation.begin_query({1, d3d12_query::timestamp, 0}), "timestamp begin rejected");
    expect(validation.begin_query({1, d3d12_query::occlusion, 0}), "ordinary begin");
    expect(!validation.begin_query({1, d3d12_query::occlusion, 0}), "duplicate active slot rejected");
    expect(!validation.end_query({1, d3d12_query::binary_occlusion, 0}), "mismatched end type rejected");
    expect(!validation.resolve_queries({1, d3d12_query::occlusion, 0, 1, 2, 0}), "active slot resolve rejected");
    expect(validation.end_query({1, d3d12_query::occlusion, 0}), "rejected operations preserve matching end");
    expect(validation.begin_query({1, d3d12_query::occlusion, 0}) && validation.end_query({1,
           d3d12_query::occlusion, 0}), "same-list slot reuse");
    expect(!validation.resolve_queries({1, d3d12_query::occlusion, UINT32_MAX, 2, 2, 0}),
           "overflowing query range rejected");
    expect(!validation.resolve_queries({1, d3d12_query::occlusion, 0, 1, 2, 1}), "unaligned resolve offset rejected");
    expect(!validation.set_predication({3, 1, 0}) && !validation.set_predication({3, 0, invalid_operation}),
           "invalid nonnull predicate rejected");
    auto n = validation.commands().size();
    expect(validation.resolve_queries({1, d3d12_query::occlusion, 1, 0, 2, 0}) && validation.copy_buffer({2,
           UINT64_MAX, 1, UINT64_MAX, 0}) && validation.commands().size() == n, "zero-count commands record no IR");
    expect(validation.close() && !validation.copy_buffer({2, 0, 1, 0, 0}), "closed list rejects even zero-byte copies");
    CommandList open;
    expect(open.begin_query({1, d3d12_query::occlusion, 0}) && open.close(), "close synthetic IR with active query");
    PreparedList invalid;
    expect(prepare(open, resources, vk, invalid) == ReplayResult::invalid_query, "replay rejects unended query");
    pred.size = predicate_offset + word_size - 1;
    CommandList short_pred;
    expect(short_pred.set_predication({3, predicate_offset, 0}) && short_pred.close(), "record short predicate");
    expect(prepare(short_pred, resources, vk, invalid) == ReplayResult::invalid_predication,
           "all eight predicate bytes required");
}

VulkanDeviceFunctions driver;
std::vector<std::pair<VkQueryType, unsigned>> allocations;
std::set<VkQueryPool> destroyed;
constexpr unsigned first_pool_id = 100;
unsigned next_pool = first_pool_id;
bool fail_submit = false;
PFN_vkVoidFunction VKAPI_CALL get(VkDevice, const char* name) {
#define MOCK_DEVICE_FUNCTION(n)                                       \
    if (std::string_view(name) == #n) {                                \
        return reinterpret_cast<PFN_vkVoidFunction>(driver.n);         \
    }
    LEONIDA_VK_DEVICE_FUNCTIONS(MOCK_DEVICE_FUNCTION)
#undef MOCK_DEVICE_FUNCTION
    return nullptr;
}

void test_queue_segments() {
    constexpr unsigned list_count = 2;
    constexpr unsigned overlap_segments = 3;
    constexpr unsigned occlusion_slots_needed = list_count * overlap_segments;
    constexpr unsigned statistics_slots_needed = 2 + 3;
    constexpr unsigned successful_occlusion_pool = first_pool_id + 2;
    constexpr unsigned successful_statistics_pool = first_pool_id + 3;
    constexpr VkQueryPipelineStatisticFlags statistics_mask = (1u << statistics_counters) - 1;
    constexpr VkDeviceSize arena_offset = 24;
    constexpr VkDeviceSize probe_occlusion_offset = 16;
    constexpr VkDeviceSize probe_statistics_offset = 32;
    constexpr unsigned resource_bytes = 128;
    constexpr unsigned backing_bytes = 160;

    MockReplay c;
    auto r = c.replay_functions();
    // Only the entry points used by a query-only queue are supplied. No Vulkan loader is involved.
    driver.vkCmdResetQueryPool = r.reset_query_pool;
    driver.vkCmdBeginQuery = r.begin_query;
    driver.vkCmdEndQuery = r.end_query;
    driver.vkCmdWriteTimestamp = r.write_timestamp;
    driver.vkCmdCopyQueryPoolResults = r.copy_query_pool_results;
    driver.vkCmdFillBuffer = r.fill_buffer;
    driver.vkCreateCommandPool = +[](VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*,
        VkCommandPool* p) {
        *p = handle<VkCommandPool>(1);
        return VK_SUCCESS;
    };
    driver.vkDestroyCommandPool = +[](VkDevice, VkCommandPool, const VkAllocationCallbacks*) {};
    driver.vkCreateSemaphore = +[](VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*,
        VkSemaphore* s) {
        *s = handle<VkSemaphore>(1);
        return VK_SUCCESS;
    };
    driver.vkDestroySemaphore = +[](VkDevice, VkSemaphore, const VkAllocationCallbacks*) {};
    driver.vkGetSemaphoreCounterValue = +[](VkDevice, VkSemaphore, std::uint64_t* v) {
        *v = UINT64_MAX;
        return VK_SUCCESS;
    };
    driver.vkQueueWaitIdle = +[](VkQueue) {
        return VK_SUCCESS;
    };
    driver.vkAllocateCommandBuffers = +[](VkDevice, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* b) {
        expect(info->commandBufferCount == 1, "one command buffer");
        *b = handle<VkCommandBuffer>(1);
        return VK_SUCCESS;
    };
    driver.vkResetCommandBuffer = +[](VkCommandBuffer, VkCommandBufferResetFlags) {
        return VK_SUCCESS;
    };
    driver.vkBeginCommandBuffer = +[](VkCommandBuffer, const VkCommandBufferBeginInfo*) {
        return VK_SUCCESS;
    };
    driver.vkEndCommandBuffer = +[](VkCommandBuffer) {
        return VK_SUCCESS;
    };
    driver.vkQueueSubmit = +[](VkQueue, unsigned, const VkSubmitInfo*, VkFence) {
        return fail_submit ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
    };
    driver.vkCreateQueryPool = +[](VkDevice, const VkQueryPoolCreateInfo* info, const VkAllocationCallbacks*,
        VkQueryPool* out) {
        expect(info->queryCount > 0, "no empty pool allocation");
        if (info->queryType == VK_QUERY_TYPE_PIPELINE_STATISTICS) {
            expect(info->pipelineStatistics == statistics_mask, "all 11 counters selected");
        }
        allocations.emplace_back(info->queryType, info->queryCount);
        *out = handle<VkQueryPool>(next_pool++);
        MockReplay::current->values[*out].resize(info->queryCount);
        return VK_SUCCESS;
    };
    driver.vkDestroyQueryPool = +[](VkDevice, VkQueryPool p, const VkAllocationCallbacks*) {
        expect(destroyed.insert(p).second, "pool destroyed twice");
    };
    VulkanContext context(get, handle<VkDevice>(1), true);
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryTypeCount = 1;
    context.set_occlusion_resolve_memory(memory);
    QueryPool heap{handle<VkQueryPool>(2), d3d12_query_heap::occlusion, 4, false, std::vector<bool>(4), {}};
    QueryPool stat{handle<VkQueryPool>(3), d3d12_query_heap::pipeline_statistics, 2, false, std::vector<bool>(2), {}};
    c.values[heap.pool].resize(4);
    c.values[stat.pool].resize(2);
    context.add_query_pool(1, heap);
    context.add_query_pool(2, stat);
    ResourceBinding dst{};
    dst.buffer = handle<VkBuffer>(1);
    dst.size = resource_bytes;
    dst.buffer_offset = arena_offset;
    context.add_resource(1, dst);
    c.buffers[dst.buffer].assign(backing_bytes, guard_byte);
    {
        VulkanQueue queue(context, handle<VkQueue>(1), 0);
        expect(queue.ok(), "mock queue initialized");
        CommandList a, b;
        for (unsigned i = 0; i < list_count; ++i) {
            auto& list = i ? b : a;
            expect(list.begin_query({1, d3d12_query::occlusion, 0}) && list.begin_query({1, d3d12_query::occlusion,
                   1}) && list.end_query({1, d3d12_query::occlusion, 0}) && list.end_query({1,
                   d3d12_query::occlusion, 1}), "overlap scope");
            expect(list.begin_query({2, d3d12_query::pipeline_statistics, i}), "statistics begin");
            for (unsigned k = 0; k <= i; ++k) {
                expect(list.resolve_queries({1, d3d12_query::occlusion, 3, 1, 1, 0}), "split statistics on zero fill");
            }
            expect(list.end_query({2, d3d12_query::pipeline_statistics, i}) && list.close(), "statistics end");
        }
        const CommandList* lists[]{&a, &b};
        fail_submit = true;
        expect(queue.execute(lists) == ReplayResult::unsupported, "injected transient submit failure");
        fail_submit = false;
        const std::vector<std::pair<VkQueryType, unsigned>> want{{VK_QUERY_TYPE_OCCLUSION, occlusion_slots_needed},
            {VK_QUERY_TYPE_PIPELINE_STATISTICS, statistics_slots_needed}};
        expect(allocations == want && destroyed.size() == 2, "failed submission frees exact six/five-slot pools");
        // Published query state must still say 'never ended'. If commit ran on failure this would need a sum helper.
        CommandList probe;
        expect(probe.resolve_queries({1, d3d12_query::occlusion, 0, 2, 1, probe_occlusion_offset})
               && probe.resolve_queries({2, d3d12_query::pipeline_statistics, 0, 1, 1, probe_statistics_offset})
               && probe.close(), "failure-state probe");
        const CommandList* pp = &probe;
        expect(queue.execute({&pp, 1}) == ReplayResult::success, "failed query results were not published");
        expect(c.word(dst.buffer, arena_offset + probe_occlusion_offset) == 0 && c.word(dst.buffer,
               arena_offset + probe_occlusion_offset + word_size) == 0, "ordinary failure state zero-filled");
        for (unsigned j = 0; j < statistics_counters; ++j) {
            expect(c.word(dst.buffer, arena_offset + probe_statistics_offset + word_size * j) == 0,
                   "statistics failure state zero-filled");
        }
        expect(allocations == want, "resolve-only batch allocates no segment pools");
        allocations.clear();
        c.resets.clear();
        expect(queue.execute(lists) == ReplayResult::success, "retry multi-list submission");
        expect(allocations == want, "actual queue sums all list segment counts");
        std::vector<unsigned> occlusion_slots, statistics_slots;
        for (auto [p, i] : c.resets) {
            if (p == handle<VkQueryPool>(successful_occlusion_pool)) {
                occlusion_slots.push_back(i);
            }
            if (p == handle<VkQueryPool>(successful_statistics_pool)) {
                statistics_slots.push_back(i);
            }
        }
        expect(occlusion_slots == std::vector<unsigned>({0, 1, 2, 3, 4, 5})
               && statistics_slots == std::vector<unsigned>({0, 1, 2, 3, 4}),
               "second-list bases use every allocated slot exactly once");
        // Mock reset/end entry points bounds-check every index against the actual allocations. Nonzero bases in the
        // second list must cover occlusion slots 3..5 and statistics slots 2..4, without overlap or overflow.
        for (unsigned i = first_pool_id; i <= successful_statistics_pool; ++i) {
            expect(c.active[handle<VkQueryPool>(i)].empty(), "all physical queries ended");
        }
        expect(!destroyed.count(handle<VkQueryPool>(successful_occlusion_pool))
               && !destroyed.count(handle<VkQueryPool>(successful_statistics_pool)), "successful pools remain alive");
        context.remove_query_pool(1);
        context.remove_query_pool(2);
        expect(!destroyed.count(handle<VkQueryPool>(successful_occlusion_pool))
               && !destroyed.count(handle<VkQueryPool>(successful_statistics_pool)),
               "in-flight command slot retains pools after heap removal");
        CommandList empty;
        expect(empty.close(), "empty list");
        const CommandList* ep = &empty;
        expect(queue.execute({&ep, 1}) == ReplayResult::success, "reuse completed command slot");
        expect(destroyed.count(handle<VkQueryPool>(successful_occlusion_pool))
               && destroyed.count(handle<VkQueryPool>(successful_statistics_pool)),
               "completed slot releases pools after final owner disappears");
    }
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*run)();
    };
    const TestCase cases[] = {
        {"helper chunk boundaries", test_helper_chunks},
        {"query boundaries and persistent state", test_query_state},
        {"predication and recording checks", test_predicated_copies},
        {"queue segment allocation and failed submit", test_queue_segments},
    };
    unsigned failures = 0;
    for (const auto& test : cases) {
        try {
            test.run();
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL: %s: %s\n", test.name, error.what());
            ++failures;
        }
    }
    if (failures) {
        return 1;
    }
    std::puts("replay_query_edges_test: heap boundaries, chunking, persistent queries, "
              "predicated copies and queue ownership passed");
    return 0;
}
