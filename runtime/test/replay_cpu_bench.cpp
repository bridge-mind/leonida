// CPU benchmark of the D3D12 -> Vulkan replay: how long the layer itself takes per command to record a list
// (CommandList), check it (prepare) and write it (emit). Every Vulkan entry point is a no-op that only counts its
// calls, and resources are fake bindings, so no GPU, driver or game files are involved and the numbers are the
// layer's own cost. The real ExecuteCommandLists adds the driver's vkCmd* cost on top.
// A binary occlusion resolve counts as one mock call at the emit boundary. The benchmark excludes recording
// the Vulkan commands inside the queue's binary resolve helper.
//
// Two measurements:
//   - per command type: a list of N commands of one kind (transitions come in pairs, queries as Begin+End),
//     timed on its own, so one slow kind stands out;
//   - a frame: L lists of P passes, each pass a mix of what a frame records (barriers, whole-view and rectangle
//     clears, buffer and texture copies, a UAV clear, queries, predication and indexed and non-indexed draws).
// prepare includes refilling the PreparedList storage kept from the last execution (as VulkanQueue::execute keeps
// prepared_) and planning the query batch. emit includes committing query state, as execute does.
// Each figure is the median over --reps runs after --warmup runs. Heap allocations (global operator new) and
// Vulkan calls are counted per command; both are deterministic, unlike the times.
//
// Usage:
//   replay_cpu_bench [--lists L] [--passes P] [--count N] [--reps R] [--warmup W] [--csv] [--self-test]
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_replay.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

// Heap allocations, counted for the whole process; read around one phase at a time.
static std::size_t g_allocations = 0;

void* operator new(std::size_t size) {
    ++g_allocations;
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using namespace leonida;

namespace {

std::size_t g_vk_calls = 0;

// A no-op for any vkCmd* signature that counts its calls.
template<typename F>
struct Nop;
template<typename R, typename... A>
struct Nop<R(A...)> {
    static R VKAPI_CALL call(A...) {
        ++g_vk_calls;
        if constexpr (!std::is_void_v<R>) return R{};
    }
};
template<typename PFN>
PFN nop() {
    return &Nop<std::remove_pointer_t<PFN>>::call;
}

template<typename H>
H fake_handle(std::uintptr_t value) {
    return reinterpret_cast<H>(value);
}

constexpr VkFormat rgba8 = VK_FORMAT_R8G8B8A8_UNORM;

// The resources one frame uses, registered like VulkanContext does: bindings by ID in an unordered_map, image
// views cached in a std::map keyed like VulkanContext::view.
struct Scene {
    enum : ResourceId {
        color = 1,
        depth,
        texture,  // 256x256 RGBA8, 9 mips: a copy destination and shader resource
        upload,   // 64 MiB upload buffer: copy source
        constants,
        uav,
        predicate,
        results,  // resolved query data
        occlusion_heap = 100,
        timestamp_heap,
    };
    static constexpr std::uint32_t queries = 4096;

    std::unordered_map<ResourceId, ResourceBinding> resources;
    std::map<std::tuple<VkImage, VkImageAspectFlags, std::uint32_t, std::uint32_t, std::uint32_t, VkFormat>,
             VkImageView>
        views;
    QueryPool occlusion, timestamps;
    // One graphics pipeline: an RGBA8 target, D32 depth and two vertex input slots, as VulkanContext keeps them.
    GraphicsPipeline pipeline;
    static constexpr std::uintptr_t pipeline_id = 0x7000;
    // A second one with no depth and one vertex slot, for the self-test's draw-storage check.
    GraphicsPipeline bare;
    static constexpr std::uintptr_t bare_id = 0x7001;
    static constexpr std::uint64_t geometry_va = 0x100000000ull;  // a 16 MiB vertex and index arena
    static constexpr std::uint64_t geometry_size = 16ull << 20;
    LayoutTracker tracker;
    ReplayFunctions vk;

    Scene() {
        const auto image = [&](ResourceId id, VkFormat format, VkImageAspectFlags aspects, std::uint32_t w,
                               std::uint32_t h, std::uint32_t mips, bool attachment) {
            ResourceBinding b{};
            b.image = fake_handle<VkImage>(0x1000 + id);
            b.format = format;
            b.aspects = aspects;
            b.width = w;
            b.height = h;
            b.mip_levels = mips;
            b.color_attachment = attachment;
            resources[id] = b;
        };
        const auto buffer = [&](ResourceId id, VkDeviceSize size) {
            ResourceBinding b{};
            b.buffer = fake_handle<VkBuffer>(0x2000 + id);
            b.size = size;
            resources[id] = b;
        };
        image(color, rgba8, VK_IMAGE_ASPECT_COLOR_BIT, 1920, 1080, 1, true);
        image(depth, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, 1920, 1080, 1, false);
        image(texture, rgba8, VK_IMAGE_ASPECT_COLOR_BIT, 256, 256, 9, false);
        buffer(upload, 64ull << 20);
        buffer(constants, 1 << 20);
        buffer(uav, 1 << 20);
        buffer(predicate, 4096);
        buffer(results, queries * 8);
        tracker.track(color, resources[color], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        tracker.track(depth, resources[depth], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        tracker.track(texture, resources[texture], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        occlusion = {fake_handle<VkQueryPool>(0x3000), d3d12_query_heap::occlusion, queries, true,
                     std::vector<bool>(queries)};
        timestamps = {fake_handle<VkQueryPool>(0x3001), d3d12_query_heap::timestamp, queries, false,
                      std::vector<bool>(queries)};

        vk.pipeline_barrier = nop<PFN_vkCmdPipelineBarrier>();
        vk.clear_color_image = nop<PFN_vkCmdClearColorImage>();
        vk.clear_depth_stencil_image = nop<PFN_vkCmdClearDepthStencilImage>();
        vk.pipeline_barrier2 = nop<PFN_vkCmdPipelineBarrier2>();
        vk.begin_rendering = nop<PFN_vkCmdBeginRendering>();
        vk.end_rendering = nop<PFN_vkCmdEndRendering>();
        vk.clear_attachments = nop<PFN_vkCmdClearAttachments>();
        vk.copy_buffer = nop<PFN_vkCmdCopyBuffer>();
        vk.copy_buffer_to_image = nop<PFN_vkCmdCopyBufferToImage>();
        vk.copy_image_to_buffer = nop<PFN_vkCmdCopyImageToBuffer>();
        vk.copy_image = nop<PFN_vkCmdCopyImage>();
        vk.resolve_image = nop<PFN_vkCmdResolveImage>();
        vk.fill_buffer = nop<PFN_vkCmdFillBuffer>();
        vk.update_buffer = nop<PFN_vkCmdUpdateBuffer>();
        vk.begin_conditional_rendering = nop<PFN_vkCmdBeginConditionalRenderingEXT>();
        vk.end_conditional_rendering = nop<PFN_vkCmdEndConditionalRenderingEXT>();
        vk.prepare_predicate = [](VkBuffer source, VkDeviceSize offset, PredicateCapture& out) {
            out = {source, offset, fake_handle<VkBuffer>(0x5000), 0, VK_NULL_HANDLE};
            return true;
        };
        vk.capture_predicate = [](VkCommandBuffer, const PredicateCapture&) {};
        vk.reset_query_pool = nop<PFN_vkCmdResetQueryPool>();
        vk.begin_query = nop<PFN_vkCmdBeginQuery>();
        vk.end_query = nop<PFN_vkCmdEndQuery>();
        vk.write_timestamp = nop<PFN_vkCmdWriteTimestamp>();
        vk.copy_query_pool_results = nop<PFN_vkCmdCopyQueryPoolResults>();
        vk.resolve_binary_occlusion = [](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer,
                                          VkDeviceSize) { ++g_vk_calls; };
        vk.image_view = [this](const ResourceBinding& r, const VkImageSubresourceRange& range) {
            const auto key = std::make_tuple(r.image, range.aspectMask, range.baseMipLevel, range.baseArrayLayer,
                                             range.layerCount, r.format);
            if (const auto it = views.find(key); it != views.end()) return it->second;
            const VkImageView view = fake_handle<VkImageView>(0x4000 + views.size());
            views[key] = view;
            return view;
        };
        vk.query_pool = [this](QueryHeapId id) -> QueryPool* {
            if (id == occlusion_heap) return &occlusion;
            if (id == timestamp_heap) return &timestamps;
            return nullptr;
        };

        pipeline.pipeline = fake_handle<VkPipeline>(0x6000);
        pipeline.layout = fake_handle<VkPipelineLayout>(0x6001);
        pipeline.push_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pipeline.color_formats = {rgba8};
        pipeline.depth_format = VK_FORMAT_D32_SFLOAT;
        pipeline.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
        pipeline.vertex_slots = 0b11;
        pipeline.variant = [](std::uint32_t) { return fake_handle<VkPipeline>(0x6002); };
        bare.pipeline = fake_handle<VkPipeline>(0x6010);
        bare.layout = pipeline.layout;
        bare.push_stages = pipeline.push_stages;
        bare.color_formats = {rgba8};
        bare.vertex_slots = 0b1;
        vk.graphics_pipeline = [this](std::uintptr_t id) -> const GraphicsPipeline* {
            if (id == pipeline_id) return &pipeline;
            if (id == bare_id) return &bare;
            return nullptr;
        };
        vk.buffer_at = [](std::uint64_t va, std::uint64_t& size, VkBuffer& buffer, VkDeviceSize& offset) {
            if (va < geometry_va || va >= geometry_va + geometry_size) return false;
            buffer = fake_handle<VkBuffer>(0x2100);
            offset = va - geometry_va;
            size = std::min<std::uint64_t>(size, geometry_va + geometry_size - va);
            return true;
        };
        vk.zero_buffer = fake_handle<VkBuffer>(0x2101);
        vk.zero_size = 65536;
        vk.bind_pipeline = nop<PFN_vkCmdBindPipeline>();
        vk.push_constants = nop<PFN_vkCmdPushConstants>();
        vk.bind_descriptor_sets = nop<PFN_vkCmdBindDescriptorSets>();
        vk.bind_vertex_buffers2 = nop<PFN_vkCmdBindVertexBuffers2>();
        vk.bind_index_buffer = nop<PFN_vkCmdBindIndexBuffer>();
        vk.set_viewport = nop<PFN_vkCmdSetViewport>();
        vk.set_scissor = nop<PFN_vkCmdSetScissor>();
        vk.set_primitive_topology = nop<PFN_vkCmdSetPrimitiveTopology>();
        vk.set_primitive_restart_enable = nop<PFN_vkCmdSetPrimitiveRestartEnable>();
        vk.set_stencil_reference = nop<PFN_vkCmdSetStencilReference>();
        vk.set_blend_constants = nop<PFN_vkCmdSetBlendConstants>();
        vk.draw = nop<PFN_vkCmdDraw>();
        vk.draw_indexed = nop<PFN_vkCmdDrawIndexed>();
    }

    const ResourceBinding* resolve(ResourceId id) const {
        const auto it = resources.find(id);
        return it == resources.end() ? nullptr : &it->second;
    }
};

namespace s = d3d12_state;
namespace y = d3d12_sync;
namespace a = d3d12_access;
namespace q = d3d12_query;

constexpr ClearColor black{0, 0, 0, 1};
constexpr Rect two_rects[] = {{0, 0, 640, 360}, {1280, 720, 1920, 1080}};
const TargetView color_view{Scene::color, 0, 0, 1};
const TargetView depth_view{Scene::depth, 0, 0, 1};

// One kind of command, recorded `count` times (a pair or a Begin+End counts as two).
struct Kind {
    const char* name;
    bool (*record)(CommandList&, std::uint32_t index);
    std::uint32_t commands_per_record;
};

bool texture_copy(CommandList& list, std::uint32_t index) {
    // A 64x64 block into mip 0 from a placed footprint (row pitch 256, offsets 512-aligned).
    CopyTexture copy{};
    copy.dst = {Scene::texture, false, 0, 0, 0, 0, 0, 0};
    copy.src = {Scene::upload, true, 0, (index % 1024) * 16384ull, 64, 64, 1, 256};
    copy.x = (index % 4) * 64;
    copy.y = (index / 4 % 4) * 64;
    return list.copy_texture(copy);
}

bool uav_clear(CommandList& list, std::uint32_t index) {
    const UavView view{Scene::uav, true, VK_FORMAT_R32_UINT, (index % 256) * 4096ull, 4096, 4, 0, 0, 1};
    return list.clear_unordered_access(view, false, {index, 0, 0, 0});
}

// A draw into the colour and depth targets, as a scene's mesh draws are: two vertex streams, and for `indexed`
// a 16-bit index buffer; each draw reads its own part of the geometry arena.
bool draw(CommandList& list, std::uint32_t index, bool indexed) {
    Draw d{};
    d.pipeline = Scene::pipeline_id;
    auto views = std::make_shared<DrawViews>();  // out of line since D3D-391
    views->render_targets = {color_view};
    d.depth = depth_view;
    const std::uint64_t base = Scene::geometry_va + (index % 256) * 65536ull;
    views->vertex_buffers[0] = {base, 24576, 12};  // by input slot
    views->vertex_buffers[1] = {base + 24576, 16384, 8};
    d.views = std::move(views);
    if (indexed) d.index = {base + 40960, 6144, 57};
    d.topology = 4;  // D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST
    d.viewport = {0, 0, 1920, 1080, 0, 1};
    d.scissor = {0, 0, 1920, 1080};
    d.blend_factor = {1, 1, 1, 1};
    d.indexed = indexed;
    d.count = indexed ? 3072 : 2048;
    d.instances = 1;
    return list.draw(d);
}

constexpr char texture_barrier_kind[] = "texture barrier pair (enh.)";

const Kind kinds[] = {
    {"transition pair",
     [](CommandList& l, std::uint32_t) {
         return l.transition({Scene::color, all_subresources, s::pixel_shader_resource, s::render_target}) &&
                l.transition({Scene::color, all_subresources, s::render_target, s::pixel_shader_resource});
     },
     2},
    {"buffer barrier (enhanced)",
     [](CommandList& l, std::uint32_t) {
         return l.barrier(BufferBarrier{y::copy, y::pixel_shading, a::copy_dest, a::shader_resource, Scene::constants});
     },
     1},
    {texture_barrier_kind,
     [](CommandList& l, std::uint32_t) {
         const BarrierSubresourceRange all{all_subresources, 0, 0, 0, 0, 0};
         return l.barrier(TextureBarrier{y::pixel_shading, y::copy, a::shader_resource, a::copy_dest,
                                         d3d12_layout::shader_resource, d3d12_layout::copy_dest, Scene::texture, all,
                                         0}) &&
                l.barrier(TextureBarrier{y::copy, y::pixel_shading, a::copy_dest, a::shader_resource,
                                         d3d12_layout::copy_dest, d3d12_layout::shader_resource, Scene::texture, all,
                                         0});
     },
     2},
    {"clear RTV, whole view", [](CommandList& l, std::uint32_t) { return l.clear_render_target(color_view, black); }, 1},
    {"clear RTV, 2 rects",
     [](CommandList& l, std::uint32_t) { return l.clear_render_target(color_view, black, two_rects); }, 1},
    {"clear DSV, whole view",
     [](CommandList& l, std::uint32_t) { return l.clear_depth_stencil(depth_view, clear_depth, 1, 0); },
     1},
    {"copy buffer region",
     [](CommandList& l, std::uint32_t i) {
         return l.copy_buffer({Scene::constants, (i % 256) * 256ull, Scene::upload, (i % 4096) * 256ull, 256});
     },
     1},
    {"copy texture region", texture_copy, 1},
    {"clear UAV (buffer)", uav_clear, 1},
    {"occlusion Begin+End",
     [](CommandList& l, std::uint32_t i) {
         const std::uint32_t index = i % Scene::queries;
         return l.begin_query({Scene::occlusion_heap, q::binary_occlusion, index}) &&
                l.end_query({Scene::occlusion_heap, q::binary_occlusion, index});
     },
     2},
    {"timestamp",
     [](CommandList& l, std::uint32_t i) {
         return l.end_query({Scene::timestamp_heap, q::timestamp, i % Scene::queries});
     },
     1},
    {"resolve 16 queries",
     [](CommandList& l, std::uint32_t i) {
         const std::uint32_t first = (i * 16) % Scene::queries;
         return l.resolve_queries({Scene::occlusion_heap, q::binary_occlusion, first, 16, Scene::results, first * 8ull});
     },
     1},
    {"draw", [](CommandList& l, std::uint32_t i) { return draw(l, i, false); }, 1},
    {"draw indexed", [](CommandList& l, std::uint32_t i) { return draw(l, i, true); }, 1},
    {"predication on+off",
     [](CommandList& l, std::uint32_t) {
         return l.set_predication({Scene::predicate, 0, d3d12_predication::equal_zero}) &&
                l.set_predication({0, 0, 0});
     },
     2},
};

// Draws per pass; a frame's draws outnumber its other commands.
constexpr std::uint32_t draws_per_pass = 16;

// One pass of the frame.
bool frame_pass(CommandList& l, std::uint32_t pass) {
    const std::uint32_t query = (pass * 2) % Scene::queries;
    bool ok = l.end_query({Scene::timestamp_heap, q::timestamp, query}) &&
              l.transition({Scene::color, all_subresources, s::pixel_shader_resource, s::render_target}) &&
              l.clear_render_target(color_view, black) && l.clear_depth_stencil(depth_view, clear_depth, 1, 0) &&
              l.begin_query({Scene::occlusion_heap, q::binary_occlusion, query}) &&
              l.clear_render_target(color_view, black, two_rects);
    for (std::uint32_t i = 0; ok && i < draws_per_pass; ++i)
        ok = draw(l, pass * draws_per_pass + i, i % 4 != 3);  // three of four indexed
    return ok &&
           l.end_query({Scene::occlusion_heap, q::binary_occlusion, query}) &&
           l.transition({Scene::color, all_subresources, s::render_target, s::pixel_shader_resource}) &&
           l.copy_buffer({Scene::constants, (pass % 256) * 256ull, Scene::upload, (pass % 4096) * 256ull, 256}) &&
           l.barrier(BufferBarrier{y::copy, y::pixel_shading, a::copy_dest, a::shader_resource, Scene::constants}) &&
           l.transition({Scene::texture, 0, s::pixel_shader_resource, s::copy_dest}) && texture_copy(l, pass) &&
           l.transition({Scene::texture, 0, s::copy_dest, s::pixel_shader_resource}) &&
           l.transition({Scene::uav, all_subresources, s::common, s::unordered_access}) && uav_clear(l, pass) &&
           l.transition({Scene::uav, all_subresources, s::unordered_access, s::common}) &&
           l.set_predication({Scene::predicate, 0, d3d12_predication::not_equal_zero}) &&
           l.clear_render_target({Scene::color, 0, 0, 1}, black, two_rects) && l.set_predication({0, 0, 0}) &&
           l.end_query({Scene::timestamp_heap, q::timestamp, query + 1}) &&
           l.resolve_queries({Scene::timestamp_heap, q::timestamp, query, 2, Scene::results, query * 8ull});
}

struct Phase {
    double ns = 0;            // median, per command
    double allocations = 0;   // per command
    double vk_calls = 0;      // per command (emit only)
};

struct Result {
    std::size_t commands = 0;
    Phase record, prepare, emit;
    ReplayResult replay = ReplayResult::success;
    bool recorded = true;
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

// Records `lists` lists with `build`, prepares and emits them, `reps` times after `warmup` runs.
template<typename Build>
Result run(Scene& scene, std::size_t lists, Build&& build, int reps, int warmup) {
    using clock = std::chrono::steady_clock;
    std::vector<CommandList> recorded(lists);
    std::vector<PreparedList> prepared;
    const VkCommandBuffer command_buffer = fake_handle<VkCommandBuffer>(0x10);
    const auto resolve = [&scene](ResourceId id) { return scene.resolve(id); };
    std::vector<double> t_record, t_prepare, t_emit;
    Result out;
    for (int rep = 0; rep < warmup + reps; ++rep) {
        const std::size_t a0 = g_allocations;
        const auto t0 = clock::now();
        for (auto& list : recorded) {
            list.reset();
            if (!build(list)) out.recorded = false;
            list.close();
        }
        const auto t1 = clock::now();
        const std::size_t a1 = g_allocations;
        clock::time_point t2;
        std::size_t a2, v2;
        {
            // The PreparedList storage is kept between executions, as VulkanQueue::execute keeps prepared_.
            prepared.resize(lists);
            QueryReplay queries;
            for (std::size_t i = 0; i < lists; ++i) {
                if (const auto r = prepare(recorded[i], resolve, scene.vk, prepared[i]); r != ReplayResult::success) {
                    out.replay = r;
                    break;
                }
            }
            if (out.replay == ReplayResult::success) {
                for (auto& list : prepared) {
                    if (const auto r = queries.plan(list); r != ReplayResult::success) {
                        out.replay = r;
                        break;
                    }
                }
            }
            t2 = clock::now();
            a2 = g_allocations;
            v2 = g_vk_calls;
            if (out.replay == ReplayResult::success) {
                std::uint32_t open_labels = 0;
                for (const auto& list : prepared)
                    emit(list, command_buffer, scene.vk, &scene.tracker, &open_labels);
                close_labels(command_buffer, scene.vk, open_labels);
                queries.commit();  // every simulated submission completes immediately
            }
        }
        const auto t3 = clock::now();
        const std::size_t a3 = g_allocations, v3 = g_vk_calls;

        std::size_t commands = 0;
        for (const auto& list : recorded) commands += list.commands().size();
        out.commands = commands;
        if (rep < warmup || !commands) continue;
        const auto ns = [&](auto from, auto to) {
            return std::chrono::duration<double, std::nano>(to - from).count() / double(commands);
        };
        t_record.push_back(ns(t0, t1));
        t_prepare.push_back(ns(t1, t2));
        t_emit.push_back(ns(t2, t3));
        out.record.allocations = double(a1 - a0) / double(commands);
        out.prepare.allocations = double(a2 - a1) / double(commands);
        out.emit.allocations = double(a3 - a2) / double(commands);
        out.emit.vk_calls = double(v3 - v2) / double(commands);
    }
    out.record.ns = median(t_record);
    out.prepare.ns = median(t_prepare);
    out.emit.ns = median(t_emit);
    return out;
}

const char* result_name(ReplayResult r) {
    switch (r) {
        case ReplayResult::success: return "success";
        case ReplayResult::list_open: return "list_open";
        case ReplayResult::unresolved_resource: return "unresolved_resource";
        case ReplayResult::invalid_subresource: return "invalid_subresource";
        case ReplayResult::invalid_clear: return "invalid_clear";
        case ReplayResult::invalid_copy: return "invalid_copy";
        case ReplayResult::invalid_predication: return "invalid_predication";
        case ReplayResult::invalid_query: return "invalid_query";
        case ReplayResult::invalid_indirect_arguments: return "invalid_indirect_arguments";
        case ReplayResult::invalid_draw: return "invalid_draw";
        case ReplayResult::unsupported: return "unsupported";
    }
    return "?";
}

void print_header(bool csv) {
    if (csv)
        std::printf("case,commands,record_ns,prepare_ns,emit_ns,total_ns,record_allocs,prepare_allocs,emit_allocs,"
                    "vk_calls\n");
    else
        std::printf("%-28s %8s | %9s %9s %9s %9s | %6s %6s %6s | %5s\n", "ns per command", "commands", "record",
                    "prepare", "emit", "total", "alloc:r", "p", "e", "vk");
}

void print_row(const char* name, const Result& r, bool csv) {
    const double total = r.record.ns + r.prepare.ns + r.emit.ns;
    if (csv)
        std::printf("%s,%zu,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%.2f,%.2f\n", name, r.commands, r.record.ns, r.prepare.ns,
                    r.emit.ns, total, r.record.allocations, r.prepare.allocations, r.emit.allocations, r.emit.vk_calls);
    else
        std::printf("%-28s %8zu | %9.1f %9.1f %9.1f %9.1f | %6.2f %6.2f %6.2f | %5.2f\n", name, r.commands,
                    r.record.ns, r.prepare.ns, r.emit.ns, total, r.record.allocations, r.prepare.allocations,
                    r.emit.allocations, r.emit.vk_calls);
}

int usage() {
    std::fprintf(stderr,
                 "usage: replay_cpu_bench [--lists L] [--passes P] [--count N] [--reps R] [--warmup W] [--csv] "
                 "[--self-test]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t lists = 8, passes = 128, count = 4096;
    int reps = 25, warmup = 5;
    bool csv = false, self_test = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&](auto& out) {
            if (i + 1 >= argc) return false;
            const long long v = std::atoll(argv[++i]);
            if (v <= 0) return false;
            out = static_cast<std::remove_reference_t<decltype(out)>>(v);
            return true;
        };
        if (arg == "--lists") { if (!value(lists)) return usage(); }
        else if (arg == "--passes") { if (!value(passes)) return usage(); }
        else if (arg == "--count") { if (!value(count)) return usage(); }
        else if (arg == "--reps") { if (!value(reps)) return usage(); }
        else if (arg == "--warmup") { if (!value(warmup)) return usage(); }
        else if (arg == "--csv") csv = true;
        else if (arg == "--self-test") self_test = true;
        else return usage();
    }
    if (self_test) {  // small and quick: every kind and the frame must record and replay, deterministically
        lists = 2;
        passes = 4;
        count = 16;
        reps = 2;
        warmup = 1;
    }

    Scene scene;
    int failures = 0;
    const auto check = [&](const char* name, const Result& r) {
        if (r.recorded && r.replay == ReplayResult::success && r.commands && r.emit.vk_calls > 0) return;
        std::fprintf(stderr, "FAIL %s: recorded=%d replay=%s commands=%zu vk_calls=%.2f\n", name, r.recorded,
                     result_name(r.replay), r.commands, r.emit.vk_calls);
        ++failures;
    };

    if (!self_test) print_header(csv);
    for (const Kind& kind : kinds) {
        const std::uint32_t records = std::uint32_t(std::max<std::size_t>(1, count / kind.commands_per_record));
        const Result r = run(
            scene, 1,
            [&](CommandList& list) {
                for (std::uint32_t i = 0; i < records; ++i)
                    if (!kind.record(list, i)) return false;
                return true;
            },
            reps, warmup);
        check(kind.name, r);
        // One image barrier for each piece: emit keeps the usual one or two on the stack.
        if (std::strcmp(kind.name, texture_barrier_kind) == 0 && r.emit.allocations != 0) {
            std::fprintf(stderr, "FAIL %s: emit makes %.2f heap allocations per command\n", kind.name,
                         r.emit.allocations);
            ++failures;
        }
        if (!self_test) print_row(kind.name, r, csv);
    }
    const auto frame = [&](CommandList& list) {
        for (std::uint32_t pass = 0; pass < passes; ++pass)
            if (!frame_pass(list, pass)) return false;
        return true;
    };
    const Result f = run(scene, lists, frame, reps, warmup);
    check("frame", f);
    if (self_test) {
        // A PreparedList keeps its draws' storage from one execution to the next. A draw prepared in an entry an
        // indexed draw with depth used before must not keep that draw's depth attachment or index buffer.
        CommandList earlier, later;
        Draw plain{};
        plain.pipeline = Scene::bare_id;
        auto plain_views = std::make_shared<DrawViews>();
        plain_views->render_targets = {color_view};
        plain.views = std::move(plain_views);
        plain.topology = 4;
        plain.viewport = {0, 0, 1920, 1080, 0, 1};
        plain.scissor = {0, 0, 1920, 1080};
        plain.count = 3;
        plain.instances = 1;
        PreparedList prepared;
        const auto resolve = [&scene](ResourceId id) { return scene.resolve(id); };
        const bool recorded = draw(earlier, 0, true) && draw(earlier, 1, true) && earlier.close() &&
                              later.draw(plain) && later.close();
        const bool replayed = recorded && prepare(earlier, resolve, scene.vk, prepared) == ReplayResult::success &&
                              prepare(later, resolve, scene.vk, prepared) == ReplayResult::success;
        const detail::PreparedDraw* d =
            replayed && prepared.draw_count == 1 ? &prepared.draws[prepared.commands[0].draw] : nullptr;
        if (!d || d->graphics != &scene.bare || d->pipeline != scene.bare.pipeline || d->depth.id ||
            d->index_buffer || d->vertex_buffers.size() != 1 || d->vertex_buffers[0] != scene.vk.zero_buffer ||
            d->colors.size() != 1 || d->colors[0].id != Scene::color) {
            std::fprintf(stderr, "FAIL draw storage: a draw kept state from the draw prepared there before\n");
            ++failures;
        }

        // The counts don't depend on timing: a second run must make exactly the same allocations and calls.
        const Result again = run(scene, lists, frame, reps, warmup);
        if (again.emit.vk_calls != f.emit.vk_calls || again.prepare.allocations != f.prepare.allocations) {
            std::fprintf(stderr, "FAIL frame: counts differ between runs\n");
            ++failures;
        }
        std::printf("replay_cpu_bench self-test: %zu kinds and a %zu-command frame recorded and replayed, %s\n",
                    std::size(kinds), f.commands, failures ? "FAILED" : "ok");
        return failures ? 1 : 0;
    }
    char name[64];
    std::snprintf(name, sizeof name, "frame (%zu lists x %zu passes)", lists, passes);
    print_row(name, f, csv);

    // The floor under prepare: one PreparedList entry per command, refilled in storage kept from the last
    // execution (what execute does), and for comparison made in new storage and freed every time.
    const auto entries_ns = [&](bool kept) {
        std::vector<double> ns;
        PreparedList storage;
        for (int rep = 0; rep < warmup + reps; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            if (kept) {
                storage.commands.assign(f.commands, {});
            } else {
                PreparedList fresh;
                fresh.commands.assign(f.commands, {});
            }
            const auto t1 = std::chrono::steady_clock::now();
            if (rep >= warmup) ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / double(f.commands));
        }
        return median(ns);
    };
    const double kept = entries_ns(true), fresh = entries_ns(false);
    if (csv) {
        std::printf("prepared_entries_kept,%zu,0,%.1f,0,%.1f,0,0,0,0\n", f.commands, kept, kept);
        std::printf("prepared_entries_fresh,%zu,0,%.1f,0,%.1f,0,0,0,0\n", f.commands, fresh, fresh);
        return failures ? 1 : 0;
    }
    std::printf("%-28s %8zu | %9s %9.1f %9s %9.1f |\n", "  entries, storage kept", f.commands, "", kept, "", kept);
    std::printf("%-28s %8zu | %9s %9.1f %9s %9.1f |\n", "  entries, made fresh", f.commands, "", fresh, "", fresh);
    std::printf("\nframe: %.1f us per ExecuteCommandLists of %zu commands, layer only (no driver)\n",
                (f.record.ns + f.prepare.ns + f.emit.ns) * double(f.commands) / 1000.0, f.commands);
    std::printf("one PreparedList entry (detail::Prepared): %zu bytes; one recorded Command: %zu bytes\n",
                sizeof(detail::Prepared), sizeof(Command));
    return failures ? 1 : 0;
}
