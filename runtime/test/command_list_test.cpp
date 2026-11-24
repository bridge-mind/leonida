#include "../src/command_list.h"

#include <cstdio>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

int main() {
    using namespace leonida;
    namespace s = d3d12_state;
    CommandList list;
    TargetView descriptor{12, 1, 2, 3};
    ClearColor color{1, 0, 0, 1};
    CHECK(!list.closed());
    CHECK(!list.clear_render_target({}, color));
    CHECK(list.clear_render_target(descriptor, color));
    // The CPU descriptor slot is reused; the recorded view keeps the old contents.
    descriptor = {24, 0, 0, 1};
    color[0] = 0;
    color[1] = 1;
    CHECK(list.clear_render_target(descriptor, color));

    // Barriers: a split barrier records only its end half.
    CHECK(!list.transition({0, 0, s::common, s::render_target}));
    CHECK(list.transition({24, all_subresources, s::render_target, s::copy_source}, barrier_begin_only));
    CHECK(list.transition({24, all_subresources, s::render_target, s::copy_source}, barrier_end_only));
    CHECK(!list.transition({24, 0, s::common, s::copy_dest}, barrier_begin_only | barrier_end_only));
    CHECK(!list.transition({24, 0, s::common, s::copy_dest}, 0x4));
    CHECK(list.uav_barrier({0}));
    CHECK(list.aliasing_barrier({0, 24}));

    // Depth-stencil clears: at least one flag, no unknown flags, depth in [0, 1].
    const TargetView depth{48, 0, 0, 1};
    CHECK(!list.clear_depth_stencil(depth, 0, 1.0f, 0));
    CHECK(!list.clear_depth_stencil(depth, 0x4, 1.0f, 0));
    CHECK(!list.clear_depth_stencil(depth, clear_depth, 1.5f, 0));
    CHECK(!list.clear_depth_stencil(depth, clear_depth, -0.0001f, 0));
    CHECK(!list.clear_depth_stencil({}, clear_depth, 1.0f, 0));
    CHECK(list.clear_depth_stencil(depth, clear_stencil, 7.0f, 0x5a));  // depth is ignored without clear_depth
    CHECK(list.clear_depth_stencil(depth, clear_depth | clear_stencil, 0.25f, 3));

    CHECK(list.close());
    CHECK(list.closed());
    CHECK(!list.close());
    CHECK(!list.clear_render_target(descriptor, color));
    CHECK(!list.transition({24, 0, s::copy_source, s::render_target}));
    CHECK(!list.uav_barrier({0}));

    const auto commands = list.commands();
    CHECK(commands.size() == 7);
    const auto* first = std::get_if<ClearRenderTarget>(&commands[0]);
    const auto* second = std::get_if<ClearRenderTarget>(&commands[1]);
    CHECK(first && second);
    CHECK(first->target.resource == 12 && second->target.resource == 24);
    CHECK(first->target.mip == 1 && first->target.first_layer == 2 && first->target.layer_count == 3);
    const ClearColor red{1, 0, 0, 1};
    CHECK(first->color == red && second->color == color);
    const auto* transition = std::get_if<TransitionBarrier>(&commands[2]);
    CHECK(transition && transition->resource == 24 && transition->subresource == all_subresources);
    CHECK(transition->before == s::render_target && transition->after == s::copy_source);
    CHECK(std::get_if<UavBarrier>(&commands[3]) && std::get<UavBarrier>(commands[3]).resource == 0);
    CHECK(std::get_if<AliasingBarrier>(&commands[4]) && std::get<AliasingBarrier>(commands[4]).after == 24);
    const auto* stencil = std::get_if<ClearDepthStencil>(&commands[5]);
    CHECK(stencil && stencil->flags == clear_stencil && stencil->stencil == 0x5a);
    const auto* both = std::get_if<ClearDepthStencil>(&commands[6]);
    CHECK(both && both->flags == (clear_depth | clear_stencil) && both->depth == 0.25f && both->stencil == 3);

    // Rectangles are copied while recording.
    CHECK(list.reset());
    Rect rects[] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    CHECK(list.clear_render_target({36, 0, 0, 1}, color, rects));
    CHECK(list.clear_depth_stencil({48, 0, 0, 1}, clear_depth, 0.5f, 0, {rects, 1}));
    rects[0] = {};
    CHECK(list.close());
    const auto& rect_clear = std::get<ClearRenderTarget>(list.commands()[0]);
    CHECK(rect_clear.rects.size() == 2 && rect_clear.rects[0].left == 1 && rect_clear.rects[1].bottom == 8);
    CHECK(std::get<ClearDepthStencil>(list.commands()[1]).rects.size() == 1);

    // Enhanced barriers: split begin halves are dropped, end halves kept; invalid shapes are refused.
    namespace y = d3d12_sync;
    namespace a = d3d12_access;
    namespace l = d3d12_layout;
    CHECK(list.reset());
    const BarrierSubresourceRange all{all_subresources, 0, 0, 0, 0, 0};
    CHECK(list.barrier(GlobalBarrier{y::all, y::all, a::common, a::common}));
    CHECK(list.barrier(GlobalBarrier{y::compute_shading, y::split, a::unordered_access, a::shader_resource}));
    CHECK(list.barrier(GlobalBarrier{y::split, y::pixel_shading, a::unordered_access, a::shader_resource}));
    CHECK(!list.barrier(GlobalBarrier{y::split, y::split, 0, 0}));
    CHECK(list.barrier(BufferBarrier{y::copy, y::vertex_shading, a::copy_dest, a::vertex_buffer, 7}));
    CHECK(!list.barrier(BufferBarrier{y::copy, y::vertex_shading, a::copy_dest, a::vertex_buffer, 0}));
    CHECK(list.barrier(TextureBarrier{y::none, y::render_target, a::no_access, a::render_target, l::undefined,
                                      l::render_target, 9, all, texture_barrier_discard}));
    CHECK(!list.barrier(TextureBarrier{y::all, y::all, a::common, a::common, l::common, l::undefined, 9, all, 0}));
    CHECK(!list.barrier(TextureBarrier{y::all, y::all, a::common, a::common, l::common, l::common, 9, all, 0x2}));
    CHECK(!list.barrier(TextureBarrier{y::all, y::all, a::common, a::common, l::common, l::common, 9,
                                       {0, 1, 0, 0, 0, 1}, 0}));  // zero layers
    CHECK(list.close());
    CHECK(list.commands().size() == 4);
    CHECK(std::get<GlobalBarrier>(list.commands()[1]).sync_before == y::split);
    CHECK(std::get<BufferBarrier>(list.commands()[2]).resource == 7);
    CHECK(std::get<TextureBarrier>(list.commands()[3]).flags == texture_barrier_discard);

    // Copies: D3D12's own call rules (aligned footprints; degenerate boxes and DEFAULT-heap
    // CopyResource self-copies are accepted as no-ops and must not record Vulkan commands).
    CHECK(list.reset());
    CHECK(list.copy_buffer({1, 0, 2, 16, 64}));
    CHECK(list.copy_buffer({1, 0, 2, 16, 0}));
    CHECK(!list.copy_buffer({0, 0, 2, 16, 64}));
    const CopyLocation texture{3, false, 0, 0, 0, 0, 0, 0};
    const CopyLocation placed{4, true, 0, 512, 8, 8, 1, 256};
    CHECK(list.copy_texture({texture, 0, 0, 0, placed, false, {}}));
    CHECK(list.copy_texture({placed, 0, 0, 0, texture, true, {2, 2, 0, 6, 6, 1}}));
    CHECK(!list.copy_texture({placed, 0, 0, 0, placed, false, {}}));                 // footprint to footprint
    CHECK(!list.copy_texture({texture, 0, 0, 0, {4, true, 0, 100, 8, 8, 1, 256}, false, {}}));  // offset not 512
    CHECK(!list.copy_texture({texture, 0, 0, 0, {4, true, 0, 512, 8, 8, 1, 100}, false, {}}));  // pitch not 256
    CHECK(list.commands().size() == 3);
    CHECK(list.copy_texture({texture, 0, 0, 0, texture, true, {2, 2, 0, 2, 6, 1}}));           // empty box, no-op
    CHECK(list.commands().size() == 3);
    CHECK(list.copy_resource({5, 6}));
    CHECK(list.commands().size() == 4);
    CHECK(list.copy_resource({5, 5}));  // DEFAULT-heap self-copy; COM layer rejects UPLOAD self-copies
    CHECK(list.commands().size() == 4);
    // A volume's box has a z range, and its footprint a depth; a resolve needs two different resources.
    CHECK(list.copy_texture({texture, 0, 0, 2, {4, true, 0, 512, 4, 4, 4, 256}, true, {0, 0, 1, 4, 4, 3}}));
    CHECK(!list.copy_texture({texture, 0, 0, 0, {4, true, 0, 512, 4, 4, 0, 256}, false, {}}));  // no slices
    CHECK(list.copy_texture({texture, 0, 0, 0, texture, true, {0, 0, 2, 4, 4, 2}}));            // empty in z, no-op
    CHECK(list.commands().size() == 5);
    CHECK(list.resolve_subresource({7, 1, 8, 0}));
    CHECK(!list.resolve_subresource({7, 0, 7, 0}));
    CHECK(!list.resolve_subresource({0, 0, 8, 0}));
    CHECK(!list.resolve_subresource({7, 0, 0, 0}));
    CHECK(list.close());
    CHECK(!list.resolve_subresource({7, 1, 8, 0}));
    CHECK(list.commands().size() == 6);
    CHECK(std::get<CopyBuffer>(list.commands()[0]).src_offset == 16);
    CHECK(std::get<CopyTexture>(list.commands()[1]).src.footprint && !std::get<CopyTexture>(list.commands()[1]).has_box);
    CHECK(std::get<CopyTexture>(list.commands()[2]).box.right == 6);
    CHECK(std::get<CopyResource>(list.commands()[3]).src == 6);
    const auto& slices = std::get<CopyTexture>(list.commands()[4]);
    CHECK(slices.z == 2 && slices.src.depth == 4 && slices.box.front == 1 && slices.box.back == 3);
    const auto& resolved = std::get<ResolveSubresource>(list.commands()[5]);
    CHECK(resolved.dst == 7 && resolved.dst_subresource == 1 && resolved.src == 8 && resolved.src_subresource == 0);

    // Dispatches keep the user data they were recorded with; no pipeline is refused, no groups records nothing.
    CHECK(list.reset());
    Dispatch dispatch{0x1234, {}, 2, 1, 1};
    dispatch.user_data[0] = 7;
    dispatch.user_data[31] = 9;
    CHECK(list.dispatch(dispatch));
    dispatch.user_data[0] = 8;  // the caller's block changes after recording
    CHECK(!list.dispatch({0, {}, 1, 1, 1}));
    CHECK(list.dispatch({0x1234, {}, 0, 1, 1}));
    CHECK(list.close());
    CHECK(!list.dispatch(dispatch));
    CHECK(list.commands().size() == 1);
    const auto& recorded = std::get<Dispatch>(list.commands()[0]);
    CHECK(recorded.pipeline == 0x1234 && recorded.user_data[0] == 7 && recorded.user_data[31] == 9 && recorded.x == 2);

    CHECK(list.reset());
    CHECK(!list.closed() && list.commands().empty());
    CHECK(!list.reset());
    CHECK(list.clear_render_target({36, 0, 0, 1}, color));
    CHECK(list.close());
    CHECK(list.commands().size() == 1);
    CHECK(std::get<ClearRenderTarget>(list.commands()[0]).target.resource == 36);

    // D3D-26: discards, immediate writes and atomic copies.
    CHECK(list.reset());
    CHECK(list.discard({40, 0, 0, false, true}));                  // the whole resource
    CHECK(list.discard({40, 2, 3, true}));
    CHECK(list.discard({40, 0, 0, false}));                       // nothing to discard: not recorded
    CHECK(!list.discard({0, 0, 0, false, true}));
    CHECK(!list.discard({40, 0xfffffff0u, 0x20, false}));         // past the last subresource index
    CHECK(!list.discard({40, 1, all_subresources, false}));       // an explicit count of UINT_MAX is a range
    CHECK(!list.discard({40, all_subresources, 1, false}));
    CHECK(list.write_immediate({0x1000, 7, write_immediate_mode::marker_out}));
    CHECK(!list.write_immediate({0, 7, 0}));                      // null address
    CHECK(!list.write_immediate({0x1002, 7, 0}));                 // not 4-byte aligned
    CHECK(!list.write_immediate({0x1000, 7, 3}));                 // not a mode
    const DependentRange dependent{41, 0, 0, 64};
    CHECK(list.atomic_copy({42, 8, 43, 16, 8, {dependent}}));
    CHECK(list.atomic_copy({42, 4, 42, 8, 4, {dependent}}));      // within one buffer
    CHECK(!list.atomic_copy({42, 8, 42, 8, 8, {dependent}}));     // onto itself
    CHECK(!list.atomic_copy({42, 4, 43, 16, 8, {dependent}}));    // UINT64 at an offset that is not 8-aligned
    CHECK(!list.atomic_copy({42, 8, 43, 16, 2, {dependent}}));    // neither UINT nor UINT64
    CHECK(list.atomic_copy({42, 8, 43, 16, 4, {}}));              // zero dependencies is valid on Windows D3D12
    CHECK(!list.atomic_copy({42, 8, 43, 16, 4, {{41, 0, 8, 4}}})); // inverted dependent range
    CHECK(!list.atomic_copy({42, 8, 43, 16, 4, {{41, 0, 4, 4}}})); // empty dependent range
    CHECK(!list.atomic_copy({42, 8, 43, 16, 4, {{0, 0, 0, 4}}})); // a null dependent resource
    CHECK(!list.atomic_copy({0, 8, 43, 16, 4, {dependent}}));
    CHECK(list.close());
    CHECK(!list.discard({40, 0, 0, false, true}) && !list.discard({40, 0, 0, false}));
    CHECK(list.commands().size() == 6);
    CHECK(std::get<DiscardResource>(list.commands()[0]).whole);
    CHECK(std::get<DiscardResource>(list.commands()[1]).first_subresource == 2 &&
          std::get<DiscardResource>(list.commands()[1]).partial && !std::get<DiscardResource>(list.commands()[1]).whole);
    CHECK(std::get<WriteImmediate>(list.commands()[2]).value == 7);
    const auto& atomic = std::get<AtomicCopyBuffer>(list.commands()[3]);
    CHECK(atomic.bytes == 8 && atomic.dst_offset == 8 && atomic.dependencies.size() == 1 &&
          atomic.dependencies[0].end == 64);

    // Synthetic IR values here test copy-by-value mechanics only; the COM test below enforces D3D12 bundle
    // restrictions on every currently implemented forbidden operation.
    // Bundles are copied into the parent at ExecuteBundle time and can be reused independently.
    CommandList parent, bundle;
    CHECK(parent.copy_resource({1, 2}));
    CHECK(bundle.copy_resource({3, 4}));
    CHECK(bundle.copy_buffer({5, 0, 6, 16, 32}));
    CHECK(bundle.close());
    CHECK(parent.execute_bundle(bundle));
    CHECK(parent.copy_resource({7, 8}));
    CHECK(parent.execute_bundle(bundle));  // the same closed bundle can execute more than once
    CHECK(parent.close());
    CHECK(parent.commands().size() == 6);
    CHECK(std::get<CopyResource>(parent.commands()[0]).src == 2);
    CHECK(std::get<CopyResource>(parent.commands()[1]).src == 4);
    CHECK(std::get<CopyBuffer>(parent.commands()[2]).size == 32);
    CHECK(std::get<CopyResource>(parent.commands()[3]).src == 8);
    CHECK(std::get<CopyResource>(parent.commands()[4]).src == 4);
    CHECK(std::get<CopyBuffer>(parent.commands()[5]).size == 32);

    // Resetting/rerecording the source bundle cannot mutate the parent's copied commands.
    CHECK(bundle.reset());
    CHECK(bundle.copy_resource({9, 10}));
    CHECK(bundle.close());
    CHECK(parent.commands().size() == 6);
    CHECK(std::get<CopyResource>(parent.commands()[1]).src == 4);
    CHECK(!parent.execute_bundle(bundle));  // parent is closed

    // The range form: [first, last) of the bundle, with last clamped to its size.
    {
        CommandList source, target, other;
        CHECK(source.copy_resource({1, 2}) && source.copy_resource({3, 4}) && source.copy_resource({5, 6}));
        CHECK(source.close());
        CHECK(target.execute_bundle(source, 1, 2));  // the middle command only
        CHECK(target.commands().size() == 1 && std::get<CopyResource>(target.commands()[0]).src == 4);
        CHECK(target.execute_bundle(source, 2, 100));  // last beyond the size is clamped
        CHECK(target.commands().size() == 2 && std::get<CopyResource>(target.commands()[1]).src == 6);
        CHECK(target.execute_bundle(source, 1, 1) && target.commands().size() == 2);  // first == last: nothing
        CHECK(target.execute_bundle(source, 3, 3) && target.execute_bundle(source, 7, 9));  // first beyond the size
        CHECK(target.execute_bundle(source, 2, 1) && target.commands().size() == 2);  // first after last: nothing
        CHECK(target.commands().size() == 2);
        CHECK(other.execute_bundle(source));  // the default arguments copy all of it, as before
        CHECK(other.commands().size() == 3 && std::get<CopyResource>(other.commands()[0]).src == 2 &&
              std::get<CopyResource>(other.commands()[2]).src == 6);
    }

    CommandList open_parent, open_bundle;
    CHECK(!open_parent.execute_bundle(open_bundle));  // bundle must be closed
    CHECK(!open_parent.execute_bundle(open_parent));  // a list cannot execute itself
    CHECK(open_parent.commands().empty());

    // SetPredication (D3D-18): the op must be one of the two D3D12 values and the offset 8-byte aligned; a null
    // buffer turns it off whatever the other arguments are.
    namespace pred = d3d12_predication;
    CommandList predicated;
    CHECK(predicated.set_predication({7, 0, pred::equal_zero}));
    CHECK(predicated.set_predication({7, 16, pred::not_equal_zero}));
    CHECK(!predicated.set_predication({7, 4, pred::equal_zero}));   // not 8-byte aligned
    CHECK(!predicated.set_predication({7, 0, 2}));                  // no such op
    CHECK(predicated.set_predication({0, 4, 2}));                   // off: offset and op ignored
    CHECK(predicated.close());
    CHECK(!predicated.set_predication({7, 0, pred::equal_zero}));  // closed
    CHECK(predicated.commands().size() == 3);
    CHECK(std::get<Predication>(predicated.commands()[0]).op == pred::equal_zero);
    CHECK(std::get<Predication>(predicated.commands()[1]).offset == 16);
    const auto& off = std::get<Predication>(predicated.commands()[2]);
    CHECK(off.buffer == 0 && off.offset == 0 && off.op == 0);
    // Queries: timestamps only end; other queries are begun and ended in one list, with the same type, and are not
    // begun twice or resolved while running. Resolve offsets are multiples of 8; a resolve of nothing records nothing.
    namespace q = d3d12_query;
    namespace h = d3d12_query_heap;
    CHECK(list.reset());
    constexpr QueryHeapId heap = 70, other = 71;
    CHECK(!list.begin_query({0, q::occlusion, 0}));
    CHECK(!list.begin_query({heap, q::timestamp, 0}));
    CHECK(list.end_query({heap, q::timestamp, 3}));
    CHECK(list.end_query({heap, q::timestamp, 3}));  // a timestamp can be written again
    CHECK(!list.end_query({heap, q::occlusion, 1}));  // never begun
    CHECK(list.begin_query({heap, q::occlusion, 1}));
    CHECK(!list.begin_query({heap, q::binary_occlusion, 1}));  // already running
    CHECK(list.begin_query({other, q::occlusion, 1}));         // same index, another heap
    CHECK(!list.end_query({heap, q::binary_occlusion, 1}));    // begun with another type
    CHECK(!list.resolve_queries({heap, q::occlusion, 0, 2, 9, 0}));  // query 1 is running
    CHECK(list.resolve_queries({heap, q::occlusion, 2, 2, 9, 0}));
    CHECK(list.queries_open());
    CHECK(list.end_query({heap, q::occlusion, 1}));
    CHECK(list.end_query({other, q::occlusion, 1}));
    CHECK(!list.queries_open());
    CHECK(!list.resolve_queries({heap, q::occlusion, 0, 2, 9, 4}));  // offset not a multiple of 8
    CHECK(!list.resolve_queries({heap, q::occlusion, 0, 2, 0, 8}));  // no destination
    CHECK(!list.resolve_queries({heap, q::occlusion, 0xffffffff, 2, 9, 8}));  // range wraps
    CHECK(list.resolve_queries({heap, q::occlusion, 0, 2, 9, 16}));
    CHECK(list.resolve_queries({heap, q::occlusion, 0, 0, 9, 16}));  // nothing to resolve
    CHECK(list.close());
    CHECK(list.commands().size() == 8);
    CHECK(std::get<EndQuery>(list.commands()[0]).index == 3 && std::get<EndQuery>(list.commands()[0]).type == q::timestamp);
    CHECK(std::get<BeginQuery>(list.commands()[2]).heap == heap && std::get<BeginQuery>(list.commands()[3]).heap == other);
    const auto& query_resolve = std::get<ResolveQueries>(list.commands()[7]);
    CHECK(query_resolve.first == 0 && query_resolve.count == 2 && query_resolve.dst == 9 && query_resolve.offset == 16);

    // A query still running at Close leaves the list closed and marked; Reset clears it.
    CHECK(list.reset());
    CHECK(list.begin_query({heap, q::pipeline_statistics, 0}));
    CHECK(list.close() && list.queries_open());
    CHECK(list.reset() && !list.queries_open());
    CHECK(list.begin_query({heap, q::pipeline_statistics, 0}));  // free again after the reset

    CHECK(query_fits_heap(h::occlusion, q::binary_occlusion) && !query_fits_heap(h::occlusion, q::timestamp));
    CHECK(query_fits_heap(h::copy_queue_timestamp, q::timestamp) && !query_fits_heap(h::timestamp, q::occlusion));
    CHECK(query_fits_heap(h::so_statistics, q::so_statistics_stream3) && !query_fits_heap(6, q::occlusion));
    CHECK(query_result_size(q::occlusion) == 8 && query_result_size(q::timestamp) == 8);
    CHECK(query_result_size(q::pipeline_statistics) == 88 && query_result_size(q::so_statistics_stream0) == 16);

    // Acceleration-structure commands: 256-byte AS/scratch alignment, 8-byte postbuild alignment,
    // valid copy modes (0..4), valid postbuild info types (0..3), and PERFORM_UPDATE source checks.
    CHECK(list.close());
    CHECK(list.reset());
    BuildAccelerationStructure build{};
    build.dst_va = 0x1000;
    build.scratch_va = 0x2000;
    CHECK(list.build_acceleration_structure(build));
    build.dst_va = 0;
    CHECK(!list.build_acceleration_structure(build));
    build.dst_va = 0x1080;  // misaligned
    CHECK(!list.build_acceleration_structure(build));
    build.dst_va = 0x1000;
    build.scratch_va = 0;
    CHECK(!list.build_acceleration_structure(build));
    build.scratch_va = 0x2080;  // misaligned
    CHECK(!list.build_acceleration_structure(build));
    build.scratch_va = 0x2000;
    build.update = true;
    build.src_va = 0;
    CHECK(!list.build_acceleration_structure(build));
    build.src_va = 0x3080;  // misaligned update source
    CHECK(!list.build_acceleration_structure(build));
    build.src_va = 0x3000;
    CHECK(list.build_acceleration_structure(build));
    build.update = false;
    build.src_va = 0x3080;  // ignored when not updating
    CHECK(list.build_acceleration_structure(build));

    CHECK(list.copy_acceleration_structure({0x1000, 0x2000, 0}));
    CHECK(list.copy_acceleration_structure({0x1000, 0x2000, 4}));
    CHECK(!list.copy_acceleration_structure({0x1000, 0x2000, 5}));
    CHECK(!list.copy_acceleration_structure({0, 0x2000, 0}));
    CHECK(!list.copy_acceleration_structure({0x1080, 0x2000, 0}));
    CHECK(!list.copy_acceleration_structure({0x1000, 0, 0}));
    CHECK(!list.copy_acceleration_structure({0x1000, 0x2080, 0}));

    CHECK(list.acceleration_structure_postbuild({0x4000, 0, {0x1000}}));
    CHECK(list.acceleration_structure_postbuild({0x4008, 3, {0x1000, 0x2000}}));
    CHECK(!list.acceleration_structure_postbuild({0x4000, 4, {0x1000}}));
    CHECK(!list.acceleration_structure_postbuild({0, 0, {0x1000}}));
    CHECK(!list.acceleration_structure_postbuild({0x4004, 0, {0x1000}}));
    CHECK(!list.acceleration_structure_postbuild({0x4000, 0, {}}));
    CHECK(!list.acceleration_structure_postbuild({0x4000, 0, {0}}));
    CHECK(!list.acceleration_structure_postbuild({0x4000, 0, {0x1080}}));
    CHECK(list.close());
    CHECK(list.commands().size() == 7);
    return 0;
}
