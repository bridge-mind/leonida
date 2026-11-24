#pragma once

// The command-list IR (docs/d3d12_vulkan.md section 3.2): what a D3D12 command list records, kept until
// ExecuteCommandLists replays it on Vulkan (vulkan_replay.h). This slice covers legacy and enhanced barriers and
// render-target, depth-stencil and unordered-access clears (whole views or rectangles), copies, resolves,
// predication, discards, immediate buffer writes, atomic copies, queries, debug markers and dispatches. It has
// no Windows or Vulkan dependency.

#include <algorithm>
#include <array>
#include <utility>
#include <cstdint>
#include <optional>
#include <span>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace leonida {

using ResourceId = std::uintptr_t;
using ClearColor = std::array<float, 4>;

// D3D12 enum values from the public d3d12.h, so the IR builds without the Windows SDK.
// d3d12_state_map_test checks every one against d3d12.h on Windows.
namespace d3d12_state {
inline constexpr std::uint32_t common = 0;
inline constexpr std::uint32_t vertex_and_constant_buffer = 0x1;
inline constexpr std::uint32_t index_buffer = 0x2;
inline constexpr std::uint32_t render_target = 0x4;
inline constexpr std::uint32_t unordered_access = 0x8;
inline constexpr std::uint32_t depth_write = 0x10;
inline constexpr std::uint32_t depth_read = 0x20;
inline constexpr std::uint32_t non_pixel_shader_resource = 0x40;
inline constexpr std::uint32_t pixel_shader_resource = 0x80;
inline constexpr std::uint32_t stream_out = 0x100;
inline constexpr std::uint32_t indirect_argument = 0x200;
inline constexpr std::uint32_t predication = 0x200;  // D3D12_RESOURCE_STATE_PREDICATION, the same bit
inline constexpr std::uint32_t copy_dest = 0x400;
inline constexpr std::uint32_t copy_source = 0x800;
inline constexpr std::uint32_t resolve_dest = 0x1000;
inline constexpr std::uint32_t resolve_source = 0x2000;
inline constexpr std::uint32_t raytracing_acceleration_structure = 0x400000;
inline constexpr std::uint32_t shading_rate_source = 0x1000000;
inline constexpr std::uint32_t generic_read = 0xac3;
inline constexpr std::uint32_t present = 0;
} // namespace d3d12_state

// D3D12_BARRIER_SYNC (enhanced barriers).
namespace d3d12_sync {
inline constexpr std::uint32_t none = 0;
inline constexpr std::uint32_t all = 0x1;
inline constexpr std::uint32_t draw = 0x2;
inline constexpr std::uint32_t index_input = 0x4;
inline constexpr std::uint32_t vertex_shading = 0x8;
inline constexpr std::uint32_t pixel_shading = 0x10;
inline constexpr std::uint32_t depth_stencil = 0x20;
inline constexpr std::uint32_t render_target = 0x40;
inline constexpr std::uint32_t compute_shading = 0x80;
inline constexpr std::uint32_t raytracing = 0x100;
inline constexpr std::uint32_t copy = 0x200;
inline constexpr std::uint32_t resolve = 0x400;
inline constexpr std::uint32_t execute_indirect = 0x800;
inline constexpr std::uint32_t predication = 0x800;
inline constexpr std::uint32_t all_shading = 0x1000;
inline constexpr std::uint32_t non_pixel_shading = 0x2000;
inline constexpr std::uint32_t emit_raytracing_acceleration_structure_postbuild_info = 0x4000;
inline constexpr std::uint32_t clear_unordered_access_view = 0x8000;
inline constexpr std::uint32_t video_decode = 0x100000;
inline constexpr std::uint32_t video_process = 0x200000;
inline constexpr std::uint32_t video_encode = 0x400000;
inline constexpr std::uint32_t build_raytracing_acceleration_structure = 0x800000;
inline constexpr std::uint32_t copy_raytracing_acceleration_structure = 0x1000000;
inline constexpr std::uint32_t split = 0x80000000;
} // namespace d3d12_sync

// D3D12_BARRIER_ACCESS.
namespace d3d12_access {
inline constexpr std::uint32_t common = 0;
inline constexpr std::uint32_t vertex_buffer = 0x1;
inline constexpr std::uint32_t constant_buffer = 0x2;
inline constexpr std::uint32_t index_buffer = 0x4;
inline constexpr std::uint32_t render_target = 0x8;
inline constexpr std::uint32_t unordered_access = 0x10;
inline constexpr std::uint32_t depth_stencil_write = 0x20;
inline constexpr std::uint32_t depth_stencil_read = 0x40;
inline constexpr std::uint32_t shader_resource = 0x80;
inline constexpr std::uint32_t stream_output = 0x100;
inline constexpr std::uint32_t indirect_argument = 0x200;
inline constexpr std::uint32_t predication = 0x200;
inline constexpr std::uint32_t copy_dest = 0x400;
inline constexpr std::uint32_t copy_source = 0x800;
inline constexpr std::uint32_t resolve_dest = 0x1000;
inline constexpr std::uint32_t resolve_source = 0x2000;
inline constexpr std::uint32_t raytracing_acceleration_structure_read = 0x4000;
inline constexpr std::uint32_t raytracing_acceleration_structure_write = 0x8000;
inline constexpr std::uint32_t shading_rate_source = 0x10000;
inline constexpr std::uint32_t video_decode_read = 0x20000;
inline constexpr std::uint32_t video_decode_write = 0x40000;
inline constexpr std::uint32_t video_process_read = 0x80000;
inline constexpr std::uint32_t video_process_write = 0x100000;
inline constexpr std::uint32_t video_encode_read = 0x200000;
inline constexpr std::uint32_t video_encode_write = 0x400000;
inline constexpr std::uint32_t no_access = 0x80000000;
} // namespace d3d12_access

// D3D12_BARRIER_LAYOUT.
namespace d3d12_layout {
inline constexpr std::uint32_t undefined = 0xffffffff;
inline constexpr std::uint32_t common = 0;
inline constexpr std::uint32_t present = 0;
inline constexpr std::uint32_t generic_read = 1;
inline constexpr std::uint32_t render_target = 2;
inline constexpr std::uint32_t unordered_access = 3;
inline constexpr std::uint32_t depth_stencil_write = 4;
inline constexpr std::uint32_t depth_stencil_read = 5;
inline constexpr std::uint32_t shader_resource = 6;
inline constexpr std::uint32_t copy_source = 7;
inline constexpr std::uint32_t copy_dest = 8;
inline constexpr std::uint32_t resolve_source = 9;
inline constexpr std::uint32_t resolve_dest = 10;
inline constexpr std::uint32_t shading_rate_source = 11;
inline constexpr std::uint32_t video_decode_read = 12;
inline constexpr std::uint32_t video_decode_write = 13;
inline constexpr std::uint32_t video_process_read = 14;
inline constexpr std::uint32_t video_process_write = 15;
inline constexpr std::uint32_t video_encode_read = 16;
inline constexpr std::uint32_t video_encode_write = 17;
inline constexpr std::uint32_t direct_queue_common = 18;
inline constexpr std::uint32_t direct_queue_generic_read = 19;
inline constexpr std::uint32_t direct_queue_unordered_access = 20;
inline constexpr std::uint32_t direct_queue_shader_resource = 21;
inline constexpr std::uint32_t direct_queue_copy_source = 22;
inline constexpr std::uint32_t direct_queue_copy_dest = 23;
inline constexpr std::uint32_t compute_queue_common = 24;
inline constexpr std::uint32_t compute_queue_generic_read = 25;
inline constexpr std::uint32_t compute_queue_unordered_access = 26;
inline constexpr std::uint32_t compute_queue_shader_resource = 27;
inline constexpr std::uint32_t compute_queue_copy_source = 28;
inline constexpr std::uint32_t compute_queue_copy_dest = 29;
inline constexpr std::uint32_t video_queue_common = 30;
} // namespace d3d12_layout

// D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAGS, D3D12_CLEAR_FLAGS and
// D3D12_TEXTURE_BARRIER_FLAGS.
inline constexpr std::uint32_t all_subresources = 0xffffffff;
inline constexpr std::uint32_t barrier_begin_only = 0x1;
inline constexpr std::uint32_t barrier_end_only = 0x2;
inline constexpr std::uint32_t clear_depth = 0x1;
inline constexpr std::uint32_t clear_stencil = 0x2;
inline constexpr std::uint32_t texture_barrier_discard = 0x1;

// The format a color view names, as a VkFormat number (0, `resource`, is the resource's own). The named values are
// the RGBA8 family's; a typeless resource of any other family stores its view's VkFormat the same way (D3D-275).
enum class ColorViewFormat : std::uint32_t {
    resource = 0, rgba8_unorm = 37, rgba8_snorm = 38, rgba8_uint = 41, rgba8_sint = 42, rgba8_srgb = 43
};

// D3D12_QUERY_HEAP_TYPE and D3D12_QUERY_TYPE.
namespace d3d12_query_heap {
inline constexpr std::uint32_t occlusion = 0;
inline constexpr std::uint32_t timestamp = 1;
inline constexpr std::uint32_t pipeline_statistics = 2;
inline constexpr std::uint32_t so_statistics = 3;
inline constexpr std::uint32_t video_decode_statistics = 4;
inline constexpr std::uint32_t copy_queue_timestamp = 5;
inline constexpr std::uint32_t pipeline_statistics1 = 7;
} // namespace d3d12_query_heap

namespace d3d12_query {
inline constexpr std::uint32_t occlusion = 0;
inline constexpr std::uint32_t binary_occlusion = 1;
inline constexpr std::uint32_t timestamp = 2;
inline constexpr std::uint32_t pipeline_statistics = 3;
inline constexpr std::uint32_t so_statistics_stream0 = 4;
inline constexpr std::uint32_t so_statistics_stream3 = 7;
inline constexpr std::uint32_t video_decode_statistics = 8;
inline constexpr std::uint32_t pipeline_statistics1 = 10;
} // namespace d3d12_query

// A stream-output statistics query (SO_STATISTICS_STREAM0..3).
constexpr bool is_so_statistics(std::uint32_t query_type) {
    return query_type >= d3d12_query::so_statistics_stream0 && query_type <= d3d12_query::so_statistics_stream3;
}

// Whether a heap of `heap_type` holds queries of `query_type` (the D3D12 pairing rules).
constexpr bool query_fits_heap(std::uint32_t heap_type, std::uint32_t query_type) {
    namespace h = d3d12_query_heap;
    namespace q = d3d12_query;
    switch (heap_type) {
    case h::occlusion: return query_type == q::occlusion || query_type == q::binary_occlusion;
    case h::timestamp:
    case h::copy_queue_timestamp: return query_type == q::timestamp;
    case h::pipeline_statistics: return query_type == q::pipeline_statistics;
    case h::so_statistics: return query_type >= q::so_statistics_stream0 && query_type <= q::so_statistics_stream3;
    case h::video_decode_statistics: return query_type == q::video_decode_statistics;
    case h::pipeline_statistics1: return query_type == q::pipeline_statistics1;
    default: return false;
    }
}

// Bytes one resolved query takes: a UINT64 for occlusion and timestamps, D3D12_QUERY_DATA_PIPELINE_STATISTICS
// (11 UINT64s) for pipeline statistics, D3D12_QUERY_DATA_SO_STATISTICS (2 UINT64s) for stream output. 0 for the
// types this layer does not resolve.
constexpr std::uint32_t query_result_size(std::uint32_t query_type) {
    namespace q = d3d12_query;
    switch (query_type) {
    case q::occlusion:
    case q::binary_occlusion:
    case q::timestamp: return 8;
    case q::pipeline_statistics: return 11 * 8;
    // D3D12_QUERY_DATA_SO_STATISTICS: NumPrimitivesWritten, PrimitivesStorageNeeded.
    case q::so_statistics_stream0:
    case q::so_statistics_stream0 + 1:
    case q::so_statistics_stream0 + 2:
    case q::so_statistics_stream3: return 2 * 8;
    default: return 0;
    }
}

// A view captured by value while recording: the CPU descriptor slot may be rewritten after the call.
struct TargetView {
    ResourceId resource;
    std::uint32_t mip;
    std::uint32_t first_layer;
    std::uint32_t layer_count;
    ColorViewFormat format = ColorViewFormat::resource;
};

// D3D12_RECT: right and bottom are exclusive.
struct Rect {
    std::int32_t left;
    std::int32_t top;
    std::int32_t right;
    std::int32_t bottom;
};

// D3D12 subresource index: mip + layer * mips + plane * mips * layers, or all_subresources.
struct TransitionBarrier {
    ResourceId resource;
    std::uint32_t subresource;
    std::uint32_t before;
    std::uint32_t after;
};

// resource 0 means every UAV access, as a null pResource does in D3D12.
struct UavBarrier {
    ResourceId resource;
};

// Either resource may be 0 ("any placed or reserved resource").
struct AliasingBarrier {
    ResourceId before;
    ResourceId after;
};

// Enhanced barriers (ID3D12GraphicsCommandList7::Barrier).
struct GlobalBarrier {
    std::uint32_t sync_before;
    std::uint32_t sync_after;
    std::uint32_t access_before;
    std::uint32_t access_after;
};

// Offset and size are not kept: D3D12 requires 0 and UINT64_MAX (the whole buffer).
struct BufferBarrier {
    std::uint32_t sync_before;
    std::uint32_t sync_after;
    std::uint32_t access_before;
    std::uint32_t access_after;
    ResourceId resource;
};

// D3D12_BARRIER_SUBRESOURCE_RANGE: mip_count == 0 means index_or_first_mip is a subresource index
// (all_subresources for every one).
struct BarrierSubresourceRange {
    std::uint32_t index_or_first_mip;
    std::uint32_t mip_count;
    std::uint32_t first_layer;
    std::uint32_t layer_count;
    std::uint32_t first_plane;
    std::uint32_t plane_count;
};

struct TextureBarrier {
    std::uint32_t sync_before;
    std::uint32_t sync_after;
    std::uint32_t access_before;
    std::uint32_t access_after;
    std::uint32_t layout_before;
    std::uint32_t layout_after;
    ResourceId resource;
    BarrierSubresourceRange range;
    std::uint32_t flags;  // texture_barrier_discard
};

// rects empty: the whole view.
// CopyBufferRegion.
struct CopyBuffer {
    ResourceId dst;
    std::uint64_t dst_offset;
    ResourceId src;
    std::uint64_t src_offset;
    std::uint64_t size;
    ResourceId tiled_resource = 0;  // CopyTiles uses the padded size of the sparse side
};

// D3D12_TEXTURE_COPY_LOCATION: a texture subresource, or a placed footprint in a buffer. The footprint's format is
// not kept: the copy uses the texture's own format for the plane, which D3D12 requires to be compatible.
struct CopyLocation {
    ResourceId resource;
    bool footprint;
    std::uint32_t subresource;  // texture
    std::uint64_t offset;       // footprint: byte offset in the buffer
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t depth;
    std::uint32_t row_pitch;
};

// D3D12_BOX: right, bottom and back are exclusive.
struct CopyBox {
    std::uint32_t left;
    std::uint32_t top;
    std::uint32_t front;
    std::uint32_t right;
    std::uint32_t bottom;
    std::uint32_t back;
};

// CopyTextureRegion. Without a box, the whole source subresource or footprint is copied.
struct CopyTexture {
    CopyLocation dst;
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    CopyLocation src;
    bool has_box;
    CopyBox box;
    bool tile_footprint = false;  // RGBA8 CopyTiles: rows in a linear 64 KiB tile, at any byte offset
};

// CopyResource: every subresource, or the whole buffer.
struct CopyResource {
    ResourceId dst;
    ResourceId src;
};

// D3D12_RESOLVE_MODE values a region resolve replays.
namespace d3d12_resolve_mode {
inline constexpr std::uint32_t minimum = 1;
inline constexpr std::uint32_t maximum = 2;
inline constexpr std::uint32_t average = 3;
}

struct ResolveRegion {
    std::uint32_t dst_x;
    std::uint32_t dst_y;
    std::optional<Rect> src_rect;
    std::uint32_t mode = d3d12_resolve_mode::average;  // colour: average only; stencil: minimum or maximum
};

// Without a region, both subresources have the same extent. A region with no source rectangle takes the
// entire source to the given destination offset. The resolve uses the images' own format.
struct ResolveSubresource {
    ResourceId dst;
    std::uint32_t dst_subresource;
    ResourceId src;
    std::uint32_t src_subresource;
    std::optional<ResolveRegion> region = std::nullopt;
};

// Dispatch, with what it uses captured when it is recorded: the pipeline (an opaque id the replay resolves) and
// the 32 user-data dwords the root arguments produced (s0..s31 of the translated shader, section 3.5).
struct Dispatch {
    std::uintptr_t pipeline;
    std::array<std::uint32_t, 32> user_data;
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
};

// D3D12_VIEWPORT.
struct Viewport {
    float x;
    float y;
    float width;
    float height;
    float min_depth;
    float max_depth;
};

// D3D12_VERTEX_BUFFER_VIEW / D3D12_INDEX_BUFFER_VIEW: console VAs, resolved to buffers at replay.
struct VertexBufferView {
    std::uint64_t va;
    std::uint32_t size;
    std::uint32_t stride;
};

struct IndexBufferView {
    std::uint64_t va;
    std::uint32_t size;
    std::uint32_t format;  // DXGI_FORMAT_R16_UINT (57) or DXGI_FORMAT_R32_UINT (42)
};

// The views a draw captured, at the D3D12 maxima. Kept by value because the CPU descriptor slots and the bound
// views can change after the draw is recorded, and out of line because the command variant is sized by its largest
// member: inline they made every recorded command, barriers included, 976 bytes.
struct DrawViews {
    std::array<TargetView, 8> render_targets{};
    std::array<VertexBufferView, 32> vertex_buffers{};  // by input slot
    std::uint32_t render_target_count = 8;
};

// DrawInstanced / DrawIndexedInstanced / DispatchMesh, with everything they use captured when they are recorded:
// D3D12 reads render-target and depth descriptors at OMSetRenderTargets, and every other piece of state is the
// list's state at the draw. resource 0 marks an unbound render-target slot or no depth buffer. A mesh draw
// (DispatchMesh) takes its thread-group counts in count (x), instances (y) and first (z), and ignores the vertex
// and index buffers and the topology.
// Render targets and vertex buffer views are inlined (8 simultaneous targets, 32 input slots), eliminating heap allocations
// during command list recording (D3D-289).
struct Draw {
    std::uintptr_t pipeline = 0;
    std::array<std::uint32_t, 32> user_data{};
    std::shared_ptr<const DrawViews> views;  // null: no render targets or vertex buffers captured
    TargetView depth{};
    IndexBufferView index{};
    std::uint32_t topology = 0;  // D3D_PRIMITIVE_TOPOLOGY
    Viewport viewport{};       // the first; more viewports are not modelled yet
    Rect scissor{};            // the first; D3D12 always scissors
    std::array<float, 4> blend_factor{1, 1, 1, 1};
    std::uint32_t stencil_front = 0;
    std::uint32_t stencil_back = 0;
    bool indexed = false;
    std::uint32_t count = 0;  // vertices or indices per instance
    std::uint32_t instances = 0;
    std::uint32_t first = 0;  // first vertex or first index
    std::int32_t base_vertex = 0;
    std::uint32_t first_instance = 0;
    bool mesh = false;
};

// D3D12_TEXTURE_DATA_PITCH_ALIGNMENT and D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT.
inline constexpr std::uint32_t texture_data_pitch_alignment = 256;
inline constexpr std::uint32_t texture_data_placement_alignment = 512;

struct ClearRenderTarget {
    TargetView target;
    ClearColor color;
    std::vector<Rect> rects;
};

struct ClearDepthStencil {
    TargetView target;
    std::uint32_t flags;  // clear_depth | clear_stencil
    float depth;
    std::uint8_t stencil;
    std::vector<Rect> rects;
};

// DiscardResource. whole: the whole resource (a null region); first_subresource and count are then ignored.
// Otherwise count subresources from first_subresource, an explicit range even when count is all_subresources.
// partial: the region had rectangles, so only part of each subresource is discarded.
struct DiscardResource {
    ResourceId resource;
    std::uint32_t first_subresource;
    std::uint32_t count;
    bool partial;
    bool whole = false;
};

// D3D12_WRITEBUFFERIMMEDIATE_MODE.
namespace write_immediate_mode {
inline constexpr std::uint32_t normal = 0;      // DEFAULT: ordered like any copy write
inline constexpr std::uint32_t marker_in = 1;   // after every earlier command has started
inline constexpr std::uint32_t marker_out = 2;  // after every earlier command has finished
} // namespace write_immediate_mode

// One D3D12_WRITEBUFFERIMMEDIATE_PARAMETER: a 32-bit value written at a GPU virtual address.
struct WriteImmediate {
    std::uint64_t address;
    std::uint32_t value;
    std::uint32_t mode;
};

// D3D12_SUBRESOURCE_RANGE_UINT64 of a dependent resource: [begin, end) in bytes of one subresource
// (end == UINT64_MAX is the D3D12 sentinel meaning through the end of the subresource).
struct DependentRange {
    ResourceId resource;
    std::uint32_t subresource;
    std::uint64_t begin;
    std::uint64_t end;
};

// AtomicCopyBufferUINT (bytes 4) and AtomicCopyBufferUINT64 (bytes 8). The copy runs once every earlier write to
// the dependent resources has finished.
struct AtomicCopyBuffer {
    ResourceId dst;
    std::uint64_t dst_offset;
    ResourceId src;
    std::uint64_t src_offset;
    std::uint32_t bytes;
    std::vector<DependentRange> dependencies;
};

// What a UAV descriptor said when ClearUnorderedAccessView* was recorded (the CPU slot may be rewritten after).
// A buffer view is a byte range of the buffer and an element format; a texture view is one mip and a layer range.
struct UavView {
    ResourceId resource;
    bool buffer;
    std::uint32_t format;        // the view's VkFormat value; 0 (VK_FORMAT_UNDEFINED): a raw or structured buffer
    std::uint64_t offset;        // buffer: first byte of the view
    std::uint64_t size;          // buffer: bytes in the view
    std::uint32_t element_size;  // buffer: bytes an element (rects count elements), 4 for a raw view
    std::uint32_t mip;           // texture
    std::uint32_t first_layer;
    std::uint32_t layer_count;
};

// ClearUnorderedAccessViewUint / ...Float; the resource must be in unordered_access state at execution. values are
// the four UINTs, or the four FLOATs' bits. rects empty: the whole view. A buffer view's rects are element ranges
// [left, right) (top and bottom must cover row 0).
struct ClearUnorderedAccess {
    UavView view;
    bool is_float;
    std::array<std::uint32_t, 4> values;
    std::vector<Rect> rects;
};

// D3D12_PREDICATION_OP.
namespace d3d12_predication {
inline constexpr std::uint32_t equal_zero = 0;      // skip the predicated commands when the value is 0
inline constexpr std::uint32_t not_equal_zero = 1;  // skip them when it isn't
} // namespace d3d12_predication

// D3D12 requires the predicate's offset to be a multiple of 8 (the value is 64 bits).
inline constexpr std::uint64_t predication_alignment = 8;

// SetPredication. buffer 0 turns predication off. A command list starts without predication, and it does not carry
// over to the next list.
struct Predication {
    ResourceId buffer;
    std::uint64_t offset;
    std::uint32_t op;  // d3d12_predication
};

// Query heaps are identified like resources, by the COM object's pointer value. A heap's type and size are checked
// at replay, when the heap is known.
using QueryHeapId = std::uintptr_t;

// BeginQuery: occlusion and pipeline statistics only (D3D12 has no Begin for timestamps).
struct BeginQuery {
    QueryHeapId heap;
    std::uint32_t type;
    std::uint32_t index;
};

// EndQuery; for a timestamp, the time at which every earlier command has finished.
struct EndQuery {
    QueryHeapId heap;
    std::uint32_t type;
    std::uint32_t index;
};

// ResolveQueryData: `count` results from `first` on, written to the buffer `dst` at `offset`.
struct ResolveQueries {
    QueryHeapId heap;
    std::uint32_t type;
    std::uint32_t first;
    std::uint32_t count;
    ResourceId dst;
    std::uint64_t offset;
};

// SetMarker, BeginEvent and EndEvent, replayed as VK_EXT_debug_utils labels when the device has them (and skipped
// when it does not). The label is UTF-8; an end has none.
struct DebugMarker {
    enum class Kind : std::uint8_t { set, begin, end };
    Kind kind;
    std::string label;
};

// ExecuteIndirect argument kind (from ID3D12CommandSignature).
enum class IndirectKind : std::uint8_t {
    draw = 0,
    draw_indexed = 1,
    dispatch = 2,
};

struct ExecuteIndirect {
    IndirectKind kind = IndirectKind::draw;
    std::uint32_t stride = 0;
    std::uint32_t max_count = 0;
    ResourceId argument_buffer = 0;
    std::uint64_t argument_offset = 0;
    ResourceId count_buffer = 0;  // 0: fixed max_count; non-zero: vkCmdDraw*IndirectCount
    std::uint64_t count_offset = 0;
    // kind == dispatch: the compute pipeline and root arguments in effect, as Dispatch records them, so the dispatches
    // run that pipeline with those arguments whatever the command buffer had bound. A pipeline of 0 leaves the binding
    // as it is.
    std::uintptr_t pipeline = 0;
    std::array<std::uint32_t, 32> user_data{};
};

// Triangle geometry descriptor for a BLAS build (ID3D12GraphicsCommandList4::BuildRaytracingAccelerationStructure).
struct RtTriangleGeometry {
    std::uint64_t vertex_va = 0;
    std::uint32_t vertex_count = 0;
    std::uint32_t vertex_stride = 0;
    std::uint32_t vertex_format = 0;  // VkFormat (e.g. VK_FORMAT_R32G32B32_SFLOAT)
    std::uint64_t index_va = 0;
    std::uint32_t index_count = 0;
    std::uint32_t index_type = 0;     // 0=NONE, 1=UINT16, 2=UINT32
    std::uint64_t transform_va = 0;
    bool opaque = true;
    std::uint32_t flags = 0;          // D3D12_RAYTRACING_GEOMETRY_FLAGS (bit 0 opaque, bit 1 no duplicate any-hit)
};

// BLAS or TLAS build command (VK_KHR_acceleration_structure).
struct BuildAccelerationStructure {
    bool is_tlas = false;
    bool update = false;
    std::uint32_t build_flags = 0;
    std::uint64_t dst_va = 0;
    std::uint64_t src_va = 0;
    std::uint64_t scratch_va = 0;
    std::uint64_t instances_va = 0;
    std::uint32_t instance_count = 0;
    // D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS: instances_va is an array of instance_count GPU virtual addresses, one
    // per 64-byte D3D12_RAYTRACING_INSTANCE_DESC; otherwise the descs themselves, consecutively.
    bool instance_pointers = false;
    std::vector<RtTriangleGeometry> triangles;
};

// CopyRaytracingAccelerationStructure. `mode` is D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE; replay supports
// CLONE (0) and COMPACT (1) for XDXR structures.
struct CopyAccelerationStructure {
    std::uint64_t dst_va = 0;
    std::uint64_t src_va = 0;
    std::uint32_t mode = 0;
};

// EmitRaytracingAccelerationStructurePostbuildInfo, and each postbuild desc of BuildRaytracingAccelerationStructure:
// one value per source, consecutively from `dest_va`. `type` is D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE;
// replay supports COMPACTED_SIZE (0) and CURRENT_SIZE (3), 8 bytes each.
struct AccelerationStructurePostbuild {
    std::uint64_t dest_va = 0;
    std::uint32_t type = 0;
    std::vector<std::uint64_t> sources;
};

using Command = std::variant<TransitionBarrier, UavBarrier, AliasingBarrier, GlobalBarrier, BufferBarrier,
                             TextureBarrier, ClearRenderTarget, ClearDepthStencil, CopyBuffer, CopyTexture,
                             CopyResource, ResolveSubresource, Predication, ClearUnorderedAccess, DiscardResource,
                             WriteImmediate, AtomicCopyBuffer, BeginQuery, EndQuery, ResolveQueries, DebugMarker,
                             ExecuteIndirect, BuildAccelerationStructure, Dispatch, Draw, CopyAccelerationStructure,
                             AccelerationStructurePostbuild>;

class CommandList {
public:
    // D3D12 split barriers: the begin half only marks the resource unusable until the end half, so the transition
    // is recorded once, at the end. A full barrier (flags 0) is recorded where it is issued.
    bool transition(const TransitionBarrier& barrier, std::uint32_t flags = 0) {
        if (closed_ || !barrier.resource) return false;
        if (flags & ~(barrier_begin_only | barrier_end_only)) return false;
        if (flags == (barrier_begin_only | barrier_end_only)) return false;
        if (flags & barrier_begin_only) return true;
        commands_.push_back(barrier);
        return true;
    }

    bool uav_barrier(const UavBarrier& barrier) { return append(barrier); }
    bool aliasing_barrier(const AliasingBarrier& barrier) { return append(barrier); }

    // Enhanced split barriers follow the same rule: the half with sync_after == split is dropped, and the half with
    // sync_before == split is replayed as a full barrier.
    bool barrier(const GlobalBarrier& barrier) {
        if (!valid_sync(barrier.sync_before, barrier.sync_after)) return false;
        if (barrier.sync_after == d3d12_sync::split) return !closed_;
        return append(barrier);
    }

    bool barrier(const BufferBarrier& barrier) {
        if (!barrier.resource || !valid_sync(barrier.sync_before, barrier.sync_after)) return false;
        if (barrier.sync_after == d3d12_sync::split) return !closed_;
        return append(barrier);
    }

    bool barrier(const TextureBarrier& barrier) {
        if (!barrier.resource || !valid_sync(barrier.sync_before, barrier.sync_after)) return false;
        if (barrier.layout_after == d3d12_layout::undefined && barrier.layout_before != d3d12_layout::undefined)
            return false;
        if (barrier.flags & ~texture_barrier_discard) return false;
        if (barrier.range.mip_count && (!barrier.range.layer_count || !barrier.range.plane_count)) return false;
        if (barrier.sync_after == d3d12_sync::split) return !closed_;
        return append(barrier);
    }

    // Copies. The rules D3D12 states for the call itself are checked here; ranges against the resources are
    // checked at replay, when the resources are known.
    bool copy_buffer(const CopyBuffer& copy) {
        if (!copy.dst || !copy.src) return false;
        if (!copy.size) return !closed_; // Native D3D12 accepts a zero-byte CopyBufferRegion as a no-op.
        return append(copy);
    }

    bool copy_texture(const CopyTexture& copy) {
        const auto alignment = copy.tile_footprint ? 1u : texture_data_placement_alignment;
        if (!valid(copy.dst, alignment) || !valid(copy.src, alignment) || (copy.dst.footprint && copy.src.footprint))
            return false;
        if (copy.has_box && (copy.box.right <= copy.box.left || copy.box.bottom <= copy.box.top ||
                             copy.box.back <= copy.box.front))
            return !closed_;  // D3D12 accepts empty/inverted source boxes; emit no zero-extent Vulkan region.
        return append(copy);
    }

    bool copy_resource(const CopyResource& copy) {
        if (!copy.dst || !copy.src) return false;
        if (copy.dst == copy.src) return !closed_;  // A self-copy is a legal no-op, not a failed command list.
        return append(copy);
    }

    // A resource has one sample count, so it cannot be resolved into itself.
    bool resolve_subresource(const ResolveSubresource& resolve) {
        if (!resolve.dst || !resolve.src || resolve.dst == resolve.src) return false;
        if (const auto& region = resolve.region) {
            if (region->dst_x > INT32_MAX || region->dst_y > INT32_MAX) return false;
            if (region->mode < d3d12_resolve_mode::minimum || region->mode > d3d12_resolve_mode::average) return false;
            if (const auto& r = region->src_rect;
                r && (r->left < 0 || r->top < 0 || r->right <= r->left || r->bottom <= r->top))
                return false;
        }
        return append(resolve);
    }

    // A dispatch with no groups does nothing in D3D12, and is not recorded.
    bool dispatch(const Dispatch& dispatch) {
        if (!dispatch.pipeline) return false;
        if (!dispatch.x || !dispatch.y || !dispatch.z) return !closed_;
        return append(dispatch);
    }

    // A draw with no vertices or instances, or a mesh draw with no thread groups, does nothing in D3D12, and is
    // not recorded.
    bool draw(const Draw& draw) {
        if (!draw.pipeline) return false;
        if (!draw.count || !draw.instances || (draw.mesh && !draw.first)) return !closed_;
        return append(draw);
    }
    bool draw(Draw&& draw) {
        if (!draw.pipeline) return false;
        if (!draw.count || !draw.instances) return !closed_;
        return append(std::move(draw));
    }

    // ClearRenderTargetView; the resource must be in render_target state at execution. No rects: the whole view.
    bool clear_render_target(const TargetView& target, const ClearColor& color, std::span<const Rect> rects = {}) {
        if (!valid(target)) return false;
        return append(ClearRenderTarget{target, color, {rects.begin(), rects.end()}});
    }

    // ClearDepthStencilView; the cleared planes must be in depth_write state at execution.
    // D3D12 requires depth in [0, 1] and at least one clear flag.
    bool clear_depth_stencil(const TargetView& target, std::uint32_t flags, float depth, std::uint8_t stencil,
                             std::span<const Rect> rects = {}) {
        if (!valid(target)) return false;
        if (!flags || (flags & ~(clear_depth | clear_stencil))) return false;
        if ((flags & clear_depth) && !(depth >= 0.0f && depth <= 1.0f)) return false;
        return append(ClearDepthStencil{target, flags, depth, stencil, {rects.begin(), rects.end()}});
    }

    // DiscardResource: a count of 0 discards nothing, as in D3D12. An explicit range must end before
    // all_subresources, the largest index plus one; whether it fits the resource is checked at replay.
    bool discard(const DiscardResource& discard) {
        if (!discard.resource) return false;
        if (discard.whole) return append(DiscardResource{discard.resource, 0, 0, discard.partial, true});
        if (discard.count == 0) return !closed_;
        if (std::uint64_t(discard.first_subresource) + discard.count > all_subresources) return false;
        return append(discard);
    }

    // WriteBufferImmediate: D3D12 requires a 4-byte aligned, non-null address and one of the three modes.
    bool write_immediate(const WriteImmediate& write) {
        if (!write.address || write.address % 4 || write.mode > write_immediate_mode::marker_out) return false;
        return append(write);
    }

    // AtomicCopyBufferUINT/UINT64: both offsets aligned to the copy's size and each dependency on a named resource with
    // a non-empty, non-inverted range (zero dependencies is accepted by the Windows 11 WARP and RTX 5090 D3D12
    // runtimes). A copy of a value onto itself (vkCmdCopyBuffer forbids the overlap) is refused too.
    bool atomic_copy(const AtomicCopyBuffer& copy) {
        if ((copy.bytes != 4 && copy.bytes != 8) || !copy.dst || !copy.src) return false;
        if (copy.dst == copy.src && copy.dst_offset == copy.src_offset) return false;
        if (copy.dst_offset % copy.bytes || copy.src_offset % copy.bytes) return false;
        for (const auto& d : copy.dependencies)
            if (!d.resource || d.begin >= d.end) return false;
        return append(copy);
    }

    // A view of nothing (no resource, an empty buffer range or no layers) is invalid, as is a rectangle that is
    // inside out. Ranges against the resource are checked at replay.
    bool clear_unordered_access(const UavView& view, bool is_float, const std::array<std::uint32_t, 4>& values,
                                std::span<const Rect> rects = {}) {
        if (!view.resource) return false;
        if (view.buffer ? (!view.size || !view.element_size) : !view.layer_count) return false;
        for (const Rect& r : rects)
            if (r.right < r.left || r.bottom < r.top) return false;
        return append(ClearUnorderedAccess{view, is_float, values, {rects.begin(), rects.end()}});
    }

    // ExecuteBundle copies the closed bundle's commands [first, last) at the call site. The COM layer validates D3D12
    // list types and bundle-only restrictions before calling this method, and calls it once per stretch between the
    // bundle's deferred state calls and draws (d3d12_com.cpp, GraphicsCommandList::ExecuteBundle).
    bool execute_bundle(const CommandList& bundle, std::size_t first = 0, std::size_t last = std::size_t(-1)) {
        if (this == &bundle || closed_ || !bundle.closed_) return false;
        last = std::min(last, bundle.commands_.size());
        if (first < last)
            commands_.insert(commands_.end(), bundle.commands_.begin() + std::ptrdiff_t(first),
                             bundle.commands_.begin() + std::ptrdiff_t(last));
        return true;
    }

    // SetPredication. Turning it off ignores the offset and op, as D3D12 does.
    bool set_predication(const Predication& predication) {
        if (!predication.buffer) return append(Predication{0, 0, 0});
        if (predication.op != d3d12_predication::equal_zero && predication.op != d3d12_predication::not_equal_zero)
            return false;
        if (predication.offset % predication_alignment) return false;
        return append(predication);
    }

    // Queries. D3D12 rules for the calls themselves: timestamps have only an EndQuery; any other query is begun and
    // ended in the same list, with the same type, and is not begun again while it is running.
    bool begin_query(const BeginQuery& query) {
        if (closed_ || !query.heap || query.type == d3d12_query::timestamp) return false;
        if (find_active(query.heap, query.index) != active_.end()) return false;
        active_.push_back(query);
        return append(query);
    }

    bool end_query(const EndQuery& query) {
        if (closed_ || !query.heap) return false;
        if (query.type != d3d12_query::timestamp) {
            const auto it = find_active(query.heap, query.index);
            if (it == active_.end() || it->type != query.type) return false;
            active_.erase(it);
        }
        return append(query);
    }

    // The offset is a multiple of 8, as D3D12 requires. A resolve of no queries records nothing.
    bool resolve_queries(const ResolveQueries& resolve) {
        if (closed_ || !resolve.heap || !resolve.dst || resolve.offset % 8) return false;
        if (resolve.count > ~std::uint32_t(0) - resolve.first) return false;
        for (const BeginQuery& running : active_)
            if (running.heap == resolve.heap && running.index - resolve.first < resolve.count) return false;
        return resolve.count ? append(resolve) : true;
    }

    // A query begun and not ended. Such a list still closes (D3D12's Close reports the error), but is not replayed.
    bool queries_open() const { return !active_.empty(); }

    // Debug markers. D3D12 does not ask a list to balance its events (one list may end what another began), so
    // they are recorded as given; the replay balances them per submission (vulkan_replay.h, emit).
    bool marker(DebugMarker::Kind kind, std::string label = {}) {
        if (kind == DebugMarker::Kind::end) label.clear();
        return append(DebugMarker{kind, std::move(label)});
    }

    bool execute_indirect(const ExecuteIndirect& indirect) {
        if (!indirect.argument_buffer || !indirect.stride || indirect.stride % 4) return false;
        if (indirect.argument_offset % 4) return false;
        if (!indirect.count_buffer && indirect.count_offset) return false;
        if (indirect.count_buffer && indirect.count_offset % 4) return false;
        if (!indirect.max_count) return !closed_;  // the PC runtime takes MaxCommandCount 0 as a no-op
        return append(indirect);
    }

    // D3D12: acceleration structures and build scratch are 256-byte aligned; postbuild destinations 8-byte aligned.
    bool build_acceleration_structure(BuildAccelerationStructure build) {
        if (!build.dst_va || build.dst_va % 256 || !build.scratch_va || build.scratch_va % 256) return false;
        if (build.update && (!build.src_va || build.src_va % 256)) return false;
        return append(std::move(build));
    }
    bool copy_acceleration_structure(CopyAccelerationStructure copy) {
        if (!copy.dst_va || !copy.src_va || copy.dst_va % 256 || copy.src_va % 256 || copy.mode > 4) return false;
        return append(copy);
    }
    bool acceleration_structure_postbuild(AccelerationStructurePostbuild info) {
        if (!info.dest_va || info.dest_va % 8 || info.type > 3 || info.sources.empty()) return false;
        for (const auto va : info.sources)
            if (!va || va % 256) return false;
        return append(std::move(info));
    }
    bool close() {
        if (closed_) return false;
        closed_ = true;
        return true;
    }

    bool reset() {
        if (!closed_) return false;
        commands_.clear();
        active_.clear();
        closed_ = false;
        return true;
    }

    bool closed() const { return closed_; }
    std::span<const Command> commands() const { return commands_; }

private:
    static bool valid(const TargetView& target) { return target.resource && target.layer_count; }

    // A placed footprint needs aligned offsets and pitches and a non-empty size.
    static bool valid(const CopyLocation& location, std::uint32_t alignment) {
        if (!location.resource) return false;
        if (!location.footprint) return true;
        return location.width && location.height && location.depth && location.row_pitch &&
               location.offset % alignment == 0 &&
               location.row_pitch % texture_data_pitch_alignment == 0;
    }

    // A split begin and end cannot be the same barrier.
    static bool valid_sync(std::uint32_t before, std::uint32_t after) {
        return !(before == d3d12_sync::split && after == d3d12_sync::split);
    }

    bool append(const Command& command) {
        if (closed_) return false;
        commands_.push_back(command);
        return true;
    }
    // The recording calls pass a temporary: moved, so a command's rectangles are not copied a second time.
    bool append(Command&& command) {
        if (closed_) return false;
        commands_.push_back(std::move(command));
        return true;
    }

    std::vector<BeginQuery>::iterator find_active(QueryHeapId heap, std::uint32_t index) {
        return std::find_if(active_.begin(), active_.end(),
                            [&](const BeginQuery& q) { return q.heap == heap && q.index == index; });
    }

    std::vector<Command> commands_;
    std::vector<BeginQuery> active_;  // queries begun and not yet ended
    bool closed_ = false;
};

} // namespace leonida
