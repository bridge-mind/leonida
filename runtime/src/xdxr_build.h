// XDXR BVH4 acceleration-structure builds on the GPU (xdxr_build.comp): what BuildRaytracingAccelerationStructure
// has to produce for the game's translated traversal shaders, which read the console's node records themselves
// (docs/d3d12_vulkan.md 4.8, docs/bvh.md 7a). tools/xdxr_bvh.py is the CPU reference.
//
// XdxrLayout gives the byte layout of a structure from its element count alone: the same numbers answer
// GetRaytracingAccelerationStructurePrebuildInfo and place the records when the build is recorded. The region
// order (header, parent table, elements, nodes) is the one tools/xdxr_bvh.py writes (an ASSUMPTION there: a shader
// reaches each region through the header). Nodes are BOX32 (128 bytes, float boxes) or, when the layout asks for
// them, BOX16 (64 bytes, half boxes rounded outward): a binary radix tree over the elements' Morton keys, collapsed
// to four children per node by keeping every other level; the root is node 0. Their count depends on the keys, so
// the layout has room for the most there can be and the header counts the ones a build wrote.
//
// XdxrBuilder records the dispatches into a command buffer. Every address is a buffer device address; `bias` is
// added to the BLAS addresses read from instance descs (the distance between the application's GPU virtual
// addresses and device addresses; 0 where they are the same). An INSTANCE record holds its BLAS address in 256-byte
// units in 32 bits, so the application's BLAS addresses must be below 2^40, as console GPU addresses are; where
// device addresses are wider (llvmpipe's are host pointers), the layer must present a narrower window and a bias.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "xdxr_build_spv.h"

namespace leonida {

struct XdxrLayout {
    static constexpr std::uint32_t header_size = 64;

    bool tlas = false;
    bool box16 = false;                // inner nodes are BOX16
    std::uint32_t node_size = 128;     // BOX32 128 bytes, BOX16 64
    std::uint32_t elements = 0;        // triangles (BLAS) or instances (TLAS)
    std::uint32_t element_size = 64;   // a leaf, or a 128-byte instance record
    std::uint32_t nodes = 0;           // the most BOX32 nodes a build can write
    std::uint32_t parents_off = header_size;
    std::uint32_t elements_off = 0;
    std::uint32_t nodes_off = 0;
    std::uint32_t size = 0;            // result data size in bytes, a multiple of 64
    std::uint32_t parent_entries = 0;  // one per 64-byte unit from elements_off on
    std::uint32_t sort_size = 0;       // keys the radix sort orders: one per element
    static constexpr std::uint32_t radix_tile = 256;  // xdxr_build.comp RADIX_TILE
    static constexpr std::uint32_t radix_bins = 16;
    static constexpr std::uint32_t radix_passes = 8;
    static constexpr std::uint32_t small_sort_max = 1024;  // xdxr_build.comp SMALL_SORT_MAX: one-workgroup sort
    std::uint64_t scratch = 0;         // scratch bytes

    // False when the count would overflow the 32-bit offsets of the format. `half_boxes`: BOX16 inner nodes.
    bool compute(std::uint32_t count, bool top_level, bool half_boxes = false) {
        *this = XdxrLayout{};
        tlas = top_level;
        box16 = half_boxes;
        node_size = box16 ? 64u : 128u;
        elements = count;
        element_size = tlas ? 128u : 64u;
        // A kept inner node has two children when both of its binary children are elements, which takes two
        // elements of its own, and three or four otherwise. With K nodes and a of the first kind, the child slots
        // count K - 1 + N >= 2a + 3(K - a), and a <= N / 2, so K <= (3N - 2) / 4. An empty structure or a single
        // element keeps one root.
        nodes = count <= 1 ? 1u : std::uint32_t((3ull * count - 2) / 4);
        // BOX16: one spare 64-byte unit after the last node a build writes (xdxr_build.comp MODE_HEADER zeroes it
        // and the header's size counts it). The translator's image_bvh_intersect_ray loads a box node's 112 bytes of
        // BOX32 bounds before it knows the type, so without it the last BOX16 node (64 bytes) would be read 48 bytes
        // past the structure, a compacted one too.
        const std::uint64_t pad = box16 ? 64u : 0u;
        const std::uint64_t body_units =
            (std::uint64_t(count) * element_size + std::uint64_t(nodes) * node_size + pad) / 64;
        const std::uint64_t parent_bytes = (4 * body_units + 63) / 64 * 64;
        const std::uint64_t e_off = header_size + parent_bytes;
        const std::uint64_t n_off = e_off + std::uint64_t(count) * element_size;
        const std::uint64_t end = n_off + std::uint64_t(nodes) * node_size + pad;
        // node pointers hold the 64-byte unit index in 29 bits
        if (end > (std::uint64_t(1) << 32) - 64 || (end >> 6) >= (std::uint64_t(1) << 29)) return false;
        elements_off = std::uint32_t(e_off);
        nodes_off = std::uint32_t(n_off);
        size = std::uint32_t(end);
        parent_entries = std::uint32_t(body_units);
        sort_size = count;
        // centroid bounds and the node count, keys and element indices twice, digit counts per tile and their prefix
        // sums, a box per element, a box accumulator per node, then per inner node of the binary tree its two
        // children, its parent, its kept flag and its node index, a kept count per tile with its prefix sum, and
        // each element's pointer in the tree
        scratch = 64 + 16ull * sort_size + 8ull * radix_bins * tiles() + 24ull * count + 32ull * nodes +
                  20ull * inner() + 8ull * kept_tiles() + 4ull * count;
        return true;
    }
    std::uint32_t tiles() const { return (sort_size + radix_tile - 1) / radix_tile; }
    std::uint32_t inner() const { return elements > 1 ? elements - 1 : 0; }   // binary tree inner nodes
    std::uint32_t kept_tiles() const { return (inner() + radix_tile - 1) / radix_tile; }
};

// One geometry of a bottom-level build, as D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC gives it.
struct XdxrTriangles {
    std::uint64_t vertices = 0;       // device address
    std::uint32_t vertex_stride = 12;
    bool xy_only = false;             // DXGI_FORMAT_R32G32_FLOAT (z = 0); otherwise R32G32B32_FLOAT
    std::uint64_t indices = 0;        // device address, 0 = none
    std::uint32_t index_type = 0;     // 0 none, 1 R16_UINT, 2 R32_UINT
    std::uint32_t triangle_count = 0;
    std::uint64_t transform = 0;      // device address of a 3x4 row-major float matrix, 0 = none
    std::uint32_t flags = 0;          // D3D12_RAYTRACING_GEOMETRY_FLAGS (bit 0 opaque)
};

class XdxrBuilder {
public:
    XdxrBuilder(PFN_vkGetDeviceProcAddr get, VkDevice device) : device_(device) {
        loaded_ = load(get) && make_pipeline();
    }
    ~XdxrBuilder() {
        if (pipeline_) vk_.vkDestroyPipeline(device_, pipeline_, nullptr);
        if (layout_) vk_.vkDestroyPipelineLayout(device_, layout_, nullptr);
    }
    XdxrBuilder(const XdxrBuilder&) = delete;
    XdxrBuilder& operator=(const XdxrBuilder&) = delete;

    bool loaded() const { return loaded_; }

    // Records a bottom-level build into `dst` (layout.size bytes) using `scratch` (layout.scratch bytes). `layout`
    // must come from compute() with the sum of the geometries' triangle counts.
    //
    // With `update_from` set, an update (D3D12 PERFORM_UPDATE) instead: the structure at `update_from` (the
    // destination itself, or another one, which is copied first) was built from the same topology, and is refitted
    // to the new positions. Its tree is kept and only its boxes change, which is valid for any motion and costs no
    // sort, though a refitted tree traverses more slowly as the geometry moves away from where it was built.
    //
    // `pair`: sibling triangles that share an edge go into one leaf as TRI0 and TRI1 (xdxr_build.comp MODE_NODES).
    // The layout and the tree are the same either way. A refit rewrites every leaf as one triangle, so a structure
    // that will be updated must be built without pairing (the layer pairs only builds without ALLOW_UPDATE), and an
    // update ignores `pair`.
    bool record_blas(VkCommandBuffer cb, const XdxrLayout& layout, const std::vector<XdxrTriangles>& geometries,
                     std::uint64_t dst, std::uint64_t scratch, std::uint64_t update_from = 0, bool pair = false) {
        if (!loaded_ || layout.tlas) return false;
        Push p = base(layout, dst, scratch);
        if (pair && !update_from) p.flags |= flag_pair;
        if (update_from) refit_begin(cb, layout, p, update_from); else begin(cb, layout, p);
        std::uint32_t first = 0;
        for (std::size_t g = 0; g < geometries.size(); ++g) {
            const auto& geo = geometries[g];
            if (!geo.triangle_count) continue;
            Push t = p;
            t.mode = mode_triangles;
            t.count = geo.triangle_count;
            t.first = first;
            t.src0 = split(geo.vertices);
            t.src1 = split(geo.indices);
            t.src2 = split(geo.transform);
            t.arg0 = geo.vertex_stride;
            t.arg1 = geo.xy_only ? 1u : 0u;
            t.arg2 = geo.index_type;
            t.geometry = std::uint32_t(g) << 8 | (geo.flags & 0xffu);
            dispatch(cb, t, t.count);
            first += geo.triangle_count;
        }
        if (first != layout.elements) return false;
        if (update_from) refit_finish(cb, layout, p); else finish(cb, layout, p);
        return true;
    }

    // Records a top-level build from `count` D3D12_RAYTRACING_INSTANCE_DESCs at `instances`.
    // `update_from`: as for record_blas; the instances may move, and name other BLASes.
    bool record_tlas(VkCommandBuffer cb, const XdxrLayout& layout, std::uint64_t instances, std::uint64_t dst,
                     std::uint64_t scratch, std::uint64_t bias = 0, std::uint64_t update_from = 0) {
        if (!loaded_ || !layout.tlas) return false;
        Push p = base(layout, dst, scratch);
        p.bias = split(bias);
        if (update_from) refit_begin(cb, layout, p, update_from); else begin(cb, layout, p);
        if (layout.elements) {
            Push t = p;
            t.mode = mode_instances;
            t.count = layout.elements;
            t.src0 = split(instances);
            dispatch(cb, t, t.count);
        }
        if (update_from) refit_finish(cb, layout, p); else finish(cb, layout, p);
        return true;
    }

    // CopyRaytracingAccelerationStructure CLONE or COMPACT: the structure at `src` to `dst`, as many bytes as its
    // header says, at most `max_bytes` (the room the destination has). The structures are position-independent.
    bool record_copy(VkCommandBuffer cb, std::uint64_t src, std::uint64_t dst, std::uint64_t max_bytes) {
        if (!loaded_ || !src || !dst || max_bytes < XdxrLayout::header_size) return false;
        if (src == dst) return true;
        vk_.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        Push p{};
        p.mode = mode_copy_as;
        p.src0 = split(src);
        p.dst = split(dst);
        p.count = std::uint32_t(std::min<std::uint64_t>(max_bytes, 0xffffffffull) / 4);
        dispatch(cb, p, p.count, false);
        return true;
    }

    // One postbuild value (COMPACTED_SIZE or CURRENT_SIZE, the same for these builds): 8 bytes at `dst` for the
    // structure at `src`. A barrier first: a build or copy recorded just before writes the header it reads.
    bool record_postbuild(VkCommandBuffer cb, std::uint64_t src, std::uint64_t dst) {
        if (!loaded_ || !src || !dst) return false;
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vk_.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                 &b, 0, nullptr, 0, nullptr);
        vk_.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        Push p{};
        p.mode = mode_postbuild;
        p.src0 = split(src);
        p.dst = split(dst);
        p.count = 1;
        dispatch(cb, p, 1, false);
        return true;
    }

private:
    enum : std::uint32_t {
        mode_init = 0, mode_triangles = 1, mode_instances = 2, mode_morton = 3, mode_karras = 4, mode_nodes = 5,
        mode_header = 6, mode_radix_count = 7, mode_radix_scan = 8, mode_radix_scatter = 9, mode_small_sort = 10,
        mode_copy_as = 12, mode_postbuild = 13, mode_kept = 14, mode_kept_scan = 15,
        mode_kept_index = 16, mode_propagate = 17, mode_clear = 18
    };
    struct Addr { std::uint32_t lo = 0, hi = 0; };
    struct Push {  // xdxr_build.comp `Push`, std430 push-constant layout
        Addr dst, scratch, src0, src1, src2, bias;
        std::uint32_t mode = 0, count = 0, first = 0, arg0 = 0, arg1 = 0, arg2 = 0, geometry = 0, elements = 0,
                      sort_size = 0, elements_off = 0, nodes_off = 0, parents_off = 0, total_size = 0, flags = 0;
    };
    enum : std::uint32_t { flag_tlas = 1, flag_box16 = 2, flag_pair = 4 };  // xdxr_build.comp FLAG_*
    static_assert(sizeof(Push) == 104, "push constants must match xdxr_build.comp");

    static Addr split(std::uint64_t a) { return {std::uint32_t(a), std::uint32_t(a >> 32)}; }

    static Push base(const XdxrLayout& l, std::uint64_t dst, std::uint64_t scratch) {
        Push p{};
        p.dst = split(dst);
        p.scratch = split(scratch);
        p.elements = l.elements;
        p.sort_size = l.sort_size;
        p.elements_off = l.elements_off;
        p.nodes_off = l.nodes_off;
        p.parents_off = l.parents_off;
        p.total_size = l.size;
        p.flags = (l.tlas ? flag_tlas : 0u) | (l.box16 ? flag_box16 : 0u);
        p.arg0 = l.nodes;                               // where the scratch arrays after the accumulators start
        return p;
    }

    void begin(VkCommandBuffer cb, const XdxrLayout& l, const Push& p) {
        vk_.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        Push init = p;
        init.mode = mode_init;
        init.count = l.parent_entries > 16 ? l.parent_entries : 16;   // the parent table has at least 16 entries
        dispatch(cb, init, init.count);
    }

    // An update: the source's tree is copied unless it is the destination, as many bytes as its header says (a
    // compacted source is shorter than the layout). Its parent table stays as it is, so there is no MODE_INIT.
    void refit_begin(VkCommandBuffer cb, const XdxrLayout& l, const Push& p, std::uint64_t source) {
        vk_.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        const std::uint64_t dst = std::uint64_t(p.dst.hi) << 32 | p.dst.lo;
        if (source == dst) return;
        Push c = p;
        c.mode = mode_copy_as;
        c.count = l.size / 4;
        c.src0 = split(source);
        dispatch(cb, c, c.count);
    }

    // Boxes bottom-up through the existing nodes, then the header (its bounds change; nothing else does).
    void refit_finish(VkCommandBuffer cb, const XdxrLayout& l, const Push& p) {
        Push c = p;
        c.mode = mode_clear;
        c.count = l.nodes;
        dispatch(cb, c, c.count);
        propagate(cb, l, p, true);
        Push h = p;
        h.mode = mode_header;
        h.count = 1;
        h.arg2 = 1;                                     // an update: the bounds only
        dispatch(cb, h, 1, false);
    }

    // `update`: the elements' pointers are their own (an update never pairs), not the ones MODE_NODES recorded.
    void propagate(VkCommandBuffer cb, const XdxrLayout& l, const Push& p, bool update = false) {
        if (!l.elements) return;
        Push b = p;
        b.mode = mode_propagate;
        b.count = l.elements;
        b.arg2 = update ? 1u : 0u;
        dispatch(cb, b, b.count);
    }

    void finish(VkCommandBuffer cb, const XdxrLayout& l, const Push& p) {
        Push m = p;
        m.mode = mode_morton;
        m.count = l.sort_size;
        dispatch(cb, m, m.count);
        // stable LSD radix sort, 4 bits a pass; an even pass count leaves the result where the keys started
        static_assert(XdxrLayout::radix_passes % 2 == 0, "the sorted keys must end in the first buffer");
        if (l.sort_size > 1 && l.sort_size <= XdxrLayout::small_sort_max) {
            Push s = p;
            s.mode = mode_small_sort;
            dispatch(cb, s, 64);                       // one workgroup, shared memory
        } else if (l.sort_size > 1) {
            for (std::uint32_t pass = 0; pass < XdxrLayout::radix_passes; ++pass) {
                Push s = p;
                s.first = pass;
                s.mode = mode_radix_count;
                dispatch(cb, s, 64 * l.tiles());       // one workgroup of 64 per tile
                s.mode = mode_radix_scan;
                dispatch(cb, s, 64);                   // one workgroup
                s.mode = mode_radix_scatter;
                dispatch(cb, s, 64 * l.tiles());
            }
        }
        // the binary radix tree, which of its inner nodes become BOX32 nodes, and their indices
        if (l.inner()) {
            Push t = p;
            t.mode = mode_karras;
            t.count = l.inner();
            dispatch(cb, t, t.count);
            t.mode = mode_kept;
            dispatch(cb, t, 64 * l.kept_tiles());      // one workgroup of 64 per tile
            t.mode = mode_kept_scan;
            dispatch(cb, t, 64);                       // one workgroup
            t.mode = mode_kept_index;
            dispatch(cb, t, 64 * l.kept_tiles());
        }
        Push n = p;
        n.mode = mode_nodes;
        n.count = l.inner() ? l.inner() : 1;          // one root of its own for zero or one element
        dispatch(cb, n, n.count);
        propagate(cb, l, p);
        Push h = p;
        h.mode = mode_header;
        h.count = 1;
        dispatch(cb, h, 1, false);
    }

    // One dispatch, then a compute-to-compute dependency so the next step sees its writes. The last step leaves the
    // dependency on its results to the caller's barrier (the application's UAV or AS barrier).
    void dispatch(VkCommandBuffer cb, const Push& p, std::uint32_t threads, bool barrier = true) {
        vk_.vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &p);
        vk_.vkCmdDispatch(cb, (threads + 63) / 64, 1, 1);
        if (!barrier) return;
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vk_.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                 &b, 0, nullptr, 0, nullptr);
    }

    struct Functions {
#define LEONIDA_XDXR_FUNCTIONS(X)                                                                              \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout)      \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCmdBindPipeline) X(vkCmdPushConstants)                \
    X(vkCmdDispatch) X(vkCmdPipelineBarrier)
#define LEONIDA_XDXR_MEMBER(name) PFN_##name name = nullptr;
        LEONIDA_XDXR_FUNCTIONS(LEONIDA_XDXR_MEMBER)
#undef LEONIDA_XDXR_MEMBER
    };

    bool load(PFN_vkGetDeviceProcAddr get) {
        if (!get || !device_) return false;
        bool all = true;
#define LEONIDA_XDXR_LOAD(name)                                     \
    vk_.name = reinterpret_cast<PFN_##name>(get(device_, #name));   \
    all = all && vk_.name != nullptr;
        LEONIDA_XDXR_FUNCTIONS(LEONIDA_XDXR_LOAD)
#undef LEONIDA_XDXR_LOAD
#undef LEONIDA_XDXR_FUNCTIONS
        return all;
    }

    bool make_pipeline() {
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        if (vk_.vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS) return false;
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = sizeof(xdxr_build_spirv);
        module_info.pCode = xdxr_build_spirv;
        VkShaderModule module = VK_NULL_HANDLE;
        if (vk_.vkCreateShaderModule(device_, &module_info, nullptr, &module) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                      module, "main", nullptr};
        info.layout = layout_;
        const bool ok = vk_.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline_) == VK_SUCCESS;
        vk_.vkDestroyShaderModule(device_, module, nullptr);
        return ok;
    }

    VkDevice device_ = VK_NULL_HANDLE;
    Functions vk_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    bool loaded_ = false;
};

}  // namespace leonida
