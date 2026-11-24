// The XDXR BVH4 GPU builder (src/xdxr_build.comp, xdxr_build.h) on synthetic scenes. Each case is built on the
// device the Vulkan loader picks and written to <out dir>/<case>.bin, the acceleration structure as the GPU left it,
// with <case>.json describing the scene: its address and the triangles (or instances) it was built from.
// `python -I tools/xdxr_bvh.py check-build <out dir>` then validates every structure and traces rays through it
// against brute force (ctest runs both; see CMakeLists.txt).
//
// Cases: one triangle; five (two levels); 1000 random triangles; an indexed R16 grid with a 16-byte vertex stride;
// R32 indices with R32G32 vertices; a geometry transform; two geometries with an inactive (NaN) triangle; no
// triangles; 5000 triangles (many radix-sort tiles); updates (refits) in place and into another buffer; clones and a
// compaction (CopyRaytracingAccelerationStructure) with the postbuild size; a TLAS of four instances (rotated and scaled, one with mask 0, one with no BLAS);
// BOX16 builds (blas_box16_*: half boxes, which check-build compares bit for bit with tools/xdxr_bvh.py encode_box16),
// including coordinates past the half range and below its smallest normal, and a refit; TRI1 pairing (blas_paired_*:
// sibling triangles that share an edge in one leaf), on grids, BOX16, a transform, inactive triangles, triangles that
// share an edge in the same direction or across geometries (never paired), and the radix-sort path.
// Exit 77 when there is no Vulkan device with bufferDeviceAddress.
// `vk_xdxr_build_test <out dir> <triangles>` instead times BLAS builds of that many random triangles (wall clock of
// one submission each, after one warm-up; result and scratch in device-local memory), then in-place updates
// (refits) of the same structure, and writes nothing.
#include "vk_test_device.h"
#include "xdxr_build_output.h"
#include "../src/xdxr_build.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

struct Tri { float v[3][3]; std::uint32_t geometry, primitive; };

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    void* map = nullptr;
    std::uint64_t address = 0;
};

Buffer make(vktest::Device& gpu, std::size_t bytes, bool device_local = false) {
    Buffer b;
    b.buffer = gpu.buffer(bytes ? bytes : 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, device_local ? nullptr : &b.map);
    if (b.buffer) b.address = gpu.device_address(b.buffer);
    return b;
}

void barrier_to_host(vktest::Device& gpu, VkCommandBuffer cb) {
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    gpu.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                            VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0,
                                                            nullptr);
}

std::string num(double x) {
    if (std::isnan(x)) return "\"nan\"";
    char s[64];
    std::snprintf(s, sizeof s, "%.9g", x);
    return s;
}

bool write_file(const std::string& path, const void* data, std::size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char*>(data), std::streamsize(bytes));
    return bool(f);
}

// The size a structure's header gives (totalSizeIn64BNodes does not count the header): the nodes the build wrote,
// at most the layout's size, which leaves room for the most a tree of that many elements can have.
std::size_t header_size(const void* structure) {
    std::uint32_t units = 0;
    std::memcpy(&units, static_cast<const char*>(structure) + 4, 4);
    return (std::size_t(units) + 1) * 64;
}

std::string tris_json(const std::vector<Tri>& tris) {
    std::string s = "[";
    for (std::size_t i = 0; i < tris.size(); ++i) {
        const Tri& t = tris[i];
        s += i ? ",\n  [" : "\n  [";
        for (int k = 0; k < 3; ++k)
            s += (k ? ", [" : "[") + num(t.v[k][0]) + ", " + num(t.v[k][1]) + ", " + num(t.v[k][2]) + "]";
        s += ", " + std::to_string(t.geometry) + ", " + std::to_string(t.primitive) + "]";
    }
    return s + "]";
}

struct Geometry {
    std::vector<float> vertex_data;          // as the buffer holds it (stride / 4 floats per vertex)
    std::uint32_t stride = 12;
    bool xy = false;
    std::vector<std::uint32_t> indices;       // empty = none
    bool u16 = false;
    bool transform = false;
    float m[12] = {};
    std::uint32_t triangle_count = 0;
};

// The world-space triangles a geometry describes, as the reference must see them.
void expand(const Geometry& g, std::uint32_t geometry, std::vector<Tri>& out) {
    const std::uint32_t floats = g.stride / 4;
    for (std::uint32_t t = 0; t < g.triangle_count; ++t) {
        Tri tri{};
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t v = g.indices.empty() ? 3 * t + k : g.indices[3 * t + k];
            float p[3] = {g.vertex_data[v * floats], g.vertex_data[v * floats + 1],
                          g.xy ? 0.0f : g.vertex_data[v * floats + 2]};
            for (int c = 0; c < 3; ++c)
                tri.v[k][c] = g.transform ? g.m[4 * c] * p[0] + g.m[4 * c + 1] * p[1] + g.m[4 * c + 2] * p[2] + g.m[4 * c + 3]
                                          : p[c];
        }
        tri.geometry = geometry;
        tri.primitive = t;
        out.push_back(tri);
    }
}

struct Built {
    std::uint64_t address = 0;
    std::size_t size = 0;              // as the header gives it
    std::vector<Tri> tris;
};

// With `moved` (the same geometries with new vertex positions) the BLAS is then updated: refitted in place, or with
// `into_other` into a second buffer, in which case the untouched source is written too, as <name>_source.
// `box16`: BOX16 inner nodes. `pair`: TRI1 pairing; the case's JSON then asks check-build for at least one TRI1
// node, or with `expect_no_pairs` for none.
int build_blas(vktest::Device& gpu, XdxrBuilder& builder, const std::string& dir, const std::string& name,
               const std::vector<Geometry>& geometries, Built& built, unsigned rays = 0,
               const std::vector<Geometry>* moved = nullptr, bool into_other = false, bool box16 = false,
               bool pair = false, bool expect_no_pairs = false) {
    std::vector<Buffer> keep, vertex_buffers;
    std::vector<XdxrTriangles> descs;
    std::uint32_t total = 0;
    for (std::size_t g = 0; g < geometries.size(); ++g) {
        const Geometry& geo = geometries[g];
        Buffer v = make(gpu, geo.vertex_data.size() * 4);
        EXPECT(v.buffer);
        std::memcpy(v.map, geo.vertex_data.data(), geo.vertex_data.size() * 4);
        XdxrTriangles d;
        d.vertices = v.address;
        d.vertex_stride = geo.stride;
        d.xy_only = geo.xy;
        d.triangle_count = geo.triangle_count;
        d.flags = 1;
        keep.push_back(v);
        vertex_buffers.push_back(v);
        if (!geo.indices.empty()) {
            Buffer ib = make(gpu, geo.indices.size() * 4);
            EXPECT(ib.buffer);
            if (geo.u16) {
                auto* p = static_cast<std::uint16_t*>(ib.map);
                for (std::size_t i = 0; i < geo.indices.size(); ++i) p[i] = std::uint16_t(geo.indices[i]);
            } else {
                std::memcpy(ib.map, geo.indices.data(), geo.indices.size() * 4);
            }
            d.indices = ib.address;
            d.index_type = geo.u16 ? 1 : 2;
            keep.push_back(ib);
        }
        if (geo.transform) {
            Buffer tb = make(gpu, 48);
            EXPECT(tb.buffer);
            std::memcpy(tb.map, geo.m, 48);
            d.transform = tb.address;
            keep.push_back(tb);
        }
        descs.push_back(d);
        expand(geo, std::uint32_t(g), built.tris);
        total += geo.triangle_count;
    }
    XdxrLayout layout;
    EXPECT(layout.compute(total, false, box16));
    Buffer dst = make(gpu, layout.size + 256), scratch = make(gpu, std::size_t(layout.scratch));
    EXPECT(dst.buffer && scratch.buffer);
    const std::uint64_t base = (dst.address + 255) & ~std::uint64_t(255);   // 256-byte aligned, as D3D12 requires
    bool recorded = false;
    EXPECT(gpu.run([&](VkCommandBuffer cb) {
        recorded = builder.record_blas(cb, layout, descs, base, scratch.address, 0, pair);
        barrier_to_host(gpu, cb);
        return recorded;
    }));
    built.address = base;
    built.size = header_size(static_cast<const char*>(dst.map) + (base - dst.address));
    auto dump = [&](const std::string& case_name, const Buffer& buffer, std::uint64_t at, const std::vector<Tri>& tris) {
        const auto* bytes = static_cast<const char*>(buffer.map) + (at - buffer.address);
        const std::string json = "{\"kind\": \"blas\", \"address\": " + std::to_string(at) +
                                 (rays ? ", \"rays\": " + std::to_string(rays) : std::string()) +
                                 (pair && !moved ? (expect_no_pairs ? ", \"pairs\": false" : ", \"pairs\": true")
                                                 : "") +
                                 (box16 ? ", \"box16\": true" : ", \"box16\": false") +
                                 (name == "blas_paired_rotations" ? ", \"all_rotations\": true" : "") +
                                 ", \"triangles\": " + tris_json(tris) + "}\n";
        const std::size_t size = header_size(bytes);
        if (size > layout.size) return false;
        return write_file(dir + "/" + case_name + ".bin", bytes, size) &&
               write_file(dir + "/" + case_name + ".json", json.data(), json.size());
    };
    if (!moved) {
        EXPECT(dump(name, dst, base, built.tris));
        std::printf("%s: %u triangles, %u bytes, %u nodes\n", name.c_str(), total, layout.size, layout.nodes);
        return 0;
    }
    // the update: new positions in the same vertex buffers, then a refit of the structure just built
    EXPECT(moved->size() == geometries.size());
    std::vector<Tri> moved_tris;
    for (std::size_t g = 0; g < moved->size(); ++g) {
        EXPECT((*moved)[g].vertex_data.size() == geometries[g].vertex_data.size());
        std::memcpy(vertex_buffers[g].map, (*moved)[g].vertex_data.data(), (*moved)[g].vertex_data.size() * 4);
        expand((*moved)[g], std::uint32_t(g), moved_tris);
    }
    Buffer other = into_other ? make(gpu, layout.size + 256) : Buffer{};
    const std::uint64_t target = into_other ? (other.address + 255) & ~std::uint64_t(255) : base;
    EXPECT(!into_other || other.buffer);
    EXPECT(gpu.run([&](VkCommandBuffer cb) {
        recorded = builder.record_blas(cb, layout, descs, target, scratch.address, base);
        barrier_to_host(gpu, cb);
        return recorded;
    }));
    if (into_other) EXPECT(dump(name + "_source", dst, base, built.tris));     // a refit must not touch its source
    EXPECT(dump(name, into_other ? other : dst, target, moved_tris));
    built.tris = moved_tris;
    built.address = target;
    std::printf("%s: %u triangles refitted %s, %u bytes\n", name.c_str(), total,
                into_other ? "into another buffer" : "in place", layout.size);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: vk_xdxr_build_test <out dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    if (const auto ec = prepare_xdxr_build_output(dir, argc > 2)) {
        std::fprintf(stderr, "cannot create output directory %s: %s\n", dir.c_str(), ec.message().c_str());
        return 1;
    }
    vktest::Device gpu;
    gpu.request_buffer_device_address = true;
    const auto status = gpu.initialize();
    if (status == vktest::DeviceStatus::unavailable || (status == vktest::DeviceStatus::ready && !gpu.buffer_device_address)) {
        std::printf("vk_xdxr_build_test: no Vulkan device with bufferDeviceAddress, skipped\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;
    XdxrBuilder builder(gpu.device_proc, gpu.device);
    EXPECT(builder.loaded());

    if (argc > 2) {                                   // timing only
        const std::uint32_t n = std::uint32_t(std::strtoul(argv[2], nullptr, 10));
        EXPECT(n > 0);
        std::mt19937 trng(3);
        std::uniform_real_distribution<float> u(-100.0f, 100.0f), j(-0.5f, 0.5f);
        Buffer v = make(gpu, std::size_t(n) * 36);
        EXPECT(v.buffer);
        auto* f = static_cast<float*>(v.map);
        for (std::uint32_t t = 0; t < n; ++t) {
            const float c[3] = {u(trng), u(trng), u(trng)};
            for (int k = 0; k < 9; ++k) f[9 * t + k] = c[k % 3] + j(trng);
        }
        XdxrTriangles d;
        d.vertices = v.address;
        d.triangle_count = n;
        XdxrLayout layout;
        EXPECT(layout.compute(n, false));
        // in VRAM, where the layer keeps acceleration structures (host-visible memory on a discrete GPU is PCIe)
        Buffer dst = make(gpu, layout.size, true), scratch = make(gpu, std::size_t(layout.scratch), true);
        EXPECT(dst.buffer && scratch.buffer);
        double best = 1e30;
        for (int rep = 0; rep < 4; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            EXPECT(gpu.run([&](VkCommandBuffer cb) { return builder.record_blas(cb, layout, {d}, dst.address, scratch.address); }));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (rep) best = std::min(best, ms);
        }
        double refit = 1e30;                          // an update of the structure just built, in place
        for (int rep = 0; rep < 4; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            EXPECT(gpu.run([&](VkCommandBuffer cb) {
                return builder.record_blas(cb, layout, {d}, dst.address, scratch.address, dst.address);
            }));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (rep) refit = std::min(refit, ms);
        }
        std::printf("%u triangles: %.2f ms (best of 3), refit %.2f ms, result %u bytes, scratch %llu bytes, sort size %u\n",
                    n, best, refit, layout.size, static_cast<unsigned long long>(layout.scratch), layout.sort_size);
        return 0;
    }

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> spread(-4.0f, 4.0f), jitter(-0.6f, 0.6f);
    auto soup = [&](std::uint32_t n) {
        Geometry g;
        g.triangle_count = n;
        for (std::uint32_t t = 0; t < n; ++t) {
            const float c[3] = {spread(rng), spread(rng), spread(rng)};
            for (int k = 0; k < 3; ++k)
                for (int a = 0; a < 3; ++a) g.vertex_data.push_back(c[a] + jitter(rng));
        }
        return g;
    };
    auto grid = [&](std::uint32_t n, std::uint32_t floats_per_vertex, bool xy) {
        Geometry g;
        g.stride = 4 * floats_per_vertex;
        g.xy = xy;
        for (std::uint32_t j = 0; j <= n; ++j)
            for (std::uint32_t i = 0; i <= n; ++i) {
                const float p[4] = {float(i) - n * 0.5f, float(j) - n * 0.5f, xy ? 0.0f : 0.25f * float((i * 7 + j * 3) % 5), 9.0f};
                for (std::uint32_t f = 0; f < floats_per_vertex; ++f) g.vertex_data.push_back(p[f]);
            }
        for (std::uint32_t j = 0; j < n; ++j)
            for (std::uint32_t i = 0; i < n; ++i) {
                const std::uint32_t a = j * (n + 1) + i;
                const std::uint32_t q[6] = {a, a + 1, a + n + 2, a, a + n + 2, a + n + 1};
                g.indices.insert(g.indices.end(), q, q + 6);
            }
        g.triangle_count = 2 * n * n;
        return g;
    };

    Built one, five, random, grid16, xy32, moved, multi, empty;
    if (build_blas(gpu, builder, dir, "blas_one", {soup(1)}, one)) return 1;
    if (build_blas(gpu, builder, dir, "blas_five", {soup(5)}, five)) return 1;
    if (build_blas(gpu, builder, dir, "blas_random", {soup(1000)}, random)) return 1;
    Geometry g16 = grid(12, 4, false);
    g16.u16 = true;
    if (build_blas(gpu, builder, dir, "blas_grid_u16_stride16", {g16}, grid16)) return 1;
    if (build_blas(gpu, builder, dir, "blas_grid_u32_xy", {grid(9, 2, true)}, xy32)) return 1;
    Geometry gt = grid(6, 3, false);
    gt.transform = true;
    const float m[12] = {0, 2, 0, 1, 0, 0, 1, -2, 2, 0, 0, 3};   // exact in float: permute, scale by 2, translate
    std::memcpy(gt.m, m, sizeof m);
    if (build_blas(gpu, builder, dir, "blas_transform", {gt}, moved)) return 1;
    Geometry a = soup(40), b = soup(25);
    b.vertex_data[3 * 3 * 7] = std::nanf("");                    // triangle 7 of geometry 1: inactive
    if (build_blas(gpu, builder, dir, "blas_two_geometries", {a, b}, multi)) return 1;
    if (build_blas(gpu, builder, dir, "blas_empty", {}, empty)) return 1;
    // many radix-sort tiles (RADIX_TILE elements each), the last one partial (fewer rays: the brute force is slow)
    Built tiles;
    if (build_blas(gpu, builder, dir, "blas_three_sort_tiles", {soup(5000)}, tiles, 200)) return 1;

    // Updates. Every triangle moves (its own offset, and a stretch), so each box changes; the tree is kept.
    auto move = [&](Geometry g, float amount) {
        std::uniform_real_distribution<float> d(-amount, amount);
        const std::uint32_t floats = g.stride / 4;
        for (std::size_t v = 0; v + 2 < g.vertex_data.size(); v += floats) {
            const float dx = d(rng), dy = d(rng);
            g.vertex_data[v] = g.vertex_data[v] * 1.25f + dx;
            g.vertex_data[v + 1] = g.vertex_data[v + 1] * 0.8f + dy;
            if (!g.xy) g.vertex_data[v + 2] += d(rng);
        }
        return g;
    };
    {
        Built refitted;
        const Geometry big = soup(3000);                    // the radix-sort path
        const std::vector<Geometry> after = {move(big, 1.5f)};
        if (build_blas(gpu, builder, dir, "blas_refit_in_place", {big}, refitted, 300, &after)) return 1;
        Built copied;
        const Geometry g = grid(12, 4, false);              // R32 indices, stride 16; the small-sort path
        const std::vector<Geometry> after_grid = {move(g, 0.3f)};
        if (build_blas(gpu, builder, dir, "blas_refit_into_other", {g}, copied, 0, &after_grid, true)) return 1;
    }

    // BOX16 inner nodes: the same tree with half boxes rounded outward (64-byte nodes instead of 128).
    {
        Built h0, h1, h2, h3, h4;                          // build_blas appends to a Built's triangles
        if (build_blas(gpu, builder, dir, "blas_box16_random", {soup(1000)}, h0, 0, nullptr, false, true)) return 1;
        if (build_blas(gpu, builder, dir, "blas_box16_sort_tiles", {soup(5000)}, h1, 200, nullptr, false, true)) return 1;
        Geometry a16 = soup(40), b16 = soup(25);
        b16.vertex_data[3 * 3 * 3] = std::nanf("");                // an inactive triangle: an empty box
        if (build_blas(gpu, builder, dir, "blas_box16_two_geometries", {a16, b16}, h2, 0, nullptr, false, true)) return 1;
        // Coordinates the half format cannot hold: past 65504 (inf outward, HALF_MAX inward), between 2^-14 and
        // 2^-24 and below (no denormal halves: 0 or +-2^-14), exact halves, and ordinary triangles the rays hit.
        Geometry x = soup(24);
        const float odd[][9] = {
            {70000.0f, 1.0f, 2.0f, 70001.0f, 3.0f, 2.5f, 69999.0f, 2.0f, 1.5f},
            {-65520.0f, -65504.0f, 65504.0f, -65519.0f, -65500.0f, 65505.0f, -65000.0f, -65503.0f, 65490.0f},
            {3.0e-5f, -2.0e-6f, 1.0e-7f, 6.0e-5f, 4.0e-5f, -1.0e-9f, -3.0e-5f, 6.103515625e-5f, 0.0f},
            {-0.0f, 1.0e-30f, -1.0e-30f, 2.0f, 0.5f, 0.25f, 1.0f, -0.125f, 0.0f},
            {1e30f, -1e30f, 5.0f, 1e30f, -1e30f, 6.0f, -1e30f, 1e30f, 5.5f},
            {1.0009766f, -2.0019531f, 3.0f, 1.0f, -2.0f, 3.0029297f, 1.0004883f, -2.0f, 3.0f},
        };
        for (const auto& t : odd) x.vertex_data.insert(x.vertex_data.end(), t, t + 9);
        x.triangle_count += sizeof odd / sizeof odd[0];
        if (build_blas(gpu, builder, dir, "blas_box16_extremes", {x}, h3, 0, nullptr, false, true)) return 1;
        const Geometry big = soup(3000);
        const std::vector<Geometry> after = {move(big, 1.5f)};
        if (build_blas(gpu, builder, dir, "blas_box16_refit_in_place", {big}, h4, 300, &after, false, true)) return 1;
    }

    // TRI1 pairing: the same tree, with sibling triangles that share an edge in one leaf.
    {
        Built p0, p1, p2, p3, p4;
        Geometry pg = grid(12, 4, false);
        pg.u16 = true;
        if (build_blas(gpu, builder, dir, "blas_paired_grid", {pg}, p0, 0, nullptr, false, false, true)) return 1;
        if (build_blas(gpu, builder, dir, "blas_paired_grid_box16", {grid(9, 2, true)}, p1, 0, nullptr, false, true,
                       true))
            return 1;
        Geometry ptf = grid(6, 3, false);
        ptf.transform = true;
        std::memcpy(ptf.m, m, sizeof m);
        if (build_blas(gpu, builder, dir, "blas_paired_transform", {ptf}, p2, 0, nullptr, false, false, true)) return 1;
        // A grid with a NaN vertex (its triangles are inactive); quads split across two geometries (each edge shared
        // the right way, but a leaf holds one geometry, so a pair would report the wrong one); and triangle pairs that
        // share an edge in the same direction (mirrored winding: not a valid pair).
        Geometry holed = grid(8, 3, false);
        holed.vertex_data[3 * 40] = std::nanf("");
        // Each pair is small and the pairs are spread through the volume, so a pair's two triangles have nearly the
        // same Morton key and end up siblings: the rules, not the sort, must keep them apart.
        Geometry cross_a, cross_b, same_dir;
        for (int i = 0; i < 32; ++i) {
            const float cx = spread(rng), cy = spread(rng), cz = spread(rng);
            const float a[3] = {cx - 0.2f, cy, cz}, b[3] = {cx + 0.2f, cy, cz};
            const float up[3] = {cx, cy + 0.3f, cz + 0.05f}, down[3] = {cx, cy - 0.3f, cz - 0.05f};
            const float ta[9] = {a[0], a[1], a[2], b[0], b[1], b[2], up[0], up[1], up[2]};        // A -> B
            const float tb[9] = {b[0], b[1], b[2], a[0], a[1], a[2], down[0], down[1], down[2]};  // B -> A
            const float td[9] = {a[0], a[1], a[2], b[0], b[1], b[2], down[0], down[1], down[2]};  // A -> B again
            if (i < 16) {                                    // across two geometries, the reverse edge
                cross_a.vertex_data.insert(cross_a.vertex_data.end(), ta, ta + 9);
                cross_b.vertex_data.insert(cross_b.vertex_data.end(), tb, tb + 9);
            } else {                                         // one geometry, the same direction
                same_dir.vertex_data.insert(same_dir.vertex_data.end(), ta, ta + 9);
                same_dir.vertex_data.insert(same_dir.vertex_data.end(), td, td + 9);
            }
        }
        cross_a.triangle_count = cross_b.triangle_count = 16;
        same_dir.triangle_count = 32;
        if (build_blas(gpu, builder, dir, "blas_paired_mixed", {holed, cross_a, cross_b, same_dir}, p3, 0, nullptr,
                       false, false, true))
            return 1;
        // on their own, so that the triangles that must not pair are siblings: no TRI1 at all
        Built q0, q1;
        if (build_blas(gpu, builder, dir, "blas_unpaired_geometries", {cross_a, cross_b}, q0, 0, nullptr, false, false,
                       true, true))
            return 1;
        if (build_blas(gpu, builder, dir, "blas_unpaired_same_direction", {same_dir}, q1, 0, nullptr, false, false,
                       true, true))
            return 1;
        // Every triangle of a grid starts at another of its vertices (the same winding), so pairs come in all nine
        // combinations of the first triangle's rotation k and the sibling's kd (check-build asserts it).
        Geometry rotated = grid(16, 3, false);
        for (std::size_t t = 0; t + 2 < rotated.indices.size(); t += 3) {
            const std::uint32_t r = std::uint32_t((t / 3) * 7 / 3 % 3), i0 = rotated.indices[t],
                                i1 = rotated.indices[t + 1], i2 = rotated.indices[t + 2];
            const std::uint32_t v[3] = {i0, i1, i2};
            for (std::uint32_t k = 0; k < 3; ++k) rotated.indices[t + k] = v[(k + r) % 3];
        }
        Built p5;
        if (build_blas(gpu, builder, dir, "blas_paired_rotations", {rotated}, p5, 0, nullptr, false, false, true))
            return 1;
        // 7,200 triangles: the radix-sort path
        if (build_blas(gpu, builder, dir, "blas_paired_sort_tiles", {grid(60, 3, false)}, p4, 200, nullptr, false,
                       false, true))
            return 1;
    }

    // TLAS: instances over blas_random and blas_grid_u16_stride16. An instance names its BLAS by a 40-bit address
    // (256-byte units in 32 bits), as on the console; device addresses can be wider (llvmpipe's are host pointers).
    // The instances hold application-style addresses, device address - bias, as the layer's address window does.
    const std::uint64_t bias = (std::min(random.address, grid16.address) - 0x10000) & ~std::uint64_t(0xffff);
    auto va = [&](std::uint64_t device) { return device ? device - bias : 0; };
    struct Inst { float m[12]; std::uint32_t id_mask; std::uint32_t contrib; std::uint64_t blas; int which; };
    const float c = std::cos(0.7f), s = std::sin(0.7f);
    const Inst insts[] = {
        {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, 0xff000000u | 10, 0, va(random.address), 0},
        {{c, -s, 0, 9, s, c, 0, -1, 0, 0, 1, 0.5f}, 0x01000000u | 11, 3, va(grid16.address), 1},
        {{2, 0, 0, -9, 0, 0.5f, 0, 2, 0, 0, 1.5f, 1}, 0x00000000u | 12, 0, va(random.address), 0},   // mask 0
        {{c, 0, s, 0, 0, 1, 0, 9, -s, 0, c, -2}, 0xff000000u | 13, 0, 0, -1},                  // no BLAS
    };
    const std::uint32_t count = sizeof insts / sizeof insts[0];
    for (const Inst& i : insts) EXPECT(i.blas < (std::uint64_t(1) << 40));
    Buffer descs = make(gpu, 64 * count);
    EXPECT(descs.buffer);
    for (std::uint32_t i = 0; i < count; ++i) {
        auto* d = static_cast<char*>(descs.map) + 64 * i;
        std::memcpy(d, insts[i].m, 48);
        std::memcpy(d + 48, &insts[i].id_mask, 4);
        std::memcpy(d + 52, &insts[i].contrib, 4);
        std::memcpy(d + 56, &insts[i].blas, 8);
    }
    XdxrLayout layout;
    EXPECT(layout.compute(count, true));
    Buffer dst = make(gpu, layout.size + 256), scratch = make(gpu, std::size_t(layout.scratch));
    EXPECT(dst.buffer && scratch.buffer);
    const std::uint64_t base = (dst.address + 255) & ~std::uint64_t(255);
    bool recorded = false;
    EXPECT(gpu.run([&](VkCommandBuffer cb) {
        recorded = builder.record_tlas(cb, layout, descs.address, base, scratch.address, bias);
        barrier_to_host(gpu, cb);
        return recorded;
    }));
    const char* tlas_bytes = static_cast<const char*>(dst.map) + (base - dst.address);
    EXPECT(header_size(tlas_bytes) <= layout.size);
    EXPECT(write_file(dir + "/tlas.bin", tlas_bytes, header_size(tlas_bytes)));
    std::string json = "{\"kind\": \"tlas\", \"address\": " + std::to_string(base) + ", \"bias\": " +
                       std::to_string(bias) + ", \"blas\": [\"blas_random\", " +
                       "\"blas_grid_u16_stride16\"], \"instances\": [";
    for (std::uint32_t i = 0; i < count; ++i) {
        json += i ? ", {" : "{";
        json += "\"transform\": [";
        for (int k = 0; k < 12; ++k) json += (k ? ", " : "") + num(insts[i].m[k]);
        json += "], \"mask\": " + std::to_string(insts[i].id_mask >> 24) + ", \"blas\": " + std::to_string(insts[i].which) + "}";
    }
    json += "]}\n";
    EXPECT(write_file(dir + "/tlas.json", json.data(), json.size()));

    // The TLAS updated in place: every instance moves and turns; the masked and the BLAS-less ones stay so.
    Inst moved_insts[sizeof insts / sizeof insts[0]];
    std::memcpy(moved_insts, insts, sizeof insts);
    const float c2 = std::cos(1.9f), s2 = std::sin(1.9f);
    const float turned[12] = {c2, 0, s2, 4, 0, 1, 0, -3, -s2, 0, c2, 2};
    std::memcpy(moved_insts[0].m, turned, sizeof turned);
    moved_insts[1].m[3] -= 7;
    moved_insts[1].m[7] += 5;
    moved_insts[2].m[11] -= 4;
    for (std::uint32_t i = 0; i < count; ++i) std::memcpy(static_cast<char*>(descs.map) + 64 * i, moved_insts[i].m, 48);
    EXPECT(gpu.run([&](VkCommandBuffer cb) {
        recorded = builder.record_tlas(cb, layout, descs.address, base, scratch.address, bias, base);
        barrier_to_host(gpu, cb);
        return recorded;
    }));
    EXPECT(write_file(dir + "/tlas_refit.bin", tlas_bytes, header_size(tlas_bytes)));
    std::string json2 = "{\"kind\": \"tlas\", \"address\": " + std::to_string(base) + ", \"bias\": " +
                        std::to_string(bias) + ", \"blas\": [\"blas_random\", \"blas_grid_u16_stride16\"], \"instances\": [";
    for (std::uint32_t i = 0; i < count; ++i) {
        json2 += i ? ", {" : "{";
        json2 += "\"transform\": [";
        for (int k = 0; k < 12; ++k) json2 += (k ? ", " : "") + num(moved_insts[i].m[k]);
        json2 += "], \"mask\": " + std::to_string(moved_insts[i].id_mask >> 24) + ", \"blas\": " +
                 std::to_string(moved_insts[i].which) + "}";
    }
    json2 += "]}\n";
    EXPECT(write_file(dir + "/tlas_refit.json", json2.data(), json2.size()));

    // CopyRaytracingAccelerationStructure: a clone of blas_random, its COMPACTED_SIZE (postbuild), a compaction into
    // a buffer of exactly that size, and a clone of the refitted TLAS. The copies are position-independent.
    {
        XdxrLayout blas_layout;
        EXPECT(blas_layout.compute(1000, false));
        Buffer clone = make(gpu, blas_layout.size + 256), info = make(gpu, 16);
        EXPECT(clone.buffer && info.buffer);
        const std::uint64_t clone_at = (clone.address + 255) & ~std::uint64_t(255);
        std::memset(info.map, 0xab, 16);
        EXPECT(gpu.run([&](VkCommandBuffer cb) {
            const bool ok = builder.record_copy(cb, random.address, clone_at, blas_layout.size) &&
                            builder.record_postbuild(cb, random.address, info.address) &&
                            builder.record_postbuild(cb, clone_at, info.address + 8);
            barrier_to_host(gpu, cb);
            return ok;
        }));
        std::uint64_t compacted = 0, clone_size = 0;
        std::memcpy(&compacted, info.map, 8);
        std::memcpy(&clone_size, static_cast<char*>(info.map) + 8, 8);
        // a build is already as small as it gets: the size its header gives, within the layout's room
        EXPECT(compacted == random.size && compacted <= blas_layout.size);
        EXPECT(clone_size == compacted);
        auto dump_blas = [&](const std::string& name, const Buffer& b, std::uint64_t at, std::size_t bytes) {
            const std::string j = "{\"kind\": \"blas\", \"address\": " + std::to_string(at) +
                                  ", \"triangles\": " + tris_json(random.tris) + "}\n";
            return write_file(dir + "/" + name + ".bin", static_cast<const char*>(b.map) + (at - b.address), bytes) &&
                   write_file(dir + "/" + name + ".json", j.data(), j.size());
        };
        EXPECT(dump_blas("blas_clone", clone, clone_at, std::size_t(compacted)));
        // COMPACT into exactly the compacted size; a guard word after it must survive
        Buffer packed = make(gpu, std::size_t(compacted) + 256 + 4);
        EXPECT(packed.buffer);
        const std::uint64_t packed_at = (packed.address + 255) & ~std::uint64_t(255);
        const std::uint32_t guard = 0x600dc0deu;
        std::memcpy(static_cast<char*>(packed.map) + (packed_at - packed.address) + compacted, &guard, 4);
        EXPECT(gpu.run([&](VkCommandBuffer cb) {
            const bool ok = builder.record_copy(cb, clone_at, packed_at, compacted + 4);   // more room than needed
            barrier_to_host(gpu, cb);
            return ok;
        }));
        std::uint32_t after = 0;
        std::memcpy(&after, static_cast<char*>(packed.map) + (packed_at - packed.address) + compacted, 4);
        EXPECT(after == guard);                          // the copy stops at the source's own size
        EXPECT(dump_blas("blas_compacted", packed, packed_at, std::size_t(compacted)));

        Buffer tclone = make(gpu, layout.size + 256);
        EXPECT(tclone.buffer);
        const std::uint64_t tclone_at = (tclone.address + 255) & ~std::uint64_t(255);
        EXPECT(gpu.run([&](VkCommandBuffer cb) {
            const bool ok = builder.record_copy(cb, base, tclone_at, layout.size);
            barrier_to_host(gpu, cb);
            return ok;
        }));
        EXPECT(write_file(dir + "/tlas_refit_clone.bin", static_cast<const char*>(tclone.map) + (tclone_at - tclone.address),
                          header_size(tlas_bytes)));
        std::string json3 = json2;
        json3.replace(json3.find(std::to_string(base)), std::to_string(base).size(), std::to_string(tclone_at));
        EXPECT(write_file(dir + "/tlas_refit_clone.json", json3.data(), json3.size()));
        std::printf("copies: blas_clone, blas_compacted (%llu bytes), tlas_refit_clone\n",
                    static_cast<unsigned long long>(compacted));
    }
    std::printf("tlas: %u instances, %u bytes\nvk_xdxr_build_test: built every case\n", count, layout.size);
    return 0;
}
