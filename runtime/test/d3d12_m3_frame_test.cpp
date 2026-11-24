// Milestone M3, the synthetic part (docs/d3d12_vulkan.md section 6): frames with the structure a game's have, through the
// D3D12 API on the proxy, with vertex and pixel shaders that are translator output (test/shaders/vs_m3.asm, ps_m3.asm,
// vs_m2.asm, ps_m3_post.asm). Render passes (BeginRenderPass, D3D-27) are not part of it: the draws use
// OMSetRenderTargets.
// One frame is two passes:
//   pass 1  two render targets (R8G8B8A8_UNORM and R32_FLOAT) and a D24_UNORM_S8_UINT depth/stencil buffer (the layer
//           maps it to D32_SFLOAT_S8_UINT where the device has no 24-bit depth). Three rectangles from root constants:
//           N (near, depth 0.25, depth LESS + write, stencil REPLACE with reference 1), then F (far, depth 0.75, the
//           same state, stencil reference 2), then G (no depth test, stencil
//           EQUAL to reference 1, so it only reaches pixels N won). ResolveQueryData writes the probe's count to the
//           readback buffer. Before F, a single-output probe of F uses the same depth/stencil tests without writing
//           color, depth or stencil. Only that probe is inside BeginQuery/EndQuery(OCCLUSION): Vulkan permits MRT
//           occlusion counts multiplied by the number of written color outputs even with PRECISE (D3D-209).
//   barrier the first target goes RENDER_TARGET -> PIXEL_SHADER_RESOURCE | COPY_SOURCE, the second to COPY_SOURCE.
//   pass 2  a descriptor-table SRV of the first target (the M2 path: heaps, T#, S#, hash table) sampled by a translated PS
//           that swaps red and blue and writes a root constant to alpha, into a third target.
// Six frames go through a ring of three slots (three frames in flight): every slot has its own allocator, list, targets,
// depth buffer, query and fence. A frame waits for the fence its slot signalled three frames earlier
// (SetEventOnCompletion with an event), reads that frame back and only then records the slot again; the root constants
// (rectangles, colours, alpha) differ per frame.
// Compared with a CPU reference: every pixel of all three targets byte for byte (the float target bit for bit), and the
// exact number of samples F made visible (rectangles are pixel-aligned, so the count is the area of F minus N).
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation (0 errors required).
// --first-pass-only isolates depth/stencil, MRT, queries and slot reuse from the descriptor-sampling second pass.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cstring>

#ifndef LEONIDA_TEST_M3_DIR
#error "LEONIDA_TEST_M3_DIR must name the folder build.py wrote (spirv/index.json, fssm/*.fssm)"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX

namespace {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

constexpr UINT dim = 64;                  // every target is 64x64
constexpr UINT pitch = dim * 4;           // 256, both for RGBA8 and for R32F
constexpr UINT image_bytes = pitch * dim;
constexpr UINT slots = 3, frames = 6;      // three frames in flight, six frames
// The readback buffer of a slot: the query result at 0, then the three images, each at a 512-byte aligned offset.
constexpr UINT64 offset_rt0 = 512, offset_rt1 = offset_rt0 + image_bytes, offset_rt2 = offset_rt1 + image_bytes;
constexpr UINT64 readback_bytes = offset_rt2 + image_bytes;

struct Rect { int x0, y0, x1, y1; };  // pixels, half open, y down
struct Draw {
    Rect rect;
    float depth;
    std::uint8_t k0, k1, k2;            // the colour (k0, k1, k2)/255 and the float target's (k1 + k2)/255
    UINT stencil_ref;
};

bool inside(const Rect& r, int x, int y) { return x >= r.x0 && x < r.x1 && y >= r.y0 && y < r.y1; }
float unorm(std::uint8_t k) { return float(k) / 255.0f; }

// The three draws of a frame (the same for the GPU and the reference).
struct Frame {
    Draw n, f, g;
    std::uint8_t alpha;  // pass 2's root constant, alpha / 255
};
Frame frame_of(UINT n) {
    const int s = int(n);
    return {{{6 + 2 * s, 8, 38 + 2 * s, 44}, 0.25f, 200, 60, std::uint8_t(20 + 30 * s), 1},
            {{22 - s, 18 + s, 54 - s, 56}, 0.75f, 40, 220, std::uint8_t(90 + 10 * s), 2},
            {{2, 28, 62, 36}, 0.0f, 250, 250, std::uint8_t(100 + 5 * s), 1},
            std::uint8_t(255 - 20 * s)};
}

struct Reference {
    std::vector<std::uint8_t> rt0, rt2;  // RGBA8
    std::vector<float> rt1;
    UINT64 samples = 0;
};
Reference reference_of(const Frame& fr, bool first_pass_only) {
    Reference ref;
    ref.rt0.assign(image_bytes, 0);
    ref.rt2.assign(image_bytes, 0);
    ref.rt1.assign(dim * dim, -1.0f);
    std::vector<float> depth(dim * dim, 1.0f);
    std::vector<std::uint8_t> stencil(dim * dim, 0);
    const auto put = [&](UINT i, const Draw& d) {
        ref.rt0[i * 4 + 0] = d.k0;
        ref.rt0[i * 4 + 1] = d.k1;
        ref.rt0[i * 4 + 2] = d.k2;
        ref.rt0[i * 4 + 3] = 255;
        ref.rt1[i] = unorm(d.k1) + unorm(d.k2);
    };
    for (const Draw* d : {&fr.n, &fr.f}) {   // depth LESS + write, stencil REPLACE on a pass
        for (int y = 0; y < int(dim); ++y)
            for (int x = 0; x < int(dim); ++x) {
                const UINT i = UINT(y) * dim + UINT(x);
                if (!inside(d->rect, x, y) || !(d->depth < depth[i])) continue;
                depth[i] = d->depth;
                stencil[i] = std::uint8_t(d->stencil_ref);
                put(i, *d);
                if (d == &fr.f) ++ref.samples;
            }
    }
    for (int y = 0; y < int(dim); ++y)      // G: stencil EQUAL 1, no depth test
        for (int x = 0; x < int(dim); ++x) {
            const UINT i = UINT(y) * dim + UINT(x);
            if (inside(fr.g.rect, x, y) && stencil[i] == fr.g.stencil_ref) put(i, fr.g);
        }
    for (UINT i = 0; !first_pass_only && i < dim * dim; ++i) {  // pass 2: (b, g, r, alpha)
        ref.rt2[i * 4 + 0] = ref.rt0[i * 4 + 2];
        ref.rt2[i * 4 + 1] = ref.rt0[i * 4 + 1];
        ref.rt2[i * 4 + 2] = ref.rt0[i * 4 + 0];
        ref.rt2[i * 4 + 3] = fr.alpha;
    }
    return ref;
}

}  // namespace

int main(int argc, char** argv) {
    const bool first_pass_only = argc == 2 && std::strcmp(argv[1], "--first-pass-only") == 0;
    if (argc != 1 && !first_pass_only) {
        std::fprintf(stderr, "usage: d3d12_m3_frame_test [--first-pass-only]\n");
        return 1;
    }
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    if (!d.bindless) {
        std::printf("SKIP: the device lacks the descriptor indexing features of the bindless set 0\n");
        return 77;
    }

    const std::filesystem::path root_dir = LEONIDA_TEST_M3_DIR;
    leonida::ShaderLibrary library;
    EXPECT(library.load((root_dir / "spirv").string()));
    const auto program = [&](const char* name) { return read_file(root_dir / "fssm" / (std::string(name) + ".fssm")); };
    const auto vs1 = program("vs_m3"), ps1 = program("ps_m3"), vs2 = program("vs_m2"), ps2 = program("ps_m3_post");
    const auto ps_query = program("ps_m3_query");
    EXPECT(vs1.size() >= 4 && ps1.size() >= 4 && vs2.size() >= 4 && ps2.size() >= 4);
    EXPECT(library.find("vs", leonida::sha256_hex(vs1)) && library.find("ps", leonida::sha256_hex(ps1)));
    EXPECT(library.find("vs", leonida::sha256_hex(vs2)) && library.find("ps", leonida::sha256_hex(ps2)));
    EXPECT(ps_query.size() >= 4 && library.find("ps", leonida::sha256_hex(ps_query)));
    const auto vs1_container = shader_container(vs1), ps1_container = shader_container(ps1);
    const auto vs2_container = shader_container(vs2), ps2_container = shader_container(ps2);
    const auto query_container = shader_container(ps_query);

    unsigned mismatches = 0, frames_checked = 0;
    UINT64 total_samples = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.image_view_min_lod = d.image_view_min_lod;
        info.occlusion_query_precise = d.occlusion_query_precise;
        info.shader_dir = (root_dir / "spirv").string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        EXPECT(rt::d3d12::bindless_set(device.p) != VK_NULL_HANDLE);

        // Pass 1: [0] eight 32-bit constants (s0..s7). Pass 2: [0] one SRV (s[0:1]), [1] one sampler (s[2:3]),
        // [2] one constant (s4).
        D3D12_ROOT_PARAMETER consts1{};
        consts1.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        consts1.Constants = {0, 0, 8};
        consts1.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {1, &consts1, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root1, root2;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root1.p))));
        D3D12_DESCRIPTOR_RANGE srv_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_DESCRIPTOR_RANGE sampler_range{D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER params2[3] = {};
        params2[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params2[0].DescriptorTable = {1, &srv_range};
        params2[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        params2[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params2[1].DescriptorTable = {1, &sampler_range};
        params2[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        params2[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params2[2].Constants = {0, 0, 1};
        params2[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        root_desc.Desc_1_0 = {3, params2, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root2.p))));

        // A: depth LESS + write, stencil ALWAYS/REPLACE. Query: same tests, no writes, one color export.
        // C: no depth, stencil EQUAL. P: pass 2.
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root1.p;
        pso_desc.VS = {vs1_container.data(), vs1_container.size()};
        pso_desc.PS = {ps1_container.data(), ps1_container.size()};
        for (auto& rt : pso_desc.BlendState.RenderTarget) rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 2;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.RTVFormats[1] = DXGI_FORMAT_R32_FLOAT;
        pso_desc.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
        pso_desc.SampleDesc.Count = 1;
        auto& ds = pso_desc.DepthStencilState;
        ds.DepthEnable = TRUE;
        ds.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        ds.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        ds.StencilEnable = TRUE;
        ds.StencilReadMask = ds.StencilWriteMask = 0xff;
        ds.FrontFace = ds.BackFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_REPLACE,
                                      D3D12_COMPARISON_FUNC_ALWAYS};
        Ref<ID3D12PipelineState> pso_a, pso_c, pso_p, pso_query;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso_a.p))));
        auto probe_desc = pso_desc;
        probe_desc.PS = {query_container.data(), query_container.size()};
        probe_desc.NumRenderTargets = 1;
        probe_desc.RTVFormats[1] = DXGI_FORMAT_UNKNOWN;
        probe_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0;
        probe_desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        probe_desc.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
        probe_desc.DepthStencilState.BackFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&probe_desc, IID_PPV_ARGS(&pso_query.p))));
        ds.DepthEnable = FALSE;
        ds.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        ds.FrontFace = ds.BackFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                      D3D12_COMPARISON_FUNC_EQUAL};
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso_c.p))));
        pso_desc.pRootSignature = root2.p;
        pso_desc.VS = {vs2_container.data(), vs2_container.size()};
        pso_desc.PS = {ps2_container.data(), ps2_container.size()};
        pso_desc.DepthStencilState = {};
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[1] = DXGI_FORMAT_UNKNOWN;
        pso_desc.DSVFormat = DXGI_FORMAT_UNKNOWN;
        if (!first_pass_only)
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso_p.p))));

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC image_desc{};
        image_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        image_desc.Width = image_desc.Height = dim;
        image_desc.DepthOrArraySize = 1;
        image_desc.MipLevels = 1;
        image_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        image_desc.SampleDesc.Count = 1;
        image_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        auto float_desc = image_desc;
        float_desc.Format = DXGI_FORMAT_R32_FLOAT;
        auto depth_desc = image_desc;
        depth_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        depth_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        constexpr D3D12_RESOURCE_STATES rt0_rest = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_COPY_SOURCE;
        constexpr D3D12_RESOURCE_STATES copy_rest = D3D12_RESOURCE_STATE_COPY_SOURCE;

        // The index buffer: the quad, in an upload buffer.
        Ref<ID3D12Resource> index_buffer;
        const auto upload_desc = buffer_desc(256);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&index_buffer.p))));
        const std::uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(index_buffer->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memcpy(up, indices, sizeof(indices));
        index_buffer->Unmap(0, nullptr);

        // Heaps: the shader-visible SRV heap (a slot per frame slot) and sampler heap, the RTV heap (three views a
        // slot) and the DSV heap (one per slot). The start handles come back in RAX (heap slots 9 and 10).
        const D3D12_DESCRIPTOR_HEAP_DESC srv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, slots,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1,
                                                           D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 3 * slots,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC dsv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, slots,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> srv_heap, sampler_heap, rtv_heap, dsv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&srv_heap_desc, IID_PPV_ARGS(&srv_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&sampler_heap_desc, IID_PPV_ARGS(&sampler_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&dsv_heap_desc, IID_PPV_ARGS(&dsv_heap.p))));
        const auto cpu_start = [](ID3D12DescriptorHeap* h) {
            return reinterpret_cast<XboxHeapHandleFn>(slot(h, 9))(h);
        };
        const auto gpu_start = [](ID3D12DescriptorHeap* h) {
            return reinterpret_cast<XboxHeapHandleFn>(slot(h, 10))(h);
        };
        const std::uint64_t srv_cpu = cpu_start(srv_heap.p), srv_gpu = gpu_start(srv_heap.p);
        const std::uint64_t sampler_cpu = cpu_start(sampler_heap.p), sampler_gpu = gpu_start(sampler_heap.p);
        const std::uint64_t rtv_cpu = cpu_start(rtv_heap.p), dsv_cpu = cpu_start(dsv_heap.p);
        const UINT srv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const UINT rtv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        const UINT dsv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        EXPECT(srv_cpu && sampler_cpu && rtv_cpu && dsv_cpu && srv_gpu >= 0x10000 && sampler_gpu >= 0x10000);
        EXPECT(srv_gpu < (1ull << 32) && sampler_gpu < (1ull << 32) && srv_step >= 32 && rtv_step && dsv_step);

        D3D12_SAMPLER_DESC sampler_desc{};
        sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler_desc.MaxAnisotropy = 1;
        sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
        device->CreateSampler(&sampler_desc, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu)});

        D3D12_QUERY_HEAP_DESC query_desc{D3D12_QUERY_HEAP_TYPE_OCCLUSION, slots, 0};
        Ref<ID3D12QueryHeap> query_heap;
        EXPECT(SUCCEEDED(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&query_heap.p))));

        struct Slot {
            Ref<ID3D12Resource> rt0, rt1, rt2, depth, readback;
            Ref<ID3D12CommandAllocator> allocator;
            Ref<ID3D12GraphicsCommandList> list;
            Ref<ID3D12Fence> fence;
            UINT64 fence_value = 0;
            int frame = -1;                       // the frame in flight in this slot, -1 when none
        };
        std::array<Slot, slots> ring;
        D3D12_CPU_DESCRIPTOR_HANDLE rtvs[slots][3], dsvs[slots];
        std::uint32_t keys[slots][8];
        const auto readback_desc = buffer_desc(readback_bytes);
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        for (UINT s = 0; s < slots; ++s) {
            auto& slot_ = ring[s];
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &image_desc, rt0_rest,
                                                             nullptr, IID_PPV_ARGS(&slot_.rt0.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &float_desc, copy_rest,
                                                             nullptr, IID_PPV_ARGS(&slot_.rt1.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &image_desc, copy_rest,
                                                             nullptr, IID_PPV_ARGS(&slot_.rt2.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &depth_desc,
                                                             D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                             IID_PPV_ARGS(&slot_.depth.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&slot_.readback.p))));
            for (UINT i = 0; i < 3; ++i) rtvs[s][i] = {SIZE_T(rtv_cpu + std::uint64_t(s * 3 + i) * rtv_step)};
            device->CreateRenderTargetView(slot_.rt0.p, nullptr, rtvs[s][0]);
            device->CreateRenderTargetView(slot_.rt1.p, nullptr, rtvs[s][1]);
            device->CreateRenderTargetView(slot_.rt2.p, nullptr, rtvs[s][2]);
            dsvs[s] = {SIZE_T(dsv_cpu + std::uint64_t(s) * dsv_step)};
            device->CreateDepthStencilView(slot_.depth.p, nullptr, dsvs[s]);
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D = {0, 1, 0, 0.0f};
            device->CreateShaderResourceView(slot_.rt0.p, &srv, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(srv_cpu + std::uint64_t(s) * srv_step)});
            std::memcpy(keys[s], reinterpret_cast<const void*>(srv_cpu + std::uint64_t(s) * srv_step), 32);
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&slot_.allocator.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, slot_.allocator.p, nullptr,
                                                       IID_PPV_ARGS(&slot_.list.p))));
            EXPECT(SUCCEEDED(slot_.list->Close()));
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&slot_.fence.p))));
        }
        // The T# of each render target's SRV has an index in the hash table, and no two share one.
        std::uint32_t indices_seen[slots];
        for (UINT s = 0; s < slots; ++s) {
            indices_seen[s] = rt::d3d12::bindless_hash_lookup(device.p, keys[s]);
            EXPECT(indices_seen[s] != 0);
            for (UINT t = 0; t < s; ++t) EXPECT(indices_seen[s] != indices_seen[t]);
        }
        const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        EXPECT(event != nullptr);

        const D3D12_VIEWPORT viewport{0, 0, float(dim), float(dim), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(dim), LONG(dim)};
        const D3D12_INDEX_BUFFER_VIEW ibv{index_buffer->GetGPUVirtualAddress(), sizeof(indices), DXGI_FORMAT_R16_UINT};
        ID3D12DescriptorHeap* heaps[] = {srv_heap.p, sampler_heap.p};

        const auto draw = [&](ID3D12GraphicsCommandList* list, const Draw& dr, UINT stencil_ref) {
            const auto clip_x = [](int px) { return float(px) / 32.0f - 1.0f; };
            const auto clip_y = [](int py) { return 1.0f - float(py) / 32.0f; };
            const float c[8] = {clip_x(dr.rect.x0), clip_y(dr.rect.y0), clip_x(dr.rect.x1), clip_y(dr.rect.y1), dr.depth,
                                unorm(dr.k0), unorm(dr.k1), unorm(dr.k2)};
            list->SetGraphicsRoot32BitConstants(0, 8, c, 0);
            list->OMSetStencilRef(stencil_ref);
            list->DrawIndexedInstanced(6, 1, 0, 0, 0);
        };
        const auto copy_image = [&](ID3D12GraphicsCommandList* list, Slot& slot_, ID3D12Resource* image, DXGI_FORMAT format,
                                    UINT64 offset) {
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = slot_.readback.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = {offset, {format, dim, dim, 1, pitch}};
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = image;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        };

        // Records and submits frame n in its slot.
        const auto submit = [&](UINT n) -> bool {
            const UINT s = n % slots;
            auto& slot_ = ring[s];
            const Frame fr = frame_of(n);
            auto* list = slot_.list.p;
            if (FAILED(slot_.allocator->Reset()) || FAILED(list->Reset(slot_.allocator.p, pso_a.p))) return false;
            D3D12_RESOURCE_BARRIER begin[3] = {
                transition(slot_.rt0.p, rt0_rest, D3D12_RESOURCE_STATE_RENDER_TARGET),
                transition(slot_.rt1.p, copy_rest, D3D12_RESOURCE_STATE_RENDER_TARGET),
                transition(slot_.rt2.p, copy_rest, D3D12_RESOURCE_STATE_RENDER_TARGET)};
            list->ResourceBarrier(3, begin);
            const FLOAT black[4] = {0, 0, 0, 0}, minus_one[4] = {-1.0f, 0, 0, 0};
            list->ClearRenderTargetView(rtvs[s][0], black, 0, nullptr);
            list->ClearRenderTargetView(rtvs[s][1], minus_one, 0, nullptr);
            list->ClearRenderTargetView(rtvs[s][2], black, 0, nullptr);
            list->ClearDepthStencilView(dsvs[s], D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

            // Pass 1.
            list->SetGraphicsRootSignature(root1.p);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list->IASetIndexBuffer(&ibv);
            list->RSSetViewports(1, &viewport);
            list->RSSetScissorRects(1, &scissor);
            list->OMSetRenderTargets(2, rtvs[s], FALSE, &dsvs[s]);
            draw(list, fr.n, fr.n.stencil_ref);
            // The probe must leave F's depth/stencil/color inputs intact for the real two-output draw below.
            list->SetPipelineState(pso_query.p);
            list->OMSetRenderTargets(1, rtvs[s], FALSE, &dsvs[s]);
            list->BeginQuery(query_heap.p, D3D12_QUERY_TYPE_OCCLUSION, s);
            draw(list, fr.f, fr.f.stencil_ref);
            list->EndQuery(query_heap.p, D3D12_QUERY_TYPE_OCCLUSION, s);
            list->SetPipelineState(pso_a.p);
            list->OMSetRenderTargets(2, rtvs[s], FALSE, &dsvs[s]);
            draw(list, fr.f, fr.f.stencil_ref);
            list->SetPipelineState(pso_c.p);
            draw(list, fr.g, fr.g.stencil_ref);
            list->ResolveQueryData(query_heap.p, D3D12_QUERY_TYPE_OCCLUSION, s, 1, slot_.readback.p, 0);

            D3D12_RESOURCE_BARRIER between[2] = {transition(slot_.rt0.p, D3D12_RESOURCE_STATE_RENDER_TARGET, rt0_rest),
                                                 transition(slot_.rt1.p, D3D12_RESOURCE_STATE_RENDER_TARGET, copy_rest)};
            list->ResourceBarrier(2, between);

            // Pass 2: the first target through an SRV in a descriptor table.
            if (!first_pass_only) {
                list->SetPipelineState(pso_p.p);
                list->SetGraphicsRootSignature(root2.p);
                list->SetDescriptorHeaps(2, heaps);
                list->SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{srv_gpu + std::uint64_t(s) * srv_step});
                list->SetGraphicsRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{sampler_gpu});
                const float alpha = unorm(fr.alpha);
                list->SetGraphicsRoot32BitConstants(2, 1, &alpha, 0);
                list->OMSetRenderTargets(1, &rtvs[s][2], FALSE, nullptr);
                list->DrawIndexedInstanced(6, 1, 0, 0, 0);
            }

            D3D12_RESOURCE_BARRIER end = transition(slot_.rt2.p, D3D12_RESOURCE_STATE_RENDER_TARGET, copy_rest);
            list->ResourceBarrier(1, &end);
            copy_image(list, slot_, slot_.rt0.p, DXGI_FORMAT_R8G8B8A8_UNORM, offset_rt0);
            copy_image(list, slot_, slot_.rt1.p, DXGI_FORMAT_R32_FLOAT, offset_rt1);
            copy_image(list, slot_, slot_.rt2.p, DXGI_FORMAT_R8G8B8A8_UNORM, offset_rt2);
            if (list->Close() != S_OK) return false;
            ID3D12CommandList* lists[] = {list};
            queue->ExecuteCommandLists(1, lists);
            if (rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) return false;
            slot_.fence_value += 1;
            slot_.frame = int(n);
            return SUCCEEDED(queue->Signal(slot_.fence.p, slot_.fence_value));
        };

        // Waits for the frame in a slot (SetEventOnCompletion with an event) and compares its readback.
        const auto retire = [&](UINT s) -> bool {
            auto& slot_ = ring[s];
            if (slot_.frame < 0) return true;
            if (FAILED(slot_.fence->SetEventOnCompletion(slot_.fence_value, event))) return false;
            if (WaitForSingleObject(event, 20000) != WAIT_OBJECT_0) {
                std::fprintf(stderr, "frame %d: the fence never reached %llu\n", slot_.frame,
                             static_cast<unsigned long long>(slot_.fence_value));
                return false;
            }
            if (slot_.fence->GetCompletedValue() < slot_.fence_value) return false;
            const UINT n = UINT(slot_.frame);
            slot_.frame = -1;
            const Reference want = reference_of(frame_of(n), first_pass_only);
            void* mapped = nullptr;
            if (FAILED(slot_.readback->Map(0, nullptr, &mapped))) return false;
            const auto* base = static_cast<const std::uint8_t*>(mapped);
            UINT64 samples = 0;
            std::memcpy(&samples, base, 8);
            const auto compare = [&](const char* what, UINT64 offset, const void* expected, std::size_t unit) {
                const auto* got = base + offset;
                const auto* exp = static_cast<const std::uint8_t*>(expected);
                for (UINT i = 0; i < dim * dim; ++i) {
                    if (std::memcmp(got + i * unit, exp + i * unit, unit) == 0) continue;
                    if (mismatches++ < 12)
                        std::fprintf(stderr, "frame %u %s pixel (%u, %u): got %08x want %08x\n", n, what, i % dim, i / dim,
                                     unit == 4 ? *reinterpret_cast<const std::uint32_t*>(got + i * unit) : 0u,
                                     unit == 4 ? *reinterpret_cast<const std::uint32_t*>(exp + i * unit) : 0u);
                }
            };
            compare("rt0", offset_rt0, want.rt0.data(), 4);
            compare("rt1", offset_rt1, want.rt1.data(), 4);
            compare("rt2", offset_rt2, want.rt2.data(), 4);
            const bool precise = d.occlusion_query_precise;
            if (precise ? samples != want.samples : samples == 0) {
                std::fprintf(stderr, "frame %u: occlusion query counted %llu samples, want %llu\n", n,
                             static_cast<unsigned long long>(samples), static_cast<unsigned long long>(want.samples));
                ++mismatches;
            }
            total_samples += samples;
            ++frames_checked;
            slot_.readback->Unmap(0, nullptr);
            return true;
        };

        for (UINT n = 0; n < frames; ++n) {
            EXPECT(retire(n % slots));   // the slot's frame from three frames ago: wait, read back, then reuse
            EXPECT(submit(n));
        }
        for (UINT s = 0; s < slots; ++s) EXPECT(retire(s));
        CloseHandle(event);

        for (auto& slot_ : ring) {
            slot_.rt0.reset(); slot_.rt1.reset(); slot_.rt2.reset(); slot_.depth.reset(); slot_.readback.reset();
            slot_.list.reset(); slot_.allocator.reset(); slot_.fence.reset();
        }
        index_buffer.reset();
        pso_a.reset(); pso_c.reset(); pso_p.reset(); pso_query.reset();
        root1.reset(); root2.reset();
        srv_heap.reset(); sampler_heap.reset(); rtv_heap.reset(); dsv_heap.reset();
        query_heap.reset();
        queue.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    EXPECT(frames_checked == frames);
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("M3: %u frames, %u in flight, two render targets + D24S8 depth/stencil, barriers, an occlusion query and "
                "%s matched all %u pixels of three targets per frame "
                "(error 0) and every sample count (%llu samples in all%s)\n",
                frames, slots, first_pass_only ? "first-pass isolation (third target stays clear)" :
                    "a second pass sampling the first through an SRV", dim * dim,
                static_cast<unsigned long long>(total_samples),
                d.occlusion_query_precise ? ", exact" : ", imprecise device: non-zero only");
    return 0;
}
