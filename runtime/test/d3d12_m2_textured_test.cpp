// Milestone M2 (docs/d3d12_vulkan.md section 6): a textured draw through the whole D3D12 descriptor path, with a
// pixel shader that is translator output. Everything the application does goes through the D3D12 API on the proxy:
//   - a shader-visible CBV/SRV/UAV heap and a sampler heap (the start handles come back in RAX, heap slots 9/10);
//   - two 8x8 textures, a gradient and its point reflection, uploaded with Map and CopyTextureRegion;
//   - CreateShaderResourceView for each (the layer writes the 32-byte GFX10 T# into the heap slot) and CreateSampler
//     with a POINT filter (the 16-byte S#);
//   - a root signature of two descriptor tables, [0] one SRV and [1] one sampler, and SetDescriptorHeaps,
//     SetGraphicsRootDescriptorTable and DrawIndexedInstanced of a quad (a vertex-index-driven vs_m2.asm, six indices).
// test/shaders/ps_m2.asm s_loads the T# from the table in s[0:1] and the S# from the table in s[2:3] and image_samples
// with them; the translated shader hashes both and finds their bindless indices in the descriptor hash table (set 0,
// binding 2), so the pipeline layout holds set 0 and every draw binds it. The PSO is found by the SHA-256 of the FSSM
// program in the DXBC container, as in the M1.5 test.
// Expected: the quad covers the 64x64 target, so each 8x8 block of pixels is one texel. Scissor rectangles split it:
// the left half is drawn from the gradient, the right half (the second draw switches the SRV table to slot 1) from
// the reflected texture. Compared byte for byte (error 0). The test also checks the hash table: the T# of each view and the S#
// have distinct non-zero indices, and bytes that were never created have none (the fallback, 0).
// The same quad is also drawn with LINEAR/CLAMP. Each RGBA8 channel is checked against the four clamped
// texels at that pixel center, allowing one quantization level.
//
// Sampler control (suggested and measured by @Vonsek in #1489): with U in range, POINT/CLAMP, POINT/WRAP and the
// fallback sampler all give the same pixels, so the draw above cannot tell which sampler it used. A second pass draws
// with ps_m2_uoff.asm, which adds 1 to U just before image_sample, and two sampler-heap slots, [0] POINT/CLAMP and
// [1] POINT/WRAP. The left half takes the WRAP sampler (U + 1 wraps back: the gradient again), the right half the
// CLAMP one (U + 1 clamps to the last column: each row is that row's right-edge texel of the reflected texture). The
// pixels are predicted on the CPU, error 0, and 1536 of them differ from the plain reflection.
// The control also keeps one unbounded SRV table bound and selects its two entries by a root constant.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation (0 errors required).
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#ifndef LEONIDA_TEST_M2_DIR
#error "LEONIDA_TEST_M2_DIR must name the folder build.py wrote (spirv/index.json, fssm/*.fssm)"
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

using Texel = std::array<std::uint8_t, 4>;
constexpr UINT texture_size = 8;
constexpr UINT target_size = 64;  // the target: each texel covers 8x8 pixels
constexpr UINT target_pitch = target_size * 4;  // 256

// The synthetic gradient: red rises with x, green with y, blue falls along the diagonal.
Texel gradient(UINT x, UINT y) {
    return {std::uint8_t(x * 36), std::uint8_t(y * 36), std::uint8_t(255 - (x + y) * 16), 255};
}
// The second texture: the first one point-reflected, so no texel of it is in the same place.
Texel reflected(UINT x, UINT y) { return gradient(texture_size - 1 - x, texture_size - 1 - y); }

Texel linear_clamp(UINT col, UINT row) {
    const float x = (col + 0.5f) * texture_size / target_size - 0.5f;
    const float y = (row + 0.5f) * texture_size / target_size - 0.5f;
    const int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    const float fx = x - x0, fy = y - y0;
    const auto texel = col < target_size / 2 ? gradient : reflected;
    auto tap = [&](int dx, int dy) {
        return texel(std::clamp(x0 + dx, 0, int(texture_size) - 1),
                     std::clamp(y0 + dy, 0, int(texture_size) - 1));
    };
    const Texel a = tap(0, 0), b = tap(1, 0), c = tap(0, 1), d = tap(1, 1);
    Texel result{};
    for (unsigned k = 0; k < result.size(); ++k) {
        const float top = a[k] + fx * (b[k] - a[k]);
        const float bottom = c[k] + fx * (d[k] - c[k]);
        result[k] = std::uint8_t(std::lround(top + fy * (bottom - top)));
    }
    return result;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    if (!d.bindless) {
        std::printf("SKIP: the device lacks the descriptor indexing features of the bindless set 0\n");
        return 77;
    }

    const std::filesystem::path root_dir = LEONIDA_TEST_M2_DIR;
    const auto vs_program = read_file(root_dir / "fssm" / "vs_m2.fssm");
    const auto ps_program = read_file(root_dir / "fssm" / "ps_m2.fssm");
    EXPECT(vs_program.size() >= 4 && ps_program.size() >= 4);
    leonida::ShaderLibrary library;
    EXPECT(library.load((root_dir / "spirv").string()));
    const auto* vs_entry = library.find("vs", leonida::sha256_hex(vs_program));
    const auto* ps_entry = library.find("ps", leonida::sha256_hex(ps_program));
    EXPECT(vs_entry && ps_entry);
    EXPECT(ps_entry->binding_count > 0 && !vs_entry->binding_count);  // the PS declares set-0 bindings
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    const auto uoff_program = read_file(root_dir / "fssm" / "ps_m2_uoff.fssm");
    EXPECT(uoff_program.size() >= 4 && library.find("ps", leonida::sha256_hex(uoff_program)));
    const auto uoff_container = shader_container(uoff_program);
    // ps_m2_literal builds its S# from immediates: the index lists it under literal_samplers.
    const auto literal_program = read_file(root_dir / "fssm" / "ps_m2_literal.fssm");
    const auto* literal_entry = library.find("ps", leonida::sha256_hex(literal_program));
    EXPECT(literal_program.size() >= 4 && literal_entry);
    const leonida::gfx10::SamplerWords literal_words{0x92u, 0xc00000u, 0x14500000u, 0u};
    EXPECT(literal_entry->literal_samplers.size() == 1 && literal_entry->literal_samplers[0] == literal_words);
    const auto literal_container = shader_container(literal_program);

    std::vector<std::uint8_t> pixels(std::size_t(target_pitch) * target_size), control_pixels = pixels, linear_pixels = pixels;
    unsigned image_hits = 0, sampler_hits = 0;
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
        info.shader_dir = (root_dir / "spirv").string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        EXPECT(rt::d3d12::bindless_set(device.p) != VK_NULL_HANDLE);

        // Root signature: [0] a table of one SRV (s0:s1), [1] a table of one sampler (s2:s3).
        D3D12_DESCRIPTOR_RANGE srv_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_DESCRIPTOR_RANGE sampler_range{D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable = {1, &srv_range};
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable = {1, &sampler_range};
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.SampleDesc.Count = 1;
        Ref<ID3D12PipelineState> pso, uoff_pso;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));
        D3D12_DESCRIPTOR_RANGE indexed_range = srv_range;
        indexed_range.NumDescriptors = UINT32_MAX;
        D3D12_ROOT_PARAMETER indexed_params[3] = {params[0], params[1], {}};
        indexed_params[0].DescriptorTable = {1, &indexed_range};
        indexed_params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        indexed_params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        indexed_params[2].Constants = {0, 0, 2};  // s4 = SRV index, s5 = SRV stride in bytes
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC indexed_desc = root_desc;
        indexed_desc.Desc_1_0.NumParameters = 3;
        indexed_desc.Desc_1_0.pParameters = indexed_params;
        const auto indexed_blob = serialize_root_signature(indexed_desc);
        EXPECT(!indexed_blob.empty());
        Ref<ID3D12RootSignature> indexed_root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, indexed_blob.data(), indexed_blob.size(),
                                                    IID_PPV_ARGS(&indexed_root.p))));
        pso_desc.pRootSignature = indexed_root.p;
        pso_desc.PS = {uoff_container.data(), uoff_container.size()};
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&uoff_pso.p))));
        {
            // A pipeline with a literal-S# shader registers that S#: before it the hash table has no key for the
            // words (a lookup gives the fallback, 0), after it a sampler index. Nothing else names this S#.
            std::uint32_t key[8];
            std::memcpy(key, literal_words.data(), 16);
            std::fill(key + 4, key + 8, 0xffffffffu);
            EXPECT(rt::d3d12::bindless_hash_lookup(device.p, key) == 0);
            D3D12_GRAPHICS_PIPELINE_STATE_DESC literal_desc = pso_desc;
            literal_desc.pRootSignature = root.p;
            literal_desc.PS = {literal_container.data(), literal_container.size()};
            Ref<ID3D12PipelineState> literal_pso;
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&literal_desc, IID_PPV_ARGS(&literal_pso.p))));
            EXPECT(rt::d3d12::bindless_hash_lookup(device.p, key) != 0);
        }

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC texture_desc{};
        texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture_desc.Width = texture_desc.Height = texture_size;
        texture_desc.DepthOrArraySize = 1;
        texture_desc.MipLevels = 1;
        texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture_desc.SampleDesc.Count = 1;
        Ref<ID3D12Resource> textures[2], upload, readback, readback2, target;
        for (auto& texture : textures)
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&texture.p))));
        auto rt_desc = texture_desc;
        rt_desc.Width = rt_desc.Height = target_size;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p))));
        // The upload buffer: texture A at 0, texture B at 2048 (8 rows of the 256-byte pitch), the indices at 4096.
        constexpr UINT64 b_offset = 2048, index_offset = 4096;
        const auto upload_desc = buffer_desc(8192);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto readback_desc = buffer_desc(std::size_t(target_pitch) * target_size);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback2.p))));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        device->GetCopyableFootprints(&texture_desc, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);
        EXPECT(footprint.Footprint.RowPitch == 256 && footprint.Footprint.RowPitch * texture_size <= b_offset);
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        for (UINT y = 0; y < texture_size; ++y)
            for (UINT x = 0; x < texture_size; ++x) {
                const Texel a = gradient(x, y), b = reflected(x, y);
                std::memcpy(up + y * footprint.Footprint.RowPitch + x * 4, a.data(), 4);
                std::memcpy(up + b_offset + y * footprint.Footprint.RowPitch + x * 4, b.data(), 4);
            }
        const std::uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
        std::memcpy(up + index_offset, indices, sizeof(indices));

        // The two heaps; their start handles come back in RAX (heap slots 9 and 10).
        const D3D12_DESCRIPTOR_HEAP_DESC srv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 3,
                                                           D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> srv_heap, sampler_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&srv_heap_desc, IID_PPV_ARGS(&srv_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&sampler_heap_desc, IID_PPV_ARGS(&sampler_heap.p))));
        const std::uint64_t srv_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(srv_heap.p, 9))(srv_heap.p);
        const std::uint64_t srv_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(srv_heap.p, 10))(srv_heap.p);
        const std::uint64_t sampler_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 9))(sampler_heap.p);
        const std::uint64_t sampler_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 10))(sampler_heap.p);
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(srv_cpu && sampler_cpu && srv_gpu >= 0x10000 && sampler_gpu >= 0x10000);
        EXPECT(srv_gpu < (1ull << 32) && sampler_gpu < (1ull << 32) && step >= 32);

        // The SRVs go into slots 0 and 1 (offset by the increment size), the POINT sampler into the sampler heap.
        for (UINT i = 0; i < 2; ++i) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D = {0, 1, 0, 0.0f};
            device->CreateShaderResourceView(textures[i].p, &srv, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(srv_cpu + i * step)});
        }
        D3D12_SAMPLER_DESC sampler_desc{};
        sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler_desc.MaxAnisotropy = 1;
        sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
        device->CreateSampler(&sampler_desc, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu)});
        // Slot 1: the same sampler, wrapping in U (the control pass picks between the two).
        const UINT sampler_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        EXPECT(sampler_step >= 16);
        sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        device->CreateSampler(&sampler_desc, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu + sampler_step)});
        sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler_desc.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
        device->CreateSampler(&sampler_desc, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu + 2 * sampler_step)});


        // The hash table, after the descriptors were created and before anything has run: a T# is 8 dwords of the
        // heap slot, an S# 4 dwords followed by 0xFFFFFFFF x4. Each of the three has an index, they differ, and
        // bytes nothing was created for have none.
        std::uint32_t key_a[8], key_b[8], key_s[8], key_unknown[8];
        std::memcpy(key_a, reinterpret_cast<const void*>(srv_cpu), 32);
        std::memcpy(key_b, reinterpret_cast<const void*>(srv_cpu + step), 32);
        std::memcpy(key_s, reinterpret_cast<const void*>(sampler_cpu), 16);
        std::fill(key_s + 4, key_s + 8, 0xffffffffu);
        for (unsigned i = 0; i < 8; ++i) key_unknown[i] = 0x5a5a0000u + i;
        EXPECT(std::memcmp(key_a, key_b, 32) != 0);
        EXPECT(std::any_of(key_a, key_a + 8, [](std::uint32_t w) { return w != 0; }));
        const std::uint32_t index_a = rt::d3d12::bindless_hash_lookup(device.p, key_a);
        const std::uint32_t index_b = rt::d3d12::bindless_hash_lookup(device.p, key_b);
        const std::uint32_t index_s = rt::d3d12::bindless_hash_lookup(device.p, key_s);
        std::fprintf(stderr, "hash table: T# A -> %u, T# B -> %u, S# -> %u\n", index_a, index_b, index_s);
        EXPECT(index_a && index_b && index_s && index_a != index_b);
        std::uint32_t key_wrap[8];
        std::memcpy(key_wrap, reinterpret_cast<const void*>(sampler_cpu + sampler_step), 16);
        std::fill(key_wrap + 4, key_wrap + 8, 0xffffffffu);
        const std::uint32_t index_wrap = rt::d3d12::bindless_hash_lookup(device.p, key_wrap);
        EXPECT(index_wrap && index_wrap != index_s);  // two samplers that differ in U have two entries
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, key_unknown) == 0);
        image_hits = (index_a != 0) + (index_b != 0);
        sampler_hits = index_s != 0;

        const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{
            SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
        device->CreateRenderTargetView(target.p, nullptr, rtv);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        // Upload both textures, then make them shader resources and the target a render target.
        for (UINT i = 0; i < 2; ++i) {
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = textures[i].p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = upload.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = footprint;
            src.PlacedFootprint.Offset = i ? b_offset : 0;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        D3D12_RESOURCE_BARRIER barriers[3] = {
            transition(textures[0].p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
            transition(textures[1].p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
            transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)};
        list->ResourceBarrier(3, barriers);
        const FLOAT clear[4] = {1.0f, 0.0f, 1.0f, 1.0f};  // magenta: a pixel the draws missed shows
        list->ClearRenderTargetView(rtv, clear, 0, nullptr);

        list->SetGraphicsRootSignature(root.p);
        ID3D12DescriptorHeap* heaps[] = {srv_heap.p, sampler_heap.p};
        list->SetDescriptorHeaps(2, heaps);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_INDEX_BUFFER_VIEW ibv{upload->GetGPUVirtualAddress() + index_offset, sizeof(indices),
                                          DXGI_FORMAT_R16_UINT};
        list->IASetIndexBuffer(&ibv);
        const D3D12_VIEWPORT viewport{0, 0, float(target_size), float(target_size), 0, 1};
        list->RSSetViewports(1, &viewport);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        auto draw_halves = [&](std::uint64_t sampler) {
            list->SetGraphicsRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{sampler});
            for (UINT half = 0; half < 2; ++half) {
                // The same quad over the whole target; the scissor rectangle keeps one half, and the SRV table names
                // the texture of that half.
                const D3D12_RECT scissor{LONG(half * target_size / 2), 0, LONG((half + 1) * target_size / 2), LONG(target_size)};
                list->RSSetScissorRects(1, &scissor);
                list->SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{srv_gpu + std::uint64_t(half) * step});
                list->DrawIndexedInstanced(6, 1, 0, 0, 0);
            }
        };
        draw_halves(sampler_gpu);

        D3D12_RESOURCE_BARRIER to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, target_size, target_size, 1, target_pitch}};
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        auto to_render = transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &to_render);
        list->ClearRenderTargetView(rtv, clear, 0, nullptr);
        draw_halves(sampler_gpu + 2 * sampler_step);
        list->ResourceBarrier(1, &to_copy);
        dst.pResource = readback2.p;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        std::memcpy(pixels.data(), mapped, pixels.size());

        void* mapped_linear = nullptr;
        EXPECT(SUCCEEDED(readback2->Map(0, nullptr, &mapped_linear)));
        std::memcpy(linear_pixels.data(), mapped_linear, linear_pixels.size());
        readback2->Unmap(0, nullptr);

        // The control pass: U + 1 in the PS, WRAP sampler (slot 1) on the left, CLAMP (slot 0) on the right.
        Ref<ID3D12CommandAllocator> allocator2;
        Ref<ID3D12GraphicsCommandList> list2;
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator2.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator2.p, uoff_pso.p,
                                                   IID_PPV_ARGS(&list2.p))));
        D3D12_RESOURCE_BARRIER back = transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                 D3D12_RESOURCE_STATE_RENDER_TARGET);
        list2->ResourceBarrier(1, &back);
        list2->ClearRenderTargetView(rtv, clear, 0, nullptr);
        list2->SetGraphicsRootSignature(indexed_root.p);
        list2->SetDescriptorHeaps(2, heaps);
        list2->SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{srv_gpu});
        list2->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list2->IASetIndexBuffer(&ibv);
        list2->RSSetViewports(1, &viewport);
        list2->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        for (UINT half = 0; half < 2; ++half) {
            const D3D12_RECT scissor{LONG(half * target_size / 2), 0, LONG((half + 1) * target_size / 2), LONG(target_size)};
            list2->RSSetScissorRects(1, &scissor);
            const UINT srv_args[] = {half, step};
            list2->SetGraphicsRoot32BitConstants(2, 2, srv_args, 0);
            list2->SetGraphicsRootDescriptorTable(
                1, D3D12_GPU_DESCRIPTOR_HANDLE{sampler_gpu + std::uint64_t(half == 0 ? sampler_step : 0)});
            list2->DrawIndexedInstanced(6, 1, 0, 0, 0);
        }
        list2->ResourceBarrier(1, &to_copy);
        dst.pResource = readback2.p;
        list2->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        EXPECT(list2->Close() == S_OK);
        ID3D12CommandList* lists2[] = {list2.p};
        queue->ExecuteCommandLists(1, lists2);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 2)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(2, nullptr)));
        void* mapped2 = nullptr;
        EXPECT(SUCCEEDED(readback2->Map(0, nullptr, &mapped2)));
        std::memcpy(control_pixels.data(), mapped2, control_pixels.size());

        // The entries are still there after the draws.
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, key_a) == index_a);
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, key_s) == index_s);

        for (auto& texture : textures) texture.reset();
        for (auto* r : {&upload, &readback, &readback2, &target}) r->reset();
        pso.reset();
        uoff_pso.reset();
        list2.reset();
        allocator2.reset();
        indexed_root.reset();
        root.reset();
        srv_heap.reset();
        sampler_heap.reset();
        rtv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    unsigned mismatches = 0, linear_mismatches = 0;
    int max_linear_error = 0;
    for (UINT row = 0; row < target_size; ++row) {
        for (UINT col = 0; col < target_size; ++col) {
            const UINT tx = col / (target_size / texture_size), ty = row / (target_size / texture_size);
            const Texel want = col < target_size / 2 ? gradient(tx, ty) : reflected(tx, ty);
            const std::uint8_t* got = &pixels[std::size_t(row) * target_pitch + col * 4];
            const Texel filtered = linear_clamp(col, row);
            const auto* linear = &linear_pixels[std::size_t(row) * target_pitch + col * 4];
            for (unsigned k = 0; k < filtered.size(); ++k) {
                const int error = std::abs(int(linear[k]) - filtered[k]);
                max_linear_error = std::max(max_linear_error, error);
                if (error > 1 && linear_mismatches++ < 8)
                    std::fprintf(stderr, "LINEAR (%u, %u) channel %u: got %u want %u\n",
                                 col, row, k, unsigned(linear[k]), unsigned(filtered[k]));
            }
            if (std::memcmp(got, want.data(), 4) == 0) continue;
            if (mismatches++ < 8)
                std::fprintf(stderr, "pixel (%u, %u): got %u,%u,%u,%u want %u,%u,%u,%u\n", col, row, got[0], got[1],
                             got[2], got[3], want[0], want[1], want[2], want[3]);
        }
    }
    EXPECT(!mismatches);
    EXPECT(!linear_mismatches);
    std::printf("M2 LINEAR: %u pixels, maximum RGBA8 channel error %d\n", target_size * target_size, max_linear_error);

    // The control pass: left half the gradient (WRAP: U + 1 wraps back), right half the reflected texture's last
    // column per row (CLAMP: U + 1 is past the edge). Pixels that differ from the plain pass tell the clamp was used.
    unsigned control_mismatches = 0, differing = 0;
    for (UINT row = 0; row < target_size; ++row) {
        for (UINT col = 0; col < target_size; ++col) {
            const UINT tx = col / (target_size / texture_size), ty = row / (target_size / texture_size);
            const Texel want = col < target_size / 2 ? gradient(tx, ty) : reflected(texture_size - 1, ty);
            const std::uint8_t* got = &control_pixels[std::size_t(row) * target_pitch + col * 4];
            const std::uint8_t* plain = &pixels[std::size_t(row) * target_pitch + col * 4];
            differing += std::memcmp(got, plain, 4) != 0;
            if (std::memcmp(got, want.data(), 4) == 0) continue;
            if (control_mismatches++ < 8)
                std::fprintf(stderr, "control pixel (%u, %u): got %u,%u,%u,%u want %u,%u,%u,%u\n", col, row, got[0],
                             got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
        }
    }
    EXPECT(!control_mismatches);
    EXPECT(differing == 1536);  // the right half's columns 4..6 of texels: 3 * 8 * 64
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("M2: a textured quad through descriptor heaps, SRVs, a POINT sampler and two root descriptor tables "
                "matched all %u pixels (error 0, and the sampler control: WRAP left, CLAMP right, %u pixels differ from the plain pass): translated PS sampled a T#/S# found in the hash table (%u image "
                "entries, %u sampler entry)\n",
                target_size * target_size, differing, image_hits, sampler_hits);
    return 0;
}
