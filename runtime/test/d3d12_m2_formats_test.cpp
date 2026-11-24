// Milestone M2 (docs/d3d12_vulkan.md section 6), the part d3d12_m2_textured_test leaves open: texture formats other
// than R8G8B8A8_UNORM, mips and arrays through a PSO. The same descriptor path (heaps with start handles from heap
// slots 9/10, CreateShaderResourceView, CreateSampler, two root descriptor tables, translated vs_m2/ps_m2) samples:
//   - bc1: an 8x8 DXGI_FORMAT_BC1_UNORM texture with 4 mips (8, 4, 2, 1), four-colour blocks built by hand;
//   - bc7: the same in DXGI_FORMAT_BC7_UNORM, mode-6 blocks built by hand;
//   - bc6h: identical mode-11 blocks in BC6H_UF16 and BC6H_SF16, sampled into an unclamped RGBA32F target;
//   - arr: an 8x8 R8G8B8A8_UNORM 2D array of 3 slices, read by ps_m2_array.asm (dim:SQ_RSRC_IMG_2D_ARRAY, slice 1.0).
// Every subresource is uploaded at the layout GetCopyableFootprints returns, with one CopyTextureRegion each.
// The quad magnifies (64x64 pixels), so a POINT sampler with MinLOD = MaxLOD = k reads level k of the view: four
// samplers pin levels 0..3. A view with MostDetailedMip 1 checks that the level is relative to the view, and two
// array views (slices 0..2 and 1..2) that the slice coordinate is. The blocks only use the endpoint colours (BC1
// indices 0/1, BC7 indices 0/15), whose decode is exact on every GPU, so the BC1, BC7 and array cases are compared byte for byte with
// the CPU (error 0). Each level and slice holds different texels, so a draw that read the wrong one fails.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation (0 errors required).
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#ifndef LEONIDA_TEST_M2F_DIR
#error "LEONIDA_TEST_M2F_DIR must name the folder build.py wrote (spirv/index.json, fssm/*.fssm)"
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
constexpr UINT mip_count = 4;    // 8, 4, 2, 1
constexpr UINT slice_count = 3;
constexpr UINT target_size = 64;
constexpr UINT target_pitch = target_size * 4;  // 256

enum Kind : UINT { bc1, bc7, arr, bc6u, bc6s, kind_count };

// BC6H mode 11: prefix 00011 (LSB first), then endpoint 0 RGB = (512, 511, 0)
// in consecutive 10-bit fields. Endpoint 1 and all indices are zero.
constexpr std::array<std::uint8_t, 16> bc6_block{0x03, 0xc0, 0xff};
// UF16 unquantizes the first two channels to 32800 and 32736, then scales by
// 31/64 to half bits 0x3e0f and 0x3df0. SF16 sign-extends to -512 and +511,
// the two extremes, yielding half bits 0xfbff and 0x7bff. BC6H supplies alpha 1.
using FloatTexel = std::array<float, 4>;
constexpr FloatTexel unsigned_endpoint{1.5146484375f, 1.484375f, 0, 1};
constexpr FloatTexel signed_endpoint{-65504, 65504, 0, 1};
constexpr auto float_format = DXGI_FORMAT_R32G32B32A32_FLOAT;

// BC1 block (bx, by) of mip m: c0 > c1 (four-colour mode, red is the top field and r0 >= 16 > r1), texel index
// (x + y + m) & 1, so every texel is one endpoint exactly.
struct Bc1Colors { UINT r0, g0, b0, r1, g1, b1; };
Bc1Colors bc1_colors(UINT m, UINT bx, UINT by) {
    return {16 + 4 * m + 2 * bx + by, 8 + 12 * m + 3 * bx + by, 31 - 5 * m - bx - 2 * by,
            2 + m, 60 - 10 * m - bx, 4 + bx + by + m};
}
std::array<std::uint8_t, 8> bc1_block(UINT m, UINT bx, UINT by) {
    const auto c = bc1_colors(m, bx, by);
    const std::uint16_t c0 = std::uint16_t(c.r0 << 11 | c.g0 << 5 | c.b0), c1 = std::uint16_t(c.r1 << 11 | c.g1 << 5 | c.b1);
    std::uint32_t indices = 0;
    for (UINT i = 0; i < 16; ++i) indices |= std::uint32_t(((i % 4) + (i / 4) + m) & 1) << (2 * i);
    std::array<std::uint8_t, 8> block{};
    std::memcpy(block.data(), &c0, 2);
    std::memcpy(block.data() + 2, &c1, 2);
    std::memcpy(block.data() + 4, &indices, 4);
    return block;
}
Texel bc1_texel(UINT m, UINT x, UINT y) {
    const auto c = bc1_colors(m, x / 4, y / 4);
    const bool second = ((x % 4) + (y % 4) + m) & 1;
    const UINT r5 = second ? c.r1 : c.r0, g6 = second ? c.g1 : c.g0, b5 = second ? c.b1 : c.b0;
    return {std::uint8_t(r5 << 3 | r5 >> 2), std::uint8_t(g6 << 2 | g6 >> 4), std::uint8_t(b5 << 3 | b5 >> 2), 255};
}

// BC7 mode-6 block (bx, by) of mip m: 7-bit endpoints with p-bit 1, alpha 255; texel i takes index 0 (endpoint 0)
// or 15 (endpoint 1), whose weights 0 and 64 give the endpoints exactly. Texel 0 is the anchor: index 0.
struct Bc7Endpoints { UINT r0, g0, b0, r1, g1, b1; };
Bc7Endpoints bc7_endpoints(UINT m, UINT bx, UINT by) {
    return {10 + 20 * m + 7 * bx, 100 - 15 * m - 9 * by, 30 + 30 * bx + 15 * by, 120 - 10 * m, 20 + 20 * bx + 5 * by,
            90 - 20 * m};
}
bool bc7_second(UINT m, UINT x, UINT y) {  // x, y inside the block
    return (x || y) && (x + 2 * y + m) % 3 != 0;
}
std::array<std::uint8_t, 16> bc7_block(UINT m, UINT bx, UINT by) {
    std::array<std::uint8_t, 16> block{};
    UINT at = 0;
    const auto put = [&](UINT value, UINT bits) {
        for (UINT i = 0; i < bits; ++i, ++at)
            if (value >> i & 1) block[at / 8] |= std::uint8_t(1u << (at % 8));
    };
    const auto e = bc7_endpoints(m, bx, by);
    put(1u << 6, 7);  // mode 6
    for (UINT v : {e.r0, e.r1, e.g0, e.g1, e.b0, e.b1, 127u, 127u}) put(v, 7);
    put(1, 1);  // P0
    put(1, 1);  // P1
    for (UINT i = 0; i < 16; ++i) put(bc7_second(m, i % 4, i / 4) ? 15 : 0, i ? 4 : 3);
    return at == 128 ? block : std::array<std::uint8_t, 16>{};
}
Texel bc7_texel(UINT m, UINT x, UINT y) {
    const auto e = bc7_endpoints(m, x / 4, y / 4);
    const bool second = bc7_second(m, x % 4, y % 4);
    const UINT r = second ? e.r1 : e.r0, g = second ? e.g1 : e.g0, b = second ? e.b1 : e.b0;
    return {std::uint8_t(r << 1 | 1), std::uint8_t(g << 1 | 1), std::uint8_t(b << 1 | 1), 255};
}

Texel arr_texel(UINT s, UINT x, UINT y) {
    return {std::uint8_t(x * 30 + s * 7), std::uint8_t(y * 30), std::uint8_t(60 + s * 90), 255};
}

Texel texel(Kind kind, UINT level, UINT slice, UINT x, UINT y) {
    return kind == bc1 ? bc1_texel(level, x, y) : kind == bc7 ? bc7_texel(level, x, y) : arr_texel(slice, x, y);
}

// The texel pixel (px, py) of the target reads at a level: the quad's UVs span the level.
Texel expected(Kind kind, UINT level, UINT slice, UINT px, UINT py) {
    const UINT w = std::max(1u, texture_size >> level);
    return texel(kind, level, slice, px * w / target_size, py * w / target_size);
}

unsigned differing(Kind ka, UINT la, UINT sa, Kind kb, UINT lb, UINT sb) {
    unsigned n = 0;
    for (UINT y = 0; y < target_size; ++y)
        for (UINT x = 0; x < target_size; ++x) n += expected(ka, la, sa, x, y) != expected(kb, lb, sb, x, y);
    return n;
}

struct Case {
    const char* name;
    bool array_pso;
    UINT srv, sampler;  // heap slots
    Kind kind;
    UINT level, slice;  // what the draw must read
};

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

    const Case cases[] = {
        {"bc6h UF16", false, 5, 0, bc6u, 0, 0},
        {"bc6h SF16", false, 6, 0, bc6s, 0, 0},
        {"bc1 mip 0", false, 0, 0, bc1, 0, 0},
        {"bc1 mip 1", false, 0, 1, bc1, 1, 0},
        {"bc1 mip 2", false, 0, 2, bc1, 2, 0},
        {"bc1 mip 3", false, 0, 3, bc1, 3, 0},
        {"bc7 mip 0", false, 1, 0, bc7, 0, 0},
        {"bc7 mip 1", false, 1, 1, bc7, 1, 0},
        {"bc7 mip 2", false, 1, 2, bc7, 2, 0},
        {"bc7 mip 3", false, 1, 3, bc7, 3, 0},
        {"bc1 view from mip 1, lod 0", false, 2, 0, bc1, 1, 0},
        {"bc1 view from mip 1, lod 2", false, 2, 2, bc1, 3, 0},
        {"array view of slices 0..2, slice 1", true, 3, 0, arr, 0, 1},
        {"array view of slices 1..2, slice 1", true, 4, 0, arr, 0, 2},
    };
    // The cases can only fail when the wrong level or slice is read if those hold other texels.
    for (UINT m = 1; m < mip_count; ++m) {
        EXPECT(differing(bc1, m - 1, 0, bc1, m, 0) > 0);
        EXPECT(differing(bc7, m - 1, 0, bc7, m, 0) > 0);
    }
    EXPECT(differing(arr, 0, 0, arr, 0, 1) > 0 && differing(arr, 0, 1, arr, 0, 2) > 0);
    EXPECT(bc7_block(0, 0, 0)[0] == 0x40);  // the encoder wrote all 128 bits (mode 6: bit 6 of byte 0)

    const std::filesystem::path root_dir = LEONIDA_TEST_M2F_DIR;
    const auto vs_program = read_file(root_dir / "fssm" / "vs_m2.fssm");
    const auto ps_program = read_file(root_dir / "fssm" / "ps_m2.fssm");
    const auto array_program = read_file(root_dir / "fssm" / "ps_m2_array.fssm");
    EXPECT(vs_program.size() >= 4 && ps_program.size() >= 4 && array_program.size() >= 4);
    leonida::ShaderLibrary library;
    EXPECT(library.load((root_dir / "spirv").string()));
    const auto* vs_entry = library.find("vs", leonida::sha256_hex(vs_program));
    const auto* ps_entry = library.find("ps", leonida::sha256_hex(ps_program));
    const auto* array_entry = library.find("ps", leonida::sha256_hex(array_program));
    EXPECT(vs_entry && ps_entry && array_entry);
    EXPECT(ps_entry->binding_count > 0 && array_entry->binding_count > 0);
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program),
               array_container = shader_container(array_program);

    unsigned failed = 0;
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

        // Root signature: [0] a table of one SRV (s0:s1), [1] a table of one sampler (s2:s3), as in the M2 test.
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
        Ref<ID3D12PipelineState> pso, array_pso, float_pso;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));
        pso_desc.PS = {array_container.data(), array_container.size()};
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&array_pso.p))));

        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.RTVFormats[0] = float_format;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&float_pso.p))));

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC descs[kind_count] = {};
        for (auto& desc : descs) {
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width = desc.Height = texture_size;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = mip_count;
            desc.SampleDesc.Count = 1;
        }
        descs[bc1].Format = DXGI_FORMAT_BC1_UNORM;
        descs[bc7].Format = DXGI_FORMAT_BC7_UNORM;
        descs[arr].Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        descs[arr].DepthOrArraySize = slice_count;
        descs[arr].MipLevels = 1;
        descs[bc6u].Format = DXGI_FORMAT_BC6H_UF16;
        descs[bc6s].Format = DXGI_FORMAT_BC6H_SF16;
        descs[bc6u].MipLevels = descs[bc6s].MipLevels = 1;
        Ref<ID3D12Resource> textures[kind_count], uploads[kind_count], index_buffer, readback, target;
        for (UINT k = 0; k < kind_count; ++k)
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &descs[k],
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&textures[k].p))));
        auto rt_desc = descs[arr];
        rt_desc.Width = rt_desc.Height = target_size;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p))));
        Ref<ID3D12Resource> float_target;
        rt_desc.Format = float_format;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&float_target.p))));
        const auto readback_desc = buffer_desc(sizeof(FloatTexel) * target_size * target_size);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        const std::uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
        const auto index_desc = buffer_desc(256);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &index_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&index_buffer.p))));
        void* index_data = nullptr;
        EXPECT(SUCCEEDED(index_buffer->Map(0, nullptr, &index_data)));
        std::memcpy(index_data, indices, sizeof(indices));

        // Each texture's upload buffer is laid out by GetCopyableFootprints; a BC row is a row of 4x4 blocks.
        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts[kind_count];
        for (UINT k = 0; k < kind_count; ++k) {
            const UINT count = descs[k].MipLevels * descs[k].DepthOrArraySize;
            layouts[k].resize(count);
            std::vector<UINT> rows(count);
            std::vector<UINT64> row_bytes(count);
            UINT64 total = 0;
            device->GetCopyableFootprints(&descs[k], 0, count, 0, layouts[k].data(), rows.data(), row_bytes.data(),
                                          &total);
            EXPECT(total && total != UINT64_MAX);
            const auto upload_desc = buffer_desc(total);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                             IID_PPV_ARGS(&uploads[k].p))));
            std::uint8_t* up = nullptr;
            EXPECT(SUCCEEDED(uploads[k]->Map(0, nullptr, reinterpret_cast<void**>(&up))));
            for (UINT i = 0; i < count; ++i) {
                const UINT level = i % descs[k].MipLevels, slice = i / descs[k].MipLevels;
                std::uint8_t* base = up + layouts[k][i].Offset;
                const UINT pitch = layouts[k][i].Footprint.RowPitch;
                if (k == arr) {
                    EXPECT(rows[i] == texture_size && row_bytes[i] == texture_size * 4);
                    for (UINT y = 0; y < texture_size; ++y)
                        for (UINT x = 0; x < texture_size; ++x)
                            std::memcpy(base + y * pitch + x * 4, arr_texel(slice, x, y).data(), 4);
                    continue;
                }
                const UINT block_bytes = k == bc1 ? 8 : 16;
                const UINT blocks = (std::max(1u, texture_size >> level) + 3) / 4;
                EXPECT(rows[i] == blocks && row_bytes[i] == UINT64(blocks) * block_bytes);
                for (UINT by = 0; by < blocks; ++by)
                    for (UINT bx = 0; bx < blocks; ++bx) {
                        std::uint8_t* at = base + by * pitch + bx * block_bytes;
                        if (k == bc1) std::memcpy(at, bc1_block(level, bx, by).data(), 8);
                        else if (k == bc7) std::memcpy(at, bc7_block(level, bx, by).data(), 16);
                        else std::memcpy(at, bc6_block.data(), bc6_block.size());
                    }
            }
        }

        // The heaps; their start handles come back in RAX (heap slots 9 and 10).
        const D3D12_DESCRIPTOR_HEAP_DESC srv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 7,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, mip_count,
                                                           D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> srv_heap, sampler_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&srv_heap_desc, IID_PPV_ARGS(&srv_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&sampler_heap_desc, IID_PPV_ARGS(&sampler_heap.p))));
        const std::uint64_t srv_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(srv_heap.p, 9))(srv_heap.p);
        const std::uint64_t srv_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(srv_heap.p, 10))(srv_heap.p);
        const std::uint64_t sampler_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 9))(sampler_heap.p);
        const std::uint64_t sampler_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 10))(sampler_heap.p);
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const UINT sampler_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        EXPECT(srv_cpu && sampler_cpu && srv_gpu >= 0x10000 && sampler_gpu >= 0x10000);
        EXPECT(srv_gpu < (1ull << 32) && sampler_gpu < (1ull << 32) && step >= 32 && sampler_step >= 16);

        // SRV slots: 0 bc1 (mips 0..3), 1 bc7 (mips 0..3), 2 bc1 from mip 1 (mips 1..3), 3 arr slices 0..2,
        // 4 arr slices 1..2.
        const auto make_srv = [&](UINT slot_index, Kind kind, UINT first, UINT count) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = descs[kind].Format;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            if (kind == arr) {
                srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                srv.Texture2DArray = {0, 1, first, count, 0, 0.0f};
            } else {
                srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                srv.Texture2D = {first, count, 0, 0.0f};
            }
            device->CreateShaderResourceView(textures[kind].p, &srv,
                                             D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(srv_cpu + slot_index * step)});
        };
        make_srv(0, bc1, 0, mip_count);
        make_srv(1, bc7, 0, mip_count);
        make_srv(2, bc1, 1, mip_count - 1);
        make_srv(3, arr, 0, slice_count);
        make_srv(4, arr, 1, slice_count - 1);
        make_srv(5, bc6u, 0, 1);
        make_srv(6, bc6s, 0, 1);
        // Sampler slot k: POINT, CLAMP, MinLOD = MaxLOD = k.
        for (UINT k = 0; k < mip_count; ++k) {
            D3D12_SAMPLER_DESC sampler_desc{};
            sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
            sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            sampler_desc.MaxAnisotropy = 1;
            sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
            sampler_desc.MinLOD = sampler_desc.MaxLOD = float(k);
            device->CreateSampler(&sampler_desc, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu + k * sampler_step)});
        }

        const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{
            SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
        device->CreateRenderTargetView(target.p, nullptr, rtv);
        const D3D12_CPU_DESCRIPTOR_HANDLE float_rtv{
            rtv.ptr + device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)};
        device->CreateRenderTargetView(float_target.p, nullptr, float_rtv);

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
        UINT64 fence_value = 0;
        const auto submit = [&]() -> bool {
            if (list->Close() != S_OK) return false;
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            if (rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) return false;
            return SUCCEEDED(queue->Signal(fence.p, ++fence_value)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr));
        };

        // Upload: one CopyTextureRegion per subresource, at its footprint.
        for (UINT k = 0; k < kind_count; ++k)
            for (UINT i = 0; i < layouts[k].size(); ++i) {
                D3D12_TEXTURE_COPY_LOCATION dst{};
                dst.pResource = textures[k].p;
                dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                dst.SubresourceIndex = i;
                D3D12_TEXTURE_COPY_LOCATION src{};
                src.pResource = uploads[k].p;
                src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                src.PlacedFootprint = layouts[k][i];
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
        D3D12_RESOURCE_BARRIER barriers[kind_count + 2];
        for (UINT k = 0; k < kind_count; ++k)
            barriers[k] = transition(textures[k].p, D3D12_RESOURCE_STATE_COPY_DEST,
                                     D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        barriers[kind_count] = transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        barriers[kind_count + 1] = transition(float_target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(UINT(std::size(barriers)), barriers);
        EXPECT(submit());

        const D3D12_INDEX_BUFFER_VIEW ibv{index_buffer->GetGPUVirtualAddress(), sizeof(indices), DXGI_FORMAT_R16_UINT};
        const D3D12_VIEWPORT viewport{0, 0, float(target_size), float(target_size), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(target_size), LONG(target_size)};
        for (const Case& c : cases) {
            const bool hdr = c.kind == bc6u || c.kind == bc6s;
            auto* draw_target = hdr ? float_target.p : target.p;
            const auto draw_rtv = hdr ? float_rtv : rtv;
            const UINT pitch = hdr ? target_size * UINT(sizeof(FloatTexel)) : target_pitch;
            std::vector<std::uint8_t> pixels(std::size_t(pitch) * target_size);
            EXPECT(SUCCEEDED(allocator->Reset()));
            EXPECT(SUCCEEDED(list->Reset(allocator.p, hdr ? float_pso.p : c.array_pso ? array_pso.p : pso.p)));
            const FLOAT clear[4] = {1.0f, 0.0f, 1.0f, 1.0f};  // magenta: a pixel the draw missed shows
            list->ClearRenderTargetView(draw_rtv, clear, 0, nullptr);
            list->SetGraphicsRootSignature(root.p);
            ID3D12DescriptorHeap* heaps[] = {srv_heap.p, sampler_heap.p};
            list->SetDescriptorHeaps(2, heaps);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list->IASetIndexBuffer(&ibv);
            list->RSSetViewports(1, &viewport);
            list->RSSetScissorRects(1, &scissor);
            list->OMSetRenderTargets(1, &draw_rtv, FALSE, nullptr);
            list->SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{srv_gpu + std::uint64_t(c.srv) * step});
            list->SetGraphicsRootDescriptorTable(
                1, D3D12_GPU_DESCRIPTOR_HANDLE{sampler_gpu + std::uint64_t(c.sampler) * sampler_step});
            list->DrawIndexedInstanced(6, 1, 0, 0, 0);
            D3D12_RESOURCE_BARRIER to_copy = transition(draw_target, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                        D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &to_copy);
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = readback.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = {0, {hdr ? float_format : DXGI_FORMAT_R8G8B8A8_UNORM, target_size, target_size, 1, pitch}};
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = draw_target;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            D3D12_RESOURCE_BARRIER back = transition(draw_target, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                     D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ResourceBarrier(1, &back);
            EXPECT(submit());
            void* mapped = nullptr;
            EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
            std::memcpy(pixels.data(), mapped, pixels.size());
            readback->Unmap(0, nullptr);

            unsigned mismatches = 0;
            for (UINT y = 0; y < target_size; ++y)
                for (UINT x = 0; x < target_size; ++x) {
                    if (hdr) {
                        const auto& want = c.kind == bc6u ? unsigned_endpoint : signed_endpoint;
                        FloatTexel got;
                        std::memcpy(got.data(), &pixels[y * pitch + x * sizeof(got)], sizeof(got));
                        if (got != want && mismatches++ < 4)
                            std::fprintf(stderr, "%s: pixel (%u, %u) got %g,%g,%g,%g want %g,%g,%g,%g\n",
                                         c.name, x, y, got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                        continue;
                    }
                    const Texel want = expected(c.kind, c.level, c.slice, x, y);
                    const std::uint8_t* got = &pixels[std::size_t(y) * target_pitch + x * 4];
                    if (std::memcmp(got, want.data(), 4) == 0) continue;
                    if (mismatches++ < 4)
                        std::fprintf(stderr, "%s: pixel (%u, %u) got %u,%u,%u,%u want %u,%u,%u,%u\n", c.name, x, y,
                                     got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                }
            std::printf("%s %s (%u of %u pixels differ)\n", mismatches ? "FAIL" : "PASS", c.name, mismatches,
                        target_size * target_size);
            failed += mismatches != 0;
        }

        index_buffer->Unmap(0, nullptr);
        for (auto& upload : uploads) upload.reset();
        for (auto& texture : textures) texture.reset();
        index_buffer.reset();
        readback.reset();
        float_target.reset();
        float_pso.reset();
        target.reset();
        pso.reset();
        array_pso.reset();
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

    EXPECT(!failed);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("d3d12_m2_formats: PASS (%zu cases)\n", std::size(cases));
    return 0;
}
