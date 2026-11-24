// First light through the port's D3D12 layer (docs/first_light.md section 5): the real model and the real translated
// VS/PS that tools/first_light.py assembled from the user's package, drawn through the Xbox-ABI device proxy over the
// Vulkan device. Package-gated: it reads everything at run time from LEONIDA_FIRST_LIGHT_DIR (the first_light.py
// output folder) and prints SKIP and exits 77 when that is unset. No game data is in the repository.
//   - CreateGraphicsPipelineState gets the game's own DXBC containers (vs.dxbc, ps.dxbc) and finds both stages by the
//     SHA-256 of the FSSM program inside them, in the translator index first_light.py wrote (index.json, vs.spv,
//     ps.spv); nothing bypasses the lookup. The hashes must also be the ones manifest.json records;
//   - the input layout (POSITION float3, TEXCOORD float2, stride 20) is matched to the VS input signature by
//     semantic; mesh.bin's vertices and indices go through an upload heap into DEFAULT buffers in the GPU address
//     arena, and one DrawIndexedInstanced draws the whole mesh with no depth test and no culling, as vk_harness does;
//   - the R32G32B32A32_FLOAT target is cleared to -1 (vk_harness's clear) and read back through CopyTextureRegion.
// The oracle is vk_harness's (tests/test_first_light.py): coverage in the manifest's window, the clear corner keeps
// -1, and the centre pixel is the UV the mesh's last covering triangle gives there. D3D12 clip space is y-up and the
// layer flips the viewport, while vk_harness draws with a y-down viewport, so vk_harness row r is row H-1-r here.
// When the folder also has vk_harness's image.raw, every pixel must equal it with the rows flipped.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX
using Pixel = std::array<float, 4>;

namespace {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

double number(const leonida::Json& object, std::string_view key, double fallback = -1) {
    const leonida::Json* value = object.find(key);
    return value && value->kind == leonida::Json::Kind::number ? value->number : fallback;
}

// [x, y] from the manifest, or false when it is missing.
bool pixel_of(const leonida::Json& object, std::string_view key, std::array<double, 2>& out) {
    const leonida::Json* value = object.find(key);
    if (!value || value->kind != leonida::Json::Kind::array || value->array.size() != 2) return false;
    for (int i = 0; i < 2; ++i) {
        if (value->array[i].kind != leonida::Json::Kind::number) return false;
        out[i] = value->array[i].number;
    }
    return true;
}

std::string stage_hash(const leonida::Json& manifest, std::string_view stage) {
    const leonida::Json* entry = manifest.find(stage);
    const leonida::Json* hash = entry ? entry->find("fssm_sha256") : nullptr;
    return hash && hash->kind == leonida::Json::Kind::string ? hash->string : std::string();
}

}  // namespace

int main() {
    const char* dir_env = std::getenv("LEONIDA_FIRST_LIGHT_DIR");
    if (!dir_env || !*dir_env) {
        std::printf("SKIP: LEONIDA_FIRST_LIGHT_DIR is not set (run tools/first_light.py first)\n");
        return 77;
    }
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    // The work folder: the manifest, the mesh, the game's two containers and the translator's index for them.
    const std::filesystem::path dir = dir_env;
    const auto manifest_text = read_file(dir / "manifest.json");
    leonida::Json manifest;
    EXPECT(leonida::JsonReader(std::string_view(reinterpret_cast<const char*>(manifest_text.data()),
                                                manifest_text.size()))
               .parse(manifest));
    for (const char* name : {"vs.dxbc", "ps.dxbc", "index.json", "vs.spv", "ps.spv", "mesh.bin"}) {
        if (!std::filesystem::is_regular_file(dir / name)) {
            std::fprintf(stderr, "%s has no %s: re-run tools/first_light.py (it writes the D3D12 inputs too)\n",
                         dir.string().c_str(), name);
            return 1;
        }
    }
    const UINT width = UINT(number(manifest, "width")), height = UINT(number(manifest, "height"));
    EXPECT(width > 0 && width <= 4096 && height > 0 && height <= 4096);
    const UINT pitch = (width * sizeof(Pixel) + 255) / 256 * 256;

    const auto vs_container = read_file(dir / "vs.dxbc"), ps_container = read_file(dir / "ps.dxbc");
    const std::string vs_hash = leonida::sha256_hex(leonida::fssm_program(vs_container));
    const std::string ps_hash = leonida::sha256_hex(leonida::fssm_program(ps_container));
    EXPECT(vs_hash == stage_hash(manifest, "vs") && ps_hash == stage_hash(manifest, "ps"));
    leonida::ShaderLibrary library;
    EXPECT(library.load(dir.string()));
    EXPECT(library.find("vs", vs_hash) && library.find("ps", ps_hash));
    // The lookup is keyed by stage as well as hash: each program must not be found as the other stage.
    EXPECT(!library.find("vs", ps_hash) && !library.find("ps", vs_hash));

    // mesh.bin: u32 vertex_count, u32 index_count, (x, y, z, u, v) f32 per vertex, u32 indices.
    const auto mesh = read_file(dir / "mesh.bin");
    EXPECT(mesh.size() >= 8);
    std::uint32_t vertex_count = 0, index_count = 0;
    std::memcpy(&vertex_count, mesh.data(), 4);
    std::memcpy(&index_count, mesh.data() + 4, 4);
    constexpr UINT stride = 5 * sizeof(float);
    const std::size_t vertex_bytes = std::size_t(vertex_count) * stride, index_bytes = std::size_t(index_count) * 4;
    EXPECT(vertex_count && index_count && mesh.size() == 8 + vertex_bytes + index_bytes);
    EXPECT(vertex_count == number(manifest, "vertex_count") && index_count == number(manifest, "index_count"));
    const std::size_t index_offset = (vertex_bytes + 255) / 256 * 256;

    std::vector<Pixel> pixels(std::size_t(width) * height);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.subgroup = d.subgroup;
        info.shader_dir = dir.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

        // No root parameters: the VS applies no matrix and the PS has no bindings (vk_harness pushes zeros too).
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        const D3D12_INPUT_ELEMENT_DESC elements[2] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.InputLayout = {elements, 2};
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        pso_desc.SampleDesc.Count = 1;
        {
            // The pixel shader's container as the VS: the proxy finds no vs program with that hash, so no PSO.
            D3D12_GRAPHICS_PIPELINE_STATE_DESC swapped = pso_desc;
            swapped.VS = pso_desc.PS;
            Ref<ID3D12PipelineState> wrong;
            EXPECT(FAILED(device->CreateGraphicsPipelineState(&swapped, IID_PPV_ARGS(&wrong.p))) && !wrong.p);
        }
        Ref<ID3D12PipelineState> pso;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> vertex_buffer, index_buffer, upload, readback, target;
        const auto vb_desc = buffer_desc(vertex_bytes), ib_desc = buffer_desc(index_bytes);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &vb_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&vertex_buffer.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &ib_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&index_buffer.p))));
        const auto upload_desc = buffer_desc(index_offset + index_bytes);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto readback_desc = buffer_desc(UINT64(pitch) * height);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = width;
        rt_desc.Height = height;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p))));

        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memcpy(up, mesh.data() + 8, vertex_bytes);
        std::memcpy(up + index_offset, mesh.data() + 8 + vertex_bytes, index_bytes);

        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
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

        list->CopyBufferRegion(vertex_buffer.p, 0, upload.p, 0, vertex_bytes);
        list->CopyBufferRegion(index_buffer.p, 0, upload.p, index_offset, index_bytes);
        D3D12_RESOURCE_BARRIER barriers[3] = {
            transition(vertex_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER),
            transition(index_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDEX_BUFFER),
            transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)};
        list->ResourceBarrier(3, barriers);
        const FLOAT clear[4] = {-1, -1, -1, -1};  // vk_harness's clear: marks untouched pixels
        list->ClearRenderTargetView(rtv, clear, 0, nullptr);

        list->SetGraphicsRootSignature(root.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VERTEX_BUFFER_VIEW vbv{vertex_buffer->GetGPUVirtualAddress(), UINT(vertex_bytes), stride};
        list->IASetVertexBuffers(0, 1, &vbv);
        const D3D12_INDEX_BUFFER_VIEW ibv{index_buffer->GetGPUVirtualAddress(), UINT(index_bytes),
                                          DXGI_FORMAT_R32_UINT};
        list->IASetIndexBuffer(&ibv);
        const D3D12_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(width), LONG(height)};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        list->DrawIndexedInstanced(index_count, 1, 0, 0, 0);

        D3D12_RESOURCE_BARRIER to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_R32G32B32A32_FLOAT, width, height, 1, pitch}};
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        std::uint8_t* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
        for (UINT row = 0; row < height; ++row)
            std::memcpy(&pixels[std::size_t(row) * width], mapped + std::size_t(row) * pitch, width * sizeof(Pixel));

        for (auto* r : {&vertex_buffer, &index_buffer, &upload, &readback, &target}) r->reset();
        pso.reset();
        root.reset();
        rtv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    // vk_harness's (x, y) is (x, H-1-y) here: same clip-space point, opposite viewport direction.
    const auto at = [&](UINT x, UINT vk_y) -> const Pixel& { return pixels[std::size_t(height - 1 - vk_y) * width + x]; };
    unsigned covered = 0;
    for (const Pixel& p : pixels) covered += p[3] > 0.5f;
    const double ratio = double(covered) / (double(width) * height);
    std::printf("covered %u/%u (%.3f)\n", covered, width * height, ratio);
    EXPECT(ratio >= number(manifest, "min_coverage", 0.0) && ratio <= number(manifest, "max_coverage", 1.0));
    std::array<double, 2> clear_xy{}, centre_xy{}, centre_uv{};
    if (pixel_of(manifest, "clear_pixel", clear_xy)) {
        const Pixel& p = at(UINT(clear_xy[0]), UINT(clear_xy[1]));
        const float want = float(number(manifest, "clear", -1.0));
        std::printf("clear pixel (%u, %u): %g %g %g %g\n", UINT(clear_xy[0]), height - 1 - UINT(clear_xy[1]), p[0],
                    p[1], p[2], p[3]);
        for (float c : p) EXPECT(std::fabs(c - want) <= 1e-5f);
    }
    if (pixel_of(manifest, "centre_uv", centre_uv)) {
        if (!pixel_of(manifest, "centre_pixel", centre_xy)) centre_xy = {double(width / 2), double(height / 2)};
        const Pixel& p = at(UINT(centre_xy[0]), UINT(centre_xy[1]));
        std::printf("centre pixel (%u, %u): %.5f %.5f %.5f %.5f (mesh UV %.5f %.5f)\n", UINT(centre_xy[0]),
                    height - 1 - UINT(centre_xy[1]), p[0], p[1], p[2], p[3], centre_uv[0], centre_uv[1]);
        const double want[4] = {centre_uv[0], centre_uv[1], 0.0, 1.0};
        for (int c = 0; c < 4; ++c) EXPECT(std::fabs(p[c] - want[c]) <= 2e-3);
    }
    // first_light.py's probes: pixels where U and V differ, so a swapped or mirrored UV cannot pass (#1906).
    if (const leonida::Json* probes = manifest.find("uv_probes"); probes && probes->kind == leonida::Json::Kind::array) {
        for (const leonida::Json& probe : probes->array) {
            std::array<double, 2> xy{}, uv{};
            EXPECT(pixel_of(probe, "pixel", xy) && pixel_of(probe, "uv", uv));
            const Pixel& p = at(UINT(xy[0]), UINT(xy[1]));
            const double want[4] = {uv[0], uv[1], 0.0, 1.0};
            for (int c = 0; c < 4; ++c) EXPECT(std::fabs(p[c] - want[c]) <= 2e-3);
        }
        std::printf("%zu UV probes match\n", probes->array.size());
    }

    // vk_harness's own image of the same draw, which first_light.py --run writes; required, so the whole-image check
    // cannot be skipped silently. The PS exports with v_cvt_pkrtz_f16_f32 (round toward zero to f16), so an f32
    // interpolant a few ulps apart, which the opposite viewport direction gives, can land on the adjacent f16 value:
    // measured on an RTX 3060, 1 of 16,384 pixels, and vk_harness drawing the mesh mirrored in y reproduces this
    // image bit for bit. Coverage (alpha) must match exactly.
    const auto vk_raw = read_file(dir / "image.raw");
    if (vk_raw.empty()) {
        std::fprintf(stderr, "%s has no image.raw: run tools/first_light.py with --run first\n", dir.string().c_str());
        return 1;
    }
    {
        EXPECT(vk_raw.size() == pixels.size() * sizeof(Pixel));
        // Both values must be exact halves (the PS exported f16) and adjacent in half order: ordered index = the
        // half's bits for +, minus the magnitude bits for -, so 0x3bff and 0x3c00 are one step and so are 0 and 0x0001.
        const auto half_index = [](float f, int& out) {
            const float a = std::fabs(f);
            if (!(a <= 65504.0f)) return false;
            int e = 0;
            const float m = std::frexp(a, &e);                       // a = m * 2^e, m in [0.5, 1)
            int bits;
            if (a == 0.0f) {
                bits = 0;
            } else if (e <= -14) {                                    // subnormal half: a = k * 2^-24
                const float k = std::ldexp(a, 24);
                if (k != std::floor(k)) return false;
                bits = int(k);
            } else {
                const float k = std::ldexp(m, 11);                    // 11 significant bits, implicit one included
                if (k != std::floor(k)) return false;
                bits = ((e + 14) << 10) | (int(k) & 0x3ff);
            }
            out = std::signbit(f) ? -bits : bits;
            return true;
        };
        const auto one_f16_step = [&](float a, float b) {
            int ia = 0, ib = 0;
            return half_index(a, ia) && half_index(b, ib) && std::abs(ia - ib) <= 1;
        };
        EXPECT(one_f16_step(1.0f, 0.99951171875f) && !one_f16_step(1.0f, 0.9990234375f));   // 0x3c00 vs 0x3bff, 0x3bfe
        EXPECT(one_f16_step(0.0f, 5.9604645e-8f) && !one_f16_step(0.0f, 1.1920929e-7f));     // 0 vs 0x0001, 0x0002
        unsigned exact = 0, f16_step = 0, differ = 0;
        for (UINT y = 0; y < height; ++y) {
            for (UINT x = 0; x < width; ++x) {
                Pixel want;
                std::memcpy(want.data(), vk_raw.data() + (std::size_t(y) * width + x) * sizeof(Pixel), sizeof(Pixel));
                const Pixel& got = at(x, y);
                if (std::memcmp(got.data(), want.data(), sizeof(Pixel)) == 0) {
                    ++exact;
                    continue;
                }
                const bool close = got[3] == want[3] && one_f16_step(got[0], want[0]) &&
                                   one_f16_step(got[1], want[1]) && one_f16_step(got[2], want[2]);
                ++(close ? f16_step : differ);
                std::fprintf(stderr, "vk_harness (%u, %u): %g %g %g %g, here (%u, %u): %g %g %g %g%s\n", x, y,
                             want[0], want[1], want[2], want[3], x, height - 1 - y, got[0], got[1], got[2], got[3],
                             close ? " (one f16 step)" : "");
                if (differ > 8) return 1;
            }
        }
        std::printf("against vk_harness's image.raw (rows flipped): %u of %u pixels bit-exact, %u one f16 step apart, "
                    "%u different\n", exact, width * height, f16_step, differ);
        EXPECT(differ == 0);
        // An adjacent half only appears where the f32 interpolant sits a few ulps from an f16 truncation boundary,
        // about 2^-11 of values per channel: 1 pixel of 12,804 covered on the RTX 3060. Allow at most one covered
        // pixel in a thousand, so a regression that moves the whole image by one half step (thousands) fails.
        EXPECT(f16_step <= covered / 1000);
    }
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("first light through D3D12: the game's VS+PS found by FSSM hash drew %u triangles\n", index_count / 3);
    return 0;
}
