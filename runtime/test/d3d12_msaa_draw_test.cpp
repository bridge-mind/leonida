// Multisampled draws through the Xbox-ABI proxy over the Vulkan device. A 4x R8G8B8A8 render target and a 4x
// D32_FLOAT depth target (depth test LESS, cleared to 1) take two draws, and ResolveSubresource averages the
// target into a single-sampled one that is read back:
//   - a red triangle over the half of the image above the diagonal from the bottom-left to the top-right corner.
//     Pixels well inside are red and pixels well outside blue. Pixels the edge crosses resolve to a mix, which
//     only a multisampled target produces (a single-sampled one gives 0 or 255);
//   - a green quad over the bottom-right quadrant from a PSO with SampleMask 0x1: one sample of four is written,
//     so those pixels resolve to a quarter green and three quarters blue.
// PSO quality: D3D12_STANDARD_MULTISAMPLE_PATTERN is accepted, as on native D3D12; quality 1 (past NumQualityLevels)
// and 3 samples are E_INVALIDARG. A 4x PSO drawing into a single-sampled target fails the list at execute
// (invalid_subresource) instead of reaching the driver. Shaders: test/glsl/vs_draw.vert, ps_draw.frag. No game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cstdlib>
#include <cstring>

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_VS_SPV and LEONIDA_TEST_PS_SPV must name the compiled test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

struct Vertex {
    float x, y;
    std::uint8_t r, g, b, a;
};

D3D12_CPU_DESCRIPTOR_HANDLE heap_start(ID3D12DescriptorHeap* heap) {
    return {SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(heap, 9))(heap))};
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU (hosted CI): ctest skips
    EXPECT(device_status == vktest::DeviceStatus::ready);

    const auto vs_program = fssm_program({0x56530001, 0xbf810000});
    const auto ps_program = fssm_program({0x50530001, 0xbf810000});
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    const testfiles::ScratchDir scratch("leonida_msaa_draw_test");
    const auto& library = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_SPV), library / "vs_draw.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), library / "ps_draw.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_draw\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}, "
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 1, \"mask\": 3, \"rw_mask\": 3, "
                 "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }

    constexpr UINT size = 64;
    constexpr UINT samples = 4;
    std::vector<std::uint8_t> pixels(size * 256);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.shader_dir = library.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        if (rt::d3d12::device_push_constants_need_fallback(inner)) {
            std::printf("d3d12_msaa_draw_test: skipped, the PSOs need 144 push-constant bytes and this GPU has %u\n",
                        rt::d3d12::device_max_push_constants_size(inner));
            return 77;
        }

        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[1].Descriptor = {1, 0};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        const D3D12_INPUT_ELEMENT_DESC elements[2] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D12_APPEND_ALIGNED_ELEMENT,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.DepthStencilState.DepthEnable = TRUE;  // the VS writes z = 0, under the cleared 1
        pso_desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        pso_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        pso_desc.InputLayout = {elements, 2};
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        pso_desc.SampleDesc = {samples, 0};
        Ref<ID3D12PipelineState> pso, masked, three, single;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));
        pso_desc.SampleMask = 0x1;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&masked.p))));
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.SampleDesc = {3, 0};  // not a power of two
        EXPECT(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&three.p)) == E_INVALIDARG && !three.p);
        pso_desc.SampleDesc = {samples, 1};  // past NumQualityLevels (1): D3D12 returns E_INVALIDARG
        EXPECT(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&three.p)) == E_INVALIDARG && !three.p);
        pso_desc.SampleDesc = {samples, UINT(DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN)};  // accepted natively
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&three.p))));
        three.reset();

        // Vertices 0-2: the triangle above the diagonal; 3-6: the bottom-right quadrant.
        const Vertex vertices[7] = {
            {-1, 1, 255, 0, 0, 255}, {1, 1, 255, 0, 0, 255}, {-1, -1, 255, 0, 0, 255},
            {0, 0, 255, 0, 0, 255}, {1, 0, 255, 0, 0, 255}, {0, -1, 255, 0, 0, 255}, {1, -1, 255, 0, 0, 255}};
        const std::uint16_t indices[6] = {3, 4, 5, 5, 4, 6};
        const float tints[2][3] = {{1, 0, 1}, {0, 1, 0}};  // red stays red; then green

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> vertex_buffer, upload, readback, target, depth, resolved;
        const auto vb_desc = buffer_desc(sizeof(vertices));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &vb_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&vertex_buffer.p))));
        const auto upload_desc = buffer_desc(4096);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto readback_desc = buffer_desc(size * 256);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = rt_desc.Height = size;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rt_desc.SampleDesc = {samples, 0};
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                         IID_PPV_ARGS(&target.p))));
        D3D12_RESOURCE_DESC ds_desc = rt_desc;
        ds_desc.Format = DXGI_FORMAT_D32_FLOAT;
        ds_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &ds_desc,
                                                         D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                         IID_PPV_ARGS(&depth.p))));
        D3D12_RESOURCE_DESC single_desc = rt_desc;
        single_desc.SampleDesc = {1, 0};
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &single_desc,
                                                         D3D12_RESOURCE_STATE_RESOLVE_DEST, nullptr,
                                                         IID_PPV_ARGS(&resolved.p))));

        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memcpy(up, vertices, sizeof(vertices));
        std::memcpy(up + 256, indices, sizeof(indices));
        std::memcpy(up + 512, tints[0], sizeof(tints[0]));
        std::memcpy(up + 768, tints[1], sizeof(tints[1]));
        const D3D12_GPU_VIRTUAL_ADDRESS upload_va = upload->GetGPUVirtualAddress();

        const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC dsv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap, dsv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&dsv_heap_desc, IID_PPV_ARGS(&dsv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv = heap_start(rtv_heap.p);
        const D3D12_CPU_DESCRIPTOR_HANDLE single_rtv{rtv.ptr + rt::d3d12::descriptor_size};
        const D3D12_CPU_DESCRIPTOR_HANDLE dsv = heap_start(dsv_heap.p);
        D3D12_RENDER_TARGET_VIEW_DESC ms_view{};
        ms_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        ms_view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
        device->CreateRenderTargetView(target.p, &ms_view, rtv);
        device->CreateRenderTargetView(resolved.p, nullptr, single_rtv);
        D3D12_DEPTH_STENCIL_VIEW_DESC ms_depth{};
        ms_depth.Format = DXGI_FORMAT_D32_FLOAT;
        ms_depth.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
        device->CreateDepthStencilView(depth.p, &ms_depth, dsv);

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

        const auto common_state = [&](D3D12_CPU_DESCRIPTOR_HANDLE target_view, const D3D12_CPU_DESCRIPTOR_HANDLE* ds) {
            list->SetGraphicsRootSignature(root.p);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            const D3D12_VERTEX_BUFFER_VIEW vbv{vertex_buffer->GetGPUVirtualAddress(), sizeof(vertices),
                                               sizeof(Vertex)};
            list->IASetVertexBuffers(0, 1, &vbv);
            const D3D12_INDEX_BUFFER_VIEW ibv{upload_va + 256, sizeof(indices), DXGI_FORMAT_R16_UINT};
            list->IASetIndexBuffer(&ibv);
            const D3D12_VIEWPORT viewport{0, 0, float(size), float(size), 0, 1};
            const D3D12_RECT scissor{0, 0, LONG(size), LONG(size)};
            list->RSSetViewports(1, &viewport);
            list->RSSetScissorRects(1, &scissor);
            list->OMSetRenderTargets(1, &target_view, FALSE, ds);
            const float origin[2] = {0, 0};
            list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
        };

        list->CopyBufferRegion(vertex_buffer.p, 0, upload.p, 0, sizeof(vertices));
        D3D12_RESOURCE_BARRIER to_vertex = transition(vertex_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                                      D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        list->ResourceBarrier(1, &to_vertex);
        const FLOAT blue[4] = {0, 0, 1, 1};
        list->ClearRenderTargetView(rtv, blue, 0, nullptr);
        list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        common_state(rtv, &dsv);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 512);  // red
        list->DrawInstanced(3, 1, 0, 0);
        list->SetPipelineState(masked.p);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 768);  // green, sample 0 only
        list->DrawIndexedInstanced(6, 1, 0, 0, 0);

        D3D12_RESOURCE_BARRIER to_resolve = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                       D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
        list->ResourceBarrier(1, &to_resolve);
        list->ResolveSubresource(resolved.p, 0, target.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
        D3D12_RESOURCE_BARRIER to_copy = transition(resolved.p, D3D12_RESOURCE_STATE_RESOLVE_DEST,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, size, size, 1, 256}};
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = resolved.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
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

        // A 4x PSO into the single-sampled target: D3D12 requires matching counts, so the list fails at execute.
        EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, pso.p)));
        D3D12_RESOURCE_BARRIER to_target = transition(resolved.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &to_target);
        common_state(single_rtv, nullptr);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
        list->DrawInstanced(3, 1, 0, 0);
        EXPECT(list->Close() == S_OK);
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::invalid_subresource);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 2)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(2, nullptr)));

        for (auto* r : {&vertex_buffer, &upload, &readback, &target, &depth, &resolved}) r->reset();
        pso.reset();
        masked.reset();
        root.reset();
        rtv_heap.reset();
        dsv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    // Pixel centre in clip space; the triangle covers y > x. Sample offsets are under half a pixel on each axis,
    // so a pixel whose centre is more than one pixel width (2 / size in clip space, over both axes) from the edge
    // is fully inside or outside.
    unsigned mismatches = 0, edge_mixed = 0, edge = 0;
    const auto report = [&](UINT col, UINT row, const std::uint8_t* got, const char* want) {
        if (mismatches++ < 6)
            std::fprintf(stderr, "pixel (%u, %u): got %u,%u,%u,%u want %s\n", col, row, got[0], got[1], got[2],
                         got[3], want);
    };
    for (UINT row = 0; row < size; ++row) {
        for (UINT col = 0; col < size; ++col) {
            const std::uint8_t* got = pixels.data() + row * 256 + col * 4;
            const float x = (col + 0.5f) * 2 / size - 1, y = 1 - (row + 0.5f) * 2 / size;
            const float distance = y - x, margin = 2.0f / size;
            if (col >= size / 2 && row >= size / 2) {
                // One of four samples green, three blue: 63.75 and 191.25, rounded either way by the resolve.
                if (got[0] != 0 || std::abs(int(got[1]) - 64) > 1 || std::abs(int(got[2]) - 191) > 1 || got[3] != 255)
                    report(col, row, got, "0,64,191,255 (+-1)");
            } else if (distance > margin) {
                if (!(got[0] == 255 && got[1] == 0 && got[2] == 0 && got[3] == 255)) report(col, row, got, "red");
            } else if (distance < -margin) {
                if (!(got[0] == 0 && got[1] == 0 && got[2] == 255 && got[3] == 255)) report(col, row, got, "blue");
            } else {
                ++edge;
                if (got[0] > 0 && got[0] < 255) ++edge_mixed;
                if (got[1] != 0 || got[3] != 255 || int(got[0]) + int(got[2]) < 254 || int(got[0]) + int(got[2]) > 256)
                    report(col, row, got, "a red/blue mix");
            }
        }
    }
    EXPECT(!mismatches);
    EXPECT(edge_mixed > 0);  // single-sampled rendering leaves no partial pixels
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("MSAA: 4x draws resolved, %u pixels matched; %u of %u edge pixels are partial (red/blue mix); "
                "SampleMask 0x1 wrote one sample of four; 3 samples and a mismatched target refused\n",
                size * size, edge_mixed, edge);
    return 0;
}
