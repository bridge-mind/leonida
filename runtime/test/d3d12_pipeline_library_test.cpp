// Synthetic GPU test for ID3D12Device1::CreatePipelineLibrary and ID3D12PipelineLibrary / ID3D12PipelineLibrary1:
//   - empty library creation (0-byte blob), null-output capability check (S_FALSE), and invalid/foreign/corrupt
//     blob rejection (E_INVALIDARG for null pointer with non-zero size, D3D12_ERROR_DRIVER_VERSION_MISMATCH for
//     unrecognized or corrupt headers);
//   - StorePipeline (unique names, duplicate rejection, null/empty name rejection, null/foreign PSO rejection);
//   - LoadComputePipeline, LoadGraphicsPipeline, and ID3D12PipelineLibrary1::LoadPipeline (including cross-type
//     mismatch rejection when a compute PSO is requested through LoadGraphicsPipeline or vice versa);
//   - GetSerializedSize + Serialize round-trip into a second CreatePipelineLibrary;
//   - GPU execution of both loaded compute and loaded graphics PSOs under LEONIDA_VK_VALIDATION=1.
// No game data.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#if !defined(LEONIDA_TEST_CS_SPV) || !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_CS_SPV, LEONIDA_TEST_VS_SPV and LEONIDA_TEST_PS_SPV must be defined"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

struct Vertex {
    float x, y;
    std::uint8_t r, g, b, a;
};

D3D12_RESOURCE_DESC tex2d_desc(UINT w, UINT h, DXGI_FORMAT fmt, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = fmt;
    d.SampleDesc = {1, 0};
    d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d.Flags = flags;
    return d;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);

    const auto cs_prog = fssm_program({0x43532001, 0xbf810000});
    const auto vs_prog = fssm_program({0x56532001, 0xbf810000});
    const auto ps_prog = fssm_program({0x50532001, 0xbf810000});
    const auto cs_dxbc = shader_container(cs_prog);
    const auto vs_dxbc = shader_container(vs_prog);
    const auto ps_dxbc = shader_container(ps_prog);

    const testfiles::ScratchDir scratch("leonida_pipeline_library_test");
    const auto& lib_dir = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), lib_dir / "cs.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_SPV), lib_dir / "vs.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), lib_dir / "ps.spv"));
    {
        std::ofstream idx(lib_dir / "index.json", std::ios::binary);
        idx << "{\"cs\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(cs_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144},\n"
               " \"vs\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
               "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
               "\"system_value\": 0, \"component_type\": \"float\"}, "
               "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 1, \"mask\": 3, \"rw_mask\": 3, "
               "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
               " \"ps\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = true;
    info.shader_dir = lib_dir.string();

    Ref<ID3D12Device1> device;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

    // 1. CreatePipelineLibrary creation, capability check (null output -> S_FALSE), and invalid/corrupt blobs.
    {
        EXPECT(device->CreatePipelineLibrary(nullptr, 0, __uuidof(ID3D12PipelineLibrary), nullptr) == S_FALSE);
        Ref<ID3D12PipelineLibrary> bad;
        EXPECT(device->CreatePipelineLibrary(nullptr, 16, IID_PPV_ARGS(&bad.p)) == E_INVALIDARG && !bad.p);
        // A foreign or truncated blob is E_INVALIDARG, as on the host runtime (d3d12_pipeline_library_host_test);
        // one of ours from another version is D3D12_ERROR_DRIVER_VERSION_MISMATCH ("LPL1", version 2).
        const std::uint32_t foreign[4] = {0xDEADBEEFu, 1u, 0u, 0u};
        EXPECT(device->CreatePipelineLibrary(foreign, sizeof(foreign), IID_PPV_ARGS(&bad.p)) == E_INVALIDARG && !bad.p);
        const std::uint8_t tiny[4] = {1, 2, 3, 4};
        EXPECT(device->CreatePipelineLibrary(tiny, sizeof(tiny), IID_PPV_ARGS(&bad.p)) == E_INVALIDARG && !bad.p);
        const std::uint32_t other_version[4] = {0x314C504Cu, 2u, 0u, 0u};
        EXPECT(device->CreatePipelineLibrary(other_version, sizeof(other_version), IID_PPV_ARGS(&bad.p)) ==
                   D3D12_ERROR_DRIVER_VERSION_MISMATCH &&
               !bad.p);
    }

    if (rt::d3d12::device_push_constants_need_fallback(device.p)) {
        std::printf("d3d12_pipeline_library_test: skipped, the PSOs need 144 push-constant bytes and this GPU has %u\n",
                    rt::d3d12::device_max_push_constants_size(device.p));
        return 77;
    }

    // 2. Build one compute PSO and one graphics PSO.
    D3D12_DESCRIPTOR_RANGE uav_range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, 0};
    D3D12_ROOT_PARAMETER cs_params[3]{};
    cs_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    cs_params[0].Constants = {0, 0, 1};
    cs_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    cs_params[1].Descriptor = {0, 0};
    cs_params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    cs_params[2].DescriptorTable = {1, &uav_range};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC cs_rs_desc{};
    cs_rs_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    cs_rs_desc.Desc_1_0 = {3, cs_params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto cs_rs_blob = serialize_root_signature(cs_rs_desc);
    Ref<ID3D12RootSignature> cs_rs;
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, cs_rs_blob.data(), cs_rs_blob.size(), IID_PPV_ARGS(&cs_rs.p))));

    D3D12_COMPUTE_PIPELINE_STATE_DESC cs_desc{};
    cs_desc.pRootSignature = cs_rs.p;
    cs_desc.CS = {cs_dxbc.data(), cs_dxbc.size()};
    Ref<ID3D12PipelineState> cs_pso;
    EXPECT(SUCCEEDED(device->CreateComputePipelineState(&cs_desc, IID_PPV_ARGS(&cs_pso.p))));

    D3D12_ROOT_PARAMETER gfx_params[2]{};
    gfx_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    gfx_params[0].Constants = {0, 0, 2};
    gfx_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    gfx_params[1].Descriptor = {1, 0};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC gfx_rs_desc{};
    gfx_rs_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    gfx_rs_desc.Desc_1_0 = {2, gfx_params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
    const auto gfx_rs_blob = serialize_root_signature(gfx_rs_desc);
    Ref<ID3D12RootSignature> gfx_rs;
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, gfx_rs_blob.data(), gfx_rs_blob.size(), IID_PPV_ARGS(&gfx_rs.p))));

    const D3D12_INPUT_ELEMENT_DESC input_elems[2] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D12_APPEND_ALIGNED_ELEMENT,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC gfx_desc{};
    gfx_desc.pRootSignature = gfx_rs.p;
    gfx_desc.VS = {vs_dxbc.data(), vs_dxbc.size()};
    gfx_desc.PS = {ps_dxbc.data(), ps_dxbc.size()};
    gfx_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    gfx_desc.SampleMask = 0xFFFFFFFFu;
    gfx_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    gfx_desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    gfx_desc.RasterizerState.DepthClipEnable = TRUE;
    gfx_desc.InputLayout = {input_elems, 2};
    gfx_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    gfx_desc.NumRenderTargets = 1;
    gfx_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    gfx_desc.SampleDesc = {1, 0};
    Ref<ID3D12PipelineState> gfx_pso;
    EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&gfx_desc, IID_PPV_ARGS(&gfx_pso.p))));

    // 3. Store, load, cross-type rejection, and serialize round-trip.
    Ref<ID3D12PipelineLibrary1> lib;
    EXPECT(SUCCEEDED(device->CreatePipelineLibrary(nullptr, 0, IID_PPV_ARGS(&lib.p))) && lib.p);
    Ref<ID3D12Device> parent_dev;
    EXPECT(SUCCEEDED(lib->GetDevice(IID_PPV_ARGS(&parent_dev.p))) && parent_dev.p);

    EXPECT(lib->StorePipeline(nullptr, cs_pso.p) == E_INVALIDARG);
    EXPECT(lib->StorePipeline(L"", cs_pso.p) == E_INVALIDARG);
    EXPECT(lib->StorePipeline(L"cs", nullptr) == E_INVALIDARG);
    EXPECT(SUCCEEDED(lib->StorePipeline(L"cs", cs_pso.p)));
    EXPECT(lib->StorePipeline(L"cs", cs_pso.p) == E_INVALIDARG);  // duplicate name
    EXPECT(SUCCEEDED(lib->StorePipeline(L"gfx", gfx_pso.p)));

    Ref<ID3D12PipelineState> loaded_cs, loaded_gfx, loaded_any, wrong;
    EXPECT(lib->LoadComputePipeline(L"missing", &cs_desc, IID_PPV_ARGS(&wrong.p)) == E_INVALIDARG && !wrong.p);
    EXPECT(lib->LoadComputePipeline(L"gfx", &cs_desc, IID_PPV_ARGS(&wrong.p)) == E_INVALIDARG && !wrong.p);
    EXPECT(lib->LoadGraphicsPipeline(L"cs", &gfx_desc, IID_PPV_ARGS(&wrong.p)) == E_INVALIDARG && !wrong.p);
    EXPECT(SUCCEEDED(lib->LoadComputePipeline(L"cs", &cs_desc, IID_PPV_ARGS(&loaded_cs.p))) && loaded_cs.p);
    EXPECT(SUCCEEDED(lib->LoadGraphicsPipeline(L"gfx", &gfx_desc, IID_PPV_ARGS(&loaded_gfx.p))) && loaded_gfx.p);

    std::uint32_t dummy_stream = 0;
    D3D12_PIPELINE_STATE_STREAM_DESC stream_desc{sizeof(dummy_stream), &dummy_stream};
    D3D12_PIPELINE_STATE_STREAM_DESC empty_stream{0, nullptr};
    EXPECT(lib->LoadPipeline(L"gfx", &empty_stream, IID_PPV_ARGS(&wrong.p)) == E_INVALIDARG && !wrong.p);
    EXPECT(SUCCEEDED(lib->LoadPipeline(L"gfx", &stream_desc, IID_PPV_ARGS(&loaded_any.p))) && loaded_any.p);

    const SIZE_T blob_size = lib->GetSerializedSize();
    EXPECT(blob_size >= 16);
    std::vector<std::uint8_t> blob(blob_size);
    EXPECT(lib->Serialize(nullptr, blob_size) == E_INVALIDARG);
    EXPECT(lib->Serialize(blob.data(), blob_size - 1) == E_INVALIDARG);
    EXPECT(SUCCEEDED(lib->Serialize(blob.data(), blob_size)));

    Ref<ID3D12PipelineLibrary> reloaded;
    EXPECT(SUCCEEDED(device->CreatePipelineLibrary(blob.data(), blob.size(), IID_PPV_ARGS(&reloaded.p))) && reloaded.p);

    // 4. Execute the loaded compute PSO and loaded graphics PSO on the GPU.
    Ref<ID3D12CommandQueue> queue;
    const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
    Ref<ID3D12CommandAllocator> alloc;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(
        device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p))));
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    constexpr UINT W = 16, H = 16;
    const D3D12_HEAP_PROPERTIES hp_def{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES hp_up{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES hp_rb{D3D12_HEAP_TYPE_READBACK};
    const auto rtd = tex2d_desc(W, H, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    const Vertex verts[4] = {
        {-1.0f, 1.0f, 255, 0, 0, 255},
        { 0.0f, 1.0f, 255, 0, 0, 255},
        { 0.0f, 0.0f, 255, 0, 0, 255},
        {-1.0f, 0.0f, 255, 0, 0, 255},
    };
    const std::uint16_t indices[6] = {0, 1, 2, 0, 2, 3};
    const float tint[4] = {1.0f, 0.0f, 1.0f, 1.0f};
    const auto vbd = buffer_desc(sizeof(verts));
    const auto ibd = buffer_desc(sizeof(indices));
    const auto cbd = buffer_desc(256);
    Ref<ID3D12Resource> rt, vb, ib, cb, rb;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&hp_def, D3D12_HEAP_FLAG_NONE, &rtd,
                                                     D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&rt.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&hp_up, D3D12_HEAP_FLAG_NONE, &vbd,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&vb.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&hp_up, D3D12_HEAP_FLAG_NONE, &ibd,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&ib.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&hp_up, D3D12_HEAP_FLAG_NONE, &cbd,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&cb.p))));
    void* p = nullptr;
    EXPECT(SUCCEEDED(vb->Map(0, nullptr, &p)));
    std::memcpy(p, verts, sizeof(verts));
    vb->Unmap(0, nullptr);
    EXPECT(SUCCEEDED(ib->Map(0, nullptr, &p)));
    std::memcpy(p, indices, sizeof(indices));
    ib->Unmap(0, nullptr);
    EXPECT(SUCCEEDED(cb->Map(0, nullptr, &p)));
    std::memcpy(p, tint, sizeof(tint));
    cb->Unmap(0, nullptr);

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 total_bytes = 0;
    device->GetCopyableFootprints(&rtd, 0, 1, 0, &fp, nullptr, nullptr, &total_bytes);
    const auto rbd = buffer_desc(total_bytes);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&hp_rb, D3D12_HEAP_FLAG_NONE, &rbd,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb.p))));

    Ref<ID3D12DescriptorHeap> rtv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC rhd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rhd, IID_PPV_ARGS(&rtv_heap.p))));
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(rt.p, nullptr, rtv);

    const float blue[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    list->ClearRenderTargetView(rtv, blue, 0, nullptr);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    const D3D12_VIEWPORT vp{0, 0, float(W), float(H), 0.0f, 1.0f};
    const D3D12_RECT sc{0, 0, LONG(W), LONG(H)};
    list->RSSetViewports(1, &vp);
    list->RSSetScissorRects(1, &sc);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), sizeof(verts), sizeof(Vertex)};
    const D3D12_INDEX_BUFFER_VIEW ibv{ib->GetGPUVirtualAddress(), sizeof(indices), DXGI_FORMAT_R16_UINT};
    list->IASetVertexBuffers(0, 1, &vbv);
    list->IASetIndexBuffer(&ibv);
    list->SetGraphicsRootSignature(gfx_rs.p);
    const float offset[2] = {0.0f, 0.0f};
    list->SetGraphicsRoot32BitConstants(0, 2, offset, 0);
    list->SetGraphicsRootConstantBufferView(1, cb->GetGPUVirtualAddress());
    list->SetPipelineState(loaded_gfx.p);
    list->DrawIndexedInstanced(6, 1, 0, 0, 0);

    const auto to_src = transition(rt.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_src);
    D3D12_TEXTURE_COPY_LOCATION src_loc{rt.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    D3D12_TEXTURE_COPY_LOCATION dst_loc{rb.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    dst_loc.PlacedFootprint = fp;
    list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

    EXPECT(SUCCEEDED(rb->Map(0, nullptr, &p)));
    const auto* bytes = static_cast<const std::uint8_t*>(p);
    std::uint32_t tl = 0, br = 0;
    std::memcpy(&tl, bytes + 2 * fp.Footprint.RowPitch + 2 * 4, 4);
    std::memcpy(&br, bytes + 12 * fp.Footprint.RowPitch + 12 * 4, 4);
    EXPECT(tl == 0xFF0000FFu);  // red quad drawn by PSO loaded from PipelineLibrary
    EXPECT(br == 0xFFFF0000u);  // blue background clear
    rb->Unmap(0, nullptr);

    EXPECT(d.validation_errors() == 0);
    std::printf("d3d12_pipeline_library_test: ok\n");
    return 0;
}
