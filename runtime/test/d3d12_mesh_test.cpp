// D3D-29: mesh-shader PSOs from pipeline state streams and DispatchMesh, through the Xbox-ABI proxy over the Vulkan
// device (docs/d3d12_vulkan.md section 5, "Pipelines").
//   - A device without VK_EXT_mesh_shader reports D3D12_MESH_SHADER_TIER_NOT_SUPPORTED and refuses a mesh PSO with
//     E_NOTIMPL; that part runs on every GPU. The rest needs the extension (exit 77 without it).
//   - With it: MeshShaderTier 1; CreatePipelineState builds an MS + PS PSO and a VS + PS PSO from streams laid out as
//     d3dx12.h lays them out (each subobject a type and its description, pointer-aligned); streams the PC runtime
//     refuses (MS with VS, CS with PS, a subobject twice or in two versions, an unknown type, a truncated stream) are E_INVALIDARG, and
//     state not built yet (an amplification shader, a depth-bounds test) is E_NOTIMPL.
//   - Draws: DispatchMesh(2, 2, 1) with root constants (0, 0) and a red tint fills the top-left quadrant;
//     DispatchMesh(1, 1, 2) moved by (1, -1) with a green tint fills the top half of the bottom-right one, so the
//     Y and Z counts reach the shader; the stream's VS PSO draws a full-screen triangle through a viewport on the
//     bottom-left quadrant (yellow). Each pixel is read back and compared.
//   - A vertex draw with the mesh PSO, and DispatchMesh with the VS one, fail at execution (ReplayResult::
//     invalid_draw); a mesh grid beyond D3D12's limits fails at Close; an empty grid records nothing.
// The shaders are GLSL written to the translator's contract (test/glsl/ms_draw.mesh, ps_draw.frag, vs_fullscreen.vert).
// No game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cstring>

#if !defined(LEONIDA_TEST_MS_SPV) || !defined(LEONIDA_TEST_PS_SPV) || !defined(LEONIDA_TEST_VS_FULLSCREEN_SPV)
#error "LEONIDA_TEST_MS_SPV, LEONIDA_TEST_PS_SPV and LEONIDA_TEST_VS_FULLSCREEN_SPV must name the compiled test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

// One pipeline state stream subobject as d3dx12.h declares it (CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT).
template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Type, typename Inner>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = Type;
    Inner inner;
};
using RootSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>;
using VsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, D3D12_SHADER_BYTECODE>;
using PsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE>;
using CsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, D3D12_SHADER_BYTECODE>;
using AsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_SHADER_BYTECODE>;
using MsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE>;
using TopologySubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE>;
using TargetsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY>;
using DepthStencil1Subobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, D3D12_DEPTH_STENCIL_DESC1>;
using DepthStencilSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC>;
using RasterizerSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC>;
using Rasterizer2Subobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER2, D3D12_RASTERIZER_DESC2>;
using MaskSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, UINT>;

// Subobjects appended one after another: each is pointer-aligned and a multiple of the pointer size, so the bytes
// are what a struct of the same members would hold.
struct Stream {
    std::vector<std::uint8_t> bytes;
    template <typename S>
    Stream& add(const S& subobject) {
        static_assert(sizeof(S) % sizeof(void*) == 0);
        const std::size_t at = bytes.size();
        bytes.resize(at + sizeof(S));
        std::memcpy(bytes.data() + at, &subobject, sizeof(S));
        return *this;
    }
    D3D12_PIPELINE_STATE_STREAM_DESC desc(std::size_t trim = 0) { return {bytes.size() - trim, bytes.data()}; }
};

D3D12_RT_FORMAT_ARRAY one_target() {
    D3D12_RT_FORMAT_ARRAY formats{};
    formats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    formats.NumRenderTargets = 1;
    return formats;
}

HRESULT create(ID3D12Device* device, Stream& stream, Ref<ID3D12PipelineState>& out, std::size_t trim = 0) {
    Ref<ID3D12Device2> device2;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device2.p)))) return E_NOINTERFACE;
    const D3D12_PIPELINE_STATE_STREAM_DESC desc = stream.desc(trim);
    out.reset();
    return device2->CreatePipelineState(&desc, IID_PPV_ARGS(&out.p));
}

D3D12_MESH_SHADER_TIER mesh_tier(ID3D12Device* device) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options, sizeof(options))))
        return D3D12_MESH_SHADER_TIER(-1);
    return options.MeshShaderTier;
}

}  // namespace

int main() {
    vktest::Device d;
    d.request_mesh_shader = true;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    const auto ms_program = fssm_program({0x4d530001, 0xbf810000});
    const auto ps_program = fssm_program({0x50530001, 0xbf810000});
    const auto vs_program = fssm_program({0x56530002, 0xbf810000});
    const auto ms_container = shader_container(ms_program), ps_container = shader_container(ps_program);
    const auto vs_container = shader_container(vs_program);
    const auto library = std::filesystem::temp_directory_path() / "leonida_mesh_test";
    std::filesystem::create_directories(library);
    std::filesystem::copy_file(LEONIDA_TEST_MS_SPV, library / "ms_draw.spv", std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(LEONIDA_TEST_PS_SPV, library / "ps_draw.spv", std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(LEONIDA_TEST_VS_FULLSCREEN_SPV, library / "vs_fullscreen.spv",
                               std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"ms_draw\": {\"stage\": \"ms\", \"fssm_sha256\": \"" << leonida::sha256_hex(ms_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [32, 1, 1], "
                 "\"max_vertices\": 4, \"max_primitives\": 2},\n"
                 " \"vs_fullscreen\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }

    const auto device_info = [&](bool mesh) {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.shader_dir = library.string();
        info.mesh_shader = mesh;
        info.pipeline_statistics_query = d.pipeline_statistics_query;
        return info;
    };
    const auto make_device = [&](bool mesh, Ref<ID3D12Device>& device) {
        ID3D12Device* inner = nullptr;
        if (FAILED(rt::d3d12::create_vulkan_device(device_info(mesh), IID_PPV_ARGS(&inner)))) return false;
        if (FAILED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                     reinterpret_cast<void**>(&device.p))))
            return false;
        rt::d3d12::set_device_outer(inner, device.p);
        return true;
    };
    const auto root_signature = [&](ID3D12Device* device, Ref<ID3D12RootSignature>& root) {
        // [0] two constants (s0, s1: offset), [1] a root CBV (s2:s3: tint), as in d3d12_draw_test.
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[1].Descriptor = {1, 0};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto blob = serialize_root_signature(root_desc);
        return !blob.empty() &&
               SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p)));
    };
    const D3D12_SHADER_BYTECODE ms{ms_container.data(), ms_container.size()};
    const D3D12_SHADER_BYTECODE ps{ps_container.data(), ps_container.size()};
    const D3D12_SHADER_BYTECODE vs{vs_container.data(), vs_container.size()};

    // Without the extension: no tier, and a mesh PSO is refused as not implemented.
    {
        Ref<ID3D12Device> device;
        Ref<ID3D12RootSignature> root;
        EXPECT(make_device(false, device));
        EXPECT(root_signature(device.p, root));
        EXPECT(mesh_tier(device.p) == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED);
        Stream stream;
        stream.add(RootSubobject{.inner = root.p}).add(MsSubobject{.inner = ms}).add(PsSubobject{.inner = ps});
        stream.add(TargetsSubobject{.inner = one_target()});
        Ref<ID3D12PipelineState> pso;
        EXPECT(create(device.p, stream, pso) == E_NOTIMPL && !pso.p);
        root.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }
    if (!d.mesh_shader) {
        std::filesystem::remove_all(library);
        std::printf("no VK_EXT_mesh_shader on this device: only the refusal without it was checked\n");
        return 77;
    }

    constexpr UINT size = 64;
    std::vector<std::uint8_t> pixels(size * 256);
    {
        Ref<ID3D12Device> device;
        Ref<ID3D12RootSignature> root;
        EXPECT(make_device(true, device));
        EXPECT(root_signature(device.p, root));
        EXPECT(mesh_tier(device.p) == D3D12_MESH_SHADER_TIER_1);

        // The PSOs. Every other subobject takes its default: back-face culling, depth test on (no DSV here), no
        // blending, all samples.
        Ref<ID3D12PipelineState> mesh_pso, vs_pso, refused;
        Stream mesh_stream;
        mesh_stream.add(RootSubobject{.inner = root.p}).add(MsSubobject{.inner = ms}).add(PsSubobject{.inner = ps});
        mesh_stream.add(TargetsSubobject{.inner = one_target()});
        EXPECT(SUCCEEDED(create(device.p, mesh_stream, mesh_pso)));
        Stream vs_stream;
        vs_stream.add(RootSubobject{.inner = root.p}).add(VsSubobject{.inner = vs}).add(PsSubobject{.inner = ps});
        vs_stream.add(TopologySubobject{.inner = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE});
        vs_stream.add(TargetsSubobject{.inner = one_target()});
        EXPECT(SUCCEEDED(create(device.p, vs_stream, vs_pso)));

        // Streams the PC runtime refuses.
        Stream bad = mesh_stream;
        bad.add(VsSubobject{.inner = vs});  // a mesh shader and a vertex shader
        EXPECT(create(device.p, bad, refused) == E_INVALIDARG && !refused.p);
        bad = mesh_stream;
        bad.add(PsSubobject{.inner = ps});  // the same subobject twice
        EXPECT(create(device.p, bad, refused) == E_INVALIDARG && !refused.p);
        bad = mesh_stream;
        bad.add(CsSubobject{.inner = ps});  // compute with graphics stages
        EXPECT(create(device.p, bad, refused) == E_INVALIDARG && !refused.p);
        EXPECT(create(device.p, mesh_stream, refused, 4) == E_INVALIDARG && !refused.p);  // truncated
        bad = mesh_stream;
        bad.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE(23), UINT>{.inner = 0});  // 23 is no subobject type
        EXPECT(create(device.p, bad, refused) == E_INVALIDARG && !refused.p);
        Stream empty;
        EXPECT(create(device.p, empty.add(MaskSubobject{.inner = UINT_MAX}), refused) == E_INVALIDARG && !refused.p);
        // Versions of one subobject count as the same subobject (D3D12_PIPELINE_STATE_STREAM_DESC): DEPTH_STENCIL with
        // DEPTH_STENCIL1 in either order, and RASTERIZER with RASTERIZER2. Each description alone is accepted.
        const D3D12_DEPTH_STENCIL_DESC depth{FALSE, D3D12_DEPTH_WRITE_MASK_ZERO, D3D12_COMPARISON_FUNC_ALWAYS};
        const D3D12_DEPTH_STENCIL_DESC1 depth1{FALSE, D3D12_DEPTH_WRITE_MASK_ZERO, D3D12_COMPARISON_FUNC_ALWAYS};
        D3D12_RASTERIZER_DESC raster{};
        raster.FillMode = D3D12_FILL_MODE_SOLID;
        raster.CullMode = D3D12_CULL_MODE_NONE;
        raster.DepthClipEnable = TRUE;  // D3D12's default; depth clamping is not built
        D3D12_RASTERIZER_DESC2 raster2{};
        raster2.FillMode = D3D12_FILL_MODE_SOLID;
        raster2.CullMode = D3D12_CULL_MODE_NONE;
        raster2.DepthClipEnable = TRUE;  // D3D12's default; depth clamping is not built
        for (int order = 0; order < 2; ++order) {
            bad = mesh_stream;
            if (order == 0) bad.add(DepthStencilSubobject{.inner = depth}).add(DepthStencil1Subobject{.inner = depth1});
            else bad.add(DepthStencil1Subobject{.inner = depth1}).add(DepthStencilSubobject{.inner = depth});
            EXPECT(create(device.p, bad, refused) == E_INVALIDARG && !refused.p);
        }
        bad = mesh_stream;
        bad.add(RasterizerSubobject{.inner = raster}).add(Rasterizer2Subobject{.inner = raster2});
        EXPECT(create(device.p, bad, refused) == E_INVALIDARG && !refused.p);
        for (int which = 0; which < 4; ++which) {
            Stream one = mesh_stream;
            if (which == 0) one.add(DepthStencilSubobject{.inner = depth});
            if (which == 1) one.add(DepthStencil1Subobject{.inner = depth1});
            if (which == 2) one.add(RasterizerSubobject{.inner = raster});
            if (which == 3) one.add(Rasterizer2Subobject{.inner = raster2});
            Ref<ID3D12PipelineState> accepted;
            const HRESULT one_hr = create(device.p, one, accepted);
            if (FAILED(one_hr)) std::fprintf(stderr, "single subobject %d: 0x%08lx\n", which, static_cast<unsigned long>(one_hr));
            EXPECT(SUCCEEDED(one_hr));
        }
        // State the layer does not build yet.
        bad = mesh_stream;
        bad.add(AsSubobject{.inner = vs});
        EXPECT(create(device.p, bad, refused) == E_NOTIMPL && !refused.p);
        bad = mesh_stream;
        D3D12_DEPTH_STENCIL_DESC1 bounds{};
        bounds.DepthBoundsTestEnable = TRUE;
        bad.add(DepthStencil1Subobject{.inner = bounds});
        EXPECT(create(device.p, bad, refused) == E_NOTIMPL && !refused.p);

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> upload, readback, target;
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
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p))));

        // Tints at 0 (red stays red), 256 (green) and 512 (white: yellow stays yellow); CBVs are 256-byte aligned.
        const float tints[3][3] = {{1, 0, 1}, {0, 1, 0}, {1, 1, 1}};
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        for (int i = 0; i < 3; ++i) std::memcpy(up + 256 * i, tints[i], sizeof(tints[i]));
        const D3D12_GPU_VIRTUAL_ADDRESS upload_va = upload->GetGPUVirtualAddress();

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
        Ref<ID3D12GraphicsCommandList6> list6;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, mesh_pso.p,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list6.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        D3D12_RESOURCE_BARRIER to_target =
            transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &to_target);
        const FLOAT blue[4] = {0, 0, 1, 1};
        list->ClearRenderTargetView(rtv, blue, 0, nullptr);
        list->SetGraphicsRootSignature(root.p);
        const D3D12_VIEWPORT viewport{0, 0, float(size), float(size), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(size), LONG(size)};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

        // Mesh draw 1: a 2x2 grid of quads, no offset, red -> the top-left quadrant. No topology was set: a mesh
        // draw does not need one. An empty grid before it records nothing.
        const float origin[2] = {0, 0}, down_right[2] = {1, -1};
        list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
        list->SetGraphicsRootConstantBufferView(1, upload_va);
        list6->DispatchMesh(0, 1, 1);
        list6->DispatchMesh(2, 2, 1);
        // Mesh draw 2: groups (0,0,0) and (0,0,1), columns 0 and 1 of row 0, moved by (1, -1), green -> the top half
        // of the bottom-right quadrant.
        list->SetGraphicsRoot32BitConstants(0, 2, down_right, 0);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 256);
        list6->DispatchMesh(1, 1, 2);
        // The stream's VS PSO: a full-screen triangle through a viewport on the bottom-left quadrant, yellow.
        const D3D12_VIEWPORT bottom_left{0, float(size / 2), float(size / 2), float(size / 2), 0, 1};
        list->RSSetViewports(1, &bottom_left);
        list->SetPipelineState(vs_pso.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
        list->DrawInstanced(3, 1, 0, 0);

        D3D12_RESOURCE_BARRIER to_copy =
            transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, size, size, 1, 256}};
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        std::memcpy(pixels.data(), mapped, pixels.size());

        // The wrong kind of draw for the pipeline fails at execution: a vertex draw with the mesh PSO, then
        // DispatchMesh with the VS PSO.
        for (const bool vertex_draw : {true, false}) {
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, vertex_draw ? mesh_pso.p
                                                                                                    : vs_pso.p)));
            list->SetGraphicsRootSignature(root.p);
            list->RSSetViewports(1, &viewport);
            list->RSSetScissorRects(1, &scissor);
            list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            if (vertex_draw) list->DrawInstanced(3, 1, 0, 0);
            else list6->DispatchMesh(1, 1, 1);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::invalid_draw);
        }
        // A grid beyond D3D12's limits (65535 per count, 2^22 groups) is a recording error.
        EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, mesh_pso.p)));
        list->SetGraphicsRootSignature(root.p);
        list->RSSetViewports(1, &viewport);
        list6->DispatchMesh(65536, 1, 1);
        EXPECT(list->Close() == E_INVALIDARG);
        EXPECT(SUCCEEDED(list->Reset(allocator.p, mesh_pso.p)));
        list->SetGraphicsRootSignature(root.p);
        list->RSSetViewports(1, &viewport);
        list6->DispatchMesh(4096, 1025, 1);
        EXPECT(list->Close() == E_INVALIDARG);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 2)) && SUCCEEDED(fence->SetEventOnCompletion(2, nullptr)));

        // D3D-291: pipeline statistics around mesh draws. Query 0 holds a vertex draw, two mesh draws and another
        // vertex draw, which the layer splits into a full, a mesh and a full segment; query 1 holds one mesh draw;
        // query 2 the two vertex draws alone. A mesh draw has no input assembly or vertex shader, so those fields
        // count only the two full-screen triangles (3 vertices, 1 primitive each). How many clipping and pixel
        // shader invocations a draw makes is the driver's (RADV's PSInvocations is not the pixel count), so query
        // 0 is checked against query 2: more of both is the mesh segment's share. Under LEONIDA_VK_VALIDATION=1 the
        // run must stay free of VUID-vkCmdDrawMeshTasksEXT-pipelineStatistics-07076.
        if (d.pipeline_statistics_query) {
            const D3D12_QUERY_HEAP_DESC statistics_desc{D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS, 3, 0};
            Ref<ID3D12QueryHeap> statistics;
            EXPECT(SUCCEEDED(device->CreateQueryHeap(&statistics_desc, IID_PPV_ARGS(&statistics.p))));
            Ref<ID3D12Resource> results;
            const auto results_desc = buffer_desc(3 * sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &results_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&results.p))));
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, mesh_pso.p)));
            D3D12_RESOURCE_BARRIER back_to_target =
                transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ResourceBarrier(1, &back_to_target);
            list->SetGraphicsRootSignature(root.p);
            list->RSSetScissorRects(1, &scissor);
            list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
            list->SetGraphicsRootConstantBufferView(1, upload_va);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            const auto vertex_draw = [&] {
                list->SetPipelineState(vs_pso.p);
                list->RSSetViewports(1, &bottom_left);
                list->DrawInstanced(3, 1, 0, 0);
            };
            const auto mesh_draw = [&](UINT x, UINT y) {
                list->SetPipelineState(mesh_pso.p);
                list->RSSetViewports(1, &viewport);
                list6->DispatchMesh(x, y, 1);
            };
            list->BeginQuery(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
            vertex_draw();
            mesh_draw(2, 2);
            mesh_draw(1, 1);
            vertex_draw();
            list->EndQuery(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
            list->BeginQuery(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 1);
            mesh_draw(2, 2);
            list->EndQuery(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 1);
            list->BeginQuery(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 2);
            vertex_draw();
            vertex_draw();
            list->EndQuery(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 2);
            list->ResolveQueryData(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0, 3, results.p, 0);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 3)) && SUCCEEDED(fence->SetEventOnCompletion(3, nullptr)));
            D3D12_QUERY_DATA_PIPELINE_STATISTICS data[3]{};
            void* stats_mapped = nullptr;
            EXPECT(SUCCEEDED(results->Map(0, nullptr, &stats_mapped)));
            std::memcpy(data, stats_mapped, sizeof(data));
            results->Unmap(0, nullptr);
            const auto& mixed = data[0];
            const auto& mesh_only = data[1];
            const auto& vertex_only = data[2];
            std::printf("statistics, mixed: IA %llu/%llu VS %llu C %llu/%llu PS %llu CS %llu; mesh only: C %llu/%llu "
                        "PS %llu; vertex only: C %llu PS %llu\n",
                        mixed.IAVertices, mixed.IAPrimitives, mixed.VSInvocations, mixed.CInvocations,
                        mixed.CPrimitives, mixed.PSInvocations, mixed.CSInvocations, mesh_only.CInvocations,
                        mesh_only.CPrimitives, mesh_only.PSInvocations, vertex_only.CInvocations,
                        vertex_only.PSInvocations);
            EXPECT(mixed.IAVertices == 6 && mixed.IAPrimitives == 2 && mixed.VSInvocations >= 1);
            EXPECT(vertex_only.IAVertices == 6 && vertex_only.IAPrimitives == 2);
            EXPECT(mixed.GSInvocations == 0 && mixed.GSPrimitives == 0 && mixed.HSInvocations == 0 &&
                   mixed.DSInvocations == 0 && mixed.CSInvocations == 0);
            EXPECT(mixed.CInvocations > vertex_only.CInvocations && mixed.PSInvocations > vertex_only.PSInvocations);
            EXPECT(mesh_only.IAVertices == 0 && mesh_only.IAPrimitives == 0 && mesh_only.VSInvocations == 0 &&
                   mesh_only.GSInvocations == 0 && mesh_only.GSPrimitives == 0 && mesh_only.HSInvocations == 0 &&
                   mesh_only.DSInvocations == 0 && mesh_only.CSInvocations == 0);
            EXPECT(mesh_only.CInvocations > 0 && mesh_only.PSInvocations > 0);
        } else {
            std::printf("pipelineStatisticsQuery not available: the D3D-291 statistics check is skipped\n");
        }

        for (auto* r : {&upload, &readback, &target}) r->reset();
        for (auto* p : {&mesh_pso, &vs_pso, &refused}) p->reset();
        root.reset();
        rtv_heap.reset();
        list6.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }
    std::filesystem::remove_all(library);

    unsigned mismatches = 0;
    for (UINT row = 0; row < size; ++row) {
        for (UINT col = 0; col < size; ++col) {
            const bool top = row < size / 2, left = col < size / 2;
            using Pixel = std::array<std::uint8_t, 4>;
            const Pixel want = top && left                            ? Pixel{255, 0, 0, 255}
                               : !top && left                         ? Pixel{255, 255, 0, 255}
                               : !top && !left && row < size * 3 / 4  ? Pixel{0, 255, 0, 255}
                                                                      : Pixel{0, 0, 255, 255};
            const std::uint8_t* got = pixels.data() + row * 256 + col * 4;
            if (std::equal(want.begin(), want.end(), got)) continue;
            if (mismatches++ < 6)
                std::fprintf(stderr, "pixel (%u, %u): got %u,%u,%u,%u want %u,%u,%u,%u\n", col, row, got[0], got[1],
                             got[2], got[3], want[0], want[1], want[2], want[3]);
        }
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("D3D-29: mesh tier 1; mesh and VS PSOs from pipeline state streams (9 refused as invalid, 2 as not "
                "implemented); DispatchMesh over X, Y and Z and a stream VS draw matched all %u pixels; wrong-kind "
                "draws fail at execution, oversized grids at Close; statistics queries around mesh draws resolve\n",
                size * size);
    return 0;
}
