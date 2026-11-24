// Typed RTVs of every TYPELESS color family (D3D-275): ClearRenderTargetView through an RTV in each typed format of
// the family, read back byte for byte and compared with the host's D3D12 on the same adapter (matched by LUID, the
// D3D-188 rule) and with WARP. Before D3D-275 only the RGBA8 family was reinterpreted: an R16_UINT or R32G32_UINT RTV
// of a typeless texture cleared through the resource's own (float or UNORM) format.
// Then draws: a full-screen pass writing a constant (integers for UINT views, floats for FLOAT and UNORM ones) into a
// typed RTV of a typeless texture, whose PSO names the typed format, compared with the same pass on the host (HLSL
// compiled by d3dcompiler_47). The layer's shaders are GLSL stand-ins for translated ones (test/glsl/vs_fullscreen.vert,
// ps_typeless_float.frag, ps_typeless_uint.frag), found by the FSSM hash of their containers in a translator-style
// index; without glslangValidator at build time only the clears run. The draws run twice: into a 4x4 2D texture and
// into a 4-texel 1D one through a TEXTURE1D RTV (a 1D attachment view).
// A view format the layer does not map yet (its CheckFeatureSupport gives no RENDER_TARGET) is reported and skipped,
// so the R8 family and the 16-bit integer and normalized formats are covered as soon as their mapping lands.
// Synthetic data only. LEONIDA_VK_VALIDATION=1 runs it under VK_LAYER_KHRONOS_validation.
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#if defined(LEONIDA_TYPELESS_VS_SPV)
#include "d3d12_test_shaders.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#endif

#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

struct Family {
    DXGI_FORMAT typeless;
    std::vector<DXGI_FORMAT> views;
};

// Values that tell the interpretations apart: a fraction (UNORM / FLOAT), 7 (UINT keeps it, UNORM clamps it to 1),
// a negative one (SINT / SNORM keep it, unsigned ones clamp it), one past every 8- and 16-bit integer range.
constexpr float kValues[2][4] = {{0.5f, 7.0f, -1.0f, 300.0f}, {0.25f, 1.0f, -128.0f, 70000.0f}};

// A full-screen pass for one device: its root signature blob and VS / PS bytecode (shader containers for the layer,
// DXBC for the host).
struct Pass {
    const std::vector<std::uint8_t>* root;
    const std::vector<std::uint8_t>* vs;
    const std::vector<std::uint8_t>* ps;
};

// Renders into a 4x4 TYPELESS texture (`line`: a 4-texel 1D one) through an RTV in `view` and reads it back: a clear
// to `value`, or (`pass` given) a full-screen draw whose PSO names `view`. False when a step failed.
bool render_and_read(ID3D12Device* device, DXGI_FORMAT typeless, DXGI_FORMAT view, const float* value,
                     const Pass* pass, std::vector<unsigned char>& bytes, bool line = false) {
    const UINT height = line ? 1 : 4;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = line ? D3D12_RESOURCE_DIMENSION_TEXTURE1D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 4;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = typeless;
    desc.SampleDesc = {1, 0};
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const D3D12_HEAP_PROPERTIES def_heap{D3D12_HEAP_TYPE_DEFAULT}, rb_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> texture, readback;
    if (FAILED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                               nullptr, IID_PPV_ARGS(&texture.p))))
        return false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes, &total);
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc = {1, 0};
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&readback.p))))
        return false;
    Ref<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p)))) return false;
    D3D12_RENDER_TARGET_VIEW_DESC rtv{};
    rtv.Format = view;
    rtv.ViewDimension = line ? D3D12_RTV_DIMENSION_TEXTURE1D : D3D12_RTV_DIMENSION_TEXTURE2D;
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(texture.p, &rtv, handle);
    Ref<ID3D12RootSignature> root;
    Ref<ID3D12PipelineState> pso;
    if (pass) {
        if (FAILED(device->CreateRootSignature(0, pass->root->data(), pass->root->size(), IID_PPV_ARGS(&root.p))))
            return false;
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = root.p;
        pd.VS = {pass->vs->data(), pass->vs->size()};
        pd.PS = {pass->ps->data(), pass->ps->size()};
        pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pd.SampleMask = UINT_MAX;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RasterizerState.DepthClipEnable = TRUE;
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.NumRenderTargets = 1;
        pd.RTVFormats[0] = view;
        pd.SampleDesc = {1, 0};
        if (FAILED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p)))) return false;
    }
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;
    if (pass) {
        const D3D12_VIEWPORT viewport{0, 0, 4, FLOAT(height), 0, 1};
        const D3D12_RECT scissor{0, 0, 4, LONG(height)};
        list->SetGraphicsRootSignature(root.p);
        list->SetPipelineState(pso.p);
        list->OMSetRenderTargets(1, &handle, FALSE, nullptr);
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->DrawInstanced(3, 1, 0, 0);
    } else {
        list->ClearRenderTargetView(handle, value, 0, nullptr);
    }
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {texture.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET,
                          D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION src{texture.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1)) || FAILED(fence->SetEventOnCompletion(1, nullptr))) return false;
    void* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, &mapped))) return false;
    bytes.clear();
    for (UINT y = 0; y < rows; ++y) {
        const auto* row = static_cast<const unsigned char*>(mapped) + footprint.Offset + y * footprint.Footprint.RowPitch;
        bytes.insert(bytes.end(), row, row + row_bytes);
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}

// The host's D3D12 device on the adapter whose LUID the Vulkan device reports (not the default adapter: D3D-188).
bool host_device(const vktest::Device& vk, Ref<ID3D12Device>& out, std::string& name) {
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    vk.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(vk.physical, &props);
    Ref<IDXGIFactory4> factory;
    if (!id.deviceLUIDValid || FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return false;
    LUID luid{};
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));
    Ref<IDXGIAdapter1> adapter;
    if (FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter.p)))) return false;
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    char utf8[256]{};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, utf8, sizeof(utf8), nullptr, nullptr);
    name = utf8;
    return SUCCEEDED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&out.p)));
}

std::string hex(const std::vector<unsigned char>& bytes, std::size_t texel) {
    std::string s;
    char b[4];
    for (std::size_t i = 0; i < texel && i < bytes.size(); ++i) {
        std::snprintf(b, sizeof(b), "%02x", bytes[i]);
        s += b;
    }
    return s;
}

// The layer and the host report the format in the same way: no RENDER_TARGET support means the layer has no mapping.
bool renders(ID3D12Device* device, DXGI_FORMAT typeless, DXGI_FORMAT view) {
    D3D12_FEATURE_DATA_FORMAT_SUPPORT typeless_support{typeless}, support{view};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &typeless_support,
                                                 sizeof(typeless_support))) &&
           (typeless_support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) &&
           SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) &&
           (support.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET);
}

#if defined(LEONIDA_TYPELESS_VS_SPV)
// HLSL for the host's pass, the same constants as the GLSL stand-ins.
constexpr char kHostVs[] =
    "float4 main(uint id : SV_VertexID) : SV_Position {\n"
    "    float2 corner = float2(id & 2, (id << 1) & 2);\n"
    "    return float4(corner * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";
constexpr char kHostPsFloat[] = "float4 main() : SV_Target { return float4(0.5, -2.0, 1.5, 0.25); }\n";
constexpr char kHostPsUint[] = "uint4 main() : SV_Target { return uint4(7, 300, 1000, 3); }\n";

bool compile_hlsl(pD3DCompile compile, const char* source, const char* target, std::vector<std::uint8_t>& out) {
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = compile(source, std::strlen(source), nullptr, nullptr, nullptr, "main", target, 0, 0, &code,
                               &errors);
    if (errors) {
        std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
        errors->Release();
    }
    if (FAILED(hr) || !code) return false;
    const auto* p = static_cast<const std::uint8_t*>(code->GetBufferPointer());
    out.assign(p, p + code->GetBufferSize());
    code->Release();
    return true;
}
#endif

}  // namespace

int main() {
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (hostwine::host_is_wine()) return 77;  // the host D3D12 runtime is the reference
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.bindless = vk.bindless;
    info.bindless_partially_bound = vk.partially_bound;
    info.host_image_copy = vk.host_image_copy;
    info.buffer_device_address = vk.buffer_device_address;
#if defined(LEONIDA_TYPELESS_VS_SPV)
    // The layer's pass: shader containers around synthetic FSSM programs, and the index that maps their hashes to the
    // compiled GLSL stand-ins.
    const auto vs_program = d3d12test::fssm_program({0x56530275, 0xbf810000});
    const auto ps_float_program = d3d12test::fssm_program({0x50530275, 0xbf810000});
    const auto ps_uint_program = d3d12test::fssm_program({0x50530276, 0xbf810000});
    const auto vs_container = d3d12test::shader_container(vs_program);
    const auto ps_float_container = d3d12test::shader_container(ps_float_program);
    const auto ps_uint_container = d3d12test::shader_container(ps_uint_program);
    const testfiles::ScratchDir scratch("leonida_typeless_families_test");
    const auto& library = scratch.path();
    if (!testfiles::copy_file(LEONIDA_TYPELESS_VS_SPV, library / "vs.spv") ||
        !testfiles::copy_file(LEONIDA_TYPELESS_PS_FLOAT_SPV, library / "ps_float.spv") ||
        !testfiles::copy_file(LEONIDA_TYPELESS_PS_UINT_SPV, library / "ps_uint.spv"))
        return 1;
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_float\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_float_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]},\n"
                 " \"ps_uint\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_uint_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }
    info.shader_dir = library.string();
#endif
    Ref<ID3D12Device> ours, host, warp;
    std::string adapter;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
    if (!host_device(vk, host, adapter)) {
        std::fprintf(stderr, "no host D3D12 device on the Vulkan device's adapter\n");
        return 77;
    }
    {
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> warp_adapter;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) &&
            SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp_adapter.p))))
            D3D12CreateDevice(warp_adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&warp.p));
    }
    std::printf("host reference: %s%s\n", adapter.c_str(), warp.p ? ", and WARP" : "");
    const std::vector<Family> families = {
        {DXGI_FORMAT_R8G8B8A8_TYPELESS, {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                                         DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R8G8B8A8_SNORM, DXGI_FORMAT_R8G8B8A8_SINT}},
        {DXGI_FORMAT_B8G8R8A8_TYPELESS, {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB}},
        {DXGI_FORMAT_R16G16B16A16_TYPELESS, {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_UNORM,
                                             DXGI_FORMAT_R16G16B16A16_UINT, DXGI_FORMAT_R16G16B16A16_SNORM,
                                             DXGI_FORMAT_R16G16B16A16_SINT}},
        {DXGI_FORMAT_R32G32B32A32_TYPELESS, {DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_UINT,
                                             DXGI_FORMAT_R32G32B32A32_SINT}},
        {DXGI_FORMAT_R32G32_TYPELESS, {DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_SINT}},
        {DXGI_FORMAT_R16G16_TYPELESS, {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_UNORM, DXGI_FORMAT_R16G16_UINT,
                                       DXGI_FORMAT_R16G16_SNORM, DXGI_FORMAT_R16G16_SINT}},
        {DXGI_FORMAT_R10G10B10A2_TYPELESS, {DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R10G10B10A2_UINT}},
        {DXGI_FORMAT_R8G8_TYPELESS, {DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8_UINT, DXGI_FORMAT_R8G8_SNORM,
                                     DXGI_FORMAT_R8G8_SINT}},
        {DXGI_FORMAT_R32_TYPELESS, {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_SINT}},
        {DXGI_FORMAT_R16_TYPELESS, {DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16_UINT,
                                    DXGI_FORMAT_R16_SNORM, DXGI_FORMAT_R16_SINT}},
        {DXGI_FORMAT_R8_TYPELESS, {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8_SNORM, DXGI_FORMAT_R8_SINT}},
    };
    unsigned compared = 0, skipped = 0, failed = 0, warp_differs = 0;
    for (const auto& family : families) {
        for (const DXGI_FORMAT view : family.views) {
            if (!renders(ours.p, family.typeless, view)) {
                std::printf("typeless %d view %d: SKIP, the layer has no render target in this format yet\n",
                            int(family.typeless), int(view));
                ++skipped;
                continue;
            }
            for (const auto& value : kValues) {
                std::vector<unsigned char> mine, reference, software;
                if (!render_and_read(host.p, family.typeless, view, value, nullptr, reference)) {
                    std::fprintf(stderr, "typeless %d view %d: the host refused\n", int(family.typeless), int(view));
                    return 1;
                }
                if (!render_and_read(ours.p, family.typeless, view, value, nullptr, mine)) {
                    std::fprintf(stderr, "typeless %d view %d: the layer failed\n", int(family.typeless), int(view));
                    return 1;
                }
                const std::size_t texel = reference.size() / 16;
                ++compared;
                if (warp.p && render_and_read(warp.p, family.typeless, view, value, nullptr, software) &&
                    software != reference)
                    ++warp_differs;
                // The layer rounds a normalised clear value to nearest even, as D3D does (#521): at a tie (0.5 on 8
                // bits) NVIDIA's D3D12 driver writes one code lower than WARP and AMD, so WARP's bytes also count.
                const bool same = mine == reference || (!software.empty() && mine == software);
                failed += !same;
                std::printf("typeless %d view %d clear (%g, %g, %g, %g): host %s, layer %s%s%s\n", int(family.typeless),
                            int(view), value[0], value[1], value[2], value[3], hex(reference, texel).c_str(),
                            hex(mine, texel).c_str(), same ? "" : "  MISMATCH",
                            software.empty() || software == reference ? "" : ("  WARP " + hex(software, texel)).c_str());
            }
        }
    }
    std::printf("%u clears compared with %s, %u differ; %u view formats skipped; WARP differs from the host in %u\n",
                compared, adapter.c_str(), failed, skipped, warp_differs);

    unsigned drawn = 0, draws_failed = 0;
#if defined(LEONIDA_TYPELESS_VS_SPV)
    // Draws: the PSO names the typed format, the attachment must be a view of the typeless image in that format.
    const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile = compiler ? reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(
                                        GetProcAddress(compiler, "D3DCompile")))
                                  : nullptr;
    std::vector<std::uint8_t> host_vs, host_ps_float, host_ps_uint;
    if (!compile || !compile_hlsl(compile, kHostVs, "vs_5_0", host_vs) ||
        !compile_hlsl(compile, kHostPsFloat, "ps_5_0", host_ps_float) ||
        !compile_hlsl(compile, kHostPsUint, "ps_5_0", host_ps_uint)) {
        std::fprintf(stderr, "d3dcompiler_47 could not compile the host's pass\n");
        return 1;
    }
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    const auto root_blob = d3d12test::serialize_root_signature(root_desc);
    if (root_blob.empty()) return 1;
    const Pass host_float{&root_blob, &host_vs, &host_ps_float}, host_uint{&root_blob, &host_vs, &host_ps_uint};
    const Pass ours_float{&root_blob, &vs_container, &ps_float_container};
    const Pass ours_uint{&root_blob, &vs_container, &ps_uint_container};
    struct DrawCase {
        DXGI_FORMAT typeless, view;
        bool uint;
    };
    const DrawCase draws[] = {
        {DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_R16G16B16A16_UINT, true},
        {DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_R16G16B16A16_FLOAT, false},
        {DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_R16G16B16A16_UNORM, false},
        {DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_UINT, true},
        {DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_FLOAT, false},
        {DXGI_FORMAT_R10G10B10A2_TYPELESS, DXGI_FORMAT_R10G10B10A2_UINT, true},
        {DXGI_FORMAT_R10G10B10A2_TYPELESS, DXGI_FORMAT_R10G10B10A2_UNORM, false},
        {DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UINT, true},
    };
    for (const bool line : {false, true}) {
        for (const auto& c : draws) {
            if (!renders(ours.p, c.typeless, c.view)) {
                std::printf("typeless %d view %d draw: SKIP, the layer has no render target in this format yet\n",
                            int(c.typeless), int(c.view));
                ++skipped;
                continue;
            }
            const char* shape = line ? "1D " : "";
            std::vector<unsigned char> mine, reference;
            if (!render_and_read(host.p, c.typeless, c.view, nullptr, c.uint ? &host_uint : &host_float, reference,
                                 line)) {
                std::fprintf(stderr, "typeless %d view %d %sdraw: the host refused\n", int(c.typeless), int(c.view),
                             shape);
                return 1;
            }
            if (!render_and_read(ours.p, c.typeless, c.view, nullptr, c.uint ? &ours_uint : &ours_float, mine,
                                 line)) {
                std::fprintf(stderr, "typeless %d view %d %sdraw: the layer failed\n", int(c.typeless), int(c.view),
                             shape);
                return 1;
            }
            ++drawn;
            const bool same = mine == reference;
            draws_failed += !same;
            const std::size_t texel = reference.size() / (line ? 4 : 16);
            std::printf("typeless %d view %d %sdraw (%s): host %s, layer %s%s\n", int(c.typeless), int(c.view), shape,
                        c.uint ? "7, 300, 1000, 3" : "0.5, -2, 1.5, 0.25", hex(reference, texel).c_str(),
                        hex(mine, texel).c_str(), same ? "" : "  MISMATCH");
        }
    }
    std::printf("%u draws compared with %s, %u differ\n", drawn, adapter.c_str(), draws_failed);
#endif
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", vk.validation_errors());
        return 1;
    }
    return failed || draws_failed ? 1 : 0;
}
