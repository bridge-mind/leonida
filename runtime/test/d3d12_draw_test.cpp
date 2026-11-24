// Milestone M1 (docs/d3d12_vulkan.md section 6) through the Xbox-ABI proxy over the Vulkan device:
//   - a graphics PSO whose VS and PS come from containers shaped like the game's, found in a translator-style
//     index by the SHA-256 of their FSSM programs; the input layout (POSITION, COLOR) is matched to the VS input
//     signature by semantic, and the VS declares them in the opposite register order;
//   - a vertex buffer copied into the GPU address arena, an index buffer read straight from an upload heap, root
//     constants (a clip-space offset) and a root CBV (a tint) through the 32 user-data dwords;
//   - a render target cleared blue, two indexed draws with different root arguments (top-left red, bottom-right
//     green), and one back-facing triangle (bottom-left) that culling must remove;
//   - the image read back through CopyTextureRegion and compared pixel by pixel. Where the quadrants land also
//     checks the viewport flip (D3D clip space is y-up) and the front-face winding;
//   - full-screen blends that read destination alpha into B8G8R8X8 and B8G8R8A8 targets cleared with alpha 0.25:
//     B8G8R8X8 has no alpha, so destination alpha reads 1 there, as on WARP and AMD's D3D12 driver;
//   - draws through RTVs in a format the texture's castable list names, in the other channel order (RGBA8 and BGRA8)
//     or across B8G8R8X8 and B8G8R8A8: the bytes land in the view's order and encoding.
// The shaders are GLSL written to the translator's contract (test/glsl/vs_draw.vert, ps_draw.frag). No game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV) || !defined(LEONIDA_TEST_VS_INT_SPV) || \
    !defined(LEONIDA_TEST_PS_UINT_SPV)
#error "LEONIDA_TEST_VS_SPV, LEONIDA_TEST_PS_SPV, LEONIDA_TEST_PS_UINT_SPV and LEONIDA_TEST_VS_INT_SPV must name the compiled test shaders"
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
    std::uint8_t marker = 0, padding = 0;
    std::uint8_t narrow_color[2] = {255, 0};
    std::uint8_t wrong_color[2] = {0, 255};
};
static_assert(offsetof(Vertex, marker) == 12);
static_assert(offsetof(Vertex, narrow_color) == 14);
static_assert(offsetof(Vertex, wrong_color) == 16);

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    // Shader programs and the index that maps their hashes to the compiled test modules.
    const auto vs_program = fssm_program({0x56530001, 0xbf810000});
    const auto ps_program = fssm_program({0x50530001, 0xbf810000});
    const auto fullscreen_program = fssm_program({0x56530002, 0xbf810000});
    const auto vs_int_program = fssm_program({0x56530003, 0xbf810000});
    const auto ps_uint_program = fssm_program({0x50530002, 0xbf810000});
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    const auto fullscreen_container = shader_container(fullscreen_program);
    const auto vs_int_container = shader_container(vs_int_program), ps_uint_container = shader_container(ps_uint_program);
    const testfiles::ScratchDir scratch("leonida_draw_test");
    const auto& library = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_SPV), library / "vs_draw.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), library / "ps_draw.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_UINT_SPV), library / "ps_draw_uint.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_FULLSCREEN_SPV),
                                library / "vs_fullscreen.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_INT_SPV), library / "vs_draw_int.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_draw\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}, "
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 1, \"mask\": 3, \"rw_mask\": 3, "
                 "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
                 " \"vs_draw_int\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_int_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"int\"}, "
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 1, \"mask\": 3, \"rw_mask\": 3, "
                 "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
                 " \"vs_fullscreen\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(fullscreen_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]},\n"
                 " \"ps_draw_uint\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_uint_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }

    constexpr UINT size = 64;
    std::vector<std::uint8_t> pixels(size * 256), second_pixels(size * 256), logic_pixels(size * 256);
    std::vector<std::uint8_t> strip16_pixels(size * 256), strip32_pixels(size * 256), strip_nocut_pixels(size * 256);
    std::vector<std::uint8_t> msaa_full_pixels(size * 256), msaa_half_pixels(size * 256);
    bool ran_msaa = false;
    std::vector<std::uint8_t> step2_pixels(size * 256), step1_pixels(size * 256);
    const bool logic_op = d.enabled.logicOp == VK_TRUE;
    const bool has_divisor = d.vertex_attribute_divisor;
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
        info.shader_dir = library.string();
        // Of the PSO features only logicOp is passed on, so the other feature refusals below still apply.
        info.features.logicOp = d.enabled.logicOp;
        info.vertex_attribute_divisor = d.vertex_attribute_divisor;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

        // Root signature: [0] two constants (s0, s1: offset), [1] a root CBV (s2:s3: tint).
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

        // The PSO. Input layout order (POSITION, COLOR) differs from the VS's registers (COLOR 0, POSITION 1).
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
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.InputLayout = {elements, 2};
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.SampleDesc.Count = 1;
        Ref<ID3D12PipelineState> pso, unfed, mrt, instanced, refused, xor_pso, fullscreen;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));
        pso_desc.InputLayout = {elements, 1};  // COLOR missing: D3D12 rejects a layout that leaves an input unfed
        EXPECT(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&unfed.p)) == E_INVALIDARG && !unfed.p);
        pso_desc.InputLayout = {elements, 2};

        // The narrow signed IA formats the draw path used to refuse; each maps to a Vulkan vertex format, so a PSO
        // whose COLOR input uses one is created.
        const DXGI_FORMAT narrow_signed_formats[] = {
            DXGI_FORMAT_R16_SNORM, DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R8G8_SNORM,
            DXGI_FORMAT_R8G8_SINT, DXGI_FORMAT_R8_SNORM, DXGI_FORMAT_R8_SINT};
        for (DXGI_FORMAT format : narrow_signed_formats) {
            D3D12_INPUT_ELEMENT_DESC narrow_elements[2] = {elements[0], elements[1]};
            narrow_elements[1].Format = format;
            D3D12_GRAPHICS_PIPELINE_STATE_DESC narrow_desc = pso_desc;
            narrow_desc.InputLayout = {narrow_elements, 2};
            // SINT formats feed an integer shader input; SNORM formats feed the float shader.
            if (format == DXGI_FORMAT_R16_SINT || format == DXGI_FORMAT_R8G8_SINT || format == DXGI_FORMAT_R8_SINT)
                narrow_desc.VS = {vs_int_container.data(), vs_int_container.size()};
            Ref<ID3D12PipelineState> narrow;
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&narrow_desc, IID_PPV_ARGS(&narrow.p))));
        }

        // Two render-target formats and a depth format, of which draws bind only RTV 0: a variant with the other
        // formats undefined is used at draw time.
        // The unused R8 element ends at byte 13. An appended R8G8 color starts at 14, while rounding to
        // four bytes would fetch the sentinel at 16 and turn the top-left draw black.
        const D3D12_INPUT_ELEMENT_DESC narrow_elements[] = {
            elements[0],
            {"MARKER", 0, DXGI_FORMAT_R8_UNORM, 0, offsetof(Vertex, marker),
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8_UNORM, 0, D3D12_APPEND_ALIGNED_ELEMENT,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC mrt_desc = pso_desc;
        mrt_desc.InputLayout = {narrow_elements, 3};
        mrt_desc.NumRenderTargets = 2;
        mrt_desc.RTVFormats[1] = DXGI_FORMAT_R8G8B8A8_UNORM;
        mrt_desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&mrt_desc, IID_PPV_ARGS(&mrt.p))));

        // COLOR per instance from slot 1 with step rate 0: every instance reads element 0.
        const D3D12_INPUT_ELEMENT_DESC instance_elements[2] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC instance_desc = pso_desc;
        instance_desc.InputLayout = {instance_elements, 2};
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&instance_desc, IID_PPV_ARGS(&instanced.p))));

        // CachedPSO on a graphics description: what the PC runtime matches, measured on an RTX 3060
        // (src/cached_pso.h). Flags is not part of it; every state below is.
        {
            Ref<ID3DBlob> blob;
            EXPECT(pso->GetCachedBlob(&blob.p) == S_OK && blob.p && blob->GetBufferSize());
            auto with_blob = [&](D3D12_GRAPHICS_PIPELINE_STATE_DESC d) {
                d.CachedPSO = {blob->GetBufferPointer(), blob->GetBufferSize()};
                Ref<ID3D12PipelineState> made;
                const HRESULT hr = device->CreateGraphicsPipelineState(&d, IID_PPV_ARGS(&made.p));
                return SUCCEEDED(hr) == (made.p != nullptr) ? hr : E_UNEXPECTED;
            };
            EXPECT(with_blob(pso_desc) == S_OK);
            D3D12_GRAPHICS_PIPELINE_STATE_DESC changed = pso_desc;
            changed.Flags = D3D12_PIPELINE_STATE_FLAG_TOOL_DEBUG;
            EXPECT(with_blob(changed) == S_OK);
            EXPECT(with_blob(mrt_desc) == E_INVALIDARG);
            EXPECT(with_blob(instance_desc) == E_INVALIDARG);
            changed = pso_desc;
            changed.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            EXPECT(with_blob(changed) == E_INVALIDARG);
            changed = pso_desc;
            changed.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_RED;
            EXPECT(with_blob(changed) == E_INVALIDARG);
            changed = pso_desc;
            changed.SampleMask = 1;
            EXPECT(with_blob(changed) == E_INVALIDARG);
            changed = pso_desc;
            changed.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF;
            EXPECT(with_blob(changed) == E_INVALIDARG);
            changed = pso_desc;
            changed.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            EXPECT(with_blob(changed) == E_INVALIDARG);
        }

        // State that needs a Vulkan feature this test device does not enable is refused, not passed to the driver.
        D3D12_GRAPHICS_PIPELINE_STATE_DESC needs_feature = pso_desc;
        needs_feature.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;  // fillModeNonSolid
        EXPECT(device->CreateGraphicsPipelineState(&needs_feature, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);
        needs_feature = mrt_desc;
        needs_feature.DSVFormat = DXGI_FORMAT_UNKNOWN;
        needs_feature.BlendState.IndependentBlendEnable = TRUE;  // independentBlend: the two targets differ
        needs_feature.BlendState.RenderTarget[1].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_RED;
        EXPECT(device->CreateGraphicsPipelineState(&needs_feature, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);
        needs_feature = pso_desc;
        needs_feature.RasterizerState.DepthBias = 4;
        needs_feature.RasterizerState.DepthBiasClamp = 0.5f;  // depthBiasClamp
        EXPECT(device->CreateGraphicsPipelineState(&needs_feature, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);
        needs_feature = pso_desc;
        needs_feature.NumRenderTargets = 9;
        EXPECT(device->CreateGraphicsPipelineState(&needs_feature, IID_PPV_ARGS(&refused.p)) == E_INVALIDARG && !refused.p);
        needs_feature = pso_desc;
        needs_feature.RasterizerState.DepthClipEnable = FALSE;  // depthClamp
        EXPECT(device->CreateGraphicsPipelineState(&needs_feature, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);

        // XOR on render target 0, which must be a UINT format (D3D12 allows logic ops on no other, and Intel Arc
        // returns the source on UNORM): applied where the GPU has logicOp (the device passes it on), refused elsewhere.
        D3D12_GRAPHICS_PIPELINE_STATE_DESC xor_desc = pso_desc;
        xor_desc.PS = {ps_uint_container.data(), ps_uint_container.size()};
        xor_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UINT;
        xor_desc.BlendState.RenderTarget[0].LogicOpEnable = TRUE;
        xor_desc.BlendState.RenderTarget[0].LogicOp = D3D12_LOGIC_OP_XOR;
        EXPECT(device->CreateGraphicsPipelineState(&xor_desc, IID_PPV_ARGS(&xor_pso.p)) ==
               (logic_op ? S_OK : E_NOTIMPL));
        // A full-screen pass: a VS with no inputs and an empty input layout, so the pipeline has no vertex bindings.
        D3D12_GRAPHICS_PIPELINE_STATE_DESC fullscreen_desc = pso_desc;
        fullscreen_desc.VS = {fullscreen_container.data(), fullscreen_container.size()};
        fullscreen_desc.InputLayout = {nullptr, 0};
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&fullscreen_desc, IID_PPV_ARGS(&fullscreen.p))));

        // Geometry: a quad in the top-left quadrant (front-facing, clockwise in D3D screen space) and a triangle in
        // the bottom-left one wound the other way (back-facing).
        const Vertex vertices[7] = {
            {-1, 1, 255, 0, 0, 255}, {0, 1, 255, 0, 0, 255}, {-1, 0, 255, 0, 0, 255}, {0, 0, 255, 0, 0, 255},
            {-1, 0, 255, 0, 0, 255}, {-1, -1, 255, 0, 0, 255}, {0, -1, 255, 0, 0, 255}};
        const std::uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
        const float tints[2][3] = {{1, 0, 1}, {0, 1, 0}};  // red stays red; then green

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> vertex_buffer, upload, readback, target, uint_target;
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

        // Buffers D3D12 refuses: no bytes, or a size that wraps when aligned. One the arena cannot hold, and a
        // shader-visible heap it cannot hold (512 MiB of descriptors in the 256 MiB default), run out of memory.
        {
            Ref<ID3D12Resource> bad;
            for (const UINT64 width : {UINT64(0), ~UINT64(0) - 65535}) {
                const auto bad_desc = buffer_desc(width);
                EXPECT(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &bad_desc,
                                                       D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                       IID_PPV_ARGS(&bad.p)) == E_INVALIDARG && !bad.p);
            }
            const auto huge = buffer_desc(UINT64(1) << 40);
            EXPECT(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &huge,
                                                   D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                   IID_PPV_ARGS(&bad.p)) == E_OUTOFMEMORY && !bad.p);
            const D3D12_DESCRIPTOR_HEAP_DESC too_many{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1u << 24,
                                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
            Ref<ID3D12DescriptorHeap> big_heap;
            EXPECT(device->CreateDescriptorHeap(&too_many, IID_PPV_ARGS(&big_heap.p)) == E_OUTOFMEMORY && !big_heap.p);
        }
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

        // Upload layout: vertices at 0, indices at 256, the tints at 512 and 768 (CBVs are 256-byte aligned).
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memcpy(up, vertices, sizeof(vertices));
        std::memcpy(up + 256, indices, sizeof(indices));
        std::memcpy(up + 512, tints[0], sizeof(tints[0]));
        std::memcpy(up + 768, tints[1], sizeof(tints[1]));
        const std::uint8_t instance_colors[2][4] = {{255, 255, 0, 255}, {255, 0, 255, 255}};  // yellow, magenta
        std::memcpy(up + 1024, instance_colors, sizeof(instance_colors));
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
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        // Upload the vertices, prepare the target.
        list->CopyBufferRegion(vertex_buffer.p, 0, upload.p, 0, sizeof(vertices));
        D3D12_RESOURCE_BARRIER barriers[2] = {
            transition(vertex_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER),
            transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)};
        list->ResourceBarrier(2, barriers);
        const FLOAT blue[4] = {0, 0, 1, 1};
        list->ClearRenderTargetView(rtv, blue, 0, nullptr);

        // State for the draws; the pipeline came with CreateCommandList.
        list->SetGraphicsRootSignature(root.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VERTEX_BUFFER_VIEW vbv{vertex_buffer->GetGPUVirtualAddress(), sizeof(vertices), sizeof(Vertex)};
        list->IASetVertexBuffers(0, 1, &vbv);
        const D3D12_INDEX_BUFFER_VIEW ibv{upload_va + 256, sizeof(indices), DXGI_FORMAT_R16_UINT};
        list->IASetIndexBuffer(&ibv);
        const D3D12_VIEWPORT viewport{0, 0, float(size), float(size), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(size), LONG(size)};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

        // Draw 1: no offset, red tint -> top-left quadrant red. The two-target PSO with a depth format, with only
        // RTV 0 and no DSV bound: it runs as a variant, validation-clean.
        const float origin[2] = {0, 0}, down_right[2] = {1, -1};
        list->SetPipelineState(mrt.p);
        list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
        list->DrawIndexedInstanced(6, 1, 0, 0, 0);
        // Draw 2: offset (1, -1), green tint -> bottom-right quadrant green. The root signature is set again after
        // the arguments; it is the same one, so they stay bound (else the quad would land top-left, untinted).
        list->SetPipelineState(pso.p);
        list->SetGraphicsRoot32BitConstants(0, 2, down_right, 0);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 768);
        list->SetGraphicsRootSignature(root.p);
        list->DrawIndexedInstanced(6, 1, 0, 0, 0);
        // Draw 3: the back-facing triangle in the bottom-left quadrant; culled, so that quadrant stays blue.
        list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
        list->DrawInstanced(3, 1, 4, 0);

        // Draws 4 and 5 through a viewport on the top-right quadrant, so each quad covers a 16x16 corner of it.
        const D3D12_VIEWPORT top_right{float(size / 2), 0, float(size / 2), float(size / 2), 0, 1};
        list->RSSetViewports(1, &top_right);
        list->SetPipelineState(instanced.p);
        // Draw 4: two instances, COLOR per instance with step rate 0, so both read yellow (red with the red tint);
        // a step rate of 1 would draw magenta over it.
        const D3D12_VERTEX_BUFFER_VIEW colors_view{upload_va + 1024, sizeof(instance_colors), 4};
        list->IASetVertexBuffers(1, 1, &colors_view);
        list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
        list->DrawIndexedInstanced(6, 2, 0, 0, 0);
        // Draw 5: slot 1 unbound. D3D12 reads zeros there, so COLOR is 0 and the corner is black, and the
        // submission must not fail.
        list->IASetVertexBuffers(1, 1, nullptr);
        list->SetGraphicsRoot32BitConstants(0, 2, down_right, 0);
        list->DrawIndexedInstanced(6, 1, 0, 0, 0);

        D3D12_RESOURCE_BARRIER to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
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
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        std::memcpy(pixels.data(), mapped, pixels.size());

        // Second pass on the same target, cleared blue again: the full-screen triangle (yellow, red tint: red)
        // through a viewport on the bottom-right quadrant, with no vertex buffer in its pipeline.
        EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, pso.p)));
        D3D12_RESOURCE_BARRIER to_target = transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &to_target);
        list->ClearRenderTargetView(rtv, blue, 0, nullptr);
        list->SetGraphicsRootSignature(root.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
        // The full-screen draw comes first, so no vertex buffer has been bound in this submission before it: a
        // pipeline that declared the stride state would break VUID-vkCmdDraw-pStrides-04913 here.
        const D3D12_VIEWPORT bottom_right{float(size / 2), float(size / 2), float(size / 2), float(size / 2), 0, 1};
        list->RSSetViewports(1, &bottom_right);
        list->SetPipelineState(fullscreen.p);
        list->DrawInstanced(3, 1, 0, 0);
        list->ResourceBarrier(1, &to_copy);
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        EXPECT(list->Close() == S_OK);
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 2)) && SUCCEEDED(fence->SetEventOnCompletion(2, nullptr)));
        std::memcpy(second_pixels.data(), mapped, second_pixels.size());

        // Logic-op pass on its own R8G8B8A8_UINT target: D3D12 applies logic ops to UINT formats only. Cleared to
        // (0,0,255,255); the red quad (255,0,0,255) XORed into the top-left quadrant gives (255,0,255,0) there.
        if (logic_op) {
            auto uint_rt_desc = rt_desc;
            uint_rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UINT;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uint_rt_desc,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&uint_target.p))));
            Ref<ID3D12DescriptorHeap> uint_rtv_heap;
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&uint_rtv_heap.p))));
            const D3D12_CPU_DESCRIPTOR_HANDLE uint_rtv{
                SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(uint_rtv_heap.p, 9))(uint_rtv_heap.p))};
            device->CreateRenderTargetView(uint_target.p, nullptr, uint_rtv);

            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, xor_pso.p)));
            D3D12_RESOURCE_BARRIER to_uint_target = transition(uint_target.p, D3D12_RESOURCE_STATE_COMMON,
                                                               D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ResourceBarrier(1, &to_uint_target);
            const FLOAT uint_blue[4] = {0, 0, 255, 255};  // a UINT target is cleared with whole numbers
            list->ClearRenderTargetView(uint_rtv, uint_blue, 0, nullptr);
            list->SetGraphicsRootSignature(root.p);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list->RSSetViewports(1, &viewport);
            list->RSSetScissorRects(1, &scissor);
            list->OMSetRenderTargets(1, &uint_rtv, FALSE, nullptr);
            list->IASetVertexBuffers(0, 1, &vbv);
            list->IASetIndexBuffer(&ibv);
            list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
            list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
            list->DrawIndexedInstanced(6, 1, 0, 0, 0);
            D3D12_RESOURCE_BARRIER uint_to_copy = transition(uint_target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                             D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &uint_to_copy);
            D3D12_TEXTURE_COPY_LOCATION uint_dst = dst, uint_src = src;
            uint_dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UINT;
            uint_src.pResource = uint_target.p;
            list->CopyTextureRegion(&uint_dst, 0, 0, 0, &uint_src, nullptr);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 3)) && SUCCEEDED(fence->SetEventOnCompletion(3, nullptr)));
            std::memcpy(logic_pixels.data(), mapped, logic_pixels.size());
        }

        // Third pass: indexed triangle-strip cuts (0xFFFF for UINT16, 0xFFFFFFFF for UINT32) and negative control
        // without a cut. Two disjoint quads (top-left and bottom-right); without the cut, bridge triangle (2, 3, 4)
        // crosses the bottom-left quadrant from (-1, 0) through (0, 0) to (1, -1).
        {
            const Vertex strip_verts[8] = {
                {-1, 1, 255, 0, 0, 255}, {0, 1, 255, 0, 0, 255}, {-1, 0, 255, 0, 0, 255}, {0, 0, 255, 0, 0, 255},
                {1, -1, 255, 0, 0, 255}, {0, -1, 255, 0, 0, 255}, {1, 0, 255, 0, 0, 255}, {0, 0, 255, 0, 0, 255},
            };
            const std::uint16_t strip_idx16[9] = {0, 1, 2, 3, 0xFFFFu, 4, 5, 6, 7};
            const std::uint32_t strip_idx32[9] = {0, 1, 2, 3, 0xFFFFFFFFu, 4, 5, 6, 7};
            const std::uint16_t strip_nocut[8] = {0, 1, 2, 3, 4, 5, 6, 7};
            void* up_ptr = nullptr;
            EXPECT(SUCCEEDED(upload->Map(0, nullptr, &up_ptr)));
            std::memcpy(static_cast<std::uint8_t*>(up_ptr) + 1024, strip_verts, sizeof(strip_verts));
            std::memcpy(static_cast<std::uint8_t*>(up_ptr) + 1280, strip_idx16, sizeof(strip_idx16));
            std::memcpy(static_cast<std::uint8_t*>(up_ptr) + 1536, strip_idx32, sizeof(strip_idx32));
            std::memcpy(static_cast<std::uint8_t*>(up_ptr) + 1792, strip_nocut, sizeof(strip_nocut));
            upload->Unmap(0, nullptr);

            auto strip_desc = pso_desc;
            strip_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            Ref<ID3D12PipelineState> strip16_pso, strip32_pso, strip_off_pso;
            strip_desc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF;
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&strip_desc, IID_PPV_ARGS(&strip16_pso.p))));
            strip_desc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&strip_desc, IID_PPV_ARGS(&strip32_pso.p))));
            strip_desc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&strip_desc, IID_PPV_ARGS(&strip_off_pso.p))));

            const D3D12_VERTEX_BUFFER_VIEW strip_vbv{upload_va + 1024, sizeof(strip_verts), sizeof(Vertex)};
            struct StripPass {
                ID3D12PipelineState* pso;
                D3D12_INDEX_BUFFER_VIEW ibv;
                UINT index_count;
                std::vector<std::uint8_t>* out;
            };
            const StripPass strip_passes[] = {
                {strip16_pso.p, {upload_va + 1280, sizeof(strip_idx16), DXGI_FORMAT_R16_UINT}, 9, &strip16_pixels},
                {strip32_pso.p, {upload_va + 1536, sizeof(strip_idx32), DXGI_FORMAT_R32_UINT}, 9, &strip32_pixels},
                {strip_off_pso.p, {upload_va + 1792, sizeof(strip_nocut), DXGI_FORMAT_R16_UINT}, 8, &strip_nocut_pixels},
            };
            UINT64 strip_fence = 10;
            for (const auto& sp : strip_passes) {
                EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, sp.pso)));
                list->ResourceBarrier(1, &to_target);
                list->ClearRenderTargetView(rtv, blue, 0, nullptr);
                list->SetGraphicsRootSignature(root.p);
                list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
                list->IASetVertexBuffers(0, 1, &strip_vbv);
                list->IASetIndexBuffer(&sp.ibv);
                list->RSSetViewports(1, &viewport);
                list->RSSetScissorRects(1, &scissor);
                list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
                list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
                list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
                list->DrawIndexedInstanced(sp.index_count, 1, 0, 0, 0);
                list->ResourceBarrier(1, &to_copy);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                EXPECT(list->Close() == S_OK);
                queue->ExecuteCommandLists(1, lists);
                EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
                ++strip_fence;
                EXPECT(SUCCEEDED(queue->Signal(fence.p, strip_fence)) &&
                       SUCCEEDED(fence->SetEventOnCompletion(strip_fence, nullptr)));
                std::memcpy(sp.out->data(), mapped, sp.out->size());
            }
        }

        // Multisampled graphics pipeline (4x MSAA): draw the top-left red quad into a 4x MSAA target and resolve.
        // Full SampleMask (0xFFFFFFFF) writes all 4 samples red; half SampleMask (0x5) writes 2 of 4 samples so the
        // resolved top-left quadrant shifts from (255, 0, 0, 255) to ~50/50 purple (128, 0, 128, 255).
        {
            // SampleDesc.Count == 0, non-power-of-two/out-of-range counts, and non-zero quality must all be rejected.
            auto bad_desc = pso_desc;
            Ref<ID3D12PipelineState> bad_pso;
            for (const DXGI_SAMPLE_DESC sd : {DXGI_SAMPLE_DESC{0, 0}, DXGI_SAMPLE_DESC{3, 0}, DXGI_SAMPLE_DESC{64, 0},
                                              DXGI_SAMPLE_DESC{4, 1}, DXGI_SAMPLE_DESC{1, 1}}) {
                bad_desc.SampleDesc = sd;
                EXPECT(device->CreateGraphicsPipelineState(&bad_desc, IID_PPV_ARGS(&bad_pso.p)) == E_INVALIDARG &&
                       !bad_pso.p);
            }
            // Any power-of-two sample count not supported by the attachment format must also be rejected.
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS ms32{DXGI_FORMAT_R8G8B8A8_UNORM, 32,
                                                               D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE, 0};
            if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &ms32, sizeof(ms32))) &&
                ms32.NumQualityLevels == 0) {
                bad_desc.SampleDesc = {32, 0};
                EXPECT(device->CreateGraphicsPipelineState(&bad_desc, IID_PPV_ARGS(&bad_pso.p)) == E_INVALIDARG &&
                       !bad_pso.p);
            }

            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS ms_levels{DXGI_FORMAT_R8G8B8A8_UNORM, 4,
                                                                    D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE, 0};
            if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &ms_levels,
                                                      sizeof(ms_levels))) &&
                ms_levels.NumQualityLevels > 0) {
                auto ms_rt_desc = rt_desc;
                ms_rt_desc.SampleDesc.Count = 4;
                Ref<ID3D12Resource> ms_target;
                EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &ms_rt_desc,
                                                                 D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                                 IID_PPV_ARGS(&ms_target.p))));
                D3D12_DESCRIPTOR_HEAP_DESC ms_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1,
                                                        D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
                Ref<ID3D12DescriptorHeap> ms_rtv_heap;
                EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&ms_heap_desc, IID_PPV_ARGS(&ms_rtv_heap.p))));
                const D3D12_CPU_DESCRIPTOR_HANDLE ms_rtv{
                    SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(ms_rtv_heap.p, 9))(ms_rtv_heap.p))};
                device->CreateRenderTargetView(ms_target.p, nullptr, ms_rtv);

                auto ms_pso_desc = pso_desc;
                ms_pso_desc.SampleDesc.Count = 4;
                Ref<ID3D12PipelineState> ms_full_pso, ms_half_pso;
                ms_pso_desc.SampleMask = 0xFFFFFFFFu;
                EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&ms_pso_desc, IID_PPV_ARGS(&ms_full_pso.p))));
                ms_pso_desc.SampleMask = 0x5u;
                EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&ms_pso_desc, IID_PPV_ARGS(&ms_half_pso.p))));

                struct MsaaPass {
                    ID3D12PipelineState* pso;
                    std::vector<std::uint8_t>* out;
                };
                const MsaaPass ms_passes[] = {{ms_full_pso.p, &msaa_full_pixels}, {ms_half_pso.p, &msaa_half_pixels}};
                UINT64 ms_fence = 20;
                for (UINT pass = 0; pass < std::size(ms_passes); ++pass) {
                    EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, ms_passes[pass].pso)));
                    if (pass > 0) {
                        const auto ms_to_rt = transition(ms_target.p, D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                                                         D3D12_RESOURCE_STATE_RENDER_TARGET);
                        list->ResourceBarrier(1, &ms_to_rt);
                    }
                    list->ClearRenderTargetView(ms_rtv, blue, 0, nullptr);
                    list->SetGraphicsRootSignature(root.p);
                    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    list->IASetVertexBuffers(0, 1, &vbv);
                    list->IASetIndexBuffer(&ibv);
                    list->RSSetViewports(1, &viewport);
                    list->RSSetScissorRects(1, &scissor);
                    list->OMSetRenderTargets(1, &ms_rtv, FALSE, nullptr);
                    list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
                    list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
                    list->DrawIndexedInstanced(6, 1, 0, 0, 0);
                    const D3D12_RESOURCE_BARRIER pre_resolve[] = {
                        transition(ms_target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE),
                        transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RESOLVE_DEST),
                    };
                    list->ResourceBarrier(2, pre_resolve);
                    list->ResolveSubresource(target.p, 0, ms_target.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
                    const auto post_resolve =
                        transition(target.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    list->ResourceBarrier(1, &post_resolve);
                    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                    EXPECT(list->Close() == S_OK);
                    queue->ExecuteCommandLists(1, lists);
                    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
                    ++ms_fence;
                    EXPECT(SUCCEEDED(queue->Signal(fence.p, ms_fence)) &&
                           SUCCEEDED(fence->SetEventOnCompletion(ms_fence, nullptr)));
                    std::memcpy(ms_passes[pass].out->data(), mapped, ms_passes[pass].out->size());
                }
                ran_msaa = true;
            }
        }

        // A draw with no viewport (never set, or unbound with a count of 0) rasterizes nothing in D3D12. It is not
        // recorded, so the list closes and executes cleanly (Vulkan forbids a 0x0 viewport).
        EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, pso.p)));
        list->SetGraphicsRootSignature(root.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->DrawInstanced(3, 1, 0, 0);
        list->RSSetViewports(1, &viewport);
        list->RSSetViewports(0, nullptr);
        list->DrawInstanced(3, 1, 0, 0);
        EXPECT(list->Close() == S_OK);
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 30)) && SUCCEEDED(fence->SetEventOnCompletion(30, nullptr)));

        // Each parent draw starts with a 0.25 blend factor, then executes one of four bundles.
        // Empty and reset bundles preserve it. Explicit zero and nullptr replace it with zero and one.
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC blend_desc = fullscreen_desc;
            blend_desc.BlendState.RenderTarget[0] = {
                TRUE, FALSE, D3D12_BLEND_BLEND_FACTOR, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
                D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP,
                D3D12_COLOR_WRITE_ENABLE_ALL};
            Ref<ID3D12PipelineState> blend_pso;
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&blend_desc, IID_PPV_ARGS(&blend_pso.p))));
            Ref<ID3D12CommandAllocator> bundle_allocator;
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE,
                                                           IID_PPV_ARGS(&bundle_allocator.p))));
            constexpr UINT strip_count = 4, strip_width = size / strip_count;
            constexpr std::array<std::uint8_t, strip_count> expected_red = {64, 0, 255, 64};
            std::array<Ref<ID3D12GraphicsCommandList>, strip_count> bundles;
            const float zero_factor[4] = {0, 0, 0, 0};
            const float parent_factor[4] = {0.25f, 0.25f, 0.25f, 0.25f};
            for (UINT i = 0; i < strip_count; ++i) {
                auto& bundle = bundles[i];
                EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, bundle_allocator.p,
                                                           nullptr, IID_PPV_ARGS(&bundle.p))));
                if (i == 1 || i == 3) bundle->OMSetBlendFactor(zero_factor);
                if (i == 2) bundle->OMSetBlendFactor(nullptr);
                EXPECT(bundle->Close() == S_OK);
                if (i == 3) {
                    EXPECT(SUCCEEDED(bundle->Reset(bundle_allocator.p, nullptr)));
                    EXPECT(bundle->Close() == S_OK);
                }
            }

            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, blend_pso.p)));
            list->ResourceBarrier(1, &to_target);
            list->ClearRenderTargetView(rtv, blue, 0, nullptr);
            list->SetGraphicsRootSignature(root.p);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list->RSSetViewports(1, &viewport);
            list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
            for (UINT i = 0; i < strip_count; ++i) {
                const D3D12_RECT strip{LONG(i * strip_width), 0, LONG((i + 1) * strip_width), LONG(size)};
                list->RSSetScissorRects(1, &strip);
                list->OMSetBlendFactor(parent_factor);
                list->ExecuteBundle(bundles[i].p);
                list->DrawInstanced(3, 1, 0, 0);
            }
            list->ResourceBarrier(1, &to_copy);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            // Above line 580's 30: a lower value is already reached, so the wait would return at once.
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 31)) && SUCCEEDED(fence->SetEventOnCompletion(31, nullptr)));
            const auto* result = static_cast<const std::uint8_t*>(mapped);
            for (UINT row = 0; row < size; ++row) {
                for (UINT col = 0; col < size; ++col) {
                    const auto* got = result + row * 256 + col * 4;
                    const auto red = expected_red[col / strip_width];
                    if (got[0] == red && got[1] == 0 && got[2] == 0 && got[3] == 255) continue;
                    std::fprintf(stderr, "bundle blend pixel (%u, %u): got %u,%u,%u,%u, expected red %u\n",
                                 col, row, got[0], got[1], got[2], got[3], red);
                    return 1;
                }
            }
        }

        // B8G8R8X8 targets have no alpha, so the blend factors that read destination alpha read 1 there, as WARP and
        // AMD's D3D12 driver blend; B8G8R8A8 with the same cleared bytes reads its stored alpha. Only red, green and
        // blue are compared: what a draw leaves in the X byte differs between them (WARP writes 0, AMD the blended
        // alpha), and no view reads it.
        {
            const float x8_tint[3] = {0.8f, 0.6f, 0.4f};  // with the full-screen VS's yellow: (0.8, 0.6, 0, 1)
            void* x8_up = nullptr;
            EXPECT(SUCCEEDED(upload->Map(0, nullptr, &x8_up)));
            std::memcpy(static_cast<std::uint8_t*>(x8_up) + 2048, x8_tint, sizeof(x8_tint));
            upload->Unmap(0, nullptr);
            const FLOAT x8_clear[4] = {0.2f, 0.4f, 0.6f, 0.25f};
            struct BlendCase {
                const char* name;
                D3D12_BLEND src, dst;
            };
            constexpr BlendCase x8_cases[] = {
                {"src*DEST_ALPHA", D3D12_BLEND_DEST_ALPHA, D3D12_BLEND_ZERO},
                {"src*INV_DEST_ALPHA", D3D12_BLEND_INV_DEST_ALPHA, D3D12_BLEND_ZERO},
                {"src*SRC_ALPHA_SAT", D3D12_BLEND_SRC_ALPHA_SAT, D3D12_BLEND_ZERO},
                {"dst*DEST_ALPHA", D3D12_BLEND_ZERO, D3D12_BLEND_DEST_ALPHA},
                {"dst*INV_DEST_ALPHA", D3D12_BLEND_ZERO, D3D12_BLEND_INV_DEST_ALPHA},
            };
            const D3D12_DESCRIPTOR_HEAP_DESC x8_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1,
                                                          D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            Ref<ID3D12DescriptorHeap> x8_heap;
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&x8_heap_desc, IID_PPV_ARGS(&x8_heap.p))));
            const D3D12_CPU_DESCRIPTOR_HANDLE x8_rtv{
                SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(x8_heap.p, 9))(x8_heap.p))};
            const D3D12_VIEWPORT x8_viewport{0, 0, 4, 4, 0, 1};
            const D3D12_RECT x8_scissor{0, 0, 4, 4};
            UINT64 x8_fence = 40;
            for (const DXGI_FORMAT format : {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8X8_UNORM}) {
                D3D12_RESOURCE_DESC x8_desc = rt_desc;
                x8_desc.Width = x8_desc.Height = 4;
                x8_desc.Format = format;
                Ref<ID3D12Resource> x8_target;
                EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &x8_desc,
                                                                 D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                                 IID_PPV_ARGS(&x8_target.p))));
                device->CreateRenderTargetView(x8_target.p, nullptr, x8_rtv);
                // The clear stores alpha 64 in both formats.
                const float dest_alpha = format == DXGI_FORMAT_B8G8R8X8_UNORM ? 1.0f : 64 / 255.0f;
                for (const BlendCase& c : x8_cases) {
                    D3D12_GRAPHICS_PIPELINE_STATE_DESC x8_pso_desc = fullscreen_desc;
                    x8_pso_desc.RTVFormats[0] = format;
                    x8_pso_desc.BlendState.RenderTarget[0] = {
                        TRUE, FALSE, c.src, c.dst, D3D12_BLEND_OP_ADD, D3D12_BLEND_ONE, D3D12_BLEND_ZERO,
                        D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
                    Ref<ID3D12PipelineState> x8_pso;
                    EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&x8_pso_desc, IID_PPV_ARGS(&x8_pso.p))));
                    EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, x8_pso.p)));
                    list->ClearRenderTargetView(x8_rtv, x8_clear, 0, nullptr);
                    list->SetGraphicsRootSignature(root.p);
                    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    list->RSSetViewports(1, &x8_viewport);
                    list->RSSetScissorRects(1, &x8_scissor);
                    list->OMSetRenderTargets(1, &x8_rtv, FALSE, nullptr);
                    list->SetGraphicsRootConstantBufferView(1, upload_va + 2048);
                    list->DrawInstanced(3, 1, 0, 0);
                    const auto x8_to_copy = transition(x8_target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
                    list->ResourceBarrier(1, &x8_to_copy);
                    D3D12_TEXTURE_COPY_LOCATION x8_dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
                    x8_dst.PlacedFootprint = {0, {format, 4, 4, 1, 256}};
                    D3D12_TEXTURE_COPY_LOCATION x8_src{x8_target.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
                    list->CopyTextureRegion(&x8_dst, 0, 0, 0, &x8_src, nullptr);
                    const auto x8_to_target = transition(x8_target.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                         D3D12_RESOURCE_STATE_RENDER_TARGET);
                    list->ResourceBarrier(1, &x8_to_target);
                    EXPECT(list->Close() == S_OK);
                    queue->ExecuteCommandLists(1, lists);
                    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
                    ++x8_fence;
                    EXPECT(SUCCEEDED(queue->Signal(fence.p, x8_fence)) &&
                           SUCCEEDED(fence->SetEventOnCompletion(x8_fence, nullptr)));
                    // Source (0.8, 0.6, 0) with alpha 1, destination the cleared bytes; SRC_ALPHA_SAT is
                    // min(1, 1 - dest alpha).
                    const float source[3] = {0.8f, 0.6f, 0.0f}, dest[3] = {51 / 255.0f, 102 / 255.0f, 153 / 255.0f};
                    const auto factor = [&](D3D12_BLEND b) {
                        return b == D3D12_BLEND_DEST_ALPHA ? dest_alpha : b == D3D12_BLEND_ZERO ? 0.0f : 1 - dest_alpha;
                    };
                    const auto* got = static_cast<const std::uint8_t*>(mapped);
                    for (UINT ch = 0; ch < 3; ++ch) {  // R, G, B, at bytes 2, 1, 0
                        const float value = source[ch] * factor(c.src) + dest[ch] * factor(c.dst);
                        const int want = int(std::min(1.0f, value) * 255 + 0.5f);
                        if (std::abs(int(got[2 - ch]) - want) <= 1) continue;
                        std::fprintf(stderr, "%s target, %s: got B,G,R %u,%u,%u, channel %u wants %d\n",
                                     format == DXGI_FORMAT_B8G8R8X8_UNORM ? "B8G8R8X8" : "B8G8R8A8", c.name, got[0],
                                     got[1], got[2], ch, want);
                        return 1;
                    }
                }
            }
            std::printf("B8G8R8X8: %u blends that read destination alpha read 1, B8G8R8A8 its stored alpha\n",
                        unsigned(std::size(x8_cases)));
        }

        // Instance step rate > 1 (VK_KHR/EXT_vertex_attribute_divisor): two per-instance colours in slot 1:
        // entry 0 = red, entry 1 = magenta (both preserved by the {1,0,1} tint). With step rate 2:
        // - Draw 1 (top-left, 2 instances from 0): instances 0 and 1 both read entry 0/2 = 0 (red).
        // - Draw 2 (bottom-right, 3 instances from 0): instance 2 reads entry 2/2 = 1 (magenta) on top.
        // Negative control (step rate 1, same 2-instance draw at top-left): instance 1 reads entry 1 (magenta) over
        // instance 0, flipping all 1,024 top-left pixels from red to magenta.
        if (has_divisor) {
            const std::uint8_t inst_colors[2][4] = {{255, 0, 0, 255}, {255, 0, 255, 255}};
            void* up_ptr = nullptr;
            EXPECT(SUCCEEDED(upload->Map(0, nullptr, &up_ptr)));
            std::memcpy(static_cast<std::uint8_t*>(up_ptr) + 1024, inst_colors, sizeof(inst_colors));
            upload->Unmap(0, nullptr);

            const D3D12_INPUT_ELEMENT_DESC step2_elements[2] = {
                {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 2},
            };
            const D3D12_INPUT_ELEMENT_DESC step1_elements[2] = {
                {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
            };
            auto step_desc = pso_desc;
            Ref<ID3D12PipelineState> step2_pso, step1_pso;
            step_desc.InputLayout = {step2_elements, 2};
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&step_desc, IID_PPV_ARGS(&step2_pso.p))));
            step_desc.InputLayout = {step1_elements, 2};
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&step_desc, IID_PPV_ARGS(&step1_pso.p))));

            const D3D12_VERTEX_BUFFER_VIEW step_vbvs[2] = {vbv, {upload_va + 1024, sizeof(inst_colors), 4}};
            const float br_offset[2] = {1.0f, -1.0f};
            UINT64 step_fence = 60;
            for (UINT pass = 0; pass < 2; ++pass) {
                EXPECT(SUCCEEDED(allocator->Reset()) &&
                       SUCCEEDED(list->Reset(allocator.p, pass == 0 ? step2_pso.p : step1_pso.p)));
                list->ResourceBarrier(1, &to_target);
                list->ClearRenderTargetView(rtv, blue, 0, nullptr);
                list->SetGraphicsRootSignature(root.p);
                list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                list->IASetVertexBuffers(0, 2, step_vbvs);
                list->IASetIndexBuffer(&ibv);
                list->RSSetViewports(1, &viewport);
                list->RSSetScissorRects(1, &scissor);
                list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
                list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
                list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
                list->DrawIndexedInstanced(6, 2, 0, 0, 0);
                if (pass == 0) {
                    list->SetGraphicsRoot32BitConstants(0, 2, br_offset, 0);
                    list->DrawIndexedInstanced(6, 3, 0, 0, 0);
                }
                list->ResourceBarrier(1, &to_copy);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                EXPECT(list->Close() == S_OK);
                queue->ExecuteCommandLists(1, lists);
                EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
                ++step_fence;
                EXPECT(SUCCEEDED(queue->Signal(fence.p, step_fence)) &&
                       SUCCEEDED(fence->SetEventOnCompletion(step_fence, nullptr)));
                auto& dst_buf = pass == 0 ? step2_pixels : step1_pixels;
                std::memcpy(dst_buf.data(), mapped, dst_buf.size());
            }
        }

        // An opaque draw through an RTV in a format the texture's castable list names, in the other channel order or
        // across B8G8R8X8 and B8G8R8A8, is stored in the view's channel order and encoding, as WARP and AMD's D3D12
        // driver store it. The X byte after a draw through an X8 view differs between them (WARP 0, AMD the alpha),
        // so it isn't compared.
        {
            const float cast_tint[3] = {0.8f, 0.6f, 0.4f};  // with the full-screen VS's yellow: (0.8, 0.6, 0, 1)
            void* cast_up = nullptr;
            EXPECT(SUCCEEDED(upload->Map(0, nullptr, &cast_up)));
            std::memcpy(static_cast<std::uint8_t*>(cast_up) + 2048, cast_tint, sizeof(cast_tint));
            upload->Unmap(0, nullptr);
            Ref<ID3D12Device10> inner10;
            EXPECT(SUCCEEDED(inner->QueryInterface(IID_PPV_ARGS(&inner10.p))));
            // The bytes of (0.8, 0.6, 0, 1) in the view's order; sRGB encodes 0.8 as 231 and 0.6 as 203.
            struct Cast { DXGI_FORMAT resource, view; const char* name; std::uint8_t bytes[4]; };
            const Cast casts[] = {
                {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, "R8G8B8A8{B8G8R8A8}", {0, 153, 204, 255}},
                {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, "B8G8R8A8{R8G8B8A8}", {204, 153, 0, 255}},
                {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, "R8G8B8A8{B8G8R8A8_SRGB}",
                 {0, 203, 231, 255}},
                {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, "B8G8R8A8{R8G8B8A8_SRGB}",
                 {231, 203, 0, 255}},
                {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, "B8G8R8A8{B8G8R8A8_SRGB}",
                 {0, 203, 231, 255}},
                {DXGI_FORMAT_B8G8R8X8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, "B8G8R8X8{B8G8R8A8}", {0, 153, 204, 255}},
                {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8X8_UNORM, "B8G8R8A8{B8G8R8X8}", {0, 153, 204, 0}},
                {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8X8_UNORM, "R8G8B8A8{B8G8R8X8}", {0, 153, 204, 0}},
            };
            const D3D12_DESCRIPTOR_HEAP_DESC cast_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1,
                                                            D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            Ref<ID3D12DescriptorHeap> cast_heap;
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&cast_heap_desc, IID_PPV_ARGS(&cast_heap.p))));
            const D3D12_CPU_DESCRIPTOR_HANDLE cast_rtv{
                SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(cast_heap.p, 9))(cast_heap.p))};
            const D3D12_VIEWPORT cast_viewport{0, 0, 4, 4, 0, 1};
            const D3D12_RECT cast_scissor{0, 0, 4, 4};
            UINT64 cast_fence = 70;  // after the step-rate block's 61 and 62
            for (const Cast& c : casts) {
                D3D12_RESOURCE_DESC1 cast_desc{};
                cast_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                cast_desc.Width = cast_desc.Height = 4;
                cast_desc.DepthOrArraySize = 1;
                cast_desc.MipLevels = 1;
                cast_desc.Format = c.resource;
                cast_desc.SampleDesc.Count = 1;
                cast_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
                Ref<ID3D12Resource> cast_target;
                EXPECT(SUCCEEDED(inner10->CreateCommittedResource3(&default_heap, D3D12_HEAP_FLAG_NONE, &cast_desc,
                    D3D12_BARRIER_LAYOUT_RENDER_TARGET, nullptr, nullptr, 1, &c.view, IID_PPV_ARGS(&cast_target.p))));
                D3D12_RENDER_TARGET_VIEW_DESC cast_view{};
                cast_view.Format = c.view;
                cast_view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                device->CreateRenderTargetView(cast_target.p, &cast_view, cast_rtv);
                D3D12_GRAPHICS_PIPELINE_STATE_DESC cast_pso_desc = fullscreen_desc;
                cast_pso_desc.RTVFormats[0] = c.view;
                Ref<ID3D12PipelineState> cast_pso;
                EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&cast_pso_desc, IID_PPV_ARGS(&cast_pso.p))));
                EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, cast_pso.p)));
                list->SetGraphicsRootSignature(root.p);
                list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                list->RSSetViewports(1, &cast_viewport);
                list->RSSetScissorRects(1, &cast_scissor);
                list->OMSetRenderTargets(1, &cast_rtv, FALSE, nullptr);
                list->SetGraphicsRootConstantBufferView(1, upload_va + 2048);
                list->DrawInstanced(3, 1, 0, 0);
                const auto cast_to_copy = transition(cast_target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                     D3D12_RESOURCE_STATE_COPY_SOURCE);
                list->ResourceBarrier(1, &cast_to_copy);
                D3D12_TEXTURE_COPY_LOCATION cast_dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
                cast_dst.PlacedFootprint = {0, {c.resource, 4, 4, 1, 256}};
                D3D12_TEXTURE_COPY_LOCATION cast_src{cast_target.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
                list->CopyTextureRegion(&cast_dst, 0, 0, 0, &cast_src, nullptr);
                EXPECT(list->Close() == S_OK);
                queue->ExecuteCommandLists(1, lists);
                EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
                ++cast_fence;
                EXPECT(SUCCEEDED(queue->Signal(fence.p, cast_fence)) &&
                       SUCCEEDED(fence->SetEventOnCompletion(cast_fence, nullptr)));
                const auto* got = static_cast<const std::uint8_t*>(mapped);
                const unsigned compared = c.view == DXGI_FORMAT_B8G8R8X8_UNORM ? 3 : 4;
                for (unsigned b = 0; b < compared; ++b) {
                    if (std::abs(int(got[b]) - int(c.bytes[b])) <= 1) continue;
                    std::fprintf(stderr, "draw through %s: got bytes %u,%u,%u,%u, byte %u wants %u\n", c.name, got[0],
                                 got[1], got[2], got[3], b, unsigned(c.bytes[b]));
                    return 1;
                }
            }
            std::printf("castable RTVs: %u draws stored in the view's channel order and encoding\n",
                        unsigned(std::size(casts)));
        }

        // A draw without a topology is invalid; Close reports it.
        EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, pso.p)));
        list->SetGraphicsRootSignature(root.p);
        list->DrawInstanced(3, 1, 0, 0);
        EXPECT(list->Close() == E_INVALIDARG);
        // So is binding more than eight render targets.
        EXPECT(SUCCEEDED(list->Reset(allocator.p, pso.p)));
        list->OMSetRenderTargets(9, &rtv, TRUE, nullptr);
        EXPECT(list->Close() == E_INVALIDARG);

        for (auto* r : {&vertex_buffer, &upload, &readback, &target, &uint_target}) r->reset();
        for (auto* p : {&mrt, &instanced, &refused, &unfed, &xor_pso, &fullscreen}) p->reset();
        pso.reset();
        root.reset();
        rtv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    unsigned mismatches = 0;
    for (UINT row = 0; row < size; ++row) {
        for (UINT col = 0; col < size; ++col) {
            const bool top = row < size / 2, left = col < size / 2;
            // In the top-right quadrant: draw 4's red corner (top-left of it) and draw 5's black one (bottom-right).
            const bool corner4 = row < size / 4 && col >= size / 2 && col < size * 3 / 4;
            const bool corner5 = top && row >= size / 4 && col >= size * 3 / 4;
            using Pixel = std::array<std::uint8_t, 4>;
            const Pixel want = (top && left) || corner4 ? Pixel{255, 0, 0, 255}
                               : corner5                ? Pixel{0, 0, 0, 255}
                               : !top && !left          ? Pixel{0, 255, 0, 255}
                                                        : Pixel{0, 0, 255, 255};
            const std::uint8_t* got = pixels.data() + row * 256 + col * 4;
            if (std::equal(want.begin(), want.end(), got)) continue;
            if (mismatches++ < 6)
                std::fprintf(stderr, "pixel (%u, %u): got %u,%u,%u,%u want %u,%u,%u,%u\n", col, row, got[0], got[1],
                             got[2], got[3], want[0], want[1], want[2], want[3]);
        }
    }
    EXPECT(!mismatches);
    // Second pass: full-screen triangle bottom-right, blue elsewhere.
    for (UINT row = 0; row < size; ++row) {
        for (UINT col = 0; col < size; ++col) {
            const bool top = row < size / 2, left = col < size / 2;
            using Pixel = std::array<std::uint8_t, 4>;
            const Pixel want = !top && !left ? Pixel{255, 0, 0, 255} : Pixel{0, 0, 255, 255};
            const std::uint8_t* got = second_pixels.data() + row * 256 + col * 4;
            if (std::equal(want.begin(), want.end(), got)) continue;
            if (mismatches++ < 6)
                std::fprintf(stderr, "second pass pixel (%u, %u): got %u,%u,%u,%u want %u,%u,%u,%u\n", col, row,
                             got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
        }
    }
    EXPECT(!mismatches);
    // Logic-op pass (UINT target): the integer XOR of source (255,0,0,255) and destination (0,0,255,255) top-left.
    if (logic_op) {
        for (UINT row = 0; row < size; ++row) {
            for (UINT col = 0; col < size; ++col) {
                const bool top = row < size / 2, left = col < size / 2;
                using Pixel = std::array<std::uint8_t, 4>;
                const Pixel dest{0, 0, 255, 255}, source{255, 0, 0, 255};
                Pixel want = dest;
                if (top && left)
                    for (std::size_t ch = 0; ch < 4; ++ch) want[ch] = std::uint8_t(source[ch] ^ dest[ch]);
                const std::uint8_t* got = logic_pixels.data() + row * 256 + col * 4;
                if (std::equal(want.begin(), want.end(), got)) continue;
                if (mismatches++ < 6)
                    std::fprintf(stderr, "logic-op pixel (%u, %u): got %u,%u,%u,%u want %u,%u,%u,%u\n", col, row,
                                 got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
            }
        }
        EXPECT(!mismatches);
    }
    // Strip-cut passes: 0xFFFF (UINT16) and 0xFFFFFFFF (UINT32) must produce only the top-left and bottom-right quads
    // (pure blue in top-right and bottom-left). Negative control without cut must paint bridge pixels in bottom-left.
    UINT nocut_bl_red = 0;
    for (UINT row = 0; row < size; ++row) {
        for (UINT col = 0; col < size; ++col) {
            const bool top = row < size / 2, left = col < size / 2;
            using Pixel = std::array<std::uint8_t, 4>;
            const Pixel want = (top && left) || (!top && !left) ? Pixel{255, 0, 0, 255} : Pixel{0, 0, 255, 255};
            for (const auto& buf : {strip16_pixels, strip32_pixels}) {
                const std::uint8_t* got = buf.data() + row * 256 + col * 4;
                if (!std::equal(want.begin(), want.end(), got) && mismatches++ < 6)
                    std::fprintf(stderr, "strip-cut pixel (%u, %u): got %u,%u,%u,%u want %u,%u,%u,%u\n", col, row,
                                 got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
            }
            if (!top && left) {
                const std::uint8_t* got = strip_nocut_pixels.data() + row * 256 + col * 4;
                if (got[0] == 255 && got[2] == 0) ++nocut_bl_red;
            }
        }
    }
    EXPECT(!mismatches);
    EXPECT(nocut_bl_red > 0);
    std::printf("strip cut (0xFFFF & 0xFFFFFFFF): 4096/4096 pixels matched each (no-cut negative control painted %u "
                "bridge pixels in bottom-left)\n",
                nocut_bl_red);
    if (ran_msaa) {
        UINT half_changed = 0;
        for (UINT row = 0; row < size; ++row) {
            for (UINT col = 0; col < size; ++col) {
                const bool top_left = row < size / 2 && col < size / 2;
                const std::uint8_t* got_full = msaa_full_pixels.data() + row * 256 + col * 4;
                const std::uint8_t* got_half = msaa_half_pixels.data() + row * 256 + col * 4;
                const std::uint8_t want_full[4] = {std::uint8_t(top_left ? 255 : 0), 0,
                                                   std::uint8_t(top_left ? 0 : 255), 255};
                if (std::memcmp(got_full, want_full, 4) != 0 && mismatches++ < 6)
                    std::fprintf(stderr, "4x MSAA full mask (%u,%u): got %u,%u,%u,%u want %u,%u,%u,%u\n", col, row,
                                 got_full[0], got_full[1], got_full[2], got_full[3], want_full[0], want_full[1],
                                 want_full[2], want_full[3]);
                if (top_left) {
                    const bool ok = std::abs(int(got_half[0]) - 128) <= 1 && got_half[1] == 0 &&
                                    std::abs(int(got_half[2]) - 128) <= 1 && got_half[3] == 255;
                    if (!ok && mismatches++ < 6)
                        std::fprintf(stderr, "4x MSAA half mask (%u,%u): got %u,%u,%u,%u want ~128,0,128,255\n", col,
                                     row, got_half[0], got_half[1], got_half[2], got_half[3]);
                    if (std::memcmp(got_half, got_full, 4) != 0) ++half_changed;
                } else if (std::memcmp(got_half, want_full, 4) != 0 && mismatches++ < 6) {
                    std::fprintf(stderr, "4x MSAA half mask bg (%u,%u): got %u,%u,%u,%u want 0,0,255,255\n", col, row,
                                 got_half[0], got_half[1], got_half[2], got_half[3]);
                }
            }
        }
        EXPECT(!mismatches);
        EXPECT(half_changed == (size / 2) * (size / 2));
        std::printf("4x MSAA graphics pipeline: 4096/4096 resolved pixels matched (SampleMask 0x5 changed %u/%u "
                    "top-left pixels to purple)\n",
                    half_changed, (size / 2) * (size / 2));
    }
    if (has_divisor) {
        UINT step1_flipped = 0;
        for (UINT row = 0; row < size; ++row) {
            for (UINT col = 0; col < size; ++col) {
                const bool top = row < size / 2, left = col < size / 2;
                using Pixel = std::array<std::uint8_t, 4>;
                const Pixel want2 = top && left     ? Pixel{255, 0, 0, 255}
                                    : !top && !left ? Pixel{255, 0, 255, 255}
                                                    : Pixel{0, 0, 255, 255};
                const std::uint8_t* got2 = step2_pixels.data() + row * 256 + col * 4;
                if (!std::equal(want2.begin(), want2.end(), got2) && mismatches++ < 6)
                    std::fprintf(stderr, "step-rate 2 pixel (%u,%u): got %u,%u,%u,%u want %u,%u,%u,%u\n", col, row,
                                 got2[0], got2[1], got2[2], got2[3], want2[0], want2[1], want2[2], want2[3]);
                if (top && left) {
                    const std::uint8_t* got1 = step1_pixels.data() + row * 256 + col * 4;
                    const Pixel want1{255, 0, 255, 255};
                    if (!std::equal(want1.begin(), want1.end(), got1) && mismatches++ < 6)
                        std::fprintf(stderr, "step-rate 1 control (%u,%u): got %u,%u,%u,%u want 255,0,255,255\n", col,
                                     row, got1[0], got1[1], got1[2], got1[3]);
                    if (std::memcmp(got1, got2, 4) != 0) ++step1_flipped;
                }
            }
        }
        EXPECT(!mismatches);
        EXPECT(step1_flipped == (size / 2) * (size / 2));
        std::printf("instance step rate 2: 4096/4096 pixels matched (step-rate 1 control flipped %u/%u top-left "
                    "pixels from red to magenta)\n",
                    step1_flipped, (size / 2) * (size / 2));
    }
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("M1: five draws and a culled triangle matched all %u pixels: input layout by semantic, root "
                "constants and a root CBV per draw, vertex buffer in the arena, index buffer in an upload heap, an "
                "unbound target and DSV, a rebound root signature, step rate 0 and an unbound vertex slot; then %s "
                "and a full-screen pass without vertex input\n",
                size * size, logic_op ? "an XOR logic op" : "(no logicOp on this GPU)");
    return 0;
}
