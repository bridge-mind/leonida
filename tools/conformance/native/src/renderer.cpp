#include "renderer.h"

#include <d3dcommon.h>

#include <cstdio>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "diagnostics.h"

namespace {

constexpr UINT kTintConstantStride = 256;  // D3D12 requires 256-byte alignment for a root CBV
// D3D12_TEXTURE_DATA_PITCH_ALIGNMENT is 256 bytes: the readback buffer's rows are padded
// to that boundary, which happens to equal width*4 at 256 pixels wide. Deriving it keeps
// the copy loop honest if the scene size ever changes.
constexpr UINT kReadbackRowPitch =
    (conform::kSceneWidth * 4 + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) /
    D3D12_TEXTURE_DATA_PITCH_ALIGNMENT * D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
constexpr UINT kReadbackRowBytes = conform::kSceneWidth * 4;
constexpr SIZE_T kUploadAlignment = 256;
constexpr UINT kFullscreenTriangleVertices = 3;

// The index buffers are declared uint16_t in the contract and bound as R16_UINT. If the
// contract ever switches to 32-bit indices these fail at compile time instead of every
// index being read as two.
static_assert(std::is_same_v<std::remove_cv_t<std::remove_reference_t<
                  decltype(conform::kDepthOrderIndices[0])>>, uint16_t>);
static_assert(std::is_same_v<std::remove_cv_t<std::remove_reference_t<
                  decltype(conform::kIndexedIndices[0])>>, uint16_t>);

// A .cso held in ordinary memory. D3D12_SHADER_BYTECODE copies nothing, so the vector
// must stay alive until the pipeline state is created.
struct ShaderFile {
    std::vector<uint8_t> bytes;

    D3D12_SHADER_BYTECODE Bytecode() const {
        return D3D12_SHADER_BYTECODE{bytes.data(), bytes.size()};
    }
};

// The input layout for conform::ClipVertex. The offsets mirror scenes.h: three floats,
// then four UNORM bytes, so the colour arrives in the shader as straight 0..1 RGBA.
// The asserts make a future edit to the struct a build failure instead of a scene that
// renders the wrong shade.
static_assert(sizeof(conform::ClipVertex) == 16,
              "ClipVertex must stay 3 floats + 4 bytes; the input layout assumes 16");
static_assert(offsetof(conform::ClipVertex, r) == 12,
              "COLOR must start at byte 12; the input layout hard-codes that offset");

const D3D12_INPUT_ELEMENT_DESC kVertexColorInputLayout[] = {
    {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
     0},
    {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
};

SIZE_T AlignToUploadBoundary(SIZE_T size) {
    return (size + (kUploadAlignment - 1)) & ~(kUploadAlignment - 1);
}

// Creates one committed resource and names it for the debug layer. Reports its own
// failure because every resource in this file is created the same way.
bool CreateCommittedResourceAt(ID3D12Device& device, D3D12_HEAP_TYPE heap_type,
                               D3D12_RESOURCE_STATES initial_state,
                               const D3D12_RESOURCE_DESC& description, const wchar_t* debug_name,
                               ComPtr<ID3D12Resource>& resource_out) {
    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = heap_type;
    heap_properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_properties.CreationNodeMask = 0;
    heap_properties.VisibleNodeMask = 0;

    const HRESULT result = device.CreateCommittedResource(
        &heap_properties, D3D12_HEAP_FLAG_NONE, &description, initial_state, nullptr,
        IID_PPV_ARGS(resource_out.GetAddressOf()));
    if (!HrSucceeded(result, __FILE__, __LINE__, "ID3D12Device::CreateCommittedResource",
                     "GPU memory is available and the heap type matches the resource; "
                     "OUT_OF_MEMORY means the adapter is full or the limit is set too low")) {
        return false;
    }
    resource_out->SetName(debug_name);
    return true;
}

// Upload-heap buffer holding a CPU-side copy of geometry or constants. The size is
// rounded up because D3D12 wants buffer widths aligned and the slack costs nothing.
bool CreateUploadBuffer(ID3D12Device& device, const void* data, SIZE_T data_size,
                        const wchar_t* debug_name, ComPtr<ID3D12Resource>& resource_out) {
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = AlignToUploadBoundary(data_size);
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = DXGI_FORMAT_UNKNOWN;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    description.Flags = D3D12_RESOURCE_FLAG_NONE;

    if (!CreateCommittedResourceAt(device, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ,
                                   description, debug_name, resource_out)) {
        return false;
    }

    void* mapped = nullptr;
    if (!HrSucceeded(resource_out->Map(0, nullptr, &mapped), __FILE__, __LINE__,
                     "ID3D12Resource::Map(upload)", "the upload heap is not exhausted")) {
        return false;
    }
    std::memcpy(mapped, data, data_size);
    resource_out->Unmap(0, nullptr);
    return true;
}

D3D12_RESOURCE_DESC MakeTextureDescription(DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    description.Width = static_cast<UINT64>(conform::kSceneWidth);
    description.Height = static_cast<UINT>(conform::kSceneHeight);
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = format;
    description.SampleDesc.Count = 1;
    description.SampleDesc.Quality = 0;
    description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    description.Flags = flags;
    return description;
}

bool CreateDescriptorHeaps(Renderer& renderer, ID3D12Device& device) {
    D3D12_DESCRIPTOR_HEAP_DESC render_target_heap{};
    render_target_heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    render_target_heap.NumDescriptors = 1;
    if (!HrSucceeded(device.CreateDescriptorHeap(&render_target_heap, IID_PPV_ARGS(
                         renderer.render_target_view_heap.GetAddressOf())),
                     __FILE__, __LINE__, "ID3D12Device::CreateDescriptorHeap(RTV)",
                     "the RTV heap limit is not exhausted and the heap type is supported")) {
        return false;
    }
    renderer.color_render_target_view =
        renderer.render_target_view_heap->GetCPUDescriptorHandleForHeapStart();

    D3D12_DESCRIPTOR_HEAP_DESC depth_stencil_heap{};
    depth_stencil_heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    depth_stencil_heap.NumDescriptors = 1;
    if (!HrSucceeded(device.CreateDescriptorHeap(&depth_stencil_heap, IID_PPV_ARGS(
                         renderer.depth_stencil_view_heap.GetAddressOf())),
                     __FILE__, __LINE__, "ID3D12Device::CreateDescriptorHeap(DSV)",
                     "the DSV heap limit is not exhausted and the heap type is supported")) {
        return false;
    }
    renderer.depth_stencil_view =
        renderer.depth_stencil_view_heap->GetCPUDescriptorHandleForHeapStart();
    return true;
}

bool CreateTargets(Renderer& renderer, ID3D12Device& device) {
    if (!CreateCommittedResourceAt(device, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                                   MakeTextureDescription(DXGI_FORMAT_R8G8B8A8_UNORM,
                                                          D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                                   L"color_target", renderer.color_target)) {
        return false;
    }
    if (!CreateCommittedResourceAt(device, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                   MakeTextureDescription(DXGI_FORMAT_D32_FLOAT,
                                                          D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                                   L"depth_target", renderer.depth_target)) {
        return false;
    }

    D3D12_RESOURCE_DESC readback_description{};
    readback_description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    readback_description.Width = static_cast<UINT64>(kReadbackRowPitch) * conform::kSceneHeight;
    readback_description.Height = 1;
    readback_description.DepthOrArraySize = 1;
    readback_description.MipLevels = 1;
    readback_description.Format = DXGI_FORMAT_UNKNOWN;
    readback_description.SampleDesc.Count = 1;
    readback_description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (!CreateCommittedResourceAt(device, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                                   readback_description, L"readback_buffer",
                                   renderer.readback_buffer)) {
        return false;
    }

    // Null descriptions inherit the resource format, which is what the pipelines expect.
    device.CreateRenderTargetView(renderer.color_target.Get(), nullptr,
                                  renderer.color_render_target_view);
    device.CreateDepthStencilView(renderer.depth_target.Get(), nullptr,
                                  renderer.depth_stencil_view);
    return true;
}

bool CreateGeometryBuffers(Renderer& renderer, ID3D12Device& device) {
    if (!CreateUploadBuffer(device, conform::kDepthOrderVertices,
                            sizeof(conform::kDepthOrderVertices), L"depth_order_vertex_buffer",
                            renderer.depth_order_vertex_buffer)) {
        return false;
    }
    if (!CreateUploadBuffer(device, conform::kDepthOrderIndices,
                            sizeof(conform::kDepthOrderIndices), L"depth_order_index_buffer",
                            renderer.depth_order_index_buffer)) {
        return false;
    }
    if (!CreateUploadBuffer(device, conform::kIndexedVertices, sizeof(conform::kIndexedVertices),
                            L"indexed_vertex_buffer", renderer.indexed_vertex_buffer)) {
        return false;
    }
    if (!CreateUploadBuffer(device, conform::kIndexedIndices, sizeof(conform::kIndexedIndices),
                            L"indexed_index_buffer", renderer.indexed_index_buffer)) {
        return false;
    }

    renderer.depth_order_vertex_view.BufferLocation =
        renderer.depth_order_vertex_buffer->GetGPUVirtualAddress();
    renderer.depth_order_vertex_view.SizeInBytes =
        static_cast<UINT>(sizeof(conform::kDepthOrderVertices));
    renderer.depth_order_vertex_view.StrideInBytes =
        static_cast<UINT>(sizeof(conform::ClipVertex));

    renderer.depth_order_index_view.BufferLocation =
        renderer.depth_order_index_buffer->GetGPUVirtualAddress();
    renderer.depth_order_index_view.SizeInBytes =
        static_cast<UINT>(sizeof(conform::kDepthOrderIndices));
    renderer.depth_order_index_view.Format = DXGI_FORMAT_R16_UINT;

    renderer.indexed_vertex_view.BufferLocation =
        renderer.indexed_vertex_buffer->GetGPUVirtualAddress();
    renderer.indexed_vertex_view.SizeInBytes = static_cast<UINT>(sizeof(conform::kIndexedVertices));
    renderer.indexed_vertex_view.StrideInBytes = static_cast<UINT>(sizeof(conform::ClipVertex));

    renderer.indexed_index_view.BufferLocation =
        renderer.indexed_index_buffer->GetGPUVirtualAddress();
    renderer.indexed_index_view.SizeInBytes = static_cast<UINT>(sizeof(conform::kIndexedIndices));
    renderer.indexed_index_view.Format = DXGI_FORMAT_R16_UINT;
    return true;
}

bool CreateTintConstants(Renderer& renderer, ID3D12Device& device) {
    // One 256-byte slot per scene. Each slot holds exactly the single float4 the contract
    // specifies at byte offset 0, already divided by 255 so the UNORM bytes come back
    // exact. Slots are filled before any draw and never rewritten.
    std::vector<uint8_t> slot_bytes(
        static_cast<size_t>(conform::kSceneCount) * kTintConstantStride, 0);
    for (int index = 0; index < conform::kSceneCount; ++index) {
        const conform::Scene& scene = conform::kScenes[index];
        const float tint[4] = {scene.tintR / 255.0f, scene.tintG / 255.0f,
                               scene.tintB / 255.0f, scene.tintA / 255.0f};
        std::memcpy(slot_bytes.data() + static_cast<size_t>(index) * kTintConstantStride, tint,
                    sizeof tint);
    }
    return CreateUploadBuffer(device, slot_bytes.data(), slot_bytes.size(), L"tint_constants",
                              renderer.tint_constants);
}

// Address of one scene's float4. Callers validate the slot; this only converts.
D3D12_GPU_VIRTUAL_ADDRESS TintAddress(const Renderer& renderer, int tint_slot) {
    return renderer.tint_constants->GetGPUVirtualAddress() +
           static_cast<uint64_t>(tint_slot) * kTintConstantStride;
}

// Serialises and creates a root signature. On a serialisation failure the compiler's
// own message is printed, because that text names the offending parameter.
bool CreateRootSignatureFromDescription(ID3D12Device& device,
                                        const D3D12_ROOT_SIGNATURE_DESC& description,
                                        const wchar_t* debug_name,
                                        ComPtr<ID3D12RootSignature>& signature_out) {
    ComPtr<ID3DBlob> serialized;
    ComPtr<ID3DBlob> error_text;
    const HRESULT serialize_result = D3D12SerializeRootSignature(
        &description, D3D_ROOT_SIGNATURE_VERSION_1, serialized.GetAddressOf(),
        error_text.GetAddressOf());
    if (FAILED(serialize_result)) {
        if (error_text != nullptr) {
            std::fprintf(stderr, "root signature error: %.*s\n",
                         static_cast<int>(error_text->GetBufferSize()),
                         static_cast<const char*>(error_text->GetBufferPointer()));
        }
        HrSucceeded(serialize_result, __FILE__, __LINE__, "D3D12SerializeRootSignature",
                    "the root signature description is valid; the printed text names the "
                    "offending field");
        return false;
    }
    if (!HrSucceeded(device.CreateRootSignature(0, serialized->GetBufferPointer(),
                                                serialized->GetBufferSize(),
                                                IID_PPV_ARGS(signature_out.GetAddressOf())),
                     __FILE__, __LINE__, "ID3D12Device::CreateRootSignature",
                     "the device is still present and the serialised blob is intact")) {
        return false;
    }
    signature_out->SetName(debug_name);
    return true;
}

bool CreateSceneRootSignature(Renderer& renderer, ID3D12Device& device) {
    // The contract's baseline: one root CBV at b0, shader visibility ALL, space 0, and
    // ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT so the vertex-colour pipeline can declare an
    // input layout. The vertex-colour shaders never read b0; every draw binds it anyway so
    // all three pipelines share one signature, exactly as the header specifies.
    D3D12_ROOT_PARAMETER tint_parameter{};
    tint_parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    tint_parameter.Descriptor.ShaderRegister = 0;
    tint_parameter.Descriptor.RegisterSpace = 0;
    tint_parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC description{};
    description.NumParameters = 1;
    description.pParameters = &tint_parameter;
    description.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    return CreateRootSignatureFromDescription(device, description, L"scene_root_signature",
                                              renderer.root_signature);
}

// The state every pipeline shares. Callers override blend and depth; the three pipelines
// share everything else. Scissor is not pipeline state in D3D12: it is per-draw state set
// in ClearAndDrawScene, which is why no pipeline mentions it.
D3D12_GRAPHICS_PIPELINE_STATE_DESC MakeBasePipelineDescription(
    ID3D12RootSignature& root_signature, const D3D12_SHADER_BYTECODE& vertex_shader,
    const D3D12_SHADER_BYTECODE& pixel_shader, const D3D12_INPUT_LAYOUT_DESC& input_layout) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC description{};
    description.pRootSignature = &root_signature;
    description.VS = vertex_shader;
    description.PS = pixel_shader;
    description.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
    description.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    description.NumRenderTargets = 1;
    description.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    description.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    description.SampleDesc.Count = 1;
    description.SampleMask = UINT_MAX;
    description.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
    description.InputLayout = input_layout;

    // Winding is irrelevant for these axis-aligned quads and a fullscreen triangle, so
    // culling is off; a cull setting must never be what makes a scene pass or fail.
    description.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    description.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    description.RasterizerState.FrontCounterClockwise = FALSE;
    description.RasterizerState.DepthBias = 0;
    description.RasterizerState.DepthBiasClamp = 0.0f;
    description.RasterizerState.SlopeScaledDepthBias = 0.0f;
    description.RasterizerState.DepthClipEnable = TRUE;
    description.RasterizerState.MultisampleEnable = FALSE;
    description.RasterizerState.AntialiasedLineEnable = FALSE;
    description.RasterizerState.ForcedSampleCount = 0;
    description.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

    description.BlendState.AlphaToCoverageEnable = FALSE;
    description.BlendState.IndependentBlendEnable = FALSE;
    description.BlendState.RenderTarget[0].BlendEnable = FALSE;
    description.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    description.DepthStencilState.DepthEnable = FALSE;
    description.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    description.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    description.DepthStencilState.StencilEnable = FALSE;
    return description;
}

bool CreateFullscreenPipeline(Renderer& renderer, ID3D12Device& device,
                              const ShaderFile& vertex_shader, const ShaderFile& pixel_shader) {
    const D3D12_INPUT_LAYOUT_DESC no_input_layout{nullptr, 0};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC description = MakeBasePipelineDescription(
        *renderer.root_signature.Get(), vertex_shader.Bytecode(),
        pixel_shader.Bytecode(), no_input_layout);
    if (!HrSucceeded(device.CreateGraphicsPipelineState(&description, IID_PPV_ARGS(
                         renderer.fullscreen_pipeline.GetAddressOf())),
                     __FILE__, __LINE__, "ID3D12Device::CreateGraphicsPipelineState(fullscreen)",
                     "the shader bytecode matches the declared formats and the shader model is "
                     "supported by this adapter")) {
        return false;
    }
    renderer.fullscreen_pipeline->SetName(L"fullscreen_pipeline");
    return true;
}

bool CreateBlendPipeline(Renderer& renderer, ID3D12Device& device,
                         const ShaderFile& vertex_shader, const ShaderFile& pixel_shader) {
    const D3D12_INPUT_LAYOUT_DESC no_input_layout{nullptr, 0};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC description = MakeBasePipelineDescription(
        *renderer.root_signature.Get(), vertex_shader.Bytecode(),
        pixel_shader.Bytecode(), no_input_layout);
    description.BlendState.RenderTarget[0].BlendEnable = TRUE;
    description.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
    description.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    description.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    // The contract states the alpha channel explicitly: ONE, INV_SRC_ALPHA, ADD, which
    // evaluates to 1.0*0.5 + 0.5*1.0 = 1.0 and stores 255. SRC_ALPHA here would store
    // 191 and fail the scene's checks.
    description.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    description.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
    description.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    if (!HrSucceeded(device.CreateGraphicsPipelineState(&description, IID_PPV_ARGS(
                         renderer.blend_pipeline.GetAddressOf())),
                     __FILE__, __LINE__, "ID3D12Device::CreateGraphicsPipelineState(blend)",
                     "the render target format supports blending; R8G8B8A8_UNORM always does")) {
        return false;
    }
    renderer.blend_pipeline->SetName(L"blend_pipeline");
    return true;
}

bool CreateVertexColorPipeline(Renderer& renderer, ID3D12Device& device,
                               const ShaderFile& vertex_shader, const ShaderFile& pixel_shader) {
    D3D12_INPUT_LAYOUT_DESC input_layout{};
    input_layout.pInputElementDescs = kVertexColorInputLayout;
    input_layout.NumElements = static_cast<UINT>(
        sizeof(kVertexColorInputLayout) / sizeof(kVertexColorInputLayout[0]));
    D3D12_GRAPHICS_PIPELINE_STATE_DESC description = MakeBasePipelineDescription(
        *renderer.root_signature.Get(), vertex_shader.Bytecode(), pixel_shader.Bytecode(),
        input_layout);
    // The only pipeline with depth: the contract asks for LESS with writes enabled so the
    // nearer green quad wins regardless of draw order.
    description.DepthStencilState.DepthEnable = TRUE;
    description.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    description.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    if (!HrSucceeded(device.CreateGraphicsPipelineState(&description, IID_PPV_ARGS(
                         renderer.vertex_color_pipeline.GetAddressOf())),
                     __FILE__, __LINE__, "ID3D12Device::CreateGraphicsPipelineState(vertex_color)",
                     "the input layout matches the shader's semantics and the depth format is "
                     "supported")) {
        return false;
    }
    renderer.vertex_color_pipeline->SetName(L"vertex_color_pipeline");
    return true;
}

bool LoadShaderFile(const std::filesystem::path& path, ShaderFile& shader_out) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        ReportFailure(__FILE__, __LINE__, "std::ifstream::open", "could not open " + path.string(),
                      "the build ran the conform_shaders target and the shaders directory sits "
                      "next to conform_native.exe; use --shaders to point elsewhere");
        return false;
    }
    const std::streamoff size = input.tellg();
    if (size <= 0) {
        ReportFailure(__FILE__, __LINE__, "std::ifstream::tellg",
                      path.string() + " is empty",
                      "rebuild with the conform_shaders target so dxc writes a real .cso");
        return false;
    }
    shader_out.bytes.resize(static_cast<size_t>(size));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(shader_out.bytes.data()), size);
    if (!input) {
        ReportFailure(__FILE__, __LINE__, "std::ifstream::read",
                      "could not read all of " + path.string(),
                      "the file is not locked by another process and the disk is readable");
        return false;
    }
    return true;
}

bool CreatePipelines(Renderer& renderer, ID3D12Device& device,
                     const std::filesystem::path& shader_directory) {
    ShaderFile fullscreen_vertex_shader;
    ShaderFile fullscreen_pixel_shader;
    ShaderFile vertex_color_vertex_shader;
    ShaderFile vertex_color_pixel_shader;
    if (!LoadShaderFile(shader_directory / "fullscreen_color.vs.cso", fullscreen_vertex_shader) ||
        !LoadShaderFile(shader_directory / "fullscreen_color.ps.cso", fullscreen_pixel_shader) ||
        !LoadShaderFile(shader_directory / "vertex_color.vs.cso", vertex_color_vertex_shader) ||
        !LoadShaderFile(shader_directory / "vertex_color.ps.cso", vertex_color_pixel_shader)) {
        return false;
    }

    if (!CreateSceneRootSignature(renderer, device) ||
        !CreateFullscreenPipeline(renderer, device, fullscreen_vertex_shader,
                                  fullscreen_pixel_shader) ||
        !CreateBlendPipeline(renderer, device, fullscreen_vertex_shader,
                             fullscreen_pixel_shader) ||
        !CreateVertexColorPipeline(renderer, device, vertex_color_vertex_shader,
                                   vertex_color_pixel_shader)) {
        return false;
    }
    return true;
}

bool CreateCommandObjects(Renderer& renderer, ID3D12Device& device) {
    if (!HrSucceeded(device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(
                         renderer.command_allocator.GetAddressOf())),
                     __FILE__, __LINE__, "ID3D12Device::CreateCommandAllocator",
                     "the direct queue was created successfully")) {
        return false;
    }
    if (!HrSucceeded(device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                              renderer.command_allocator.Get(), nullptr,
                                              IID_PPV_ARGS(renderer.command_list.GetAddressOf())),
                     __FILE__, __LINE__, "ID3D12Device::CreateCommandList",
                     "the allocator is valid and no list of this type is already recording")) {
        return false;
    }
    // A newly created list is open; close it so the first scene starts from a clean reset.
    return HrSucceeded(renderer.command_list->Close(), __FILE__, __LINE__,
                       "ID3D12GraphicsCommandList::Close",
                       "the list was created in a recordable state");
}

// Inserts one transition for every subresource. All barriers in this file go through it
// so the before-state is always the state the previous barrier left behind.
bool TransitionColorTarget(Renderer& renderer, D3D12_RESOURCE_STATES state_before,
                           D3D12_RESOURCE_STATES state_after) {
    if (state_before == state_after) {
        ReportFailure(__FILE__, __LINE__, "TransitionColorTarget",
                      "the barrier would have identical before and after states",
                      "this is a renderer bug in the scene state machine, not a driver issue");
        return false;
    }
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = renderer.color_target.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = state_before;
    barrier.Transition.StateAfter = state_after;
    renderer.command_list->ResourceBarrier(1, &barrier);
    return true;
}

// Binds the pipeline named by the plan and issues exactly one draw. Split from the
// recording function so that function stays short and the draw geometry is in one place.
void BindPipelineAndDraw(Renderer& renderer, const DrawPlan& plan) {
    ID3D12PipelineState* pipeline = nullptr;
    switch (plan.pipeline) {
        case PipelineKind::Fullscreen:
            pipeline = renderer.fullscreen_pipeline.Get();
            break;
        case PipelineKind::Blend:
            pipeline = renderer.blend_pipeline.Get();
            break;
        case PipelineKind::VertexColor:
            pipeline = renderer.vertex_color_pipeline.Get();
            break;
    }
    ID3D12GraphicsCommandList* list = renderer.command_list.Get();
    list->SetPipelineState(pipeline);
    // Bound for every draw, including vertex-colour draws whose shaders ignore it: the
    // contract's baseline names one root CBV at b0 for every scene.
    list->SetGraphicsRootSignature(renderer.root_signature.Get());
    list->SetGraphicsRootConstantBufferView(0, TintAddress(renderer, plan.tint_slot));
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    switch (plan.geometry) {
        case GeometryKind::None:
            return;
        case GeometryKind::FullscreenTriangle:
            list->DrawInstanced(kFullscreenTriangleVertices, 1, 0, 0);
            return;
        case GeometryKind::DepthOrderQuads:
            list->IASetVertexBuffers(0, 1, &renderer.depth_order_vertex_view);
            list->IASetIndexBuffer(&renderer.depth_order_index_view);
            // The index buffer carries the contract's draw order: near green first, far
            // red second. A renderer that ignores depth paints red over green.
            list->DrawIndexedInstanced(
                static_cast<UINT>(sizeof(conform::kDepthOrderIndices) / sizeof(uint16_t)), 1, 0, 0, 0);
            return;
        case GeometryKind::IndexedTriangles:
            list->IASetVertexBuffers(0, 1, &renderer.indexed_vertex_view);
            list->IASetIndexBuffer(&renderer.indexed_index_view);
            list->DrawIndexedInstanced(
                static_cast<UINT>(sizeof(conform::kIndexedIndices) / sizeof(uint16_t)), 1, 0, 0, 0);
            return;
    }
}

// Clears the colour target (and the depth target for depth_order only), then draws when
// the plan asks for geometry. The clear always happens, which is what the clear scene
// alone tests.
void ClearAndDrawScene(Renderer& renderer, const DrawPlan& plan) {
    ID3D12GraphicsCommandList* list = renderer.command_list.Get();
    // The contract's baseline binds a DSV only for depth_order; every other scene passes
    // a null DSV, which leaves depth testing inert.
    const D3D12_CPU_DESCRIPTOR_HANDLE* depth_view =
        plan.binds_depth_target ? &renderer.depth_stencil_view : nullptr;
    list->OMSetRenderTargets(1, &renderer.color_render_target_view, FALSE, depth_view);

    const float clear_color[4] = {conform::kClearR / 255.0f, conform::kClearG / 255.0f,
                                  conform::kClearB / 255.0f, conform::kClearA / 255.0f};
    list->ClearRenderTargetView(renderer.color_render_target_view, clear_color, 0, nullptr);
    if (plan.binds_depth_target) {
        list->ClearDepthStencilView(renderer.depth_stencil_view, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0,
                                    nullptr);
    }
    list->RSSetViewports(1, &renderer.viewport);
    list->RSSetScissorRects(1, &plan.scissor);
    if (plan.geometry != GeometryKind::None) {
        BindPipelineAndDraw(renderer, plan);
    }
}

// Copies the rendered target into the readback buffer at the contract's row pitch.
void CopyColorTargetToReadback(Renderer& renderer) {
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = renderer.readback_buffer.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint.Offset = 0;  // 512-byte aligned, as placed footprints require
    destination.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    destination.PlacedFootprint.Footprint.Width = static_cast<UINT>(conform::kSceneWidth);
    destination.PlacedFootprint.Footprint.Height = static_cast<UINT>(conform::kSceneHeight);
    destination.PlacedFootprint.Footprint.Depth = 1;
    destination.PlacedFootprint.Footprint.RowPitch = kReadbackRowPitch;

    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = renderer.color_target.Get();
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    source.SubresourceIndex = 0;

    const D3D12_BOX source_box{0, 0, 0, static_cast<UINT>(conform::kSceneWidth),
                               static_cast<UINT>(conform::kSceneHeight), 1};
    renderer.command_list->CopyTextureRegion(&destination, 0, 0, 0, &source, &source_box);
}

// Records one scene's whole pass: reset, clear, optional draw, copy out, back to COMMON
// so the next scene always starts from the same state.
bool RecordSceneCommands(Renderer& renderer, const DrawPlan& plan) {
    if (!HrSucceeded(renderer.command_allocator->Reset(), __FILE__, __LINE__,
                     "ID3D12CommandAllocator::Reset",
                     "the GPU finished the previous scene; RenderScene waits on a fence before "
                     "the next reset")) {
        return false;
    }
    if (!HrSucceeded(renderer.command_list->Reset(renderer.command_allocator.Get(), nullptr),
                     __FILE__, __LINE__, "ID3D12GraphicsCommandList::Reset",
                     "the list is closed and its allocator was just reset")) {
        return false;
    }

    if (!TransitionColorTarget(renderer, D3D12_RESOURCE_STATE_COMMON,
                               D3D12_RESOURCE_STATE_RENDER_TARGET)) {
        return false;
    }
    ClearAndDrawScene(renderer, plan);
    if (!TransitionColorTarget(renderer, D3D12_RESOURCE_STATE_RENDER_TARGET,
                               D3D12_RESOURCE_STATE_COPY_SOURCE)) {
        return false;
    }
    CopyColorTargetToReadback(renderer);
    if (!TransitionColorTarget(renderer, D3D12_RESOURCE_STATE_COPY_SOURCE,
                               D3D12_RESOURCE_STATE_COMMON)) {
        return false;
    }
    return HrSucceeded(renderer.command_list->Close(), __FILE__, __LINE__,
                       "ID3D12GraphicsCommandList::Close",
                       "every recording call above succeeded on a list that was open");
}

// Maps the readback buffer and copies it row by row; the row pitch is explicit because
// a buffer's mapped layout follows the placed footprint, not the image width alone.
bool CopyReadbackToSurface(Renderer& renderer, PixelSurface& surface_out) {
    void* mapped = nullptr;
    if (!HrSucceeded(renderer.readback_buffer->Map(0, nullptr, &mapped), __FILE__, __LINE__,
                     "ID3D12Resource::Map(readback)",
                     "the GPU finished the copy and the resource is a readback heap buffer")) {
        return false;
    }
    surface_out.width = conform::kSceneWidth;
    surface_out.height = conform::kSceneHeight;
    // The surface is tightly packed; the mapped buffer is not. Copy row bytes, not pitch.
    const SIZE_T image_size = static_cast<SIZE_T>(kReadbackRowBytes) * conform::kSceneHeight;
    surface_out.rgba.resize(image_size);
    for (int row = 0; row < conform::kSceneHeight; ++row) {
        std::memcpy(surface_out.rgba.data() + static_cast<SIZE_T>(row) * kReadbackRowBytes,
                    static_cast<const uint8_t*>(mapped) + static_cast<SIZE_T>(row) * kReadbackRowPitch,
                    kReadbackRowBytes);
    }
    renderer.readback_buffer->Unmap(0, nullptr);
    return true;
}

}  // namespace

bool Renderer::Initialize(GpuContext& gpu, const std::filesystem::path& shader_directory) {
    if (!CreateDescriptorHeaps(*this, *gpu.device.Get()) || !CreateTargets(*this, *gpu.device.Get()) ||
        !CreateGeometryBuffers(*this, *gpu.device.Get()) ||
        !CreateTintConstants(*this, *gpu.device.Get()) || !CreatePipelines(*this, *gpu.device.Get(),
                                                                           shader_directory) ||
        !CreateCommandObjects(*this, *gpu.device.Get())) {
        return false;
    }

    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;
    viewport.Width = static_cast<float>(conform::kSceneWidth);
    viewport.Height = static_cast<float>(conform::kSceneHeight);
    viewport.MinDepth = 0.0f;  // the clip-space z values in scenes.h map straight to 0..1
    viewport.MaxDepth = 1.0f;
    return true;
}

bool Renderer::RenderScene(GpuContext& gpu, const DrawPlan& plan, PixelSurface& surface_out) {
    if (plan.tint_slot < 0 || plan.tint_slot >= conform::kSceneCount) {
        ReportFailure(__FILE__, __LINE__, "Renderer::RenderScene",
                      "tint slot " + std::to_string(plan.tint_slot) +
                          " is outside the constant buffer",
                      "MakeDrawPlan is given an index inside conform::kScenes");
        return false;
    }
    if (!RecordSceneCommands(*this, plan)) {
        return false;
    }
    ID3D12CommandList* command_lists[] = {command_list.Get()};
    gpu.command_queue->ExecuteCommandLists(1, command_lists);
    if (!SignalAndWaitForGpu(gpu) || !DeviceStillPresent(gpu)) {
        return false;
    }
    return CopyReadbackToSurface(*this, surface_out);
}
