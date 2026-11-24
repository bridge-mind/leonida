// Native D3D12 reference for SV_InnerCoverage and the matching Vulkan capability report.
#include "vk_test_device.h"
#include "d3d12_copy_compare.h"
#include "d3d12_test_shaders.h"
#include "host_wine.h"

#include <d3dcompiler.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return false; } } while (0)

namespace {
constexpr UINT width = 32, height = 32, pitch = 256;
constexpr double required_margin = 1.0 / 64.0;
const D3D12_RECT full_scissor{0, 0, LONG(width), LONG(height)};

struct Point {
    double x, y;
};
using Triangle = std::array<Point, 3>;
using VertexData = std::array<float, 6>;

struct Fixture {
    copycmp::Run run;
    copycmp::Ptr<ID3D12RootSignature> root;
    D3D12_SHADER_BYTECODE vs{}, ps{};
};

double cross(Point a, Point b, Point p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

bool pixel_inside(const Triangle& triangle, UINT x, UINT y, double margin) {
    const double winding = cross(triangle[0], triangle[1], triangle[2]);
    if (std::abs(winding) < 1e-12) return false;
    const double orientation = winding > 0 ? 1.0 : -1.0;
    const Point corners[] = {{double(x), double(y)}, {double(x + 1), double(y)},
                             {double(x), double(y + 1)}, {double(x + 1), double(y + 1)}};
    for (unsigned edge = 0; edge < 3; ++edge) {
        const Point a = triangle[edge], b = triangle[(edge + 1) % 3];
        const double length = std::hypot(b.x - a.x, b.y - a.y);
        if (length == 0) return false;
        for (const Point corner : corners)
            if (orientation * cross(a, b, corner) / length <= margin) return false;
    }
    return true;
}

VertexData vertex_data(const Triangle& triangle) {
    VertexData out{};
    for (unsigned i = 0; i < 3; ++i) {
        out[i * 2] = float(triangle[i].x * 2 / width - 1);
        out[i * 2 + 1] = float(1 - triangle[i].y * 2 / height);
    }
    return out;
}

Triangle actual_triangle(const VertexData& input) {
    Triangle triangle{};
    for (unsigned i = 0; i < 3; ++i)
        triangle[i] = {(double(input[i * 2]) + 1) * width / 2,
                       (1 - double(input[i * 2 + 1])) * height / 2};
    return triangle;
}

bool compile_shader(pD3DCompile compile, const char* source, const char* entry, const char* profile,
                    copycmp::Ptr<ID3DBlob>& code) {
    copycmp::Ptr<ID3DBlob> error;
    const HRESULT result = compile(source, std::strlen(source), nullptr, nullptr, nullptr, entry, profile, 0, 0,
                                   &code.p, &error.p);
    if (SUCCEEDED(result)) return true;
    std::fprintf(stderr, "D3DCompile %s failed: 0x%08x", entry, unsigned(result));
    if (error.p) std::fprintf(stderr, ": %.*s", int(error->GetBufferSize()),
                              static_cast<const char*>(error->GetBufferPointer()));
    std::fputc('\n', stderr);
    return false;
}

bool setup(Fixture& fixture, ID3D12Device* device, D3D12_SHADER_BYTECODE vs, D3D12_SHADER_BYTECODE ps) {
    fixture.vs = vs;
    fixture.ps = ps;
    if (!fixture.run.init(device, false)) return false;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    const auto bytes = d3d12test::serialize_root_signature(desc);
    CHECK(!bytes.empty());
    CHECK(SUCCEEDED(device->CreateRootSignature(0, bytes.data(), bytes.size(), IID_PPV_ARGS(&fixture.root.p))));
    return true;
}

bool render(Fixture& fixture, const Triangle& triangle, std::vector<unsigned char>& rgba) {
    static const D3D12_INPUT_ELEMENT_DESC element{
        "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = fixture.root.p;
    desc.VS = fixture.vs;
    desc.PS = fixture.ps;
    desc.InputLayout = {&element, 1};
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleDesc.Count = 1;
    desc.SampleMask = UINT_MAX;

    copycmp::Ptr<ID3D12PipelineState> pso;
    const HRESULT pipeline_result = fixture.run.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p));
    if (FAILED(pipeline_result)) {
        std::fprintf(stderr, "CreateGraphicsPipelineState failed: 0x%08x\n", unsigned(pipeline_result));
        return false;
    }

    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = width;
    tex.Height = height;
    tex.DepthOrArraySize = tex.MipLevels = 1;
    tex.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    tex.SampleDesc.Count = 1;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    copycmp::Ptr<ID3D12Resource> target, vertices, readback;
    CHECK(fixture.run.texture(tex, D3D12_RESOURCE_STATE_RENDER_TARGET, target));
    CHECK(fixture.run.buffer(D3D12_HEAP_TYPE_UPLOAD, sizeof(VertexData), D3D12_RESOURCE_STATE_GENERIC_READ, vertices));
    CHECK(fixture.run.buffer(D3D12_HEAP_TYPE_READBACK, pitch * height, D3D12_RESOURCE_STATE_COPY_DEST, readback));

    const VertexData input = vertex_data(triangle);
    void* data = nullptr;
    const D3D12_RANGE none{};
    CHECK(SUCCEEDED(vertices->Map(0, &none, &data)));
    std::memcpy(data, input.data(), sizeof(input));
    vertices->Unmap(0, nullptr);

    copycmp::Ptr<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    CHECK(SUCCEEDED(fixture.run.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    fixture.run.device->CreateRenderTargetView(target.p, nullptr, handle);

    fixture.run.begin();
    const float clear[4] = {};
    fixture.run.list->ClearRenderTargetView(handle, clear, 0, nullptr);
    fixture.run.list->OMSetRenderTargets(1, &handle, FALSE, nullptr);
    fixture.run.list->SetGraphicsRootSignature(fixture.root.p);
    fixture.run.list->SetPipelineState(pso.p);
    const D3D12_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
    fixture.run.list->RSSetViewports(1, &viewport);
    fixture.run.list->RSSetScissorRects(1, &full_scissor);
    const D3D12_VERTEX_BUFFER_VIEW vb{vertices->GetGPUVirtualAddress(), sizeof(input), 8};
    fixture.run.list->IASetVertexBuffers(0, 1, &vb);
    fixture.run.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    fixture.run.list->DrawInstanced(3, 1, 0, 0);
    fixture.run.barrier(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);

    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = target.p;
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    to.pResource = readback.p;
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint.Footprint = {tex.Format, width, height, 1, pitch};
    fixture.run.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    const std::string error = fixture.run.submit();
    if (!error.empty()) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return false;
    }

    const D3D12_RANGE read_range{0, pitch * height};
    CHECK(SUCCEEDED(readback->Map(0, &read_range, &data)));
    const auto* pixels = static_cast<const unsigned char*>(data);
    rgba.resize(width * height * 4);
    for (UINT y = 0; y < height; ++y)
        std::copy_n(pixels + y * pitch, width * 4, rgba.begin() + y * width * 4);
    readback->Unmap(0, &none);
    return true;
}
} // namespace

int main() {
    if (hostwine::host_is_wine()) return 77;

    vktest::Device vk;
    const auto vk_status = vk.initialize();
    if (vk_status == vktest::DeviceStatus::unavailable) return 77;
    if (vk_status != vktest::DeviceStatus::ready) return 1;

    VkPhysicalDeviceConservativeRasterizationPropertiesEXT conservative{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT};
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    id.pNext = vk.conservative_rasterization ? &conservative : nullptr;
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    const auto get_properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        vk.instance_proc(vk.instance, "vkGetPhysicalDeviceProperties2"));
    if (!get_properties2) return 1;
    get_properties2(vk.physical, &properties);

    std::printf("Vulkan adapter: %s; VK_EXT_conservative_rasterization=%s; "
                "fullyCoveredFragmentShaderInputVariable=%s; subPixelPrecisionBits=%u\n",
                properties.properties.deviceName, vk.conservative_rasterization ? "yes" : "no",
                vk.conservative_rasterization
                    ? (conservative.fullyCoveredFragmentShaderInputVariable ? "true" : "false")
                    : "unavailable",
                properties.properties.limits.subPixelPrecisionBits);

    if (!id.deviceLUIDValid) {
        std::puts("SKIP: Vulkan did not expose a device LUID to match its native D3D12 adapter");
        return 77;
    }
    static_assert(sizeof(LUID) == VK_LUID_SIZE);
    LUID luid{};
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));
    copycmp::Ptr<ID3D12Device> device;
    std::string adapter;
    if (!copycmp::try_create_host(device, adapter, &luid)) {
        std::puts("SKIP: no native D3D12 device for the selected Vulkan adapter");
        return 77;
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)))) return 1;
    std::printf("Native D3D12 adapter: %s; ConservativeRasterizationTier=%u\n",
                adapter.c_str(), unsigned(options.ConservativeRasterizationTier));
    if (options.ConservativeRasterizationTier < D3D12_CONSERVATIVE_RASTERIZATION_TIER_3) {
        std::puts("SKIP: native D3D12 Tier 3 is required for the SV_InnerCoverage reference");
        return 77;
    }

    const HMODULE compiler_module = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile = compiler_module ? reinterpret_cast<pD3DCompile>(
        reinterpret_cast<void*>(GetProcAddress(compiler_module, "D3DCompile"))) : nullptr;
    if (!compile) {
        std::puts("SKIP: d3dcompiler_47.dll is required for the synthetic reference shader");
        return 77;
    }
    const char* hlsl =
        "float4 vs(float2 p:POSITION):SV_Position{return float4(p,0.5,1);}"
        "float4 ps(uint inner:SV_InnerCoverage):SV_Target0{"
        "return float4(1,(inner&1u)?1:0,inner==0u?1:0,1);}";
    copycmp::Ptr<ID3DBlob> vs, ps;
    if (!compile_shader(compile, hlsl, "vs", "vs_5_0", vs) ||
        !compile_shader(compile, hlsl, "ps", "ps_5_0", ps)) return 1;

    Fixture fixture;
    if (!setup(fixture, device.p, {vs->GetBufferPointer(), vs->GetBufferSize()},
               {ps->GetBufferPointer(), ps->GetBufferSize()})) return 1;

    const Triangle scenes[] = {
        {{{3.125, 3.125}, {28.125, 3.125}, {3.125, 28.125}}},
        {{{3.125, 6.125}, {27.125, 11.125}, {8.125, 28.125}}},
        {{{4.125, 24.125}, {27.125, 6.125}, {27.125, 25.125}}},
        {{{3.125, 8.125}, {28.125, 8.625}, {16.125, 25.125}}},
    };
    unsigned required_pixels = 0, inner_pixels = 0, fragments = 0;
    for (const Triangle& scene : scenes) {
        const VertexData input = vertex_data(scene);
        const Triangle triangle = actual_triangle(input);
        std::vector<unsigned char> rgba;
        if (!render(fixture, triangle, rgba)) return 1;

        unsigned scene_required = 0;
        for (UINT y = 0; y < height; ++y) {
            for (UINT x = 0; x < width; ++x) {
                const std::size_t pixel = (std::size_t(y) * width + x) * 4;
                const bool ran = rgba[pixel] == 255;
                const bool inner = rgba[pixel + 1] == 255;
                const bool zero = rgba[pixel + 2] == 255;
                if (!ran) {
                    if (inner || zero || rgba[pixel + 3]) {
                        std::fprintf(stderr, "background pixel (%u,%u) was unexpectedly written\n", x, y);
                        return 1;
                    }
                    continue;
                }
                if (rgba[pixel + 3] != 255) {
                    std::fprintf(stderr, "fragment at (%u,%u) did not write alpha\n", x, y);
                    return 1;
                }
                ++fragments;
                if (inner == zero) {
                    std::fprintf(stderr, "SV_InnerCoverage encoding at (%u,%u) is not false=0 / true=odd\n", x, y);
                    return 1;
                }
                if (inner) {
                    if (!pixel_inside(triangle, x, y, -1e-10)) {
                        std::fprintf(stderr, "false positive InnerCoverage at (%u,%u)\n", x, y);
                        return 1;
                    }
                    ++inner_pixels;
                }
                if (pixel_inside(triangle, x, y, required_margin)) {
                    ++scene_required;
                    if (!inner) {
                        std::fprintf(stderr, "missing guaranteed InnerCoverage at (%u,%u)\n", x, y);
                        return 1;
                    }
                }
            }
        }
        if (!scene_required) {
            std::fprintf(stderr, "reference scene has no pixels beyond the inner uncertainty margin\n");
            return 1;
        }
        required_pixels += scene_required;
    }

    std::printf("SV_InnerCoverage reference: %u scenes, %u conservative fragments, %u inner pixels, "
                "%u guaranteed interior pixels checked; no false positives\n",
                unsigned(std::size(scenes)), fragments, inner_pixels, required_pixels);
    return 0;
}
