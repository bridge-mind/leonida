#pragma once
// Synthetic D3D12/Vulkan conservative coverage. Required/forbidden pixels come from geometric clipping,
// not a driver snapshot: Tier 1 may over-cover by half a pixel, so uncertain pixels are not compared exactly.
#include "vk_test_device.h"
#include "d3d12_copy_compare.h"
#include "d3d12_test_shaders.h"
#include "host_wine.h"
#include "../src/conservative_raster.h"
#include "../src/shader_library.h"
#include <d3dcompiler.h>
#include <array>
#include <cmath>
#include <climits>
#include <span>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return false; } } while (0)
namespace conservative_test {
constexpr UINT width = 32, height = 32, pitch = 256;
struct Point { double x, y; };
using Triangle = std::array<Point, 3>;
using Mask = std::array<bool, width * height>;
struct Fixture {
    copycmp::Run run;
    copycmp::Ptr<ID3D12RootSignature> root;
    D3D12_SHADER_BYTECODE vs{}, ps{};
    std::span<const D3D12_INPUT_ELEMENT_DESC> input;
};
struct VertexInput {
    std::vector<float> data;
    UINT stride = 16;
    std::vector<std::uint16_t> indices;
    D3D_PRIMITIVE_TOPOLOGY topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
};

inline bool setup(Fixture& f, ID3D12Device* device, bool ours, D3D12_SHADER_BYTECODE vs, D3D12_SHADER_BYTECODE ps) {
    f.vs = vs; f.ps = ps;
    CHECK(f.run.init(device, ours));
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    const auto bytes = d3d12test::serialize_root_signature(desc);
    CHECK(!bytes.empty());
    CHECK(SUCCEEDED(device->CreateRootSignature(0, bytes.data(), bytes.size(), IID_PPV_ARGS(&f.root.p))));
    return true;
}

inline D3D12_GRAPHICS_PIPELINE_STATE_DESC description(const Fixture& f, bool conservative, UINT samples, bool variant) {
    static const D3D12_INPUT_ELEMENT_DESC element{
        "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC d{};
    d.pRootSignature = f.root.p;
    d.VS = f.vs; d.PS = f.ps;
    d.InputLayout = f.input.empty() ? D3D12_INPUT_LAYOUT_DESC{&element, 1}
                                  : D3D12_INPUT_LAYOUT_DESC{f.input.data(), UINT(f.input.size())};
    d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    d.RasterizerState.DepthClipEnable = TRUE;
    d.RasterizerState.MultisampleEnable = samples > 1;
    d.RasterizerState.ConservativeRaster = conservative ? D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON
                                                       : D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    d.NumRenderTargets = variant ? 2 : 1;
    d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    if (variant) d.RTVFormats[1] = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    d.SampleDesc.Count = samples;
    d.SampleMask = UINT_MAX;
    return d;
}

// Clip the triangle to a pixel rectangle and return its intersection area.
inline double area(const Triangle& triangle, double left, double top, double right, double bottom, bool* intersects = nullptr) {
    std::vector<Point> p(triangle.begin(), triangle.end());
    for (unsigned side = 0; side < 4 && !p.empty(); ++side) {
        const auto distance = [&](Point v) {
            return side == 0 ? v.x - left : side == 1 ? right - v.x : side == 2 ? v.y - top : bottom - v.y;
        };
        std::vector<Point> out;
        Point a = p.back(); double da = distance(a);
        for (const Point b : p) {
            const double db = distance(b);
            if ((da >= 0) != (db >= 0)) {
                const double t = da / (da - db);
                out.push_back({a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)});
            }
            if (db >= 0) out.push_back(b);
            a = b; da = db;
        }
        p = std::move(out);
    }
    double sum = 0;
    if (intersects) *intersects = !p.empty(); // a degenerate triangle may cover pixels despite having zero area
    for (std::size_t i = 0; i < p.size(); ++i) {
        const Point a = p[i], b = p[(i + 1) % p.size()];
        sum += a.x * b.y - a.y * b.x;
    }
    return std::abs(sum) * 0.5;
}

inline bool render(Fixture& f, const Triangle& triangle, bool conservative, UINT samples, bool variant,
            const D3D12_RECT& scissor, Mask& mask, D3D12_CULL_MODE cull = D3D12_CULL_MODE_NONE,
            const VertexInput* input = nullptr, std::vector<unsigned char>* rgba = nullptr) {
    auto& r = f.run;
    auto desc = description(f, conservative, samples, variant);
    desc.RasterizerState.CullMode = cull;
    copycmp::Ptr<ID3D12PipelineState> pso;
    const auto created = r.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p));
    if (FAILED(created)) {
        std::fprintf(stderr, "%s CreateGraphicsPipelineState=%08x conservative=%d samples=%u variant=%d\n",
                     r.ours ? "Vulkan" : "host", unsigned(created), int(conservative), samples, int(variant));
        return false;
    }
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = width; tex.Height = height;
    tex.DepthOrArraySize = tex.MipLevels = 1;
    tex.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    tex.SampleDesc.Count = samples;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    copycmp::Ptr<ID3D12Resource> target, resolved, vertices, indices, readback;
    CHECK(r.texture(tex, D3D12_RESOURCE_STATE_RENDER_TARGET, target));
    if (samples > 1) {
        tex.SampleDesc.Count = 1; tex.Flags = D3D12_RESOURCE_FLAG_NONE;
        CHECK(r.texture(tex, D3D12_RESOURCE_STATE_RESOLVE_DEST, resolved));
    }
    const UINT vertex_bytes = input ? UINT(input->data.size() * sizeof(float)) : 24;
    CHECK(r.buffer(D3D12_HEAP_TYPE_UPLOAD, vertex_bytes, D3D12_RESOURCE_STATE_GENERIC_READ, vertices));
    CHECK(r.buffer(D3D12_HEAP_TYPE_READBACK, pitch * height, D3D12_RESOURCE_STATE_COPY_DEST, readback));
    void* data = nullptr;
    const D3D12_RANGE none{};
    CHECK(SUCCEEDED(vertices->Map(0, &none, &data)));
    if (input) std::memcpy(data, input->data.data(), vertex_bytes);
    else {
      auto* positions = static_cast<float*>(data);
      for (unsigned i = 0; i < 3; ++i) {
          positions[i * 2] = float(triangle[i].x * 2 / width - 1);
          positions[i * 2 + 1] = float(1 - triangle[i].y * 2 / height);
      }
    }
    vertices->Unmap(0, nullptr);
    if (input && !input->indices.empty()) {
        CHECK(r.buffer(D3D12_HEAP_TYPE_UPLOAD, input->indices.size() * sizeof(std::uint16_t),
                       D3D12_RESOURCE_STATE_GENERIC_READ, indices));
        CHECK(SUCCEEDED(indices->Map(0, &none, &data)));
        std::memcpy(data, input->indices.data(), input->indices.size() * sizeof(std::uint16_t));
        indices->Unmap(0, nullptr);
    }
    copycmp::Ptr<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    CHECK(SUCCEEDED(r.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    r.device->CreateRenderTargetView(target.p, nullptr, handle);
    r.begin();
    const float clear[4] = {};
    r.list->ClearRenderTargetView(handle, clear, 0, nullptr);
    r.list->OMSetRenderTargets(1, &handle, FALSE, nullptr);
    r.list->SetGraphicsRootSignature(f.root.p);
    r.list->SetPipelineState(pso.p);
    const D3D12_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
    r.list->RSSetViewports(1, &viewport);
    r.list->RSSetScissorRects(1, &scissor);
    const D3D12_VERTEX_BUFFER_VIEW vb{vertices->GetGPUVirtualAddress(), vertex_bytes, input ? input->stride : 8};
    r.list->IASetVertexBuffers(0, 1, &vb);
    r.list->IASetPrimitiveTopology(input ? input->topology : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    if (indices.p) {
        const D3D12_INDEX_BUFFER_VIEW ib{indices->GetGPUVirtualAddress(), UINT(input->indices.size() * 2), DXGI_FORMAT_R16_UINT};
        r.list->IASetIndexBuffer(&ib);
        r.list->DrawIndexedInstanced(UINT(input->indices.size()), 1, 0, 0, 0);
    } else r.list->DrawInstanced(input ? vertex_bytes / input->stride : 3, 1, 0, 0);
    ID3D12Resource* source = target.p;
    if (samples > 1) {
        r.barrier(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
        r.list->ResolveSubresource(resolved.p, 0, target.p, 0, tex.Format);
        r.barrier(resolved.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        source = resolved.p;
    } else r.barrier(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = source; from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    to.pResource = readback.p; to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint.Footprint = {tex.Format, width, height, 1, pitch};
    r.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    const auto error = r.submit();
    if (!error.empty()) { std::fprintf(stderr, "%s\n", error.c_str()); return false; }
    CHECK(SUCCEEDED(readback->Map(0, nullptr, &data)));
    const auto* pixels = static_cast<const unsigned char*>(data);
    if (rgba) rgba->resize(width * height * 4);
    for (UINT y = 0; y < height; ++y) for (UINT x = 0; x < width; ++x) {
        const auto* p = pixels + y * pitch + x * 4;
        if (rgba) {
            std::copy(p, p + 4, rgba->begin() + (y * width + x) * 4);
            mask[y * width + x] = (p[0] | p[1] | p[2] | p[3]) != 0;
        } else {
            CHECK(p[1] == 0 && p[2] == 0 && p[0] == p[3]);
            if (conservative) CHECK(p[0] == 0 || p[0] == 255); // every MSAA sample is covered
            mask[y * width + x] = p[0] != 0;
        }
    }
    readback->Unmap(0, &none);
    return true;
}

inline bool coverage(const Triangle& triangle, const D3D12_RECT& scissor, const Mask& mask) {
    constexpr double epsilon = 1.0 / 256;
    for (UINT y = 0; y < height; ++y) for (UINT x = 0; x < width; ++x) {
        const bool in = LONG(x) >= scissor.left && LONG(x) < scissor.right && LONG(y) >= scissor.top && LONG(y) < scissor.bottom;
        const bool required = in && area(triangle, x + epsilon, y + epsilon, x + 1 - epsilon, y + 1 - epsilon) > 1e-9;
        bool intersects = false;
        area(triangle, x - 0.5 - epsilon, y - 0.5 - epsilon, x + 1.5 + epsilon, y + 1.5 + epsilon, &intersects);
        const bool allowed = in && intersects;
        if ((required && !mask[y * width + x]) || (!allowed && mask[y * width + x])) {
            std::fprintf(stderr, "coverage (%u,%u): required=%d allowed=%d actual=%d\n", x, y,
                         int(required), int(allowed), int(mask[y * width + x]));
            return false;
        }
    }
    return true;
}


} // namespace conservative_test
