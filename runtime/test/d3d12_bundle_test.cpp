// ExecuteBundle: what a bundle's draws inherit from the list that executes it, and what the bundle hands back.
// The same scenarios run on the layer (a Vulkan device, shaders written to the translator's contract, the ones
// d3d12_draw_test uses) and on Windows' own D3D12 (HLSL from D3DCompile, the default adapter and WARP), and each
// is read back as an 8x8 grid of letters: B blue (the clear), R, G, Y the draw's colour, C a green-only write over
// the clear, K black. The expected grids are what the host produced when they were measured (docs/d3d12_vulkan.md
// section 3.2); the default adapter and WARP gave the same grid in every scenario, so the layer is held to the PC
// runtime's behaviour and each host is checked against the table too.
// Only valid bundle usage is run: a bundle never calls RSSetViewports, RSSetScissorRects, OMSetRenderTargets or any
// barrier, clear or copy, sets its root signature before any root argument and sets its topology before a draw
// (the default adapter inherits the parent's topology when it does not, WARP removes the device, so that is not
// asserted). The layer alone is also checked for the order of a bundle's commands in its parent, for the trace
// (one line per call the application made, however often the bundle runs) and for object lifetimes.
// Both host adapters always run (LEONIDA_HOST_ADAPTER is not read), except under Wine, where the host is skipped.
// No game data.
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_trace.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <io.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <variant>
#include <vector>

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_VS_SPV and LEONIDA_TEST_PS_SPV must name the compiled test shaders"
#endif

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

#define REQUIRE(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

constexpr UINT target_px = 64;
constexpr UINT grid_cells = 8;

struct Vertex {
    float x, y;
    std::uint8_t r, g, b, a;
};

// Upload-heap layout, in bytes.
constexpr UINT64 vb_quad = 0;        // the top-left quadrant as a strip: (-1,1) (0,1) (-1,0) (0,0)
constexpr UINT64 vb_other = 64;      // the same shape in the bottom-right quadrant
constexpr UINT64 ib_quad = 128;      // 0 1 2 2 1 3: both triangles of the quad
constexpr UINT64 ib_half = 144;      // 3 2 1: the lower-right half
constexpr UINT64 tint_red = 256;     // (1, 0, 1): the red vertices stay red
constexpr UINT64 tint_green = 512;   // (0, 1, 0)
constexpr UINT64 tint_yellow = 768;  // (1, 1, 1)

// An expected grid: eight rows of eight letters, rows separated by a space.
#define BLANK4 "BBBBBBBB BBBBBBBB BBBBBBBB BBBBBBBB"
#define ROWS4(row) row " " row " " row " " row

using Recorder = std::function<void(struct Rig&, ID3D12GraphicsCommandList*)>;

struct Rig {
    ID3D12Device* device = nullptr;
    Ref<ID3D12RootSignature> root1, root2;  // root2 adds a third parameter, so it is a different signature
    Ref<ID3D12PipelineState> pso_a, pso_b, pso_a2;  // b writes only green and alpha; a2 is a on root2
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence;
    Ref<ID3D12Resource> upload, target, readback;
    Ref<ID3D12DescriptorHeap> rtv_heap;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
    D3D12_GPU_VIRTUAL_ADDRESS upload_va = 0;
    UINT64 fence_value = 0;
    bool bundle_failed = false;
    std::vector<Ref<ID3D12CommandAllocator>> bundle_allocators;
    std::vector<Ref<ID3D12GraphicsCommandList>> bundles;

    std::vector<std::uint8_t> vs_code, ps_code;

    // A root signature of two parameters (constants s0 s1, a CBV) or three (and one more constant).
    bool make_root(UINT count, Ref<ID3D12RootSignature>& root) {
        D3D12_ROOT_PARAMETER params[3] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[1].Descriptor = {1, 0};
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[2].Constants = {2, 0, 1};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {count, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        const auto blob = serialize_root_signature(root_desc);
        CHECK(!blob.empty());
        CHECK(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));
        return true;
    }

    bool make_pso(ID3D12RootSignature* root, UINT8 write_mask, Ref<ID3D12PipelineState>& pso) {
        static const D3D12_INPUT_ELEMENT_DESC elements[2] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D12_APPEND_ALIGNED_ELEMENT,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = root;
        desc.VS = {vs_code.data(), vs_code.size()};
        desc.PS = {ps_code.data(), ps_code.size()};
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask = write_mask;
        desc.SampleMask = 0xffffffff;
        desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        desc.RasterizerState.DepthClipEnable = TRUE;
        desc.InputLayout = {elements, 2};
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p))));
        return true;
    }

    bool init(ID3D12Device* d, const std::vector<std::uint8_t>& vs, const std::vector<std::uint8_t>& ps) {
        device = d;
        vs_code = vs;
        ps_code = ps;
        CHECK(make_root(2, root1) && make_root(3, root2));
        constexpr UINT8 all = D3D12_COLOR_WRITE_ENABLE_ALL;
        CHECK(make_pso(root1.p, all, pso_a));
        CHECK(make_pso(root1.p, D3D12_COLOR_WRITE_ENABLE_GREEN | D3D12_COLOR_WRITE_ENABLE_ALPHA, pso_b));
        CHECK(make_pso(root2.p, all, pso_a2));

        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        auto buffer = buffer_desc(4096);
        CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                        IID_PPV_ARGS(&upload.p))));
        buffer = buffer_desc(target_px * 256);
        CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                        IID_PPV_ARGS(&readback.p))));
        D3D12_RESOURCE_DESC texture{};
        texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = texture.Height = target_px;
        texture.DepthOrArraySize = texture.MipLevels = 1;
        texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture.SampleDesc.Count = 1;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&target.p))));
        const Vertex quad[4] = {{-1, 1, 255, 0, 0, 255}, {0, 1, 255, 0, 0, 255}, {-1, 0, 255, 0, 0, 255},
                                {0, 0, 255, 0, 0, 255}};
        const Vertex other[4] = {{0, 0, 255, 0, 0, 255}, {1, 0, 255, 0, 0, 255}, {0, -1, 255, 0, 0, 255},
                                 {1, -1, 255, 0, 0, 255}};
        const std::uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
        const float tints[3][3] = {{1, 0, 1}, {0, 1, 0}, {1, 1, 1}};
        std::uint8_t* up = nullptr;
        CHECK(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memset(up, 0, 4096);
        std::memcpy(up + vb_quad, quad, sizeof(quad));
        std::memcpy(up + vb_other, other, sizeof(other));
        std::memcpy(up + ib_quad, indices, sizeof(indices));
        const std::uint16_t half[3] = {3, 2, 1};
        std::memcpy(up + ib_half, half, sizeof(half));
        std::memcpy(up + tint_red, tints[0], sizeof(tints[0]));
        std::memcpy(up + tint_green, tints[1], sizeof(tints[1]));
        std::memcpy(up + tint_yellow, tints[2], sizeof(tints[2]));
        upload->Unmap(0, nullptr);
        upload_va = upload->GetGPUVirtualAddress();

        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(target.p, nullptr, rtv);
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        return true;
    }

    // A closed bundle created with `initial` as its pipeline state, kept alive until the rig goes.
    template<class F>
    ID3D12GraphicsCommandList* bundle(ID3D12PipelineState* initial, F&& record) {
        bundle_allocators.emplace_back();
        bundles.emplace_back();
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE,
                                                  IID_PPV_ARGS(&bundle_allocators.back().p))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, bundle_allocators.back().p, initial,
                                             IID_PPV_ARGS(&bundles.back().p))))
            return nullptr;
        record(bundles.back().p);
        if (FAILED(bundles.back()->Close())) return nullptr;
        return bundles.back().p;
    }

    // ExecuteBundle, unless the bundle's Close failed (reported in the output, the layer refuses such bundles).
    void execute(ID3D12GraphicsCommandList* list, ID3D12GraphicsCommandList* b) {
        if (b)
            list->ExecuteBundle(b);
        else
            bundle_failed = true;
    }

    // The parent's own targets, viewport and scissor: a bundle may not set them.
    void targets(ID3D12GraphicsCommandList* list, float x = 0, float y = 0, float w = target_px, float h = target_px) {
        const D3D12_VIEWPORT viewport{x, y, w, h, 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(target_px), LONG(target_px)};
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
    }
    void vertices(ID3D12GraphicsCommandList* list, UINT64 offset) {
        const D3D12_VERTEX_BUFFER_VIEW view{upload_va + offset, 4 * sizeof(Vertex), sizeof(Vertex)};
        list->IASetVertexBuffers(0, 1, &view);
    }
    void indices(ID3D12GraphicsCommandList* list, UINT64 offset) {
        const D3D12_INDEX_BUFFER_VIEW view{upload_va + offset, 12, DXGI_FORMAT_R16_UINT};
        list->IASetIndexBuffer(&view);
    }
    void offset(ID3D12GraphicsCommandList* list, float x, float y) {
        const float xy[2] = {x, y};
        list->SetGraphicsRoot32BitConstants(0, 2, xy, 0);
    }
    void tint(ID3D12GraphicsCommandList* list, UINT64 which) {
        list->SetGraphicsRootConstantBufferView(1, upload_va + which);
    }

    // Clears the target blue, runs `record` on the parent list and reads the target back as letters.
    bool render(const Recorder& record, std::string& out, HRESULT& close) {
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                  IID_PPV_ARGS(&list.p))));
        D3D12_RESOURCE_BARRIER to_target = transition(target.p, D3D12_RESOURCE_STATE_COMMON,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &to_target);
        const FLOAT blue[4] = {0, 0, 1, 1};
        list->ClearRenderTargetView(rtv, blue, 0, nullptr);
        record(*this, list.p);
        D3D12_RESOURCE_BARRIER to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, target_px, target_px, 1, 256}};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        close = list->Close();
        if (FAILED(close)) return true;  // reported by the caller
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        CHECK(SUCCEEDED(queue->Signal(fence.p, ++fence_value)));
        CHECK(SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
        const std::uint8_t* pixels = nullptr;
        const HRESULT mapped = readback->Map(0, nullptr, const_cast<void**>(reinterpret_cast<const void* const*>(&pixels)));
        if (FAILED(mapped)) {
            char text[96];
            std::snprintf(text, sizeof(text), "Map 0x%08lx, device removed 0x%08lx\n", static_cast<unsigned long>(mapped),
                          static_cast<unsigned long>(device->GetDeviceRemovedReason()));
            out = text;
            return true;
        }
        out = bundle_failed ? "bundle Close failed " : "";
        for (UINT row = 0; row < grid_cells; ++row) {
            for (UINT column = 0; column < grid_cells; ++column) {
                // Away from pixel centres on the quadrant diagonal (x + y = 32).
                const std::uint8_t* p = pixels + (row * 8 + 3) * 256 + (column * 8 + 3) * 4;
                char letter = '?';
                if (p[3] == 255) {
                    if (p[0] == 0 && p[1] == 0 && p[2] == 255) letter = 'B';
                    else if (p[0] == 255 && p[1] == 0 && p[2] == 0) letter = 'R';
                    else if (p[0] == 0 && p[1] == 255 && p[2] == 0) letter = 'G';
                    else if (p[0] == 255 && p[1] == 255 && p[2] == 0) letter = 'Y';
                    else if (p[0] == 0 && p[1] == 255 && p[2] == 255) letter = 'C';
                    else if (p[0] == 0 && p[1] == 0 && p[2] == 0) letter = 'K';
                }
                out += letter;
            }
            if (row + 1 < grid_cells) out += ' ';
        }
        readback->Unmap(0, nullptr);
        return true;
    }
};

struct Scenario {
    const char* name;
    Recorder record;
    const char* expected;  // null: print only
    bool warp_oracle_only = false;  // native hardware drivers may disagree with WARP about zero-count scissors
};

constexpr auto strip = D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
constexpr auto tlist = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

std::vector<Scenario> scenarios() {
    std::vector<Scenario> s;
    // The parent's state a bundle must see: targets, viewport and scissor, root signature and arguments, a vertex
    // buffer and a topology. No pipeline state is set: a bundle starts with its creation one.
    const auto parent_state = [](Rig& r, ID3D12GraphicsCommandList* l, D3D_PRIMITIVE_TOPOLOGY topology) {
        r.targets(l);
        l->SetGraphicsRootSignature(r.root1.p);
        l->IASetPrimitiveTopology(topology);
        r.vertices(l, vb_quad);
        r.offset(l, 0, 0);
        r.tint(l, tint_red);
    };
    s.push_back({"inherit: a draw sees the parent's targets, viewport, root arguments and vertex buffer",
                 [parent_state](Rig& r, ID3D12GraphicsCommandList* l) {
                     auto* b = r.bundle(r.pso_a.p, [](ID3D12GraphicsCommandList* b) {
                         b->IASetPrimitiveTopology(strip);
                         b->DrawInstanced(4, 1, 0, 0);
                     });
                     parent_state(r, l, tlist);
                     r.execute(l, b);
                 }, ROWS4("RRRRBBBB") " " BLANK4});
    s.push_back({"inherit: an indexed draw sees the parent's index buffer",
                 [parent_state](Rig& r, ID3D12GraphicsCommandList* l) {
                     auto* b = r.bundle(r.pso_a.p, [](ID3D12GraphicsCommandList* b) {
                         b->IASetPrimitiveTopology(tlist);
                         b->DrawIndexedInstanced(6, 1, 0, 0, 0);
                     });
                     parent_state(r, l, strip);
                     r.indices(l, ib_quad);
                     r.execute(l, b);
                 }, ROWS4("RRRRBBBB") " " BLANK4});
    s.push_back({"inherit: a bundle that sets the same root signature keeps the parent's root arguments",
                 [parent_state](Rig& r, ID3D12GraphicsCommandList* l) {
                     auto* b = r.bundle(r.pso_a.p, [&](ID3D12GraphicsCommandList* b) {
                         b->SetGraphicsRootSignature(r.root1.p);
                         b->IASetPrimitiveTopology(strip);
                         b->DrawInstanced(4, 1, 0, 0);
                     });
                     parent_state(r, l, tlist);
                     r.offset(l, 1, -1);
                     r.tint(l, tint_green);
                     r.execute(l, b);
                 }, BLANK4 " " ROWS4("BBBBGGGG")});
    s.push_back({"inherit: one bundle, executed twice with other root arguments and viewports in between",
                 [](Rig& r, ID3D12GraphicsCommandList* l) {
                     auto* b = r.bundle(r.pso_a.p, [](ID3D12GraphicsCommandList* b) {
                         b->IASetPrimitiveTopology(strip);
                         b->DrawInstanced(4, 1, 0, 0);
                     });
                     l->SetGraphicsRootSignature(r.root1.p);
                     r.vertices(l, vb_quad);
                     r.offset(l, 0, 0);
                     r.targets(l, 0, 0, target_px / 2, target_px);
                     r.tint(l, tint_red);
                     r.execute(l, b);
                     r.targets(l, target_px / 2, 0, target_px / 2, target_px);
                     r.tint(l, tint_green);
                     r.execute(l, b);
                 }, ROWS4("RRBBGGBB") " " BLANK4});
    s.push_back({"pipeline state: a bundle starts with its creation one, not the parent's, and hands it back",
                 [parent_state](Rig& r, ID3D12GraphicsCommandList* l) {
                     auto* b = r.bundle(r.pso_a.p, [](ID3D12GraphicsCommandList* b) {
                         b->IASetPrimitiveTopology(strip);
                         b->DrawInstanced(4, 1, 0, 0);
                     });
                     parent_state(r, l, tlist);
                     l->SetPipelineState(r.pso_b.p);
                     r.tint(l, tint_green);
                     r.execute(l, b);
                     r.offset(l, 1, -1);
                     l->DrawInstanced(4, 1, 0, 0);
                 }, ROWS4("GGGGBBBB") " " ROWS4("BBBBGGGG")});
    s.push_back({"hand back: a pipeline state the bundle sets",
                 [parent_state](Rig& r, ID3D12GraphicsCommandList* l) {
                     auto* b = r.bundle(r.pso_a.p, [&](ID3D12GraphicsCommandList* b) {
                         b->SetPipelineState(r.pso_b.p);
                         b->IASetPrimitiveTopology(strip);
                     });
                     parent_state(r, l, tlist);
                     l->SetPipelineState(r.pso_a.p);
                     r.tint(l, tint_green);
                     r.execute(l, b);
                     l->DrawInstanced(4, 1, 0, 0);
                 }, ROWS4("CCCCBBBB") " " BLANK4});
    s.push_back({"hand back: topology",
                 [parent_state](Rig& r, ID3D12GraphicsCommandList* l) {
                     auto* b = r.bundle(r.pso_a.p, [](ID3D12GraphicsCommandList* b) { b->IASetPrimitiveTopology(strip); });
                     parent_state(r, l, tlist);
                     r.execute(l, b);
                     l->DrawInstanced(4, 1, 0, 0);
                 }, ROWS4("RRRRBBBB") " " BLANK4});
    s.push_back({"hand back: root constants and a root CBV, after the same root signature",
                 [parent_state](Rig& r, ID3D12GraphicsCommandList* l) {
                     auto* b = r.bundle(r.pso_a.p, [&](ID3D12GraphicsCommandList* b) {
                         b->SetGraphicsRootSignature(r.root1.p);
                         b->IASetPrimitiveTopology(strip);
                         r.offset(b, 1, -1);
                         r.tint(b, tint_green);
                     });
                     parent_state(r, l, tlist);
                     r.execute(l, b);
                     l->DrawInstanced(4, 1, 0, 0);
                 }, BLANK4 " " ROWS4("BBBBGGGG")});
    s.push_back({"hand back: vertex and index buffers",
                 [parent_state](Rig& r, ID3D12GraphicsCommandList* l) {
                     auto* b = r.bundle(r.pso_a.p, [&](ID3D12GraphicsCommandList* b) {
                         b->IASetPrimitiveTopology(tlist);
                         r.vertices(b, vb_other);
                         r.indices(b, ib_half);
                     });
                     parent_state(r, l, strip);
                     r.indices(l, ib_quad);
                     r.execute(l, b);
                     l->DrawIndexedInstanced(3, 1, 0, 0, 0);
                 }, BLANK4 " BBBBBBBB BBBBBBBR BBBBBBRR BBBBBRRR"});
    // A different root signature: the bundle (created with pso_a2, on root2) sets it and its own arguments. The
    // parent's arguments are gone afterwards and the bundle's are the parent's.
    s.push_back({"root signature: a bundle sets another one; the parent continues with it and the bundle's arguments",
                 [parent_state](Rig& r, ID3D12GraphicsCommandList* l) {
                     auto* b = r.bundle(r.pso_a2.p, [&](ID3D12GraphicsCommandList* b) {
                         b->SetGraphicsRootSignature(r.root2.p);
                         b->IASetPrimitiveTopology(strip);
                         r.offset(b, 0, 0);
                         r.tint(b, tint_yellow);
                         const UINT extra = 0;
                         b->SetGraphicsRoot32BitConstants(2, 1, &extra, 0);
                         b->DrawInstanced(4, 1, 0, 0);
                     });
                     parent_state(r, l, tlist);
                     r.offset(l, 1, -1);
                     r.tint(l, tint_green);
                     r.execute(l, b);
                     r.vertices(l, vb_other);
                     l->DrawInstanced(4, 1, 0, 0);
                 }, ROWS4("YYYYBBBB") " " ROWS4("BBBBYYYY")});
    s.push_back({"scissor zero: reset existing rectangles before the next draw",
                 [parent_state](Rig& r, ID3D12GraphicsCommandList* l) {
                     parent_state(r, l, strip);
                     l->SetPipelineState(r.pso_a.p);
                     l->RSSetScissorRects(0, nullptr);
                     l->DrawInstanced(4, 1, 0, 0);
                 }, BLANK4 " " BLANK4, true});
    s.push_back({"scissor zero then one: restores new rectangle for draw",
                 [parent_state](Rig& r, ID3D12GraphicsCommandList* l) {
                     parent_state(r, l, strip);
                     l->SetPipelineState(r.pso_a.p);
                     l->RSSetScissorRects(0, nullptr);
                     const D3D12_RECT full{0, 0, LONG(target_px), LONG(target_px)};
                     l->RSSetScissorRects(1, &full);
                     l->DrawInstanced(4, 1, 0, 0);
                 }, ROWS4("RRRRBBBB") " " BLANK4});
    s.push_back({"scissor zero with non-null pointer still disables all rectangles",
                 [parent_state](Rig& r, ID3D12GraphicsCommandList* l) {
                     parent_state(r, l, strip);
                     l->SetPipelineState(r.pso_a.p);
                     const D3D12_RECT unused{0, 0, LONG(target_px), LONG(target_px)};
                     l->RSSetScissorRects(0, &unused);
                     l->DrawInstanced(4, 1, 0, 0);
                 }, BLANK4 " " BLANK4, true});
    s.push_back({"scissor zero on parent suppresses inherited bundle draw",
                 [parent_state](Rig& r, ID3D12GraphicsCommandList* l) {
                     auto* b = r.bundle(r.pso_a.p, [](ID3D12GraphicsCommandList* b) {
                         b->IASetPrimitiveTopology(strip);
                         b->DrawInstanced(4, 1, 0, 0);
                     });
                     parent_state(r, l, strip);
                     l->RSSetScissorRects(0, nullptr);
                     r.execute(l, b);
                 }, BLANK4 " " BLANK4, true});
    return s;
}

}  // namespace

namespace {

bool compile_hlsl(const char* source, const char* entry, const char* profile, std::vector<std::uint8_t>& out) {
    const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile = compiler ? reinterpret_cast<pD3DCompile>(GetProcAddress(compiler, "D3DCompile")) : nullptr;
    if (!compile) return false;
    ID3DBlob* blob = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = compile(source, std::strlen(source), "bundle.hlsl", nullptr, nullptr, entry, profile, 0, 0,
                               &blob, &errors);
    if (FAILED(hr) && errors) std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
    if (errors) errors->Release();
    if (FAILED(hr) || !blob) return false;
    const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
    out.assign(p, p + blob->GetBufferSize());
    blob->Release();
    return true;
}

// The same two shaders as test/glsl/vs_draw.vert and ps_draw.frag: s0, s1 offset the vertex, the root CBV tints.
constexpr char hlsl[] =
    "cbuffer Offset : register(b0) { float2 offset; };\n"
    "cbuffer Tint : register(b1) { float3 tint; };\n"
    "struct V { float4 position : SV_Position; float4 color : COLOR; };\n"
    "V vs(float2 position : POSITION, float4 color : COLOR) {\n"
    "    V v; v.position = float4(position + offset, 0, 1); v.color = color; return v; }\n"
    "float4 ps(V v) : SV_Target { return float4(v.color.r * tint.x, tint.y, v.color.b * tint.z, 1); }\n";

// Runs every scenario and stores the grids; false on a harness failure. A host scenario gets a device of its own
// (`adapter`, null for the default one): a scenario that removes a host device must not hide the ones after it.
bool run_all(ID3D12Device* shared, bool own_device, IDXGIAdapter* adapter, const std::vector<std::uint8_t>& vs, const std::vector<std::uint8_t>& ps,
             const std::vector<Scenario>& list, std::vector<std::string>& grids, std::vector<HRESULT>& closes) {
    for (const Scenario& scenario : list) {
        Rig rig;
        Ref<ID3D12Device> own;
        if (own_device) D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&own.p));
        ID3D12Device* device = own_device ? own.p : shared;
        if (!device || !rig.init(device, vs, ps)) {
            grids.push_back("device unusable\n");
            closes.push_back(E_FAIL);
            continue;
        }
        std::string grid_text;
        HRESULT close = S_OK;
        if (!rig.render(scenario.record, grid_text, close)) return false;
        grids.push_back(grid_text);
        closes.push_back(close);
    }
    return true;
}

}  // namespace

// What the layer alone can show: the order of commands in the parent, the trace, and object lifetimes.
int count_of(const std::string& text, const char* needle) {
    int count = 0;
    for (std::size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++count;
    return count;
}

// Sends stderr (where the trace goes) to a file while it lives, with the trace on.
struct TraceCapture {
    std::filesystem::path path;
    int saved = -1;
    explicit TraceCapture(const std::filesystem::path& file) : path(file) {
        std::fflush(stderr);
        std::FILE* out = std::fopen(path.string().c_str(), "wb");
        saved = _dup(2);
        if (out && saved >= 0) _dup2(_fileno(out), 2);
        if (out) std::fclose(out);
        rt::d3d12_trace::set_enabled(true);
    }
    std::string finish() {
        rt::d3d12_trace::set_enabled(false);
        std::fflush(stderr);
        if (saved >= 0) {
            _dup2(saved, 2);
            _close(saved);
            saved = -1;
        }
        std::ifstream in(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }
    ~TraceCapture() { finish(); }
};

bool layer_checks(ID3D12Device* device, const std::vector<std::uint8_t>& vs, const std::vector<std::uint8_t>& ps,
                  const std::filesystem::path& scratch) {
    constexpr UINT8 all = D3D12_COLOR_WRITE_ENABLE_ALL;
    // The parent's commands hold the bundle's markers and draws where the bundle made them, between the parent's own.
    {
        Rig r;
        CHECK(r.init(device, vs, ps));
        const auto mark = [](ID3D12GraphicsCommandList* l, const char* text) {
            l->SetMarker(1, text, UINT(std::strlen(text) + 1));
        };
        auto* b = r.bundle(r.pso_a.p, [&](ID3D12GraphicsCommandList* b) {
            mark(b, "b1");
            b->IASetPrimitiveTopology(strip);
            b->DrawInstanced(4, 1, 0, 0);
            mark(b, "b2");
            b->DrawInstanced(4, 1, 0, 0);
            mark(b, "b3");
        });
        CHECK(b);
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> parent;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                  IID_PPV_ARGS(&parent.p))));
        mark(parent.p, "p0");
        r.targets(parent.p);
        parent.p->SetGraphicsRootSignature(r.root1.p);
        r.vertices(parent.p, vb_quad);
        r.offset(parent.p, 0, 0);
        r.tint(parent.p, tint_red);
        mark(parent.p, "p1");
        parent.p->ExecuteBundle(b);
        mark(parent.p, "p2");
        std::string order;
        std::vector<leonida::Command> recorded;
        CHECK(rt::d3d12::recorded_commands(parent.p, recorded));
        for (const auto& command : recorded) {
            if (const auto* marker = std::get_if<leonida::DebugMarker>(&command)) order += marker->label + ",";
            else if (std::holds_alternative<leonida::Draw>(command)) order += "draw,";
        }
        if (order != "p0,p1,b1,draw,b2,draw,b3,p2,") {
            std::fprintf(stderr, "bundle commands in the parent: %s\n", order.c_str());
            return false;
        }
        CHECK(parent->Close() == S_OK);
    }

    // The trace has one line per call the application made on the bundle, however often it is executed, and none for
    // the pipeline state it was created or reset with.
    {
        Rig r;
        CHECK(r.init(device, vs, ps));
        TraceCapture capture(scratch / "trace.txt");
        auto* b = r.bundle(r.pso_a.p, [&](ID3D12GraphicsCommandList* b) {
            b->SetGraphicsRootSignature(r.root1.p);
            b->IASetPrimitiveTopology(strip);
            b->DrawInstanced(4, 1, 0, 0);
        });
        CHECK(b);
        std::string grid_text;
        HRESULT close = S_OK;
        CHECK(r.render([&](Rig& rig, ID3D12GraphicsCommandList* l) {
            rig.targets(l);
            rig.vertices(l, vb_quad);
            for (int i = 0; i < 3; ++i) rig.execute(l, b);
        }, grid_text, close));
        CHECK(close == S_OK);
        CHECK(SUCCEEDED(b->Reset(r.bundle_allocators.back().p, r.pso_a.p)));
        b->SetGraphicsRootSignature(r.root1.p);
        r.offset(b, 0, 0);
        r.tint(b, tint_red);
        b->IASetPrimitiveTopology(tlist);
        b->DrawIndexedInstanced(6, 1, 0, 0, 0);
        CHECK(SUCCEEDED(b->Close()));
        CHECK(r.render([&](Rig& rig, ID3D12GraphicsCommandList* l) {
            rig.targets(l);
            rig.vertices(l, vb_quad);
            rig.indices(l, ib_quad);
            rig.execute(l, b);
            rig.execute(l, b);
        }, grid_text, close));
        CHECK(close == S_OK && grid_text == ROWS4("RRRRBBBB") " " BLANK4);
        const std::string trace = capture.finish();
        const int pipeline = count_of(trace, "GraphicsCommandList::SetPipelineState called");
        const int root = count_of(trace, "GraphicsCommandList::SetGraphicsRootSignature called");
        const int draw = count_of(trace, "GraphicsCommandList::DrawInstanced called");
        const int indexed = count_of(trace, "GraphicsCommandList::DrawIndexedInstanced called");
        const int execute = count_of(trace, "GraphicsCommandList::ExecuteBundle called");
        if (pipeline != 0 || root != 2 || draw != 1 || indexed != 1 || execute != 5) {
            std::fprintf(stderr, "trace lines: SetPipelineState %d, root signature %d, draw %d, indexed %d, "
                         "ExecuteBundle %d (expected 0, 2, 1, 1, 5)\n", pipeline, root, draw, indexed, execute);
            return false;
        }
    }

    // A root-constant call of zero dwords is as valid in a bundle as on a direct list; a closed bundle takes no more
    // state calls (they are ignored) and a draw on it makes it invalid, so a parent that executes it fails.
    {
        Rig r;
        CHECK(r.init(device, vs, ps));
        const UINT none = 0;
        auto* b = r.bundle(r.pso_a.p, [&](ID3D12GraphicsCommandList* b) {
            b->SetGraphicsRootSignature(r.root1.p);
            b->SetGraphicsRoot32BitConstants(0, 0, &none, 0);
            b->IASetPrimitiveTopology(strip);
            r.offset(b, 0, 0);
            r.tint(b, tint_red);
            b->DrawInstanced(4, 1, 0, 0);
        });
        CHECK(b);
        std::string grid_text;
        HRESULT close = S_OK;
        const auto run = [&](ID3D12GraphicsCommandList* bundle) {
            return r.render([&](Rig& rig, ID3D12GraphicsCommandList* l) {
                rig.targets(l);
                rig.vertices(l, vb_quad);
                rig.execute(l, bundle);
            }, grid_text, close);
        };
        CHECK(run(b));
        CHECK(close == S_OK && grid_text == ROWS4("RRRRBBBB") " " BLANK4);
        b->IASetPrimitiveTopology(tlist);  // closed: ignored, so the bundle still draws the whole quad
        CHECK(run(b));
        CHECK(close == S_OK && grid_text == ROWS4("RRRRBBBB") " " BLANK4);
        b->DrawInstanced(4, 1, 0, 0);  // closed: invalid
        CHECK(run(b));
        CHECK(close == E_INVALIDARG);
    }

    // The application may release the pipeline state and root signature as soon as it has recorded them, and the
    // bundle itself while a list that executed it is still open.
    {
        Rig r;
        CHECK(r.init(device, vs, ps));
        ID3D12GraphicsCommandList* b = nullptr;
        {
            Ref<ID3D12RootSignature> root;
            Ref<ID3D12PipelineState> pso;
            CHECK(r.make_root(2, root) && r.make_pso(root.p, all, pso));
            b = r.bundle(pso.p, [&](ID3D12GraphicsCommandList* b) {
                b->SetGraphicsRootSignature(root.p);
                b->IASetPrimitiveTopology(strip);
                r.offset(b, 0, 0);
                r.tint(b, tint_red);
                b->DrawInstanced(4, 1, 0, 0);
            });
        }
        CHECK(b);
        std::string grid_text;
        HRESULT close = S_OK;
        bool reset_ok = false;
        CHECK(r.render([&](Rig& rig, ID3D12GraphicsCommandList* l) {
            rig.targets(l);
            rig.vertices(l, vb_quad);
            rig.execute(l, b);
        }, grid_text, close));
        if (close != S_OK || grid_text != ROWS4("RRRRBBBB") " " BLANK4) {
            std::fprintf(stderr, "first parent of a released bundle: Close 0x%08lx, %s\n",
                         static_cast<unsigned long>(close), grid_text.c_str());
            return false;
        }
        CHECK(r.render([&](Rig& rig, ID3D12GraphicsCommandList* l) {
            rig.targets(l);
            rig.vertices(l, vb_quad);
            rig.execute(l, b);
            reset_ok = SUCCEEDED(rig.bundles.back()->Reset(rig.bundle_allocators.back().p, nullptr));
            rig.bundles.clear();
            rig.bundle_allocators.clear();
            rig.offset(l, 1, -1);  // the bundle's root signature and pipeline state are the parent's now
            l->DrawInstanced(4, 1, 0, 0);
        }, grid_text, close));
        if (!reset_ok || close != S_OK || grid_text != ROWS4("RRRRBBBB") " " ROWS4("BBBBRRRR")) {
            std::fprintf(stderr, "second parent, bundle released while open: Close 0x%08lx, %s\n",
                         static_cast<unsigned long>(close), grid_text.c_str());
            return false;
        }
    }
    return true;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    REQUIRE(device_status == vktest::DeviceStatus::ready);
    const auto list = scenarios();

    // Layer shaders: FSSM-shaped containers found by hash in a translator-style index.
    const auto vs_program = fssm_program({0x56530101, 0xbf810000});
    const auto ps_program = fssm_program({0x50530101, 0xbf810000});
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    const testfiles::ScratchDir scratch("leonida_bundle_test");  // removed on every way out of main
    const auto& library = scratch.path();
    REQUIRE(testfiles::copy_file(LEONIDA_TEST_VS_SPV, library / "vs_draw.spv"));
    REQUIRE(testfiles::copy_file(LEONIDA_TEST_PS_SPV, library / "ps_draw.spv"));
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
    const unsigned errors_before = vk.validation_errors();
    std::vector<std::string> layer_grids;
    std::vector<HRESULT> layer_closes;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = true;
        info.shader_dir = library.string();
        Ref<ID3D12Device> ours;
        REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p))));
        REQUIRE(run_all(ours.p, false, nullptr, vs_container, ps_container, list, layer_grids, layer_closes));
        REQUIRE(layer_checks(ours.p, vs_container, ps_container, library));
    }

    // The host: Windows' D3D12 on the default adapter and on WARP.
    std::vector<std::uint8_t> host_vs, host_ps;
    const bool wine = hostwine::host_is_wine();
    const bool host_shaders = !wine && compile_hlsl(hlsl, "vs", "vs_5_0", host_vs) &&
                              compile_hlsl(hlsl, "ps", "ps_5_0", host_ps);
    struct HostRun {
        const char* name = "";
        bool ran = false;
        std::vector<std::string> grids;
        std::vector<HRESULT> closes;
    } hosts[2];
    hosts[0].name = "the default adapter";
    hosts[1].name = "WARP";
    if (host_shaders) {
        Ref<IDXGIFactory4> factory;
        REQUIRE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))));
        for (int i = 0; i < 2; ++i) {
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            if (i == 1 && FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            REQUIRE(run_all(nullptr, true, adapter.p, host_vs, host_ps, list, hosts[i].grids, hosts[i].closes));
            hosts[i].ran = true;
        }
    } else if (!wine) {
        std::fprintf(stderr, "no HLSL compiler: the host is not run\n");
    }

    // WARP is the reference for zero-count scissors. An AMD Windows D3D12 driver (RX 6700 XT, 26.11.2) keeps the
    // old scissor after count zero, while WARP and other tested drivers disable it. Record that native divergence
    // without failing the layer or WARP correctness test; all other default-adapter cases remain strict.
    unsigned failures = 0, compared = 0;
    const auto verdict = [&](const char* who, const Scenario& scenario, const std::string& grid, HRESULT close) {
        const bool ok = grid == scenario.expected && close == S_OK;
        if (!ok) {
            std::fprintf(stderr, "FAIL %s on %s: Close 0x%08lx\n  got      %s\n  expected %s\n", scenario.name, who,
                         static_cast<unsigned long>(close), grid.c_str(), scenario.expected);
            ++failures;
        }
    };
    for (std::size_t i = 0; i < list.size(); ++i) {
        if (!list[i].expected) {
            std::printf("%s | layer %s", list[i].name, layer_grids[i].c_str());
            for (const auto& host : hosts)
                if (host.ran) std::printf(" | %s %s", host.name, host.grids[i].c_str());
            std::printf("\n");
            ++failures;
            continue;
        }
        verdict("the layer", list[i], layer_grids[i], layer_closes[i]);
        for (const auto& host : hosts) {
            if (!host.ran) continue;
            if (&host == &hosts[0] && list[i].warp_oracle_only) {
                if (host.closes[i] != S_OK) {
                    verdict(host.name, list[i], host.grids[i], host.closes[i]);
                } else if (host.grids[i] != list[i].expected) {
                    std::printf("NOTE %s on %s differs from WARP: %s\n", list[i].name, host.name,
                                host.grids[i].c_str());
                }
                continue;
            }
            verdict(host.name, list[i], host.grids[i], host.closes[i]);
            ++compared;
        }
    }
    const unsigned validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", validation_errors);
        ++failures;
    }
    if (!hosts[0].ran && !hosts[1].ran && !hostwine::host_is_wine())
        std::fprintf(stderr, "no D3D12 reference device (default adapter or WARP)\n");
    std::printf("bundles: %zu scenarios on the layer, %u host comparisons, %u failures\n", list.size(), compared,
                failures);
    return failures ? 1 : 0;
}
