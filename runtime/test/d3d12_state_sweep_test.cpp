// Fixed-function pipeline state swept against the host's D3D12 on the same adapter (matched by LUID, the D3D-188
// rule): each case is one full-screen draw of a constant colour into a 4x4 RGBA target (and a D32_FLOAT_S8X24_UINT
// depth-stencil target) on the layer and on the host, and the bytes read back must be the same.
//   - blend: every source x destination colour factor (the SRC1 ones need dual-source output and are left out), every
//     blend operation, per-channel write masks, the blend factor, and the same on an sRGB target (blending in linear
//     space) and a float one;
//   - depth: the eight comparison functions against two clear values, with and without depth writes;
//   - stencil: the eight comparison functions against three stored values, every operation (fail, pass), read and
//     write masks, and the back face's own state through a flipped front face.
// The target and depth start from known clears; stencil results are read from the stencil plane with CopyTextureRegion.
// The layer's shaders are GLSL stand-ins for translated ones (test/glsl/vs_fullscreen.vert, ps_sweep.frag), found by
// the FSSM hash of their containers in a translator-style index; the host's are HLSL compiled by d3dcompiler_47.
// Synthetic data only.
#include "vk_test_device.h"
#include "host_wine.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

struct Programs {
    std::vector<std::uint8_t> root, vs, ps;
};

struct Case {
    std::string name;
    DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
    std::function<void(D3D12_GRAPHICS_PIPELINE_STATE_DESC&)> state;
    float clear_color[4] = {0.2f, 0.4f, 0.6f, 0.8f};
    float clear_depth = 0.5f;
    UINT8 clear_stencil = 0;
    UINT8 stencil_ref = 0;
    float blend_factor[4] = {0.3f, 0.6f, 0.9f, 0.1f};
};

constexpr DXGI_FORMAT kDepthFormat = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;

void base_state(D3D12_GRAPHICS_PIPELINE_STATE_DESC& pd, const Case& c, ID3D12RootSignature* root, const Programs& p) {
    pd = {};
    pd.pRootSignature = root;
    pd.VS = {p.vs.data(), p.vs.size()};
    pd.PS = {p.ps.data(), p.ps.size()};
    for (auto& rt : pd.BlendState.RenderTarget) {
        rt.SrcBlend = rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlend = rt.DestBlendAlpha = D3D12_BLEND_ZERO;
        rt.BlendOp = rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    }
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.DepthStencilState.DepthEnable = FALSE;
    pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    pd.DepthStencilState.StencilReadMask = pd.DepthStencilState.StencilWriteMask = 0xff;
    const D3D12_DEPTH_STENCILOP_DESC keep{D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                          D3D12_COMPARISON_FUNC_ALWAYS};
    pd.DepthStencilState.FrontFace = pd.DepthStencilState.BackFace = keep;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = c.format;
    pd.DSVFormat = kDepthFormat;
    pd.SampleDesc = {1, 0};
    if (c.state) c.state(pd);
}

struct Result {
    bool ok = false;
    HRESULT pso = S_OK;
    std::vector<unsigned char> color, stencil;
};

Result run(ID3D12Device* device, const Case& c, const Programs& p) {
    Result r;
    const D3D12_HEAP_PROPERTIES def_heap{D3D12_HEAP_TYPE_DEFAULT}, rb_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12RootSignature> root;
    if (FAILED(device->CreateRootSignature(0, p.root.data(), p.root.size(), IID_PPV_ARGS(&root.p)))) return r;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd;
    base_state(pd, c, root.p, p);
    Ref<ID3D12PipelineState> pso;
    r.pso = device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p));
    if (FAILED(r.pso)) return r;

    D3D12_RESOURCE_DESC cd{};
    cd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    cd.Width = cd.Height = 4;
    cd.DepthOrArraySize = cd.MipLevels = 1;
    cd.Format = c.format;
    cd.SampleDesc = {1, 0};
    cd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_RESOURCE_DESC dd = cd;
    dd.Format = kDepthFormat;
    dd.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    Ref<ID3D12Resource> color, depth;
    if (FAILED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &cd, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                               nullptr, IID_PPV_ARGS(&color.p))) ||
        FAILED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &dd, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                               nullptr, IID_PPV_ARGS(&depth.p))))
        return r;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT cfoot{}, sfoot{};
    UINT crows = 0, srows = 0;
    UINT64 crow_bytes = 0, srow_bytes = 0, ctotal = 0, stotal = 0;
    device->GetCopyableFootprints(&cd, 0, 1, 0, &cfoot, &crows, &crow_bytes, &ctotal);
    device->GetCopyableFootprints(&dd, 1, 1, 0, &sfoot, &srows, &srow_bytes, &stotal);
    Ref<ID3D12Resource> cread, sread;
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Height = bd.DepthOrArraySize = bd.MipLevels = 1;
    bd.SampleDesc = {1, 0};
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bd.Width = ctotal;
    if (FAILED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&cread.p))))
        return r;
    bd.Width = stotal;
    if (FAILED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&sread.p))))
        return r;
    Ref<ID3D12DescriptorHeap> rtv_heap, dsv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC rh{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, dh{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1};
    if (FAILED(device->CreateDescriptorHeap(&rh, IID_PPV_ARGS(&rtv_heap.p))) ||
        FAILED(device->CreateDescriptorHeap(&dh, IID_PPV_ARGS(&dsv_heap.p))))
        return r;
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    const auto dsv = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_RENDER_TARGET_VIEW_DESC rv{};
    rv.Format = c.format;
    rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    device->CreateRenderTargetView(color.p, &rv, rtv);
    D3D12_DEPTH_STENCIL_VIEW_DESC dv{};
    dv.Format = kDepthFormat;
    dv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device->CreateDepthStencilView(depth.p, &dv, dsv);

    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return r;
    list->ClearRenderTargetView(rtv, c.clear_color, 0, nullptr);
    list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, c.clear_depth, c.clear_stencil, 0,
                                nullptr);
    const D3D12_VIEWPORT viewport{0, 0, 4, 4, 0, 1};
    const D3D12_RECT scissor{0, 0, 4, 4};
    list->SetGraphicsRootSignature(root.p);
    list->SetPipelineState(pso.p);
    list->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
    list->OMSetStencilRef(c.stencil_ref);
    list->OMSetBlendFactor(c.blend_factor);
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    D3D12_RESOURCE_BARRIER barriers[2]{};
    barriers[0].Transition = {color.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET,
                              D3D12_RESOURCE_STATE_COPY_SOURCE};
    barriers[1].Transition = {depth.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                              D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(2, barriers);
    D3D12_TEXTURE_COPY_LOCATION cdst{cread.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    cdst.PlacedFootprint = cfoot;
    D3D12_TEXTURE_COPY_LOCATION csrc{color.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&cdst, 0, 0, 0, &csrc, nullptr);
    D3D12_TEXTURE_COPY_LOCATION sdst{sread.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    sdst.PlacedFootprint = sfoot;
    D3D12_TEXTURE_COPY_LOCATION ssrc{depth.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    ssrc.SubresourceIndex = 1;  // the stencil plane of a D32_FLOAT_S8X24_UINT texture
    list->CopyTextureRegion(&sdst, 0, 0, 0, &ssrc, nullptr);
    if (FAILED(list->Close())) return r;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1)) || FAILED(fence->SetEventOnCompletion(1, nullptr))) return r;
    const auto read = [](ID3D12Resource* buffer, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& foot, UINT rows,
                         UINT64 row_bytes, std::vector<unsigned char>& out) {
        void* mapped = nullptr;
        if (FAILED(buffer->Map(0, nullptr, &mapped))) return false;
        for (UINT y = 0; y < rows; ++y) {
            const auto* row = static_cast<const unsigned char*>(mapped) + foot.Offset + y * foot.Footprint.RowPitch;
            out.insert(out.end(), row, row + row_bytes);
        }
        const D3D12_RANGE none{};
        buffer->Unmap(0, &none);
        return true;
    };
    r.ok = read(cread.p, cfoot, crows, crow_bytes, r.color) && read(sread.p, sfoot, srows, srow_bytes, r.stencil);
    return r;
}

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

std::string hex(const std::vector<unsigned char>& bytes, std::size_t count) {
    std::string s;
    char b[4];
    for (std::size_t i = 0; i < count && i < bytes.size(); ++i) {
        std::snprintf(b, sizeof(b), "%02x", bytes[i]);
        s += b;
    }
    return s;
}

constexpr char kHostVs[] =
    "float4 main(uint id : SV_VertexID) : SV_Position {\n"
    "    float2 corner = float2(id & 2, (id << 1) & 2);\n"
    "    return float4(corner * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";
constexpr char kHostPs[] = "float4 main() : SV_Target { return float4(0.75, 0.5, 0.25, 0.5); }\n";

bool compile_hlsl(pD3DCompile compile, const char* source, std::vector<std::uint8_t>& out, bool vertex) {
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = compile(source, std::strlen(source), nullptr, nullptr, nullptr, "main", vertex ? "vs_5_0" : "ps_5_0",
                               0, 0, &code, &errors);
    if (errors) {
        std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
        errors->Release();
    }
    if (FAILED(hr) || !code) return false;
    const auto* bytes = static_cast<const std::uint8_t*>(code->GetBufferPointer());
    out.assign(bytes, bytes + code->GetBufferSize());
    code->Release();
    return true;
}

const char* kBlendNames[] = {"", "ZERO", "ONE", "SRC_COLOR", "INV_SRC_COLOR", "SRC_ALPHA", "INV_SRC_ALPHA", "DEST_ALPHA",
                             "INV_DEST_ALPHA", "DEST_COLOR", "INV_DEST_COLOR", "SRC_ALPHA_SAT", "", "", "BLEND_FACTOR",
                             "INV_BLEND_FACTOR"};
const char* kFuncNames[] = {"", "NEVER", "LESS", "EQUAL", "LESS_EQUAL", "GREATER", "NOT_EQUAL", "GREATER_EQUAL", "ALWAYS"};
const char* kOpNames[] = {"", "KEEP", "ZERO", "REPLACE", "INCR_SAT", "DECR_SAT", "INVERT", "INCR", "DECR"};

// Blend factors usable for the colour channels without dual-source output (11 = SRC_ALPHA_SAT, 14/15 the blend factor).
const D3D12_BLEND kColourFactors[] = {
    D3D12_BLEND_ZERO, D3D12_BLEND_ONE, D3D12_BLEND_SRC_COLOR, D3D12_BLEND_INV_SRC_COLOR, D3D12_BLEND_SRC_ALPHA,
    D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_DEST_ALPHA, D3D12_BLEND_INV_DEST_ALPHA, D3D12_BLEND_DEST_COLOR,
    D3D12_BLEND_INV_DEST_COLOR, D3D12_BLEND_SRC_ALPHA_SAT, D3D12_BLEND_BLEND_FACTOR, D3D12_BLEND_INV_BLEND_FACTOR};
// The factors an alpha channel takes: no *_COLOR ones.
const D3D12_BLEND kAlphaFactors[] = {
    D3D12_BLEND_ZERO, D3D12_BLEND_ONE, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_DEST_ALPHA,
    D3D12_BLEND_INV_DEST_ALPHA, D3D12_BLEND_SRC_ALPHA_SAT, D3D12_BLEND_BLEND_FACTOR, D3D12_BLEND_INV_BLEND_FACTOR};

std::vector<Case> build_cases() {
    std::vector<Case> cases;
    const auto blend = [&](const std::string& name, DXGI_FORMAT format, D3D12_BLEND sc, D3D12_BLEND dc, D3D12_BLEND_OP oc,
                           D3D12_BLEND sa, D3D12_BLEND da, D3D12_BLEND_OP oa, UINT8 mask) {
        Case c;
        c.name = name;
        c.format = format;
        c.state = [=](D3D12_GRAPHICS_PIPELINE_STATE_DESC& pd) {
            auto& rt = pd.BlendState.RenderTarget[0];
            rt.BlendEnable = TRUE;
            rt.SrcBlend = sc;
            rt.DestBlend = dc;
            rt.BlendOp = oc;
            rt.SrcBlendAlpha = sa;
            rt.DestBlendAlpha = da;
            rt.BlendOpAlpha = oa;
            rt.RenderTargetWriteMask = mask;
        };
        cases.push_back(std::move(c));
    };
    for (D3D12_BLEND s : kColourFactors)
        for (D3D12_BLEND d : kColourFactors)
            blend(std::string("blend ") + kBlendNames[s] + " / " + kBlendNames[d], DXGI_FORMAT_R8G8B8A8_UNORM, s, d,
                  D3D12_BLEND_OP_ADD, D3D12_BLEND_ONE, D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD, 0xf);
    for (D3D12_BLEND s : kAlphaFactors)
        for (D3D12_BLEND d : kAlphaFactors)
            blend(std::string("alpha blend ") + kBlendNames[s] + " / " + kBlendNames[d], DXGI_FORMAT_R8G8B8A8_UNORM,
                  D3D12_BLEND_ONE, D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD, s, d, D3D12_BLEND_OP_ADD, 0xf);
    const char* op_names[] = {"", "ADD", "SUBTRACT", "REV_SUBTRACT", "MIN", "MAX"};
    for (int oc = 1; oc <= 5; ++oc)
        for (int oa = 1; oa <= 5; ++oa)
            blend(std::string("blend op ") + op_names[oc] + " / alpha " + op_names[oa], DXGI_FORMAT_R8G8B8A8_UNORM,
                  D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_OP(oc), D3D12_BLEND_SRC_ALPHA,
                  D3D12_BLEND_DEST_ALPHA, D3D12_BLEND_OP(oa), 0xf);
    for (UINT8 mask = 0; mask < 16; ++mask) {
        blend("write mask " + std::to_string(mask) + " with blend", DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_BLEND_SRC_ALPHA,
              D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_OP_ADD, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, mask);
        Case c;
        c.name = "write mask " + std::to_string(mask) + " without blend";
        c.state = [=](D3D12_GRAPHICS_PIPELINE_STATE_DESC& pd) { pd.BlendState.RenderTarget[0].RenderTargetWriteMask = mask; };
        cases.push_back(std::move(c));
    }
    for (DXGI_FORMAT f : {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_R16G16B16A16_FLOAT,
                          DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R16G16B16A16_UNORM, DXGI_FORMAT_R16G16B16A16_SNORM,
                          DXGI_FORMAT_R8G8B8A8_SNORM, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R8_UNORM}) {
        const std::string tag = "format " + std::to_string(int(f)) + " ";
        blend(tag + "SRC_ALPHA / INV_SRC_ALPHA", f, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_OP_ADD,
              D3D12_BLEND_ONE, D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD, 0xf);
        blend(tag + "ONE / ONE", f, D3D12_BLEND_ONE, D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD, D3D12_BLEND_ONE, D3D12_BLEND_ONE,
              D3D12_BLEND_OP_ADD, 0xf);
        blend(tag + "DEST_COLOR / SRC_COLOR", f, D3D12_BLEND_DEST_COLOR, D3D12_BLEND_SRC_COLOR, D3D12_BLEND_OP_ADD,
              D3D12_BLEND_DEST_ALPHA, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_OP_ADD, 0xf);
        Case c;
        c.name = tag + "no blend";
        c.format = f;
        cases.push_back(std::move(c));
    }
    // Depth: the draw's depth is 0; the clear is 0 (EQUAL-ish) or 0.5.
    for (int func = 1; func <= 8; ++func)
        for (float clear : {0.0f, 0.5f})
            for (bool write : {false, true}) {
                Case c;
                c.name = std::string("depth ") + kFuncNames[func] + " clear " + (clear == 0.0f ? "0" : "0.5") +
                         (write ? " write" : " no write");
                c.clear_depth = clear;
                c.state = [=](D3D12_GRAPHICS_PIPELINE_STATE_DESC& pd) {
                    pd.DepthStencilState.DepthEnable = TRUE;
                    pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC(func);
                    pd.DepthStencilState.DepthWriteMask = write ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
                };
                cases.push_back(std::move(c));
            }
    // Stencil functions: reference 5 against stored 3, 5 and 7; a pass increments (wrapping), a fail inverts.
    for (int func = 1; func <= 8; ++func)
        for (UINT8 stored : {UINT8(3), UINT8(5), UINT8(7)}) {
            Case c;
            c.name = std::string("stencil ") + kFuncNames[func] + " stored " + std::to_string(stored);
            c.clear_stencil = stored;
            c.stencil_ref = 5;
            c.state = [=](D3D12_GRAPHICS_PIPELINE_STATE_DESC& pd) {
                pd.DepthStencilState.StencilEnable = TRUE;
                pd.DepthStencilState.FrontFace = {D3D12_STENCIL_OP_INVERT, D3D12_STENCIL_OP_INVERT, D3D12_STENCIL_OP_INCR,
                                                  D3D12_COMPARISON_FUNC(func)};
            };
            cases.push_back(std::move(c));
        }
    // Stencil operations as the pass operation, as the fail operation, and as the depth-fail operation.
    for (int op = 1; op <= 8; ++op)
        for (UINT8 stored : {UINT8(0), UINT8(0x5a), UINT8(0xff)}) {
            Case c;
            c.name = std::string("stencil pass op ") + kOpNames[op] + " stored " + std::to_string(stored);
            c.clear_stencil = stored;
            c.stencil_ref = 0x33;
            c.state = [=](D3D12_GRAPHICS_PIPELINE_STATE_DESC& pd) {
                pd.DepthStencilState.StencilEnable = TRUE;
                pd.DepthStencilState.FrontFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP(op),
                                                  D3D12_COMPARISON_FUNC_ALWAYS};
            };
            cases.push_back(c);
            c.name = std::string("stencil fail op ") + kOpNames[op] + " stored " + std::to_string(stored);
            c.state = [=](D3D12_GRAPHICS_PIPELINE_STATE_DESC& pd) {
                pd.DepthStencilState.StencilEnable = TRUE;
                pd.DepthStencilState.FrontFace = {D3D12_STENCIL_OP(op), D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                                  D3D12_COMPARISON_FUNC_NEVER};
            };
            cases.push_back(c);
            c.name = std::string("stencil depth-fail op ") + kOpNames[op] + " stored " + std::to_string(stored);
            c.clear_depth = 0.5f;
            c.state = [=](D3D12_GRAPHICS_PIPELINE_STATE_DESC& pd) {
                pd.DepthStencilState.StencilEnable = TRUE;
                pd.DepthStencilState.DepthEnable = TRUE;
                pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;  // the draw's depth 0 fails against 0.5
                pd.DepthStencilState.FrontFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP(op), D3D12_STENCIL_OP_KEEP,
                                                  D3D12_COMPARISON_FUNC_ALWAYS};
            };
            cases.push_back(c);
        }
    // Masks: the compare uses the read mask, the write uses the write mask.
    for (UINT8 read : {UINT8(0x0f), UINT8(0xf0), UINT8(0xff)})
        for (UINT8 write : {UINT8(0x00), UINT8(0x3c), UINT8(0xff)}) {
            Case c;
            c.name = "stencil masks read " + std::to_string(read) + " write " + std::to_string(write);
            c.clear_stencil = 0xa5;
            c.stencil_ref = 0x35;
            c.state = [=](D3D12_GRAPHICS_PIPELINE_STATE_DESC& pd) {
                pd.DepthStencilState.StencilEnable = TRUE;
                pd.DepthStencilState.StencilReadMask = read;
                pd.DepthStencilState.StencilWriteMask = write;
                pd.DepthStencilState.FrontFace = {D3D12_STENCIL_OP_ZERO, D3D12_STENCIL_OP_ZERO, D3D12_STENCIL_OP_REPLACE,
                                                  D3D12_COMPARISON_FUNC_EQUAL};
            };
            cases.push_back(std::move(c));
        }
    // The back face's state: the front face is inverted, so the triangle is back-facing and the other state applies.
    for (bool flipped : {false, true}) {
        Case c;
        c.name = std::string("stencil two-sided, ") + (flipped ? "back-facing" : "front-facing");
        c.clear_stencil = 9;
        c.stencil_ref = 9;
        c.state = [=](D3D12_GRAPHICS_PIPELINE_STATE_DESC& pd) {
            pd.RasterizerState.FrontCounterClockwise = flipped;
            pd.DepthStencilState.StencilEnable = TRUE;
            pd.DepthStencilState.FrontFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_INCR,
                                              D3D12_COMPARISON_FUNC_ALWAYS};
            pd.DepthStencilState.BackFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_DECR,
                                             D3D12_COMPARISON_FUNC_ALWAYS};
        };
        cases.push_back(std::move(c));
    }
    return cases;
}

}  // namespace

int main() {
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (hostwine::host_is_wine()) return 77;  // the host D3D12 runtime is the reference

    const auto vs_program = d3d12test::fssm_program({0x56530501, 0xbf810000});
    const auto ps_program = d3d12test::fssm_program({0x50530501, 0xbf810000});
    const auto library = std::filesystem::temp_directory_path() / "leonida_state_sweep_test";
    std::filesystem::create_directories(library);
    const auto overwrite = std::filesystem::copy_options::overwrite_existing;
    std::filesystem::copy_file(LEONIDA_SWEEP_VS_SPV, library / "vs.spv", overwrite);
    std::filesystem::copy_file(LEONIDA_SWEEP_PS_SPV, library / "ps.spv", overwrite);
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }
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
    info.shader_dir = library.string();
    Ref<ID3D12Device> ours, host;
    std::string adapter;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
    if (!host_device(vk, host, adapter)) {
        std::fprintf(stderr, "no host D3D12 device on the Vulkan device's adapter\n");
        return 77;
    }
    std::printf("host reference: %s\n", adapter.c_str());

    const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile = compiler ? reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(
                                        GetProcAddress(compiler, "D3DCompile")))
                                  : nullptr;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    Programs host_programs, our_programs;
    host_programs.root = our_programs.root = d3d12test::serialize_root_signature(root_desc);
    if (!compile || host_programs.root.empty() || !compile_hlsl(compile, kHostVs, host_programs.vs, true) ||
        !compile_hlsl(compile, kHostPs, host_programs.ps, false)) {
        std::fprintf(stderr, "d3dcompiler_47 could not compile the host's passes\n");
        return 1;
    }
    our_programs.vs = d3d12test::shader_container(vs_program);
    our_programs.ps = d3d12test::shader_container(ps_program);

    const std::vector<Case> cases = build_cases();
    unsigned compared = 0, differ = 0, skipped = 0;
    for (const Case& c : cases) {
        const Result reference = run(host.p, c, host_programs);
        const Result mine = run(ours.p, c, our_programs);
        if (!reference.ok && SUCCEEDED(reference.pso) ) {
            std::fprintf(stderr, "%s: the host failed\n", c.name.c_str());
            return 1;
        }
        if (FAILED(reference.pso)) {
            // The host refuses this pipeline: the layer must too.
            ++compared;
            if (SUCCEEDED(mine.pso)) {
                ++differ;
                std::printf("%s: the host refuses the pipeline (0x%08x), the layer accepts it  MISMATCH\n", c.name.c_str(),
                            static_cast<unsigned>(reference.pso));
            }
            continue;
        }
        if (!mine.ok) {
            // A format the layer has no render target for yet.
            std::printf("%s: SKIP, the layer failed (pipeline 0x%08x)\n", c.name.c_str(), static_cast<unsigned>(mine.pso));
            ++skipped;
            continue;
        }
        ++compared;
        if (mine.color != reference.color || mine.stencil != reference.stencil) {
            ++differ;
            std::printf("%s: colour host %s, layer %s; stencil host %s, layer %s  MISMATCH\n", c.name.c_str(),
                        hex(reference.color, 8).c_str(), hex(mine.color, 8).c_str(), hex(reference.stencil, 4).c_str(),
                        hex(mine.stencil, 4).c_str());
        }
    }
    std::printf("%u cases compared with %s, %u differ; %u skipped\n", compared, adapter.c_str(), differ, skipped);
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", vk.validation_errors());
        return 1;
    }
    return differ ? 1 : 0;
}
