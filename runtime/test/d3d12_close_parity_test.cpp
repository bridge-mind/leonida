// What ID3D12GraphicsCommandList::Close returns after one recorded call, on the layer and on Windows' own D3D12 over
// WARP (no debug layer). Each case records its calls on a fresh allocator and direct list, then closes it; nothing is
// executed. The layer is held to WARP's verdict (Close succeeds or fails; the code is printed but not compared, since
// the layer answers every refused list with E_INVALIDARG where the PC runtime also uses E_FAIL).
//
// Where the two disagree today the case is marked as a known difference, and the test checks that the difference is
// still there: a change that makes such a case match fails here until its mark is removed, so the table stays an
// accurate list of the gap. Two kinds:
//   layer_stricter  WARP closes the list, the layer fails it. Since one failed list fails the whole
//                   ExecuteCommandLists, each of these drops a submission D3D12 would run.
//   layer_looser    WARP fails the list, the layer closes it.
// The `warp_ok` column is WARP's verdict as measured on Windows 11 26200 (d3d10warp.dll 10.0.26100.9818); it is
// the reference when no WARP device can be made (Wine, whose vkd3d answers differently, or a machine without WARP),
// and a live WARP answer that differs from it is printed. Calls that crash the PC runtime (ExecuteIndirect with a
// null command signature) are left out. No game data.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>

#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

#define REQUIRE(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

constexpr HRESULT not_run = HRESULT(0x7fff0000);  // the case needs an interface the device does not give

struct Env {
    ID3D12Device* device = nullptr;
    Ref<ID3D12Resource> buffer, upload, readback, target, target2, depth, msaa, tex, mips;
    Ref<ID3D12DescriptorHeap> rtv_heap, dsv_heap;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{}, dsv{};
    Ref<ID3D12QueryHeap> occlusion;
    Ref<ID3D12CommandAllocator> closed_bundle_allocator;
    Ref<ID3D12GraphicsCommandList> closed_bundle;
    // The list being recorded, and the same list as ID3D12GraphicsCommandList2 (null when not given).
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12GraphicsCommandList2* list2 = nullptr;

    bool init(ID3D12Device* d) {
        device = d;
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT}, upload_heap{D3D12_HEAP_TYPE_UPLOAD},
            readback_heap{D3D12_HEAP_TYPE_READBACK};
        const auto create = [&](const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC& desc,
                                D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out) {
            return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                             IID_PPV_ARGS(&out.p)));
        };
        const auto texture = [](DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, UINT samples = 1, UINT16 mip_levels = 1,
                                UINT16 layers = 1) {
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width = desc.Height = 64;
            desc.DepthOrArraySize = layers;
            desc.MipLevels = mip_levels;
            desc.Format = format;
            desc.SampleDesc.Count = samples;
            desc.Flags = flags;
            return desc;
        };
        const DXGI_FORMAT rgba = DXGI_FORMAT_R8G8B8A8_UNORM;
        CHECK(create(default_heap, buffer_desc(65536), D3D12_RESOURCE_STATE_COMMON, buffer));
        CHECK(create(upload_heap, buffer_desc(1 << 20), D3D12_RESOURCE_STATE_GENERIC_READ, upload));
        CHECK(create(readback_heap, buffer_desc(1 << 20), D3D12_RESOURCE_STATE_COPY_DEST, readback));
        CHECK(create(default_heap, texture(rgba, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET), D3D12_RESOURCE_STATE_COMMON,
                     target));
        CHECK(create(default_heap, texture(rgba, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET), D3D12_RESOURCE_STATE_COMMON,
                     target2));
        CHECK(create(default_heap, texture(rgba, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON, tex));
        CHECK(create(default_heap, texture(rgba, D3D12_RESOURCE_FLAG_NONE, 1, 4, 3), D3D12_RESOURCE_STATE_COMMON,
                     mips));
        CHECK(create(default_heap, texture(DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                     D3D12_RESOURCE_STATE_DEPTH_WRITE, depth));
        CHECK(create(default_heap, texture(rgba, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 4),
                     D3D12_RESOURCE_STATE_COMMON, msaa));
        const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                                                  0};
        const D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                                                  0};
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap.p))));
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&dsv_desc, IID_PPV_ARGS(&dsv_heap.p))));
        rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        dsv = dsv_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(target.p, nullptr, rtv);
        device->CreateDepthStencilView(depth.p, nullptr, dsv);
        const D3D12_QUERY_HEAP_DESC query_desc{D3D12_QUERY_HEAP_TYPE_OCCLUSION, 8, 0};
        CHECK(SUCCEEDED(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&occlusion.p))));
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE,
                                                       IID_PPV_ARGS(&closed_bundle_allocator.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, closed_bundle_allocator.p,
                                                  nullptr, IID_PPV_ARGS(&closed_bundle.p))));
        CHECK(SUCCEEDED(closed_bundle->Close()));
        return true;
    }
};

D3D12_RESOURCE_BARRIER barrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after,
                               UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                               D3D12_RESOURCE_BARRIER_FLAGS flags = D3D12_RESOURCE_BARRIER_FLAG_NONE) {
    D3D12_RESOURCE_BARRIER b = transition(resource, before, after);
    b.Transition.Subresource = subresource;
    b.Flags = flags;
    return b;
}

D3D12_TEXTURE_COPY_LOCATION subresource(ID3D12Resource* resource, UINT index) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = resource;
    location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    location.SubresourceIndex = index;
    return location;
}

D3D12_TEXTURE_COPY_LOCATION footprint(ID3D12Resource* resource, UINT64 offset, UINT row_pitch,
                                      DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = resource;
    location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    location.PlacedFootprint.Offset = offset;
    location.PlacedFootprint.Footprint = {format, 64, 64, 1, row_pitch};
    return location;
}

enum class Known { match, layer_stricter, layer_looser };

// Records the case's calls on e.list; false when it needs an interface the device does not give.
using Recorder = std::function<bool(Env&)>;

struct Case {
    const char* name;
    bool warp_ok;  // WARP's Close verdict, measured
    Known known;
    Recorder record;
};

#define CALLS(...) [](Env& e) { __VA_ARGS__; return true; }
#define RECORD_COPY(dst_location, x, src_location, box) do { \
    const auto d = dst_location; const auto s = src_location; e.list->CopyTextureRegion(&d, x, 0, 0, &s, box); \
} while (0)

constexpr auto common = D3D12_RESOURCE_STATE_COMMON;
constexpr auto copy_dest = D3D12_RESOURCE_STATE_COPY_DEST;
constexpr auto all = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
constexpr auto begin_only = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
constexpr auto end_only = D3D12_RESOURCE_BARRIER_FLAG_END_ONLY;
constexpr DXGI_FORMAT rgba = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr auto stricter = Known::layer_stricter;
constexpr auto looser = Known::layer_looser;
constexpr auto match = Known::match;

std::vector<Case> cases() {
    return {
        {"empty list", true, match, [](Env&) { return true; }},
        {"ResourceBarrier(0, nullptr)", true, match, CALLS(e.list->ResourceBarrier(0, nullptr))},
        // Resource barriers
        {"transition COMMON -> COMMON", false, looser,
         CALLS(const auto b = barrier(e.tex.p, common, common); e.list->ResourceBarrier(1, &b))},
        {"transition COPY_DEST -> COPY_DEST", false, looser,
         CALLS(const auto a = barrier(e.tex.p, common, copy_dest);
               const auto b = barrier(e.tex.p, copy_dest, copy_dest);
               e.list->ResourceBarrier(1, &a); e.list->ResourceBarrier(1, &b))},
        {"transition with BEGIN_ONLY | END_ONLY", false, match,
         CALLS(const auto b = barrier(e.tex.p, common, copy_dest, all, begin_only | end_only);
               e.list->ResourceBarrier(1, &b))},
        {"split transition: BEGIN_ONLY then END_ONLY", true, match,
         CALLS(auto b = barrier(e.tex.p, common, copy_dest, all, begin_only); e.list->ResourceBarrier(1, &b);
               b.Flags = end_only; e.list->ResourceBarrier(1, &b);
               const auto c = barrier(e.tex.p, copy_dest, common); e.list->ResourceBarrier(1, &c))},
        {"split transition: END_ONLY without BEGIN_ONLY", true, match,
         CALLS(const auto b = barrier(e.tex.p, common, copy_dest, all, end_only); e.list->ResourceBarrier(1, &b))},
        {"transition of a null resource", false, match,
         CALLS(const auto b = barrier(nullptr, common, copy_dest); e.list->ResourceBarrier(1, &b))},
        {"transition of subresource 99 of a 1-subresource texture", false, looser,
         CALLS(const auto b = barrier(e.tex.p, common, copy_dest, 99); e.list->ResourceBarrier(1, &b))},
        {"transition to COPY_DEST | COPY_SOURCE (write and read)", false, looser,
         CALLS(const auto b = barrier(e.tex.p, common, copy_dest | D3D12_RESOURCE_STATE_COPY_SOURCE);
               e.list->ResourceBarrier(1, &b))},
        {"UAV barrier of a null resource", true, match,
         CALLS(D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; e.list->ResourceBarrier(1, &b))},
        {"UAV barrier with BEGIN_ONLY", false, match,
         CALLS(D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; b.Flags = begin_only;
               e.list->ResourceBarrier(1, &b))},
        {"aliasing barrier null -> null", true, match,
         CALLS(D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
               e.list->ResourceBarrier(1, &b))},
        {"barrier type 7", false, match,
         CALLS(D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE(7); e.list->ResourceBarrier(1, &b))},
        // CopyBufferRegion
        {"CopyBufferRegion of 0 bytes", true, match, CALLS(e.list->CopyBufferRegion(e.buffer.p, 0, e.upload.p, 0, 0))},
        {"CopyBufferRegion past the end of the destination", false, looser,
         CALLS(e.list->CopyBufferRegion(e.buffer.p, 65000, e.upload.p, 0, 4096))},
        {"CopyBufferRegion within one buffer, disjoint", true, match,
         CALLS(e.list->CopyBufferRegion(e.buffer.p, 0, e.buffer.p, 4096, 1024))},
        {"CopyBufferRegion within one buffer, overlapping", true, match,
         CALLS(e.list->CopyBufferRegion(e.buffer.p, 0, e.buffer.p, 512, 1024))},
        {"CopyBufferRegion from a null source", false, match,
         CALLS(e.list->CopyBufferRegion(e.buffer.p, 0, nullptr, 0, 16))},
        {"CopyBufferRegion into a texture", false, looser,
         CALLS(e.list->CopyBufferRegion(e.tex.p, 0, e.upload.p, 0, 16))},
        // CopyTextureRegion
        {"CopyTextureRegion upload, offset 0, row pitch 256", true, match,
         CALLS(RECORD_COPY(subresource(e.tex.p, 0), 0, footprint(e.upload.p, 0, 256), nullptr))},
        {"CopyTextureRegion upload, offset 16 (not 512-aligned)", true, stricter,
         CALLS(RECORD_COPY(subresource(e.tex.p, 0), 0, footprint(e.upload.p, 16, 256), nullptr))},
        {"CopyTextureRegion upload, offset 4 (not 512-aligned)", true, stricter,
         CALLS(RECORD_COPY(subresource(e.tex.p, 0), 0, footprint(e.upload.p, 4, 256), nullptr))},
        {"CopyTextureRegion upload, row pitch 260 (not 256-aligned)", true, stricter,
         CALLS(RECORD_COPY(subresource(e.tex.p, 0), 0, footprint(e.upload.p, 0, 260), nullptr))},
        {"CopyTextureRegion upload, row pitch 128 (shorter than a row)", false, match,
         CALLS(RECORD_COPY(subresource(e.tex.p, 0), 0, footprint(e.upload.p, 0, 128), nullptr))},
        {"CopyTextureRegion upload, footprint format UNKNOWN", false, looser,
         CALLS(RECORD_COPY(subresource(e.tex.p, 0), 0, footprint(e.upload.p, 0, 256, DXGI_FORMAT_UNKNOWN), nullptr))},
        {"CopyTextureRegion upload, footprint R32_FLOAT into RGBA8", false, looser,
         CALLS(RECORD_COPY(subresource(e.tex.p, 0), 0, footprint(e.upload.p, 0, 256, DXGI_FORMAT_R32_FLOAT), nullptr))},
        {"CopyTextureRegion upload, footprint RGBA8_SRGB into RGBA8", true, match,
         CALLS(RECORD_COPY(subresource(e.tex.p, 0), 0,
                           footprint(e.upload.p, 0, 256, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB), nullptr))},
        {"CopyTextureRegion upload with an empty box", true, match,
         CALLS(const D3D12_BOX box{4, 4, 0, 4, 8, 1};
               RECORD_COPY(subresource(e.tex.p, 0), 0, footprint(e.upload.p, 0, 256), &box))},
        {"CopyTextureRegion upload with an inverted box", true, match,
         CALLS(const D3D12_BOX box{8, 4, 0, 4, 8, 1};
               RECORD_COPY(subresource(e.tex.p, 0), 0, footprint(e.upload.p, 0, 256), &box))},
        {"CopyTextureRegion upload past the texture's edge (x 32 of 64)", false, looser,
         CALLS(RECORD_COPY(subresource(e.tex.p, 0), 32, footprint(e.upload.p, 0, 256), nullptr))},
        {"CopyTextureRegion into a readback footprint", true, match,
         CALLS(RECORD_COPY(footprint(e.readback.p, 0, 256), 0, subresource(e.tex.p, 0), nullptr))},
        {"CopyTextureRegion footprint to footprint", true, stricter,
         CALLS(RECORD_COPY(footprint(e.readback.p, 0, 256), 0, footprint(e.upload.p, 0, 256), nullptr))},
        {"CopyTextureRegion into subresource 5 of a 1-subresource texture", false, looser,
         CALLS(RECORD_COPY(subresource(e.tex.p, 5), 0, footprint(e.upload.p, 0, 256), nullptr))},
        {"CopyTextureRegion between two subresources of one texture", true, match,
         CALLS(RECORD_COPY(subresource(e.mips.p, 1), 0, subresource(e.mips.p, 5), nullptr))},
        {"CopyTextureRegion within one subresource, disjoint", true, match,
         CALLS(const D3D12_BOX box{0, 0, 0, 8, 8, 1};
               RECORD_COPY(subresource(e.tex.p, 0), 32, subresource(e.tex.p, 0), &box))},
        {"CopyTextureRegion into a null location", false, match,
         CALLS(const auto s = subresource(e.tex.p, 0); e.list->CopyTextureRegion(nullptr, 0, 0, 0, &s, nullptr))},
        // CopyResource
        {"CopyResource with the destination as the source", true, match,
         CALLS(e.list->CopyResource(e.tex.p, e.tex.p))},
        {"CopyResource buffer to texture", true, match, CALLS(e.list->CopyResource(e.tex.p, e.buffer.p))},
        {"CopyResource between buffers of different sizes", false, looser,
         CALLS(e.list->CopyResource(e.upload.p, e.buffer.p))},
        {"CopyResource texture to texture", true, match, CALLS(e.list->CopyResource(e.target2.p, e.target.p))},
        // Clears and discards
        {"ClearRenderTargetView with an inverted rectangle", true, match,
         CALLS(const FLOAT red[4] = {1, 0, 0, 1}; const D3D12_RECT r{10, 10, 5, 5};
               e.list->ClearRenderTargetView(e.rtv, red, 1, &r))},
        {"ClearRenderTargetView with a rectangle past the target", true, match,
         CALLS(const FLOAT red[4] = {1, 0, 0, 1}; const D3D12_RECT r{-8, -8, 4096, 4096};
               e.list->ClearRenderTargetView(e.rtv, red, 1, &r))},
        {"ClearRenderTargetView with NaN, infinite and negative channels", true, match,
         CALLS(const FLOAT c[4] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
               -1, 2};
               e.list->ClearRenderTargetView(e.rtv, c, 0, nullptr))},
        {"ClearDepthStencilView with flags 0", true, stricter,
         CALLS(e.list->ClearDepthStencilView(e.dsv, D3D12_CLEAR_FLAGS(0), 1, 0, 0, nullptr))},
        {"ClearDepthStencilView depth 1.5", true, stricter,
         CALLS(e.list->ClearDepthStencilView(e.dsv, D3D12_CLEAR_FLAG_DEPTH, 1.5f, 0, 0, nullptr))},
        {"ClearDepthStencilView depth -0.5", true, stricter,
         CALLS(e.list->ClearDepthStencilView(e.dsv, D3D12_CLEAR_FLAG_DEPTH, -0.5f, 0, 0, nullptr))},
        {"ClearDepthStencilView depth NaN", true, stricter,
         CALLS(e.list->ClearDepthStencilView(e.dsv, D3D12_CLEAR_FLAG_DEPTH,
                                            std::numeric_limits<float>::quiet_NaN(), 0, 0,
                                            nullptr))},
        {"ClearDepthStencilView stencil only, depth 7 (not used)", true, match,
         CALLS(e.list->ClearDepthStencilView(e.dsv, D3D12_CLEAR_FLAG_STENCIL, 7.0f, 3, 0, nullptr))},
        {"ClearDepthStencilView with flags 7 (an unknown bit)", true, stricter,
         CALLS(e.list->ClearDepthStencilView(e.dsv, D3D12_CLEAR_FLAGS(7), 1, 0, 0, nullptr))},
        {"DiscardResource with no region", true, match, CALLS(e.list->DiscardResource(e.target.p, nullptr))},
        {"DiscardResource of 0 subresources", true, match,
         CALLS(const D3D12_DISCARD_REGION r{0, nullptr, 0, 0}; e.list->DiscardResource(e.target.p, &r))},
        {"DiscardResource of a texture that is no target", true, match,
         CALLS(e.list->DiscardResource(e.tex.p, nullptr))},
        // ResolveSubresource
        {"ResolveSubresource 4x -> 1x", true, match,
         CALLS(e.list->ResolveSubresource(e.target.p, 0, e.msaa.p, 0, rgba))},
        {"ResolveSubresource with format UNKNOWN", false, looser,
         CALLS(e.list->ResolveSubresource(e.target.p, 0, e.msaa.p, 0, DXGI_FORMAT_UNKNOWN))},
        {"ResolveSubresource as RGBA8_SRGB of RGBA8 textures", true, match,
         CALLS(e.list->ResolveSubresource(e.target.p, 0, e.msaa.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB))},
        {"ResolveSubresource as R32_FLOAT of RGBA8 textures", false, looser,
         CALLS(e.list->ResolveSubresource(e.target.p, 0, e.msaa.p, 0, DXGI_FORMAT_R32_FLOAT))},
        {"ResolveSubresource from a single-sampled texture", false, looser,
         CALLS(e.list->ResolveSubresource(e.target.p, 0, e.target2.p, 0, rgba))},
        {"ResolveSubresource with the destination as the source", false, match,
         CALLS(e.list->ResolveSubresource(e.msaa.p, 0, e.msaa.p, 0, rgba))},
        // Input assembler, rasterizer, output merger
        {"IASetVertexBuffers(30, 4): slots past 31", true, stricter,
         CALLS(const D3D12_VERTEX_BUFFER_VIEW v[4] = {}; e.list->IASetVertexBuffers(30, 4, v))},
        {"IASetVertexBuffers(0, 0, nullptr)", true, match, CALLS(e.list->IASetVertexBuffers(0, 0, nullptr))},
        {"IASetIndexBuffer(nullptr)", true, match, CALLS(e.list->IASetIndexBuffer(nullptr))},
        {"IASetIndexBuffer as R16_UINT", true, match,
         CALLS(const D3D12_INDEX_BUFFER_VIEW v{e.buffer->GetGPUVirtualAddress(), 64, DXGI_FORMAT_R16_UINT};
               e.list->IASetIndexBuffer(&v))},
        {"IASetIndexBuffer as R32_UINT", true, match,
         CALLS(const D3D12_INDEX_BUFFER_VIEW v{e.buffer->GetGPUVirtualAddress(), 64, DXGI_FORMAT_R32_UINT};
               e.list->IASetIndexBuffer(&v))},
        {"IASetIndexBuffer as UNKNOWN, size 0", true, match,
         CALLS(const D3D12_INDEX_BUFFER_VIEW v{0, 0, DXGI_FORMAT_UNKNOWN}; e.list->IASetIndexBuffer(&v))},
        {"IASetIndexBuffer as UNKNOWN, size 64", true, match,
         CALLS(const D3D12_INDEX_BUFFER_VIEW v{e.buffer->GetGPUVirtualAddress(), 64, DXGI_FORMAT_UNKNOWN};
               e.list->IASetIndexBuffer(&v))},
        {"IASetIndexBuffer as R8_UINT", true, match,
         CALLS(const D3D12_INDEX_BUFFER_VIEW v{e.buffer->GetGPUVirtualAddress(), 64, DXGI_FORMAT_R8_UINT};
               e.list->IASetIndexBuffer(&v))},
        {"IASetIndexBuffer as R32_FLOAT", true, match,
         CALLS(const D3D12_INDEX_BUFFER_VIEW v{e.buffer->GetGPUVirtualAddress(), 64, DXGI_FORMAT_R32_FLOAT};
               e.list->IASetIndexBuffer(&v))},
        {"IASetPrimitiveTopology(UNDEFINED)", true, match,
         CALLS(e.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_UNDEFINED))},
        {"IASetPrimitiveTopology(TRIANGLELIST)", true, match,
         CALLS(e.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST))},
        {"IASetPrimitiveTopology(7) (gap value)", true, match,
         CALLS(e.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(7)))},
        {"IASetPrimitiveTopology(32_CONTROL_POINT_PATCHLIST)", true, match,
         CALLS(e.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_32_CONTROL_POINT_PATCHLIST))},
        {"IASetPrimitiveTopology(65) (past 32-cp patch)", true, match,
         CALLS(e.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(65)))},
        {"IASetPrimitiveTopology(99)", true, match, CALLS(e.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(99)))},
        {"OMSetBlendFactor", true, match,
         CALLS(const FLOAT factor[4] = {0.25f, 0.5f, 0.75f, 1.0f}; e.list->OMSetBlendFactor(factor))},
        {"OMSetStencilRef(7)", true, match, CALLS(e.list->OMSetStencilRef(7))},
        {"ExecuteBundle after post-close state calls on bundle", true, match,
         CALLS(const FLOAT factor[4] = {0.25f, 0.5f, 0.75f, 1.0f};
               e.closed_bundle->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
               e.closed_bundle->IASetIndexBuffer(nullptr);
               e.closed_bundle->OMSetBlendFactor(factor);
               e.closed_bundle->OMSetStencilRef(9);
               e.closed_bundle->ClearState(nullptr);
               e.list->ExecuteBundle(e.closed_bundle.p))},
        {"RSSetViewports(0, nullptr)", true, match, CALLS(e.list->RSSetViewports(0, nullptr))},
        {"RSSetViewports(17)", true, stricter,
         CALLS(D3D12_VIEWPORT v[17]; for (auto& x : v) x = {0, 0, 8, 8, 0, 1}; e.list->RSSetViewports(17, v))},
        {"RSSetViewports with MinDepth -1", true, match,
         CALLS(const D3D12_VIEWPORT v{0, 0, 8, 8, -1, 1}; e.list->RSSetViewports(1, &v))},
        {"RSSetViewports with a negative width", true, match,
         CALLS(const D3D12_VIEWPORT v{0, 0, -8, 8, 0, 1}; e.list->RSSetViewports(1, &v))},
        {"RSSetViewports with a negative height", true, match,
         CALLS(const D3D12_VIEWPORT v{0, 8, 8, -8, 0, 1}; e.list->RSSetViewports(1, &v))},
        {"RSSetViewports at x -40000", true, match,
         CALLS(const D3D12_VIEWPORT v{-40000, 0, 8, 8, 0, 1}; e.list->RSSetViewports(1, &v))},
        {"RSSetScissorRects(0, nullptr)", true, match, CALLS(e.list->RSSetScissorRects(0, nullptr))},
        {"RSSetScissorRects(17)", true, match, CALLS(const D3D12_RECT r[17] = {}; e.list->RSSetScissorRects(17, r))},
        {"RSSetScissorRects inverted", true, match,
         CALLS(const D3D12_RECT r{10, 10, 5, 5}; e.list->RSSetScissorRects(1, &r))},
        {"OMSetRenderTargets(9)", false, match,
         CALLS(D3D12_CPU_DESCRIPTOR_HANDLE h[9]; for (auto& x : h) x = e.rtv;
               e.list->OMSetRenderTargets(9, h, FALSE, nullptr))},
        {"OMSetRenderTargets(0, nullptr, depth)", true, match,
         CALLS(e.list->OMSetRenderTargets(0, nullptr, FALSE, &e.dsv))},
        // Draws, dispatches and root arguments with nothing set
        {"DrawInstanced(3) with nothing set", true, stricter, CALLS(e.list->DrawInstanced(3, 1, 0, 0))},
        {"DrawInstanced(0) with nothing set", true, stricter, CALLS(e.list->DrawInstanced(0, 1, 0, 0))},
        {"Dispatch(1, 1, 1) with nothing set", true, stricter, CALLS(e.list->Dispatch(1, 1, 1))},
        {"Dispatch(0, 0, 0) with nothing set", true, stricter, CALLS(e.list->Dispatch(0, 0, 0))},
        {"SetGraphicsRoot32BitConstant before a root signature", true, stricter,
         CALLS(e.list->SetGraphicsRoot32BitConstant(0, 1, 0))},
        {"SetGraphicsRootConstantBufferView before a root signature", true, stricter,
         CALLS(e.list->SetGraphicsRootConstantBufferView(0, e.buffer->GetGPUVirtualAddress()))},
        {"SetGraphicsRootDescriptorTable before a root signature", true, stricter,
         CALLS(e.list->SetGraphicsRootDescriptorTable(0, {}))},
        {"SetGraphicsRootSignature(nullptr)", true, match, CALLS(e.list->SetGraphicsRootSignature(nullptr))},
        {"SetDescriptorHeaps(0, nullptr)", true, match, CALLS(e.list->SetDescriptorHeaps(0, nullptr))},
        {"ClearState(nullptr)", true, match, CALLS(e.list->ClearState(nullptr))},
        // Predication and queries
        {"SetPredication(nullptr)", true, match,
         CALLS(e.list->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO))},
        {"SetPredication at offset 4", false, match,
         CALLS(e.list->SetPredication(e.buffer.p, 4, D3D12_PREDICATION_OP_EQUAL_ZERO);
               e.list->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO))},
        {"SetPredication with operation 7", false, match,
         CALLS(e.list->SetPredication(e.buffer.p, 0, D3D12_PREDICATION_OP(7));
               e.list->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO))},
        {"BeginQuery without EndQuery", false, match,
         CALLS(e.list->BeginQuery(e.occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0))},
        {"EndQuery without BeginQuery", false, match,
         CALLS(e.list->EndQuery(e.occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0))},
        {"BeginQuery twice on one index", false, match,
         CALLS(e.list->BeginQuery(e.occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
               e.list->BeginQuery(e.occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
               e.list->EndQuery(e.occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0))},
        {"BeginQuery and EndQuery at index 100 of 8", false, looser,
         CALLS(e.list->BeginQuery(e.occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 100);
               e.list->EndQuery(e.occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 100))},
        {"BeginQuery OCCLUSION, EndQuery BINARY_OCCLUSION", true, stricter,
         CALLS(e.list->BeginQuery(e.occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
               e.list->EndQuery(e.occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 0))},
        {"ResolveQueryData of 0 queries", true, match,
         CALLS(e.list->ResolveQueryData(e.occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0, 0, e.readback.p, 0))},
        {"ResolveQueryData at offset 4", false, match,
         CALLS(e.list->ResolveQueryData(e.occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0, 1, e.readback.p, 4))},
        {"ResolveQueryData of a query never begun", true, match,
         CALLS(e.list->ResolveQueryData(e.occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 3, 1, e.readback.p, 0))},
        // Markers and immediate writes
        {"EndEvent without BeginEvent", true, match, CALLS(e.list->EndEvent())},
        {"SetMarker(0, nullptr, 0)", true, match, CALLS(e.list->SetMarker(0, nullptr, 0))},
        {"WriteBufferImmediate of 0 values", true, match,
         [](Env& e) {
             if (!e.list2) return false;
             e.list2->WriteBufferImmediate(0, nullptr, nullptr);
             return true;
         }},
        {"WriteBufferImmediate at an address that is not 4-aligned", true, stricter,
         [](Env& e) {
             if (!e.list2) return false;
             const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER p{e.buffer->GetGPUVirtualAddress() + 2, 7};
             e.list2->WriteBufferImmediate(1, &p, nullptr);
             return true;
         }},
    };
}

// Records one case on a fresh allocator and list of `env`'s device and returns what Close says.
bool run_case(Env& env, const Case& c, HRESULT& close) {
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList2> list2;
    CHECK(SUCCEEDED(env.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(env.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                  IID_PPV_ARGS(&list.p))));
    if (FAILED(list->QueryInterface(IID_PPV_ARGS(&list2.p)))) list2.p = nullptr;
    env.list = list.p;
    env.list2 = list2.p;
    const bool recorded = c.record(env);
    close = recorded ? list->Close() : not_run;
    if (!recorded) list->Close();
    env.list = nullptr;
    env.list2 = nullptr;
    return true;
}

std::string text(HRESULT hr) {
    if (hr == not_run) return "n/a";
    char out[16];
    std::snprintf(out, sizeof(out), "0x%08lx", static_cast<unsigned long>(hr));
    return out;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    REQUIRE(device_status == vktest::DeviceStatus::ready);
    const auto table = cases();
    const unsigned errors_before = vk.validation_errors();

    std::vector<HRESULT> layer(table.size(), not_run), warp(table.size(), not_run);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = true;
        Ref<ID3D12Device> ours;
        REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p))));
        Env env;
        REQUIRE(env.init(ours.p));
        for (std::size_t i = 0; i < table.size(); ++i) REQUIRE(run_case(env, table[i], layer[i]));
    }

    bool warp_ran = false;
    if (!hostwine::host_is_wine()) {
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> adapter;
        Ref<ID3D12Device> host;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) &&
            SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) &&
            SUCCEEDED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
            Env env;
            REQUIRE(env.init(host.p));
            for (std::size_t i = 0; i < table.size(); ++i) REQUIRE(run_case(env, table[i], warp[i]));
            // Nothing is executed, so nothing should remove the device; if something did, its answers are suspect.
            REQUIRE(host->GetDeviceRemovedReason() == S_OK);
            warp_ran = true;
        } else {
            std::printf("no WARP device: the layer is compared with the measured WARP column\n");
        }
    }

    unsigned failures = 0, matched = 0, known_stricter = 0, known_looser = 0, warp_changed = 0, skipped = 0;
    std::printf("%-62s | %-10s | %-10s | %s\n", "case", "layer", "WARP", "expected");
    for (std::size_t i = 0; i < table.size(); ++i) {
        const Case& c = table[i];
        const bool live = warp_ran && warp[i] != not_run;
        const char* expected = c.known == Known::match ? "match"
                               : c.known == Known::layer_stricter ? "layer stricter (known)"
                                                                  : "layer looser (known)";
        std::printf("%-62s | %-10s | %-10s | %s\n", c.name, text(layer[i]).c_str(),
                    live ? text(warp[i]).c_str() : (c.warp_ok ? "(S_OK)" : "(failure)"), expected);
        if (layer[i] == not_run) {
            ++skipped;
            continue;
        }
        if (live && SUCCEEDED(warp[i]) != c.warp_ok) {
            std::printf("  WARP's verdict here differs from the measured column (%s): compared with the live answer\n",
                        c.warp_ok ? "S_OK" : "a failure");
            ++warp_changed;
        }
        const bool reference_ok = live ? SUCCEEDED(warp[i]) : c.warp_ok;
        const bool layer_ok = SUCCEEDED(layer[i]);
        const bool want_layer_ok = c.known == Known::match ? reference_ok : !reference_ok;
        if (layer_ok != want_layer_ok) {
            if (c.known == Known::match)
                std::fprintf(stderr, "FAIL %s: layer Close %s, WARP %s\n", c.name, text(layer[i]).c_str(),
                             reference_ok ? "succeeds" : "fails");
            else
                std::fprintf(stderr, "FAIL %s: now matches WARP; mark it Known::match in the table\n", c.name);
            ++failures;
            continue;
        }
        if (c.known == Known::match) ++matched;
        else if (c.known == Known::layer_stricter) ++known_stricter;
        else ++known_looser;
    }
    const unsigned validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", validation_errors);
        ++failures;
    }
    std::printf("close parity: %zu cases, %u match %s, %u known layer-stricter, %u known layer-looser, %u not run, "
                "%u where WARP differs from the measured column, %u failures\n",
                table.size(), matched, warp_ran ? "WARP" : "the measured WARP column", known_stricter, known_looser,
                skipped, warp_changed, failures);
    return failures ? 1 : 0;
}
