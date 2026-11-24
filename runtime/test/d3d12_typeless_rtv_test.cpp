
// Compare typed RTVs on one typeless RGBA8 resource against Windows D3D12/WARP.
// No game data or shaders. Run explicitly (needs Vulkan hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <algorithm>
#include <array>
#include <utility>
#include <vector>
#include <cstring>
#include <limits>
#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};
struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

struct ClearOptions {
    bool automatic_mips = false;
    bool predication = false;
    std::uint64_t predicate = 0;
    D3D12_PREDICATION_OP operation = D3D12_PREDICATION_OP_EQUAL_ZERO;
    bool overwrite_predicate = false;
};

bool clear_and_read(ID3D12Device* device, DXGI_FORMAT whole_format, DXGI_FORMAT rect_format, bool partial,
                    bool rewrite, bool uav, std::vector<unsigned char>& pixels,
                    UINT mip = 0, UINT first_layer = 0, UINT layer_count = 1, bool array = false, bool mip_chain = false,
                    ClearOptions options = {}) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    const FLOAT full_float[4] = {0.25f, 0.5f, 0.75f, 0.25f}, part_float[4] = {0.0031308f, 0.18f, 0.5f, 0.75f};
    const FLOAT full_uint[4] = {7, 128, 255, 9}, part_uint[4] = {1.5f, 2.5f, 256, -1};
    const FLOAT full_sint[4] = {-129, -23.75f, 126.75f, 130}, part_sint[4] = {-128, -0.5f, 1.75f, 127};
    const FLOAT full_snorm[4] = {-1.25f, -0.75f, 0.25f, 1.25f}, part_snorm[4] = {-0.25f, 0.75f, 0.33f, -0.33f};
    const auto clear_color = [&](DXGI_FORMAT format, bool rectangle) -> const FLOAT* {
        switch (format) {
        case DXGI_FORMAT_R8G8B8A8_UINT: return rectangle ? part_uint : full_uint;
        case DXGI_FORMAT_R8G8B8A8_SINT: return rectangle ? part_sint : full_sint;
        case DXGI_FORMAT_R8G8B8A8_SNORM: return rectangle ? part_snorm : full_snorm;
        default: return rectangle ? part_float : full_float;
        }
    };
    const FLOAT* full = clear_color(whole_format, false);
    const FLOAT* part = clear_color(rect_format, true);
    constexpr unsigned bytes = 4;
    constexpr UINT width = 5, height = 3; // Odd dimensions expose incorrect treatment of footprint padding.
    Ref<ID3D12Resource> image, readback;
    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = width;
    texture.Height = height;
    texture.DepthOrArraySize = array ? 4 : 1;
    texture.MipLevels = options.automatic_mips ? 0 : array || mip_chain ? 3 : 1;
    texture.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
    texture.SampleDesc.Count = 1;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (uav) texture.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &texture,
        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&image.p))));
    const auto resolved = image->GetDesc();
    CHECK(resolved.MipLevels == (array || mip_chain || options.automatic_mips ? 3 : 1));
    CHECK(resolved.DepthOrArraySize == texture.DepthOrArraySize);
    texture = resolved; // MipLevels=0 must be resolved before counting/reading subresources.
    const UINT count = texture.MipLevels * texture.DepthOrArraySize;
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> placed(count);
    UINT64 total = 0;
    device->GetCopyableFootprints(&texture, 0, count, 0, placed.data(), nullptr, nullptr, &total);
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = buffer.SampleDesc.Count = 1;
    buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    heap_properties.Type = D3D12_HEAP_TYPE_READBACK;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));
    Ref<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    const D3D12_CPU_DESCRIPTOR_HANDLE other{handle.ptr + device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)};
    D3D12_RENDER_TARGET_VIEW_DESC view{};
    view.ViewDimension = array ? D3D12_RTV_DIMENSION_TEXTURE2DARRAY : D3D12_RTV_DIMENSION_TEXTURE2D;
    if (array || mip_chain) {
        // Give every subresource a distinct sentinel before touching the selected view.
        view.Format = DXGI_FORMAT_R8G8B8A8_UINT;
        for (UINT layer = 0; layer < texture.DepthOrArraySize; ++layer) {
            for (UINT level = 0; level < texture.MipLevels; ++level) {
                if (array) view.Texture2DArray = {level, layer, 1, 0};
                else view.Texture2D = {level, 0};
                device->CreateRenderTargetView(image.p, &view, handle);
                const FLOAT seed[] = {FLOAT(11 + layer * 3 + level), 31, 73, 127};
                list->ClearRenderTargetView(handle, seed, 0, nullptr);
            }
        }
        if (array) view.Texture2DArray = {mip, first_layer, layer_count, 0};
        else view.Texture2D = {mip, 0};
    }
    view.Format = whole_format;
    device->CreateRenderTargetView(image.p, &view, handle);
    view.Format = rect_format;
    device->CreateRenderTargetView(image.p, &view, other);
    Ref<ID3D12Resource> predicate, predicate_upload;
    if (options.predication) {
        D3D12_RESOURCE_DESC predicate_desc = buffer;
        predicate_desc.Width = 16;
        const D3D12_HEAP_PROPERTIES upload{D3D12_HEAP_TYPE_UPLOAD};
        CHECK(SUCCEEDED(device->CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE, &predicate_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&predicate_upload.p))));
        void* mapped = nullptr;
        const D3D12_RANGE none{};
        CHECK(SUCCEEDED(predicate_upload->Map(0, &none, &mapped)) && mapped);
        const UINT64 values[] = {options.predicate, options.predicate ? 0u : 1u};
        std::memcpy(mapped, values, sizeof(values));
        const D3D12_RANGE written{0, sizeof(values)};
        predicate_upload->Unmap(0, &written);
        const auto transition_predicate = [&](D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = predicate.p;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = before; barrier.Transition.StateAfter = after;
            list->ResourceBarrier(1, &barrier);
        };
        if (options.overwrite_predicate) {
            const D3D12_HEAP_PROPERTIES local{D3D12_HEAP_TYPE_DEFAULT};
            CHECK(SUCCEEDED(device->CreateCommittedResource(&local, D3D12_HEAP_FLAG_NONE, &predicate_desc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&predicate.p))));
            list->CopyBufferRegion(predicate.p, 0, predicate_upload.p, 0, 8);
            transition_predicate(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PREDICATION);
        }
        list->SetPredication(predicate.p ? predicate.p : predicate_upload.p, 0, options.operation);
        if (options.overwrite_predicate) {
            // Choose only predicates that execute this copy on both backends. Subsequent clears must still use
            // the captured value, even though the predicate buffer now has the opposite zero/nonzero value.
            transition_predicate(D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COPY_DEST);
            list->CopyBufferRegion(predicate.p, 0, predicate_upload.p, 8, 8);
            transition_predicate(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PREDICATION);
        }
    }
    list->ClearRenderTargetView(handle, full, 0, nullptr);
    if (rewrite) {
        // ClearRenderTargetView must have captured its descriptor, including format, by value.
        view.Format = whole_format == DXGI_FORMAT_R8G8B8A8_UINT ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UINT;
        device->CreateRenderTargetView(image.p, &view, handle);
    }
    const D3D12_RECT rect = array || mip_chain ? D3D12_RECT{mip == 2 ? 0 : 1, 0,
        LONG(std::max(1u, width >> mip)), LONG(std::max(1u, height >> mip))} : D3D12_RECT{1, 1, 4, 3};
    if (partial) list->ClearRenderTargetView(other, part, 1, &rect);
    if (options.predication) list->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = image.p;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = image.p;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = readback.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    for (UINT subresource = 0; subresource < count; ++subresource) {
        src.SubresourceIndex = subresource;
        dst.PlacedFootprint = placed[subresource];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    Event event;
    CHECK(event.handle);
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, event.handle)));
    CHECK(WaitForSingleObject(event.handle, 5000) == WAIT_OBJECT_0);
    void* mapped = nullptr;
    const D3D12_RANGE range{0, SIZE_T(total)};
    CHECK(SUCCEEDED(readback->Map(0, &range, &mapped)) && mapped);
    pixels.clear();
    for (const auto& footprint : placed) {
        const auto& f = footprint.Footprint;
        for (UINT y = 0; y < f.Height; ++y) {
            const auto* row = static_cast<unsigned char*>(mapped) + footprint.Offset + y * f.RowPitch;
            pixels.insert(pixels.end(), row, row + f.Width * bytes);
        }
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}
bool rejects_view(ID3D12Device* device, DXGI_FORMAT resource_format, const D3D12_RENDER_TARGET_VIEW_DESC* view,
                  UINT16 mips = 1, UINT16 layers = 1, UINT samples = 1,
                  const DXGI_FORMAT* cast_list = nullptr, UINT32 cast_count = 0) {
    Ref<ID3D12Resource> resource;
    D3D12_HEAP_PROPERTIES heap_properties{D3D12_HEAP_TYPE_DEFAULT};
    if (cast_count) {
        Ref<ID3D12Device10> device10;
        CHECK(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device10.p))));
        D3D12_RESOURCE_DESC1 tex1{};
        tex1.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        tex1.Width = tex1.Height = 4;
        tex1.DepthOrArraySize = layers;
        tex1.MipLevels = mips;
        tex1.Format = resource_format;
        tex1.SampleDesc.Count = samples;
        tex1.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        CHECK(SUCCEEDED(device10->CreateCommittedResource3(&heap_properties, D3D12_HEAP_FLAG_NONE, &tex1,
            D3D12_BARRIER_LAYOUT_RENDER_TARGET, nullptr, nullptr, cast_count, cast_list,
            IID_PPV_ARGS(&resource.p))));
    } else {
        D3D12_RESOURCE_DESC texture{};
        texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = texture.Height = 4;
        texture.DepthOrArraySize = layers;
        texture.MipLevels = mips;
        texture.Format = resource_format;
        texture.SampleDesc.Count = samples;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &texture,
            D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&resource.p))));
    }
    Ref<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p))));
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    // For the supported shape, overwrite a valid slot too: rejection must not leave its old RTV active.
    D3D12_RENDER_TARGET_VIEW_DESC seed{};
    seed.ViewDimension = samples > 1 ? D3D12_RTV_DIMENSION_TEXTURE2DMS : D3D12_RTV_DIMENSION_TEXTURE2D;
    seed.Format = resource_format == DXGI_FORMAT_R8G8B8A8_TYPELESS ? DXGI_FORMAT_R8G8B8A8_UNORM : resource_format;
    device->CreateRenderTargetView(resource.p, &seed, handle);
    device->CreateRenderTargetView(resource.p, view, handle);
    const FLOAT color[] = {1, 1, 1, 1};
    list->ClearRenderTargetView(handle, color, 0, nullptr);
    CHECK(list->Close() == E_INVALIDARG); // The invalid list is never sent to the GPU.
    return true;
}

bool clear_and_read_msaa(ID3D12Device* device, DXGI_FORMAT resource_format,
                         const DXGI_FORMAT* cast_list, UINT32 cast_count,
                         DXGI_FORMAT whole_format, DXGI_FORMAT rect_format, bool partial,
                         bool array, UINT first_layer, UINT layer_count,
                         std::vector<unsigned char>& pixels) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    constexpr UINT width = 5, height = 3;
    const UINT16 layers = array ? 4 : 1;
    Ref<ID3D12Resource> msaa, resolved, readback;
    D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    if (cast_count) {
        Ref<ID3D12Device10> device10;
        CHECK(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device10.p))));
        D3D12_RESOURCE_DESC1 tex1{};
        tex1.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        tex1.Width = width;
        tex1.Height = height;
        tex1.DepthOrArraySize = layers;
        tex1.MipLevels = 1;
        tex1.Format = resource_format;
        tex1.SampleDesc.Count = 4;
        tex1.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        CHECK(SUCCEEDED(device10->CreateCommittedResource3(&default_heap, D3D12_HEAP_FLAG_NONE, &tex1,
            D3D12_BARRIER_LAYOUT_RENDER_TARGET, nullptr, nullptr, cast_count, cast_list,
            IID_PPV_ARGS(&msaa.p))));
    } else {
        D3D12_RESOURCE_DESC tex{};
        tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        tex.Width = width;
        tex.Height = height;
        tex.DepthOrArraySize = layers;
        tex.MipLevels = 1;
        tex.Format = resource_format;
        tex.SampleDesc.Count = 4;
        tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex,
            D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&msaa.p))));
    }
    D3D12_RESOURCE_DESC resolved_desc{};
    resolved_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resolved_desc.Width = width;
    resolved_desc.Height = height;
    resolved_desc.DepthOrArraySize = layers;
    resolved_desc.MipLevels = 1;
    resolved_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    resolved_desc.SampleDesc.Count = 1;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &resolved_desc,
        D3D12_RESOURCE_STATE_RESOLVE_DEST, nullptr, IID_PPV_ARGS(&resolved.p))));

    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> placed(layers);
    UINT64 total = 0;
    device->GetCopyableFootprints(&resolved_desc, 0, layers, 0, placed.data(), nullptr, nullptr, &total);
    D3D12_RESOURCE_DESC buf{};
    buf.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf.Width = total;
    buf.Height = buf.SampleDesc.Count = 1;
    buf.DepthOrArraySize = buf.MipLevels = 1;
    buf.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES rb_heap{D3D12_HEAP_TYPE_READBACK};
    CHECK(SUCCEEDED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &buf,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));

    Ref<ID3D12DescriptorHeap> rtv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap.p))));
    const auto h0 = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    const D3D12_CPU_DESCRIPTOR_HANDLE h1{h0.ptr + device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)};

    D3D12_RENDER_TARGET_VIEW_DESC view{};
    view.ViewDimension = array ? D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY : D3D12_RTV_DIMENSION_TEXTURE2DMS;
    if (array) {
        view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        for (UINT layer = 0; layer < layers; ++layer) {
            view.Texture2DMSArray = {layer, 1};
            device->CreateRenderTargetView(msaa.p, &view, h0);
            const FLOAT seed[] = {FLOAT(layer + 1) * 0.1f, 0.2f, 0.3f, 1.0f};
            list->ClearRenderTargetView(h0, seed, 0, nullptr);
        }
        view.Texture2DMSArray = {first_layer, layer_count};
    }
    view.Format = whole_format;
    device->CreateRenderTargetView(msaa.p, &view, h0);
    view.Format = rect_format;
    device->CreateRenderTargetView(msaa.p, &view, h1);
    const FLOAT full[4] = {0.25f, 0.5f, 0.75f, 0.25f};
    const FLOAT part[4] = {0.0031308f, 0.18f, 0.5f, 0.75f};
    list->ClearRenderTargetView(h0, full, 0, nullptr);
    const D3D12_RECT rect{1, 1, 4, 3};
    if (partial) list->ClearRenderTargetView(h1, part, 1, &rect);

    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {msaa.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE};
    list->ResourceBarrier(1, &b);
    for (UINT layer = 0; layer < layers; ++layer)
        list->ResolveSubresource(resolved.p, layer, msaa.p, layer, DXGI_FORMAT_R8G8B8A8_UNORM);
    b.Transition = {resolved.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = resolved.p;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = readback.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    for (UINT layer = 0; layer < layers; ++layer) {
        src.SubresourceIndex = layer;
        dst.PlacedFootprint = placed[layer];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    Event event;
    CHECK(event.handle);
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, event.handle)));
    CHECK(WaitForSingleObject(event.handle, 5000) == WAIT_OBJECT_0);
    void* mapped = nullptr;
    const D3D12_RANGE range{0, SIZE_T(total)};
    CHECK(SUCCEEDED(readback->Map(0, &range, &mapped)) && mapped);
    pixels.clear();
    for (const auto& footprint : placed) {
        const auto& f = footprint.Footprint;
        for (UINT y = 0; y < f.Height; ++y) {
            const auto* row = static_cast<unsigned char*>(mapped) + footprint.Offset + y * f.RowPitch;
            pixels.insert(pixels.end(), row, row + f.Width * 4);
        }
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}
} // namespace

int main() {
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    unsigned comparisons = 0, mismatches = 0, rejected = 0;
    {
        Ref<ID3D12Device> ours, host;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.conditional_rendering = vk.conditional_rendering;
        info.storage_buffer_8bit = vk.storage_buffer_8bit;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        if (hostwine::host_is_wine()) return 77;  // WARP is the reference for the readbacks
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> warp;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
            FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) ||
            FAILED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
            std::fprintf(stderr, "WARP reference unavailable\n");
            return 1;
        }
        const std::vector<DXGI_FORMAT> predication_formats{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UINT};
        auto formats = predication_formats;
        const auto format_properties = vk.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>(
            "vkGetPhysicalDeviceFormatProperties");
        for (const auto [dxgi_format, vk_format] : {
                 std::pair{DXGI_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R8G8B8A8_SINT},
                 std::pair{DXGI_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_SNORM}}) {
            VkFormatProperties properties{};
            format_properties(vk.physical, vk_format, &properties);
            D3D12_FEATURE_DATA_FORMAT_SUPPORT warp_support{dxgi_format};
            if (FAILED(host->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &warp_support, sizeof(warp_support))))
                return 1;
            if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) &&
                (warp_support.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET)) {
                formats.push_back(dxgi_format);
            } else {
                std::printf("RGBA8 signed format %d: skipped, color attachment support unavailable\n", int(dxgi_format));
            }
        }
        const auto tolerance_for = [](DXGI_FORMAT format) {
            // The signed vectors avoid ambiguous half-step normalized inputs.
            return format == DXGI_FORMAT_R8G8B8A8_UNORM || format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ? 1 : 0;
        };
        const auto compare = [&](DXGI_FORMAT full, DXGI_FORMAT rect, bool partial, bool uav) {
            std::vector<unsigned char> actual, expected;
            if (!clear_and_read(ours.p, full, rect, partial, true, uav, actual) ||
                !clear_and_read(host.p, full, rect, partial, true, uav, expected)) return false;
            ++comparisons;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::size_t x = (i / 4) % 5, y = i / 20;
                const DXGI_FORMAT format = partial && x >= 1 && x < 4 && y >= 1 ? rect : full;
                const int tolerance = tolerance_for(format);
                if (std::abs(int(actual[i]) - int(expected[i])) > tolerance) {
                    std::fprintf(stderr, "full %d rect %d partial %d UAV %d byte %zu: Vulkan %u, WARP %u\n",
                        int(full), int(rect), int(partial), int(uav), i, unsigned(actual[i]), unsigned(expected[i]));
                    ++mismatches;
                    break;
                }
            }
            return true;
        };
        for (const auto format : formats) {
            if (!compare(format, format, false, false)) return 1;
            for (const auto rectangle : formats)
                if (!compare(format, rectangle, true, false)) return 1;
        }
        // An sRGB attachment view must not inherit an unrelated STORAGE usage from the typeless image.
        if (!compare(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UINT, true, true)) return 1;
        // Read every mip and layer, including untouched sentinels and pixels outside the rectangle.
        for (const auto format : formats) {
            for (UINT mip : {0u, 1u, 2u}) {
                for (bool partial : {false, true}) {
                    for (const auto selection : {std::pair{0u, 1u}, std::pair{1u, 2u}, std::pair{3u, 1u}}) {
                        std::vector<unsigned char> actual, expected;
                        if (!clear_and_read(ours.p, format, format, partial, true, false, actual,
                            mip, selection.first, selection.second, true) ||
                        !clear_and_read(host.p, format, format, partial, true, false, expected,
                            mip, selection.first, selection.second, true)) return 1;
                        ++comparisons;
                        if (actual.size() != expected.size()) return 1;
                        std::size_t offset = 0;
                        for (UINT layer = 0; layer < 4; ++layer) {
                            for (UINT level = 0; level < 3; ++level) {
                                const UINT w = std::max(1u, 5u >> level), h = std::max(1u, 3u >> level);
                                for (UINT byte = 0; byte < w * h * 4; ++byte) {
                                    const std::size_t i = offset + byte;
                                    const bool selected = level == mip && layer >= selection.first &&
                                    layer - selection.first < selection.second;
                                    const int tolerance = selected ? tolerance_for(format) : 0;
                                    if (std::abs(int(actual[i]) - int(expected[i])) > tolerance) {
                                        std::fprintf(stderr, "array format %d mip %u partial %d byte %zu: Vulkan %u WARP %u\n",
                                        int(format), mip, int(partial), i, unsigned(actual[i]), unsigned(expected[i]));
                                        return 1;
                                    }
                                }
                                offset += w * h * 4;
                            }
                        }
                    }
                }
            }
        }
        // Non-array TEXTURE2D views must select higher mips too; read back the entire chain.
        for (const auto format : formats) {
            for (UINT mip : {1u, 2u}) {
                for (bool partial : {false, true}) {
                    std::vector<unsigned char> actual, expected;
                    if (!clear_and_read(ours.p, format, format, partial, true, false, actual,
                                        mip, 0, 1, false, true) ||
                        !clear_and_read(host.p, format, format, partial, true, false, expected,
                                        mip, 0, 1, false, true)) return 1;
                    ++comparisons;
                    if (actual.size() != expected.size()) return 1;
                    std::size_t offset = 0;
                    for (UINT level = 0; level < 3; ++level) {
                        const UINT size = std::max(1u, 5u >> level) * std::max(1u, 3u >> level) * 4;
                        const int tolerance = level == mip ? tolerance_for(format) : 0;
                        for (UINT byte = 0; byte < size; ++byte) {
                            if (std::abs(int(actual[offset + byte]) - int(expected[offset + byte])) > tolerance) {
                                std::fprintf(stderr, "2D mip %u format %d partial %d level %u byte %u mismatch\n",
                                             mip, int(format), int(partial), level, byte);
                                return 1;
                            }
                        }
                        offset += size;
                    }
                }
            }
        }
        // Cross formats on the same mip/range, automatic mip chains, and both predicate operations.
        // Every subresource is compared; untouched texels must also match literal UINT sentinels.
        unsigned regressions = 0;
        for (bool partial : {false, true}) {
            for (bool array : {false, true}) {
                for (bool automatic : {false, true}) {
                    for (UINT mip : {0u, 1u, 2u}) {
                        for (const auto whole : predication_formats) {
                            for (const auto part : predication_formats) {
                                if (!partial && part != whole) continue; // No rectangle: one case per whole-view format.
                                for (const auto predicate : {std::pair{false, UINT64_C(0)}, std::pair{true, UINT64_C(0)},
                                     std::pair{true, UINT64_C(1)}, std::pair{true, UINT64_C(1) << 32},
                                     std::pair{true, (UINT64_C(1) << 32) | 7}, std::pair{true, UINT64_MAX}}) {
                                    if (predicate.first && !vk.conditional_rendering) continue;
                                    for (bool inverted : {false, true}) {
                                        if (!predicate.first && inverted) continue;
                                        ClearOptions options{automatic, predicate.first, predicate.second, inverted ? D3D12_PREDICATION_OP_NOT_EQUAL_ZERO
                                                                                         : D3D12_PREDICATION_OP_EQUAL_ZERO};
                                        options.overwrite_predicate = vk.storage_buffer_8bit && automatic && predicate.first &&
                                            (inverted ? predicate.second == 0 : predicate.second != 0);
                                        const UINT first = array ? 1 : 0, layers = array ? 2 : 1;
                                        std::vector<unsigned char> actual, expected;
                                        if (!clear_and_read(ours.p, whole, part, partial, true, false, actual,
                                                            mip, first, layers, array, true, options) ||
                                            !clear_and_read(host.p, whole, part, partial, true, false, expected,
                                                            mip, first, layers, array, true, options)) return 1;
                                        ++comparisons; ++regressions;
                                        if (actual.size() != expected.size()) return 1;
                                        const bool executes = !predicate.first || (inverted ? predicate.second == 0 : predicate.second != 0);
                                        std::size_t offset = 0;
                                        for (UINT layer = 0; layer < (array ? 4u : 1u); ++layer) {
                                            for (UINT level = 0; level < 3; ++level) {
                                                const UINT w = std::max(1u, 5u >> level), h = std::max(1u, 3u >> level);
                                                const unsigned char seed[] = {static_cast<unsigned char>(11 + layer * 3 + level), 31, 73, 127};
                                                const bool selected = executes && level == mip && layer >= first && layer - first < layers;
                                                for (UINT y = 0; y < h; ++y) {
                                                    for (UINT x = 0; x < w; ++x) {
                                                        const bool in_rect = partial && x >= (mip == 2 ? 0u : 1u);
                                                        const DXGI_FORMAT written = in_rect ? part : whole;
                                                        const int tolerance = selected && written != DXGI_FORMAT_R8G8B8A8_UINT ? 1 : 0;
                                                        for (UINT channel = 0; channel < 4; ++channel) {
                                                            const std::size_t i = offset + (y * w + x) * 4 + channel;
                                                            if ((!selected && (actual[i] != seed[channel] || expected[i] != seed[channel])) ||
                                                                std::abs(int(actual[i]) - int(expected[i])) > tolerance) {
                                                                std::fprintf(stderr, "subresource regression partial %d array %d auto %d mip %u formats %d/%d predicate %llu inverted %d layer %u level %u texel %u,%u channel %u: Vulkan %u WARP %u\n",
                                                                    int(partial), int(array), int(automatic), mip, int(whole), int(part), static_cast<unsigned long long>(predicate.second), int(inverted), layer, level, x, y, channel,
                                                                    unsigned(actual[i]), unsigned(expected[i]));
                                                                return 1;
                                                            }
                                                        }
                                                    }
                                                }
                                                offset += w * h * 4;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        std::printf("typeless subresource regressions: %u mixed-format/automatic-mip/predicate cases; conditional rendering %s\n",
                    regressions, vk.conditional_rendering ? "enabled" : "unavailable (predicated cases skipped)");
        if (mismatches) return 1;
        const auto rejects = [&](DXGI_FORMAT resource_format, const D3D12_RENDER_TARGET_VIEW_DESC* view,
                                 UINT16 mips = 1, UINT16 layers = 1, UINT samples = 1,
                                 const DXGI_FORMAT* cast_list = nullptr, UINT32 cast_count = 0) {
            if (!rejects_view(ours.p, resource_format, view, mips, layers, samples, cast_list, cast_count))
                return false;
            ++rejected;
            return true;
        };
        D3D12_RENDER_TARGET_VIEW_DESC invalid{};
        invalid.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        for (const auto format : {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R16G16B16A16_FLOAT}) {
            invalid.Format = format;
            if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid)) return 1;
        }
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, nullptr)) return 1;
        for (const auto format : {DXGI_FORMAT_R8G8B8A8_SINT, DXGI_FORMAT_R8G8B8A8_SNORM, DXGI_FORMAT_R8G8B8A8_UINT}) {
            invalid.Format = format;
            if (!rejects(DXGI_FORMAT_R8G8B8A8_UNORM, &invalid)) return 1;
        }
        invalid.Texture2D.MipSlice = 1;
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid)) return 1;
        invalid.Texture2D.MipSlice = 0;
        invalid.Texture2D.PlaneSlice = 1;
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid)) return 1;
        invalid.Texture2D.PlaneSlice = 0;
        invalid.Texture2D.MipSlice = 2;
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid, 2, 1)) return 1;
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid, 1, 2)) return 1;
        invalid.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
        invalid.Texture2DArray = {0, 1, 1, 0};
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid)) return 1;
        // ArraySize -1 from a slice of the array is every slice left (d3d12_view_slices_test); past the array it is
        // refused.
        for (const auto range : {std::pair{4u, UINT_MAX}, std::pair{0u, 0u}, std::pair{4u, 1u}, std::pair{3u, 2u}}) {
            invalid.Texture2DArray = {1, range.first, range.second, 0};
            if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid, 3, 4)) return 1;
        }
        for (UINT mip : {3u, UINT_MAX}) {
            invalid.Texture2DArray = {mip, 0, 4, 0};
            if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid, 3, 4)) return 1;
        }
        for (const auto range : {std::pair{UINT_MAX, 2u}, std::pair{UINT_MAX, UINT_MAX}}) {
            invalid.Texture2DArray = {0, range.first, range.second, 0};
            if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid, 3, 4)) return 1;
        }
        // A 4x4 resource with MipLevels=0 resolves to three mips; mip 3 is out of bounds.
        invalid.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        for (UINT mip : {3u, UINT_MAX}) {
            invalid.Texture2D = {mip, 0};
            if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid, 0, 1)) return 1;
        }
        invalid.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
        invalid.Texture2DArray = {1, 1, 2, 1};
        if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &invalid, 3, 4)) return 1;

        // 4x MSAA typeless and castable RGBA8 RTVs (TEXTURE2DMS / TEXTURE2DMSARRAY) compared against WARP.
        {
            const DXGI_FORMAT srgb_list[] = {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB};
            // A host without relaxed format casting refuses a castable-format list (in-box WARP 10.0.22621.5415 on
            // Windows 11 23H2 reports RelaxedFormatCastingSupported = 0 and CreateCommittedResource3 returns
            // E_INVALIDARG). There the castable RGBA8 resource is compared with an RGBA8_TYPELESS one on the host:
            // the same views of the same bytes, so the expected readback is the same.
            bool host_casts = false;
            {
                Ref<ID3D12Device10> host10;
                D3D12_FEATURE_DATA_D3D12_OPTIONS12 options12{};
                host_casts = SUCCEEDED(host->QueryInterface(IID_PPV_ARGS(&host10.p))) &&
                             SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options12,
                                                                 sizeof(options12))) &&
                             options12.RelaxedFormatCastingSupported;
            }
            const auto compare_msaa = [&](DXGI_FORMAT resource_format, const DXGI_FORMAT* cast_list,
                                          UINT32 cast_count, DXGI_FORMAT full, DXGI_FORMAT rect, bool partial,
                                          bool array, UINT first_layer, UINT layer_count) {
                std::vector<unsigned char> actual, expected;
                const bool host_typeless = cast_count && !host_casts;
                if (!clear_and_read_msaa(ours.p, resource_format, cast_list, cast_count, full, rect, partial, array,
                                         first_layer, layer_count, actual) ||
                    !clear_and_read_msaa(host.p, host_typeless ? DXGI_FORMAT_R8G8B8A8_TYPELESS : resource_format,
                                         host_typeless ? nullptr : cast_list, host_typeless ? 0 : cast_count, full,
                                         rect, partial, array, first_layer, layer_count, expected))
                    return false;
                ++comparisons;
                if (actual.size() != expected.size()) return false;
                for (std::size_t i = 0; i < actual.size(); ++i) {
                    if (std::abs(int(actual[i]) - int(expected[i])) > 1) {
                        std::fprintf(stderr, "MSAA RTV mismatch fmt %d full %d rect %d array %d byte %zu: ours %u WARP %u\n",
                                     int(resource_format), int(full), int(rect), int(array), i, unsigned(actual[i]),
                                     unsigned(expected[i]));
                        ++mismatches;
                        return false;
                    }
                }
                return true;
            };
            if (!compare_msaa(DXGI_FORMAT_R8G8B8A8_TYPELESS, nullptr, 0, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                              DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, false, false, 0, 1))
                return 1;
            if (!compare_msaa(DXGI_FORMAT_R8G8B8A8_TYPELESS, nullptr, 0, DXGI_FORMAT_R8G8B8A8_UNORM,
                              DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, true, false, 0, 1))
                return 1;
            if (!compare_msaa(DXGI_FORMAT_R8G8B8A8_TYPELESS, nullptr, 0, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                              DXGI_FORMAT_R8G8B8A8_UNORM, true, true, 1, 2))
                return 1;
            if (!compare_msaa(DXGI_FORMAT_R8G8B8A8_UNORM, srgb_list, 2, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                              DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, false, false, 0, 1))
                return 1;
            if (!compare_msaa(DXGI_FORMAT_R8G8B8A8_UNORM, srgb_list, 2, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                              DXGI_FORMAT_R8G8B8A8_UNORM, true, true, 1, 2))
                return 1;
            if (!host_casts)
                std::printf("typeless RTV: host has no relaxed format casting; the 2 castable 4x MSAA cases were "
                            "compared with an RGBA8_TYPELESS resource on the host\n");
            // MSAA vs 1x view-dimension mismatches, out-of-range MSAA array slices, and unlisted castable formats.
            D3D12_RENDER_TARGET_VIEW_DESC bad_msaa{};
            bad_msaa.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            bad_msaa.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &bad_msaa, 1, 1, 4)) return 1;
            bad_msaa.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
            if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &bad_msaa, 1, 1, 1)) return 1;
            bad_msaa.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
            bad_msaa.Texture2DMSArray = {3, 2};
            if (!rejects(DXGI_FORMAT_R8G8B8A8_TYPELESS, &bad_msaa, 1, 4, 4)) return 1;
            bad_msaa.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
            bad_msaa.Format = DXGI_FORMAT_R8G8B8A8_UINT;
            if (!rejects(DXGI_FORMAT_R8G8B8A8_UNORM, &bad_msaa, 1, 1, 4, srgb_list, 2)) return 1;
        }
    }
    if (vk.validation_errors()) return 1;
    std::printf("typeless RTV: %u WARP readbacks matched, descriptor snapshots and format-cache order checked; %u unsupported/invalid views rejected\n",
                comparisons, rejected);
    return 0;
}
