
// Compare render-target and depth-stencil clears against native Windows D3D12/WARP, through D3D12 clear/copy/Map on
// both sides. Covers UNORM/SNORM/FLOAT/UINT/SINT render targets and depth/stencil, full and rectangle clears, and RTVs
// through castable-format lists in the other channel order.
// No game data or shaders. Run explicitly (needs Vulkan hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <array>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cstdlib>
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

// With `cast` set, the texture lists it as its castable format (CreateCommittedResource3) and the clears go through an
// RTV in it.
bool clear_and_read(ID3D12Device* device, DXGI_FORMAT format, unsigned bytes, const FLOAT* full, const FLOAT* part,
                    std::vector<unsigned char>& pixels, bool volume = false, DXGI_FORMAT cast = DXGI_FORMAT_UNKNOWN) {
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

    Ref<ID3D12Resource> image, readback;
    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = volume ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = volume ? 8 : 5;
    texture.Height = volume ? 6 : 3;
    texture.DepthOrArraySize = volume ? 12 : 1;
    texture.MipLevels = volume ? 2 : 1;
    texture.Format = format;
    texture.SampleDesc.Count = 1;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (cast == DXGI_FORMAT_UNKNOWN) {
        CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &texture,
            D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&image.p))));
    } else {
        Ref<ID3D12Device10> device10;
        CHECK(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device10.p))));
        D3D12_RESOURCE_DESC1 texture1{};
        texture1.Dimension = texture.Dimension;
        texture1.Width = texture.Width;
        texture1.Height = texture.Height;
        texture1.DepthOrArraySize = texture.DepthOrArraySize;
        texture1.MipLevels = texture.MipLevels;
        texture1.Format = texture.Format;
        texture1.SampleDesc = texture.SampleDesc;
        texture1.Flags = texture.Flags;
        CHECK(SUCCEEDED(device10->CreateCommittedResource3(&heap_properties, D3D12_HEAP_FLAG_NONE, &texture1,
            D3D12_BARRIER_LAYOUT_RENDER_TARGET, nullptr, nullptr, 1, &cast, IID_PPV_ARGS(&image.p))));
    }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed[2]{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&texture, 0, texture.MipLevels, 0, placed, nullptr, nullptr, &total);
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
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    if (cast == DXGI_FORMAT_UNKNOWN) {
        device->CreateRenderTargetView(image.p, nullptr, handle);
    } else {
        D3D12_RENDER_TARGET_VIEW_DESC listed{};
        listed.Format = cast;
        listed.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device->CreateRenderTargetView(image.p, &listed, handle);
    }
    list->ClearRenderTargetView(handle, full, 0, nullptr);
    const D3D12_RECT rect{1, 1, 4, 3};
    if (volume) {
        D3D12_RENDER_TARGET_VIEW_DESC view{};
        view.Format = format;
        view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
        const auto clear = [&](UINT first, UINT count, const FLOAT* color, const D3D12_RECT* area) {
            view.Texture3D = {1, first, count};
            device->CreateRenderTargetView(image.p, &view, handle);
            list->ClearRenderTargetView(handle, color, area ? 1 : 0, area);
        };
        const FLOAT blue[4]{0, 0, 1, 1};
        const FLOAT yellow[4]{1, 1, 0, 0};
        clear(0, UINT_MAX, blue, nullptr);
        clear(1, 2, part, nullptr);
        clear(2, UINT_MAX, yellow, &rect);
    } else {
        list->ClearRenderTargetView(handle, part, 1, &rect);
    }
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
    for (UINT mip = 0; mip < texture.MipLevels; ++mip) {
        src.SubresourceIndex = mip;
        dst.PlacedFootprint = placed[mip];
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
    for (UINT mip = 0; mip < texture.MipLevels; ++mip) {
        const auto& fp = placed[mip].Footprint;
        for (UINT row = 0; row < fp.Height * fp.Depth; ++row) {
            const auto* start = static_cast<unsigned char*>(mapped) + placed[mip].Offset + row * fp.RowPitch;
            pixels.insert(pixels.end(), start, start + fp.Width * bytes);
        }
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}

// Clears a depth-stencil target fully and a rectangle, then reads the depth plane (and the stencil plane, when the
// format has one) back. depth_bytes is the depth plane's bytes per pixel (2 for D16, 4 for D32); has_stencil selects
// the separate stencil plane (subresource 1) of a combined format. full_flags / part_flags override the clear flags
// of the two clears (0: DEPTH, plus STENCIL when the format has a stencil plane).
bool clear_depth_and_read(ID3D12Device* device, DXGI_FORMAT format, unsigned depth_bytes, bool has_stencil,
                          FLOAT depth_full, UINT8 stencil_full, FLOAT depth_part, UINT8 stencil_part,
                          std::vector<unsigned char>& depth_pixels, std::vector<unsigned char>& stencil_pixels,
                          D3D12_CLEAR_FLAGS full_flags = D3D12_CLEAR_FLAGS(0),
                          D3D12_CLEAR_FLAGS part_flags = D3D12_CLEAR_FLAGS(0)) {
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
    Ref<ID3D12Resource> image, readback;
    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = width;
    texture.Height = height;
    texture.DepthOrArraySize = texture.MipLevels = 1;
    texture.Format = format;
    texture.SampleDesc.Count = 1;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &texture,
        D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr, IID_PPV_ARGS(&image.p))));
    const UINT subresources = has_stencil ? 2u : 1u;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed[2]{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&texture, 0, subresources, 0, placed, nullptr, nullptr, &total);
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
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(image.p, nullptr, handle);
    const D3D12_CLEAR_FLAGS flags =
        static_cast<D3D12_CLEAR_FLAGS>((has_stencil ? D3D12_CLEAR_FLAG_STENCIL : 0) | D3D12_CLEAR_FLAG_DEPTH);
    list->ClearDepthStencilView(handle, full_flags ? full_flags : flags, depth_full, stencil_full, 0, nullptr);
    const D3D12_RECT rect{1, 1, 4, 3};
    list->ClearDepthStencilView(handle, part_flags ? part_flags : flags, depth_part, stencil_part, 1, &rect);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = image.p;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    for (UINT i = 0; i < subresources; ++i) {
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = image.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = i;
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = placed[i];
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
    const auto* base = static_cast<unsigned char*>(mapped);
    depth_pixels.resize(width * height * depth_bytes);
    for (UINT y = 0; y < height; ++y)
        std::memcpy(depth_pixels.data() + y * width * depth_bytes,
                    base + placed[0].Offset + y * placed[0].Footprint.RowPitch, width * depth_bytes);
    if (has_stencil) {
        stencil_pixels.resize(width * height);
        for (UINT y = 0; y < height; ++y)
            std::memcpy(stencil_pixels.data() + y * width,
                        base + placed[1].Offset + y * placed[1].Footprint.RowPitch, width);
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}

bool check_shared_exponent_target(ID3D12Device* device) {
    constexpr auto format = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    const FLOAT full[]{1, 2, 4, 0}, rect[]{4, 2, 1, 0};
    std::vector<unsigned char> pixels;
    CHECK(clear_and_read(device, format, 4, full, rect, pixels));
    for (unsigned y = 0; y < 3; ++y) {
        for (unsigned x = 0; x < 5; ++x) {
            uint32_t got;
            std::memcpy(&got, pixels.data() + (5 * y + x) * 4, 4);
            const uint32_t expected = (y >= 1 && x >= 1 && x < 4) ? 0x91010100u : 0x94010040u;
            if (got != expected) {
                std::fprintf(stderr, "RGB9E5 (%u,%u): got %08x, expected %08x\n", x, y, got, expected);
                return false;
            }
        }
    }
    std::puts("RGB9E5 RTV: full and rectangle clears match all 15 packed texels");
    return true;
}

bool check_volume_rtvs(ID3D12Device* host, ID3D12Device* ours) {
    const FLOAT red[4]{1, 0, 0, 1}, green[4]{0, 1, 0, 1};
    std::vector<unsigned char> expected;
    // Mip 0 stays red. Mip 1 starts blue; W 1 and 2 are green, with a yellow rectangle on W 2 through 5.
    for (UINT mip = 0; mip < 2; ++mip)
        for (UINT z = 0; z < (12u >> mip); ++z)
            for (UINT y = 0; y < (6u >> mip); ++y)
                for (UINT x = 0; x < (8u >> mip); ++x) {
                    if (mip == 0) expected.insert(expected.end(), {255, 0, 0, 255});
                    else if (z >= 2 && x >= 1 && y >= 1) expected.insert(expected.end(), {255, 255, 0, 0});
                    else if (z == 1 || z == 2) expected.insert(expected.end(), {0, 255, 0, 255});
                    else expected.insert(expected.end(), {0, 0, 255, 255});
                }
    const auto verify = [&](ID3D12Device* device, const char* backend) {
        std::vector<unsigned char> pixels;
        if (!clear_and_read(device, DXGI_FORMAT_R8G8B8A8_UNORM, 4, red, green, pixels, true)) return false;
        CHECK(pixels.size() == expected.size());
        size_t wrong = 0;
        for (size_t offset = 0; offset < pixels.size(); offset += 4)
            if (std::memcmp(pixels.data() + offset, expected.data() + offset, 4)) ++wrong;
        if (wrong) std::fprintf(stderr, "%s: %zu of %zu volume texels differ\n", backend, wrong, pixels.size() / 4);
        return wrong == 0;
    };
    if (!verify(host, "WARP") || !verify(ours, "Vulkan")) return false;
    std::printf("3D RTVs: %zu RGBA texels match WARP and expected slice colours\n", expected.size() / 4);
    return true;
}

// A texture with a castable list takes an RTV in its own format and in the listed one, and none in a format the
// list doesn't name (the descriptor stays empty): in the other channel order, across B8G8R8X8 and B8G8R8A8, or the
// texture's own sRGB format. Typed R8G8B8A8, B8G8R8A8 and B8G8R8X8 textures refuse alike.
bool check_unlisted_rtvs(ID3D12Device* device) {
    Ref<ID3D12Device10> device10;
    CHECK(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device10.p))));
    const D3D12_HEAP_PROPERTIES heap_properties{D3D12_HEAP_TYPE_DEFAULT};
    Ref<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    struct Listed {
        DXGI_FORMAT resource, listed;
        VkFormat listed_view;
        std::array<DXGI_FORMAT, 3> unlisted;
    };
    const Listed textures[] = {
        {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM,
         {DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8X8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB}},
        {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
         {DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8X8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB}},
        {DXGI_FORMAT_B8G8R8X8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM,
         {DXGI_FORMAT_B8G8R8X8_UNORM_SRGB, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM}},
    };
    const leonida::TargetView none{};
    unsigned refused = 0;
    for (const Listed& t : textures) {
        D3D12_RESOURCE_DESC1 texture{};
        texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = texture.Height = 4;
        texture.DepthOrArraySize = 1;
        texture.MipLevels = 1;
        texture.Format = t.resource;
        texture.SampleDesc.Count = 1;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        Ref<ID3D12Resource> image;
        CHECK(SUCCEEDED(device10->CreateCommittedResource3(&heap_properties, D3D12_HEAP_FLAG_NONE, &texture,
            D3D12_BARRIER_LAYOUT_RENDER_TARGET, nullptr, nullptr, 1, &t.listed, IID_PPV_ARGS(&image.p))));
        const auto rtv = [&](DXGI_FORMAT format) {
            D3D12_RENDER_TARGET_VIEW_DESC desc{};
            desc.Format = format;
            desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            std::memset(reinterpret_cast<void*>(handle.ptr), 0x55, sizeof(leonida::TargetView));
            device->CreateRenderTargetView(image.p, &desc, handle);
            leonida::TargetView view{};
            std::memcpy(&view, reinterpret_cast<const void*>(handle.ptr), sizeof(view));
            return view;
        };
        CHECK(rtv(t.resource).resource);
        const leonida::TargetView listed_view = rtv(t.listed);
        CHECK(listed_view.resource && listed_view.format == static_cast<leonida::ColorViewFormat>(t.listed_view));
        for (const DXGI_FORMAT unlisted : t.unlisted) {
            const leonida::TargetView view = rtv(unlisted);
            if (std::memcmp(&view, &none, sizeof(view)) != 0) {
                std::fprintf(stderr, "castable RTVs: format %d {%d} took an RTV in unlisted format %d\n",
                             int(t.resource), int(t.listed), int(unlisted));
                return false;
            }
            ++refused;
        }
    }
    std::printf("castable RTVs: %zu textures take their own and the listed format, and refuse %u formats their "
                "lists don't name\n", std::size(textures), refused);
    return true;
}
} // namespace

int main(int argc, char** argv) {
    const bool volume_only = argc == 2 && std::strcmp(argv[1], "--volume") == 0;
    const bool rgb9e5_only = argc == 2 && std::strcmp(argv[1], "--rgb9e5") == 0;
    if (argc != 1 && !volume_only && !rgb9e5_only) {
        std::fprintf(stderr, "usage: d3d12_clear_test [--volume|--rgb9e5]\n");
        return 2;
    }
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    const unsigned errors_before = vk.validation_errors();
    unsigned cases_checked = 0;
    const auto compare = [&]() -> int {
        Ref<ID3D12Device> ours, host;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        if (rgb9e5_only) {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{DXGI_FORMAT_R9G9B9E5_SHAREDEXP};
            if (FAILED(ours->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs)))) return 1;
            VkFormatProperties properties{};
            auto query = vk.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties");
            query(vk.physical, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, &properties);
            const bool native_rtv = properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
            if (native_rtv != bool(fs.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET)) {
                std::fprintf(stderr, "RGB9E5 render-target support disagrees with the native attachment capability\n");
                return 1;
            }
            if (!(fs.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET)) {
                std::puts("RGB9E5 render targets unavailable on this adapter");
                return 77;
            }
            return check_shared_exponent_target(ours.p) ? 0 : 1;
        }
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> warp;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
            FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) ||
            FAILED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
            std::fprintf(stderr, "WARP reference unavailable: differential test cannot run\n");
            return 1;
        }
        if (!check_volume_rtvs(host.p, ours.p)) return 1;
        if (volume_only) return 0;
        if (!check_unlisted_rtvs(ours.p)) return 1;
        const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
        // Integer-format values: negatives, fractions, NaN and overflow exercise the numeric float->int conversion.
        const std::array<float, 4> int_values[] = {
            {1, 2, 3, 4}, {1.5f, 2.5f, 3.5f, 4.5f}, {-1, -0.0f, 254.5f, 255.5f},
            {256, 65536, 16777216, 4294967296.0f}, {nan, inf, -inf, 0.5f},
            {0.499f, 0.501f, 1.499f, 1.501f}, {4294967040.0f, std::numeric_limits<float>::max(), 255, 0},
            // Component-range boundaries for the 8-, 16-, 10- and 2-bit formats and the signed 32-bit limits.
            {127.5f, 128, -128.5f, -129}, {32767.5f, 32768, -32768.5f, -32769},
            {65535.5f, 65536, 1023.5f, 1024}, {3.5f, 4, 2147483520.0f, -2147483648.0f}
        };
        // Float-valued formats (UNORM/SNORM/FLOAT); out-of-range values exercise clamping. NaN/inf and SRGB are
        // omitted: NaN/inf conversion is driver-defined, and the linear->sRGB encoding rounds differently at the .5
        // boundary on WARP vs Vulkan (a 1-ulp driver difference the shim cannot control).
        const std::array<float, 4> float_values[] = {
            {0, 0.25f, 0.5f, 1}, {1, 1, 1, 1}, {0.5f, 0.5f, 0.5f, 0.5f}, {-0.5f, 0, 1.5f, 2}
        };
        struct ColorFormat {
            DXGI_FORMAT format; const char* name; unsigned bytes; const std::array<float, 4>* values; unsigned count;
        };
        const unsigned nf = unsigned(std::size(float_values)), ni = unsigned(std::size(int_values));
        const ColorFormat color_formats[] = {
            {DXGI_FORMAT_R8G8B8A8_UNORM, "R8G8B8A8_UNORM", 4, float_values, nf},
            {DXGI_FORMAT_R8G8B8A8_SNORM, "R8G8B8A8_SNORM", 4, float_values, nf},
            // B8G8R8X8 is stored in a BGRA8 image; B8G8R8A8 next to it shows what the X byte changes.
            {DXGI_FORMAT_B8G8R8A8_UNORM, "B8G8R8A8_UNORM", 4, float_values, nf},
            {DXGI_FORMAT_B8G8R8X8_UNORM, "B8G8R8X8_UNORM", 4, float_values, nf},
            {DXGI_FORMAT_R32G32B32A32_FLOAT, "R32G32B32A32_FLOAT", 16, float_values, nf},
            {DXGI_FORMAT_R8G8_UINT, "R8G8_UINT", 2, int_values, ni},
            {DXGI_FORMAT_R8G8B8A8_UINT, "R8G8B8A8_UINT", 4, int_values, ni},
            {DXGI_FORMAT_R16G16_UINT, "R16G16_UINT", 4, int_values, ni},
            {DXGI_FORMAT_R16G16B16A16_UINT, "R16G16B16A16_UINT", 8, int_values, ni},
            {DXGI_FORMAT_R32_UINT, "R32_UINT", 4, int_values, ni},
            {DXGI_FORMAT_R32G32_UINT, "R32G32_UINT", 8, int_values, ni},
            {DXGI_FORMAT_R32G32B32A32_UINT, "R32G32B32A32_UINT", 16, int_values, ni},
            {DXGI_FORMAT_R10G10B10A2_UINT, "R10G10B10A2_UINT", 4, int_values, ni},
            {DXGI_FORMAT_R8G8_SINT, "R8G8_SINT", 2, int_values, ni},
            {DXGI_FORMAT_R8G8B8A8_SINT, "R8G8B8A8_SINT", 4, int_values, ni},
            {DXGI_FORMAT_R16G16_SINT, "R16G16_SINT", 4, int_values, ni},
            {DXGI_FORMAT_R16G16B16A16_SINT, "R16G16B16A16_SINT", 8, int_values, ni},
            {DXGI_FORMAT_R32_SINT, "R32_SINT", 4, int_values, ni},
            {DXGI_FORMAT_R32G32_SINT, "R32G32_SINT", 8, int_values, ni},
            {DXGI_FORMAT_R32G32B32A32_SINT, "R32G32B32A32_SINT", 16, int_values, ni},
        };
        unsigned format_failures = 0;
        for (const auto& cf : color_formats) {
            for (unsigned i = 0; i < cf.count; ++i) {
                // Different colours outside and inside the rectangle exercise both Vulkan clear paths.
                const auto& full = cf.values[i];
                const auto& part = cf.values[(i + 1) % cf.count];
                std::vector<unsigned char> actual, expected;
                if (!clear_and_read(ours.p, cf.format, cf.bytes, full.data(), part.data(), actual) ||
                    !clear_and_read(host.p, cf.format, cf.bytes, full.data(), part.data(), expected)) {
                    std::fprintf(stderr, "%s case %u: clear or readback failed\n", cf.name, i);
                    return 1;
                }
                if (actual != expected) {
                    for (std::size_t b = 0; b < actual.size(); ++b) {
                        if (actual[b] != expected[b]) {
                            std::fprintf(stderr, "%s case %u byte %zu: Vulkan %02x, WARP %02x\n",
                                cf.name, i, b, unsigned(actual[b]), unsigned(expected[b]));
                            break;
                        }
                    }
                    ++format_failures;
                    break; // first mismatch per format; keep checking the other formats
                }
                ++cases_checked;
            }
        }
        // RTVs through a castable-format list in another channel order or family, or a typed BGRA8 texture's sRGB
        // view: the clear is stored in the view's channel order and encoding, as WARP and AMD's D3D12 driver store
        // it. An X8 view leaves its X byte undefined (WARP and AMD's driver differ), so it isn't compared. The bytes
        // are compared within 1: drivers round the 0.5 tie differently (WARP 128, NVIDIA 127), and Intel's encodes
        // sRGB 1 off WARP in places (its own D3D12 driver stores the same bytes as the layer there); a wrong channel
        // order or a missing sRGB encoding is many steps off.
        // A host that can't take a castable-format list (Windows 10's d3d12.dll has no ID3D12Device10; the in-box WARP
        // 10.0.22621.5415 on Windows 11 23H2 reports RelaxedFormatCastingSupported = 0 and CreateCommittedResource3
        // fails) clears a texture in the view's own format instead: the bytes are stored in the view's channel order
        // and encoding either way, so the expected readback is the same. The layer still goes through the list.
        bool host_casts = false;
        {
            Ref<ID3D12Device10> host10;
            D3D12_FEATURE_DATA_D3D12_OPTIONS12 options12{};
            host_casts = SUCCEEDED(host.p->QueryInterface(IID_PPV_ARGS(&host10.p))) &&
                         SUCCEEDED(host.p->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options12,
                                                             sizeof(options12))) &&
                         options12.RelaxedFormatCastingSupported;
        }
        if (!host_casts)
            std::printf("castable-format clears: host has no relaxed format casting; WARP clears the view's format\n");
        struct CastFormat { DXGI_FORMAT resource, view; const char* name; };
        const CastFormat cast_formats[] = {
            {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, "R8G8B8A8_UNORM{B8G8R8A8_UNORM}"},
            {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, "B8G8R8A8_UNORM{R8G8B8A8_UNORM}"},
            {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, "R8G8B8A8_UNORM{B8G8R8A8_UNORM_SRGB}"},
            {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, "B8G8R8A8_UNORM{R8G8B8A8_UNORM_SRGB}"},
            {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, "B8G8R8A8_UNORM{B8G8R8A8_UNORM_SRGB}"},
            {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8X8_UNORM, "B8G8R8A8_UNORM{B8G8R8X8_UNORM}"},
            {DXGI_FORMAT_B8G8R8X8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, "B8G8R8X8_UNORM{B8G8R8A8_UNORM}"},
            {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8X8_UNORM, "R8G8B8A8_UNORM{B8G8R8X8_UNORM}"},
        };
        for (const auto& cf : cast_formats) {
            for (unsigned i = 0; i < nf; ++i) {
                const auto& full = float_values[i];
                const auto& part = float_values[(i + 1) % nf];
                std::vector<unsigned char> actual, expected;
                if (!clear_and_read(ours.p, cf.resource, 4, full.data(), part.data(), actual, false, cf.view) ||
                    !(host_casts
                          ? clear_and_read(host.p, cf.resource, 4, full.data(), part.data(), expected, false, cf.view)
                          : clear_and_read(host.p, cf.view, 4, full.data(), part.data(), expected))) {
                    std::fprintf(stderr, "%s case %u: clear or readback failed\n", cf.name, i);
                    ++format_failures;
                    break;
                }
                if (cf.view == DXGI_FORMAT_B8G8R8X8_UNORM) {
                    for (std::size_t b = 3; b < actual.size() && b < expected.size(); b += 4) actual[b] = expected[b] = 0;
                }
                bool close = actual.size() == expected.size();
                std::size_t off = 0;
                for (; close && off < actual.size(); ++off)
                    if (std::abs(int(actual[off]) - int(expected[off])) > 1) { close = false; break; }
                if (!close) {
                    if (actual.size() == expected.size())
                        std::fprintf(stderr, "%s case %u byte %zu: Vulkan %02x, WARP %02x\n",
                            cf.name, i, off, unsigned(actual[off]), unsigned(expected[off]));
                    else
                        std::fprintf(stderr, "%s case %u: readback sizes differ\n", cf.name, i);
                    ++format_failures;
                    break;
                }
                ++cases_checked;
            }
        }
        // Color mismatches are reported at the end so depth/stencil and the validation counter still run.
        // Depth (and stencil) clears, full and rectangle. Depth is [0,1]; stencil is a byte.
        struct DepthFormat { DXGI_FORMAT format; unsigned depth_bytes; bool has_stencil; };
        const DepthFormat depth_formats[] = {
            {DXGI_FORMAT_D32_FLOAT, 4, false},
            {DXGI_FORMAT_D16_UNORM, 2, false},
            {DXGI_FORMAT_D32_FLOAT_S8X24_UINT, 4, true},
        };
        const float depths[] = {0, 0.25f, 0.5f, 1};
        for (const auto& df : depth_formats) {
            for (unsigned i = 0; i < std::size(depths); ++i) {
                const float full = depths[i], part = depths[(i + 1) % std::size(depths)];
                const std::uint8_t stencil_full = std::uint8_t(i * 0x40 + 0x11);
                const std::uint8_t stencil_part = std::uint8_t(i * 0x40 + 0x22);
                std::vector<unsigned char> actual, expected, actual_stencil, expected_stencil;
                if (!clear_depth_and_read(ours.p, df.format, df.depth_bytes, df.has_stencil, full, stencil_full, part,
                                          stencil_part, actual, actual_stencil) ||
                    !clear_depth_and_read(host.p, df.format, df.depth_bytes, df.has_stencil, full, stencil_full, part,
                                          stencil_part, expected, expected_stencil))
                    return 1;
                if (actual != expected || actual_stencil != expected_stencil) {
                    std::size_t b = 0;
                    while (b < actual.size() && b < expected.size() && actual[b] == expected[b]) ++b;
                    if (b < actual.size() && b < expected.size())
                        std::fprintf(stderr, "depth format %d case %u byte %zu: Vulkan %02x, WARP %02x\n",
                            int(df.format), i, b, unsigned(actual[b]), unsigned(expected[b]));
                    else
                        std::fprintf(stderr, "depth format %d case %u: stencil readback mismatch\n", int(df.format), i);
                    ++format_failures;
                    break; // first mismatch per format; keep checking the other formats
                }
                ++cases_checked;
            }
        }
        // D3D-437: STENCIL on a depth-only D32_FLOAT view. The PC runtime (WARP) ignores the absent plane: DEPTH |
        // STENCIL clears the depth and STENCIL alone clears nothing; the layer used to drop the whole list instead.
        struct PlaneCase { D3D12_CLEAR_FLAGS full, part; float depth_full, depth_part; };
        const PlaneCase plane_cases[] = {
            {D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, D3D12_CLEAR_FLAG_DEPTH, 0.5f, 0.875f},
            {D3D12_CLEAR_FLAG_DEPTH, D3D12_CLEAR_FLAG_STENCIL, 0.25f, 0.75f},
            {D3D12_CLEAR_FLAG_DEPTH, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.375f, 1.0f},
        };
        for (unsigned i = 0; i < std::size(plane_cases); ++i) {
            const auto& pc = plane_cases[i];
            std::vector<unsigned char> actual, expected, no_stencil_a, no_stencil_e;
            if (!clear_depth_and_read(ours.p, DXGI_FORMAT_D32_FLOAT, 4, false, pc.depth_full, 0x11, pc.depth_part, 0x22,
                                      actual, no_stencil_a, pc.full, pc.part) ||
                !clear_depth_and_read(host.p, DXGI_FORMAT_D32_FLOAT, 4, false, pc.depth_full, 0x11, pc.depth_part, 0x22,
                                      expected, no_stencil_e, pc.full, pc.part))
                return 1;
            if (actual != expected) {
                std::fprintf(stderr, "D32_FLOAT stencil-flag case %u: depth readback differs from WARP\n", i);
                ++format_failures;
                continue;
            }
            ++cases_checked;
        }
        if (format_failures) {
            std::fprintf(stderr, "d3d12_clear_test: %u color/depth format(s) differ from WARP\n", format_failures);
            return 1;
        }
        return 0;
    }; // Destroy the D3D12 objects before checking the Vulkan validation counter.
    const int compare_failed = compare();
    const auto validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors)
        std::fprintf(stderr, "d3d12_clear_test: %llu new Vulkan validation error(s)\n",
            static_cast<unsigned long long>(validation_errors));
    if (validation_errors) return 1;
    if (compare_failed) return compare_failed;
    if (volume_only || rgb9e5_only) return 0;
    std::printf("RTV and DSV clears: %u full/rectangle readbacks matched WARP\n", cases_checked);
    return 0;
}
