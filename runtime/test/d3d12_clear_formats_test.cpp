// Clears compared with native Windows D3D12/WARP across the format classes (D3D-45): ClearRenderTargetView on
// UNORM, SNORM, SRGB, FLOAT, UINT and SINT formats and ClearDepthStencilView on every depth format, each with and
// without rectangles. Every case clears, copies the result into a readback buffer through D3D12 and compares the
// bytes with what WARP wrote. Extends d3d12_uint_clear_test.cpp (UINT only).
// No game data or shaders. Run explicitly (needs Vulkan hardware and WARP); LEONIDA_VK_VALIDATION=1 checks Vulkan
// usage. A format the layer cannot create yet is listed as "not exposed" and does not fail the test; any other
// difference does.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "    line %d: %s\n", __LINE__, #condition); return false; } \
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

constexpr UINT kWidth = 5, kHeight = 3;  // odd sizes expose a wrong treatment of footprint padding
using Color = std::array<float, 4>;
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

// Records the copy of one subresource into a readback buffer, to be read after the list has run.
struct Plane {
    Ref<ID3D12Resource> buffer;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
    UINT64 total = 0;
    unsigned bytes = 0;  // per pixel in this plane
};

struct Run {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;

    bool create(ID3D12Device* device) {
        const D3D12_COMMAND_QUEUE_DESC queue_desc{};
        CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                 IID_PPV_ARGS(&list.p))));
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        return true;
    }
    bool execute() {
        CHECK(SUCCEEDED(list->Close()));
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
        Event event;
        CHECK(event.handle);
        CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, event.handle)));
        CHECK(WaitForSingleObject(event.handle, 5000) == WAIT_OBJECT_0);
        return true;
    }
};

D3D12_RESOURCE_DESC texture_desc(DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = kWidth;
    texture.Height = kHeight;
    texture.DepthOrArraySize = texture.MipLevels = 1;
    texture.Format = format;
    texture.SampleDesc.Count = 1;
    texture.Flags = flags;
    return texture;
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* image, D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = image;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

bool add_copy(ID3D12Device* device, ID3D12GraphicsCommandList* list, ID3D12Resource* image,
              const D3D12_RESOURCE_DESC& texture, UINT subresource, unsigned bytes, Plane& plane) {
    plane.bytes = bytes;
    device->GetCopyableFootprints(&texture, subresource, 1, 0, &plane.placed, nullptr, nullptr, &plane.total);
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = plane.total;
    buffer.Height = buffer.SampleDesc.Count = 1;
    buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&plane.buffer.p))));
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = image;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = subresource;
    dst.pResource = plane.buffer.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = plane.placed;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    return true;
}

bool read_plane(Plane& plane, std::vector<unsigned char>& pixels) {
    void* mapped = nullptr;
    const D3D12_RANGE range{0, SIZE_T(plane.total)};
    CHECK(SUCCEEDED(plane.buffer->Map(0, &range, &mapped)) && mapped);
    pixels.resize(kWidth * kHeight * plane.bytes);
    for (UINT y = 0; y < kHeight; ++y)
        std::memcpy(pixels.data() + y * kWidth * plane.bytes,
                    static_cast<unsigned char*>(mapped) + y * plane.placed.Footprint.RowPitch, kWidth * plane.bytes);
    const D3D12_RANGE none{};
    plane.buffer->Unmap(0, &none);
    return true;
}

// The two results of a case: false when the device could not run it at all.
enum class Result { ok, not_created, failed };

Result clear_color(ID3D12Device* device, DXGI_FORMAT format, DXGI_FORMAT resource_format, unsigned bytes,
                   const Color& full, const Color& part, std::vector<unsigned char>& pixels) {
    Run run;
    if (!run.create(device)) return Result::failed;
    Ref<ID3D12Resource> image;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    const D3D12_RESOURCE_DESC texture = texture_desc(resource_format != DXGI_FORMAT_UNKNOWN ? resource_format : format,
                                                     D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
                                               D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&image.p))))
        return Result::not_created;
    Ref<ID3D12DescriptorHeap> rtv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&rtv_heap.p)))) return Result::failed;
    const auto handle = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_RENDER_TARGET_VIEW_DESC view{};
    view.Format = format;
    view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    device->CreateRenderTargetView(image.p, resource_format != DXGI_FORMAT_UNKNOWN ? &view : nullptr, handle);
    run.list->ClearRenderTargetView(handle, full.data(), 0, nullptr);
    const D3D12_RECT rect{1, 1, 4, 3};
    run.list->ClearRenderTargetView(handle, part.data(), 1, &rect);
    transition(run.list.p, image.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Plane plane;
    if (!add_copy(device, run.list.p, image.p, texture, 0, bytes, plane) || !run.execute() ||
        !read_plane(plane, pixels))
        return Result::failed;
    return Result::ok;
}

struct DepthValues {
    float depth_full;
    UINT8 stencil_full;
    float depth_part;
    UINT8 stencil_part;
};

// Depth and stencil planes after: a full clear of both, a depth-only clear of one rectangle and a stencil-only
// clear of another. planes[1] is empty for a depth-only format.
Result clear_depth(ID3D12Device* device, DXGI_FORMAT format, bool stencil, unsigned depth_bytes,
                   const DepthValues& v, std::vector<unsigned char> planes[2]) {
    Run run;
    if (!run.create(device)) return Result::failed;
    Ref<ID3D12Resource> image;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    const D3D12_RESOURCE_DESC texture = texture_desc(format, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
                                               D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr, IID_PPV_ARGS(&image.p))))
        return Result::not_created;
    Ref<ID3D12DescriptorHeap> dsv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&dsv_heap.p)))) return Result::failed;
    const auto handle = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(image.p, nullptr, handle);
    const D3D12_CLEAR_FLAGS both = stencil ? D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL : D3D12_CLEAR_FLAG_DEPTH;
    run.list->ClearDepthStencilView(handle, both, v.depth_full, v.stencil_full, 0, nullptr);
    const D3D12_RECT depth_rect{1, 1, 4, 3};
    run.list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, v.depth_part, 0, 1, &depth_rect);
    if (stencil) {
        const D3D12_RECT stencil_rect{0, 0, 3, 2};
        run.list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_STENCIL, 0.0f, v.stencil_part, 1, &stencil_rect);
    }
    transition(run.list.p, image.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Plane depth, stencil_plane;
    if (!add_copy(device, run.list.p, image.p, texture, 0, depth_bytes, depth) ||
        (stencil && !add_copy(device, run.list.p, image.p, texture, 1, 1, stencil_plane)) || !run.execute() ||
        !read_plane(depth, planes[0]) || (stencil && !read_plane(stencil_plane, planes[1])))
        return Result::failed;
    return Result::ok;
}

struct ColorCase {
    const char* group;
    const char* name;
    DXGI_FORMAT format;
    unsigned bytes;
    const std::vector<Color>* colors;
    DXGI_FORMAT resource = DXGI_FORMAT_UNKNOWN;  // a TYPELESS format the resource is created with; format is the view
};

// Out-of-range values clamp in the normalised classes; NaN is 0 there. NaN is left out of the float classes, whose
// payload bits are not defined.
const std::vector<Color> kUnorm = {{0, 0.25f, 0.5f, 1}, {1, 0.75f, 0.125f, 0.5f}, {-1, 2, 0.5f, 1.5f},
                                   {0.0039f, 0.5019f, 0.9961f, 0.3333f}, {kNaN, kInf, -kInf, 0.5f}};
const std::vector<Color> kSnorm = {{-1, -0.5f, 0.5f, 1}, {-2, 2, 0.001f, -0.001f}, {0, 0.25f, -0.25f, 0.7f},
                                   {kNaN, kInf, -kInf, -0.5f}};
const std::vector<Color> kFloat = {{0.1f, 1.5f, -3.25f, 65504.0f}, {1e-8f, 70000.0f, -70000.0f, 0.5f},
                                   {0, 1, 2, 3}, {kInf, -kInf, 1e-5f, 123456.0f}};
const std::vector<Color> kUint = {{1, 2, 3, 4}, {1.5f, 2.5f, 3.5f, 4.5f}, {-1, -0.0f, 254.5f, 255.5f},
                                  {256, 65536, 16777216, 4294967296.0f}, {kNaN, kInf, -kInf, 0.5f},
                                  {0.499f, 0.501f, 1.499f, 1.501f}, {4294967040.0f, 3.4e38f, 255, 0}};
const std::vector<Color> kSint = {{-1, -2, -128, 127}, {-1.5f, -2.5f, 2.5f, 3.5f}, {-129, 128, -32769, 32768},
                                  {2147483648.0f, -2147483904.0f, kNaN, kInf}, {-kInf, 0.5f, -0.5f, 100000.0f}};

const ColorCase kColorCases[] = {
    {"UNORM", "R8_UNORM", DXGI_FORMAT_R8_UNORM, 1, &kUnorm},
    {"UNORM", "R8G8_UNORM", DXGI_FORMAT_R8G8_UNORM, 2, &kUnorm},
    {"UNORM", "R8G8B8A8_UNORM", DXGI_FORMAT_R8G8B8A8_UNORM, 4, &kUnorm},
    {"UNORM", "B8G8R8A8_UNORM", DXGI_FORMAT_B8G8R8A8_UNORM, 4, &kUnorm},
    {"UNORM", "R10G10B10A2_UNORM", DXGI_FORMAT_R10G10B10A2_UNORM, 4, &kUnorm},
    {"UNORM", "R16_UNORM", DXGI_FORMAT_R16_UNORM, 2, &kUnorm},
    {"UNORM", "R16G16_UNORM", DXGI_FORMAT_R16G16_UNORM, 4, &kUnorm},
    {"UNORM", "R16G16B16A16_UNORM", DXGI_FORMAT_R16G16B16A16_UNORM, 8, &kUnorm},
    {"SNORM", "R8_SNORM", DXGI_FORMAT_R8_SNORM, 1, &kSnorm},
    {"SNORM", "R8G8_SNORM", DXGI_FORMAT_R8G8_SNORM, 2, &kSnorm},
    {"SNORM", "R8G8B8A8_SNORM", DXGI_FORMAT_R8G8B8A8_SNORM, 4, &kSnorm},
    {"SNORM", "R16_SNORM", DXGI_FORMAT_R16_SNORM, 2, &kSnorm},
    {"SNORM", "R16G16_SNORM", DXGI_FORMAT_R16G16_SNORM, 4, &kSnorm},
    {"SNORM", "R16G16B16A16_SNORM", DXGI_FORMAT_R16G16B16A16_SNORM, 8, &kSnorm},
    {"SRGB", "R8G8B8A8_UNORM_SRGB", DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 4, &kUnorm},
    {"SRGB", "B8G8R8A8_UNORM_SRGB", DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, 4, &kUnorm},
    {"FLOAT", "R16_FLOAT", DXGI_FORMAT_R16_FLOAT, 2, &kFloat},
    {"FLOAT", "R16G16_FLOAT", DXGI_FORMAT_R16G16_FLOAT, 4, &kFloat},
    {"FLOAT", "R16G16B16A16_FLOAT", DXGI_FORMAT_R16G16B16A16_FLOAT, 8, &kFloat},
    {"FLOAT", "R32_FLOAT", DXGI_FORMAT_R32_FLOAT, 4, &kFloat},
    {"FLOAT", "R32G32_FLOAT", DXGI_FORMAT_R32G32_FLOAT, 8, &kFloat},
    {"FLOAT", "R32G32B32A32_FLOAT", DXGI_FORMAT_R32G32B32A32_FLOAT, 16, &kFloat},
    {"FLOAT", "R11G11B10_FLOAT", DXGI_FORMAT_R11G11B10_FLOAT, 4, &kFloat},
    {"UINT", "R8_UINT", DXGI_FORMAT_R8_UINT, 1, &kUint},
    {"UINT", "R8G8_UINT", DXGI_FORMAT_R8G8_UINT, 2, &kUint},
    {"UINT", "R16_UINT", DXGI_FORMAT_R16_UINT, 2, &kUint},
    {"UINT", "R16G16_UINT", DXGI_FORMAT_R16G16_UINT, 4, &kUint},
    {"UINT", "R8G8B8A8_UINT", DXGI_FORMAT_R8G8B8A8_UINT, 4, &kUint},
    {"UINT", "R10G10B10A2_UINT", DXGI_FORMAT_R10G10B10A2_UINT, 4, &kUint},
    {"UINT", "R16G16B16A16_UINT", DXGI_FORMAT_R16G16B16A16_UINT, 8, &kUint},
    {"UINT", "R32_UINT", DXGI_FORMAT_R32_UINT, 4, &kUint},
    {"UINT", "R32G32_UINT", DXGI_FORMAT_R32G32_UINT, 8, &kUint},
    {"UINT", "R32G32B32A32_UINT", DXGI_FORMAT_R32G32B32A32_UINT, 16, &kUint},
    {"SINT", "R8_SINT", DXGI_FORMAT_R8_SINT, 1, &kSint},
    {"SINT", "R8G8_SINT", DXGI_FORMAT_R8G8_SINT, 2, &kSint},
    {"SINT", "R16_SINT", DXGI_FORMAT_R16_SINT, 2, &kSint},
    {"SINT", "R16G16_SINT", DXGI_FORMAT_R16G16_SINT, 4, &kSint},
    {"SINT", "R8G8B8A8_SINT", DXGI_FORMAT_R8G8B8A8_SINT, 4, &kSint},
    {"SINT", "R16G16B16A16_SINT", DXGI_FORMAT_R16G16B16A16_SINT, 8, &kSint},
    {"SINT", "R32_SINT", DXGI_FORMAT_R32_SINT, 4, &kSint},
    {"SINT", "R32G32_SINT", DXGI_FORMAT_R32G32_SINT, 8, &kSint},
    {"SINT", "R32G32B32A32_SINT", DXGI_FORMAT_R32G32B32A32_SINT, 16, &kSint},
    // A TYPELESS resource cleared through the typed view its default maps to. A view that changes the class (UINT or
    // SINT view of a TYPELESS resource) is not covered: the layer ignores the RTV's format, so the clear converts
    // for the resource's class.
    {"TYPELESS", "R8_TYPELESS as UNORM", DXGI_FORMAT_R8_UNORM, 1, &kUnorm, DXGI_FORMAT_R8_TYPELESS},
    {"TYPELESS", "R8G8_TYPELESS as UNORM", DXGI_FORMAT_R8G8_UNORM, 2, &kUnorm, DXGI_FORMAT_R8G8_TYPELESS},
    {"TYPELESS", "R16G16_TYPELESS as FLOAT", DXGI_FORMAT_R16G16_FLOAT, 4, &kFloat, DXGI_FORMAT_R16G16_TYPELESS},
    {"TYPELESS", "R32G32_TYPELESS as FLOAT", DXGI_FORMAT_R32G32_FLOAT, 8, &kFloat, DXGI_FORMAT_R32G32_TYPELESS},
};

struct DepthCase {
    const char* name;
    DXGI_FORMAT format;
    bool stencil;
    unsigned depth_bytes;
};

const DepthCase kDepthCases[] = {
    {"D16_UNORM", DXGI_FORMAT_D16_UNORM, false, 2},
    {"D32_FLOAT", DXGI_FORMAT_D32_FLOAT, false, 4},
    {"D24_UNORM_S8_UINT", DXGI_FORMAT_D24_UNORM_S8_UINT, true, 4},
    {"D32_FLOAT_S8X24_UINT", DXGI_FORMAT_D32_FLOAT_S8X24_UINT, true, 4},
};

const DepthValues kDepthValues[] = {{0.25f, 0x5A, 0.75f, 0xA5}, {1.0f, 0xFF, 0.0f, 0x00},
                                    {0.3333f, 0x01, 0.9999f, 0x80}, {0.5f, 0x7F, 0.0001f, 0x10}};

struct Tally {
    unsigned matched = 0, differ = 0, not_exposed = 0, failed = 0, skipped = 0;
};

// Names the first byte that differs.
void report(const char* group, const char* name, unsigned index, const char* what,
            const std::vector<unsigned char>& actual, const std::vector<unsigned char>& expected) {
    for (std::size_t b = 0; b < actual.size() && b < expected.size(); ++b) {
        if (actual[b] != expected[b]) {
            std::printf("DIFF %-6s %-22s case %u %s byte %zu: layer %02x, WARP %02x\n", group, name, index, what, b,
                        unsigned(actual[b]), unsigned(expected[b]));
            return;
        }
    }
    std::printf("DIFF %-6s %-22s case %u %s: sizes %zu and %zu\n", group, name, index, what, actual.size(),
                expected.size());
}
} // namespace

int main() {
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    const unsigned errors_before = vk.validation_errors();
    Tally tally;
    {
        Ref<ID3D12Device> ours, host;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> warp;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
            FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) ||
            FAILED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
            std::fprintf(stderr, "WARP reference unavailable: differential test cannot run\n");
            return 1;
        }
        for (const ColorCase& c : kColorCases) {
            const auto& colors = *c.colors;
            bool reported_exposure = false;
            for (unsigned i = 0; i < colors.size(); ++i) {
                // Different colours outside and inside the rectangle exercise both Vulkan clear paths.
                const Color& full = colors[i];
                const Color& part = colors[(i + 1) % colors.size()];
                std::vector<unsigned char> actual, expected;
                const Result want = clear_color(host.p, c.format, c.resource, c.bytes, full, part, expected);
                if (want != Result::ok) {
                    ++tally.skipped;  // WARP itself does not offer the format as a render target here
                    continue;
                }
                const Result got = clear_color(ours.p, c.format, c.resource, c.bytes, full, part, actual);
                if (got == Result::not_created) {
                    ++tally.not_exposed;
                    if (!reported_exposure) std::printf("not exposed: %-6s %s\n", c.group, c.name);
                    reported_exposure = true;
                    continue;
                }
                if (got != Result::ok) {
                    ++tally.failed;
                    std::printf("FAIL %-6s %-22s case %u: the layer could not run it\n", c.group, c.name, i);
                } else if (actual != expected) {
                    ++tally.differ;
                    report(c.group, c.name, i, "colour", actual, expected);
                } else {
                    ++tally.matched;
                }
            }
        }
        for (const DepthCase& c : kDepthCases) {
            bool reported_exposure = false;
            for (unsigned i = 0; i < std::size(kDepthValues); ++i) {
                std::vector<unsigned char> actual[2], expected[2];
                const Result want = clear_depth(host.p, c.format, c.stencil, c.depth_bytes, kDepthValues[i], expected);
                if (want != Result::ok) {
                    ++tally.skipped;
                    continue;
                }
                const Result got = clear_depth(ours.p, c.format, c.stencil, c.depth_bytes, kDepthValues[i], actual);
                if (got == Result::not_created) {
                    ++tally.not_exposed;
                    if (!reported_exposure) std::printf("not exposed: DEPTH  %s\n", c.name);
                    reported_exposure = true;
                    continue;
                }
                if (got != Result::ok) {
                    ++tally.failed;
                    std::printf("FAIL DEPTH  %-22s case %u: the layer could not run it\n", c.name, i);
                    continue;
                }
                bool same = true;
                for (int plane = 0; plane < (c.stencil ? 2 : 1); ++plane) {
                    if (actual[plane] != expected[plane]) {
                        same = false;
                        report("DEPTH", c.name, i, plane ? "stencil" : "depth", actual[plane], expected[plane]);
                    }
                }
                if (same) ++tally.matched;
                else ++tally.differ;
            }
        }
    } // Destroy the D3D12 objects before checking the Vulkan validation counter.
    std::printf("clears: %u matched WARP, %u differ, %u failed, %u in formats the layer does not expose, "
                "%u skipped (WARP does not offer the format)\n",
                tally.matched, tally.differ, tally.failed, tally.not_exposed, tally.skipped);
    if (tally.differ || tally.failed) return 1;
    if (vk.validation_errors() != errors_before) return 1;
    return 0;
}
