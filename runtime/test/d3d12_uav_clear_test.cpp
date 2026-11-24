// ClearUnorderedAccessViewUint / ClearUnorderedAccessViewFloat on the Vulkan-backed device, on a GPU. The views are
// GFX10 descriptors written into a CPU descriptor heap by hand, the bytes Create*View will write (docs/d3d12_vulkan.md
// section 3.4), so the test does not depend on which view-creation code is present. Every buffer and texture is
// cleared, copied into a readback heap with D3D12 itself and compared byte for byte with a CPU model.
//
// First, the value conversion (src/uav_clear.h) is checked against hand-computed elements, so the model the GPU
// results are compared with is pinned to D3D's rules and not only to itself.
//
// No shaders or game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation
// error fails the test.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"
#include "../src/uav_clear.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace g = leonida::gfx10;
namespace uc = leonida::uav_clear;

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

std::array<std::uint32_t, 4> floats(float r, float g_, float b, float a) {
    return {std::bit_cast<std::uint32_t>(r), std::bit_cast<std::uint32_t>(g_), std::bit_cast<std::uint32_t>(b),
            std::bit_cast<std::uint32_t>(a)};
}

std::vector<std::uint8_t> element(VkFormat format, bool is_float, const std::array<std::uint32_t, 4>& values) {
    uc::Element e{};
    if (!uc::encode(format, is_float, values, e)) return {};
    return {e.bytes.begin(), e.bytes.begin() + e.size};
}

std::uint32_t word(const std::vector<std::uint8_t>& bytes) {
    std::uint32_t w = 0;
    std::memcpy(&w, bytes.data(), std::min<std::size_t>(4, bytes.size()));
    return w;
}

int conversion_checks() {
    using V = std::vector<std::uint8_t>;
    // Uint copies the low bits of each value, whatever the type.
    EXPECT((element(VK_FORMAT_R8G8B8A8_UNORM, false, {0x1ff, 2, 3, 4}) == V{0xff, 2, 3, 4}));
    EXPECT(word(element(VK_FORMAT_R32_SFLOAT, false, {0x3f800000, 0, 0, 0})) == 0x3f800000);
    EXPECT((element(VK_FORMAT_R16_SINT, false, {0xfffe8000, 0, 0, 0}) == V{0x00, 0x80}));
    EXPECT(word(element(VK_FORMAT_A2B10G10R10_UINT_PACK32, false, {1023, 0, 1, 7})) == (1023u | 1u << 20 | 3u << 30));
    // Float converts: UNORM and SNORM clamp and round to nearest, NaN is 0.
    EXPECT((element(VK_FORMAT_R8G8B8A8_UNORM, true, floats(1.0f, 0.5f, -1.0f, NAN)) == V{0xff, 0x80, 0, 0}));
    EXPECT((element(VK_FORMAT_R8G8B8A8_SNORM, true, floats(1.0f, -1.0f, 0.5f, 2.0f)) == V{0x7f, 0x81, 0x40, 0x7f}));
    EXPECT((element(VK_FORMAT_B8G8R8A8_UNORM, true, floats(1.0f, 0.0f, 0.0f, 1.0f)) == V{0, 0, 0xff, 0xff}));
    EXPECT(word(element(VK_FORMAT_A2B10G10R10_UNORM_PACK32, true, floats(1.0f, 0.0f, 0.5f, 1.0f))) ==
           (1023u | 512u << 20 | 3u << 30));
    // A4B4G4R4 (R4G4B4A4_PACK16): red in bits 12-15, alpha in 0-3, as the host's render-target clears give.
    EXPECT((element(VK_FORMAT_R4G4B4A4_UNORM_PACK16, true, floats(1.0f, 0, 0, 0)) == V{0x00, 0xf0}));
    EXPECT((element(VK_FORMAT_R4G4B4A4_UNORM_PACK16, true, floats(0, 0, 0, 1.0f)) == V{0x0f, 0x00}));
    EXPECT((element(VK_FORMAT_R4G4B4A4_UNORM_PACK16, true, floats(0, 1.0f, 0.2f, 0)) == V{0x30, 0x0f}));
    // Half floats round to nearest even; too large is infinity; tiny values become subnormals.
    EXPECT((element(VK_FORMAT_R16_SFLOAT, true, floats(1.0f, 0, 0, 0)) == V{0x00, 0x3c}));
    EXPECT((element(VK_FORMAT_R16_SFLOAT, true, floats(-2.0f, 0, 0, 0)) == V{0x00, 0xc0}));
    // Round toward zero, a finite overflow saturating to the largest finite (D3D11.3 3.2.2), as
    // ClearUnorderedAccessViewFloat gives on WARP and on NVIDIA: 65520 and 70000 are 65504, not infinity.
    EXPECT((element(VK_FORMAT_R16_SFLOAT, true, floats(65520.0f, 0, 0, 0)) == V{0xff, 0x7b}));
    EXPECT((element(VK_FORMAT_R16_SFLOAT, true, floats(70000.0f, 0, 0, 0)) == V{0xff, 0x7b}));
    EXPECT((element(VK_FORMAT_R16_SFLOAT, true, floats(-1e10f, 0, 0, 0)) == V{0xff, 0xfb}));
    EXPECT((element(VK_FORMAT_R16_SFLOAT, true, floats(INFINITY, 0, 0, 0)) == V{0x00, 0x7c}));  // stays infinity
    EXPECT((element(VK_FORMAT_R16_SFLOAT, true, floats(1e-7f, 0, 0, 0)) == V{0x01, 0x00}));
    EXPECT((element(VK_FORMAT_R16_SFLOAT, true, floats(1.0f + 1.0f / 2048, 0, 0, 0)) == V{0x00, 0x3c}));  // truncated
    // The packed 11/11/10 floats: 1.0, 2.0 and 0.5; no sign, so negative is 0.
    EXPECT(word(element(VK_FORMAT_B10G11R11_UFLOAT_PACK32, true, floats(1.0f, 2.0f, 0.5f, 0))) ==
           (0x3c0u | 0x400u << 11 | 0x1c0u << 22));
    EXPECT(word(element(VK_FORMAT_B10G11R11_UFLOAT_PACK32, true, floats(-1.0f, 0, 0, 0))) == 0);
    // Too large is the largest finite on every channel (host: 0xf7fdffbf), and 1/3 truncates (host: 0x1aa).
    EXPECT(word(element(VK_FORMAT_B10G11R11_UFLOAT_PACK32, true, floats(1e6f, 1e6f, 1e6f, 0))) == 0xf7fdffbfu);
    EXPECT(word(element(VK_FORMAT_B10G11R11_UFLOAT_PACK32, true, floats(0, 0, 1.0f / 3, 0))) >> 22 == 0x1aau);
    // Shared-exponent 9:9:9:5 float (pinned against WARP): round half-up, bump on 512, clamp negatives/NaN to 0
    // and +Inf/overflow to 65408; Uint clears to zero.
    EXPECT(word(element(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(0, 0, 0, 1))) == 0x00000000u);
    EXPECT(word(element(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(1.0f, 2.0f, 4.0f, 0))) == 0x94010040u);
    EXPECT(word(element(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(0.5f, 0.25f, 0.125f, 0))) == 0x79010100u);
    EXPECT(word(element(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(1.0f / 3, 1.0f / 3, 1.0f / 3, 0))) == 0x7556ab55u);
    EXPECT(word(element(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true,
                        floats(1.0f + 0.5f / 256, 1.0f + 1.5f / 256, 1.0f + 2.5f / 256, 0))) == 0x840e0501u);
    EXPECT(word(element(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(1.0f - 0.25f / 256, 0, 0, 0))) == 0x80000100u);
    EXPECT(word(element(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(65408.0f, 65472.0f, INFINITY, 0))) == 0xffffffffu);
    EXPECT(word(element(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(std::ldexp(1.0f, -25), 0, 0, 0))) == 0x00000001u);
    EXPECT(word(element(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(std::ldexp(511.5f, -24), 0, 0, 0))) == 0x08000100u);
    EXPECT(word(element(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(-1.0f, 1.0f, NAN, 0))) == 0x80020000u);
    EXPECT(word(element(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, false, {0x3f800000u, 0xffffffffu, 1u, 2u})) == 0u);
    // Float into an integer channel: toward zero, clamped.
    EXPECT(word(element(VK_FORMAT_R32_SINT, true, floats(-3.7f, 0, 0, 0))) == std::uint32_t(-3));
    EXPECT((element(VK_FORMAT_R8G8_UINT, true, floats(300.0f, -5.0f, 0, 0)) == V{0xff, 0x00}));
    // Decoding the bytes in another format of the same size gives the value that writes them back.
    uc::Element e{};
    VkClearColorValue value{};
    EXPECT(uc::encode(VK_FORMAT_R32_UINT, false, {0x40490fdb, 0, 0, 0}, e) && uc::decode(VK_FORMAT_R32_SFLOAT, e, value));
    EXPECT(std::bit_cast<std::uint32_t>(value.float32[0]) == 0x40490fdb);
    EXPECT(uc::encode(VK_FORMAT_R8G8B8A8_UNORM, true, floats(1.0f, 0.5f, 0, 1.0f), e) &&
           uc::decode(VK_FORMAT_R8G8B8A8_UINT, e, value));
    EXPECT(value.uint32[0] == 0xff && value.uint32[1] == 0x80 && value.uint32[2] == 0 && value.uint32[3] == 0xff);
    EXPECT(uc::encode(VK_FORMAT_R8_UINT, false, {0x80, 0, 0, 0}, e) && uc::decode(VK_FORMAT_R8_SINT, e, value) &&
           value.int32[0] == -128);
    // R9G9B9E5 decodes both to R32_UINT (emulated backing) and to float RGB (native backing).
    EXPECT(uc::encode(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(1.0f, 2.0f, 4.0f, 0), e) &&
           uc::decode(VK_FORMAT_R32_UINT, e, value) && value.uint32[0] == 0x94010040u);
    EXPECT(uc::decode(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, e, value) && value.float32[0] == 1.0f &&
           value.float32[1] == 2.0f && value.float32[2] == 4.0f);
    EXPECT(!uc::encode(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, false, {}, e));  // not a typed-UAV format
    EXPECT(!uc::decode(VK_FORMAT_R16_UINT, e, value));                  // e is one byte, R16 is two
    return 0;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return desc;
}

// One clear. Buffers: offset, a size in elements and an element format (UNDEFINED: raw, or structured with a
// stride). Textures: a mip and a layer range in the view's format.
struct Clear {
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool is_float = false;
    std::array<std::uint32_t, 4> values{};
    std::vector<D3D12_RECT> rects;
    std::uint64_t offset = 0;  // buffer
    std::uint32_t count = 0;   // buffer: elements (raw: bytes / 4)
    std::uint32_t stride = 0;  // structured
    std::uint32_t mip = 0;     // texture
    std::uint32_t first_layer = 0;
    std::uint32_t layer_count = 1;
};

struct BufferCase {
    const char* name;
    UINT64 size;
    std::vector<Clear> clears;  // the first one should cover the whole buffer: default-heap buffers start undefined
};

struct TextureCase {
    const char* name;
    DXGI_FORMAT format;
    VkFormat image_format;  // what the device creates for `format`
    UINT width, height;
    UINT16 mips, layers;  // layers: the depth of a volume
    bool volume;
    std::vector<Clear> clears;
};

struct Harness {
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12Fence* fence = nullptr;
    UINT64 fence_value = 0;
    std::byte* slots = nullptr;  // CPU descriptor heap
    UINT step = 0;

    D3D12_CPU_DESCRIPTOR_HANDLE slot(UINT i) const { return {reinterpret_cast<SIZE_T>(slots + std::size_t(i) * step)}; }

    template<class Words>
    D3D12_CPU_DESCRIPTOR_HANDLE write(UINT i, const Words& words) const {
        std::memset(slots + std::size_t(i) * step, 0, step);
        std::memcpy(slots + std::size_t(i) * step, words.data(), sizeof(words));
        return slot(i);
    }

    // Records with `body`, executes, waits. The result of the replay.
    template<class Body>
    leonida::ReplayResult run(HRESULT& close, Body&& body) {
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))))
            return leonida::ReplayResult::unsupported;
        body(list.p);
        close = list->Close();
        if (FAILED(close)) return leonida::ReplayResult::success;
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        const auto result = rt::d3d12::last_execute_result(queue);
        queue->Signal(fence, ++fence_value);
        fence->SetEventOnCompletion(fence_value, nullptr);
        return result;
    }

    Ref<ID3D12Resource> committed(const D3D12_RESOURCE_DESC& desc, D3D12_HEAP_TYPE type) const {
        Ref<ID3D12Resource> out;
        D3D12_HEAP_PROPERTIES heap{type};
        device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                        type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
                                                                         : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                        nullptr, IID_PPV_ARGS(&out.p));
        return out;
    }
};

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after,
                                  UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {resource, subresource, before, after};
    return b;
}

g::BufferWords buffer_view(UINT64 va, const Clear& c) {
    g::Buffer buffer{};
    buffer.address = va + c.offset;
    if (c.format != VK_FORMAT_UNDEFINED) {
        buffer.format = g::format_from_vulkan(c.format);
        buffer.swizzle = g::format_swizzle(c.format);
        buffer.num_records = c.count;
        buffer.oob_select = g::OobSelect::structured;
    } else if (c.stride) {
        buffer.stride = c.stride;
        buffer.num_records = c.count;
        buffer.oob_select = g::OobSelect::structured;
    } else {
        buffer.num_records = c.count * 4;
        buffer.oob_select = g::OobSelect::raw;
    }
    return g::encode(buffer);
}

// What the clear writes into the model, the way D3D12 says it should.
void apply(std::vector<std::uint8_t>& model, const Clear& c) {
    const bool typed = c.format != VK_FORMAT_UNDEFINED;
    const std::vector<std::uint8_t> e =
        typed ? element(c.format, c.is_float, c.values)
              : std::vector<std::uint8_t>{std::uint8_t(c.values[0]), std::uint8_t(c.values[0] >> 8),
                                          std::uint8_t(c.values[0] >> 16), std::uint8_t(c.values[0] >> 24)};
    const std::uint64_t element_size = typed ? e.size() : (c.stride ? c.stride : 4);
    const std::uint64_t bytes = c.count * element_size;
    std::vector<std::array<std::uint64_t, 2>> ranges;
    if (c.rects.empty()) ranges.push_back({0, bytes});
    for (const auto& r : c.rects)
        if (r.top <= 0 && r.bottom > 0 && r.right > r.left)
            ranges.push_back({std::uint64_t(std::max<LONG>(r.left, 0)) * element_size,
                              std::min<std::uint64_t>(std::uint64_t(r.right) * element_size, bytes)});
    for (const auto& [begin, end] : ranges)
        for (std::uint64_t i = begin; i < end && c.offset + i < model.size(); ++i)
            model[std::size_t(c.offset + i)] = e[std::size_t(i % e.size())];
}

unsigned mismatches = 0;

void compare(const char* what, const std::uint8_t* got, const std::uint8_t* want, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (got[i] == want[i]) continue;
        if (mismatches++ < 12) std::fprintf(stderr, "%s: byte %zu is %02x, want %02x\n", what, i, got[i], want[i]);
    }
}

int run_buffer(Harness& h, const BufferCase& test) {
    Ref<ID3D12Resource> buffer = h.committed(buffer_desc(test.size), D3D12_HEAP_TYPE_DEFAULT);
    auto readback_desc = buffer_desc(test.size);
    readback_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    Ref<ID3D12Resource> readback = h.committed(readback_desc, D3D12_HEAP_TYPE_READBACK);
    EXPECT(buffer.p && readback.p);
    const UINT64 va = buffer->GetGPUVirtualAddress();
    std::vector<std::uint8_t> model(std::size_t(test.size), 0);
    HRESULT close = E_FAIL;
    const auto result = h.run(close, [&](ID3D12GraphicsCommandList* list) {
        for (UINT i = 0; i < test.clears.size(); ++i) {
            const Clear& c = test.clears[i];
            const auto view = h.write(i, buffer_view(va, c));  // each clear has its own slot
            const auto* rects = c.rects.empty() ? nullptr : c.rects.data();
            if (c.is_float) {
                const float v[4] = {std::bit_cast<float>(c.values[0]), std::bit_cast<float>(c.values[1]),
                                    std::bit_cast<float>(c.values[2]), std::bit_cast<float>(c.values[3])};
                list->ClearUnorderedAccessViewFloat({0}, view, buffer.p, v, UINT(c.rects.size()), rects);
            } else {
                list->ClearUnorderedAccessViewUint({0}, view, buffer.p, c.values.data(), UINT(c.rects.size()), rects);
            }
            D3D12_RESOURCE_BARRIER uav{};
            uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav.UAV.pResource = buffer.p;
            list->ResourceBarrier(1, &uav);
            apply(model, c);
        }
        const auto to_copy = transition(buffer.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        list->CopyBufferRegion(readback.p, 0, buffer.p, 0, test.size);
    });
    if (close != S_OK || result != leonida::ReplayResult::success)
        std::fprintf(stderr, "%s: Close %08lx, replay %d\n", test.name, static_cast<unsigned long>(close), int(result));
    EXPECT(close == S_OK && result == leonida::ReplayResult::success);
    void* data = nullptr;
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &data)));
    compare(test.name, static_cast<const std::uint8_t*>(data), model.data(), model.size());
    return 0;
}

std::uint32_t texel_size(VkFormat format) { return uc::layout(format)->bytes; }

int run_texture(Harness& h, const TextureCase& test) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = test.volume ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = test.width;
    desc.Height = test.height;
    desc.DepthOrArraySize = test.layers;
    desc.MipLevels = test.mips;
    desc.Format = test.format;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    Ref<ID3D12Resource> texture = h.committed(desc, D3D12_HEAP_TYPE_DEFAULT);
    EXPECT(texture.p);

    // One readback region per subresource: a volume mip is one subresource of several slices.
    const std::uint32_t texel = texel_size(test.image_format);
    const UINT array = test.volume ? 1 : test.layers;
    struct Sub { UINT mip, layer, width, height, depth; UINT64 offset; UINT pitch; };
    std::vector<Sub> subs;
    UINT64 total = 0;
    for (UINT layer = 0; layer < array; ++layer)
        for (UINT mip = 0; mip < test.mips; ++mip) {
            Sub s{mip, layer, std::max(1u, test.width >> mip), std::max(1u, test.height >> mip),
                  test.volume ? std::max(1u, UINT(test.layers) >> mip) : 1u, total, 0};
            s.pitch = (s.width * texel + 255) / 256 * 256;
            total += (UINT64(s.pitch) * s.height * s.depth + 511) / 512 * 512;
            subs.push_back(s);
        }
    auto readback_desc = buffer_desc(total);
    readback_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    Ref<ID3D12Resource> readback = h.committed(readback_desc, D3D12_HEAP_TYPE_READBACK);
    EXPECT(readback.p);

    // The model, tightly packed per subresource; committed textures start zeroed.
    std::vector<std::vector<std::uint8_t>> model(subs.size());
    for (std::size_t i = 0; i < subs.size(); ++i)
        model[i].assign(std::size_t(subs[i].width) * subs[i].height * subs[i].depth * texel, 0);

    HRESULT close = E_FAIL;
    const auto result = h.run(close, [&](ID3D12GraphicsCommandList* list) {
        for (UINT i = 0; i < test.clears.size(); ++i) {
            const Clear& c = test.clears[i];
            g::Image image{};
            image.format = g::format_from_vulkan(c.format);
            image.swizzle = g::format_swizzle(c.format);
            image.type = test.volume ? g::ImageType::image_3d : g::ImageType::image_2d_array;
            image.width = test.width;
            image.height = test.height;
            image.depth = test.volume ? test.layers : 1;
            image.first_layer = c.first_layer;
            image.layer_count = c.layer_count;
            image.first_mip = c.mip;
            image.mip_count = 1;
            image.resource_mips = test.mips;
            const auto view = h.write(i, g::encode(image));
            const auto* rects = c.rects.empty() ? nullptr : c.rects.data();
            if (c.is_float) {
                const float v[4] = {std::bit_cast<float>(c.values[0]), std::bit_cast<float>(c.values[1]),
                                    std::bit_cast<float>(c.values[2]), std::bit_cast<float>(c.values[3])};
                list->ClearUnorderedAccessViewFloat({0}, view, texture.p, v, UINT(c.rects.size()), rects);
            } else {
                list->ClearUnorderedAccessViewUint({0}, view, texture.p, c.values.data(), UINT(c.rects.size()), rects);
            }
            D3D12_RESOURCE_BARRIER uav{};
            uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav.UAV.pResource = texture.p;
            list->ResourceBarrier(1, &uav);

            const std::vector<std::uint8_t> e = element(c.format, c.is_float, c.values);
            for (std::size_t s = 0; s < subs.size(); ++s) {
                const Sub& sub = subs[s];
                if (sub.mip != c.mip) continue;
                if (!test.volume && (sub.layer < c.first_layer || sub.layer >= c.first_layer + c.layer_count)) continue;
                std::vector<D3D12_RECT> rects_or_all = c.rects;
                if (rects_or_all.empty()) rects_or_all.push_back({0, 0, LONG(sub.width), LONG(sub.height)});
                for (const auto& r : rects_or_all)
                    for (UINT z = 0; z < sub.depth; ++z)
                        for (LONG y = std::max<LONG>(r.top, 0); y < std::min<LONG>(r.bottom, sub.height); ++y)
                            for (LONG x = std::max<LONG>(r.left, 0); x < std::min<LONG>(r.right, sub.width); ++x)
                                std::memcpy(&model[s][((std::size_t(z) * sub.height + y) * sub.width + x) * texel],
                                            e.data(), texel);
            }
        }
        const auto to_copy =
            transition(texture.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        for (const Sub& s : subs) {
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = readback.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = {s.offset, {test.format, s.width, s.height, s.depth, s.pitch}};
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = texture.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            src.SubresourceIndex = s.mip + s.layer * test.mips;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
    });
    if (close != S_OK || result != leonida::ReplayResult::success)
        std::fprintf(stderr, "%s: Close %08lx, replay %d\n", test.name, static_cast<unsigned long>(close), int(result));
    EXPECT(close == S_OK && result == leonida::ReplayResult::success);
    void* data = nullptr;
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &data)));
    for (std::size_t i = 0; i < subs.size(); ++i) {
        const Sub& s = subs[i];
        for (UINT z = 0; z < s.depth; ++z)
            for (UINT y = 0; y < s.height; ++y) {
                char what[96];
                std::snprintf(what, sizeof(what), "%s mip %u layer %u z %u row %u", test.name, s.mip, s.layer, z, y);
                compare(what, static_cast<const std::uint8_t*>(data) + s.offset + (UINT64(z) * s.height + y) * s.pitch,
                        &model[i][(std::size_t(z) * s.height + y) * s.width * texel], std::size_t(s.width) * texel);
            }
    }
    return 0;
}

}  // namespace

int main() {
    if (conversion_checks()) return 1;

    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Ref<ID3D12CommandQueue> queue;
        const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        Ref<ID3D12DescriptorHeap> heap;
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 32};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));

        Harness h;
        h.device = device.p;
        h.queue = queue.p;
        h.fence = fence.p;
        h.slots = reinterpret_cast<std::byte*>(heap->GetCPUDescriptorHandleForHeapStart().ptr);
        h.step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(h.step >= sizeof(g::ImageWords));

        const auto whole = [](std::uint32_t words) {
            Clear c;
            c.values = {0xcdcdcdcd, 0, 0, 0};
            c.count = words;
            return c;
        };
        const auto typed = [](VkFormat format, bool is_float, std::array<std::uint32_t, 4> values, std::uint64_t offset,
                              std::uint32_t count, std::vector<D3D12_RECT> rects = {}) {
            Clear c;
            c.format = format;
            c.is_float = is_float;
            c.values = values;
            c.offset = offset;
            c.count = count;
            c.rects = std::move(rects);
            return c;
        };

        std::vector<BufferCase> buffers;
        {
            BufferCase raw{"raw buffer", 1024, {whole(256)}};
            Clear part;  // a raw view in the middle: value 0 in every word
            part.values = {0x11223344, 0xdead, 0xbeef, 0xf00d};
            part.offset = 64;
            part.count = 64;
            raw.clears.push_back(part);
            Clear as_float;  // Float on a raw view writes value 0's bits
            as_float.is_float = true;
            as_float.values = floats(1.0f, 2.0f, 3.0f, 4.0f);
            as_float.offset = 512;
            as_float.count = 16;
            raw.clears.push_back(as_float);
            Clear rects = part;  // element rectangles: [2, 5) and [9, 12), and one outside row 0
            rects.values = {0x55aa55aa, 0, 0, 0};
            rects.offset = 768;
            rects.count = 32;
            rects.rects = {{2, 0, 5, 1}, {9, 0, 12, 1}, {20, 1, 30, 2}};
            raw.clears.push_back(rects);
            buffers.push_back(raw);
        }
        {
            BufferCase structured{"structured buffer", 1024, {whole(256)}};
            Clear c;  // 12-byte records, two element rectangles
            c.values = {0xaabbccdd, 0, 0, 0};
            c.offset = 48;
            c.count = 10;
            c.stride = 12;
            c.rects = {{2, 0, 5, 1}, {7, 0, 9, 1}};
            structured.clears.push_back(c);
            buffers.push_back(structured);
        }
        {
            // Elements narrower than a word at unaligned offsets: fill the middle, copy the ends.
            BufferCase narrow{"typed buffer, 8- and 16-bit elements", 256, {whole(64)}};
            narrow.clears.push_back(typed(VK_FORMAT_R8_UNORM, true, floats(0.5f, 0, 0, 0), 3, 13));
            narrow.clears.push_back(typed(VK_FORMAT_R16_UINT, false, {0xbeef, 0, 0, 0}, 34, 7));
            narrow.clears.push_back(typed(VK_FORMAT_R8G8_SNORM, true, floats(-1.0f, 0.25f, 0, 0), 66, 9));
            narrow.clears.push_back(typed(VK_FORMAT_R8_UINT, false, {0x1234, 0, 0, 0}, 101, 40, {{3, 0, 9, 1}}));
            buffers.push_back(narrow);
        }
        {
            for (UINT offset = 0; offset < 8; ++offset) {
                BufferCase short_ranges{"short typed buffer ranges", 256, {whole(64)}};
                for (UINT count = 1; count <= 8; ++count)
                    short_ranges.clears.push_back(typed(VK_FORMAT_R8_UINT, false, {offset * 17 + count, 0, 0, 0},
                                                       offset, count));
                buffers.push_back(short_ranges);
            }
            BufferCase short_ranges{"short typed buffer boundary and rectangles", 256, {whole(64)}};
            short_ranges.clears.push_back(typed(VK_FORMAT_R16_UINT, false, {0xbeef, 0, 0, 0}, 254, 1));
            short_ranges.clears.push_back(typed(VK_FORMAT_R8_UINT, false, {0x7b, 0, 0, 0}, 255, 1));
            short_ranges.clears.push_back(typed(VK_FORMAT_R16_UNORM, true, floats(1.0f, 0, 0, 0), 50, 4,
                                                {{-1, 0, 1, 1}, {0, 0, 2, 1}, {3, 0, 8, 1}}));
            buffers.push_back(short_ranges);
        }
        {
            // 16-byte elements over more than vkCmdUpdateBuffer's 64 KiB: written once, then doubled by copies.
            BufferCase wide{"typed buffer, 8- and 16-byte elements", 1 << 17, {whole((1 << 17) / 4)}};
            wide.clears.push_back(typed(VK_FORMAT_R32G32B32A32_UINT, false, {1, 2, 3, 4}, 16, 8000));
            wide.clears.push_back(typed(VK_FORMAT_R16G16B16A16_SFLOAT, true, floats(1.0f, -2.0f, 0.5f, 0.0f), 128016, 64,
                                        {{4, 0, 9, 1}, {30, 0, 100, 1}}));
            wide.clears.push_back(typed(VK_FORMAT_R32G32_SINT, true, floats(-7.9f, 7.9f, 0, 0), 129536, 32));
            buffers.push_back(wide);
        }
        {
            BufferCase packed{"typed buffer, 32-bit formats", 1024, {whole(256)}};
            packed.clears.push_back(typed(VK_FORMAT_R8G8B8A8_UNORM, false, {0x1ff, 2, 3, 4}, 0, 16));
            packed.clears.push_back(typed(VK_FORMAT_B8G8R8A8_UNORM, true, floats(1.0f, 0.0f, 0.0f, 1.0f), 64, 16));
            packed.clears.push_back(typed(VK_FORMAT_B10G11R11_UFLOAT_PACK32, true, floats(1.0f, 2.0f, 0.5f, 0), 128, 16));
            packed.clears.push_back(typed(VK_FORMAT_A2B10G10R10_UNORM_PACK32, true, floats(1.0f, 0.0f, 0.5f, 1.0f), 192, 16));
            packed.clears.push_back(typed(VK_FORMAT_R32_SFLOAT, true, floats(3.5f, 0, 0, 0), 256, 64, {{10, 0, 20, 1}}));
            packed.clears.push_back(typed(VK_FORMAT_R16G16_SINT, false, {0xffff8000, 7, 0, 0}, 512, 128));
            buffers.push_back(packed);
        }
        for (const auto& test : buffers)
            if (run_buffer(h, test)) return 1;

        const auto texture_clear = [](VkFormat format, bool is_float, std::array<std::uint32_t, 4> values,
                                      std::uint32_t mip, std::uint32_t first_layer, std::uint32_t layer_count,
                                      std::vector<D3D12_RECT> rects = {}) {
            Clear c;
            c.format = format;
            c.is_float = is_float;
            c.values = values;
            c.mip = mip;
            c.first_layer = first_layer;
            c.layer_count = layer_count;
            c.rects = std::move(rects);
            return c;
        };
        const std::vector<TextureCase> textures = {
            {"RGBA32_UINT array, mips, layers and rectangles", DXGI_FORMAT_R32G32B32A32_UINT,
             VK_FORMAT_R32G32B32A32_UINT, 16, 8, 3, 4, false,
             {texture_clear(VK_FORMAT_R32G32B32A32_UINT, false, {1, 2, 3, 4}, 1, 1, 2),
              texture_clear(VK_FORMAT_R32G32B32A32_UINT, false, {9, 8, 7, 6}, 0, 3, 1, {{2, 1, 6, 4}, {10, 5, 20, 9}}),
              texture_clear(VK_FORMAT_R32G32B32A32_UINT, true, floats(2.9f, -1.0f, 1e10f, NAN), 2, 0, 4)}},
            {"RGBA8_UNORM, Float and Uint", DXGI_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, 8, 8, 1, 1, false,
             {texture_clear(VK_FORMAT_R8G8B8A8_UNORM, true, floats(1.0f, 0.5f, 0.0f, 1.0f), 0, 0, 1),
              texture_clear(VK_FORMAT_R8G8B8A8_UNORM, false, {0x10, 0x20, 0x130, 0x40}, 0, 0, 1, {{1, 1, 3, 7}})}},
            {"R32_TYPELESS through R32_UINT / R32_SINT / R32_SFLOAT (including NaN bits)", DXGI_FORMAT_R32_TYPELESS,
             VK_FORMAT_R32_SFLOAT, 8, 4, 2, 1, false,
             {texture_clear(VK_FORMAT_R32_UINT, false, {0x40490fdb, 0, 0, 0}, 0, 0, 1),
              texture_clear(VK_FORMAT_R32_UINT, false, {0x7f800001u, 0, 0, 0}, 0, 0, 1),
              texture_clear(VK_FORMAT_R32_UINT, false, {0xffffffffu, 0, 0, 0}, 0, 0, 1, {{1, 1, 6, 3}}),
              texture_clear(VK_FORMAT_R32_SINT, false, {0xff800001u, 0, 0, 0}, 1, 0, 1),
              texture_clear(VK_FORMAT_R32_SFLOAT, true, floats(-0.25f, 0, 0, 0), 1, 0, 1, {{0, 0, 2, 2}})}},
            {"R32G32_TYPELESS through R32G32_UINT (NaN bit patterns)", DXGI_FORMAT_R32G32_TYPELESS,
             VK_FORMAT_R32G32_SFLOAT, 8, 4, 1, 1, false,
             {texture_clear(VK_FORMAT_R32G32_UINT, false, {0x7f800001u, 0xffffffffu, 0, 0}, 0, 0, 1),
              texture_clear(VK_FORMAT_R32G32_SINT, false, {0xff800001u, 0x7fffffffu, 0, 0}, 0, 0, 1, {{2, 1, 6, 3}})}},
            {"R32G32B32A32_TYPELESS through R32G32B32A32_UINT (NaN bit patterns)", DXGI_FORMAT_R32G32B32A32_TYPELESS,
             VK_FORMAT_R32G32B32A32_SFLOAT, 4, 4, 1, 1, false,
             {texture_clear(VK_FORMAT_R32G32B32A32_UINT, false, {0x7f800001u, 0xffffffffu, 0x7fffffffu, 0xff800001u},
                            0, 0, 1)}},
            {"R16G16_TYPELESS through UNORM, UINT, SINT and SNORM", DXGI_FORMAT_R16G16_TYPELESS,
             VK_FORMAT_R16G16_SFLOAT, 8, 8, 2, 2, false,
             {texture_clear(VK_FORMAT_R16G16_UNORM, true, floats(1.0f, 1.0f, 0.0f, 0.0f), 0, 0, 2),
              texture_clear(VK_FORMAT_R16G16_UNORM, true, floats(0.5f, 1.0f, 0.0f, 0.0f), 0, 0, 1, {{1, 1, 5, 4}}),
              texture_clear(VK_FORMAT_R16G16_UINT, false, {0xffffu, 0x7fffu, 0, 0}, 0, 1, 1),
              texture_clear(VK_FORMAT_R16G16_UINT, false, {0x7c01u, 0xfc01u, 0, 0}, 0, 1, 1, {{2, 2, 6, 6}}),
              texture_clear(VK_FORMAT_R16G16_SINT, false, {0xffffu, 0x8000u, 0, 0}, 1, 0, 1),
              texture_clear(VK_FORMAT_R16G16_SNORM, true, floats(1.0f, -1.0f, 0.0f, 0.0f), 1, 1, 1)}},
            {"R16G16B16A16_TYPELESS through UNORM, UINT, SINT and SNORM", DXGI_FORMAT_R16G16B16A16_TYPELESS,
             VK_FORMAT_R16G16B16A16_SFLOAT, 8, 8, 2, 2, false,
             {texture_clear(VK_FORMAT_R16G16B16A16_UNORM, true, floats(1.0f, 0.5f, 0.0f, 1.0f), 0, 0, 2),
              texture_clear(VK_FORMAT_R16G16B16A16_UINT, false, {0xffffu, 0x7fffu, 0x7c01u, 0xfc01u}, 0, 1, 1),
              texture_clear(VK_FORMAT_R16G16B16A16_SINT, true, floats(-1.0f, 32767.0f, -32768.0f, 0.0f), 0, 0, 1,
                            {{1, 2, 6, 5}}),
              texture_clear(VK_FORMAT_R16G16B16A16_SNORM, true, floats(1.0f, -1.0f, 0.5f, 0.0f), 1, 0, 2)}},
            {"R8G8B8A8_TYPELESS through UINT, SINT and SNORM", DXGI_FORMAT_R8G8B8A8_TYPELESS,
             VK_FORMAT_R8G8B8A8_UNORM, 8, 8, 2, 1, false,
             {texture_clear(VK_FORMAT_R8G8B8A8_UINT, false, {0xffu, 0x80u, 0x7fu, 0x01u}, 0, 0, 1),
              texture_clear(VK_FORMAT_R8G8B8A8_SINT, true, floats(-128.0f, -1.0f, 127.0f, 0.0f), 0, 0, 1,
                            {{1, 1, 5, 6}}),
              texture_clear(VK_FORMAT_R8G8B8A8_SNORM, true, floats(1.0f, -1.0f, 0.5f, -0.5f), 1, 0, 1)}},
            {"R16_TYPELESS through UINT, SFLOAT and SNORM", DXGI_FORMAT_R16_TYPELESS,
             VK_FORMAT_R16_UNORM, 8, 4, 2, 1, false,
             {texture_clear(VK_FORMAT_R16_UINT, false, {0xffffu, 0, 0, 0}, 0, 0, 1),
              texture_clear(VK_FORMAT_R16_SFLOAT, true, floats(1.0f, 0, 0, 0), 0, 0, 1, {{2, 1, 6, 3}}),
              texture_clear(VK_FORMAT_R16_SNORM, true, floats(1.0f, 0, 0, 0), 1, 0, 1)}},
            {"RGBA16_FLOAT rectangles", DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 16, 16, 2, 2,
             false,
             {texture_clear(VK_FORMAT_R16G16B16A16_SFLOAT, true, floats(0.25f, -8.0f, 1.0f, 0.0f), 1, 0, 2,
                            {{0, 0, 3, 3}, {5, 2, 100, 4}}),
              texture_clear(VK_FORMAT_R16G16B16A16_SFLOAT, false, {0x3c00, 0x4000, 0xc000, 0x7c00}, 0, 1, 1)}},
            {"R11G11B10_FLOAT", DXGI_FORMAT_R11G11B10_FLOAT, VK_FORMAT_B10G11R11_UFLOAT_PACK32, 8, 8, 1, 1, false,
             {texture_clear(VK_FORMAT_B10G11R11_UFLOAT_PACK32, true, floats(1.0f, 2.0f, 0.5f, 0.0f), 0, 0, 1)}},
            {"R10G10B10A2_UNORM", DXGI_FORMAT_R10G10B10A2_UNORM, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 8, 8, 1, 1, false,
             {texture_clear(VK_FORMAT_A2B10G10R10_UNORM_PACK32, true, floats(1.0f, 0.0f, 0.5f, 1.0f), 0, 0, 1, {{0, 0, 4, 8}})}},
            {"R8_UNORM", DXGI_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, 16, 4, 1, 1, false,
             {texture_clear(VK_FORMAT_R8_UNORM, true, floats(0.75f, 0, 0, 0), 0, 0, 1, {{3, 0, 13, 2}})}},
            {"BGRA8_UNORM", DXGI_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, 4, 4, 1, 1, false,
             {texture_clear(VK_FORMAT_B8G8R8A8_UNORM, true, floats(1.0f, 0.5f, 0.0f, 1.0f), 0, 0, 1)}},
            {"RGBA8_UNORM volume, one mip", DXGI_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, 4, 4, 2, 4, true,
             {texture_clear(VK_FORMAT_R8G8B8A8_UNORM, false, {1, 2, 3, 4}, 1, 0, 1)}},
        };
        std::vector<TextureCase> all_textures = textures;
        {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT e5_support{DXGI_FORMAT_R9G9B9E5_SHAREDEXP};
            if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &e5_support, sizeof(e5_support))) &&
                (e5_support.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)) {
                // Full-view Float and Uint clears work whether R9G9B9E5 UAV is native or backed by mutable R32_UINT.
                TextureCase e5{"R9G9B9E5_SHAREDEXP array, Float and Uint", DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
                               VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 8, 8, 2, 2, false,
                               {texture_clear(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(1.0f, 2.0f, 4.0f, 0.0f),
                                              0, 0, 2),
                                texture_clear(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true,
                                              floats(0.5f, 0.25f, 0.125f, 0.0f), 1, 1, 1),
                                texture_clear(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, false,
                                              {0x3f800000u, 0xffffffffu, 1u, 2u}, 0, 1, 1)}};
                VkFormatProperties e5_props{};
                d.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(
                    d.physical, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, &e5_props);
                const bool native_storage = (e5_props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
                const bool native_attach = (e5_props.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0;
                if (!native_storage || native_attach) {
                    // Emulated R32_UINT images (or native ones with COLOR_ATTACHMENT) also clear partial rectangles.
                    e5.clears.push_back(texture_clear(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true,
                                                      floats(100.0f, 0.001f, 0.5f, 0.0f), 0, 0, 1, {{1, 2, 5, 6}}));
                    e5.clears.push_back(texture_clear(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, false,
                                                      {0x12345678u, 0x9abcdef0u, 0u, 0u}, 0, 0, 1, {{2, 3, 4, 5}}));
                }
                all_textures.push_back(std::move(e5));
            }
        }
        {
            // A4B4G4R4's typed UAV is the driver's call (WARP has it, AMD's D3D12 driver doesn't); where the layer
            // reports one, its clears must work. The element layout itself is checked in conversion_checks.
            D3D12_FEATURE_DATA_FORMAT_SUPPORT a4_support{DXGI_FORMAT_A4B4G4R4_UNORM};
            if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &a4_support, sizeof(a4_support))) &&
                (a4_support.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)) {
                all_textures.push_back(
                    {"A4B4G4R4_UNORM, Float and Uint", DXGI_FORMAT_A4B4G4R4_UNORM, VK_FORMAT_R4G4B4A4_UNORM_PACK16, 8, 8,
                     1, 1, false,
                     {texture_clear(VK_FORMAT_R4G4B4A4_UNORM_PACK16, true, floats(1.0f, 0.0f, 0.2f, 0.0f), 0, 0, 1),
                      texture_clear(VK_FORMAT_R4G4B4A4_UNORM_PACK16, false, {1, 2, 3, 4}, 0, 0, 1)}});
            } else {
                std::printf("A4B4G4R4_UNORM: no typed UAV on this device, its clears skipped\n");
            }
        }
        for (const auto& test : all_textures)
            if (run_texture(h, test)) return 1;

        // Refused while recording (Close says E_INVALIDARG): no values, an empty slot, no resource.
        {
            Ref<ID3D12Resource> buffer = h.committed(buffer_desc(256), D3D12_HEAP_TYPE_DEFAULT);
            const UINT values[4] = {};
            const auto check = [&](auto&& body) {
                HRESULT close = S_OK;
                h.run(close, body);
                return close;
            };
            const auto good = h.write(0, buffer_view(buffer->GetGPUVirtualAddress(), whole(64)));
            std::memset(h.slots + h.step, 0, h.step);
            EXPECT(check([&](ID3D12GraphicsCommandList* l) {
                l->ClearUnorderedAccessViewUint({0}, good, buffer.p, nullptr, 0, nullptr);
            }) == E_INVALIDARG);
            EXPECT(check([&](ID3D12GraphicsCommandList* l) {
                l->ClearUnorderedAccessViewUint({0}, h.slot(1), buffer.p, values, 0, nullptr);
            }) == E_INVALIDARG);
            EXPECT(check([&](ID3D12GraphicsCommandList* l) {
                l->ClearUnorderedAccessViewUint({0}, good, nullptr, values, 0, nullptr);
            }) == E_INVALIDARG);
            const D3D12_RECT inside_out{5, 0, 2, 1};
            EXPECT(check([&](ID3D12GraphicsCommandList* l) {
                l->ClearUnorderedAccessViewUint({0}, good, buffer.p, values, 1, &inside_out);
            }) == E_INVALIDARG);

            // Refused at execution, before anything is submitted: a buffer view of a texture, a texture view of a
            // buffer, a mip the texture does not have, a buffer view past the end, a rectangle clear of a volume.
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width = desc.Height = 4;
            desc.DepthOrArraySize = desc.MipLevels = 1;
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.SampleDesc.Count = 1;
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            Ref<ID3D12Resource> texture = h.committed(desc, D3D12_HEAP_TYPE_DEFAULT);
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
            desc.DepthOrArraySize = 4;
            Ref<ID3D12Resource> volume = h.committed(desc, D3D12_HEAP_TYPE_DEFAULT);
            g::Image image{};
            image.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
            image.width = image.height = 4;
            const auto t_view = h.write(2, g::encode(image));
            image.first_mip = 1;
            image.resource_mips = 2;
            const auto t_mip1 = h.write(3, g::encode(image));
            image = {};
            image.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
            image.type = g::ImageType::image_3d;
            image.width = image.height = image.depth = 4;
            const auto v_view = h.write(4, g::encode(image));
            Clear past = whole(64);
            past.offset = 512;
            const auto past_view = h.write(5, buffer_view(buffer->GetGPUVirtualAddress(), past));
            const D3D12_RECT rect{0, 0, 2, 2};
            const auto replay = [&](auto&& body) {
                HRESULT close = E_FAIL;
                const auto result = h.run(close, body);
                return close == S_OK ? result : leonida::ReplayResult::success;
            };
            EXPECT(replay([&](ID3D12GraphicsCommandList* l) {
                l->ClearUnorderedAccessViewUint({0}, good, texture.p, values, 0, nullptr);
            }) == leonida::ReplayResult::invalid_clear);
            EXPECT(replay([&](ID3D12GraphicsCommandList* l) {
                l->ClearUnorderedAccessViewUint({0}, t_view, buffer.p, values, 0, nullptr);
            }) == leonida::ReplayResult::invalid_clear);
            EXPECT(replay([&](ID3D12GraphicsCommandList* l) {
                l->ClearUnorderedAccessViewUint({0}, t_mip1, texture.p, values, 0, nullptr);
            }) == leonida::ReplayResult::invalid_subresource);
            EXPECT(replay([&](ID3D12GraphicsCommandList* l) {
                l->ClearUnorderedAccessViewUint({0}, past_view, buffer.p, values, 0, nullptr);
            }) == leonida::ReplayResult::invalid_subresource);
            EXPECT(replay([&](ID3D12GraphicsCommandList* l) {
                l->ClearUnorderedAccessViewUint({0}, v_view, volume.p, values, 1, &rect);
            }) == leonida::ReplayResult::unsupported);
        }
    }

    if (mismatches) {
        std::fprintf(stderr, "%u bytes differ\n", mismatches);
        return 1;
    }
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("ClearUnorderedAccessViewUint/Float: raw, structured and typed buffers and 2D, array and volume "
                "textures, whole views and rectangles, all matched\n");
    return 0;
}
