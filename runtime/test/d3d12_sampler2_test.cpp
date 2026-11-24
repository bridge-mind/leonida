// ID3D12Device11::CreateSampler2 on the Vulkan-backed device (D3D-28, part 2), compared with CreateSampler: the S#
// bytes it writes for descriptions CreateSampler can express, for UINT border colours (the S#'s border types are
// the same for integers) and for non-normalized coordinates (CreateSampler's S# with FORCE_UNNORMALIZED). Which
// descriptions the PC runtime refuses, and the D3D12_OPTIONS17 answer, are compared with the host's runtime live
// when there is one: a refused description leaves the slot as it was. No shaders or game data. Set
// LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails.
#include "vk_test_device.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

static PFN_vkGetInstanceProcAddr g_real_instance_proc = nullptr;
static PFN_vkGetDeviceProcAddr g_real_device_proc = nullptr;
static PFN_vkCreateSampler g_real_create_sampler = nullptr;
static std::vector<VkSamplerCreateInfo> g_intercepted_samplers;

static VkResult VKAPI_CALL intercept_vkCreateSampler(VkDevice device, const VkSamplerCreateInfo* pCreateInfo,
                                                     const VkAllocationCallbacks* pAllocator, VkSampler* pSampler) {
    if (pCreateInfo) g_intercepted_samplers.push_back(*pCreateInfo);
    return g_real_create_sampler(device, pCreateInfo, pAllocator, pSampler);
}

static PFN_vkVoidFunction VKAPI_CALL intercept_vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    if (pName && std::strcmp(pName, "vkCreateSampler") == 0) {
        g_real_create_sampler = reinterpret_cast<PFN_vkCreateSampler>(g_real_device_proc(device, pName));
        return reinterpret_cast<PFN_vkVoidFunction>(intercept_vkCreateSampler);
    }
    return g_real_device_proc(device, pName);
}

static PFN_vkVoidFunction VKAPI_CALL intercept_vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (pName && std::strcmp(pName, "vkGetDeviceProcAddr") == 0) {
        g_real_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(g_real_instance_proc(instance, pName));
        return reinterpret_cast<PFN_vkVoidFunction>(intercept_vkGetDeviceProcAddr);
    }
    return g_real_instance_proc(instance, pName);
}

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
    T* operator->() const { return p; }
};

constexpr unsigned char kUnwritten = 0xcd;

// A CPU sampler heap on any device, its slots filled with kUnwritten before each write.
struct Slots {
    Ref<ID3D12DescriptorHeap> heap;
    SIZE_T start = 0;
    UINT size = 0;
    UINT next = 0;
    static constexpr UINT kCount = 256;

    bool create(ID3D12Device* device) {
        const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, kCount,
                                              D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p)))) return false;
        start = heap->GetCPUDescriptorHandleForHeapStart().ptr;
        size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        return start && size;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE take() {
        const D3D12_CPU_DESCRIPTOR_HANDLE h{start + SIZE_T(next++ % kCount) * size};
        std::memset(reinterpret_cast<void*>(h.ptr), kUnwritten, size);
        return h;
    }
    std::vector<unsigned char> bytes(D3D12_CPU_DESCRIPTOR_HANDLE h) const {
        const auto* p = reinterpret_cast<const unsigned char*>(h.ptr);
        return {p, p + size};
    }
    bool unwritten(D3D12_CPU_DESCRIPTOR_HANDLE h) const {
        for (const unsigned char b : bytes(h))
            if (b != kUnwritten) return false;
        return true;
    }
};

D3D12_SAMPLER_DESC sampler(D3D12_FILTER filter, D3D12_TEXTURE_ADDRESS_MODE address, FLOAT max_lod = 1000.0f) {
    D3D12_SAMPLER_DESC d{};
    d.Filter = filter;
    d.AddressU = d.AddressV = d.AddressW = address;
    d.MaxAnisotropy = 1;
    d.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    d.MaxLOD = max_lod;
    return d;
}

D3D12_SAMPLER_DESC2 desc2(const D3D12_SAMPLER_DESC& d, D3D12_SAMPLER_FLAGS flags = D3D12_SAMPLER_FLAG_NONE) {
    D3D12_SAMPLER_DESC2 out{};
    out.Filter = d.Filter;
    out.AddressU = d.AddressU;
    out.AddressV = d.AddressV;
    out.AddressW = d.AddressW;
    out.MipLODBias = d.MipLODBias;
    out.MaxAnisotropy = d.MaxAnisotropy;
    out.ComparisonFunc = d.ComparisonFunc;
    std::memcpy(out.FloatBorderColor, d.BorderColor, sizeof(out.FloatBorderColor));
    out.MinLOD = d.MinLOD;
    out.MaxLOD = d.MaxLOD;
    out.Flags = flags;
    return out;
}

D3D12_SAMPLER_DESC with_border(D3D12_SAMPLER_DESC d, FLOAT r, FLOAT g, FLOAT b, FLOAT a) {
    d.BorderColor[0] = r;
    d.BorderColor[1] = g;
    d.BorderColor[2] = b;
    d.BorderColor[3] = a;
    return d;
}

D3D12_SAMPLER_DESC2 with_uint_border(D3D12_SAMPLER_DESC2 d, UINT r, UINT g, UINT b, UINT a) {
    d.Flags = D3D12_SAMPLER_FLAGS(d.Flags | D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR);
    d.UintBorderColor[0] = r;
    d.UintBorderColor[1] = g;
    d.UintBorderColor[2] = b;
    d.UintBorderColor[3] = a;
    return d;
}

// Descriptions CreateSampler expresses: CreateSampler2 without flags must write the same bytes.
std::vector<D3D12_SAMPLER_DESC> plain_cases() {
    const D3D12_SAMPLER_DESC border = sampler(D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_BORDER);
    D3D12_SAMPLER_DESC aniso = sampler(D3D12_FILTER_ANISOTROPIC, D3D12_TEXTURE_ADDRESS_MODE_WRAP);
    aniso.MaxAnisotropy = 8;
    aniso.MipLODBias = -1.5f;
    D3D12_SAMPLER_DESC shadow =
        sampler(D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 4.0f);
    shadow.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    D3D12_SAMPLER_DESC lods = sampler(D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_MIRROR);
    lods.MinLOD = 1.25f;
    lods.MaxLOD = 6.5f;
    D3D12_SAMPLER_DESC mixed = sampler(D3D12_FILTER_MAXIMUM_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP);
    mixed.AddressV = D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
    mixed.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    return {
        border,
        with_border(border, 0, 0, 0, 1),
        with_border(border, 1, 1, 1, 1),
        with_border(border, 0.25f, 0.5f, 0, 1),
        aniso,
        shadow,
        lods,
        mixed,
        sampler(D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0.0f),
        sampler(D3D12_FILTER_MINIMUM_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP),
    };
}

struct Edge {
    const char* name;
    bool written, want;  // `want`: whether the PC runtime writes the slot (RTX 4090, driver 621.14, and WARP)
    // The description asks for non-normalized coordinates: a host runtime without D3D12_OPTIONS17's
    // NonNormalizedCoordinateSamplersSupported answers it differently, so it is not compared with such a host.
    bool non_normalized;
};

// Which descriptions are written and which leave the slot as it was, on any ID3D12Device11: ours, or the host's PC
// runtime to compare with.
std::vector<Edge> edge_answers(ID3D12Device11* device, Slots& slots) {
    const auto edge = [&](const char* name, const D3D12_SAMPLER_DESC2* d, bool want) {
        const D3D12_CPU_DESCRIPTOR_HANDLE h = slots.take();
        device->CreateSampler2(d, h);
        const bool non_normalized = d && (d->Flags & D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES);
        return Edge{name, !slots.unwritten(h), want, non_normalized};
    };
    const D3D12_SAMPLER_DESC2 point = desc2(sampler(D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0),
                                            D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES);
    using D = D3D12_SAMPLER_DESC2;
    const auto nn = [&](const char* name, auto change, bool want) {
        D d = point;
        change(d);
        return edge(name, &d, want);
    };
    D unknown = point;
    unknown.Flags = D3D12_SAMPLER_FLAGS(0x4);
    D both = with_uint_border(point, 0, 0, 0, 1);
    both.AddressU = both.AddressV = both.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    D uint_custom = with_uint_border(desc2(sampler(D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_BORDER)),
                                     7, 0x40000000, 1, 1);
    return {
        edge("null description", nullptr, false),
        edge("unknown flag 0x4", &unknown, false),
        edge("UINT border, custom colour", &uint_custom, true),
        edge("UINT border and non-normalized", &both, true),
        edge("non-normalized, point, clamp, LOD 0", &point, true),
        nn("non-normalized, linear, mip point", [](D& d) { d.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT; }, true),
        nn("non-normalized, mip linear", [](D& d) { d.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR; }, false),
        nn("non-normalized, min point mag linear",
           [](D& d) { d.Filter = D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT; }, false),
        nn("non-normalized, anisotropic", [](D& d) { d.Filter = D3D12_FILTER_ANISOTROPIC; d.MaxAnisotropy = 8; },
           false),
        nn("non-normalized, comparison",
           [](D& d) { d.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT; d.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS; },
           false),
        nn("non-normalized, minimum", [](D& d) { d.Filter = D3D12_FILTER_MINIMUM_MIN_MAG_MIP_POINT; }, true),
        nn("non-normalized, maximum linear", [](D& d) { d.Filter = D3D12_FILTER_MAXIMUM_MIN_MAG_LINEAR_MIP_POINT; },
           true),
        nn("non-normalized, MaxAnisotropy 16 on a point filter", [](D& d) { d.MaxAnisotropy = 16; }, true),
        nn("non-normalized, ComparisonFunc on a point filter",
           [](D& d) { d.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS; }, true),
        nn("non-normalized, AddressU wrap", [](D& d) { d.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP; }, false),
        nn("non-normalized, AddressV mirror", [](D& d) { d.AddressV = D3D12_TEXTURE_ADDRESS_MODE_MIRROR; }, false),
        nn("non-normalized, AddressW wrap", [](D& d) { d.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP; }, true),
        nn("non-normalized, border", [](D& d) { d.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER; }, true),
        nn("non-normalized, MaxLOD 5", [](D& d) { d.MaxLOD = 5; }, false),
        nn("non-normalized, MaxLOD -0", [](D& d) { d.MaxLOD = -0.0f; }, true),
        nn("non-normalized, MinLOD and MaxLOD 1", [](D& d) { d.MinLOD = d.MaxLOD = 1; }, false),
        nn("non-normalized, MipLODBias 1", [](D& d) { d.MipLODBias = 1; }, false),
    };
}

leonida::gfx10::SamplerWords words_of(const std::vector<unsigned char>& bytes) {
    leonida::gfx10::SamplerWords w{};
    std::memcpy(w.data(), bytes.data(), sizeof(w));
    return w;
}

}  // namespace

int main() {
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
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.custom_border_color = d.custom_border_color;
        info.custom_border_color_without_format = d.custom_border_color_without_format;
        Ref<ID3D12Device11> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Slots slots;
        EXPECT(slots.create(device.p));

        const auto via1 = [&](const D3D12_SAMPLER_DESC& desc) {
            const D3D12_CPU_DESCRIPTOR_HANDLE h = slots.take();
            device->CreateSampler(&desc, h);
            return slots.bytes(h);
        };
        const auto via2 = [&](const D3D12_SAMPLER_DESC2& desc) {
            const D3D12_CPU_DESCRIPTOR_HANDLE h = slots.take();
            device->CreateSampler2(&desc, h);
            return slots.bytes(h);
        };

        // Without flags, the same bytes as CreateSampler.
        int plain = 0;
        for (const D3D12_SAMPLER_DESC& desc : plain_cases()) {
            EXPECT(via2(desc2(desc)) == via1(desc));
            ++plain;
        }

        // A comparison that always returns zero still needs a comparison-enabled Vulkan sampler,
        // alongside an ordinary sampler with otherwise identical settings in the same set.
        const auto regular = sampler(D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
        auto comparison = regular;
        comparison.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
        const auto regular_words = words_of(via1(regular));
        const auto never_words = words_of(via1(comparison));
        EXPECT(never_words != regular_words);
        EXPECT(words_of(via2(desc2(comparison))) == never_words);
        const auto regular_info = leonida::gfx10::vulkan_sampler(leonida::gfx10::decode(regular_words));
        const auto never_info = leonida::gfx10::vulkan_sampler(leonida::gfx10::decode(never_words));
        EXPECT(!regular_info.compareEnable);
        EXPECT(never_info.compareEnable && never_info.compareOp == VK_COMPARE_OP_NEVER);

        // UINT border colours take the S#'s border types as the float ones do, marked as integers in
        // BORDER_COLOR_PTR (kIntegerBorderPtr), so that they are their own key and VK_BORDER_COLOR_INT_* in Vulkan.
        // Anything else is the application's colour, including the bits of 1.0f, which as integers are not white.
        const D3D12_SAMPLER_DESC border = sampler(D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_BORDER);
        const D3D12_SAMPLER_DESC2 border2 = desc2(border);
        const auto marked = [](std::vector<unsigned char> bytes) {
            leonida::gfx10::SamplerWords w = words_of(bytes);
            w[3] |= leonida::gfx10::kIntegerBorderPtr;
            std::memcpy(bytes.data(), w.data(), sizeof(w));
            return bytes;
        };
        for (const std::array<UINT, 4> c : {std::array<UINT, 4>{0, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 1, 1}}) {
            const std::vector<unsigned char> got = via2(with_uint_border(border2, c[0], c[1], c[2], c[3]));
            EXPECT(got == marked(via1(with_border(border, FLOAT(c[0]), FLOAT(c[1]), FLOAT(c[2]), FLOAT(c[3])))));
            const leonida::gfx10::Sampler s = leonida::gfx10::decode(words_of(got));
            EXPECT(s.integer_border && s.border_color_index == 0);
            EXPECT(leonida::gfx10::vulkan_sampler(s).borderColor ==
                   (c[3] == 0   ? VK_BORDER_COLOR_INT_TRANSPARENT_BLACK
                    : c[0] == 0 ? VK_BORDER_COLOR_INT_OPAQUE_BLACK
                                : VK_BORDER_COLOR_INT_OPAQUE_WHITE));
        }
        // Custom float border colours and custom UINT border colours are registered in the device table
        // (D3D-73) and mapped to VK_BORDER_COLOR_FLOAT_CUSTOM_EXT / VK_BORDER_COLOR_INT_CUSTOM_EXT via
        // VK_EXT_custom_border_color. Identical colours share the table index; distinct colours or integer vs
        // float with the same bit pattern get distinct table indices and distinct S# descriptors.
        const std::vector<unsigned char> custom_float_1 = via1(with_border(border, 0.25f, 0, 0, 1));
        const std::vector<unsigned char> custom_float_2 = via1(with_border(border, 0.25f, 0, 0, 1));
        EXPECT(custom_float_1 == custom_float_2);
        const leonida::gfx10::Sampler s_cust_f = leonida::gfx10::decode(words_of(custom_float_1));
        EXPECT(s_cust_f.border_color == leonida::gfx10::BorderColor::table);

        const std::vector<unsigned char> custom_uint_1 = via2(with_uint_border(border2, 7, 9, 0, 1));
        const std::vector<unsigned char> custom_uint_2 = via2(with_uint_border(border2, 7, 9, 0, 1));
        EXPECT(custom_uint_1 == custom_uint_2);
        const leonida::gfx10::Sampler s_cust_u = leonida::gfx10::decode(words_of(custom_uint_1));
        EXPECT(s_cust_u.border_color == leonida::gfx10::BorderColor::table);
        EXPECT(custom_uint_1 != custom_float_1);
        EXPECT(s_cust_f.border_color_index != s_cust_u.border_color_index);

        // Integer border with 1.0f bits is not white; it is a custom integer table colour.
        const UINT one = 0x3f800000;
        const std::vector<unsigned char> custom_uint_one = via2(with_uint_border(border2, one, one, one, one));
        EXPECT(custom_uint_one != via1(with_border(border, 1, 1, 1, 1)));
        EXPECT(leonida::gfx10::decode(words_of(custom_uint_one)).border_color == leonida::gfx10::BorderColor::table);
        // ...and without the flag, the union's bits are floats.
        D3D12_SAMPLER_DESC2 float_bits = border2;
        for (UINT& c : float_bits.UintBorderColor) c = one;
        EXPECT(via2(float_bits) == via1(with_border(border, 1, 1, 1, 1)));

        // vulkan_sampler defaults for table borders follow integer_border.
        leonida::gfx10::Sampler s_table_int{};
        s_table_int.border_color = leonida::gfx10::BorderColor::table;
        s_table_int.integer_border = true;
        EXPECT(leonida::gfx10::vulkan_sampler(s_table_int).borderColor == VK_BORDER_COLOR_INT_TRANSPARENT_BLACK);
        leonida::gfx10::Sampler s_table_flt{};
        s_table_flt.border_color = leonida::gfx10::BorderColor::table;
        s_table_flt.integer_border = false;
        EXPECT(leonida::gfx10::vulkan_sampler(s_table_flt).borderColor == VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);

        // Fallback device without VK_EXT_custom_border_color: bind_sampler selects
        // VK_BORDER_COLOR_INT_TRANSPARENT_BLACK for UINT custom borders and
        // VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK for float custom borders.
        {
            g_intercepted_samplers.clear();
            g_real_instance_proc = d.instance_proc;
            rt::d3d12::VulkanDeviceInfo fallback_info = info;
            fallback_info.get_instance_proc = intercept_vkGetInstanceProcAddr;
            fallback_info.custom_border_color = false;
            fallback_info.custom_border_color_without_format = false;
            Ref<ID3D12Device11> fallback_device;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(fallback_info, IID_PPV_ARGS(&fallback_device.p))));
            Slots fallback_slots;
            EXPECT(fallback_slots.create(fallback_device.p));

            g_intercepted_samplers.clear();
            const D3D12_CPU_DESCRIPTOR_HANDLE h_uint = fallback_slots.take();
            const D3D12_SAMPLER_DESC2 uint_s = with_uint_border(border2, 9, 33, 117, 211);
            fallback_device->CreateSampler2(&uint_s, h_uint);
            EXPECT(!g_intercepted_samplers.empty());
            EXPECT(g_intercepted_samplers.back().borderColor == VK_BORDER_COLOR_INT_TRANSPARENT_BLACK);

            g_intercepted_samplers.clear();
            const D3D12_CPU_DESCRIPTOR_HANDLE h_flt = fallback_slots.take();
            const D3D12_SAMPLER_DESC flt_s = with_border(border, 0.25f, 0.5f, 0.75f, 1.0f);
            fallback_device->CreateSampler(&flt_s, h_flt);
            EXPECT(!g_intercepted_samplers.empty());
            EXPECT(g_intercepted_samplers.back().borderColor == VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
        }

        // Non-normalized coordinates: CreateSampler's S# with FORCE_UNNORMALIZED set, nothing else changed.
        for (const D3D12_FILTER filter : {D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
                                          D3D12_FILTER_MINIMUM_MIN_MAG_MIP_POINT}) {
            D3D12_SAMPLER_DESC desc = sampler(filter, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0.0f);
            desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            const leonida::gfx10::SamplerWords normalized = words_of(via1(desc));
            const leonida::gfx10::SamplerWords got =
                words_of(via2(desc2(desc, D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES)));
            leonida::gfx10::SamplerWords want = normalized;
            want[0] |= 1u << 15;
            EXPECT(got == want);
            EXPECT(leonida::gfx10::decode(got).unnormalized);
            EXPECT(!leonida::gfx10::decode(normalized).unnormalized);
        }

        // A null handle is ignored.
        const D3D12_SAMPLER_DESC2 any = desc2(border);
        device->CreateSampler2(&any, D3D12_CPU_DESCRIPTOR_HANDLE{0});

        // D3D12_OPTIONS17: non-normalized samplers, no manual write tracking.
        D3D12_FEATURE_DATA_D3D12_OPTIONS17 ours17{};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS17, &ours17, sizeof(ours17))));
        EXPECT(ours17.NonNormalizedCoordinateSamplersSupported && !ours17.ManualWriteTrackingResourceSupported);

        // The edge cases, each with the PC runtime's answer, and compared with it live when the host has a device.
        const std::vector<Edge> ours = edge_answers(device.p, slots);
        for (const Edge& e : ours) {
            if (e.written == e.want) continue;
            std::fprintf(stderr, "%s: written %d, want %d\n", e.name, e.written, e.want);
            return 1;
        }
        {
            const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            const auto create_host = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                                 reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                                           : nullptr;
            Ref<ID3D12Device11> host;
            Slots host_slots;
            if (create_host && !hostwine::host_is_wine() &&
                SUCCEEDED(create_host(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p))) &&
                host_slots.create(host.p)) {
                D3D12_FEATURE_DATA_D3D12_OPTIONS17 theirs17{};
                EXPECT(SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS17, &theirs17,
                                                           sizeof(theirs17))));
                // Without non-normalized samplers the host leaves those descriptions unwritten, so only the other
                // edge cases are compared with it; the layer's answers to all of them were checked above.
                const bool host_nn = theirs17.NonNormalizedCoordinateSamplersSupported;
                EXPECT(!host_nn || ours17.NonNormalizedCoordinateSamplersSupported);
                const std::vector<Edge> theirs = edge_answers(host.p, host_slots);
                std::size_t compared = 0;
                for (std::size_t i = 0; i < ours.size(); ++i) {
                    if (ours[i].non_normalized && !host_nn) continue;
                    ++compared;
                    if (ours[i].written == theirs[i].written) continue;
                    std::fprintf(stderr, "%s: ours written %d, host runtime %d\n", ours[i].name, ours[i].written,
                                 theirs[i].written);
                    return 1;
                }
                if (!host_nn)
                    std::printf("host reports no non-normalized samplers: %zu of %zu edge cases compared with it, "
                                "the non-normalized ones against their recorded answers only\n",
                                compared, ours.size());
                // The host also tells the UINT and float border colours apart, and normalized from not.
                const auto host2 = [&](const D3D12_SAMPLER_DESC2& desc) {
                    const D3D12_CPU_DESCRIPTOR_HANDLE h = host_slots.take();
                    host->CreateSampler2(&desc, h);
                    return host_slots.bytes(h);
                };
                const D3D12_SAMPLER_DESC point = sampler(D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0);
                if (host_nn)
                    EXPECT(host2(desc2(point)) != host2(desc2(point, D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES)));
            } else {
                std::printf("no host D3D12 device: edge cases checked against their recorded answers only\n");
            }
        }

        if (d.validation_errors()) {
            std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
            return 1;
        }
        std::printf("D3D12 CreateSampler2: %d plain descriptions, 6 border colours and 3 non-normalized samplers "
                    "match CreateSampler; %zu edge cases match the PC runtime\n",
                    plain, ours.size());
    }
    return 0;
}
