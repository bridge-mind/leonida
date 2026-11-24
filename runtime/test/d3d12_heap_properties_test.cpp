// ID3D12Resource::GetHeapProperties on the Vulkan-backed device, compared with the host PC runtime: the properties
// (type, CPU page property, memory pool, node masks) and heap flags it reports for committed resources in every heap
// type (custom ones included) and with SHARED, ALLOW_DISPLAY, CREATE_NOT_ZEROED and CREATE_NOT_RESIDENT, for placed
// resources in heaps of each kind, and for a reserved resource, with null arguments too. d3d12_node_mask_test covers
// the node masks and DENY bits of a few committed resources; this test covers the rest. Every value is recorded,
// with whether the call wrote its outputs (they start as a sentinel), and the two records must be the same. A
// resource one device refuses to create is reported and its steps aren't compared. No shaders or game data. Skips
// (77) without a Vulkan GPU or the standalone device; without a host D3D12 device only the layer's own record is
// printed. The host runs on the layer's adapter (host_adapter.h; LEONIDA_HOST_ADAPTER picks another, "warp" WARP).
// Set LEONIDA_D3D49_VERBOSE=1 to print every value.
#include "../shims/d3d12_com.h"
#include "host_adapter.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

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

constexpr std::uint64_t kFaulted = 0xFA017EDull;
constexpr std::uint64_t kNotCreated = 0x5C1F1Dull;  // the resource couldn't be created on this device
constexpr UINT kSentinel = 0xCDCDCDCDu;

D3D12_RESOURCE_DESC buffer(UINT64 bytes = 64 * 1024) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_DESC texture(DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM,
                            D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = 64;
    d.Height = 64;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.Flags = flags;
    return d;
}

D3D12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE type) { return D3D12_HEAP_PROPERTIES{type}; }
D3D12_HEAP_PROPERTIES custom(D3D12_CPU_PAGE_PROPERTY page, D3D12_MEMORY_POOL pool) {
    return D3D12_HEAP_PROPERTIES{D3D12_HEAP_TYPE_CUSTOM, page, pool};
}

enum class Kind { committed, placed, reserved };

struct Case {
    const char* name;
    Kind kind;
    D3D12_HEAP_PROPERTIES props;  // the committed resource's, or the placed resource's heap
    D3D12_HEAP_FLAGS flags;
    D3D12_RESOURCE_DESC desc;
    D3D12_RESOURCE_STATES state;
};

const D3D12_HEAP_FLAGS kNone = D3D12_HEAP_FLAG_NONE;
const D3D12_RESOURCE_STATES kCommon = D3D12_RESOURCE_STATE_COMMON;

std::vector<Case> cases() {
    const auto rt = texture(DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    const auto ds = texture(DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    const auto uav = texture(DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto uav_buffer = buffer();
    uav_buffer.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto wb = custom(D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0);
    const auto wc = custom(D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0);
    const auto none_l0 = custom(D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE, D3D12_MEMORY_POOL_L0);
    const auto none_l1 = custom(D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE, D3D12_MEMORY_POOL_L1);
    auto masks = heap(D3D12_HEAP_TYPE_DEFAULT);
    masks.CreationNodeMask = 1;
    masks.VisibleNodeMask = 1;
    return {
        {"committed buffer, DEFAULT", Kind::committed, heap(D3D12_HEAP_TYPE_DEFAULT), kNone, buffer(), kCommon},
        {"committed UAV buffer, DEFAULT", Kind::committed, heap(D3D12_HEAP_TYPE_DEFAULT), kNone, uav_buffer, kCommon},
        {"committed buffer, UPLOAD", Kind::committed, heap(D3D12_HEAP_TYPE_UPLOAD), kNone, buffer(),
         D3D12_RESOURCE_STATE_GENERIC_READ},
        {"committed buffer, READBACK", Kind::committed, heap(D3D12_HEAP_TYPE_READBACK), kNone, buffer(),
         D3D12_RESOURCE_STATE_COPY_DEST},
        {"committed buffer, CUSTOM write-back L0", Kind::committed, wb, kNone, buffer(), kCommon},
        {"committed buffer, CUSTOM write-combine L0", Kind::committed, wc, kNone, buffer(), kCommon},
        {"committed buffer, CUSTOM not-available L0", Kind::committed, none_l0, kNone, buffer(), kCommon},
        {"committed buffer, CUSTOM not-available L1", Kind::committed, none_l1, kNone, buffer(), kCommon},
        {"committed buffer, DEFAULT with node masks 1", Kind::committed, masks, kNone, buffer(), kCommon},
        {"committed buffer, DEFAULT, SHARED", Kind::committed, heap(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_SHARED,
         buffer(), kCommon},
        {"committed buffer, DEFAULT, CREATE_NOT_ZEROED", Kind::committed, heap(D3D12_HEAP_TYPE_DEFAULT),
         D3D12_HEAP_FLAG_CREATE_NOT_ZEROED, buffer(), kCommon},
        {"committed buffer, DEFAULT, CREATE_NOT_RESIDENT", Kind::committed, heap(D3D12_HEAP_TYPE_DEFAULT),
         D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT, buffer(), kCommon},
        {"committed buffer, DEFAULT, SHARED | CREATE_NOT_ZEROED", Kind::committed, heap(D3D12_HEAP_TYPE_DEFAULT),
         D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED, buffer(), kCommon},
        {"committed texture, DEFAULT", Kind::committed, heap(D3D12_HEAP_TYPE_DEFAULT), kNone, texture(), kCommon},
        {"committed render target, DEFAULT", Kind::committed, heap(D3D12_HEAP_TYPE_DEFAULT), kNone, rt, kCommon},
        {"committed depth texture, DEFAULT", Kind::committed, heap(D3D12_HEAP_TYPE_DEFAULT), kNone, ds,
         D3D12_RESOURCE_STATE_DEPTH_WRITE},
        {"committed UAV texture, DEFAULT", Kind::committed, heap(D3D12_HEAP_TYPE_DEFAULT), kNone, uav, kCommon},
        {"committed render target, DEFAULT, ALLOW_DISPLAY", Kind::committed, heap(D3D12_HEAP_TYPE_DEFAULT),
         D3D12_HEAP_FLAG_ALLOW_DISPLAY, rt, kCommon},
        {"committed render target, DEFAULT, SHARED", Kind::committed, heap(D3D12_HEAP_TYPE_DEFAULT),
         D3D12_HEAP_FLAG_SHARED, rt, kCommon},
        {"committed texture, CUSTOM write-back L0", Kind::committed, wb, kNone, texture(), kCommon},
        {"committed texture, CUSTOM write-combine L0", Kind::committed, wc, kNone, texture(), kCommon},

        {"placed buffer, DEFAULT heap, ALLOW_ONLY_BUFFERS", Kind::placed, heap(D3D12_HEAP_TYPE_DEFAULT),
         D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS, buffer(), kCommon},
        {"placed buffer, DEFAULT heap, ALLOW_ALL", Kind::placed, heap(D3D12_HEAP_TYPE_DEFAULT),
         D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES, buffer(), kCommon},
        {"placed texture, DEFAULT heap, ALLOW_ONLY_NON_RT_DS_TEXTURES", Kind::placed, heap(D3D12_HEAP_TYPE_DEFAULT),
         D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES, texture(), kCommon},
        {"placed render target, DEFAULT heap, ALLOW_ONLY_RT_DS_TEXTURES", Kind::placed,
         heap(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES, rt, kCommon},
        {"placed texture, DEFAULT heap, ALLOW_ALL", Kind::placed, heap(D3D12_HEAP_TYPE_DEFAULT),
         D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES, texture(), kCommon},
        {"placed buffer, UPLOAD heap", Kind::placed, heap(D3D12_HEAP_TYPE_UPLOAD),
         D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS, buffer(), D3D12_RESOURCE_STATE_GENERIC_READ},
        {"placed buffer, READBACK heap", Kind::placed, heap(D3D12_HEAP_TYPE_READBACK),
         D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS, buffer(), D3D12_RESOURCE_STATE_COPY_DEST},
        {"placed buffer, CUSTOM write-back L0 heap", Kind::placed, wb, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS, buffer(),
         kCommon},
        {"placed buffer, DEFAULT heap, SHARED", Kind::placed, heap(D3D12_HEAP_TYPE_DEFAULT),
         D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_SHARED, buffer(), kCommon},
        {"placed buffer, DEFAULT heap, CREATE_NOT_ZEROED", Kind::placed, heap(D3D12_HEAP_TYPE_DEFAULT),
         D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED, buffer(), kCommon},

        {"reserved buffer", Kind::reserved, {}, kNone, buffer(), kCommon},
        {"reserved texture", Kind::reserved, {}, kNone, [] {
             auto d = texture();
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), kCommon},
    };
}

struct Record {
    std::vector<std::string> names;
    std::vector<std::uint64_t> values;
    void add(const std::string& name, std::uint64_t value) {
        names.push_back(name);
        values.push_back(value);
    }
};

HRESULT create(ID3D12Device* device, const Case& c, Ref<ID3D12Heap>& h, Ref<ID3D12Resource>& r) {
    switch (c.kind) {
    case Kind::committed:
        return device->CreateCommittedResource(&c.props, c.flags, &c.desc, c.state, nullptr, IID_PPV_ARGS(&r.p));
    case Kind::placed: {
        D3D12_HEAP_DESC hd{};
        hd.SizeInBytes = 4 * 1024 * 1024;
        hd.Properties = c.props;
        hd.Flags = c.flags;
        const HRESULT hr = device->CreateHeap(&hd, IID_PPV_ARGS(&h.p));
        if (FAILED(hr)) return hr;
        return device->CreatePlacedResource(h.p, 0, &c.desc, c.state, nullptr, IID_PPV_ARGS(&r.p));
    }
    case Kind::reserved:
        return device->CreateReservedResource(&c.desc, c.state, nullptr, IID_PPV_ARGS(&r.p));
    }
    return E_FAIL;
}

HRESULT get(ID3D12Resource* r, D3D12_HEAP_PROPERTIES* p, D3D12_HEAP_FLAGS* f) {
    __try {
        return r->GetHeapProperties(p, f);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return HRESULT(kFaulted);
    }
}

void record_call(Record& out, const std::string& what, ID3D12Resource* r, bool want_props, bool want_flags) {
    D3D12_HEAP_PROPERTIES p;
    std::memset(&p, 0xCD, sizeof(p));
    auto f = D3D12_HEAP_FLAGS(kSentinel);
    const HRESULT hr = get(r, want_props ? &p : nullptr, want_flags ? &f : nullptr);
    out.add(what + ": hr", std::uint32_t(hr));
    if (want_props) {
        out.add(what + ": Type", std::uint32_t(p.Type));
        out.add(what + ": CPUPageProperty", std::uint32_t(p.CPUPageProperty));
        out.add(what + ": MemoryPoolPreference", std::uint32_t(p.MemoryPoolPreference));
        out.add(what + ": CreationNodeMask", p.CreationNodeMask);
        out.add(what + ": VisibleNodeMask", p.VisibleNodeMask);
    }
    if (want_flags) out.add(what + ": flags", std::uint32_t(f));
}

void run(ID3D12Device* device, Record& out) {
    for (const Case& c : cases()) {
        Ref<ID3D12Heap> h;
        Ref<ID3D12Resource> r;
        const HRESULT created = create(device, c, h, r);
        out.add(std::string(c.name) + ": created", std::uint32_t(created));
        if (FAILED(created) || !r.p) {
            out.add(std::string(c.name) + " [not created]", kNotCreated);
            continue;
        }
        record_call(out, c.name, r.p, true, true);
        record_call(out, std::string(c.name) + ", properties only", r.p, true, false);
        record_call(out, std::string(c.name) + ", flags only", r.p, false, true);
        record_call(out, std::string(c.name) + ", both null", r.p, false, false);
    }
}

}  // namespace

int main() {
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))) || !device.p) {
        std::printf("heap properties: no Vulkan device, skipped\n");
        return 77;
    }
    const bool verbose = GetEnvironmentVariableA("LEONIDA_D3D49_VERBOSE", nullptr, 0) > 0;
    Record ours;
    run(device.p, ours);

    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create_host = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                         reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                                   : nullptr;
    // The host on the layer's adapter: on a PC with two GPUs D3D12's default adapter can be the other one.
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> adapter;
    std::string host_name = "the default adapter";
    const LUID luid = device->GetAdapterLuid();
    // No factory, or no adapter for the pass (LEONIDA_HOST_ADAPTER=warp without WARP): don't compare, rather than
    // fall back to the default adapter and compare with a GPU the layer doesn't run on.
    const bool have_adapter =
        SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) &&
        hostadapter::reference(factory.p, &luid, hostadapter::wanted() == "warp", &adapter.p, host_name);
    Ref<ID3D12Device> host;
    if (!have_adapter) {
        std::printf("heap properties: no host adapter to compare with; %zu values recorded, not compared\n",
                    ours.values.size());
    } else if (!create_host || FAILED(create_host(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
        std::printf("heap properties: no host D3D12 device to compare with; %zu values recorded, not compared\n",
                    ours.values.size());
    } else {
        Record theirs;
        run(host.p, theirs);
        // D3D12 refuses MEMORY_POOL_L1 on a UMA adapter. The layer answers UMA on an integrated GPU and refuses L1
        // there too, so with the host on the layer's own adapter every case is compared; a host of the other
        // architecture (LEONIDA_HOST_ADAPTER, or WARP) refusing or accepting the L1 cases is not a layer bug.
        D3D12_FEATURE_DATA_ARCHITECTURE host_arch{}, our_arch{};
        const bool uma_differs =
            SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &host_arch, sizeof(host_arch))) &&
            SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &our_arch, sizeof(our_arch))) &&
            host_arch.UMA != our_arch.UMA;
        std::vector<std::string> l1_cases;
        for (const Case& c : cases())
            if (c.props.Type == D3D12_HEAP_TYPE_CUSTOM && c.props.MemoryPoolPreference == D3D12_MEMORY_POOL_L1)
                l1_cases.emplace_back(c.name);
        const auto in_l1_case = [&](const std::string& name) {
            for (const std::string& c : l1_cases)
                if (name.compare(0, c.size(), c) == 0 && name.size() > c.size() && (name[c.size()] == ':' ||
                    name[c.size()] == ',' || name[c.size()] == ' ')) return true;
            return false;
        };
        std::printf("heap properties: host %s (UMA %d, the layer's UMA %d)\n", host_name.c_str(), host_arch.UMA,
                    our_arch.UMA);
        // A resource one side couldn't create has fewer entries: compare case by case on the names both have.
        int mismatches = 0, compared = 0, skipped = 0;
        std::size_t j = 0;
        for (std::size_t i = 0; i < ours.names.size(); ++i) {
            if (uma_differs && in_l1_case(ours.names[i])) {
                if (verbose || ours.names[i].ends_with(": created"))
                    std::printf("heap properties %s: not comparable on this host (MEMORY_POOL_L1 on a %s host)\n",
                                ours.names[i].c_str(), host_arch.UMA ? "UMA" : "NUMA");
                ++skipped;
                continue;
            }
            while (j < theirs.names.size() && theirs.names[j] != ours.names[i]) ++j;
            if (j == theirs.names.size()) {
                std::printf("heap properties %s: ours 0x%llx, the host has no such value (not compared)\n",
                            ours.names[i].c_str(), static_cast<unsigned long long>(ours.values[i]));
                ++skipped;
                j = 0;
                continue;
            }
            const std::uint64_t a = ours.values[i], b = theirs.values[j];
            if (b == kFaulted) {
                std::printf("heap properties %s: the host faults, not compared (ours 0x%llx)\n", ours.names[i].c_str(),
                            static_cast<unsigned long long>(a));
                ++skipped;
                continue;
            }
            ++compared;
            const bool same = a == b;
            if (verbose || !same)
                std::printf("heap properties %s: ours 0x%llx host 0x%llx%s\n", ours.names[i].c_str(),
                            static_cast<unsigned long long>(a), static_cast<unsigned long long>(b),
                            same ? "" : "  MISMATCH");
            mismatches += !same;
        }
        std::printf("heap properties: %d values compared with the host (%d more not compared), %d mismatches\n",
                    compared, skipped, mismatches);
        EXPECT(mismatches == 0);
    }
    std::printf("d3d12_heap_properties_test: OK\n");
    return 0;
}
