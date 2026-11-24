// Compare what ID3D12Resource::GetHeapProperties, ID3D12DescriptorHeap::GetDesc and ID3D12CommandQueue::GetDesc
// report on the Vulkan-backed device with Windows D3D12 (the layer's own adapter and WARP). On a single adapter a node
// mask of 0 reads back as the one node (1), and a committed resource reports the heap flags the runtime resolves:
// the caller's, plus the DENY bits of the
// resource classes it is not (a buffer 0xC0, a texture without render-target or depth use 0x44, the rest 0x84).
// No game data or shaders. Run explicitly (needs Vulkan hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <utility>
#include <vector>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

D3D12_RESOURCE_DESC buffer(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.SampleDesc.Count = 1;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

D3D12_RESOURCE_DESC texture(D3D12_RESOURCE_FLAGS flags, DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = 64;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

struct HeapCase {
    const char* name;
    D3D12_RESOURCE_DESC desc;
    D3D12_HEAP_TYPE type;
    UINT creation_mask, visible_mask;
    D3D12_HEAP_FLAGS flags;
};

struct HeapView {
    D3D12_HEAP_TYPE type;
    D3D12_CPU_PAGE_PROPERTY cpu_page;
    D3D12_MEMORY_POOL pool;
    UINT creation_mask, visible_mask;
    D3D12_HEAP_FLAGS flags;
};

HeapView read(ID3D12Resource* resource) {
    D3D12_HEAP_PROPERTIES heap{};
    D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE;
    resource->GetHeapProperties(&heap, &flags);
    return {heap.Type, heap.CPUPageProperty, heap.MemoryPoolPreference, heap.CreationNodeMask, heap.VisibleNodeMask,
            flags};
}

void print(const char* who, const HeapView& v) {
    std::fprintf(stderr, "  %-6s type %d cpu %d pool %d masks %u/%u flags 0x%x\n", who, int(v.type), int(v.cpu_page),
                 int(v.pool), v.creation_mask, v.visible_mask, unsigned(v.flags));
}

bool same(const HeapView& a, const HeapView& b) {
    return a.type == b.type && a.cpu_page == b.cpu_page && a.pool == b.pool && a.creation_mask == b.creation_mask &&
           a.visible_mask == b.visible_mask && a.flags == b.flags;
}

struct Totals {
    unsigned matched = 0, mismatched = 0, host_refused = 0, ours_refused = 0;
};

Totals compare_heaps(ID3D12Device* ours, ID3D12Device* host, const char* host_name,
                     const std::vector<HeapCase>& cases) {
    Totals totals;
    for (const HeapCase& c : cases) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = c.type;
        heap.CreationNodeMask = c.creation_mask;
        heap.VisibleNodeMask = c.visible_mask;
        const D3D12_RESOURCE_STATES state = c.type == D3D12_HEAP_TYPE_DEFAULT   ? D3D12_RESOURCE_STATE_COMMON
                                            : c.type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
                                                                                 : D3D12_RESOURCE_STATE_GENERIC_READ;
        Ref<ID3D12Resource> expected, actual;
        if (FAILED(host->CreateCommittedResource(&heap, c.flags, &c.desc, state, nullptr,
                                                 IID_PPV_ARGS(&expected.p))) ||
            !expected.p) {
            ++totals.host_refused;
            continue;
        }
        const HRESULT hr =
            ours->CreateCommittedResource(&heap, c.flags, &c.desc, state, nullptr, IID_PPV_ARGS(&actual.p));
        if (FAILED(hr) || !actual.p) {
            std::fprintf(stderr, "%s: not created by the layer (0x%08lx), not compared with %s\n", c.name,
                         static_cast<unsigned long>(hr), host_name);
            ++totals.ours_refused;
            continue;
        }
        const HeapView a = read(actual.p), e = read(expected.p);
        if (same(a, e)) {
            ++totals.matched;
            continue;
        }
        ++totals.mismatched;
        std::fprintf(stderr, "%s: GetHeapProperties differs from %s\n", c.name, host_name);
        std::fprintf(stderr, "  asked  masks %u/%u flags 0x%x\n", c.creation_mask, c.visible_mask, unsigned(c.flags));
        print("ours", a);
        print("host", e);
    }
    return totals;
}

struct DescriptorCase {
    const char* name;
    D3D12_DESCRIPTOR_HEAP_DESC desc;
};

bool same(const D3D12_DESCRIPTOR_HEAP_DESC& a, const D3D12_DESCRIPTOR_HEAP_DESC& b) {
    return a.Type == b.Type && a.NumDescriptors == b.NumDescriptors && a.Flags == b.Flags && a.NodeMask == b.NodeMask;
}

void print(const char* who, const D3D12_DESCRIPTOR_HEAP_DESC& d) {
    std::fprintf(stderr, "  %-6s type %d num %u flags 0x%x node mask %u\n", who, int(d.Type), d.NumDescriptors,
                 unsigned(d.Flags), d.NodeMask);
}

Totals compare_descriptor_heaps(ID3D12Device* ours, ID3D12Device* host, const char* host_name,
                                const std::vector<DescriptorCase>& cases) {
    Totals totals;
    for (const DescriptorCase& c : cases) {
        Ref<ID3D12DescriptorHeap> expected, actual;
        if (FAILED(host->CreateDescriptorHeap(&c.desc, IID_PPV_ARGS(&expected.p))) || !expected.p) {
            ++totals.host_refused;
            continue;
        }
        if (FAILED(ours->CreateDescriptorHeap(&c.desc, IID_PPV_ARGS(&actual.p))) || !actual.p) {
            std::fprintf(stderr, "%s: not created by the layer, not compared with %s\n", c.name, host_name);
            ++totals.ours_refused;
            continue;
        }
        const D3D12_DESCRIPTOR_HEAP_DESC a = actual->GetDesc(), e = expected->GetDesc();
        if (same(a, e)) {
            ++totals.matched;
            continue;
        }
        ++totals.mismatched;
        std::fprintf(stderr, "%s: GetDesc differs from %s\n", c.name, host_name);
        print("ours", a);
        print("host", e);
    }
    return totals;
}

// Command queues of each type with node mask 0 and 1; GetDesc should read back what the host reports.
Totals compare_queues(ID3D12Device* ours, ID3D12Device* host, const char* host_name) {
    Totals totals;
    for (const auto type : {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY})
        for (const UINT mask : {0u, 1u}) {
            const D3D12_COMMAND_QUEUE_DESC desc{type, D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
                                                D3D12_COMMAND_QUEUE_FLAG_NONE, mask};
            Ref<ID3D12CommandQueue> expected, actual;
            if (FAILED(host->CreateCommandQueue(&desc, IID_PPV_ARGS(&expected.p))) || !expected.p) {
                ++totals.host_refused;
                continue;
            }
            if (FAILED(ours->CreateCommandQueue(&desc, IID_PPV_ARGS(&actual.p))) || !actual.p) {
                std::fprintf(stderr, "queue type %d, mask %u: not created by the layer\n", int(type), mask);
                ++totals.ours_refused;
                continue;
            }
            const D3D12_COMMAND_QUEUE_DESC a = actual->GetDesc(), e = expected->GetDesc();
            if (a.Type == e.Type && a.Priority == e.Priority && a.Flags == e.Flags && a.NodeMask == e.NodeMask) {
                ++totals.matched;
                continue;
            }
            ++totals.mismatched;
            std::fprintf(stderr, "queue type %d, mask %u: GetDesc differs from %s (ours node mask %u, host %u)\n",
                         int(type), mask, host_name, a.NodeMask, e.NodeMask);
        }
    return totals;
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    unsigned matched = 0, mismatched = 0, references = 0;
    {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;

        constexpr auto none = D3D12_RESOURCE_FLAG_NONE, target = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                       depth = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                       uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        constexpr auto default_heap = D3D12_HEAP_TYPE_DEFAULT;
        constexpr auto no_flags = D3D12_HEAP_FLAG_NONE;
        constexpr auto shader_visible = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        constexpr auto rgba8 = DXGI_FORMAT_R8G8B8A8_UNORM;
        const D3D12_RESOURCE_DESC buf = buffer(256), buf_uav = [&] {
            D3D12_RESOURCE_DESC d = buffer(256);
            d.Flags = uav;
            return d;
        }();
        // Each heap node mask 0 or 1 (a single adapter has one node; 2 or more is refused by the host).
        std::vector<HeapCase> heaps;
        const std::pair<UINT, UINT> masks[] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        for (const auto& [creation, visible] : masks) {
            heaps.push_back({"default buffer", buf, default_heap, creation, visible, no_flags});
            heaps.push_back({"upload buffer", buf, D3D12_HEAP_TYPE_UPLOAD, creation, visible, no_flags});
            heaps.push_back({"readback buffer", buf, D3D12_HEAP_TYPE_READBACK, creation, visible, no_flags});
        }
        // Resolved heap flags by resource class; the creation-only CREATE_NOT_ZEROED is not reported back.
        heaps.push_back({"default buffer, UAV", buf_uav, default_heap, 1, 1, no_flags});
        heaps.push_back({"default buffer, not zeroed", buf, default_heap, 1, 1, D3D12_HEAP_FLAG_CREATE_NOT_ZEROED});
        heaps.push_back({"default 2D, sampled", texture(none, rgba8), default_heap, 1, 1, no_flags});
        heaps.push_back({"default 2D, UAV", texture(uav, rgba8), default_heap, 1, 1, no_flags});
        heaps.push_back({"default 2D, render target", texture(target, rgba8), default_heap, 1, 1, no_flags});
        heaps.push_back({"default 2D, depth", texture(depth, DXGI_FORMAT_D32_FLOAT), default_heap, 1, 1, no_flags});

        std::vector<DescriptorCase> descriptors;
        for (const UINT mask : {0u, 1u}) {
            descriptors.push_back(
                {"CBV/SRV/UAV, shader visible", {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8, shader_visible, mask}});
            descriptors.push_back(
                {"sampler, shader visible", {D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 8, shader_visible, mask}});
            descriptors.push_back({"RTV", {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, mask}});
            descriptors.push_back({"DSV", {D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, mask}});
        }

        if (hostwine::host_is_wine()) return 77;  // every case is a comparison with the host
        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            std::fprintf(stderr, "no DXGI factory\n");
            return 1;
        }
        for (const bool warp : {false, true}) {
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            std::string adapter_name;
            if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            const char* name = adapter_name.c_str();
            const Totals heap_totals = compare_heaps(ours.p, host.p, name, heaps);
            const Totals desc_totals = compare_descriptor_heaps(ours.p, host.p, name, descriptors);
            const Totals queue_totals = compare_queues(ours.p, host.p, name);
            std::printf("heap properties against %s: %u of %zu match, %u differ, %u refused by the host, %u not "
                        "created by the layer\n",
                        name, heap_totals.matched, heaps.size(), heap_totals.mismatched, heap_totals.host_refused,
                        heap_totals.ours_refused);
            std::printf("descriptor heap GetDesc against %s: %u of %zu match, %u differ, %u refused by the host, "
                        "%u not created by the layer\n",
                        name, desc_totals.matched, descriptors.size(), desc_totals.mismatched, desc_totals.host_refused,
                        desc_totals.ours_refused);
            std::printf("command queue GetDesc against %s: %u of 6 match, %u differ, %u refused by the host, %u "
                        "not created by the layer\n",
                        name, queue_totals.matched, queue_totals.mismatched, queue_totals.host_refused,
                        queue_totals.ours_refused);
            matched += heap_totals.matched + desc_totals.matched + queue_totals.matched;
            mismatched += heap_totals.mismatched + desc_totals.mismatched + queue_totals.mismatched;
            ++references;
        }
    }
    if (!references) {
        std::fprintf(stderr, "no D3D12 reference device (hardware adapter or WARP)\n");
        return 1;
    }
    if (mismatched || !matched) return 1;
    if (vk.validation_errors()) return 1;
    return 0;
}
