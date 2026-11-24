// Query heaps and query calls on the Vulkan-backed device against Windows D3D12 (the layer's own adapter and WARP):
// CreateQueryHeap results by type, count and node mask, Close() results of lists that use queries on each list type,
// and timestamps resolved on direct, compute and copy queues (pipeline statistics heaps are left out: the test
// device does not enable pipelineStatisticsQuery). No game data or shaders. Run explicitly (needs Vulkan
// hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
    T** put() { if (p) { p->Release(); p = nullptr; } return &p; }
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

struct Env {
    ID3D12Device* device;
    Ref<ID3D12QueryHeap> occlusion, timestamps, copy_timestamps;
    Ref<ID3D12Resource> dst;       // default buffer, COPY_DEST
    Ref<ID3D12Resource> readback;  // readback buffer
};

std::string heap_cases(ID3D12Device* device) {
    std::string out;
    const D3D12_QUERY_HEAP_TYPE types[] = {D3D12_QUERY_HEAP_TYPE_OCCLUSION, D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                                           D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP, D3D12_QUERY_HEAP_TYPE(99)};
    for (const auto type : types)
        for (const UINT count : {0u, 1u})
            for (const UINT mask : {0u, 1u, 2u, 3u}) {
                Ref<ID3D12QueryHeap> heap;
                D3D12_QUERY_HEAP_DESC desc{type, count, mask};
                const HRESULT hr = device->CreateQueryHeap(&desc, IID_PPV_ARGS(heap.put()));
                char line[128];
                std::snprintf(line, sizeof line, "heap type %d count %u mask %u: %s\n", int(type), count, mask,
                              SUCCEEDED(hr) ? "created" : "refused");
                out += line;
            }
    return out;
}

// Records one list of `type`, closes it and returns Close's result.
HRESULT close_result(Env& e, D3D12_COMMAND_LIST_TYPE type, const std::function<void(ID3D12GraphicsCommandList*)>& rec) {
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(e.device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator.p)))) return 0x7fff0001;
    if (FAILED(e.device->CreateCommandList(0, type, allocator.p, nullptr, IID_PPV_ARGS(&list.p)))) return 0x7fff0002;
    rec(list.p);
    return list->Close();
}

// Executes one list on a new queue of its type and waits; true when it ran.
bool run(Env& e, D3D12_COMMAND_LIST_TYPE type, const std::function<void(ID3D12GraphicsCommandList*)>& rec,
         UINT64* frequency = nullptr, int* result = nullptr) {
    Ref<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd{type};
    if (FAILED(e.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p)))) return false;
    if (frequency && FAILED(queue->GetTimestampFrequency(frequency))) *frequency = 0;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(e.device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator.p)))) return false;
    if (FAILED(e.device->CreateCommandList(0, type, allocator.p, nullptr, IID_PPV_ARGS(&list.p)))) return false;
    rec(list.p);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (result) *result = int(rt::d3d12::last_execute_result(queue.p));
    Ref<ID3D12Fence> fence;
    if (FAILED(e.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return false;
    queue->Signal(fence.p, 1);
    for (int i = 0; i < 5000 && fence->GetCompletedValue() < 1; ++i) Sleep(1);
    return fence->GetCompletedValue() >= 1;
}

UINT64 read_u64(Env& e, UINT64 offset) {
    void* data = nullptr;
    D3D12_RANGE range{0, 4096};
    if (FAILED(e.readback->Map(0, &range, &data))) return ~0ull;
    UINT64 v;
    std::memcpy(&v, static_cast<char*>(data) + offset, 8);
    D3D12_RANGE none{0, 0};
    e.readback->Unmap(0, &none);
    return v;
}

void fill_readback(Env& e, unsigned char byte) {
    void* data = nullptr;
    D3D12_RANGE none{0, 0};
    if (FAILED(e.readback->Map(0, &none, &data))) return;
    std::memset(data, byte, 4096);
    e.readback->Unmap(0, nullptr);
}

bool setup(Env& e) {
    D3D12_QUERY_HEAP_DESC desc{D3D12_QUERY_HEAP_TYPE_OCCLUSION, 4, 0};
    if (FAILED(e.device->CreateQueryHeap(&desc, IID_PPV_ARGS(&e.occlusion.p)))) return false;
    desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    if (FAILED(e.device->CreateQueryHeap(&desc, IID_PPV_ARGS(&e.timestamps.p)))) return false;
    desc.Type = D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP;
    e.device->CreateQueryHeap(&desc, IID_PPV_ARGS(&e.copy_timestamps.p));
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto bd = buffer(4096);
    if (FAILED(e.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                 nullptr, IID_PPV_ARGS(&e.dst.p))))
        return false;
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    return SUCCEEDED(e.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd,
                                                       D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                       IID_PPV_ARGS(&e.readback.p)));
}

std::string list_cases(Env& e) {
    std::string out;
    using L = ID3D12GraphicsCommandList*;
    constexpr auto direct = D3D12_COMMAND_LIST_TYPE_DIRECT, compute = D3D12_COMMAND_LIST_TYPE_COMPUTE,
                   copy = D3D12_COMMAND_LIST_TYPE_COPY, bundle = D3D12_COMMAND_LIST_TYPE_BUNDLE;
    auto* occ = e.occlusion.p;
    auto* ts = e.timestamps.p;
    auto* cts = e.copy_timestamps.p ? e.copy_timestamps.p : e.timestamps.p;
    auto* dst = e.dst.p;
    struct Case {
        const char* name;
        D3D12_COMMAND_LIST_TYPE type;
        std::function<void(L)> rec;
    };
    const Case cases[] = {
        {"direct timestamp", direct, [&](L l) { l->EndQuery(ts, D3D12_QUERY_TYPE_TIMESTAMP, 0); }},
        {"direct occlusion begin/end", direct,
         [&](L l) { l->BeginQuery(occ, D3D12_QUERY_TYPE_OCCLUSION, 0); l->EndQuery(occ, D3D12_QUERY_TYPE_OCCLUSION, 0); }},
        {"direct resolve timestamp", direct,
         [&](L l) { l->ResolveQueryData(ts, D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, dst, 0); }},
        {"direct end without begin", direct, [&](L l) { l->EndQuery(occ, D3D12_QUERY_TYPE_OCCLUSION, 0); }},
        {"direct begin twice", direct, [&](L l) {
             l->BeginQuery(occ, D3D12_QUERY_TYPE_OCCLUSION, 0); l->BeginQuery(occ, D3D12_QUERY_TYPE_OCCLUSION, 0);
             l->EndQuery(occ, D3D12_QUERY_TYPE_OCCLUSION, 0); }},
        {"direct begin timestamp", direct, [&](L l) { l->BeginQuery(ts, D3D12_QUERY_TYPE_TIMESTAMP, 0); }},
        {"direct begin never ended", direct, [&](L l) { l->BeginQuery(occ, D3D12_QUERY_TYPE_OCCLUSION, 0); }},
        {"direct resolve offset 4", direct,
         [&](L l) { l->ResolveQueryData(ts, D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, dst, 4); }},
        {"direct predication offset 4", direct, [&](L l) {
             l->SetPredication(dst, 4, D3D12_PREDICATION_OP_EQUAL_ZERO); l->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO); }},
        {"compute timestamp", compute, [&](L l) { l->EndQuery(ts, D3D12_QUERY_TYPE_TIMESTAMP, 0); }},
        {"compute resolve timestamp", compute,
         [&](L l) { l->ResolveQueryData(ts, D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, dst, 0); }},
        {"compute occlusion begin/end", compute,
         [&](L l) { l->BeginQuery(occ, D3D12_QUERY_TYPE_OCCLUSION, 0); l->EndQuery(occ, D3D12_QUERY_TYPE_OCCLUSION, 0); }},
        {"compute predication", compute, [&](L l) {
             l->SetPredication(dst, 0, D3D12_PREDICATION_OP_EQUAL_ZERO); l->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO); }},
        {"copy timestamp (copy heap)", copy, [&](L l) { l->EndQuery(cts, D3D12_QUERY_TYPE_TIMESTAMP, 0); }},
        {"copy resolve timestamp", copy,
         [&](L l) { l->ResolveQueryData(cts, D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, dst, 0); }},
        {"copy occlusion begin/end", copy,
         [&](L l) { l->BeginQuery(occ, D3D12_QUERY_TYPE_OCCLUSION, 0); l->EndQuery(occ, D3D12_QUERY_TYPE_OCCLUSION, 0); }},
        {"copy predication", copy, [&](L l) {
             l->SetPredication(dst, 0, D3D12_PREDICATION_OP_EQUAL_ZERO); l->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO); }},
        {"bundle occlusion begin/end", bundle,
         [&](L l) { l->BeginQuery(occ, D3D12_QUERY_TYPE_OCCLUSION, 0); l->EndQuery(occ, D3D12_QUERY_TYPE_OCCLUSION, 0); }},
        {"bundle timestamp", bundle, [&](L l) { l->EndQuery(ts, D3D12_QUERY_TYPE_TIMESTAMP, 0); }},
        {"bundle resolve", bundle, [&](L l) { l->ResolveQueryData(ts, D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, dst, 0); }},
        {"compute binary begin/end", compute, [&](L l) {
             l->BeginQuery(occ, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 0); l->EndQuery(occ, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 0); }},
        {"compute resolve occlusion", compute,
         [&](L l) { l->ResolveQueryData(occ, D3D12_QUERY_TYPE_OCCLUSION, 0, 1, dst, 0); }},
        {"copy resolve occlusion", copy,
         [&](L l) { l->ResolveQueryData(occ, D3D12_QUERY_TYPE_OCCLUSION, 0, 1, dst, 0); }},
        {"bundle predication", bundle, [&](L l) { l->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO); }},
    };
    for (const auto& c : cases) {
        char line[160];
        // Success or failure only: the layer reports E_INVALIDARG where the host sometimes gives E_FAIL.
        std::snprintf(line, sizeof line, "close %s: %s\n", c.name,
                      SUCCEEDED(close_result(e, c.type, c.rec)) ? "ok" : "fails");
        out += line;
    }
    return out;
}

// Timestamps on each queue type: written, resolved, copied to readback; reports whether each value is plausible.
std::string timestamp_runs(Env& e) {
    std::string out;
    const std::pair<const char*, D3D12_COMMAND_LIST_TYPE> types[] = {
        {"direct", D3D12_COMMAND_LIST_TYPE_DIRECT}, {"compute", D3D12_COMMAND_LIST_TYPE_COMPUTE},
        {"copy", D3D12_COMMAND_LIST_TYPE_COPY}};
    for (const auto& [name, type] : types) {
        ID3D12QueryHeap* heap = type == D3D12_COMMAND_LIST_TYPE_COPY ? e.copy_timestamps.p : e.timestamps.p;
        char line[200];
        if (!heap) {
            std::snprintf(line, sizeof line, "timestamps on %s: no heap\n", name);
            out += line;
            continue;
        }
        fill_readback(e, 0xcd);
        UINT64 frequency = 0;
        const bool ran = run(e, type, [&](ID3D12GraphicsCommandList* l) {
            l->EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            l->EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            l->ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, e.readback.p, 0);
        }, &frequency);
        const UINT64 t0 = read_u64(e, 0), t1 = read_u64(e, 8);
        const bool plausible = ran && t0 && t0 != 0xcdcdcdcdcdcdcdcdull && t1 >= t0 && frequency &&
                               double(t1 - t0) / double(frequency) < 1.0;
        std::snprintf(line, sizeof line, "timestamps on %s: %s\n", name, plausible ? "plausible" : "wrong");
        out += line;
    }
    return out;
}

std::string all_cases(ID3D12Device* device) {
    Env e{device};
    if (!setup(e)) return "setup failed\n";
    return heap_cases(device) + list_cases(e) + timestamp_runs(e);
}

std::vector<std::string> lines(const std::string& s) {
    std::vector<std::string> out;
    size_t at = 0;
    while (at < s.size()) {
        const size_t nl = s.find('\n', at);
        out.push_back(s.substr(at, nl - at));
        at = nl + 1;
    }
    return out;
}
} // namespace

int main() {
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;
    if (device_status != vktest::DeviceStatus::ready) return 1;
    std::string mine;
    unsigned mismatched = 0;
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
        mine = all_cases(ours.p);
        {  // A timestamp heap of the wrong kind for the list: the host refuses it at Close (E_INVALIDARG); the layer
           // checks heap types at execution, so it must refuse the list there.
            Env e{ours.p};
            if (setup(e)) {
                const std::pair<const char*, std::pair<D3D12_COMMAND_LIST_TYPE, ID3D12QueryHeap*>> m[] = {
                    {"copy list, TIMESTAMP heap", {D3D12_COMMAND_LIST_TYPE_COPY, e.timestamps.p}},
                    {"direct list, COPY heap", {D3D12_COMMAND_LIST_TYPE_DIRECT, e.copy_timestamps.p}},
                    {"compute list, COPY heap", {D3D12_COMMAND_LIST_TYPE_COMPUTE, e.copy_timestamps.p}}};
                for (const auto& [name, tc] : m) {
                    int result = -1;
                    auto* heap = tc.second;
                    const bool ran = run(e, tc.first, [&](ID3D12GraphicsCommandList* l) {
                        l->EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                        l->ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, e.readback.p, 0);
                    }, nullptr, &result);
                    if (result != int(leonida::ReplayResult::invalid_query)) {
                        ++mismatched;
                        std::printf("[ours] %s: executed (ran %d, result %d), host refuses it\n", name, ran, result);
                    }
                }
            }
        }
    }
    Ref<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return 1;
    unsigned differ = 0;
    for (const bool warp : {false, true}) {
        if (hostwine::host_is_wine()) break;
        Ref<IDXGIAdapter> adapter;
        Ref<ID3D12Device> host;
        std::string adapter_name;
        if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
        if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
        const auto a = lines(mine), b = lines(all_cases(host.p));
        for (size_t i = 0; i < a.size() || i < b.size(); ++i) {
            const std::string x = i < a.size() ? a[i] : "", y = i < b.size() ? b[i] : "";
            if (x != y) {
                ++differ;
                std::printf("[%s] ours: %s | host: %s\n", adapter_name.c_str(), x.c_str(), y.c_str());
            } else if (!warp) {
                std::printf("same: %s\n", x.c_str());
            }
        }
    }
    if (vk.validation_errors()) {
        std::printf("validation errors\n");
        return 1;
    }
    return differ || mismatched ? 1 : 0;
}
