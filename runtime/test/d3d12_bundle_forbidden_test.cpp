// Calls a bundle may not make: what Close of the bundle and of the parent that executes it return for each, on the
// layer and on Windows' own D3D12 (no debug layer; the default adapter and WARP). One call is recorded per bundle,
// with valid arguments, on a bundle of its own; the parent direct list executes it, is closed and submitted, and the
// device's removed reason is read after the fence. A host device is created per call, so a call that removes the
// device does not touch the next one. Microsoft documents RSSetViewports, RSSetScissorRects, OMSetRenderTargets,
// ClearState and the calls below as not allowed in a bundle; the table is what the PC runtime does without the debug
// layer (every one of them fails the bundle's Close with E_FAIL, on both adapters, and executing such a bundle
// crashes the PC runtime, so the parent is only closed for bundles that closed). The layer is held to the same
// verdict for every call where both host adapters agree; its code for a refused bundle is E_INVALIDARG.
// Both host adapters always run (LEONIDA_HOST_ADAPTER is not read), except under Wine, where the host is skipped.
// No game data.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>

#include <cstdio>
#include <functional>
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

constexpr HRESULT not_run = HRESULT(0x7fff0000);  // no parent Close: the bundle could not be executed

struct Env {
    ID3D12Device* device = nullptr;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence;
    Ref<ID3D12Resource> target, buffer;
    Ref<ID3D12DescriptorHeap> rtv_heap;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
    UINT64 fence_value = 0;
    Ref<ID3D12CommandAllocator> other_allocator;  // a closed bundle for ExecuteBundle inside a bundle
    Ref<ID3D12GraphicsCommandList> other_bundle;

    bool init(ID3D12Device* d) {
        device = d;
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC texture{};
        texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = texture.Height = 16;
        texture.DepthOrArraySize = texture.MipLevels = 1;
        texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture.SampleDesc.Count = 1;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture,
                                                        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                        IID_PPV_ARGS(&target.p))));
        const auto desc = buffer_desc(256);
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&buffer.p))));
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(target.p, nullptr, rtv);
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE,
                                                       IID_PPV_ARGS(&other_allocator.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, other_allocator.p, nullptr,
                                                  IID_PPV_ARGS(&other_bundle.p))));
        CHECK(SUCCEEDED(other_bundle->Close()));
        return true;
    }
};

using Recorder = std::function<void(Env&, ID3D12GraphicsCommandList*)>;

struct Call {
    const char* name;
    Recorder record;
};

struct Outcome {
    HRESULT bundle_close = E_FAIL, parent_close = not_run, removed = S_OK;
    HRESULT direct_close = E_FAIL;  // the same call, same arguments, on a direct list: the control
};

std::vector<Call> calls() {
    return {
        // Calls the layer did not refuse in a bundle before this test. Every call here is also valid on a direct
        // list, which run_call checks as a control (none is bundle-only or meaningless there).
        {"RSSetViewports", [](Env&, ID3D12GraphicsCommandList* b) {
             const D3D12_VIEWPORT viewport{0, 0, 16, 16, 0, 1};
             b->RSSetViewports(1, &viewport); }},
        {"RSSetScissorRects", [](Env&, ID3D12GraphicsCommandList* b) {
             const D3D12_RECT scissor{0, 0, 16, 16};
             b->RSSetScissorRects(1, &scissor); }},
        {"OMSetRenderTargets", [](Env& e, ID3D12GraphicsCommandList* b) {
             b->OMSetRenderTargets(1, &e.rtv, FALSE, nullptr); }},
        {"ClearState", [](Env&, ID3D12GraphicsCommandList* b) { b->ClearState(nullptr); }},
        // Controls: calls the layer already refuses in a bundle.
        {"ResourceBarrier", [](Env& e, ID3D12GraphicsCommandList* b) {
             const auto barrier = transition(e.target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                             D3D12_RESOURCE_STATE_COPY_SOURCE);
             b->ResourceBarrier(1, &barrier); }},
        {"ClearRenderTargetView", [](Env& e, ID3D12GraphicsCommandList* b) {
             const FLOAT color[4] = {0, 1, 0, 1};
             b->ClearRenderTargetView(e.rtv, color, 0, nullptr); }},
        {"SetPredication(null)", [](Env&, ID3D12GraphicsCommandList* b) {
             b->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO); }},
        {"ExecuteBundle", [](Env& e, ID3D12GraphicsCommandList* b) { b->ExecuteBundle(e.other_bundle.p); }},
        // Controls: calls a bundle may make.
        {"OMSetStencilRef", [](Env&, ID3D12GraphicsCommandList* b) { b->OMSetStencilRef(1); }},
        {"OMSetBlendFactor", [](Env&, ID3D12GraphicsCommandList* b) {
             const FLOAT factor[4] = {1, 1, 1, 1};
             b->OMSetBlendFactor(factor); }},
    };
}

// One call on one bundle, executed by a parent list, on `device` (its own for a host).
bool run_call(Env& env, const Call& call, Outcome& out) {
    ID3D12Device* device = env.device;
    Ref<ID3D12CommandAllocator> bundle_allocator, allocator;
    Ref<ID3D12GraphicsCommandList> bundle, list;
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(&bundle_allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, bundle_allocator.p, nullptr,
                                              IID_PPV_ARGS(&bundle.p))));
    // The control: the call is valid, so its refusal in a bundle comes from the bundle rule and not its arguments.
    {
        Ref<ID3D12CommandAllocator> direct_allocator;
        Ref<ID3D12GraphicsCommandList> direct;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                       IID_PPV_ARGS(&direct_allocator.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, direct_allocator.p, nullptr,
                                                  IID_PPV_ARGS(&direct.p))));
        call.record(env, direct.p);
        out.direct_close = direct->Close();
    }
    call.record(env, bundle.p);
    out.bundle_close = bundle->Close();
    // Executing a bundle whose Close failed crashes the PC runtime (measured), so the parent is left out then.
    if (FAILED(out.bundle_close)) return true;
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                              IID_PPV_ARGS(&list.p))));
    list->ExecuteBundle(bundle.p);
    out.parent_close = list->Close();
    if (SUCCEEDED(out.parent_close)) {
        ID3D12CommandList* lists[] = {list.p};
        env.queue->ExecuteCommandLists(1, lists);
        // A device that is removed here makes the wait fail; GetDeviceRemovedReason below reports it.
        if (SUCCEEDED(env.queue->Signal(env.fence.p, ++env.fence_value)))
            env.fence->SetEventOnCompletion(env.fence_value, nullptr);
    }
    out.removed = device->GetDeviceRemovedReason();
    return true;
}

struct Run {
    const char* name = "";
    bool ran = false;
    std::vector<Outcome> outcomes;
};

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
    const auto list = calls();
    const unsigned errors_before = vk.validation_errors();

    Run layer;
    layer.name = "layer";
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
        for (const Call& call : list) {
            layer.outcomes.emplace_back();
            REQUIRE(run_call(env, call, layer.outcomes.back()));
        }
        layer.ran = true;
    }

    Run hosts[2];
    hosts[0].name = "default adapter";
    hosts[1].name = "WARP";
    if (hostwine::host_is_wine()) {
        // Wine's vkd3d answers differently; the host is not compared.
    } else {
        Ref<IDXGIFactory4> factory;
        REQUIRE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))));
        for (int i = 0; i < 2; ++i) {
            bool usable = true;
            for (const Call& call : list) {
                Ref<IDXGIAdapter> adapter;
                Ref<ID3D12Device> host;
                if (i == 1 && FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) usable = false;
                if (!usable || FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
                    usable = false;
                    break;
                }
                Env env;
                REQUIRE(env.init(host.p));
                hosts[i].outcomes.emplace_back();
                REQUIRE(run_call(env, call, hosts[i].outcomes.back()));
            }
            hosts[i].ran = usable;
        }
    }

    // The layer's bundle Close must match the host's wherever the adapters that ran agree; where the two differ the
    // case is reported and not compared. The direct-list control must succeed everywhere.
    const auto text = [](HRESULT hr) {
        char out[16];
        if (hr == not_run) return std::string("n/a");
        std::snprintf(out, sizeof(out), "0x%08lx", static_cast<unsigned long>(hr));
        return std::string(out);
    };
    unsigned failures = 0, compared = 0, disagreed = 0;
    std::printf("%-24s | %-26s | %-26s | %-26s\n", "call (bundle/parent/removed)", "layer", hosts[0].name, hosts[1].name);
    for (std::size_t i = 0; i < list.size(); ++i) {
        std::printf("%-24s", list[i].name);
        const auto cell = [&](const Run& run) {
            if (!run.ran) return std::printf(" | %-26s", "(not run)");
            const Outcome& o = run.outcomes[i];
            return std::printf(" | %s %s %s", text(o.bundle_close).c_str(), text(o.parent_close).c_str(),
                               text(o.removed).c_str());
        };
        cell(layer);
        cell(hosts[0]);
        cell(hosts[1]);
        std::printf("\n");
        for (const Run* run : {&layer, &hosts[0], &hosts[1]}) {
            if (run->ran && FAILED(run->outcomes[i].direct_close)) {
                std::fprintf(stderr, "FAIL %s on %s: Close of the direct-list control is %s\n", list[i].name,
                             run->name, text(run->outcomes[i].direct_close).c_str());
                ++failures;
            }
        }
        if (!hosts[0].ran && !hosts[1].ran) continue;
        const Run& first = hosts[0].ran ? hosts[0] : hosts[1];
        const HRESULT a = first.outcomes[i].bundle_close;
        if (hosts[0].ran && hosts[1].ran && a != hosts[1].outcomes[i].bundle_close) {
            std::printf("  the default adapter and WARP disagree on %s: not compared\n", list[i].name);
            ++disagreed;
            continue;
        }
        ++compared;
        // The same verdict, not the same code: a refused bundle's Close is E_FAIL on the PC runtime and E_INVALIDARG here.
        if (SUCCEEDED(layer.outcomes[i].bundle_close) != SUCCEEDED(a)) {
            std::fprintf(stderr, "FAIL %s: layer bundle Close %s, host %s\n", list[i].name,
                         text(layer.outcomes[i].bundle_close).c_str(), text(a).c_str());
            ++failures;
        }
    }
    const unsigned validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", validation_errors);
        ++failures;
    }
    if (!hosts[0].ran && !hosts[1].ran && !hostwine::host_is_wine())
        std::fprintf(stderr, "no D3D12 reference device (default adapter or WARP)\n");
    std::printf("bundle forbidden calls: %zu on the layer, %u compared with the host, %u hosts disagree, %u failures\n",
                list.size(), compared, disagreed, failures);
    return failures ? 1 : 0;
}
