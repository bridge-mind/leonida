// ExecuteIndirect with MaxCommandCount 0 on the Vulkan-backed device against Windows D3D12 (the layer's own adapter
// and WARP): the list's Close() result. A zero-command ExecuteIndirect is a no-op, not an error that fails the list.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

namespace {
template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

// Close() of a direct list holding ExecuteIndirect(signature of `type`, 0 commands), or the failing step's HRESULT.
HRESULT close_zero_indirect(ID3D12Device* device, D3D12_INDIRECT_ARGUMENT_TYPE type, UINT stride) {
    D3D12_INDIRECT_ARGUMENT_DESC arg{};
    arg.Type = type;
    D3D12_COMMAND_SIGNATURE_DESC sd{};
    sd.ByteStride = stride;
    sd.NumArgumentDescs = 1;
    sd.pArgumentDescs = &arg;
    Ref<ID3D12CommandSignature> sig;
    HRESULT hr = device->CreateCommandSignature(&sd, nullptr, IID_PPV_ARGS(&sig.p));
    if (FAILED(hr)) return hr;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = 256;
    rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> args;
    hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                         IID_PPV_ARGS(&args.p));
    if (FAILED(hr)) return hr;
    Ref<ID3D12CommandAllocator> alloc;
    hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p));
    if (FAILED(hr)) return hr;
    Ref<ID3D12GraphicsCommandList> list;
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p));
    if (FAILED(hr)) return hr;
    list->ExecuteIndirect(sig.p, 0, args.p, 0, nullptr, 0);
    return list->Close();
}

struct Case {
    const char* name;
    D3D12_INDIRECT_ARGUMENT_TYPE type;
    UINT stride;
};
}  // namespace

int main() {
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;
    if (device_status != vktest::DeviceStatus::ready) return 1;
    if (hostwine::host_is_wine()) return 77;
    const Case cases[] = {{"dispatch", D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, 12},
                          {"draw", D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, 16},
                          {"draw indexed", D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, 20}};
    unsigned mismatched = 0, references = 0;
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
        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return 1;
        for (const bool warp : {false, true}) {
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            std::string adapter_name;
            if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            ++references;
            for (const Case& c : cases) {
                const HRESULT a = close_zero_indirect(ours.p, c.type, c.stride);
                const HRESULT e = close_zero_indirect(host.p, c.type, c.stride);
                const bool same = SUCCEEDED(a) == SUCCEEDED(e);
                if (!same) ++mismatched;
                std::printf("%s %s, MaxCommandCount 0: ours Close 0x%08lx, %s 0x%08lx\n", same ? "[ok]  " : "[FAIL]",
                            c.name, static_cast<unsigned long>(a), adapter_name.c_str(),
                            static_cast<unsigned long>(e));
            }
        }
    }
    if (!references) return 1;
    if (vk.validation_errors()) return 1;
    return mismatched ? 1 : 0;
}
