// Exercises the Xbox-ABI device proxy against a real WARP PC device, with no game executable.
//
// It creates the proxy through the registered d3d12_xs!D3D12XboxCreateDevice shim, then calls the measured Xbox
// slots exactly as the guest would (raw function pointers, not the C++ interface):
//   slot 14  CreateDescriptorHeap (returns a heap proxy)
//   heap 9/10 GetCPU/GetGPUDescriptorHandleForHeapStart (Xbox: 8-byte handle in RAX; the PC heap uses the
//             hidden return pointer in RDX, so slot 14 wraps it)
//   slot 62  CreateComputePipelineState
//   slot 76  resource creation with an Xbox layout word (0x101)
//   slot 8/36 CreateCommandQueue/CreateFence (return queue and fence proxies that run GPU work)
// Slot 9 on the device is CreateCommandAllocator on the PC vtable and must still forward, which proves the handle
// getters were not wedged into the device slots. Console output also shows the proxy's first-call slot log.
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <wrl/client.h>

#include "d3d12_trace.h"
#include "d3d12_xs_device.h"
#include "shims.h"

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

using Microsoft::WRL::ComPtr;

using XboxCreateDeviceFn = HRESULT(WINAPI*)(IUnknown*, const void*, const IID*, void**);
using XboxCreateResourceFn = HRESULT(WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
                                              const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
                                              const D3D12_CLEAR_VALUE*, REFIID, void**);
// Xbox slot 62: (desc, two words, iid, out), as XboxBvhBuilder.dll calls it (zeros for the two words).
using XboxCreateComputePsoFn = HRESULT(WINAPI*)(void*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, std::uint64_t,
                                                std::uint64_t, REFIID, void**);
using GetCpuHandleFn = D3D12_CPU_DESCRIPTOR_HANDLE(WINAPI*)(ID3D12DescriptorHeap*);
using GetGpuHandleFn = D3D12_GPU_DESCRIPTOR_HANDLE(WINAPI*)(ID3D12DescriptorHeap*);

D3D12_RESOURCE_DESC buffer_desc() {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 4096;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);  // measured Xbox tiled/buffer word
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    return desc;
}

D3D12_RESOURCE_DESC texture_desc() {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 4;
    desc.Height = 4;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    return desc;
}

D3D12_HEAP_PROPERTIES default_heap() {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    return heap;
}

bool load_compute_shader(ComPtr<ID3DBlob>& bytecode) {
    HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    check(compiler != nullptr, "d3dcompiler_47.dll loads");
    auto compile = compiler ? reinterpret_cast<pD3DCompile>(GetProcAddress(compiler, "D3DCompile")) : nullptr;
    check(compile != nullptr, "d3dcompiler_47.dll exports D3DCompile");
    if (!compile) return false;

    static const char kSource[] =
        "[numthreads(1, 1, 1)]\n"
        "void main(uint3 id : SV_DispatchThreadID) { }\n";
    ComPtr<ID3DBlob> errors;
    HRESULT hr = compile(kSource, sizeof(kSource) - 1, "device_proxy_test.hlsl", nullptr, nullptr, "main",
                         "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, bytecode.GetAddressOf(),
                         errors.GetAddressOf());
    if (FAILED(hr) && errors) {
        std::printf("shader compile error: %s\n", static_cast<const char*>(errors->GetBufferPointer()));
    }
    check(SUCCEEDED(hr) && bytecode, "trivial compute shader compiles (cs_5_0)");
    return SUCCEEDED(hr) && bytecode;
}

}  // namespace

int main(int argc, char** argv) {
    // --trace-d3d12, as gta6rt takes it: device_proxy_trace checks that the parameter words still print then.
    if (argc == 2 && std::strcmp(argv[1], "--trace-d3d12") == 0) rt::d3d12_trace::set_enabled(true);
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("device_proxy_test: Xbox-ABI device proxy over WARP\n");

    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf()));
    check(SUCCEEDED(hr) && factory, "DXGI factory created");
    ComPtr<IDXGIAdapter> warp;
    if (factory) hr = factory->EnumWarpAdapter(IID_PPV_ARGS(warp.GetAddressOf()));
    if (!factory) {
        std::printf("device_proxy_test: FAILED (no DXGI factory)\n");
        return 1;
    }
    if (!warp) {
        // Windows always has WARP; Wine doesn't. CTest reads exit code 77 as a skip (SKIP_RETURN_CODE).
        std::printf("SKIP device_proxy_test: no WARP adapter (0x%08lX)\n", static_cast<unsigned long>(hr));
        return 77;
    }
    check(true, "WARP adapter enumerated");

    auto create_device = reinterpret_cast<XboxCreateDeviceFn>(
        rt::shims::find("d3d12_xs.dll", "D3D12XboxCreateDevice"));
    check(create_device != nullptr, "shim registry exposes d3d12_xs!D3D12XboxCreateDevice");
    if (!create_device) return 1;

    std::uint32_t params[24] = {0x1234abcd};  // logged only, not decoded; word 0 is device_proxy_trace's marker
    const IID iid_device = __uuidof(ID3D12Device);
    void* created = nullptr;
    hr = create_device(warp.Get(), params, &iid_device, &created);
    check(SUCCEEDED(hr) && created != nullptr, "D3D12XboxCreateDevice returns a device");
    if (!created) {
        std::printf("device_proxy_test: FAILED (no device)\n");
        return 1;
    }

    ComPtr<ID3D12Device> device;
    device.Attach(static_cast<ID3D12Device*>(created));
    check(rt::d3d12_xs::is_device_proxy(device.Get()), "returned pointer is an Xbox-ABI proxy");

    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device.Get());
    check(real != nullptr, "proxy exposes its PC device");
    unsigned slots = rt::d3d12_xs::device_proxy_slot_count(device.Get());
    std::printf("proxy forwards %u PC vtable slots (ID3D12Device..Device14 max is 83)\n", slots);
    check(slots >= 44, "proxy forwards at least the base ID3D12Device vtable");
    check(rt::d3d12_xs::device_slot_overridden(9), "device slot 9 is CreateCommandAllocator (allocator proxies)");
    check(!rt::d3d12_xs::device_slot_overridden(10), "device slot 10 is forwarded (PC CreateGraphicsPipelineState)");
    check(rt::d3d12_xs::device_slot_overridden(14), "device slot 14 is the descriptor-heap override");
    check(rt::d3d12_xs::device_slot_overridden(62), "device slot 62 is the compute-PSO override");
    check(rt::d3d12_xs::device_slot_overridden(76), "device slot 76 is the resource override");
    check(rt::d3d12_xs::device_slot_overridden(67),
          "device slot 67 is the DeserializeComputePipelineStateX override (console native-code compute pipeline, not PC CreateProtectedResourceSession1)");

    void** proxy_vtbl = *reinterpret_cast<void***>(device.Get());
    void** real_vtbl = real ? *reinterpret_cast<void***>(real) : nullptr;
    check(proxy_vtbl != nullptr && proxy_vtbl[14] != real_vtbl[14], "slot 14 thunk differs from the PC entry");
    if (slots > 62) check(proxy_vtbl[62] != real_vtbl[62], "slot 62 thunk differs from PC CreateStateObject");
    if (slots > 67) check(proxy_vtbl[67] != real_vtbl[67], "slot 67 thunk differs from PC CreateProtectedResourceSession1");
    if (slots > 76) check(proxy_vtbl[76] != real_vtbl[76], "slot 76 thunk differs from PC CreateCommittedResource3");

    // The hit counters are for tests and count from the first read or reset, so that a title's calls write no
    // shared counter. The QueryInterface of the creation above was before that: its flag is set, its hit is not.
    check(rt::d3d12_xs::device_slot_first_call_seen(0), "slot 0 first-call flag is set by the creation");
    check(rt::d3d12_xs::device_slot_hits(0) == 0, "slot 0 has no hit before the first read of a counter");
    device->AddRef();
    device->Release();
    check(rt::d3d12_xs::device_slot_hits(1) == 1 && rt::d3d12_xs::device_slot_hits(2) == 1,
          "slots 1 and 2 count the calls after the first read");

    rt::d3d12_xs::reset_device_proxy_stats();

    // QI/refcount behaviour: device interfaces resolve to the same proxy, non-device IIDs do not.
    ID3D12Device* same = nullptr;
    hr = device->QueryInterface(__uuidof(ID3D12Device), reinterpret_cast<void**>(&same));
    check(SUCCEEDED(hr) && same == device.Get(), "QueryInterface(IID_ID3D12Device) returns the same proxy");
    if (same) same->Release();
    ComPtr<ID3D12Device1> device1;
    hr = device->QueryInterface(IID_PPV_ARGS(device1.GetAddressOf()));
    check(SUCCEEDED(hr) && device1 != nullptr, "QueryInterface(IID_ID3D12Device1) succeeds");
    ComPtr<ID3D12CommandQueue> not_a_device;
    hr = device->QueryInterface(IID_PPV_ARGS(not_a_device.GetAddressOf()));
    check(FAILED(hr) && !not_a_device, "QueryInterface rejects a non-device IID");
    check(rt::d3d12_xs::device_slot_hits(0) >= 3, "slot 0 hit counter moves");

    // Generic-forwarder coverage: slot 43 returns an 8-byte struct through the hidden pointer, and slot 30 has
    // two stack arguments. Both must reach the PC device unchanged (they are not measured overrides).
    const LUID proxy_luid = device->GetAdapterLuid();
    const LUID real_luid = real ? real->GetAdapterLuid() : LUID{};
    check(proxy_luid.LowPart == real_luid.LowPart && proxy_luid.HighPart == real_luid.HighPart,
          "slot 43 forwards GetAdapterLuid (struct return)");
    D3D12_RESOURCE_DESC reserved_desc = buffer_desc();
    reserved_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> reserved;
    hr = device->CreateReservedResource(&reserved_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                        IID_PPV_ARGS(reserved.GetAddressOf()));
    check(SUCCEEDED(hr) && reserved != nullptr, "slot 30 forwards CreateReservedResource (stack arguments)");

    // Slot 14: descriptor heap through the proxy. The proxy wraps the real PC heap for the handle ABI.
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 8;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    const unsigned long refs_before_heap = rt::d3d12_xs::device_proxy_ref_count(device.Get());
    hr = device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(heap.GetAddressOf()));
    check(SUCCEEDED(hr) && heap != nullptr, "slot 14 creates a descriptor heap");
    check(rt::d3d12_xs::device_slot_hits(14) == 1, "slot 14 hit counter is 1");
    check(heap && rt::d3d12_xs::device_proxy_ref_count(device.Get()) == refs_before_heap + 1,
          "heap holds one device proxy reference");

    if (heap) {
        check(rt::d3d12_xs::is_heap_proxy(heap.Get()), "slot 14 returns a heap proxy");
        ID3D12DescriptorHeap* real_heap = rt::d3d12_xs::heap_proxy_real_heap(heap.Get());
        check(real_heap != nullptr, "heap proxy exposes the PC heap");
        check(rt::d3d12_xs::heap_slot_overridden(9), "heap slot 9 is the CPU-handle override");
        check(rt::d3d12_xs::heap_slot_overridden(10), "heap slot 10 is the GPU-handle override");
        ID3D12DescriptorHeap* same_heap = nullptr;
        hr = heap->QueryInterface(__uuidof(ID3D12DescriptorHeap), reinterpret_cast<void**>(&same_heap));
        check(SUCCEEDED(hr) && same_heap == heap.Get(), "heap QueryInterface returns the same heap proxy");
        if (same_heap) same_heap->Release();

        // ID3D12DeviceChild::GetDevice must return the Xbox-ABI device, not the raw PC device, and AddRef it.
        check(rt::d3d12_xs::heap_slot_overridden(7), "heap slot 7 is the GetDevice override");
        ID3D12Device* heap_device = nullptr;
        const unsigned long refs_before_get = rt::d3d12_xs::device_proxy_ref_count(device.Get());
        hr = heap->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(&heap_device));
        check(SUCCEEDED(hr) && heap_device == device.Get() && rt::d3d12_xs::is_device_proxy(heap_device),
              "heap GetDevice returns the device proxy");
        check(rt::d3d12_xs::device_proxy_ref_count(device.Get()) == refs_before_get + 1,
              "GetDevice AddRefs the device proxy");
        check(rt::d3d12_xs::heap_slot_hits(7) == 1, "heap slot 7 hit counter is 1");
        if (heap_device) heap_device->Release();
        check(rt::d3d12_xs::device_proxy_ref_count(device.Get()) == refs_before_get,
              "GetDevice release balances");

        // The Xbox call shape: `this` only, 8-byte handle in RAX. The proxy converts from the PC heap's
        // hidden return pointer (RDX). A PC C++ call through the proxy uses the PC convention, so it is not
        // used here; the real heap below shows that convention and the conversion is compared against it.
        void** proxy_heap_vtbl = *reinterpret_cast<void***>(heap.Get());
        auto get_cpu = reinterpret_cast<GetCpuHandleFn>(proxy_heap_vtbl[9]);
        auto get_gpu = reinterpret_cast<GetGpuHandleFn>(proxy_heap_vtbl[10]);
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = get_cpu(heap.Get());
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = get_gpu(heap.Get());
        check(cpu.ptr != 0, "heap proxy slot 9 returns a CPU handle in RAX");
        check(gpu.ptr != 0, "heap proxy slot 10 returns a GPU handle in RAX");
        if (real_heap) {
            void** real_heap_vtbl = *reinterpret_cast<void***>(real_heap);
            check(proxy_heap_vtbl[9] != real_heap_vtbl[9], "heap slot 9 thunk differs from the PC heap entry");
            check(proxy_heap_vtbl[10] != real_heap_vtbl[10], "heap slot 10 thunk differs from the PC heap entry");
        }

        // PC reality check: the real heap takes a hidden return pointer in RDX. This is the deviation that
        // makes the wrapper necessary; a one-argument call on the real heap is invalid and is not made.
        if (real_heap) {
            void** real_heap_vtbl = *reinterpret_cast<void***>(real_heap);
            using GetCpuOutFn = void(WINAPI*)(ID3D12DescriptorHeap*, D3D12_CPU_DESCRIPTOR_HANDLE*);
            using GetGpuOutFn = void(WINAPI*)(ID3D12DescriptorHeap*, D3D12_GPU_DESCRIPTOR_HANDLE*);
            constexpr SIZE_T kSentinel = 0x1111111111111111ull;
            D3D12_CPU_DESCRIPTOR_HANDLE cpu_pc{};
            D3D12_GPU_DESCRIPTOR_HANDLE gpu_pc{};
            cpu_pc.ptr = kSentinel;
            gpu_pc.ptr = kSentinel;
            reinterpret_cast<GetCpuOutFn>(real_heap_vtbl[9])(real_heap, &cpu_pc);
            reinterpret_cast<GetGpuOutFn>(real_heap_vtbl[10])(real_heap, &gpu_pc);
            check(cpu_pc.ptr != kSentinel, "PC heap slot 9 writes through the RDX return pointer");
            check(gpu_pc.ptr != kSentinel, "PC heap slot 10 writes through the RDX return pointer");
            check(cpu_pc.ptr == cpu.ptr, "proxy CPU handle equals the PC heap's hidden-pointer result");
            check(gpu_pc.ptr == gpu.ptr, "proxy GPU handle equals the PC heap's hidden-pointer result");
        }

        const D3D12_DESCRIPTOR_HEAP_DESC got_desc = heap->GetDesc();
        check(got_desc.NumDescriptors == heap_desc.NumDescriptors, "heap slot 8 forwards GetDesc");
        check(rt::d3d12_xs::heap_slot_hits(9) >= 1, "heap slot 9 hit counter moves");
        check(rt::d3d12_xs::heap_slot_hits(10) >= 1, "heap slot 10 hit counter moves");
        check(rt::d3d12_xs::heap_slot_first_call_seen(9), "heap slot 9 first-call flag is set");
    }

    // Slot 9 on the device must remain CreateCommandAllocator, not a handle getter.
    ComPtr<ID3D12CommandAllocator> allocator;
    hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.GetAddressOf()));
    check(SUCCEEDED(hr) && allocator != nullptr, "device slot 9 forwards CreateCommandAllocator");
    // A PC runtime allocator is not wrapped: the PC runtime's lists could not resolve an allocator proxy.
    check(!rt::d3d12_xs::is_allocator_proxy(allocator.Get()), "slot 9 hands out the PC allocator unwrapped");
    check(rt::d3d12_xs::device_slot_hits(9) == 1, "slot 9 hit counter is 1");

    // Slot 76: buffer resource with the measured Xbox layout word 0x101; PC committed resources take ROW_MAJOR.
    const IID iid_resource = __uuidof(ID3D12Resource);
    auto create_resource = reinterpret_cast<XboxCreateResourceFn>(proxy_vtbl[76]);
    const D3D12_HEAP_PROPERTIES heap_properties = default_heap();
    D3D12_RESOURCE_DESC xbox_buffer = buffer_desc();
    ComPtr<ID3D12Resource> buffer;
    hr = create_resource(device.Get(), &heap_properties, D3D12_HEAP_FLAG_NONE, &xbox_buffer,
                         D3D12_RESOURCE_STATE_COMMON, nullptr, iid_resource,
                         reinterpret_cast<void**>(buffer.GetAddressOf()));
    check(SUCCEEDED(hr) && buffer != nullptr, "slot 76 creates a committed buffer");
    if (buffer) {
        const D3D12_RESOURCE_DESC got = buffer->GetDesc();
        check(got.Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR, "slot 76 maps 0x101 to ROW_MAJOR for a buffer");
        check(got.Width == 4096, "slot 76 preserves the resource width");
    }

    // The same Xbox word on a texture maps to UNKNOWN, which is what PC committed textures require.
    D3D12_RESOURCE_DESC xbox_texture = texture_desc();
    ComPtr<ID3D12Resource> texture;
    hr = create_resource(device.Get(), &heap_properties, D3D12_HEAP_FLAG_NONE, &xbox_texture,
                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr, iid_resource,
                         reinterpret_cast<void**>(texture.GetAddressOf()));
    check(SUCCEEDED(hr) && texture != nullptr, "slot 76 creates a committed texture");
    if (texture) {
        check(texture->GetDesc().Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN,
              "slot 76 maps 0x101 to UNKNOWN for a texture");
    }
    check(rt::d3d12_xs::device_slot_hits(76) == 2, "slot 76 hit counter is 2");
    check(rt::d3d12_xs::device_slot_first_call_seen(76), "slot 76 first-call flag is set");

    // Xbox device slots from the public HLE. On the Vulkan device the PC entries at 78/79 are
    // CreateReservedResource2/CreateSampler2 and 80 is past the 80-slot table, so success here can only come from
    // the routes: frame events return S_OK and a fresh token, CreateCommandQueueX returns a queue proxy and
    // CreateCommandListX (desc, allocator, initialState, riid, out; no node mask) reaches PC CreateCommandList.
    {
        using SetFrameIntervalFn = HRESULT(WINAPI*)(void*, void*, UINT, UINT, UINT);
        using ScheduleFrameEventFn = HRESULT(WINAPI*)(void*, UINT, UINT, void*, UINT);
        using WaitFrameEventFn = HRESULT(WINAPI*)(void*, UINT, UINT, void*, UINT, UINT64*);
        using CreateQueueXFn = HRESULT(WINAPI*)(void*, const D3D12_COMMAND_QUEUE_DESC*, const IID*, void**);
        using CreateListXFn = HRESULT(WINAPI*)(void*, const void*, ID3D12CommandAllocator*, ID3D12PipelineState*,
                                               const IID*, void**);

        hr = reinterpret_cast<SetFrameIntervalFn>(proxy_vtbl[78])(device.Get(), nullptr, 16667, 1, 0);
        check(hr == S_OK, "slot 78 SetFrameIntervalX returns S_OK");
        hr = reinterpret_cast<ScheduleFrameEventFn>(proxy_vtbl[79])(device.Get(), 0, 0, nullptr, 0);
        check(hr == S_OK, "slot 79 ScheduleFrameEventX returns S_OK");

        std::uint64_t token = 0, token2 = 0;
        hr = reinterpret_cast<WaitFrameEventFn>(proxy_vtbl[80])(device.Get(), 0, INFINITE, nullptr, 0, &token);
        check(hr == S_OK && token != 0, "slot 80 WaitFrameEventX returns a non-zero token");
        hr = reinterpret_cast<WaitFrameEventFn>(proxy_vtbl[80])(device.Get(), 0, 0, nullptr, 0, &token2);
        check(hr == S_OK && token2 != 0 && token2 != token, "slot 80 returns a fresh token on the next wait");
        hr = reinterpret_cast<WaitFrameEventFn>(proxy_vtbl[80])(device.Get(), 0, 0, nullptr, 0, nullptr);
        check(hr == E_POINTER, "slot 80 refuses a null token with E_POINTER");
        check(rt::d3d12_xs::device_slot_hits(78) == 1 && rt::d3d12_xs::device_slot_hits(79) == 1 &&
                  rt::d3d12_xs::device_slot_hits(80) == 3,
              "slots 78/79/80 hit counters are 1/1/3");

        D3D12_COMMAND_QUEUE_DESC xbox_queue{};
        xbox_queue.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        const IID iid_queue = __uuidof(ID3D12CommandQueue);
        ComPtr<ID3D12CommandQueue> xbox_queue_proxy;
        hr = reinterpret_cast<CreateQueueXFn>(proxy_vtbl[48])(device.Get(), &xbox_queue, &iid_queue,
                                                               reinterpret_cast<void**>(xbox_queue_proxy.GetAddressOf()));
        check(hr == S_OK && rt::d3d12_xs::is_queue_proxy(xbox_queue_proxy.Get()),
              "slot 48 CreateCommandQueueX returns a queue proxy");
        check(rt::d3d12_xs::device_slot_hits(48) == 1, "slot 48 hit counter is 1");

        ComPtr<ID3D12CommandAllocator> xbox_allocator;
        hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            IID_PPV_ARGS(xbox_allocator.GetAddressOf()));
        struct XboxCommandListDesc {
            UINT Type;
            UINT NodeMask;
        } list_desc{D3D12_COMMAND_LIST_TYPE_DIRECT, 7};  // the node mask is ignored by the console call
        const IID iid_list = __uuidof(ID3D12GraphicsCommandList);
        ComPtr<ID3D12GraphicsCommandList> xbox_list;
        hr = SUCCEEDED(hr) ? reinterpret_cast<CreateListXFn>(proxy_vtbl[73])(
                                 device.Get(), &list_desc, xbox_allocator.Get(), nullptr, &iid_list,
                                 reinterpret_cast<void**>(xbox_list.GetAddressOf()))
                           : hr;
        check(hr == S_OK && xbox_list != nullptr, "slot 73 CreateCommandListX returns a command list");
        check(rt::d3d12_xs::device_slot_hits(73) == 1, "slot 73 hit counter is 1");
    }

    // Slot 62: an empty root signature and a trivial compute shader through the Xbox signature. The PC entry at
    // slot 62 is CreateStateObject, so success here proves the override reached CreateComputePipelineState.
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ComPtr<ID3DBlob> root_blob;
    ComPtr<ID3DBlob> root_error;
    hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, root_blob.GetAddressOf(),
                                     root_error.GetAddressOf());
    check(SUCCEEDED(hr) && root_blob != nullptr, "empty root signature serializes");
    ComPtr<ID3D12RootSignature> root_signature;
    if (root_blob) {
        hr = device->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(),
                                         IID_PPV_ARGS(root_signature.GetAddressOf()));
    }
    check(SUCCEEDED(hr) && root_signature != nullptr, "device slot 16 forwards CreateRootSignature");

    ComPtr<ID3DBlob> bytecode;
    if (load_compute_shader(bytecode) && root_signature) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root_signature.Get();
        pso_desc.CS.pShaderBytecode = bytecode->GetBufferPointer();
        pso_desc.CS.BytecodeLength = bytecode->GetBufferSize();
        pso_desc.NodeMask = 0;
        pso_desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

        auto create_pso = reinterpret_cast<XboxCreateComputePsoFn>(proxy_vtbl[62]);
        const IID iid_pso = __uuidof(ID3D12PipelineState);
        ComPtr<ID3D12PipelineState> pso;
        hr = create_pso(device.Get(), &pso_desc, 0, 0, iid_pso, reinterpret_cast<void**>(pso.GetAddressOf()));
        check(SUCCEEDED(hr) && pso != nullptr, "slot 62 creates a compute PSO via CreateComputePipelineState");
        check(rt::d3d12_xs::device_slot_hits(62) == 1, "slot 62 hit counter is 1");
        check(rt::d3d12_xs::device_slot_first_call_seen(62), "slot 62 first-call flag is set");

        // Slot 67: Xbox native compute pipeline creation. Maps the console descriptor (+0x00 root signature,
        // +0x28 kind 6 code blob, +0x30 code size) to D3D12_COMPUTE_PIPELINE_STATE_DESC and calls CreateComputePipelineState.
        struct XboxNativeComputeDesc {
            ID3D12RootSignature* root_signature;
            const void* kind0;
            std::uint64_t kind0_size;
            const void* kind1;
            std::uint64_t kind1_size;
            const void* kind6;
            std::uint64_t kind6_size;
            const void* kind7;
            std::uint64_t kind7_size;
        } xbox_desc{};
        xbox_desc.root_signature = root_signature.Get();
        xbox_desc.kind6 = bytecode->GetBufferPointer();
        xbox_desc.kind6_size = bytecode->GetBufferSize();

        using XboxCreateNativeComputePsoFn = HRESULT(WINAPI*)(void*, const void*, const IID*, void**);
        auto create_native_pso = reinterpret_cast<XboxCreateNativeComputePsoFn>(proxy_vtbl[67]);
        ComPtr<ID3D12PipelineState> pso67;
        hr = create_native_pso(device.Get(), &xbox_desc, &iid_pso, reinterpret_cast<void**>(pso67.GetAddressOf()));
        check(SUCCEEDED(hr) && pso67 != nullptr, "slot 67 creates a compute PSO via CreateComputePipelineState");
        check(rt::d3d12_xs::device_slot_hits(67) == 1, "slot 67 hit counter is 1");
        check(rt::d3d12_xs::device_slot_first_call_seen(67), "slot 67 first-call flag is set");
    }

    // _COM_Outptr_ (d3d12.h): a failed create leaves NULL in the output. WARP accepts invalid heap descriptions,
    // so the PC-failure case for slot 14 is in d3d12_vulkan_device_test; these are the proxy's own early returns.
    {
        void* proxy_out = reinterpret_cast<void*>(0x1111);
        using Slot14Fn = HRESULT(WINAPI*)(void*, const D3D12_DESCRIPTOR_HEAP_DESC*, const IID*, void**);
        hr = reinterpret_cast<Slot14Fn>(proxy_vtbl[14])(device.Get(), &heap_desc, nullptr, &proxy_out);
        check(FAILED(hr) && proxy_out == nullptr, "slot 14 nulls its output for a null IID");
        using Slot62Fn = HRESULT(WINAPI*)(void*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, std::uint64_t,
                                          std::uint64_t, const IID*, void**);
        proxy_out = reinterpret_cast<void*>(0x1111);
        D3D12_COMPUTE_PIPELINE_STATE_DESC empty_pso{};
        hr = reinterpret_cast<Slot62Fn>(proxy_vtbl[62])(device.Get(), &empty_pso, 0, 0, nullptr, &proxy_out);
        check(FAILED(hr) && proxy_out == nullptr, "slot 62 nulls its output for a null IID");
        using Slot67Fn = HRESULT(WINAPI*)(void*, const void*, const IID*, void**);
        proxy_out = reinterpret_cast<void*>(0x1111);
        const IID iid_pso = __uuidof(ID3D12PipelineState);
        hr = reinterpret_cast<Slot67Fn>(proxy_vtbl[67])(device.Get(), nullptr, &iid_pso, &proxy_out);
        check(FAILED(hr) && proxy_out == nullptr, "slot 67 nulls its output for a null description");
        proxy_out = reinterpret_cast<void*>(0x1111);
        hr = reinterpret_cast<Slot67Fn>(proxy_vtbl[67])(device.Get(), &root_desc, nullptr, &proxy_out);
        check(FAILED(hr) && proxy_out == nullptr, "slot 67 nulls its output for a null IID");
        using Slot76Fn = HRESULT(WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
                                          const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
                                          const D3D12_CLEAR_VALUE*, const IID*, void**);
        proxy_out = reinterpret_cast<void*>(0x1111);
        hr = reinterpret_cast<Slot76Fn>(proxy_vtbl[76])(device.Get(), &heap_properties, D3D12_HEAP_FLAG_NONE,
                                                         nullptr, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         &iid_resource, &proxy_out);
        check(FAILED(hr) && proxy_out == nullptr, "slot 76 nulls its output for a null description");
    }

    // Slots 8 and 36 on the device the shim made: a title gets queue and fence proxies, and they run GPU work.
    {
        using CreateQueueFn = HRESULT(WINAPI*)(void*, const D3D12_COMMAND_QUEUE_DESC*, const IID*, void**);
        using CreateFenceFn = HRESULT(WINAPI*)(void*, UINT64, D3D12_FENCE_FLAGS, const IID*, void**);
        void** vtable = *reinterpret_cast<void***>(device.Get());
        const D3D12_COMMAND_QUEUE_DESC queue_desc{};
        const IID iid_queue = __uuidof(ID3D12CommandQueue);
        const IID iid_fence = __uuidof(ID3D12Fence);
        ComPtr<ID3D12CommandQueue> queue;
        ComPtr<ID3D12Fence> fence;
        hr = reinterpret_cast<CreateQueueFn>(vtable[8])(device.Get(), &queue_desc, &iid_queue,
                                                        reinterpret_cast<void**>(queue.GetAddressOf()));
        check(SUCCEEDED(hr) && rt::d3d12_xs::is_queue_proxy(queue.Get()), "slot 8 returns a queue proxy");
        hr = reinterpret_cast<CreateFenceFn>(vtable[36])(device.Get(), 0, D3D12_FENCE_FLAG_NONE, &iid_fence,
                                                         reinterpret_cast<void**>(fence.GetAddressOf()));
        check(SUCCEEDED(hr) && rt::d3d12_xs::is_fence_proxy(fence.Get()), "slot 36 returns a fence proxy");
        if (queue && fence) {
            HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            check(event && SUCCEEDED(queue->Signal(fence.Get(), 1)) &&
                      SUCCEEDED(fence->SetEventOnCompletion(1, event)) && WaitForSingleObject(event, 5000) == 0,
                  "the queue proxy signals the fence proxy on the GPU");
            if (event) CloseHandle(event);
            ComPtr<ID3D12Device> owner;
            check(SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(owner.GetAddressOf()))) && owner.Get() == device.Get(),
                  "the queue proxy's GetDevice is the device the shim returned");
        }
    }

    // Hold a reference to the PC device so the final check can prove the proxy's Release balances its own.
    if (real) real->AddRef();
    // Drop every device child before the proxy so the PC device is not released first.
    root_signature.Reset();
    reserved.Reset();
    texture.Reset();
    buffer.Reset();
    allocator.Reset();
    device1.Reset();
    const unsigned long refs_before_heap_release = rt::d3d12_xs::device_proxy_ref_count(device.Get());
    heap.Reset();
    check(rt::d3d12_xs::device_proxy_ref_count(device.Get()) + 1 == refs_before_heap_release,
          "heap release drops its device reference");
    device.Reset();
    if (real) {
        check(real->GetNodeCount() > 0, "PC device survives the proxy's final Release");
        real->Release();
    }

    std::printf("device_proxy_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
