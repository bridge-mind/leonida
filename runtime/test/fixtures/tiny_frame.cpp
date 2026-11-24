// A stand-in title for `gta6rt --run` that reaches the D3D12 layer the way a title does: it imports
// d3d12_xs.dll!D3D12XboxCreateDevice (from a link-time stub; the shim registry answers it) and then does the least
// a title does before its first frame, on the device that call returns: a direct queue, an allocator, a command
// list, a fence, an RTV heap whose start handle is read in the console shape (the 8-byte handle in RAX, heap slot 9),
// a committed render target, a clear, a copy into a readback buffer, Signal and a fence wait, Map, and a check of
// every pixel. It then returns from main with the device, the queue and the fence still alive, as a title's exit
// does. With LEONIDA_D3D12=vulkan (check_tiny_frame.py sets it) that device is the Vulkan-backed one, so this is the
// loader, the game thread, the Xbox-ABI proxies and the Vulkan device in one process. Hand-written test code, public
// D3D12 API only; no game data and no shader.
#include <windows.h>
#include <d3d12.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" __declspec(dllimport) HRESULT __stdcall D3D12XboxCreateDevice(IUnknown* adapter, const void* params,
                                                                        const IID* riid, void** device);

namespace {

int g_failures = 0;

bool check(bool ok, const char* what) {
    std::printf("tiny_frame: %s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
    return ok;
}

int finish() {
    std::printf("tiny_frame: %s\n", g_failures ? "FAILED" : "OK");
    std::fflush(stdout);
    return g_failures ? 1 : 0;
}

// Descriptor-heap slot 9 on the console returns the handle in RAX (docs/d3d12_vulkan.md section 7); the PC header's
// GetCPUDescriptorHandleForHeapStart returns it through a hidden pointer, which the heap proxy does not serve.
using HeapStartFn = std::uint64_t(__stdcall*)(void* heap);

constexpr UINT kWidth = 8, kHeight = 4;

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // The entry point's second argument is a block the shim logs and does not interpret; a title passes its own.
    std::uint32_t params[24] = {};
    const IID device_iid = __uuidof(ID3D12Device);
    ID3D12Device* device = nullptr;
    HRESULT hr = D3D12XboxCreateDevice(nullptr, params, &device_iid, reinterpret_cast<void**>(&device));
    if (!check(hr == S_OK && device, "D3D12XboxCreateDevice returns a device")) {
        std::printf("tiny_frame: hr=0x%08lx\n", static_cast<unsigned long>(hr));
        return finish();
    }

    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    check(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))) && queue, "a direct command queue");
    check(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) &&
              allocator, "a command allocator");
    if (allocator)
        check(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr,
                                                  IID_PPV_ARGS(&list))) && list, "a command list");
    check(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) && fence, "a fence");

    ID3D12DescriptorHeap* rtv_heap = nullptr;
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heap_desc.NumDescriptors = 1;
    check(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap))) && rtv_heap, "an RTV heap");
    if (!queue || !list || !fence || !rtv_heap) return finish();
    const std::uint64_t start = reinterpret_cast<HeapStartFn>((*reinterpret_cast<void***>(rtv_heap))[9])(rtv_heap);
    check(start != 0, "the heap's start handle comes back in RAX (heap slot 9)");
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv{static_cast<SIZE_T>(start)};

    D3D12_HEAP_PROPERTIES default_heap{}, readback_heap{};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC target_desc{};
    target_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    target_desc.Width = kWidth;
    target_desc.Height = kHeight;
    target_desc.DepthOrArraySize = 1;
    target_desc.MipLevels = 1;
    target_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    target_desc.SampleDesc.Count = 1;
    target_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    ID3D12Resource* target = nullptr;
    check(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &target_desc,
                                                    D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                    IID_PPV_ARGS(&target))) && target, "a committed render target");

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    device->GetCopyableFootprints(&target_desc, 0, 1, 0, &placed, &rows, &row_bytes, &total);
    check(rows == kHeight && row_bytes == kWidth * 4 && placed.Footprint.RowPitch >= kWidth * 4 && total != 0,
          "GetCopyableFootprints describes the target");
    D3D12_RESOURCE_DESC buffer_desc{};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Width = total ? total : 4096;
    buffer_desc.Height = 1;
    buffer_desc.DepthOrArraySize = 1;
    buffer_desc.MipLevels = 1;
    buffer_desc.SampleDesc.Count = 1;
    buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* readback = nullptr;
    check(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&readback))) && readback, "a readback buffer");
    if (!target || !readback) return finish();

    device->CreateRenderTargetView(target, nullptr, rtv);
    const FLOAT colour[4] = {64.0f / 255.0f, 128.0f / 255.0f, 191.0f / 255.0f, 1.0f};
    list->ClearRenderTargetView(rtv, colour, 0, nullptr);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = target;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = target;
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    to.pResource = readback;
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint = placed;
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    check(SUCCEEDED(list->Close()), "the list closes");
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);

    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    check(SUCCEEDED(queue->Signal(fence, 1)), "Signal");
    check(done && SUCCEEDED(fence->SetEventOnCompletion(1, done)) && WaitForSingleObject(done, 30000) == WAIT_OBJECT_0,
          "the fence reaches the signalled value");
    check(fence->GetCompletedValue() >= 1, "GetCompletedValue is past the signal");
    if (done) CloseHandle(done);

    void* bytes = nullptr;
    if (check(SUCCEEDED(readback->Map(0, nullptr, &bytes)) && bytes, "Map of the readback buffer")) {
        const std::uint8_t want[4] = {64, 128, 191, 255};
        unsigned wrong = 0;
        for (UINT y = 0; y < kHeight; ++y)
            for (UINT x = 0; x < kWidth; ++x)
                wrong += std::memcmp(static_cast<const std::uint8_t*>(bytes) + placed.Offset +
                                         y * placed.Footprint.RowPitch + x * 4, want, 4) != 0;
        check(wrong == 0, "all 32 pixels are the clear colour (64, 128, 191, 255)");
        readback->Unmap(0, nullptr);
    }

    // What a frame owns is released; the device, the queue and the fence stay alive across the return from main, so
    // the run ends through the CRT's exit with the layer still up.
    readback->Release();
    target->Release();
    rtv_heap->Release();
    list->Release();
    allocator->Release();
    return finish();
}
