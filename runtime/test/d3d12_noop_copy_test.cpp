// Legal D3D12 no-op copies must not poison Close or emit invalid Vulkan copy commands.
// Synthetic resources and native host WARP comparator; no game content.
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <cstdio>
#include <cstring>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

int check(ID3D12Device* device, const char* name) {
    Ref<ID3D12Resource> buffer, default_buffer, texture;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 65536;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                              D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&buffer.p)))) return 1;
    unsigned char* bytes = nullptr;
    D3D12_RANGE no_read{0, 0};
    if (FAILED(buffer->Map(0, &no_read, reinterpret_cast<void**>(&bytes))) || !bytes) return 1;
    for (unsigned i = 0; i < 65536; ++i) bytes[i] = static_cast<unsigned char>((i * 31 + 7) & 255);
    D3D12_RANGE written{0, 65536};
    buffer->Unmap(0, &written);

    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                              D3D12_RESOURCE_STATE_COMMON, nullptr,
                                              IID_PPV_ARGS(&default_buffer.p)))) return 1;
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 8;
    desc.Height = 8;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                              D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr, IID_PPV_ARGS(&texture.p)))) return 1;

    Ref<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue.p)))) return 1;
    for (int test = 0; test < 5; ++test) {
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p)))) return 1;
        if (test == 0) list->CopyResource(texture.p, texture.p);
        else if (test == 1) list->CopyResource(default_buffer.p, default_buffer.p);
        else {
            D3D12_TEXTURE_COPY_LOCATION location{};
            location.pResource = texture.p;
            location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            location.SubresourceIndex = 0;
            D3D12_TEXTURE_COPY_LOCATION upload{};
            upload.pResource = buffer.p;
            upload.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            upload.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, 8, 8, 1, 256};
            const D3D12_BOX box = test == 2 ? D3D12_BOX{2, 0, 0, 2, 3, 1} :
                                  test == 3 ? D3D12_BOX{2, 4, 0, 3, 4, 1} :
                                              D3D12_BOX{5, 0, 0, 3, 1, 1};
            list->CopyTextureRegion(&location, 0, 0, 0, &upload, &box);
        }
        const HRESULT closed = list->Close();
        if (FAILED(closed)) {
            std::fprintf(stderr, "%s no-op case %d: Close failed %08x\n", name, test,
                         static_cast<unsigned>(closed));
            return 1;
        }
        ID3D12CommandList* execute[] = {list.p};
        queue->ExecuteCommandLists(1, execute);
        // No GPU resources/allocators may be torn down until this submission completes.
        Ref<ID3D12Fence> done;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done.p))) ||
            FAILED(queue->Signal(done.p, 1))) return 1;
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) return 1;
        const bool ok = SUCCEEDED(done->SetEventOnCompletion(1, event)) &&
                        WaitForSingleObject(event, 5000) == WAIT_OBJECT_0;
        CloseHandle(event);
        if (!ok) return 1;
    }
    // Upload memory is not a copy destination. This is invalid even for a self-copy: accepting every
    // same-resource pair as a no-op would silently loosen validation compared to native D3D12.
    {
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p)))) return 1;
        list->CopyResource(buffer.p, buffer.p);
        if (SUCCEEDED(list->Close())) {
            std::fprintf(stderr, "%s incorrectly accepted upload-heap self-copy\n", name);
            return 1;
        }
    }
    if (FAILED(buffer->Map(0, &no_read, reinterpret_cast<void**>(&bytes))) || !bytes) return 1;
    unsigned mismatches = 0;
    for (unsigned i = 0; i < 65536; ++i) mismatches += bytes[i] != static_cast<unsigned char>((i * 31 + 7) & 255);
    buffer->Unmap(0, &no_read);
    std::printf("%s: 5 legal no-op copy submissions, invalid upload copy rejected; pattern mismatches=%u\n",
                name, mismatches);
    return mismatches ? 1 : 0;
}
} // namespace

int main() {
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> warp;
    Ref<ID3D12Device> host;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) ||
        FAILED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) return 77;
    if (check(host.p, "native WARP")) return 1;
    Ref<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&layer.p)))) return 77;
    return check(layer.p, "Leonida Vulkan");
}
