// Minimal smoke test for D3D12 1D texture creation and RTV/DSV/SRV/UAV views.

#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"

#include <cstdio>
#include <cstring>

namespace {

int fail(const char* msg, HRESULT hr = S_OK) {
    std::fprintf(stderr, "FAIL: %s (hr=0x%08X)\n", msg, static_cast<unsigned>(hr));
    return 1;
}

D3D12_HEAP_PROPERTIES default_heap() {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    return hp;
}

}  // namespace

int main() {
    ID3D12Device* dev = nullptr;
    HRESULT hr = rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&dev));
    if (FAILED(hr) || !dev) {
        std::printf("SKIP: no Vulkan device (hr=0x%08X)\n", static_cast<unsigned>(hr));
        return 77;
    }

    D3D12_DESCRIPTOR_HEAP_DESC cbv_heap_desc{};
    cbv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbv_heap_desc.NumDescriptors = 8;
    ID3D12DescriptorHeap* cbv_heap = nullptr;
    hr = dev->CreateDescriptorHeap(&cbv_heap_desc, IID_PPV_ARGS(&cbv_heap));
    if (FAILED(hr) || !cbv_heap) return fail("CreateDescriptorHeap(CBV_SRV_UAV)", hr);

    D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{};
    rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv_heap_desc.NumDescriptors = 2;
    ID3D12DescriptorHeap* rtv_heap = nullptr;
    hr = dev->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap));
    if (FAILED(hr) || !rtv_heap) return fail("CreateDescriptorHeap(RTV)", hr);

    D3D12_DESCRIPTOR_HEAP_DESC dsv_heap_desc{};
    dsv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsv_heap_desc.NumDescriptors = 2;
    ID3D12DescriptorHeap* dsv_heap = nullptr;
    hr = dev->CreateDescriptorHeap(&dsv_heap_desc, IID_PPV_ARGS(&dsv_heap));
    if (FAILED(hr) || !dsv_heap) return fail("CreateDescriptorHeap(DSV)", hr);

    const UINT cbv_inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT rtv_inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    const UINT dsv_inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

    // 1D color texture (DepthOrArraySize = 1) and 1D array (DepthOrArraySize = 4)
    D3D12_RESOURCE_DESC tex1d_single_desc{};
    tex1d_single_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
    tex1d_single_desc.Width = 64;
    tex1d_single_desc.Height = 1;
    tex1d_single_desc.DepthOrArraySize = 1;
    tex1d_single_desc.MipLevels = 2;
    tex1d_single_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    tex1d_single_desc.SampleDesc = {1, 0};
    tex1d_single_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    tex1d_single_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_RESOURCE_DESC tex1d_desc = tex1d_single_desc;
    tex1d_desc.DepthOrArraySize = 4;

    ID3D12Resource* tex1d_single = nullptr;
    ID3D12Resource* tex1d = nullptr;
    const D3D12_HEAP_PROPERTIES hp = default_heap();
    hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &tex1d_single_desc, D3D12_RESOURCE_STATE_COMMON,
                                      nullptr, IID_PPV_ARGS(&tex1d_single));
    if (FAILED(hr) || !tex1d_single) return fail("CreateCommittedResource(TEXTURE1D single)", hr);
    hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &tex1d_desc, D3D12_RESOURCE_STATE_COMMON,
                                      nullptr, IID_PPV_ARGS(&tex1d));
    if (FAILED(hr) || !tex1d) return fail("CreateCommittedResource(TEXTURE1D array)", hr);

    D3D12_CPU_DESCRIPTOR_HANDLE cbv_cpu = cbv_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE rtv_cpu = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE dsv_cpu = dsv_heap->GetCPUDescriptorHandleForHeapStart();

    // SRV: default (nullptr) on 1D single, explicit TEXTURE1D on 1D single (mip 1) and array (slice 0)
    dev->CreateShaderResourceView(tex1d_single, nullptr, cbv_cpu);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv1d_mip1{};
    srv1d_mip1.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv1d_mip1.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
    srv1d_mip1.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv1d_mip1.Texture1D.MostDetailedMip = 1;
    srv1d_mip1.Texture1D.MipLevels = 1;
    dev->CreateShaderResourceView(tex1d_single, &srv1d_mip1, {cbv_cpu.ptr + cbv_inc * 1});

    D3D12_SHADER_RESOURCE_VIEW_DESC srv1d{};
    srv1d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv1d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
    srv1d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv1d.Texture1D.MipLevels = 2;
    dev->CreateShaderResourceView(tex1d, &srv1d, {cbv_cpu.ptr + cbv_inc * 2});

    // UAV: default (nullptr) on 1D single, explicit TEXTURE1D on 1D single (mip 0) and array (slice 0, mip 1)
    dev->CreateUnorderedAccessView(tex1d_single, nullptr, nullptr, {cbv_cpu.ptr + cbv_inc * 3});

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav1d_single{};
    uav1d_single.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uav1d_single.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
    uav1d_single.Texture1D.MipSlice = 0;
    dev->CreateUnorderedAccessView(tex1d_single, nullptr, &uav1d_single, {cbv_cpu.ptr + cbv_inc * 4});

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav1d{};
    uav1d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uav1d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
    uav1d.Texture1D.MipSlice = 1;
    dev->CreateUnorderedAccessView(tex1d, nullptr, &uav1d, {cbv_cpu.ptr + cbv_inc * 5});

    // Default (nullptr) SRV and UAV on a 1D array must write a null descriptor (set 0 has no 1D-array binding).
    std::memset(reinterpret_cast<void*>(cbv_cpu.ptr + cbv_inc * 6), 0x77, cbv_inc * 2);
    dev->CreateShaderResourceView(tex1d, nullptr, {cbv_cpu.ptr + cbv_inc * 6});
    dev->CreateUnorderedAccessView(tex1d, nullptr, nullptr, {cbv_cpu.ptr + cbv_inc * 7});

    namespace g = leonida::gfx10;
    const auto decode_and_lookup = [&](UINT slot, g::Image& img) {
        g::ImageWords words{};
        std::memcpy(words.data(), reinterpret_cast<const void*>(cbv_cpu.ptr + cbv_inc * slot), sizeof(words));
        if (!g::is_image(words) || !g::decode(words, img)) return false;
        std::uint32_t key[8]{};
        std::memcpy(key, words.data(), sizeof(key));
        return rt::d3d12::bindless_hash_lookup(dev, key) != 0;
    };
    g::Image img{};
    if (!decode_and_lookup(0, img) || img.type != g::ImageType::image_1d || img.width != 64 || img.mip_count != 2 ||
        img.layer_count != 1)
        return fail("default 1D SRV T# or bindless lookup mismatch");
    if (!decode_and_lookup(1, img) || img.type != g::ImageType::image_1d || img.first_mip != 1 ||
        img.mip_count != 1 || img.layer_count != 1)
        return fail("explicit 1D mip-1 SRV T# or bindless lookup mismatch");
    if (!decode_and_lookup(2, img) || img.type != g::ImageType::image_1d || img.layer_count != 1 || img.mip_count != 2)
        return fail("explicit 1D SRV on array T# or bindless lookup mismatch");
    if (!decode_and_lookup(3, img) || img.type != g::ImageType::image_1d || img.mip_count != 1 || img.layer_count != 1)
        return fail("default 1D UAV T# or bindless lookup mismatch");
    if (!decode_and_lookup(4, img) || img.type != g::ImageType::image_1d || img.first_mip != 0 ||
        img.mip_count != 1 || img.layer_count != 1)
        return fail("explicit 1D UAV T# or bindless lookup mismatch");
    if (!decode_and_lookup(5, img) || img.type != g::ImageType::image_1d || img.first_mip != 1 || img.mip_count != 1 ||
        img.layer_count != 1)
        return fail("explicit 1D UAV on array T# or bindless lookup mismatch");
    for (UINT slot = 6; slot < 8; ++slot) {
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(cbv_cpu.ptr + cbv_inc * slot);
        for (UINT b = 0; b < 32; ++b)
            if (bytes[b] != 0) return fail("default 1DArray view should write a null descriptor");
    }

    // RTV / DSV 1D bounds validation (out-of-bounds mip/slice must write a null TargetView)
    D3D12_RENDER_TARGET_VIEW_DESC bad_rtv{};
    bad_rtv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    bad_rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1DARRAY;
    bad_rtv.Texture1DArray.MipSlice = 5;
    bad_rtv.Texture1DArray.FirstArraySlice = 0;
    bad_rtv.Texture1DArray.ArraySize = 1;
    dev->CreateRenderTargetView(tex1d, &bad_rtv, {rtv_cpu.ptr + rtv_inc});
    if (*reinterpret_cast<const std::uint64_t*>(rtv_cpu.ptr + rtv_inc) != 0)
        return fail("out-of-bounds 1D RTV should be null");

    D3D12_DEPTH_STENCIL_VIEW_DESC bad_dsv{};
    bad_dsv.Format = DXGI_FORMAT_D32_FLOAT;
    bad_dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1DARRAY;
    bad_dsv.Texture1DArray.MipSlice = 0;
    bad_dsv.Texture1DArray.FirstArraySlice = 4;
    bad_dsv.Texture1DArray.ArraySize = 1;
    dev->CreateDepthStencilView(tex1d, &bad_dsv, {dsv_cpu.ptr + dsv_inc});
    if (*reinterpret_cast<const std::uint64_t*>(dsv_cpu.ptr + dsv_inc) != 0)
        return fail("out-of-bounds 1D DSV should be null");

    // GPU round-trip: upload buffer -> CopyTextureRegion -> TEXTURE1D -> CopyTextureRegion -> readback buffer
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT num_rows = 0;
    UINT64 row_bytes = 0, total_bytes = 0;
    dev->GetCopyableFootprints(&tex1d_desc, 0, 1, 0, &footprint, &num_rows, &row_bytes, &total_bytes);
    if (total_bytes == 0 || footprint.Footprint.Height != 1 || footprint.Footprint.Depth != 1) {
        return fail("GetCopyableFootprints(TEXTURE1D)");
    }

    D3D12_HEAP_PROPERTIES upload_hp{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_HEAP_PROPERTIES readback_hp{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC buf_desc{};
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Width = total_bytes;
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.MipLevels = 1;
    buf_desc.Format = DXGI_FORMAT_UNKNOWN;
    buf_desc.SampleDesc = {1, 0};
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ID3D12Resource* upload_buf = nullptr;
    ID3D12Resource* readback_buf = nullptr;
    hr = dev->CreateCommittedResource(&upload_hp, D3D12_HEAP_FLAG_NONE, &buf_desc,
                                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload_buf));
    if (FAILED(hr) || !upload_buf) return fail("CreateCommittedResource(upload)", hr);
    hr = dev->CreateCommittedResource(&readback_hp, D3D12_HEAP_FLAG_NONE, &buf_desc,
                                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_buf));
    if (FAILED(hr) || !readback_buf) return fail("CreateCommittedResource(readback)", hr);

    unsigned char* mapped = nullptr;
    hr = upload_buf->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr) || !mapped) return fail("upload_buf->Map", hr);
    for (UINT i = 0; i < 64 * 4; ++i) mapped[i] = static_cast<unsigned char>((i * 7 + 13) & 0xFF);
    upload_buf->Unmap(0, nullptr);

    D3D12_COMMAND_QUEUE_DESC qd{};
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* cmd = nullptr;
    ID3D12Fence* fence = nullptr;
    if (FAILED(hr = dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)))) return fail("CreateCommandQueue", hr);
    if (FAILED(hr = dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))))
        return fail("CreateCommandAllocator", hr);
    if (FAILED(hr = dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&cmd))))
        return fail("CreateCommandList", hr);
    if (FAILED(hr = dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return fail("CreateFence", hr);

    D3D12_TEXTURE_COPY_LOCATION src_loc{upload_buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, footprint};
    D3D12_TEXTURE_COPY_LOCATION dst_loc{tex1d, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    cmd->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {tex1d, 0, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE};
    cmd->ResourceBarrier(1, &barrier);

    D3D12_TEXTURE_COPY_LOCATION rb_loc{readback_buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, footprint};
    cmd->CopyTextureRegion(&rb_loc, 0, 0, 0, &dst_loc, nullptr);
    if (FAILED(hr = cmd->Close())) return fail("cmd->Close", hr);

    ID3D12CommandList* lists[] = {cmd};
    queue->ExecuteCommandLists(1, lists);
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (FAILED(hr = queue->Signal(fence, 1))) return fail("queue->Signal", hr);
    if (fence->GetCompletedValue() < 1) {
        fence->SetEventOnCompletion(1, ev);
        WaitForSingleObject(ev, INFINITE);
    }
    CloseHandle(ev);

    unsigned char* rb_mapped = nullptr;
    hr = readback_buf->Map(0, nullptr, reinterpret_cast<void**>(&rb_mapped));
    if (FAILED(hr) || !rb_mapped) return fail("readback_buf->Map", hr);
    for (UINT i = 0; i < 64 * 4; ++i) {
        const unsigned char expected = static_cast<unsigned char>((i * 7 + 13) & 0xFF);
        if (rb_mapped[i] != expected) return fail("1D texture readback mismatch");
    }
    readback_buf->Unmap(0, nullptr);

    fence->Release();
    cmd->Release();
    alloc->Release();
    queue->Release();
    readback_buf->Release();
    upload_buf->Release();
    tex1d->Release();
    tex1d_single->Release();
    dsv_heap->Release();
    rtv_heap->Release();
    cbv_heap->Release();
    dev->Release();

    std::printf("PASS: d3d12_texture1d_test (1D/1DArray SRV, UAV + CopyTextureRegion round-trip)\n");
    return 0;
}
