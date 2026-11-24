// End-to-end GPU + host-parity tests for:
// 1. ClearState resetting active SetPredication (proved on Vulkan layer, Windows 11 WARP, and native HW D3D12).
// 2. SetPredication recording-time validation parity against live Windows 11 WARP and native HW D3D12.
// 3. AtomicCopyBufferUINT / AtomicCopyBufferUINT64 recording-time validation parity and GPU execution.
// 100% generic standard D3D12 API parity; no game data.

#include "vk_test_device.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

#define REQUIRE(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
    T* operator->() const { return p; }
};

bool create_buffer(ID3D12Device* device, UINT64 size, D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_STATES state,
                   Ref<ID3D12Resource>& out) {
    const D3D12_HEAP_PROPERTIES heap{heap_type};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                    IID_PPV_ARGS(&out.p))));
    return true;
}

bool create_texture2d(ID3D12Device* device, UINT width, UINT height, UINT16 mips, UINT16 layers,
                      D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out) {
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = layers;
    desc.MipLevels = mips;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                    IID_PPV_ARGS(&out.p))));
    return true;
}

bool write_buffer(ID3D12Resource* resource, const void* data, SIZE_T size) {
    void* mapped = nullptr;
    const D3D12_RANGE none{};
    CHECK(SUCCEEDED(resource->Map(0, &none, &mapped)) && mapped);
    std::memcpy(mapped, data, size);
    const D3D12_RANGE written{0, size};
    resource->Unmap(0, &written);
    return true;
}

bool read_buffer(ID3D12Resource* resource, void* out, SIZE_T size) {
    void* mapped = nullptr;
    const D3D12_RANGE range{0, size};
    CHECK(SUCCEEDED(resource->Map(0, &range, &mapped)) && mapped);
    std::memcpy(out, mapped, size);
    const D3D12_RANGE none{};
    resource->Unmap(0, &none);
    return true;
}

bool submit_and_wait(ID3D12GraphicsCommandList* list, ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64 value) {
    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence, value)));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    CHECK(event);
    const HRESULT hr = fence->SetEventOnCompletion(value, event);
    const DWORD wait = SUCCEEDED(hr) ? WaitForSingleObject(event, 10000) : WAIT_FAILED;
    CloseHandle(event);
    CHECK(SUCCEEDED(hr) && wait == WAIT_OBJECT_0);
    return true;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

// GPU proof: SetPredication(predicate=0, EQUAL_ZERO) skips a clear, then ClearState(nullptr) resets predication so
// the subsequent clear writes blue instead of being skipped. Also checks CopyBufferRegion after ClearState.
bool test_clear_state_resets_predication_gpu(const char* label, ID3D12Device* device, bool is_layer) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC qdesc{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                              IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    constexpr UINT kDim = 4;
    constexpr UINT kRowPitch = 256;
    Ref<ID3D12Resource> target, predicate_upload, predicate_def, readback;
    CHECK(create_texture2d(device, kDim, kDim, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                           D3D12_RESOURCE_STATE_RENDER_TARGET, target));
    CHECK(create_buffer(device, 64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, predicate_upload));
    CHECK(create_buffer(device, 64, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, predicate_def));
    CHECK(create_buffer(device, kRowPitch * kDim, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, readback));

    const UINT64 zero = 0;
    CHECK(write_buffer(predicate_upload.p, &zero, sizeof(zero)));

    Ref<ID3D12DescriptorHeap> rtv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap.p))));
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.p, nullptr, rtv);

    // Upload predicate = 0 and transition to PREDICATION.
    list->CopyBufferRegion(predicate_def.p, 0, predicate_upload.p, 0, 8);
    const auto to_pred = transition(predicate_def.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PREDICATION);
    list->ResourceBarrier(1, &to_pred);

    // 1. Unpredicated clear to red.
    const FLOAT red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const FLOAT green[4] = {0.0f, 1.0f, 0.0f, 1.0f};
    const FLOAT blue[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    list->ClearRenderTargetView(rtv, red, 0, nullptr);

    // 2. Enable predication (predicate == 0 with EQUAL_ZERO -> skip). Green clear must be skipped.
    list->SetPredication(predicate_def.p, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    list->ClearRenderTargetView(rtv, green, 0, nullptr);

    // 3. ClearState must reset predication to NULL. Blue clear must execute!
    list->ClearState(nullptr);
    list->ClearRenderTargetView(rtv, blue, 0, nullptr);

    const auto to_src = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_src);

    D3D12_TEXTURE_COPY_LOCATION dst_loc{};
    dst_loc.pResource = readback.p;
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst_loc.PlacedFootprint.Offset = 0;
    dst_loc.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, kDim, kDim, 1, kRowPitch};
    D3D12_TEXTURE_COPY_LOCATION src_loc{};
    src_loc.pResource = target.p;
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_loc.SubresourceIndex = 0;
    list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);

    CHECK(submit_and_wait(list.p, queue.p, fence.p, 1));
    if (is_layer) CHECK(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);

    std::vector<std::uint8_t> pixels(kRowPitch * kDim);
    CHECK(read_buffer(readback.p, pixels.data(), pixels.size()));
    for (UINT y = 0; y < kDim; ++y) {
        for (UINT x = 0; x < kDim; ++x) {
            const std::uint8_t* px = pixels.data() + y * kRowPitch + x * 4;
            if (px[0] != 0 || px[1] != 0 || px[2] != 255 || px[3] != 255) {
                std::fprintf(stderr, "FAIL [%s] ClearState predication reset pixel (%u,%u): got (%u,%u,%u,%u), want (0,0,255,255)\n",
                             label, x, y, px[0], px[1], px[2], px[3]);
                return false;
            }
        }
    }
    std::printf("PASS [%s] ClearState resets predication (GPU readback verified blue)\n", label);
    return true;
}

// GPU proof on Vulkan layer: AtomicCopyBufferUINT with count=0 and UINT64_MAX end sentinels
// ({0, UINT64_MAX}, {1, UINT64_MAX}, {63, UINT64_MAX}), plus replay bounds check on {64, UINT64_MAX}.
bool test_atomic_copy_gpu(ID3D12Device* device) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList1> list1;
    Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC qdesc{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                              IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list1.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    Ref<ID3D12Resource> src, dst;
    CHECK(create_buffer(device, 64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, src));
    CHECK(create_buffer(device, 64, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, dst));

    const std::array<std::uint32_t, 4> input{0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u};
    CHECK(write_buffer(src.p, input.data(), sizeof(input)));

    ID3D12Resource* dep = src.p;
    const D3D12_SUBRESOURCE_RANGE_UINT64 full_buf{0, {0, UINT64_MAX}};
    const D3D12_SUBRESOURCE_RANGE_UINT64 tail_1{0, {1, UINT64_MAX}};
    const D3D12_SUBRESOURCE_RANGE_UINT64 tail_63{0, {63, UINT64_MAX}};
    list1->AtomicCopyBufferUINT(dst.p, 0, src.p, 0, 0, nullptr, nullptr);
    list1->AtomicCopyBufferUINT(dst.p, 4, src.p, 4, 1, &dep, &full_buf);
    list1->AtomicCopyBufferUINT(dst.p, 8, src.p, 8, 1, &dep, &tail_1);
    list1->AtomicCopyBufferUINT(dst.p, 12, src.p, 12, 1, &dep, &tail_63);

    CHECK(submit_and_wait(list.p, queue.p, fence.p, 1));
    CHECK(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);

    std::array<std::uint32_t, 4> out{};
    CHECK(read_buffer(dst.p, out.data(), sizeof(out)));
    CHECK(out == input);

    // {64, UINT64_MAX} on a 64-byte buffer passes Close() (matching WARP + HW D3D12) but fails replay bounds check.
    CHECK(SUCCEEDED(allocator->Reset()));
    CHECK(SUCCEEDED(list->Reset(allocator.p, nullptr)));
    const D3D12_SUBRESOURCE_RANGE_UINT64 oob_begin{0, {64, UINT64_MAX}};
    list1->AtomicCopyBufferUINT(dst.p, 0, src.p, 0, 1, &dep, &oob_begin);
    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::invalid_copy);

    std::printf("PASS [layer] AtomicCopyBufferUINT count=0 and UINT64_MAX sentinel GPU readback\n");
    return true;
}

struct Env {
    ID3D12Device* device = nullptr;
    Ref<ID3D12Resource> buf64, buf64_2, upload, readback, tex2d, tex_mips;

    bool init(ID3D12Device* d) {
        device = d;
        return create_buffer(device, 64, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, buf64) &&
               create_buffer(device, 64, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, buf64_2) &&
               create_buffer(device, 64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, upload) &&
               create_buffer(device, 64, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, readback) &&
               create_texture2d(device, 64, 64, 1, 1, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, tex2d) &&
               create_texture2d(device, 64, 64, 3, 2, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, tex_mips);
    }
};

struct ParityCase {
    const char* name;
    D3D12_COMMAND_LIST_TYPE list_type;
    bool expected_ok;       // Measured on Windows 11 WARP (and RTX 5090 for all SetPredication + AtomicCopyBufferUINT)
    bool compare_hw_d3d12;  // False for AtomicCopyBufferUINT64 valid cases where NVIDIA HW D3D12 does not support 64-bit atomic copy
    std::function<void(Env&, ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList1*)> record;
};

std::vector<ParityCase> build_cases() {
    constexpr auto D = D3D12_COMMAND_LIST_TYPE_DIRECT;
    constexpr auto C = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    constexpr auto CP = D3D12_COMMAND_LIST_TYPE_COPY;
    constexpr auto B = D3D12_COMMAND_LIST_TYPE_BUNDLE;
    constexpr auto eq0 = D3D12_PREDICATION_OP_EQUAL_ZERO;
    constexpr auto ne0 = D3D12_PREDICATION_OP_NOT_EQUAL_ZERO;

    const D3D12_SUBRESOURCE_RANGE_UINT64 dep_ok{0, {0, 64}};

    return {
        // --- SetPredication ---
        {"SetPredication valid (buf, 0, EQUAL_ZERO)", D, true, true,
         [=](Env& e, auto* l, auto*) { l->SetPredication(e.buf64.p, 0, eq0); }},
        {"SetPredication valid (buf, 56, NOT_EQUAL_ZERO)", D, true, true,
         [=](Env& e, auto* l, auto*) { l->SetPredication(e.buf64.p, 56, ne0); }},
        {"SetPredication valid on COMPUTE list", C, true, true,
         [=](Env& e, auto* l, auto*) { l->SetPredication(e.buf64.p, 0, eq0); }},
        {"SetPredication on COPY list", CP, false, true,
         [=](Env& e, auto* l, auto*) { l->SetPredication(e.buf64.p, 0, eq0); }},
        {"SetPredication on BUNDLE", B, false, true,
         [=](Env& e, auto* l, auto*) { l->SetPredication(e.buf64.p, 0, eq0); }},
        {"SetPredication(nullptr, 0, EQUAL_ZERO)", D, true, true,
         [=](Env&, auto* l, auto*) { l->SetPredication(nullptr, 0, eq0); }},
        {"SetPredication(nullptr, 4, EQUAL_ZERO)", D, false, true,
         [=](Env&, auto* l, auto*) { l->SetPredication(nullptr, 4, eq0); }},
        {"SetPredication(nullptr, 8, EQUAL_ZERO)", D, false, true,
         [=](Env&, auto* l, auto*) { l->SetPredication(nullptr, 8, eq0); }},
        {"SetPredication(nullptr, 0, NOT_EQUAL_ZERO)", D, false, true,
         [=](Env&, auto* l, auto*) { l->SetPredication(nullptr, 0, ne0); }},
        {"SetPredication(nullptr, 0, op=7)", D, false, true,
         [=](Env&, auto* l, auto*) { l->SetPredication(nullptr, 0, D3D12_PREDICATION_OP(7)); }},
        {"SetPredication(tex2d, 0, EQUAL_ZERO)", D, false, true,
         [=](Env& e, auto* l, auto*) { l->SetPredication(e.tex2d.p, 0, eq0); }},
        {"SetPredication(buf, 4, EQUAL_ZERO) unaligned", D, false, true,
         [=](Env& e, auto* l, auto*) { l->SetPredication(e.buf64.p, 4, eq0); }},
        {"SetPredication(buf, 64, EQUAL_ZERO) OOB exact end", D, false, true,
         [=](Env& e, auto* l, auto*) { l->SetPredication(e.buf64.p, 64, eq0); }},
        {"SetPredication(buf, 128, EQUAL_ZERO) OOB past end", D, false, true,
         [=](Env& e, auto* l, auto*) { l->SetPredication(e.buf64.p, 128, eq0); }},
        {"SetPredication(buf, UINT64_MAX-7, EQUAL_ZERO) wrap", D, false, true,
         [=](Env& e, auto* l, auto*) { l->SetPredication(e.buf64.p, UINT64_MAX - 7, eq0); }},
        {"SetPredication(buf, 0, op=7) invalid op", D, false, true,
         [=](Env& e, auto* l, auto*) { l->SetPredication(e.buf64.p, 0, D3D12_PREDICATION_OP(7)); }},
        {"SetPredication(upload, 0, EQUAL_ZERO)", D, true, true,
         [=](Env& e, auto* l, auto*) { l->SetPredication(e.upload.p, 0, eq0); }},
        {"SetPredication(readback, 0, EQUAL_ZERO)", D, false, true,
         [=](Env& e, auto* l, auto*) { l->SetPredication(e.readback.p, 0, eq0); }},

        // --- AtomicCopyBufferUINT / UINT64 ---
        {"AtomicCopyUINT valid (count=1)", D, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT64 valid (count=1)", D, true, false,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT64(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT valid on COMPUTE", C, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT valid on COPY", CP, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT on BUNDLE", B, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT count=0, null/null", D, true, true,
         [=](Env& e, auto*, auto* l1) { l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 0, nullptr, nullptr); }},
        {"AtomicCopyUINT64 count=0, null/null", D, true, false,
         [=](Env& e, auto*, auto* l1) {
             l1->AtomicCopyBufferUINT64(e.buf64.p, 0, e.buf64_2.p, 0, 0, nullptr, nullptr);
         }},
        {"AtomicCopyUINT null dst", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(nullptr, 0, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT null src", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, nullptr, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT dst=tex2d", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.tex2d.p, 0, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT src=tex2d", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.tex2d.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT dst_offset unaligned (2)", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.buf64.p, 2, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT64 src_offset unaligned (4)", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT64(e.buf64.p, 0, e.buf64_2.p, 4, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT dst_offset=60 (exact 4B fit)", D, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.buf64.p, 60, e.buf64_2.p, 60, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT64 dst_offset=56 (exact 8B fit)", D, true, false,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT64(e.buf64.p, 56, e.buf64_2.p, 56, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT dst_offset OOB (64)", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.buf64.p, 64, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT64 dst_offset OOB (60)", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT64(e.buf64.p, 60, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT src_offset OOB (64)", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 64, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT dst_offset wrap (UINT64_MAX-3)", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.buf64.p, UINT64_MAX - 3, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT64 dst_offset wrap (UINT64_MAX-7)", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT64(e.buf64.p, UINT64_MAX - 7, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT dst=UPLOAD", D, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.upload.p, 0, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT src=READBACK", D, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.readback.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT dst=READBACK, src=UPLOAD", D, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.readback.p, 0, e.upload.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT dep buf full-range {0, UINT64_MAX}", D, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{0, {0, UINT64_MAX}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep buf sentinel {1, UINT64_MAX}", D, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{0, {1, UINT64_MAX}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep buf sentinel {63, UINT64_MAX}", D, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{0, {63, UINT64_MAX}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep buf sentinel {64, UINT64_MAX}", D, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{0, {64, UINT64_MAX}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep buf sentinel {65, UINT64_MAX}", D, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{0, {65, UINT64_MAX}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep buf sentinel {MAX, MAX}", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{0, {UINT64_MAX, UINT64_MAX}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep buf empty range {0, 0}", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{0, {0, 0}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep buf inverted range {32, 16}", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{0, {32, 16}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep buf OOB end {0, 65}", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{0, {0, 65}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep buf OOB subresource 1", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{1, {0, 64}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep buf sub=ALL_SUBRESOURCES", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, {0, 64}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep tex sub=5 {0, UINT64_MAX}", D, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.tex_mips.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{5, {0, UINT64_MAX}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep tex sub=ALL {0, UINT64_MAX}", D, true, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.tex_mips.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, {0, UINT64_MAX}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep tex sub=6 OOB {0, UINT64_MAX}", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.tex_mips.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{6, {0, UINT64_MAX}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep tex partial range {0, 64}", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.tex2d.p;
             const D3D12_SUBRESOURCE_RANGE_UINT64 rng{0, {0, 64}};
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &rng);
         }},
        {"AtomicCopyUINT dep null resource element", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = nullptr;
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, &dep_ok);
         }},
        {"AtomicCopyUINT count=1, resources=nullptr", D, false, true,
         [=](Env& e, auto*, auto* l1) { l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, nullptr, &dep_ok); }},
        {"AtomicCopyUINT count=1, ranges=nullptr", D, false, true,
         [=](Env& e, auto*, auto* l1) {
             ID3D12Resource* r = e.buf64_2.p;
             l1->AtomicCopyBufferUINT(e.buf64.p, 0, e.buf64_2.p, 0, 1, &r, nullptr);
         }},
    };
}

bool run_close_case(Env& env, const ParityCase& c, HRESULT& close_hr) {
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList1> list1;
    CHECK(SUCCEEDED(env.device->CreateCommandAllocator(c.list_type, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(env.device->CreateCommandList(0, c.list_type, allocator.p, nullptr, IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list1.p))));
    c.record(env, list.p, list1.p);
    close_hr = list->Close();
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    REQUIRE(status == vktest::DeviceStatus::ready);
    const unsigned errors_before = vk.validation_errors();

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = true;
    info.conditional_rendering = vk.conditional_rendering;

    Ref<ID3D12Device> ours;
    REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p))));

    // 1. GPU end-to-end ClearState predication reset proof (requires VK_EXT_conditional_rendering on Vulkan device).
    if (vk.conditional_rendering) {
        REQUIRE(test_clear_state_resets_predication_gpu("layer", ours.p, true));
    } else {
        std::printf("NOTE: VK_EXT_conditional_rendering unavailable on this Vulkan device; skipping layer GPU predication check\n");
    }
    REQUIRE(test_atomic_copy_gpu(ours.p));

    // 2. Host devices: Windows 11 WARP and native HW D3D12.
    Ref<ID3D12Device> warp_dev, hw_dev;
    std::string hw_name;
    if (!hostwine::host_is_wine()) {
        Ref<IDXGIFactory4> factory;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            Ref<IDXGIAdapter> warp_adapter;
            if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp_adapter.p)))) {
                if (FAILED(D3D12CreateDevice(warp_adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&warp_dev.p))))
                    warp_dev.p = nullptr;
            }
            Ref<IDXGIAdapter1> hw_adapter;
            for (UINT i = 0; factory->EnumAdapters1(i, &hw_adapter.p) != DXGI_ERROR_NOT_FOUND; ++i) {
                DXGI_ADAPTER_DESC1 desc{};
                hw_adapter->GetDesc1(&desc);
                if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                    if (SUCCEEDED(D3D12CreateDevice(hw_adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&hw_dev.p)))) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf), "HW: %ls", desc.Description);
                        hw_name = buf;
                        break;
                    }
                }
                hw_adapter.p->Release();
                hw_adapter.p = nullptr;
            }
        }
    }

    if (warp_dev.p) REQUIRE(test_clear_state_resets_predication_gpu("WARP", warp_dev.p, false));
    if (hw_dev.p) REQUIRE(test_clear_state_resets_predication_gpu(hw_name.c_str(), hw_dev.p, false));

    // 3. Full SetPredication + AtomicCopyBuffer Close() parity matrix.
    const auto cases = build_cases();
    std::vector<HRESULT> layer_hr(cases.size()), warp_hr(cases.size()), hw_hr(cases.size());

    {
        Env env;
        REQUIRE(env.init(ours.p));
        for (std::size_t i = 0; i < cases.size(); ++i) REQUIRE(run_close_case(env, cases[i], layer_hr[i]));
    }
    if (warp_dev.p) {
        Env env;
        REQUIRE(env.init(warp_dev.p));
        for (std::size_t i = 0; i < cases.size(); ++i) REQUIRE(run_close_case(env, cases[i], warp_hr[i]));
        REQUIRE(warp_dev->GetDeviceRemovedReason() == S_OK);
    }
    if (hw_dev.p) {
        Env env;
        REQUIRE(env.init(hw_dev.p));
        for (std::size_t i = 0; i < cases.size(); ++i) REQUIRE(run_close_case(env, cases[i], hw_hr[i]));
        REQUIRE(hw_dev->GetDeviceRemovedReason() == S_OK);
    }

    unsigned failures = 0;
    std::printf("%-54s | %-10s | %-10s | %-10s\n", "case", "layer", "WARP", "HW D3D12");
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        const bool l_ok = SUCCEEDED(layer_hr[i]);
        const bool w_ok = warp_dev.p ? SUCCEEDED(warp_hr[i]) : c.expected_ok;
        const bool h_ok = hw_dev.p ? SUCCEEDED(hw_hr[i]) : c.expected_ok;
        std::printf("%-54s | 0x%08lx | %-10s | %-10s\n", c.name, static_cast<unsigned long>(layer_hr[i]),
                    warp_dev.p ? (w_ok ? "S_OK" : "FAIL") : (c.expected_ok ? "(S_OK)" : "(FAIL)"),
                    hw_dev.p ? (h_ok ? "S_OK" : "FAIL") : "n/a");
        if (l_ok != w_ok) {
            std::fprintf(stderr, "FAIL %s: layer=%s, WARP=%s\n", c.name, l_ok ? "S_OK" : "FAIL", w_ok ? "S_OK" : "FAIL");
            ++failures;
        }
        if (hw_dev.p && c.compare_hw_d3d12 && l_ok != h_ok) {
            std::fprintf(stderr, "FAIL %s: layer=%s, %s=%s\n", c.name, l_ok ? "S_OK" : "FAIL", hw_name.c_str(),
                         h_ok ? "S_OK" : "FAIL");
            ++failures;
        }
    }

    const unsigned validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors) {
        std::fprintf(stderr, "FAIL: %u Vulkan validation errors\n", validation_errors);
        ++failures;
    }
    std::printf("d3d12_predication_atomic_copy_test: %zu cases, %u failures, %u Vulkan validation errors\n",
                cases.size(), failures, validation_errors);
    return failures ? 1 : 0;
}
