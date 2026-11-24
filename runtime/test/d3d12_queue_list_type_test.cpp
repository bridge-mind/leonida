// ExecuteCommandLists runs a list only on a queue of its own type (the debug layer's
// D3D12_MESSAGE_ID_EXECUTECOMMANDLISTS_COMMANDLISTMISMATCH, 836 in d3d12sdklayers.h). Every list type on every queue
// type, then a batch whose second list doesn't match: nothing of it may run, so the first list's copy must not land.
// No game files; exits 77 without a Vulkan device.
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

const char* name(D3D12_COMMAND_LIST_TYPE type) {
    switch (type) {
    case D3D12_COMMAND_LIST_TYPE_DIRECT: return "DIRECT";
    case D3D12_COMMAND_LIST_TYPE_COMPUTE: return "COMPUTE";
    case D3D12_COMMAND_LIST_TYPE_COPY: return "COPY";
    default: return "?";
    }
}

bool wait_idle(ID3D12Device* device, ID3D12CommandQueue* queue) {
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return false;
    return SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr));
}

HRESULT make_buffer(ID3D12Device* device, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state, ID3D12Resource** out) {
    D3D12_HEAP_PROPERTIES props{};
    props.Type = heap;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 256;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                           IID_PPV_ARGS(out));
}
}  // namespace

int main() {
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p)))) return 77;

    const D3D12_COMMAND_LIST_TYPE types[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             D3D12_COMMAND_LIST_TYPE_COPY};
    Ref<ID3D12CommandQueue> queues[3];
    Ref<ID3D12CommandAllocator> allocators[3];
    Ref<ID3D12GraphicsCommandList> lists[3];
    for (int t = 0; t < 3; ++t) {
        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = types[t];
        if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queues[t].p)))) return 77;
        if (FAILED(device->CreateCommandAllocator(types[t], IID_PPV_ARGS(&allocators[t].p))) ||
            FAILED(device->CreateCommandList(0, types[t], allocators[t].p, nullptr, IID_PPV_ARGS(&lists[t].p))) ||
            FAILED(lists[t]->Close())) {
            std::fprintf(stderr, "could not create a %s list\n", name(types[t]));
            return 1;
        }
    }

    // 1. Every list type on every queue type: only the matching pairs run.
    for (int q = 0; q < 3; ++q) {
        for (int l = 0; l < 3; ++l) {
            ID3D12CommandList* batch[] = {lists[l].p};
            queues[q]->ExecuteCommandLists(1, batch);
            const bool ran = rt::d3d12::last_execute_result(queues[q].p) == leonida::ReplayResult::success;
            char what[96];
            std::snprintf(what, sizeof what, "a %s list on a %s queue %s", name(types[l]), name(types[q]),
                          l == q ? "runs" : "is refused");
            check(ran == (l == q), what);
        }
        check(wait_idle(device.p, queues[q].p), "the queue still signals and completes after the refusals");
    }

    // 2. A batch with a mismatched list runs none of its lists: the DIRECT list's copy must not land.
    Ref<ID3D12Resource> upload, readback;
    if (FAILED(make_buffer(device.p, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, &upload.p)) ||
        FAILED(make_buffer(device.p, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, &readback.p)))
        return 1;
    const std::uint32_t marker = 0x4c454f4e;  // "LEON"
    void* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, &mapped))) return 1;
    std::memcpy(mapped, &marker, sizeof marker);
    upload->Unmap(0, nullptr);
    if (FAILED(readback->Map(0, nullptr, &mapped))) return 1;
    std::memset(mapped, 0, 4);
    readback->Unmap(0, nullptr);

    Ref<ID3D12CommandAllocator> copy_allocator;
    Ref<ID3D12GraphicsCommandList> copy_list;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&copy_allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, copy_allocator.p, nullptr,
                                         IID_PPV_ARGS(&copy_list.p))))
        return 1;
    copy_list->CopyBufferRegion(readback.p, 0, upload.p, 0, sizeof marker);
    if (FAILED(copy_list->Close())) return 1;

    auto read_word = [&] {
        std::uint32_t word = 0;
        const D3D12_RANGE range{0, sizeof word};
        if (SUCCEEDED(readback->Map(0, &range, &mapped))) {
            std::memcpy(&word, mapped, sizeof word);
            const D3D12_RANGE none{0, 0};
            readback->Unmap(0, &none);
        }
        return word;
    };

    ID3D12CommandList* mixed[] = {copy_list.p, lists[2].p};  // DIRECT, then COPY, on the DIRECT queue
    queues[0]->ExecuteCommandLists(2, mixed);
    check(rt::d3d12::last_execute_result(queues[0].p) != leonida::ReplayResult::success,
          "a batch with a COPY list on the DIRECT queue is refused");
    check(wait_idle(device.p, queues[0].p) && read_word() == 0, "and its matching DIRECT list did not run either");

    // 3. Null command-list array: count == 0 is a no-op, count > 0 is refused without crashing.
    queues[0]->ExecuteCommandLists(0, nullptr);
    check(rt::d3d12::last_execute_result(queues[0].p) == leonida::ReplayResult::success,
          "ExecuteCommandLists(0, nullptr) succeeds as a no-op");
    queues[0]->ExecuteCommandLists(1, nullptr);
    check(rt::d3d12::last_execute_result(queues[0].p) != leonida::ReplayResult::success,
          "ExecuteCommandLists(1, nullptr) is refused without crashing");
    check(wait_idle(device.p, queues[0].p), "the queue still signals after a null-array refusal");

    // 4. A closed-but-invalid list (Close returned E_INVALIDARG after recording a copy) must not execute.
    Ref<ID3D12CommandAllocator> invalid_allocator;
    Ref<ID3D12GraphicsCommandList> invalid_list;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&invalid_allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, invalid_allocator.p, nullptr,
                                         IID_PPV_ARGS(&invalid_list.p))))
        return 1;
    invalid_list->CopyBufferRegion(readback.p, 0, upload.p, 0, sizeof marker);
    invalid_list->CopyBufferRegion(readback.p, 0, nullptr, 0, sizeof marker);  // invalidates recording
    check(invalid_list->Close() == E_INVALIDARG, "invalid recording causes Close to return E_INVALIDARG");

    ID3D12CommandList* invalid_alone[] = {invalid_list.p};
    queues[0]->ExecuteCommandLists(1, invalid_alone);
    check(rt::d3d12::last_execute_result(queues[0].p) != leonida::ReplayResult::success,
          "a closed-but-invalid list is refused by ExecuteCommandLists");
    check(wait_idle(device.p, queues[0].p) && read_word() == 0,
          "and its earlier CopyBufferRegion did not modify the destination buffer");

    ID3D12CommandList* valid_then_invalid[] = {copy_list.p, invalid_list.p};
    queues[0]->ExecuteCommandLists(2, valid_then_invalid);
    check(rt::d3d12::last_execute_result(queues[0].p) != leonida::ReplayResult::success,
          "a batch containing an invalid list is refused");
    check(wait_idle(device.p, queues[0].p) && read_word() == 0,
          "and the valid list in that batch did not run either");

    ID3D12CommandList* alone[] = {copy_list.p};
    queues[0]->ExecuteCommandLists(1, alone);
    check(rt::d3d12::last_execute_result(queues[0].p) == leonida::ReplayResult::success,
          "the DIRECT list alone runs on the DIRECT queue");
    check(wait_idle(device.p, queues[0].p) && read_word() == marker, "and its copy lands");

    std::printf("d3d12_queue_list_type_test: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
