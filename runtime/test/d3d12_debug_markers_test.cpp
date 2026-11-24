// D3D-24 on a GPU: SetMarker/BeginEvent/EndEvent on command lists and queues become VK_EXT_debug_utils labels,
// and private data (SetPrivateData, SetPrivateDataInterface, SetName) works on every object of the Vulkan device.
// The device is built behind the Xbox-ABI proxy, as in d3d12_vulkan_device_test, on a vkGetInstanceProcAddr that
// records every label call before forwarding it to the driver, so the test sees exactly what reached Vulkan.
// Without VK_EXT_debug_utils the label part is skipped; the no-extension path (markers are no-ops) and private
// data always run. No shaders or game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation;
// any validation error fails the test.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

// Every label call the layer made, in order: "cmd insert <label>", "cmd begin <label>", "cmd end", "queue ...".
std::vector<std::string> g_calls;
PFN_vkGetInstanceProcAddr g_real = nullptr;
PFN_vkCmdInsertDebugUtilsLabelEXT g_cmd_insert = nullptr;
PFN_vkCmdBeginDebugUtilsLabelEXT g_cmd_begin = nullptr;
PFN_vkCmdEndDebugUtilsLabelEXT g_cmd_end = nullptr;
PFN_vkQueueInsertDebugUtilsLabelEXT g_queue_insert = nullptr;
PFN_vkQueueBeginDebugUtilsLabelEXT g_queue_begin = nullptr;
PFN_vkQueueEndDebugUtilsLabelEXT g_queue_end = nullptr;

std::string label_of(const VkDebugUtilsLabelEXT* label) {
    return label && label->pLabelName ? label->pLabelName : "(null)";
}

VKAPI_ATTR void VKAPI_CALL cmd_insert(VkCommandBuffer cb, const VkDebugUtilsLabelEXT* label) {
    g_calls.push_back("cmd insert " + label_of(label));
    g_cmd_insert(cb, label);
}
VKAPI_ATTR void VKAPI_CALL cmd_begin(VkCommandBuffer cb, const VkDebugUtilsLabelEXT* label) {
    g_calls.push_back("cmd begin " + label_of(label));
    g_cmd_begin(cb, label);
}
VKAPI_ATTR void VKAPI_CALL cmd_end(VkCommandBuffer cb) {
    g_calls.push_back("cmd end");
    g_cmd_end(cb);
}
VKAPI_ATTR void VKAPI_CALL queue_insert(VkQueue queue, const VkDebugUtilsLabelEXT* label) {
    g_calls.push_back("queue insert " + label_of(label));
    g_queue_insert(queue, label);
}
VKAPI_ATTR void VKAPI_CALL queue_begin(VkQueue queue, const VkDebugUtilsLabelEXT* label) {
    g_calls.push_back("queue begin " + label_of(label));
    g_queue_begin(queue, label);
}
VKAPI_ATTR void VKAPI_CALL queue_end(VkQueue queue) {
    g_calls.push_back("queue end");
    g_queue_end(queue);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL recording_proc(VkInstance instance, const char* name) {
    const struct {
        const char* name;
        PFN_vkVoidFunction recorder;
    } table[] = {
        {"vkCmdInsertDebugUtilsLabelEXT", reinterpret_cast<PFN_vkVoidFunction>(cmd_insert)},
        {"vkCmdBeginDebugUtilsLabelEXT", reinterpret_cast<PFN_vkVoidFunction>(cmd_begin)},
        {"vkCmdEndDebugUtilsLabelEXT", reinterpret_cast<PFN_vkVoidFunction>(cmd_end)},
        {"vkQueueInsertDebugUtilsLabelEXT", reinterpret_cast<PFN_vkVoidFunction>(queue_insert)},
        {"vkQueueBeginDebugUtilsLabelEXT", reinterpret_cast<PFN_vkVoidFunction>(queue_begin)},
        {"vkQueueEndDebugUtilsLabelEXT", reinterpret_cast<PFN_vkVoidFunction>(queue_end)},
    };
    for (const auto& entry : table)
        if (std::strcmp(name, entry.name) == 0) return g_real(instance, name) ? entry.recorder : nullptr;
    return g_real(instance, name);
}

// A Vulkan device behind the Xbox-ABI proxy, with or without debug labels.
HRESULT make_device(const vktest::Device& d, bool debug_utils, ID3D12Device** out) {
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = recording_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.debug_utils = debug_utils;
    ID3D12Device* inner = nullptr;
    HRESULT hr = rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner));
    if (FAILED(hr)) return hr;
    hr = rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device), reinterpret_cast<void**>(out));
    if (FAILED(hr)) {
        inner->Release();
        return hr;
    }
    rt::d3d12::set_device_outer(inner, *out);
    return S_OK;
}

// Records events across two lists of one ExecuteCommandLists and around it on the queue, runs them, and waits.
int run_markers(ID3D12Device* device) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> first, second;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&first.p))));
    // One list at a time records into an allocator (D3D-51), and both lists are open at once here.
    Ref<ID3D12CommandAllocator> second_allocator;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    IID_PPV_ARGS(&second_allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, second_allocator.p, nullptr,
                                               IID_PPV_ARGS(&second.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    // Metadata 0: UTF-16, terminated (the size counts the terminator); 1: ANSI, cut at `size`; 2: a PIX3 blob.
    const wchar_t frame[] = L"frame";
    const wchar_t nested[] = L"nested café";
    const char ansi[] = "ansi pass and more";
    const wchar_t spans[] = L"spans two lists";
    const std::uint64_t blob[2] = {0x1234, 0x5678};
    first->SetMarker(0, frame, sizeof(frame));
    first->SetMarker(0, nullptr, 0);
    first->BeginEvent(1, ansi, 9);  // "ansi pass"
    first->BeginEvent(0, nested, sizeof(nested));
    first->EndEvent();
    first->BeginEvent(0, spans, sizeof(spans));  // left open: the next list ends it
    EXPECT(first->Close() == S_OK);
    second->EndEvent();  // "spans two lists"
    second->EndEvent();  // "ansi pass"
    second->EndEvent();  // nothing open: dropped
    second->BeginEvent(2, blob, sizeof(blob));  // left open: ended with the submission
    EXPECT(second->Close() == S_OK);

    const wchar_t queue_frame[] = L"queue frame";
    const char after[] = "after";
    queue->BeginEvent(0, queue_frame, sizeof(queue_frame));
    ID3D12CommandList* lists[] = {first.p, second.p};
    queue->ExecuteCommandLists(2, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    queue->SetMarker(1, after, sizeof(after));
    queue->EndEvent();
    queue->EndEvent();  // nothing open: dropped
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

    // A list replayed on its own balances its own events: run the second list again by itself.
    queue->ExecuteCommandLists(1, lists + 1);
    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 2)));
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(2, nullptr)));
    return 0;
}

// A COM object that only counts its references.
struct Counted final : IUnknown {
    ULONG refs = 1;
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = riid == __uuidof(IUnknown) ? this : nullptr;
        if (!*out) return E_NOINTERFACE;
        AddRef();
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
    ULONG STDMETHODCALLTYPE Release() override { return --refs; }
};

// {d3d24a01-0000-4000-8000-000000000001} and ...02: two private-data keys of the test's own.
constexpr GUID kKey{0xd3d24a01, 0x0000, 0x4000, {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01}};
constexpr GUID kOtherKey{0xd3d24a01, 0x0000, 0x4000, {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02}};
// WKPDID_D3DDebugObjectNameW (d3dcommon.h).
constexpr GUID kDebugObjectNameW{0x4cca5fd8, 0x921f, 0x42c8, {0x85, 0x66, 0x70, 0xca, 0xf2, 0xa9, 0xb7, 0x41}};

int check_private_data(ID3D12Object* object, const char* what) {
    std::printf("private data: %s\n", what);
    Counted counted;
    EXPECT(object->SetPrivateDataInterface(kKey, &counted) == S_OK);
    EXPECT(counted.refs == 2);
    UINT size = 0;
    EXPECT(object->GetPrivateData(kKey, &size, nullptr) == S_OK && size == sizeof(IUnknown*));
    IUnknown* got = nullptr;
    size = 4;  // too small: no reference added, and the size to retry with comes back
    EXPECT(object->GetPrivateData(kKey, &size, &got) == DXGI_ERROR_MORE_DATA && !got && counted.refs == 2);
    EXPECT(size == sizeof(IUnknown*));
    EXPECT(object->GetPrivateData(kKey, &size, &got) == S_OK && got == &counted && counted.refs == 3);
    got->Release();

    // Bytes replace the interface, which is released; an interface replaces bytes.
    const std::uint32_t value = 0xd3d24;
    EXPECT(object->SetPrivateData(kKey, sizeof(value), &value) == S_OK && counted.refs == 1);
    std::uint32_t read = 0;
    size = sizeof(read);
    EXPECT(object->GetPrivateData(kKey, &size, &read) == S_OK && size == sizeof(read) && read == value);
    std::uint16_t too_small = 0;
    size = sizeof(too_small);  // too small for the bytes: nothing copied, the size to retry with comes back
    EXPECT(object->GetPrivateData(kKey, &size, &too_small) == DXGI_ERROR_MORE_DATA && too_small == 0);
    EXPECT(size == sizeof(value));
    read = 0;
    EXPECT(object->GetPrivateData(kKey, &size, &read) == S_OK && size == sizeof(value) && read == value);
    // Null data removes the entry whatever the size: S_OK, then S_FALSE once there is none (the host runtime's
    // answers, d3d12_private_data_test).
    EXPECT(object->SetPrivateData(kKey, 4, nullptr) == S_OK);
    EXPECT(object->GetPrivateData(kKey, &size, nullptr) == DXGI_ERROR_NOT_FOUND && size == 0);
    EXPECT(object->SetPrivateData(kKey, 4, nullptr) == S_FALSE);
    EXPECT(object->SetPrivateDataInterface(kKey, &counted) == S_OK && counted.refs == 2);
    EXPECT(object->SetPrivateDataInterface(kKey, &counted) == S_OK && counted.refs == 2);  // same one again
    // A null interface releases the one held and stays as a pointer-sized entry that reads back as null.
    EXPECT(object->SetPrivateDataInterface(kKey, nullptr) == S_OK && counted.refs == 1);
    size = 99;
    EXPECT(object->GetPrivateData(kKey, &size, nullptr) == S_OK && size == sizeof(IUnknown*));
    got = &counted;
    EXPECT(object->GetPrivateData(kKey, &size, &got) == S_OK && got == nullptr && counted.refs == 1);
    EXPECT(object->SetPrivateData(kKey, 0, nullptr) == S_OK);
    EXPECT(object->GetPrivateData(kKey, &size, nullptr) == DXGI_ERROR_NOT_FOUND && size == 0);

    // SetName is readable as WKPDID_D3DDebugObjectNameW, terminator included, as on the host runtime. A null name
    // (which the host faults on) changes nothing.
    const wchar_t name[] = L"leonida object";
    EXPECT(object->SetName(name) == S_OK);
    wchar_t stored[32] = {};
    size = sizeof(stored);
    EXPECT(object->GetPrivateData(kDebugObjectNameW, &size, stored) == S_OK);
    EXPECT(size == sizeof(name) && std::wcscmp(stored, name) == 0);
    EXPECT(object->SetName(nullptr) == E_INVALIDARG);
    size = sizeof(stored);
    EXPECT(object->GetPrivateData(kDebugObjectNameW, &size, stored) == S_OK && size == sizeof(name));

    // Left set: released when the object dies (checked for a fence below).
    EXPECT(object->SetPrivateDataInterface(kOtherKey, nullptr) == S_OK);
    return 0;
}

int run_private_data(ID3D12Device* device) {
    EXPECT(check_private_data(device, "device") == 0);
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12DescriptorHeap> heap;
    Ref<ID3D12Resource> buffer;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
    D3D12_HEAP_PROPERTIES upload{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 256;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE, &desc,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                     IID_PPV_ARGS(&buffer.p))));
    EXPECT(check_private_data(queue.p, "command queue") == 0);
    EXPECT(check_private_data(allocator.p, "command allocator") == 0);
    EXPECT(check_private_data(list.p, "command list") == 0);
    EXPECT(check_private_data(heap.p, "descriptor heap") == 0);
    EXPECT(check_private_data(buffer.p, "resource") == 0);

    // The held interface outlives nothing: destroying the object releases it.
    Counted counted;
    {
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        EXPECT(check_private_data(fence.p, "fence") == 0);
        EXPECT(fence->SetPrivateDataInterface(kOtherKey, &counted) == S_OK && counted.refs == 2);
    }
    EXPECT(counted.refs == 1);
    return 0;
}

}  // namespace

int main() {
    vktest::Device d;
    d.request_debug_utils = true;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    g_real = d.instance_proc;

    if (d.debug_utils) {
        g_cmd_insert = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
            d.instance_proc(d.instance, "vkCmdInsertDebugUtilsLabelEXT"));
        g_cmd_begin = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
            d.instance_proc(d.instance, "vkCmdBeginDebugUtilsLabelEXT"));
        g_cmd_end = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
            d.instance_proc(d.instance, "vkCmdEndDebugUtilsLabelEXT"));
        g_queue_insert = reinterpret_cast<PFN_vkQueueInsertDebugUtilsLabelEXT>(
            d.instance_proc(d.instance, "vkQueueInsertDebugUtilsLabelEXT"));
        g_queue_begin = reinterpret_cast<PFN_vkQueueBeginDebugUtilsLabelEXT>(
            d.instance_proc(d.instance, "vkQueueBeginDebugUtilsLabelEXT"));
        g_queue_end = reinterpret_cast<PFN_vkQueueEndDebugUtilsLabelEXT>(
            d.instance_proc(d.instance, "vkQueueEndDebugUtilsLabelEXT"));
        EXPECT(g_cmd_insert && g_cmd_begin && g_cmd_end && g_queue_insert && g_queue_begin && g_queue_end);

        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(make_device(d, true, &device.p)));
        EXPECT(run_markers(device.p) == 0);
        const std::vector<std::string> want = {
            "queue begin queue frame",
            "cmd insert frame",
            "cmd insert ",
            "cmd begin ansi pass",
            "cmd begin nested caf\xc3\xa9",
            "cmd end",
            "cmd begin spans two lists",
            "cmd end",
            "cmd end",
            "cmd begin PIX event",
            "cmd end",  // closed with the submission
            "queue insert after",
            "queue end",
            // the second list alone: its two leading ends have nothing open, the PIX event is closed at its end
            "cmd begin PIX event",
            "cmd end",
        };
        for (std::size_t i = 0; i < g_calls.size() || i < want.size(); ++i) {
            const char* got = i < g_calls.size() ? g_calls[i].c_str() : "(none)";
            const char* expected = i < want.size() ? want[i].c_str() : "(none)";
            if (std::strcmp(got, expected) != 0) std::fprintf(stderr, "label call %zu: got \"%s\", want \"%s\"\n", i, got, expected);
        }
        EXPECT(g_calls == want);
        std::printf("debug labels: %zu label calls as expected\n", g_calls.size());
    } else {
        std::printf("debug labels: VK_EXT_debug_utils not available, skipped\n");
    }

    // Without the extension (info.debug_utils false) the same calls reach no label function and still execute.
    g_calls.clear();
    {
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(make_device(d, false, &device.p)));
        EXPECT(run_markers(device.p) == 0);
        EXPECT(g_calls.empty());
        std::printf("debug labels off: markers are no-ops\n");
        EXPECT(run_private_data(device.p) == 0);
    }

    if (d.validation) EXPECT(d.validation_errors() == 0);
    std::printf("d3d12_debug_markers_test: all checks passed\n");
    return 0;
}
