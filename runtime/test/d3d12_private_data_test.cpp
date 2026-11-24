// ID3D12Object private data on the Vulkan-backed device, compared with the host PC runtime: GetPrivateData,
// SetPrivateData, SetPrivateDataInterface and SetName (runtime/shims/d3d12_com_object.h, D3D-24). The same script
// runs on the device and on a fence, a committed buffer, a command allocator and a descriptor heap of each device,
// and records every HRESULT, every size the call writes back, the bytes it reads and the reference count of an
// interface the object holds. The two records must be the same, step by step, and the same as Microsoft's answers
// recorded in d3d12_private_data_answers.h, which is what Wine and Proton check (their host D3D12 is vkd3d). No
// shaders or game data. Skips (77) without a Vulkan GPU. Set LEONIDA_D3D49_VERBOSE=1 to print every step, and
// LEONIDA_D3D49_DUMP=1 to print the host's record in the answers file's format (Windows only).
#include "vk_test_device.h"
#include "host_wine.h"
#include "d3d12_private_data_answers.h"
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
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

// An interface for SetPrivateDataInterface that counts its references, the same object type on both devices.
class Counted final : public IUnknown {
public:
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        if (riid == __uuidof(IUnknown)) {
            *out = this;
            AddRef();
            return S_OK;
        }
        *out = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
    ULONG STDMETHODCALLTYPE Release() override { return --refs; }  // lives on the stack; never deleted
    ULONG refs = 1;
};

// WKPDID_D3DDebugObjectName / ...NameW (d3dcommon.h), spelled out so nothing needs dxguid.lib.
constexpr GUID kName{0x429b8c22, 0x9188, 0x4b0c, {0x87, 0x42, 0xac, 0xb0, 0xbf, 0x85, 0xc2, 0x00}};
constexpr GUID kNameW{0x4cca5fd8, 0x921f, 0x42c8, {0x85, 0x66, 0x70, 0xca, 0xf2, 0xa9, 0xb7, 0x41}};

// A key of its own for each step, so no step sees another's entry.
GUID key(unsigned step) { return GUID{0x5eed0000u + step, 0x1d3d, 0x4c12, {0x9a, 0x41, 0x77, 0x10, 0x2b, 0x6c, 0x00, 0x01}}; }

struct Record {
    std::vector<std::string> names;
    std::vector<std::uint64_t> values;
    void add(const std::string& name, std::uint64_t value) {
        names.push_back(name);
        values.push_back(value);
    }
};

// A call the host may fault on is recorded as this value instead of crashing the test.
constexpr std::uint64_t kFaulted = 0xC0000005;

HRESULT get_guarded(ID3D12Object* o, REFGUID guid, UINT* size, void* data) {
    __try {
        return o->GetPrivateData(guid, size, data);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return HRESULT(kFaulted);
    }
}

HRESULT set_guarded(ID3D12Object* o, REFGUID guid, UINT size, const void* data) {
    __try {
        return o->SetPrivateData(guid, size, data);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return HRESULT(kFaulted);
    }
}

HRESULT name_guarded(ID3D12Object* o, LPCWSTR name) {
    __try {
        return o->SetName(name);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return HRESULT(kFaulted);
    }
}

// GetPrivateData with a buffer of `capacity` bytes (null when `with_buffer` is false), starting the size at
// `capacity`. Records the HRESULT, the size written back and the first 8 bytes of the buffer (filled with 0xCD).
void get(Record& r, const std::string& what, ID3D12Object* o, REFGUID guid, UINT capacity, bool with_buffer = true) {
    std::uint8_t buffer[64];
    std::memset(buffer, 0xCD, sizeof(buffer));
    UINT size = capacity;
    const HRESULT hr = get_guarded(o, guid, &size, with_buffer ? buffer : nullptr);
    r.add(what + ": hr", std::uint32_t(hr));
    r.add(what + ": size", size);
    std::uint64_t head = 0;
    std::memcpy(&head, buffer, sizeof(head));
    if (with_buffer) r.add(what + ": bytes", head);
}

void script(Record& r, ID3D12Object* o) {
    const std::uint8_t eight[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::uint8_t four[4] = {9, 10, 11, 12};
    unsigned step = 0;

    // A key that was never set.
    {
        const GUID g = key(++step);
        get(r, "missing, 16-byte buffer", o, g, 16);
        get(r, "missing, size query", o, g, 16, false);
        get(r, "missing, size 0", o, g, 0);
        r.add("missing, null size: hr", std::uint32_t(get_guarded(o, g, nullptr, nullptr)));
    }
    // Eight bytes, read back with buffers of every size around them.
    {
        const GUID g = key(++step);
        r.add("set 8 bytes: hr", std::uint32_t(set_guarded(o, g, 8, eight)));
        get(r, "8 bytes, size query", o, g, 0, false);
        get(r, "8 bytes, size query from 100", o, g, 100, false);
        get(r, "8 bytes, 4-byte buffer", o, g, 4);
        get(r, "8 bytes, 0-byte buffer", o, g, 0);
        get(r, "8 bytes, 8-byte buffer", o, g, 8);
        get(r, "8 bytes, 64-byte buffer", o, g, 64);
        r.add("8 bytes, null size: hr", std::uint32_t(get_guarded(o, g, nullptr, nullptr)));
        r.add("overwrite with 4 bytes: hr", std::uint32_t(set_guarded(o, g, 4, four)));
        get(r, "after overwrite", o, g, 64);
    }
    // Removal: size 0, null data, or both.
    {
        const GUID g = key(++step);
        set_guarded(o, g, 8, eight);
        r.add("set 8 with null data: hr", std::uint32_t(set_guarded(o, g, 8, nullptr)));
        get(r, "after 8 with null data", o, g, 64);
    }
    {
        const GUID g = key(++step);
        set_guarded(o, g, 8, eight);
        r.add("set 0 bytes with data: hr", std::uint32_t(set_guarded(o, g, 0, eight)));
        get(r, "after 0 bytes with data", o, g, 64);
    }
    {
        const GUID g = key(++step);
        set_guarded(o, g, 8, eight);
        r.add("set 0 bytes, null data: hr", std::uint32_t(set_guarded(o, g, 0, nullptr)));
        get(r, "after 0 bytes, null data", o, g, 64);
    }
    {
        const GUID g = key(++step);
        r.add("set 8 with null data on a missing key: hr", std::uint32_t(set_guarded(o, g, 8, nullptr)));
        r.add("set 0 bytes on a missing key: hr", std::uint32_t(set_guarded(o, g, 0, nullptr)));
        get(r, "missing key after removals", o, g, 64);
    }
    {
        const GUID g = key(++step);
        r.add("set 0 bytes with data on a missing key: hr", std::uint32_t(set_guarded(o, g, 0, eight)));
        get(r, "after 0 bytes with data on a missing key", o, g, 64);
        get(r, "after 0 bytes with data on a missing key, size query", o, g, 64, false);
        r.add("then 0 bytes, null data: hr", std::uint32_t(set_guarded(o, g, 0, nullptr)));
        get(r, "after removing the empty entry", o, g, 64);
    }
    {
        const GUID g = key(++step);
        r.add("null interface on a missing key: hr", std::uint32_t(o->SetPrivateDataInterface(g, nullptr)));
        get(r, "after a null interface, 4-byte buffer", o, g, 4);
        get(r, "after a null interface, 64-byte buffer", o, g, 64);
        r.add("null interface again: hr", std::uint32_t(o->SetPrivateDataInterface(g, nullptr)));
        r.add("remove the null interface: hr", std::uint32_t(set_guarded(o, g, 0, nullptr)));
        get(r, "after removing the null interface", o, g, 64);
    }
    // An interface entry: the object's references to it, and what reading it gives.
    {
        const GUID g = key(++step);
        Counted payload;
        r.add("interface: set hr", std::uint32_t(o->SetPrivateDataInterface(g, &payload)));
        r.add("interface: refs after set", payload.refs);
        get(r, "interface, size query", o, g, 0, false);
        r.add("interface, size query: refs", payload.refs);
        get(r, "interface, 4-byte buffer", o, g, 4);
        r.add("interface, 4-byte buffer: refs", payload.refs);
        {
            IUnknown* read = nullptr;
            UINT size = sizeof(read);
            const HRESULT hr = get_guarded(o, g, &size, &read);
            r.add("interface, pointer-size buffer: hr", std::uint32_t(hr));
            r.add("interface, pointer-size buffer: size", size);
            r.add("interface, pointer-size buffer: same object", read == &payload);
            r.add("interface, pointer-size buffer: refs", payload.refs);
            if (read == &payload) read->Release();
        }
        r.add("interface: set again hr", std::uint32_t(o->SetPrivateDataInterface(g, &payload)));
        r.add("interface: refs after setting again", payload.refs);
        r.add("interface: bytes over it hr", std::uint32_t(set_guarded(o, g, 4, four)));
        r.add("interface: refs after bytes over it", payload.refs);
        get(r, "interface, then bytes", o, g, 64);
        r.add("interface: set over bytes hr", std::uint32_t(o->SetPrivateDataInterface(g, &payload)));
        r.add("interface: refs after set over bytes", payload.refs);
        r.add("interface: null hr", std::uint32_t(o->SetPrivateDataInterface(g, nullptr)));
        r.add("interface: refs after null", payload.refs);
        get(r, "interface, after null", o, g, 64);
        // Removed with SetPrivateData(0, null) instead.
        o->SetPrivateDataInterface(g, &payload);
        r.add("interface: removed with bytes hr", std::uint32_t(set_guarded(o, g, 0, nullptr)));
        r.add("interface: refs after removal with bytes", payload.refs);
        get(r, "interface, after removal with bytes", o, g, 64);
        o->SetPrivateDataInterface(g, nullptr);
    }
    // SetName: where the name goes, and what an empty or null name does.
    {
        r.add("SetName abc: hr", std::uint32_t(name_guarded(o, L"abc")));
        get(r, "name W after abc", o, kNameW, 64);
        get(r, "name A after abc", o, kName, 64);
        r.add("SetName empty: hr", std::uint32_t(name_guarded(o, L"")));
        get(r, "name W after empty", o, kNameW, 64);
        get(r, "name A after empty", o, kName, 64);
        r.add("SetName null: hr", std::uint32_t(name_guarded(o, nullptr)));
        get(r, "name W after null", o, kNameW, 64);
        get(r, "name A after null", o, kName, 64);
        // A name set through the narrow key, and SetName after it.
        const char narrow[] = "xyz";
        r.add("set narrow name: hr", std::uint32_t(set_guarded(o, kName, sizeof(narrow), narrow)));
        get(r, "name W after narrow", o, kNameW, 64);
        r.add("SetName def after narrow: hr", std::uint32_t(name_guarded(o, L"def")));
        get(r, "name A after def", o, kName, 64);
        get(r, "name W after def", o, kNameW, 64);
        set_guarded(o, kName, 0, nullptr);
        name_guarded(o, nullptr);
    }
}

// The objects the script runs on, made on `device`: the device itself and four children.
struct Objects {
    Ref<ID3D12Fence> fence;
    Ref<ID3D12Resource> buffer;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12DescriptorHeap> heap;
};

bool make(ID3D12Device* device, Objects& o) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 256;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const D3D12_HEAP_PROPERTIES upload{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4};
    return SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&o.fence.p))) &&
           SUCCEEDED(device->CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE, &desc,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                     IID_PPV_ARGS(&o.buffer.p))) &&
           SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&o.allocator.p))) &&
           SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&o.heap.p)));
}

// The records of every object of `device`, in a fixed order, each name prefixed with the object's kind.
bool record(ID3D12Device* device, Record& r) {
    Objects o;
    if (!make(device, o)) return false;
    const struct {
        const char* kind;
        ID3D12Object* object;
    } objects[] = {{"device", device}, {"fence", o.fence.p}, {"buffer", o.buffer.p},
                   {"allocator", o.allocator.p}, {"descriptor heap", o.heap.p}};
    for (const auto& entry : objects) {
        Record one;
        script(one, entry.object);
        for (std::size_t i = 0; i < one.names.size(); ++i) r.add(std::string(entry.kind) + ": " + one.names[i], one.values[i]);
    }
    // An object that dies holding an interface releases it.
    Counted payload;
    {
        Ref<ID3D12Fence> fence;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return false;
        fence->SetPrivateDataInterface(key(100), &payload);
        r.add("dying fence: refs while held", payload.refs);
    }
    r.add("dying fence: refs after release", payload.refs);
    return true;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    const bool verbose = GetEnvironmentVariableA("LEONIDA_D3D49_VERBOSE", nullptr, 0) > 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Record ours;
        EXPECT(record(device.p, ours));

        const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        const auto create_host = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                             reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                                       : nullptr;
        Ref<ID3D12Device> host;
        Record theirs;
        if (create_host && !hostwine::host_is_wine() &&
            SUCCEEDED(create_host(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p))))
            EXPECT(record(host.p, theirs));
        if (GetEnvironmentVariableA("LEONIDA_D3D49_DUMP", nullptr, 0) > 0) {
            if (!host.p) {
                std::fprintf(stderr, "LEONIDA_D3D49_DUMP prints Microsoft's runtime's record: run it on Windows\n");
                return 1;
            }
            for (std::size_t i = 0; i < theirs.values.size(); ++i)
                std::printf("    {\"%s\", 0x%llxull},\n", theirs.names[i].c_str(),
                            static_cast<unsigned long long>(theirs.values[i]));
            return 0;
        }
        Record recorded;
        for (const PrivateDataAnswer& a : kPrivateDataAnswers) recorded.add(a.name, a.value);
        const auto compare = [&](const char* against, const Record& ref) {
            if (ref.names != ours.names) {
                std::fprintf(stderr, "private data: the steps differ from %s's\n", against);
                return 1;
            }
            int mismatches = 0, faulted = 0;
            for (std::size_t i = 0; i < ours.values.size(); ++i) {
                if (ref.values[i] == kFaulted) {  // SetName(nullptr): the host faults; the layer clears the name
                    if (verbose)
                        std::printf("private data %s: the host faults, not compared (ours 0x%llx)\n",
                                    ours.names[i].c_str(), static_cast<unsigned long long>(ours.values[i]));
                    ++faulted;
                    continue;
                }
                const bool same = ours.values[i] == ref.values[i];
                if (verbose || !same)
                    std::printf("private data %s: ours 0x%llx, %s 0x%llx%s\n", ours.names[i].c_str(),
                                static_cast<unsigned long long>(ours.values[i]), against,
                                static_cast<unsigned long long>(ref.values[i]), same ? "" : "  MISMATCH");
                mismatches += !same;
            }
            std::printf("private data: %zu steps compared with %s (%d more the host faults on), %d mismatches\n",
                        ours.values.size() - faulted, against, faulted, mismatches);
            return mismatches;
        };
        if (host.p) EXPECT(compare("the host", theirs) == 0);
        EXPECT(compare("Microsoft's recorded answers", recorded) == 0);
    }
    std::printf("d3d12_private_data_test: OK\n");
    return 0;
}
