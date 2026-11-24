// Residency on the Vulkan-backed device, compared with the host PC runtime: MakeResident, Evict,
// SetResidencyPriority (ID3D12Device1) and EnqueueMakeResident (ID3D12Device3). The same script runs on both devices
// with pageable objects of each (a committed buffer, a heap, a placed buffer, a descriptor heap, a query heap and a
// fence): counts of 0, null arrays, null entries, the same object twice, evicting twice, every named priority and a
// few others, residency flags, a null fence, and the fence values EnqueueMakeResident signals (the value the fence
// reaches is recorded too). Every HRESULT is recorded and the two records must be the same, step by step. A step the
// host runtime faults on is reported and not compared. No shaders or game data. Skips (77) without a Vulkan GPU;
// without a host D3D12 device only the layer's own record is printed. Set LEONIDA_D3D49_VERBOSE=1 to print every
// step.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

constexpr std::uint64_t kFaulted = 0xFA017EDull;  // the call raised an exception

struct Objects {
    ID3D12Device1* device1 = nullptr;
    ID3D12Device3* device3 = nullptr;
    ID3D12Device* device = nullptr;
    ID3D12Pageable* all[6] = {};  // buffer, heap, placed, descriptor heap, query heap, fence
    ID3D12Fence* fence = nullptr;
};

using Call = HRESULT (*)(Objects&);

HRESULT guarded(Call call, Objects* o) {
    __try {
        return call(*o);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return HRESULT(kFaulted);
    }
}

HRESULT priority(Objects& o, std::uint32_t value) {
    const auto p = D3D12_RESIDENCY_PRIORITY(value);
    return o.device1->SetResidencyPriority(1, o.all, &p);
}

HRESULT priority1(Objects& o, int which) {
    const D3D12_RESIDENCY_PRIORITY p = D3D12_RESIDENCY_PRIORITY_NORMAL;
    return o.device1->SetResidencyPriority(1, o.all + which, &p);
}

struct Step {
    const char* name;
    Call call;
    UINT64 fence_value = 0;  // EnqueueMakeResident: also record the value the fence reaches (waits for this one)
    bool isolated = false;   // on the host, run in a process of its own (see below)
};

const Step kSteps[] = {
    {"MakeResident 0, null", [](Objects& o) { return o.device->MakeResident(0, nullptr); }},
    {"MakeResident all six", [](Objects& o) { return o.device->MakeResident(6, o.all); }},
    {"MakeResident 0 with an array", [](Objects& o) { return o.device->MakeResident(0, o.all); }},
    {"MakeResident the buffer twice", [](Objects& o) {
         ID3D12Pageable* twice[] = {o.all[0], o.all[0]};
         return o.device->MakeResident(2, twice);
     }},
    {"MakeResident a resident buffer", [](Objects& o) { return o.device->MakeResident(1, o.all); }},
    {"Evict 0, null", [](Objects& o) { return o.device->Evict(0, nullptr); }},
    {"Evict 0 with an array", [](Objects& o) { return o.device->Evict(0, o.all); }},
    {"Evict the buffer", [](Objects& o) { return o.device->Evict(1, o.all); }},
    {"Evict the buffer again", [](Objects& o) { return o.device->Evict(1, o.all); }},
    {"MakeResident the evicted buffer", [](Objects& o) { return o.device->MakeResident(1, o.all); }},
    {"Evict all six", [](Objects& o) { return o.device->Evict(6, o.all); }},
    {"MakeResident all six after Evict", [](Objects& o) { return o.device->MakeResident(6, o.all); }},
    {"Evict the heap and the placed buffer", [](Objects& o) { return o.device->Evict(2, o.all + 1); }},
    {"MakeResident the placed buffer alone", [](Objects& o) { return o.device->MakeResident(1, o.all + 2); }},
    {"MakeResident the heap", [](Objects& o) { return o.device->MakeResident(1, o.all + 1); }},

    {"SetResidencyPriority 0, null, null", [](Objects& o) { return o.device1->SetResidencyPriority(0, nullptr, nullptr); }},
    {"SetResidencyPriority 0 with arrays", [](Objects& o) {
         const D3D12_RESIDENCY_PRIORITY p = D3D12_RESIDENCY_PRIORITY_NORMAL;
         return o.device1->SetResidencyPriority(0, o.all, &p);
     }},
    {"SetResidencyPriority the buffer", [](Objects& o) { return priority1(o, 0); }},
    {"SetResidencyPriority the heap", [](Objects& o) { return priority1(o, 1); }},
    {"SetResidencyPriority MINIMUM", [](Objects& o) { return priority(o, std::uint32_t(D3D12_RESIDENCY_PRIORITY_MINIMUM)); }},
    {"SetResidencyPriority LOW", [](Objects& o) { return priority(o, std::uint32_t(D3D12_RESIDENCY_PRIORITY_LOW)); }},
    {"SetResidencyPriority HIGH", [](Objects& o) { return priority(o, std::uint32_t(D3D12_RESIDENCY_PRIORITY_HIGH)); }},
    {"SetResidencyPriority MAXIMUM", [](Objects& o) { return priority(o, std::uint32_t(D3D12_RESIDENCY_PRIORITY_MAXIMUM)); }},
    {"SetResidencyPriority 0", [](Objects& o) { return priority(o, 0); }},
    {"SetResidencyPriority 1", [](Objects& o) { return priority(o, 1); }},
    {"SetResidencyPriority NORMAL + 0x1234", [](Objects& o) { return priority(o, std::uint32_t(D3D12_RESIDENCY_PRIORITY_NORMAL) + 0x1234); }},
    {"SetResidencyPriority 0x12345678", [](Objects& o) { return priority(o, 0x12345678u); }},
    {"SetResidencyPriority 0x9fffffff", [](Objects& o) { return priority(o, 0x9fffffffu); }},
    {"SetResidencyPriority 0xc8000000", [](Objects& o) { return priority(o, 0xc8000000u); }},
    {"SetResidencyPriority 0xffffffff", [](Objects& o) { return priority(o, 0xffffffffu); }},
    {"SetResidencyPriority an evicted object", [](Objects& o) {
         o.device->Evict(1, o.all);
         const HRESULT hr = priority(o, std::uint32_t(D3D12_RESIDENCY_PRIORITY_HIGH));
         o.device->MakeResident(1, o.all);
         return hr;
     }},

    {"EnqueueMakeResident NONE, two objects, value 7", [](Objects& o) {
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 2, o.all, o.fence, 7);
     }, 7},
    {"EnqueueMakeResident DENY_OVERBUDGET, 0, null, value 9", [](Objects& o) {
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET, 0, nullptr, o.fence, 9);
     }, 9},
    {"EnqueueMakeResident all six, value 10", [](Objects& o) {
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 6, o.all, o.fence, 10);
     }, 10},
    {"EnqueueMakeResident a null fence", [](Objects& o) {
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1, o.all, nullptr, 11);
     }},
    {"EnqueueMakeResident flags 0x2", [](Objects& o) {
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAGS(0x2), 1, o.all, o.fence, 11);
     }},
    {"EnqueueMakeResident flags 0xffffffff", [](Objects& o) {
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAGS(0xffffffffu), 1, o.all, o.fence, 11);
     }},
    {"EnqueueMakeResident 0 with an array, value 12", [](Objects& o) {
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 0, o.all, o.fence, 12);
     }, 12},
    {"EnqueueMakeResident an evicted buffer, value 13", [](Objects& o) {
         o.device->Evict(1, o.all);
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1, o.all, o.fence, 13);
     }, 13},
    {"EnqueueMakeResident the buffer twice, value 14", [](Objects& o) {
         ID3D12Pageable* twice[] = {o.all[0], o.all[0]};
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 2, twice, o.fence, 14);
     }, 14},
    {"EnqueueMakeResident the fence it signals, value 15", [](Objects& o) {
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1, o.all + 5, o.fence, 15);
     }, 15},
    {"EnqueueMakeResident a value below the fence's [backwards], value 3", [](Objects& o) {
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1, o.all, o.fence, 3);
     }, 3},
    {"EnqueueMakeResident the same value again [backwards], value 3", [](Objects& o) {
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1, o.all, o.fence, 3);
     }, 3},

    // Steps after which the host runtime isn't usable any more. On the host each runs in a process of its own with a
    // fresh device (--host-step): SetResidencyPriority on anything but a committed resource or a heap (the host
    // answers E_INVALIDARG and removes its device, DXGI_ERROR_INVALID_CALL), and a count with a null array or a null
    // entry in an array (the host dereferences them).
    {"SetResidencyPriority the placed buffer", [](Objects& o) { return priority1(o, 2); }, 0, true},
    {"SetResidencyPriority the descriptor heap", [](Objects& o) { return priority1(o, 3); }, 0, true},
    {"SetResidencyPriority the query heap", [](Objects& o) { return priority1(o, 4); }, 0, true},
    {"SetResidencyPriority the fence", [](Objects& o) { return priority1(o, 5); }, 0, true},
    {"SetResidencyPriority all six NORMAL", [](Objects& o) {
         const D3D12_RESIDENCY_PRIORITY p[6] = {D3D12_RESIDENCY_PRIORITY_NORMAL, D3D12_RESIDENCY_PRIORITY_NORMAL,
                                                D3D12_RESIDENCY_PRIORITY_NORMAL, D3D12_RESIDENCY_PRIORITY_NORMAL,
                                                D3D12_RESIDENCY_PRIORITY_NORMAL, D3D12_RESIDENCY_PRIORITY_NORMAL};
         return o.device1->SetResidencyPriority(6, o.all, p);
     }, 0, true},
    {"SetResidencyPriority the buffer and the placed buffer", [](Objects& o) {
         ID3D12Pageable* two[] = {o.all[0], o.all[2]};
         const D3D12_RESIDENCY_PRIORITY p[2] = {D3D12_RESIDENCY_PRIORITY_NORMAL, D3D12_RESIDENCY_PRIORITY_NORMAL};
         return o.device1->SetResidencyPriority(2, two, p);
     }, 0, true},
    {"MakeResident 1, null", [](Objects& o) { return o.device->MakeResident(1, nullptr); }, 0, true},
    {"Evict 1, null", [](Objects& o) { return o.device->Evict(1, nullptr); }, 0, true},
    {"SetResidencyPriority 1, objects, null", [](Objects& o) {
         return o.device1->SetResidencyPriority(1, o.all, nullptr);
     }, 0, true},
    {"SetResidencyPriority 1, null, priorities", [](Objects& o) {
         const D3D12_RESIDENCY_PRIORITY p = D3D12_RESIDENCY_PRIORITY_NORMAL;
         return o.device1->SetResidencyPriority(1, nullptr, &p);
     }, 0, true},
    {"EnqueueMakeResident 1, null objects", [](Objects& o) {
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1, nullptr, o.fence, 11);
     }, 0, true},
    {"MakeResident 1, [null]", [](Objects& o) {
         ID3D12Pageable* none[] = {nullptr};
         return o.device->MakeResident(1, none);
     }, 0, true},
    {"Evict 1, [null]", [](Objects& o) {
         ID3D12Pageable* none[] = {nullptr};
         return o.device->Evict(1, none);
     }, 0, true},
    {"SetResidencyPriority 1, [null]", [](Objects& o) {
         ID3D12Pageable* none[] = {nullptr};
         const D3D12_RESIDENCY_PRIORITY p = D3D12_RESIDENCY_PRIORITY_NORMAL;
         return o.device1->SetResidencyPriority(1, none, &p);
     }, 0, true},
    {"EnqueueMakeResident 1, [null]", [](Objects& o) {
         ID3D12Pageable* none[] = {nullptr};
         return o.device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1, none, o.fence, 20);
     }, 0, true},
};

struct Record {
    std::vector<std::string> names;
    std::vector<std::uint64_t> values;
    void add(const std::string& name, std::uint64_t value) {
        names.push_back(name);
        values.push_back(value);
    }
};

// The value `fence` settles at after an EnqueueMakeResident that asked for `want`: the host signals it from a
// paging queue, so wait (up to 2 s) for it to reach `want`, or, for a value below the current one, to change.
UINT64 settled(ID3D12Fence* fence, UINT64 before, UINT64 want) {
    for (int i = 0; i < 200; ++i) {
        const UINT64 now = fence->GetCompletedValue();
        if (now == want || (want < before && now != before)) return now;
        Sleep(10);
    }
    return fence->GetCompletedValue();
}

// The pageable objects of one device, and the references that keep them alive.
struct Holder {
    Ref<ID3D12Device1> device1;
    Ref<ID3D12Device3> device3;
    Ref<ID3D12Resource> buffer, placed;
    Ref<ID3D12Heap> heap;
    Ref<ID3D12DescriptorHeap> descriptors;
    Ref<ID3D12QueryHeap> queries;
    Ref<ID3D12Fence> fence;
    Objects o;
};

bool make(ID3D12Device* device, Holder& h) {
    h.o.device = device;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&h.device1.p))) ||
        FAILED(device->QueryInterface(IID_PPV_ARGS(&h.device3.p))))
        return false;
    h.o.device1 = h.device1.p;
    h.o.device3 = h.device3.p;
    const D3D12_HEAP_PROPERTIES gpu{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 64 * 1024;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_DESC heap_desc{};
    heap_desc.SizeInBytes = 64 * 1024;
    heap_desc.Properties = gpu;
    heap_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    const D3D12_DESCRIPTOR_HEAP_DESC dh{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16,
                                        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    const D3D12_QUERY_HEAP_DESC qh{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 4, 0};
    if (FAILED(device->CreateCommittedResource(&gpu, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&h.buffer.p))) ||
        FAILED(device->CreateHeap(&heap_desc, IID_PPV_ARGS(&h.heap.p))) ||
        FAILED(device->CreatePlacedResource(h.heap.p, 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                            IID_PPV_ARGS(&h.placed.p))) ||
        FAILED(device->CreateDescriptorHeap(&dh, IID_PPV_ARGS(&h.descriptors.p))) ||
        FAILED(device->CreateQueryHeap(&qh, IID_PPV_ARGS(&h.queries.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&h.fence.p))))
        return false;
    h.o.all[0] = h.buffer.p;
    h.o.all[1] = h.heap.p;
    h.o.all[2] = h.placed.p;
    h.o.all[3] = h.descriptors.p;
    h.o.all[4] = h.queries.p;
    h.o.all[5] = h.fence.p;
    h.o.fence = h.fence.p;
    return true;
}

ID3D12Device* create_host() {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                    reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                              : nullptr;
    ID3D12Device* host = nullptr;
    if (!create || FAILED(create(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host)))) return nullptr;
    return host;
}

constexpr std::uint64_t kNoChild = 0xC41DFA11ull;  // the child process couldn't run the step
constexpr int kMaxNotCompared = 11;  // values the host can't give: see the end of main

// Step `index` on a fresh host device, in a child process (this executable with --host-step): its exit code is the
// HRESULT, or a crash status, which counts as a fault.
std::uint64_t isolated(std::size_t index) {
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring command = L"\"" + std::wstring(exe) + L"\" --host-step " + std::to_wstring(index);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr,
                        &startup, &process))
        return kNoChild;
    DWORD code = DWORD(kNoChild);
    if (WaitForSingleObject(process.hProcess, 60000) == WAIT_OBJECT_0) GetExitCodeProcess(process.hProcess, &code);
    else TerminateProcess(process.hProcess, DWORD(kNoChild));
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    // Checked before the crash statuses below, which kNoChild would otherwise fall into: a child that couldn't run
    // its step is a broken test, not a host fault, and fails it.
    if (code == DWORD(kNoChild)) return kNoChild;
    if ((code & 0xF0000000u) == 0xC0000000u) return kFaulted;  // an access violation or fail-fast the guard missed
    return code;
}

// Runs the script on `device` and records it. `host`: run the isolated steps in child processes.
bool run(ID3D12Device* device, bool host, Record& out) {
    Holder h;
    if (!make(device, h)) return false;
    for (std::size_t i = 0; i < sizeof(kSteps) / sizeof(kSteps[0]); ++i) {
        const Step& s = kSteps[i];
        if (s.isolated) {
            out.add(s.name, host ? isolated(i) : std::uint64_t(std::uint32_t(guarded(s.call, &h.o))));
            continue;
        }
        const UINT64 before = h.fence->GetCompletedValue();
        const HRESULT hr = guarded(s.call, &h.o);
        out.add(s.name, std::uint64_t(std::uint32_t(hr)));
        out.add(std::string(s.name) + " (removed reason)", std::uint64_t(std::uint32_t(device->GetDeviceRemovedReason())));
        if (s.fence_value)
            out.add(std::string(s.name) + " (fence)",
                    hr == HRESULT(kFaulted) ? kFaulted : settled(h.fence.p, before, s.fence_value));
    }
    return true;
}

// --host-step N: one isolated step on a fresh host device; the exit code is its HRESULT.
int host_step(std::size_t index) {
    if (index >= sizeof(kSteps) / sizeof(kSteps[0])) return int(kNoChild);
    Ref<ID3D12Device> host;
    host.p = create_host();
    Holder h;
    if (!host.p || !make(host.p, h)) return int(kNoChild);
    return int(guarded(kSteps[index].call, &h.o));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--host-step") == 0) return host_step(std::size_t(std::atoi(argv[2])));
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
        info.buffer_device_address = true;  // heaps live in the GPU address arena
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Record ours;
        EXPECT(run(device.p, false, ours));

        Ref<ID3D12Device> host;
        host.p = create_host();
        if (!host.p) {
            std::printf("residency: no host D3D12 device to compare with; %zu steps recorded, not compared\n",
                        ours.values.size());
        } else {
            Record theirs;
            EXPECT(run(host.p, true, theirs));
            EXPECT(theirs.names == ours.names);
            int mismatches = 0, faulted = 0;
            for (std::size_t i = 0; i < ours.values.size(); ++i) {
                if (theirs.values[i] == kNoChild)
                    std::printf("residency %s: the host step's child process didn't run it\n", ours.names[i].c_str());
                EXPECT(theirs.values[i] != kNoChild);
                if (theirs.values[i] == kFaulted) {
                    std::printf("residency %s: the host faults, not compared (ours 0x%llx)\n", ours.names[i].c_str(),
                                static_cast<unsigned long long>(ours.values[i]));
                    ++faulted;
                    continue;
                }
                // A fence the layer would have to move backwards: Vulkan timelines only go up, so the layer's fence
                // stays where it was (docs/d3d12_vulkan.md); the HRESULT is still compared.
                if (ours.names[i].find("[backwards]") != std::string::npos &&
                    ours.names[i].find("(fence)") != std::string::npos) {
                    std::printf("residency %s: ours 0x%llx host 0x%llx (a fence can't go backwards here; not "
                                "compared)\n", ours.names[i].c_str(), static_cast<unsigned long long>(ours.values[i]),
                                static_cast<unsigned long long>(theirs.values[i]));
                    ++faulted;
                    continue;
                }
                const bool same = ours.values[i] == theirs.values[i];
                if (verbose || !same)
                    std::printf("residency %s: ours 0x%llx host 0x%llx%s\n", ours.names[i].c_str(),
                                static_cast<unsigned long long>(ours.values[i]),
                                static_cast<unsigned long long>(theirs.values[i]), same ? "" : "  MISMATCH");
                mismatches += !same;
            }
            std::printf("residency: %zu values from %zu steps compared with the host (%d more not compared), "
                        "%d mismatches\n", ours.values.size() - faulted, sizeof(kSteps) / sizeof(kSteps[0]), faulted,
                        mismatches);
            EXPECT(mismatches == 0);
            // The values left out are the null-pointer steps the host faults on and the backwards fence. A host that
            // faulted on more would make the comparison vacuous while still passing, so the count is held to that.
            EXPECT(faulted <= kMaxNotCompared);
        }
    }
    std::printf("d3d12_residency_test: OK\n");
    return 0;
}
