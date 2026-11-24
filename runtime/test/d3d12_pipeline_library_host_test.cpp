// CreatePipelineLibrary and ID3D12PipelineLibrary / ID3D12PipelineLibrary1 on the Vulkan-backed device, compared
// with the host PC runtime, for every call that needs no pipeline state object: null, empty, foreign, truncated and
// corrupt blobs, the other device's serialized blob, the null-output capability check and other interfaces, and on
// an empty library StorePipeline / LoadComputePipeline / LoadGraphicsPipeline / LoadPipeline with null, empty and
// unknown names and null PSOs, descriptions or outputs, and Serialize with null, short and long buffers. Every HRESULT
// is recorded, with whether the output pointer was written; the two records must be the same, step by step. The
// serialized size and bytes are each device's own format and are not compared. A step the host runtime faults on is
// reported and not compared. No shaders or game data. Skips (77) without a Vulkan GPU; without a host D3D12 device
// only the layer's own record is printed. Set LEONIDA_D3D49_VERBOSE=1 to print every step.
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

struct Context {
    ID3D12Device1* device = nullptr;
    ID3D12PipelineLibrary1* library = nullptr;  // an empty library of this device
    std::vector<std::uint8_t> own;              // this device's serialized empty library
    std::vector<std::uint8_t> other;            // the other device's, or empty without one
    void* out = nullptr;                        // the step's output pointer, released after it
    std::vector<std::uint8_t> buffer;
};

using Call = HRESULT (*)(Context&);

HRESULT guarded(Call call, Context* c) {
    __try {
        return call(*c);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return HRESULT(kFaulted);
    }
}

HRESULT create(Context& c, const void* blob, SIZE_T size, REFIID riid = __uuidof(ID3D12PipelineLibrary)) {
    return c.device->CreatePipelineLibrary(blob, size, riid, &c.out);
}

const std::vector<std::uint8_t> kForeignBytes(64, 0xAB);
const std::uint8_t* const kForeign = kForeignBytes.data();
const std::uint8_t kZeros[64] = {};

D3D12_COMPUTE_PIPELINE_STATE_DESC compute_desc() { return {}; }
D3D12_GRAPHICS_PIPELINE_STATE_DESC graphics_desc() { return {}; }

struct Step {
    const char* name;
    Call call;
};

const Step kSteps[] = {
    // CreatePipelineLibrary.
    {"create: null blob, size 0", [](Context& c) { return create(c, nullptr, 0); }},
    {"create: null blob, size 0, null output", [](Context& c) {
         return c.device->CreatePipelineLibrary(nullptr, 0, __uuidof(ID3D12PipelineLibrary), nullptr);
     }},
    {"create: null blob, size 16", [](Context& c) { return create(c, nullptr, 16); }},
    {"create: null blob, size 16, null output", [](Context& c) {
         return c.device->CreatePipelineLibrary(nullptr, 16, __uuidof(ID3D12PipelineLibrary), nullptr);
     }},
    {"create: a blob, size 0", [](Context& c) { return create(c, kForeign, 0); }},
    {"create: 64 foreign bytes", [](Context& c) { return create(c, kForeign, kForeignBytes.size()); }},
    {"create: 64 zero bytes", [](Context& c) { return create(c, kZeros, sizeof(kZeros)); }},
    {"create: 4 foreign bytes", [](Context& c) { return create(c, kForeign, 4); }},
    {"create: 1 zero byte", [](Context& c) { return create(c, kZeros, 1); }},
    {"create: 64 foreign bytes, null output", [](Context& c) {
         return c.device->CreatePipelineLibrary(kForeign, kForeignBytes.size(), __uuidof(ID3D12PipelineLibrary), nullptr);
     }},
    {"create: as ID3D12PipelineLibrary1", [](Context& c) {
         return create(c, nullptr, 0, __uuidof(ID3D12PipelineLibrary1));
     }},
    {"create: as IUnknown", [](Context& c) { return create(c, nullptr, 0, __uuidof(IUnknown)); }},
    {"create: as ID3D12Pageable", [](Context& c) { return create(c, nullptr, 0, __uuidof(ID3D12Pageable)); }},
    {"create: as ID3D12Fence", [](Context& c) { return create(c, nullptr, 0, __uuidof(ID3D12Fence)); }},
    {"create: as ID3D12Fence, a foreign blob", [](Context& c) {
         return create(c, kForeign, kForeignBytes.size(), __uuidof(ID3D12Fence));
     }},
    {"create: its own serialized empty library", [](Context& c) { return create(c, c.own.data(), c.own.size()); }},
    {"create: its own blob, one byte short", [](Context& c) { return create(c, c.own.data(), c.own.size() - 1); }},
    {"create: its own blob, 64 bytes longer", [](Context& c) {
         std::vector<std::uint8_t> longer(c.own);
         longer.resize(longer.size() + 64, 0);
         return create(c, longer.data(), longer.size());
     }},
    {"create: its own blob, last byte flipped", [](Context& c) {
         std::vector<std::uint8_t> flipped(c.own);
         flipped.back() ^= 0xFF;
         return create(c, flipped.data(), flipped.size());
     }},
    {"create: its own blob, first byte flipped", [](Context& c) {
         std::vector<std::uint8_t> flipped(c.own);
         flipped.front() ^= 0xFF;
         return create(c, flipped.data(), flipped.size());
     }},
    {"create: the other device's serialized empty library", [](Context& c) {
         return c.other.empty() ? HRESULT(S_OK) : create(c, c.other.data(), c.other.size());
     }},

    // Serialize on an empty library (the size itself is the device's own; see run()).
    {"Serialize null, its size", [](Context& c) { return c.library->Serialize(nullptr, c.own.size()); }},
    {"Serialize null, 0", [](Context& c) { return c.library->Serialize(nullptr, 0); }},
    {"Serialize a buffer, 0", [](Context& c) {
         c.buffer.assign(c.own.size() + 64, 0xCD);
         return c.library->Serialize(c.buffer.data(), 0);
     }},
    {"Serialize one byte short", [](Context& c) {
         c.buffer.assign(c.own.size() + 64, 0xCD);
         return c.library->Serialize(c.buffer.data(), c.own.size() - 1);
     }},
    {"Serialize its size", [](Context& c) {
         c.buffer.assign(c.own.size() + 64, 0xCD);
         return c.library->Serialize(c.buffer.data(), c.own.size());
     }},
    {"Serialize 64 bytes longer", [](Context& c) {
         c.buffer.assign(c.own.size() + 64, 0xCD);
         return c.library->Serialize(c.buffer.data(), c.own.size() + 64);
     }},

    // StorePipeline without a PSO.
    {"StorePipeline null, null", [](Context& c) { return c.library->StorePipeline(nullptr, nullptr); }},
    {"StorePipeline empty name, null", [](Context& c) { return c.library->StorePipeline(L"", nullptr); }},
    {"StorePipeline a name, null", [](Context& c) { return c.library->StorePipeline(L"pso", nullptr); }},

    // Loads from an empty library.
    {"LoadComputePipeline unknown name", [](Context& c) {
         const auto d = compute_desc();
         return c.library->LoadComputePipeline(L"missing", &d, __uuidof(ID3D12PipelineState), &c.out);
     }},
    {"LoadComputePipeline null name", [](Context& c) {
         const auto d = compute_desc();
         return c.library->LoadComputePipeline(nullptr, &d, __uuidof(ID3D12PipelineState), &c.out);
     }},
    {"LoadComputePipeline empty name", [](Context& c) {
         const auto d = compute_desc();
         return c.library->LoadComputePipeline(L"", &d, __uuidof(ID3D12PipelineState), &c.out);
     }},
    {"LoadComputePipeline null description", [](Context& c) {
         return c.library->LoadComputePipeline(L"missing", nullptr, __uuidof(ID3D12PipelineState), &c.out);
     }},
    {"LoadComputePipeline null output", [](Context& c) {
         const auto d = compute_desc();
         return c.library->LoadComputePipeline(L"missing", &d, __uuidof(ID3D12PipelineState), nullptr);
     }},
    {"LoadComputePipeline as ID3D12Fence", [](Context& c) {
         const auto d = compute_desc();
         return c.library->LoadComputePipeline(L"missing", &d, __uuidof(ID3D12Fence), &c.out);
     }},
    {"LoadGraphicsPipeline unknown name", [](Context& c) {
         const auto d = graphics_desc();
         return c.library->LoadGraphicsPipeline(L"missing", &d, __uuidof(ID3D12PipelineState), &c.out);
     }},
    {"LoadGraphicsPipeline null name", [](Context& c) {
         const auto d = graphics_desc();
         return c.library->LoadGraphicsPipeline(nullptr, &d, __uuidof(ID3D12PipelineState), &c.out);
     }},
    {"LoadGraphicsPipeline empty name", [](Context& c) {
         const auto d = graphics_desc();
         return c.library->LoadGraphicsPipeline(L"", &d, __uuidof(ID3D12PipelineState), &c.out);
     }},
    {"LoadGraphicsPipeline null description", [](Context& c) {
         return c.library->LoadGraphicsPipeline(L"missing", nullptr, __uuidof(ID3D12PipelineState), &c.out);
     }},
    {"LoadGraphicsPipeline null output", [](Context& c) {
         const auto d = graphics_desc();
         return c.library->LoadGraphicsPipeline(L"missing", &d, __uuidof(ID3D12PipelineState), nullptr);
     }},
    {"LoadPipeline unknown name, empty stream", [](Context& c) {
         const D3D12_PIPELINE_STATE_STREAM_DESC d{};
         return c.library->LoadPipeline(L"missing", &d, __uuidof(ID3D12PipelineState), &c.out);
     }},
    {"LoadPipeline unknown name, a stream", [](Context& c) {
         struct alignas(void*) {
             D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK;
             UINT mask = 0;
         } stream;
         const D3D12_PIPELINE_STATE_STREAM_DESC d{sizeof(stream), &stream};
         return c.library->LoadPipeline(L"missing", &d, __uuidof(ID3D12PipelineState), &c.out);
     }},
    {"LoadPipeline null name", [](Context& c) {
         const D3D12_PIPELINE_STATE_STREAM_DESC d{};
         return c.library->LoadPipeline(nullptr, &d, __uuidof(ID3D12PipelineState), &c.out);
     }},
    {"LoadPipeline null description", [](Context& c) {
         return c.library->LoadPipeline(L"missing", nullptr, __uuidof(ID3D12PipelineState), &c.out);
     }},
    {"LoadPipeline null output", [](Context& c) {
         const D3D12_PIPELINE_STATE_STREAM_DESC d{};
         return c.library->LoadPipeline(L"missing", &d, __uuidof(ID3D12PipelineState), nullptr);
     }},
};

struct Record {
    std::vector<std::string> names;
    std::vector<std::uint64_t> values;
    void add(const std::string& name, std::uint64_t value) {
        names.push_back(name);
        values.push_back(value);
    }
};

// An empty library of `device`, and its serialized bytes.
bool empty_library(ID3D12Device1* device, Ref<ID3D12PipelineLibrary1>& library, std::vector<std::uint8_t>& blob) {
    if (FAILED(device->CreatePipelineLibrary(nullptr, 0, IID_PPV_ARGS(&library.p))) || !library.p) return false;
    blob.assign(library->GetSerializedSize(), 0);
    return !blob.empty() && SUCCEEDED(library->Serialize(blob.data(), blob.size()));
}

// One step on `device`'s empty library: its HRESULT, and the output pointer's state (0 nulled, 1 an object, 2 left
// alone).
void one_step(Context& c, const Step& s, std::uint64_t& hr_out, std::uint64_t& output) {
    c.out = reinterpret_cast<void*>(std::uintptr_t(0x5EED));
    const HRESULT hr = guarded(s.call, &c);
    hr_out = std::uint64_t(std::uint32_t(hr));
    const bool written = c.out != reinterpret_cast<void*>(std::uintptr_t(0x5EED));
    output = hr == HRESULT(kFaulted) ? kFaulted : !written ? 2 : c.out ? 1 : 0;
    if (written && c.out && hr != HRESULT(kFaulted)) static_cast<IUnknown*>(c.out)->Release();
}

std::string hex(const std::vector<std::uint8_t>& bytes) {
    static const char digits[] = "0123456789abcdef";
    std::string text;
    for (const std::uint8_t b : bytes) {
        text += digits[b >> 4];
        text += digits[b & 15];
    }
    return text.empty() ? "-" : text;
}

std::vector<std::uint8_t> unhex(const char* text) {
    std::vector<std::uint8_t> bytes;
    for (std::size_t i = 0; text[i] && text[i + 1] && text[0] != '-'; i += 2) {
        const auto nibble = [](char ch) { return std::uint8_t(ch <= '9' ? ch - '0' : ch - 'a' + 10); };
        bytes.push_back(std::uint8_t(nibble(text[i]) << 4 | nibble(text[i + 1])));
    }
    return bytes;
}

ID3D12Device1* create_host() {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                    reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                              : nullptr;
    ID3D12Device1* host = nullptr;
    if (!create || FAILED(create(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host)))) return nullptr;
    return host;
}

// Step `index` on the host, in a child process (this executable with --host-step) with a fresh device: a call the
// host runtime faults on (Serialize(nullptr, n)) leaves it broken, and in-process the next call failed fast, which no
// exception handler survives.
void isolated(std::size_t index, const std::vector<std::uint8_t>& other, std::uint64_t& hr, std::uint64_t& output) {
    hr = output = kFaulted;
    char exe[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, exe, MAX_PATH);
    const std::string command = "\"\"" + std::string(exe) + "\" --host-step " + std::to_string(index) + " " +
                                hex(other) + "\"";
    FILE* child = _popen(command.c_str(), "r");
    if (!child) return;
    unsigned long long a = 0, b = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), child))
        if (sscanf_s(line, "result %llx %llx", &a, &b) == 2) {
            hr = a;
            output = b;
        }
    _pclose(child);
}

// Records the script on `device` (in-process), or on the host with every step in a child process.
bool run(ID3D12Device1* device, bool host, const std::vector<std::uint8_t>& other, Record& out) {
    Context c;
    c.device = device;
    c.other = other;
    Ref<ID3D12PipelineLibrary1> library;
    if (!empty_library(device, library, c.own)) return false;
    c.library = library.p;
    // The serialized size is each device's own format: recorded, not compared.
    out.add("GetSerializedSize of an empty library [own format]", c.own.size());
    const bool verbose = GetEnvironmentVariableA("LEONIDA_D3D49_VERBOSE", nullptr, 0) > 0;
    for (std::size_t i = 0; i < sizeof(kSteps) / sizeof(kSteps[0]); ++i) {
        const Step& s = kSteps[i];
        if (verbose) {  // flushed first: a runtime that dies on a step leaves its name behind
            std::printf("pipeline library step (%s): %s\n", host ? "host" : "ours", s.name);
            std::fflush(stdout);
        }
        std::uint64_t hr = 0, output = 0;
        if (host) isolated(i, other, hr, output);
        else one_step(c, s, hr, output);
        out.add(s.name, hr);
        out.add(std::string(s.name) + " (output)", output);
    }
    out.add("GetSerializedSize after the steps [own format]", library->GetSerializedSize());
    return true;
}

// --host-step N <other blob in hex>: one step on a fresh host device; prints "result <hr> <output>".
int host_step(std::size_t index, const char* other_hex) {
    if (index >= sizeof(kSteps) / sizeof(kSteps[0])) return 2;
    Ref<ID3D12Device1> host;
    host.p = create_host();
    if (!host.p) return 2;
    Context c;
    c.device = host.p;
    c.other = unhex(other_hex);
    Ref<ID3D12PipelineLibrary1> library;
    if (!empty_library(host.p, library, c.own)) return 2;
    c.library = library.p;
    std::uint64_t hr = 0, output = 0;
    one_step(c, kSteps[index], hr, output);
    std::printf("result %llx %llx\n", static_cast<unsigned long long>(hr), static_cast<unsigned long long>(output));
    std::fflush(stdout);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[1], "--host-step") == 0)
        return host_step(std::size_t(std::strtoul(argv[2], nullptr, 10)), argv[3]);
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
        Ref<ID3D12Device1> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

        Ref<ID3D12Device1> host;
        host.p = create_host();

        // Each device gets the other's serialized empty library as a foreign blob.
        std::vector<std::uint8_t> ours_blob, host_blob;
        {
            Ref<ID3D12PipelineLibrary1> library;
            EXPECT(empty_library(device.p, library, ours_blob));
        }
        if (host.p) {
            Ref<ID3D12PipelineLibrary1> library;
            if (!empty_library(host.p, library, host_blob)) host.reset();  // no pipeline libraries on this host
        }
        Record ours;
        EXPECT(run(device.p, false, host_blob, ours));
        if (!host.p) {
            std::printf("pipeline library: no host D3D12 device with pipeline libraries; %zu steps recorded, not "
                        "compared\n", ours.values.size());
        } else {
            Record theirs;
            EXPECT(run(host.p, true, ours_blob, theirs));
            EXPECT(theirs.names == ours.names);
            int mismatches = 0, skipped = 0;
            for (std::size_t i = 0; i < ours.values.size(); ++i) {
                const bool own_format = ours.names[i].find("[own format]") != std::string::npos;
                if (theirs.values[i] == kFaulted || own_format) {
                    std::printf("pipeline library %s: ours 0x%llx host 0x%llx (%s; not compared)\n",
                                ours.names[i].c_str(), static_cast<unsigned long long>(ours.values[i]),
                                static_cast<unsigned long long>(theirs.values[i]),
                                own_format ? "each device's own format" : "the host faults");
                    ++skipped;
                    continue;
                }
                const bool same = ours.values[i] == theirs.values[i];
                if (verbose || !same)
                    std::printf("pipeline library %s: ours 0x%llx host 0x%llx%s\n", ours.names[i].c_str(),
                                static_cast<unsigned long long>(ours.values[i]),
                                static_cast<unsigned long long>(theirs.values[i]), same ? "" : "  MISMATCH");
                mismatches += !same;
            }
            std::printf("pipeline library: %zu steps compared with the host (%d more not compared), %d mismatches\n",
                        ours.values.size() - skipped, skipped, mismatches);
            EXPECT(mismatches == 0);
        }
    }
    std::printf("d3d12_pipeline_library_host_test: OK\n");
    return 0;
}
