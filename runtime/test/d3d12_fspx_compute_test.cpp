// Compute PSOs from the FidelityFX DLL's FSPX programs (docs/fspx.md, "Run-time lookup"), with synthetic inputs:
//   - device slot 62 (the Xbox shape) with a DXBC container whose ACCC part is an FSPX payload, what the DLL passes
//     when slot 67 fails;
//   - CreateComputePipelineState with a bare kind-6 program, what the slot-67 description carries at +0x28 and what
//     a proxy mapping of slot 67 forwards;
//   - both found through make_game.py's <shaders>/fspx/index.json by fspx_code_sha256, the SHA-256 of the kind-6
//     bytes; each PSO is dispatched once and its shader reads its code base, which must be the kind-6 program;
//   - a malformed directory, a program without its translation and a container without a usable ACCC part are
//     refused with E_INVALIDARG, as bytecode without a program is today.
// The shader (test/glsl/cs_user_data.comp) stands in for a translated one. No game data. Set LEONIDA_VK_VALIDATION=1
// to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"
#include "../shims/d3d12_xs_device.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#ifndef LEONIDA_TEST_CS_SPV
#error "LEONIDA_TEST_CS_SPV must name the compiled test shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX
using XboxComputePipelineFn = HRESULT(WINAPI*)(void* device, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, std::uint64_t,
                                               std::uint64_t, const IID*, void**);
// Device slot 67 as the FidelityFX DLL calls it: (desc, riid, out), the description as docs/fspx.md lays it out.
struct XboxNativeDesc {
    ID3D12RootSignature* root_signature;
    const void* kind0;
    std::uint64_t kind0_size;
    const void* kind1;
    std::uint64_t kind1_size;
    const void* kind6;
    std::uint64_t kind6_size;
    const void* kind7;
    std::uint64_t kind7_size;
};
using XboxNativeComputePipelineFn = HRESULT(WINAPI*)(void* device, const XboxNativeDesc*, const IID*, void**);
using SerializeFn =HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);

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

void* slot(void* object, unsigned index) { return (*reinterpret_cast<void***>(object))[index]; }

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

// A kind-6 program: the 36-byte prologue (three VOP3 words with opcode 0x180, bits 31..16 = 0xD580, and literals),
// a filler word and s_endpgm. Dword 1, which the test shader reads through its code base, is `marker`.
std::vector<std::uint8_t> kind6(std::uint32_t marker) {
    std::vector<std::uint8_t> out;
    for (std::uint32_t i = 0; i < 3; ++i) {
        put32(out, 0xd5800000u | i);
        put32(out, i == 0 ? marker : 0xffu);
        put32(out, 0x1000u + i);
    }
    put32(out, 0x7e000280u);
    put32(out, 0xbf810000u);
    return out;
}

// An FSPX payload with kinds 0, 1, 6 and 7 after a four-entry directory; `code_offset` and `code_size` override the
// kind-6 entry to make a malformed directory.
std::vector<std::uint8_t> payload(const std::vector<std::uint8_t>& code, std::uint32_t code_offset = 128,
                                  std::int64_t code_size = -1, std::uint8_t state_fill = 0x11) {
    std::vector<std::uint8_t> out = {'F', 'S', 'P', 'X'};
    const auto n = std::uint32_t(code.size());
    put32(out, 1);
    put32(out, 4);
    put32(out, 128 + n + 8);
    const std::uint32_t size6 = code_size < 0 ? n : std::uint32_t(code_size);
    const std::array<std::array<std::uint32_t, 3>, 4> directory = {
        {{0, 80, 28}, {1, 108, 20}, {6, code_offset, size6}, {7, 128 + n, 8}}};
    for (const auto& entry : directory) {
        put32(out, entry[0]);
        put32(out, entry[1]);
        put32(out, entry[2]);
        put32(out, 0);
    }
    out.insert(out.end(), 28, state_fill);  // kind 0
    out.insert(out.end(), 20, 0x22);
    out.insert(out.end(), code.begin(), code.end());
    out.insert(out.end(), 8, 0x77);
    return out;
}

// DXBC { HASH, ACCC = payload }, the parts that matter of the DLL's shader containers.
std::vector<std::uint8_t> container(const std::vector<std::uint8_t>& accc) {
    std::vector<std::uint8_t> out = {'D', 'X', 'B', 'C'};
    out.resize(28, 0);
    put32(out, 2);
    put32(out, 40);
    put32(out, 40 + 8 + 20);
    out.insert(out.end(), {'H', 'A', 'S', 'H'});
    put32(out, 20);
    out.insert(out.end(), 20, 0x5a);
    out.insert(out.end(), {'A', 'C', 'C', 'C'});
    put32(out, std::uint32_t(accc.size()));
    out.insert(out.end(), accc.begin(), accc.end());
    return out;
}

// The slot-67 description built the way amd_fidelityfx_upscaler_xbox.dll builds it (docs/fspx.md, "How the DLL reads
// it", pinned from its code):
//   - its FSPX reader (RVA 0x1a20, and the same loop inline for a container) zeroes four (pointer, size) pairs, walks
//     the directory with the reader's bounds rules and stores kind 0 in pair 0, kind 1 in pair 1, kind 6 in pair 2 and
//     kind 7 in pair 3, the last entry of a kind winning; the pointers point into the payload itself;
//   - it copies kind 6 into an allocation rounded up to 256 bytes (here filled with 0xcc past the program, so a size
//     taken from the allocation would show) and passes that copy with kind 6's own size;
//   - it copies the pairs into the description in that order: pair 0 at +0x08, pair 1 at +0x18, the copy at +0x28,
//     pair 3 at +0x38.
// So the test does not reuse the layout the proxy reads: it follows the DLL's reader over the payload's own directory.
struct DllSlot67 {
    XboxNativeDesc desc{};
    std::vector<std::uint8_t> copy;  // the kind-6 copy (the DLL's XMemAlloc block)
};

DllSlot67 dll_slot67(ID3D12RootSignature* root, const std::vector<std::uint8_t>& container_bytes) {
    DllSlot67 out;
    out.desc.root_signature = root;
    const std::span<const std::uint8_t> payload = leonida::dxbc_part(container_bytes, "ACCC");
    const auto u32 = [&](std::size_t at) {
        std::uint32_t v;
        std::memcpy(&v, payload.data() + at, 4);
        return v;
    };
    if (payload.size() < 16 || std::memcmp(payload.data(), "FSPX", 4) != 0) return out;
    const std::uint64_t total = std::min<std::uint64_t>(u32(12), payload.size());
    const std::uint64_t end_dir = 16 + 16 * std::uint64_t(u32(8));
    if (end_dir > total) return out;
    const void* pointer[4] = {};
    std::uint64_t size[4] = {};
    for (std::uint64_t at = 16; at < end_dir; at += 16) {
        const std::uint32_t kind = u32(std::size_t(at)), offset = u32(std::size_t(at) + 4),
                            length = u32(std::size_t(at) + 8);
        if (offset < end_dir || std::uint64_t(offset) + length > total) continue;
        const int pair = kind == 0 ? 0 : kind == 1 ? 1 : kind == 6 ? 2 : kind == 7 ? 3 : -1;
        if (pair < 0) continue;
        pointer[pair] = payload.data() + offset;
        size[pair] = length;
    }
    out.copy.assign((size[2] + 255) & ~std::uint64_t(255), 0xcc);
    if (size[2]) std::memcpy(out.copy.data(), pointer[2], std::size_t(size[2]));
    out.desc.kind0 = pointer[0];
    out.desc.kind0_size = size[0];
    out.desc.kind1 = pointer[1];
    out.desc.kind1_size = size[1];
    out.desc.kind6 = out.copy.data();
    out.desc.kind6_size = size[2];
    out.desc.kind7 = pointer[3];
    out.desc.kind7_size = size[3];
    return out;
}

std::vector<std::uint8_t> serialize(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc) {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto fn = d3d12 ? reinterpret_cast<SerializeFn>(
                                reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeVersionedRootSignature")))
                          : nullptr;
    std::vector<std::uint8_t> out;
    ID3DBlob* blob = nullptr;
    if (fn && SUCCEEDED(fn(&desc, &blob, nullptr)) && blob) {
        const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
        out.assign(p, p + blob->GetBufferSize());
        blob->Release();
    }
    return out;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    return barrier;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    // Two programs, one translated entry each in <library>/fspx/index.json; the main index holds only an FSSM entry.
    constexpr std::uint32_t marker_a = 5000, marker_b = 7000;
    const auto code_a = kind6(marker_a), code_b = kind6(marker_b);
    const auto container_a = container(payload(code_a));
    const testfiles::ScratchDir scratch("leonida_fspx_compute_test");
    const auto& library = scratch.path();
    std::filesystem::create_directories(library / "fspx");
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV),
                                library / "fspx" / "cs_user_data.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV),
                                library / "fspx" / "cs_user_data_b.spv"));
    // The review case on #1911: program C in two containers whose kind-0 entries differ, translated to two
    // different modules (the second has another generator word in its header, still a valid module).
    const auto code_c = kind6(9100);
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV),
                                library / "fspx" / "cs_c_state1.spv"));
    {
        std::ifstream in(LEONIDA_TEST_CS_SPV, std::ios::binary);
        std::vector<char> words((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        EXPECT(words.size() > 12);
        words[8] ^= 0x5a;  // word 2: the generator
        std::ofstream out(library / "fspx" / "cs_c_state2.spv", std::ios::binary);
        out.write(words.data(), std::streamsize(words.size()));
    }
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"cs_game\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << std::string(64, '0') << "\"}}";
        std::ofstream sub(library / "fspx" / "index.json", std::ios::binary);
        const auto entry = [](const std::vector<std::uint8_t>& code, std::uint8_t state_fill) {
            return "{\"stage\": \"cs\", \"fspx_code_sha256\": \"" + leonida::sha256_hex(code) +
                   "\", \"fspx_state_sha256\": \"" + leonida::sha256_hex(std::vector<std::uint8_t>(28, state_fill)) +
                   "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                   "\"uses_code_base\": true, \"wave64\": false}";
        };
        sub << "{\"cs_user_data\": " << entry(code_a, 0x11) << ",\n \"cs_user_data_b\": " << entry(code_b, 0x11)
            << ",\n \"cs_c_state1\": " << entry(code_c, 0x31) << ",\n \"cs_c_state2\": " << entry(code_c, 0x32)
            << ",\n \"cs_c_state3\": " << entry(code_c, 0x33) << "}";  // no cs_c_state3.spv: see the slot-67 block
    }

    constexpr UINT elements = 64, bytes = elements * 4;
    const std::uint32_t table_values[4] = {1000, 2000, 3000, 4000};
    std::vector<std::uint32_t> first(elements), second(elements);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.shader_dir = library.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        if (rt::d3d12::device_push_constants_need_fallback(inner)) {
            std::printf("d3d12_fspx_compute_test: skipped, the PSOs need 144 push-constant bytes and this GPU has %u\n",
                        rt::d3d12::device_max_push_constants_size(inner));
            return 77;
        }

        // Root signature: [0] two constants (s4, s5), [1] a root UAV (s6:s7), [2] a one-CBV table (s8, its 32-bit VA):
        // the console compute layout (rt::rootsig::pack_stage, D3D-448).
        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 1, 0, 0};
        D3D12_ROOT_PARAMETER params[3] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].Descriptor = {0, 0};
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[2].DescriptorTable = {1, &range};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        // Program A through slot 62 in its container; program B bare, as the slot-67 description carries it.
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.CS = {container_a.data(), container_a.size()};
        const IID pso_iid = __uuidof(ID3D12PipelineState);
        Ref<ID3D12PipelineState> pso_a, pso_b;
        EXPECT(SUCCEEDED(reinterpret_cast<XboxComputePipelineFn>(slot(device.p, 62))(
            device.p, &pso_desc, 0, 0, &pso_iid, reinterpret_cast<void**>(&pso_a.p))));
        pso_desc.CS = {code_b.data(), code_b.size()};
        EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso_b.p))));
        {
            // Program A bare and program B in a container resolve too: the key is the kind-6 bytes either way.
            const auto container_b = container(payload(code_b));
            Ref<ID3D12PipelineState> bare_a, wrapped_b;
            pso_desc.CS = {code_a.data(), code_a.size()};
            EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&bare_a.p))));
            pso_desc.CS = {container_b.data(), container_b.size()};
            EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&wrapped_b.p))));
        }

        // Program C: each container finds its own translation by its kind-0 state; one with a kind 0 nothing was
        // translated for is refused; the bare program is refused as ambiguous, since its two translations differ.
        {
            const auto c1 = container(payload(code_c, 128, -1, 0x31)), c2 = container(payload(code_c, 128, -1, 0x32));
            const auto c_other = container(payload(code_c, 128, -1, 0x11));
            Ref<ID3D12PipelineState> pso_c1, pso_c2, none_other, none_bare;
            pso_desc.CS = {c1.data(), c1.size()};
            EXPECT(SUCCEEDED(reinterpret_cast<XboxComputePipelineFn>(slot(device.p, 62))(
                device.p, &pso_desc, 0, 0, &pso_iid, reinterpret_cast<void**>(&pso_c1.p))));
            pso_desc.CS = {c2.data(), c2.size()};
            EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso_c2.p))));
            EXPECT(pso_c1.p != pso_c2.p);
            pso_desc.CS = {c_other.data(), c_other.size()};
            EXPECT(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&none_other.p)) == E_INVALIDARG &&
                   !none_other.p);
            pso_desc.CS = {code_c.data(), code_c.size()};
            EXPECT(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&none_bare.p)) == E_INVALIDARG &&
                   !none_bare.p);
        }

        // Slot 67 with all four pairs (RT-99): the proxy passes kind 0 on, so program C resolves by its state there
        // too. State 0x31 finds cs_c_state1 (S_OK); state 0x33 finds cs_c_state3, whose module is missing on purpose
        // (E_FAIL, not the E_INVALIDARG of an unknown program), so a lookup that ignored kind 0 would show. The
        // pipeline's console slot 11 answers with a blob holding 0, and every object is released again.
        {
            const std::vector<std::uint8_t> state31(28, 0x31), state33(28, 0x33), kind1(20, 0x22), kind7(8, 0x77);
            XboxNativeDesc native{};
            native.root_signature = root.p;
            native.kind0 = state31.data();
            native.kind0_size = state31.size();
            native.kind1 = kind1.data();
            native.kind1_size = kind1.size();
            native.kind6 = code_c.data();
            native.kind6_size = code_c.size();
            native.kind7 = kind7.data();
            native.kind7_size = kind7.size();
            const auto slot67 = reinterpret_cast<XboxNativeComputePipelineFn>(slot(device.p, 67));
            Ref<ID3D12PipelineState> pso67, missing67;
            EXPECT(SUCCEEDED(slot67(device.p, &native, &pso_iid, reinterpret_cast<void**>(&pso67.p))) && pso67.p);
            native.kind0 = state33.data();
            EXPECT(slot67(device.p, &native, &pso_iid, reinterpret_cast<void**>(&missing67.p)) == E_FAIL &&
                   !missing67.p);
            native.kind0 = state31.data();

            // Inconsistent pairs are refused before anything is read (the wild pointer is never dereferenced).
            const void* wild = reinterpret_cast<const void*>(std::uintptr_t(0x10));
            const auto refused67 = [&](XboxNativeDesc bad) {
                Ref<ID3D12PipelineState> none;
                return slot67(device.p, &bad, &pso_iid, reinterpret_cast<void**>(&none.p)) == E_INVALIDARG && !none.p;
            };
            XboxNativeDesc bad = native;
            bad.kind1_size = 0;  // a pointer without a size
            EXPECT(refused67(bad));
            bad = native;
            bad.kind7 = nullptr;  // a size without a pointer
            EXPECT(refused67(bad));
            bad = native;
            bad.kind0 = nullptr;  // kind 0's size without its pointer
            EXPECT(refused67(bad));
            bad = native;
            bad.kind6 = wild;
            bad.kind6_size = std::uint64_t(1) << 40;  // far over the limit
            EXPECT(refused67(bad));

            // Slot 11 as the DLL calls it: (0, 0, &obj); obj's slot 3 gives a pointer to the dword it multiplies by
            // 1024, slot 2 releases it. The pipeline's own count is unchanged by the call.
            using Slot11Fn = HRESULT(WINAPI*)(void*, std::uint64_t, std::uint64_t, void**);
            using BufferPointerFn = void*(WINAPI*)(void*);
            using SizeFn = SIZE_T(WINAPI*)(void*);
            using ReleaseFn = ULONG(WINAPI*)(void*);
            for (ID3D12PipelineState* pipeline : {pso67.p, pso_a.p}) {
                const ULONG before = pipeline->AddRef() - 1;
                pipeline->Release();
                void* obj = nullptr;
                EXPECT(reinterpret_cast<Slot11Fn>(slot(pipeline, 11))(pipeline, 0, 0, &obj) == S_OK && obj);
                const auto* value = static_cast<const std::uint32_t*>(reinterpret_cast<BufferPointerFn>(slot(obj, 3))(obj));
                EXPECT(value && *value == 0);
                EXPECT(reinterpret_cast<SizeFn>(slot(obj, 4))(obj) == 4);
                EXPECT(reinterpret_cast<ReleaseFn>(slot(obj, 2))(obj) == 0);
                EXPECT(reinterpret_cast<Slot11Fn>(slot(pipeline, 11))(pipeline, 0, 0, nullptr) == E_POINTER);
                const ULONG after = pipeline->AddRef() - 1;
                pipeline->Release();
                EXPECT(after == before);
            }
        }

        // Slot 67 with the description built the way the DLL builds it (dll_slot67): each program resolves to its
        // own translation with its own state. Program A (state 0x11) and program C with states 0x31 and 0x32 create a
        // pipeline; C with 0x33 finds its own entry, whose module is missing (E_FAIL, not E_INVALIDARG); C with 0x11,
        // a state nothing was translated for, is refused like any unknown program.
        {
            const auto slot67 = reinterpret_cast<XboxNativeComputePipelineFn>(slot(device.p, 67));
            const auto create = [&](const std::vector<std::uint8_t>& box) {
                const DllSlot67 call = dll_slot67(root.p, box);
                Ref<ID3D12PipelineState> made;
                const HRESULT hr = slot67(device.p, &call.desc, &pso_iid, reinterpret_cast<void**>(&made.p));
                if (FAILED(hr) && made.p) return E_UNEXPECTED;  // a failure must leave no object
                return hr;
            };
            EXPECT(dll_slot67(root.p, container_a).desc.kind0_size == 28);       // the builder found the entries
            EXPECT(dll_slot67(root.p, container_a).copy.size() % 256 == 0);
            EXPECT(create(container_a) == S_OK);
            EXPECT(create(container(payload(code_c, 128, -1, 0x31))) == S_OK);
            EXPECT(create(container(payload(code_c, 128, -1, 0x32))) == S_OK);
            EXPECT(create(container(payload(code_c, 128, -1, 0x33))) == E_FAIL);
            EXPECT(create(container(payload(code_c, 128, -1, 0x11))) == E_INVALIDARG);
        }

        // Refused with E_INVALIDARG, and no object: an untranslated program, bare and in a container; directories
        // whose kind-6 entry is out of bounds or missing; a container without ACCC; bytes without the prologue.
        {
            const auto untranslated = kind6(9000);
            auto no_prologue = code_a;
            no_prologue[2] ^= 0x01;
            const auto padded = [&] {  // one more byte than the program: a different key, not a translated one
                auto out = code_a;
                out.insert(out.end(), {0, 0, 0, 0});
                return out;
            }();
            std::vector<std::uint8_t> hash_only = container(payload(code_a));
            std::memcpy(hash_only.data() + 40 + 8 + 20, "ACCX", 4);
            const std::vector<std::vector<std::uint8_t>> refused = {
                untranslated,
                container(payload(untranslated)),
                container(payload(code_a, 140)),                   // kind 6 ends past the total
                container(payload(code_a, 16)),                    // kind 6 inside the directory
                container(payload(code_a, 0xffffff00u, 0x200)),    // offset + size wraps 32 bits
                container(payload(code_a, 128, 0xffffffffu)),      // size past the payload
                container(payload(code_a, 128, 0)),                // kind 6 empty: entry ignored, kind 6 missing
                hash_only,
                no_prologue,
                padded,
            };
            for (std::size_t i = 0; i < refused.size(); ++i) {
                Ref<ID3D12PipelineState> none;
                pso_desc.CS = {refused[i].data(), refused[i].size()};
                const HRESULT hr = device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&none.p));
                if (hr != E_INVALIDARG || none.p) std::fprintf(stderr, "refused input %zu: 0x%08x\n", i, unsigned(hr));
                EXPECT(hr == E_INVALIDARG && !none.p);
            }
            Ref<ID3D12PipelineState> none;
            pso_desc.CS = {refused[2].data(), refused[2].size()};
            EXPECT(reinterpret_cast<XboxComputePipelineFn>(slot(device.p, 62))(
                       device.p, &pso_desc, 0, 0, &pso_iid, reinterpret_cast<void**>(&none.p)) == E_INVALIDARG &&
                   !none.p);
        }

        // One dispatch per PSO; each shader adds dword 1 of its code base, which is its program's marker.
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> out_a, out_b, constants, readback;
        const auto uav = buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        for (auto* out : {&out_a, &out_b})
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav,
                                                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                             IID_PPV_ARGS(&out->p))));
        const auto constants_desc = buffer_desc(256, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &constants_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&constants.p))));
        const auto readback_desc = buffer_desc(2 * bytes, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(constants->Map(0, nullptr, &mapped)));
        std::memcpy(mapped, table_values, sizeof(table_values));

        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
        const std::uint64_t cpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 9))(heap.p);
        const std::uint64_t gpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 10))(heap.p);
        EXPECT(cpu && gpu);
        const std::uint64_t target = constants->GetGPUVirtualAddress();
        std::memcpy(reinterpret_cast<void*>(cpu), &target, sizeof(target));

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        ID3D12DescriptorHeap* heaps[] = {heap.p};
        list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootSignature(root.p);
        const std::uint32_t base_and_step[2] = {7, 3};
        list->SetComputeRoot32BitConstants(0, 2, base_and_step, 0);
        list->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{gpu});
        list->SetPipelineState(pso_a.p);
        list->SetComputeRootUnorderedAccessView(1, out_a->GetGPUVirtualAddress());
        list->Dispatch(1, 1, 1);
        list->SetPipelineState(pso_b.p);
        list->SetComputeRootUnorderedAccessView(1, out_b->GetGPUVirtualAddress());
        list->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER barriers[2] = {transition(out_a.p), transition(out_b.p)};
        list->ResourceBarrier(2, barriers);
        list->CopyBufferRegion(readback.p, 0, out_a.p, 0, bytes);
        list->CopyBufferRegion(readback.p, bytes, out_b.p, 0, bytes);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* results = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &results)));
        std::memcpy(first.data(), results, bytes);
        std::memcpy(second.data(), static_cast<const std::uint8_t*>(results) + bytes, bytes);

        for (auto* r : {&out_a, &out_b, &constants, &readback}) r->reset();
        pso_a.reset();
        pso_b.reset();
        root.reset();
        heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    unsigned mismatches = 0;
    for (std::uint32_t i = 0; i < elements; ++i) {
        const std::uint32_t a = 7 + 3 * i + table_values[i % 4] + marker_a;
        const std::uint32_t b = 7 + 3 * i + table_values[i % 4] + marker_b;
        if ((first[i] != a || second[i] != b) && mismatches++ < 4)
            std::fprintf(stderr, "element %u: got %u and %u, want %u and %u\n", i, first[i], second[i], a, b);
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("fspx compute: PSOs from an ACCC container (slot 62) and a bare kind-6 program found their translated "
                "modules by fspx_code_sha256; each dispatch read its kind-6 program as its code base; malformed and "
                "untranslated inputs refused\n");
    return 0;
}
