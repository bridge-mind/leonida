// The compute path end to end, through the Xbox-ABI proxy over the Vulkan device:
//   - a root signature serialized by the PC runtime, created with CreateRootSignature;
//   - a compute PSO created through device slot 62 (the Xbox shape) from a shader container shaped like the
//     game's (DXBC, DXIL, XDAT 0x0D, XBCH, zlib), found by the SHA-256 of its FSSM program in a translator index;
//   - buffers and a shader-visible descriptor heap in the GPU address arena, an upload buffer reached through its
//     biased console VA;
//   - root constants, a root UAV and a descriptor table set on the command list, two dispatches that each keep the
//     arguments they were recorded with, and the results copied to a readback buffer.
// The shader (test/glsl/cs_user_data.comp) is written to the translator's push-constant and address contract.
// No game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"
#include "../src/cached_pso.h"
#include "../src/shader_library.h"
#include "../shims/d3d12_xs_device.h"

#include <dxgi1_4.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#ifndef LEONIDA_TEST_CS_SPV
#error "LEONIDA_TEST_CS_SPV must name the compiled test shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX
// Device slot 62 as XboxBvhBuilder.dll calls it on 1.0.2.1: (desc, 0, 0, riid, out) (main's 28cf235).
using XboxComputePipelineFn = HRESULT(WINAPI*)(void* device, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, std::uint64_t,
                                               std::uint64_t, const IID*, void**);
// Device slot 67 as amd_fidelityfx_upscaler_xbox.dll calls it: (desc, riid, out).
using XboxNativeComputePipelineFn = HRESULT(WINAPI*)(void* device, const void*, const IID*, void**);
using SerializeFn = HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);

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

// zlib with one stored block (RFC 1950/1951), enough to wrap a test program.
std::vector<std::uint8_t> zlib_stored(const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> out = {0x78, 0x01, 0x01};
    const auto length = std::uint16_t(data.size());
    out.insert(out.end(), {std::uint8_t(length), std::uint8_t(length >> 8), std::uint8_t(~length),
                           std::uint8_t(~length >> 8)});
    out.insert(out.end(), data.begin(), data.end());
    std::uint32_t a = 1, b = 0;
    for (const std::uint8_t x : data) {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    const std::uint32_t adler = (b << 16) | a;
    out.insert(out.end(), {std::uint8_t(adler >> 24), std::uint8_t(adler >> 16), std::uint8_t(adler >> 8),
                           std::uint8_t(adler)});
    return out;
}

// DXBC { DXIL { stub, XDAT { 0x0D: XBCH + zlib(fssm) } } }, the layout tools/fssm_unpack.py reads.
std::vector<std::uint8_t> shader_container(const std::vector<std::uint8_t>& fssm) {
    const auto stream = zlib_stored(fssm);
    std::vector<std::uint8_t> xbch = {'X', 'B', 'C', 'H'};
    for (const std::uint32_t v : {1u, std::uint32_t(stream.size()), 0u, std::uint32_t(fssm.size()), 0u}) put32(xbch, v);
    xbch.insert(xbch.end(), stream.begin(), stream.end());
    std::vector<std::uint8_t> dxil(16, 0);
    dxil.insert(dxil.end(), {'X', 'D', 'A', 'T'});
    put32(dxil, 1);
    put32(dxil, 12);
    put32(dxil, 0x0d);
    put32(dxil, 16 + 12 + 12);
    put32(dxil, std::uint32_t(xbch.size()));
    dxil.insert(dxil.end(), xbch.begin(), xbch.end());
    std::vector<std::uint8_t> out = {'D', 'X', 'B', 'C'};
    out.resize(28, 0);
    put32(out, 1);
    put32(out, 36);
    out.insert(out.end(), {'D', 'X', 'I', 'L'});
    put32(out, std::uint32_t(dxil.size()));
    out.insert(out.end(), dxil.begin(), dxil.end());
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

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

// The CachedPSO cases both devices answer, in this order: the same description, another Flags, an equal root
// signature object, a root signature with other content, a truncated blob, a blob of junk, its first, middle and last
// byte changed, a pointer with size 0, a null pointer with a size.
constexpr int cached_case_count = 11;

template<class Device>
std::vector<HRESULT> cached_pso_cases(Device* device, D3D12_COMPUTE_PIPELINE_STATE_DESC desc,
                                      const std::vector<std::uint8_t>& root_blob,
                                      const std::vector<std::uint8_t>& other_root_blob) {
    std::vector<HRESULT> out;
    auto create = [&](D3D12_COMPUTE_PIPELINE_STATE_DESC d, const void* p, SIZE_T n) {
        d.CachedPSO = {p, n};
        ID3D12PipelineState* made = nullptr;
        HRESULT hr = device->CreateComputePipelineState(&d, IID_PPV_ARGS(&made));
        if (SUCCEEDED(hr) != (made != nullptr)) hr = E_UNEXPECTED;  // a result and its object disagree
        if (made) made->Release();
        return hr;
    };
    ID3D12PipelineState* pso = nullptr;
    if (FAILED(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) return out;
    ID3DBlob* blob = nullptr;
    const HRESULT got = pso->GetCachedBlob(&blob);
    const HRESULT got_null = pso->GetCachedBlob(nullptr);
    pso->Release();
    if (FAILED(got) || !blob || got_null != S_OK) {
        if (blob) blob->Release();
        return out;
    }
    const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
    std::vector<std::uint8_t> bytes(p, p + blob->GetBufferSize());
    blob->Release();
    ID3D12RootSignature* same_root = nullptr;
    ID3D12RootSignature* other_root = nullptr;
    device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&same_root));
    device->CreateRootSignature(0, other_root_blob.data(), other_root_blob.size(), IID_PPV_ARGS(&other_root));
    if (!same_root || !other_root) return out;

    out.push_back(create(desc, bytes.data(), bytes.size()));
    D3D12_COMPUTE_PIPELINE_STATE_DESC flags = desc;
    flags.Flags = D3D12_PIPELINE_STATE_FLAG_TOOL_DEBUG;
    out.push_back(create(flags, bytes.data(), bytes.size()));
    D3D12_COMPUTE_PIPELINE_STATE_DESC same = desc;
    same.pRootSignature = same_root;
    out.push_back(create(same, bytes.data(), bytes.size()));
    D3D12_COMPUTE_PIPELINE_STATE_DESC other = desc;
    other.pRootSignature = other_root;
    out.push_back(create(other, bytes.data(), bytes.size()));
    out.push_back(create(desc, bytes.data(), bytes.size() / 2));
    const std::vector<std::uint8_t> junk(bytes.size(), 0xab);
    out.push_back(create(desc, junk.data(), junk.size()));
    for (const std::size_t at : {std::size_t(0), bytes.size() / 2, bytes.size() - 1}) {
        bytes[at] ^= 0xff;
        out.push_back(create(desc, bytes.data(), bytes.size()));
        bytes[at] ^= 0xff;
    }
    out.push_back(create(desc, bytes.data(), 0));
    out.push_back(create(desc, nullptr, 16));
    same_root->Release();
    other_root->Release();
    return out;
}

// The same cases on the PC runtime, with an empty compute shader from d3dcompiler_47: the default adapter, then WARP.
// Empty when the host has no D3D12 device or no compiler, or is Wine's, which the caller reports as a skip: vkd3d has
// both, but its GetCachedBlob is a stub, so every host answered nothing and the comparison failed on each run.
std::vector<std::vector<HRESULT>> host_cached_pso_cases(const std::vector<std::uint8_t>& root_blob,
                                                        const std::vector<std::uint8_t>& other_root_blob) {
    if (hostwine::host_is_wine()) return {};
    using CompileFn = HRESULT(WINAPI*)(LPCVOID, SIZE_T, LPCSTR, const void*, void*, LPCSTR, LPCSTR, UINT, UINT,
                                       ID3DBlob**, ID3DBlob**);
    using CreateDeviceFn = decltype(&::D3D12CreateDevice);
    std::vector<std::vector<HRESULT>> out;
    const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const HMODULE dxgi = LoadLibraryExA("dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!compiler || !d3d12 || !dxgi) return out;
    const auto compile = reinterpret_cast<CompileFn>(reinterpret_cast<void*>(GetProcAddress(compiler, "D3DCompile")));
    const auto create_device =
        reinterpret_cast<CreateDeviceFn>(reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")));
    using FactoryFn = HRESULT(WINAPI*)(REFIID, void**);
    const auto create_factory =
        reinterpret_cast<FactoryFn>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
    if (!compile || !create_device || !create_factory) return out;
    static const char source[] = "[numthreads(1, 1, 1)] void main() {}";
    ID3DBlob* shader = nullptr;
    if (FAILED(compile(source, sizeof(source) - 1, nullptr, nullptr, nullptr, "main", "cs_5_0", 0, 0, &shader,
                       nullptr)) || !shader)
        return out;
    IDXGIFactory4* factory = nullptr;
    if (SUCCEEDED(create_factory(IID_PPV_ARGS(&factory)))) {
        IDXGIAdapter* warp = nullptr;
        factory->EnumWarpAdapter(IID_PPV_ARGS(&warp));
        for (IUnknown* adapter : {static_cast<IUnknown*>(nullptr), static_cast<IUnknown*>(warp)}) {
            ID3D12Device* host = nullptr;
            if (FAILED(create_device(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host)))) continue;
            ID3D12RootSignature* root = nullptr;
            if (SUCCEEDED(host->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root)))) {
                D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
                desc.pRootSignature = root;
                desc.CS = {shader->GetBufferPointer(), shader->GetBufferSize()};
                out.push_back(cached_pso_cases(host, desc, root_blob, other_root_blob));
                root->Release();
            }
            host->Release();
        }
        if (warp) warp->Release();
        factory->Release();
    }
    shader->Release();
    return out;
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

    // The shader program the PSO carries, and a translator index that maps its hash to the test module.
    constexpr std::uint32_t code_word = 5000;
    std::vector<std::uint8_t> fssm = {'F', 'S', 'S', 'M'};
    put32(fssm, code_word);
    for (std::uint32_t i = 0; i < 14; ++i) put32(fssm, 0xbf810000u + i);  // filler, s_endpgm-like words
    const auto container = shader_container(fssm);
    EXPECT(leonida::fssm_program(container) == fssm);
    const testfiles::ScratchDir scratch("leonida_compute_test");
    const auto& library = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), library / "cs_user_data.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        // An entry for the same program with set-0 bindings comes first, as in one group of the corpus: the PSO
        // must take the binding-free entry, which needs no set 0 (the bound one has no module either).
        index << "{\"cs_user_data_bound\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(fssm)
              << "\", \"bindings\": [{\"binding\": 0}], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1]},\n"
                 " \"cs_user_data\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(fssm)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": true, \"wave64\": false}}";
    }
    // A second program whose only translation is in <library>/bvh/index.json, where make_game.py puts the programs
    // of XboxBvhBuilder.dll (docs/embedded_shaders.md): the same container shape and the same key, another folder.
    auto builder_fssm = fssm;
    builder_fssm[4] ^= 0x40;  // another code word, so another hash
    const auto builder_container = shader_container(builder_fssm);
    std::filesystem::create_directories(library / "bvh");
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV),
                                library / "bvh" / "BuilderProgram.spv"));
    {
        std::ofstream index(library / "bvh" / "index.json", std::ios::binary);
        index << "{\"BuilderProgram\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(builder_fssm)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": true, \"wave64\": false}}";
    }

    constexpr UINT elements = 64, bytes = elements * 4;
    const std::uint32_t table_values[4] = {1000, 2000, 3000, 4000};
    std::vector<std::uint32_t> first(elements), second(elements), third(elements);
    std::vector<std::uint32_t> cbv_a_out(elements), cbv_oob_out(elements), cbv_b_out(elements), cbv_null_out(elements);
    std::vector<std::uint32_t> srv_raw_a(elements), srv_raw_b(elements), srv_struct_a(elements),
        srv_struct_b(elements), srv_typed(elements), srv_null(elements);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.subgroup = d.subgroup;
        info.shader_dir = library.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

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
        Ref<ID3D12RootSignature> junk;
        EXPECT(device->CreateRootSignature(0, "junk", 4, IID_PPV_ARGS(&junk.p)) == E_INVALIDARG && !junk.p);

        // The PSO through slot 62, the Xbox call shape; an unknown program is refused.
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.CS = {container.data(), container.size()};
        const IID pso_iid = __uuidof(ID3D12PipelineState);
        Ref<ID3D12PipelineState> pso, unknown;
        EXPECT(SUCCEEDED(reinterpret_cast<XboxComputePipelineFn>(slot(device.p, 62))(
            device.p, &pso_desc, 0, 0, &pso_iid, reinterpret_cast<void**>(&pso.p))));
        auto other = fssm;
        other[8] ^= 1;
        const auto other_container = shader_container(other);
        pso_desc.CS = {other_container.data(), other_container.size()};
        EXPECT(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&unknown.p)) == E_INVALIDARG && !unknown.p);
        {
            // XboxBvhBuilder.dll's call: slot 62 with a container whose program is translated in <library>/bvh.
            Ref<ID3D12PipelineState> builder_pso;
            pso_desc.CS = {builder_container.data(), builder_container.size()};
            EXPECT(SUCCEEDED(reinterpret_cast<XboxComputePipelineFn>(slot(device.p, 62))(
                device.p, &pso_desc, 0, 0, &pso_iid, reinterpret_cast<void**>(&builder_pso.p))));
            EXPECT(builder_pso.p && builder_pso.p != pso.p);
        }
        pso_desc.CS = {container.data(), container.size()};

        // The PSO through slot 67, compute pipeline from native code; maps to CreateComputePipelineState.
        struct XboxNativeComputeDesc {
            ID3D12RootSignature* root_signature;
            const void* kind0;
            std::uint64_t kind0_size;
            const void* kind1;
            std::uint64_t kind1_size;
            const void* kind6;
            std::uint64_t kind6_size;
            const void* kind7;
            std::uint64_t kind7_size;
        } native_desc{};
        native_desc.root_signature = root.p;
        native_desc.kind6 = fssm.data();
        native_desc.kind6_size = fssm.size();
        Ref<ID3D12PipelineState> pso67, unknown67;
        EXPECT(SUCCEEDED(reinterpret_cast<XboxNativeComputePipelineFn>(slot(device.p, 67))(
            device.p, &native_desc, &pso_iid, reinterpret_cast<void**>(&pso67.p))));
        EXPECT(pso67.p != nullptr);
        native_desc.kind6 = other.data();
        native_desc.kind6_size = other.size();
        EXPECT(reinterpret_cast<XboxNativeComputePipelineFn>(slot(device.p, 67))(
            device.p, &native_desc, &pso_iid, reinterpret_cast<void**>(&unknown67.p)) == E_INVALIDARG && !unknown67.p);
        void* out_null = reinterpret_cast<void*>(0x1111);
        EXPECT(reinterpret_cast<XboxNativeComputePipelineFn>(slot(device.p, 67))(
            device.p, nullptr, &pso_iid, &out_null) == E_INVALIDARG && !out_null);
        out_null = reinterpret_cast<void*>(0x1111);
        native_desc.kind6 = fssm.data();
        native_desc.kind6_size = fssm.size();
        EXPECT(reinterpret_cast<XboxNativeComputePipelineFn>(slot(device.p, 67))(
            device.p, &native_desc, nullptr, &out_null) == E_POINTER && !out_null);
        EXPECT(reinterpret_cast<XboxNativeComputePipelineFn>(slot(device.p, 67))(
            device.p, &native_desc, &pso_iid, nullptr) == E_POINTER);
        // GetCachedBlob and CachedPSO answer as the PC runtime does (src/cached_pso.h).
        {
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC other_desc = root_desc;
            other_desc.Desc_1_0.NumParameters = 2;
            const auto other_root_blob = serialize(other_desc);
            EXPECT(!other_root_blob.empty());
            const std::vector<HRESULT> layer = cached_pso_cases(device.p, pso_desc, root_blob, other_root_blob);
            EXPECT(int(layer.size()) == cached_case_count);
            const std::vector<HRESULT> expected = {S_OK,         S_OK,         S_OK,         E_INVALIDARG,
                                                   E_INVALIDARG, E_INVALIDARG, E_INVALIDARG, E_INVALIDARG,
                                                   E_INVALIDARG, E_INVALIDARG, S_OK};
            EXPECT(layer == expected);
            const auto hosts = host_cached_pso_cases(root_blob, other_root_blob);
            if (hosts.empty())
                std::printf("SKIP: no host D3D12 reference (Wine, or no device or compiler) for the CachedPSO "
                            "comparison\n");
            for (const auto& host : hosts) EXPECT(host == layer);
            if (!hosts.empty()) std::printf("cached PSO: %zu host adapter(s) gave the same %d answers\n", hosts.size(), cached_case_count);

            // Every byte of the blob counts, and the adapter and driver fields give the runtime's own errors.
            Ref<ID3DBlob> blob;
            EXPECT(pso->GetCachedBlob(&blob.p) == S_OK && blob.p);
            const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
            std::vector<std::uint8_t> blob_bytes(p, p + blob->GetBufferSize());
            auto attempt = [&](const std::vector<std::uint8_t>& b) {
                D3D12_COMPUTE_PIPELINE_STATE_DESC d = pso_desc;
                d.CachedPSO = {b.data(), b.size()};
                Ref<ID3D12PipelineState> made;
                return device->CreateComputePipelineState(&d, IID_PPV_ARGS(&made.p));
            };
            for (std::size_t at = 0; at < blob_bytes.size(); ++at) {
                auto changed = blob_bytes;
                changed[at] ^= 0x01;
                EXPECT(attempt(changed) == E_INVALIDARG);
            }
            auto resealed = [](std::vector<std::uint8_t> b, std::size_t at) {
                b[at] ^= 0x01;
                const std::uint32_t sum = leonida::cached_pso::fnv1a(b.data(), b.size() - 4);
                for (int i = 0; i < 4; ++i) b[b.size() - 4 + i] = std::uint8_t(sum >> (8 * i));
                return b;
            };
            EXPECT(attempt(resealed(blob_bytes, 8)) == D3D12_ERROR_ADAPTER_NOT_FOUND);   // vendor
            EXPECT(attempt(resealed(blob_bytes, 12)) == D3D12_ERROR_ADAPTER_NOT_FOUND);  // device
            EXPECT(attempt(resealed(blob_bytes, 16)) == D3D12_ERROR_DRIVER_VERSION_MISMATCH);
            EXPECT(attempt(resealed(blob_bytes, 20)) == D3D12_ERROR_DRIVER_VERSION_MISMATCH);  // pipelineCacheUUID
            EXPECT(attempt(resealed(blob_bytes, 4)) == E_INVALIDARG);  // version
        }

        // A device whose maxPushConstantsSize is the Vulkan minimum (128 bytes) cannot hold the 144-byte block, so
        // its PSOs are refused instead of creating a pipeline layout the spec forbids (#224's fallback flag).
        {
            rt::d3d12::VulkanDeviceInfo limited_info = info;
            limited_info.max_push_constants_size = 128;
            limited_info.push_constants_need_fallback = true;
            Ref<ID3D12Device> limited;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(limited_info, IID_PPV_ARGS(&limited.p))));
            EXPECT(rt::d3d12::device_push_constants_need_fallback(limited.p));
            Ref<ID3D12RootSignature> limited_root;
            EXPECT(SUCCEEDED(limited->CreateRootSignature(0, root_blob.data(), root_blob.size(),
                                                        IID_PPV_ARGS(&limited_root.p))));
            D3D12_COMPUTE_PIPELINE_STATE_DESC limited_desc = pso_desc;
            limited_desc.pRootSignature = limited_root.p;
            Ref<ID3D12PipelineState> refused;
            EXPECT(limited->CreateComputePipelineState(&limited_desc, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);
        }

        // The GPU address arena is no larger than the device's maxBufferSize and maxMemoryAllocationSize
        // (src/arena_size.h), whatever LEONIDA_ARENA_MB asks for: with LEONIDA_ARENA_MB=3000 on a GPU that reports
        // 2 GiB (RX 6800), a run under LEONIDA_VK_VALIDATION=1 has no vkAllocateMemory error. A device that reports
        // a 32 MiB maxMemoryAllocationSize (simulated) gets at most 32 MiB from the default 256, and a buffer past
        // that is refused.
        {
            VkPhysicalDeviceMaintenance4Properties m4{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
            VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES, &m4};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m3};
            d.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(d.physical, &properties);
            const std::uint64_t arena = rt::d3d12::device_arena_size(device.p);
            EXPECT(arena > 0 && arena <= m4.maxBufferSize && arena <= m3.maxMemoryAllocationSize);

            constexpr std::uint64_t capped_limit = 32ull << 20;
            rt::d3d12::VulkanDeviceInfo capped_info = info;
            capped_info.max_memory_allocation_size = capped_limit;
            Ref<ID3D12Device> capped;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(capped_info, IID_PPV_ARGS(&capped.p))));
            const std::uint64_t capped_arena = rt::d3d12::device_arena_size(capped.p);
            EXPECT(capped_arena > 0 && capped_arena <= capped_limit);
            const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
            const auto fits = buffer_desc(1u << 20, D3D12_RESOURCE_FLAG_NONE);
            const auto too_large = buffer_desc(capped_limit + (1u << 20), D3D12_RESOURCE_FLAG_NONE);
            Ref<ID3D12Resource> inside, past;
            EXPECT(SUCCEEDED(capped->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &fits,
                                                            D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                            IID_PPV_ARGS(&inside.p))));
            EXPECT(inside->GetGPUVirtualAddress() + (1u << 20) <= capped_arena);
            EXPECT(capped->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &too_large, D3D12_RESOURCE_STATE_COMMON,
                                                  nullptr, IID_PPV_ARGS(&past.p)) == E_OUTOFMEMORY && !past.p);
            std::printf("arena: %llu MiB (device limits: maxBufferSize %llu MiB, maxMemoryAllocationSize %llu MiB); "
                        "%llu MiB with a simulated 32 MiB maxMemoryAllocationSize\n",
                        static_cast<unsigned long long>(arena >> 20),
                        static_cast<unsigned long long>(m4.maxBufferSize >> 20),
                        static_cast<unsigned long long>(m3.maxMemoryAllocationSize >> 20),
                        static_cast<unsigned long long>(capped_arena >> 20));
        }

        // Buffers in the arena, an upload buffer outside it, a readback buffer for the results.
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> out_a, out_b, out_c, constants, readback;
        const auto uav = buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&out_a.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&out_b.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&out_c.p))));
        const auto constants_desc = buffer_desc(256, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &constants_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&constants.p))));
        const auto readback_desc = buffer_desc(3 * bytes, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        const D3D12_GPU_VIRTUAL_ADDRESS va_a = out_a->GetGPUVirtualAddress(), va_b = out_b->GetGPUVirtualAddress(),
                                        va_c = out_c->GetGPUVirtualAddress();
        EXPECT(va_a >= 0x10000 && va_b >= 0x10000 && va_c >= 0x10000 && va_a != va_b && va_a != va_c && va_b != va_c &&
               va_a < (1ull << 32) && va_b < (1ull << 32) && va_c < (1ull << 32));
        EXPECT(va_a % D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT == 0);
        EXPECT(constants->GetGPUVirtualAddress() != 0);  // device address - heap base: wraps, still valid
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(constants->Map(0, nullptr, &mapped)));
        std::memcpy(mapped, table_values, sizeof(table_values));

        // A shader-visible heap in the arena; its first slot gets the upload buffer's VA, written as raw bytes.
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
        const std::uint64_t cpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 9))(heap.p);
        const std::uint64_t gpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 10))(heap.p);
        EXPECT(cpu && gpu >= 0x10000 && gpu < (1ull << 32));
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

        // Misuse is reported by Close: a dispatch without a pipeline, a root view set on a constants parameter.
        list->Dispatch(1, 1, 1);
        EXPECT(list->Close() == E_INVALIDARG);
        EXPECT(SUCCEEDED(list->Reset(allocator.p, pso.p)));
        list->SetComputeRootSignature(root.p);
        list->SetComputeRootUnorderedAccessView(0, va_a);
        EXPECT(list->Close() == E_INVALIDARG);

        // Two dispatches: the second changes s4 and the UAV; each must keep what it was recorded with.
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        ID3D12DescriptorHeap* heaps[] = {heap.p};
        list->SetDescriptorHeaps(1, heaps);
        // This root signature and PSO are released right after being set, as D3D12 allows: the list keeps them,
        // so the arguments below still have a layout and ExecuteCommandLists still finds the pipeline.
        ID3D12RootSignature* held_root = nullptr;
        {
            Ref<ID3D12RootSignature> own_root;
            EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(),
                                                         IID_PPV_ARGS(&own_root.p))));
            D3D12_COMPUTE_PIPELINE_STATE_DESC own_desc = pso_desc;
            own_desc.pRootSignature = own_root.p;
            Ref<ID3D12PipelineState> own_pso;
            EXPECT(SUCCEEDED(device->CreateComputePipelineState(&own_desc, IID_PPV_ARGS(&own_pso.p))));
            list->SetComputeRootSignature(own_root.p);
            list->SetPipelineState(own_pso.p);
            held_root = own_root.p;  // still alive: the list holds it
        }
        // Setting the bound root signature again keeps the arguments, as D3D12 does (TEST-280), and setting a
        // graphics root signature does not disturb the compute root arguments.
        const std::uint32_t base_and_step[2] = {7, 3};
        list->SetComputeRoot32BitConstant(0, base_and_step[0], 0);
        list->SetComputeRootSignature(held_root);
        list->SetComputeRoot32BitConstant(0, base_and_step[1], 1);
        list->SetComputeRootUnorderedAccessView(1, va_a);
        list->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{gpu});
        list->SetComputeRootSignature(held_root);
        list->SetGraphicsRootSignature(root.p);
        list->SetGraphicsRoot32BitConstant(0, 999, 0);
        list->SetGraphicsRootSignature(root.p);
        list->SetGraphicsRoot32BitConstant(0, 888, 1);
        list->Dispatch(1, 1, 1);
        list->SetComputeRoot32BitConstant(0, 100, 0);
        list->SetComputeRootUnorderedAccessView(1, va_b);
        list->Dispatch(1, 1, 1);
        // Switching to a different root-signature object (root.p != held_root) resets the bound arguments to zero
        // (clearing s0 = 100 to 0), whereas rebinding root.p afterwards keeps the arguments set on it (s1 = 5).
        list->SetComputeRootSignature(root.p);
        list->SetComputeRoot32BitConstant(0, 5, 1);  // offset 1 (s1 = 5); offset 0 (s0) stays 0 from the switch
        list->SetComputeRootUnorderedAccessView(1, va_c);
        list->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{gpu});
        list->SetComputeRootSignature(root.p);
        list->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER barriers[3] = {
            transition(out_a.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transition(out_b.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transition(out_c.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)};
        list->ResourceBarrier(3, barriers);
        list->CopyBufferRegion(readback.p, 0, out_a.p, 0, bytes);
        list->CopyBufferRegion(readback.p, bytes, out_b.p, 0, bytes);
        list->CopyBufferRegion(readback.p, 2 * bytes, out_c.p, 0, bytes);
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
        std::memcpy(third.data(), static_cast<const std::uint8_t*>(results) + 2 * bytes, bytes);

        // CreateConstantBufferView: populate shader-visible descriptor heap slots with GFX10 V# descriptors for a
        // 512-byte arena constant buffer (two 256-byte CBVs: A at +0, B at +256) and null CBVs.
        {
            const UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            EXPECT(inc == 32);
            const auto cb_desc = buffer_desc(512, D3D12_RESOURCE_FLAG_NONE);
            Ref<ID3D12Resource> cb_upload, cb_default, cb_readback;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &cb_desc,
                                                             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                             IID_PPV_ARGS(&cb_upload.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &cb_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&cb_default.p))));
            const auto cb_rb_desc = buffer_desc(4 * bytes, D3D12_RESOURCE_FLAG_NONE);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &cb_rb_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&cb_readback.p))));
            void* cb_mapped = nullptr;
            EXPECT(SUCCEEDED(cb_upload->Map(0, nullptr, &cb_mapped)));
            auto* cb_words = static_cast<std::uint32_t*>(cb_mapped);
            for (UINT i = 0; i < elements; ++i) {
                cb_words[i] = 0x10000u + i;
                cb_words[elements + i] = 0x20000u + i;
            }
            const D3D12_GPU_VIRTUAL_ADDRESS cb_va = cb_default->GetGPUVirtualAddress();
            EXPECT(cb_va >= 0x10000 && cb_va < (1ull << 32));

            const auto slot_cpu = [&](UINT idx) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu + idx * inc)}; };
            const auto slot_gpu = [&](UINT idx) { return D3D12_GPU_DESCRIPTOR_HANDLE{gpu + idx * inc}; };

            // Pre-fill slot 3 with non-zero bytes to prove null CBV zeroes the entire 32-byte slot.
            std::memset(reinterpret_cast<void*>(slot_cpu(3).ptr), 0xA5, inc);
            device->CreateConstantBufferView(nullptr, slot_cpu(3));
            std::uint8_t zeros[32] = {};
            EXPECT(std::memcmp(reinterpret_cast<const void*>(slot_cpu(3).ptr), zeros, sizeof(zeros)) == 0);

            // Zero BufferLocation or zero SizeInBytes also produces a zeroed null V#.
            const D3D12_CONSTANT_BUFFER_VIEW_DESC zero_loc{0, 256};
            std::memset(reinterpret_cast<void*>(slot_cpu(3).ptr), 0x5A, inc);
            device->CreateConstantBufferView(&zero_loc, slot_cpu(3));
            EXPECT(std::memcmp(reinterpret_cast<const void*>(slot_cpu(3).ptr), zeros, sizeof(zeros)) == 0);
            const D3D12_CONSTANT_BUFFER_VIEW_DESC zero_size{cb_va, 0};
            std::memset(reinterpret_cast<void*>(slot_cpu(3).ptr), 0x5A, inc);
            device->CreateConstantBufferView(&zero_size, slot_cpu(3));
            EXPECT(std::memcmp(reinterpret_cast<const void*>(slot_cpu(3).ptr), zeros, sizeof(zeros)) == 0);

            const D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_a{cb_va, 256};
            const D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_b{cb_va + 256, 256};
            device->CreateConstantBufferView(&cbv_a, slot_cpu(1));
            device->CreateConstantBufferView(&cbv_b, slot_cpu(2));

            // Decode the written V# in slots 1 and 2 and verify every field + zeroed trailing 16 bytes.
            for (const auto& [slot_idx, expected_va] :
                 {std::pair{1u, std::uint64_t(cb_va)}, std::pair{2u, std::uint64_t(cb_va + 256)}}) {
                leonida::gfx10::BufferWords raw{};
                std::memcpy(raw.data(), reinterpret_cast<const void*>(slot_cpu(slot_idx).ptr), sizeof(raw));
                leonida::gfx10::Buffer decoded{};
                EXPECT(leonida::gfx10::decode(raw, decoded));
                EXPECT(decoded.address == expected_va);
                EXPECT(decoded.stride == 0);
                EXPECT(decoded.num_records == 256);
                EXPECT(decoded.format == leonida::gfx10::format_from_vulkan(VK_FORMAT_R32_UINT));
                EXPECT(decoded.swizzle == leonida::gfx10::identity_swizzle);
                EXPECT(decoded.oob_select == leonida::gfx10::OobSelect::raw);
                EXPECT(std::memcmp(reinterpret_cast<const std::uint8_t*>(slot_cpu(slot_idx).ptr) + 16, zeros, 16) == 0);
            }

            // Copy CBV data into the arena buffer, then run 4 dispatches:
            //   1) CBV A at byte offset 0 (all 64 dwords in-bounds)
            //   2) CBV A at byte offset 128 (dwords 0..31 in-bounds, dwords 32..63 OOB -> 0)
            //   3) CBV B at byte offset 0 (negative control: all 64 dwords flip to 0x20000 + i)
            //   4) Null CBV at byte offset 0 (negative control: all 64 dwords become 0)
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, pso.p)));
            list->CopyBufferRegion(cb_default.p, 0, cb_upload.p, 0, 512);
            const D3D12_RESOURCE_BARRIER cb_ready[] = {
                transition(cb_default.p, D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER),
                transition(out_a.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            };
            list->ResourceBarrier(2, cb_ready);
            list->SetDescriptorHeaps(1, heaps);
            list->SetComputeRootSignature(root.p);
            list->SetComputeRootUnorderedAccessView(1, va_a);

            const struct CbvDispatch {
                UINT byte_offset;
                UINT slot_idx;
                UINT readback_offset;
            } cbv_dispatches[] = {
                {0, 1, 0 * bytes},
                {128, 1, 1 * bytes},
                {0, 2, 2 * bytes},
                {0, 3, 3 * bytes},
            };
            for (const auto& d_cfg : cbv_dispatches) {
                const std::uint32_t rc[2] = {d_cfg.byte_offset, 0};  // s1 == 0 selects CBV V# mode
                list->SetComputeRoot32BitConstants(0, 2, rc, 0);
                list->SetComputeRootDescriptorTable(2, slot_gpu(d_cfg.slot_idx));
                list->Dispatch(1, 1, 1);
                const auto to_src =
                    transition(out_a.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                list->ResourceBarrier(1, &to_src);
                list->CopyBufferRegion(cb_readback.p, d_cfg.readback_offset, out_a.p, 0, bytes);
                const auto to_uav =
                    transition(out_a.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                list->ResourceBarrier(1, &to_uav);
            }
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 2)));
            EXPECT(SUCCEEDED(fence->SetEventOnCompletion(2, nullptr)));

            void* cb_res = nullptr;
            EXPECT(SUCCEEDED(cb_readback->Map(0, nullptr, &cb_res)));
            const auto* src_bytes = static_cast<const std::uint8_t*>(cb_res);
            std::memcpy(cbv_a_out.data(), src_bytes + 0 * bytes, bytes);
            std::memcpy(cbv_oob_out.data(), src_bytes + 1 * bytes, bytes);
            std::memcpy(cbv_b_out.data(), src_bytes + 2 * bytes, bytes);
            std::memcpy(cbv_null_out.data(), src_bytes + 3 * bytes, bytes);
        }

        // CreateShaderResourceView(buffer): raw, structured, and typed GFX10 V# descriptors in the shader-visible heap.
        {
            const UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            EXPECT(inc == 32);
            // 1024-byte arena buffer = 256 uint32 words = 64 16-byte structs / float4 elements.
            constexpr UINT buf_bytes = 1024;
            const auto srv_buf_desc = buffer_desc(buf_bytes, D3D12_RESOURCE_FLAG_NONE);
            const auto deny_buf_desc = buffer_desc(256, D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
            Ref<ID3D12Resource> srv_upload, srv_default, srv_deny, srv_readback;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &srv_buf_desc,
                                                             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                             IID_PPV_ARGS(&srv_upload.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &srv_buf_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&srv_default.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &deny_buf_desc,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&srv_deny.p))));
            const auto srv_rb_desc = buffer_desc(6 * bytes, D3D12_RESOURCE_FLAG_NONE);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &srv_rb_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&srv_readback.p))));
            void* srv_mapped = nullptr;
            EXPECT(SUCCEEDED(srv_upload->Map(0, nullptr, &srv_mapped)));
            auto* words = static_cast<std::uint32_t*>(srv_mapped);
            for (UINT w = 0; w < buf_bytes / 4; ++w) words[w] = 0x50000u + w;
            const D3D12_GPU_VIRTUAL_ADDRESS srv_va = srv_default->GetGPUVirtualAddress();
            EXPECT(srv_va >= 0x10000 && srv_va < (1ull << 32));

            const auto slot_cpu = [&](UINT idx) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu + idx * inc)}; };
            const auto slot_gpu = [&](UINT idx) { return D3D12_GPU_DESCRIPTOR_HANDLE{gpu + idx * inc}; };
            std::uint8_t zeros[32] = {};

            // Slot 1: Raw buffer SRV A (FirstElement = 0, NumElements = 32 -> 128 bytes; words 32..63 must read 0).
            D3D12_SHADER_RESOURCE_VIEW_DESC raw_a{};
            raw_a.Format = DXGI_FORMAT_R32_TYPELESS;
            raw_a.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            raw_a.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            raw_a.Buffer = {0, 32, 0, D3D12_BUFFER_SRV_FLAG_RAW};
            device->CreateShaderResourceView(srv_default.p, &raw_a, slot_cpu(1));

            // Slot 2: Raw buffer SRV B (FirstElement = 64, NumElements = 32 -> negative control shifted by +64 dwords).
            D3D12_SHADER_RESOURCE_VIEW_DESC raw_b = raw_a;
            raw_b.Buffer.FirstElement = 64;
            device->CreateShaderResourceView(srv_default.p, &raw_b, slot_cpu(2));

            // Slot 3: Structured buffer SRV A (stride 16, FirstElement = 0, NumElements = 32 -> elements 32..63 OOB).
            D3D12_SHADER_RESOURCE_VIEW_DESC struct_a{};
            struct_a.Format = DXGI_FORMAT_UNKNOWN;
            struct_a.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            struct_a.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            struct_a.Buffer = {0, 32, 16, D3D12_BUFFER_SRV_FLAG_NONE};
            device->CreateShaderResourceView(srv_default.p, &struct_a, slot_cpu(3));

            // Slot 4: Structured buffer SRV B (stride 16, FirstElement = 16, NumElements = 32 -> shifted +16 structs).
            D3D12_SHADER_RESOURCE_VIEW_DESC struct_b = struct_a;
            struct_b.Buffer.FirstElement = 16;
            device->CreateShaderResourceView(srv_default.p, &struct_b, slot_cpu(4));

            // Slot 5: Typed buffer SRV (R32G32B32A32_FLOAT, stride 16, FirstElement = 8, NumElements = 24, custom swizzle).
            D3D12_SHADER_RESOURCE_VIEW_DESC typed_desc{};
            typed_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
            typed_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            typed_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(2, 1, 0, 5);  // B,G,R,1
            typed_desc.Buffer = {8, 24, 0, D3D12_BUFFER_SRV_FLAG_NONE};
            device->CreateShaderResourceView(srv_default.p, &typed_desc, slot_cpu(5));

            // Slot 6: Typed RGB96 buffer SRV (R32G32B32_FLOAT, stride 12, FirstElement = 2, NumElements = 10).
            D3D12_SHADER_RESOURCE_VIEW_DESC rgb96_desc{};
            rgb96_desc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
            rgb96_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            rgb96_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            rgb96_desc.Buffer = {2, 10, 0, D3D12_BUFFER_SRV_FLAG_NONE};
            device->CreateShaderResourceView(srv_default.p, &rgb96_desc, slot_cpu(6));

            // CPU V# decode verification for slots 1..6.
            const auto decode_slot = [&](UINT idx, leonida::gfx10::Buffer& dec) -> bool {
                leonida::gfx10::BufferWords rw{};
                std::memcpy(rw.data(), reinterpret_cast<const void*>(slot_cpu(idx).ptr), sizeof(rw));
                return leonida::gfx10::decode(rw, dec) &&
                       std::memcmp(reinterpret_cast<const std::uint8_t*>(slot_cpu(idx).ptr) + 16, zeros, 16) == 0;
            };
            {
                leonida::gfx10::Buffer d1{}, d2{}, d3{}, d4{}, d5{}, d6{};
                EXPECT(decode_slot(1, d1));
                EXPECT(d1.address == srv_va && d1.stride == 0 && d1.num_records == 128 &&
                       d1.format == leonida::gfx10::format_from_vulkan(VK_FORMAT_R32_UINT) &&
                       d1.oob_select == leonida::gfx10::OobSelect::raw);
                EXPECT(decode_slot(2, d2));
                EXPECT(d2.address == srv_va + 256 && d2.stride == 0 && d2.num_records == 128 &&
                       d2.oob_select == leonida::gfx10::OobSelect::raw);
                EXPECT(decode_slot(3, d3));
                EXPECT(d3.address == srv_va && d3.stride == 16 && d3.num_records == 32 &&
                       d3.format == leonida::gfx10::format_from_vulkan(VK_FORMAT_R32_UINT) &&
                       d3.oob_select == leonida::gfx10::OobSelect::structured);
                EXPECT(decode_slot(4, d4));
                EXPECT(d4.address == srv_va + 256 && d4.stride == 16 && d4.num_records == 32 &&
                       d4.oob_select == leonida::gfx10::OobSelect::structured);
                EXPECT(decode_slot(5, d5));
                const leonida::gfx10::Swizzle want_swz = {leonida::gfx10::select_z, leonida::gfx10::select_y,
                                                          leonida::gfx10::select_x, leonida::gfx10::select_one};
                EXPECT(d5.address == srv_va + 8 * 16 && d5.stride == 16 && d5.num_records == 24 &&
                       d5.format == leonida::gfx10::format_from_vulkan(VK_FORMAT_R32G32B32A32_SFLOAT) &&
                       d5.swizzle == want_swz && d5.oob_select == leonida::gfx10::OobSelect::structured);
                EXPECT(decode_slot(6, d6));
                EXPECT(d6.address == srv_va + 2 * 12 && d6.stride == 12 && d6.num_records == 10 &&
                       d6.format == leonida::gfx10::format_from_vulkan(VK_FORMAT_R32G32B32_SFLOAT) &&
                       d6.oob_select == leonida::gfx10::OobSelect::structured);
            }

            // Invalid/refused buffer SRVs must all leave slot 7 zeroed.
            const auto expect_zero_srv = [&](ID3D12Resource* res, const D3D12_SHADER_RESOURCE_VIEW_DESC* d) -> bool {
                std::memset(reinterpret_cast<void*>(slot_cpu(7).ptr), 0xA5, inc);
                device->CreateShaderResourceView(res, d, slot_cpu(7));
                return std::memcmp(reinterpret_cast<const void*>(slot_cpu(7).ptr), zeros, sizeof(zeros)) == 0;
            };
            EXPECT(expect_zero_srv(srv_default.p, nullptr));          // null desc on buffer
            EXPECT(expect_zero_srv(srv_deny.p, &raw_a));              // DENY_SHADER_RESOURCE
            auto bad = raw_a;
            bad.Format = DXGI_FORMAT_R32_UINT;
            EXPECT(expect_zero_srv(srv_default.p, &bad));             // RAW requires R32_TYPELESS
            bad = raw_a;
            bad.Buffer.StructureByteStride = 4;
            EXPECT(expect_zero_srv(srv_default.p, &bad));             // RAW requires stride 0
            bad = struct_a;
            bad.Format = DXGI_FORMAT_R32_FLOAT;
            EXPECT(expect_zero_srv(srv_default.p, &bad));             // structured requires DXGI_FORMAT_UNKNOWN
            bad = struct_a;
            bad.Buffer.FirstElement = 60;
            bad.Buffer.NumElements = 10;
            EXPECT(expect_zero_srv(srv_default.p, &bad));             // out of bounds (70 * 16 > 1024)
            EXPECT(expect_zero_srv(nullptr, &raw_a));                 // null resource -> null V#

            // Upload buffer contents to arena and dispatch 6 GPU passes through the shader-visible descriptor table.
            // out_a is already in D3D12_RESOURCE_STATE_UNORDERED_ACCESS at the end of the CBV pass.
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, pso.p)));
            list->CopyBufferRegion(srv_default.p, 0, srv_upload.p, 0, buf_bytes);
            const auto srv_ready = transition(srv_default.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &srv_ready);
            list->SetDescriptorHeaps(1, heaps);
            list->SetComputeRootSignature(root.p);
            list->SetComputeRootUnorderedAccessView(1, va_a);

            const struct SrvDispatch {
                UINT slot_idx;
                UINT readback_offset;
            } srv_dispatches[] = {
                {1, 0 * bytes},
                {2, 1 * bytes},
                {3, 2 * bytes},
                {4, 3 * bytes},
                {5, 4 * bytes},
                {7, 5 * bytes},
            };
            const std::uint32_t srv_mode[2] = {0u, 0xFFFFFFFFu};  // s1 == 0xFFFFFFFF selects buffer SRV V# mode
            list->SetComputeRoot32BitConstants(0, 2, srv_mode, 0);
            for (const auto& cfg : srv_dispatches) {
                list->SetComputeRootDescriptorTable(2, slot_gpu(cfg.slot_idx));
                list->Dispatch(1, 1, 1);
                const auto to_src =
                    transition(out_a.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                list->ResourceBarrier(1, &to_src);
                list->CopyBufferRegion(srv_readback.p, cfg.readback_offset, out_a.p, 0, bytes);
                const auto to_uav =
                    transition(out_a.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                list->ResourceBarrier(1, &to_uav);
            }
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 3)));
            EXPECT(SUCCEEDED(fence->SetEventOnCompletion(3, nullptr)));

            void* srv_res = nullptr;
            EXPECT(SUCCEEDED(srv_readback->Map(0, nullptr, &srv_res)));
            const auto* rb = static_cast<const std::uint8_t*>(srv_res);
            std::memcpy(srv_raw_a.data(), rb + 0 * bytes, bytes);
            std::memcpy(srv_raw_b.data(), rb + 1 * bytes, bytes);
            std::memcpy(srv_struct_a.data(), rb + 2 * bytes, bytes);
            std::memcpy(srv_struct_b.data(), rb + 3 * bytes, bytes);
            std::memcpy(srv_typed.data(), rb + 4 * bytes, bytes);
            std::memcpy(srv_null.data(), rb + 5 * bytes, bytes);
        }

        for (auto* r : {&out_a, &out_b, &out_c, &constants, &readback}) r->reset();
        pso67.reset();
        unknown67.reset();
        pso.reset();
        pso67.reset();  // the slot-67 PSO holds the device like any other; released before the count is checked
        unknown.reset();
        unknown67.reset();
        junk.reset();
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
        const std::uint32_t a = 7 + 3 * i + table_values[i % 4] + code_word;
        const std::uint32_t b = 100 + 3 * i + table_values[i % 4] + code_word;
        const std::uint32_t c = 5 * i + table_values[i % 4] + code_word;
        if ((first[i] != a || second[i] != b || third[i] != c) && mismatches++ < 4)
            std::fprintf(stderr, "element %u: got %u, %u and %u, want %u, %u and %u\n", i, first[i], second[i],
                         third[i], a, b, c);
    }
    EXPECT(!mismatches);
    UINT cbv_b_diff = 0, cbv_null_diff = 0;
    for (std::uint32_t i = 0; i < elements; ++i) {
        const std::uint32_t want_a = 0x10000u + i;
        const std::uint32_t want_oob = i < 32 ? (0x10000u + 32u + i) : 0u;
        const std::uint32_t want_b = 0x20000u + i;
        if (cbv_a_out[i] != want_a && mismatches++ < 4)
            std::fprintf(stderr, "CBV A[%u]: got 0x%x want 0x%x\n", i, cbv_a_out[i], want_a);
        if (cbv_oob_out[i] != want_oob && mismatches++ < 4)
            std::fprintf(stderr, "CBV OOB[%u]: got 0x%x want 0x%x\n", i, cbv_oob_out[i], want_oob);
        if (cbv_b_out[i] != want_b && mismatches++ < 4)
            std::fprintf(stderr, "CBV B[%u]: got 0x%x want 0x%x\n", i, cbv_b_out[i], want_b);
        if (cbv_null_out[i] != 0u && mismatches++ < 4)
            std::fprintf(stderr, "CBV null[%u]: got 0x%x want 0\n", i, cbv_null_out[i]);
        if (cbv_b_out[i] != cbv_a_out[i]) ++cbv_b_diff;
        if (cbv_null_out[i] != cbv_a_out[i]) ++cbv_null_diff;
    }
    EXPECT(!mismatches);
    EXPECT(cbv_b_diff == elements && cbv_null_diff == elements);
    std::printf("CreateConstantBufferView: 64/64 dwords matched for CBV A, 32 in-bounds + 32 OOB-zero matched at "
                "+128, negative controls flipped %u/%u (CBV B) and %u/%u (null CBV)\n",
                cbv_b_diff, elements, cbv_null_diff, elements);
    UINT raw_diff = 0, struct_diff = 0, null_diff = 0;
    for (std::uint32_t i = 0; i < elements; ++i) {
        const std::uint32_t want_raw_a = i < 32 ? (0x50000u + i) : 0u;
        const std::uint32_t want_raw_b = i < 32 ? (0x50000u + 64u + i) : 0u;
        const std::uint32_t want_struct_a = i < 32 ? (0x50000u + i * 4u) : 0u;
        const std::uint32_t want_struct_b = i < 32 ? (0x50000u + (16u + i) * 4u) : 0u;
        const std::uint32_t want_typed = i < 24 ? (0x50000u + (8u + i) * 4u) : 0u;
        if (srv_raw_a[i] != want_raw_a && mismatches++ < 4)
            std::fprintf(stderr, "SRV raw A[%u]: got 0x%x want 0x%x\n", i, srv_raw_a[i], want_raw_a);
        if (srv_raw_b[i] != want_raw_b && mismatches++ < 4)
            std::fprintf(stderr, "SRV raw B[%u]: got 0x%x want 0x%x\n", i, srv_raw_b[i], want_raw_b);
        if (srv_struct_a[i] != want_struct_a && mismatches++ < 4)
            std::fprintf(stderr, "SRV struct A[%u]: got 0x%x want 0x%x\n", i, srv_struct_a[i], want_struct_a);
        if (srv_struct_b[i] != want_struct_b && mismatches++ < 4)
            std::fprintf(stderr, "SRV struct B[%u]: got 0x%x want 0x%x\n", i, srv_struct_b[i], want_struct_b);
        if (srv_typed[i] != want_typed && mismatches++ < 4)
            std::fprintf(stderr, "SRV typed[%u]: got 0x%x want 0x%x\n", i, srv_typed[i], want_typed);
        if (srv_null[i] != 0u && mismatches++ < 4)
            std::fprintf(stderr, "SRV null[%u]: got 0x%x want 0\n", i, srv_null[i]);
        if (srv_raw_b[i] != srv_raw_a[i]) ++raw_diff;
        if (srv_struct_b[i] != srv_struct_a[i]) ++struct_diff;
        if (srv_null[i] != srv_raw_a[i]) ++null_diff;
    }
    EXPECT(!mismatches);
    EXPECT(raw_diff == 32 && struct_diff == 32 && null_diff == 32);
    std::printf("CreateShaderResourceView(buffer): raw (32 in-bounds + 32 OOB-zero), structured (32 + 32 OOB-zero), "
                "typed (24 + 40 OOB-zero) matched; negative controls flipped %u/32 (raw B), %u/32 (struct B), %u/32 "
                "(null)\n",
                raw_diff, struct_diff, null_diff);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("compute: root constants, a root UAV and a descriptor table reached the shader through the GPU "
                "address arena; %u elements matched in each of three dispatches\n", elements);
    return 0;
}
