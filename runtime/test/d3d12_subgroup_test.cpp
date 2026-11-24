// Subgroup size control in PSOs, end to end through the Xbox-ABI proxy over the Vulkan device (src/subgroup_size.h):
// three translated compute shaders from runtime/test/shaders, each dispatched through a D3D12 PSO found by the
// SHA-256 of its FSSM program, writing eight words through the V# in s[4:7] (four root constants in the console compute layout), read back:
//   w64 native    cs_w64x_readlane translated natively: v_readlane across both halves of a wave64. Exact only in a
//                 64-wide subgroup, so the PSO must ask for 64 (and full subgroups) on a device that can give it.
//                 On a device that cannot (NVIDIA: 32 only) the PSO reports it, and the result is not judged.
//   w64 emulated  the same shader translated with --wave64-emulate: exact on any size, the PSO asks for nothing.
//   w32 native    cs_wqm_w32, a wave32 shader with whole-quad mode on lane masks; the PSO asks for 32 where the
//                 device's default or range would otherwise change it.
// Each case also prints a control run with the layer's size request switched off, to show what the request changes
// on this GPU; the control is not judged. No game data. Set LEONIDA_VK_VALIDATION=1 to run under
// VK_LAYER_KHRONOS_validation; validation errors fail the test.
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"
#include "../shims/d3d12_xs_device.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#ifndef LEONIDA_TEST_SUBGROUP_SHADERS
#error "LEONIDA_TEST_SUBGROUP_SHADERS must name the folder with the translated test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XboxComputePipelineFn = HRESULT(WINAPI*)(void* device, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, std::uint64_t,
                                               std::uint64_t, const IID*, void**);
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX
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

// runtime/test/shaders/compute_fixture.py's header for a compute shader without "; group:" or "; rsrc2:" lines:
// 0x148 bytes, COMPUTE_PGM_RSRC2 0x8088 (four user SGPRs, TGID_X, 512 bytes of LDS) at 0x138 and a 64x1x1 group.
// The synthetic header build.py gives these fixtures: compute_fixture.py's default RSRC2 0x8088 (four user SGPRs,
// TGID_X) with USER_SGPR raised by LEONIDA_USER_DATA_SHIFT=4, so 0x8090 (eight: the V# moves to s[4:7], D3D-448).
std::vector<std::uint8_t> fixture_fssm() {
    std::vector<std::uint8_t> fssm(0x138, 0);
    for (const std::uint32_t v : {0x8090u, 64u, 1u, 1u}) put32(fssm, v);
    return fssm;
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

using Words = std::array<std::uint32_t, 8>;

struct Shader {
    const char* label;
    const char* folder;  // under LEONIDA_TEST_SUBGROUP_SHADERS; its spirv/ holds index.json and the module
    Words expected;
};

constexpr Shader w64_native{"w64 native (cs_w64x_readlane)", "w64_native", {121, 121, 16, 97, 16, 100, 100, 121}};
constexpr Shader w64_emulated{"w64 emulated (cs_w64x_readlane, --wave64-emulate)", "w64_emulated",
                              {121, 121, 16, 97, 16, 100, 100, 121}};
constexpr Shader w32_native{"w32 native (cs_wqm_w32)", "w32_native", {12, 1, 4, 0, 8, 15728880, 4, 32}};

struct Result {
    int failure = 0;      // a line number, 0 when the case ran
    Words words{};
    std::vector<leonida::SubgroupRequest> requests;
};

// One dispatch of the shader's PSO on a device created with `caps` as what the layer knows of the GPU.
Result run(vktest::Device& d, const std::filesystem::path& library, const leonida::SubgroupSizeCaps& caps) {
    Result result;
    const auto fail = [&](int line) {
        result.failure = line;
        return result;
    };
#define CASE_EXPECT(condition) do { if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return fail(__LINE__); } } while (0)

    const auto fssm = fixture_fssm();
    const auto container = shader_container(fssm);
    CASE_EXPECT(leonida::fssm_program(container) == fssm);
    {
        leonida::ShaderLibrary index;
        CASE_EXPECT(index.load(library.string()));
        CASE_EXPECT(index.find("cs", leonida::sha256_hex(fssm)));  // the translator's header is the fixture's
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = true;
    info.shader_dir = library.string();
    info.subgroup = caps;
    ID3D12Device* inner = nullptr;
    CASE_EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    CASE_EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                            reinterpret_cast<void**>(&device.p))));
    rt::d3d12::set_device_outer(inner, device.p);

    // Root signature: [0] four constants (s4..s7 in the console compute layout, D3D-448), the V# the shaders store
    // through. The asm fixtures are built with LEONIDA_USER_DATA_SHIFT=4 to read it there (runtime/CMakeLists.txt).
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants = {0, 0, 4};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto root_blob = serialize(root_desc);
    CASE_EXPECT(!root_blob.empty());
    Ref<ID3D12RootSignature> root;
    CASE_EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.p;
    pso_desc.CS = {container.data(), container.size()};
    const IID pso_iid = __uuidof(ID3D12PipelineState);
    Ref<ID3D12PipelineState> pso;
    CASE_EXPECT(SUCCEEDED(reinterpret_cast<XboxComputePipelineFn>(slot(device.p, 62))(
        device.p, &pso_desc, 0, 0, &pso_iid, reinterpret_cast<void**>(&pso.p))));
    const auto* requests = rt::d3d12::pipeline_subgroup_requests(pso.p);
    CASE_EXPECT(requests && requests->size() == 1);
    result.requests = *requests;

    constexpr UINT bytes = 256;
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> out, readback;
    const auto uav = buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    CASE_EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav,
                                                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                          IID_PPV_ARGS(&out.p))));
    const auto readback_desc = buffer_desc(bytes, D3D12_RESOURCE_FLAG_NONE);
    CASE_EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                          IID_PPV_ARGS(&readback.p))));
    const D3D12_GPU_VIRTUAL_ADDRESS va = out->GetGPUVirtualAddress();
    CASE_EXPECT(va >= 0x10000 && va < (1ull << 32));

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    CASE_EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CASE_EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CASE_EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                    IID_PPV_ARGS(&list.p))));
    CASE_EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    // The V# the shaders read from s[4:7]: base = the buffer's console VA (the shader adds the heap base), stride 0,
    // num_records unlimited, OOB_SELECT 3 (raw: a stride of 0 is a bounds check in the other modes).
    list->SetComputeRootSignature(root.p);
    const std::uint32_t vsharp[4] = {std::uint32_t(va), std::uint32_t(va >> 32) & 0xffff, 0xffffffffu, 0x31016FACu};
    list->SetComputeRoot32BitConstants(0, 4, vsharp, 0);
    list->Dispatch(1, 1, 1);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {out.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(readback.p, 0, out.p, 0, bytes);
    CASE_EXPECT(list->Close() == S_OK);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CASE_EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    CASE_EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
    CASE_EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
    void* mapped = nullptr;
    CASE_EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    std::memcpy(result.words.data(), mapped, sizeof(result.words));

    for (auto* r : {&out, &readback}) r->reset();
    pso.reset();
    root.reset();
    list.reset();
    allocator.reset();
    queue.reset();
    fence.reset();
    CASE_EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
#undef CASE_EXPECT
    return result;
}

std::string words_text(const Words& w) {
    std::string s;
    for (const std::uint32_t v : w) s += (s.empty() ? "" : " ") + std::to_string(v);
    return s;
}

// What the decision function says about the shader's own index entry on this device.
leonida::SubgroupRequest expected_request(const Shader& shader, const leonida::SubgroupSizeCaps& caps) {
    leonida::ShaderLibrary index;
    const auto dir = std::filesystem::path(LEONIDA_TEST_SUBGROUP_SHADERS) / shader.folder / "spirv";
    if (!index.load(dir.string())) return {};
    const auto* entry = index.find("cs", leonida::sha256_hex(fixture_fssm()));
    return entry ? leonida::decide_subgroup_size(*entry, caps) : leonida::SubgroupRequest{};
}

std::filesystem::path shader_dir(const Shader& shader) {
    return std::filesystem::path(LEONIDA_TEST_SUBGROUP_SHADERS) / shader.folder / "spirv";
}

// ---- Probe shaders (GLSL, test/glsl) and the graphics path -----------------------------------------------------------
// What the translator would write into an index entry for a shader of each kind.
struct Mode {
    const char* label;
    bool wave64;
    std::uint32_t size;  // subgroup_size
    bool emulated;       // --wave64-emulate: subgroup_size null, wave64_exchange "workgroup"
};
constexpr Mode mode_w64{"wave64, subgroup_size 64", true, 64, false};
constexpr Mode mode_w32{"wave32, subgroup_size 32", false, 32, false};
constexpr Mode mode_emulated{"emulated (wave64_exchange)", true, 0, true};

std::string mode_json(const Mode& m) {
    std::string s = std::string("\"wave64\": ") + (m.wave64 ? "true" : "false") + ", \"subgroup_size\": " +
                    (m.emulated ? std::string("null") : std::to_string(m.size));
    if (m.emulated) s += ", \"wave64_exchange\": \"workgroup\"";
    return s;
}

struct LibraryEntry {
    std::string name, stage, tail;  // tail: the rest of the index entry's members
    std::filesystem::path spv;
};

// A translator-style folder with the entries' modules, every entry keyed by the fixture program's hash.
std::filesystem::path make_library(const std::string& tag, const std::vector<LibraryEntry>& entries) {
    const auto dir = testfiles::process_temp_dir("leonida_subgroup_" + tag);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::ofstream index(dir / "index.json", std::ios::binary);
    index << "{";
    bool first = true;
    for (const LibraryEntry& e : entries) {
        testfiles::copy_file(e.spv, dir / (e.name + ".spv"));  // a missing module is reported, and the PSO fails
        index << (first ? "" : ",\n") << "\"" << e.name << "\": {\"stage\": \"" << e.stage << "\", \"fssm_sha256\": \""
              << leonida::sha256_hex(fixture_fssm()) << "\", \"bindings\": [], \"push_constant_bytes\": 144, "
              << e.tail << "}";
        first = false;
    }
    index << "}";
    return dir;
}

std::filesystem::path glsl_spv(const char* name) {
    return std::filesystem::path(LEONIDA_TEST_SUBGROUP_SHADERS) / "glsl" / (std::string(name) + ".spv");
}

// The compute probe: a library with cs_sg_probe in the given mode.
std::filesystem::path compute_probe_library(const Mode& mode) {
    return make_library("cs_probe", {{"cs_sg_probe", "cs",
                                      "\"local_size\": [64, 1, 1], \"uses_code_base\": false, " + mode_json(mode),
                                      glsl_spv("cs_sg_probe.comp")}});
}

constexpr UINT draw_size = 64;
using Pixel = std::array<std::uint32_t, 4>;

struct DrawResult {
    int failure = 0;
    std::vector<Pixel> pixels;  // draw_size * draw_size
    std::vector<leonida::SubgroupRequest> requests;  // vertex, pixel
};

// One draw of a full-viewport triangle through a graphics PSO whose VS is vs_fullscreen and whose PS is `ps` of the
// library, into a 64x64 R32G32B32A32_FLOAT target that is read back as bit patterns.
DrawResult draw(vktest::Device& d, const std::filesystem::path& library, const leonida::SubgroupSizeCaps& caps) {
    DrawResult result;
    const auto fail = [&](int line) {
        result.failure = line;
        return result;
    };
#define CASE_EXPECT(condition) do { if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return fail(__LINE__); } } while (0)
    const auto container = shader_container(fixture_fssm());

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = true;
    info.shader_dir = library.string();
    info.subgroup = caps;
    ID3D12Device* inner = nullptr;
    CASE_EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    CASE_EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                            reinterpret_cast<void**>(&device.p))));
    rt::d3d12::set_device_outer(inner, device.p);

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto root_blob = serialize(root_desc);
    CASE_EXPECT(!root_blob.empty());
    Ref<ID3D12RootSignature> root;
    CASE_EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.p;
    pso_desc.VS = {container.data(), container.size()};
    pso_desc.PS = {container.data(), container.size()};
    pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso_desc.SampleMask = 0xffffffff;
    pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso_desc.RasterizerState.DepthClipEnable = TRUE;
    pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    pso_desc.SampleDesc.Count = 1;
    Ref<ID3D12PipelineState> pso;
    CASE_EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));
    const auto* requests = rt::d3d12::pipeline_subgroup_requests(pso.p);
    CASE_EXPECT(requests && requests->size() == 2);
    result.requests = *requests;

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    constexpr UINT pitch = draw_size * 16;
    Ref<ID3D12Resource> readback, target;
    const auto readback_desc = buffer_desc(pitch * draw_size, D3D12_RESOURCE_FLAG_NONE);
    CASE_EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                          IID_PPV_ARGS(&readback.p))));
    D3D12_RESOURCE_DESC rt_desc{};
    rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rt_desc.Width = rt_desc.Height = draw_size;
    rt_desc.DepthOrArraySize = 1;
    rt_desc.MipLevels = 1;
    rt_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    rt_desc.SampleDesc.Count = 1;
    rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    CASE_EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          IID_PPV_ARGS(&target.p))));
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> rtv_heap;
    CASE_EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv{SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
    device->CreateRenderTargetView(target.p, nullptr, rtv);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    CASE_EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CASE_EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CASE_EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                    IID_PPV_ARGS(&list.p))));
    CASE_EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    D3D12_RESOURCE_BARRIER to_target{};
    to_target.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_target.Transition = {target.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_RENDER_TARGET};
    list->ResourceBarrier(1, &to_target);
    const FLOAT zero[4] = {0, 0, 0, 0};
    list->ClearRenderTargetView(rtv, zero, 0, nullptr);
    list->SetGraphicsRootSignature(root.p);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VIEWPORT viewport{0, 0, float(draw_size), float(draw_size), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(draw_size), LONG(draw_size)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->DrawInstanced(3, 1, 0, 0);
    D3D12_RESOURCE_BARRIER to_copy{};
    to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_copy.Transition = {target.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET,
                          D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &to_copy);
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = readback.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = {0, {DXGI_FORMAT_R32G32B32A32_FLOAT, draw_size, draw_size, 1, pitch}};
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = target.p;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    CASE_EXPECT(list->Close() == S_OK);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CASE_EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    CASE_EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
    CASE_EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
    void* mapped = nullptr;
    CASE_EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    result.pixels.resize(draw_size * draw_size);
    std::memcpy(result.pixels.data(), mapped, result.pixels.size() * sizeof(Pixel));

    for (auto* r : {&readback, &target}) r->reset();
    pso.reset();
    root.reset();
    rtv_heap.reset();
    list.reset();
    allocator.reset();
    queue.reset();
    fence.reset();
    CASE_EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
#undef CASE_EXPECT
    return result;
}

// The vertex shader of every draw. It is marked emulated so that it asks for nothing: only the pixel stage is under test.
LibraryEntry fullscreen_vs() {
    return {"vs_fullscreen", "vs", "\"input_signature\": [], " + mode_json(mode_emulated), glsl_spv("vs_sg_fullscreen.vert")};
}

LibraryEntry ps_probe(const Mode& mode) {
    return {"ps_sg_probe", "ps", "\"outputs\": [0], " + mode_json(mode), glsl_spv("ps_sg_probe.frag")};
}

// The wave64 pixel shader with readlane across both halves, translated by the build (test/shaders/ps_w64x_readlane.asm).
LibraryEntry ps_readlane(const Mode& mode) {
    return {"ps_w64x_readlane", "ps", "\"outputs\": [0], " + mode_json(mode),
            std::filesystem::path(LEONIDA_TEST_SUBGROUP_SHADERS) / "ps_w64_native" / "spirv" / "ps_w64x_readlane.spv"};
}

// The subgroup size a probe pixel reports, or 0 when the pixels disagree or do not look like a probe's.
std::uint32_t probed_size(const std::vector<Pixel>& pixels) {
    std::uint32_t size = 0;
    for (const Pixel& p : pixels) {
        if ((p[0] & 0xffff0000u) != 0x40000000u) return 0;
        const std::uint32_t s = p[0] & 0xffff;
        if (size && s != size) return 0;
        size = s;
    }
    return size;
}

const Pixel readlane_expected = {0x40000079u, 0x40000010u, 0x40000064u, 0x400000beu};

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    const leonida::SubgroupSizeCaps& caps = d.subgroup;
    leonida::SubgroupSizeCaps off = caps;  // the control: the layer knows nothing of the device's sizes
    off.min_size = off.max_size = off.default_size = 0;
    off.subgroup_size_control = off.compute_full_subgroups = false;
    off.required_stages = 0;

    unsigned judged = 0;
    for (const Shader* shader : {&w64_native, &w64_emulated, &w32_native}) {
        const leonida::SubgroupRequest want = expected_request(*shader, caps);
        const Result result = run(d, shader_dir(*shader), caps);
        EXPECT(!result.failure);
        EXPECT(result.requests.size() == 1);
        const leonida::SubgroupRequest& got = result.requests[0];
        // The PSO asked for what the decision function says.
        EXPECT(got.outcome == want.outcome && got.size == want.size && got.flags == want.flags);
        EXPECT(got.stage == VK_SHADER_STAGE_COMPUTE_BIT);

        const bool exact = result.words == shader->expected;
        std::printf("%s: outcome %d, required size %u, full subgroups %s; result %s (%s)\n", shader->label,
                    int(got.outcome), got.size, got.flags ? "yes" : "no", exact ? "exact" : "DIFFERS",
                    words_text(result.words).c_str());
        if (shader == &w64_native && got.outcome == leonida::SubgroupOutcome::unavailable) {
            // This GPU cannot give a wave64 compute shader 64 lanes. The layer said so (a log line, above); the
            // build for this GPU should have used --wave64-emulate, which the next case checks.
            std::printf("  this device has no 64-wide compute subgroup (sizes %u..%u): native wave64 not judged\n",
                        caps.min_size, caps.max_size);
        } else {
            if (!exact) std::fprintf(stderr, "%s: wanted %s\n", shader->label, words_text(shader->expected).c_str());
            EXPECT(exact);
            ++judged;
        }
        if (shader == &w64_native && got.outcome != leonida::SubgroupOutcome::unavailable) {
            EXPECT(got.outcome != leonida::SubgroupOutcome::unknown_device);
            EXPECT(got.outcome == leonida::SubgroupOutcome::required || got.outcome == leonida::SubgroupOutcome::device_fixed ||
                   got.outcome == leonida::SubgroupOutcome::device_default);
            if (got.outcome == leonida::SubgroupOutcome::required) {
                EXPECT(got.size == 64);
                EXPECT(!caps.compute_full_subgroups ||
                       got.flags == VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT);
            }
        }
        if (shader == &w64_emulated) EXPECT(got.outcome == leonida::SubgroupOutcome::emulated && !got.size && !got.flags);

        // Control, not judged: the same PSO with no size asked for.
        if (shader != &w64_emulated) {
            const Result control = run(d, shader_dir(*shader), off);
            EXPECT(!control.failure);
            std::printf("  control, nothing requested: %s (%s)\n", control.words == shader->expected ? "exact" : "differs",
                        words_text(control.words).c_str());
        }
    }
    // --- The driver's choice, observed: probe shaders that store gl_SubgroupSize ------------------------------------
    const auto exact_outcome = [](leonida::SubgroupOutcome o) {
        return o == leonida::SubgroupOutcome::required || o == leonida::SubgroupOutcome::device_fixed;
    };
    for (const Mode* mode : {&mode_w64, &mode_w32, &mode_emulated}) {
        const auto library = compute_probe_library(*mode);
        const Result r = run(d, library, caps);
        EXPECT(!r.failure && r.requests.size() == 1);
        const leonida::SubgroupRequest& req = r.requests[0];
        const std::uint32_t seen = r.words[0];
        std::printf("compute probe, %s: outcome %d (asked %u), the driver ran it %u-wide\n", mode->label,
                    int(req.outcome), req.size, seen);
        if (exact_outcome(req.outcome)) EXPECT(seen == req.wanted);  // the size the PSO asked for is the size it got
        const Result control = run(d, library, off);
        EXPECT(!control.failure);
        std::printf("  control, nothing requested: %u-wide (device default %u)\n", control.words[0], caps.default_size);
        // The request matters: a wave32 shader on a device whose default is not 32 runs at the default without it.
        if (mode == &mode_w32 && req.outcome == leonida::SubgroupOutcome::required && caps.default_size != 32)
            EXPECT(control.words[0] != 32);
        std::filesystem::remove_all(library);
    }

    std::uint32_t fragment_default = 0;
    {
        const auto library = make_library("ps_probe", {fullscreen_vs(), ps_probe(mode_w32)});
        const DrawResult control = draw(d, library, off);
        EXPECT(!control.failure);
        fragment_default = probed_size(control.pixels);
        EXPECT(fragment_default);
        std::printf("pixel probe, nothing requested: the driver runs pixel shaders %u-wide\n", fragment_default);
        std::filesystem::remove_all(library);
    }
    for (const Mode* mode : {&mode_w64, &mode_w32, &mode_emulated}) {
        const auto library = make_library("ps_probe", {fullscreen_vs(), ps_probe(*mode)});
        const DrawResult r = draw(d, library, caps);
        EXPECT(!r.failure && r.requests.size() == 2);
        const leonida::SubgroupRequest& vs_req = r.requests[0];
        const leonida::SubgroupRequest& req = r.requests[1];
        EXPECT(vs_req.outcome == leonida::SubgroupOutcome::emulated && vs_req.stage == VK_SHADER_STAGE_VERTEX_BIT);
        EXPECT(req.stage == VK_SHADER_STAGE_FRAGMENT_BIT);
        const std::uint32_t seen = probed_size(r.pixels);
        EXPECT(seen);
        std::printf("pixel probe, %s: outcome %d (asked %u), the driver ran it %u-wide\n", mode->label,
                    int(req.outcome), req.size, seen);
        if (exact_outcome(req.outcome)) EXPECT(seen == req.wanted);
        if (req.outcome == leonida::SubgroupOutcome::required && fragment_default != req.wanted)
            std::printf("  the request changes it: %u-wide without, %u-wide with\n", fragment_default, seen);
        std::filesystem::remove_all(library);
    }

    // --- A wave64 pixel shader with a cross-half v_readlane, through a graphics PSO ---------------------------------
    {
        const auto library = make_library("ps_readlane", {fullscreen_vs(), ps_readlane(mode_w64)});
        const auto count_exact = [](const std::vector<Pixel>& pixels) {
            std::size_t n = 0;
            for (const Pixel& p : pixels) n += p == readlane_expected;
            return n;
        };
        const DrawResult r = draw(d, library, caps);
        EXPECT(!r.failure && r.requests.size() == 2);
        const leonida::SubgroupRequest& req = r.requests[1];
        const std::size_t exact = count_exact(r.pixels), total = r.pixels.size();
        std::printf("wave64 pixel shader (ps_w64x_readlane): outcome %d, required size %u; %zu of %zu pixels exact\n",
                    int(req.outcome), req.size, exact, total);
        if (exact_outcome(req.outcome)) {
            if (exact != total)
                std::fprintf(stderr, "first pixel %08x %08x %08x %08x\n", r.pixels[0][0], r.pixels[0][1],
                             r.pixels[0][2], r.pixels[0][3]);
            EXPECT(exact == total);
            ++judged;
        } else {
            std::printf("  this device cannot require 64 for a wave64 pixel shader: not judged\n");
        }
        const DrawResult control = draw(d, library, off);
        EXPECT(!control.failure);
        const std::size_t control_exact = count_exact(control.pixels);
        std::printf("  control, nothing requested (pixel shaders run %u-wide): %zu of %zu pixels exact\n",
                    fragment_default, control_exact, total);
        // Where the default is not 64, the shader crosses halves that are not there: the request must matter.
        if (req.outcome == leonida::SubgroupOutcome::required && fragment_default != 64) EXPECT(control_exact < total);
        std::filesystem::remove_all(library);
    }
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("subgroup size: %u shaders judged exact through the layer's PSOs (sizes %u..%u, default %u)\n",
                judged, caps.min_size, caps.max_size, caps.default_size);
    return 0;
}
