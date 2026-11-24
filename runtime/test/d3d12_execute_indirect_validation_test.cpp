// Host-parity validation and GPU execution sanity test for ID3D12GraphicsCommandList::ExecuteIndirect.
// Compares Close() verdicts between the Vulkan layer, live Windows WARP, and live hardware D3D12 (0 disagreements),
// then executes valid ExecuteIndirect dispatches on the Vulkan GPU with validation enabled. No game data.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <dxgi1_4.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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

using namespace d3d12test;

namespace {

// Minimal valid SPIR-V 1.0 compute shader: `void main() {}` with LocalSize(1, 1, 1).
constexpr std::array<std::uint32_t, 35> kMinimalComputeSpv = {
    0x07230203u, 0x00010000u, 0u, 6u, 0u,
    (2u << 16) | 17u, 1u,                                 // OpCapability Shader
    (3u << 16) | 14u, 0u, 1u,                             // OpMemoryModel Logical GLSL450
    (5u << 16) | 15u, 5u, 4u, 0x6e69616du, 0x00000000u,   // OpEntryPoint GLCompute %4 "main"
    (6u << 16) | 16u, 4u, 17u, 1u, 1u, 1u,                // OpExecutionMode %4 LocalSize 1 1 1
    (2u << 16) | 19u, 2u,                                 // %2 = OpTypeVoid
    (3u << 16) | 33u, 3u, 2u,                             // %3 = OpTypeFunction %2
    (5u << 16) | 54u, 2u, 4u, 0u, 3u,                     // %4 = OpFunction %2 None %3
    (2u << 16) | 248u, 5u,                                // %5 = OpLabel
    (1u << 16) | 253u,                                    // OpReturn
    (1u << 16) | 56u,                                     // OpFunctionEnd
};

struct Env {
    ID3D12Device* device = nullptr;
    Ref<ID3D12Resource> buf256;
    Ref<ID3D12Resource> count256;
    Ref<ID3D12Resource> buf3;
    Ref<ID3D12Resource> tex;
    Ref<ID3D12CommandSignature> draw16;
    Ref<ID3D12CommandSignature> draw20;
    Ref<ID3D12CommandSignature> draw_indexed20;
    Ref<ID3D12CommandSignature> draw_indexed24;
    Ref<ID3D12CommandSignature> dispatch12;
    Ref<ID3D12CommandSignature> dispatch20;
    Ref<ID3D12RootSignature> empty_root;
    Ref<ID3D12PipelineState> compute_pso;
    ID3D12GraphicsCommandList* list = nullptr;

    bool init(ID3D12Device* d, const std::vector<std::uint8_t>* layer_cs_container = nullptr) {
        device = d;
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const auto mkbuf = [&](UINT64 bytes, Ref<ID3D12Resource>& out) {
            const auto desc = buffer_desc(bytes);
            return SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&out.p)));
        };
        CHECK(mkbuf(256, buf256));
        CHECK(mkbuf(256, count256));
        CHECK(mkbuf(3, buf3));

        D3D12_RESOURCE_DESC td{};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = 64;
        td.Height = 64;
        td.DepthOrArraySize = 1;
        td.MipLevels = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &td,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&tex.p))));

        const auto mksig = [&](D3D12_INDIRECT_ARGUMENT_TYPE type, UINT stride, Ref<ID3D12CommandSignature>& out) {
            D3D12_INDIRECT_ARGUMENT_DESC arg{};
            arg.Type = type;
            D3D12_COMMAND_SIGNATURE_DESC desc{stride, 1, &arg, 0};
            return SUCCEEDED(device->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&out.p)));
        };
        CHECK(mksig(D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, 16, draw16));
        CHECK(mksig(D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, 20, draw20));
        CHECK(mksig(D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, 20, draw_indexed20));
        CHECK(mksig(D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, 24, draw_indexed24));
        CHECK(mksig(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, 12, dispatch12));
        CHECK(mksig(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, 20, dispatch20));

        if (layer_cs_container) {
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC rs_desc{};
            rs_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
            rs_desc.Desc_1_0 = {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
            const auto serialized = serialize_root_signature(rs_desc);
            CHECK(!serialized.empty());
            CHECK(SUCCEEDED(device->CreateRootSignature(0, serialized.data(), serialized.size(),
                                                        IID_PPV_ARGS(&empty_root.p))));
            D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
            pso_desc.pRootSignature = empty_root.p;
            pso_desc.CS = {layer_cs_container->data(), layer_cs_container->size()};
            CHECK(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&compute_pso.p))));
        }
        return true;
    }

    void bind_compute_if_available() const {
        if (empty_root.p) list->SetComputeRootSignature(empty_root.p);
        if (compute_pso.p) list->SetPipelineState(compute_pso.p);
    }
};

struct Case {
    const char* name;
    bool expected_ok;  // Measured on Windows 11 WARP and RTX 5090 HW D3D12
    std::function<void(Env&)> record;
};

bool run_case(Env& env, const Case& c, HRESULT& out_hr) {
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    CHECK(SUCCEEDED(env.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
    CHECK(SUCCEEDED(env.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                  IID_PPV_ARGS(&list.p))));
    env.list = list.p;
    c.record(env);
    env.list = nullptr;
    out_hr = list->Close();
    return true;
}

std::vector<Case> parity_cases() {
    constexpr UINT64 kU64Max = std::numeric_limits<UINT64>::max();
    constexpr UINT kU32Max = std::numeric_limits<UINT>::max();
    return {
        // Valid baselines
        {"valid draw max=1 arg=0 cnt=null", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, nullptr, 0); }},
        {"valid draw max=1 arg=0 cnt=buf@0", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, e.count256.p, 0); }},
        {"valid draw max=0 arg=0 cnt=null", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.buf256.p, 0, nullptr, 0); }},
        {"valid draw_indexed max=1 arg=0 cnt=null", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw_indexed20.p, 1, e.buf256.p, 0, nullptr, 0); }},
        {"valid dispatch max=1 arg=0 cnt=null", true,
         [](Env& e) {
             e.bind_compute_if_available();
             e.list->ExecuteIndirect(e.dispatch12.p, 1, e.buf256.p, 0, nullptr, 0);
         }},
        {"valid same buffer for arg and count", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, e.buf256.p, 252); }},

        // Non-buffer resources (Dimension != BUFFER)
        {"argument_buffer is texture (max=1)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.tex.p, 0, nullptr, 0); }},
        {"argument_buffer is texture (max=0)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.tex.p, 0, nullptr, 0); }},
        {"count_buffer is texture (max=1)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, e.tex.p, 0); }},
        {"count_buffer is texture (max=0)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.buf256.p, 0, e.tex.p, 0); }},

        // 4-byte offset alignment
        {"arg_offset=1 unaligned (max=1)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 1, nullptr, 0); }},
        {"arg_offset=2 unaligned (max=1)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 2, nullptr, 0); }},
        {"arg_offset=2 unaligned (max=0)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.buf256.p, 2, nullptr, 0); }},
        {"arg_offset=4 aligned (max=1)", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 4, nullptr, 0); }},
        {"count_offset=1 unaligned (max=1)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, e.count256.p, 1); }},
        {"count_offset=2 unaligned (max=1)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, e.count256.p, 2); }},
        {"count_offset=2 unaligned (max=0)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.buf256.p, 0, e.count256.p, 2); }},
        {"count_offset=4 aligned (max=1)", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, e.count256.p, 4); }},

        // Null count buffer with non-zero count offset
        {"count_buffer=null count_offset=4 (max=1)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, nullptr, 4); }},
        {"count_buffer=null count_offset=2 (max=1)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, nullptr, 2); }},
        {"count_buffer=null count_offset=4 (max=0)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.buf256.p, 0, nullptr, 4); }},
        {"count_buffer=null count_offset=U64MAX (max=1)", false,
         [=](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, nullptr, kU64Max); }},

        // Argument buffer bounds (exact fit, OOB, overflow wrap, max=0 vs max>0)
        {"arg_offset=240 exact fit (240+16==256, max=1)", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 240, nullptr, 0); }},
        {"arg_offset=244 OOB (244+16>256, max=1)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 244, nullptr, 0); }},
        {"arg_offset=256 ==width (max=1)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 256, nullptr, 0); }},
        {"arg_offset=260 >width (max=1)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 260, nullptr, 0); }},
        {"arg_offset=256 ==width (max=0 no-op)", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.buf256.p, 256, nullptr, 0); }},
        {"arg_offset=260 >width (max=0 no-op)", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.buf256.p, 260, nullptr, 0); }},
        {"arg_offset=U64MAX-3 wrap (max=1)", false,
         [=](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, kU64Max - 3, nullptr, 0); }},
        {"arg_offset=U64MAX-3 aligned (max=0 no-op)", true,
         [=](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.buf256.p, kU64Max - 3, nullptr, 0); }},
        {"arg_offset=U64MAX unaligned+wrap (max=1)", false,
         [=](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, kU64Max, nullptr, 0); }},
        {"arg_offset=240 max=2 cnt=null (OOB)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 2, e.buf256.p, 240, nullptr, 0); }},
        {"arg_offset=240 max=2 cnt=buf (OOB)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 2, e.buf256.p, 240, e.count256.p, 0); }},
        {"arg_offset=0 max=U32MAX cnt=null (OOB)", false,
         [=](Env& e) { e.list->ExecuteIndirect(e.draw16.p, kU32Max, e.buf256.p, 0, nullptr, 0); }},
        {"arg_offset=0 max=U32MAX cnt=buf (OOB)", false,
         [=](Env& e) { e.list->ExecuteIndirect(e.draw16.p, kU32Max, e.buf256.p, 0, e.count256.p, 0); }},
        {"arg_buf width=3 (max=0 no-op)", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.buf3.p, 0, nullptr, 0); }},
        {"arg_buf width=3 (max=1 OOB)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf3.p, 0, nullptr, 0); }},

        // Padded stride exact-fit: (max_count - 1) * stride + record_size
        {"draw stride=20 off=240 max=1 (240+16==256)", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw20.p, 1, e.buf256.p, 240, nullptr, 0); }},
        {"draw stride=20 off=220 max=2 (220+20+16==256)", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw20.p, 2, e.buf256.p, 220, nullptr, 0); }},
        {"dispatch stride=20 off=244 max=1 (244+12==256)", true,
         [](Env& e) {
             e.bind_compute_if_available();
             e.list->ExecuteIndirect(e.dispatch20.p, 1, e.buf256.p, 244, nullptr, 0);
         }},
        {"draw_indexed stride=24 off=240 max=1 (OOB)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw_indexed24.p, 1, e.buf256.p, 240, nullptr, 0); }},

        // Count buffer bounds (checked even when max=0)
        {"count_offset=252 exact fit (252+4==256, max=1)", true,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, e.count256.p, 252); }},
        {"count_offset=256 ==width (max=1)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, e.count256.p, 256); }},
        {"count_offset=256 ==width (max=0)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.buf256.p, 0, e.count256.p, 256); }},
        {"count_offset=260 >width (max=0)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.buf256.p, 0, e.count256.p, 260); }},
        {"count_offset=U64MAX-3 wrap (max=1)", false,
         [=](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, e.count256.p, kU64Max - 3); }},
        {"count_offset=U64MAX-3 wrap (max=0)", false,
         [=](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.buf256.p, 0, e.count256.p, kU64Max - 3); }},
        {"count_buf width=3 off=0 (max=1)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, e.buf256.p, 0, e.buf3.p, 0); }},
        {"count_buf width=3 off=0 (max=0)", false,
         [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, e.buf256.p, 0, e.buf3.p, 0); }},
    };
}

bool run_positive_gpu_dispatch(Env& env) {
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto desc256 = buffer_desc(256);
    Ref<ID3D12Resource> upload, args;
    CHECK(SUCCEEDED(env.device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &desc256,
                                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                        IID_PPV_ARGS(&upload.p))));
    CHECK(SUCCEEDED(env.device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc256,
                                                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                        IID_PPV_ARGS(&args.p))));
    void* mapped = nullptr;
    CHECK(SUCCEEDED(upload->Map(0, nullptr, &mapped)) && mapped);
    std::memset(mapped, 0, 256);
    const D3D12_DISPATCH_ARGUMENTS dispatch_args{1, 1, 1};
    std::memcpy(static_cast<std::uint8_t*>(mapped) + 244, &dispatch_args, sizeof(dispatch_args));
    upload->Unmap(0, nullptr);

    const D3D12_COMMAND_QUEUE_DESC qdesc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    CHECK(SUCCEEDED(env.device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(env.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
    CHECK(SUCCEEDED(env.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, env.compute_pso.p,
                                                  IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(env.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    list->CopyBufferRegion(args.p, 244, upload.p, 244, sizeof(dispatch_args));
    const auto to_indirect = transition(args.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    list->ResourceBarrier(1, &to_indirect);
    list->SetComputeRootSignature(env.empty_root.p);
    list->SetPipelineState(env.compute_pso.p);
    // Exact-fit padded-stride indirect dispatch (244 + 12 == 256) plus max_command_count == 0 no-op.
    list->ExecuteIndirect(env.dispatch20.p, 1, args.p, 244, nullptr, 0);
    list->ExecuteIndirect(env.dispatch20.p, 0, args.p, 260, nullptr, 0);
    CHECK(SUCCEEDED(list->Close()));

    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    const HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    CHECK(ev != nullptr);
    const bool done = SUCCEEDED(fence->SetEventOnCompletion(1, ev)) && WaitForSingleObject(ev, 20000) == WAIT_OBJECT_0;
    CloseHandle(ev);
    CHECK(done);
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    REQUIRE(status == vktest::DeviceStatus::ready);
    const unsigned errors_before = vk.validation_errors();

    // Write minimal synthetic compute shader fixture for positive ExecuteIndirect dispatch.
    std::vector<std::uint8_t> program = fssm_program({0x2200u, 0x11u, 0x22u, 0x33u});
    const auto container = shader_container(program);
    const auto fixture = std::filesystem::temp_directory_path() /
                         ("leonida_exec_ind_val_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(fixture);
    {
        std::ofstream spv(fixture / "noop.spv", std::ios::binary);
        spv.write(reinterpret_cast<const char*>(kMinimalComputeSpv.data()),
                  static_cast<std::streamsize>(kMinimalComputeSpv.size() * sizeof(std::uint32_t)));
        std::ofstream index(fixture / "index.json", std::ios::binary);
        index << "{\"noop\":{\"stage\":\"cs\",\"fssm_sha256\":\"" << leonida::sha256_hex(program)
              << "\",\"bindings\":[],\"push_constant_bytes\":0,\"local_size\":[1,1,1],"
                 "\"uses_code_base\":false,\"wave64\":false}}";
    }

    const auto table = parity_cases();
    std::vector<HRESULT> layer_hr(table.size(), E_FAIL);
    std::vector<HRESULT> warp_hr(table.size(), E_FAIL);
    std::vector<HRESULT> hw_hr(table.size(), E_FAIL);

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = vk.buffer_device_address;
        info.shader_dir = fixture.string();
        Ref<ID3D12Device> ours;
        REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p))));
        Env env;
        REQUIRE(env.init(ours.p, &container));
        for (std::size_t i = 0; i < table.size(); ++i) REQUIRE(run_case(env, table[i], layer_hr[i]));

        // Layer-only null-pointer safety checks (Windows D3D12 dereferences these and crashes).
        HRESULT null_hr = S_OK;
        REQUIRE(run_case(env, {"null arg max=1", false,
                               [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 1, nullptr, 0, nullptr, 0); }},
                         null_hr));
        REQUIRE(FAILED(null_hr));
        REQUIRE(run_case(env, {"null arg max=0", false,
                               [](Env& e) { e.list->ExecuteIndirect(e.draw16.p, 0, nullptr, 0, nullptr, 0); }},
                         null_hr));
        REQUIRE(FAILED(null_hr));
        REQUIRE(run_case(env, {"null sig max=1", false,
                               [](Env& e) { e.list->ExecuteIndirect(nullptr, 1, e.buf256.p, 0, nullptr, 0); }},
                         null_hr));
        REQUIRE(FAILED(null_hr));

        // Positive valid GPU ExecuteIndirect dispatch on Vulkan.
        REQUIRE(run_positive_gpu_dispatch(env));
    }

    bool warp_ran = false;
    bool hw_ran = false;
    std::string hw_name;
    if (!hostwine::host_is_wine()) {
        Ref<IDXGIFactory4> factory;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            Ref<IDXGIAdapter> warp_adapter;
            Ref<ID3D12Device> warp_dev;
            if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp_adapter.p))) &&
                SUCCEEDED(D3D12CreateDevice(warp_adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&warp_dev.p)))) {
                Env env;
                REQUIRE(env.init(warp_dev.p));
                for (std::size_t i = 0; i < table.size(); ++i) REQUIRE(run_case(env, table[i], warp_hr[i]));
                REQUIRE(warp_dev->GetDeviceRemovedReason() == S_OK);
                warp_ran = true;
            }
            for (UINT i = 0;; ++i) {
                Ref<IDXGIAdapter1> adapter;
                if (factory->EnumAdapters1(i, &adapter.p) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC1 desc{};
                adapter->GetDesc1(&desc);
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
                Ref<ID3D12Device> hw_dev;
                if (SUCCEEDED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&hw_dev.p)))) {
                    Env env;
                    REQUIRE(env.init(hw_dev.p));
                    for (std::size_t j = 0; j < table.size(); ++j) REQUIRE(run_case(env, table[j], hw_hr[j]));
                    REQUIRE(hw_dev->GetDeviceRemovedReason() == S_OK);
                    char name_buf[256]{};
                    std::snprintf(name_buf, sizeof(name_buf), "%ls", desc.Description);
                    hw_name = name_buf;
                    hw_ran = true;
                    break;
                }
            }
        }
    }

    std::error_code ec;
    std::filesystem::remove_all(fixture, ec);

    unsigned failures = 0;
    std::printf("%-48s | %-10s | %-10s | %-10s\n", "case", "layer", "WARP", hw_ran ? hw_name.c_str() : "HW");
    for (std::size_t i = 0; i < table.size(); ++i) {
        const bool layer_ok = SUCCEEDED(layer_hr[i]);
        const bool warp_ok = warp_ran ? SUCCEEDED(warp_hr[i]) : table[i].expected_ok;
        const bool hw_ok = hw_ran ? SUCCEEDED(hw_hr[i]) : table[i].expected_ok;
        std::printf("%-48s | 0x%08lx | %-10s | %-10s\n",
                    table[i].name,
                    static_cast<unsigned long>(layer_hr[i]),
                    warp_ran ? (SUCCEEDED(warp_hr[i]) ? "S_OK" : "FAIL") : (warp_ok ? "(S_OK)" : "(FAIL)"),
                    hw_ran ? (SUCCEEDED(hw_hr[i]) ? "S_OK" : "FAIL") : (hw_ok ? "(S_OK)" : "(FAIL)"));
        if (layer_ok != warp_ok || layer_ok != hw_ok || layer_ok != table[i].expected_ok) {
            std::fprintf(stderr, "FAIL %s: layer=%d warp=%d hw=%d expected=%d\n",
                         table[i].name, int(layer_ok), int(warp_ok), int(hw_ok), int(table[i].expected_ok));
            ++failures;
        }
    }

    const unsigned validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", validation_errors);
        ++failures;
    }
    std::printf("execute_indirect_validation: %zu parity cases (WARP=%s, HW=%s), positive GPU dispatch ok, "
                "%u validation errors, %u failures\n",
                table.size(), warp_ran ? "live" : "measured", hw_ran ? hw_name.c_str() : "measured",
                validation_errors, failures);
    return failures ? 1 : 0;
}
