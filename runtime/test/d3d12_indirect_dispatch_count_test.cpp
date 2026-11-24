// D3D12 ExecuteIndirect with a GPU count buffer and one DISPATCH argument per record.
// The same synthetic workload runs on native WARP and on the Vulkan layer, with actual GPU readback.
// Set LEONIDA_VK_VALIDATION=1 to enable Vulkan core and synchronization validation.
#include "vk_test_device.h"
#include "shader_fixture_path.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <winver.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef LEONIDA_TEST_INDIRECT_COUNT_SPV
#error "LEONIDA_TEST_INDIRECT_COUNT_SPV must name the synthetic GLSL compute module"
#endif

using namespace d3d12test;
namespace {
constexpr UINT kMaximum = 3, kStride = 20, kArgsOffset = 32, kCountOffset = 44;
const char native_shader[] = R"(
RWByteAddressBuffer output : register(u0);
[numthreads(1,1,1)]
void main(uint3 group : SV_GroupID) {
    uint old;
    output.InterlockedAdd(0, 1, old);
    output.InterlockedAdd(4, group.x + 1, old);
}
)";

bool run(ID3D12Device* dev, const std::vector<std::uint8_t>& shader, UINT count, bool gpu_update,
         std::array<UINT, 2>& result, bool shim) {
    Ref<ID3D12RootSignature> root;
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    param.Descriptor = {0, 0};
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto serialized = serialize_root_signature(root_desc);
    if (serialized.empty() ||
        FAILED(dev->CreateRootSignature(0, serialized.data(), serialized.size(), IID_PPV_ARGS(&root.p))))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root.p;
    desc.CS = {shader.data(), shader.size()};
    Ref<ID3D12PipelineState> pso;
    if (FAILED(dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso.p)))) return false;
    const D3D12_INDIRECT_ARGUMENT_DESC argument{D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH};
    const D3D12_COMMAND_SIGNATURE_DESC signature_desc{kStride, 1, &argument, 0};
    Ref<ID3D12CommandSignature> signature;
    if (FAILED(dev->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature.p))))
        return false;
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> source, args, counts, output, readback;
    const auto create = [&](Ref<ID3D12Resource>& dest, D3D12_HEAP_PROPERTIES heap,
                            D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags) {
        const auto buffer = buffer_desc(256, flags);
        return SUCCEEDED(dev->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &buffer, state, nullptr, IID_PPV_ARGS(&dest.p)));
    };
    if (!create(source, upload_heap, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE) ||
        !create(args, default_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE) ||
        !create(counts, default_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE) ||
        !create(output, default_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ||
        !create(readback, readback_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE))
        return false;
    std::array<UINT, 64> input{};
    for (UINT i = 0; i < kMaximum; ++i) {
        const UINT base = (kArgsOffset + i * kStride) / 4;
        input[base] = i + 1;   // number of groups in the record, 1 then 2 then 3
        input[base + 1] = input[base + 2] = 1;
        input[base + 3] = 0xa5a5a5a5;  // padding: must be ignored
        input[base + 4] = 0x5a5a5a5a;
    }
    input[40] = count;      // source offset 160, copied to destination +44
    input[41] = 3;          // update count from source offset 164
    void* mapped = nullptr;
    if (FAILED(source->Map(0, nullptr, &mapped)) || !mapped) return false;
    std::memcpy(mapped, input.data(), sizeof(input));
    source->Unmap(0, nullptr);

    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_COMPUTE};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    if (FAILED(dev->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.p,
                                     pso.p, IID_PPV_ARGS(&list.p))) ||
        FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;
    list->CopyBufferRegion(args.p, kArgsOffset, source.p, kArgsOffset, kMaximum * kStride);
    list->CopyBufferRegion(counts.p, kCountOffset, source.p, 160, 4);
    list->CopyBufferRegion(output.p, 0, source.p, 0, 8); // initialize the UAV to zero
    const D3D12_RESOURCE_BARRIER ready[] = {
        transition(args.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        transition(counts.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        transition(output.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
    list->ResourceBarrier(3, ready);
    list->SetComputeRootSignature(root.p);
    list->SetPipelineState(pso.p);
    list->SetComputeRootUnorderedAccessView(0, output->GetGPUVirtualAddress());
    list->ExecuteIndirect(signature.p, kMaximum, args.p, kArgsOffset, counts.p, kCountOffset);
    if (gpu_update) {
        auto to_copy = transition(counts.p, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                                  D3D12_RESOURCE_STATE_COPY_DEST);
        list->ResourceBarrier(1, &to_copy);
        list->CopyBufferRegion(counts.p, kCountOffset, source.p, 164, 4);
        auto to_indirect = transition(counts.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                      D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        list->ResourceBarrier(1, &to_indirect);
        // Repeat with a count produced by a GPU copy after the previous indirect dispatch.
        list->ExecuteIndirect(signature.p, kMaximum, args.p, kArgsOffset, counts.p, kCountOffset);
    }
    const auto to_copy = transition(output.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    list->CopyBufferRegion(readback.p, 0, output.p, 0, 8);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (shim && rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) return false;
    if (FAILED(queue->Signal(fence.p, 1))) return false;
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    const bool completed = SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                           WaitForSingleObject(event, 20000) == WAIT_OBJECT_0;
    CloseHandle(event);
    if (!completed || FAILED(readback->Map(0, nullptr, &mapped)) || !mapped) return false;
    std::memcpy(result.data(), mapped, sizeof(result));
    readback->Unmap(0, nullptr);
    return true;
}

// The second dispatch of a gpu_update run reads the count written by the copy before it (3).
// With second_count == count instead, this is what a host that reuses the first count gives.
std::array<UINT, 2> expected(UINT count, bool update, UINT second_count = 3) {
    auto result = std::array<UINT, 2>{};
    const auto accumulate = [&](UINT c) {
        for (UINT i = 0; i < std::min(c, kMaximum); ++i) {
            result[0] += i + 1;
            result[1] += (i + 1) * (i + 2) / 2;
        }
    };
    accumulate(count);
    if (update) accumulate(second_count);
    return result;
}

// Build and revision of a loaded system DLL ("22621.5415"; Windows shows it as 10.0.22621.5415), or "unknown".
std::string module_version(const wchar_t* name) {
    const HMODULE module = GetModuleHandleW(name);
    wchar_t path[MAX_PATH]{};
    if (!module || !GetModuleFileNameW(module, path, MAX_PATH)) return "unknown";
    DWORD ignored = 0;
    const DWORD size = GetFileVersionInfoSizeW(path, &ignored);
    std::vector<std::uint8_t> data(size);
    VS_FIXEDFILEINFO* info = nullptr;
    UINT length = 0;
    if (!size || !GetFileVersionInfoW(path, 0, size, data.data()) ||
        !VerQueryValueW(data.data(), L"\\", reinterpret_cast<void**>(&info), &length) || !info)
        return "unknown";
    char text[64];
    std::snprintf(text, sizeof text, "%u.%u", HIWORD(info->dwFileVersionLS), LOWORD(info->dwFileVersionLS));
    return text;
}
}  // namespace

int main() {
    using Compile = decltype(&D3DCompile);
    const HMODULE compiler = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!compiler) return 77;
    const auto compile = reinterpret_cast<Compile>(GetProcAddress(compiler, "D3DCompile"));
    if (!compile) return 77;
    Ref<ID3DBlob> native, errors;
    if (FAILED(compile(native_shader, std::strlen(native_shader), nullptr, nullptr, nullptr,
                       "main", "cs_5_0", 0, 0, &native.p, &errors.p)) || !native.p)
        return 1;
    const auto* native_bytes = static_cast<const std::uint8_t*>(native->GetBufferPointer());
    const std::vector<std::uint8_t> native_code(native_bytes, native_bytes + native->GetBufferSize());
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> warp;
    Ref<ID3D12Device> host;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) ||
        FAILED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p))))
        return 77;

    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!vk.buffer_device_address) return 77;
    std::vector<std::uint8_t> program = fssm_program({0x1100u, 0x44u, 0x55u, 0x66u, 0x77u, 0x88u});
    for (UINT i = 0; i < 12; ++i) put32(program, 0xbf810000 + i);
    const auto container = shader_container(program);
    const auto fixture = std::filesystem::temp_directory_path() /
                         ("leonida_indirect_dispatch_count_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(fixture);
    std::filesystem::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_INDIRECT_COUNT_SPV),
                               fixture / "count.spv", std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(fixture / "index.json", std::ios::binary);
        index << "{\"count\":{\"stage\":\"cs\",\"fssm_sha256\":\"" << leonida::sha256_hex(program)
              << "\",\"bindings\":[],\"push_constant_bytes\":144,\"local_size\":[1,1,1],"
                 "\"uses_code_base\":false,\"wave64\":false}}";
    }
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = vk.buffer_device_address;
    info.shader_dir = fixture.string();
    Ref<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&layer.p)))) return 1;

    // The layer is checked against expected(), the D3D12 rule that the count is read when the GPU executes the
    // command. WARP must match it too, with one measured exception: in-box WARP 10.0.22621.5415 (Windows 11 23H2)
    // reuses the first count for the second ExecuteIndirect after the GPU copy overwrites it (0 -> 3 gives 0/0,
    // 1 -> 3 gives 2/2). That case is reported, not failed, when WARP gives exactly the stale-count result.
    unsigned cases = 0, stale_host = 0;
    for (UINT count : {0u, 1u, 3u, 7u}) {
        for (bool update : {false, true}) {
            std::array<UINT, 2> native_result{}, layer_result{};
            const auto want = expected(count, update);
            if (!run(host.p, native_code, count, update, native_result, false) ||
                !run(layer.p, container, count, update, layer_result, true) || layer_result != want ||
                (native_result != want && !(update && native_result == expected(count, update, count)))) {
                std::fprintf(stderr, "count %u update %d: host %u/%u, layer %u/%u, expected %u/%u\n",
                             count, int(update), native_result[0], native_result[1],
                             layer_result[0], layer_result[1], want[0], want[1]);
                return 1;
            }
            if (native_result != want) {
                std::printf("count %u update 1: host WARP build %s reused the first count (%u/%u, expected %u/%u); "
                            "the layer matches the expected result\n",
                            count, module_version(L"d3d10warp.dll").c_str(), native_result[0], native_result[1],
                            want[0], want[1]);
                ++stale_host;
            }
            ++cases;
        }
    }
    if (stale_host)
        std::printf("%u GPU-count-overwrite case(s) checked against the expected result only (stale WARP count)\n",
                    stale_host);
    if (vk.validation_errors()) {
        std::fprintf(stderr, "validation errors: %u\n", vk.validation_errors());
        return 1;
    }
    std::printf("counted ExecuteIndirect(DISPATCH): %u native WARP / Vulkan GPU comparisons passed; "
                "count=0,1,max,>max, nonzero offsets and GPU count overwrite, no validation errors\n", cases);
    return 0;
}
