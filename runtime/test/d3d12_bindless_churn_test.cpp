// Bindless table churn without a device-idle wait (D3D-314). A game that makes and releases textures every frame
// changes the descriptor hash table of set 0 at every ExecuteCommandLists: the table grows, and keys that are still
// live move or change their index. Before D3D-314 each such flush waited for the whole device (vkDeviceWaitIdle) so
// the old table could be freed; now the flush makes a new generation (table buffers and descriptor set) and the old
// one is freed once every queue's submissions up to then are done.
//
// This test runs the churn through D3D12 alone, on the Xbox-ABI proxy over the Vulkan device:
//   - a direct queue and a compute queue, three frames in flight, each frame on both queues;
//   - every frame creates 24 new 8x8 textures with their SRVs (a ring of descriptor slots in a shader-visible heap)
//     and releases the textures of seven frames ago, so keys are erased and the live set grows to about 190 keys;
//   - the direct queue uploads the textures; the compute queue waits for that on the GPU (ID3D12CommandQueue::Wait);
//   - each queue dispatches the sampling shader of d3d12_set0_compute_test (test/glsl/cs_set0_sample.comp, written
//     to the translator's runtime contract) eight times, on SRVs of the current frame and of the three before it;
//   - every result is compared byte for byte with the texels the CPU wrote.
// It runs twice on a fresh device each: as the layer is now (the number of vkDeviceWaitIdle calls for the bindless
// set must be 0, and generations must have been made), and with LEONIDA_BINDLESS_IDLE_WAIT=1, which restores the
// wait of before D3D-314 so the frame times can be compared (the results must be exact there too; its wait count
// must be above 0). The times are printed, not asserted.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation with synchronization validation (0 errors
// required).
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#ifndef LEONIDA_TEST_CS_SET0_SAMPLE_SPV
#error "LEONIDA_TEST_CS_SET0_SAMPLE_SPV must name the compiled test shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

#define MAIN_EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX

namespace {

using Texel = std::array<std::uint8_t, 4>;
constexpr UINT texture_size = 8, texels = texture_size * texture_size, run_bytes = texels * 4;
constexpr UINT frames = 96, in_flight = 3, per_frame = 24, use_span = 3;
constexpr UINT life = use_span + in_flight + 1;      // frames a texture is kept: no list that samples it still runs
constexpr UINT ring = life + in_flight + 1;          // frames of descriptor slots before one is written again
constexpr UINT probes = 8;
constexpr UINT queues = 2;                           // 0 direct, 1 compute

// What frame `g` wrote into its texture `i`.
Texel texel(UINT g, UINT i, UINT x, UINT y) {
    return {std::uint8_t(x * 29 + g * 7 + i), std::uint8_t(y * 31 + g * 3 + i * 5),
            std::uint8_t(g * 13 + i * 17 + x * y), 255};
}

struct Probe {
    UINT frame, index;
};
// The SRVs a frame's dispatches sample: four of its own and four from up to three frames before.
// With `async`, the compute queue (q 1) samples only textures of `use_span` frames ago, which the CPU has seen
// finished: it has no D3D12 fence on the direct queue, so what it must wait for is the table the direct queue's
// flush just grew, nothing else.
bool async_compute(bool async, UINT q, UINT f) { return async && q == 1 && f >= use_span; }
Probe probe_of(UINT f, UINT p, bool async = false, UINT q = 0) {
    static constexpr UINT ages[probes] = {0, 0, 0, 0, 1, 2, 3, 1};
    const UINT wanted = async_compute(async, q, f) ? use_span : ages[p];
    const UINT age = wanted <= f ? wanted : 0;
    return {f - age, (p * 5 + f * 3) % per_frame};
}

struct Result {
    double total_ms = 0, worst_frame_ms = 0;
    rt::d3d12::BindlessStats stats{};
    unsigned mismatches = 0, compared = 0;
    std::uint64_t max_retired_pending = 0;  // seen right after an ExecuteCommandLists, with work in flight
};

bool run(vktest::Device& d, const std::filesystem::path& library, bool legacy_idle, bool async, Result& result) {
    _putenv_s("LEONIDA_BINDLESS_IDLE_WAIT", legacy_idle ? "1" : "0");
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    // Distinct VkQueues for the D3D12 queues where the family has them: two queues on one VkQueue are ordered by
    // submission, which hides a missing wait between them.
    if (d.queues.size() >= 2) info.command_queues = d.queues;
    info.buffer_device_address = true;
    info.bindless = d.bindless;
    info.bindless_partially_bound = d.partially_bound;
    info.image_view_min_lod = d.image_view_min_lod;
    info.shader_dir = library.string();
    ID3D12Device* inner = nullptr;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                       reinterpret_cast<void**>(&device.p))));
    rt::d3d12::set_device_outer(inner, device.p);
    EXPECT(rt::d3d12::bindless_set(device.p) != VK_NULL_HANDLE);

    // Root signature as in d3d12_set0_compute_test: [0] two constants, [1] a root UAV (the output), [2] the table of
    // views (here: pointed at the SRV's own slot, so s0 is 0), [3] the table of samplers.
    D3D12_DESCRIPTOR_RANGE view_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_DESCRIPTOR_RANGE sampler_range{D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[4] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, 2};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor = {0, 0};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable = {1, &view_range};
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[3].DescriptorTable = {1, &sampler_range};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {4, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto root_blob = serialize_root_signature(root_desc);
    EXPECT(!root_blob.empty());
    Ref<ID3D12RootSignature> root;
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

    const auto program = fssm_program({0x43530003, 0xbf810000});
    const auto container = shader_container(program);
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.p;
    pso_desc.CS = {container.data(), container.size()};
    Ref<ID3D12PipelineState> pso;
    EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC texture_desc{};
    texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture_desc.Width = texture_desc.Height = texture_size;
    texture_desc.DepthOrArraySize = 1;
    texture_desc.MipLevels = 1;
    texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture_desc.SampleDesc.Count = 1;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    device->GetCopyableFootprints(&texture_desc, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);
    const UINT pitch = footprint.Footprint.RowPitch;
    EXPECT(pitch == 256);
    constexpr UINT64 slice_bytes = UINT64(256) * texture_size;  // one texture in the upload buffer

    // Per frame-in-flight: an upload buffer (direct queue only), and per queue an output buffer and its readback.
    Ref<ID3D12Resource> upload[in_flight], output[queues][in_flight], readback[queues][in_flight];
    std::uint8_t* upload_ptr[in_flight] = {};
    bool output_in_uav[queues][in_flight] = {};
    for (UINT k = 0; k < in_flight; ++k) {
        const auto upload_desc = buffer_desc(slice_bytes * per_frame);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload[k].p))));
        EXPECT(SUCCEEDED(upload[k]->Map(0, nullptr, reinterpret_cast<void**>(&upload_ptr[k]))));
        for (UINT q = 0; q < queues; ++q) {
            const auto output_desc = buffer_desc(probes * run_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &output_desc,
                                                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                             IID_PPV_ARGS(&output[q][k].p))));
            output_in_uav[q][k] = true;
            const auto readback_desc = buffer_desc(probes * run_bytes);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&readback[q][k].p))));
        }
    }

    // The view heap holds `ring` frames of `per_frame` SRVs; the sampler heap one wrapping POINT sampler.
    const D3D12_DESCRIPTOR_HEAP_DESC view_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, ring * per_frame,
                                                    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    const D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Ref<ID3D12DescriptorHeap> view_heap, sampler_heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&view_heap_desc, IID_PPV_ARGS(&view_heap.p))));
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&sampler_heap_desc, IID_PPV_ARGS(&sampler_heap.p))));
    const std::uint64_t view_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(view_heap.p, 9))(view_heap.p);
    const std::uint64_t view_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(view_heap.p, 10))(view_heap.p);
    const std::uint64_t sampler_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 9))(sampler_heap.p);
    const std::uint64_t sampler_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 10))(sampler_heap.p);
    const UINT view_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    EXPECT(view_cpu && sampler_cpu && view_gpu >= 0x10000 && sampler_gpu >= 0x10000);
    EXPECT(view_gpu < (1ull << 32) && sampler_gpu < (1ull << 32));
    {
        D3D12_SAMPLER_DESC sampler_desc{};
        sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.MaxAnisotropy = 1;
        sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
        device->CreateSampler(&sampler_desc, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu)});
    }

    // The queues: each with its own fence, an allocator per frame in flight and one list.
    const D3D12_COMMAND_LIST_TYPE types[queues] = {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE};
    Ref<ID3D12CommandQueue> queue[queues];
    Ref<ID3D12CommandAllocator> allocator[queues][in_flight];
    Ref<ID3D12GraphicsCommandList> list[queues];
    Ref<ID3D12Fence> fence[queues];
    for (UINT q = 0; q < queues; ++q) {
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = types[q];
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue[q].p))));
        for (UINT k = 0; k < in_flight; ++k)
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(types[q], IID_PPV_ARGS(&allocator[q][k].p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, types[q], allocator[q][0].p, nullptr, IID_PPV_ARGS(&list[q].p))));
        EXPECT(list[q]->Close() == S_OK);
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence[q].p))));
    }

    std::array<std::array<Ref<ID3D12Resource>, per_frame>, life> textures;  // frame g in [g % life]

    const auto check = [&](UINT g) {  // results of frame g, once both queues finished it
        const UINT k = g % in_flight;
        for (UINT q = 0; q < queues; ++q) {
            void* mapped = nullptr;
            if (FAILED(readback[q][k]->Map(0, nullptr, &mapped))) {
                ++result.mismatches;
                continue;
            }
            const auto* bytes = static_cast<const std::uint8_t*>(mapped);
            for (UINT p = 0; p < probes; ++p) {
                const Probe probe = probe_of(g, p, async, q);
                for (UINT i = 0; i < texels; ++i) {
                    const Texel want = texel(probe.frame, probe.index, i % texture_size, i / texture_size);
                    const std::uint8_t* got = bytes + p * run_bytes + i * 4;
                    ++result.compared;
                    if (std::memcmp(got, want.data(), 4) == 0) continue;
                    if (result.mismatches++ < 8)
                        std::fprintf(stderr, "frame %u queue %u probe %u (frame %u texture %u) texel %u: got "
                                             "%u,%u,%u,%u want %u,%u,%u,%u\n",
                                     g, q, p, probe.frame, probe.index, i, got[0], got[1], got[2], got[3], want[0],
                                     want[1], want[2], want[3]);
                }
            }
            D3D12_RANGE none{0, 0};
            readback[q][k]->Unmap(0, &none);
        }
        return true;
    };

    using Clock = std::chrono::steady_clock;
    const auto begin_all = Clock::now();
    UINT checked = 0;
    for (UINT f = 0; f < frames; ++f) {
        const auto frame_start = Clock::now();
        const UINT k = f % in_flight;
        // Frame f - in_flight is done on both queues before its allocators, upload buffer and outputs are reused.
        if (f >= in_flight) {
            for (UINT q = 0; q < queues; ++q) EXPECT(SUCCEEDED(fence[q]->SetEventOnCompletion(f - in_flight + 1, nullptr)));
            EXPECT(check(f - in_flight));
            ++checked;
        }
        // The textures of frame f - life are not sampled by any frame still in flight: release them (their keys go).
        if (f >= life)
            for (auto& texture : textures[f % life]) texture.reset();

        // New textures and their SRVs, and their texels in the upload buffer.
        for (UINT i = 0; i < per_frame; ++i) {
            auto& texture = textures[f % life][i];
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&texture.p))));
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D = {0, 1, 0, 0.0f};
            const UINT heap_slot = (f % ring) * per_frame + i;
            device->CreateShaderResourceView(texture.p, &srv,
                                             D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(view_cpu + UINT64(heap_slot) * view_step)});
            for (UINT y = 0; y < texture_size; ++y)
                for (UINT x = 0; x < texture_size; ++x) {
                    const Texel t = texel(f, i, x, y);
                    std::memcpy(upload_ptr[k] + i * slice_bytes + y * pitch + x * 4, t.data(), 4);
                }
        }

        for (UINT q = 0; q < queues; ++q) {
            EXPECT(SUCCEEDED(allocator[q][k]->Reset()));
            EXPECT(SUCCEEDED(list[q]->Reset(allocator[q][k].p, nullptr)));
            if (q == 0) {  // upload, and make the textures shader resources for both queues
                D3D12_RESOURCE_BARRIER sampled[per_frame];
                for (UINT i = 0; i < per_frame; ++i) {
                    D3D12_TEXTURE_COPY_LOCATION dst{};
                    dst.pResource = textures[f % life][i].p;
                    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    D3D12_TEXTURE_COPY_LOCATION src{};
                    src.pResource = upload[k].p;
                    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                    src.PlacedFootprint = footprint;
                    src.PlacedFootprint.Offset = i * slice_bytes;
                    list[q]->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                    sampled[i] = transition(textures[f % life][i].p, D3D12_RESOURCE_STATE_COPY_DEST,
                                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                }
                list[q]->ResourceBarrier(per_frame, sampled);
            }
            if (!output_in_uav[q][k]) {
                const auto back = transition(output[q][k].p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                list[q]->ResourceBarrier(1, &back);
            }
            ID3D12DescriptorHeap* heaps[] = {view_heap.p, sampler_heap.p};
            list[q]->SetDescriptorHeaps(2, heaps);
            list[q]->SetComputeRootSignature(root.p);
            list[q]->SetPipelineState(pso.p);
            list[q]->SetComputeRootUnorderedAccessView(1, output[q][k]->GetGPUVirtualAddress());
            list[q]->SetComputeRootDescriptorTable(3, D3D12_GPU_DESCRIPTOR_HANDLE{sampler_gpu});
            for (UINT p = 0; p < probes; ++p) {
                const Probe probe = probe_of(f, p, async, q);
                const UINT heap_slot = (probe.frame % ring) * per_frame + probe.index;
                list[q]->SetComputeRootDescriptorTable(
                    2, D3D12_GPU_DESCRIPTOR_HANDLE{view_gpu + UINT64(heap_slot) * view_step});
                const std::uint32_t offset_and_first[2] = {0, p * texels};
                list[q]->SetComputeRoot32BitConstants(0, 2, offset_and_first, 0);
                list[q]->Dispatch(1, 1, 1);
            }
            const auto to_copy = transition(output[q][k].p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                            D3D12_RESOURCE_STATE_COPY_SOURCE);
            list[q]->ResourceBarrier(1, &to_copy);
            output_in_uav[q][k] = false;
            list[q]->CopyBufferRegion(readback[q][k].p, 0, output[q][k].p, 0, probes * run_bytes);
            EXPECT(list[q]->Close() == S_OK);
            // The compute queue samples what the direct queue uploads: that is a wait on the GPU, not on the CPU.
            if (q == 1 && !async_compute(async, q, f)) EXPECT(SUCCEEDED(queue[1]->Wait(fence[0].p, f + 1)));
            ID3D12CommandList* lists[] = {list[q].p};
            queue[q]->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue[q].p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue[q]->Signal(fence[q].p, f + 1)));
            // A generation the flush retired stays until the submissions that may have used it are done.
            result.max_retired_pending = std::max(result.max_retired_pending,
                                                  rt::d3d12::bindless_stats(device.p).retired_pending);
        }
        const double frame_ms = std::chrono::duration<double, std::milli>(Clock::now() - frame_start).count();
        result.worst_frame_ms = std::max(result.worst_frame_ms, frame_ms);
    }
    for (; checked < frames; ++checked) {
        for (UINT q = 0; q < queues; ++q) EXPECT(SUCCEEDED(fence[q]->SetEventOnCompletion(checked + 1, nullptr)));
        EXPECT(check(checked));
    }
    result.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin_all).count();
    result.stats = rt::d3d12::bindless_stats(device.p);

    for (auto& frame : textures)
        for (auto& texture : frame) texture.reset();
    for (UINT k = 0; k < in_flight; ++k) {
        upload[k].reset();
        for (UINT q = 0; q < queues; ++q) {
            output[q][k].reset();
            readback[q][k].reset();
            allocator[q][k].reset();
        }
    }
    for (UINT q = 0; q < queues; ++q) {
        list[q].reset();
        queue[q].reset();
        fence[q].reset();
    }
    pso.reset();
    root.reset();
    view_heap.reset();
    sampler_heap.reset();
    EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    // `--validated` runs under the validation layer with synchronization validation, and skips when it is not there.
    const bool validated = argc > 1 && std::strcmp(argv[1], "--validated") == 0;
    if (validated) _putenv_s("LEONIDA_VK_VALIDATION", "1");
    vktest::Device d;
    d.queue_count_wanted = 2;  // a VkQueue for each of the two D3D12 queues, where the family has two
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (validated && device_status != vktest::DeviceStatus::ready) {
        std::printf("SKIP: the validation layer is not available\n");
        return 77;
    }
    if (device_status != vktest::DeviceStatus::ready) return 1;
    if (!d.bindless) {
        std::printf("SKIP: the device lacks the descriptor indexing features of the bindless set 0\n");
        return 77;
    }

    // The shader's program and a translator index that maps its hash to the test module (as d3d12_set0_compute_test).
    const auto program = fssm_program({0x43530003, 0xbf810000});
    const auto library = std::filesystem::temp_directory_path() / "leonida_bindless_churn_test";
    std::filesystem::create_directories(library);
    std::filesystem::copy_file(LEONIDA_TEST_CS_SET0_SAMPLE_SPV, library / "cs_set0_sample.spv",
                               std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"cs_set0_sample\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(program)
              << "\", \"bindings\": [0, 1, 2], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": false, \"wave64\": false}}";
    }

    const bool distinct = d.queues.size() >= 2 && d.queues[0] != d.queues[1];
    std::printf("D3D12 queues on %s VkQueues (%zu created in family %u)\n",
                distinct ? "distinct" : "ONE shared (the family has a single queue: the cross-queue wait is only "
                                        "checked for validity)", d.queues.size(), d.family);
    Result now, async, legacy, async_legacy;
    bool ok = run(d, library, false, false, now);
    if (ok) ok = run(d, library, false, true, async);
    if (ok) ok = run(d, library, true, false, legacy);
    if (ok) ok = run(d, library, true, true, async_legacy);  // the old wait, and no wait between the queues
    std::filesystem::remove_all(library);
    if (!ok) return 1;

    const auto report = [](const char* name, const Result& r) {
        std::printf("%-22s %u frames in %.1f ms (%.2f ms/frame, worst %.2f ms); %llu generations made, %llu "
                    "vkDeviceWaitIdle for the set, %llu retired not yet freed at the end; %u texels compared, "
                    "%u wrong\n",
                    name, frames, r.total_ms, r.total_ms / frames, r.worst_frame_ms,
                    static_cast<unsigned long long>(r.stats.generations_made),
                    static_cast<unsigned long long>(r.stats.device_idle_waits),
                    static_cast<unsigned long long>(r.stats.retired_pending), r.compared, r.mismatches);
    };
    report("deferred retirement:", now);
    report("async compute:", async);
    report("old wait:", legacy);
    report("old wait, async:", async_legacy);
    MAIN_EXPECT(now.mismatches == 0 && async.mismatches == 0 && legacy.mismatches == 0 &&
                async_legacy.mismatches == 0);
    MAIN_EXPECT(async.compared == now.compared && async.stats.device_idle_waits == 0 && async.stats.generations_made > 0);
    MAIN_EXPECT(now.max_retired_pending > 0 && async.max_retired_pending > 0);
    MAIN_EXPECT(now.compared == frames * queues * probes * texels);
    MAIN_EXPECT(now.stats.device_idle_waits == 0);
    MAIN_EXPECT(now.stats.generations_made > 0);
    MAIN_EXPECT(legacy.stats.device_idle_waits > 0);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("bindless churn: %u frames on a direct and a compute queue, 3 in flight, %u SRVs made and released "
                "per frame; every sampled texel matched, the table moved to %llu new generations and no queue waited "
                "for the device (the old path did %llu times)\n",
                frames, per_frame, static_cast<unsigned long long>(now.stats.generations_made),
                static_cast<unsigned long long>(legacy.stats.device_idle_waits));
    return 0;
}
