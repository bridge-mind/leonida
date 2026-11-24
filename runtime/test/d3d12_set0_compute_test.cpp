// Set 0 from the compute bind point, through the Xbox-ABI proxy over the Vulkan device. A compute PSO whose shader
// declares set-0 bindings is created and dispatched through D3D12 alone; nothing in the test touches Vulkan's
// descriptor sets. The dispatches only give the right words if the layer has the device's bindless set 0 in the
// pipeline layout and binds it at the dispatch:
//   - two 8x8 textures, a gradient and its point reflection, uploaded with Map and CopyTextureRegion;
//   - a shader-visible CBV/SRV/UAV heap with an SRV for each in slots 0 and 1 and 32 bytes no view was made for in
//     slot 2, and a sampler heap with a POINT sampler that wraps in slot 0 and one that clamps in slot 1;
//   - a root signature of two constants, a root UAV and the two descriptor tables.
// test/glsl/cs_set0_sample.comp loads the T# and the S# from the tables in the GPU address arena, looks them up in
// the descriptor hash table (set 0, binding 2) and samples one tile to the right in U. Four dispatches:
//   1. slot 0 with the wrapping sampler: the gradient, texel for texel;
//   2. slot 1 with the wrapping sampler: the reflected texture, so the T# decides the image;
//   3. slot 2 with the wrapping sampler: the fallback image, one colour everywhere;
//   4. slot 0 with the clamping sampler: the gradient's right edge in every column, so the S# decides the sampler.
// A device without the bindless set refuses the PSO. No game data: the program is two marker words and the module is
// GLSL written to the translator's contract.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation (0 errors required).
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#ifndef LEONIDA_TEST_CS_SET0_SAMPLE_SPV
#error "LEONIDA_TEST_CS_SET0_SAMPLE_SPV must name the compiled test shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX

namespace {

using Texel = std::array<std::uint8_t, 4>;
constexpr UINT texture_size = 8, texels = texture_size * texture_size, run_bytes = texels * 4, runs = 4;

// The synthetic gradient: red rises with x, green with y, blue falls along the diagonal.
Texel gradient(UINT x, UINT y) {
    return {std::uint8_t(x * 36), std::uint8_t(y * 36), std::uint8_t(255 - (x + y) * 16), 255};
}
// The second texture: the first one point-reflected, so no texel of it is in the same place.
Texel reflected(UINT x, UINT y) { return gradient(texture_size - 1 - x, texture_size - 1 - y); }

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    if (!d.bindless) {
        std::printf("SKIP: the device lacks the descriptor indexing features of the bindless set 0\n");
        return 77;
    }

    // The program the PSO carries, and a translator index that maps its hash to the test module. The index lists
    // set-0 bindings for the shader, as the translator does for one that samples.
    const auto program = fssm_program({0x43530003, 0xbf810000});
    const auto container = shader_container(program);
    const testfiles::ScratchDir scratch("leonida_set0_compute_test");
    const auto& library = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SET0_SAMPLE_SPV),
                                library / "cs_set0_sample.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"cs_set0_sample\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(program)
              << "\", \"bindings\": [0, 1, 2], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": false, \"wave64\": false}}";
    }

    std::vector<std::uint8_t> results(runs * run_bytes);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
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
        if (rt::d3d12::device_push_constants_need_fallback(inner)) {
            std::printf("d3d12_set0_compute_test: skipped, the PSOs need 144 push-constant bytes and this GPU has %u\n",
                        rt::d3d12::device_max_push_constants_size(inner));
            return 77;
        }
        EXPECT(rt::d3d12::bindless_set(device.p) != VK_NULL_HANDLE);

        // Root signature, in the console compute layout (rt::rootsig::pack_stage, D3D-448): [0] two constants (s4: the
        // view slot's offset in dwords, s5: the first output element), [1] a root UAV (s6:s7, the output buffer),
        // [2] the table of views (s8, its 32-bit VA), [3] the table of samplers (s9).
        D3D12_DESCRIPTOR_RANGE view_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, 0};
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

        // The PSO: its shader has set-0 bindings, so its layout holds the device's set.
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.CS = {container.data(), container.size()};
        Ref<ID3D12PipelineState> pso;
        EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

        // A device without the bindless set has nothing such a shader could find its images in: the PSO is refused.
        {
            rt::d3d12::VulkanDeviceInfo plain_info = info;
            plain_info.bindless = plain_info.bindless_partially_bound = false;
            Ref<ID3D12Device> plain;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(plain_info, IID_PPV_ARGS(&plain.p))));
            EXPECT(rt::d3d12::bindless_set(plain.p) == VK_NULL_HANDLE);
            Ref<ID3D12RootSignature> plain_root;
            EXPECT(SUCCEEDED(plain->CreateRootSignature(0, root_blob.data(), root_blob.size(),
                                                      IID_PPV_ARGS(&plain_root.p))));
            D3D12_COMPUTE_PIPELINE_STATE_DESC plain_desc = pso_desc;
            plain_desc.pRootSignature = plain_root.p;
            Ref<ID3D12PipelineState> refused;
            EXPECT(plain->CreateComputePipelineState(&plain_desc, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);
        }

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
        Ref<ID3D12Resource> textures[2], upload, output, readback;
        for (auto& texture : textures)
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&texture.p))));
        // The upload buffer: texture A at 0, texture B at 2048 (8 rows of the 256-byte pitch).
        constexpr UINT64 b_offset = 2048;
        const auto upload_desc = buffer_desc(4096);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto output_desc = buffer_desc(results.size(), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &output_desc,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&output.p))));
        const auto readback_desc = buffer_desc(results.size());
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        device->GetCopyableFootprints(&texture_desc, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);
        EXPECT(footprint.Footprint.RowPitch == 256 && footprint.Footprint.RowPitch * texture_size <= b_offset);
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        for (UINT y = 0; y < texture_size; ++y)
            for (UINT x = 0; x < texture_size; ++x) {
                const Texel a = gradient(x, y), b = reflected(x, y);
                std::memcpy(up + y * footprint.Footprint.RowPitch + x * 4, a.data(), 4);
                std::memcpy(up + b_offset + y * footprint.Footprint.RowPitch + x * 4, b.data(), 4);
            }

        // The two heaps; their start handles come back in RAX (heap slots 9 and 10). Shader-visible heaps live in
        // the GPU address arena, so the GPU handles are console VAs and the bytes the views and the samplers wrote
        // are what the shader loads.
        const D3D12_DESCRIPTOR_HEAP_DESC view_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                                        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2,
                                                           D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> view_heap, sampler_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&view_heap_desc, IID_PPV_ARGS(&view_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&sampler_heap_desc, IID_PPV_ARGS(&sampler_heap.p))));
        const std::uint64_t view_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(view_heap.p, 9))(view_heap.p);
        const std::uint64_t view_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(view_heap.p, 10))(view_heap.p);
        const std::uint64_t sampler_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 9))(sampler_heap.p);
        const std::uint64_t sampler_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 10))(sampler_heap.p);
        const UINT view_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const UINT sampler_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        EXPECT(view_cpu && sampler_cpu && view_gpu >= 0x10000 && sampler_gpu >= 0x10000);
        EXPECT(view_gpu < (1ull << 32) && sampler_gpu < (1ull << 32));
        EXPECT(view_step >= 32 && view_step % 4 == 0 && sampler_step >= 16);

        for (UINT i = 0; i < 2; ++i) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D = {0, 1, 0, 0.0f};
            device->CreateShaderResourceView(textures[i].p, &srv,
                                             D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(view_cpu + i * view_step)});
        }
        // Slot 2: bytes no view was made for. The hash table has no entry for them.
        std::uint32_t unknown[8];
        for (std::uint32_t i = 0; i < 8; ++i) unknown[i] = 0x5a5a0000u + i;
        std::memcpy(reinterpret_cast<void*>(view_cpu + 2 * view_step), unknown, sizeof(unknown));
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, unknown) == 0);
        for (UINT i = 0; i < 2; ++i) {
            D3D12_SAMPLER_DESC sampler_desc{};
            sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
            sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW =
                i ? D3D12_TEXTURE_ADDRESS_MODE_CLAMP : D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            sampler_desc.MaxAnisotropy = 1;
            sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
            sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
            device->CreateSampler(&sampler_desc, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu + i * sampler_step)});
        }

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

        // Upload both textures and make them shader resources for a compute shader.
        for (UINT i = 0; i < 2; ++i) {
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = textures[i].p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = upload.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = footprint;
            src.PlacedFootprint.Offset = i ? b_offset : 0;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        D3D12_RESOURCE_BARRIER sampled[2] = {
            transition(textures[0].p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(textures[1].p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)};
        list->ResourceBarrier(2, sampled);

        // The four dispatches. Each keeps the slot, the output position and the sampler table it was recorded with.
        const D3D12_GPU_VIRTUAL_ADDRESS output_va = output->GetGPUVirtualAddress();
        EXPECT(output_va >= 0x10000 && output_va < (1ull << 32));
        ID3D12DescriptorHeap* heaps[] = {view_heap.p, sampler_heap.p};
        list->SetDescriptorHeaps(2, heaps);
        list->SetComputeRootSignature(root.p);
        list->SetPipelineState(pso.p);
        list->SetComputeRootUnorderedAccessView(1, output_va);
        list->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{view_gpu});
        const UINT view_slot[runs] = {0, 1, 2, 0};
        const UINT sampler_slot[runs] = {0, 0, 0, 1};
        for (UINT run = 0; run < runs; ++run) {
            const std::uint32_t offset_and_first[2] = {view_slot[run] * view_step / 4, run * texels};
            list->SetComputeRoot32BitConstants(0, 2, offset_and_first, 0);
            list->SetComputeRootDescriptorTable(
                3, D3D12_GPU_DESCRIPTOR_HANDLE{sampler_gpu + std::uint64_t(sampler_slot[run]) * sampler_step});
            list->Dispatch(1, 1, 1);
        }
        D3D12_RESOURCE_BARRIER to_copy = transition(output.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        list->CopyBufferRegion(readback.p, 0, output.p, 0, results.size());
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        std::memcpy(results.data(), mapped, results.size());

        for (auto& texture : textures) texture.reset();
        for (auto* r : {&upload, &output, &readback}) r->reset();
        pso.reset();
        root.reset();
        view_heap.reset();
        sampler_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    // Runs 0, 1 and 3 against the textures; run 2 is the fallback image, whose colour the test does not fix: it
    // must be one colour everywhere, which neither texture gives.
    const char* const names[runs] = {"slot 0, wrapping", "slot 1, wrapping", "the unknown slot", "slot 0, clamping"};
    unsigned mismatches = 0;
    for (UINT run = 0; run < runs; ++run) {
        const std::uint8_t* got_run = results.data() + run * run_bytes;
        for (UINT i = 0; i < texels; ++i) {
            const UINT x = i % texture_size, y = i / texture_size;
            const std::uint8_t* got = got_run + i * 4;
            Texel want{};
            if (run == 0) want = gradient(x, y);
            else if (run == 1) want = reflected(x, y);
            else if (run == 3) want = gradient(texture_size - 1, y);
            else std::memcpy(want.data(), got_run, 4);  // the fallback: texel 0 of the run, everywhere
            if (std::memcmp(got, want.data(), 4) == 0) continue;
            if (mismatches++ < 8)
                std::fprintf(stderr, "%s, texel %u: got %u,%u,%u,%u want %u,%u,%u,%u\n", names[run], i, got[0], got[1],
                             got[2], got[3], want[0], want[1], want[2], want[3]);
        }
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("set 0 from compute: four dispatches sampled through the bindless set the layer bound; %u texels "
                "matched for each of two SRVs and for a clamping sampler, and bytes no view was made for gave the "
                "fallback image\n", texels);
    return 0;
}
