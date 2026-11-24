// Milestone M4, the synthetic part (docs/d3d12_vulkan.md section 6): a compute workload with the structure a game's asset
// streaming and compute passes have, through the D3D12 API on the proxy, with compute shaders that are translator output
// (test/shaders/cs_m4_fill.asm and cs_m4_indirect.asm, hand-written RDNA2).
// Three queues' worth of work, each its own command list type:
//   COPY queue     streams an input buffer: CopyBufferRegion from an UPLOAD buffer into a DEFAULT buffer that was created
//                  in COMMON (promoted to COPY_DEST there), then Signal(copy fence, 1).
//   COMPUTE queue  Wait(copy fence, 1), then one list: a barrier COMMON -> UNORDERED_ACCESS for the input, a compute root
//                  signature (four root constants and a descriptor table of five UAVs: a structured buffer, a raw
//                  buffer, an R32_UINT RWTexture2D, the raw input and a raw output), a compute PSO, and Dispatch with
//                  four groups. That shader fills the structured buffer and the image from the input and writes the
//                  arguments of the next dispatch into the raw buffer. Barriers (a transition of the argument buffer to
//                  INDIRECT_ARGUMENT and UAV barriers on the buffer and the image) and ExecuteIndirect with a dispatch
//                  command signature runs a second PSO whose group count is that buffer's first word: it reads the
//                  structured buffer and the image back and writes the raw output. Everything then goes to COMMON and
//                  the queue does Signal(compute fence, 1).
//   COPY queue     Wait(compute fence, 1), a COPY list that copies the five results (four buffers and the image through
//                  CopyTextureRegion) to a READBACK buffer, Signal(copy fence, 2). The CPU waits for that with
//                  SetEventOnCompletion and an event.
// When the Vulkan family has two queues, the copy and the compute D3D12 queue get their own VkQueue (so the fences are
// real cross-queue semaphores); with one, they share it. Either way the compute batch is submitted before the copy that feeds it: the layer holds it until that Signal
// has been submitted, with the PSOs and root signature the application released in between.
// Compared with a CPU reference, word by word, error 0: the input after the round trip, the structured buffer, the
// argument buffer, the image and the indirect dispatch's output (which also shows that exactly the 3 groups the
// first dispatch asked for ran).
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation (0 errors required).
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cstring>

#ifndef LEONIDA_TEST_M4_DIR
#error "LEONIDA_TEST_M4_DIR must name the folder build.py wrote (spirv/index.json, fssm/cs_*.fssm)"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX

namespace {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

constexpr UINT records = 256;                 // fill: 4 groups of 64
constexpr UINT second_groups = 3;             // the indirect dispatch: 3 groups of 64
constexpr UINT image_dim = 16;                // 16 x 16 R32_UINT, one texel a record
constexpr UINT seed = 0x00c0ffee, multiplier = 0x01000193;
// The readback buffer: the structured buffer, the argument buffer's 16 bytes, the second output, the input, the image.
constexpr UINT64 data_bytes = records * 16, args_bytes = 16, out2_bytes = second_groups * 64 * 8, in_bytes = records * 4;
constexpr UINT64 offset_data = 0, offset_args = offset_data + data_bytes, offset_out2 = offset_args + args_bytes,
                 offset_in = offset_out2 + out2_bytes;
constexpr UINT64 offset_image = (offset_in + in_bytes + 511) / 512 * 512;
constexpr UINT image_pitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
constexpr UINT64 readback_bytes = offset_image + UINT64(image_dim) * image_pitch;

std::uint32_t rotr(std::uint32_t v, unsigned n) { return (v >> n) | (v << (32 - n)); }
std::uint32_t input_word(UINT g) { return (g + 1) * 0x9e3779b1u ^ (g << 7); }

struct Reference {
    std::vector<std::uint32_t> data, args, in, image, out2;
};
Reference reference() {
    Reference ref;
    ref.args = {second_groups, 1, 1, 0xc0de0001u};
    ref.data.resize(records * 4);
    ref.image.resize(image_dim * image_dim);
    for (UINT g = 0; g < records; ++g) {
        const std::uint32_t in = input_word(g);
        ref.in.push_back(in);
        ref.data[g * 4 + 0] = in + seed;
        ref.data[g * 4 + 1] = g * multiplier;
        ref.data[g * 4 + 2] = in ^ (g << 8);
        ref.data[g * 4 + 3] = rotr(in, 13);
        const UINT x = g & 15, y = g >> 4;
        ref.image[y * image_dim + x] = in * 3 + x * 0x10001u + y;
    }
    for (UINT g = 0; g < second_groups * 64; ++g) {
        const std::uint32_t t = ref.image[g];   // (g & 15, g >> 4) is texel g of the row-major image
        ref.out2.push_back((ref.data[g * 4 + 0] + t) ^ ref.data[g * 4 + 3]);
        ref.out2.push_back((ref.data[g * 4 + 2] ^ t) + ref.data[g * 4 + 1]);
    }
    return ref;
}

}  // namespace

int main() {
    vktest::Device d;
    d.queue_count_wanted = 2;   // the copy queue and the compute queue each get a VkQueue where the family has two
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    if (!d.bindless) {
        std::printf("SKIP: the device lacks the descriptor indexing features of the bindless set 0\n");
        return 77;
    }

    const std::filesystem::path root_dir = LEONIDA_TEST_M4_DIR;
    leonida::ShaderLibrary library;
    EXPECT(library.load((root_dir / "spirv").string()));
    const auto program = [&](const char* name) { return read_file(root_dir / "fssm" / (std::string(name) + ".fssm")); };
    const auto fill = program("cs_m4_fill"), indirect = program("cs_m4_indirect");
    EXPECT(fill.size() >= 0x150 && indirect.size() >= 0x150 && fill != indirect);
    EXPECT(library.find("cs", leonida::sha256_hex(fill)) && library.find("cs", leonida::sha256_hex(indirect)));
    const auto fill_container = shader_container(fill), indirect_container = shader_container(indirect);

    const bool distinct_queues = d.queues.size() >= 2;
    Reference want = reference();
    std::vector<std::uint8_t> got;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        if (distinct_queues) info.command_queues = d.queues;
        info.buffer_device_address = true;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.image_view_min_lod = d.image_view_min_lod;
        info.shader_dir = (root_dir / "spirv").string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        EXPECT(rt::d3d12::bindless_set(device.p) != VK_NULL_HANDLE);

        // [0] four 32-bit constants (s4..s7), [1] a table of five UAVs (s8, its 32-bit VA): the console compute layout
        // (D3D-448). The fixtures are built with LEONIDA_USER_DATA_SHIFT=4 (runtime/CMakeLists.txt).
        D3D12_DESCRIPTOR_RANGE uav_range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 5, 0, 0, 0};
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 4};
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable = {1, &uav_range};
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.CS = {fill_container.data(), fill_container.size()};
        Ref<ID3D12PipelineState> pso_fill, pso_indirect;
        EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso_fill.p))));
        pso_desc.CS = {indirect_container.data(), indirect_container.size()};
        EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso_indirect.p))));

        D3D12_INDIRECT_ARGUMENT_DESC argument{};
        argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        D3D12_COMMAND_SIGNATURE_DESC signature_desc{};
        signature_desc.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
        signature_desc.NumArgumentDescs = 1;
        signature_desc.pArgumentDescs = &argument;
        Ref<ID3D12CommandSignature> signature;
        EXPECT(SUCCEEDED(device->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature.p))));

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        constexpr auto uav_flag = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        constexpr auto uav_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        constexpr auto common = D3D12_RESOURCE_STATE_COMMON;
        Ref<ID3D12Resource> upload, input, data, args, out2, image, readback;
        const auto create_buffer = [&](Ref<ID3D12Resource>& out, const D3D12_HEAP_PROPERTIES& heap, UINT64 bytes,
                                       D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
            const auto desc = buffer_desc(bytes, flags);
            return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                             IID_PPV_ARGS(&out.p)));
        };
        EXPECT(create_buffer(upload, upload_heap, in_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ));
        EXPECT(create_buffer(input, default_heap, in_bytes, uav_flag, common));
        EXPECT(create_buffer(data, default_heap, data_bytes, uav_flag, uav_state));
        EXPECT(create_buffer(args, default_heap, 256, uav_flag, uav_state));
        EXPECT(create_buffer(out2, default_heap, out2_bytes, uav_flag, uav_state));
        EXPECT(create_buffer(readback, readback_heap, readback_bytes, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_COPY_DEST));
        D3D12_RESOURCE_DESC image_desc{};
        image_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        image_desc.Width = image_desc.Height = image_dim;
        image_desc.DepthOrArraySize = 1;
        image_desc.MipLevels = 1;
        image_desc.Format = DXGI_FORMAT_R32_UINT;
        image_desc.SampleDesc.Count = 1;
        image_desc.Flags = uav_flag;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &image_desc, uav_state,
                                                         nullptr, IID_PPV_ARGS(&image.p))));
        {
            std::uint32_t* mapped = nullptr;
            EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
            for (UINT g = 0; g < records; ++g) mapped[g] = input_word(g);
            upload->Unmap(0, nullptr);
        }

        // The table: slot 0 structured buffer (stride 16), 1 raw argument buffer, 2 image, 3 raw input, 4 raw output.
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 5,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
        const std::uint64_t cpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 9))(heap.p);
        const std::uint64_t gpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 10))(heap.p);
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(cpu && gpu >= 0x10000 && gpu < (1ull << 32) && step == 32);   // the shaders step 32 bytes a slot
        const auto handle = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu + std::uint64_t(i) * step)}; };
        const auto buffer_view = [](DXGI_FORMAT format, UINT count, UINT stride, D3D12_BUFFER_UAV_FLAGS flags) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
            view.Format = format;
            view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            view.Buffer = {0, count, stride, 0, flags};
            return view;
        };
        auto view = buffer_view(DXGI_FORMAT_UNKNOWN, records, 16, D3D12_BUFFER_UAV_FLAG_NONE);
        device->CreateUnorderedAccessView(data.p, nullptr, &view, handle(0));
        view = buffer_view(DXGI_FORMAT_R32_TYPELESS, 4, 0, D3D12_BUFFER_UAV_FLAG_RAW);   // the 16 bytes of arguments
        device->CreateUnorderedAccessView(args.p, nullptr, &view, handle(1));
        D3D12_UNORDERED_ACCESS_VIEW_DESC image_view{};
        image_view.Format = DXGI_FORMAT_R32_UINT;
        image_view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device->CreateUnorderedAccessView(image.p, nullptr, &image_view, handle(2));
        view = buffer_view(DXGI_FORMAT_R32_TYPELESS, records, 0, D3D12_BUFFER_UAV_FLAG_RAW);
        device->CreateUnorderedAccessView(input.p, nullptr, &view, handle(3));
        view = buffer_view(DXGI_FORMAT_R32_TYPELESS, UINT(out2_bytes / 4), 0, D3D12_BUFFER_UAV_FLAG_RAW);
        device->CreateUnorderedAccessView(out2.p, nullptr, &view, handle(4));
        // The V# of the structured view and the T# of the image view are what the shader loads from the table.
        std::uint32_t v_data[4], v_args[4];
        std::memcpy(v_data, reinterpret_cast<const void*>(handle(0).ptr), sizeof(v_data));
        std::memcpy(v_args, reinterpret_cast<const void*>(handle(1).ptr), sizeof(v_args));
        EXPECT(v_data[0] == std::uint32_t(data->GetGPUVirtualAddress()) && (v_data[1] >> 16 & 0x3fff) == 16 &&
               v_data[2] == records);
        EXPECT(v_args[0] == std::uint32_t(args->GetGPUVirtualAddress()) && (v_args[1] >> 16 & 0x3fff) == 0 &&
               v_args[2] == 16);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> copy_queue, compute_queue;
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&copy_queue.p))));
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&compute_queue.p))));
        Ref<ID3D12CommandAllocator> stream_alloc, compute_alloc, readback_alloc;
        Ref<ID3D12GraphicsCommandList> stream_list, compute_list, readback_list;
        Ref<ID3D12Fence> copy_fence, compute_fence;
        const auto make_list = [&](D3D12_COMMAND_LIST_TYPE type, Ref<ID3D12CommandAllocator>& alloc,
                                   Ref<ID3D12GraphicsCommandList>& list) {
            return SUCCEEDED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&alloc.p))) &&
                   SUCCEEDED(device->CreateCommandList(0, type, alloc.p, nullptr, IID_PPV_ARGS(&list.p)));
        };
        EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COPY, stream_alloc, stream_list));
        EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COMPUTE, compute_alloc, compute_list));
        EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COPY, readback_alloc, readback_list));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&copy_fence.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&compute_fence.p))));

        // Copy list 1: stream the input in (the buffer is in COMMON and is promoted to COPY_DEST by the copy). This is the
        // first submission after the UAV views were written, and a copy queue: it must not initialize the storage image
        // (the compute queue does, in the submission that dispatches), or the data written later would be at risk.
        stream_list->CopyBufferRegion(input.p, 0, upload.p, 0, in_bytes);
        EXPECT(stream_list->Close() == S_OK);

        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {offset_image, {DXGI_FORMAT_R32_UINT, image_dim, image_dim, 1, image_pitch}};
        src.pResource = image.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

        // Compute list.
        ID3D12DescriptorHeap* heaps[] = {heap.p};
        D3D12_RESOURCE_BARRIER acquire = transition(input.p, common, uav_state);
        compute_list->ResourceBarrier(1, &acquire);
        compute_list->SetDescriptorHeaps(1, heaps);
        compute_list->SetComputeRootSignature(root.p);
        compute_list->SetPipelineState(pso_fill.p);
        const std::uint32_t constants[4] = {seed, multiplier, second_groups, 0};
        compute_list->SetComputeRoot32BitConstants(0, 4, constants, 0);
        compute_list->SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{gpu});
        compute_list->Dispatch(records / 64, 1, 1);
        D3D12_RESOURCE_BARRIER between[3] = {transition(args.p, uav_state, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
                                             {}, {}};
        for (int i : {1, 2}) {
            between[i].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            between[i].UAV.pResource = i == 1 ? data.p : image.p;
        }
        compute_list->ResourceBarrier(3, between);
        compute_list->SetPipelineState(pso_indirect.p);   // the root arguments stay
        compute_list->ExecuteIndirect(signature.p, 1, args.p, 0, nullptr, 0);
        D3D12_RESOURCE_BARRIER release[5] = {transition(data.p, uav_state, common),
                                             transition(args.p, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, common),
                                             transition(out2.p, uav_state, common),
                                             transition(input.p, uav_state, common),
                                             transition(image.p, uav_state, common)};
        compute_list->ResourceBarrier(5, release);
        EXPECT(compute_list->Close() == S_OK);

        // Copy list 2: the results to the readback buffer, once the compute queue is done.
        readback_list->CopyBufferRegion(readback.p, offset_data, data.p, 0, data_bytes);
        readback_list->CopyBufferRegion(readback.p, offset_args, args.p, 0, args_bytes);
        readback_list->CopyBufferRegion(readback.p, offset_out2, out2.p, 0, out2_bytes);
        readback_list->CopyBufferRegion(readback.p, offset_in, input.p, 0, in_bytes);
        readback_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        EXPECT(readback_list->Close() == S_OK);

        const auto execute = [&](ID3D12CommandQueue* queue, ID3D12GraphicsCommandList* list) {
            ID3D12CommandList* lists[] = {list};
            queue->ExecuteCommandLists(1, lists);
            return rt::d3d12::last_execute_result(queue) == leonida::ReplayResult::success;
        };
        const auto submit_copy = [&] {
            return execute(copy_queue.p, stream_list.p) && SUCCEEDED(copy_queue->Signal(copy_fence.p, 1));
        };
        const auto submit_readback = [&] {
            return SUCCEEDED(copy_queue->Wait(compute_fence.p, 1)) && execute(copy_queue.p, readback_list.p) &&
                   SUCCEEDED(copy_queue->Signal(copy_fence.p, 2));
        };
        const auto submit_compute = [&] {
            return SUCCEEDED(compute_queue->Wait(copy_fence.p, 1)) && execute(compute_queue.p, compute_list.p) &&
                   SUCCEEDED(compute_queue->Signal(compute_fence.p, 1));
        };
        // The compute list goes first, before the copy that feeds it: it waits for a Signal not yet submitted, so the
        // layer holds it until the copy has been submitted, and its layouts follow the dependency order (docs section
        // 4.4; d3d12_queue_handover_test covers every order). The application may release what the list uses as soon
        // as ExecuteCommandLists returns: the pipeline states and the root signature go and the list is Reset, all
        // while the batch is still held, so the layer has to keep them until it runs.
        EXPECT(submit_compute());
        pso_fill.reset();
        pso_indirect.reset();
        root.reset();
        EXPECT(compute_list->Reset(compute_alloc.p, nullptr) == S_OK && compute_list->Close() == S_OK);
        EXPECT(submit_copy() && submit_readback());

        const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        EXPECT(event != nullptr);
        EXPECT(SUCCEEDED(copy_fence->SetEventOnCompletion(2, event)));
        if (WaitForSingleObject(event, 20000) != WAIT_OBJECT_0) {
            std::fprintf(stderr, "the copy fence never reached 2 (compute fence at %llu)\n",
                         static_cast<unsigned long long>(compute_fence->GetCompletedValue()));
            return 1;
        }
        CloseHandle(event);
        EXPECT(copy_fence->GetCompletedValue() >= 2 && compute_fence->GetCompletedValue() >= 1);

        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        got.assign(static_cast<const std::uint8_t*>(mapped), static_cast<const std::uint8_t*>(mapped) + readback_bytes);
        readback->Unmap(0, nullptr);

        for (auto* r : {&upload, &input, &data, &args, &out2, &image, &readback}) r->reset();
        stream_list.reset(); compute_list.reset(); readback_list.reset();
        stream_alloc.reset(); compute_alloc.reset(); readback_alloc.reset();
        copy_fence.reset(); compute_fence.reset();
        signature.reset(); pso_fill.reset(); pso_indirect.reset(); root.reset(); heap.reset();
        copy_queue.reset(); compute_queue.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    unsigned mismatches = 0, words = 0;
    const auto compare = [&](const char* what, UINT64 offset, const std::vector<std::uint32_t>& expected, UINT pitch,
                             UINT row_words) {
        unsigned bad = 0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            std::uint32_t value = 0;
            const UINT64 at = offset + (pitch ? (i / row_words) * pitch + (i % row_words) * 4 : i * 4);
            std::memcpy(&value, got.data() + at, 4);
            ++words;
            if (value == expected[i]) continue;
            ++mismatches;
            if (bad++ < 4) std::fprintf(stderr, "%s word %zu: got %08x want %08x\n", what, i, value, expected[i]);
        }
        if (bad) std::fprintf(stderr, "%s: %u of %zu words differ\n", what, bad, expected.size());
    };
    compare("input (copy queue round trip)", offset_in, want.in, 0, 0);
    compare("structured buffer", offset_data, want.data, 0, 0);
    compare("dispatch arguments", offset_args, want.args, 0, 0);
    compare("image", offset_image, want.image, image_pitch, image_dim);
    compare("indirect dispatch output", offset_out2, want.out2, 0, 0);
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("M4: a COPY queue streamed an input buffer in, a COMPUTE queue ran a compute PSO over a UAV table "
                "(structured buffer, raw buffers, an R32_UINT storage image) and a second dispatch through "
                "ExecuteIndirect from arguments the first wrote, and a COPY queue behind a cross-queue fence read the "
                "results back (%s): %u words matched the CPU reference (error 0)\n",
                distinct_queues ? "two VkQueues" : "one VkQueue", words);
    return 0;
}
