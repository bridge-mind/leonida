// Typed buffer UAVs and UAV counters (docs/d3d12_vulkan.md section 3.4, "Buffer UAVs"), through the D3D12 API on the
// proxy, with compute shaders that are translator output (test/shaders/cs_buf_typed.asm and cs_buf_counter.asm,
// hand-written RDNA2), plus D3D12-only behaviour compared with the host runtime on the same adapter.
//
// Part 1, one DIRECT list: a compute root signature (four root constants and a table of six UAVs), two compute PSOs.
//   cs_buf_typed    writes typed buffer views (R32_UINT with an atomic add, R16G16B16A16_FLOAT, R8G8B8A8_UNORM from
//                   FirstElement 16, R32G32B32A32_UINT, R32_SINT) and reads an R32_FLOAT view. Two views are shorter
//                   than the dispatch (240 and 250 elements of 256): the V#'s NUM_RECORDS is the bound, so stores past
//                   it leave the buffer's sentinel and loads past it give 0.
//   cs_buf_counter  a structured UAV with a counter at CounterOffsetInBytes 4096 of another buffer: 256 lanes append
//                   (buffer_atomic_add on the counter V# in the slot's upper half, store at the returned index), then
//                   128 lanes consume (buffer_atomic_sub, load the record just below). "CopyStructureCount" in D3D12
//                   is CopyBufferRegion of the counter's 4 bytes into a buffer: read after the append (256) and after
//                   the consume (128). Lane order is up to the GPU, so records are compared as sets.
//   Everything is compared with a CPU reference, word by word, error 0.
// Part 2, the layer's device (no proxy) and the host runtime on the SAME adapter (LUID): typed buffer UAVs created with
// CreateUnorderedAccessView and cleared with ClearUnorderedAccessViewUint / Float (thirteen formats, sub-ranges, an
// element range ending off a word boundary), and a structured UAV with a counter: a clear of the structured view and
// of the counter's own word through a raw view. The bytes read back are compared with the host's; with no host D3D12
// the layer's are compared with the model of uav_clear.h.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation (0 errors required).
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "d3d12_copy_compare.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/gfx10_descriptors.h"
#include "../src/shader_library.h"
#include "../src/uav_clear.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>

#ifndef LEONIDA_TEST_BUFFER_UAV_DIR
#error "LEONIDA_TEST_BUFFER_UAV_DIR must name the folder build.py wrote (spirv/index.json, fssm/cs_*.fssm)"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX
namespace g = leonida::gfx10;
namespace uc = leonida::uav_clear;

namespace {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

constexpr UINT lanes = 256;
constexpr UINT seed = 0x00c0ffee, multiplier = 0x01000193;
constexpr std::uint32_t sentinel = 0xa5a5a5a5u;
constexpr UINT out0_elements = 240, in4_elements = 250, out2_first = 16;
constexpr UINT64 counter_offset = 4096, counter_buffer_bytes = 8192;
constexpr UINT consumed = 128;

// The buffers of part 1: sizes, in the order of their slices of the upload and of the readback buffer.
enum Buf { out0, out1, out2, out3, in4, out5, records, counter, consumed_out, buffer_count };
constexpr UINT64 sizes[buffer_count] = {lanes * 4, lanes * 8, (out2_first + lanes) * 4, lanes * 16, lanes * 4,
                                        lanes * 4, lanes * 8, counter_buffer_bytes, consumed * 8};
constexpr UINT64 slice(Buf which) {
    UINT64 at = 0;
    for (int i = 0; i < which; ++i) at += (sizes[i] + 255) / 256 * 256;
    return at;
}
constexpr UINT64 total_bytes = slice(consumed_out) + (sizes[consumed_out] + 255) / 256 * 256;
constexpr UINT64 count_after_append = total_bytes, count_after_consume = total_bytes + 4;
constexpr UINT64 readback_bytes = total_bytes + 8;

std::uint16_t half_bits(float value) {   // exact for the values the reference uses (finite, in the normal range)
    const std::uint32_t f = std::bit_cast<std::uint32_t>(value);
    if (!(f & 0x7fffffffu)) return std::uint16_t(f >> 16);
    const int exponent = int((f >> 23) & 0xff) - 127 + 15;
    return std::uint16_t(((f >> 16) & 0x8000) | (exponent << 10) | ((f >> 13) & 0x3ff));
}

float input_value(UINT g) { return float(g) - 100.5f; }

struct Reference {
    std::vector<std::uint32_t> words[buffer_count];
};

Reference reference() {
    Reference ref;
    for (int i = 0; i < buffer_count; ++i) ref.words[i].assign(sizes[i] / 4, sentinel);
    for (UINT g = 0; g < lanes; ++g) {
        if (g < out0_elements) ref.words[out0][g] = (g * multiplier + seed) + g;
        const float x = float(g);
        ref.words[out1][g * 2] = half_bits(x) | std::uint32_t(half_bits(x + 0.5f)) << 16;
        ref.words[out1][g * 2 + 1] = half_bits(-1.0f - x) | std::uint32_t(half_bits(0.25f)) << 16;
        const std::uint32_t k = g & 255;
        ref.words[out2][out2_first + g] = k | (255 - k) << 8 | 0xff000000u;
        ref.words[out3][g * 4 + 0] = g;
        ref.words[out3][g * 4 + 1] = g ^ 0xffff;
        ref.words[out3][g * 4 + 2] = g * 7;
        ref.words[out3][g * 4 + 3] = 0xa5;
        ref.words[in4][g] = std::bit_cast<std::uint32_t>(input_value(g));
        ref.words[out5][g] = g < in4_elements ? std::uint32_t(int(2.0f * input_value(g))) : 0;
    }
    return ref;
}

D3D12_RESOURCE_BARRIER uav_barrier(ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = resource;
    return b;
}

D3D12_UNORDERED_ACCESS_VIEW_DESC buffer_view(DXGI_FORMAT format, UINT64 first, UINT count, UINT stride,
                                             UINT64 counter_at = 0, D3D12_BUFFER_UAV_FLAGS flags = D3D12_BUFFER_UAV_FLAG_NONE) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.Format = format;
    view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    view.Buffer = {first, count, stride, counter_at, flags};
    return view;
}

// What a V# in a slot says, as the shader loads it.
struct Vsharp {
    std::uint64_t address;
    std::uint32_t stride, records, format, oob;
};
Vsharp vsharp(const void* slot_pointer, unsigned dword = 0) {
    std::uint32_t w[4];
    std::memcpy(w, static_cast<const std::uint32_t*>(slot_pointer) + dword, sizeof(w));
    return {(std::uint64_t(w[1] & 0xffff) << 32) | w[0], w[1] >> 16 & 0x3fff, w[2], w[3] >> 12 & 0x7f, w[3] >> 28 & 3};
}

// ---- part 1 -----------------------------------------------------------------------------------------------------

int run_shaders(vktest::Device& d, const std::filesystem::path& root_dir, unsigned& words_checked) {
    leonida::ShaderLibrary library;
    EXPECT(library.load((root_dir / "spirv").string()));
    const auto program = [&](const char* name) { return read_file(root_dir / "fssm" / (std::string(name) + ".fssm")); };
    const auto typed_fssm = program("cs_buf_typed"), counter_fssm = program("cs_buf_counter");
    EXPECT(typed_fssm.size() >= 0x150 && counter_fssm.size() >= 0x150 && typed_fssm != counter_fssm);
    EXPECT(library.find("cs", leonida::sha256_hex(typed_fssm)) && library.find("cs", leonida::sha256_hex(counter_fssm)));
    const auto typed_container = shader_container(typed_fssm), counter_container = shader_container(counter_fssm);

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
    info.shader_dir = (root_dir / "spirv").string();
    ID3D12Device* inner = nullptr;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                       reinterpret_cast<void**>(&device.p))));
    rt::d3d12::set_device_outer(inner, device.p);
    EXPECT(rt::d3d12::bindless_set(device.p) != VK_NULL_HANDLE);

    D3D12_DESCRIPTOR_RANGE uav_range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 6, 0, 0, 0};
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
    pso_desc.CS = {typed_container.data(), typed_container.size()};
    Ref<ID3D12PipelineState> pso_typed, pso_counter;
    EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso_typed.p))));
    pso_desc.CS = {counter_container.data(), counter_container.size()};
    EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso_counter.p))));

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT}, upload_heap{D3D12_HEAP_TYPE_UPLOAD},
        readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> upload, readback, buf[buffer_count];
    const auto create = [&](Ref<ID3D12Resource>& out, const D3D12_HEAP_PROPERTIES& heap, UINT64 bytes,
                            D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
        const auto desc = buffer_desc(bytes, flags);
        return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                         IID_PPV_ARGS(&out.p)));
    };
    EXPECT(create(upload, upload_heap, total_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ));
    EXPECT(create(readback, readback_heap, readback_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST));
    for (int i = 0; i < buffer_count; ++i)
        EXPECT(create(buf[i], default_heap, sizes[i], D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                      D3D12_RESOURCE_STATE_COPY_DEST));
    Reference want = reference();
    {
        std::uint8_t* mapped = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
        for (UINT64 i = 0; i < total_bytes / 4; ++i) std::memcpy(mapped + i * 4, &sentinel, 4);
        // The input buffer holds floats; the counter's own word starts at 0 (the rest of its buffer is sentinel).
        std::memcpy(mapped + slice(in4), want.words[in4].data(), sizes[in4]);
        std::memset(mapped + slice(counter) + counter_offset, 0, 4);
        upload->Unmap(0, nullptr);
    }

    // The table: slots 0..5 typed views for cs_buf_typed; slot 6 a structured UAV with a counter, slot 7 a raw output
    // for cs_buf_counter (its table starts at slot 6).
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 12,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Ref<ID3D12DescriptorHeap> heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
    const std::uint64_t cpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 9))(heap.p);
    const std::uint64_t gpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 10))(heap.p);
    const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    EXPECT(cpu && gpu >= 0x10000 && gpu < (1ull << 32) && step == 32);   // the shaders step 32 bytes a slot
    const auto handle = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu + std::uint64_t(i) * step)}; };
    const auto make = [&](UINT i, Buf which, const D3D12_UNORDERED_ACCESS_VIEW_DESC& view, ID3D12Resource* counter_res = nullptr) {
        device->CreateUnorderedAccessView(buf[which].p, counter_res, &view, handle(i));
    };
    make(0, out0, buffer_view(DXGI_FORMAT_R32_UINT, 0, out0_elements, 0));
    make(1, out1, buffer_view(DXGI_FORMAT_R16G16B16A16_FLOAT, 0, lanes, 0));
    make(2, out2, buffer_view(DXGI_FORMAT_R8G8B8A8_UNORM, out2_first, lanes, 0));
    make(3, out3, buffer_view(DXGI_FORMAT_R32G32B32A32_UINT, 0, lanes, 0));
    make(4, in4, buffer_view(DXGI_FORMAT_R32_FLOAT, 0, in4_elements, 0));
    make(5, out5, buffer_view(DXGI_FORMAT_R32_SINT, 0, lanes, 0));
    make(6, records, buffer_view(DXGI_FORMAT_UNKNOWN, 0, lanes, 8, counter_offset), buf[counter].p);
    make(7, consumed_out, buffer_view(DXGI_FORMAT_R32_TYPELESS, 0, consumed * 2, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW));

    // The V#s the shaders load: format code, element size as stride, element count, OOB_SELECT 1 (structured); the
    // R8G8B8A8 view starts 16 elements in; the counter V# is raw (stride 0, 4 bytes, OOB_SELECT 3) in the slot's upper half.
    const auto at = [&](UINT i) { return reinterpret_cast<const void*>(handle(i).ptr); };
    const auto expect_typed = [&](UINT i, Buf which, UINT code, UINT bytes, UINT elements, UINT first) {
        const Vsharp v = vsharp(at(i));
        return v.address == buf[which]->GetGPUVirtualAddress() + UINT64(first) * bytes && v.stride == bytes &&
               v.records == elements && v.format == code && v.oob == 1;
    };
    EXPECT(expect_typed(0, out0, 20, 4, out0_elements, 0));          // GFX10_FORMAT_32_UINT
    EXPECT(expect_typed(1, out1, 71, 8, lanes, 0));                  // 16_16_16_16_FLOAT
    EXPECT(expect_typed(2, out2, 56, 4, lanes, out2_first));         // 8_8_8_8_UNORM
    EXPECT(expect_typed(3, out3, 75, 16, lanes, 0));                 // 32_32_32_32_UINT
    EXPECT(expect_typed(4, in4, 22, 4, in4_elements, 0));            // 32_FLOAT
    EXPECT(expect_typed(5, out5, 21, 4, lanes, 0));                  // 32_SINT
    {
        const Vsharp v = vsharp(at(6)), c = vsharp(at(6), 4);
        EXPECT(v.address == buf[records]->GetGPUVirtualAddress() && v.stride == 8 && v.records == lanes && v.format == 0 &&
               v.oob == 0);
        EXPECT(c.address == buf[counter]->GetGPUVirtualAddress() + counter_offset && c.stride == 0 && c.records == 4 &&
               c.format == 0 && c.oob == 3);
        const Vsharp r = vsharp(at(7));
        EXPECT(r.address == buf[consumed_out]->GetGPUVirtualAddress() && r.stride == 0 && r.records == consumed * 8 &&
               r.oob == 3);
        // A view with no counter leaves the slot's upper half zero.
        const Vsharp none = vsharp(at(7), 4);
        EXPECT(!none.address && !none.records);
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    constexpr auto uav_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    for (int i = 0; i < buffer_count; ++i) list->CopyBufferRegion(buf[i].p, 0, upload.p, slice(Buf(i)), sizes[i]);
    D3D12_RESOURCE_BARRIER barriers[buffer_count];
    for (int i = 0; i < buffer_count; ++i) barriers[i] = transition(buf[i].p, D3D12_RESOURCE_STATE_COPY_DEST, uav_state);
    list->ResourceBarrier(buffer_count, barriers);
    ID3D12DescriptorHeap* heaps[] = {heap.p};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(root.p);
    const auto constants = [&](std::uint32_t a, std::uint32_t b) {
        const std::uint32_t values[4] = {a, b, 0, 0};
        list->SetComputeRoot32BitConstants(0, 4, values, 0);
    };
    list->SetPipelineState(pso_typed.p);
    constants(seed, multiplier);
    list->SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{gpu});
    list->Dispatch(lanes / 64, 1, 1);
    list->SetPipelineState(pso_counter.p);
    constants(0, 0);   // append
    list->SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{gpu + 6ull * step});
    list->Dispatch(lanes / 64, 1, 1);
    D3D12_RESOURCE_BARRIER sync[2] = {uav_barrier(buf[records].p), uav_barrier(buf[counter].p)};
    list->ResourceBarrier(2, sync);
    // CopyStructureCount: the counter's 4 bytes into a buffer, after the append.
    D3D12_RESOURCE_BARRIER to_copy = transition(buf[counter].p, uav_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    list->CopyBufferRegion(readback.p, count_after_append, buf[counter].p, counter_offset, 4);
    D3D12_RESOURCE_BARRIER back = transition(buf[counter].p, D3D12_RESOURCE_STATE_COPY_SOURCE, uav_state);
    list->ResourceBarrier(1, &back);
    constants(1, 0);   // consume
    list->Dispatch(consumed / 64, 1, 1);
    for (int i = 0; i < buffer_count; ++i) barriers[i] = transition(buf[i].p, uav_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(buffer_count, barriers);
    for (int i = 0; i < buffer_count; ++i) list->CopyBufferRegion(readback.p, slice(Buf(i)), buf[i].p, 0, sizes[i]);
    list->CopyBufferRegion(readback.p, count_after_consume, buf[counter].p, counter_offset, 4);
    EXPECT(list->Close() == S_OK);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(event != nullptr);
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, event)));
    if (WaitForSingleObject(event, 20000) != WAIT_OBJECT_0) {
        std::fprintf(stderr, "the fence never reached 1\n");
        return 1;
    }
    CloseHandle(event);

    std::vector<std::uint8_t> got;
    {
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        got.assign(static_cast<const std::uint8_t*>(mapped), static_cast<const std::uint8_t*>(mapped) + readback_bytes);
        readback->Unmap(0, nullptr);
    }
    const auto word_at = [&](UINT64 offset) {
        std::uint32_t value = 0;
        std::memcpy(&value, got.data() + offset, 4);
        return value;
    };

    unsigned mismatches = 0;
    const auto compare = [&](const char* what, Buf which, const std::vector<std::uint32_t>& expected) {
        unsigned bad = 0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            ++words_checked;
            const std::uint32_t value = word_at(slice(which) + i * 4);
            if (value == expected[i]) continue;
            ++mismatches;
            if (bad++ < 4) std::fprintf(stderr, "%s word %zu: got %08x want %08x\n", what, i, value, expected[i]);
        }
        if (bad) std::fprintf(stderr, "%s: %u of %zu words differ\n", what, bad, expected.size());
    };
    compare("R32_UINT store + atomic add (240 of 256 elements)", out0, want.words[out0]);
    compare("R16G16B16A16_FLOAT store", out1, want.words[out1]);
    compare("R8G8B8A8_UNORM store (FirstElement 16)", out2, want.words[out2]);
    compare("R32G32B32A32_UINT store", out3, want.words[out3]);
    compare("R32_FLOAT input untouched", in4, want.words[in4]);
    compare("R32_FLOAT load -> R32_SINT store (250 of 256 loads in range)", out5, want.words[out5]);

    // The counter: 256 after the append, 128 after the consume, and nothing else of its buffer touched.
    ++words_checked;
    if (word_at(count_after_append) != lanes) {
        std::fprintf(stderr, "counter after the append: %u, want %u\n", word_at(count_after_append), lanes);
        ++mismatches;
    }
    ++words_checked;
    if (word_at(count_after_consume) != lanes - consumed) {
        std::fprintf(stderr, "counter after the consume: %u, want %u\n", word_at(count_after_consume), lanes - consumed);
        ++mismatches;
    }
    for (UINT64 i = 0; i < counter_buffer_bytes / 4; ++i) {
        ++words_checked;
        const std::uint32_t expected = i == counter_offset / 4 ? lanes - consumed : sentinel;
        if (word_at(slice(counter) + i * 4) != expected && mismatches++ < 12)
            std::fprintf(stderr, "counter buffer word %llu: got %08x want %08x\n", static_cast<unsigned long long>(i),
                         word_at(slice(counter) + i * 4), expected);
    }
    // The appended records: every g exactly once with its second field, in whatever order the lanes took the counter.
    std::vector<std::array<std::uint32_t, 2>> appended, taken;
    for (UINT i = 0; i < lanes; ++i)
        appended.push_back({word_at(slice(records) + i * 8), word_at(slice(records) + i * 8 + 4)});
    std::vector<std::array<std::uint32_t, 2>> sorted = appended;
    std::sort(sorted.begin(), sorted.end());
    for (UINT i = 0; i < lanes; ++i) {
        ++words_checked;
        if (sorted[i][0] != i || sorted[i][1] != i * 3 + 1) {
            if (mismatches++ < 12)
                std::fprintf(stderr, "appended record %u (sorted): { %u, %u }, want { %u, %u }\n", i, sorted[i][0],
                             sorted[i][1], i, i * 3 + 1);
        }
    }
    // The consumed ones: the records at indices 128..255 (the counter went from 256 down to 128), each once.
    for (UINT i = 0; i < consumed; ++i)
        taken.push_back({word_at(slice(consumed_out) + i * 8), word_at(slice(consumed_out) + i * 8 + 4)});
    std::vector<std::array<std::uint32_t, 2>> tail(appended.begin() + (lanes - consumed), appended.end());
    std::sort(taken.begin(), taken.end());
    std::sort(tail.begin(), tail.end());
    for (UINT i = 0; i < consumed; ++i) {
        ++words_checked;
        if (taken[i] != tail[i] && mismatches++ < 12)
            std::fprintf(stderr, "consumed record %u (sorted): { %u, %u }, want { %u, %u }\n", i, taken[i][0], taken[i][1],
                         tail[i][0], tail[i][1]);
    }
    EXPECT(!mismatches);

    for (auto& b : buf) b.reset();
    upload.reset(); readback.reset(); list.reset(); allocator.reset(); fence.reset(); queue.reset();
    pso_typed.reset(); pso_counter.reset(); root.reset(); heap.reset();
    EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    return 0;
}

// ---- part 2 -----------------------------------------------------------------------------------------------------

struct ClearCase {
    DXGI_FORMAT format;
    VkFormat vulkan;   // for the model
    bool is_float;
    std::array<std::uint32_t, 4> values;
    UINT64 byte_offset;
    UINT elements;
};
std::uint32_t fb(float f) { return std::bit_cast<std::uint32_t>(f); }

// Values in the domain D3D12 defines the same on every driver: no exact .5 ties in a float -> UNORM rounding (the RTX 5070's
// D3D12 rounds 127.5 down, the layer and the spec's round-to-nearest up) and no integer wider than its channel (the low bits
// are kept here, one vendor clamps).
const ClearCase clear_cases[] = {
    {DXGI_FORMAT_R32_UINT, VK_FORMAT_R32_UINT, false, {0x12345678, 0, 0, 0}, 0, 16},
    {DXGI_FORMAT_R32_SINT, VK_FORMAT_R32_SINT, false, {0xfffffff0, 0, 0, 0}, 256, 8},
    {DXGI_FORMAT_R32_FLOAT, VK_FORMAT_R32_SFLOAT, true, {fb(1.5f), 0, 0, 0}, 512, 8},
    {DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, true, {fb(1.0f), fb(0.5f), fb(-2.0f), fb(0.25f)}, 768, 8},
    {DXGI_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, true, {fb(1.0f), fb(0.4f), fb(0.2f), fb(2.0f)}, 1024, 16},
    {DXGI_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_UINT, false, {0xf1, 2, 3, 0x80}, 1280, 16},
    {DXGI_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_UINT, false, {1, 2, 3, 4}, 1536, 4},
    {DXGI_FORMAT_R16_UNORM, VK_FORMAT_R16_UNORM, true, {fb(0.25f), 0, 0, 0}, 1792, 16},
    {DXGI_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, true, {fb(0.2f), 0, 0, 0}, 2048, 21},   // ends off a word boundary
    {DXGI_FORMAT_R10G10B10A2_UNORM, VK_FORMAT_A2B10G10R10_UNORM_PACK32, true, {fb(0.25f), fb(0.4f), fb(0.75f), fb(1.0f)}, 2304, 8},
    {DXGI_FORMAT_R11G11B10_FLOAT, VK_FORMAT_B10G11R11_UFLOAT_PACK32, true, {fb(1.0f), fb(2.0f), fb(0.5f), 0}, 2560, 8},
    {DXGI_FORMAT_R16G16_FLOAT, VK_FORMAT_R16G16_SFLOAT, true, {fb(1.5f), fb(-0.5f), 0, 0}, 2816, 8},
    {DXGI_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_UINT, false, {0xaa, 0xbb, 0, 0}, 3072, 10},
};
constexpr UINT64 api_data_bytes = 4096, api_records_bytes = 4096;
constexpr UINT api_readback = api_data_bytes + api_records_bytes + counter_buffer_bytes;

// The same scenario on `device`: bytes read back, or empty on failure. host: a real heap pair (a non-shader-visible one the
// views are made in, a shader-visible one the clear's GPU handle comes from), as the PC runtime wants; the layer takes the
// CPU handle and ignores the GPU one.
bool run_api(ID3D12Device* device, bool host, std::vector<std::uint8_t>& out, std::vector<bool>& supported_by_host) {
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT}, upload_heap{D3D12_HEAP_TYPE_UPLOAD},
        readback_heap{D3D12_HEAP_TYPE_READBACK};
    const auto create = [&](Ref<ID3D12Resource>& r, const D3D12_HEAP_PROPERTIES& heap, UINT64 bytes,
                            D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
        const auto desc = buffer_desc(bytes, flags);
        return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                         IID_PPV_ARGS(&r.p)));
    };
    constexpr auto uav_flag = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    Ref<ID3D12Resource> data, structured, counter_buffer, upload, readback;
    if (!create(data, default_heap, api_data_bytes, uav_flag, D3D12_RESOURCE_STATE_UNORDERED_ACCESS) ||
        !create(structured, default_heap, api_records_bytes, uav_flag, D3D12_RESOURCE_STATE_COPY_DEST) ||
        !create(counter_buffer, default_heap, counter_buffer_bytes, uav_flag, D3D12_RESOURCE_STATE_COPY_DEST) ||
        !create(upload, upload_heap, api_records_bytes + counter_buffer_bytes, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ) ||
        !create(readback, readback_heap, api_readback, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST))
        return std::fprintf(stderr, "resource creation failed\n"), false;
    {
        std::uint8_t* mapped = nullptr;
        if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return std::fprintf(stderr, "run_api: line %d\n", __LINE__), false;
        std::memset(mapped, 0x11, api_records_bytes + counter_buffer_bytes);
        const std::uint32_t seven = 7;
        std::memcpy(mapped + api_records_bytes + counter_offset, &seven, 4);   // the counter's word
        upload->Unmap(0, nullptr);
    }
    constexpr UINT slots = 24;
    Ref<ID3D12DescriptorHeap> cpu_heap, gpu_heap;
    D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, slots, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&cpu_heap.p)))) return std::fprintf(stderr, "run_api: line %d\n", __LINE__), false;
    if (host) {
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&gpu_heap.p)))) return std::fprintf(stderr, "run_api: line %d\n", __LINE__), false;
    }
    const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const auto cpu = [&](UINT i) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = cpu_heap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += SIZE_T(i) * step;
        return h;
    };
    const auto gpu = [&](UINT i) {
        if (!host) return D3D12_GPU_DESCRIPTOR_HANDLE{0};
        D3D12_GPU_DESCRIPTOR_HANDLE h = gpu_heap->GetGPUDescriptorHandleForHeapStart();
        h.ptr += UINT64(i) * step;
        return h;
    };
    UINT next_slot = 0;
    const auto make = [&](ID3D12Resource* resource, ID3D12Resource* counter, const D3D12_UNORDERED_ACCESS_VIEW_DESC& view) {
        const UINT i = next_slot++;
        device->CreateUnorderedAccessView(resource, counter, &view, cpu(i));
        if (host) {
            D3D12_CPU_DESCRIPTOR_HANDLE visible = gpu_heap->GetCPUDescriptorHandleForHeapStart();
            visible.ptr += SIZE_T(i) * step;
            device->CopyDescriptorsSimple(1, visible, cpu(i), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
        return i;
    };

    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr, IID_PPV_ARGS(&list.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return std::fprintf(stderr, "run_api: line %d\n", __LINE__), false;
    ID3D12DescriptorHeap* heaps[] = {gpu_heap.p};
    if (host) list->SetDescriptorHeaps(1, heaps);

    const auto clear = [&](UINT slot_index, ID3D12Resource* resource, bool is_float, const std::array<std::uint32_t, 4>& v) {
        if (is_float) {
            const float f[4] = {std::bit_cast<float>(v[0]), std::bit_cast<float>(v[1]), std::bit_cast<float>(v[2]),
                                std::bit_cast<float>(v[3])};
            list->ClearUnorderedAccessViewFloat(gpu(slot_index), cpu(slot_index), resource, f, 0, nullptr);
        } else {
            list->ClearUnorderedAccessViewUint(gpu(slot_index), cpu(slot_index), resource, v.data(), 0, nullptr);
        }
        const D3D12_RESOURCE_BARRIER barrier = uav_barrier(resource);
        list->ResourceBarrier(1, &barrier);
    };
    // The whole buffer 0xcdcdcdcd through a raw view, then each case's elements through a typed view.
    clear(make(data.p, nullptr, buffer_view(DXGI_FORMAT_R32_TYPELESS, 0, api_data_bytes / 4, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW)),
          data.p, false, {0xcdcdcdcd, 0, 0, 0});
    supported_by_host.assign(std::size(clear_cases), true);
    for (std::size_t c = 0; c < std::size(clear_cases); ++c) {
        const ClearCase& test = clear_cases[c];
        if (host) {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{test.format};
            supported_by_host[c] = SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) &&
                                   (support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
        }
        if (!supported_by_host[c]) continue;
        const UINT size = uc::layout(test.vulkan)->bytes;
        clear(make(data.p, nullptr, buffer_view(test.format, test.byte_offset / size, test.elements, 0)), data.p,
              test.is_float, test.values);
    }
    // A structured view with a counter. The clear writes the records and leaves the counter (preset to 7) alone ...
    list->CopyBufferRegion(structured.p, 0, upload.p, 0, api_records_bytes);
    list->CopyBufferRegion(counter_buffer.p, 0, upload.p, api_records_bytes, counter_buffer_bytes);
    D3D12_RESOURCE_BARRIER to_uav[2] = {transition(structured.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                                        transition(counter_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
    list->ResourceBarrier(2, to_uav);
    clear(make(structured.p, counter_buffer.p, buffer_view(DXGI_FORMAT_UNKNOWN, 0, 256, 16, counter_offset)), structured.p,
          false, {0xab, 0xcd, 0xef, 0x01});
    // ... and a raw view of the counter's own word (element 1024) clears it, as a game resets its counter.
    clear(make(counter_buffer.p, nullptr, buffer_view(DXGI_FORMAT_R32_TYPELESS, counter_offset / 4, 1, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW)),
          counter_buffer.p, false, {3, 0, 0, 0});
    D3D12_RESOURCE_BARRIER to_copy[3] = {
        transition(data.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        transition(structured.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        transition(counter_buffer.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)};
    list->ResourceBarrier(3, to_copy);
    list->CopyBufferRegion(readback.p, 0, data.p, 0, api_data_bytes);
    list->CopyBufferRegion(readback.p, api_data_bytes, structured.p, 0, api_records_bytes);
    list->CopyBufferRegion(readback.p, api_data_bytes + api_records_bytes, counter_buffer.p, 0, counter_buffer_bytes);
    if (const HRESULT hr = list->Close(); FAILED(hr)) return std::fprintf(stderr, "Close: %08lx\n", static_cast<unsigned long>(hr)), false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (!host && rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success)
        return std::fprintf(stderr, "replay result %d\n", int(rt::d3d12::last_execute_result(queue.p))), false;
    if (FAILED(queue->Signal(fence.p, 1))) return std::fprintf(stderr, "run_api: line %d\n", __LINE__), false;
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event || FAILED(fence->SetEventOnCompletion(1, event)) || WaitForSingleObject(event, 20000) != WAIT_OBJECT_0)
        return std::fprintf(stderr, "run_api: line %d\n", __LINE__), false;
    CloseHandle(event);
    void* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, &mapped))) return std::fprintf(stderr, "run_api: line %d\n", __LINE__), false;
    out.assign(static_cast<const std::uint8_t*>(mapped), static_cast<const std::uint8_t*>(mapped) + api_readback);
    readback->Unmap(0, nullptr);
    return true;
}

// What D3D12 says the same sequence leaves, from uav_clear.h: the model for a run with no host runtime to compare with.
std::vector<std::uint8_t> model(const std::vector<bool>& supported) {
    std::vector<std::uint8_t> bytes(api_readback, 0x11);
    std::fill(bytes.begin(), bytes.begin() + api_data_bytes, std::uint8_t(0xcd));
    for (std::size_t c = 0; c < std::size(clear_cases); ++c) {
        const ClearCase& test = clear_cases[c];
        uc::Element e{};
        if (!supported[c] || !uc::encode(test.vulkan, test.is_float, test.values, e)) continue;
        for (UINT64 i = 0; i < UINT64(test.elements) * e.size; ++i) bytes[test.byte_offset + i] = e.bytes[i % e.size];
    }
    // A structured view clears every 32-bit word to the first value (D3D12 ignores the other three).
    for (UINT64 i = 0; i < api_records_bytes; ++i) bytes[api_data_bytes + i] = i % 4 == 0 ? 0xab : 0;
    const std::uint32_t three = 3;
    std::memcpy(bytes.data() + api_data_bytes + api_records_bytes + counter_offset, &three, 4);
    return bytes;
}

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
    const std::filesystem::path root_dir = LEONIDA_TEST_BUFFER_UAV_DIR;

    unsigned words = 0;
    if (const int status = run_shaders(d, root_dir, words)) return status;

    // Part 2: the layer's device and the host's on the same adapter.
    std::vector<std::uint8_t> ours, host_bytes;
    std::vector<bool> supported;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;  // the V# holds the buffer's address
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        EXPECT(run_api(device.p, false, ours, supported));
    }
    std::string host_name = "none";
    bool compared = false;
    std::uint32_t vendor = 0;
    {
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
        d.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(d.physical, &properties);
        vendor = properties.properties.vendorID;
        copycmp::Ptr<ID3D12Device> host;
        LUID luid{};
        std::memcpy(&luid, id.deviceLUID, sizeof(luid));
        if (id.deviceLUIDValid && copycmp::try_create_host(host, host_name, &luid)) {
            std::vector<bool> host_supported;
            EXPECT(run_api(host.p, true, host_bytes, host_supported));
            supported = host_supported;
            compared = true;
        }
    }
    if (!compared) {
        std::printf("no host D3D12 device on the layer's adapter: the layer's bytes are compared with uav_clear.h\n");
        host_bytes = model(supported);
        host_name = "the uav_clear.h model";
    } else {
        // The layer ran every case; the host only those its adapter can store typed. Mask the others in both.
        for (std::size_t c = 0; c < std::size(clear_cases); ++c) {
            if (supported[c]) continue;
            const ClearCase& test = clear_cases[c];
            const UINT size = uc::layout(test.vulkan)->bytes;
            std::fill_n(ours.begin() + test.byte_offset, UINT64(test.elements) * size, std::uint8_t(0));
            std::fill_n(host_bytes.begin() + test.byte_offset, UINT64(test.elements) * size, std::uint8_t(0));
            std::printf("host adapter has no typed UAV store for DXGI format %d: that case is not compared\n",
                        int(test.format));
        }
        // One measured host quirk, not a layer bug: Intel's D3D12 (Xe-LP, 32.0.101.7092) writes the R11G11B10_FLOAT
        // clear (1.0, 2.0, 0.5) as 0x701e03c0, G = 1.0, as if the values were clamped to [0, 1]. Report it, don't
        // fail, only when the host's bytes are exactly that clamped encoding. The layer and model() share
        // uav_clear::encode, so the host stays the independent reference for every other format, adapter and value.
        for (std::size_t c = 0; c < std::size(clear_cases); ++c) {
            const ClearCase& test = clear_cases[c];
            if (vendor != 0x8086 || test.format != DXGI_FORMAT_R11G11B10_FLOAT || !supported[c] ||
                ours.size() != api_readback || host_bytes.size() != api_readback) continue;
            std::array<std::uint32_t, 4> clamped = test.values;
            for (auto& v : clamped) v = fb(std::clamp(std::bit_cast<float>(v), 0.0f, 1.0f));
            uc::Element e{};
            if (!uc::encode(test.vulkan, true, clamped, e)) continue;
            const UINT64 begin = test.byte_offset, end = begin + UINT64(test.elements) * e.size;
            bool host_clamped = true;
            for (UINT64 i = begin; i < end; ++i) host_clamped = host_clamped && host_bytes[i] == e.bytes[(i - begin) % e.size];
            const std::vector<std::uint8_t> reference = model(supported);
            if (!host_clamped || !std::equal(ours.begin() + begin, ours.begin() + end, reference.begin() + begin) ||
                std::equal(ours.begin() + begin, ours.begin() + end, host_bytes.begin() + begin)) continue;
            UINT64 first = begin;
            while (ours[first] == host_bytes[first]) ++first;
            std::uint32_t host_word = 0;
            std::memcpy(&host_word, host_bytes.data() + begin, 4);
            std::printf("DXGI format %d: host quirk: %s writes 0x%08x, the clear clamped to [0, 1] (first differing "
                        "byte %llu: layer %02x, host %02x); not compared\n", int(test.format), host_name.c_str(),
                        host_word, static_cast<unsigned long long>(first), ours[first], host_bytes[first]);
            std::copy(ours.begin() + begin, ours.begin() + end, host_bytes.begin() + begin);
        }
    }
    EXPECT(ours.size() == api_readback && host_bytes.size() == api_readback);
    unsigned differing = 0;
    for (std::size_t i = 0; i < ours.size(); ++i) {
        ++words;
        if (ours[i] == host_bytes[i]) continue;
        if (differing++ < 12) {
            const char* where = i < api_data_bytes ? "typed clears" : i < api_data_bytes + api_records_bytes ? "structured clear with a counter" : "counter buffer";
            std::fprintf(stderr, "%s byte %zu: layer %02x, %s %02x\n", where, i, ours[i], host_name.c_str(), host_bytes[i]);
        }
    }
    if (differing) {
        // Say which clear case the first differing byte belongs to.
        for (std::size_t c = 0; c < std::size(clear_cases); ++c) {
            const ClearCase& test = clear_cases[c];
            const UINT size = uc::layout(test.vulkan)->bytes;
            for (UINT64 i = 0; i < UINT64(test.elements) * size; ++i)
                if (ours[test.byte_offset + i] != host_bytes[test.byte_offset + i]) {
                    std::fprintf(stderr, "case %zu (DXGI format %d) differs\n", c, int(test.format));
                    break;
                }
        }
        std::fprintf(stderr, "%u bytes differ\n", differing);
        return 1;
    }
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("Buffer UAVs: typed views (R32_UINT with an atomic, R16G16B16A16_FLOAT, R8G8B8A8_UNORM from FirstElement 16, "
                "R32G32B32A32_UINT, R32_FLOAT loads, R32_SINT) written by a translated compute shader with NUM_RECORDS "
                "as the bound, and a structured UAV with a counter at offset 4096 appended to by 256 lanes and consumed "
                "by 128 (counter 256 then 128 through CopyBufferRegion) matched the CPU reference; typed-buffer and "
                "counter clears matched %s: %u values compared, error 0\n",
                host_name.c_str(), words);
    return 0;
}
