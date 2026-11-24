// Milestone M1.5 (D3D-53, docs/d3d12_vulkan.md section 6): a draw whose VS and PS are real translator output, not GLSL
// written to the contract. test/shaders/vs_m15.asm and ps_m15.asm are hand-written RDNA2 (gfx1013); the build runs
// them through llvm-mc and tools/rdna2spirv.py (test/shaders/build.py), which writes spirv/index.json with each
// program's fssm_sha256. The proxy's CreateGraphicsPipelineState then finds both stages by the SHA-256 of the FSSM
// program inside the DXBC container, as it would for the game; nothing here bypasses that lookup. What it covers:
//   - the user-data contract: a descriptor-table root argument (s0:s1, the table's console VA) and four root constants
//     (s2..s5) reach the VS and the PS through the 144-byte push-constant block;
//   - vertex fetch: the VS has no input layout; it s_loads a V# the application wrote into a shader-visible descriptor
//     slot and fetches position and colour with buffer_load_dwordx4 ... idxen. The V# points at a DEFAULT-heap vertex
//     buffer (a range of the GPU address arena, copied from an upload buffer). A V# cannot point at the upload
//     buffer itself: its base is 48 bits, and the upload buffer's console VA (device address minus heap base) is
//     negative, so heap base + the truncated base is a wild address (measured on both test GPUs: the draw faults the
//     queue). Root views and table pointers are 64 bits and do not have the problem; see docs/d3d12_vulkan.md 3.3;
//   - the code base: the PS starts with s_getpc_b64 and s_loads four floats that sit in its own FSSM program at offset
//     0x100, so heap base + code base + 0x100 must be where the layer put that program;
//   - interpolation of param0 and the mrt0 export into a R32G32B32A32_FLOAT render target, read back through
//     CopyTextureRegion and compared bit for bit.
// All draws use StartVertexLocation 0 and StartInstanceLocation 0, so the result is the same with and without the
// translator's zero-based VS ids (PR 436: v0 = VertexIndex - BaseVertex), which is not in this tree: with 436's
// translator change applied the result was identical (checked on both GPUs); the test never relies on either.
// Expected pixels (64x64, clip y up, so rows run top to bottom; the quadrants are 32x32):
//   clear               (0.5, 0.125, 0.25, 0.5)
//   draw 1, top left    quad, colour (0.25, 0.5, 0.75, 1), offset (0, 0),   bias 0     -> (0.125, 1.0,   0.75,  1)
//   draw 2, bottom right the same quad moved by (1, -1),   bias 0.125                  -> (0.25,  1.125, 0.875, 1)
//   draw 3, bottom left  the triangle below the anti-diagonal, colour (1, 0, 0.5, 1), bias 0.25, from the second V#
//                                                                                      -> (0.75,  0.25,  0.75,  1)
//   top right           not drawn: the clear colour
// The triangle's long edge passes exactly through the centres of the pixels with row - col == 32 (32 of them); the
// top-left rule leaves them uncovered (a right edge), so they must be the clear colour; they are counted apart. Every
// other pixel (4064 of 4096) is compared exactly. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
// Built a second time with LEONIDA_M15_GUEST, this file is tiny_guest_draw.exe, which runs the same draws and the same
// pixel checks as a `gta6rt --run` guest (test/run_guest_draw.cmake). Its device comes from an imported
// d3d12_xs.dll!D3D12XboxCreateDevice, which gta6rt's shim registry answers (LEONIDA_D3D12=vulkan, LEONIDA_SHADERS=
// <folder>/spirv); the folder build.py wrote is argv[1]. It links nothing from src/ or shims/, so the checks that need
// the runtime's internals (the shader library, the replay result, the proxy's ref count, validation) stay in the host
// build. On success it prints one marker line, after every check.
#ifndef LEONIDA_M15_GUEST
#include "vk_test_device.h"
#endif
#include "d3d12_test_shaders.h"
#ifndef LEONIDA_M15_GUEST
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#endif
#include "../src/gfx10_descriptors.h"
#ifndef LEONIDA_M15_GUEST
#include "../src/shader_library.h"
#endif

#ifdef LEONIDA_GUEST_SCENE
#include "guest_scene_state.h"
#include <cstdlib>
#endif

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>

#ifdef LEONIDA_M15_GUEST
// Resolved at link time against test/fixtures/d3d12_xs_stub.cpp (which fails), at run time by gta6rt's shim registry.
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown* adapter, const void* params,
                                                                      const IID* riid, void** out);
#elif !defined(LEONIDA_TEST_M15_DIR)
#error "LEONIDA_TEST_M15_DIR must name the folder build.py wrote (spirv/index.json, fssm/*.fssm)"
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

// One 32-byte vertex record, as vs_m15.asm reads it: position xyzw, then colour rgba.
struct Vertex {
    float position[4];
    float color[4];
};

using Pixel = std::array<float, 4>;
constexpr UINT target_extent = 64;
constexpr UINT pitch = target_extent * sizeof(Pixel);  // 1024, a multiple of 256

// A V# for the vertex records: stride 32, `records` records, format 32_32_32_32_FLOAT, identity swizzle. The
// out-of-range mode is structured (an index checked against num_records), which is what the hardware does with an
// idxen fetch; raw mode would check the byte offset against num_records and return 0 past the first record.
std::array<std::uint32_t, 4> make_vsharp(std::uint64_t base, std::uint32_t records) {
    leonida::gfx10::Buffer buffer;
    buffer.address = base;
    buffer.stride = 32;
    buffer.num_records = records;
    buffer.format = 77;  // GFX10_FORMAT_32_32_32_32_FLOAT
    buffer.oob_select = leonida::gfx10::OobSelect::structured;
    return leonida::gfx10::encode(buffer);
}

}  // namespace

#ifdef LEONIDA_M15_GUEST
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // the marker reaches the launcher even if the process dies after it
    if (argc < 2) {
        std::fprintf(stderr, "usage: tiny_guest_draw <folder build.py wrote (spirv/index.json, fssm/*.fssm)>\n");
        return 2;
    }
    // The FSSM programs the index entries are keyed by; the shim's device finds the index through LEONIDA_SHADERS.
    const std::filesystem::path root_dir = argv[1];
#else
int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    EXPECT(device_status == vktest::DeviceStatus::ready);

    // The translator-style library the build wrote, and the FSSM programs its index entries are keyed by.
    const std::filesystem::path root_dir = LEONIDA_TEST_M15_DIR;
#endif
    const auto vs_program = read_file(root_dir / "fssm" / "vs_m15.fssm");
    const auto ps_program = read_file(root_dir / "fssm" / "ps_m15.fssm");
    EXPECT(vs_program.size() >= 4 && ps_program.size() == 0x110);
    const float scale[4] = {0.5f, 2.0f, 1.0f, 1.0f};  // ps_m15.asm's table at offset 0x100
    EXPECT(std::memcmp(ps_program.data() + 0x100, scale, sizeof(scale)) == 0);
#ifndef LEONIDA_M15_GUEST
    leonida::ShaderLibrary library;
    EXPECT(library.load((root_dir / "spirv").string()));
    EXPECT(library.find("vs", leonida::sha256_hex(vs_program)) && library.find("ps", leonida::sha256_hex(ps_program)));
    EXPECT(library.find("ps", leonida::sha256_hex(ps_program))->uses_code_base);
#endif
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    // A program the index does not know (the VS's bytes with one word changed) must be refused by the PSO.
    auto unknown_program = vs_program;
    unknown_program.back() ^= 0x5a;
    const auto unknown_container = shader_container(unknown_program);

    // Geometry: nine vertices. 0..5: a quad over the top-left quadrant (two triangles), colour (0.25, 0.5, 0.75, 1).
    // 6..8: a triangle under the anti-diagonal of the bottom-left quadrant, colour (1, 0, 0.5, 1).
    const float c1[4] = {0.25f, 0.5f, 0.75f, 1.0f}, c2[4] = {1.0f, 0.0f, 0.5f, 1.0f};
    const auto vertex = [](float x, float y, const float* c) {
        return Vertex{{x, y, 0.0f, 1.0f}, {c[0], c[1], c[2], c[3]}};
    };
    const Vertex vertices[9] = {vertex(-1, 1, c1), vertex(0, 1, c1), vertex(-1, 0, c1),   //
                                vertex(-1, 0, c1), vertex(0, 1, c1), vertex(0, 0, c1),    //
                                vertex(-1, 0, c2), vertex(-1, -1, c2), vertex(0, -1, c2)};
    static_assert(sizeof(Vertex) == 32 && sizeof(vertices) == 288);

    std::vector<Pixel> pixels(target_extent * target_extent);
    {
#ifdef LEONIDA_M15_GUEST
        // The import the game uses is the guest's only way to a device. With LEONIDA_D3D12=vulkan the shim returns the
        // Vulkan device behind the Xbox-ABI proxy; the host's D3D12 device cannot build the PSO from FSSM containers,
        // and the link-time stub returns no device, so either of those fails below.
        const std::uint32_t device_params[24] = {};  // the 24-word block the shim logs; it does not create from it
        const IID device_iid = __uuidof(ID3D12Device);
        Ref<ID3D12Device> device;
        const HRESULT created = D3D12XboxCreateDevice(nullptr, device_params, &device_iid, reinterpret_cast<void**>(&device.p));
        if (FAILED(created) || !device.p) {
            std::fprintf(stderr, "D3D12XboxCreateDevice failed (hr=0x%08lx)\n", static_cast<unsigned long>(created));
            return 1;
        }
#else
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.shader_dir = (root_dir / "spirv").string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
#endif

        // Root signature: [0] a descriptor table of one SRV (s0:s1, the table's console VA), [1] four 32-bit
        // constants (s2, s3: the draw's offset; s4: the PS's bias; s5 unused). No input layout: the VS fetches.
        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable = {1, &range};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[1].Constants = {0, 0, 4};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        // The PSO, from containers whose FSSM hashes are in the index.
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        pso_desc.SampleDesc.Count = 1;
        Ref<ID3D12PipelineState> pso, unknown;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));
        pso_desc.VS = {unknown_container.data(), unknown_container.size()};
        EXPECT(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&unknown.p)) == E_INVALIDARG && !unknown.p);

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> vertex_buffer, upload, readback, target;
        const auto vb_desc = buffer_desc(sizeof(vertices));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &vb_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&vertex_buffer.p))));
        const auto upload_desc = buffer_desc(1024);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto readback_desc = buffer_desc(pitch * target_extent);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = rt_desc.Height = target_extent;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p))));

        // The upload buffer holds the vertices at 0; they go into the DEFAULT buffer by a copy.
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memcpy(up, vertices, sizeof(vertices));

        // A shader-visible heap in the arena; each slot gets the 16 bytes of a V#, written as raw bytes.
        const D3D12_DESCRIPTOR_HEAP_DESC srv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> srv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&srv_heap_desc, IID_PPV_ARGS(&srv_heap.p))));
        const std::uint64_t cpu = reinterpret_cast<XboxHeapHandleFn>(slot(srv_heap.p, 9))(srv_heap.p);
        const std::uint64_t gpu = reinterpret_cast<XboxHeapHandleFn>(slot(srv_heap.p, 10))(srv_heap.p);
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(cpu && gpu >= 0x10000 && gpu < (1ull << 32) && step >= 16);
        const std::uint64_t buffer_va = vertex_buffer->GetGPUVirtualAddress();
        EXPECT(buffer_va >= 0x10000);
        const std::array<std::uint32_t, 4> vsharps[2] = {make_vsharp(buffer_va, 9),                       // 9 vertices
                                                         make_vsharp(buffer_va + 6 * sizeof(Vertex), 3)};  // 6..8
        for (unsigned i = 0; i < 2; ++i)
            std::memcpy(reinterpret_cast<void*>(cpu + std::uint64_t(i) * step), vsharps[i].data(), 16);

        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{
            SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
        device->CreateRenderTargetView(target.p, nullptr, rtv);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        list->CopyBufferRegion(vertex_buffer.p, 0, upload.p, 0, sizeof(vertices));
        D3D12_RESOURCE_BARRIER barriers[2] = {
            transition(vertex_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)};
        list->ResourceBarrier(2, barriers);
        // The first command list already contains the vertex upload. Reuse it for subsequent frames
        // after the preceding submission's fence has completed and its readback is available.
        const auto render_frame = [&](UINT64 frame, float x_offset, float bias) -> int {
            if (frame > 1) {
                EXPECT(SUCCEEDED(allocator->Reset()));
                EXPECT(SUCCEEDED(list->Reset(allocator.p, pso.p)));
                auto to_target = transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                            D3D12_RESOURCE_STATE_RENDER_TARGET);
                list->ResourceBarrier(1, &to_target);
            }
            const FLOAT clear[4] = {0.5f, 0.125f, 0.25f, 0.5f};
            list->ClearRenderTargetView(rtv, clear, 0, nullptr);

            list->SetGraphicsRootSignature(root.p);
            ID3D12DescriptorHeap* heaps[] = {srv_heap.p};
            list->SetDescriptorHeaps(1, heaps);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            const D3D12_VIEWPORT viewport{0, 0, float(target_extent), float(target_extent), 0, 1};
            const D3D12_RECT scissor{0, 0, LONG(target_extent), LONG(target_extent)};
            list->RSSetViewports(1, &viewport);
            list->RSSetScissorRects(1, &scissor);
            list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

            struct Draw {
                unsigned slot;
                float constants[4];  // s2, s3: offset; s4: bias; s5: unused
                UINT vertices;
            };
            const Draw draws[3] = {{0, {x_offset, 0, bias, 0}, 6},     // top left
                                   {0, {1, -1, 0.125f + bias, 0}, 6},  // bottom right: same V#, other constants
                                   {1, {0, 0, 0.25f + bias, 0}, 3}};   // bottom left: the second V#
            for (const Draw& draw : draws) {
                list->SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{gpu + std::uint64_t(draw.slot) * step});
                list->SetGraphicsRoot32BitConstants(1, 4, draw.constants, 0);
                list->DrawInstanced(draw.vertices, 1, 0, 0);
            }

            D3D12_RESOURCE_BARRIER to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                        D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &to_copy);
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = readback.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = {0, {DXGI_FORMAT_R32G32B32A32_FLOAT, target_extent, target_extent, 1, pitch}};
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = target.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            EXPECT(list->Close() == S_OK);
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
#ifndef LEONIDA_M15_GUEST
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
#endif
            EXPECT(SUCCEEDED(queue->Signal(fence.p, frame)));
            EXPECT(SUCCEEDED(fence->SetEventOnCompletion(frame, nullptr)));
            void* mapped = nullptr;
            EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
            std::memcpy(pixels.data(), mapped, pixels.size() * sizeof(Pixel));
            readback->Unmap(0, nullptr);
            return 0;
        };

#ifdef LEONIDA_GUEST_SCENE
        if (std::getenv("LEONIDA_SCENE_MODE")) {
            guestscene::Session scene;
            EXPECT(scene.begin());
            UINT64 count = 0;
            while (!scene.done()) {
                EXPECT(scene.step());
                EXPECT(render_frame(++count, scene.x_offset(), scene.bias()) == 0);
                EXPECT(scene.finish_frame(pixels.front().data(), target_extent));
            }
            EXPECT(scene.finish());
            std::printf("GUEST_SCENE_PASS frames=%llu\n", static_cast<unsigned long long>(count));
            return 0;
        }
#endif
        EXPECT(render_frame(1, 0.0f, 0.0f) == 0);
        for (auto* r : {&vertex_buffer, &upload, &readback, &target}) r->reset();
        pso.reset();
        root.reset();
        srv_heap.reset();
        rtv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
#ifndef LEONIDA_M15_GUEST
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
#endif
    }

    const Pixel clear_color = {0.5f, 0.125f, 0.25f, 0.5f};
    const Pixel top_left = {0.125f, 1.0f, 0.75f, 1.0f}, bottom_right = {0.25f, 1.125f, 0.875f, 1.0f};
    const Pixel bottom_left = {0.75f, 0.25f, 0.75f, 1.0f};
    unsigned mismatches = 0, edge = 0, edge_triangle = 0, edge_clear = 0;
    for (UINT row = 0; row < target_extent; ++row) {
        for (UINT col = 0; col < target_extent; ++col) {
            const bool top = row < target_extent / 2, left = col < target_extent / 2;
            const Pixel& got = pixels[row * target_extent + col];
            const auto bits_equal = [&](const Pixel& want) { return std::memcmp(&got, &want, sizeof(Pixel)) == 0; };
            const Pixel* want;
            if (top) {
                want = left ? &top_left : &clear_color;
            } else if (!left) {
                want = &bottom_right;
            } else if (row - col == 32) {
                ++edge;
                if (bits_equal(bottom_left)) ++edge_triangle;
                else if (bits_equal(clear_color)) ++edge_clear;
                else if (mismatches++ < 8)
                    std::fprintf(stderr, "edge pixel (%u, %u): got %g,%g,%g,%g is neither colour\n", col, row, got[0],
                                 got[1], got[2], got[3]);
                continue;
            } else {
                want = row - col > 32 ? &bottom_left : &clear_color;
            }
            if (bits_equal(*want)) continue;
            if (mismatches++ < 8)
                std::fprintf(stderr, "pixel (%u, %u): got %g,%g,%g,%g want %g,%g,%g,%g\n", col, row, got[0], got[1],
                             got[2], got[3], (*want)[0], (*want)[1], (*want)[2], (*want)[3]);
        }
    }
    // The long edge is a right edge of the triangle, which the top-left rule leaves uncovered: all 32 are the clear colour.
    EXPECT(edge == 32 && edge_triangle + edge_clear == edge && edge_triangle == 0);
    EXPECT(!mismatches);
#ifdef LEONIDA_M15_GUEST
    // Printed only here, after every check: test/run_guest_draw.cmake wants exactly this line and exit code 0.
    std::printf("M1.5 guest: PASS, %u of %u pixels exact, %u edge pixels the clear colour\n",
                target_extent * target_extent - edge, target_extent * target_extent, edge_clear);
#else
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("M1.5: translated VS+PS found by FSSM hash drew three draws; %u of %u pixels exact, %u edge "
                "pixels on the triangle's diagonal (%u its colour, %u the clear colour). User data from a descriptor "
                "table and root constants, V# vertex fetch from the arena, PS constants "
                "through the code base\n",
                target_extent * target_extent - edge, target_extent * target_extent, edge, edge_triangle, edge_clear);
#endif
    return 0;
}
