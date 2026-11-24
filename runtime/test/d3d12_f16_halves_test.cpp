// End-to-end Vulkan GPU test for per-pipeline `--f16-interp-fallback` linking (SHADER-378).
//
// Exercises two graphics PSOs through ID3D12Device::CreateGraphicsPipelineState on a real Vulkan 1.3 GPU, without
// needing glslangValidator, llvm-mc or game files (both SPIR-V 1.5 modules are built directly in memory):
//   1. A 16-param VS (param0..param15, static f16_halves only for param0..param6) writes packed f16 pairs into
//      param7: .x = packHalf2x16(0.25, 0.75) = 0x3A003400, .y = packHalf2x16(0.5, 1.0) = 0x3C003800.
//      PS #1 reads raw attr0..attr14 plus attr7's decoded f16 halves at Locations 20 & 21 (which collide with VS's
//      static param4 halves at 20 & 21). `link_f16_halves` demotes VS's unneeded static halves to Private,
//      synthesizes UnpackHalf2x16 for param7 at Locations 20 & 21, and the GPU renders RGBA (64, 191, 128, 255).
//   2. PS #2 reads raw attr0..attr13 plus attr7's decoded f16 halves at Locations 14 & 15 (colliding with VS's raw
//      param14 & param15). `link_f16_halves` relocates BOTH PS and VS halves to Locations 28 & 29 and renders the
//      exact same RGBA (64, 191, 128, 255).
// Skips (77) when no usable Vulkan 1.3 device is present. No game data.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"
#include "../src/spirv_f16_halves.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

#define EXPECT(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

using namespace d3d12test;
using namespace leonida::spirv_detail;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

void put_inst(std::vector<std::uint32_t>& w, std::uint16_t op, std::initializer_list<std::uint32_t> args) {
    w.push_back((std::uint32_t(1 + args.size()) << 16) | op);
    for (std::uint32_t a : args) w.push_back(a);
}

void put_inst_vec(std::vector<std::uint32_t>& w, std::uint16_t op, const std::vector<std::uint32_t>& args) {
    w.push_back((std::uint32_t(1 + args.size()) << 16) | op);
    w.insert(w.end(), args.begin(), args.end());
}

void write_spv(const std::filesystem::path& path, const std::vector<std::uint32_t>& words) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
}

// Fullscreen triangle VS (from VertexIndex 0,1,2) exporting 16 raw params (0..15) + static f16_halves for 0..6.
// param7 carries packed f16 bits: .x = 0x3A003400 (lo=0.25, hi=0.75), .y = 0x3C003800 (lo=0.5, hi=1.0).
std::vector<std::uint32_t> make_16param_vs() {
    // Fixed IDs:
    // 1=%void, 2=%fn, 3=%float, 4=%v4float, 5=%int, 6=%ptr_in_int, 7=%ptr_out_v4
    // 8=%vid_var, 9=%pos_var, 10=%null_v4
    // 11=%f_m1(-1.0), 12=%f_3(3.0), 13=%f_0(0.0), 14=%f_1(1.0)
    // 15=%c_p0(-1,-1,0,1), 16=%c_p1(3,-1,0,1), 17=%c_p2(-1,3,0,1)
    // 18=%i_0, 19=%i_1, 20=%bool
    // 21=%f_pack_x (bitcast 0x3A003400), 22=%f_pack_y (bitcast 0x3C003800), 23=%v_param7
    // 24=%main, 25=%entry
    // 26=%vid, 27=%is0, 28=%is1, 29=%p12, 30=%pos
    // 31..46 = raw param0..param15 Output vars
    // 47..60 = static f16_halves for param0..param6 (14 Output vars: 28,29, 26,27, ..., 16,17)
    constexpr std::uint32_t kBound = 61;
    std::vector<std::uint32_t> w = {kSpirvMagic, 0x00010500u, 0, kBound, 0};
    put_inst(w, kOpCapability, {1});        // Shader
    put_inst(w, kOpMemoryModel, {0, 1});    // Logical GLSL450
    std::vector<std::uint32_t> ep = {kExecModelVertex, 24, 0x6e69616du, 0x00000000u, 8, 9};
    for (std::uint32_t id = 31; id <= 60; ++id) ep.push_back(id);
    put_inst_vec(w, kOpEntryPoint, ep);

    put_inst(w, kOpDecorate, {8, 11, 42});  // BuiltIn VertexIndex
    put_inst(w, kOpDecorate, {9, 11, 0});   // BuiltIn Position
    for (std::uint32_t p = 0; p < 16; ++p) put_inst(w, kOpDecorate, {31 + p, kDecorationLocation, p});
    for (std::uint32_t n = 0; n <= 6; ++n) {
        put_inst(w, kOpDecorate, {47 + 2 * n, kDecorationLocation, 28 - 2 * n});
        put_inst(w, kOpDecorate, {48 + 2 * n, kDecorationLocation, 29 - 2 * n});
    }

    put_inst(w, 19, {1});                   // OpTypeVoid
    put_inst(w, 33, {2, 1});                // OpTypeFunction %void
    put_inst(w, kOpTypeFloat, {3, 32});
    put_inst(w, kOpTypeVector, {4, 3, 4});
    put_inst(w, kOpTypeInt, {5, 32, 1});    // %int
    put_inst(w, kOpTypePointer, {6, kStorageInput, 5});
    put_inst(w, kOpTypePointer, {7, kStorageOutput, 4});
    put_inst(w, 20, {20});                  // OpTypeBool

    put_inst(w, kOpConstantNull, {4, 10});
    put_inst(w, 43, {3, 11, 0xbf800000u});  // -1.0f
    put_inst(w, 43, {3, 12, 0x40400000u});  //  3.0f
    put_inst(w, 43, {3, 13, 0x00000000u});  //  0.0f
    put_inst(w, 43, {3, 14, 0x3f800000u});  //  1.0f
    put_inst(w, 44, {4, 15, 11, 11, 13, 14});  // (-1, -1, 0, 1)
    put_inst(w, 44, {4, 16, 12, 11, 13, 14});  // ( 3, -1, 0, 1)
    put_inst(w, 44, {4, 17, 11, 12, 13, 14});  // (-1,  3, 0, 1)
    put_inst(w, 43, {5, 18, 0u});
    put_inst(w, 43, {5, 19, 1u});
    put_inst(w, 43, {3, 21, 0x3a003400u});  // packHalf2x16(0.25, 0.75) as raw f32 bits
    put_inst(w, 43, {3, 22, 0x3c003800u});  // packHalf2x16(0.50, 1.00) as raw f32 bits
    put_inst(w, 44, {4, 23, 21, 22, 13, 14});

    put_inst(w, kOpVariable, {6, 8, kStorageInput});
    put_inst(w, kOpVariable, {7, 9, kStorageOutput});
    for (std::uint32_t id = 31; id <= 60; ++id) put_inst(w, kOpVariable, {7, id, kStorageOutput});

    put_inst(w, kOpFunction, {1, 24, 0, 2});
    put_inst(w, 248, {25});                 // OpLabel
    put_inst(w, kOpLoad, {5, 26, 8});       // %vid
    put_inst(w, 170, {20, 27, 26, 18});     // OpIEqual %vid 0
    put_inst(w, 170, {20, 28, 26, 19});     // OpIEqual %vid 1
    put_inst(w, 169, {4, 29, 28, 16, 17});  // OpSelect %is1 %p1 %p2
    put_inst(w, 169, {4, 30, 27, 15, 29});  // OpSelect %is0 %p0 %p12
    put_inst(w, kOpStore, {9, 30});
    for (std::uint32_t p = 0; p < 16; ++p) put_inst(w, kOpStore, {31 + p, p == 7 ? 23u : 10u});
    for (std::uint32_t id = 47; id <= 60; ++id) put_inst(w, kOpStore, {id, 10});
    put_inst(w, kOpReturn, {});
    put_inst(w, kOpFunctionEnd, {});
    return w;
}

// Fragment shader reading `num_raw` raw inputs (Location 0..num_raw-1) plus decoded halves of param7 at
// `lo_loc` and `hi_loc`, writing vec4(lo.x, hi.x, lo.y, hi.y) to MRT 0.
std::vector<std::uint32_t> make_f16_ps(std::uint32_t num_raw, std::uint32_t lo_loc, std::uint32_t hi_loc) {
    // 1=%void, 2=%fn, 3=%float, 4=%v4float, 5=%ptr_in_v4, 6=%ptr_out_v4, 7=%out0, 8=%in_lo, 9=%in_hi,
    // 10=%main, 11=%entry, 12=%vlo, 13=%vhi, 14=%lox, 15=%hix, 16=%loy, 17=%hiy, 18=%rgba,
    // 19.. = raw inputs 0..num_raw-1
    const std::uint32_t bound = 19 + num_raw;
    std::vector<std::uint32_t> w = {kSpirvMagic, 0x00010500u, 0, bound, 0};
    put_inst(w, kOpCapability, {1});
    put_inst(w, kOpMemoryModel, {0, 1});
    std::vector<std::uint32_t> ep = {kExecModelFragment, 10, 0x6e69616du, 0x00000000u, 7, 8, 9};
    for (std::uint32_t i = 0; i < num_raw; ++i) ep.push_back(19 + i);
    put_inst_vec(w, kOpEntryPoint, ep);
    put_inst(w, kOpExecutionMode, {10, 7});  // OriginUpperLeft

    put_inst(w, kOpDecorate, {7, kDecorationLocation, 0});
    put_inst(w, kOpDecorate, {8, kDecorationLocation, lo_loc});
    put_inst(w, kOpDecorate, {9, kDecorationLocation, hi_loc});
    for (std::uint32_t i = 0; i < num_raw; ++i) put_inst(w, kOpDecorate, {19 + i, kDecorationLocation, i});

    put_inst(w, 19, {1});
    put_inst(w, 33, {2, 1});
    put_inst(w, kOpTypeFloat, {3, 32});
    put_inst(w, kOpTypeVector, {4, 3, 4});
    put_inst(w, kOpTypePointer, {5, kStorageInput, 4});
    put_inst(w, kOpTypePointer, {6, kStorageOutput, 4});

    put_inst(w, kOpVariable, {6, 7, kStorageOutput});
    put_inst(w, kOpVariable, {5, 8, kStorageInput});
    put_inst(w, kOpVariable, {5, 9, kStorageInput});
    for (std::uint32_t i = 0; i < num_raw; ++i) put_inst(w, kOpVariable, {5, 19 + i, kStorageInput});

    put_inst(w, kOpFunction, {1, 10, 0, 2});
    put_inst(w, 248, {11});
    put_inst(w, kOpLoad, {4, 12, 8});
    put_inst(w, kOpLoad, {4, 13, 9});
    put_inst(w, kOpCompositeExtract, {3, 14, 12, 0});  // lo.x = 0.25
    put_inst(w, kOpCompositeExtract, {3, 15, 13, 0});  // hi.x = 0.75
    put_inst(w, kOpCompositeExtract, {3, 16, 12, 1});  // lo.y = 0.50
    put_inst(w, kOpCompositeExtract, {3, 17, 13, 1});  // hi.y = 1.00
    put_inst(w, kOpCompositeConstruct, {4, 18, 14, 15, 16, 17});
    put_inst(w, kOpStore, {7, 18});
    put_inst(w, kOpReturn, {});
    put_inst(w, kOpFunctionEnd, {});
    return w;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);

    const auto vs_prog = fssm_program({0x56530378, 0xbf810000});
    const auto ps1_prog = fssm_program({0x50530378, 0x00000001});
    const auto ps2_prog = fssm_program({0x50530378, 0x00000002});
    const auto vs_c = shader_container(vs_prog);
    const auto ps1_c = shader_container(ps1_prog);
    const auto ps2_c = shader_container(ps2_prog);

    const auto lib_dir = std::filesystem::temp_directory_path() / "leonida_f16_halves_gpu_test";
    std::filesystem::create_directories(lib_dir);
    write_spv(lib_dir / "vs_16p.spv", make_16param_vs());
    write_spv(lib_dir / "ps_packed.spv", make_f16_ps(15, 20, 21));
    write_spv(lib_dir / "ps_reloc.spv", make_f16_ps(14, 14, 15));
    {
        std::ofstream idx(lib_dir / "index.json", std::ios::binary | std::ios::trunc);
        idx << "{\"vs_16p\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": [], "
               "\"f16_halves\": {\"0\": [28, 29], \"1\": [26, 27], \"2\": [24, 25], \"3\": [22, 23], "
               "\"4\": [20, 21], \"5\": [18, 19], \"6\": [16, 17]}},\n"
               " \"ps_packed\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps1_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0], "
               "\"f16_halves\": {\"7\": [20, 21]}},\n"
               " \"ps_reloc\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps2_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0], "
               "\"f16_halves\": {\"7\": [14, 15]}}}";
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = true;
    info.subgroup = d.subgroup;
    info.shader_dir = lib_dir.string();

    ID3D12Device* inner = nullptr;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                       reinterpret_cast<void**>(&device.p))));
    rt::d3d12::set_device_outer(inner, device.p);

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    const auto root_blob = serialize_root_signature(root_desc);
    EXPECT(!root_blob.empty());
    Ref<ID3D12RootSignature> root;
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
    Ref<ID3D12CommandAllocator> alloc;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(list->Close()));

    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    constexpr UINT kSize = 16;
    D3D12_HEAP_PROPERTIES default_hp{};
    default_hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rt_desc{};
    rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rt_desc.Width = kSize;
    rt_desc.Height = kSize;
    rt_desc.DepthOrArraySize = 1;
    rt_desc.MipLevels = 1;
    rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rt_desc.SampleDesc.Count = 1;
    rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    Ref<ID3D12Resource> rt_res;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_hp, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                     D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                     IID_PPV_ARGS(&rt_res.p))));

    D3D12_DESCRIPTOR_HEAP_DESC rtv_hd{};
    rtv_hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv_hd.NumDescriptors = 1;
    Ref<ID3D12DescriptorHeap> rtv_heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_hd, IID_PPV_ARGS(&rtv_heap.p))));
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv_h{
        SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
    device->CreateRenderTargetView(rt_res.p, nullptr, rtv_h);

    D3D12_HEAP_PROPERTIES rb_hp{};
    rb_hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rb_desc{};
    rb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rb_desc.Width = kSize * 256;
    rb_desc.Height = 1;
    rb_desc.DepthOrArraySize = 1;
    rb_desc.MipLevels = 1;
    rb_desc.SampleDesc.Count = 1;
    rb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> rb_res;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&rb_hp, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                     IID_PPV_ARGS(&rb_res.p))));

    UINT64 fence_val = 0;
    const auto run_case = [&](const char* label, const std::vector<std::uint8_t>& ps_container) -> bool {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = root.p;
        pd.VS = {vs_c.data(), vs_c.size()};
        pd.PS = {ps_container.data(), ps_container.size()};
        pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pd.SampleMask = 0xffffffffu;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RasterizerState.DepthClipEnable = TRUE;
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.NumRenderTargets = 1;
        pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pd.SampleDesc.Count = 1;

        Ref<ID3D12PipelineState> pso;
        if (FAILED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p))) || !pso.p) {
            std::fprintf(stderr, "%s: CreateGraphicsPipelineState failed\n", label);
            return false;
        }
        if (FAILED(alloc->Reset()) || FAILED(list->Reset(alloc.p, pso.p))) return false;
        const float clear_col[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        list->ClearRenderTargetView(rtv_h, clear_col, 0, nullptr);
        list->OMSetRenderTargets(1, &rtv_h, FALSE, nullptr);
        const D3D12_VIEWPORT vp = {0.0f, 0.0f, float(kSize), float(kSize), 0.0f, 1.0f};
        const D3D12_RECT sc = {0, 0, LONG(kSize), LONG(kSize)};
        list->RSSetViewports(1, &vp);
        list->RSSetScissorRects(1, &sc);
        list->SetGraphicsRootSignature(root.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->DrawInstanced(3, 1, 0, 0);

        D3D12_RESOURCE_BARRIER to_copy{};
        to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        to_copy.Transition = {rt_res.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                              D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE};
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = rb_res.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, kSize, kSize, 1, 256};
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = rt_res.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        D3D12_RESOURCE_BARRIER to_rt = to_copy;
        to_rt.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        to_rt.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        list->ResourceBarrier(1, &to_rt);
        if (FAILED(list->Close())) return false;

        ID3D12CommandList* cl = list.p;
        queue->ExecuteCommandLists(1, &cl);
        ++fence_val;
        if (FAILED(queue->Signal(fence.p, fence_val))) return false;
        if (FAILED(fence->SetEventOnCompletion(fence_val, nullptr))) return false;

        void* mapped = nullptr;
        const D3D12_RANGE read_range = {0, kSize * 256};
        if (FAILED(rb_res->Map(0, &read_range, &mapped)) || !mapped) return false;
        const auto* px = static_cast<const std::uint8_t*>(mapped) + (kSize / 2) * 256 + (kSize / 2) * 4;
        const int r = px[0], g = px[1], b = px[2], a = px[3];
        rb_res->Unmap(0, nullptr);
        // Expected: (0.25, 0.75, 0.50, 1.00) -> (64, 191, 128, 255) within +-1 LSB of UNORM rounding.
        const bool ok = std::abs(r - 64) <= 1 && std::abs(g - 191) <= 1 && std::abs(b - 128) <= 1 && a == 255;
        if (!ok) {
            std::fprintf(stderr, "%s: pixel=(%d,%d,%d,%d), expected~(64,191,128,255)\n", label, r, g, b, a);
        } else {
            std::printf("%s: rendered pixel=(%d,%d,%d,%d) OK\n", label, r, g, b, a);
        }
        return ok;
    };

    EXPECT(run_case("ps_packed (synth param7 + demote static halves)", ps1_c));
    EXPECT(run_case("ps_reloc (relocate colliding PS + VS halves to 28,29)", ps2_c));
    return 0;
}
