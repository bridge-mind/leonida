// Graphics and compute pipeline state descriptor validation parity against live Windows D3D12 (WARP and HW).
// Covers null InputLayout.pInputElementDescs crash prevention, InputLayout bounds/semantics/step-rate/slot-class,
// BlendState LogicOp/IndependentBlend rules, trailing RTVFormats, DSVFormat validation, NodeMask, and Flags.
// No game data. Set LEONIDA_VK_VALIDATION=1 to check Vulkan usage under VK_LAYER_KHRONOS_validation.
#include "d3d12_test_shaders.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV) || !defined(LEONIDA_TEST_PS_UINT_SPV) || \
    !defined(LEONIDA_TEST_VS_FULLSCREEN_SPV) || !defined(LEONIDA_TEST_CS_SPV)
#error "Test shader SPIR-V paths must be defined"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

struct HostContext {
    std::string name;
    Ref<ID3D12Device> device;
    Ref<ID3D12RootSignature> root;
    D3D12_SHADER_BYTECODE vs_fullscreen{};
    D3D12_SHADER_BYTECODE vs_pos{};
    D3D12_SHADER_BYTECODE ps_unorm{};
    D3D12_SHADER_BYTECODE ps_uint{};
    D3D12_SHADER_BYTECODE cs{};
};

bool compile_hlsl(pD3DCompile compile, const char* src, const char* profile, Ref<ID3DBlob>& out) {
    Ref<ID3DBlob> err;
    const HRESULT hr = compile(src, std::strlen(src), nullptr, nullptr, nullptr, "main", profile, 0, 0, &out.p, &err.p);
    if (SUCCEEDED(hr) && out.p) return true;
    std::fprintf(stderr, "D3DCompile (%s) failed: 0x%08lx\n", profile, static_cast<unsigned long>(hr));
    return false;
}

HRESULT create_graphics(ID3D12Device* dev, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc) {
    Ref<ID3D12PipelineState> pso;
    return dev->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p));
}

HRESULT create_compute(ID3D12Device* dev, const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc) {
    Ref<ID3D12PipelineState> pso;
    return dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso.p));
}

}  // namespace

int main() {
    vktest::Device vk_dev;
    const auto status = vk_dev.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);

    // Prepare translated shader library for our Vulkan-backed D3D12 device.
    const auto vs_pos_prog = fssm_program({0x56531001, 0xbf810000});
    const auto vs_fs_prog = fssm_program({0x56531002, 0xbf810000});
    const auto ps_unorm_prog = fssm_program({0x50531001, 0xbf810000});
    const auto ps_uint_prog = fssm_program({0x50531002, 0xbf810000});
    const auto cs_prog = fssm_program({0x43531001, 0xbf810000});

    const auto vs_pos_container = shader_container(vs_pos_prog);
    const auto vs_fs_container = shader_container(vs_fs_prog);
    const auto ps_unorm_container = shader_container(ps_unorm_prog);
    const auto ps_uint_container = shader_container(ps_uint_prog);
    const auto cs_container = shader_container(cs_prog);

    const testfiles::ScratchDir scratch("leonida_pso_validation_test");
    const auto& lib_dir = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_SPV), lib_dir / "vs_draw.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_FULLSCREEN_SPV),
                                lib_dir / "vs_fullscreen.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), lib_dir / "ps_draw.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_UINT_SPV), lib_dir / "ps_draw_uint.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), lib_dir / "cs_user_data.spv"));
    {
        std::ofstream index(lib_dir / "index.json", std::ios::binary);
        index << "{\"vs_draw\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_pos_prog)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}, "
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 1, \"mask\": 3, \"rw_mask\": 3, "
                 "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
                 " \"vs_fullscreen\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_fs_prog)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_unorm_prog)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]},\n"
                 " \"ps_draw_uint\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_uint_prog)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]},\n"
                 " \"cs_user_data\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(cs_prog)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144}}";
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk_dev.instance_proc;
    info.instance = vk_dev.instance;
    info.physical = vk_dev.physical;
    info.device = vk_dev.device;
    info.queue = vk_dev.queue;
    info.queue_family = vk_dev.family;
    info.buffer_device_address = true;
    info.subgroup = vk_dev.subgroup;
    info.shader_dir = lib_dir.string();
    info.features.logicOp = vk_dev.enabled.logicOp;
    info.vertex_attribute_divisor = vk_dev.vertex_attribute_divisor;

    Ref<ID3D12Device> layer_dev;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&layer_dev.p))));

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    const auto root_blob = serialize_root_signature(root_desc);
    EXPECT(!root_blob.empty());
    Ref<ID3D12RootSignature> layer_root;
    EXPECT(SUCCEEDED(layer_dev->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&layer_root.p))));

    // Optional live host D3D12 devices (WARP and HW) on native Windows for exact HRESULT parity.
    Ref<ID3DBlob> hlsl_vs_fs, hlsl_vs_pos, hlsl_ps_unorm, hlsl_ps_uint, hlsl_cs;
    std::vector<std::unique_ptr<HostContext>> hosts;
    HMODULE compiler_dll = nullptr;
    if (!hostwine::host_is_wine()) {
        compiler_dll = LoadLibraryA("d3dcompiler_47.dll");
        auto compile = compiler_dll ? reinterpret_cast<pD3DCompile>(GetProcAddress(compiler_dll, "D3DCompile")) : nullptr;
        if (compile) {
            const char* kVsFs = "float4 main(uint id : SV_VertexID) : SV_Position { return float4(0,0,0,1); }";
            const char* kVsPos =
                "float4 main(float2 p : POSITION, float4 c : COLOR) : SV_Position { return float4(p, c.x, 1); }";
            const char* kPsUnorm = "float4 main() : SV_Target { return float4(1,0,0,1); }";
            const char* kPsUint = "uint4 main() : SV_Target { return uint4(1,0,0,1); }";
            const char* kCs = "[numthreads(1,1,1)] void main() {}";
            if (compile_hlsl(compile, kVsFs, "vs_5_0", hlsl_vs_fs) &&
                compile_hlsl(compile, kVsPos, "vs_5_0", hlsl_vs_pos) &&
                compile_hlsl(compile, kPsUnorm, "ps_5_0", hlsl_ps_unorm) &&
                compile_hlsl(compile, kPsUint, "ps_5_0", hlsl_ps_uint) &&
                compile_hlsl(compile, kCs, "cs_5_0", hlsl_cs)) {
                Ref<IDXGIFactory4> factory;
                if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
                    for (const bool warp : {false, true}) {
                        Ref<IDXGIAdapter> adapter;
                        std::string name;
                        if (!hostadapter::reference(factory.p, vk_dev, warp, &adapter.p, name)) continue;
                        auto hc = std::make_unique<HostContext>();
                        hc->name = name;
                        if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&hc->device.p))))
                            continue;
                        if (FAILED(hc->device->CreateRootSignature(0, root_blob.data(), root_blob.size(),
                                                                   IID_PPV_ARGS(&hc->root.p))))
                            continue;
                        hc->vs_fullscreen = {hlsl_vs_fs->GetBufferPointer(), hlsl_vs_fs->GetBufferSize()};
                        hc->vs_pos = {hlsl_vs_pos->GetBufferPointer(), hlsl_vs_pos->GetBufferSize()};
                        hc->ps_unorm = {hlsl_ps_unorm->GetBufferPointer(), hlsl_ps_unorm->GetBufferSize()};
                        hc->ps_uint = {hlsl_ps_uint->GetBufferPointer(), hlsl_ps_uint->GetBufferSize()};
                        hc->cs = {hlsl_cs->GetBufferPointer(), hlsl_cs->GetBufferSize()};
                        hosts.push_back(std::move(hc));
                    }
                }
            }
        }
    }

    auto make_gfx = [&](ID3D12RootSignature* rs, D3D12_SHADER_BYTECODE vs, D3D12_SHADER_BYTECODE ps) {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC d{};
        d.pRootSignature = rs;
        d.VS = vs;
        d.PS = ps;
        d.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        d.SampleMask = 0xffffffffu;
        d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        d.RasterizerState.DepthClipEnable = TRUE;
        d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        d.NumRenderTargets = 1;
        d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.SampleDesc.Count = 1;
        return d;
    };

    const D3D12_SHADER_BYTECODE layer_vs_fs{vs_fs_container.data(), vs_fs_container.size()};
    const D3D12_SHADER_BYTECODE layer_vs_pos{vs_pos_container.data(), vs_pos_container.size()};
    const D3D12_SHADER_BYTECODE layer_ps_unorm{ps_unorm_container.data(), ps_unorm_container.size()};
    const D3D12_SHADER_BYTECODE layer_ps_uint{ps_uint_container.data(), ps_uint_container.size()};
    const D3D12_SHADER_BYTECODE layer_cs{cs_container.data(), cs_container.size()};

    unsigned checked = 0, host_compared = 0;

    auto check_gfx_parity = [&](const char* label, HRESULT expected, bool use_pos_vs, bool use_uint_ps,
                                auto mutator) -> bool {
        auto desc = make_gfx(layer_root.p, use_pos_vs ? layer_vs_pos : layer_vs_fs,
                             use_uint_ps ? layer_ps_uint : layer_ps_unorm);
        if (use_uint_ps) desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UINT;
        mutator(desc);
        const HRESULT actual = create_graphics(layer_dev.p, desc);
        if (actual != expected) {
            std::fprintf(stderr, "FAIL [%s]: layer returned 0x%08lx, expected 0x%08lx\n", label,
                         static_cast<unsigned long>(actual), static_cast<unsigned long>(expected));
            return false;
        }
        ++checked;
        for (const auto& hcp : hosts) {
            const HostContext& hc = *hcp;
            auto hdesc = make_gfx(hc.root.p, use_pos_vs ? hc.vs_pos : hc.vs_fullscreen,
                                  use_uint_ps ? hc.ps_uint : hc.ps_unorm);
            if (use_uint_ps) hdesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UINT;
            mutator(hdesc);
            const HRESULT hhr = create_graphics(hc.device.p, hdesc);
            if (hhr != actual) {
                std::fprintf(stderr, "FAIL [%s]: layer 0x%08lx != host %s 0x%08lx\n", label,
                             static_cast<unsigned long>(actual), hc.name.c_str(), static_cast<unsigned long>(hhr));
                return false;
            }
            ++host_compared;
        }
        return true;
    };

    auto check_gfx_layer_only = [&](const char* label, HRESULT expected, auto mutator) -> bool {
        auto desc = make_gfx(layer_root.p, layer_vs_fs, layer_ps_unorm);
        mutator(desc);
        const HRESULT actual = create_graphics(layer_dev.p, desc);
        if (actual != expected) {
            std::fprintf(stderr, "FAIL [%s]: layer returned 0x%08lx, expected 0x%08lx\n", label,
                         static_cast<unsigned long>(actual), static_cast<unsigned long>(expected));
            return false;
        }
        ++checked;
        return true;
    };

    auto check_cs_parity = [&](const char* label, HRESULT expected, auto mutator) -> bool {
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = layer_root.p;
        desc.CS = layer_cs;
        mutator(desc);
        const HRESULT actual = create_compute(layer_dev.p, desc);
        if (actual != expected) {
            std::fprintf(stderr, "FAIL [%s]: layer returned 0x%08lx, expected 0x%08lx\n", label,
                         static_cast<unsigned long>(actual), static_cast<unsigned long>(expected));
            return false;
        }
        ++checked;
        for (const auto& hcp : hosts) {
            const HostContext& hc = *hcp;
            D3D12_COMPUTE_PIPELINE_STATE_DESC hdesc{};
            hdesc.pRootSignature = hc.root.p;
            hdesc.CS = hc.cs;
            mutator(hdesc);
            const HRESULT hhr = create_compute(hc.device.p, hdesc);
            if (hhr != actual) {
                std::fprintf(stderr, "FAIL [%s]: layer 0x%08lx != host %s 0x%08lx\n", label,
                             static_cast<unsigned long>(actual), hc.name.c_str(), static_cast<unsigned long>(hhr));
                return false;
            }
            ++host_compared;
        }
        return true;
    };

    // 1. Baseline & null pInputElementDescs crash guards.
    EXPECT(check_gfx_parity("gfx: baseline", S_OK, false, false, [](auto&) {}));
    EXPECT(check_gfx_parity("gfx: InputLayout null ptr, N=1, VS requires inputs", E_INVALIDARG, true, false,
                            [](auto& d) { d.InputLayout = {nullptr, 1}; }));
    EXPECT(check_gfx_layer_only("gfx: InputLayout null ptr, N=1, VS has no inputs", E_INVALIDARG,
                                [](auto& d) { d.InputLayout = {nullptr, 1}; }));
    EXPECT(check_gfx_layer_only("gfx: InputLayout null ptr, N=33", E_INVALIDARG,
                                [](auto& d) { d.InputLayout = {nullptr, 33}; }));

    // 2. InputLayout validation (bounds, SemanticName, InputSlot, InputSlotClass, InstanceDataStepRate).
    std::vector<D3D12_INPUT_ELEMENT_DESC> elems32(32), elems33(33);
    for (UINT i = 0; i < 32; ++i)
        elems32[i] = {"TEXCOORD", i, DXGI_FORMAT_R32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
                      D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    for (UINT i = 0; i < 33; ++i)
        elems33[i] = {"TEXCOORD", i, DXGI_FORMAT_R32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
                      D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};

    EXPECT(check_gfx_parity("gfx: InputLayout NumElements=32", S_OK, false, false,
                            [&](auto& d) { d.InputLayout = {elems32.data(), 32}; }));
    EXPECT(check_gfx_parity("gfx: InputLayout NumElements=33", E_INVALIDARG, false, false,
                            [&](auto& d) { d.InputLayout = {elems33.data(), 33}; }));

    const D3D12_INPUT_ELEMENT_DESC e_null_sem{nullptr, 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
                                              D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    const D3D12_INPUT_ELEMENT_DESC e_empty_sem{"", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
                                               D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    const D3D12_INPUT_ELEMENT_DESC e_slot31{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 31, 0,
                                            D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    const D3D12_INPUT_ELEMENT_DESC e_slot32{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 32, 0,
                                            D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    const D3D12_INPUT_ELEMENT_DESC e_pv_step1{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
                                              D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 1};
    const D3D12_INPUT_ELEMENT_DESC e_pi_step0{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
                                              D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 0};
    const D3D12_INPUT_ELEMENT_DESC e_pi_step1{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
                                              D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1};
    const D3D12_INPUT_ELEMENT_DESC e_bad_cls{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
                                             static_cast<D3D12_INPUT_CLASSIFICATION>(2), 0};

    EXPECT(check_gfx_parity("gfx: InputLayout SemanticName=nullptr", E_INVALIDARG, false, false,
                            [&](auto& d) { d.InputLayout = {&e_null_sem, 1}; }));
    EXPECT(check_gfx_parity("gfx: InputLayout SemanticName empty string", S_OK, false, false,
                            [&](auto& d) { d.InputLayout = {&e_empty_sem, 1}; }));
    EXPECT(check_gfx_parity("gfx: InputLayout InputSlot=31", S_OK, false, false,
                            [&](auto& d) { d.InputLayout = {&e_slot31, 1}; }));
    EXPECT(check_gfx_parity("gfx: InputLayout InputSlot=32", E_INVALIDARG, false, false,
                            [&](auto& d) { d.InputLayout = {&e_slot32, 1}; }));
    EXPECT(check_gfx_parity("gfx: InputLayout PER_VERTEX + stepRate=1", E_INVALIDARG, false, false,
                            [&](auto& d) { d.InputLayout = {&e_pv_step1, 1}; }));
    EXPECT(check_gfx_parity("gfx: InputLayout PER_INSTANCE + stepRate=0", S_OK, false, false,
                            [&](auto& d) { d.InputLayout = {&e_pi_step0, 1}; }));
    EXPECT(check_gfx_parity("gfx: InputLayout PER_INSTANCE + stepRate=1", S_OK, false, false,
                            [&](auto& d) { d.InputLayout = {&e_pi_step1, 1}; }));
    EXPECT(check_gfx_parity("gfx: InputLayout invalid InputSlotClass=2", E_INVALIDARG, false, false,
                            [&](auto& d) { d.InputLayout = {&e_bad_cls, 1}; }));

    // 3. BlendState validation (BlendEnable && LogicOpEnable, IndependentBlendEnable && LogicOpEnable, LogicOp range).
    EXPECT(check_gfx_parity("gfx: BlendEnable && LogicOpEnable (N=1)", E_INVALIDARG, false, true, [](auto& d) {
        d.BlendState.RenderTarget[0].BlendEnable = TRUE;
        d.BlendState.RenderTarget[0].LogicOpEnable = TRUE;
    }));
    EXPECT(check_gfx_parity("gfx: BlendEnable && LogicOpEnable (N=0)", E_INVALIDARG, false, false, [](auto& d) {
        d.NumRenderTargets = 0;
        d.RTVFormats[0] = DXGI_FORMAT_UNKNOWN;
        d.BlendState.RenderTarget[0].BlendEnable = TRUE;
        d.BlendState.RenderTarget[0].LogicOpEnable = TRUE;
    }));
    EXPECT(check_gfx_parity("gfx: IndependentBlend && RT0.LogicOp (N=1)", E_INVALIDARG, false, true, [](auto& d) {
        d.BlendState.IndependentBlendEnable = TRUE;
        d.BlendState.RenderTarget[0].LogicOpEnable = TRUE;
    }));
    EXPECT(check_gfx_parity("gfx: IndependentBlend && RT0.LogicOp (N=0)", E_INVALIDARG, false, false, [](auto& d) {
        d.NumRenderTargets = 0;
        d.RTVFormats[0] = DXGI_FORMAT_UNKNOWN;
        d.BlendState.IndependentBlendEnable = TRUE;
        d.BlendState.RenderTarget[0].LogicOpEnable = TRUE;
    }));
    EXPECT(check_gfx_parity("gfx: IndependentBlend && RT1.LogicOp (N=2)", E_INVALIDARG, false, true, [](auto& d) {
        d.NumRenderTargets = 2;
        d.RTVFormats[1] = DXGI_FORMAT_R8G8B8A8_UINT;
        d.BlendState.IndependentBlendEnable = TRUE;
        d.BlendState.RenderTarget[1].LogicOpEnable = TRUE;
    }));
    EXPECT(check_gfx_parity("gfx: IndependentBlend=0 ignores RT1 Blend+LogicOp", S_OK, false, false, [](auto& d) {
        d.BlendState.IndependentBlendEnable = FALSE;
        d.BlendState.RenderTarget[1].BlendEnable = TRUE;
        d.BlendState.RenderTarget[1].LogicOpEnable = TRUE;
    }));
    EXPECT(check_gfx_parity("gfx: IndependentBlend=1 checks RT1 even when N=1", E_INVALIDARG, false, false,
                            [](auto& d) {
                                d.BlendState.IndependentBlendEnable = TRUE;
                                d.BlendState.RenderTarget[1].BlendEnable = TRUE;
                                d.BlendState.RenderTarget[1].LogicOpEnable = TRUE;
                            }));
    EXPECT(check_gfx_parity("gfx: LogicOp out of range (N=1)", E_INVALIDARG, false, true, [](auto& d) {
        d.BlendState.RenderTarget[0].LogicOpEnable = TRUE;
        d.BlendState.RenderTarget[0].LogicOp = static_cast<D3D12_LOGIC_OP>(16);
    }));
    EXPECT(check_gfx_parity("gfx: LogicOp out of range (N=0)", E_INVALIDARG, false, false, [](auto& d) {
        d.NumRenderTargets = 0;
        d.RTVFormats[0] = DXGI_FORMAT_UNKNOWN;
        d.BlendState.RenderTarget[0].LogicOpEnable = TRUE;
        d.BlendState.RenderTarget[0].LogicOp = static_cast<D3D12_LOGIC_OP>(16);
    }));
    if (vk_dev.enabled.logicOp == VK_TRUE) {
        EXPECT(check_gfx_parity("gfx: valid LogicOp XOR on R8G8B8A8_UINT", S_OK, false, true, [](auto& d) {
            d.BlendState.RenderTarget[0].LogicOpEnable = TRUE;
            d.BlendState.RenderTarget[0].LogicOp = D3D12_LOGIC_OP_XOR;
        }));
    }

    // 4. RTVFormats validation.
    EXPECT(check_gfx_parity("gfx: RTVFormats[1]!=UNKNOWN with N=1", E_INVALIDARG, false, false,
                            [](auto& d) { d.RTVFormats[1] = DXGI_FORMAT_R8G8B8A8_UNORM; }));
    EXPECT(check_gfx_parity("gfx: RTVFormats[0]!=UNKNOWN with N=0", E_INVALIDARG, false, false, [](auto& d) {
        d.NumRenderTargets = 0;
        d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    }));
    EXPECT(check_gfx_parity("gfx: RTVFormats[0]==UNKNOWN with N=1", S_OK, false, false,
                            [](auto& d) { d.RTVFormats[0] = DXGI_FORMAT_UNKNOWN; }));
    EXPECT(check_gfx_parity("gfx: NumRenderTargets=9", E_INVALIDARG, false, false,
                            [](auto& d) { d.NumRenderTargets = 9; }));

    // 5. DSVFormat validation.
    EXPECT(check_gfx_parity("gfx: DSVFormat=D32_FLOAT", S_OK, false, false,
                            [](auto& d) { d.DSVFormat = DXGI_FORMAT_D32_FLOAT; }));
    EXPECT(check_gfx_parity("gfx: DSVFormat=D16_UNORM", S_OK, false, false,
                            [](auto& d) { d.DSVFormat = DXGI_FORMAT_D16_UNORM; }));
    EXPECT(check_gfx_parity("gfx: DSVFormat=D24_UNORM_S8_UINT", S_OK, false, false,
                            [](auto& d) { d.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT; }));
    EXPECT(check_gfx_parity("gfx: DSVFormat=D32_FLOAT_S8X24_UINT", S_OK, false, false,
                            [](auto& d) { d.DSVFormat = DXGI_FORMAT_D32_FLOAT_S8X24_UINT; }));
    EXPECT(check_gfx_parity("gfx: DSVFormat=9999 (unknown)", E_INVALIDARG, false, false,
                            [](auto& d) { d.DSVFormat = static_cast<DXGI_FORMAT>(9999); }));
    // Layer-strict DSVFormat checks: reject typeless and non-depth formats to avoid invalid Vulkan depthAttachmentFormat.
    EXPECT(check_gfx_layer_only("gfx: DSVFormat=R32_TYPELESS", E_INVALIDARG,
                                [](auto& d) { d.DSVFormat = DXGI_FORMAT_R32_TYPELESS; }));
    EXPECT(check_gfx_layer_only("gfx: DSVFormat=R16_TYPELESS", E_INVALIDARG,
                                [](auto& d) { d.DSVFormat = DXGI_FORMAT_R16_TYPELESS; }));
    EXPECT(check_gfx_layer_only("gfx: DSVFormat=R24G8_TYPELESS", E_INVALIDARG,
                                [](auto& d) { d.DSVFormat = DXGI_FORMAT_R24G8_TYPELESS; }));
    EXPECT(check_gfx_layer_only("gfx: DSVFormat=R32G8X24_TYPELESS", E_INVALIDARG,
                                [](auto& d) { d.DSVFormat = DXGI_FORMAT_R32G8X24_TYPELESS; }));
    EXPECT(check_gfx_layer_only("gfx: DSVFormat=R8G8B8A8_UNORM (color)", E_INVALIDARG,
                                [](auto& d) { d.DSVFormat = DXGI_FORMAT_R8G8B8A8_UNORM; }));
    EXPECT(check_gfx_layer_only("gfx: DSVFormat=R32_FLOAT (color)", E_INVALIDARG,
                                [](auto& d) { d.DSVFormat = DXGI_FORMAT_R32_FLOAT; }));

    // 6. NodeMask and Flags validation (Graphics & Compute).
    EXPECT(check_gfx_parity("gfx: NodeMask=0", S_OK, false, false, [](auto& d) { d.NodeMask = 0; }));
    EXPECT(check_gfx_parity("gfx: NodeMask=1", S_OK, false, false, [](auto& d) { d.NodeMask = 1; }));
    EXPECT(check_gfx_parity("gfx: NodeMask=2", E_INVALIDARG, false, false, [](auto& d) { d.NodeMask = 2; }));
    EXPECT(check_gfx_parity("gfx: NodeMask=3", E_INVALIDARG, false, false, [](auto& d) { d.NodeMask = 3; }));
    EXPECT(check_gfx_parity("gfx: NodeMask=0x80000000", E_INVALIDARG, false, false,
                            [](auto& d) { d.NodeMask = 0x80000000u; }));
    EXPECT(check_gfx_parity("gfx: Flags=DYNAMIC_DEPTH_BIAS", S_OK, false, false,
                            [](auto& d) { d.Flags = D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS; }));
    EXPECT(check_gfx_parity("gfx: Flags=DYNAMIC_INDEX_BUFFER_STRIP_CUT", S_OK, false, false,
                            [](auto& d) { d.Flags = D3D12_PIPELINE_STATE_FLAG_DYNAMIC_INDEX_BUFFER_STRIP_CUT; }));
    EXPECT(check_gfx_parity("gfx: Flags=0x10 (unknown bit)", E_INVALIDARG, false, false,
                            [](auto& d) { d.Flags = static_cast<D3D12_PIPELINE_STATE_FLAGS>(0x10); }));
    EXPECT(check_gfx_parity("gfx: Flags=0x80000000 (unknown bit)", E_INVALIDARG, false, false,
                            [](auto& d) { d.Flags = static_cast<D3D12_PIPELINE_STATE_FLAGS>(0x80000000u); }));
    EXPECT(check_gfx_layer_only("gfx: Flags=TOOL_DEBUG (cached-PSO parity)", S_OK,
                                [](auto& d) { d.Flags = D3D12_PIPELINE_STATE_FLAG_TOOL_DEBUG; }));

    EXPECT(check_cs_parity("cs: baseline", S_OK, [](auto&) {}));
    EXPECT(check_cs_parity("cs: NodeMask=0", S_OK, [](auto& d) { d.NodeMask = 0; }));
    EXPECT(check_cs_parity("cs: NodeMask=1", S_OK, [](auto& d) { d.NodeMask = 1; }));
    EXPECT(check_cs_parity("cs: NodeMask=2", E_INVALIDARG, [](auto& d) { d.NodeMask = 2; }));
    EXPECT(check_cs_parity("cs: NodeMask=3", E_INVALIDARG, [](auto& d) { d.NodeMask = 3; }));
    EXPECT(check_cs_parity("cs: NodeMask=0x80000000", E_INVALIDARG, [](auto& d) { d.NodeMask = 0x80000000u; }));
    EXPECT(check_cs_parity("cs: Flags=DYNAMIC_DEPTH_BIAS", S_OK,
                           [](auto& d) { d.Flags = D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS; }));
    EXPECT(check_cs_parity("cs: Flags=DYNAMIC_INDEX_BUFFER_STRIP_CUT", S_OK,
                           [](auto& d) { d.Flags = D3D12_PIPELINE_STATE_FLAG_DYNAMIC_INDEX_BUFFER_STRIP_CUT; }));
    EXPECT(check_cs_parity("cs: Flags=0x10 (unknown bit)", E_INVALIDARG,
                           [](auto& d) { d.Flags = static_cast<D3D12_PIPELINE_STATE_FLAGS>(0x10); }));
    EXPECT(check_cs_parity("cs: Flags=0x80000000 (unknown bit)", E_INVALIDARG,
                           [](auto& d) { d.Flags = static_cast<D3D12_PIPELINE_STATE_FLAGS>(0x80000000u); }));
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC cs_debug{};
        cs_debug.pRootSignature = layer_root.p;
        cs_debug.CS = layer_cs;
        cs_debug.Flags = D3D12_PIPELINE_STATE_FLAG_TOOL_DEBUG;
        EXPECT(create_compute(layer_dev.p, cs_debug) == S_OK);
        ++checked;
    }

    const std::size_t host_count = hosts.size();
    hosts.clear();
    hlsl_vs_fs.reset();
    hlsl_vs_pos.reset();
    hlsl_ps_unorm.reset();
    hlsl_ps_uint.reset();
    hlsl_cs.reset();
    if (compiler_dll) FreeLibrary(compiler_dll);
    std::printf("d3d12_pso_validation_test: PASS (%u layer checks, %u host comparisons across %zu host device(s))\n",
                checked, host_compared, host_count);
    return 0;
}
