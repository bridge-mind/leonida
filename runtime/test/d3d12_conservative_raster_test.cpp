// Synthetic D3D12/Vulkan conservative coverage. Required/forbidden pixels come from geometric clipping,
// not a driver snapshot: Tier 1 may over-cover by half a pixel, so uncertain pixels are not compared exactly.
#include "vk_test_device.h"
#include "d3d12_copy_compare.h"
#include "d3d12_test_shaders.h"
#include "host_wine.h"
#include "../src/conservative_raster.h"
#include "../src/shader_library.h"
#include <d3dcompiler.h>
#include <array>
#include <cmath>
#include <climits>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return false; } } while (0)
namespace {
constexpr UINT width = 32, height = 32, pitch = 256;
struct Point { double x, y; };
using Triangle = std::array<Point, 3>;
using Mask = std::array<bool, width * height>;
struct Fixture {
    copycmp::Run run;
    copycmp::Ptr<ID3D12RootSignature> root;
    D3D12_SHADER_BYTECODE vs{}, ps{};
};

bool setup(Fixture& f, ID3D12Device* device, bool ours, D3D12_SHADER_BYTECODE vs, D3D12_SHADER_BYTECODE ps) {
    f.vs = vs; f.ps = ps;
    CHECK(f.run.init(device, ours));
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    const auto bytes = d3d12test::serialize_root_signature(desc);
    CHECK(!bytes.empty());
    CHECK(SUCCEEDED(device->CreateRootSignature(0, bytes.data(), bytes.size(), IID_PPV_ARGS(&f.root.p))));
    return true;
}

D3D12_GRAPHICS_PIPELINE_STATE_DESC description(const Fixture& f, bool conservative, UINT samples, bool variant) {
    static const D3D12_INPUT_ELEMENT_DESC element{
        "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC d{};
    d.pRootSignature = f.root.p;
    d.VS = f.vs; d.PS = f.ps;
    d.InputLayout = {&element, 1};
    d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    d.RasterizerState.DepthClipEnable = TRUE;
    d.RasterizerState.MultisampleEnable = samples > 1;
    d.RasterizerState.ConservativeRaster = conservative ? D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON
                                                       : D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    d.NumRenderTargets = variant ? 2 : 1;
    d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    if (variant) d.RTVFormats[1] = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    d.SampleDesc.Count = samples;
    d.SampleMask = UINT_MAX;
    return d;
}

// Clip the triangle to a pixel rectangle and return its intersection area.
double area(const Triangle& triangle, double left, double top, double right, double bottom, bool* intersects = nullptr) {
    std::vector<Point> p(triangle.begin(), triangle.end());
    for (unsigned side = 0; side < 4 && !p.empty(); ++side) {
        const auto distance = [&](Point v) {
            return side == 0 ? v.x - left : side == 1 ? right - v.x : side == 2 ? v.y - top : bottom - v.y;
        };
        std::vector<Point> out;
        Point a = p.back(); double da = distance(a);
        for (const Point b : p) {
            const double db = distance(b);
            if ((da >= 0) != (db >= 0)) {
                const double t = da / (da - db);
                out.push_back({a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)});
            }
            if (db >= 0) out.push_back(b);
            a = b; da = db;
        }
        p = std::move(out);
    }
    double sum = 0;
    if (intersects) *intersects = !p.empty(); // a degenerate triangle may cover pixels despite having zero area
    for (std::size_t i = 0; i < p.size(); ++i) {
        const Point a = p[i], b = p[(i + 1) % p.size()];
        sum += a.x * b.y - a.y * b.x;
    }
    return std::abs(sum) * 0.5;
}

bool render(Fixture& f, const Triangle& triangle, bool conservative, UINT samples, bool variant,
            const D3D12_RECT& scissor, Mask& mask, D3D12_CULL_MODE cull = D3D12_CULL_MODE_NONE) {
    auto& r = f.run;
    auto desc = description(f, conservative, samples, variant);
    desc.RasterizerState.CullMode = cull;
    copycmp::Ptr<ID3D12PipelineState> pso;
    const auto created = r.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p));
    if (FAILED(created)) {
        std::fprintf(stderr, "%s CreateGraphicsPipelineState=%08x conservative=%d samples=%u variant=%d\n",
                     r.ours ? "Vulkan" : "host", unsigned(created), int(conservative), samples, int(variant));
        return false;
    }
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = width; tex.Height = height;
    tex.DepthOrArraySize = tex.MipLevels = 1;
    tex.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    tex.SampleDesc.Count = samples;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    copycmp::Ptr<ID3D12Resource> target, resolved, vertices, readback;
    CHECK(r.texture(tex, D3D12_RESOURCE_STATE_RENDER_TARGET, target));
    if (samples > 1) {
        tex.SampleDesc.Count = 1; tex.Flags = D3D12_RESOURCE_FLAG_NONE;
        CHECK(r.texture(tex, D3D12_RESOURCE_STATE_RESOLVE_DEST, resolved));
    }
    CHECK(r.buffer(D3D12_HEAP_TYPE_UPLOAD, 24, D3D12_RESOURCE_STATE_GENERIC_READ, vertices));
    CHECK(r.buffer(D3D12_HEAP_TYPE_READBACK, pitch * height, D3D12_RESOURCE_STATE_COPY_DEST, readback));
    void* data = nullptr;
    const D3D12_RANGE none{};
    CHECK(SUCCEEDED(vertices->Map(0, &none, &data)));
    auto* positions = static_cast<float*>(data);
    for (unsigned i = 0; i < 3; ++i) {
        positions[i * 2] = float(triangle[i].x * 2 / width - 1);
        positions[i * 2 + 1] = float(1 - triangle[i].y * 2 / height);
    }
    vertices->Unmap(0, nullptr);
    copycmp::Ptr<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    CHECK(SUCCEEDED(r.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    r.device->CreateRenderTargetView(target.p, nullptr, handle);
    r.begin();
    const float clear[4] = {};
    r.list->ClearRenderTargetView(handle, clear, 0, nullptr);
    r.list->OMSetRenderTargets(1, &handle, FALSE, nullptr);
    r.list->SetGraphicsRootSignature(f.root.p);
    r.list->SetPipelineState(pso.p);
    const D3D12_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
    r.list->RSSetViewports(1, &viewport);
    r.list->RSSetScissorRects(1, &scissor);
    const D3D12_VERTEX_BUFFER_VIEW vb{vertices->GetGPUVirtualAddress(), 24, 8};
    r.list->IASetVertexBuffers(0, 1, &vb);
    r.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    r.list->DrawInstanced(3, 1, 0, 0);
    ID3D12Resource* source = target.p;
    if (samples > 1) {
        r.barrier(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
        r.list->ResolveSubresource(resolved.p, 0, target.p, 0, tex.Format);
        r.barrier(resolved.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        source = resolved.p;
    } else r.barrier(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = source; from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    to.pResource = readback.p; to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint.Footprint = {tex.Format, width, height, 1, pitch};
    r.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    const auto error = r.submit();
    if (!error.empty()) { std::fprintf(stderr, "%s\n", error.c_str()); return false; }
    CHECK(SUCCEEDED(readback->Map(0, nullptr, &data)));
    const auto* pixels = static_cast<const unsigned char*>(data);
    for (UINT y = 0; y < height; ++y) for (UINT x = 0; x < width; ++x) {
        const auto* p = pixels + y * pitch + x * 4;
        CHECK(p[1] == 0 && p[2] == 0 && p[0] == p[3]);
        if (conservative) CHECK(p[0] == 0 || p[0] == 255); // every MSAA sample is covered
        mask[y * width + x] = p[0] != 0;
    }
    readback->Unmap(0, &none);
    return true;
}

bool coverage(const Triangle& triangle, const D3D12_RECT& scissor, const Mask& mask) {
    constexpr double epsilon = 1.0 / 256;
    for (UINT y = 0; y < height; ++y) for (UINT x = 0; x < width; ++x) {
        const bool in = LONG(x) >= scissor.left && LONG(x) < scissor.right && LONG(y) >= scissor.top && LONG(y) < scissor.bottom;
        const bool required = in && area(triangle, x + epsilon, y + epsilon, x + 1 - epsilon, y + 1 - epsilon) > 1e-9;
        bool intersects = false;
        area(triangle, x - 0.5 - epsilon, y - 0.5 - epsilon, x + 1.5 + epsilon, y + 1.5 + epsilon, &intersects);
        const bool allowed = in && intersects;
        if ((required && !mask[y * width + x]) || (!allowed && mask[y * width + x])) {
            std::fprintf(stderr, "coverage (%u,%u): required=%d allowed=%d actual=%d\n", x, y,
                         int(required), int(allowed), int(mask[y * width + x]));
            return false;
        }
    }
    return true;
}

bool rejected_states(Fixture& f, bool enabled) {
    auto desc = description(f, true, 1, false);
    copycmp::Ptr<ID3D12PipelineState> pso;
    if (!enabled) {
        CHECK(f.run.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p)) == E_INVALIDARG);
        desc.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
        CHECK(SUCCEEDED(f.run.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p))));
        return true;
    }
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;
    CHECK(f.run.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p)) == E_INVALIDARG);
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
    CHECK(f.run.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p)) == E_INVALIDARG);
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.RasterizerState.ForcedSampleCount = 4;
    CHECK(f.run.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p)) == E_NOTIMPL);
    desc.RasterizerState.ForcedSampleCount = 0;
    desc.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE(7);
    CHECK(f.run.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p)) == E_INVALIDARG);
    return true;
}

bool owned_device_caps() {
    copycmp::Ptr<ID3D12Device> device;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))));
    const auto* info = rt::d3d12::vulkan_device_info(device.p);
    CHECK(info);
    const auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        info->get_instance_proc(info->instance, "vkEnumerateDeviceExtensionProperties"));
    UINT count = 0;
    CHECK(enumerate(info->physical, nullptr, &count, nullptr) == VK_SUCCESS);
    std::vector<VkExtensionProperties> extensions(count);
    CHECK(enumerate(info->physical, nullptr, &count, extensions.data()) == VK_SUCCESS);
    const bool available = std::any_of(extensions.begin(), extensions.end(), [](const auto& e) {
        return std::strcmp(e.extensionName, VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME) == 0;
    });
    CHECK(info->conservative_rasterization == available);
    VkPhysicalDeviceConservativeRasterizationPropertiesEXT conservative{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    if (available) properties.pNext = &conservative;
    reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(info->get_instance_proc(info->instance,
        "vkGetPhysicalDeviceProperties2"))(info->physical, &properties);
    const auto expected = leonida::conservative_raster::tier(available,
        properties.properties.limits.subPixelPrecisionBits, conservative.primitiveOverestimationSize,
        conservative.degenerateTrianglesRasterized == VK_TRUE);
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    CHECK(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))));
    CHECK(unsigned(options.ConservativeRasterizationTier) == expected);
    return true;
}
} // namespace

int main(int argc, char** argv) {
    const bool compare_warp_msaa4 = argc == 2 && std::strcmp(argv[1], "--warp-msaa4") == 0;
    if (argc != 1 && !compare_warp_msaa4) return 2;
    if (hostwine::host_is_wine()) return 77;
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    VkPhysicalDeviceConservativeRasterizationPropertiesEXT conservative_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    if (vk.conservative_rasterization) properties.pNext = &conservative_properties;
    reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(vk.instance_proc(vk.instance, "vkGetPhysicalDeviceProperties2"))(
        vk.physical, &properties);
    if (!vk.buffer_device_address || properties.properties.limits.maxPushConstantsSize < 144) {
        std::puts("SKIP: current graphics path needs bufferDeviceAddress and 144 push-constant bytes"); return 77;
    }
    const auto vs_program = d3d12test::fssm_program({0x43525653, 0xbf810000});
    const auto ps_program = d3d12test::fssm_program({0x43525053, 0xbf810000});
    const auto vs_bytes = d3d12test::shader_container(vs_program), ps_bytes = d3d12test::shader_container(ps_program);
    const auto directory = std::filesystem::temp_directory_path() / ("leonida_conservative_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(directory);
    std::filesystem::copy_file(LEONIDA_CONSERVATIVE_VS, directory / "vs.spv", std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(LEONIDA_CONSERVATIVE_PS, directory / "ps.spv", std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(directory / "index.json", std::ios::binary);
        index << "{\"vs\":{\"stage\":\"vs\",\"fssm_sha256\":\"" << leonida::sha256_hex(vs_program)
              << "\",\"bindings\":[],\"push_constant_bytes\":0,\"input_signature\":[{\"semantic\":\"POSITION\",\"semantic_index\":0,\"register\":0,\"mask\":3,\"rw_mask\":3,\"system_value\":0,\"component_type\":\"float\"}]},"
                 "\"ps\":{\"stage\":\"ps\",\"fssm_sha256\":\"" << leonida::sha256_hex(ps_program)
              << "\",\"bindings\":[],\"push_constant_bytes\":0,\"outputs\":[0]}}";
    }
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc; info.instance = vk.instance; info.physical = vk.physical;
    info.device = vk.device; info.queue = vk.queue; info.queue_family = vk.family;
    info.buffer_device_address = vk.buffer_device_address;
    info.conservative_rasterization = vk.conservative_rasterization;
    info.shader_dir = directory.string();
    copycmp::Ptr<ID3D12Device> ours, host;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (FAILED(ours->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)))) return 1;
    const auto expected_tier = leonida::conservative_raster::tier(vk.conservative_rasterization,
        properties.properties.limits.subPixelPrecisionBits, conservative_properties.primitiveOverestimationSize,
        conservative_properties.degenerateTrianglesRasterized == VK_TRUE);
    if (unsigned(options.ConservativeRasterizationTier) != expected_tier) return 1;
    if (options.ConservativeRasterizationTier == D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED) {
        std::puts("SKIP: Vulkan device does not meet conservative raster Tier 1 requirements"); return 77;
    }
    if (options.ConservativeRasterizationTier > D3D12_CONSERVATIVE_RASTERIZATION_TIER_2) return 1;
    if (!owned_device_caps()) return 1;
    std::string adapter;
    if (!copycmp::create_host(host, adapter)) return 1;
    copycmp::Ptr<IDXGIFactory4> factory;
    copycmp::Ptr<IDXGIAdapter1> host_adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumAdapterByLuid(host->GetAdapterLuid(), IID_PPV_ARGS(&host_adapter.p)))) return 1;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    if (FAILED(host_adapter->GetDesc1(&adapter_desc))) return 1;
    const bool software_reference = (adapter_desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
    if (FAILED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)))) return 1;
    if (options.ConservativeRasterizationTier == D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED) return 77;
    copycmp::Ptr<ID3DBlob> vs, ps, error;
    const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile = compiler ? reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(
        GetProcAddress(compiler, "D3DCompile"))) : nullptr;
    if (!compile) { std::puts("SKIP: d3dcompiler_47 is needed for the native reference shaders"); return 77; }
    const char* hlsl = "float4 vs(float2 p:POSITION):SV_Position{return float4(p,0.5,1);} float4 ps():SV_Target{return float4(1,0,0,1);}";
    if (FAILED(compile(hlsl, std::strlen(hlsl), nullptr, nullptr, nullptr, "vs", "vs_5_0", 0, 0, &vs.p, &error.p)) ||
        FAILED(compile(hlsl, std::strlen(hlsl), nullptr, nullptr, nullptr, "ps", "ps_5_0", 0, 0, &ps.p, nullptr))) return 1;
    Fixture a, b;
    if (!setup(a, ours.p, true, {vs_bytes.data(), vs_bytes.size()}, {ps_bytes.data(), ps_bytes.size()}) ||
        !setup(b, host.p, false, {vs->GetBufferPointer(), vs->GetBufferSize()}, {ps->GetBufferPointer(), ps->GetBufferSize()})) return 1;
    if (!rejected_states(a, true)) return 1;
    {
        info.conservative_rasterization = false;
        copycmp::Ptr<ID3D12Device> disabled;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&disabled.p)))) return 1;
        if (FAILED(disabled->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
            options.ConservativeRasterizationTier != D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED) return 1;
        Fixture f;
        if (!setup(f, disabled.p, true, {vs_bytes.data(), vs_bytes.size()}, {ps_bytes.data(), ps_bytes.size()}) ||
            !rejected_states(f, false)) return 1;
    }
    bool msaa4 = true;
    for (auto* device : {ours.p, host.p}) {
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS levels{DXGI_FORMAT_R8G8B8A8_UNORM, 4};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &levels, sizeof(levels))) ||
            !levels.NumQualityLevels) msaa4 = false;
    }
    const Triangle scenes[] = {
        {{{3.0625,3.0625},{3.3125,3.0625},{3.0625,3.3125}}},
        {{{2.125,5.125},{25.125,6.125},{2.125,5.25}}},
        {{{4.125,4.125},{24.125,4.125},{4.125,24.125}}},
        {{{-0.25,2.125},{5.125,2.125},{2.125,15.125}}},
        {{{4.125,24.125},{24.125,4.125},{4.125,4.125}}},
        {{{3.0625,3.0625},{3.3125,3.0625},{3.0625,3.0625}}}, // Tier 1 permits degenerate triangles to be culled
    };
    unsigned comparisons = 0, host_comparisons = 0, host_not_comparable = 0;
    const auto count = [](const Mask& m) { return unsigned(std::count(m.begin(), m.end(), true)); };
    for (const UINT samples : {1u, 4u}) {
      if (samples == 4 && !msaa4) continue;
      // WARP's 4x path was observed to cover outside the Tier 1 uncertainty bound for the subpixel fixture.
      // Keep Vulkan's 4x geometric oracle strict; use 1x for the ordinary software-reference comparisons.
      // --warp-msaa4 explicitly reproduces that reference discrepancy without weakening either oracle.
      const bool compare_host = samples == 1 || !software_reference || compare_warp_msaa4;
      for (const bool variant : {false, true})
        for (const auto scissor : {D3D12_RECT{0,0,32,32}, D3D12_RECT{4,4,20,20}})
            for (const auto& triangle : scenes) {
                Mask normal_a{}, normal_b{}, conservative_a{}, conservative_b{};
                if (!render(a, triangle, false, samples, variant, scissor, normal_a) ||
                    (compare_host && !render(b, triangle, false, samples, variant, scissor, normal_b)) ||
                    !render(a, triangle, true, samples, variant, scissor, conservative_a) ||
                    (compare_host && !render(b, triangle, true, samples, variant, scissor, conservative_b))) return 1;
                const bool ours_ok = coverage(triangle, scissor, conservative_a);
                if (!ours_ok) {
                    std::fprintf(stderr, "scene %zu samples %u variant %d scissor-left %ld: Vulkan fails the oracle\n",
                                 std::size_t(&triangle - scenes), samples, int(variant), scissor.left);
                    return 1;
                }
                // The oracle is geometric, so the layer is judged by it alone. A native D3D12 render that fails it
                // (TEST-907: an RTX 5060 Ti on Windows 10 drew nothing for scene 4 with conservative raster on) is
                // reported, with both of its masks, and that scenario is not compared with the host.
                bool host_ok = compare_host && coverage(triangle, scissor, conservative_b);
                if (compare_host && !host_ok && compare_warp_msaa4) {
                    // --warp-msaa4 exists to reproduce WARP's known 4x discrepancy, so there it stays a failure.
                    std::fprintf(stderr, "scene %zu samples %u variant %d scissor-left %ld: host fails the oracle\n",
                                 std::size_t(&triangle - scenes), samples, int(variant), scissor.left);
                    return 1;
                }
                if (compare_host && !host_ok) {
                    unsigned normal_only = 0;
                    for (unsigned i = 0; i < normal_b.size(); ++i) normal_only += normal_b[i] && !conservative_b[i];
                    std::printf("host not comparable: scene %zu samples %u variant %d scissor-left %ld: host normal "
                                "mask %u pixels, conservative %u (%u normal pixels missing), Vulkan conservative %u\n",
                                std::size_t(&triangle - scenes), samples, int(variant), scissor.left, count(normal_b),
                                count(conservative_b), normal_only, count(conservative_a));
                    ++host_not_comparable;
                }
                for (unsigned i = 0; i < normal_a.size(); ++i)
                    if ((normal_a[i] && !conservative_a[i]) || (host_ok && normal_b[i] && !conservative_b[i])) return 1;
                if (&triangle == &scenes[0] && samples == 1 && scissor.left == 0) {
                    if (std::any_of(normal_a.begin(), normal_a.end(), [](bool x) { return x; }) ||
                        (host_ok && std::any_of(normal_b.begin(), normal_b.end(), [](bool x) { return x; })) ||
                        !std::any_of(conservative_a.begin(), conservative_a.end(), [](bool x) { return x; }) ||
                        (host_ok && !std::any_of(conservative_b.begin(), conservative_b.end(), [](bool x) { return x; }))) return 1;
                }
                ++comparisons;
                if (host_ok) ++host_comparisons;
            }
    }
    for (auto* f : {&a, &b}) {
        Mask front{}, back{}, all{};
        const D3D12_RECT full{0,0,32,32};
        if (!render(*f, scenes[2], true, 1, false, full, all) ||
            !render(*f, scenes[2], true, 1, false, full, front, D3D12_CULL_MODE_FRONT) ||
            !render(*f, scenes[2], true, 1, false, full, back, D3D12_CULL_MODE_BACK)) return 1;
        for (unsigned i = 0; i < all.size(); ++i) if ((front[i] && back[i]) || ((front[i] || back[i]) != all[i])) return 1;
    }
    std::printf("conservative raster: %u Vulkan and %u %s scenario pairs; geometric bounds, culling, scissor, variants, rejections; Vulkan 4x MSAA %s\n",
                comparisons, host_comparisons, adapter.c_str(), msaa4 ? "checked" : "unavailable");
    if (host_not_comparable)
        std::printf("%u scenarios not compared: the %s render fails the geometric oracle there (listed above)\n",
                    host_not_comparable, adapter.c_str());
    if (software_reference && !compare_warp_msaa4)
        std::puts("WARP comparison is 1x; --warp-msaa4 separately checks its observed 4x coverage discrepancy");
    return vk.validation_errors() ? 1 : 0;
}
