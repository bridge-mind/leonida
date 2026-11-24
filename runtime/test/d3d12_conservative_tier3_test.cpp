// Explicit FullyCoveredEXT library contract versus native SV_InnerCoverage. Synthetic data only.
#include "conservative_raster_fixture.h"
using namespace conservative_test;

namespace {
double cross(Point a, Point b, Point p) {
    return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x);
}

// Same geometric construction as #3970. The conservative margin avoids implementation-dependent edge pixels.
bool inside(const Triangle& t, UINT x, UINT y, double margin) {
    const double winding = cross(t[0],t[1],t[2]);
    if (std::abs(winding)<1e-12) return false;
    const double sign = winding>0 ? 1.0 : -1.0;
    const Point corners[]={{double(x),double(y)},{double(x+1),double(y)},
                           {double(x),double(y+1)},{double(x+1),double(y+1)}};
    for (unsigned i=0;i<3;++i) {
        const Point a=t[i], b=t[(i+1)%3];
        const double length=std::hypot(b.x-a.x,b.y-a.y);
        if (!length) return false;
        for (const auto p:corners) if (sign*cross(a,b,p)/length<=margin) return false;
    }
    return true;
}

bool oracle(const Triangle& t, const D3D12_RECT& scissor, const Mask& mask,
            const std::vector<unsigned char>& rgba, bool log=true) {
    for (UINT y=0;y<height;++y) for (UINT x=0;x<width;++x) {
        const unsigned at=y*width+x;
        const auto* p=rgba.data()+4*at;
        const bool in=LONG(x)>=scissor.left && LONG(x)<scissor.right && LONG(y)>=scissor.top && LONG(y)<scissor.bottom;
        // Check required interior even if the shader never ran at this pixel.
        const bool required=in && inside(t,x,y,1.0/64);
        const bool inner=p[1]==255;
        if ((required && (!mask[at] || !inner)) ||
            (inner && (!in || !inside(t,x,y,-1e-10))) ||
            (mask[at] && (p[0]!=255 || p[3]!=255 || (p[1]!=0 && p[1]!=255) || p[2]!=(inner?0:255))) ||
            (!mask[at] && (p[0]|p[1]|p[2]|p[3]))) {
            if (log) std::fprintf(stderr,"InnerCoverage (%u,%u): required=%d ran=%d rgba=%u,%u,%u,%u\n",
                                  x,y,int(required),int(mask[at]),p[0],p[1],p[2],p[3]);
            return false;
        }
    }
    return true;
}

bool oracle_controls() {
    const Triangle t{{{3.125,3.125},{28.125,3.125},{3.125,28.125}}};
    const D3D12_RECT full{0,0,32,32};
    Mask mask{}; std::vector<unsigned char> rgba(width*height*4);
    for (UINT y=0;y<height;++y) for (UINT x=0;x<width;++x) {
        if (area(t,x,y,x+1,y+1)==0) continue;
        const unsigned at=y*width+x;
        mask[at]=true;
        const bool inner=inside(t,x,y,1.0/64);
        rgba[4*at]=rgba[4*at+3]=255; rgba[4*at+1]=inner?255:0; rgba[4*at+2]=inner?0:255;
    }
    CHECK(oracle(t,full,mask,rgba));
    auto bad=rgba;
    bad[4*(3*width+3)+1]=255; bad[4*(3*width+3)+2]=0; // partial pixel incorrectly declared inner
    CHECK(!oracle(t,full,mask,bad,false));
    bad=rgba;
    bad[4*(4*width+4)+1]=0; bad[4*(4*width+4)+2]=255; // fully interior but reported partial
    CHECK(!oracle(t,full,mask,bad,false));
    bad=rgba; auto absent=mask;
    absent[4*width+4]=false;
    std::fill_n(bad.begin()+4*(4*width+4),4,0); // a missing invocation must also fail
    CHECK(!oracle(t,full,absent,bad,false));
    return true;
}

bool refused(rt::d3d12::VulkanDeviceInfo info, D3D12_SHADER_BYTECODE vs, D3D12_SHADER_BYTECODE ps,
             unsigned expected_tier) {
    copycmp::Ptr<ID3D12Device> device;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info,IID_PPV_ARGS(&device.p))));
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    CHECK(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,&options,sizeof(options))));
    CHECK(unsigned(options.ConservativeRasterizationTier)==expected_tier);
    Fixture fixture; CHECK(setup(fixture,device.p,true,vs,ps));
    for (const bool conservative:{false,true}) {
        auto desc=description(fixture,conservative,1,false);
        copycmp::Ptr<ID3D12PipelineState> pso;
        CHECK(device->CreateGraphicsPipelineState(&desc,IID_PPV_ARGS(&pso.p))==E_NOTIMPL);
        CHECK(!pso.p);
    }
    return true;
}
} // namespace

int main(int argc, char** argv) {
    if (!oracle_controls()) return 1;
    if (argc>1 && std::strcmp(argv[1],"--oracle-only")==0) return 0;
    const bool guards_only=argc>1 && std::strcmp(argv[1],"--guards-only")==0;
    const bool native_only=argc>1 && std::strcmp(argv[1],"--native-only")==0;
    if (hostwine::host_is_wine()) return 77;
    vktest::Device vk; vk.request_buffer_device_address=true;
    const auto status=vk.initialize();
    if (status==vktest::DeviceStatus::unavailable) return 77;
    if (status!=vktest::DeviceStatus::ready) return 1;
    VkPhysicalDeviceConservativeRasterizationPropertiesEXT cr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT};
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    id.pNext=vk.conservative_rasterization?&cr:nullptr;
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,&id};
    const auto get_properties=reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(vk.instance_proc(vk.instance,"vkGetPhysicalDeviceProperties2"));
    if (!get_properties) return 1;
    get_properties(vk.physical,&properties);
    if (!vk.buffer_device_address || properties.properties.limits.maxPushConstantsSize<144) return 77;
    const auto vs_program=d3d12test::fssm_program({0x54335653,0xbf810000});
    const auto ps_program=d3d12test::fssm_program({0x54335053,0xbf810000});
    const auto vs_bytes=d3d12test::shader_container(vs_program), ps_bytes=d3d12test::shader_container(ps_program);
    const auto dir=std::filesystem::temp_directory_path()/("leonida_conservative_tier3_"+std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(dir);
    std::filesystem::copy_file(LEONIDA_CONSERVATIVE_VS,dir/"vs.spv",std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(LEONIDA_CONSERVATIVE_TIER3_PS,dir/"ps.spv",std::filesystem::copy_options::overwrite_existing);
    {
        // Omit optional capability metadata deliberately: the actual SPIR-V must enforce the unsupported guard.
        std::ofstream out(dir/"index.json",std::ios::binary);
        out<<"{\"vs\":{\"stage\":\"vs\",\"fssm_sha256\":\""<<leonida::sha256_hex(vs_program)
           <<"\",\"bindings\":[],\"push_constant_bytes\":0,\"input_signature\":[{\"semantic\":\"POSITION\","
             "\"register\":0,\"component_type\":\"float\"}]},\"ps\":{\"stage\":\"ps\",\"fssm_sha256\":\""
           <<leonida::sha256_hex(ps_program)<<"\",\"bindings\":[],\"push_constant_bytes\":0,\"outputs\":[0]}}";
    }
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc=vk.instance_proc; info.instance=vk.instance; info.physical=vk.physical;
    info.device=vk.device; info.queue=vk.queue; info.queue_family=vk.family;
    info.buffer_device_address=vk.buffer_device_address; info.conservative_rasterization=vk.conservative_rasterization;
    info.shader_dir=dir.string();
    const D3D12_SHADER_BYTECODE vs_code{vs_bytes.data(),vs_bytes.size()}, ps_code{ps_bytes.data(),ps_bytes.size()};
    const auto base=leonida::conservative_raster::tier(vk.conservative_rasterization,
        properties.properties.limits.subPixelPrecisionBits,cr.primitiveOverestimationSize,cr.degenerateTrianglesRasterized==VK_TRUE);
    if (!refused(info,vs_code,ps_code,base)) return 1; // ABI opt-in false, even on a capable GPU
    info.inner_coverage_abi=true;
    const auto tier=leonida::conservative_raster::tier(vk.conservative_rasterization,
        properties.properties.limits.subPixelPrecisionBits,cr.primitiveOverestimationSize,
        cr.degenerateTrianglesRasterized==VK_TRUE,cr.fullyCoveredFragmentShaderInputVariable==VK_TRUE,true);
    std::printf("Tier 3 explicit ABI: %s; fullyCoveredFragmentShaderInputVariable=%u; tier=%u\n",
                properties.properties.deviceName,cr.fullyCoveredFragmentShaderInputVariable,tier);
    if (tier<3 && !refused(info,vs_code,ps_code,tier)) return 1;
    if (vk.validation_errors()) return 1;
    if (guards_only) { std::puts("InnerCoverage unsupported/ABI guards passed"); return 0; }
    if (tier<3 && !native_only) { std::puts("SKIP: positive Tier 3 path needs a FullyCoveredEXT-capable device"); return 77; }
    if (!id.deviceLUIDValid) return 77;
    LUID luid{}; static_assert(sizeof(luid)==VK_LUID_SIZE); std::memcpy(&luid,id.deviceLUID,sizeof(luid));
    copycmp::Ptr<ID3D12Device> host,ours; std::string adapter;
    if (!copycmp::try_create_host(host,adapter,&luid)) return 77;
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (FAILED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,&options,sizeof(options)))) return 1;
    if (options.ConservativeRasterizationTier<D3D12_CONSERVATIVE_RASTERIZATION_TIER_3) return 77;
    const HMODULE dll=LoadLibraryExA("d3dcompiler_47.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile=dll?reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(GetProcAddress(dll,"D3DCompile"))):nullptr;
    if (!compile) return 77;
    const char* hlsl="float4 vs(float2 p:POSITION):SV_Position{return float4(p,0.5,1);}"
                     "float4 ps(uint inner:SV_InnerCoverage):SV_Target0{return float4(1,(inner&1u)?1:0,inner==0u?1:0,1);}";
    copycmp::Ptr<ID3DBlob> vs,ps,error;
    if (FAILED(compile(hlsl,std::strlen(hlsl),nullptr,nullptr,nullptr,"vs","vs_5_0",0,0,&vs.p,&error.p)) ||
        FAILED(compile(hlsl,std::strlen(hlsl),nullptr,nullptr,nullptr,"ps","ps_5_0",0,0,&ps.p,nullptr))) return 1;
    Fixture a,b;
    if (!native_only) {
        if (FAILED(rt::d3d12::create_vulkan_device(info,IID_PPV_ARGS(&ours.p)))) return 1;
        if (FAILED(ours->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,&options,sizeof(options))) ||
            options.ConservativeRasterizationTier!=D3D12_CONSERVATIVE_RASTERIZATION_TIER_3) return 1;
        if (!setup(a,ours.p,true,vs_code,ps_code)) return 1;
    }
    if (!setup(b,host.p,false,{vs->GetBufferPointer(),vs->GetBufferSize()},{ps->GetBufferPointer(),ps->GetBufferSize()})) return 1;
    const Triangle scenes[]={
        {{{3.125,3.125},{28.125,3.125},{3.125,28.125}}},
        {{{3.125,6.125},{27.125,11.125},{8.125,28.125}}},
        {{{4.125,24.125},{27.125,6.125},{27.125,25.125}}},
        {{{3.125,8.125},{28.125,8.625},{16.125,25.125}}},
        {{{3.0625,3.0625},{3.3125,3.0625},{3.0625,3.3125}}},
        {{{4.125,16.125},{24.125,16.125},{12.125,16.125}}},
    };
    unsigned comparisons=0;
    for (const auto& triangle:scenes) for (const UINT samples:{1u,4u})
      for (const bool variant:{false,true}) for (const D3D12_RECT scissor:{D3D12_RECT{0,0,32,32},D3D12_RECT{8,8,24,24}}) {
        // InnerCoverage is undefined in D3D12 when conservative rasterization is off: no off-mode oracle.
        for (auto* fixture:{&a,&b}) {
            if (native_only && fixture==&a) continue;
            Mask mask{}; std::vector<unsigned char> rgba;
            if (!render(*fixture,triangle,true,samples,variant,scissor,mask,D3D12_CULL_MODE_NONE,nullptr,&rgba) ||
                !coverage(triangle,scissor,mask) || !oracle(triangle,scissor,mask,rgba)) {
                std::fprintf(stderr,"%s Tier 3 scene=%zu samples=%u variant=%d scissor=%ld\n",
                             fixture->run.ours?"Vulkan":adapter.c_str(),std::size_t(&triangle-scenes),samples,int(variant),scissor.left);
                return 1;
            }
        }
        ++comparisons;
    }
    std::printf("Tier 3 explicit ABI: %u %s %s scenarios; inner positives/negatives, 1x/4x, scissor and variants\n",
                comparisons,native_only?"native-only":"Vulkan/native paired",adapter.c_str());
    return vk.validation_errors()?1:0;
}
