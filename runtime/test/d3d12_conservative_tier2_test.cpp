// Tier 2: tight uncertainty, exact and post-snap degenerates. RGB independently encodes varying colour,
// constant depth and facing. No game data.
#include "conservative_raster_fixture.h"
using namespace conservative_test;

namespace {
double cross(const Triangle& t) {
    return (t[1].x-t[0].x)*(t[2].y-t[0].y) - (t[1].y-t[0].y)*(t[2].x-t[0].x);
}

bool tight_coverage(const Triangle& original, const Triangle* snapped, const D3D12_RECT& scissor,
                    const Mask& mask, bool log = true) {
    constexpr double epsilon = 1e-7, uncertainty = 1.0 / 256;
    unsigned required_count = 0, covered = 0;
    for (UINT y=0;y<height;++y) for (UINT x=0;x<width;++x) {
        const bool in = LONG(x)>=scissor.left && LONG(x)<scissor.right && LONG(y)>=scissor.top && LONG(y)<scissor.bottom;
        bool hits = false;
        const double intersection = area(snapped ? *snapped : original,
            x+epsilon,y+epsilon,x+1-epsilon,y+1-epsilon,&hits);
        const bool required = in && (snapped ? hits : intersection > 1e-12);
        area(original,x-uncertainty-epsilon,y-uncertainty-epsilon,
             x+1+uncertainty+epsilon,y+1+uncertainty+epsilon,&hits);
        const bool allowed = in && hits;
        if ((required && !mask[y*width+x]) || (!allowed && mask[y*width+x])) {
            if (log) std::fprintf(stderr,"Tier 2 coverage (%u,%u): required=%d allowed=%d actual=%d\n",
                                 x,y,int(required),int(allowed),int(mask[y*width+x]));
            return false;
        }
        required_count += unsigned(required); covered += unsigned(mask[y*width+x]);
    }
    // A snapped point on a pixel corner has no required open-pixel intersection, but must cover at least one
    // adjacent pixel. Its particular edge/corner tie-breaking is not asserted.
    if (snapped && !required_count && !covered && log) std::fprintf(stderr,"snapped corner produced no fragments\n");
    return !snapped || required_count || covered;
}

VertexInput vertices(const std::vector<Point>& points, std::vector<std::uint16_t> indices,
                     D3D_PRIMITIVE_TOPOLOGY topology) {
    // Keep depth planar: native D3D12 and Vulkan may extrapolate SV_Position.z for screen-space degenerates.
    static constexpr float depths[] = {0.375f,0.375f,0.375f,0.375f};
    static constexpr float markers[] = {0.25f,0.75f,0.375f,0.875f};
    VertexInput out; out.indices=std::move(indices); out.topology=topology;
    for (std::size_t i=0;i<points.size();++i) {
        out.data.push_back(float(points[i].x*2/width-1));
        out.data.push_back(float(1-points[i].y*2/height));
        out.data.push_back(depths[i]); out.data.push_back(markers[i]);
    }
    return out;
}

Triangle actual_triangle(const VertexInput& input, unsigned first = 0) {
    Triangle t{};
    for (unsigned i=0;i<3;++i) {
        const auto at=std::size_t(input.indices[first+i])*4;
        t[i]={(double(input.data[at])+1)*width/2,(1-double(input.data[at+1]))*height/2};
    }
    return t;
}

Triangle quantized(Triangle t, unsigned bits) {
    const double grid=std::ldexp(1.0,int(bits));
    for (auto& p:t) { p.x=std::nearbyint(p.x*grid)/grid; p.y=std::nearbyint(p.y*grid)/grid; }
    return t;
}

bool attributes(const Mask& mask, const std::vector<unsigned char>& rgba, unsigned provoking) {
    static constexpr std::array<unsigned char,4> expected[] = {
        {64,96,0,255},{191,96,0,255},{96,96,0,255},{223,96,0,255}};
    for (unsigned i=0;i<mask.size();++i) {
        const auto* p=rgba.data()+i*4;
        if (!mask[i]) { CHECK(!(p[0]|p[1]|p[2]|p[3])); continue; }
        for (unsigned lane=0;lane<4;++lane) {
            if (p[lane]!=expected[provoking][lane]) {
                std::fprintf(stderr,"post-snap pixel %u lane %u: got %02x, provoking %u expects %02x\n",
                             i,lane,unsigned(p[lane]),provoking,unsigned(expected[provoking][lane]));
                return false;
            }
        }
    }
    return true;
}

bool oracle_control() {
    const Triangle triangle{{{8.015625,4.25},{9.25,4.25},{8.015625,24.25}}};
    const D3D12_RECT full{0,0,32,32};
    Mask mask{};
    for (UINT y=0;y<height;++y) for (UINT x=0;x<width;++x)
        mask[y*width+x]=area(triangle,x+1e-7,y+1e-7,x+1-1e-7,y+1-1e-7)>1e-12;
    CHECK(tight_coverage(triangle,nullptr,full,mask));
    mask[8*width+7]=true; // within Tier 1's half-pixel margin, outside Tier 2's 1/256 margin
    CHECK(coverage(triangle,full,mask));
    CHECK(!tight_coverage(triangle,nullptr,full,mask,false));
    return true;
}
} // namespace

int main(int argc, char** argv) {
    const bool warp_corner = argc > 1 && std::strcmp(argv[1], "--warp-corner") == 0;
    if (!oracle_control()) return 1;
    if (hostwine::host_is_wine()) return 77;
    vktest::Device vk; vk.request_buffer_device_address=true;
    const auto status=vk.initialize();
    if(status==vktest::DeviceStatus::unavailable)return 77;
    if(status!=vktest::DeviceStatus::ready)return 1;
    VkPhysicalDeviceConservativeRasterizationPropertiesEXT cr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    if(vk.conservative_rasterization)properties.pNext=&cr;
    reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(vk.instance_proc(vk.instance,"vkGetPhysicalDeviceProperties2"))(vk.physical,&properties);
    const auto tier=leonida::conservative_raster::tier(vk.conservative_rasterization,
        properties.properties.limits.subPixelPrecisionBits,cr.primitiveOverestimationSize,cr.degenerateTrianglesRasterized==VK_TRUE);
    if(!vk.buffer_device_address||properties.properties.limits.maxPushConstantsSize<144)return 77;
    const auto vs_program=d3d12test::fssm_program({0x54325653,0xbf810000});
    const auto ps_program=d3d12test::fssm_program({0x54325053,0xbf810000});
    const auto vs_bytes=d3d12test::shader_container(vs_program),ps_bytes=d3d12test::shader_container(ps_program);
    const auto directory=std::filesystem::temp_directory_path()/("leonida_conservative_tier2_"+std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(directory);
    std::filesystem::copy_file(LEONIDA_CONSERVATIVE_TIER2_VS,directory/"vs.spv",std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(LEONIDA_CONSERVATIVE_TIER2_PS,directory/"ps.spv",std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(directory/"index.json",std::ios::binary);
        index<<"{\"vs\":{\"stage\":\"vs\",\"fssm_sha256\":\""<<leonida::sha256_hex(vs_program)
             <<"\",\"bindings\":[],\"push_constant_bytes\":0,\"input_signature\":["
               "{\"semantic\":\"POSITION\",\"semantic_index\":0,\"register\":0,\"mask\":7,\"rw_mask\":7,\"system_value\":0,\"component_type\":\"float\"},"
               "{\"semantic\":\"COLOR\",\"semantic_index\":0,\"register\":1,\"mask\":1,\"rw_mask\":1,\"system_value\":0,\"component_type\":\"float\"}]},"
               "\"ps\":{\"stage\":\"ps\",\"fssm_sha256\":\""<<leonida::sha256_hex(ps_program)
             <<"\",\"bindings\":[],\"push_constant_bytes\":0,\"outputs\":[0]}}";
    }
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc=vk.instance_proc;info.instance=vk.instance;info.physical=vk.physical;
    info.device=vk.device;info.queue=vk.queue;info.queue_family=vk.family;
    info.buffer_device_address=vk.buffer_device_address;info.conservative_rasterization=vk.conservative_rasterization;
    info.shader_dir=directory.string();
    copycmp::Ptr<ID3D12Device> ours,host;
    if(FAILED(rt::d3d12::create_vulkan_device(info,IID_PPV_ARGS(&ours.p))))return 1;
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if(FAILED(ours->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,&options,sizeof(options)))||unsigned(options.ConservativeRasterizationTier)!=tier)return 1;
    if(tier<2){std::puts("SKIP: device does not meet conservative raster Tier 2 requirements");return 77;}
    std::string adapter;if(!copycmp::create_host(host,adapter))return 1;
    if(FAILED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,&options,sizeof(options))))return 1;
    if(options.ConservativeRasterizationTier<D3D12_CONSERVATIVE_RASTERIZATION_TIER_2)return 77;
    const HMODULE compiler=LoadLibraryExA("d3dcompiler_47.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile=compiler?reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(GetProcAddress(compiler,"D3DCompile"))):nullptr;
    if(!compile)return 77;
    const char* hlsl="struct O{float4 p:SV_Position;float m:COLOR;}; O vs(float3 p:POSITION,float m:COLOR){O o;o.p=float4(p,1);o.m=m;return o;} float4 ps(O o,bool front:SV_IsFrontFace):SV_Target{return float4(o.m,o.p.z,front?1:0,1);}";
    copycmp::Ptr<ID3DBlob> vs,ps,error;
    if(FAILED(compile(hlsl,std::strlen(hlsl),nullptr,nullptr,nullptr,"vs","vs_5_0",0,0,&vs.p,&error.p))||
       FAILED(compile(hlsl,std::strlen(hlsl),nullptr,nullptr,nullptr,"ps","ps_5_0",0,0,&ps.p,nullptr)))return 1;
    Fixture a,b;
    if(!setup(a,ours.p,true,{vs_bytes.data(),vs_bytes.size()},{ps_bytes.data(),ps_bytes.size()})||
       !setup(b,host.p,false,{vs->GetBufferPointer(),vs->GetBufferSize()},{ps->GetBufferPointer(),ps->GetBufferSize()}))return 1;
    const D3D12_INPUT_ELEMENT_DESC elements[]={
        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"COLOR",0,DXGI_FORMAT_R32_FLOAT,0,12,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}};
    a.input=b.input=elements;
    const D3D12_RECT full{0,0,32,32},inset{8,8,24,24};
    unsigned tight_cases=0,degenerate_cases=0;
    const Triangle tight_scenes[]={
        {{{8.015625,4.25},{9.25,4.25},{8.015625,24.25}}},
        {{{3.125,8.001953125},{28.125,8.005859375},{3.125,9.125}}},
        {{{3.125,3.125},{28.125,28.125},{3.125,3.140625}}},
        {{{-0.125,8.125},{12.125,8.125},{8.125,28.125}}},
    };
    for(auto* fixture:{&a,&b})for(const auto& triangle:tight_scenes)for(const auto scissor:{full,inset}) {
        Mask mask{};std::vector<unsigned char> rgba;
        const auto input=vertices(std::vector<Point>(triangle.begin(),triangle.end()),{0,1,2},D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        if(!render(*fixture,triangle,true,1,false,scissor,mask,D3D12_CULL_MODE_NONE,&input,&rgba)||
           !tight_coverage(triangle,nullptr,scissor,mask))return 1;
        ++tight_cases;
    }
    constexpr double delta=1.0/(1u<<20);
    const std::vector<std::vector<Point>> collapsed={
        {{16.125,16.125},{16.125+delta,16.125},{16.125,16.125+delta}},
        {{4.125,16.125},{24.125,16.125},{4.125,16.125+delta}},
        {{4.125,4.125},{24.125,24.125},{4.125,4.125+delta}},
        {{16,16},{16+delta,16},{16,16+delta}},
        {{4.125,16.125},{24.125,16.125},{12.125,16.125}},
        {{16.125,16.125},{16.125,16.125},{16.125,16.125}},
        {{16.125,16.125},{16.125+delta,16.125},{16.125,16.125+delta},{16.125+delta,16.125+delta}},
    };
    for(auto* fixture:{&a,&b})for(const auto& points:collapsed)
      for(const auto scissor:{full,inset})for(const bool variant:{false,true}) {
        std::vector<std::vector<std::uint16_t>> orders=points.size()==4
            ?std::vector<std::vector<std::uint16_t>>{{0,1,2,3},{3,2,1,0}}
            :std::vector<std::vector<std::uint16_t>>{{0,1,2},{2,1,0},{1,2,0}};
        // WARP drops the exact pixel-corner fixture; retain a strict opt-in reproducer.
        if (!fixture->run.ours && adapter.find("Microsoft Basic") != std::string::npos && points[0].x == 16 && !warp_corner) {
            std::puts("WARP reference: corner fixture excluded (use --warp-corner to reproduce); Vulkan still checks it");
            continue;
        }
        for(const auto& order:orders) {
            const bool strip=points.size()==4;
            const auto input=vertices(points,order,strip?D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP:D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            const auto triangle=actual_triangle(input);
            const auto snapped=quantized(triangle,properties.properties.limits.subPixelPrecisionBits);
            const bool must_have_area = points.size() == 4 || points[2].y != points[0].y;
            if (cross(snapped) != 0 || (must_have_area && cross(triangle) == 0)) {
                std::fprintf(stderr,"fixture lost its intended pre/post-snap area\n"); return 1;
            }
            for(const auto cull:{D3D12_CULL_MODE_NONE,D3D12_CULL_MODE_FRONT,D3D12_CULL_MODE_BACK}) {
                Mask mask{};std::vector<unsigned char> rgba;
                if(!render(*fixture,triangle,true,1,variant,scissor,mask,cull,&input,&rgba))return 1;
                if(cull==D3D12_CULL_MODE_BACK) {
                    if(std::any_of(mask.begin(),mask.end(),[](bool x){return x;})){std::fprintf(stderr,"post-snap degenerate was not back-facing\n");return 1;}
                } else if(!tight_coverage(triangle,&snapped,scissor,mask)||!attributes(mask,rgba,order[strip?1:0])) {
                    std::fprintf(stderr,"%s degenerate points=%zu first=%u strip=%d cull=%u\n",
                                 fixture->run.ours?"Vulkan":adapter.c_str(),points.size(),order[0],int(strip),unsigned(cull)); std::fprintf(stderr,"point=(%.9f,%.9f) scissor=%ld variant=%d\n",points[0].x,points[0].y,scissor.left,int(variant));return 1;
                }
                ++degenerate_cases;
            }
        }
    }
    std::printf("Tier 2: %u tight-bound scenes and %u degenerate draw cases across Vulkan and %s; colour/constant-depth/facing, index permutations, strips, culling, scissor and variants checked\n",
                tight_cases,degenerate_cases,adapter.c_str());
    return vk.validation_errors()?1:0;
}
