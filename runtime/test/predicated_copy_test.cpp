// Preparation-only checks: no Vulkan calls, loader, shaders, GPU or game data.
#include "../src/vulkan_replay.h"
#include <cstdio>
#define EXPECT(x) do { if(!(x)){std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1;} }while(0)
// Non-null capability markers; prepare() never calls these entry points. Nothing is emitted by this test.
template<class T>T available(){return reinterpret_cast<T>(+[](){});}
template<class T>T handle(std::uintptr_t n){return reinterpret_cast<T>(n);}
int main(){
    using namespace leonida;
    ReplayFunctions vk{};
    vk.copy_buffer=available<PFN_vkCmdCopyBuffer>();vk.copy_buffer_to_image=available<PFN_vkCmdCopyBufferToImage>();
    vk.copy_image_to_buffer=available<PFN_vkCmdCopyImageToBuffer>();vk.copy_image=available<PFN_vkCmdCopyImage>();
    vk.begin_conditional_rendering=available<PFN_vkCmdBeginConditionalRenderingEXT>();
    vk.end_conditional_rendering=available<PFN_vkCmdEndConditionalRenderingEXT>();
    vk.prepare_predicate=[](VkBuffer,VkDeviceSize,PredicateCapture& out){out.scratch=handle<VkBuffer>(4);return true;};
    vk.capture_predicate=[](VkCommandBuffer,const PredicateCapture&){};
    ResourceBinding source{},destination{},predicate{};
    source.buffer=handle<VkBuffer>(1);source.size=64;source.buffer_offset=(UINT64_C(1)<<33)+1;
    destination.buffer=handle<VkBuffer>(2);destination.size=64;destination.buffer_offset=(UINT64_C(1)<<34)+3;
    predicate.buffer=handle<VkBuffer>(3);predicate.size=16;
    auto resolve=[&](ResourceId id)->const ResourceBinding*{return id==1?&source:id==2?&destination:id==3?&predicate:nullptr;};
    VkDeviceSize got_src=0,got_dst=0,got_bytes=0;unsigned prepared=0;
    auto copier=[&](VkBuffer a,VkBuffer b,VkDeviceSize src,VkDeviceSize dst,VkDeviceSize bytes,std::vector<PredicatedCopyDispatch>& jobs){
        if(a!=source.buffer||b!=destination.buffer)return false;
        got_src=src;got_dst=dst;got_bytes=bytes;++prepared;
        jobs.push_back({handle<VkDescriptorSet>(5),0,0,static_cast<std::uint32_t>(bytes)});return true;
    };
    PreparedList result;
    auto check=[&](CopyBuffer copy,bool predicated=true){
        CommandList list;
        if(predicated)list.set_predication({3,8,d3d12_predication::equal_zero});
        list.copy_buffer(copy);list.close();
        return prepare(list,resolve,vk,result);
    };
    EXPECT(check({2,3,1,1,7})==ReplayResult::unsupported); // optional capability absent
    vk.prepare_predicated_copy=copier;
    EXPECT(check({2,3,1,1,7})==ReplayResult::unsupported); // emitter absent
    vk.emit_predicated_copy=[](VkCommandBuffer,const std::vector<PredicatedCopyDispatch>&){};
    EXPECT(check({2,3,1,1,7})==ReplayResult::success);
    EXPECT(prepared==1&&got_src==source.buffer_offset+1&&got_dst==destination.buffer_offset+3&&got_bytes==7);
    EXPECT(result.unpredicated_copies==0&&result.commands.back().buffer_copies.empty());
    EXPECT(check({2,63,1,63,1},false)==ReplayResult::success);
    EXPECT(result.commands.back().buffer_copies[0].srcOffset==source.buffer_offset+63);
    EXPECT(result.commands.back().buffer_copies[0].dstOffset==destination.buffer_offset+63&&prepared==1);
    EXPECT(check({2,64,1,64,0})==ReplayResult::success&&prepared==1); // native no-op; no shader/descriptor
    EXPECT(check({2,64,1,0,1})==ReplayResult::invalid_copy);
    EXPECT(check({2,0,1,64,1})==ReplayResult::invalid_copy);
    EXPECT(check({9,0,1,0,1})==ReplayResult::unresolved_resource);
    const auto saved=source;
    source.buffer_offset=UINT64_MAX;
    EXPECT(check({2,0,1,1,1})==ReplayResult::invalid_copy); // offset addition
    EXPECT(check({2,0,1,0,2})==ReplayResult::invalid_copy); // end addition
    source=saved;
    destination.buffer=source.buffer;destination.buffer_offset=source.buffer_offset;
    EXPECT(check({2,1,1,0,7})==ReplayResult::invalid_copy); // physical overlap across resource IDs
    EXPECT(check({2,8,1,0,7})==ReplayResult::success);     // same buffer, disjoint ranges
    destination.buffer=handle<VkBuffer>(2);destination.buffer_offset=0;
    vk.prepare_predicated_copy=[](VkBuffer,VkBuffer,VkDeviceSize,VkDeviceSize,VkDeviceSize,std::vector<PredicatedCopyDispatch>&){return false;};
    EXPECT(check({2,0,1,0,1})==ReplayResult::unsupported); // allocation/preparation failure cannot fall back
    vk.prepare_predicated_copy=copier;
    CommandList resource_copy;
    EXPECT(resource_copy.set_predication({3,0,d3d12_predication::equal_zero}));
    EXPECT(resource_copy.copy_resource({2,1})&&resource_copy.close());
    EXPECT(prepare(resource_copy,resolve,vk,result)==ReplayResult::success&&result.unpredicated_copies==1);
    vk.reset_query_pool=available<PFN_vkCmdResetQueryPool>();vk.begin_query=available<PFN_vkCmdBeginQuery>();
    vk.end_query=available<PFN_vkCmdEndQuery>();vk.write_timestamp=available<PFN_vkCmdWriteTimestamp>();
    vk.copy_query_pool_results=available<PFN_vkCmdCopyQueryPoolResults>();vk.fill_buffer=available<PFN_vkCmdFillBuffer>();
    QueryPool pool{};pool.pool=handle<VkQueryPool>(6);pool.heap_type=d3d12_query_heap::pipeline_statistics;pool.count=1;
    vk.query_pool=[&](QueryHeapId){return &pool;};
    CommandList statistics;
    EXPECT(statistics.set_predication({3,0,d3d12_predication::equal_zero}));
    EXPECT(statistics.begin_query({1,d3d12_query::pipeline_statistics,0}));
    EXPECT(statistics.copy_buffer({2,0,1,0,1}));
    EXPECT(statistics.end_query({1,d3d12_query::pipeline_statistics,0})&&statistics.close());
    const auto before=prepared;
    EXPECT(prepare(statistics,resolve,vk,result)==ReplayResult::unsupported && prepared==before);
    EXPECT(prepare(statistics,resolve,vk,result,true)==ReplayResult::success && prepared==before+1);
    EXPECT(result.statistics_segment_count==2);
    EXPECT(result.commands[2].statistics_end==0 && result.commands[2].statistics_begin==1);
    // Readback visibility follows the actual producer, including mixed command lists.
    destination.host_readable = false;
    EXPECT(check({2,0,1,0,1})==ReplayResult::success && !result.wrote_readback && !result.readback_src_stages);
    destination.host_readable = true;
    EXPECT(check({2,0,1,0,1})==ReplayResult::success && result.wrote_readback);
    EXPECT(result.readback_src_stages==VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    EXPECT(check({2,0,1,0,1},false)==ReplayResult::success && result.wrote_readback);
    EXPECT(result.readback_src_stages==VK_PIPELINE_STAGE_2_COPY_BIT);
    CommandList mixed;
    EXPECT(mixed.set_predication({3,0,d3d12_predication::equal_zero}));
    EXPECT(mixed.copy_buffer({2,0,1,0,1}) && mixed.set_predication({0,0,0}));
    EXPECT(mixed.copy_buffer({2,1,1,1,1}) && mixed.close());
    EXPECT(prepare(mixed,resolve,vk,result)==ReplayResult::success && result.wrote_readback);
    EXPECT(result.readback_src_stages==(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_2_COPY_BIT));
    EXPECT(check({2,0,1,0,0})==ReplayResult::success && !result.wrote_readback && !result.readback_src_stages);
    PredicatedBufferCopier missing(+[](VkDevice,const char*)->PFN_vkVoidFunction{return nullptr;},VK_NULL_HANDLE,{});
    std::vector<PredicatedCopyDispatch> jobs;
    EXPECT(!missing.batch()->prepare(source.buffer,destination.buffer,0,0,1,jobs));
    std::puts("predicated copy preparation: capability, routing, 64-bit offsets, bounds, overflow, overlap, zero size, failure and statistics-query controls passed");
}
