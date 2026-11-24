// Synthetic byte-for-byte CopyBufferRegion oracle. --warp-only measures the native reference without Vulkan.
#include "vk_test_device.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <array>
#include <vector>
#include <cstring>
#include <cstdio>
#include <utility>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return false; } } while (0)
namespace {
template<class T> struct Ref { T* p = nullptr; ~Ref() { if (p) p->Release(); } T* operator->() const { return p; } };
struct Case { UINT64 source_bytes, destination_bytes, source_offset, destination_offset, bytes; };
bool buffer(ID3D12Device* device, UINT64 size, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out) {
    const D3D12_HEAP_PROPERTIES properties{heap};
    D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size; desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1; desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&out.p))));
    return true;
}
bool write(ID3D12Resource* resource, const void* bytes, SIZE_T size) {
    void* mapped = nullptr; const D3D12_RANGE none{};
    CHECK(SUCCEEDED(resource->Map(0, &none, &mapped)) && mapped);
    std::memcpy(mapped, bytes, size); const D3D12_RANGE written{0, size}; resource->Unmap(0, &written); return true;
}
bool finish(ID3D12GraphicsCommandList* list, ID3D12CommandQueue* queue, ID3D12Fence* fence) {
    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list}; queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence, 1)));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr); CHECK(event);
    const HRESULT hr = fence->SetEventOnCompletion(1, event);
    const DWORD wait = SUCCEEDED(hr) ? WaitForSingleObject(event, 10000) : WAIT_FAILED;
    CloseHandle(event); CHECK(SUCCEEDED(hr) && wait == WAIT_OBJECT_0);
    return true;
}
void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before; barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}
bool run(ID3D12Device* device, const Case& c, UINT64 predicate, D3D12_PREDICATION_OP op,
         bool enabled, std::vector<unsigned char>& actual, bool statistics_query = false) {
    Ref<ID3D12Resource> source, destination, seed, condition, readback;
    CHECK(buffer(device,c.source_bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,source));
    CHECK(buffer(device,c.destination_bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST,destination));
    CHECK(buffer(device,c.destination_bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,seed));
    CHECK(buffer(device,24,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,condition));
    CHECK(buffer(device,c.destination_bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,readback));
    std::vector<unsigned char> input(SIZE_T(c.source_bytes)), sentinel(SIZE_T(c.destination_bytes));
    for (SIZE_T i=0;i<input.size();++i) input[i]=static_cast<unsigned char>(i*37+11);
    for (SIZE_T i=0;i<sentinel.size();++i) sentinel[i]=static_cast<unsigned char>(i*13+173);
    const UINT64 conditions[3]={UINT64_MAX,predicate,0};
    CHECK(write(source.p,input.data(),input.size()) && write(seed.p,sentinel.data(),sentinel.size()) && write(condition.p,conditions,sizeof(conditions)));
    Ref<ID3D12CommandQueue> queue; Ref<ID3D12CommandAllocator> allocator; Ref<ID3D12GraphicsCommandList> list; Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC desc{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&desc,IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.p,nullptr,IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence.p))));
    list->CopyBufferRegion(destination.p,0,seed.p,0,c.destination_bytes);
    // Explicit dependency also covers the unpredicated control path.
    D3D12_RESOURCE_BARRIER seed_barrier{}; seed_barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    seed_barrier.Transition.pResource=destination.p;
    seed_barrier.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    seed_barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST;
    seed_barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_COMMON;
    list->ResourceBarrier(1,&seed_barrier);
    std::swap(seed_barrier.Transition.StateBefore,seed_barrier.Transition.StateAfter);
    list->ResourceBarrier(1,&seed_barrier);
    Ref<ID3D12QueryHeap> statistics;
    Ref<ID3D12Resource> counters;
    if (statistics_query) {
        const D3D12_QUERY_HEAP_DESC query_desc{D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS,1,0};
        CHECK(SUCCEEDED(device->CreateQueryHeap(&query_desc,IID_PPV_ARGS(&statistics.p))));
        CHECK(buffer(device,88,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,counters));
        list->BeginQuery(statistics.p,D3D12_QUERY_TYPE_PIPELINE_STATISTICS,0);
    }
    if(enabled) list->SetPredication(condition.p,8,op);
    list->CopyBufferRegion(destination.p,c.destination_offset,source.p,c.source_offset,c.bytes);
    list->SetPredication(nullptr,0,D3D12_PREDICATION_OP_EQUAL_ZERO);
    if (statistics_query) {
        list->EndQuery(statistics.p,D3D12_QUERY_TYPE_PIPELINE_STATISTICS,0);
        list->ResolveQueryData(statistics.p,D3D12_QUERY_TYPE_PIPELINE_STATISTICS,0,1,counters.p,0);
    }
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource=destination.p; barrier.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST; barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1,&barrier); list->CopyBufferRegion(readback.p,0,destination.p,0,c.destination_bytes);
    CHECK(finish(list.p, queue.p, fence.p));
    if (statistics_query) {
        void* values=nullptr; const D3D12_RANGE range{0,88}, none{};
        CHECK(SUCCEEDED(counters->Map(0,&range,&values)) && values);
        bool zero=true;
        for (unsigned word=0; word<11; ++word) {
            UINT64 value=UINT64_MAX;
            std::memcpy(&value,static_cast<const char*>(values)+8*word,8);
            if (value) { std::fprintf(stderr,"statistics word %u: %llu, expected zero\n",word,value); zero=false; }
        }
        counters->Unmap(0,&none); CHECK(zero);
    }
    void* mapped=nullptr; const D3D12_RANGE range{0,SIZE_T(c.destination_bytes)};
    CHECK(SUCCEEDED(readback->Map(0,&range,&mapped))&&mapped);
    actual.assign(static_cast<unsigned char*>(mapped),static_cast<unsigned char*>(mapped)+c.destination_bytes);
    const D3D12_RANGE none{}; readback->Unmap(0,&none);
    const bool executes=!enabled || (op==D3D12_PREDICATION_OP_EQUAL_ZERO ? predicate!=0 : predicate==0);
    if(executes && c.bytes) std::memcpy(sentinel.data()+c.destination_offset,input.data()+c.source_offset,SIZE_T(c.bytes));
    for(SIZE_T i=0;i<sentinel.size();++i) if(actual[i]!=sentinel[i]) {
        std::fprintf(stderr,"copy %llu/%llu +%llu/%llu size %llu predicate %016llx op %u enabled %d byte %zu got %02x expected %02x\n",
            c.source_bytes,c.destination_bytes,c.source_offset,c.destination_offset,c.bytes,predicate,unsigned(op),int(enabled),i,unsigned(actual[i]),unsigned(sentinel[i])); return false;
    }
    return true;
}

// A real empty occlusion query produces zero; an executing GPU copy can overwrite it with a high-word-only value.
// Keeping/recapturing the same address distinguishes the captured predicate from the live application buffer.
// The query is intentionally empty: this test does not claim visible-draw or nonzero-occlusion coverage.
bool run_query_copy(ID3D12Device* device, D3D12_QUERY_TYPE query_type, D3D12_PREDICATION_OP op,
                    UINT64 offset, bool overwrite, bool recapture, std::vector<unsigned char>& actual) {
    constexpr SIZE_T destination_bytes = 128, condition_bytes = 48, total_bytes = 184;
    Ref<ID3D12Resource> source, seed, condition_seed, destination, condition, readback, replacement_source;
    CHECK(!overwrite || op==D3D12_PREDICATION_OP_NOT_EQUAL_ZERO);
    CHECK(buffer(device,8,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,replacement_source));
    const UINT64 high_word=UINT64_C(1)<<32;
    CHECK(write(replacement_source.p,&high_word,sizeof(high_word)));
    CHECK(buffer(device,32,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,source));
    CHECK(buffer(device,destination_bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,seed));
    CHECK(buffer(device,condition_bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,condition_seed));
    CHECK(buffer(device,destination_bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST,destination));
    CHECK(buffer(device,condition_bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST,condition));
    CHECK(buffer(device,total_bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,readback));
    std::array<unsigned char,32> input{};
    std::array<unsigned char,destination_bytes> expected{}; expected.fill(0xcd);
    std::array<unsigned char,condition_bytes> guard{}; guard.fill(0xa7);
    for(SIZE_T i=0;i<input.size();++i) input[i]=static_cast<unsigned char>(37*i+11);
    CHECK(write(source.p,input.data(),input.size()) && write(seed.p,expected.data(),expected.size()) &&
          write(condition_seed.p,guard.data(),guard.size()));
    Ref<ID3D12QueryHeap> occlusion;
    const D3D12_QUERY_HEAP_DESC occlusion_desc{D3D12_QUERY_HEAP_TYPE_OCCLUSION,1,0};
    CHECK(SUCCEEDED(device->CreateQueryHeap(&occlusion_desc,IID_PPV_ARGS(&occlusion.p))));
    Ref<ID3D12CommandQueue> queue; Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list; Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc,IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.p,nullptr,IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence.p))));
    list->CopyBufferRegion(destination.p,0,seed.p,0,destination_bytes);
    list->CopyBufferRegion(condition.p,0,condition_seed.p,0,condition_bytes);
    for(auto* resource:{destination.p,condition.p}) {
        transition(list.p,resource,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);
        transition(list.p,resource,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
    }
    list->BeginQuery(occlusion.p,query_type,0);
    list->EndQuery(occlusion.p,query_type,0);
    list->ResolveQueryData(occlusion.p,query_type,0,1,condition.p,offset);
    transition(list.p,condition.p,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_PREDICATION);
    list->SetPredication(condition.p,offset,op);
    list->CopyBufferRegion(destination.p,3,source.p,2,7);
    if(overwrite) {
        // With zero + NOT_EQUAL_ZERO, this copy executes on both backends. Verify its eight bytes below.
        transition(list.p,condition.p,D3D12_RESOURCE_STATE_PREDICATION,D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(condition.p,offset,replacement_source.p,0,8);
        transition(list.p,condition.p,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_PREDICATION);
    }
    list->CopyBufferRegion(destination.p,25,source.p,5,11);
    if(recapture) list->SetPredication(condition.p,offset,op);
    list->CopyBufferRegion(destination.p,57,source.p,9,13);
    list->SetPredication(nullptr,0,D3D12_PREDICATION_OP_EQUAL_ZERO);
    list->CopyBufferRegion(destination.p,91,source.p,1,17);
    transition(list.p,destination.p,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(list.p,condition.p,D3D12_RESOURCE_STATE_PREDICATION,D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.p,0,destination.p,0,destination_bytes);
    list->CopyBufferRegion(readback.p,destination_bytes,condition.p,0,condition_bytes);
    list->ResolveQueryData(occlusion.p,query_type,0,1,readback.p,176);
    CHECK(finish(list.p,queue.p,fence.p));
    void* mapped=nullptr; const D3D12_RANGE range{0,total_bytes};
    CHECK(SUCCEEDED(readback->Map(0,&range,&mapped)) && mapped);
    std::array<unsigned char,total_bytes> bytes{}; std::memcpy(bytes.data(),mapped,bytes.size());
    const D3D12_RANGE none{}; readback->Unmap(0,&none);
    UINT64 zero=UINT64_MAX, overwritten=0;
    const UINT64 replacement=overwrite ? high_word : 0;
    std::memcpy(&zero,bytes.data()+176,8);
    std::memcpy(&overwritten,bytes.data()+destination_bytes+offset,8);
    if(zero!=0 || overwritten!=replacement) {
        std::fprintf(stderr,"query evidence type %u op %u offset %llu overwrite %d recapture %d: occlusion=%llu predicate=%llu\n",
            unsigned(query_type),unsigned(op),offset,int(overwrite),int(recapture),zero,overwritten);
        return false;
    }
    std::memcpy(guard.data()+offset,&replacement,8);
    CHECK(std::memcmp(bytes.data()+destination_bytes,guard.data(),guard.size())==0);
    const bool initially_runs=op==D3D12_PREDICATION_OP_NOT_EQUAL_ZERO;
    if(initially_runs) {
        std::memcpy(expected.data()+3,input.data()+2,7);
        std::memcpy(expected.data()+25,input.data()+5,11);
    }
    if((overwrite && recapture) ? !initially_runs : initially_runs) std::memcpy(expected.data()+57,input.data()+9,13);
    std::memcpy(expected.data()+91,input.data()+1,17);
    actual.assign(bytes.begin(),bytes.begin()+destination_bytes);
    for(SIZE_T i=0;i<expected.size();++i) if(actual[i]!=expected[i]) {
        std::fprintf(stderr,"query-copy type %u op %u offset %llu overwrite %d recapture %d byte %zu got %02x expected %02x\n",
            unsigned(query_type),unsigned(op),offset,int(overwrite),int(recapture),i,unsigned(actual[i]),unsigned(expected[i]));
        return false;
    }
    return true;
}

}
int main(int argc,char** argv) {
    const bool native_only=argc==2&&std::strcmp(argv[1],"--warp-only")==0;
    if(argc!=1&&!native_only) return 2;
    if(hostwine::host_is_wine()) return 77;  // WARP is the reference for every case
    vktest::Device vk;
    Ref<IDXGIFactory4> factory; Ref<IDXGIAdapter> adapter; Ref<ID3D12Device> host,ours;
    if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) ||
       FAILED(D3D12CreateDevice(adapter.p,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&host.p)))) return 1;
    if(!native_only) {
        const auto status = vk.initialize();
        if (status == vktest::DeviceStatus::unavailable) {
            std::puts("predicated buffer copies: skipped (no Vulkan device)");
            return 77;
        }
        if (status != vktest::DeviceStatus::ready) return 1;
        if(!vk.conditional_rendering || !vk.storage_buffer_8bit) {
            std::puts("predicated buffer copies: skipped (requires conditional rendering and storageBuffer8BitAccess)");
            return 0;
        }
        rt::d3d12::VulkanDeviceInfo info{}; info.get_instance_proc=vk.instance_proc; info.instance=vk.instance;
        info.physical=vk.physical; info.device=vk.device; info.queue=vk.queue; info.queue_family=vk.family;
        info.conditional_rendering=vk.conditional_rendering;
        info.storage_buffer_8bit=vk.storage_buffer_8bit;
        info.occlusion_query_precise=vk.occlusion_query_precise;
        info.pipeline_statistics_query=vk.pipeline_statistics_query;
        if(FAILED(rt::d3d12::create_vulkan_device(info,IID_PPV_ARGS(&ours.p))))return 1;
    }
    const Case cases[]={{1,1,0,0,1},{2,3,1,2,1},{7,11,1,3,5},{13,17,3,1,7},{33,35,0,3,32},
        {257,259,1,3,256},{4099,4103,3,7,4096},{4099,4103,4098,4102,1},{13,17,13,17,0}};
    unsigned count=0;
    for(const auto& c:cases)for(UINT64 value:{UINT64_C(0),UINT64_C(1),UINT64_C(1)<<32,(UINT64_C(1)<<32)|7,UINT64_MAX})
        for(auto op:{D3D12_PREDICATION_OP_EQUAL_ZERO,D3D12_PREDICATION_OP_NOT_EQUAL_ZERO})for(bool enabled:{false,true}) {
            std::vector<unsigned char> expected,actual;
            if(!run(host.p,c,value,op,enabled,expected)) {std::fprintf(stderr,"WARP reference failed\n");return 1;}
            if(!native_only&&(!run(ours.p,c,value,op,enabled,actual)||actual!=expected)){std::fprintf(stderr,"Vulkan comparison failed\n");return 1;}
            ++count;
        }
    unsigned query_cases=0, recapture_controls=0;
    for(auto type:{D3D12_QUERY_TYPE_OCCLUSION,D3D12_QUERY_TYPE_BINARY_OCCLUSION})
        for(auto op:{D3D12_PREDICATION_OP_EQUAL_ZERO,D3D12_PREDICATION_OP_NOT_EQUAL_ZERO})
            for(UINT64 offset:{UINT64_C(0),UINT64_C(16)}) {
                std::vector<unsigned char> without_recapture;
                // Baseline on both operators; overwrite controls only where the predicate permits the write.
                const unsigned variants=op==D3D12_PREDICATION_OP_NOT_EQUAL_ZERO ? 3 : 1;
                for(unsigned variant=0;variant<variants;++variant) {
                    const bool overwrite=variant!=0, recapture=variant==2;
                    std::vector<unsigned char> expected,actual;
                    if(!run_query_copy(host.p,type,op,offset,overwrite,recapture,expected)) {
                        std::fprintf(stderr,"WARP query-copy oracle failed\n");return 1;
                    }
                    if(!native_only && (!run_query_copy(ours.p,type,op,offset,overwrite,recapture,actual) || actual!=expected)) {
                        std::fprintf(stderr,"Vulkan query-copy comparison failed\n");return 1;
                    }
                    if(variant==1) without_recapture=expected;
                    if(recapture) {
                        unsigned changed=0;
                        for(SIZE_T i=0;i<expected.size();++i) if(expected[i]!=without_recapture[i]) {
                            if(i<57 || i>=70)return 1;
                            ++changed;
                        }
                        if(changed!=13)return 1;
                        ++recapture_controls;
                    }
                    ++query_cases;
                }
            }
    // D3D-162: capture and byte-copy helpers are invisible in every statistics counter, whether executed or skipped.
    if (!native_only && vk.pipeline_statistics_query) {
        for (UINT64 value : {UINT64_C(0),UINT64_C(1)<<32})
            for (auto op : {D3D12_PREDICATION_OP_EQUAL_ZERO,D3D12_PREDICATION_OP_NOT_EQUAL_ZERO}) {
                std::vector<unsigned char> expected,actual;
                if (!run(host.p,cases[3],value,op,true,expected,true) ||
                    !run(ours.p,cases[3],value,op,true,actual,true) || actual!=expected) return 1;
            }
    }
    if(!native_only&&vk.validation_errors())return 1;
    std::printf("query-to-predicated-copy: %u exact cases, %u recapture controls; GPU overwrite and predicate guards exact\n",query_cases,recapture_controls);
    std::printf("predicated buffer copies: %u exact cases (%s)\n",count,native_only?"WARP + literal byte oracle":"Vulkan + WARP + literal byte oracle");
}
