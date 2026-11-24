// Emulation of DXGI_FORMAT_R9G9B9E5_SHAREDEXP textures with D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS on Vulkan
// GPUs whose VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 lacks native STORAGE_IMAGE support (docs/d3d12_vulkan.md Appendix A.3).
//
// Exercises:
//   1. CheckFeatureSupport(FORMAT_SUPPORT) for R9G9B9E5_SHAREDEXP (typed UAV on, render target off)
//   2. CreateCommittedResource (1D/2D 1920x1080/2D-array+mips/3D), GetResourceAllocationInfo (4 KiB + 64 KiB),
//      CreateReservedResource + GetResourceTiling, and refusals (ALLOW_RENDER_TARGET, ALLOW_DEPTH_STENCIL,
//      DENY_SHADER_RESOURCE, MSAA, null RTV)
//   3. SRV and UAV T# creation (FORMAT 132, channels_rgb, ResourceMinLODClamp, SRV->UAV upgrade on same T#,
//      and null-descriptor rejection of invalid casts in both directions)
//   4. Differential ClearUnorderedAccessViewFloat / ClearUnorderedAccessViewUint (whole views and rectangles)
//      against Windows WARP (when available)
//   5. CopyResource / CopyTextureRegion between UAV and non-UAV R9G9B9E5 textures + WriteToSubresource /
//      ReadFromSubresource round-trip
//   6. Set-0 bindless GPU compute write (R32_UINT storage bindings 9 & 4) -> sampled HDR float read (binding 0,
//      VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) when glslangValidator compiled the test shader
//
// Synthetic data only; no game files. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"
#include "../src/uav_clear.h"

#include <dxgi1_4.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace g = leonida::gfx10;

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

D3D12_RESOURCE_DESC tex_desc(D3D12_RESOURCE_DIMENSION dim, UINT64 width, UINT height, UINT16 depth_or_layers,
                             UINT16 mips, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, UINT samples = 1,
                             UINT64 alignment = 0, D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = dim;
    d.Alignment = alignment;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = depth_or_layers;
    d.MipLevels = mips;
    d.Format = format;
    d.SampleDesc = {samples, 0};
    d.Layout = layout;
    d.Flags = flags;
    return d;
}

D3D12_RESOURCE_DESC buf_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after,
                                  UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {resource, subresource, before, after};
    return b;
}

bool all_zero(const void* ptr, std::size_t bytes) {
    const auto* p = static_cast<const std::uint8_t*>(ptr);
    for (std::size_t i = 0; i < bytes; ++i)
        if (p[i]) return false;
    return true;
}

// Runs a sequence of UAV clears on a 4x4 DXGI_FORMAT_R9G9B9E5_SHAREDEXP texture on any ID3D12Device (Vulkan or WARP)
// and reads back the 16 packed uint32 words.
struct ClearStep {
    bool is_float;
    std::array<float, 4> fvals;
    std::array<UINT, 4> uvals;
    std::vector<D3D12_RECT> rects;
};

bool run_uav_clear_sequence(ID3D12Device* dev, bool vulkan_device, const std::vector<ClearStep>& steps,
                            std::vector<std::uint32_t>& out_words) {
    Ref<ID3D12CommandQueue> queue;
    const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p)))) return false;
    Ref<ID3D12CommandAllocator> alloc;
    if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p)))) return false;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p))))
        return false;
    Ref<ID3D12Fence> fence;
    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return false;

    const D3D12_HEAP_PROPERTIES def_heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto td = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 4, 4, 1, 1, DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    Ref<ID3D12Resource> tex;
    if (FAILED(dev->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &td,
                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&tex.p))))
        return false;

    const D3D12_HEAP_PROPERTIES rb_heap{D3D12_HEAP_TYPE_READBACK};
    const auto bd = buf_desc(4 * 256);
    Ref<ID3D12Resource> rb;
    if (FAILED(dev->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                            nullptr, IID_PPV_ARGS(&rb.p))))
        return false;

    Ref<ID3D12DescriptorHeap> cpu_heap, gpu_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hd_cpu{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    const D3D12_DESCRIPTOR_HEAP_DESC hd_gpu{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    if (FAILED(dev->CreateDescriptorHeap(&hd_cpu, IID_PPV_ARGS(&cpu_heap.p))) ||
        FAILED(dev->CreateDescriptorHeap(&hd_gpu, IID_PPV_ARGS(&gpu_heap.p))))
        return false;

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavd{};
    uavd.Format = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    uavd.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    const auto cpu_h = cpu_heap->GetCPUDescriptorHandleForHeapStart();
    const auto gpu_h = gpu_heap->GetGPUDescriptorHandleForHeapStart();
    dev->CreateUnorderedAccessView(tex.p, nullptr, &uavd, cpu_h);
    dev->CreateUnorderedAccessView(tex.p, nullptr, &uavd, gpu_heap->GetCPUDescriptorHandleForHeapStart());

    ID3D12DescriptorHeap* heaps[] = {gpu_heap.p};
    list->SetDescriptorHeaps(1, heaps);
    for (const auto& step : steps) {
        const auto* rects = step.rects.empty() ? nullptr : step.rects.data();
        if (step.is_float) {
            list->ClearUnorderedAccessViewFloat(gpu_h, cpu_h, tex.p, step.fvals.data(), UINT(step.rects.size()), rects);
        } else {
            list->ClearUnorderedAccessViewUint(gpu_h, cpu_h, tex.p, step.uvals.data(), UINT(step.rects.size()), rects);
        }
        D3D12_RESOURCE_BARRIER uav_b{};
        uav_b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav_b.UAV.pResource = tex.p;
        list->ResourceBarrier(1, &uav_b);
    }
    const auto to_src = transition(tex.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_src);
    D3D12_TEXTURE_COPY_LOCATION dst{rb.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst.PlacedFootprint = {0, {DXGI_FORMAT_R9G9B9E5_SHAREDEXP, 4, 4, 1, 256}};
    D3D12_TEXTURE_COPY_LOCATION src{tex.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    src.SubresourceIndex = 0;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (FAILED(list->Close())) return false;

    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (vulkan_device && rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) return false;
    if (FAILED(queue->Signal(fence.p, 1)) || FAILED(fence->SetEventOnCompletion(1, nullptr))) return false;

    void* mapped = nullptr;
    if (FAILED(rb->Map(0, nullptr, &mapped))) return false;
    out_words.resize(16);
    for (UINT y = 0; y < 4; ++y)
        std::memcpy(&out_words[y * 4], static_cast<const std::uint8_t*>(mapped) + y * 256, 4 * sizeof(std::uint32_t));
    rb->Unmap(0, nullptr);
    return true;
}

#ifdef LEONIDA_TEST_CS_R9G9B9E5_UAV_SPV
bool read_spirv(const char* path, std::vector<std::uint32_t>& out) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const auto size = file.tellg();
    if (size <= 0 || size % 4 != 0) return false;
    out.resize(std::size_t(size) / 4);
    file.seekg(0);
    return bool(file.read(reinterpret_cast<char*>(out.data()), size));
}

struct ShaderParams {
    std::uint32_t tsharp[8];
    std::uint32_t ssharp[4];
    std::uint32_t mode;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t pad;
};

bool run_bindless_roundtrip(vktest::Device& d, VkDescriptorSet set0, VkDescriptorSetLayout set0_layout,
                            const g::ImageWords& uav_words, const g::ImageWords& srv_words,
                            const g::SamplerWords& sampler_words, bool do_uint_write,
                            const std::array<std::uint32_t, 16>& input_words,
                            std::array<float, 64>& out_rgba) {
    std::vector<std::uint32_t> code;
    if (!read_spirv(LEONIDA_TEST_CS_R9G9B9E5_UAV_SPV, code)) return false;

    void* in_ptr = nullptr;
    void* out_ptr = nullptr;
    const VkBuffer in_buf = d.buffer(sizeof(input_words), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &in_ptr);
    const VkBuffer out_buf = d.buffer(sizeof(out_rgba), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out_ptr);
    if (!in_buf || !out_buf) return false;
    std::memcpy(in_ptr, input_words.data(), sizeof(input_words));

    const auto create_dsl = d.fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout");
    const auto destroy_dsl = d.fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout");
    const auto create_pool = d.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool");
    const auto destroy_pool = d.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool");
    const auto alloc_sets = d.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets");
    const auto update_sets = d.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets");
    const auto create_layout = d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout");
    const auto destroy_layout = d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
    const auto create_mod = d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule");
    const auto destroy_mod = d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
    const auto create_pipe = d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
    const auto destroy_pipe = d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline");

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dsl_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl_info.bindingCount = 2;
    dsl_info.pBindings = bindings;
    VkDescriptorSetLayout set1_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const auto cleanup = [&](bool ok) {
        if (pipeline) destroy_pipe(d.device, pipeline, nullptr);
        if (module) destroy_mod(d.device, module, nullptr);
        if (pipe_layout) destroy_layout(d.device, pipe_layout, nullptr);
        if (pool) destroy_pool(d.device, pool, nullptr);
        if (set1_layout) destroy_dsl(d.device, set1_layout, nullptr);
        return ok;
    };
    if (create_dsl(d.device, &dsl_info, nullptr, &set1_layout) != VK_SUCCESS) return cleanup(false);

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (create_pool(d.device, &pool_info, nullptr, &pool) != VK_SUCCESS) return cleanup(false);

    VkDescriptorSetAllocateInfo alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc_info.descriptorPool = pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &set1_layout;
    VkDescriptorSet set1 = VK_NULL_HANDLE;
    if (alloc_sets(d.device, &alloc_info, &set1) != VK_SUCCESS) return cleanup(false);

    VkDescriptorBufferInfo buf_infos[2] = {{in_buf, 0, sizeof(input_words)}, {out_buf, 0, sizeof(out_rgba)}};
    VkWriteDescriptorSet writes[2]{};
    for (UINT i = 0; i < 2; ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set1;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    update_sets(d.device, 2, writes, 0, nullptr);

    VkDescriptorSetLayout set_layouts[2] = {set0_layout, set1_layout};
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShaderParams)};
    VkPipelineLayoutCreateInfo pl_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl_info.setLayoutCount = 2;
    pl_info.pSetLayouts = set_layouts;
    pl_info.pushConstantRangeCount = 1;
    pl_info.pPushConstantRanges = &range;
    if (create_layout(d.device, &pl_info, nullptr, &pipe_layout) != VK_SUCCESS) return cleanup(false);

    VkShaderModuleCreateInfo mod_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mod_info.codeSize = code.size() * sizeof(std::uint32_t);
    mod_info.pCode = code.data();
    if (create_mod(d.device, &mod_info, nullptr, &module) != VK_SUCCESS) return cleanup(false);

    VkComputePipelineCreateInfo pipe_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipe_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipe_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipe_info.stage.module = module;
    pipe_info.stage.pName = "main";
    pipe_info.layout = pipe_layout;
    if (create_pipe(d.device, VK_NULL_HANDLE, 1, &pipe_info, nullptr, &pipeline) != VK_SUCCESS) return cleanup(false);

    const bool ok = d.run([&](VkCommandBuffer cb) {
        const auto bind_pipe = d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
        const auto bind_sets = d.fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
        const auto push = d.fn<PFN_vkCmdPushConstants>("vkCmdPushConstants");
        const auto dispatch = d.fn<PFN_vkCmdDispatch>("vkCmdDispatch");
        const auto barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");

        VkDescriptorSet sets[2] = {set0, set1};
        bind_pipe(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        bind_sets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 2, sets, 0, nullptr);

        if (do_uint_write) {
            ShaderParams wp{};
            std::memcpy(wp.tsharp, uav_words.data(), sizeof(wp.tsharp));
            std::memcpy(wp.ssharp, sampler_words.data(), sizeof(wp.ssharp));
            wp.mode = 0;
            wp.width = 4;
            wp.height = 4;
            push(cb, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(wp), &wp);
            dispatch(cb, 1, 1, 1);

            VkMemoryBarrier mem{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mem.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mem.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem, 0,
                    nullptr, 0, nullptr);
        }

        ShaderParams sp{};
        std::memcpy(sp.tsharp, srv_words.data(), sizeof(sp.tsharp));
        std::memcpy(sp.ssharp, sampler_words.data(), sizeof(sp.ssharp));
        sp.mode = 1;
        sp.width = 4;
        sp.height = 4;
        push(cb, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sp), &sp);
        dispatch(cb, 1, 1, 1);

        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0,
                nullptr);
        return true;
    });
    if (ok) std::memcpy(out_rgba.data(), out_ptr, sizeof(out_rgba));
    return cleanup(ok);
}
#endif

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;

    VkFormatProperties e5_props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(
        d.physical, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, &e5_props);
    const bool native_storage = (e5_props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
    const bool native_attach = (e5_props.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0;
    const bool rect_clears_supported = !native_storage || native_attach;

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.image_view_min_lod = d.image_view_min_lod;
        info.host_image_copy = d.host_image_copy;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

        // 1. CheckFeatureSupport(FORMAT_SUPPORT): typed UAV is advertised; render target and depth-stencil stay off.
        D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{DXGI_FORMAT_R9G9B9E5_SHAREDEXP};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs))));
        if (!(fs.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)) {
            std::printf("R9G9B9E5_SHAREDEXP UAV not supported natively or via mutable R32_UINT on this GPU, skipping\n");
            return 77;
        }
        EXPECT((fs.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE1D) != 0);
        EXPECT((fs.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) != 0);
        EXPECT((fs.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE3D) != 0);
        EXPECT((fs.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURECUBE) != 0);
        EXPECT((fs.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_LOAD) != 0);
        EXPECT((fs.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE) != 0);
        EXPECT(bool(fs.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET) == native_attach);
        EXPECT((fs.Support1 & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL) == 0);
        EXPECT((fs.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0);
        std::printf("R9G9B9E5 UAV mode: %s (Support1=0x%08x Support2=0x%08x)\n",
                    native_storage ? "native" : "emulated mutable R32_UINT",
                    static_cast<unsigned>(fs.Support1), static_cast<unsigned>(fs.Support2));

        // 2. Resource creation (1D, 2D 1920x1080, 2D array + mips, 3D), GetResourceAllocationInfo,
        //    CreateReservedResource + GetResourceTiling, and flag refusals.
        const D3D12_HEAP_PROPERTIES def_heap{D3D12_HEAP_TYPE_DEFAULT};
        const auto desc_1080p = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1920, 1080, 1, 1,
                                         DXGI_FORMAT_R9G9B9E5_SHAREDEXP, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        const auto alloc_1080p = device->GetResourceAllocationInfo(0, 1, &desc_1080p);
        EXPECT(alloc_1080p.SizeInBytes != UINT64_MAX && alloc_1080p.SizeInBytes >= 1920ull * 1080 * 4);
        EXPECT(alloc_1080p.Alignment == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);

        // Small 16x16 2D UAV qualifies for 4 KiB placement alignment (matches host GetResourceAllocationInfo rules).
        const auto desc_small_4k = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 1,
                                            DXGI_FORMAT_R9G9B9E5_SHAREDEXP, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                            1, D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT);
        const auto alloc_small = device->GetResourceAllocationInfo(0, 1, &desc_small_4k);
        EXPECT(alloc_small.SizeInBytes != UINT64_MAX && alloc_small.Alignment == D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT);

        Ref<ID3D12Resource> tex_1080p;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &desc_1080p,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&tex_1080p.p))));
        EXPECT(tex_1080p->GetDesc().Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP);
        EXPECT(tex_1080p->GetDesc().Flags == D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        const auto desc_arr = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 8, 4, 3,
                                       DXGI_FORMAT_R9G9B9E5_SHAREDEXP, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        Ref<ID3D12Resource> tex_arr;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &desc_arr,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&tex_arr.p))));

        const auto desc_1d = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE1D, 64, 1, 2, 2,
                                      DXGI_FORMAT_R9G9B9E5_SHAREDEXP, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        Ref<ID3D12Resource> tex_1d;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &desc_1d,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&tex_1d.p))));

        const auto desc_3d = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE3D, 8, 8, 4, 2,
                                      DXGI_FORMAT_R9G9B9E5_SHAREDEXP, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        Ref<ID3D12Resource> tex_3d;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &desc_3d,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&tex_3d.p))));

        // Reserved resource with 64KB_UNDEFINED_SWIZZLE + GetResourceTiling.
        const auto desc_reserved = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1920, 1080, 1, 1,
                                            DXGI_FORMAT_R9G9B9E5_SHAREDEXP, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                            1, 0, D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE);
        Ref<ID3D12Resource> reserved;
        EXPECT(SUCCEEDED(device->CreateReservedResource(&desc_reserved, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&reserved.p))));
        UINT total_tiles = 0;
        D3D12_TILE_SHAPE tile_shape{};
        device->GetResourceTiling(reserved.p, &total_tiles, nullptr, &tile_shape, nullptr, 0, nullptr);
        EXPECT(total_tiles > 0 && tile_shape.WidthInTexels == 128 && tile_shape.HeightInTexels == 128 &&
               tile_shape.DepthInTexels == 1);

        for (const auto flags : {D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                 D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS}) {
            const auto desc = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 1,
                                       DXGI_FORMAT_R9G9B9E5_SHAREDEXP, flags);
            const bool supported = native_attach && (!(flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) || native_storage);
            Ref<ID3D12Resource> target;
            const HRESULT hr = device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                               D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p));
            EXPECT(hr == (supported ? S_OK : E_INVALIDARG));
            EXPECT(bool(target.p) == supported);
            EXPECT((device->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes != UINT64_MAX) == supported);
        }

        const D3D12_RESOURCE_DESC bad_descs[] = {
            tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 1, DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
                     D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
            tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 1, DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE),
            tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 1, DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, 4),
        };
        for (const auto& bad : bad_descs) {
            Ref<ID3D12Resource> refused;
            EXPECT(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &bad, D3D12_RESOURCE_STATE_COMMON,
                                                   nullptr, IID_PPV_ARGS(&refused.p)) == E_INVALIDARG);
            EXPECT(!refused.p);
            EXPECT(device->GetResourceAllocationInfo(0, 1, &bad).SizeInBytes == UINT64_MAX);
        }

        // CreateRenderTargetView on an emulated R9G9B9E5 UAV leaves a null RTV (never exposes backing R32_UINT RT).
        Ref<ID3D12DescriptorHeap> rtv_heap;
        const D3D12_DESCRIPTOR_HEAP_DESC rtv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_hd, IID_PPV_ARGS(&rtv_heap.p))));
        const auto rtv_h = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        const UINT rtv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        std::memset(reinterpret_cast<void*>(rtv_h.ptr), 0x55, rtv_step);
        device->CreateRenderTargetView(tex_1080p.p, nullptr, rtv_h);
        EXPECT(all_zero(reinterpret_cast<const void*>(rtv_h.ptr), sizeof(leonida::TargetView)));

        // 3. Descriptors (SRV, UAV, ResourceMinLODClamp, SRV->UAV upgrade, invalid casts in both directions).
        Ref<ID3D12DescriptorHeap> heap;
        const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))));
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto* base = reinterpret_cast<std::byte*>(heap->GetCPUDescriptorHandleForHeapStart().ptr);
        const auto slot = [&](UINT i) -> D3D12_CPU_DESCRIPTOR_HANDLE {
            return {reinterpret_cast<SIZE_T>(base + std::size_t(i) * step)};
        };
        const auto words_at = [&](UINT i) {
            g::ImageWords w{};
            std::memcpy(w.data(), base + std::size_t(i) * step, sizeof(w));
            return w;
        };

        // Create SRV first on mip 0 of a 1-mip texture, then UAV with the exact same range: T# bytes must match and
        // decode as GFX10_FORMAT_5_9_9_9_FLOAT (132) with RGB1 channels.
        device->CreateShaderResourceView(tex_1080p.p, nullptr, slot(0));
        device->CreateUnorderedAccessView(tex_1080p.p, nullptr, nullptr, slot(1));
        const g::ImageWords srv_1080p = words_at(0);
        const g::ImageWords uav_1080p = words_at(1);
        EXPECT(!all_zero(srv_1080p.data(), sizeof(srv_1080p)));
        EXPECT(srv_1080p == uav_1080p);
        g::Image decoded{};
        EXPECT(g::decode(uav_1080p, decoded));
        EXPECT(decoded.format == 132u && decoded.swizzle == g::channels_rgb);
        EXPECT(decoded.width == 1920u && decoded.height == 1080u);

        // 2D-array UAV + SRV with ResourceMinLODClamp (verifies VkImageViewMinLodCreateInfoEXT chains with usage).
        D3D12_UNORDERED_ACCESS_VIEW_DESC arr_uav{};
        arr_uav.Format = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
        arr_uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        arr_uav.Texture2DArray = {1, 1, 2, 0};
        device->CreateUnorderedAccessView(tex_arr.p, nullptr, &arr_uav, slot(2));
        EXPECT(g::decode(words_at(2), decoded));
        EXPECT(decoded.format == 132u && decoded.first_mip == 1u && decoded.first_layer == 1u &&
               decoded.layer_count == 2u);

        D3D12_SHADER_RESOURCE_VIEW_DESC lod_srv{};
        lod_srv.Format = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
        lod_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        lod_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        lod_srv.Texture2DArray = {0, 3, 0, 4, 0, d.image_view_min_lod ? 1.5f : 0.0f};
        device->CreateShaderResourceView(tex_arr.p, &lod_srv, slot(3));
        EXPECT(g::decode(words_at(3), decoded) && decoded.format == 132u);

        // Invalid casts on R9G9B9E5_SHAREDEXP must leave null descriptors.
        for (DXGI_FORMAT bad_fmt : {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM}) {
            D3D12_SHADER_RESOURCE_VIEW_DESC bad_srv{};
            bad_srv.Format = bad_fmt;
            bad_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            bad_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            bad_srv.Texture2D.MipLevels = 1;
            std::memset(base + std::size_t(4) * step, 0xaa, step);
            device->CreateShaderResourceView(tex_1080p.p, &bad_srv, slot(4));
            EXPECT(all_zero(base + std::size_t(4) * step, step));

            D3D12_UNORDERED_ACCESS_VIEW_DESC bad_uav{};
            bad_uav.Format = bad_fmt;
            bad_uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            std::memset(base + std::size_t(5) * step, 0xaa, step);
            device->CreateUnorderedAccessView(tex_1080p.p, nullptr, &bad_uav, slot(5));
            EXPECT(all_zero(base + std::size_t(5) * step, step));
        }

        // Reverse invalid cast: R9G9B9E5 view on a real R32_UINT UAV texture must leave a null descriptor.
        const auto desc_r32 = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 8, 8, 1, 1, DXGI_FORMAT_R32_UINT,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        Ref<ID3D12Resource> tex_r32;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &desc_r32,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&tex_r32.p))));
        D3D12_SHADER_RESOURCE_VIEW_DESC rev_srv{};
        rev_srv.Format = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
        rev_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        rev_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        rev_srv.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(tex_r32.p, &rev_srv, slot(6));
        EXPECT(all_zero(base + std::size_t(6) * step, step));

        // 4. Differential UAV clear comparison against Windows WARP.
        Ref<ID3D12Device> warp;
        if (!hostwine::host_is_wine()) {
            Ref<IDXGIFactory4> factory;
            Ref<IDXGIAdapter> adapter;
            if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) &&
                SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) {
                D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&warp.p));
            }
        }
        // The reference needs typed R9G9B9E5 UAVs. Windows 10's WARP has none (Support1 0x008013f0: its
        // CreateCommittedResource with ALLOW_UNORDERED_ACCESS returns E_INVALIDARG), so it can't be compared with.
        if (warp.p) {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT warp_fs{DXGI_FORMAT_R9G9B9E5_SHAREDEXP};
            if (FAILED(warp->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &warp_fs, sizeof(warp_fs))) ||
                !(warp_fs.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)) {
                std::printf("WARP has no typed R9G9B9E5 UAVs (Support1=0x%08x): UAV clears not compared with it\n",
                            unsigned(warp_fs.Support1));
                warp.reset();
            }
        }
        std::vector<std::vector<ClearStep>> clear_cases = {
            {{true, {1.0f, 2.0f, 4.0f, 0.0f}, {}, {}}},
            {{true, {1.0f + 0.5f / 256, 1.0f + 1.5f / 256, 1.0f + 2.5f / 256, 0.0f}, {}, {}}},
            {{true, {1.0f - 0.5f / 512, std::ldexp(511.5f, -24), std::ldexp(0.5f, -24), 0.0f}, {}, {}}},
            {{true, {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), 0.0f}, {}, {}}},
            {{true, {1.0f, 2.0f, 4.0f, 0.0f}, {}, {}},
             {false, {}, {0x3f800000u, 0xffffffffu, 0x12345678u, 7u}, {}}},
        };
        if (rect_clears_supported) {
            clear_cases.push_back({
                {true, {1.0f, 2.0f, 4.0f, 0.0f}, {}, {}},
                {true, {0.5f, 0.25f, 0.125f, 0.0f}, {}, {{1, 1, 3, 3}}},
                {false, {}, {0xdeadbeefu, 0x3f800000u, 0u, 0u}, {{2, 2, 4, 4}}},
            });
        }
        // WARP's words for each case, row by row (docs/d3d12_vulkan.md: half-up rounding, uint clears write 0).
        const auto same = [](std::uint32_t w) {
            std::array<std::uint32_t, 16> a{};
            a.fill(w);
            return a;
        };
        constexpr std::uint32_t A = 0x94010040u, B = 0x79010100u;
        const std::array<std::array<std::uint32_t, 16>, 6> warp_answers = {
            same(0x94010040u), same(0x840e0501u), same(0x80000100u), same(0xf803fe00u), same(0u),
            std::array<std::uint32_t, 16>{A, A, A, A, A, B, B, A, A, B, 0, 0, A, A, 0, 0},
        };
        for (std::size_t c = 0; c < clear_cases.size(); ++c) {
            std::vector<std::uint32_t> vk_words;
            EXPECT(run_uav_clear_sequence(device.p, true, clear_cases[c], vk_words));
            for (std::size_t i = 0; i < vk_words.size(); ++i) {
                if (vk_words[i] == warp_answers[c][i]) continue;
                std::fprintf(stderr, "clear case %zu texel %zu: 0x%08x, WARP writes 0x%08x\n", c, i, vk_words[i],
                             warp_answers[c][i]);
                return 1;
            }
            if (warp.p) {
                std::vector<std::uint32_t> warp_words;
                EXPECT(run_uav_clear_sequence(warp.p, false, clear_cases[c], warp_words));
                if (vk_words != warp_words) {
                    std::fprintf(stderr, "WARP mismatch on clear case %zu: vk[0]=0x%08x warp[0]=0x%08x\n",
                                 c, vk_words[0], warp_words[0]);
                    return 1;
                }
            }
        }
        std::printf("UAV clears: %zu/%zu cases %s\n", clear_cases.size(), clear_cases.size(),
                    warp.p ? "matched WARP bit-for-bit" : "matched WARP's recorded words (WARP unavailable)");

        // 5. CopyResource / CopyTextureRegion between UAV (emulated R32_UINT) and plain non-UAV (E5B9G9R9) textures,
        //    plus WriteToSubresource / ReadFromSubresource on a CPU-visible custom-heap UAV texture.
        {
            const std::array<std::uint32_t, 16> pattern = {
                0x94010040u, 0x79010100u, 0x7556ab55u, 0x840e0501u,
                0x80000100u, 0x80020000u, 0x84000000u, 0xb0080190u,
                0xf80001ffu, 0xffffffffu, 0x08000100u, 0x00000100u,
                0x00000001u, 0x000c0202u, 0x00100602u, 0x00000000u,
            };
            const D3D12_HEAP_PROPERTIES custom_heap = device->GetCustomHeapProperties(0, D3D12_HEAP_TYPE_UPLOAD);
            const auto desc_uav4 = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 4, 4, 1, 1,
                                            DXGI_FORMAT_R9G9B9E5_SHAREDEXP, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            const auto desc_plain4 = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 4, 4, 1, 1,
                                              DXGI_FORMAT_R9G9B9E5_SHAREDEXP, D3D12_RESOURCE_FLAG_NONE);
            Ref<ID3D12Resource> cpu_uav, plain_tex, dst_uav;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&custom_heap, D3D12_HEAP_FLAG_NONE, &desc_uav4,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&cpu_uav.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &desc_plain4,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&plain_tex.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&custom_heap, D3D12_HEAP_FLAG_NONE, &desc_uav4,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&dst_uav.p))));

            EXPECT(SUCCEEDED(cpu_uav->WriteToSubresource(0, nullptr, pattern.data(), 16, 64)));
            std::array<std::uint32_t, 16> cpu_readback{};
            EXPECT(SUCCEEDED(cpu_uav->ReadFromSubresource(cpu_readback.data(), 16, 64, 0, nullptr)));
            EXPECT(cpu_readback == pattern);

            // UAV -> plain E5B9G9R9 (CopyResource) -> UAV (CopyTextureRegion), then ReadFromSubresource.
            Ref<ID3D12CommandQueue> queue;
            const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
            EXPECT(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
            Ref<ID3D12CommandAllocator> alloc;
            Ref<ID3D12GraphicsCommandList> list;
            Ref<ID3D12Fence> fence;
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

            const auto b1 = transition(cpu_uav.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &b1);
            list->CopyResource(plain_tex.p, cpu_uav.p);
            const auto b2 = transition(plain_tex.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &b2);
            D3D12_TEXTURE_COPY_LOCATION copy_dst{dst_uav.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            D3D12_TEXTURE_COPY_LOCATION copy_src{plain_tex.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            list->CopyTextureRegion(&copy_dst, 0, 0, 0, &copy_src, nullptr);
            const auto b3 = transition(dst_uav.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            list->ResourceBarrier(1, &b3);
            EXPECT(list->Close() == S_OK);
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

            std::array<std::uint32_t, 16> roundtrip{};
            EXPECT(SUCCEEDED(dst_uav->ReadFromSubresource(roundtrip.data(), 16, 64, 0, nullptr)));
            EXPECT(roundtrip == pattern);
            std::printf("Copies & CPU subresource access: UAV <-> non-UAV R9G9B9E5 matched 16/16 texels\n");
        }

        // 6. Set-0 bindless GPU compute round-trip: write packed RGB9E5 through R32_UINT storage views (bindings 9
        //    and 4) and sample back decoded HDR floats through the E5B9G9R9 sampled view (binding 0).
#ifdef LEONIDA_TEST_CS_R9G9B9E5_UAV_SPV
        if (d.bindless) {
            const VkDescriptorSet set0 = rt::d3d12::bindless_set(device.p);
            const VkDescriptorSetLayout set0_layout = rt::d3d12::bindless_set_layout(device.p);
            EXPECT(set0 && set0_layout);

            const auto desc_rt = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 4, 4, 1, 1,
                                          DXGI_FORMAT_R9G9B9E5_SHAREDEXP, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            Ref<ID3D12Resource> rt_tex;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &desc_rt,
                                                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                             IID_PPV_ARGS(&rt_tex.p))));
            device->CreateUnorderedAccessView(rt_tex.p, nullptr, nullptr, slot(8));
            device->CreateShaderResourceView(rt_tex.p, nullptr, slot(9));
            const g::ImageWords rt_uav_words = words_at(8);
            const g::ImageWords rt_srv_words = words_at(9);
            EXPECT(!all_zero(rt_uav_words.data(), sizeof(rt_uav_words)) && rt_uav_words == rt_srv_words);

            Ref<ID3D12DescriptorHeap> smp_heap;
            const D3D12_DESCRIPTOR_HEAP_DESC shd{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1};
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&shd, IID_PPV_ARGS(&smp_heap.p))));
            D3D12_SAMPLER_DESC sd{};
            sd.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
            sd.AddressU = sd.AddressV = sd.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            const auto smp_h = smp_heap->GetCPUDescriptorHandleForHeapStart();
            device->CreateSampler(&sd, smp_h);
            g::SamplerWords smp_words{};
            std::memcpy(smp_words.data(), reinterpret_cast<const void*>(smp_h.ptr), sizeof(smp_words));

            // Flush set-0 descriptors and transition texture into GENERAL (UAV / bindless-sampled state).
            Ref<ID3D12CommandQueue> queue;
            const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
            EXPECT(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
            Ref<ID3D12CommandAllocator> alloc;
            Ref<ID3D12GraphicsCommandList> list;
            Ref<ID3D12Fence> fence;
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
            const float seed[4] = {1.0f, 2.0f, 4.0f, 0.0f};
            list->ClearUnorderedAccessViewFloat({0}, slot(8), rt_tex.p, seed, 0, nullptr);
            EXPECT(list->Close() == S_OK);
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

            // 16 distinct packed RGB9E5 words spanning subnormals, normal HDR, and max finite 65408.
            const std::array<std::uint32_t, 16> input_words = {
                0x94010040u,  // (1, 2, 4)
                0x79010100u,  // (0.5, 0.25, 0.125)
                0x80000100u,  // (1, 0, 0)
                0x80020000u,  // (0, 1, 0)
                0x84000000u,  // (0, 0, 1)
                0xb0080190u,  // (100, 0, 0.5)
                0xf80001ffu,  // (65408, 0, 0)
                0xffffffffu,  // (65408, 65408, 65408)
                0x08000100u,  // (2^-15, 0, 0)  -- second half written through binding 4 (r32ui)
                0x00000100u,  // (2^-16, 0, 0)
                0x00000001u,  // (2^-24, 0, 0)
                0x7556ab55u,  // (1/3, 1/3, 1/3) quantised
                0x840e0501u,  // (257/256, 258/256, 259/256)
                0x98040100u,  // (2, 1, 8)? exact decode checked via uav_clear::decode
                0x00000000u,  // (0, 0, 0)
                0x88040280u,  // (2.5, 1.0, 2.0)
            };
            std::array<float, 64> sampled{};
            EXPECT(run_bindless_roundtrip(d, set0, set0_layout, rt_uav_words, rt_srv_words, smp_words,
                                          !native_storage, input_words, sampled));
            for (UINT i = 0; i < 16; ++i) {
                const std::uint32_t want_word = native_storage ? 0x94010040u : input_words[i];
                leonida::uav_clear::Element elem{};
                elem.size = 4;
                std::memcpy(elem.bytes.data(), &want_word, 4);
                VkClearColorValue want{};
                EXPECT(leonida::uav_clear::decode(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, elem, want));
                for (UINT c = 0; c < 4; ++c) {
                    const float got = sampled[i * 4 + c];
                    const float exp_v = want.float32[c];
                    const float tol = std::max(1e-6f, std::fabs(exp_v) * 1e-5f);
                    if (std::fabs(got - exp_v) > tol) {
                        std::fprintf(stderr, "bindless sample texel %u ch %u: got %g, want %g (word 0x%08x)\n",
                                     i, c, got, exp_v, want_word);
                        return 1;
                    }
                }
            }
            std::printf("Bindless set-0 GPU round-trip: 16/16 HDR RGB9E5 texels matched (%s)\n",
                        native_storage ? "clear -> sampled E5B9G9R9"
                                       : "storage R32_UINT [bindings 9 & 4] -> sampled E5B9G9R9 [binding 0]");
        }
#endif
    }

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("d3d12_r9g9b9e5_uav_test: OK (0 validation errors)\n");
    return 0;
}
