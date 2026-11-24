// Descriptors through the Xbox-ABI proxy over the Vulkan device.
//   - CopyDescriptors and CopyDescriptorsSimple, checked on the bytes of the heap slots.
//   - Texture views and samplers (milestone M2, the device's part): CreateShaderResourceView,
//     CreateUnorderedAccessView and CreateSampler write a T# or an S# into the heap slot and put the view or the
//     sampler into the device's bindless set 0. The slots are decoded and checked. Then a compute shader samples
//     an uploaded 8x8 texture by those bytes: it looks the T# and the S# up in the descriptor hash table, as a
//     translated shader does. It checks the view's own slot, a copied slot, a slot that must give the fallback
//     image, and a packed RGB9E5 view with distinct channels and varying shared exponents. The sampler wraps
//     in U and the shader samples one tile to the right, so a missed S# would show as well (the fallback
//     sampler clamps).
//   - Default and mip-1 volume SRVs: sample a 4x4x4 grid through each view of one uploaded 8x8x4 texture.
//   - UINT border colours (#538, Vonsek's case): an R8G8B8A8_UINT texture sampled one tile to the right through
//     CreateSampler2 samplers that clamp to a UINT (0,0,0,0) and (1,1,1,1) border, and a float white border on
//     the UNORM texture. Every texel is the border: 0, 1 and 255 in each channel. Both white borders are built in
//     at the same S# type, so this also checks that they are two samplers, not one.
//   - B8G8R8X8 (D3D-87): a B8G8R8X8_UNORM and a B8G8R8A8_UNORM texture uploaded with the same bytes, the fourth
//     never 255. Their default views read the same red, green and blue; B8G8R8X8 reads alpha 1, B8G8R8A8 the
//     stored byte. Views between the two typeless families give a null descriptor, unless a castable-format list
//     names the view's format: then a B8G8R8X8 view of a B8G8R8A8 texture reads alpha 1, and a B8G8R8A8 view of a
//     B8G8R8X8 one the stored byte, as on WARP and AMD's D3D12 driver.
// Multisample SRVs: four masked triangles seed different values per sample and array layer. A compute shader reads
// them back using the T#s for the 2D, full-array and sliced-array views. The test builds its pipelines in Vulkan
// and binds the device's set (rt::d3d12::bindless_set). Without glslangValidator the GPU readbacks are skipped.
// No game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/gfx10_descriptors.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: 8-byte handle in RAX

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

void* slot(void* object, unsigned index) { return (*reinterpret_cast<void***>(object))[index]; }

// A root signature blob from the PC runtime's own serializer; empty when it has none for that version.
std::vector<std::uint8_t> serialize_root_signature(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc) {
    using SerializeFn = HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto fn = d3d12 ? reinterpret_cast<SerializeFn>(
                                reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeVersionedRootSignature")))
                          : nullptr;
    std::vector<std::uint8_t> out;
    ID3DBlob* blob = nullptr;
    if (fn && SUCCEEDED(fn(&desc, &blob, nullptr)) && blob) {
        const auto* bytes = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
        out.assign(bytes, bytes + blob->GetBufferSize());
        blob->Release();
    }
    return out;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

// The uploaded texture: a gradient no two texels of which are equal.
std::array<std::uint8_t, 4> texel(UINT x, UINT y) {
    return {std::uint8_t(x * 32), std::uint8_t(y * 32), std::uint8_t(128 + x + 8 * y), 255};
}

// The same texel in B, G, R memory order, with a fourth byte that is never 255: what a B8G8R8X8 texture and a
// B8G8R8A8 one with the same bytes are uploaded with. Through a view the first reads alpha 1, the second this byte.
std::array<std::uint8_t, 4> bgrx_texel(UINT x, UINT y) {
    const auto t = texel(x, y);
    return {t[2], t[1], t[0], std::uint8_t(3 * (y * 8 + x))};
}

#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
// What cs_sample.comp is pushed: a T#, an S#, and where in the output buffer its 64 texels go.
struct SampleKeys {
    std::uint32_t tsharp[8];
    std::uint32_t ssharp[4];
    std::uint32_t first;
    std::uint32_t mode;  // 0: normalized, 1: uint, 2: multisampled, 3: multisampled array, 4: 1D
    std::uint32_t layer;
};

VkShaderModule load_shader(vktest::Device& d, const char* path) {
    std::ifstream file(vktest::shader_fixture_path(path), std::ios::binary | std::ios::ate);
    if (!file) return VK_NULL_HANDLE;
    std::vector<std::uint32_t> code(std::size_t(file.tellg()) / 4);
    file.seekg(0);
    if (code.empty() || !file.read(reinterpret_cast<char*>(code.data()), std::streamsize(code.size() * 4)))
        return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = code.size() * 4;
    info.pCode = code.data();
    VkShaderModule module{};
    return d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &info, nullptr, &module) == VK_SUCCESS
        ? module : VK_NULL_HANDLE;
}

// Seed a 2D image and three layers of an array. Each triangle writes one sample through gl_SampleMask,
// so a device without sample-rate shading can still exercise reads of four independently known values.
bool seed_ms_images(vktest::Device& d, VkImage single, VkImage array) {
    const VkShaderModule vs = load_shader(d, LEONIDA_TEST_MSAA_VERT_SPV);
    const VkShaderModule fs = load_shader(d, LEONIDA_TEST_MSAA_FRAG_SPV);
    VkPipelineLayout layout{};
    VkPipeline pipeline{};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    bool ok = vs && fs && d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(
                              d.device, &layout_info, nullptr, &layout) == VK_SUCCESS;
    const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &format;
    VkPipelineShaderStageCreateInfo stages[] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main"}};
    VkPipelineVertexInputStateCreateInfo vertices{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    const VkRect2D area{{0, 0}, {4, 4}};
    const VkViewport viewport{0, 0, 4, 4, 0, 1};
    VkPipelineViewportStateCreateInfo viewports{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewports.viewportCount = viewports.scissorCount = 1;
    viewports.pViewports = &viewport;
    viewports.pScissors = &area;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_4_BIT;
    VkPipelineColorBlendAttachmentState color{};
    color.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &color;
    VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
    pi.stageCount = 2;
    pi.pStages = stages;
    pi.pVertexInputState = &vertices;
    pi.pInputAssemblyState = &assembly;
    pi.pViewportState = &viewports;
    pi.pRasterizationState = &raster;
    pi.pMultisampleState = &ms;
    pi.pColorBlendState = &blend;
    pi.layout = layout;
    ok = ok && d.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(
                   d.device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline) == VK_SUCCESS;
    VkImageView views[4]{};
    for (UINT k = 0; ok && k < std::size(views); ++k) {
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = k == 0 ? single : array;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = format;
        info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, k == 0 ? 0 : k - 1, 1};
        ok = d.fn<PFN_vkCreateImageView>("vkCreateImageView")(d.device, &info, nullptr, &views[k]) == VK_SUCCESS;
    }
    ok = ok && d.run([&](VkCommandBuffer cb) {
        d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        for (UINT k = 0; k < std::size(views); ++k) {
            VkRenderingAttachmentInfo attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            attachment.imageView = views[k];
            attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo draw{VK_STRUCTURE_TYPE_RENDERING_INFO};
            draw.renderArea = area;
            draw.layerCount = draw.colorAttachmentCount = 1;
            draw.pColorAttachments = &attachment;
            d.fn<PFN_vkCmdBeginRendering>("vkCmdBeginRendering")(cb, &draw);
            d.fn<PFN_vkCmdDraw>("vkCmdDraw")(cb, 3, 4, 0, 4 * k);
            d.fn<PFN_vkCmdEndRendering>("vkCmdEndRendering")(cb);
        }
        VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        ready.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(
            cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &ready, 0, nullptr, 0, nullptr);
        return true;
    });
    for (VkImageView view : views) d.fn<PFN_vkDestroyImageView>("vkDestroyImageView")(d.device, view, nullptr);
    d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(d.device, pipeline, nullptr);
    d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(d.device, layout, nullptr);
    d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(d.device, fs, nullptr);
    d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(d.device, vs, nullptr);
    return ok;
}

// Runs test/glsl/cs_sample.comp once for each of `keys` on `set0`, the device's bindless set, and copies the
// texels they wrote to `results` (64 RGBA8 texels each). Set 1 is the test's output buffer.
bool sample_through_set(vktest::Device& d, VkDescriptorSet set0, VkDescriptorSetLayout set0_layout,
                        std::span<const SampleKeys> keys, void* results, const char* shader = LEONIDA_TEST_CS_SAMPLE_SPV) {
    const VkDevice dev = d.device;

    constexpr VkDeviceSize texels = 64;
    void* mapped = nullptr;
    const VkBuffer output = d.buffer(keys.size() * texels * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &mapped);
    if (!output) return false;

    const VkShaderModule module = load_shader(d, shader);
    VkDescriptorSetLayout set1_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet set1 = VK_NULL_HANDLE;

    const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                                               nullptr};
    VkDescriptorSetLayoutCreateInfo set1_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set1_info.bindingCount = 1;
    set1_info.pBindings = &binding;
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    bool ok = module && d.fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(dev, &set1_info, nullptr,
                                                                                   &set1_layout) == VK_SUCCESS &&
              d.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(dev, &pool_info, nullptr, &pool) == VK_SUCCESS;
    if (ok) {
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = pool;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &set1_layout;
        ok = d.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(dev, &allocate, &set1) == VK_SUCCESS;
    }
    if (ok) {
        const VkDescriptorBufferInfo buffer_info{output, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set1;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer_info;
        d.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(dev, 1, &write, 0, nullptr);

        const VkDescriptorSetLayout layouts[2] = {set0_layout, set1_layout};
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SampleKeys)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 2;
        layout_info.pSetLayouts = layouts;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        ok = d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(dev, &layout_info, nullptr, &layout) == VK_SUCCESS;
        pipeline_info.layout = layout;
        ok = ok && d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(dev, VK_NULL_HANDLE, 1, &pipeline_info,
                                                                                  nullptr, &pipeline) == VK_SUCCESS;
    }
    if (ok) {
        const auto bind_sets = d.fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
        const auto push_constants = d.fn<PFN_vkCmdPushConstants>("vkCmdPushConstants");
        const auto dispatch = d.fn<PFN_vkCmdDispatch>("vkCmdDispatch");
        ok = d.run([&](VkCommandBuffer cb) {
            d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            const VkDescriptorSet sets[2] = {set0, set1};
            bind_sets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, sets, 0, nullptr);
            for (const SampleKeys& k : keys) {
                push_constants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(k), &k);
                dispatch(cb, 1, 1, 1);
            }
            return true;
        });
    }
    if (ok) std::memcpy(results, mapped, keys.size() * texels * 4);
    if (pipeline) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(dev, pipeline, nullptr);
    if (layout) d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(dev, layout, nullptr);
    if (pool) d.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(dev, pool, nullptr);
    if (set1_layout) d.fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(dev, set1_layout, nullptr);
    if (module) d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(dev, module, nullptr);
    return ok;
}
#endif

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;

    constexpr UINT size = 8, texels = size * size, bytes = texels * 4;
    std::vector<std::uint8_t> results(4 * bytes), after_destruction(bytes), borders(3 * bytes), bgr_results(6 * bytes);
    bool sampled_through_set = false;
    UINT lod_mismatches = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.texture_compression_bc = d.bc;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.image_view_min_lod = d.image_view_min_lod;
        info.sampler_filter_minmax = d.sampler_filter_minmax;
        info.custom_border_color = d.custom_border_color;
        info.custom_border_color_without_format = d.custom_border_color_without_format;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);  // the proxy now owns `inner`

        D3D12_FEATURE_DATA_D3D12_OPTIONS8 options8{};
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS8, &options8, sizeof(options8)) == S_OK);
        EXPECT(options8.UnalignedBlockTexturesSupported == (d.bc ? TRUE : FALSE));
        for (UINT query_size : {0u, UINT(sizeof(options8) - 1), UINT(sizeof(options8) + 1)}) {
            D3D12_FEATURE_DATA_D3D12_OPTIONS8 output[] = {{83}, {42}};
            EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS8, output, query_size) == E_INVALIDARG);
            EXPECT(output[0].UnalignedBlockTexturesSupported == 83 && output[1].UnalignedBlockTexturesSupported == 42);
        }

        // --- Copies. Two host-side heaps; slot i of the source holds 32 bytes of (0x10 * (i + 1) + byte index).
        {
            const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            constexpr UINT source_slots = 4, destination_slots = 8;
            Ref<ID3D12DescriptorHeap> source_heap, destination_heap;
            D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, source_slots,
                                                 D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&source_heap.p))));
            heap_desc.NumDescriptors = destination_slots;
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&destination_heap.p))));
            auto* const source = reinterpret_cast<std::uint8_t*>(
                reinterpret_cast<XboxHeapHandleFn>(slot(source_heap.p, 9))(source_heap.p));
            auto* const destination = reinterpret_cast<std::uint8_t*>(
                reinterpret_cast<XboxHeapHandleFn>(slot(destination_heap.p, 9))(destination_heap.p));
            EXPECT(source && destination && step >= 16);
            for (UINT i = 0; i < source_slots; ++i)
                for (UINT b = 0; b < step; ++b) source[i * step + b] = std::uint8_t(0x10 * (i + 1) + b);
            const auto from = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(source + i * step)}; };
            const auto to = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(destination + i * step)}; };
            // Destination slot `at` holds source slot `want`, or is still zero (want < 0).
            const auto holds = [&](UINT at, int want) {
                const std::uint8_t* got = destination + at * step;
                if (want < 0) return std::all_of(got, got + step, [](std::uint8_t byte) { return byte == 0; });
                return std::memcmp(got, source + UINT(want) * step, step) == 0;
            };
            const auto clear = [&] { std::memset(destination, 0, std::size_t(destination_slots) * step); };
            const auto rtv = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;

            // CopyDescriptorsSimple: one run of descriptors.
            device->CopyDescriptorsSimple(2, to(1), from(2), rtv);
            EXPECT(holds(0, -1) && holds(1, 2) && holds(2, 3) && holds(3, -1));
            device->CopyDescriptorsSimple(0, to(5), from(0), rtv);
            EXPECT(holds(5, -1));

            // CopyDescriptors walks both range lists together: destination ranges {3, 4} and {0}, source ranges
            // {0} and {1, 2}, so 3 <- 0, 4 <- 1, 0 <- 2.
            clear();
            {
                const D3D12_CPU_DESCRIPTOR_HANDLE destinations[] = {to(3), to(0)}, sources[] = {from(0), from(1)};
                const UINT destination_sizes[] = {2, 1}, source_sizes[] = {1, 2};
                device->CopyDescriptors(2, destinations, destination_sizes, 2, sources, source_sizes, rtv);
                EXPECT(holds(3, 0) && holds(4, 1) && holds(0, 2));
                EXPECT(holds(1, -1) && holds(2, -1) && holds(5, -1) && holds(6, -1) && holds(7, -1));
            }
            // Null size arrays: every range is one descriptor.
            clear();
            {
                const D3D12_CPU_DESCRIPTOR_HANDLE destinations[] = {to(6), to(2)}, sources[] = {from(3), from(1)};
                device->CopyDescriptors(2, destinations, nullptr, 2, sources, nullptr, rtv);
                EXPECT(holds(6, 3) && holds(2, 1) && holds(7, -1) && holds(3, -1));
            }
            // One side sized, the other not, and a destination longer than the source: the copy stops with the
            // source.
            clear();
            {
                const D3D12_CPU_DESCRIPTOR_HANDLE destinations[] = {to(1)}, sources[] = {from(0), from(3)};
                const UINT destination_sizes[] = {5};
                device->CopyDescriptors(1, destinations, destination_sizes, 2, sources, nullptr, rtv);
                EXPECT(holds(1, 0) && holds(2, 3) && holds(3, -1) && holds(4, -1) && holds(5, -1) && holds(0, -1));
            }
            // Nothing to copy from or to.
            clear();
            {
                const D3D12_CPU_DESCRIPTOR_HANDLE destinations[] = {to(0)}, sources[] = {from(0)};
                device->CopyDescriptors(1, destinations, nullptr, 0, sources, nullptr, rtv);
                device->CopyDescriptors(1, nullptr, nullptr, 1, sources, nullptr, rtv);
                device->CopyDescriptors(1, destinations, nullptr, 1, nullptr, nullptr, rtv);
                EXPECT(holds(0, -1));
            }
        }

        // --- Sampling through descriptors.
        if (!d.bindless) {
            std::printf("descriptors: copies matched; sampling skipped, the device lacks a feature the bindless set "
                        "needs\n");
            return d.validation_errors() ? 1 : 0;
        }
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        D3D12_RESOURCE_DESC texture_desc{};
        texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture_desc.Width = size;
        texture_desc.Height = size;
        texture_desc.DepthOrArraySize = 1;
        texture_desc.MipLevels = 1;
        texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture_desc.SampleDesc.Count = 1;
        auto sampled_desc = texture_desc;
        sampled_desc.MipLevels = 3;
        Ref<ID3D12Resource> texture, upload;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &sampled_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&texture.p))));
        const auto upload_desc = buffer_desc(131072, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT levels[3]{};
        device->GetCopyableFootprints(&sampled_desc, 0, 3, 0, levels, nullptr, nullptr, nullptr);
        const auto& placed = levels[0];
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, &mapped)));
        for (UINT y = 0; y < size; ++y)
            for (UINT x = 0; x < size; ++x)
                std::memcpy(static_cast<std::uint8_t*>(mapped) + placed.Offset + y * placed.Footprint.RowPitch + x * 4,
                            texel(x, y).data(), 4);
        constexpr std::array<std::uint8_t, 4> mip_colors[] = {{16, 160, 64, 255}, {208, 48, 240, 255}};
        for (UINT mip = 1; mip < 3; ++mip) {
            const auto& level = levels[mip];
            for (UINT y = 0; y < level.Footprint.Height; ++y)
                for (UINT x = 0; x < level.Footprint.Width; ++x)
                    std::memcpy(static_cast<std::uint8_t*>(mapped) + level.Offset + y * level.Footprint.RowPitch + x * 4,
                                mip_colors[mip - 1].data(), 4);
        }
        // The same bytes in an R8G8B8A8_UINT texture, from further into the upload buffer.
        D3D12_RESOURCE_DESC uint_desc = texture_desc;
        uint_desc.Format = DXGI_FORMAT_R8G8B8A8_UINT;
        Ref<ID3D12Resource> uint_texture;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uint_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&uint_texture.p))));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT uint_placed{};
        device->GetCopyableFootprints(&uint_desc, 0, 1, 32768, &uint_placed, nullptr, nullptr, nullptr);
        for (UINT y = 0; y < size; ++y)
            for (UINT x = 0; x < size; ++x)
                std::memcpy(static_cast<std::uint8_t*>(mapped) + uint_placed.Offset +
                                y * uint_placed.Footprint.RowPitch + x * 4,
                            texel(x, y).data(), 4);

        constexpr DXGI_FORMAT byte_formats[] = {DXGI_FORMAT_A8_UNORM, DXGI_FORMAT_R8_UNORM};
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{DXGI_FORMAT_A8_UNORM};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))));
        EXPECT((support.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE) != 0);
        EXPECT((support.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_GATHER) != 0);
        EXPECT((support.Support1 & D3D12_FORMAT_SUPPORT1_MIP) != 0);
        EXPECT(!(support.Support1 & (D3D12_FORMAT_SUPPORT1_RENDER_TARGET | D3D12_FORMAT_SUPPORT1_BLENDABLE |
                                     D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)) && !support.Support2);
        for (UINT count : {1u, 4u}) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS quality_levels{DXGI_FORMAT_A8_UNORM, count};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &quality_levels,
                                                         sizeof(quality_levels))));
            EXPECT(quality_levels.NumQualityLevels == (count == 1 ? 1u : 0u));
        }
        for (auto flag : {D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS}) {
            auto desc = texture_desc;
            desc.Format = DXGI_FORMAT_A8_UNORM;
            desc.Flags = flag;
            EXPECT(FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          __uuidof(ID3D12Resource), nullptr)));
            EXPECT(device->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes == UINT64_MAX);
        }
        Ref<ID3D12Resource> byte_textures[2];
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT byte_placed[2]{}, a8_mip_placed{};
        for (UINT f = 0; f < 2; ++f) {
            auto desc = texture_desc;
            desc.Format = byte_formats[f];
            if (desc.Format == DXGI_FORMAT_A8_UNORM) {
                desc.MipLevels = 2;
                device->GetCopyableFootprints(&desc, 1, 1, 24576, &a8_mip_placed, nullptr, nullptr, nullptr);
            }
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                              D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                              IID_PPV_ARGS(&byte_textures[f].p))));
            device->GetCopyableFootprints(&desc, 0, 1, 16384, &byte_placed[f], nullptr, nullptr, nullptr);
            EXPECT(byte_placed[f].Footprint.Format == desc.Format);
        }
        EXPECT(byte_placed[0].Offset == byte_placed[1].Offset &&
               byte_placed[0].Footprint.RowPitch == byte_placed[1].Footprint.RowPitch);
        for (UINT y = 0; y < size; ++y) {
            auto* row = static_cast<std::uint8_t*>(mapped) + byte_placed[0].Offset + y * byte_placed[0].Footprint.RowPitch;
            for (UINT x = 0; x < size; ++x) row[x] = std::uint8_t(1 + 3 * (y * size + x));
        }

        constexpr UINT mip_size = size / 2;
        EXPECT(a8_mip_placed.Offset == 24576 && a8_mip_placed.Footprint.Width == mip_size &&
               a8_mip_placed.Footprint.Height == mip_size &&
               a8_mip_placed.Footprint.RowPitch == D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
        for (UINT y = 0; y < mip_size; ++y) {
            auto* row = static_cast<std::uint8_t*>(mapped) + a8_mip_placed.Offset + y * a8_mip_placed.Footprint.RowPitch;
            for (UINT x = 0; x < mip_size; ++x) row[x] = std::uint8_t(208 + y * mip_size + x);
        }

        // Three cubes with distinct colours for each face and mip. The 2D array views use layers 1 and 2.
        auto array_desc = sampled_desc;
        array_desc.Width = array_desc.Height = 4;
        constexpr UINT face_count = 18;
        array_desc.DepthOrArraySize = face_count;
        Ref<ID3D12Resource> array_texture;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &array_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&array_texture.p))));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT array_levels[face_count * 3]{};
        device->GetCopyableFootprints(&array_desc, 0, UINT(std::size(array_levels)), 49152, array_levels,
                                      nullptr, nullptr, nullptr);
        const auto array_color = [](UINT layer, UINT mip) {
            return std::array<std::uint8_t, 4>{std::uint8_t(40 + 60 * layer), std::uint8_t(20 + 80 * mip), 64, 255};
        };
        for (UINT sub = 0; sub < std::size(array_levels); ++sub) {
            const auto& placed_array = array_levels[sub];
            const auto& fp = placed_array.Footprint;
            EXPECT(placed_array.Offset + UINT64(fp.RowPitch) * fp.Height <= upload_desc.Width);
            const auto color = array_color(sub / array_desc.MipLevels, sub % array_desc.MipLevels);
            for (UINT y = 0; y < fp.Height; ++y) {
                auto* row = static_cast<std::uint8_t*>(mapped) + placed_array.Offset + y * fp.RowPitch;
                for (UINT x = 0; x < fp.Width; ++x) std::memcpy(row + 4 * x, color.data(), 4);
            }
        }
        // Five textures uploaded with the same bgrx_texel bytes past the cube array's footprints: a B8G8R8X8_UNORM and
        // a B8G8R8A8_UNORM one for their default views, then a B8G8R8A8_UNORM texture whose castable-format list names
        // B8G8R8X8_UNORM and a B8G8R8X8_UNORM one whose list names B8G8R8A8_UNORM, each viewed in the other family,
        // and two R8G8B8A8_UNORM ones whose lists name B8G8R8X8_UNORM and B8G8R8A8_UNORM, viewed through them (red
        // and blue swap places).
        constexpr UINT bgr_count = 6;
        constexpr DXGI_FORMAT bgr_formats[bgr_count] = {DXGI_FORMAT_B8G8R8X8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM,
                                                        DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8X8_UNORM,
                                                        DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM};
        constexpr DXGI_FORMAT bgr_casts[bgr_count] = {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN,
                                                      DXGI_FORMAT_B8G8R8X8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM,
                                                      DXGI_FORMAT_B8G8R8X8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM};
        Ref<ID3D12Resource> bgr_textures[bgr_count];
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT bgr_placed[bgr_count]{};
        Ref<ID3D12Device10> bgr_device10;
        EXPECT(SUCCEEDED(inner->QueryInterface(IID_PPV_ARGS(&bgr_device10.p))));
        const auto& last_array = array_levels[std::size(array_levels) - 1];
        constexpr UINT64 placement = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
        const UINT64 bgr_start =
            (last_array.Offset + UINT64(last_array.Footprint.RowPitch) * last_array.Footprint.Height + placement - 1) &
            ~(placement - 1);
        for (UINT f = 0; f < bgr_count; ++f) {
            D3D12_RESOURCE_DESC desc = texture_desc;
            desc.Format = bgr_formats[f];
            if (bgr_casts[f] == DXGI_FORMAT_UNKNOWN) {
                EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                                 D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                                 IID_PPV_ARGS(&bgr_textures[f].p))));
            } else {
                D3D12_RESOURCE_DESC1 desc1{};
                desc1.Dimension = desc.Dimension;
                desc1.Width = desc.Width;
                desc1.Height = desc.Height;
                desc1.DepthOrArraySize = desc.DepthOrArraySize;
                desc1.MipLevels = desc.MipLevels;
                desc1.Format = desc.Format;
                desc1.SampleDesc = desc.SampleDesc;
                desc1.Layout = desc.Layout;
                desc1.Flags = desc.Flags;
                EXPECT(SUCCEEDED(bgr_device10->CreateCommittedResource3(
                    &default_heap, D3D12_HEAP_FLAG_NONE, &desc1, D3D12_BARRIER_LAYOUT_COPY_DEST, nullptr, nullptr, 1,
                    &bgr_casts[f], IID_PPV_ARGS(&bgr_textures[f].p))));
            }
            device->GetCopyableFootprints(&desc, 0, 1, bgr_start + 8192 * f, &bgr_placed[f], nullptr, nullptr, nullptr);
            EXPECT(bgr_placed[f].Footprint.Format == bgr_formats[f]);
            EXPECT(bgr_placed[f].Offset + UINT64(bgr_placed[f].Footprint.RowPitch) * size <= upload_desc.Width);
            for (UINT y = 0; y < size; ++y)
                for (UINT x = 0; x < size; ++x)
                    std::memcpy(static_cast<std::uint8_t*>(mapped) + bgr_placed[f].Offset +
                                    y * bgr_placed[f].Footprint.RowPitch + x * 4,
                                bgrx_texel(x, y).data(), 4);
        }
        EXPECT(bgr_placed[0].Footprint.RowPitch == bgr_placed[1].Footprint.RowPitch);

        // Shader-visible heaps, start handles in the Xbox shape.
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const UINT sampler_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        Ref<ID3D12DescriptorHeap> view_heap, sampler_heap;
        struct LodView { UINT first, count; float clamp; bool force_alpha_one = false; };
        constexpr LodView lod_views[] = {
            {0, 3, 0.25f}, {0, 3, 0.75f}, {1, 2, 1.75f}, {0, 2, 1.75f}, {1, 2, 0.75f}, {0, 2, 1.75f, true}
        };
        constexpr UINT first_lod_slot = 10;
        constexpr float array_clamps[] = {1.0f, 1.75f, 2.0f, 2.25f};
        struct CubeLodView { D3D12_SRV_DIMENSION dimension; UINT first_face, faces, minimum_mip; };
        constexpr CubeLodView cube_lods[] = {
            {D3D12_SRV_DIMENSION_TEXTURECUBE, 0, 6, 1}, {D3D12_SRV_DIMENSION_TEXTURECUBE, 0, 6, 2},
            {D3D12_SRV_DIMENSION_TEXTURECUBEARRAY, 6, 12, 1}, {D3D12_SRV_DIMENSION_TEXTURECUBEARRAY, 6, 12, 2},
        };
        constexpr LodView volume_lod_views[] = {
            {0, 2, 1.0f}, {1, 1, 0.5f}, {0, 2, 2.0f}, {0, 2, 2.0f, true},
        };
        constexpr UINT lod_view_count = std::size(lod_views);
        constexpr UINT array_clamp_count = std::size(array_clamps);
        constexpr UINT cube_lod_count = std::size(cube_lods);
        constexpr UINT volume_lod_count = std::size(volume_lod_views);
        // Keep multisample views after the preceding volume and minimum-LOD views.
        enum : UINT {
            array_lod_slot = first_lod_slot + lod_view_count,
            cube_lod_slot = array_lod_slot + array_clamp_count,
            volume_lod_slot = cube_lod_slot + cube_lod_count,
            ms_slot = volume_lod_slot + volume_lod_count, ms_array_slot, ms_slice_slot, ms_first_layer_slot,
            ms_uint_slot, ms_uint_array_slot, ms_sint_slot, ms_sint_array_slot,
            line_default, line_mip, line_clamped, bgrx_slot, bgra_slot, typed_bgrx_slot, cast_bgrx_slot,
            cast_bgra_slot, cast_rgbx_slot, cast_rgba_slot, view_slot_count
        };
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, view_slot_count,
                                             D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&view_heap.p))));
        heap_desc = {D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&sampler_heap.p))));
        const std::uint64_t view_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(view_heap.p, 9))(view_heap.p);
        const std::uint64_t sampler_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 9))(sampler_heap.p);
        EXPECT(view_cpu && sampler_cpu);
        const auto view_slot = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(view_cpu + i * step)}; };
        const auto view_bytes = [&](UINT i) { return reinterpret_cast<std::uint8_t*>(view_cpu + i * step); };

        auto line_desc = sampled_desc;
        line_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
        line_desc.Height = 1;
        Ref<ID3D12Resource> line;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &line_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&line.p))));
        device->CreateShaderResourceView(line.p, nullptr, view_slot(line_default));
        D3D12_SHADER_RESOURCE_VIEW_DESC line_view{};
        line_view.Format = line_desc.Format;
        line_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
        line_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        line_view.Texture1D = {1, 1, 0};
        device->CreateShaderResourceView(line.p, &line_view, view_slot(line_mip));
        line_view.Texture1D = {0, UINT_MAX, 2};
        device->CreateShaderResourceView(line.p, &line_view, view_slot(line_clamped));

        for (UINT i = 0; i < std::size(array_clamps); ++i) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = array_desc.Format;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2DArray = {1, 2, 1, 2, 0, array_clamps[i]};
            device->CreateShaderResourceView(array_texture.p, &srv, view_slot(array_lod_slot + i));
        }

        for (UINT i = 0; i < std::size(cube_lods); ++i) {
            const auto& cube_lod = cube_lods[i];
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = array_desc.Format;
            srv.ViewDimension = cube_lod.dimension;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            if (cube_lod.dimension == D3D12_SRV_DIMENSION_TEXTURECUBE)
                srv.TextureCube = {1, 2, float(cube_lod.minimum_mip)};
            else
                srv.TextureCubeArray = {1, 2, cube_lod.first_face, cube_lod.faces / 6, float(cube_lod.minimum_mip)};
            device->CreateShaderResourceView(array_texture.p, &srv, view_slot(cube_lod_slot + i));
        }

        for (UINT i = 0; i < std::size(lod_views); ++i) {
            const auto& v = lod_views[i];
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = sampled_desc.Format;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = v.force_alpha_one
                ? D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0, 1, 2, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1)
                : D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D = {v.first, v.count, 0, v.clamp};
            device->CreateShaderResourceView(texture.p, &srv, view_slot(first_lod_slot + i));
        }

        // The view in slot 1, a copy of it in slot 5, and in slot 3 bytes no view was created for.
        constexpr UINT own = 1, copied = 5, unknown = 3;
        device->CreateShaderResourceView(texture.p, nullptr, view_slot(own));
        EXPECT(!std::all_of(view_bytes(own), view_bytes(own) + step, [](std::uint8_t byte) { return byte == 0; }));
        device->CopyDescriptorsSimple(1, view_slot(copied), view_slot(own), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(std::memcmp(view_bytes(copied), view_bytes(own), step) == 0);
        std::memset(view_bytes(unknown), 0x5a, step);
        D3D12_SAMPLER_DESC sampler_desc{};
        // Wrapping in U, so that it differs from the fallback sampler (nearest, clamping): the shader samples one
        // tile to the right, which gives the texture with this sampler and its right edge with the fallback.
        sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.AddressV = sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler_desc.MaxAnisotropy = 1;
        sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
        device->CreateSampler(&sampler_desc, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu)});
        const auto* sampler_bytes = reinterpret_cast<const std::uint8_t*>(sampler_cpu);
        EXPECT(!std::all_of(sampler_bytes, sampler_bytes + 16, [](std::uint8_t byte) { return byte == 0; }));
        EXPECT(sampler_step >= 16);

        // What the slots hold says what the view and the sampler are, in the structures the console's shaders read.
        namespace g = leonida::gfx10;
        const auto image_at = [&](UINT i, g::Image& image) {
            g::ImageWords words{};
            std::memcpy(words.data(), view_bytes(i), sizeof(words));
            return g::is_image(words) && g::decode(words, image);
        };
        g::Image line_image{};
        EXPECT(image_at(line_default, line_image) && line_image.type == g::ImageType::image_1d &&
               line_image.width == size && line_image.height == 1 && line_image.first_mip == 0 &&
               line_image.mip_count == line_desc.MipLevels && line_image.layer_count == 1);
        EXPECT(image_at(line_mip, line_image) && line_image.type == g::ImageType::image_1d &&
               line_image.first_mip == 1 && line_image.mip_count == 1);
        const auto null_at = [&](UINT i) {
            return std::all_of(view_bytes(i), view_bytes(i) + step, [](std::uint8_t byte) { return byte == 0; });
        };
        // An explicit D3D12_SRV_DIMENSION_TEXTURE1D view on a 1D texture array selects layer zero (like TEXTURE2D
        // and TEXTURE2DMS on 2D arrays), while a default (nullptr) SRV on a 1D array remains a null descriptor.
        {
            auto line_array_desc = line_desc;
            line_array_desc.DepthOrArraySize = 3;
            Ref<ID3D12Resource> line_array;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &line_array_desc,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&line_array.p))));
            D3D12_DESCRIPTOR_HEAP_DESC la_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2,
                                                    D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            Ref<ID3D12DescriptorHeap> la_heap;
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&la_heap_desc, IID_PPV_ARGS(&la_heap.p))));
            const std::uint64_t la_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(la_heap.p, 9))(la_heap.p);
            const D3D12_CPU_DESCRIPTOR_HANDLE la_default{SIZE_T(la_cpu)}, la_slice0{SIZE_T(la_cpu + step)};
            std::memset(reinterpret_cast<void*>(la_default.ptr), 0x77, step);
            device->CreateShaderResourceView(line_array.p, nullptr, la_default);
            EXPECT(std::all_of(reinterpret_cast<const std::uint8_t*>(la_default.ptr),
                               reinterpret_cast<const std::uint8_t*>(la_default.ptr) + step,
                               [](std::uint8_t byte) { return byte == 0; }));
            D3D12_SHADER_RESOURCE_VIEW_DESC la_view{};
            la_view.Format = DXGI_FORMAT_UNKNOWN;
            la_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            la_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            la_view.Texture1D = {1, 1, 0};
            device->CreateShaderResourceView(line_array.p, &la_view, la_slice0);
            g::ImageWords la_words{};
            std::memcpy(la_words.data(), reinterpret_cast<const void*>(la_slice0.ptr), sizeof(la_words));
            g::Image la_image{};
            EXPECT(g::is_image(la_words) && g::decode(la_words, la_image) &&
                   la_image.type == g::ImageType::image_1d && la_image.first_mip == 1 && la_image.mip_count == 1 &&
                   la_image.first_layer == 0 && la_image.layer_count == 1);
            std::uint32_t la_key[8]{};
            std::memcpy(la_key, la_words.data(), sizeof(la_key));
            EXPECT(rt::d3d12::bindless_hash_lookup(device.p, la_key) != 0);
        }
        D3D12_RESOURCE_DESC ms_desc = texture_desc;
        ms_desc.Width = ms_desc.Height = 4;
        ms_desc.SampleDesc.Count = 4;
        ms_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        Ref<ID3D12Resource> ms_texture, ms_array;
        D3D12_FEATURE_DATA_FORMAT_SUPPORT float_support{ms_desc.Format};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &float_support, sizeof(float_support))));
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS ms_quality{ms_desc.Format, 4};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &ms_quality, sizeof(ms_quality))));
        if ((float_support.Support1 & D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD) && ms_quality.NumQualityLevels) {
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &ms_desc,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&ms_texture.p))));
            ms_desc.DepthOrArraySize = 3;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &ms_desc,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&ms_array.p))));
            device->CreateShaderResourceView(ms_texture.p, nullptr, view_slot(ms_slot));
            device->CreateShaderResourceView(ms_array.p, nullptr, view_slot(ms_array_slot));
            g::Image ms_image{};
            EXPECT(image_at(ms_slot, ms_image) && ms_image.type == g::ImageType::image_2d_msaa &&
                   ms_image.samples == 4 && ms_image.first_mip == 0 && ms_image.mip_count == 1 && ms_image.layer_count == 1);
            EXPECT(image_at(ms_array_slot, ms_image) && ms_image.type == g::ImageType::image_2d_msaa_array &&
                   ms_image.samples == 4 && ms_image.first_layer == 0 && ms_image.layer_count == 3);
            D3D12_SHADER_RESOURCE_VIEW_DESC ms_view{};
            ms_view.Format = ms_desc.Format;
            ms_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            ms_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
            device->CreateShaderResourceView(ms_texture.p, &ms_view, view_slot(ms_slice_slot));
            EXPECT(std::memcmp(view_bytes(ms_slot), view_bytes(ms_slice_slot), step) == 0);
            // An explicit UNKNOWN format inherits the texture format. A 2DMS view of an array selects layer zero.
            ms_view.Format = DXGI_FORMAT_UNKNOWN;
            device->CreateShaderResourceView(ms_texture.p, &ms_view, view_slot(ms_slice_slot));
            EXPECT(std::memcmp(view_bytes(ms_slot), view_bytes(ms_slice_slot), step) == 0);
            device->CreateShaderResourceView(ms_array.p, &ms_view, view_slot(ms_first_layer_slot));
            EXPECT(image_at(ms_first_layer_slot, ms_image) && ms_image.type == g::ImageType::image_2d_msaa &&
                   ms_image.first_layer == 0 && ms_image.layer_count == 1);
            device->CreateShaderResourceView(texture.p, &ms_view, view_slot(ms_slice_slot));
            EXPECT(null_at(ms_slice_slot));  // an MS view of a single-sampled image
            ms_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            ms_view.Texture2D.MipLevels = 1;
            device->CreateShaderResourceView(ms_texture.p, &ms_view, view_slot(ms_slice_slot));
            EXPECT(null_at(ms_slice_slot));  // and the inverse
            ms_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
            ms_view.Texture2DMSArray.FirstArraySlice = 3;
            ms_view.Texture2DMSArray.ArraySize = 1;
            device->CreateShaderResourceView(ms_array.p, &ms_view, view_slot(ms_slice_slot));
            EXPECT(null_at(ms_slice_slot));
            ms_view.Texture2DMSArray.FirstArraySlice = 1;
            for (UINT invalid_count : {0u, 3u, UINT_MAX - 1}) {
                ms_view.Texture2DMSArray.ArraySize = invalid_count;
                device->CreateShaderResourceView(ms_array.p, &ms_view, view_slot(ms_slice_slot));
                EXPECT(null_at(ms_slice_slot));
            }
            ms_view.Texture2DMSArray.ArraySize = 2;
            ms_view.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(6, 1, 2, 3);
            device->CreateShaderResourceView(ms_array.p, &ms_view, view_slot(ms_slice_slot));
            EXPECT(null_at(ms_slice_slot));
            ms_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            for (UINT remaining : {2u, UINT_MAX}) {
                ms_view.Texture2DMSArray.ArraySize = remaining;
                device->CreateShaderResourceView(ms_array.p, &ms_view, view_slot(ms_slice_slot));
                EXPECT(image_at(ms_slice_slot, ms_image) && ms_image.type == g::ImageType::image_2d_msaa_array &&
                       ms_image.samples == 4 && ms_image.first_layer == 1 && ms_image.layer_count == 2);
            }
        } else {
            std::printf("MSAA SRVs: device does not advertise 4x RGBA8 multisample loads, skipped\n");
        }

        g::Image image{};
        EXPECT(image_at(own, image));
        EXPECT(image.type == g::ImageType::image_2d && image.width == size && image.height == size);
        EXPECT(image.format == g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM) && image.format != 0);
        EXPECT(image.first_mip == 0 && image.mip_count == 3 && image.layer_count == 1 && image.resource_mips == 3);
        EXPECT(image.swizzle == g::identity_swizzle && image.address >= (1ull << 32) && image.address % 256 == 0);
        g::SamplerWords sampler_words{};
        std::memcpy(sampler_words.data(), sampler_bytes, sizeof(sampler_words));
        const g::Sampler sampler = g::decode(sampler_words);
        EXPECT(sampler.mag_filter == g::Filter::point && sampler.min_filter == g::Filter::point);
        EXPECT(sampler.mip_filter == g::MipFilter::point && sampler.max_anisotropy == 1);
        EXPECT(sampler.address_u == g::AddressMode::wrap && sampler.address_v == g::AddressMode::clamp_last_texel &&
               sampler.address_w == g::AddressMode::clamp_last_texel);
        EXPECT(sampler.compare == g::CompareFunction::never && sampler.min_lod == 0.0f && sampler.max_lod > 14.9f);
        // Two literal S#s, as a shader builds from immediates: the same wrapping sampler with other LOD ranges, so
        // other keys. The first is registered without a heap slot (twice: the second time changes nothing), the
        // second not, so a shader finds the first and gets the fallback for the second.
        g::Sampler literal = sampler;
        literal.max_lod = 4.0f;
        const g::SamplerWords literal_words = g::encode(literal);
        literal.max_lod = 5.0f;
        const g::SamplerWords unregistered_words = g::encode(literal);
        EXPECT(literal_words != sampler_words && unregistered_words != literal_words);
        EXPECT(rt::d3d12::register_samplers(device.p, {&literal_words, 1}) == S_OK);
        EXPECT(rt::d3d12::register_samplers(device.p, {&literal_words, 1}) == S_OK);
        EXPECT(rt::d3d12::register_samplers(nullptr, {&literal_words, 1}) == E_INVALIDARG);

        // GFX10 literal S#s can encode a larger LOD bias than vkCreateSampler accepts on this device.
        literal.lod_bias = -32.0f;
        const auto negative_bias = g::encode(literal);
        literal.lod_bias = 31.0f;
        const auto positive_bias = g::encode(literal);
        const g::SamplerWords biased_samplers[] = {negative_bias, positive_bias};
        EXPECT(rt::d3d12::register_samplers(device.p, biased_samplers) == S_OK);

        // Static samplers of a root signature (docs/d3d12_vulkan.md section 3.5): no heap slot and no root argument,
        // the shader holds the S# by value, so CreateRootSignature has to register it. One per blob version, each
        // the wrapping sampler above with its own LOD range (so its own key), and one that only the fourth has:
        // a border sampler with the UINT white border of the static enum. None is in the hash table before its
        // root signature is created, all are after. A version the PC serializer here does not write is skipped.
        const auto sampler_index = [&](const g::SamplerWords& words) {
            std::uint32_t key[8]{~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
            std::memcpy(key, words.data(), sizeof(words));
            return rt::d3d12::bindless_hash_lookup(device.p, key);
        };
        struct StaticCase {
            D3D_ROOT_SIGNATURE_VERSION version;
            const char* name;
            float max_lod;
            bool uint_white_border;
            g::SamplerWords words;
            bool created;
        } static_cases[] = {{D3D_ROOT_SIGNATURE_VERSION_1_0, "1.0", 6.0f, false, {}, false},
                            {D3D_ROOT_SIGNATURE_VERSION_1_1, "1.1", 7.0f, false, {}, false},
                            {D3D_ROOT_SIGNATURE_VERSION_1_2, "1.2", 8.0f, false, {}, false},
                            {D3D_ROOT_SIGNATURE_VERSION_1_1, "1.1, UINT white border", 9.0f, true, {}, false}};
        std::vector<Ref<ID3D12RootSignature>> static_roots(std::size(static_cases));
        UINT static_created = 0;
        for (std::size_t c = 0; c < std::size(static_cases); ++c) {
            StaticCase& test = static_cases[c];
            g::Sampler expected = sampler;
            expected.max_lod = test.max_lod;
            if (test.uint_white_border) {
                expected.address_u = g::AddressMode::clamp_border;
                expected.border_color = g::BorderColor::opaque_white;
                expected.integer_border = true;
            }
            test.words = g::encode(expected);
            EXPECT(test.words != sampler_words && sampler_index(test.words) == 0);

            D3D12_STATIC_SAMPLER_DESC1 fixed{};
            fixed.Filter = sampler_desc.Filter;
            fixed.AddressU = test.uint_white_border ? D3D12_TEXTURE_ADDRESS_MODE_BORDER : sampler_desc.AddressU;
            fixed.AddressV = sampler_desc.AddressV;
            fixed.AddressW = sampler_desc.AddressW;
            fixed.MaxAnisotropy = sampler_desc.MaxAnisotropy;
            fixed.ComparisonFunc = sampler_desc.ComparisonFunc;
            fixed.BorderColor = test.uint_white_border ? D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE_UINT
                                                       : D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
            fixed.MaxLOD = test.max_lod;
            fixed.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            // D3D12_STATIC_SAMPLER_DESC is D3D12_STATIC_SAMPLER_DESC1 without the trailing Flags.
            D3D12_STATIC_SAMPLER_DESC fixed0{};
            std::memcpy(&fixed0, &fixed, sizeof(fixed0));
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
            root_desc.Version = test.version;
            if (test.version == D3D_ROOT_SIGNATURE_VERSION_1_0) {
                root_desc.Desc_1_0.NumStaticSamplers = 1;
                root_desc.Desc_1_0.pStaticSamplers = &fixed0;
            } else if (test.version == D3D_ROOT_SIGNATURE_VERSION_1_1) {
                root_desc.Desc_1_1.NumStaticSamplers = 1;
                root_desc.Desc_1_1.pStaticSamplers = &fixed0;
            } else {
                root_desc.Desc_1_2.NumStaticSamplers = 1;
                root_desc.Desc_1_2.pStaticSamplers = &fixed;
            }
            const std::vector<std::uint8_t> blob = serialize_root_signature(root_desc);
            if (blob.empty()) {
                std::printf("static sampler, root signature %s: the PC serializer does not write it, skipped\n",
                            test.name);
                continue;
            }
            EXPECT(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(),
                                                         IID_PPV_ARGS(&static_roots[c].p))));
            EXPECT(sampler_index(test.words) != 0);
            test.created = true;
            ++static_created;
        }
        // The entries outlive their root signatures, as a heap slot's S# outlives the heap: the shader pass below
        // samples with them after the last reference is gone.
        for (std::size_t c = 0; c < std::size(static_cases); ++c) {
            const std::uint32_t before = sampler_index(static_cases[c].words);
            static_roots[c].reset();
            EXPECT(sampler_index(static_cases[c].words) == before);
        }
        std::printf("static samplers: %u of %u root signatures registered their S#\n", static_created,
                    UINT(std::size(static_cases)));
        EXPECT(static_created >= 3);  // 1.0 and 1.1 are in every d3d12.dll; 1.2 came later

        // The UINT texture's view in slot 0, and three border samplers after the first: UINT (0,0,0,0) and
        // (1,1,1,1) from CreateSampler2, which the proxy does not have (the inner device does), and float white.
        constexpr UINT uint_slot = 0, uint_black = 1, uint_white = 2, float_white = 3;
        device->CreateShaderResourceView(uint_texture.p, nullptr, view_slot(uint_slot));
        {
            Ref<ID3D12Device11> inner11;
            EXPECT(SUCCEEDED(inner->QueryInterface(IID_PPV_ARGS(&inner11.p))));
            D3D12_SAMPLER_DESC2 border_desc{};
            border_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
            border_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            border_desc.AddressV = border_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            border_desc.MaxAnisotropy = 1;
            border_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
            const auto at = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu + i * sampler_step)}; };
            border_desc.Flags = D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR;
            inner11->CreateSampler2(&border_desc, at(uint_black));
            for (UINT& c : border_desc.UintBorderColor) c = 1;
            inner11->CreateSampler2(&border_desc, at(uint_white));
            border_desc.Flags = D3D12_SAMPLER_FLAG_NONE;
            for (FLOAT& c : border_desc.FloatBorderColor) c = 1.0f;
            inner11->CreateSampler2(&border_desc, at(float_white));
        }
        const auto sampler_at = [&](UINT i) {
            return reinterpret_cast<const std::uint8_t*>(sampler_cpu + i * sampler_step);
        };
        // The two white borders differ only in the integer mark.
        g::SamplerWords white_words{}, float_white_words{};
        std::memcpy(white_words.data(), sampler_at(uint_white), sizeof(white_words));
        std::memcpy(float_white_words.data(), sampler_at(float_white), sizeof(float_white_words));
        EXPECT(white_words != float_white_words && g::decode(white_words).integer_border);
        EXPECT(white_words[3] == (float_white_words[3] | g::kIntegerBorderPtr));

        // A view with every field away from its default: mips 1 and 2 of a BGRA cube, with the view's mapping
        // swapping red and blue and forcing alpha to one. The texture is another one, so its address differs.
        Ref<ID3D12Resource> cube, writable;
        D3D12_RESOURCE_DESC cube_desc = texture_desc;
        cube_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        cube_desc.MipLevels = 4;
        cube_desc.DepthOrArraySize = 6;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &cube_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&cube.p))));
        D3D12_SHADER_RESOURCE_VIEW_DESC cube_view{};
        cube_view.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        cube_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        cube_view.Shader4ComponentMapping =
            D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(2, 1, 0, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1);
        cube_view.TextureCube.MostDetailedMip = 1;
        cube_view.TextureCube.MipLevels = 2;
        constexpr UINT cube_slot = 6, read_slot = 4, write_slot = 7, null_slot = 2;
        device->CreateShaderResourceView(cube.p, &cube_view, view_slot(cube_slot));
        g::Image cube_image{};
        EXPECT(image_at(cube_slot, cube_image));
        EXPECT(cube_image.type == g::ImageType::cube && cube_image.layer_count == 6 && cube_image.first_layer == 0);
        EXPECT(cube_image.first_mip == 1 && cube_image.mip_count == 2 && cube_image.resource_mips == 4);
        EXPECT(cube_image.format == g::format_from_vulkan(VK_FORMAT_B8G8R8A8_UNORM));
        // BGRA keeps blue in the hardware's first channel, so reading blue as red is X, and so on.
        EXPECT((cube_image.swizzle == g::Swizzle{g::select_x, g::select_y, g::select_z, g::select_one}));
        EXPECT(cube_image.address != image.address);

        // An unordered-access view is the same T# as a shader-resource view of the same mip: one key, one entry.
        D3D12_RESOURCE_DESC writable_desc = texture_desc;
        writable_desc.Format = DXGI_FORMAT_R32_UINT;
        writable_desc.MipLevels = 2;
        writable_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &writable_desc,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&writable.p))));
        D3D12_SHADER_RESOURCE_VIEW_DESC read_view{};
        read_view.Format = DXGI_FORMAT_R32_UINT;
        read_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        read_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        read_view.Texture2D.MostDetailedMip = 1;
        read_view.Texture2D.MipLevels = 1;
        D3D12_UNORDERED_ACCESS_VIEW_DESC write_view{};
        write_view.Format = DXGI_FORMAT_R32_UINT;
        write_view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        write_view.Texture2D.MipSlice = 1;
        device->CreateShaderResourceView(writable.p, &read_view, view_slot(read_slot));
        device->CreateUnorderedAccessView(writable.p, nullptr, &write_view, view_slot(write_slot));
        g::Image write_image{};
        EXPECT(image_at(write_slot, write_image));
        EXPECT(write_image.first_mip == 1 && write_image.mip_count == 1 && write_image.resource_mips == 2);
        EXPECT(write_image.format == g::format_from_vulkan(VK_FORMAT_R32_UINT));
        EXPECT(std::memcmp(view_bytes(read_slot), view_bytes(write_slot), step) == 0);

        // Null descriptors: no resource, a view the texture cannot have, a view past its mips.
        std::memset(view_bytes(null_slot), 0x77, step);
        device->CreateShaderResourceView(nullptr, &read_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        std::memset(view_bytes(null_slot), 0x77, step);
        device->CreateUnorderedAccessView(texture.p, nullptr, nullptr, view_slot(null_slot));  // not ALLOW_UNORDERED_ACCESS
        EXPECT(null_at(null_slot));
        std::memset(view_bytes(null_slot), 0x77, step);
        read_view.Texture2D.MostDetailedMip = 2;
        device->CreateShaderResourceView(writable.p, &read_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        // A volume has a depth, one Vulkan array layer and a view-relative mip range.
        D3D12_RESOURCE_DESC volume_desc = texture_desc;
        volume_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        volume_desc.DepthOrArraySize = 4;
        volume_desc.MipLevels = 2;
        Ref<ID3D12Resource> volume;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &volume_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&volume.p))));
        constexpr UINT volume_all = 8, volume_mip = 9;
        device->CreateShaderResourceView(volume.p, nullptr, view_slot(volume_all));
        g::Image vol{};
        EXPECT(image_at(volume_all, vol) && vol.type == g::ImageType::image_3d);
        EXPECT(vol.width == size && vol.height == size && vol.depth == 4 && vol.layer_count == 1);
        EXPECT(vol.first_mip == 0 && vol.mip_count == 2);
        D3D12_SHADER_RESOURCE_VIEW_DESC volume_view{};
        volume_view.Format = volume_desc.Format;
        volume_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        volume_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        volume_view.Texture3D = {1, 1, 0};
        device->CreateShaderResourceView(volume.p, &volume_view, view_slot(volume_mip));
        EXPECT(image_at(volume_mip, vol) && vol.type == g::ImageType::image_3d);
        EXPECT(vol.first_mip == 1 && vol.mip_count == 1 && vol.depth == 4 && vol.resource_mips == 2);
        auto typeless_desc = volume_desc;
        typeless_desc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
        Ref<ID3D12Resource> typeless_volume;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &typeless_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&typeless_volume.p))));
        const D3D12_DESCRIPTOR_HEAP_DESC typeless_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2,
                                                           D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> typeless_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&typeless_heap_desc, IID_PPV_ARGS(&typeless_heap.p))));
        const auto linear_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(typeless_heap.p, 9))(typeless_heap.p);
        const D3D12_CPU_DESCRIPTOR_HANDLE linear_slot{SIZE_T(linear_cpu)}, srgb_slot{SIZE_T(linear_cpu + step)};
        auto typed_view = volume_view;
        device->CreateShaderResourceView(typeless_volume.p, &typed_view, linear_slot);
        typed_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        device->CreateShaderResourceView(typeless_volume.p, &typed_view, srgb_slot);
        device->CopyDescriptorsSimple(1, view_slot(null_slot), view_slot(volume_mip),
                                      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        device->CreateShaderResourceView(typeless_volume.p, nullptr, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        for (const auto invalid : {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_B8G8R8A8_UNORM}) {
            typed_view.Format = invalid;
            device->CopyDescriptorsSimple(1, view_slot(null_slot), view_slot(volume_mip),
                                          D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            device->CreateShaderResourceView(typeless_volume.p, &typed_view, view_slot(null_slot));
            EXPECT(null_at(null_slot));
        }
        // D3D12 requires DENY_SHADER_RESOURCE to accompany ALLOW_DEPTH_STENCIL. If the layer accepts
        // this transfer-only volume at creation, requesting an SRV must still clear the destination.
        D3D12_RESOURCE_DESC denied_desc = volume_desc;
        denied_desc.Flags = D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
        Ref<ID3D12Resource> denied_volume;
        const HRESULT denied_hr = device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &denied_desc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&denied_volume.p));
        EXPECT(denied_hr == S_OK || denied_hr == E_INVALIDARG);
        if (denied_hr == S_OK) {
            device->CopyDescriptorsSimple(1, view_slot(null_slot), view_slot(volume_mip),
                                          D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            EXPECT(!null_at(null_slot));
            device->CreateShaderResourceView(denied_volume.p, &volume_view, view_slot(null_slot));
            EXPECT(null_at(null_slot));
            denied_volume.reset();
        }
        volume_view.Texture3D.MostDetailedMip = 2;
        device->CreateShaderResourceView(volume.p, &volume_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        volume_view.Texture3D.MostDetailedMip = 0;
        device->CreateShaderResourceView(texture.p, &volume_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        for (UINT i = 0; i < volume_lod_count; ++i) {
            const auto& v = volume_lod_views[i];
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = volume_desc.Format;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            srv.Shader4ComponentMapping = v.force_alpha_one
                ? D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0, 1, 2, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1)
                : D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture3D = {v.first, v.count, v.clamp};
            device->CreateShaderResourceView(volume.p, &srv, view_slot(volume_lod_slot + i));
            EXPECT(image_at(volume_lod_slot + i, vol) && vol.type == g::ImageType::image_3d);
            EXPECT(vol.first_mip == v.first && vol.mip_count == v.count && vol.min_lod == v.clamp);
        }
        volume_view.Format = DXGI_FORMAT_R8G8B8A8_UINT;
        device->CreateShaderResourceView(volume.p, &volume_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        D3D12_SHADER_RESOURCE_VIEW_DESC flat_view{};
        flat_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        flat_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        flat_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        flat_view.Texture2D.MipLevels = 1;
        std::memset(view_bytes(null_slot), 0x77, step);
        device->CreateShaderResourceView(volume.p, &flat_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT volume_placed[2]{};
        // Place both volume mips after the UINT texture, leaving the lower offsets for the RGBA8 and RGB9E5 uploads.
        const UINT64 volume_start = uint_placed.Offset + UINT64(uint_placed.Footprint.RowPitch) * uint_placed.Footprint.Height;
        device->GetCopyableFootprints(&volume_desc, 0, 2, volume_start, volume_placed, nullptr, nullptr, nullptr);
        const auto& last = volume_placed[1];
        EXPECT(last.Offset + UINT64(last.Footprint.RowPitch) * last.Footprint.Height * last.Footprint.Depth <= upload_desc.Width);
        for (UINT mip = 0; mip < 2; ++mip) {
            const auto& fp = volume_placed[mip].Footprint;
            for (UINT z = 0; z < fp.Depth; ++z)
                for (UINT y = 0; y < fp.Height; ++y)
                    for (UINT x = 0; x < fp.Width; ++x) {
                        const std::uint8_t rgba[]{std::uint8_t(x * 19), std::uint8_t(y * 23),
                                                  std::uint8_t(z * 53), std::uint8_t(mip ? 101 : 237)};
                        const auto offset = volume_placed[mip].Offset + (z * fp.Height + y) * fp.RowPitch + 4 * x;
                        std::memcpy(static_cast<std::uint8_t*>(mapped) + offset, rgba, 4);
                    }
        }

        constexpr UINT a8_mip_slot = 4, byte_view_count = a8_mip_slot + 1;
        D3D12_DESCRIPTOR_HEAP_DESC byte_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, byte_view_count,
                                                  D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> byte_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&byte_heap_desc, IID_PPV_ARGS(&byte_heap.p))));
        const auto byte_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(byte_heap.p, 9))(byte_heap.p);
        const UINT byte_step = device->GetDescriptorHandleIncrementSize(byte_heap_desc.Type);
        EXPECT(byte_cpu && byte_step >= sizeof(g::ImageWords));
        const auto byte_slot = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(byte_cpu + i * byte_step)}; };
        const auto byte_bytes = [&](UINT i) { return reinterpret_cast<std::uint8_t*>(byte_slot(i).ptr); };
        D3D12_SHADER_RESOURCE_VIEW_DESC byte_view{};
        byte_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        byte_view.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
            D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0,
            D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0);
        byte_view.Texture2D.MipLevels = 1;
        for (UINT f = 0; f < 2; ++f) {
            byte_view.Format = byte_formats[1 - f];
            std::memset(byte_bytes(2 * f), 0x77, byte_step);
            device->CreateShaderResourceView(byte_textures[f].p, &byte_view, byte_slot(2 * f));
            EXPECT(std::all_of(byte_bytes(2 * f), byte_bytes(2 * f) + byte_step,
                               [](std::uint8_t byte) { return byte == 0; }));
            device->CreateShaderResourceView(byte_textures[f].p, nullptr, byte_slot(2 * f));
            byte_view.Format = byte_formats[f];
            device->CreateShaderResourceView(byte_textures[f].p, &byte_view, byte_slot(2 * f + 1));
        }
        byte_view.Format = DXGI_FORMAT_A8_UNORM;
        byte_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        byte_view.Texture2D.MostDetailedMip = 1;
        device->CreateShaderResourceView(byte_textures[0].p, &byte_view, byte_slot(a8_mip_slot));

        // D3D-75: plane 1 of a depth-stencil texture is its stencil. The T# is 8_UINT at the texture's own address,
        // with the eight bits read as G and R, B, A as 0, 0, 1 (D3D12's X24_TYPELESS_G8_UINT / X32_..._G8X24_UINT).
        // A stencil view of a colour texture, or one in another format, is a null descriptor.
        // Plane 0, as R24_UNORM_X8_TYPELESS or R32_FLOAT_X8X24_TYPELESS, is a depth T# of the format backing the
        // texture at the same address, the depth read as R. A GPU without a usable D24 backs R24G8 with
        // D32_SFLOAT_S8_UINT (the layer's depth_usable rule), so the expected code follows the backing. Each depth
        // view format belongs to its own family only: the other one's is a null descriptor, as D3D12 refuses it.
        VkFormatProperties d24_properties{};
        d.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(
            d.physical, VK_FORMAT_D24_UNORM_S8_UINT, &d24_properties);
        const VkFormat d24_backing =
            (d24_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
                ? VK_FORMAT_D24_UNORM_S8_UINT
                : VK_FORMAT_D32_SFLOAT_S8_UINT;
        for (const auto& [resource_format, stencil_view_format, depth_view_format, other_depth_view_format, depth_vk] :
             {std::tuple{DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_X24_TYPELESS_G8_UINT,
                         DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, d24_backing},
              std::tuple{DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_X32_TYPELESS_G8X24_UINT,
                         DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
                         VK_FORMAT_D32_SFLOAT_S8_UINT}}) {
            D3D12_RESOURCE_DESC depth_desc = texture_desc;
            depth_desc.Format = resource_format;
            depth_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            Ref<ID3D12Resource> depth_stencil;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &depth_desc,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&depth_stencil.p))));
            D3D12_SHADER_RESOURCE_VIEW_DESC plane_view{};
            plane_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            plane_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            plane_view.Texture2D.MipLevels = 1;
            plane_view.Format = stencil_view_format;
            plane_view.Texture2D.PlaneSlice = 1;
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateShaderResourceView(depth_stencil.p, &plane_view, view_slot(null_slot));
            g::Image stencil_image{};
            EXPECT(image_at(null_slot, stencil_image));
            EXPECT(stencil_image.format == g::stencil_format && stencil_image.type == g::ImageType::image_2d);
            EXPECT((stencil_image.swizzle == g::Swizzle{g::select_zero, g::select_x, g::select_zero, g::select_one}));
            EXPECT(stencil_image.width == size && stencil_image.mip_count == 1);

            plane_view.Format = depth_view_format;
            plane_view.Texture2D.PlaneSlice = 0;
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateShaderResourceView(depth_stencil.p, &plane_view, view_slot(null_slot));
            g::Image depth_image{};
            EXPECT(image_at(null_slot, depth_image));
            EXPECT(depth_image.format == g::format_from_vulkan(depth_vk) && depth_image.format != 0);
            EXPECT(depth_image.address == stencil_image.address && depth_image.swizzle[0] == g::select_x);

            plane_view.Format = other_depth_view_format;  // the other depth family's view format
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateShaderResourceView(depth_stencil.p, &plane_view, view_slot(null_slot));
            EXPECT(null_at(null_slot));

            plane_view.Format = DXGI_FORMAT_R8_UINT;  // not a stencil view format
            plane_view.Texture2D.PlaneSlice = 1;
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateShaderResourceView(depth_stencil.p, &plane_view, view_slot(null_slot));
            EXPECT(null_at(null_slot));

            // The other family's stencil view format is refused too, as D3D12 refuses it: X24_TYPELESS_G8_UINT
            // belongs to R24G8 only, X32_TYPELESS_G8X24_UINT to R32G8X24 only.
            plane_view.Format = stencil_view_format == DXGI_FORMAT_X24_TYPELESS_G8_UINT
                                    ? DXGI_FORMAT_X32_TYPELESS_G8X24_UINT
                                    : DXGI_FORMAT_X24_TYPELESS_G8_UINT;
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateShaderResourceView(depth_stencil.p, &plane_view, view_slot(null_slot));
            EXPECT(null_at(null_slot));

            // Both planes of a depth-stencil resource can be denied to shaders. Start each negative
            // case with a live SRV so leaving a stale T# in the destination also fails.
            depth_desc.Flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
            Ref<ID3D12Resource> denied;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &depth_desc,
                            D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr, IID_PPV_ARGS(&denied.p))));
            for (UINT plane = 0; plane < 2; ++plane) {
                plane_view.Format = plane == 0 ? depth_view_format : stencil_view_format;
                plane_view.Texture2D.PlaneSlice = plane;
                device->CreateShaderResourceView(depth_stencil.p, &plane_view, view_slot(null_slot));
                EXPECT(!null_at(null_slot));
                device->CreateShaderResourceView(denied.p, &plane_view, view_slot(null_slot));
                EXPECT(null_at(null_slot));
            }

            // The family without ALLOW_DEPTH_STENCIL, as a copy of a depth buffer kept for sampling, has the same
            // two planes.
            depth_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
            Ref<ID3D12Resource> plain;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &depth_desc,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&plain.p))));
            plane_view.Format = depth_view_format;
            plane_view.Texture2D.PlaneSlice = 0;
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateShaderResourceView(plain.p, &plane_view, view_slot(null_slot));
            g::Image plain_depth{};
            EXPECT(image_at(null_slot, plain_depth) && plain_depth.format == g::format_from_vulkan(depth_vk));
            plane_view.Format = stencil_view_format;
            plane_view.Texture2D.PlaneSlice = 1;
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateShaderResourceView(plain.p, &plane_view, view_slot(null_slot));
            g::Image plain_stencil{};
            EXPECT(image_at(null_slot, plain_stencil) && plain_stencil.format == g::stencil_format);
            EXPECT(plain_depth.address == plain_stencil.address);
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC colour_plane{};
        colour_plane.Format = DXGI_FORMAT_X24_TYPELESS_G8_UINT;
        colour_plane.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        colour_plane.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        colour_plane.Texture2D.MipLevels = 1;
        colour_plane.Texture2D.PlaneSlice = 1;
        std::memset(view_bytes(null_slot), 0x77, step);
        device->CreateShaderResourceView(texture.p, &colour_plane, view_slot(null_slot));
        EXPECT(null_at(null_slot));

        // B8G8R8X8 is a BGRA8 image whose views read alpha as 1, so its T# selects one where B8G8R8A8's selects W.
        // B8G8R8X8 and B8G8R8A8 are different typeless families: a view never crosses from one to the other.
        device->CreateShaderResourceView(bgr_textures[0].p, nullptr, view_slot(bgrx_slot));
        device->CreateShaderResourceView(bgr_textures[1].p, nullptr, view_slot(bgra_slot));
        g::Image bgrx_image{}, bgra_image{};
        EXPECT(image_at(bgrx_slot, bgrx_image) && image_at(bgra_slot, bgra_image));
        EXPECT(bgrx_image.format == g::format_from_vulkan(VK_FORMAT_B8G8R8A8_UNORM) &&
               bgra_image.format == bgrx_image.format);
        EXPECT((bgrx_image.swizzle == g::Swizzle{g::select_z, g::select_y, g::select_x, g::select_one}));
        EXPECT((bgra_image.swizzle == g::Swizzle{g::select_z, g::select_y, g::select_x, g::select_w}));
        // A castable-format list lets a view cross the two families, and the view's own format decides the alpha:
        // WARP and AMD's D3D12 driver read 1 through a B8G8R8X8 view of a B8G8R8A8 texture, and the stored byte
        // through a B8G8R8A8 view of a B8G8R8X8 one. A format the list doesn't name stays refused.
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC cast_view = flat_view;
            cast_view.Format = DXGI_FORMAT_B8G8R8X8_UNORM;
            device->CreateShaderResourceView(bgr_textures[2].p, &cast_view, view_slot(cast_bgrx_slot));
            cast_view.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            device->CreateShaderResourceView(bgr_textures[3].p, &cast_view, view_slot(cast_bgra_slot));
            cast_view.Format = DXGI_FORMAT_B8G8R8X8_UNORM;
            device->CreateShaderResourceView(bgr_textures[4].p, &cast_view, view_slot(cast_rgbx_slot));
            cast_view.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            device->CreateShaderResourceView(bgr_textures[5].p, &cast_view, view_slot(cast_rgba_slot));
            g::Image cast_bgrx_image{}, cast_bgra_image{}, cast_rgbx_image{}, cast_rgba_image{};
            EXPECT(image_at(cast_bgrx_slot, cast_bgrx_image) && image_at(cast_bgra_slot, cast_bgra_image) &&
                   image_at(cast_rgbx_slot, cast_rgbx_image) && image_at(cast_rgba_slot, cast_rgba_image));
            EXPECT(cast_bgrx_image.format == bgrx_image.format && cast_bgra_image.format == bgrx_image.format &&
                   cast_rgbx_image.format == bgrx_image.format && cast_rgba_image.format == bgrx_image.format);
            // The T#'s format code is the same for both channel orders, so the swizzle carries the view's order.
            EXPECT(cast_bgrx_image.swizzle == bgrx_image.swizzle && cast_bgra_image.swizzle == bgra_image.swizzle);
            EXPECT(cast_rgba_image.swizzle == bgra_image.swizzle);  // B8G8R8A8 view of R8G8B8A8, no B8G8R8X8 involved
            EXPECT(cast_rgbx_image.swizzle == bgrx_image.swizzle);
            cast_view.Format = DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
            for (ID3D12Resource* listed : {bgr_textures[2].p, bgr_textures[3].p, bgr_textures[4].p, bgr_textures[5].p}) {
                std::memset(view_bytes(null_slot), 0x77, step);
                device->CreateShaderResourceView(listed, &cast_view, view_slot(null_slot));
                EXPECT(null_at(null_slot));
            }
        }
        Ref<ID3D12Resource> bgrx_typeless;
        D3D12_RESOURCE_DESC bgrx_typeless_desc = texture_desc;
        bgrx_typeless_desc.Format = DXGI_FORMAT_B8G8R8X8_TYPELESS;
        bgrx_typeless_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;  // allowed, but no format gives a UAV
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &bgrx_typeless_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&bgrx_typeless.p))));
        D3D12_SHADER_RESOURCE_VIEW_DESC bgr_view = flat_view;
        bgr_view.Format = DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
        device->CreateShaderResourceView(bgrx_typeless.p, &bgr_view, view_slot(typed_bgrx_slot));
        g::Image typed_bgrx_image{};
        EXPECT(image_at(typed_bgrx_slot, typed_bgrx_image));
        EXPECT(typed_bgrx_image.format == g::format_from_vulkan(VK_FORMAT_B8G8R8A8_SRGB));
        EXPECT(typed_bgrx_image.swizzle == bgrx_image.swizzle && typed_bgrx_image.address != bgrx_image.address);
        bgr_view.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        for (ID3D12Resource* x8 : {bgr_textures[0].p, bgrx_typeless.p}) {
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateShaderResourceView(x8, &bgr_view, view_slot(null_slot));
            EXPECT(null_at(null_slot));
        }
        bgr_view.Format = DXGI_FORMAT_B8G8R8X8_UNORM;
        std::memset(view_bytes(null_slot), 0x77, step);
        device->CreateShaderResourceView(bgr_textures[1].p, &bgr_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        D3D12_UNORDERED_ACCESS_VIEW_DESC bgrx_write{};
        bgrx_write.Format = DXGI_FORMAT_B8G8R8X8_UNORM;
        bgrx_write.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        const D3D12_UNORDERED_ACCESS_VIEW_DESC* const bgrx_uavs[2] = {nullptr, &bgrx_write};
        for (const D3D12_UNORDERED_ACCESS_VIEW_DESC* uav : bgrx_uavs) {
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateUnorderedAccessView(bgrx_typeless.p, nullptr, uav, view_slot(null_slot));
            EXPECT(null_at(null_slot));
        }
        bgrx_typeless.reset();
        cube.reset();
        writable.reset();

        // A packed RGB9E5 SRV uses the same four-byte copy footprint as RGBA8. Its shared exponent
        // changes on each diagonal. The last pattern permutes the mantissas to detect channel swaps.
        Ref<ID3D12Resource> packed;
        D3D12_RESOURCE_DESC packed_desc = texture_desc;
        packed_desc.Format = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &packed_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&packed.p))));
        constexpr UINT packed_slot = cube_slot;  // The cube was released above.
        device->CreateShaderResourceView(packed.p, nullptr, view_slot(packed_slot));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT packed_placed{};
        device->GetCopyableFootprints(&packed_desc, 0, 1, 4096, &packed_placed, nullptr, nullptr, nullptr);
        EXPECT(packed_placed.Offset == 4096 && packed_placed.Footprint.RowPitch == D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
        constexpr auto rgb9e5 = [](UINT r, UINT g, UINT b, UINT e) { return r | (g << 9) | (b << 18) | (e << 27); };
        constexpr UINT packed_values[] = {rgb9e5(64, 128, 256, 14), rgb9e5(64, 128, 256, 15),
                                          rgb9e5(64, 128, 256, 16), rgb9e5(256, 64, 128, 14)};
        for (UINT y = 0; y < size; ++y) {
            auto* row = static_cast<std::uint8_t*>(mapped) + packed_placed.Offset + y * packed_placed.Footprint.RowPitch;
            for (UINT x = 0; x < size; ++x)
                std::memcpy(row + x * 4, &packed_values[(x + y) % 4], 4);
        }

        // OPTIONS8 applies to the top-level mip. The four BC1 blocks cover 4x4, 3x4, 4x1 and 3x1 texels.
        constexpr UINT bc_width = 7, bc_height = 5;
        Ref<ID3D12Resource> bc;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT bc_placed{};
        if (options8.UnalignedBlockTexturesSupported) {
            auto bc_desc = texture_desc;
            bc_desc.Width = bc_width;
            bc_desc.Height = bc_height;
            bc_desc.Format = DXGI_FORMAT_BC1_UNORM;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &bc_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&bc.p))));
            device->CreateShaderResourceView(bc.p, nullptr, view_slot(null_slot));
            device->GetCopyableFootprints(&bc_desc, 0, 1, 8192, &bc_placed, nullptr, nullptr, nullptr);
            EXPECT(bc_placed.Offset == 8192 && bc_placed.Footprint.RowPitch == D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
            // Endpoint 0 is RGB565 red, green, blue or white. Every two-bit selector picks that endpoint.
            constexpr std::uint64_t blocks[2][2] = {{0xf800, 0x07e0}, {0x001f, 0xffff}};
            for (UINT row = 0; row < 2; ++row)
                std::memcpy(static_cast<std::uint8_t*>(mapped) + bc_placed.Offset + row * bc_placed.Footprint.RowPitch,
                            blocks[row], sizeof(blocks[row]));
        }

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        for (UINT f = 0; f < 2; ++f) {
            D3D12_TEXTURE_COPY_LOCATION to{}, from{};
            to.pResource = byte_textures[f].p;
            to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            from.pResource = upload.p;
            from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            from.PlacedFootprint = byte_placed[f];
            list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            if (byte_formats[f] == DXGI_FORMAT_A8_UNORM) {
                to.SubresourceIndex = 1;
                from.PlacedFootprint = a8_mip_placed;
                list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            }
            const auto ready = transition(to.pResource, D3D12_RESOURCE_STATE_COPY_DEST,
                                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &ready);
        }
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        // Upload, then the texture into a shader-resource state. Executing the list also brings set 0 up to date.
        D3D12_TEXTURE_COPY_LOCATION to_texture{}, from_upload{};
        to_texture.pResource = texture.p;
        to_texture.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        from_upload.pResource = upload.p;
        from_upload.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        for (UINT mip = 0; mip < 3; ++mip) {
            to_texture.SubresourceIndex = mip;
            from_upload.PlacedFootprint = levels[mip];
            list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_upload, nullptr);
            // The first row of the same upload supplies the full 1D mip.
            D3D12_TEXTURE_COPY_LOCATION to_line = to_texture;
            to_line.pResource = line.p;
            from_upload.PlacedFootprint.Footprint.Height = 1;
            list->CopyTextureRegion(&to_line, 0, 0, 0, &from_upload, nullptr);
        }
        to_texture.SubresourceIndex = 0;
        to_texture.pResource = uint_texture.p;
        from_upload.PlacedFootprint = uint_placed;
        list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_upload, nullptr);
        to_texture.pResource = packed.p;
        from_upload.PlacedFootprint = packed_placed;
        list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_upload, nullptr);
        if (bc.p) {
            to_texture.pResource = bc.p;
            from_upload.PlacedFootprint = bc_placed;
            list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_upload, nullptr);
            const auto barrier =
                transition(bc.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &barrier);
        }
        for (UINT f = 0; f < bgr_count; ++f) {
            to_texture.pResource = bgr_textures[f].p;
            from_upload.PlacedFootprint = bgr_placed[f];
            list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_upload, nullptr);
        }
        to_texture.pResource = volume.p;
        for (UINT mip = 0; mip < 2; ++mip) {
            to_texture.SubresourceIndex = mip;
            from_upload.PlacedFootprint = volume_placed[mip];
            list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_upload, nullptr);
        }
        to_texture.pResource = typeless_volume.p;
        to_texture.SubresourceIndex = 1;
        from_upload.PlacedFootprint = volume_placed[1];
        list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_upload, nullptr);
        const D3D12_RESOURCE_BARRIER sampled[] = {
            transition(typeless_volume.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(texture.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(line.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(packed.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(uint_texture.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(volume.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(bgr_textures[0].p, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(bgr_textures[1].p, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(bgr_textures[2].p, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(bgr_textures[3].p, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(bgr_textures[4].p, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(bgr_textures[5].p, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)};
        list->ResourceBarrier(UINT(std::size(sampled)), sampled);
        to_texture.pResource = array_texture.p;
        for (UINT sub = 0; sub < std::size(array_levels); ++sub) {
            to_texture.SubresourceIndex = sub;
            from_upload.PlacedFootprint = array_levels[sub];
            list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_upload, nullptr);
        }
        const auto array_ready = transition(array_texture.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &array_ready);
        if (ms_texture.p) {
            const D3D12_RESOURCE_BARRIER ms_sampled[] = {
                transition(ms_texture.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition(ms_array.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)};
            list->ResourceBarrier(UINT(std::size(ms_sampled)), ms_sampled);
        }
#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
        struct IntegerMS {
            DXGI_FORMAT format;
            VkFormat vk_format;
            const char* shader;
            std::uint32_t value;
            UINT slots[2];
            Ref<ID3D12Resource> textures[2];  // single layer and two-layer array
            SampleKeys keys[4]{};             // two hits and a missed key in each dimension
        };
        IntegerMS integer_ms[] = {
            {DXGI_FORMAT_R32_UINT, VK_FORMAT_R32_UINT, LEONIDA_TEST_MS_UINT_SPV, 0xf1234567u,
             {ms_uint_slot, ms_uint_array_slot}},
            {DXGI_FORMAT_R32_SINT, VK_FORMAT_R32_SINT, LEONIDA_TEST_MS_SINT_SPV, std::uint32_t(-1234567),
             {ms_sint_slot, ms_sint_array_slot}}};
        for (auto& test : integer_ms) {
            VkImageFormatProperties props{};
            const auto query = d.instance_fn<PFN_vkGetPhysicalDeviceImageFormatProperties>("vkGetPhysicalDeviceImageFormatProperties");
            constexpr auto usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            if (query(d.physical, test.vk_format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage, 0, &props) != VK_SUCCESS ||
                !(props.sampleCounts & (VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT))) {
                std::printf("MSAA SRV format %u: no 2x or 4x sampled render target, skipped\n", UINT(test.format));
                continue;
            }
            D3D12_FEATURE_DATA_FORMAT_SUPPORT integer_support{test.format};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &integer_support, sizeof(integer_support))));
            if (!(integer_support.Support1 & D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD)) {
                std::printf("MSAA SRV format %u: device does not advertise multisample loads, skipped\n",
                            UINT(test.format));
                continue;
            }
            D3D12_RESOURCE_DESC desc = ms_desc;
            desc.Format = test.format;
            desc.SampleDesc.Count = (props.sampleCounts & VK_SAMPLE_COUNT_4_BIT) ? 4 : 2;
            for (UINT shape = 0; shape < 2; ++shape) {
                desc.DepthOrArraySize = UINT16(1 + shape);
                auto& resource = test.textures[shape];
                EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                             D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource.p))));
                const UINT at = test.slots[shape];
                device->CreateShaderResourceView(resource.p, nullptr, view_slot(at));
                EXPECT(!null_at(at));
                std::memcpy(test.keys[shape].tsharp, view_bytes(at), sizeof(test.keys[shape].tsharp));
                test.keys[shape].mode = test.keys[2 + shape].mode = 2 + shape;
                test.keys[shape].first = texels * shape;
                test.keys[2 + shape].first = texels * (2 + shape);
                test.keys[shape].layer = shape;
                const auto barrier = transition(resource.p, D3D12_RESOURCE_STATE_COMMON,
                                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                list->ResourceBarrier(1, &barrier);
            }
        }
#endif
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

        // Set 0 now holds what the views and the sampler named. Four runs of the shader: the view's own slot, its
        // copy, the slot of bytes no view was made for, and the packed RGB view; each with the sampler's S#.
        // Read again before each use: an ExecuteCommandLists that grows the table makes a new set (D3D-314).
        const VkDescriptorSet set0 = rt::d3d12::bindless_set(device.p);
        const VkDescriptorSetLayout set0_layout = rt::d3d12::bindless_set_layout(device.p);
        EXPECT(set0 && set0_layout);
#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
        const UINT tables[] = {own, copied, unknown, packed_slot};
        SampleKeys keys[std::size(tables)] = {};
        for (UINT k = 0; k < std::size(tables); ++k) {
            std::memcpy(keys[k].tsharp, view_bytes(tables[k]), sizeof(keys[k].tsharp));
            std::memcpy(keys[k].ssharp, sampler_bytes, sizeof(keys[k].ssharp));
            keys[k].first = k * texels;
        }
        EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, keys, results.data()));
        sampled_through_set = true;
        {
            const UINT slots[] = {line_default, line_mip, line_clamped, unknown};
            SampleKeys line_keys[std::size(slots)]{};
            for (UINT k = 0; k < std::size(slots); ++k) {
                std::memcpy(line_keys[k].tsharp, view_bytes(slots[k]), sizeof(line_keys[k].tsharp));
                std::memcpy(line_keys[k].ssharp, sampler_bytes, sizeof(line_keys[k].ssharp));
                line_keys[k].first = k * texels;
                line_keys[k].mode = 4;  // Sample from the 1D binding, including on a lookup miss.
            }
            std::array<std::uint8_t, std::size(slots) * bytes> pixels{};
            EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, line_keys, pixels.data()));
            UINT matched[std::size(slots)]{};
            for (UINT i = 0; i < texels; ++i) {
                using Pixel = std::array<std::uint8_t, 4>;
                const Pixel expected[] = {texel(i % size, 0), mip_colors[0],
                                          d.image_view_min_lod ? mip_colors[1] : Pixel{}, Pixel{}};
                for (UINT k = 0; k < std::size(slots); ++k)
                    matched[k] += std::memcmp(pixels.data() + k * bytes + 4 * i, expected[k].data(), 4) == 0;
            }
            std::printf("1D SRVs (default, mip 1, min LOD 2, missing): %u/%u %u/%u %u/%u %u/%u texels matched\n",
                        matched[0], texels, matched[1], texels, matched[2], texels, matched[3], texels);
            for (UINT count : matched) EXPECT(count == texels);
        }
        {
            // The registered literal S# wraps (the texture, texel for texel); the other one clamps as the fallback
            // sampler does (the right edge).
            SampleKeys literal_keys[2] = {keys[0], keys[0]};
            std::memcpy(literal_keys[0].ssharp, literal_words.data(), sizeof(literal_keys[0].ssharp));
            std::memcpy(literal_keys[1].ssharp, unregistered_words.data(), sizeof(literal_keys[1].ssharp));
            literal_keys[1].first = texels;
            std::array<std::uint8_t, 2 * bytes> literal_pixels{};
            EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, literal_keys, literal_pixels.data()));
            UINT wrapped = 0, clamped = 0;
            for (UINT i = 0; i < texels; ++i) {
                wrapped += std::memcmp(literal_pixels.data() + 4 * i, texel(i % size, i / size).data(), 4) == 0;
                clamped += std::memcmp(literal_pixels.data() + bytes + 4 * i, texel(size - 1, i / size).data(), 4) == 0;
            }
            std::printf("literal S#s: registered %u/%u texels wrapped, unregistered %u/%u clamped\n", wrapped, texels,
                        clamped, texels);
            EXPECT(wrapped == texels && clamped == texels);

            // The static samplers' S#s: the three wrapping ones give the texture, texel for texel, as the registered
            // literal does. Unregistered (CreateRootSignature dropping them) they would clamp like the second above.
            for (const StaticCase& test : static_cases) {
                if (!test.created || test.uint_white_border) continue;
                SampleKeys key = keys[0];
                std::memcpy(key.ssharp, test.words.data(), sizeof(key.ssharp));
                std::array<std::uint8_t, bytes> pixels{};
                EXPECT(sample_through_set(d, set0, set0_layout, {&key, 1}, pixels.data()));
                UINT static_wrapped = 0, static_clamped = 0;
                for (UINT i = 0; i < texels; ++i) {
                    static_wrapped += std::memcmp(pixels.data() + 4 * i, texel(i % size, i / size).data(), 4) == 0;
                    static_clamped += std::memcmp(pixels.data() + 4 * i, texel(size - 1, i / size).data(), 4) == 0;
                }
                std::printf("static sampler, root signature %s: %u/%u texels wrapped (%u would match a clamp)\n",
                            test.name, static_wrapped, texels, static_clamped);
                EXPECT(static_wrapped == texels && static_clamped < texels);
            }

            // With LOD zero requested, opposite bias extremes select the first and last mips.
            for (const auto& words : biased_samplers) {
                SampleKeys key = keys[0];
                std::memcpy(key.ssharp, words.data(), sizeof(key.ssharp));
                std::array<std::uint8_t, bytes> pixels{};
                EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, {&key, 1}, pixels.data()));
                UINT matches = 0;
                for (UINT i = 0; i < texels; ++i) {
                    const auto expected = words == negative_bias ? texel(i % size, i / size) : mip_colors[1];
                    matches += std::memcmp(pixels.data() + 4 * i, expected.data(), 4) == 0;
                }
                std::printf("literal sampler bias %.0f: %u/%u samples match, first RGBA %u,%u,%u,%u\n",
                            g::decode(words).lod_bias, matches, texels, pixels[0], pixels[1], pixels[2], pixels[3]);
                EXPECT(matches == texels);
            }
        }
        if (bc.p) {
            SampleKeys bc_key = keys[0];
            std::memcpy(bc_key.tsharp, view_bytes(null_slot), sizeof(bc_key.tsharp));
            std::array<std::uint8_t, bytes> samples{};
            EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, {&bc_key, 1}, samples.data()));
            constexpr std::uint8_t colors[2][2][4] = {{{255, 0, 0, 255}, {0, 255, 0, 255}},
                                                      {{0, 0, 255, 255}, {255, 255, 255, 255}}};
            for (UINT y = 0; y < size; ++y) {
                for (UINT x = 0; x < size; ++x) {
                    // cs_sample.comp samples at the centres of an 8x8 grid with nearest filtering.
                    const UINT tex_x = (2 * x + 1) * bc_width / (2 * size);
                    const UINT tex_y = (2 * y + 1) * bc_height / (2 * size);
                    EXPECT(std::memcmp(samples.data() + 4 * (y * size + x), colors[tex_y / 4][tex_x / 4], 4) == 0);
                }
            }
            std::printf("OPTIONS8: 64 point samples of a 7x5 BC1 top mip matched\n");
        }

        // The same point sampler requests LOD zero from each view. The clamp is in resource mip space.
        SampleKeys lod_keys[std::size(lod_views)]{};
        std::array<std::uint8_t, std::size(lod_views) * bytes> lod_pixels{};
        for (UINT i = 0; i < std::size(lod_views); ++i) {
            std::memcpy(lod_keys[i].tsharp, view_bytes(first_lod_slot + i), sizeof(lod_keys[i].tsharp));
            std::memcpy(lod_keys[i].ssharp, sampler_bytes, sizeof(lod_keys[i].ssharp));
            lod_keys[i].first = i * texels;
        }
        EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, lod_keys, lod_pixels.data()));
        // A fractional image-view minimum LOD is not exact on every driver: Vulkan defines the view's minimum as
        // minLodFloat (preferred) or minLodInteger = floor(minLodFloat) (alternative), and an implementation may use
        // either (Vulkan spec, "Image Level(s) Selection" and VkImageViewMinLodCreateInfoEXT). The UHD 770 floors it.
        // So a view with a fractional clamp inside its mip range passes if all of its texels follow one of the two
        // rules; the rest of the views have one answer either way.
        std::array<UINT, std::size(lod_views)> lod_errors{}, lod_errors_floored{};
        std::array<bool, std::size(lod_views)> lod_fractional{};
        char lod_first[std::size(lod_views)][160]{};
        for (UINT i = 0; i < texels; ++i) {
            using Pixel = std::array<std::uint8_t, 4>;
            const Pixel expected[] = {
                d.image_view_min_lod ? texel(i % size, i / size) : Pixel{},
                d.image_view_min_lod ? mip_colors[0] : Pixel{},
                d.image_view_min_lod ? mip_colors[1] : Pixel{},
                {}, mip_colors[0], {0, 0, 0, 255}
            };
            // The same views with the minimum floored where the extension is used (views 1 and 2: 0.75 -> 0, 1.75 -> 1).
            const Pixel floored[] = {
                expected[0], d.image_view_min_lod ? texel(i % size, i / size) : Pixel{},
                d.image_view_min_lod ? mip_colors[0] : Pixel{}, expected[3], expected[4], expected[5]
            };
            static_assert(std::size(expected) == std::size(lod_views));
            lod_fractional[1] = lod_fractional[2] = d.image_view_min_lod;
            for (UINT view = 0; view < std::size(lod_views); ++view) {
                const auto* sample = &lod_pixels[view * bytes + 4 * i];
                if (std::memcmp(sample, floored[view].data(), 4)) ++lod_errors_floored[view];
                if (std::memcmp(sample, expected[view].data(), 4) && lod_errors[view]++ == 0) {
                    std::snprintf(lod_first[view], sizeof(lod_first[view]),
                                  "minimum LOD view %u texel %u: got %u %u %u %u, expected %u %u %u %u\n", view, i,
                                  sample[0], sample[1], sample[2], sample[3], expected[view][0], expected[view][1],
                                  expected[view][2], expected[view][3]);
                }
            }
        }
        for (UINT view = 0; view < lod_errors.size(); ++view) {
            if (lod_fractional[view] && lod_errors[view] && !lod_errors_floored[view]) {
                std::printf("2D LOD view %u: the driver floors the fractional minimum LOD (%.2f), as Vulkan allows\n",
                            view, lod_views[view].clamp);
                lod_errors[view] = 0;
            }
            if (lod_errors[view]) {
                std::fputs(lod_first[view], stderr);
                std::fprintf(stderr, "2D LOD view %u: %u/%u mismatched\n", view, lod_errors[view], texels);
            }
            lod_mismatches += lod_errors[view];
        }

        SampleKeys array_keys[std::size(array_clamps)]{};
        for (UINT i = 0; i < std::size(array_keys); ++i) {
            array_keys[i] = keys[0];
            std::memcpy(array_keys[i].tsharp, view_bytes(array_lod_slot + i), sizeof(array_keys[i].tsharp));
            array_keys[i].first = i * texels;
        }
        std::array<std::uint8_t, std::size(array_keys) * bytes> array_pixels{};
        EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, array_keys, array_pixels.data()));
        std::array<UINT, std::size(array_clamps)> array_errors{}, array_errors_floored{};
        char array_first[std::size(array_clamps)][160]{};
        for (UINT i = 0; i < texels; ++i) {
            const UINT layer = 1 + i / 32;
            const std::array<std::uint8_t, 4> zero{};
            const auto mip2 = d.image_view_min_lod ? array_color(layer, 2) : zero;
            const std::array expected{array_color(layer, 1), mip2, mip2, zero};
            // View 1 (minimum 1.75) floors to mip 1 where the driver uses minLodInteger; see the 2D views above.
            const std::array floored{array_color(layer, 1), d.image_view_min_lod ? array_color(layer, 1) : zero, mip2, zero};
            static_assert(expected.size() == std::size(array_clamps));
            for (UINT view = 0; view < expected.size(); ++view) {
                const auto* got = array_pixels.data() + view * bytes + i * 4;
                if (std::memcmp(got, floored[view].data(), 4)) ++array_errors_floored[view];
                if (std::memcmp(got, expected[view].data(), 4) && array_errors[view]++ == 0) {
                    std::snprintf(array_first[view], sizeof(array_first[view]),
                                  "array LOD %.2f, layer %u: got %u,%u,%u,%u; expected %u,%u,%u,%u\n",
                                  array_clamps[view], layer, got[0], got[1], got[2], got[3], expected[view][0],
                                  expected[view][1], expected[view][2], expected[view][3]);
                }
            }
        }
        if (d.image_view_min_lod && array_errors[1] && !array_errors_floored[1]) {
            std::printf("array LOD %.2f: the driver floors the fractional minimum LOD, as Vulkan allows\n", array_clamps[1]);
            array_errors[1] = 0;
        }
        UINT array_mismatches = 0;
        for (UINT view = 0; view < array_errors.size(); ++view) {
            if (array_errors[view]) {
                std::fputs(array_first[view], stderr);
                std::fprintf(stderr, "array LOD %.2f: %u/%u mismatched\n", array_clamps[view], array_errors[view], texels);
            }
            array_mismatches += array_errors[view];
        }
        lod_mismatches += array_mismatches;
        std::printf("array minimum LOD: %u/%u samples matched across two layers and four views\n",
                    UINT(array_pixels.size() / 4) - array_mismatches, UINT(array_pixels.size() / 4));

        SampleKeys cube_keys[std::size(cube_lods)]{};
        std::array<std::uint8_t, std::size(cube_lods) * bytes> cube_pixels{};
        for (UINT i = 0; i < std::size(cube_lods); ++i) {
            cube_keys[i] = keys[0];
            std::memcpy(cube_keys[i].tsharp, view_bytes(cube_lod_slot + i), sizeof(cube_keys[i].tsharp));
            cube_keys[i].layer = cube_lods[i].faces;
            cube_keys[i].first = i * texels;
        }
        EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, cube_keys, cube_pixels.data()));
        UINT cube_errors = 0;
        for (UINT view = 0; view < std::size(cube_lods); ++view) {
            const auto& v = cube_lods[view];
            UINT errors = 0;
            for (UINT i = 0; i < texels; ++i) {
                const UINT face = v.first_face + i % v.faces;
                const auto want = v.minimum_mip == 1 || d.image_view_min_lod
                    ? array_color(face, v.minimum_mip) : std::array<std::uint8_t, 4>{};
                const auto* sample = cube_pixels.data() + (view * texels + i) * 4;
                if (std::memcmp(sample, want.data(), 4) && errors++ == 0)
                    std::fprintf(stderr, "cube view %u, face %u, min mip %u: got %u,%u,%u,%u; expected %u,%u,%u,%u\n",
                                 view, face, v.minimum_mip, sample[0], sample[1], sample[2], sample[3],
                                 want[0], want[1], want[2], want[3]);
            }
            cube_errors += errors;
        }
        lod_mismatches += cube_errors;
        std::printf("cube minimum LOD: %u/%u samples matched across cube and cube-array views\n",
                    UINT(cube_pixels.size() / 4) - cube_errors, UINT(cube_pixels.size() / 4));

        const D3D12_CPU_DESCRIPTOR_HANDLE volume_slots[] = {view_slot(volume_all), view_slot(volume_mip),
                                                            linear_slot, srgb_slot};
        SampleKeys volumes[std::size(volume_slots)]{};
        std::array<std::uint8_t, std::size(volume_slots) * bytes> volume_pixels{};
        for (UINT view = 0; view < std::size(volume_slots); ++view) {
            std::memcpy(volumes[view].tsharp, reinterpret_cast<const void*>(volume_slots[view].ptr),
                        sizeof(volumes[view].tsharp));
            std::memcpy(volumes[view].ssharp, sampler_bytes, sizeof(volumes[view].ssharp));
            volumes[view].first = view * texels;
        }
        EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, volumes, volume_pixels.data()));
        unsigned volume_errors = 0;
        for (UINT view = 0; view < std::size(volume_slots); ++view)
            for (UINT i = 0; i < texels; ++i) {
                const bool mip = volume_slots[view].ptr != view_slot(volume_all).ptr;
                const UINT x = i % 4, y = (i / 4) % 4, z = i / 16;
                std::uint8_t expected[]{
                    std::uint8_t((mip ? x : 2 * x + 1) * 19), std::uint8_t((mip ? y : 2 * y + 1) * 23),
                    std::uint8_t((mip ? z / 2 : z) * 53), std::uint8_t(mip ? 101 : 237)};
                const bool srgb = volume_slots[view].ptr == srgb_slot.ptr;
                if (srgb) {
                    // The inverse sRGB transfer function, rounded to bytes, for the uploaded mip-1 RGB values.
                    constexpr std::uint8_t red[] = {0, 2, 5, 10}, green[] = {0, 2, 7, 15}, blue[] = {0, 9};
                    expected[0] = red[x];
                    expected[1] = green[y];
                    expected[2] = blue[z / 2];
                }
                const auto* sample = &volume_pixels[(view * texels + i) * 4];
                // Allow a byte of sRGB conversion rounding. UNORM and alpha are exact.
                bool differs = sample[3] != expected[3];
                for (UINT c = 0; c < 3; ++c)
                    differs |= std::abs(int(sample[c]) - expected[c]) > (srgb ? 1 : 0);
                if (differs) ++volume_errors;
            }
        std::printf("volume SRVs: %u of %zu sampled RGBA texels differ\n", volume_errors, std::size(volume_slots) * texels);
        EXPECT(volume_errors == 0);

        SampleKeys volume_lod_keys[volume_lod_count]{};
        std::array<std::uint8_t, volume_lod_count * bytes> volume_lod_pixels{};
        for (UINT i = 0; i < volume_lod_count; ++i) {
            std::memcpy(volume_lod_keys[i].tsharp, view_bytes(volume_lod_slot + i), sizeof(volume_lod_keys[i].tsharp));
            std::memcpy(volume_lod_keys[i].ssharp, sampler_bytes, sizeof(volume_lod_keys[i].ssharp));
            volume_lod_keys[i].first = i * texels;
        }
        EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, volume_lod_keys, volume_lod_pixels.data()));
        UINT volume_lod_errors = 0;
        UINT volume_clamp_diffs = 0;
        for (UINT i = 0; i < texels; ++i) {
            using Pixel = std::array<std::uint8_t, 4>;
            const UINT x = i % 4, y = (i / 4) % 4, z = i / 16;
            const Pixel mip1{std::uint8_t(x * 19), std::uint8_t(y * 23), std::uint8_t((z / 2) * 53), 101};
            const Pixel expected[volume_lod_count] = {
                d.image_view_min_lod ? mip1 : Pixel{}, mip1, {}, {0, 0, 0, 255},
            };
            for (UINT view = 0; view < volume_lod_count; ++view) {
                const auto* sample = &volume_lod_pixels[view * bytes + 4 * i];
                if (std::memcmp(sample, expected[view].data(), 4) && volume_lod_errors++ == 0) {
                    std::fprintf(stderr, "volume LOD view %u texel %u: got %u,%u,%u,%u; expected %u,%u,%u,%u\n",
                                 view, i, sample[0], sample[1], sample[2], sample[3],
                                 expected[view][0], expected[view][1], expected[view][2], expected[view][3]);
                }
            }
            // Negative control: clamp 1.0 vs unclamped volume_all (clamp 0.0) must differ at every texel.
            if (std::memcmp(&volume_lod_pixels[4 * i], &volume_pixels[4 * i], 4) != 0) ++volume_clamp_diffs;
        }
        lod_mismatches += volume_lod_errors;
        std::printf("volume minimum LOD: %u/%u samples matched across %u views (clamp 1.0 vs 0.0: %u/%u texels differed)\n",
                    UINT(volume_lod_pixels.size() / 4) - volume_lod_errors, UINT(volume_lod_pixels.size() / 4),
                    volume_lod_count, volume_clamp_diffs, texels);
        EXPECT(volume_lod_errors == 0);
        if (d.image_view_min_lod) EXPECT(volume_clamp_diffs == texels);

        SampleKeys byte_keys[byte_view_count]{};
        for (UINT k = 0; k < byte_view_count; ++k) {
            std::memcpy(byte_keys[k].tsharp, byte_bytes(k), sizeof(byte_keys[k].tsharp));
            std::memcpy(byte_keys[k].ssharp, sampler_bytes, sizeof(byte_keys[k].ssharp));
            byte_keys[k].first = k * texels;
        }
        std::array<std::uint8_t, byte_view_count * bytes> byte_results{};
        EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, byte_keys, byte_results.data()));
        for (UINT i = 0; i < texels; ++i) {
            const auto value = std::uint8_t(1 + 3 * i);
            const std::uint8_t expected[4][4] = {{0, 0, 0, value}, {value, 0, 255, 0},
                                                 {value, 0, 0, 255}, {255, value, 255, 0}};
            for (UINT k = 0; k < 4; ++k)
                EXPECT(std::memcmp(byte_results.data() + k * bytes + 4 * i, expected[k], 4) == 0);
            const UINT mip_texel = (i / size / 2) * mip_size + (i % size) / 2;
            const std::uint8_t mip_expected[] = {0, 0, 0, std::uint8_t(208 + mip_texel)};
            EXPECT(std::memcmp(byte_results.data() + a8_mip_slot * bytes + 4 * i, mip_expected, 4) == 0);
        }
        std::printf("A8 mip 1: %u samples matched across %u texels\n", texels, mip_size * mip_size);
        std::printf("A8 and R8 default and remapped views: %u texels matched each\n", texels);

        // A gather across the center of each 2x2 block uses the A8 view's R-to-A swizzle.
        // The remapped view of the same image forces alpha to zero.
        std::array<std::uint8_t, 2 * bytes> gather_results{};
        EXPECT(sample_through_set(d, set0, set0_layout, std::span(byte_keys).first(2), gather_results.data(),
                                  LEONIDA_TEST_GATHER_ALPHA_SPV));
        for (UINT y = 0; y + 1 < size; ++y) {
            for (UINT x = 0; x + 1 < size; ++x) {
                const UINT i = y * size + x;
                const std::uint8_t alpha[] = {std::uint8_t(1 + 3 * (i + size)),
                                              std::uint8_t(1 + 3 * (i + size + 1)),
                                              std::uint8_t(1 + 3 * (i + 1)), std::uint8_t(1 + 3 * i)};
                const std::uint8_t zero[4]{};
                EXPECT(std::memcmp(gather_results.data() + 4 * i, alpha, sizeof(alpha)) == 0);
                EXPECT(std::memcmp(gather_results.data() + bytes + 4 * i, zero, sizeof(zero)) == 0);
            }
        }
        std::printf("A8 alpha gather: %u footprints matched for both default and remapped views\n",
                    (size - 1) * (size - 1));

        // The border samplers, one tile to the right: every texel is the border.
        SampleKeys border_keys[3] = {};
        const UINT border_views[3] = {uint_slot, uint_slot, own};
        const UINT border_samplers[3] = {uint_black, uint_white, float_white};
        for (UINT k = 0; k < 3; ++k) {
            std::memcpy(border_keys[k].tsharp, view_bytes(border_views[k]), sizeof(border_keys[k].tsharp));
            std::memcpy(border_keys[k].ssharp, sampler_at(border_samplers[k]), sizeof(border_keys[k].ssharp));
            border_keys[k].first = k * texels;
            border_keys[k].mode = border_views[k] == uint_slot;
        }
        EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, border_keys, borders.data()));

        if (ms_texture.p) {
            EXPECT(seed_ms_images(d, rt::d3d12::vulkan_image(ms_texture.p), rt::d3d12::vulkan_image(ms_array.p)));
            // The last key uses the non-array binding to read layer zero of the array.
            const struct { UINT slot, mode, view_layer, seed_layer; } ms_reads[] = {
                {ms_slot, 2, 0, 0}, {ms_array_slot, 3, 0, 1}, {ms_array_slot, 3, 1, 2}, {ms_array_slot, 3, 2, 3},
                {ms_slice_slot, 3, 0, 2}, {ms_slice_slot, 3, 1, 3}, {ms_first_layer_slot, 2, 0, 1}};
            SampleKeys ms_keys[std::size(ms_reads)]{};
            for (UINT k = 0; k < std::size(ms_keys); ++k) {
                const auto type = ms_reads[k].mode == 2 ? g::ImageType::image_2d_msaa
                                                        : g::ImageType::image_2d_msaa_array;
                EXPECT(image_at(ms_reads[k].slot, image) && image.type == type);
                std::memcpy(ms_keys[k].tsharp, view_bytes(ms_reads[k].slot), sizeof(ms_keys[k].tsharp));
                ms_keys[k].first = k * texels;
                ms_keys[k].mode = ms_reads[k].mode;
                ms_keys[k].layer = ms_reads[k].view_layer;
            }
            std::array<std::uint8_t, std::size(ms_keys) * bytes> ms_results{};
            EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, ms_keys, ms_results.data(), LEONIDA_TEST_MS_FLOAT_SPV));
            for (UINT k = 0; k < std::size(ms_keys); ++k)
                for (UINT i = 0; i < texels; ++i) {
                    const std::uint8_t expected[] = {std::uint8_t(i / 4 % 4), std::uint8_t(i / 16),
                                                     std::uint8_t(4 * ms_reads[k].seed_layer + i % 4 + 1), 255};
                    const auto* got = ms_results.data() + 4 * (k * texels + i);
                    if (std::memcmp(got, expected, 4) != 0) {
                        std::fprintf(stderr, "MSAA view %u layer %u texel (%u,%u) sample %u: got %u,%u,%u,%u, "
                                     "expected %u,%u,%u,%u\n", ms_reads[k].slot, ms_reads[k].view_layer, i / 4 % 4, i / 16, i % 4,
                                     got[0], got[1], got[2], got[3], expected[0], expected[1], expected[2], expected[3]);
                        return 1;
                    }
                }
            std::printf("MSAA SRVs: %zu per-sample values matched through 2D, full-array and sliced-array views\n",
                        std::size(ms_keys) * texels);
        }

        for (auto& test : integer_ms) {
            if (!test.textures[0].p) continue;
            EXPECT(d.run([&](VkCommandBuffer cb) {
                for (UINT shape = 0; shape < 2; ++shape) {
                    VkClearColorValue color{};
                    color.uint32[0] = test.value + shape;
                    const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 + shape};
                    d.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(cb,
                        rt::d3d12::vulkan_image(test.textures[shape].p), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &all);
                }
                VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &ready, 0, nullptr, 0, nullptr);
                return true;
            }));
            std::array<std::uint32_t, 4 * texels> values{};
            for (UINT shape = 0; shape < 2; ++shape)
                EXPECT(std::memcmp(test.keys[shape].tsharp, view_bytes(test.slots[shape]), sizeof(test.keys[shape].tsharp)) == 0);
            EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, test.keys, values.data(), test.shader));
            for (UINT k = 0; k < 4; ++k) {
                const auto expected = k < 2 ? test.value + k : 0u;
                for (UINT i = 0; i < texels; ++i) EXPECT(values[k * texels + i] == expected);
            }
            for (auto& resource : test.textures) resource.reset();
            std::printf("MSAA format %u: 2D and array integer reads and zero fallbacks matched\n", UINT(test.format));
        }

        // More than 4096 SRVs and more than a full shader-visible sampler heap (2048): 5000 distinct views of the
        // texture (mip ranges by swizzles), then one more that samples as the texture, and 2100 distinct wrapping
        // samplers (MaxLOD apart). The last view through the last sampler must give the texture, one tile to the
        // right; a fallback view or the fallback (clamping) sampler would not. A set with less room than that
        // (MoltenVK's keeps the default 4096 2D images and 1024 samplers, docs/d3d12_vulkan.md section 5) gets as
        // many as fill each short array exactly: its last view and sampler must still give the texture, and one
        // more of each must be refused and read as the fallbacks.
        {
            constexpr UINT wanted_views = 5000, wanted_samplers = 2100, heap_samplers = 2048;
            const auto room = rt::d3d12::bindless_room(device.p);
            EXPECT(room.images_2d >= 2 && room.samplers >= 1);
            const bool views_capped = room.images_2d < wanted_views + 1, samplers_capped = room.samplers < wanted_samplers;
            const UINT many_views = views_capped ? room.images_2d - 1 : wanted_views;
            const UINT many_samplers = samplers_capped ? room.samplers : wanted_samplers;
            Ref<ID3D12DescriptorHeap> many_view_heap, many_sampler_heaps[2];
            D3D12_DESCRIPTOR_HEAP_DESC many_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, many_views + 2,
                                                 D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&many_desc, IID_PPV_ARGS(&many_view_heap.p))));
            many_desc = {D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, heap_samplers, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                         0};
            for (auto& heap : many_sampler_heaps)
                EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&many_desc, IID_PPV_ARGS(&heap.p))));
            const auto cpu_start = [&](ID3D12DescriptorHeap* heap) {
                return reinterpret_cast<XboxHeapHandleFn>(slot(heap, 9))(heap);
            };
            const std::uint64_t many_view_cpu = cpu_start(many_view_heap.p);
            std::vector<std::array<std::uint32_t, 8>> tsharps;
            constexpr UINT mip_ranges[][2] = {{0, 1}, {0, 2}, {1, 1}, {1, 2}};
            for (UINT i = 0; i < many_views; ++i) {
                const UINT swizzle = i / UINT(std::size(mip_ranges));
                const UINT c0 = swizzle % 6, c1 = swizzle / 6 % 6, c2 = swizzle / 36 % 6, c3 = swizzle / 216 % 6;
                D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
                srv.Format = sampled_desc.Format;
                srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                srv.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(c0, c1, c2, c3);
                const auto& mips = mip_ranges[i % std::size(mip_ranges)];
                srv.Texture2D = {mips[0], mips[1], 0, 0};
                const D3D12_CPU_DESCRIPTOR_HANDLE at{SIZE_T(many_view_cpu + i * step)};
                device->CreateShaderResourceView(texture.p, &srv, at);
                std::memcpy(tsharps.emplace_back().data(), reinterpret_cast<const void*>(many_view_cpu + i * step), 32);
            }
            // Every mip with alpha forced to one: the texture's texels (alpha 255), under a T# no other view has.
            // The one past it, a short array's extra, forces alpha to zero.
            D3D12_SHADER_RESOURCE_VIEW_DESC last_srv{};
            last_srv.Format = sampled_desc.Format;
            last_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            last_srv.Shader4ComponentMapping =
                D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0, 1, 2, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1);
            last_srv.Texture2D = {0, UINT(-1), 0, 0};
            const std::uint64_t last_view = many_view_cpu + many_views * step, extra_view = last_view + step;
            device->CreateShaderResourceView(texture.p, &last_srv, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(last_view)});
            std::memcpy(tsharps.emplace_back().data(), reinterpret_cast<const void*>(last_view), 32);
            if (views_capped) {
                EXPECT(rt::d3d12::bindless_room(device.p).images_2d == 0);
                last_srv.Shader4ComponentMapping =
                    D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0, 1, 2, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0);
                device->CreateShaderResourceView(texture.p, &last_srv, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(extra_view)});
                std::memcpy(tsharps.emplace_back().data(), reinterpret_cast<const void*>(extra_view), 32);
            }
            std::memcpy(tsharps.emplace_back().data(), view_bytes(own), 32);
            std::sort(tsharps.begin(), tsharps.end());
            EXPECT(std::unique(tsharps.begin(), tsharps.end()) == tsharps.end());  // distinct set entries

            std::vector<std::array<std::uint32_t, 4>> ssharps;
            std::uint64_t last_sampler = 0;
            const UINT created_samplers = many_samplers + (samplers_capped ? 1 : 0);
            for (UINT i = 0; i < created_samplers; ++i) {
                if (i == many_samplers) EXPECT(rt::d3d12::bindless_room(device.p).samplers == 0);
                D3D12_SAMPLER_DESC many_sampler = sampler_desc;
                many_sampler.MaxLOD = float(i + 1) / 256.0f;  // the S# keeps 8 fraction bits; LOD 0 is sampled
                const std::uint64_t at =
                    cpu_start(many_sampler_heaps[i / heap_samplers].p) + (i % heap_samplers) * sampler_step;
                device->CreateSampler(&many_sampler, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(at)});
                std::memcpy(ssharps.emplace_back().data(), reinterpret_cast<const void*>(at), 16);
                if (i < many_samplers) last_sampler = at;
            }
            const std::uint64_t extra_sampler = last_sampler + sampler_step;  // many_samplers < heap_samplers then
            std::memcpy(ssharps.emplace_back().data(), sampler_bytes, 16);
            std::sort(ssharps.begin(), ssharps.end());
            EXPECT(std::unique(ssharps.begin(), ssharps.end()) == ssharps.end());

            // The next ExecuteCommandLists writes them into the set.
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            d.fn<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(d.device);
            const auto in_table = [&](std::uint64_t descriptor, bool sampler) {
                std::uint32_t key[8]{~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
                std::memcpy(key, reinterpret_cast<const void*>(descriptor), sampler ? 16 : 32);
                return rt::d3d12::bindless_hash_lookup(device.p, key) != 0;
            };
            EXPECT(in_table(last_view, false) && in_table(last_sampler, true));
            if (views_capped) EXPECT(!in_table(extra_view, false));
            if (samplers_capped) EXPECT(!in_table(extra_sampler, true));

            // The last view through the last sampler; then a short array's extra in its place, which reads as the
            // fallback: zeros for the view, and the clamping sampler's edge column for the sampler.
            SampleKeys last_keys[3]{};
            std::memcpy(last_keys[0].tsharp, reinterpret_cast<const void*>(last_view), sizeof(last_keys[0].tsharp));
            std::memcpy(last_keys[0].ssharp, reinterpret_cast<const void*>(last_sampler), sizeof(last_keys[0].ssharp));
            UINT key_count = 1;
            if (views_capped) {
                last_keys[key_count] = last_keys[0];
                std::memcpy(last_keys[key_count].tsharp, reinterpret_cast<const void*>(extra_view),
                            sizeof(last_keys[key_count].tsharp));
                last_keys[key_count].first = key_count * texels;
                ++key_count;
            }
            if (samplers_capped) {
                last_keys[key_count] = last_keys[0];
                std::memcpy(last_keys[key_count].ssharp, reinterpret_cast<const void*>(extra_sampler),
                            sizeof(last_keys[key_count].ssharp));
                last_keys[key_count].first = key_count * texels;
                ++key_count;
            }
            std::array<std::uint8_t, 3 * bytes> last_texels{};
            EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, {last_keys, key_count}, last_texels.data()));
            const char* const names[] = {"the last view through the last sampler", "the refused view",
                                         "the refused sampler"};
            UINT differ = 0;
            for (UINT k = 0; k < key_count; ++k) {
                const UINT kind = k == 0 ? 0 : views_capped && k == 1 ? 1 : 2;
                for (UINT i = 0; i < texels; ++i) {
                    const auto want = kind == 0 ? texel(i % size, i / size)
                                      : kind == 1 ? std::array<std::uint8_t, 4>{}
                                                  : texel(size - 1, i / size);
                    const auto* got = last_texels.data() + k * bytes + 4 * i;
                    if (std::memcmp(got, want.data(), 4) != 0 && differ++ < 4)
                        std::fprintf(stderr, "%s, texel %u: got %u,%u,%u,%u; expected %u,%u,%u,%u\n", names[kind], i,
                                     got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                }
            }
            EXPECT(differ == 0);
            if (!views_capped && !samplers_capped)
                std::printf("bindless capacity: the %uth SRV through the %uth sampler gave the texture\n",
                            many_views + 1, many_samplers);
            else
                std::printf("bindless capacity: the set had room for %u more 2D views and %u more samplers, fewer than "
                            "%u and %u: the last of each gave the texture, and one more %s refused and read as the "
                            "fallback\n", room.images_2d, room.samplers, wanted_views + 1, wanted_samplers,
                            views_capped && samplers_capped ? "of each was"
                            : views_capped                  ? "view was"
                                                            : "sampler was");
        }

        // 96-bit textures (D3D-62): R32G32B32_FLOAT holding the texture's red, green and blue as n/255,
        // R32G32B32_UINT holding them as integers, and R32G32B32_TYPELESS holding the floats. Their texels are 12
        // bytes; the layer keeps them in RGBA32 images whose fourth channel no upload writes, and a shader must
        // read alpha as one. Default views of the typed textures, then the TYPELESS one through a FLOAT view and a
        // UINT view (the bits of n/255 as integers: the shader clamps to 255, so 0 stays 0 and the rest is 255).
        {
            constexpr UINT rgb96_count = 3, rgb96_views = 4, rgb96_null = rgb96_views;
            constexpr DXGI_FORMAT rgb96_formats[rgb96_count] = {DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R32G32B32_UINT,
                                                                DXGI_FORMAT_R32G32B32_TYPELESS};
            constexpr UINT rgb96_view_texture[rgb96_views] = {0, 1, 2, 2};
            constexpr bool rgb96_view_uint[rgb96_views] = {false, true, false, true};
            Ref<ID3D12Resource> rgb96_textures[rgb96_count], rgb96_upload;
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT rgb96_placed[rgb96_count]{};
            const auto rgb96_upload_desc = buffer_desc(16384, D3D12_RESOURCE_FLAG_NONE);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &rgb96_upload_desc,
                                                             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                             IID_PPV_ARGS(&rgb96_upload.p))));
            void* rgb96_mapped = nullptr;
            EXPECT(SUCCEEDED(rgb96_upload->Map(0, nullptr, &rgb96_mapped)));
            for (UINT k = 0; k < rgb96_count; ++k) {
                D3D12_RESOURCE_DESC rgb96_desc = texture_desc;
                rgb96_desc.Format = rgb96_formats[k];
                EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rgb96_desc,
                                                                 D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                                 IID_PPV_ARGS(&rgb96_textures[k].p))));
                device->GetCopyableFootprints(&rgb96_desc, 0, 1, k * 4096, &rgb96_placed[k], nullptr, nullptr, nullptr);
                EXPECT(rgb96_placed[k].Footprint.Format == rgb96_formats[k]);
                for (UINT y = 0; y < size; ++y)
                    for (UINT x = 0; x < size; ++x)
                        for (UINT c = 0; c < 3; ++c) {
                            const float as_float = float(texel(x, y)[c]) / 255.0f;
                            const std::uint32_t as_uint = texel(x, y)[c];
                            std::memcpy(static_cast<std::uint8_t*>(rgb96_mapped) + rgb96_placed[k].Offset +
                                            y * rgb96_placed[k].Footprint.RowPitch + x * 12 + c * 4,
                                        k == 1 ? static_cast<const void*>(&as_uint) : static_cast<const void*>(&as_float), 4);
                        }
            }
            rgb96_upload->Unmap(0, nullptr);

            Ref<ID3D12DescriptorHeap> rgb96_heap;
            const D3D12_DESCRIPTOR_HEAP_DESC rgb96_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, rgb96_views + 1,
                                                             D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rgb96_heap_desc, IID_PPV_ARGS(&rgb96_heap.p))));
            const std::uint64_t rgb96_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(rgb96_heap.p, 9))(rgb96_heap.p);
            const auto rgb96_slot = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(rgb96_cpu + i * step)}; };
            const auto rgb96_bytes = [&](UINT i) { return reinterpret_cast<std::uint8_t*>(rgb96_cpu + i * step); };
            D3D12_SHADER_RESOURCE_VIEW_DESC rgb96_view{};
            rgb96_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            rgb96_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            rgb96_view.Texture2D.MipLevels = 1;
            for (UINT k = 0; k < rgb96_views; ++k) {
                rgb96_view.Format = rgb96_view_uint[k] ? DXGI_FORMAT_R32G32B32_UINT : DXGI_FORMAT_R32G32B32_FLOAT;
                device->CreateShaderResourceView(rgb96_textures[rgb96_view_texture[k]].p, k < 2 ? nullptr : &rgb96_view,
                                                 rgb96_slot(k));
                // The T# names the three-channel format (GFX10_FORMAT_32_32_32_UINT is 72, _FLOAT 74).
                leonida::gfx10::ImageWords rgb96_words{};
                std::memcpy(rgb96_words.data(), rgb96_bytes(k), sizeof(rgb96_words));
                leonida::gfx10::Image rgb96_image{};
                EXPECT(leonida::gfx10::is_image(rgb96_words) && leonida::gfx10::decode(rgb96_words, rgb96_image));
                EXPECT(rgb96_image.format == (rgb96_view_uint[k] ? 72u : 74u));
            }
            // A 96-bit texture has no view of a four-channel format, a four-channel one none of a 96-bit format, and
            // a typed 96-bit texture none of another type (only the TYPELESS one is cast).
            const std::pair<ID3D12Resource*, DXGI_FORMAT> wrong[] = {
                {rgb96_textures[0].p, DXGI_FORMAT_R32G32B32A32_FLOAT},
                {texture.p, DXGI_FORMAT_R32G32B32_FLOAT},
                {rgb96_textures[0].p, DXGI_FORMAT_R32G32B32_UINT},
                {rgb96_textures[2].p, DXGI_FORMAT_R32G32B32A32_UINT}};
            for (const auto& [resource, format] : wrong) {
                rgb96_view.Format = format;
                std::memset(rgb96_bytes(rgb96_null), 0x77, step);
                device->CreateShaderResourceView(resource, &rgb96_view, rgb96_slot(rgb96_null));
                EXPECT(std::all_of(rgb96_bytes(rgb96_null), rgb96_bytes(rgb96_null) + step,
                                   [](std::uint8_t byte) { return byte == 0; }));
            }

            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            D3D12_TEXTURE_COPY_LOCATION to_rgb96{}, from_rgb96{};
            to_rgb96.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            from_rgb96.pResource = rgb96_upload.p;
            from_rgb96.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            for (UINT k = 0; k < rgb96_count; ++k) {
                to_rgb96.pResource = rgb96_textures[k].p;
                from_rgb96.PlacedFootprint = rgb96_placed[k];
                list->CopyTextureRegion(&to_rgb96, 0, 0, 0, &from_rgb96, nullptr);
                const auto ready = transition(rgb96_textures[k].p, D3D12_RESOURCE_STATE_COPY_DEST,
                                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                list->ResourceBarrier(1, &ready);
            }
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            d.fn<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(d.device);

            SampleKeys rgb96_keys[rgb96_views] = {};
            for (UINT k = 0; k < rgb96_views; ++k) {
                std::memcpy(rgb96_keys[k].tsharp, rgb96_bytes(k), sizeof(rgb96_keys[k].tsharp));
                std::memcpy(rgb96_keys[k].ssharp, sampler_bytes, sizeof(rgb96_keys[k].ssharp));
                rgb96_keys[k].first = k * texels;
                rgb96_keys[k].mode = rgb96_view_uint[k] ? 1 : 0;
            }
            std::vector<std::uint8_t> rgb96_pixels(rgb96_views * bytes);
            EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, rgb96_keys, rgb96_pixels.data()));
            const char* rgb96_names[rgb96_views] = {"FLOAT", "UINT", "TYPELESS as FLOAT", "TYPELESS as UINT"};
            UINT rgb96_differ = 0;
            for (UINT k = 0; k < rgb96_views; ++k)
                for (UINT i = 0; i < texels; ++i) {
                    std::array<std::uint8_t, 4> want = texel(i % size, i / size);
                    if (k == 3)
                        for (UINT c = 0; c < 3; ++c) want[c] = want[c] ? 255 : 0;
                    want[3] = rgb96_view_uint[k] ? 1 : 255;  // alpha is one: 1 as an integer, 255 as UNORM
                    const std::uint8_t* got = rgb96_pixels.data() + k * bytes + i * 4;
                    if (std::memcmp(got, want.data(), 4) != 0 && rgb96_differ++ < 4)
                        std::fprintf(stderr, "R32G32B32_%s texel %u: got %u,%u,%u,%u, want %u,%u,%u,%u\n", rgb96_names[k],
                                     i, unsigned(got[0]), unsigned(got[1]), unsigned(got[2]), unsigned(got[3]),
                                     unsigned(want[0]), unsigned(want[1]), unsigned(want[2]), unsigned(want[3]));
                }
            EXPECT(rgb96_differ == 0);
        }
        // The B8G8R8X8 and B8G8R8A8 textures through their default views, then the two castable ones through a view
        // of the other family.
        SampleKeys bgr_keys[bgr_count] = {};
        const UINT bgr_slots[bgr_count] = {bgrx_slot,      bgra_slot,      cast_bgrx_slot,
                                           cast_bgra_slot, cast_rgbx_slot, cast_rgba_slot};
        for (UINT k = 0; k < bgr_count; ++k) {
            std::memcpy(bgr_keys[k].tsharp, view_bytes(bgr_slots[k]), sizeof(bgr_keys[k].tsharp));
            std::memcpy(bgr_keys[k].ssharp, sampler_bytes, sizeof(bgr_keys[k].ssharp));
            bgr_keys[k].first = k * texels;
        }
        EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, bgr_keys, bgr_results.data()));
#endif

        // A destroyed texture takes its views out of the set: the next ExecuteCommandLists writes that, and nothing
        // may still name the destroyed image views (the validation layer reports a set that does).
        bc.reset();
        packed.reset();
        texture.reset();
        EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
        EXPECT(list->Close() == S_OK);
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 2)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(2, nullptr)));
#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
        // The view's bytes are still in the slot, but they find nothing any more: the fallback.
        SampleKeys gone = keys[0];
        gone.first = 0;
        EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, {&gone, 1}, after_destruction.data()));
#endif

        struct PackedFormat {
            DXGI_FORMAT dxgi;
            VkFormat vk;
            const char* name;
        };
        constexpr PackedFormat packed_formats[] = {
            {DXGI_FORMAT_B4G4R4A4_UNORM, VK_FORMAT_A4R4G4B4_UNORM_PACK16, "B4G4R4A4"},
            {DXGI_FORMAT_A4B4G4R4_UNORM, VK_FORMAT_R4G4B4A4_UNORM_PACK16, "A4B4G4R4"},
            {DXGI_FORMAT_B5G5R5A1_UNORM, VK_FORMAT_A1R5G5B5_UNORM_PACK16, "B5G5R5A1"}
        };
        UINT64 packed_signal = 2;
        for (const auto& format : packed_formats) {
            // Vulkan requires sampling A4R4G4B4 (1.3) and A1R5G5B5 but not R4G4B4A4: skip A4B4G4R4 where the Vulkan
            // driver can't sample it (the driver's answer, not the layer's, so a dropped mapping still fails).
            if (format.dxgi == DXGI_FORMAT_A4B4G4R4_UNORM) {
                VkFormatProperties props{};
                d.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(
                    d.physical, format.vk, &props);
                if (!(props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
                    std::printf("%s: the Vulkan driver can't sample it, skipped\n", format.name);
                    continue;
                }
            }
            D3D12_RESOURCE_DESC format_desc = texture_desc;
            format_desc.Format = format.dxgi;
            Ref<ID3D12Resource> format_texture;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &format_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&format_texture.p))));
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT format_footprint{};
            UINT64 row_size = 0;
            device->GetCopyableFootprints(&format_desc, 0, 1, 49152, &format_footprint, nullptr, &row_size, nullptr);
            EXPECT(format_footprint.Footprint.Format == format.dxgi && row_size == size * 2 &&
                   format_footprint.Footprint.RowPitch == 256);
            constexpr std::uint16_t b5_words[] = {0xfc00, 0x7c00, 0x83e1, 0xc117};
            constexpr std::array<std::uint8_t, 4> b5_rgba[] = {
                {255, 0, 0, 255}, {255, 0, 0, 0}, {0, 255, 8, 255}, {132, 66, 189, 255}
            };
            std::array<std::array<std::uint8_t, 4>, texels> expected{};
            for (UINT y = 0; y < size; ++y)
                for (UINT x = 0; x < size; ++x) {
                    auto& rgba = expected[y * size + x];
                    std::uint16_t word;
                    if (format.dxgi == DXGI_FORMAT_B5G5R5A1_UNORM) {
                        const UINT i = (x + y) % std::size(b5_words);
                        word = b5_words[i];
                        rgba = b5_rgba[i];
                    } else {
                        const UINT r = x, g = y + 8, b = (x + y) & 15, a = 15 - x;
                        // B4G4R4A4 keeps blue in bits 0-3 and alpha in 12-15, A4B4G4R4 alpha in 0-3 and red in 12-15.
                        word = format.dxgi == DXGI_FORMAT_A4B4G4R4_UNORM
                                   ? std::uint16_t(a | (b << 4) | (g << 8) | (r << 12))
                                   : std::uint16_t(b | (g << 4) | (r << 8) | (a << 12));
                        rgba = {std::uint8_t(r * 17), std::uint8_t(g * 17), std::uint8_t(b * 17), std::uint8_t(a * 17)};
                    }
                    std::memcpy(static_cast<std::uint8_t*>(mapped) + format_footprint.Offset +
                                    y * format_footprint.Footprint.RowPitch + x * 2,
                                &word, 2);
                }
            // Set 0 takes in a view at the next ExecuteCommandLists, so it is made before the upload runs.
            device->CreateShaderResourceView(format_texture.p, nullptr, view_slot(null_slot));
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            D3D12_TEXTURE_COPY_LOCATION to_packed{format_texture.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
            to_packed.SubresourceIndex = 0;
            D3D12_TEXTURE_COPY_LOCATION from_packed{upload.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
            from_packed.PlacedFootprint = format_footprint;
            list->CopyTextureRegion(&to_packed, 0, 0, 0, &from_packed, nullptr);
            const D3D12_RESOURCE_BARRIER readable =
                transition(format_texture.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &readable);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            ++packed_signal;
            EXPECT(SUCCEEDED(queue->Signal(fence.p, packed_signal)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(packed_signal, nullptr)));

            g::Image packed_image{};
            EXPECT(image_at(null_slot, packed_image));
            EXPECT(packed_image.format == g::format_from_vulkan(format.vk) && packed_image.format != 0);
            if (format.dxgi == DXGI_FORMAT_B5G5R5A1_UNORM) EXPECT(packed_image.swizzle == g::channels_bgra);
            if (format.dxgi == DXGI_FORMAT_A4B4G4R4_UNORM) EXPECT(packed_image.swizzle == g::channels_abgr);
#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
            SampleKeys packed_keys{};
            std::memcpy(packed_keys.tsharp, view_bytes(null_slot), sizeof(packed_keys.tsharp));
            std::memcpy(packed_keys.ssharp, sampler_bytes, sizeof(packed_keys.ssharp));
            std::vector<std::uint8_t> packed_results(bytes);
            EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, {&packed_keys, 1}, packed_results.data()));
            for (UINT i = 0; i < texels; ++i) {
                const auto& want = expected[i];
                if (std::memcmp(&packed_results[i * 4], want.data(), 4) != 0) {
                    std::fprintf(stderr, "%s texel %u: got %02x %02x %02x %02x, want %02x %02x %02x %02x\n", format.name, i,
                                 packed_results[i * 4], packed_results[i * 4 + 1], packed_results[i * 4 + 2],
                                 packed_results[i * 4 + 3], want[0], want[1], want[2], want[3]);
                    EXPECT(false);
                }
            }
            std::printf("%s sampled: %u texels matched\n", format.name, texels);
#endif
        }

        // Minimum/maximum filter reduction modes on a 2x2 RGBA8 texture: every sample of cs_sample.comp's 8x8 grid
        // lands strictly between texel centres, so all four texels have positive weight under WRAP addressing.
        {
            D3D12_RESOURCE_DESC minmax_desc = texture_desc;
            minmax_desc.Width = minmax_desc.Height = 2;
            minmax_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            Ref<ID3D12Resource> minmax_texture;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &minmax_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&minmax_texture.p))));
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT minmax_fp{};
            device->GetCopyableFootprints(&minmax_desc, 0, 1, 49152, &minmax_fp, nullptr, nullptr, nullptr);
            constexpr std::uint8_t minmax_texels[2][2][4] = {
                {{16, 240, 80, 255}, {240, 16, 160, 255}},
                {{96, 128, 240, 255}, {176, 64, 16, 255}},
            };
            for (UINT y = 0; y < 2; ++y)
                for (UINT x = 0; x < 2; ++x)
                    std::memcpy(static_cast<std::uint8_t*>(mapped) + minmax_fp.Offset + y * minmax_fp.Footprint.RowPitch +
                                    x * 4,
                                minmax_texels[y][x], 4);
            device->CreateShaderResourceView(minmax_texture.p, nullptr, view_slot(null_slot));

            D3D12_DESCRIPTOR_HEAP_DESC mm_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 3,
                                                    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
            Ref<ID3D12DescriptorHeap> mm_samplers;
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&mm_heap_desc, IID_PPV_ARGS(&mm_samplers.p))));
            const auto mm_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(mm_samplers.p, 9))(mm_samplers.p);
            const auto mm_slot = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(mm_cpu + i * sampler_step)}; };
            const auto mm_bytes = [&](UINT i) { return reinterpret_cast<const std::uint8_t*>(mm_cpu + i * sampler_step); };
            constexpr D3D12_FILTER mm_filters[3] = {
                D3D12_FILTER_MINIMUM_MIN_MAG_MIP_LINEAR,
                D3D12_FILTER_MAXIMUM_MIN_MAG_MIP_LINEAR,
                D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            };
            constexpr g::FilterMode mm_modes[3] = {g::FilterMode::minimum, g::FilterMode::maximum, g::FilterMode::blend};
            for (UINT i = 0; i < 3; ++i) {
                D3D12_SAMPLER_DESC sd = sampler_desc;
                sd.Filter = mm_filters[i];
                sd.AddressU = sd.AddressV = sd.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
                device->CreateSampler(&sd, mm_slot(i));
                g::SamplerWords sw{};
                std::memcpy(sw.data(), mm_bytes(i), sizeof(sw));
                EXPECT(g::decode(sw).filter_mode == mm_modes[i]);
            }

            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            D3D12_TEXTURE_COPY_LOCATION to_mm{minmax_texture.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
            to_mm.SubresourceIndex = 0;
            D3D12_TEXTURE_COPY_LOCATION from_mm{upload.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
            from_mm.PlacedFootprint = minmax_fp;
            list->CopyTextureRegion(&to_mm, 0, 0, 0, &from_mm, nullptr);
            const D3D12_RESOURCE_BARRIER mm_readable =
                transition(minmax_texture.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &mm_readable);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            ++packed_signal;
            EXPECT(SUCCEEDED(queue->Signal(fence.p, packed_signal)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(packed_signal, nullptr)));

#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
            SampleKeys mm_keys[3]{};
            for (UINT i = 0; i < 3; ++i) {
                std::memcpy(mm_keys[i].tsharp, view_bytes(null_slot), sizeof(mm_keys[i].tsharp));
                std::memcpy(mm_keys[i].ssharp, mm_bytes(i), sizeof(mm_keys[i].ssharp));
                mm_keys[i].first = i * texels;
            }
            std::array<std::uint8_t, 3 * bytes> mm_pixels{};
            EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, mm_keys, mm_pixels.data()));
            const auto get_fmt_props =
                d.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties");
            VkFormatProperties rgba8_props{}, bc1_props{};
            get_fmt_props(d.physical, VK_FORMAT_R8G8B8A8_UNORM, &rgba8_props);
            get_fmt_props(d.physical, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, &bc1_props);
            const bool rgba8_minmax =
                d.sampler_filter_minmax &&
                (rgba8_props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT) != 0;
            const bool bc1_minmax =
                d.sampler_filter_minmax &&
                (bc1_props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT) != 0;
            constexpr std::array<std::uint8_t, 4> want_min{16, 16, 16, 255};
            constexpr std::array<std::uint8_t, 4> want_max{240, 240, 240, 255};
            UINT min_matches = 0, max_matches = 0, blend_diff_min = 0, blend_diff_max = 0;
            for (UINT i = 0; i < texels; ++i) {
                const auto* got_min = &mm_pixels[0 * bytes + i * 4];
                const auto* got_max = &mm_pixels[1 * bytes + i * 4];
                const auto* got_blend = &mm_pixels[2 * bytes + i * 4];
                if (rgba8_minmax) {
                    min_matches += std::memcmp(got_min, want_min.data(), 4) == 0;
                    max_matches += std::memcmp(got_max, want_max.data(), 4) == 0;
                    blend_diff_min += std::memcmp(got_blend, want_min.data(), 4) != 0;
                    blend_diff_max += std::memcmp(got_blend, want_max.data(), 4) != 0;
                } else {
                    min_matches += std::memcmp(got_min, got_blend, 4) == 0;
                    max_matches += std::memcmp(got_max, got_blend, 4) == 0;
                }
            }
            std::printf("sampler reduction min/max (RGBA8=%s, BC1=%s): min %u/%u, max %u/%u (blend differed: %u/%u vs "
                        "min, %u/%u vs max)\n",
                        rgba8_minmax ? "yes" : "no", bc1_minmax ? "yes" : "no", min_matches, texels, max_matches,
                        texels, blend_diff_min, texels, blend_diff_max, texels);
            EXPECT(min_matches == texels && max_matches == texels);
            if (rgba8_minmax) EXPECT(blend_diff_min == texels && blend_diff_max == texels);
#endif
        }

        // Cross-family castable SRV: write_image_descriptor must pass resource.has_view_format_list() into a single
        // bind_image call so a listed R8G8B8A8_UINT -> R32_UINT view registers cleanly without a spurious first-pass
        // failure (and without double-binding every view).
        if (d.bindless) {
            Ref<ID3D12Device10> inner10;
            EXPECT(SUCCEEDED(inner->QueryInterface(IID_PPV_ARGS(&inner10.p))));
            D3D12_RESOURCE_DESC1 cast_desc{};
            cast_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            cast_desc.Width = 4;
            cast_desc.Height = 4;
            cast_desc.DepthOrArraySize = 1;
            cast_desc.MipLevels = 1;
            cast_desc.Format = DXGI_FORMAT_R8G8B8A8_UINT;
            cast_desc.SampleDesc = {1, 0};
            cast_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            const DXGI_FORMAT cast_formats[] = {DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R32_UINT};
            Ref<ID3D12Resource> cast_tex;
            EXPECT(SUCCEEDED(inner10->CreateCommittedResource3(
                &default_heap, D3D12_HEAP_FLAG_NONE, &cast_desc, D3D12_BARRIER_LAYOUT_COMMON, nullptr, nullptr,
                2, cast_formats, IID_PPV_ARGS(&cast_tex.p))));
            const D3D12_DESCRIPTOR_HEAP_DESC cast_hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                                     D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            Ref<ID3D12DescriptorHeap> cast_heap;
            EXPECT(SUCCEEDED(inner10->CreateDescriptorHeap(&cast_hd, IID_PPV_ARGS(&cast_heap.p))));
            const D3D12_CPU_DESCRIPTOR_HANDLE cast_slot = cast_heap->GetCPUDescriptorHandleForHeapStart();
            D3D12_SHADER_RESOURCE_VIEW_DESC cast_srv{};
            cast_srv.Format = DXGI_FORMAT_R32_UINT;
            cast_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            cast_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            cast_srv.Texture2D.MipLevels = 1;
            inner10->CreateShaderResourceView(cast_tex.p, &cast_srv, cast_slot);
            g::ImageWords cast_words{};
            std::memcpy(cast_words.data(), reinterpret_cast<const void*>(cast_slot.ptr), sizeof(cast_words));
            g::Image decoded{};
            EXPECT(g::decode(cast_words, decoded) && decoded.type == g::ImageType::image_2d);
            EXPECT(decoded.format == g::format_from_vulkan(VK_FORMAT_R32_UINT));
            std::uint32_t cast_key[8]{};
            std::memcpy(cast_key, cast_words.data(), sizeof(cast_key));
            EXPECT(rt::d3d12::bindless_hash_lookup(device.p, cast_key) != 0);
        }

        // Depth SRVs: D16_UNORM / R16_TYPELESS (depth) and D32_FLOAT / R32_TYPELESS (depth) only allow matching depth
        // view formats (R16_UNORM / R32_FLOAT, plus UNKNOWN on typed D16/D32). Colour casts on depth textures
        // (R16_FLOAT/SNORM/UINT/SINT on D16, R32_UINT/SINT on D32, or any colour format on D24S8/D32S8) must produce
        // null descriptors with zero Vulkan validation errors.
        {
            const unsigned validation_before_depth = d.validation_errors();
            constexpr UINT depth_tex_count = 4;
            constexpr DXGI_FORMAT depth_tex_formats[depth_tex_count] = {
                DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_TYPELESS};
            constexpr DXGI_FORMAT depth_dsv_formats[depth_tex_count] = {
                DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_D32_FLOAT};
            // Multiples of 1/255 are exact in both D16_UNORM (65535 = 255 * 257) and D32_FLOAT.
            constexpr std::uint8_t depth_clear_u8[depth_tex_count] = {51, 102, 153, 204};
            Ref<ID3D12Resource> depth_textures[depth_tex_count];
            for (UINT i = 0; i < depth_tex_count; ++i) {
                D3D12_RESOURCE_DESC dd = texture_desc;
                dd.Format = depth_tex_formats[i];
                dd.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
                EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &dd,
                                                                 D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                                 IID_PPV_ARGS(&depth_textures[i].p))));
            }
            Ref<ID3D12DescriptorHeap> depth_dsv_heap, depth_srv_heap;
            const D3D12_DESCRIPTOR_HEAP_DESC depth_dsv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, depth_tex_count,
                                                          D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&depth_dsv_hd, IID_PPV_ARGS(&depth_dsv_heap.p))));
            const UINT dsv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
            const std::uint64_t dsv_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(depth_dsv_heap.p, 9))(depth_dsv_heap.p);
            const auto dsv_slot = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(dsv_cpu + i * dsv_step)}; };
            for (UINT i = 0; i < depth_tex_count; ++i) {
                D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
                dsv.Format = depth_dsv_formats[i];
                dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                device->CreateDepthStencilView(depth_textures[i].p, &dsv, dsv_slot(i));
            }

            // Valid depth SRVs: 0: D16 default, 1: D16->R16_UNORM, 2: R16_TYPELESS->R16_UNORM,
            //                   3: D32 default, 4: D32->R32_FLOAT, 5: R32_TYPELESS->R32_FLOAT; 6: scratch for invalid.
            constexpr UINT valid_depth_srvs = 6, depth_scratch = 6;
            const D3D12_DESCRIPTOR_HEAP_DESC depth_srv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, valid_depth_srvs + 1,
                                                          D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&depth_srv_hd, IID_PPV_ARGS(&depth_srv_heap.p))));
            const std::uint64_t depth_srv_cpu =
                reinterpret_cast<XboxHeapHandleFn>(slot(depth_srv_heap.p, 9))(depth_srv_heap.p);
            const auto depth_slot = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(depth_srv_cpu + i * step)}; };
            const auto depth_bytes = [&](UINT i) { return reinterpret_cast<std::uint8_t*>(depth_srv_cpu + i * step); };
            const auto depth_null = [&](UINT i) {
                return std::all_of(depth_bytes(i), depth_bytes(i) + step, [](std::uint8_t b) { return b == 0; });
            };

            struct ValidDepthSrv {
                UINT tex_index;
                DXGI_FORMAT view_format;  // UNKNOWN => pass nullptr
                VkFormat expected_vk;
            };
            constexpr ValidDepthSrv valid_cases[valid_depth_srvs] = {
                {0, DXGI_FORMAT_UNKNOWN, VK_FORMAT_D16_UNORM},
                {0, DXGI_FORMAT_R16_UNORM, VK_FORMAT_D16_UNORM},
                {1, DXGI_FORMAT_R16_UNORM, VK_FORMAT_D16_UNORM},
                {2, DXGI_FORMAT_UNKNOWN, VK_FORMAT_D32_SFLOAT},
                {2, DXGI_FORMAT_R32_FLOAT, VK_FORMAT_D32_SFLOAT},
                {3, DXGI_FORMAT_R32_FLOAT, VK_FORMAT_D32_SFLOAT},
            };
            for (UINT i = 0; i < valid_depth_srvs; ++i) {
                const auto& vc = valid_cases[i];
                D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
                srv.Format = vc.view_format;
                srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srv.Texture2D.MipLevels = 1;
                device->CreateShaderResourceView(depth_textures[vc.tex_index].p,
                                                 vc.view_format == DXGI_FORMAT_UNKNOWN ? nullptr : &srv, depth_slot(i));
                g::ImageWords words{};
                std::memcpy(words.data(), depth_bytes(i), sizeof(words));
                g::Image decoded{};
                EXPECT(g::is_image(words) && g::decode(words, decoded));
                EXPECT(decoded.type == g::ImageType::image_2d);
                EXPECT(decoded.format == g::format_from_vulkan(vc.expected_vk));
                EXPECT(decoded.swizzle == g::channels_r);
                std::uint32_t key[8]{};
                std::memcpy(key, words.data(), sizeof(key));
                EXPECT(rt::d3d12::bindless_hash_lookup(device.p, key) != 0);
            }

            // Invalid depth SRVs: every non-depth colour format and mismatched/typeless format must leave a null
            // descriptor and emit zero Vulkan validation errors.
            const auto expect_depth_srv_null = [&](ID3D12Resource* res, const D3D12_SHADER_RESOURCE_VIEW_DESC* srv) {
                std::memset(depth_bytes(depth_scratch), 0x77, step);
                device->CreateShaderResourceView(res, srv, depth_slot(depth_scratch));
                return depth_null(depth_scratch);
            };
            const auto check_bad_formats = [&](ID3D12Resource* res, std::initializer_list<DXGI_FORMAT> bad_formats) {
                for (const DXGI_FORMAT fmt : bad_formats) {
                    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
                    srv.Format = fmt;
                    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    srv.Texture2D.MipLevels = 1;
                    if (!expect_depth_srv_null(res, &srv)) return false;
                }
                return true;
            };
            // D16_UNORM and R16_TYPELESS (depth): R16_FLOAT, R16_SNORM, R16_UINT, R16_SINT and other mismatched views.
            EXPECT(check_bad_formats(depth_textures[0].p,
                                     {DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_SNORM, DXGI_FORMAT_R16_UINT,
                                      DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_D16_UNORM,
                                      DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM}));
            EXPECT(expect_depth_srv_null(depth_textures[1].p, nullptr));  // UNKNOWN on R16_TYPELESS (depth)
            EXPECT(check_bad_formats(depth_textures[1].p,
                                     {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_SNORM,
                                      DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R16_TYPELESS,
                                      DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM}));
            // D32_FLOAT and R32_TYPELESS (depth): R32_UINT, R32_SINT and other mismatched views.
            EXPECT(check_bad_formats(depth_textures[2].p,
                                     {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32_TYPELESS,
                                      DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM}));
            EXPECT(expect_depth_srv_null(depth_textures[3].p, nullptr));  // UNKNOWN on R32_TYPELESS (depth)
            EXPECT(check_bad_formats(depth_textures[3].p,
                                     {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_SINT,
                                      DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R16_UNORM,
                                      DXGI_FORMAT_R8G8B8A8_UNORM}));

            // D24_UNORM_S8_UINT / R24G8_TYPELESS and D32_FLOAT_S8X24_UINT / R32G8X24_TYPELESS: colour view formats
            // must also be rejected, while depth-plane and stencil-plane views continue working.
            for (const auto& [res_fmt, valid_depth_fmt, valid_stencil_fmt, is_typeless] :
                 {std::tuple{DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
                             DXGI_FORMAT_X24_TYPELESS_G8_UINT, false},
                  std::tuple{DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
                             DXGI_FORMAT_X24_TYPELESS_G8_UINT, true},
                  std::tuple{DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
                             DXGI_FORMAT_X32_TYPELESS_G8X24_UINT, false},
                  std::tuple{DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
                             DXGI_FORMAT_X32_TYPELESS_G8X24_UINT, true}}) {
                D3D12_RESOURCE_DESC ds_desc = texture_desc;
                ds_desc.Format = res_fmt;
                ds_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
                Ref<ID3D12Resource> ds_tex;
                EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &ds_desc,
                                                                 D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                 IID_PPV_ARGS(&ds_tex.p))));
                if (is_typeless) EXPECT(expect_depth_srv_null(ds_tex.p, nullptr));
                EXPECT(check_bad_formats(ds_tex.p,
                                         {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_SINT,
                                          DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM,
                                          DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
                                          DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_R32G8X24_TYPELESS}));
                D3D12_SHADER_RESOURCE_VIEW_DESC ds_srv{};
                ds_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                ds_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                ds_srv.Texture2D.MipLevels = 1;
                ds_srv.Format = valid_depth_fmt;
                ds_srv.Texture2D.PlaneSlice = 0;
                device->CreateShaderResourceView(ds_tex.p, &ds_srv, depth_slot(depth_scratch));
                EXPECT(!depth_null(depth_scratch));
                ds_srv.Format = valid_stencil_fmt;
                ds_srv.Texture2D.PlaneSlice = 1;
                device->CreateShaderResourceView(ds_tex.p, &ds_srv, depth_slot(depth_scratch));
                EXPECT(!depth_null(depth_scratch));
            }

            // Colour R16_TYPELESS and R32_TYPELESS (without ALLOW_DEPTH_STENCIL) still allow colour casts,
            // while rejecting depth view formats on colour textures.
            for (const auto& [colour_typeless, good_colour, bad_depth] :
                 {std::tuple{DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_D16_UNORM},
                  std::tuple{DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_D32_FLOAT}}) {
                D3D12_RESOURCE_DESC col_desc = texture_desc;
                col_desc.Format = colour_typeless;
                Ref<ID3D12Resource> col_tex;
                EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &col_desc,
                                                                 D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                 IID_PPV_ARGS(&col_tex.p))));
                D3D12_SHADER_RESOURCE_VIEW_DESC col_srv{};
                col_srv.Format = good_colour;
                col_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                col_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                col_srv.Texture2D.MipLevels = 1;
                device->CreateShaderResourceView(col_tex.p, &col_srv, depth_slot(depth_scratch));
                EXPECT(!depth_null(depth_scratch));
                EXPECT(check_bad_formats(col_tex.p, {bad_depth}));
            }

            // Depth resource with a castable format list (allow_channel_cast = true): valid depth SRV still works
            // and colour SRVs still leave a null descriptor with zero validation errors.
            {
                Ref<ID3D12Device10> inner10;
                EXPECT(SUCCEEDED(inner->QueryInterface(IID_PPV_ARGS(&inner10.p))));
                D3D12_RESOURCE_DESC1 d16_cast_desc{};
                d16_cast_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                d16_cast_desc.Width = size;
                d16_cast_desc.Height = size;
                d16_cast_desc.DepthOrArraySize = 1;
                d16_cast_desc.MipLevels = 1;
                d16_cast_desc.Format = DXGI_FORMAT_D16_UNORM;
                d16_cast_desc.SampleDesc = {1, 0};
                d16_cast_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
                const DXGI_FORMAT d16_list[] = {DXGI_FORMAT_D16_UNORM};
                Ref<ID3D12Resource> d16_cast_tex;
                EXPECT(SUCCEEDED(inner10->CreateCommittedResource3(
                    &default_heap, D3D12_HEAP_FLAG_NONE, &d16_cast_desc, D3D12_BARRIER_LAYOUT_COMMON, nullptr, nullptr,
                    1, d16_list, IID_PPV_ARGS(&d16_cast_tex.p))));
                D3D12_SHADER_RESOURCE_VIEW_DESC d16_srv{};
                d16_srv.Format = DXGI_FORMAT_R16_UNORM;
                d16_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                d16_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                d16_srv.Texture2D.MipLevels = 1;
                device->CreateShaderResourceView(d16_cast_tex.p, &d16_srv, depth_slot(depth_scratch));
                EXPECT(!depth_null(depth_scratch));
                EXPECT(check_bad_formats(d16_cast_tex.p,
                                         {DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_SNORM, DXGI_FORMAT_R16_UINT,
                                          DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R8G8B8A8_UNORM}));
            }

            // Clear each depth texture and sample all 6 valid depth SRVs on the GPU to confirm they read back their
            // cleared depth value in R (with G=0, B=0, A=1).
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            for (UINT i = 0; i < depth_tex_count; ++i) {
                list->ClearDepthStencilView(dsv_slot(i), D3D12_CLEAR_FLAG_DEPTH, float(depth_clear_u8[i]) / 255.0f, 0,
                                            0, nullptr);
                const D3D12_RESOURCE_BARRIER barrier = transition(
                    depth_textures[i].p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                list->ResourceBarrier(1, &barrier);
            }
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            ++packed_signal;
            EXPECT(SUCCEEDED(queue->Signal(fence.p, packed_signal)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(packed_signal, nullptr)));
#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
            SampleKeys depth_keys[valid_depth_srvs]{};
            for (UINT i = 0; i < valid_depth_srvs; ++i) {
                std::memcpy(depth_keys[i].tsharp, depth_bytes(i), sizeof(depth_keys[i].tsharp));
                std::memcpy(depth_keys[i].ssharp, sampler_bytes, sizeof(depth_keys[i].ssharp));
                depth_keys[i].first = i * texels;
            }
            std::vector<std::uint8_t> depth_pixels(valid_depth_srvs * bytes);
            EXPECT(sample_through_set(d, rt::d3d12::bindless_set(device.p), set0_layout, depth_keys, depth_pixels.data()));
            UINT depth_differ = 0;
            for (UINT k = 0; k < valid_depth_srvs; ++k) {
                const std::uint8_t want[4] = {depth_clear_u8[valid_cases[k].tex_index], 0, 0, 255};
                for (UINT i = 0; i < texels; ++i) {
                    const std::uint8_t* got = depth_pixels.data() + k * bytes + i * 4;
                    if (std::memcmp(got, want, 4) != 0 && depth_differ++ < 4)
                        std::fprintf(stderr, "depth SRV %u texel %u: got %u,%u,%u,%u, want %u,%u,%u,%u\n", k, i,
                                     unsigned(got[0]), unsigned(got[1]), unsigned(got[2]), unsigned(got[3]),
                                     unsigned(want[0]), unsigned(want[1]), unsigned(want[2]), unsigned(want[3]));
                }
            }
            EXPECT(depth_differ == 0);
            std::printf("depth SRVs: %u valid D16/D32 views sampled cleared depth values; invalid colour casts rejected "
                        "as null descriptors (validation errors: %u)\n",
                        valid_depth_srvs, d.validation_errors() - validation_before_depth);
#endif
            EXPECT(d.validation_errors() == validation_before_depth);
        }

        upload.reset();
        array_texture.reset();
        line.reset();
        volume.reset();
        typeless_volume.reset();
        typeless_heap.reset();
        uint_texture.reset();
        for (auto& resource : byte_textures) resource.reset();
        byte_heap.reset();
        ms_texture.reset();
        ms_array.reset();
        for (auto& resource : bgr_textures) resource.reset();
        view_heap.reset();
        sampler_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    if (!sampled_through_set) {
        if (d.validation_errors()) {
            std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
            return 1;
        }
        std::printf("descriptors: copies matched; views and samplers wrote the T#s and S#s expected; the sampling "
                    "shader was not built (no glslangValidator)\n");
        return 0;
    }

    // The view's slot and its copy give the texture, texel for texel; the unknown descriptor gives the fallback
    // image, one colour everywhere.
    unsigned mismatches = 0;
    for (UINT i = 0; i < texels; ++i) {
        // Each RGB channel is mantissa * 2^(exponent - 24). The shader packs the result to RGBA8.
        constexpr std::uint8_t packed_expected[][4] = {{16, 32, 64, 255}, {32, 64, 128, 255},
                                                       {64, 128, 255, 255}, {64, 16, 32, 255}};
        const auto* packed_want = packed_expected[(i % size + i / size) % 4];
        const auto* packed_got = results.data() + 3 * bytes + i * 4;
        if (std::memcmp(packed_got, packed_want, 4) != 0 && mismatches++ < 4)
            std::fprintf(stderr, "RGB9E5 texel %u: got %u,%u,%u,%u, want %u,%u,%u,%u\n", i,
                         packed_got[0], packed_got[1], packed_got[2], packed_got[3],
                         packed_want[0], packed_want[1], packed_want[2], packed_want[3]);
        const auto want = texel(i % size, i / size);
        for (UINT k = 0; k < 2; ++k) {
            const std::uint8_t* got = results.data() + k * bytes + i * 4;
            if (std::memcmp(got, want.data(), 4) != 0 && mismatches++ < 4)
                std::fprintf(stderr, "table %u texel %u: got %u,%u,%u,%u, want %u,%u,%u,%u\n", k, i, unsigned(got[0]),
                             unsigned(got[1]), unsigned(got[2]), unsigned(got[3]), unsigned(want[0]), unsigned(want[1]),
                             unsigned(want[2]), unsigned(want[3]));
        }
        if (std::memcmp(results.data() + 2 * bytes + i * 4, results.data() + 2 * bytes, 4) != 0 && mismatches++ < 4)
            std::fprintf(stderr, "fallback texel %u differs from texel 0\n", i);
    }
    EXPECT(!mismatches);
    EXPECT(std::memcmp(results.data() + 2 * bytes, texel(0, 0).data(), 4) != 0);  // not the texture
    // After the texture was destroyed its T# finds the fallback too: the same colour as the unknown descriptor.
    for (UINT i = 0; i < texels; ++i) {
        if (std::memcmp(after_destruction.data() + i * 4, results.data() + 2 * bytes, 4) != 0 && mismatches++ < 4)
            std::fprintf(stderr, "texel %u after the texture was destroyed is not the fallback\n", i);
    }
    EXPECT(!mismatches);
    // UINT (0,0,0,0), UINT (1,1,1,1) and float white borders: 0, 1 and 255 in every channel of every texel.
    const std::uint8_t border_values[3] = {0, 1, 255};
    const char* border_names[3] = {"UINT transparent black", "UINT opaque white", "float opaque white"};
    for (UINT k = 0; k < 3; ++k)
        for (UINT i = 0; i < texels * 4; ++i) {
            const std::uint8_t got = borders[k * bytes + i];
            if (got != border_values[k] && mismatches++ < 4)
                std::fprintf(stderr, "%s border, texel %u channel %u: got %u, want %u\n", border_names[k], i / 4,
                             i % 4, unsigned(got), unsigned(border_values[k]));
        }
    EXPECT(!mismatches);
    EXPECT(lod_mismatches == 0);
    // All read red, green and blue from the same bytes; a B8G8R8X8 view reads alpha 1, a B8G8R8A8 view the stored
    // byte, whatever the texture's own format (R8G8B8A8 textures read through B8G8R8X8 and B8G8R8A8 views included).
    const char* bgr_names[] = {"B8G8R8X8_UNORM", "B8G8R8A8_UNORM", "B8G8R8X8_UNORM view of castable B8G8R8A8",
                               "B8G8R8A8_UNORM view of castable B8G8R8X8", "B8G8R8X8_UNORM view of castable R8G8B8A8",
                               "B8G8R8A8_UNORM view of castable R8G8B8A8"};
    for (UINT k = 0; k < 6; ++k)
        for (UINT i = 0; i < texels; ++i) {
            auto want = texel(i % size, i / size);
            if (k == 1 || k == 3 || k == 5) want[3] = bgrx_texel(i % size, i / size)[3];
            const std::uint8_t* got = bgr_results.data() + k * bytes + i * 4;
            if (std::memcmp(got, want.data(), 4) != 0 && mismatches++ < 4)
                std::fprintf(stderr, "%s texel %u: got %u,%u,%u,%u, want %u,%u,%u,%u\n", bgr_names[k], i,
                             unsigned(got[0]), unsigned(got[1]), unsigned(got[2]), unsigned(got[3]),
                             unsigned(want[0]), unsigned(want[1]), unsigned(want[2]), unsigned(want[3]));
        }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("descriptors: copies matched; a T# and an S# from CreateShaderResourceView and CreateSampler were "
                "found in the device's bindless set by a shader, %u RGBA8 and RGB9E5 texels matched, and an unknown descriptor "
                "and a destroyed texture's gave the fallback; UINT black, UINT white and float white borders read "
                "0, 1 and 255; B8G8R8X8 views read alpha 1 where B8G8R8A8 views read the same bytes' alpha, also across "
                "a castable-format list (%u texels each)\n",
                texels, texels);
    return 0;
}
