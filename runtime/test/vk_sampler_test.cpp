// GPU test of the S# and T# mip fields as src/gfx10_descriptors.h maps them to Vulkan. The pixel shader
// (ps_ssharp.asm) loads a T# and an S# from the heap, the translator hashes both, and this program binds the image
// and sampler those keys name. Unlike vk_texture_test, which binds a fixed nearest sampler and one level, the
// VkSampler here is vulkan_sampler(decode(S#)) and the view's levels come from decode(T#).
//
//   vk_sampler_test <vs_ids.spv> <ps_ssharp.spv> <out.raw> <s0> <s1> <s2> <s3> [<base level> <last level>]
//
// S# words are hex. The T# is encode() of a 128x128 RGBA8 image with two levels, with BASE_LEVEL and LAST_LEVEL as
// given (default 0 and 1). Level 0 has four solid quadrants: red, green, blue and yellow. Level 1 is solid grey
// (128/255). vs_ids runs UV 0..1 over the 64x64 target, so implicit sampling sees two level-0 texels per pixel
// (LOD 1) and image_sample_lz asks for LOD 0. The 64x64 RGBA32F result is written to <out.raw>; check.py judges it.
// Exit 77 with "unsupported: ..." when the device lacks a feature the shader or sampler needs.
#define VK_NO_PROTOTYPES
#include "vulkan_loader.h"
#include "vk_shader_fma.h"
#include "../src/vulkan_device_filter.h"
#include "vk_pc_tail.h"
#include "vk_shader_fma.h"

#include "../src/descriptor_table.h"
#include "../src/gfx10_descriptors.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#define CHECK(x)                                                                     \
    do {                                                                             \
        VkResult r_ = (x);                                                           \
        if (r_ != VK_SUCCESS) {                                                      \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, r_, __LINE__);   \
            return 1;                                                                \
        }                                                                            \
    } while (0)

namespace {

constexpr int unsupported = 77;

std::vector<std::uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
    std::vector<std::uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), words.size() * 4);
    return words;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 8 && argc != 10) {
        std::fprintf(stderr, "usage: vk_sampler_test <vs.spv> <fs.spv> <out.raw> <s0> <s1> <s2> <s3> [<base> <last>]\n");
        return 2;
    }
    namespace gfx10 = leonida::gfx10;
    constexpr std::uint32_t W = 64, H = 64, TW = 128, levels = 2;
    constexpr std::uint32_t table_va = 256;  // console VA of the T# (8 dwords); the S# follows at +32
    gfx10::SamplerWords ssharp{};
    for (int i = 0; i < 4; ++i) ssharp[i] = std::uint32_t(std::strtoul(argv[4 + i], nullptr, 16));
    gfx10::Image image;
    image.address = 0x100000;
    image.format = gfx10::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
    image.width = image.height = TW;
    image.resource_mips = levels;
    image.first_mip = argc == 10 ? std::uint32_t(std::atoi(argv[8])) : 0;
    image.mip_count = (argc == 10 ? std::uint32_t(std::atoi(argv[9])) : levels - 1) - image.first_mip + 1;
    const gfx10::ImageWords tsharp = gfx10::encode(image);
    gfx10::Image view_of;
    if (!gfx10::decode(tsharp, view_of) || view_of.first_mip + view_of.mip_count > levels) {
        std::fprintf(stderr, "the T# does not decode to a view of the %u levels\n", levels);
        return 2;
    }
    const VkSamplerCreateInfo sampler_info = gfx10::vulkan_sampler(gfx10::decode(ssharp));
    std::printf("S# %08x %08x %08x %08x: mipmapMode %d, minLod %g, maxLod %g, mipLodBias %g; view levels %u..%u\n",
                ssharp[0], ssharp[1], ssharp[2], ssharp[3], int(sampler_info.mipmapMode), sampler_info.minLod,
                sampler_info.maxLod, sampler_info.mipLodBias, view_of.first_mip, view_of.first_mip + view_of.mip_count - 1);

    const PFN_vkGetInstanceProcAddr gipa = load_vulkan();
    if (!gipa) { std::fprintf(stderr, "the Vulkan loader was not found\n"); return 1; }
    VkInstance inst = VK_NULL_HANDLE;
#define LI(n) const auto n = reinterpret_cast<PFN_##n>(gipa(inst, #n))
    LI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    LI(vkEnumeratePhysicalDevices); LI(vkGetPhysicalDeviceProperties); LI(vkGetPhysicalDeviceQueueFamilyProperties);
    LI(vkGetPhysicalDeviceMemoryProperties); LI(vkGetPhysicalDeviceFeatures2); LI(vkCreateDevice); LI(vkGetDeviceProcAddr);
    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    int best_rank = -1;
    for (auto d : pds) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.apiVersion < VK_API_VERSION_1_2 || !leonida::vulkan_device_wanted(p.deviceName)) continue;
        // the runtime's pick: the first device of the highest rank (native driver over Dozen, #915)
        if (const int rank = leonida::vulkan_device_rank(static_cast<int>(p.deviceType), p.vendorID);
            rank > best_rank) {
            best_rank = rank;
            pd = d;
            props = p;
        }
    }
    if (!pd && !leonida::vulkan_device_filter().empty()) {  // a filter matching nothing fails, never skips
        std::fprintf(stderr, "LEONIDA_VK_DEVICE=%s: no Vulkan 1.2 device matches\n",
                     leonida::vulkan_device_filter().c_str());
        return 1;
    }
    if (!pd) { std::printf("unsupported: no Vulkan 1.2 device\n"); return unsupported; }
    std::printf("device: %s, driver 0x%x, Vulkan %u.%u.%u\n", props.deviceName, props.driverVersion,
                VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
                VK_API_VERSION_PATCH(props.apiVersion));
    // 144 push-constant bytes by default, or 128 under LEONIDA_PC_TAIL_UBO=1 (rdna2spirv.py --pc-tail-ubo).
    // A 128-byte device cannot declare a 144-byte range (VUID-VkPushConstantRange-size-00298): exit 77 (D3D-14).
    const bool pc_tail = leonida::vk_test::pc_tail_requested();
    const std::uint32_t push_bytes = pc_tail ? 128u : 144u;
    if (props.limits.maxPushConstantsSize < push_bytes) {
        std::printf("unsupported: need %u push constant bytes, device has %u\n", push_bytes,
                    props.limits.maxPushConstantsSize);
        return unsupported;
    }
    VkPhysicalDeviceVulkan11Features have11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &have11};
    VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &have12};
    vkGetPhysicalDeviceFeatures2(pd, &have);
    if (!have12.bufferDeviceAddress || !have12.runtimeDescriptorArray || !have.features.shaderInt64) {
        std::printf("unsupported: the device lacks bufferDeviceAddress, runtimeDescriptorArray or shaderInt64\n");
        return unsupported;
    }
    const auto fragment_code = read_spv(argv[2]);
    const bool dynamic_sampled = leonida::vk_test::spirv_declares_capability(
        fragment_code.data(), fragment_code.size() * 4, leonida::vk_test::spv_sampled_image_array_dynamic_indexing);
    if (dynamic_sampled && !have.features.shaderSampledImageArrayDynamicIndexing) {
        std::printf("unsupported: shaderSampledImageArrayDynamicIndexing\n");
        return unsupported;
    }
    if (sampler_info.anisotropyEnable && !have.features.samplerAnisotropy) {
        std::printf("unsupported: the S# asks for anisotropy and the device has no samplerAnisotropy\n");
        return unsupported;
    }
    std::uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    std::uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT)) ++qi;
    if (qi == qn) { std::printf("unsupported: no graphics queue\n"); return unsupported; }
    const float priority = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    enable12.bufferDeviceAddress = VK_TRUE;
    enable12.runtimeDescriptorArray = VK_TRUE;
    enable12.descriptorIndexing = have12.descriptorIndexing;
    // The translated vs_ids reads BaseVertex / BaseInstance (DrawParameters) for the zero-based v0 / v3, as in
    // vk_harness and vk_texture_test. Without the feature, creating its module breaks
    // VUID-VkShaderModuleCreateInfo-pCode-08740.
    VkPhysicalDeviceVulkan11Features enable11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    enable11.shaderDrawParameters = have11.shaderDrawParameters;
    enable12.pNext = &enable11;
    VkPhysicalDeviceFeatures2 enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &enable12};
    enable.features.shaderInt64 = VK_TRUE;
    enable.features.shaderSampledImageArrayDynamicIndexing = dynamic_sampled;
    enable.features.samplerAnisotropy = sampler_info.anisotropyEnable;
    leonida::vk_test::ShaderFma fma;      // OpFmaKHR (rdna2spirv.py --fma khr), when the device has it
    fma.query(gipa, inst, pd);
    // A device without VK_KHR_shader_fma cannot create a pipeline from a stage that uses OpFmaKHR: skip the pair.
    if (!fma.accepts_file(argv[1]) || !fma.accepts_file(argv[2])) return unsupported;
    fma.link_after(&enable);
    const char* fma_extension = fma.extension();
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enable};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (fma.supported()) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &fma_extension; }
    VkDevice dev = VK_NULL_HANDLE;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LD(n) const auto n = reinterpret_cast<PFN_##n>(vkGetDeviceProcAddr(dev, #n))
    LD(vkGetDeviceQueue); LD(vkCreateImage); LD(vkGetImageMemoryRequirements); LD(vkAllocateMemory);
    LD(vkBindImageMemory); LD(vkCreateImageView); LD(vkCreateRenderPass); LD(vkCreateFramebuffer);
    LD(vkCreateShaderModule); LD(vkCreatePipelineLayout); LD(vkCreateGraphicsPipelines); LD(vkCreateCommandPool);
    LD(vkAllocateCommandBuffers); LD(vkBeginCommandBuffer); LD(vkCmdBeginRenderPass); LD(vkCmdBindPipeline);
    LD(vkCmdDraw); LD(vkCmdEndRenderPass); LD(vkCmdCopyImageToBuffer); LD(vkCmdCopyBufferToImage);
    LD(vkEndCommandBuffer); LD(vkQueueSubmit); LD(vkQueueWaitIdle); LD(vkCreateBuffer);
    LD(vkGetBufferMemoryRequirements); LD(vkBindBufferMemory); LD(vkMapMemory); LD(vkGetBufferDeviceAddress);
    LD(vkCreateSampler); LD(vkCreateDescriptorSetLayout); LD(vkCreateDescriptorPool); LD(vkAllocateDescriptorSets);
    LD(vkUpdateDescriptorSets); LD(vkCmdBindDescriptorSets); LD(vkCmdPushConstants); LD(vkCmdPipelineBarrier);
    VkQueue queue;
    vkGetDeviceQueue(dev, qi, 0, &queue);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    const auto memtype = [&](std::uint32_t bits, VkMemoryPropertyFlags want) {
        for (std::uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0u;
    };
    const auto make_buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* b, void** map) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = usage;
        VkResult r = vkCreateBuffer(dev, &bi, nullptr, b);
        if (r) return r;
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, *b, &mr);
        VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.pNext = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ? &fi : nullptr;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory m;
        r = vkAllocateMemory(dev, &ai, nullptr, &m);
        if (r) return r;
        r = vkBindBufferMemory(dev, *b, m, 0);
        if (r) return r;
        return vkMapMemory(dev, m, 0, size, 0, map);
    };
    const auto make_image = [&](VkFormat format, std::uint32_t w, std::uint32_t h, std::uint32_t mips,
                                VkImageUsageFlags usage, VkImage* img) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = format;
        ii.extent = {w, h, 1};
        ii.mipLevels = mips;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.usage = usage;
        VkResult r = vkCreateImage(dev, &ii, nullptr, img);
        if (r) return r;
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(dev, *img, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m;
        r = vkAllocateMemory(dev, &ai, nullptr, &m);
        return r ? r : vkBindImageMemory(dev, *img, m, 0);
    };
    const auto make_view = [&](VkImage img, VkFormat format, std::uint32_t base, std::uint32_t count, VkImageView* view) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, base, count, 0, 1};
        return vkCreateImageView(dev, &vi, nullptr, view);
    };

    // The heap (the console's GPU address window): zeros, with the T# at table_va and the S# behind it.
    VkBuffer heap;
    void* heap_map;
    CHECK(make_buffer(65536, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, &heap, &heap_map));
    std::memset(heap_map, 0, 65536);
    std::memcpy(static_cast<char*>(heap_map) + table_va, tsharp.data(), 32);
    std::memcpy(static_cast<char*>(heap_map) + table_va + 32, ssharp.data(), 16);
    VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address_info.buffer = heap;
    const VkDeviceAddress heap_base = vkGetBufferDeviceAddress(dev, &address_info);
    // Descriptor hash table: the T# names image 1, the S# sampler 1. Index 0 of each is a magenta fallback.
    leonida::DescriptorTable table(16);
    table.insert(tsharp.data(), 1);
    table.insert(leonida::make_ssharp_key(ssharp.data()).data(), 1);
    const auto table_words = table.build_gpu_buffer(true);
    VkBuffer table_buffer;
    void* table_map;
    CHECK(make_buffer(table_words.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &table_buffer, &table_map));
    std::memcpy(table_map, table_words.data(), table_words.size() * 4);

    // Textures. RGBA8 in memory order; level 1 grey has r = g = 128/255, unlike every level-0 colour.
    constexpr VkFormat texture_format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr std::uint32_t quadrant[2][2] = {{0xff0000ffu, 0xff00ff00u}, {0xffff0000u, 0xff00ffffu}};
    constexpr std::uint32_t grey = 0xff808080u, magenta = 0xffff00ffu;
    std::vector<std::uint32_t> texels;
    for (std::uint32_t y = 0; y < TW; y++)
        for (std::uint32_t x = 0; x < TW; x++) texels.push_back(quadrant[y >= TW / 2][x >= TW / 2]);
    texels.resize(texels.size() + (TW / 2) * (TW / 2), grey);
    VkImage fallback, texture;
    VkImageView fallback_view, texture_view;
    CHECK(make_image(texture_format, 1, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, &fallback));
    CHECK(make_view(fallback, texture_format, 0, 1, &fallback_view));
    CHECK(make_image(texture_format, TW, TW, levels, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, &texture));
    CHECK(make_view(texture, gfx10::vulkan_format(view_of.format), view_of.first_mip, view_of.mip_count, &texture_view));
    VkBuffer upload;
    void* upload_map;
    const VkDeviceSize texture_offset = 256, texture_bytes = texels.size() * 4;
    CHECK(make_buffer(texture_offset + texture_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &upload, &upload_map));
    std::memcpy(upload_map, &magenta, 4);
    std::memcpy(static_cast<char*>(upload_map) + texture_offset, texels.data(), texture_bytes);
    VkSampler samplers[2];
    VkSamplerCreateInfo fallback_sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    CHECK(vkCreateSampler(dev, &fallback_sampler, nullptr, &samplers[0]));
    CHECK(vkCreateSampler(dev, &sampler_info, nullptr, &samplers[1]));

    // Render target, pass and framebuffer.
    constexpr VkFormat target_format = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkImage target;
    VkImageView target_view;
    CHECK(make_image(target_format, W, H, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &target));
    CHECK(make_view(target, target_format, 0, 1, &target_view));
    VkAttachmentDescription ad{};
    ad.format = target_format;
    ad.samples = VK_SAMPLE_COUNT_1_BIT;
    ad.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ad.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ad.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ad.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    ad.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ad.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const VkAttachmentReference ar{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sd{};
    sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sd.colorAttachmentCount = 1;
    sd.pColorAttachments = &ar;
    VkSubpassDependency after{};
    after.srcSubpass = 0;
    after.dstSubpass = VK_SUBPASS_EXTERNAL;
    after.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    after.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    after.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 1;
    rpi.pAttachments = &ad;
    rpi.subpassCount = 1;
    rpi.pSubpasses = &sd;
    rpi.dependencyCount = 1;
    rpi.pDependencies = &after;
    VkRenderPass pass;
    CHECK(vkCreateRenderPass(dev, &rpi, nullptr, &pass));
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = pass;
    fbi.attachmentCount = 1;
    fbi.pAttachments = &target_view;
    fbi.width = W;
    fbi.height = H;
    fbi.layers = 1;
    VkFramebuffer framebuffer;
    CHECK(vkCreateFramebuffer(dev, &fbi, nullptr, &framebuffer));

    // Set 0 as the translator expects it: binding 0 images[], 1 samplers[], 2 the descriptor hash table.
    VkDescriptorSetLayoutBinding b[3]{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER, 2, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 3;
    dli.pBindings = b;
    VkDescriptorSetLayout dsl;
    CHECK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
    const VkDescriptorPoolSize sizes[3]{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2}, {VK_DESCRIPTOR_TYPE_SAMPLER, 2},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = 3;
    dpi.pPoolSizes = sizes;
    VkDescriptorPool dp;
    CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dp;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &dsl;
    VkDescriptorSet ds;
    CHECK(vkAllocateDescriptorSets(dev, &dai, &ds));
    const VkDescriptorImageInfo images[2]{{VK_NULL_HANDLE, fallback_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                          {VK_NULL_HANDLE, texture_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    const VkDescriptorImageInfo sampler_infos[2]{{samplers[0]}, {samplers[1]}};
    const VkDescriptorBufferInfo table_info{table_buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet writes[3]{};
    for (auto& w : writes) {
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = ds;
    }
    writes[0].dstBinding = 0; writes[0].descriptorCount = 2; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[0].pImageInfo = images;
    writes[1].dstBinding = 1; writes[1].descriptorCount = 2; writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[1].pImageInfo = sampler_infos;
    writes[2].dstBinding = 2; writes[2].descriptorCount = 1; writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &table_info;
    vkUpdateDescriptorSets(dev, 3, writes, 0, nullptr);

    // Pipeline.
    VkShaderModule modules[2];
    const auto vertex_code = read_spv(argv[1]);
    // LEONIDA_PC_TAIL_UBO=1 (rdna2spirv.py --pc-tail-ubo): s0..s31 in 128 push-constant bytes; the heap base, if a
    // stage reads it, in a uniform block at set 1 (vk_pc_tail.h).
    bool tail_block = false;
    for (int stage = 0; stage < 2; ++stage) {
        const auto& code = stage == 0 ? vertex_code : fragment_code;
        if (code.empty()) {
            std::fprintf(stderr, "cannot read %s\n", argv[1 + stage]);
            return 1;
        }
        if (pc_tail && leonida::vk_test::spirv_declares_set(code.data(), code.size() * 4, leonida::vk_test::pc_tail_set))
            tail_block = true;
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size() * 4;
        smi.pCode = code.data();
        CHECK(vkCreateShaderModule(dev, &smi, nullptr, &modules[stage]));
    }
    VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                              {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = modules[0];
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = modules[1];
    stages[1].pName = "main";
    const VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &push;
    leonida::vk_test::PcTail tail;
    if (tail_block && !tail.create(vkGetDeviceProcAddr, dev, mp, props.limits.minUniformBufferOffsetAlignment,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1)) {
        std::fprintf(stderr, "FAILED creating the --pc-tail-ubo block\n");
        return 1;
    }
    const VkDescriptorSetLayout set_layouts[] = {dsl, tail.set_layout};
    if (tail_block) { pli.setLayoutCount = 2; pli.pSetLayouts = set_layouts; }
    VkPipelineLayout pipeline_layout;
    CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pipeline_layout));
    const VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    const VkViewport viewport{0, 0, float(W), float(H), 0, 1};
    const VkRect2D scissor{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo viewports{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewports.viewportCount = 1;
    viewports.pViewports = &viewport;
    viewports.scissorCount = 1;
    viewports.pScissors = &scissor;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = 2;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vertex_input;
    gpi.pInputAssemblyState = &assembly;
    gpi.pViewportState = &viewports;
    gpi.pRasterizationState = &raster;
    gpi.pMultisampleState = &multisample;
    gpi.pColorBlendState = &blend;
    gpi.layout = pipeline_layout;
    gpi.renderPass = pass;
    VkPipeline pipeline;
    CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipeline));
    VkBuffer readback;
    void* readback_map;
    CHECK(make_buffer(VkDeviceSize(W) * H * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback, &readback_map));

    // Upload every level, draw once, read back.
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qi;
    VkCommandPool pool;
    CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &begin));
    for (int t = 0; t < 2; ++t) {
        const std::uint32_t mips = t ? levels : 1;
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        ib.image = t ? texture : fallback;
        ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
        ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
        VkDeviceSize offset = t ? texture_offset : 0;
        for (std::uint32_t level = 0; level < mips; ++level) {
            const std::uint32_t side = t ? TW >> level : 1;
            VkBufferImageCopy c{};
            c.bufferOffset = offset;
            c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
            c.imageExtent = {side, side, 1};
            vkCmdCopyBufferToImage(cmd, upload, ib.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
            offset += VkDeviceSize(side) * side * 4;
        }
        ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    }
    VkClearValue clear{};
    for (float& channel : clear.color.float32) channel = -1.0f;  // no texel holds this
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = pass;
    rbi.framebuffer = framebuffer;
    rbi.renderArea = scissor;
    rbi.clearValueCount = 1;
    rbi.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &ds, 0, nullptr);
    std::uint32_t block[36] = {};
    block[0] = table_va;  // s0:s1 = console VA of the T# (the shader's s1 stays 0)
    block[32] = std::uint32_t(heap_base);
    block[33] = std::uint32_t(heap_base >> 32);
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, push_bytes, block);
    if (tail_block) {
        tail.write(0, block, 4);
        const std::uint32_t offset = tail.offset(0);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, leonida::vk_test::pc_tail_set, 1,
                                &tail.set, 1, &offset);
    }
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    VkBufferImageCopy rc{};
    rc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    rc.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &rc);
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(queue));
    std::ofstream output(argv[3], std::ios::binary);
    output.write(static_cast<const char*>(readback_map), std::streamsize(W) * H * 16);
    output.close();
    if (!output) {
        std::fprintf(stderr, "failed to write %s\n", argv[3]);
        return 1;
    }
    std::printf("ok -> %s\n", argv[3]);
    return 0;
}
