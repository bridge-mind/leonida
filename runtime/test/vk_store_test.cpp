// Headless Vulkan test for translated typed image stores and loads (runtime/test/shaders/cs_imgstore.asm and
// cs_imgload.asm).
// Usage: vk_store_test cs.spv [load|load-d16|words|signed-zero]
// The shader builds three T#s (word1 FORMAT = 56 rgba8 unorm, 20 r32 uint, 21 r32 sint) and stores to three 8x8
// images. The translator picks the float / uint / sint view (set 0, bindings 8 / 9 / 10 for 2D) from the FORMAT
// field, and the bindless index from the descriptor hash table (binding 2). As in the runtime, each image has its
// own index in the 2D arrays (1, 2, 3; index 0 is the fallback) and its view in its own format at that index of the
// sampled array (binding 0), which typed image_load fetches from. Checks every texel.
// "load" (cs_imgload.asm): the uint and sint images are filled before the dispatch, and the shader loads both
// texels with image_load and stores their sum into the sint image. The device is made without
// shaderStorageImageReadWithoutFormat, and a module that declares StorageImageReadWithoutFormat fails (SHADER-151).
// "words" (cs_imgatomic.asm, cs_imgatomic3d.asm, cs_resinfo.asm): the uint and sint images are filled as for "load"
// and a 4x4x4 r32 uint volume with 200 + i, and nothing is judged here: the 64 words of each are printed after the
// dispatch (texel (x, y) at i = 8y + x, (x, y, z) at i = 16z + 4y + x) for check.py to compare. Three things are
// bound for these shaders in every mode: the uint image as an r32ui storage image at set 0 binding 4 and the volume
// at binding 7, where 2D and 3D image atomics address their image, and at index 4 of the sampled array a 16x4 rgba8
// image with all five mip levels (never read, only queried), behind the T# the runtime writes for such an image.
// "signed-zero" checks three direct float stores and all remaining RGBA32 guard words (SHADER-605).
#include "vulkan_loader.h"
#include "vk_device_pick.h"
#include "vk_shader_fma.h"
#include "vk_pc_tail.h"
#include "spirv_capability.h"
#include "../src/gfx10_descriptors.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

static PFN_vkGetInstanceProcAddr gipa;
#define CHECK(x)                                                                         \
    do {                                                                                 \
        VkResult r_ = (x);                                                               \
        if (r_ != VK_SUCCESS) {                                                          \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, r_, __LINE__);       \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

static uint32_t key_hash(const uint32_t k[8]) {     // same FNV-1a as the translator's desc_lookup
    uint32_t h = 2166136261u;
    for (int i = 0; i < 8; i++) h = (h ^ k[i]) * 16777619u;
    return h;
}

static void table_insert(std::vector<uint32_t>& t, const uint32_t key[8], uint32_t index) {
    uint32_t mask = t[0], h = key_hash(key);
    for (uint32_t k = 0; k < 8; k++) {
        uint32_t base = ((h + k) & mask) * 10;
        bool used = false;
        for (int j = 0; j < 8; j++) used |= t[base + 1 + j] != 0;
        if (used) continue;
        for (int j = 0; j < 8; j++) t[base + 1 + j] = key[j];
        t[base + 9] = index;
        return;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: vk_store_test cs.spv [load|load-d16|words|signed-zero]\n"); return 2; }
    gipa = load_vulkan();
    if (!gipa) return 1;
#define LOADI(name) auto name = reinterpret_cast<PFN_##name>(gipa(inst, #name))
    VkInstance inst = VK_NULL_HANDLE;
    LOADI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    LOADI(vkEnumeratePhysicalDevices); LOADI(vkGetPhysicalDeviceProperties); LOADI(vkCreateDevice);
    LOADI(vkGetPhysicalDeviceQueueFamilyProperties); LOADI(vkGetDeviceProcAddr);
    LOADI(vkGetPhysicalDeviceMemoryProperties); LOADI(vkGetPhysicalDeviceFeatures);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    std::vector<VkPhysicalDeviceProperties> candidates(pds.size());
    for (std::size_t i = 0; i < pds.size(); ++i) vkGetPhysicalDeviceProperties(pds[i], &candidates[i]);
    const std::size_t chosen = leonida::vk_test::pick_device(candidates);
    VkPhysicalDevice pd = chosen < pds.size() ? pds[chosen] : VK_NULL_HANDLE;
    if (!pd) {
        leonida::vk_test::report_no_device();
        return 1;
    }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    std::fprintf(stderr, "device: %s\n", props.deviceName);
    // 144 push-constant bytes by default, or 128 under LEONIDA_PC_TAIL_UBO=1 (rdna2spirv.py --pc-tail-ubo).
    // A 128-byte device cannot declare a 144-byte range (VUID-VkPushConstantRange-size-00298): exit 77 (D3D-14).
    const bool pc_tail = leonida::vk_test::pc_tail_requested();
    const uint32_t pc_bytes = pc_tail ? 128u : 144u;
    if (props.limits.maxPushConstantsSize < pc_bytes) {
        std::printf("unsupported: need %u push constant bytes, device has %u\n", pc_bytes,
                    props.limits.maxPushConstantsSize);
        return 77;
    }
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_COMPUTE_BIT)) qi++;
    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    // The typed stores need format-less storage image writes: without them, report that and exit 77 (unsupported,
    // as vk_compute_test does) instead of failing vkCreateDevice. Typed loads need no storage image feature.
    VkPhysicalDeviceFeatures have{};
    vkGetPhysicalDeviceFeatures(pd, &have);
    if (!have.shaderStorageImageWriteWithoutFormat) {
        std::printf("unsupported: shaderStorageImageWriteWithoutFormat=0\n");
        return 77;
    }
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.runtimeDescriptorArray = VK_TRUE; f12.bufferDeviceAddress = VK_TRUE;
    f12.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    // "load" mode (cs_imgload.asm): the sint image receives the sum of the two texels read back with image_load.
    const bool packed_load = argc > 2 && !std::strcmp(argv[2], "load-d16");
    const bool load_mode = packed_load || (argc > 2 && !std::strcmp(argv[2], "load"));
    // "words" mode: the two images filled as for "load"; the shader's results are whatever it leaves in them.
    const bool words_mode = argc > 2 && !std::strcmp(argv[2], "words");
    const bool signed_zero = argc > 2 && !std::strcmp(argv[2], "signed-zero");
    if (signed_zero) {
        LOADI(vkGetPhysicalDeviceProperties2);
        VkPhysicalDeviceFloatControlsProperties controls{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &controls;
        vkGetPhysicalDeviceProperties2(pd, &properties);
        if (!controls.shaderSignedZeroInfNanPreserveFloat32) {
            std::printf("unsupported: shaderSignedZeroInfNanPreserveFloat32=0\n");
            return 77;
        }
    }
    const bool filled = load_mode || words_mode;
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
    if (code.size() < 20 || code.size() % 4) { std::fprintf(stderr, "%s: not a SPIR-V module\n", argv[1]); return 1; }
    // shaderInt64 only for a module with 64-bit integers: SHADER-423's address pairs need none (TEST-468).
    const bool int64 = leonida::vk_test::needs_shader_int64(code.data(), code.size());
    if (int64 && !have.shaderInt64) {
        std::printf("unsupported: %s declares Int64 and the device has no shaderInt64\n", argv[1]);
        return 77;
    }
    using leonida::vk_test::spirv_declares_capability;
    const bool dynamic_sampled = spirv_declares_capability(code.data(), code.size(),
        leonida::vk_test::spv_sampled_image_array_dynamic_indexing);
    if (dynamic_sampled && !have.shaderSampledImageArrayDynamicIndexing) {
        std::printf("unsupported: shaderSampledImageArrayDynamicIndexing\n");
        return 77;
    }
    f2.features.shaderSampledImageArrayDynamicIndexing = dynamic_sampled;
    const bool dynamic_storage = spirv_declares_capability(code.data(), code.size(),
        leonida::vk_test::spv_storage_image_array_dynamic_indexing);
    if (dynamic_storage && !have.shaderStorageImageArrayDynamicIndexing) {
        std::printf("unsupported: shaderStorageImageArrayDynamicIndexing\n");
        return 77;
    }
    f2.features.shaderStorageImageArrayDynamicIndexing = dynamic_storage;
    // Typed image_load fetches through the sampled arrays without StorageImageReadWithoutFormat (55, SHADER-151).
    if (filled && spirv_declares_capability(code.data(), code.size(), 55)) {
        std::printf("FAILED: the module declares StorageImageReadWithoutFormat\n");
        return 1;
    }
    f2.features.shaderInt64 = int64 ? VK_TRUE : VK_FALSE;
    f2.pNext = &f12;
    leonida::vk_test::ShaderFma fma;      // OpFmaKHR (rdna2spirv.py --fma khr)
    fma.query(gipa, inst, pd);
    if (!fma.accepts(code.data(), code.size(), argv[1])) return 77;
    fma.link_after(&f2);
    const char* fma_extension = fma.extension();
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.pNext = &f2;
    if (fma.supported()) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &fma_extension; }
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LOADD(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(dev, #name))
    LOADD(vkGetDeviceQueue); LOADD(vkCreateBuffer); LOADD(vkGetBufferMemoryRequirements); LOADD(vkAllocateMemory);
    LOADD(vkBindBufferMemory); LOADD(vkMapMemory); LOADD(vkCreateShaderModule); LOADD(vkCreateImage);
    LOADD(vkGetImageMemoryRequirements); LOADD(vkBindImageMemory); LOADD(vkCreateImageView); LOADD(vkCreateSampler);
    LOADD(vkCreateDescriptorSetLayout); LOADD(vkCreatePipelineLayout); LOADD(vkCreateComputePipelines);
    LOADD(vkCreateDescriptorPool); LOADD(vkAllocateDescriptorSets); LOADD(vkUpdateDescriptorSets);
    LOADD(vkCreateCommandPool); LOADD(vkAllocateCommandBuffers); LOADD(vkBeginCommandBuffer);
    LOADD(vkCmdBindPipeline); LOADD(vkCmdBindDescriptorSets); LOADD(vkCmdPushConstants); LOADD(vkCmdDispatch);
    LOADD(vkCmdPipelineBarrier); LOADD(vkCmdCopyImageToBuffer); LOADD(vkCmdCopyBufferToImage);
    LOADD(vkEndCommandBuffer); LOADD(vkQueueSubmit);
    LOADD(vkQueueWaitIdle);
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memtype = [&](uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0u;
    };
    auto make_buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* b, void** map) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = usage;
        VkResult r = vkCreateBuffer(dev, &bi, nullptr, b);
        if (r) return r;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, *b, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory m;
        if ((r = vkAllocateMemory(dev, &mai, nullptr, &m))) return r;
        if ((r = vkBindBufferMemory(dev, *b, m, 0))) return r;
        return vkMapMemory(dev, m, 0, size, 0, map);
    };

    // T#s exactly as the shader builds them (cs_imgstore.asm): word0 = 0x1000 * (k + 1), word1 = FORMAT << 20,
    // words 2..7 = 0x11 * (k + 1).
    const uint32_t fmts[3] = {56, 20, 21};
    const VkFormat compact_formats[3] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT};
    const VkFormat rgba32_formats[3] = {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT,
                                        VK_FORMAT_R32G32B32A32_SINT};
    const VkFormat* vkf = signed_zero ? rgba32_formats : compact_formats;
    std::vector<uint32_t> table(1 + 16 * 10 + 10, 0);
    table[0] = 15;
    for (uint32_t k = 0; k < 3; k++) {
        uint32_t t[8] = {0x1000 * (k + 1), fmts[k] << 20};
        for (int j = 2; j < 8; j++) t[j] = 0x11 * (k + 1);
        table_insert(table, t, k + 1);     // its index in the 2D arrays; 0 is the fallback, as in the runtime
    }
    // Index 4: a 16x4 rgba8 image with its whole mip chain (16x4, 8x2, 4x1, 2x1, 1x1) for image_get_resinfo. Its key
    // is the T# of exactly that image (gfx10_descriptors.h), which cs_resinfo.asm builds word for word.
    const uint32_t WIDE_W = 16, WIDE_H = 4, WIDE_MIPS = 5;
    leonida::gfx10::Image wide_desc{};
    wide_desc.address = 0x400000; wide_desc.format = fmts[0]; wide_desc.width = WIDE_W; wide_desc.height = WIDE_H;
    wide_desc.mip_count = wide_desc.resource_mips = WIDE_MIPS;
    const leonida::gfx10::ImageWords wide_key = leonida::gfx10::encode(wide_desc);
    table_insert(table, wide_key.data(), 4);
    // The volume of the 3D atomics: index 1 of the 3D arrays, behind a T# of the 2D images' pattern (k = 4).
    const uint32_t VOL = 4;
    {
        uint32_t t[8] = {0x5000, fmts[1] << 20, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55};
        table_insert(table, t, 1);
    }
    VkBuffer tb; void* tbp;
    CHECK(make_buffer(table.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tb, &tbp));
    std::memcpy(tbp, table.data(), table.size() * 4);

    const uint32_t W = 8;
    VkImage img[3]; VkImageView view[3];
    for (int k = 0; k < 3; k++) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = vkf[k]; ii.extent = {W, W, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        CHECK(vkCreateImage(dev, &ii, nullptr, &img[k]));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, img[k], &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size; mai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m; CHECK(vkAllocateMemory(dev, &mai, nullptr, &m));
        CHECK(vkBindImageMemory(dev, img[k], m, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img[k]; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = vkf[k];
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        CHECK(vkCreateImageView(dev, &vi, nullptr, &view[k]));
    }
    VkImage wide; VkImageView wide_view;
    {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = vkf[0]; ii.extent = {WIDE_W, WIDE_H, 1};
        ii.mipLevels = WIDE_MIPS; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        CHECK(vkCreateImage(dev, &ii, nullptr, &wide));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, wide, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size; mai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m; CHECK(vkAllocateMemory(dev, &mai, nullptr, &m));
        CHECK(vkBindImageMemory(dev, wide, m, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = wide; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = vkf[0];
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, WIDE_MIPS, 0, 1};
        CHECK(vkCreateImageView(dev, &vi, nullptr, &wide_view));
    }
    VkImage vol; VkImageView vol_view;
    {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_3D; ii.format = vkf[1]; ii.extent = {VOL, VOL, VOL};
        ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        CHECK(vkCreateImage(dev, &ii, nullptr, &vol));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, vol, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size; mai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m; CHECK(vkAllocateMemory(dev, &mai, nullptr, &m));
        CHECK(vkBindImageMemory(dev, vol, m, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = vol; vi.viewType = VK_IMAGE_VIEW_TYPE_3D; vi.format = vkf[1];
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        CHECK(vkCreateImageView(dev, &vi, nullptr, &vol_view));
    }
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    VkSampler sampler; CHECK(vkCreateSampler(dev, &sci, nullptr, &sampler));
    // Set 0: binding 0 = sampled images, 1 = samplers, 2 = hash table, 8/9/10 = float/uint/sint 2D storage image arrays.
    // Five elements each: index 0 stands for the runtime's fallback, image k is at index k + 1 and the 16x4 image at
    // index 4. Binding 0 holds each image's view in its own format (typed image_load reads the uint and sint ones
    // through integer aliases); every element of a storage array holds the image of its class, so all of them are
    // valid descriptors. Bindings 4 and 7 = the 2D and 3D r32ui storage images of the image atomics: the uint image
    // and the volume, in every element.
    constexpr uint32_t N = 5;
    VkDescriptorSetLayoutBinding b[8]{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, N, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    for (int k = 0; k < 3; k++) b[3 + k] = {uint32_t(8 + k), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, N, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[6] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, N, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[7] = {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, N, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 8; dli.pBindings = b;
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
    VkDescriptorPoolSize ps[4] = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, N},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5 * N},
    };
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 4; dpi.pPoolSizes = ps;
    VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev, &dai, &ds));
    VkDescriptorBufferInfo tbi{tb, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo sii[N] = {{VK_NULL_HANDLE, view[0], VK_IMAGE_LAYOUT_GENERAL},
                                    {VK_NULL_HANDLE, view[0], VK_IMAGE_LAYOUT_GENERAL},
                                    {VK_NULL_HANDLE, view[1], VK_IMAGE_LAYOUT_GENERAL},
                                    {VK_NULL_HANDLE, view[2], VK_IMAGE_LAYOUT_GENERAL},
                                    {VK_NULL_HANDLE, wide_view, VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorImageInfo smpi{sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    VkDescriptorImageInfo dii[3][N];
    VkDescriptorImageInfo r32ui[N], r32ui_3d[N];
    VkWriteDescriptorSet wr[8]{};
    for (auto& w : wr) { w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet = ds; w.descriptorCount = 1; }
    wr[0].dstBinding = 0; wr[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; wr[0].pImageInfo = sii;
    wr[0].descriptorCount = N;
    wr[1].dstBinding = 1; wr[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; wr[1].pImageInfo = &smpi;
    wr[2].dstBinding = 2; wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[2].pBufferInfo = &tbi;
    for (int k = 0; k < 3; k++) {
        for (uint32_t e = 0; e < N; e++) dii[k][e] = {VK_NULL_HANDLE, view[k], VK_IMAGE_LAYOUT_GENERAL};
        wr[3 + k].dstBinding = 8 + k; wr[3 + k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wr[3 + k].pImageInfo = dii[k]; wr[3 + k].descriptorCount = N;
    }
    for (uint32_t e = 0; e < N; e++) r32ui[e] = {VK_NULL_HANDLE, view[1], VK_IMAGE_LAYOUT_GENERAL};
    wr[6].dstBinding = 4; wr[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr[6].pImageInfo = r32ui; wr[6].descriptorCount = N;
    for (uint32_t e = 0; e < N; e++) r32ui_3d[e] = {VK_NULL_HANDLE, vol_view, VK_IMAGE_LAYOUT_GENERAL};
    wr[7].dstBinding = 7; wr[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr[7].pImageInfo = r32ui_3d; wr[7].descriptorCount = N;
    vkUpdateDescriptorSets(dev, 8, wr, 0, nullptr);

    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size(); smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
    // LEONIDA_PC_TAIL_UBO=1 (rdna2spirv.py --pc-tail-ubo): s0..s31 only, 128 bytes; a module that reads the tail
    // gets it, all zero as without the flag, from a uniform block at set 1 (vk_pc_tail.h).
    const bool tail_block = pc_tail &&
        leonida::vk_test::spirv_declares_set(code.data(), code.size(), leonida::vk_test::pc_tail_set);
    leonida::vk_test::PcTail tail;
    if (tail_block && !tail.create(vkGetDeviceProcAddr, dev, mp, props.limits.minUniformBufferOffsetAlignment,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 1)) {
        std::fprintf(stderr, "FAILED creating the --pc-tail-ubo block\n");
        return 1;
    }
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    const VkDescriptorSetLayout set_layouts[] = {dsl, tail.set_layout};
    if (tail_block) { pli.setLayoutCount = 2; pli.pSetLayouts = set_layouts; }
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
    cpi.layout = pl;
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));

    const uint32_t image_words = W * W * (signed_zero ? 4 : 1);
    VkBuffer rb[3]; void* rbp[3];
    for (int k = 0; k < 3; k++) CHECK(make_buffer(image_words * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb[k], &rbp[k]));
    // Load and words modes: what the uint and sint images hold before the dispatch, texel (x, y) at i = 8y + x.
    // The last row of the D16 fixture ("load-d16" only) exercises both sides of each 16-bit boundary.
    struct PackedLoad { uint32_t u; int32_t s, sum; };
    constexpr PackedLoad boundary[] = {
        {65535, 0, 65535}, {65536, 0, 65535}, {65537, 0, 65535}, {0, 32767, 32767},
        {0, 32768, 32767}, {0, -32768, -32768}, {0, -32769, -32768}, {65536, -32769, 32767},
    };
    static_assert(sizeof(boundary) / sizeof(boundary[0]) == W);
    // Load/words modes seed the integer images. The signed-zero probe seeds every component of all three images.
    const int first_fill = signed_zero ? 0 : filled ? 1 : 3;
    VkBuffer fill[3]{};
    for (int k = first_fill; k < 3; ++k) {
        void* mapped;
        CHECK(make_buffer(image_words * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &fill[k], &mapped));
        auto* words = static_cast<uint32_t*>(mapped);
        for (uint32_t i = 0; i < image_words; ++i) {
            const bool edge = packed_load && i / W == W - 1;
            if (signed_zero) words[i] = 0x39000000u + k * image_words + i;
            else if (k == 1) words[i] = edge ? boundary[i % W].u : 100 + i;
            else words[i] = static_cast<uint32_t>(edge ? boundary[i % W].s : -2 * int32_t(i));
        }
    }
    // The volume's fill and its read-back, used in words mode.
    const uint32_t vol_texels = VOL * VOL * VOL;
    VkBuffer vol_fill{}, vol_rb{}; void* vol_fillp = nullptr; void* vol_rbp = nullptr;
    if (words_mode) {
        CHECK(make_buffer(vol_texels * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &vol_fill, &vol_fillp));
        CHECK(make_buffer(vol_texels * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &vol_rb, &vol_rbp));
        for (uint32_t i = 0; i < vol_texels; i++) static_cast<uint32_t*>(vol_fillp)[i] = 200 + i;
    }
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = qi;
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &cbi));
    VkImageMemoryBarrier ib[3]{};
    for (int k = 0; k < 3; k++) {
        ib[k].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; ib[k].image = img[k];
        ib[k].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; ib[k].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib[k].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        ib[k].srcQueueFamilyIndex = ib[k].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib[k].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (k >= first_fill) {   // filled by a copy first
            ib[k].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib[k].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        }
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 3, ib);
    VkImageMemoryBarrier wide_ib = ib[0];   // every level: the descriptor names the whole image in GENERAL
    wide_ib.image = wide; wide_ib.newLayout = VK_IMAGE_LAYOUT_GENERAL; wide_ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    wide_ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, WIDE_MIPS, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &wide_ib);
    VkImageMemoryBarrier vol_ib = ib[0];
    vol_ib.image = vol; vol_ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    vol_ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    const VkImageSubresourceLayers vol_layers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    if (words_mode) {   // filled by a copy first
        vol_ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; vol_ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &vol_ib);
        VkBufferImageCopy bic{};
        bic.imageSubresource = vol_layers; bic.imageExtent = {VOL, VOL, VOL};
        vkCmdCopyBufferToImage(cmd, vol_fill, vol, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
        vol_ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; vol_ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vol_ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vol_ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &vol_ib);
    } else {
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &vol_ib);
    }
    if (first_fill < 3) {
        for (int k = first_fill; k < 3; k++) {
            VkBufferImageCopy bic{};
            bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bic.imageExtent = {W, W, 1};
            vkCmdCopyBufferToImage(cmd, fill[k], img[k], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
            ib[k].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib[k].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            ib[k].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            ib[k].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 3 - first_fill, ib + first_fill);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    uint32_t pc[36] = {};
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes, pc);
    if (tail_block) {
        const uint32_t offset = tail.offset(0);   // PcTail starts zeroed: the tail stays all zero, as pushed before
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, leonida::vk_test::pc_tail_set, 1, &tail.set,
                                1, &offset);
    }
    vkCmdDispatch(cmd, 1, 1, 1);
    for (int k = 0; k < 3; k++) {
        ib[k].oldLayout = VK_IMAGE_LAYOUT_GENERAL; ib[k].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        ib[k].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        ib[k].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 3, ib);
    for (int k = 0; k < 3; k++) {
        VkBufferImageCopy bic{};
        bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bic.imageExtent = {W, W, 1};
        vkCmdCopyImageToBuffer(cmd, img[k], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb[k], 1, &bic);
    }
    if (words_mode) {
        vol_ib.oldLayout = VK_IMAGE_LAYOUT_GENERAL; vol_ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        vol_ib.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vol_ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &vol_ib);
        VkBufferImageCopy bic{};
        bic.imageSubresource = vol_layers; bic.imageExtent = {VOL, VOL, VOL};
        vkCmdCopyImageToBuffer(cmd, vol, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vol_rb, 1, &bic);
    }
    // Make every copy's readback writes visible to the host before it reads the mapped buffers.
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &host, 0, nullptr, 0, nullptr);
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(q));

    if (signed_zero) {
        constexpr uint32_t neg = 0x80000000;
        constexpr uint32_t stores[] = {neg, 0, 0, 0, neg, 0x3f800000, 0, 0, neg, neg, 0, 0};
        const uint32_t total = 3 * image_words;
        uint32_t bad = 0;
        for (uint32_t i = 0; i < total; ++i) {
            const uint32_t expected = i < 12 ? stores[i] : 0x39000000u + i;
            const uint32_t actual = static_cast<const uint32_t*>(rbp[i / image_words])[i % image_words];
            if (actual != expected) {
                std::printf("word %u: expected %08x, observed %08x\n", i, expected, actual);
                ++bad;
            }
        }
        std::printf("%u/%u words exact\n", total - bad, total);
        return bad ? 1 : 0;
    }
    if (words_mode) {
        const char* names[3] = {"uint", "sint", "volume"};
        const void* words[3] = {rbp[1], rbp[2], vol_rbp};
        static_assert(W * W == VOL * VOL * VOL, "64 words an image");
        for (int k = 0; k < 3; k++) {
            std::printf("%s words:", names[k]);
            for (uint32_t i = 0; i < W * W; i++) std::printf(" 0x%08x", static_cast<const uint32_t*>(words[k])[i]);
            std::printf("\n");
        }
        return 0;
    }
    uint32_t bad[3] = {};
    for (uint32_t y = 0; y < W; y++)
        for (uint32_t x = 0; x < W; x++) {
            uint32_t i = y * W + x;
            const uint8_t* c = static_cast<const uint8_t*>(rbp[0]) + 4 * i;
            int er = int(std::lround(x / 7.0 * 255)), eg = int(std::lround(y / 7.0 * 255));
            if (std::abs(c[0] - er) > 1 || std::abs(c[1] - eg) > 1 || std::abs(c[2] - 128) > 1 || c[3] != 255) bad[0]++;
            const bool edge = packed_load && y == W - 1;
            if (static_cast<const uint32_t*>(rbp[1])[i] != (edge ? boundary[x].u : 100 + i)) bad[1]++;
            // load: the uint texel (100 + 8y + x) plus the sint one (-2 (8y + x)), both read with image_load
            const int32_t want_s = edge ? boundary[x].sum : load_mode ? 100 - int32_t(i) : -int32_t(i);
            if (static_cast<const int32_t*>(rbp[2])[i] != want_s) bad[2]++;
        }
    const uint8_t* c = static_cast<const uint8_t*>(rbp[0]) + 4 * (3 * W + 5);
    std::printf("rgba8 unorm (float view): %u bad, texel(5,3) = %u %u %u %u\n", bad[0], c[0], c[1], c[2], c[3]);
    std::printf("r32 uint (uint view): %u bad, texel(5,3) = %u\n", bad[1], static_cast<const uint32_t*>(rbp[1])[3 * W + 5]);
    std::printf("r32 sint (sint view): %u bad, texel(5,3) = %d\n", bad[2], static_cast<const int32_t*>(rbp[2])[3 * W + 5]);
    return (bad[0] || bad[1] || bad[2]) ? 1 : 0;
}
