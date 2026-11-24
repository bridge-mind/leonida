// GPU test of the bindless set (src/vulkan_bindless.h) and the descriptor hash table (src/descriptor_table.h) with
// a translated shader. ps_tex.asm reads a T# and an S# from a descriptor table in the heap, looks both up in the
// table at set 0 binding 2 and samples the image and sampler the indices name.
//
//   vk_bindless_test <vs_ids.spv> <ps_tex.spv>
//
// A full-screen triangle is drawn into 64x64 RGBA32F and read back, eight times:
//   1. the T# and S# are in the table behind keys that share the T#'s first slot, so the shader has to probe:
//      every pixel is the texel of an 8x8 gradient under it;
//   2. the T# is erased: every pixel is the fallback at index 0 (zeros);
//   3. the T# is back among 200 more keys and the table has grown into a new buffer: the gradient again;
//   4. the T# is rebound to the decoy index: that image's colour;
//   5. the T# is rebound to the original index: the gradient again;
//   6. the T# is rebound again, but the command buffer that carried the upload is never submitted and the set is
//      told so: the retry still has to show the decoy's colour, not the old mapping;
//   7. the original index is given another view: the decoy image's colour;
//   8. the original index is removed: the fallback.
// This runs with the descriptor-indexing flags the device has (update after bind, partially bound) and again
// without any. Exit 77 with "unsupported: ..." when the device lacks a feature the shaders need.
// LEONIDA_VK_VALIDATION=1 runs under VK_LAYER_KHRONOS_validation; a validation error fails the test.
#define VK_NO_PROTOTYPES
#include "vulkan_loader.h"
#include "vk_device_pick.h"
#include "vk_shader_fma.h"
#include "vk_pc_tail.h"
#include "spirv_capability.h"
#include "../src/vulkan_bindless.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>

#define CHECK(x)                                                                     \
    do {                                                                             \
        VkResult r_ = (x);                                                           \
        if (r_ != VK_SUCCESS) {                                                      \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, r_, __LINE__);   \
            return 1;                                                                \
        }                                                                            \
    } while (0)
#define EXPECT(condition)                                                            \
    do {                                                                             \
        if (!(condition)) {                                                          \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);             \
            return 1;                                                                \
        }                                                                            \
    } while (0)

namespace {

using namespace leonida;

constexpr int unsupported = 77;
unsigned g_validation_errors = 0;

VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                          VkDebugUtilsMessageTypeFlagsEXT types,
                                          const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    // Only validation-type errors count: a broken third-party layer must not fail the test.
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) && (types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT))
        ++g_validation_errors;
    std::fprintf(stderr, "validation: %s\n", data && data->pMessage ? data->pMessage : "(no message)");
    return VK_FALSE;
}

std::vector<std::uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
    std::vector<std::uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), words.size() * 4);
    return words;
}

// Expected RGBA values in this test are finite. Report the worst channel error per pixel.
float pixel_error(const float* pixel, const float* want) {
    float error = 0;
    for (int c = 0; c < 4; ++c) {
        if (!std::isfinite(pixel[c])) return std::numeric_limits<float>::infinity();
        error = std::max(error, std::fabs(pixel[c] - want[c]));
    }
    return error;
}

DescriptorKey numbered(std::uint32_t n) {
    return {n, 0x9e3779b9u * n, n ^ 0x55555555u, n >> 3, 7, n * n, 0x80000000u, ~n};
}

// How many slots the shader reads before it finds `key` (8: it does not), from the buffer contents alone.
std::uint32_t probe_depth(std::span<const std::uint32_t> table, const DescriptorKey& key) {
    const std::uint32_t h = hash_descriptor_key(key), mask = table[0];
    for (std::uint32_t k = 0; k < 8; ++k) {
        const std::uint32_t base = ((h + k) & mask) * 10;
        bool match = true;
        for (int j = 0; j < 8; ++j) match = match && table[base + 1 + j] == key[j];
        if (match) return k;
    }
    return 8;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: vk_bindless_test <vs_ids.spv> <ps_tex.spv | ps_imgatomic.spv>\n");
        return 2;
    }
    // A bad readback in a single channel must still fail when all three others match.
    const float black[4]{};
    for (int c = 0; c < 4; ++c) {
        float pixel[4]{};
        pixel[c] = std::numeric_limits<float>::quiet_NaN();
        EXPECT(std::isinf(pixel_error(pixel, black)));
    }
    constexpr std::uint32_t W = 64, H = 64, TW = 8;
    constexpr std::uint32_t table_va = 256;  // console VA of the descriptor table: T# at +0, S# at +32
    // Any 8 dwords as the lookup key, except FORMAT (word 1 [28:20]): the translator picks the float, uint or sint
    // view of the image from it, and every view this test binds is R8G8B8A8_UNORM, so it is 56 (8_8_8_8_UNORM).
    // 0x22222222 had 34 (10_11_11_UINT): every sample went through the uint alias of a UNORM view, which Vulkan
    // leaves undefined (VUID-vkCmdDraw-format-07753, reported by GPU-assisted validation).
    constexpr std::uint32_t tsharp_word1 = (0x22222222u & ~(0x1FFu << 20)) | (56u << 20);
    static_assert(((tsharp_word1 >> 20) & 0x1FF) == 56, "the T#'s FORMAT must be 8_8_8_8_UNORM");
    const DescriptorKey tsharp{0x11111111, tsharp_word1, 0x33333333, 0x44444444,
                               0x55555555, 0x66666666, 0x77777777, 0x88888888};
    const std::array<std::uint32_t, 4> ssharp{0x5a5a0001, 0x00000002, 0x00000003, 0x00000004};

    const PFN_vkGetInstanceProcAddr gipa = load_vulkan();
    if (!gipa) return 1;
    VkInstance inst = VK_NULL_HANDLE;
#define LI(n) const auto n = reinterpret_cast<PFN_##n>(gipa(inst, #n))
    LI(vkCreateInstance);
    LI(vkEnumerateInstanceLayerProperties);
    const char* env = std::getenv("LEONIDA_VK_VALIDATION");
    const bool validation = env && std::strcmp(env, "0") != 0;
    const char* layer = "VK_LAYER_KHRONOS_validation";
    const char* debug_utils = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    VkDebugUtilsMessengerCreateInfoEXT messenger_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messenger_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    messenger_info.pfnUserCallback = on_message;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (validation) {
        std::uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> layers(count);
        vkEnumerateInstanceLayerProperties(&count, layers.data());
        bool found = false;
        for (const auto& properties : layers) found = found || std::strcmp(properties.layerName, layer) == 0;
        if (!found) {
            std::fprintf(stderr, "LEONIDA_VK_VALIDATION is set but %s is not installed\n", layer);
            return 1;
        }
        ici.pNext = &messenger_info;
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &layer;
        ici.enabledExtensionCount = 1;
        ici.ppEnabledExtensionNames = &debug_utils;
    }
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (validation) {
        LI(vkCreateDebugUtilsMessengerEXT);
        CHECK(vkCreateDebugUtilsMessengerEXT(inst, &messenger_info, nullptr, &messenger));
    }
    LI(vkEnumeratePhysicalDevices); LI(vkGetPhysicalDeviceProperties); LI(vkGetPhysicalDeviceQueueFamilyProperties);
    LI(vkGetPhysicalDeviceMemoryProperties); LI(vkGetPhysicalDeviceFeatures2); LI(vkGetPhysicalDeviceProperties2);
    LI(vkCreateDevice); LI(vkGetDeviceProcAddr);
    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    // The same choice as the other tools (vk_device_pick.h): the best-ranked Vulkan 1.2 device that LEONIDA_VK_DEVICE
    // allows. It used to rank every device and ignore the filter, so a run meant for one GPU took another (TEST-469).
    // A filter that matches nothing fails, as in vk_compute_test and vk_test_device.h (as #2657 does it).
    std::vector<VkPhysicalDeviceProperties> candidates;
    std::vector<VkPhysicalDevice> eligible;
    for (auto d : pds) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.apiVersion < VK_API_VERSION_1_2) continue;
        candidates.push_back(p);
        eligible.push_back(d);
    }
    const std::size_t chosen = leonida::vk_test::pick_device(candidates);
    VkPhysicalDevice pd = chosen < eligible.size() ? eligible[chosen] : VK_NULL_HANDLE;
    if (!pd) {
        if (!leonida::vulkan_device_filter().empty()) {
            leonida::vk_test::report_no_device();
            return 1;
        }
        std::printf("unsupported: no Vulkan 1.2 device\n");
        return unsupported;
    }
    VkPhysicalDeviceVulkan12Properties props12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties props11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &props12};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &props11};
    vkGetPhysicalDeviceProperties2(pd, &props);
    std::printf("device: %s%s\n", props.properties.deviceName, validation ? " (validation on)" : "");
    // The pipeline layout pushes the 144-byte translated user-data block. A device with fewer push constant bytes
    // (128 on Intel Arc and some AMD drivers) can't create it: a device limit, not a wrong result (TEST-149).
    // LEONIDA_PC_TAIL_UBO=1 (rdna2spirv.py --pc-tail-ubo): s0..s31 only, 128 bytes; the heap base, if a stage reads
    // it, is in a uniform block at set 1 (vk_pc_tail.h).
    const bool pc_tail = leonida::vk_test::pc_tail_requested();
    const std::uint32_t push_bytes = pc_tail ? 128u : 144u;
    if (props.properties.limits.maxPushConstantsSize < push_bytes) {
        std::printf("unsupported: need %u push constant bytes, device has %u\n", push_bytes,
                    props.properties.limits.maxPushConstantsSize);
        return unsupported;
    }

    // The features ps_tex's metadata names (bufferDeviceAddress, runtimeDescriptorArray), dynamic indexing of the
    // image array, and whichever descriptor-indexing flags the set can use. shaderInt64 only when a stage has 64-bit
    // integers: SHADER-423's address pairs need none (TEST-468).
    VkPhysicalDeviceVulkan11Features have11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &have11};
    VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &have12};
    vkGetPhysicalDeviceFeatures2(pd, &have);
    const bool int64 = leonida::vk_test::needs_shader_int64(read_spv(argv[1])) ||
                       leonida::vk_test::needs_shader_int64(read_spv(argv[2]));
    if (!have12.bufferDeviceAddress || !have12.runtimeDescriptorArray || (int64 && !have.features.shaderInt64) ||
        !have.features.shaderSampledImageArrayDynamicIndexing) {
        std::printf("unsupported: the device lacks bufferDeviceAddress, runtimeDescriptorArray, %s"
                    "shaderSampledImageArrayDynamicIndexing\n", int64 ? "shaderInt64 (a stage declares Int64) or " : "");
        return unsupported;
    }
    BindlessFeatures device_flags;
    device_flags.update_after_bind = have12.descriptorBindingSampledImageUpdateAfterBind &&
                                     have12.descriptorBindingStorageImageUpdateAfterBind &&
                                     have12.descriptorBindingStorageBufferUpdateAfterBind;
    device_flags.partially_bound = have12.descriptorBindingPartiallyBound == VK_TRUE;
    // Translated vertex shaders read BaseVertex / BaseInstance (DrawParameters) for the zero-based v0 / v3.
    VkPhysicalDeviceVulkan11Features enable11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    enable11.shaderDrawParameters = have11.shaderDrawParameters;
    VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &enable11};
    enable12.bufferDeviceAddress = VK_TRUE;
    enable12.runtimeDescriptorArray = VK_TRUE;
    enable12.descriptorIndexing = have12.descriptorIndexing;
    enable12.descriptorBindingSampledImageUpdateAfterBind = device_flags.update_after_bind;
    enable12.descriptorBindingStorageImageUpdateAfterBind = device_flags.update_after_bind;
    enable12.descriptorBindingStorageBufferUpdateAfterBind = device_flags.update_after_bind;
    enable12.descriptorBindingPartiallyBound = device_flags.partially_bound;
    VkPhysicalDeviceFeatures2 enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &enable12};
    enable.features.shaderInt64 = int64 ? VK_TRUE : VK_FALSE;
    enable.features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    const std::vector<std::uint32_t> shader_code[] = {read_spv(argv[1]), read_spv(argv[2])};
    bool dynamic_storage = false;
    for (int stage = 0; stage < 2; ++stage) {
        const auto& code = shader_code[stage];
        if (code.empty()) {
            std::fprintf(stderr, "cannot read %s\n", argv[1 + stage]);
            return 1;
        }
        dynamic_storage = dynamic_storage || leonida::vk_test::spirv_declares_capability(
            code.data(), code.size() * sizeof(code[0]), leonida::vk_test::spv_storage_image_array_dynamic_indexing);
    }
    if (dynamic_storage && !have.features.shaderStorageImageArrayDynamicIndexing) {
        std::printf("unsupported: shaderStorageImageArrayDynamicIndexing\n");
        return unsupported;
    }
    enable.features.shaderStorageImageArrayDynamicIndexing = dynamic_storage;
    // ps_imgatomic: image atomics in a pixel shader need fragment stores and atomics.
    const bool atomic_mode = std::strstr(argv[2], "imgatomic") != nullptr;
    if (atomic_mode) {
        if (!have.features.fragmentStoresAndAtomics) {
            std::printf("unsupported: fragmentStoresAndAtomics (image atomics in a pixel shader)\n");
            return unsupported;
        }
        enable.features.fragmentStoresAndAtomics = VK_TRUE;
    }

    std::uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    std::uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT)) ++qi;
    EXPECT(qi < qn);
    const float priority = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    leonida::vk_test::ShaderFma fma;      // OpFmaKHR (rdna2spirv.py --fma khr), when the device has it
    fma.query(gipa, inst, pd);
    // A device without VK_KHR_shader_fma cannot create a pipeline from a stage that uses OpFmaKHR: skip the pair.
    if (!fma.accepts_file(argv[1]) || !fma.accepts_file(argv[2])) return 77;
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
    LD(vkEndCommandBuffer); LD(vkQueueSubmit); LD(vkQueueWaitIdle); LD(vkCreateBuffer); LD(vkResetCommandBuffer);
    LD(vkGetBufferMemoryRequirements); LD(vkBindBufferMemory); LD(vkMapMemory); LD(vkGetBufferDeviceAddress);
    LD(vkCreateSampler); LD(vkCmdBindDescriptorSets); LD(vkCmdPushConstants); LD(vkCmdPipelineBarrier);
    LD(vkDeviceWaitIdle); LD(vkDestroyPipeline); LD(vkDestroyPipelineLayout);
    LD(vkDestroyCommandPool); LD(vkDestroyShaderModule); LD(vkDestroyFramebuffer); LD(vkDestroyRenderPass);
    LD(vkDestroySampler); LD(vkDestroyImageView); LD(vkDestroyImage); LD(vkDestroyBuffer);
    LD(vkFreeMemory); LD(vkDestroyDevice); LD(vkCmdClearColorImage);
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev, qi, 0, &queue);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    const auto memtype = [&](std::uint32_t bits, VkMemoryPropertyFlags want) {
        for (std::uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0u;
    };
    std::vector<VkDeviceMemory> allocations;
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
        allocations.push_back(m);
        r = vkBindBufferMemory(dev, *b, m, 0);
        if (r) return r;
        return vkMapMemory(dev, m, 0, size, 0, map);
    };
    const auto make_image = [&](VkFormat format, std::uint32_t w, std::uint32_t h, VkImageUsageFlags usage, VkImage* image,
                                VkImageView* view) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = format;
        ii.extent = {w, h, 1};
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.usage = usage;
        VkResult r = vkCreateImage(dev, &ii, nullptr, image);
        if (r) return r;
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(dev, *image, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m;
        r = vkAllocateMemory(dev, &ai, nullptr, &m);
        if (r) return r;
        allocations.push_back(m);
        r = vkBindImageMemory(dev, *image, m, 0);
        if (r) return r;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = *image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        return vkCreateImageView(dev, &vi, nullptr, view);
    };

    // The heap (the console's GPU address window): zeros, with the descriptor table at table_va.
    VkBuffer heap;
    void* heap_map;
    CHECK(make_buffer(65536, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, &heap, &heap_map));
    std::memset(heap_map, 0, 65536);
    std::memcpy(static_cast<char*>(heap_map) + table_va, tsharp.data(), 32);
    std::memcpy(static_cast<char*>(heap_map) + table_va + 32, ssharp.data(), 16);
    VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address_info.buffer = heap;
    const VkDeviceAddress heap_base = vkGetBufferDeviceAddress(dev, &address_info);

    // Textures: an 8x8 gradient (r = x / 8, g = y / 8, b = 0.5, a = 1 in steps of 32/255), a 1x1 orange decoy, and
    // a 1x1 image with storage usage to go through the storage bindings.
    constexpr VkImageUsageFlags sampled_usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImage gradient, decoy, stored;
    VkImageView gradient_view, decoy_view, stored_view;
    CHECK(make_image(VK_FORMAT_R8G8B8A8_UNORM, TW, TW, sampled_usage, &gradient, &gradient_view));
    CHECK(make_image(VK_FORMAT_R8G8B8A8_UNORM, 1, 1, sampled_usage, &decoy, &decoy_view));
    CHECK(make_image(VK_FORMAT_R8G8B8A8_UNORM, 1, 1, sampled_usage | VK_IMAGE_USAGE_STORAGE_BIT, &stored, &stored_view));
    std::vector<std::uint32_t> texels(TW * TW);
    for (std::uint32_t y = 0; y < TW; y++)
        for (std::uint32_t x = 0; x < TW; x++) texels[y * TW + x] = (x * 32u) | ((y * 32u) << 8) | (128u << 16) | (255u << 24);
    constexpr std::uint32_t orange = 0xff0080ffu;  // r = 255, g = 128, b = 0, a = 255
    VkBuffer upload;
    void* upload_map;
    CHECK(make_buffer(4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &upload, &upload_map));
    std::memcpy(upload_map, texels.data(), texels.size() * 4);
    std::memcpy(static_cast<char*>(upload_map) + 1024, &orange, 4);
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler nearest;
    CHECK(vkCreateSampler(dev, &sci, nullptr, &nearest));

    // Render target, pass and framebuffer.
    constexpr VkFormat target_format = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkImage target;
    VkImageView target_view;
    CHECK(make_image(target_format, W, H, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &target,
                     &target_view));
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
    // The read back that follows the pass waits for its last write and its final layout transition.
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
    VkBuffer readback;
    void* readback_map;
    CHECK(make_buffer(VkDeviceSize(W) * H * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback, &readback_map));

    VkShaderModule vs, fs;
    bool tail_block = false;
    for (int stage = 0; stage < 2; ++stage) {
        const auto& code = shader_code[stage];
        if (pc_tail && leonida::vk_test::spirv_declares_set(code.data(), code.size() * 4, leonida::vk_test::pc_tail_set))
            tail_block = true;
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size() * 4;
        smi.pCode = code.data();
        CHECK(vkCreateShaderModule(dev, &smi, nullptr, stage ? &fs : &vs));
    }
    // One tail block for every variant below: each draw's tail is the same heap base.
    leonida::vk_test::PcTail tail;
    if (tail_block && !tail.create(vkGetDeviceProcAddr, dev, mp, props.properties.limits.minUniformBufferOffsetAlignment,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1)) {
        std::fprintf(stderr, "FAILED creating the --pc-tail-ubo block\n");
        return 1;
    }
    auto bind_tail = [&](VkCommandBuffer cb, VkPipelineLayout layout, const std::uint32_t* block) {
        if (!tail_block) return;
        tail.write(0, block, 4);
        const std::uint32_t offset = tail.offset(0);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, leonida::vk_test::pc_tail_set, 1, &tail.set,
                                1, &offset);
    };
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = qi;
    VkCommandPool pool;
    CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    const auto submit = [&]() {
        VkResult r = vkEndCommandBuffer(cmd);
        if (r) return r;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        return r ? r : vkQueueWaitIdle(queue);
    };

    // Upload the textures. The sampled ones end in SHADER_READ_ONLY_OPTIMAL, the storage one in GENERAL.
    CHECK(vkBeginCommandBuffer(cmd, &begin));
    const struct { VkImage image; std::uint32_t size; VkDeviceSize offset; VkImageLayout layout; } uploads[] = {
        {gradient, TW, 0, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {decoy, 1, 1024, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {stored, 1, 0, VK_IMAGE_LAYOUT_GENERAL}};
    for (const auto& u : uploads) {
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        ib.image = u.image;
        ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
        VkBufferImageCopy c{};
        c.bufferOffset = u.offset;
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {u.size, u.size, 1};
        vkCmdCopyBufferToImage(cmd, upload, u.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        ib.newLayout = u.layout;
        ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    }
    CHECK(submit());

    // An adapter at the original 5632 sampled-image limit must keep the original set. The per-stage
    // resource budget can also leave room for just one class (9985 original resources + 512).
    constexpr auto ms_classes = [](std::uint32_t sampled, std::uint32_t total) {
        BindlessFeatures f{};
        f.ms_samples.fill(VK_SAMPLE_COUNT_4_BIT);
        bindless::limit_ms_bindings(f, sampled, total);
        return f.ms_samples;
    };
    static_assert(ms_classes(5632, 20000) ==
                  std::array{VK_SAMPLE_COUNT_1_BIT, VK_SAMPLE_COUNT_1_BIT, VK_SAMPLE_COUNT_1_BIT});
    static_assert(ms_classes(7168, 10497) ==
                  std::array{VK_SAMPLE_COUNT_4_BIT, VK_SAMPLE_COUNT_1_BIT, VK_SAMPLE_COUNT_1_BIT});

    // Does a set with these flags fit the device? Every binding is visible to every stage.
    const auto descriptors = [](VkDescriptorType type, const BindlessFeatures& f) {
        std::uint32_t count = 0;
        for (std::uint32_t b = 0; b < bindless::bindings.size(); ++b)
            if (bindless::bindings[b].type == type && bindless::binding_enabled(b, f))
                count += f.counts[b];
        return count;
    };
    const auto fits = [&](const BindlessFeatures& f) {
        const auto sampled_total = descriptors(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, f);
        const auto sampler_total = descriptors(VK_DESCRIPTOR_TYPE_SAMPLER, f);
        const auto storage_total = descriptors(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, f);
        const auto stage_resources = sampled_total + storage_total + 2;  // table buffer and fragment output
        const VkPhysicalDeviceLimits& l = props.properties.limits;
        if (f.update_after_bind)
            return props12.maxPerStageDescriptorUpdateAfterBindSampledImages >= sampled_total &&
                   props12.maxPerStageDescriptorUpdateAfterBindSamplers >= sampler_total &&
                   props12.maxPerStageDescriptorUpdateAfterBindStorageImages >= storage_total &&
                   props12.maxPerStageUpdateAfterBindResources >= stage_resources &&
                   props12.maxUpdateAfterBindDescriptorsInAllPools >= sampled_total + sampler_total + storage_total + 1 &&
                   props12.maxDescriptorSetUpdateAfterBindSampledImages >= sampled_total &&
                   props12.maxDescriptorSetUpdateAfterBindSamplers >= sampler_total &&
                   props12.maxDescriptorSetUpdateAfterBindStorageImages >= storage_total;
        return l.maxPerStageDescriptorSampledImages >= sampled_total && l.maxPerStageDescriptorSamplers >= sampler_total &&
               l.maxPerStageDescriptorStorageImages >= storage_total &&
               l.maxPerStageResources >= stage_resources &&
               l.maxDescriptorSetSampledImages >= sampled_total && l.maxDescriptorSetSamplers >= sampler_total &&
               l.maxDescriptorSetStorageImages >= storage_total;
    };

    unsigned mismatches = 0, runs = 0, readbacks = 0;
    float worst = 0;
    // What the 64x64 target must hold: the texel under each pixel, or one colour everywhere.
    const auto compare = [&](const char* what, const float* colour) {
        ++readbacks;
        const float* got = static_cast<const float*>(readback_map);
        for (std::uint32_t y = 0; y < H; ++y) {
            for (std::uint32_t x = 0; x < W; ++x) {
                const float gradient_texel[4] = {float(x * TW / W * 32) / 255.0f, float(y * TW / H * 32) / 255.0f, 128.0f / 255.0f, 1.0f};
                const float* want = colour ? colour : gradient_texel;
                const float* pixel = got + (std::size_t(y) * W + x) * 4;
                const float error = pixel_error(pixel, want);
                worst = std::max(worst, error);
                if (error <= 1e-6f) continue;
                if (mismatches++ < 6)
                    std::fprintf(stderr, "%s pixel (%u, %u): got %g %g %g %g, want %g %g %g %g\n", what, x, y, pixel[0],
                                 pixel[1], pixel[2], pixel[3], want[0], want[1], want[2], want[3]);
            }
        }
    };

    BindlessFeatures configurations[2] = {device_flags, {}};
    const int configuration_count = (device_flags.update_after_bind || device_flags.partially_bound) ? 2 : 1;
    for (int configuration = 0; configuration < configuration_count; ++configuration) {
        BindlessFeatures flags = configurations[configuration];
        flags.ms_samples[0] = VK_SAMPLE_COUNT_4_BIT;
        // A device below the default counts gets a smaller set; this test needs at least those.
        const bool full = bindless::size_bindings(flags, props.properties.limits, props11, props12);
        const char* name = flags.update_after_bind && flags.partially_bound ? "update after bind, partially bound"
                           : flags.update_after_bind ? "update after bind" : flags.partially_bound ? "partially bound" : "no flags";
        if (!full || !fits(flags)) {
            std::printf("%s: skipped, the set does not fit this device's descriptor limits\n", name);
            continue;
        }
        BindlessSet set(vkGetDeviceProcAddr, dev, mp, flags);
        EXPECT(set.ok() && set.layout() && set.set());

        const VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, push_bytes};
        const VkDescriptorSetLayout set_layout = set.layout();
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &set_layout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &push;
        const VkDescriptorSetLayout set_layouts[] = {set_layout, tail.set_layout};
        if (tail_block) { pli.setLayoutCount = 2; pli.pSetLayouts = set_layouts; }
        VkPipelineLayout pipeline_layout;
        CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pipeline_layout));
        VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName = "main";
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName = "main";
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

        // One draw and read back. The first also initialises the set's fallback images.
        const auto draw = [&]() {
            vkResetCommandBuffer(cmd, 0);
            VkResult r = vkBeginCommandBuffer(cmd, &begin);
            if (r) return r;
            if (!set.fallbacks_ready()) set.record_fallback_init(cmd);
            VkClearValue clear{};
            for (float& channel : clear.color.float32) channel = 0.25f;  // neither a texel nor the fallback
            VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rbi.renderPass = pass;
            rbi.framebuffer = framebuffer;
            rbi.renderArea = scissor;
            rbi.clearValueCount = 1;
            rbi.pClearValues = &clear;
            vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            const VkDescriptorSet bound = set.set();
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &bound, 0, nullptr);
            std::uint32_t block[36] = {};
            block[2] = table_va;  // s2:s3 = console VA of the descriptor table
            block[32] = std::uint32_t(heap_base);
            block[33] = std::uint32_t(heap_base >> 32);
            vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, push_bytes, block);
            bind_tail(cmd, pipeline_layout, block);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRenderPass(cmd);
            VkBufferImageCopy rc{};
            rc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            rc.imageExtent = {W, H, 1};
            vkCmdCopyImageToBuffer(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &rc);
            return submit();
        };

        // Uploads the table where the device does: the copy and its barrier are recorded in a command buffer that
        // runs before the draws that read binding 2.
        const auto flush = [&]() {
            vkResetCommandBuffer(cmd, 0);
            if (const VkResult r = vkBeginCommandBuffer(cmd, &begin); r != VK_SUCCESS) return false;
            // Ticket 1: what a flush that replaces the table stamps its retired generation with; release_retired(0)
            // below must keep it, as the owner's completed ticket 0 means nothing has finished.
            const bool ok = set.flush(cmd, 1);
            return submit() == VK_SUCCESS && ok;
        };

        // Index spaces. An image with storage views gets an index the storage arrays hold; one without goes above
        // them where the dimension has room. A dimension that is full gives 0, and removed indices come back.
        StorageViews as_storage;
        as_storage.as_float = stored_view;
        const std::uint32_t stored_index = set.add_image(ImageDimension::dim_2d, stored_view, VK_IMAGE_LAYOUT_GENERAL, as_storage);
        const std::uint32_t decoy_index = set.add_image(ImageDimension::dim_2d, decoy_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const std::uint32_t gradient_index = set.add_image(ImageDimension::dim_2d, gradient_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        EXPECT(stored_index >= 1 && stored_index < set.storage_capacity(ImageDimension::dim_2d));
        EXPECT(decoy_index >= set.storage_capacity(ImageDimension::dim_2d) && gradient_index > decoy_index);
        EXPECT(gradient_index < set.image_capacity(ImageDimension::dim_2d));
        std::vector<std::uint32_t> filled;
        for (std::uint32_t i = 1; i < set.image_capacity(ImageDimension::dim_1d); ++i)
            filled.push_back(set.add_image(ImageDimension::dim_1d, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL));
        EXPECT(filled.front() == 1 && filled.back() == set.image_capacity(ImageDimension::dim_1d) - 1);
        EXPECT(set.free_images(ImageDimension::dim_1d) == 0);
        EXPECT(set.add_image(ImageDimension::dim_1d, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL) == 0);
        set.remove_image(ImageDimension::dim_1d, 7);
        EXPECT(set.add_image(ImageDimension::dim_1d, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL) == 7);
        for (const std::uint32_t index : filled) set.remove_image(ImageDimension::dim_1d, index);
        EXPECT(set.free_images(ImageDimension::dim_1d) == set.image_capacity(ImageDimension::dim_1d) - 1);
        EXPECT(set.free_samplers() == set.sampler_capacity() - 1);
        const std::uint32_t decoy_sampler = set.add_sampler(nearest);
        const std::uint32_t sampler_index = set.add_sampler(nearest);
        EXPECT(decoy_sampler == 1 && sampler_index == 2 && set.free_samplers() == set.sampler_capacity() - 3);

        if (atomic_mode) {
            // ps_imgatomic.asm: image_atomic_umax/umin/or on three 8x8 r32ui images and umax on a 4x4x4 one, from
            // every pixel of the 64x64 draw (64 pixels per 2D texel), each found through its own T# key. Odd
            // pixels are discarded after their atomics, which still count.
            constexpr std::uint32_t atomic_va = 1024, AW = 8, A3 = 4;
            const std::array<DescriptorKey, 4> keys{{{0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8},
                                                     {0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8},
                                                     {0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8},
                                                     {0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8}}};
            for (std::uint32_t k = 0; k < 4; ++k)
                std::memcpy(static_cast<char*>(heap_map) + atomic_va + 32 * k, keys[k].data(), 32);
            const VkImageUsageFlags atomic_usage =
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            VkImage images[4];
            VkImageView views[4];
            for (int k = 0; k < 3; ++k) CHECK(make_image(VK_FORMAT_R32_UINT, AW, AW, atomic_usage, &images[k], &views[k]));
            {   // the 3D image
                VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                ii.imageType = VK_IMAGE_TYPE_3D;
                ii.format = VK_FORMAT_R32_UINT;
                ii.extent = {A3, A3, A3};
                ii.mipLevels = 1;
                ii.arrayLayers = 1;
                ii.samples = VK_SAMPLE_COUNT_1_BIT;
                ii.usage = atomic_usage;
                CHECK(vkCreateImage(dev, &ii, nullptr, &images[3]));
                VkMemoryRequirements mr;
                vkGetImageMemoryRequirements(dev, images[3], &mr);
                VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                ai.allocationSize = mr.size;
                ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                VkDeviceMemory m;
                CHECK(vkAllocateMemory(dev, &ai, nullptr, &m));
                allocations.push_back(m);
                CHECK(vkBindImageMemory(dev, images[3], m, 0));
                VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                vi.image = images[3];
                vi.viewType = VK_IMAGE_VIEW_TYPE_3D;
                vi.format = VK_FORMAT_R32_UINT;
                vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                CHECK(vkCreateImageView(dev, &vi, nullptr, &views[3]));
            }
            std::uint32_t indices[4];
            for (int k = 0; k < 4; ++k) {
                StorageViews storage;
                storage.as_uint = views[k];
                storage.r32ui = views[k];
                indices[k] = set.add_image(k == 3 ? ImageDimension::dim_3d : ImageDimension::dim_2d, views[k],
                                           VK_IMAGE_LAYOUT_GENERAL, storage);
                EXPECT(indices[k] != 0);
                EXPECT(set.insert_key(keys[k], indices[k]));
            }
            EXPECT(flush());

            // Clear: umax and or start at 0, umin at all ones; then GENERAL for the shader's atomics.
            CHECK(vkResetCommandBuffer(cmd, 0));
            CHECK(vkBeginCommandBuffer(cmd, &begin));
            if (!set.fallbacks_ready()) set.record_fallback_init(cmd);
            for (int k = 0; k < 4; ++k) {
                VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = images[k];
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                     nullptr, 0, nullptr, 1, &b);
                VkClearColorValue value{};
                for (auto& c : value.uint32) c = k == 1 ? 0xFFFFFFFFu : 0u;
                vkCmdClearColorImage(cmd, images[k], VK_IMAGE_LAYOUT_GENERAL, &value, 1, &b.subresourceRange);
                b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                b.oldLayout = b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                                     nullptr, 0, nullptr, 1, &b);
            }
            CHECK(submit());

            // The draw, then the images and the render target read back.
            const VkDeviceSize image_bytes = 3 * AW * AW * 4 + A3 * A3 * A3 * 4;
            VkBuffer atomic_readback;
            void* atomic_map;
            CHECK(make_buffer(image_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &atomic_readback, &atomic_map));
            CHECK(vkResetCommandBuffer(cmd, 0));
            CHECK(vkBeginCommandBuffer(cmd, &begin));
            VkClearValue clear{};
            for (float& channel : clear.color.float32) channel = 0.25f;
            VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rbi.renderPass = pass;
            rbi.framebuffer = framebuffer;
            rbi.renderArea = scissor;
            rbi.clearValueCount = 1;
            rbi.pClearValues = &clear;
            vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            const VkDescriptorSet bound = set.set();
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &bound, 0, nullptr);
            std::uint32_t block[36] = {};
            block[2] = atomic_va;
            block[32] = std::uint32_t(heap_base);
            block[33] = std::uint32_t(heap_base >> 32);
            vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               push_bytes, block);
            bind_tail(cmd, pipeline_layout, block);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRenderPass(cmd);
            VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            after.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &after,
                                 0, nullptr, 0, nullptr);
            VkDeviceSize offset = 0;
            for (int k = 0; k < 4; ++k) {
                VkBufferImageCopy rc{};
                rc.bufferOffset = offset;
                rc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                rc.imageExtent = k == 3 ? VkExtent3D{A3, A3, A3} : VkExtent3D{AW, AW, 1};
                vkCmdCopyImageToBuffer(cmd, images[k], VK_IMAGE_LAYOUT_GENERAL, atomic_readback, 1, &rc);
                offset += k == 3 ? A3 * A3 * A3 * 4 : AW * AW * 4;
            }
            VkBufferImageCopy tc{};
            tc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            tc.imageExtent = {W, H, 1};
            vkCmdCopyImageToBuffer(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &tc);
            CHECK(submit());

            // The model: every pixel's atomics, discarded or not.
            std::vector<std::uint32_t> want(3 * AW * AW + A3 * A3 * A3, 0);
            for (std::uint32_t t = 0; t < AW * AW; ++t) want[AW * AW + t] = 0xFFFFFFFFu;
            for (std::uint32_t py = 0; py < H; ++py)
                for (std::uint32_t px = 0; px < W; ++px) {
                    const std::uint32_t id = px + 64 * py, t = (py >> 3) * AW + (px >> 3);
                    want[t] = std::max(want[t], id);
                    want[AW * AW + t] = std::min(want[AW * AW + t], id);
                    want[2 * AW * AW + t] |= 1u << (id & 31);
                    const std::uint32_t v = ((px & 3) * A3 + (py >> 4)) * A3 + (px >> 4);
                    want[3 * AW * AW + v] = std::max(want[3 * AW * AW + v], id);
                }
            const auto* got = static_cast<const std::uint32_t*>(atomic_map);
            const char* names[] = {"2D umax", "2D umin", "2D or", "3D umax"};
            unsigned wrong = 0;
            for (std::size_t i = 0; i < want.size(); ++i) {
                if (got[i] == want[i]) continue;
                const int k = i < 3 * AW * AW ? int(i / (AW * AW)) : 3;
                if (wrong++ < 8)
                    std::printf("  %s texel %zu: got 0x%08x, want 0x%08x\n", names[k],
                                k == 3 ? i - 3 * AW * AW : i % (AW * AW), got[i], want[i]);
            }
            const float* rt = static_cast<const float*>(readback_map);
            unsigned bad_rt = 0;
            for (std::uint32_t py = 0; py < H; ++py)
                for (std::uint32_t px = 0; px < W; ++px)
                    bad_rt += rt[(py * W + px) * 4] != ((px & 1) ? 0.25f : 1.0f);
            std::printf("ps_imgatomic: %u of %zu texels wrong (2D umax/umin/or 8x8 each, 3D umax 4x4x4); "
                        "%u of %u render-target pixels wrong (odd columns discarded)\n",
                        wrong, want.size(), bad_rt, W * H);
            vkDeviceWaitIdle(dev);
            if (g_validation_errors) {
                std::printf("ps_imgatomic: %u validation errors\n", g_validation_errors);
                return 1;
            }
            return wrong || bad_rt ? 1 : 0;
        }

        // A view that comes and goes before the set is flushed is never written: a texture created and destroyed
        // between two submissions. Its slots are written once, with the fallback; a write of the destroyed view
        // would be a validation error.
        {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = stored;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = VK_FORMAT_R8G8B8A8_UNORM;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageView brief = VK_NULL_HANDLE;
            CHECK(vkCreateImageView(dev, &vi, nullptr, &brief));
            StorageViews brief_storage;
            brief_storage.as_float = brief;
            const std::uint32_t brief_index = set.add_image(ImageDimension::dim_2d, brief, VK_IMAGE_LAYOUT_GENERAL, brief_storage);
            EXPECT(brief_index >= 1 && brief_index < set.storage_capacity(ImageDimension::dim_2d) &&
                   brief_index != stored_index);
            set.remove_image(ImageDimension::dim_2d, brief_index);
            vkDestroyImageView(dev, brief, nullptr);
        }

        const DescriptorTable& table = set.table();
        const auto slot_count = [&] { return std::uint32_t(table.capacity()); };
        const std::uint32_t slots = slot_count(), home = hash_descriptor_key(tsharp) & (slots - 1);
        // Fill eight adjacent home slots across the end of the table. The ninth key
        // shares the first home and can fit only by moving a resident already on the GPU.
        {
            std::array<DescriptorKey, kDescriptorProbes + 1> keys{};
            for (std::uint32_t i = 0; i < keys.size(); ++i) {
                auto& key = keys[i];
                key[0] = i + 1;
                const auto target = (slots - 4 + i % kDescriptorProbes) & (slots - 1);
                while ((hash_descriptor_key(key) & (slots - 1)) != target) ++key[7];
            }
            for (std::uint32_t i = 0; i < kDescriptorProbes; ++i) set.insert_key(keys[i], decoy_index);
            EXPECT(!set.flush_replaces_table() && flush());
            EXPECT(set.insert_key(keys.back(), gradient_index));
            EXPECT(slot_count() == slots && set.flush_replaces_table());
            EXPECT(flush() && !set.flush_replaces_table());
            for (const auto& key : keys) {
                EXPECT(table.lookup(key) == (key == keys.back() ? gradient_index : decoy_index));
                EXPECT(set.erase_key(key));
            }
            EXPECT(table.size() == 0 && flush());
        }

        // 1. The T# behind three keys that share its first slot, the S#, and other keys that name the decoy.
        std::uint32_t ahead = 0;
        for (std::uint32_t k = 1; ahead < 3; ++k) {
            if ((hash_descriptor_key(numbered(k)) & (slots - 1)) != home) continue;
            set.insert_key(numbered(k), decoy_index);
            ++ahead;
        }
        set.insert_key(tsharp, gradient_index);
        set.insert_key(make_ssharp_key(ssharp.data()), sampler_index);
        for (std::uint32_t k = 100000; k < 100006; ++k) set.insert_key(numbered(k), decoy_index);
        const std::uint32_t depth = probe_depth(table.build_gpu_buffer(), tsharp);
        EXPECT(slot_count() == slots && depth == 3 && !set.flush_replaces_table());
        EXPECT(flush());
        CHECK(draw());
        compare("1 gradient", nullptr);

        // 2. Without its key the T# is a miss: index 0, the fallback image of zeros.
        const float zeros[4] = {0, 0, 0, 0};
        EXPECT(set.erase_key(tsharp) && !set.erase_key(tsharp));
        EXPECT(!set.flush_replaces_table());  // erasing empties its slot only: the three keys ahead stay where they are
        EXPECT(flush() && !set.flush_replaces_table());
        CHECK(draw());
        compare("2 miss", zeros);

        // 3. The table grows: a new, larger buffer behind binding 2.
        set.insert_key(tsharp, gradient_index);
        EXPECT(!set.flush_replaces_table());  // one more entry, in place
        for (std::uint32_t k = 200000; k < 200200; ++k) set.insert_key(numbered(k), decoy_index);
        EXPECT(slot_count() > slots && table.lookup(tsharp) == gradient_index && set.flush_replaces_table());
        const VkDescriptorSet before_growth = set.set();
        const std::uint64_t made_before = set.generations_made();
        EXPECT(flush());
        // A new generation: another descriptor set (every descriptor written so far copied into it) and table, the
        // old one retired but intact until released (D3D-314).
        EXPECT(set.set() != before_growth && set.generations_made() == made_before + 1 && set.retired_count() >= 1);
        CHECK(draw());
        compare("3 grown", nullptr);
        set.release_retired(0);  // a ticket past what completed (nothing): keeps the generation
        EXPECT(set.retired_count() >= 1);
        set.release_retired();
        EXPECT(set.retired_count() == 0);

        const float orange_colour[4] = {1.0f, 128.0f / 255.0f, 0.0f, 1.0f};
        // Rewriting the same mapping is a no-op. Changing it makes a new generation even with no table growth.
        const std::uint32_t grown_slots = slot_count();
        EXPECT(set.insert_key(tsharp, gradient_index) && !set.flush_replaces_table());
        EXPECT(set.insert_key(tsharp, decoy_index) && slot_count() == grown_slots && set.flush_replaces_table());
        EXPECT(flush() && !set.flush_replaces_table());
        CHECK(draw());
        compare("4 rebound", orange_colour);
        EXPECT(set.insert_key(tsharp, gradient_index) && set.flush_replaces_table());
        EXPECT(flush());
        CHECK(draw());
        compare("5 restored", nullptr);

        // A submit that never reached the queue must keep the table upload pending: the retry has to see the new
        // mapping instead of the fallback or the old index (a recovered submit failure, the case a device reports
        // through table_upload_result(false)).
        EXPECT(set.insert_key(tsharp, decoy_index) && set.flush_replaces_table());
        vkResetCommandBuffer(cmd, 0);
        CHECK(vkBeginCommandBuffer(cmd, &begin));
        EXPECT(set.flush(cmd, 1));  // recorded in a command buffer that is never submitted
        CHECK(vkEndCommandBuffer(cmd));
        set.table_upload_result(false);
        EXPECT(set.flush_replaces_table());  // the retry rewrites the whole table, in a new generation
        EXPECT(flush());
        CHECK(draw());
        compare("6 rebound after a failed submit", orange_colour);
        EXPECT(set.insert_key(tsharp, gradient_index) && set.flush_replaces_table());
        EXPECT(flush());

        // Another view at the same index, then removal of the index.
        set.update_image(ImageDimension::dim_2d, gradient_index, decoy_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        EXPECT(flush());
        CHECK(draw());
        compare("7 other view", orange_colour);
        set.remove_image(ImageDimension::dim_2d, gradient_index);
        EXPECT(flush());
        CHECK(draw());
        compare("8 removed", zeros);
        EXPECT(set.add_image(ImageDimension::dim_2d, gradient_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) == gradient_index);

        // 9. More than 4096 2D images and more than a full D3D12 sampler heap (2048): the gradient and the S#
        // through the last indices of both arrays, where the device's limits gave the set room for them.
        std::uint32_t far_image = 0, far_sampler = 0;
        if (set.image_capacity(ImageDimension::dim_2d) > 5000 + 8 && set.sampler_capacity() > 2100 + 8) {
            for (std::uint32_t i = 0; i < 5000; ++i)
                EXPECT(set.add_image(ImageDimension::dim_2d, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL) != 0);
            far_image = set.add_image(ImageDimension::dim_2d, gradient_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            for (std::uint32_t i = 0; i < 2100; ++i) far_sampler = set.add_sampler(nearest);
            EXPECT(far_image > 5000 && far_sampler > 2100);
            set.insert_key(tsharp, far_image);
            set.insert_key(make_ssharp_key(ssharp.data()), far_sampler);
            EXPECT(flush());
            CHECK(draw());
            compare("9 past 4096 images and 2048 samplers", nullptr);

            // Keep another high slot live while the sampled slot is removed and then reused.
            const auto neighbour = set.add_image(ImageDimension::dim_2d, decoy_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            EXPECT(neighbour > 5000 && neighbour != far_image);
            set.remove_image(ImageDimension::dim_2d, far_image);
            EXPECT(flush());
            CHECK(draw());
            compare("10 freed high slot", zeros);

            const auto replacement = set.add_image(ImageDimension::dim_2d, stored_view, VK_IMAGE_LAYOUT_GENERAL);
            EXPECT(replacement == far_image);
            EXPECT(flush());
            CHECK(draw());
            const float blue[4] = {0, 0, 128.0f / 255.0f, 1};  // stored image uses the first gradient texel
            compare("11 reused high slot", blue);

            EXPECT(set.insert_key(tsharp, neighbour));
            EXPECT(flush());
            CHECK(draw());
            compare("12 live neighbour", orange_colour);
        } else {
            std::printf("%s: no room past 5000 2D images and 2100 samplers on this device; step 9 skipped\n", name);
        }

        std::printf("%s: descriptor table in %s memory\n", name,
                    set.table_in_device_local_memory() ? "device-local" : "host-visible");
        std::printf("%s: gradient at the %uth slot probed, miss, table grown %u -> %u slots, rebound, restored, "
                    "other view, removed, index %u of %u 2D images and %u of %u samplers\n", name, depth + 1, slots,
                    slot_count(), far_image, set.image_capacity(ImageDimension::dim_2d), far_sampler,
                    set.sampler_capacity());
        ++runs;
        vkDeviceWaitIdle(dev);
        vkDestroyPipeline(dev, pipeline, nullptr);
        vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
    }

    // Let validation observe resource and device destruction before reporting its final error count.
    vkDestroyCommandPool(dev, pool, nullptr);
    vkDestroyFramebuffer(dev, framebuffer, nullptr);
    vkDestroyRenderPass(dev, pass, nullptr);
    vkDestroyShaderModule(dev, fs, nullptr);
    vkDestroyShaderModule(dev, vs, nullptr);
    vkDestroySampler(dev, nearest, nullptr);
    for (const auto view : {target_view, stored_view, decoy_view, gradient_view})
        vkDestroyImageView(dev, view, nullptr);
    for (const auto image : {target, stored, decoy, gradient}) vkDestroyImage(dev, image, nullptr);
    for (const auto buffer : {readback, upload, heap}) vkDestroyBuffer(dev, buffer, nullptr);
    for (const auto memory : allocations) vkFreeMemory(dev, memory, nullptr);
    tail.destroy(vkGetDeviceProcAddr, dev);  // the --pc-tail-ubo block, when the shaders use one
    vkDestroyDevice(dev, nullptr);
    if (messenger) {
        LI(vkDestroyDebugUtilsMessengerEXT);
        vkDestroyDebugUtilsMessengerEXT(inst, messenger, nullptr);
    }
    LI(vkDestroyInstance);
    vkDestroyInstance(inst, nullptr);

    if (!runs) {
        std::printf("unsupported: the bindless set does not fit this device's descriptor limits\n");
        return unsupported;
    }
    std::printf("bindless: %u draws read back, %u bad pixels, largest error %g\n", readbacks, mismatches, double(worst));
    if (g_validation_errors) {
        std::fprintf(stderr, "%u validation errors\n", g_validation_errors);
        return 1;
    }
    return mismatches ? 1 : 0;
}
