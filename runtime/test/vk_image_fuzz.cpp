// Headless fuzz harness for the translator's `image_load` path:
//   vk_image_fuzz <vs.spv> <fs.spv> <out.raw> <spec.txt>
// The generated fragment shader does `image_load ..., s[8:15]`, so push constants [8..15] carry the T#.
// A full-screen triangle is drawn into 64x64 RGBA32F and read back into <out.raw>.
// Exit 0 done, 1 error (`spec: ...` for spec errors), 77 when the device cannot do the work.
// LEONIDA_VK_VALIDATION=1 runs under VK_LAYER_KHRONOS_validation; a validation error fails the run.
#define VK_NO_PROTOTYPES
#include "vulkan_loader.h"
#include "../src/vulkan_device_filter.h"
#include "../src/vulkan_bindless.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#define CHECK(x)                                                                   \
    do {                                                                           \
        VkResult r_ = (x);                                                         \
        if (r_ != VK_SUCCESS) {                                                    \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, r_, __LINE__); \
            return 1;                                                              \
        }                                                                          \
    } while (0)

namespace {
using namespace leonida;
constexpr int unsupported = 77;
unsigned g_validation_errors = 0;
VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                          VkDebugUtilsMessageTypeFlagsEXT types,
                                          const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    // Only validation-type errors count: a broken third-party layer must not fail the run.
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) && (types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT))
        ++g_validation_errors;
    std::fprintf(stderr, "validation: %s\n", data && data->pMessage ? data->pMessage : "(no message)");
    return VK_FALSE;
}
std::vector<std::uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
    std::vector<std::uint32_t> words(bytes.size() / 4);
    return std::memcpy(words.data(), bytes.data(), words.size() * 4), words;
}
std::string trim(const std::string& s) {
    std::size_t b = s.find_first_not_of(" \t\r\n"), e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}
// LEONIDA_PC_TAIL_UBO=1: shaders were translated with --pc-tail-ubo (128-byte push constants for s0..s31; the
// image_load fuzz shaders read neither heap base nor code base, so they declare no set-1 block).
bool pc_tail_requested() {
    const char* v = std::getenv("LEONIDA_PC_TAIL_UBO");
    std::string s = trim(v ? v : "");
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s == "1" || s == "true" || s == "yes" || s == "on";
}
// Decimal or 0x-prefixed hex, whole string, no exceptions (strtoul with base 0 handles both).
bool parse_u32(const std::string& text, std::uint32_t& out) {
    if (text.empty() || text.find_first_of(" \t\r\n") != std::string::npos) return false;
    char* end = nullptr;
    const unsigned long v = std::strtoul(text.c_str(), &end, 0);
    if (end == text.c_str() || *end || v > 0xfffffffful) return false;
    return out = std::uint32_t(v), true;
}
struct Spec {
    std::uint32_t width = 0, height = 0, bytes_per_texel = 0, tsharp[8] = {};
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::string format_name, texels_path;
    bool has_width = false, has_height = false, has_format = false, has_texels = false, has_tsharp = false;
};
bool parse_spec(const char* path, Spec& s, std::string& error) {
    std::ifstream f(path);
    if (!f) return error = "cannot open spec file", false;
    std::string line;
    for (unsigned lineno = 0; std::getline(f, line);) {
        ++lineno;
        std::string text = trim(line);
        const std::size_t hash = text.find('#'); // '#' starts a comment, even trailing one
        if (hash != std::string::npos) text = trim(text.substr(0, hash));
        if (text.empty()) continue;
        const std::size_t eq = text.find('=');
        if (eq == std::string::npos) return error = "line " + std::to_string(lineno) + ": missing '='", false;
        const std::string key = trim(text.substr(0, eq)), value = trim(text.substr(eq + 1));
        const std::string where = "line " + std::to_string(lineno) + ": ";
        if (key == "width" || key == "height") {
            std::uint32_t v = 0;
            if (!parse_u32(value, v) || v < 1 || v > 4096) return error = where + "width/height must be 1..4096", false;
            if (key == "width") s.width = v, s.has_width = true;
            else s.height = v, s.has_height = true;
        } else if (key == "format") {
            s.format_name = value;
            if (value == "R8G8B8A8_UNORM") s.format = VK_FORMAT_R8G8B8A8_UNORM, s.bytes_per_texel = 4;
            else if (value == "R16G16B16A16_SFLOAT") s.format = VK_FORMAT_R16G16B16A16_SFLOAT, s.bytes_per_texel = 8;
            else if (value == "R32G32B32A32_SFLOAT") s.format = VK_FORMAT_R32G32B32A32_SFLOAT, s.bytes_per_texel = 16;
            else if (value == "R32_SFLOAT") s.format = VK_FORMAT_R32_SFLOAT, s.bytes_per_texel = 4;
            else if (value == "R8G8B8A8_UINT") s.format = VK_FORMAT_R8G8B8A8_UINT, s.bytes_per_texel = 4;
            else if (value == "R8G8B8A8_SINT") s.format = VK_FORMAT_R8G8B8A8_SINT, s.bytes_per_texel = 4;
            else return error = where + "unknown format '" + value + "'", false;
            s.has_format = true;
        } else if (key == "texels") {
            s.texels_path = value, s.has_texels = true;
        } else if (key == "tsharp") {
            std::size_t start = 0;
            for (int i = 0; i < 8; ++i) {
                const std::size_t comma = i == 7 ? std::string::npos : value.find(',', start);
                if ((i < 7) == (comma == std::string::npos)) return error = where + "tsharp needs 8 uint32", false;
                if (!parse_u32(trim(value.substr(start, comma == std::string::npos ? comma : comma - start)), s.tsharp[i]))
                    return error = where + "tsharp needs 8 uint32 (decimal or 0x hex)", false;
                if (comma != std::string::npos) start = comma + 1;
            }
            s.has_tsharp = true;
        } else return error = where + "unknown key '" + key + "'", false;
    }
    if (!s.has_width || !s.has_height) error = "missing width/height";
    else if (!s.has_format) error = "missing format";
    else if (!s.has_texels) error = "missing texels";
    else if (!s.has_tsharp) error = "missing tsharp";
    else return true;
    return false;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: vk_image_fuzz <vs.spv> <fs.spv> <out.raw> <spec.txt>\n");
        return 2;
    }
    Spec spec;
    {
        std::string error;
        if (!parse_spec(argv[4], spec, error)) {
            std::fprintf(stderr, "spec: %s\n", error.c_str());
            return 1;
        }
    }
    std::vector<char> texels;
    {
        std::ifstream f(spec.texels_path.c_str(), std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "spec: cannot open texels '%s'\n", spec.texels_path.c_str());
            return 1;
        }
        texels.assign(std::istreambuf_iterator<char>(f), {});
        const std::uint64_t want = std::uint64_t(spec.width) * spec.height * spec.bytes_per_texel;
        if (std::uint64_t(texels.size()) != want) {
            std::fprintf(stderr, "spec: texels '%s' is %llu bytes, want %llu\n", spec.texels_path.c_str(),
                         static_cast<unsigned long long>(texels.size()), static_cast<unsigned long long>(want));
            return 1;
        }
    }
    constexpr std::uint32_t W = 64, H = 64;
    const PFN_vkGetInstanceProcAddr gipa = load_vulkan();
    if (!gipa) return 1;
    VkInstance inst = VK_NULL_HANDLE;
#define LI(n) const auto n = reinterpret_cast<PFN_##n>(gipa(inst, #n))
    LI(vkCreateInstance); LI(vkEnumerateInstanceLayerProperties);
    const char* env = std::getenv("LEONIDA_VK_VALIDATION");
    const bool validation = env && std::strcmp(env, "0") != 0;
    const char* layer = "VK_LAYER_KHRONOS_validation", *debug_utils = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    VkDebugUtilsMessengerCreateInfoEXT messenger_info{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                                                     .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                                                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                                                     .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                                                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
                                                     .pfnUserCallback = on_message};
    VkApplicationInfo app{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    if (validation) {
        std::uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> layers(count);
        vkEnumerateInstanceLayerProperties(&count, layers.data());
        bool found = false;
        for (const auto& p : layers) found = found || std::strcmp(p.layerName, layer) == 0;
        if (!found) {
            std::fprintf(stderr, "LEONIDA_VK_VALIDATION is set but %s is not installed\n", layer);
            return 1;
        }
        ici.pNext = &messenger_info, ici.enabledLayerCount = ici.enabledExtensionCount = 1;
        ici.ppEnabledLayerNames = &layer, ici.ppEnabledExtensionNames = &debug_utils;
    }
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (validation) {
        LI(vkCreateDebugUtilsMessengerEXT);
        CHECK(vkCreateDebugUtilsMessengerEXT(inst, &messenger_info, nullptr, &messenger));
    }
    LI(vkEnumeratePhysicalDevices); LI(vkGetPhysicalDeviceProperties); LI(vkGetPhysicalDeviceQueueFamilyProperties);
    LI(vkGetPhysicalDeviceMemoryProperties); LI(vkGetPhysicalDeviceFeatures2); LI(vkGetPhysicalDeviceProperties2);
    LI(vkCreateDevice); LI(vkGetDeviceProcAddr); LI(vkGetPhysicalDeviceFormatProperties);
    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = VK_NULL_HANDLE;
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
        }
    }
    if (!pd && !leonida::vulkan_device_filter().empty()) {  // a filter matching nothing fails, never skips
        std::fprintf(stderr, "LEONIDA_VK_DEVICE=%s: no Vulkan 1.2 device matches\n",
                     leonida::vulkan_device_filter().c_str());
        return 1;
    }
    if (!pd) {
        std::printf("unsupported: no Vulkan 1.2 device\n");
        return unsupported;
    }
    VkPhysicalDeviceVulkan12Properties props12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &props12};
    vkGetPhysicalDeviceProperties2(pd, &props);
    std::printf("device: %s%s\n", props.properties.deviceName, validation ? " (validation on)" : "");
    // Every format, UINT/SINT included, is a sampled image: a translated image_load fetches it from the sampled
    // array, through a uint/sint alias for integer formats (SHADER-151), as the runtime binds a read-only texture.
    VkFormatProperties format_props{};
    vkGetPhysicalDeviceFormatProperties(pd, spec.format, &format_props);
    if (!(format_props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
        std::printf("unsupported: %s lacks the required image feature\n", spec.format_name.c_str());
        return unsupported;
    }
    // The pipeline layout pushes the translated user-data block (144 bytes by default, or 128 bytes under
    // LEONIDA_PC_TAIL_UBO=1 for rdna2spirv.py --pc-tail-ubo). A device with fewer push constant bytes (128 on
    // Intel Arc and AMD's Windows driver for RDNA2 iGPUs) can't take 144: a device limit, not a wrong result (TEST-149).
    const bool pc_tail = pc_tail_requested();
    const std::uint32_t push_bytes = pc_tail ? 128u : 144u;
    if (props.properties.limits.maxPushConstantsSize < push_bytes) {
        std::printf("unsupported: need %u push constant bytes, device has %u\n", push_bytes,
                    props.properties.limits.maxPushConstantsSize);
        return unsupported;
    }
    VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &have12};
    vkGetPhysicalDeviceFeatures2(pd, &have);
    // Name the missing feature(s): one message per run is not enough to tell which one stopped it.
    std::string missing;
    const auto need = [&missing](bool present, const char* name) {
        if (present) return;
        if (!missing.empty()) missing += ", ";
        missing += name;
    };
    need(have12.bufferDeviceAddress, "bufferDeviceAddress");
    need(have12.runtimeDescriptorArray, "runtimeDescriptorArray");
    need(have.features.shaderInt64, "shaderInt64");
    need(have.features.shaderSampledImageArrayDynamicIndexing, "shaderSampledImageArrayDynamicIndexing");
    need(have.features.fragmentStoresAndAtomics, "fragmentStoresAndAtomics");
    // No without-format storage-image feature: a translated image_load fetches every format from the sampled
    // arrays (SHADER-151), so Intel Xe-LP, which lacks shaderStorageImageReadWithoutFormat, runs these cases too.
    if (!missing.empty()) {
        std::printf("unsupported: the device lacks %s\n", missing.c_str());
        return unsupported;
    }
    BindlessFeatures device_flags{
        have12.descriptorBindingSampledImageUpdateAfterBind && have12.descriptorBindingStorageImageUpdateAfterBind &&
            have12.descriptorBindingStorageBufferUpdateAfterBind,
        have12.descriptorBindingPartiallyBound == VK_TRUE};
    VkPhysicalDeviceVulkan12Features enable12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                              .descriptorIndexing = have12.descriptorIndexing,
                                              .descriptorBindingSampledImageUpdateAfterBind = device_flags.update_after_bind,
                                              .descriptorBindingStorageImageUpdateAfterBind = device_flags.update_after_bind,
                                              .descriptorBindingStorageBufferUpdateAfterBind = device_flags.update_after_bind,
                                              .descriptorBindingPartiallyBound = device_flags.partially_bound,
                                              .runtimeDescriptorArray = VK_TRUE,
                                              .bufferDeviceAddress = VK_TRUE};
    VkPhysicalDeviceFeatures2 enable{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                      .pNext = &enable12,
                                      .features = {.fragmentStoresAndAtomics = VK_TRUE,
                                                   .shaderSampledImageArrayDynamicIndexing = VK_TRUE,
                                                   .shaderInt64 = VK_TRUE}};
    std::uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    std::uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT)) ++qi;
    if (qi >= qn) {
        std::fprintf(stderr, "FAILED graphics queue not found (line %d)\n", __LINE__);
        return 1;
    }
    const float priority = 1;
    VkDeviceQueueCreateInfo qci{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                .queueFamilyIndex = qi,
                                .queueCount = 1,
                                .pQueuePriorities = &priority};
    VkDeviceCreateInfo dci{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &enable, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
    VkDevice dev = VK_NULL_HANDLE;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LD(n) const auto n = reinterpret_cast<PFN_##n>(vkGetDeviceProcAddr(dev, #n))
    LD(vkGetDeviceQueue); LD(vkCreateImage); LD(vkGetImageMemoryRequirements); LD(vkAllocateMemory); LD(vkBindImageMemory); LD(vkCreateImageView);
    LD(vkCreateRenderPass); LD(vkCreateFramebuffer); LD(vkCreateShaderModule); LD(vkCreatePipelineLayout); LD(vkCreateGraphicsPipelines); LD(vkCreateCommandPool);
    LD(vkAllocateCommandBuffers); LD(vkBeginCommandBuffer); LD(vkCmdBeginRenderPass); LD(vkCmdBindPipeline); LD(vkCmdDraw); LD(vkCmdEndRenderPass);
    LD(vkCmdCopyImageToBuffer); LD(vkCmdCopyBufferToImage); LD(vkEndCommandBuffer); LD(vkQueueSubmit); LD(vkQueueWaitIdle); LD(vkCreateBuffer);
    LD(vkResetCommandBuffer); LD(vkGetBufferMemoryRequirements); LD(vkBindBufferMemory); LD(vkMapMemory); LD(vkCmdBindDescriptorSets); LD(vkCmdPushConstants);
    LD(vkCmdPipelineBarrier); LD(vkDeviceWaitIdle); LD(vkDestroyPipeline); LD(vkDestroyPipelineLayout); LD(vkDestroyCommandPool); LD(vkDestroyShaderModule);
    LD(vkDestroyFramebuffer); LD(vkDestroyRenderPass); LD(vkDestroyImageView); LD(vkDestroyImage); LD(vkDestroyBuffer); LD(vkFreeMemory); LD(vkDestroyDevice);
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
        VkBufferCreateInfo bi{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
        if (VkResult r = vkCreateBuffer(dev, &bi, nullptr, b)) return r;
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, *b, &mr);
        VkMemoryAllocateInfo ai{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = mr.size,
                                .memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
        VkDeviceMemory m;
        if (VkResult r = vkAllocateMemory(dev, &ai, nullptr, &m)) return r;
        allocations.push_back(m);
        if (VkResult r = vkBindBufferMemory(dev, *b, m, 0)) return r;
        return vkMapMemory(dev, m, 0, size, 0, map);
    };
    const auto make_image = [&](VkFormat format, std::uint32_t w, std::uint32_t h, VkImageUsageFlags usage,
                                VkImage* image, VkImageView* view) {
        VkImageCreateInfo ii{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                             .imageType = VK_IMAGE_TYPE_2D,
                             .format = format,
                             .extent = {w, h, 1},
                             .mipLevels = 1,
                             .arrayLayers = 1,
                             .samples = VK_SAMPLE_COUNT_1_BIT,
                             .tiling = VK_IMAGE_TILING_OPTIMAL,
                             .usage = usage};
        if (VkResult r = vkCreateImage(dev, &ii, nullptr, image)) return r;
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(dev, *image, &mr);
        VkMemoryAllocateInfo ai{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = mr.size,
                                .memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        VkDeviceMemory m;
        if (VkResult r = vkAllocateMemory(dev, &ai, nullptr, &m)) return r;
        allocations.push_back(m);
        if (VkResult r = vkBindImageMemory(dev, *image, m, 0)) return r;
        VkImageViewCreateInfo vi{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                 .image = *image,
                                 .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                 .format = format,
                                 .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        return vkCreateImageView(dev, &vi, nullptr, view);
    };
    // The sampled image from the spec, uploaded through a staging buffer.
    VkBuffer staging;
    void* staging_map = nullptr;
    CHECK(make_buffer(static_cast<VkDeviceSize>(texels.size()), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &staging, &staging_map));
    std::memcpy(staging_map, texels.data(), texels.size());
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    const VkImageLayout image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    CHECK(make_image(spec.format, spec.width, spec.height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                     &image, &view));
    // Render target, pass and framebuffer.
    constexpr VkFormat target_format = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkImage target = VK_NULL_HANDLE;
    VkImageView target_view = VK_NULL_HANDLE;
    CHECK(make_image(target_format, W, H, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &target,
                     &target_view));
    VkAttachmentDescription ad{.format = target_format,
                               .samples = VK_SAMPLE_COUNT_1_BIT,
                               .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                               .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                               .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                               .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                               .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                               .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
    const VkAttachmentReference ar{.attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sd{.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1, .pColorAttachments = &ar};
    // The read back that follows the pass waits for its last write and its final layout transition.
    VkSubpassDependency after{.srcSubpass = 0,
                              .dstSubpass = VK_SUBPASS_EXTERNAL,
                              .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                              .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
                              .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};
    VkRenderPassCreateInfo rpi{.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                               .attachmentCount = 1,
                               .pAttachments = &ad,
                               .subpassCount = 1,
                               .pSubpasses = &sd,
                               .dependencyCount = 1,
                               .pDependencies = &after};
    VkRenderPass pass = VK_NULL_HANDLE;
    CHECK(vkCreateRenderPass(dev, &rpi, nullptr, &pass));
    VkFramebufferCreateInfo fbi{.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                .renderPass = pass,
                                .attachmentCount = 1,
                                .pAttachments = &target_view,
                                .width = W,
                                .height = H,
                                .layers = 1};
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    CHECK(vkCreateFramebuffer(dev, &fbi, nullptr, &framebuffer));
    VkBuffer readback = VK_NULL_HANDLE;
    void* readback_map = nullptr;
    CHECK(make_buffer(VkDeviceSize(W) * H * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback, &readback_map));
    VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
    for (int stage = 0; stage < 2; ++stage) {
        const auto code = read_spv(argv[1 + stage]);
        if (code.empty()) {
            std::fprintf(stderr, "cannot read %s\n", argv[1 + stage]);
            return 1;
        }
        VkShaderModuleCreateInfo smi{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = code.size() * 4, .pCode = code.data()};
        CHECK(vkCreateShaderModule(dev, &smi, nullptr, stage ? &fs : &vs));
    }
    VkCommandPoolCreateInfo cpi{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                .queueFamilyIndex = qi};
    VkCommandPool pool = VK_NULL_HANDLE;
    CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .commandBufferCount = 1};
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    const auto submit = [&]() {
        if (VkResult r = vkEndCommandBuffer(cmd)) return r;
        VkSubmitInfo si{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
        if (VkResult r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)) return r;
        return vkQueueWaitIdle(queue);
    };
    // Upload the texels, then transition to the layout used by the shader's descriptor.
    CHECK(vkBeginCommandBuffer(cmd, &begin));
    {
        VkImageMemoryBarrier ib{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &ib);
        VkBufferImageCopy c{.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {spec.width, spec.height, 1}};
        vkCmdCopyBufferToImage(cmd, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ib.newLayout = image_layout;
        ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &ib);
    }
    CHECK(submit());
    // Does a set with these flags fit the device? Every binding is visible to every stage.
    constexpr std::uint32_t sampled_total = 4096 + 1024 + 256 + 256, sampler_total = 1024, storage_total = 1024 + 13 * 256;
    const auto fits = [&](const BindlessFeatures& f) {
        const VkPhysicalDeviceLimits& l = props.properties.limits;
        if (f.update_after_bind)
            return props12.maxPerStageDescriptorUpdateAfterBindSampledImages >= sampled_total &&
                   props12.maxPerStageDescriptorUpdateAfterBindSamplers >= sampler_total &&
                   props12.maxPerStageDescriptorUpdateAfterBindStorageImages >= storage_total &&
                   props12.maxPerStageUpdateAfterBindResources >= sampled_total + sampler_total + storage_total + 1 &&
                   props12.maxDescriptorSetUpdateAfterBindSampledImages >= sampled_total &&
                   props12.maxDescriptorSetUpdateAfterBindSamplers >= sampler_total &&
                   props12.maxDescriptorSetUpdateAfterBindStorageImages >= storage_total;
        return l.maxPerStageDescriptorSampledImages >= sampled_total && l.maxPerStageDescriptorSamplers >= sampler_total &&
               l.maxPerStageDescriptorStorageImages >= storage_total &&
               l.maxPerStageResources >= sampled_total + sampler_total + storage_total + 1 &&
               l.maxDescriptorSetSampledImages >= sampled_total && l.maxDescriptorSetSamplers >= sampler_total &&
               l.maxDescriptorSetStorageImages >= storage_total;
    };
    BindlessFeatures flags = device_flags;
    if (!fits(flags)) flags = {};
    if (!fits(flags)) {
        std::printf("unsupported: the bindless set does not fit this device's descriptor limits\n");
        return unsupported;
    }
    std::optional<BindlessSet> set;
    set.emplace(vkGetDeviceProcAddr, dev, mp, flags);
    if (!set->ok() || !set->layout() || !set->set()) {
        std::printf("unsupported: cannot create the bindless set\n");
        return unsupported;
    }
    const std::uint32_t index = set->add_image(ImageDimension::dim_2d, view, image_layout);
    if (index < 1) {
        std::fprintf(stderr, "FAILED add_image -> 0 (line %d)\n", __LINE__);
        return 1;
    }
    set->insert_key(make_tsharp_key(spec.tsharp), index);
    const VkPushConstantRange push{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, .size = push_bytes};
    const VkDescriptorSetLayout set_layout = set->layout();
    VkPipelineLayoutCreateInfo pli{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                   .setLayoutCount = 1,
                                   .pSetLayouts = &set_layout,
                                   .pushConstantRangeCount = 1,
                                   .pPushConstantRanges = &push};
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(dev, &pli, nullptr, &pipeline_layout) != VK_SUCCESS) {
        std::printf("unsupported: cannot create the pipeline layout\n");
        return unsupported;
    }
    VkPipelineShaderStageCreateInfo stages[2]{{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                                                .module = vs,
                                                .pName = "main"},
                                              {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                                                .module = fs,
                                                .pName = "main"}};
    const VkPipelineVertexInputStateCreateInfo vertex_input{.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    const VkViewport viewport{.x = 0, .y = 0, .width = float(W), .height = float(H), .maxDepth = 1};
    const VkRect2D scissor{.offset = {0, 0}, .extent = {W, H}};
    VkPipelineViewportStateCreateInfo viewports{.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                                .viewportCount = 1,
                                                .pViewports = &viewport,
                                                .scissorCount = 1,
                                                .pScissors = &scissor};
    VkPipelineRasterizationStateCreateInfo raster{.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                  .cullMode = VK_CULL_MODE_NONE,
                                                  .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo multisample{.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                     .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState blend_attachment{.colorWriteMask = 0xf};
    VkPipelineColorBlendStateCreateInfo blend{.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = 1,
                                              .pAttachments = &blend_attachment};
    VkGraphicsPipelineCreateInfo gpi{.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                     .stageCount = 2,
                                     .pStages = stages,
                                     .pVertexInputState = &vertex_input,
                                     .pInputAssemblyState = &assembly,
                                     .pViewportState = &viewports,
                                     .pRasterizationState = &raster,
                                     .pMultisampleState = &multisample,
                                     .pColorBlendState = &blend,
                                     .layout = pipeline_layout,
                                     .renderPass = pass};
    VkPipeline pipeline = VK_NULL_HANDLE;
    CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipeline));
    // Publish the image index and the table, then draw and read back.
    vkResetCommandBuffer(cmd, 0);
    CHECK(vkBeginCommandBuffer(cmd, &begin));
    if (!set->flush(cmd)) {
        std::fprintf(stderr, "FAILED set->flush (line %d)\n", __LINE__);
        return 1;
    }
    CHECK(submit());
    vkResetCommandBuffer(cmd, 0);
    CHECK(vkBeginCommandBuffer(cmd, &begin));
    if (!set->fallbacks_ready()) set->record_fallback_init(cmd);
    VkClearValue clear{};
    VkRenderPassBeginInfo rbi{.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                              .renderPass = pass,
                              .framebuffer = framebuffer,
                              .renderArea = scissor,
                              .clearValueCount = 1,
                              .pClearValues = &clear};
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkDescriptorSet bound = set->set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &bound, 0, nullptr);
    std::uint32_t block[36] = {};
    std::memcpy(block + 8, spec.tsharp, sizeof(spec.tsharp)); // s[8:15]: the T# the fragment shader loads
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, push_bytes, block);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    VkBufferImageCopy rc{.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {W, H, 1}};
    vkCmdCopyImageToBuffer(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &rc);
    CHECK(submit());
    {
        std::ofstream out(argv[3], std::ios::binary);
        out.write(static_cast<const char*>(readback_map), VkDeviceSize(W) * H * 16);
        if (!out) {
            std::fprintf(stderr, "cannot write %s\n", argv[3]);
            return 1;
        }
    }
    std::printf("vk_image_fuzz: %ux%u %s wrote %s\n", spec.width, spec.height, spec.format_name.c_str(), argv[3]);
    // Let validation observe resource and device destruction before reporting its final error count.
    vkDeviceWaitIdle(dev);
    set.reset();  // its buffers and descriptor set must die before the device
    vkDestroyPipeline(dev, pipeline, nullptr), vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
    vkDestroyCommandPool(dev, pool, nullptr), vkDestroyFramebuffer(dev, framebuffer, nullptr);
    vkDestroyRenderPass(dev, pass, nullptr), vkDestroyShaderModule(dev, fs, nullptr), vkDestroyShaderModule(dev, vs, nullptr);
    vkDestroyImageView(dev, target_view, nullptr), vkDestroyImageView(dev, view, nullptr);
    vkDestroyImage(dev, target, nullptr), vkDestroyImage(dev, image, nullptr);
    vkDestroyBuffer(dev, readback, nullptr), vkDestroyBuffer(dev, staging, nullptr);
    for (const auto memory : allocations) vkFreeMemory(dev, memory, nullptr);
    vkDestroyDevice(dev, nullptr);
    if (messenger) {
        LI(vkDestroyDebugUtilsMessengerEXT);
        vkDestroyDebugUtilsMessengerEXT(inst, messenger, nullptr);
    }
    LI(vkDestroyInstance);
    vkDestroyInstance(inst, nullptr);
    if (g_validation_errors) {
        std::fprintf(stderr, "%u validation errors\n", g_validation_errors);
        return 1;
    }
    return 0;
}
