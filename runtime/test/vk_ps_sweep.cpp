// Draws one translated pixel shader with the runtime's bindless set 0 and a host heap, on whichever GPU the loader
// gives it, and writes every colour target as raw RGBA32F. It is the per-shader worker of a corpus sweep: the
// vertex shader is a generated one that outputs vec4 varyings for the pixel shader's input locations
// (tools/ps_sweep.py), so any pixel shader can be drawn without its real vertex stage.
//
//   vk_ps_sweep <vs.spv> <ps.spv> <out.raw> <width> <height> <render-targets>
//
// Inputs follow the contract in docs/design.md and are identical on every device:
//   set 0       the BindlessSet of runtime/src/vulkan_bindless.h with only its fallbacks: an empty descriptor
//               table, so every T# and S# the shader reads resolves to index 0 (a zero image, a nearest sampler)
//   VKS_TEX=1   replaces the fallback (index 0) sampled images and sampler with an 8x8 gradient (r = x/8, g = y/8,
//               b = 0.25 + 0.15 * z or layer, a = 1) in every sampled dimension (2D, 2D array of 6, 1D, 3D of
//               depth 4) and a linear clamping sampler, so a shader whose T# and S# miss the table (a zeroed heap)
//               samples texels; the storage bindings keep their 1x1 fallbacks
//   VKS_TEX=sint|uint selects an RGBA8 integer view at 2D element zero for bounded constant texelFetch.
//               --screen-integer <ps.spv> checks applicability without initializing Vulkan.
//   VKS_GS=<gs.spv>
//               a geometry shader between the two (tools/bary_fallback_gs.py's pass-through GS for a pixel shader
//               translated with --bary-fallback: it writes the barycentric input that shader reads); needs the
//               geometryShader feature
//   push        s0..s31 from VKS_SGPRS (like VKH_SGPRS), then the heap base
//   heap        VKS_HEAP_KB KiB (default 1024), zero unless VKS_HEAP_FILL=<offset>,<bytes>,<seed> fills a range with
//               floats in [0.25, 1)
// Output: the targets back to back, each width * height * 16 bytes, cleared to -1 so a pixel that was never
// written stays clear. --depth also writes <out.raw>.depth, one D32_SFLOAT word per pixel, cleared to 1.
// It enables precise occlusion counting, depth test ALWAYS, depth writes, and no stencil or depth bounds test.
//
// Exit codes: 0 drawn; 77 the device lacks a feature the module needs (stdout says which); 1 a Vulkan call failed
// (stderr says which); 3 the fence did not signal in VKS_TIMEOUT_S seconds (default 20); 4 the device was lost.
#define VK_NO_PROTOTYPES
#include "vulkan_loader.h"
#include "vk_device_pick.h"
#include "ps_sweep_integer_fetch.h"
#include "vk_shader_fma.h"
#include "../src/vulkan_bindless.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

static PFN_vkGetInstanceProcAddr gipa;
#define CHECK(x)                                                                         \
    do {                                                                                 \
        VkResult r_ = (x);                                                               \
        if (r_ == VK_ERROR_DEVICE_LOST) {                                                \
            std::fprintf(stderr, "DEVICE_LOST %s (line %d)\n", #x, __LINE__);            \
            return 4;                                                                    \
        }                                                                                \
        if (r_ != VK_SUCCESS) {                                                          \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, int(r_), __LINE__);  \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

static std::vector<uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    if (b.empty() || b.size() % 4) return {};
    std::vector<uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), w.size() * 4);
    return w;
}

static std::vector<uint32_t> capabilities(const std::vector<uint32_t>& code) {
    std::vector<uint32_t> caps;
    for (std::size_t i = 5; i + 1 < code.size();) {
        const uint32_t words = code[i] >> 16, op = code[i] & 0xffffu;
        if (!words) break;
        if (op == 17) caps.push_back(code[i + 1]);
        i += words;
    }
    return caps;
}

// VKS_MAX_RECONV=1: add SPV_KHR_maximal_reconvergence and OpExecutionMode MaximallyReconvergesKHR to the module.
static void add_maximal_reconvergence(std::vector<uint32_t>& code) {
    std::size_t ext_at = 0, mode_at = 0;
    uint32_t entry = 0;
    for (std::size_t i = 5; i < code.size();) {
        const uint32_t words = code[i] >> 16, op = code[i] & 0xffffu;
        if (!words) break;
        if (!ext_at && (op == 11 || op == 14)) ext_at = i;   // first OpExtInstImport / OpMemoryModel
        if (op == 15) { entry = code[i + 2]; mode_at = i + words; }
        i += words;
    }
    if (!ext_at || !mode_at) return;
    const char ext[] = "SPV_KHR_maximal_reconvergence";
    std::vector<uint32_t> name((sizeof(ext) + 3) / 4, 0);
    std::memcpy(name.data(), ext, sizeof(ext));
    std::vector<uint32_t> ins_ext{uint32_t((1 + name.size()) << 16 | 10)};
    ins_ext.insert(ins_ext.end(), name.begin(), name.end());
    const std::vector<uint32_t> ins_mode{3u << 16 | 16u, entry, 6023u};   // MaximallyReconvergesKHR
    code.insert(code.begin() + std::ptrdiff_t(mode_at), ins_mode.begin(), ins_mode.end());   // later position first
    code.insert(code.begin() + std::ptrdiff_t(ext_at), ins_ext.begin(), ins_ext.end());
}

static bool parse_words(const char* s, uint32_t* out, uint32_t max) {
    for (uint32_t i = 0; *s; ++i) {
        if (i == max) return false;
        char* end;
        out[i] = uint32_t(std::strtoull(s, &end, 0));
        if (end == s) return false;
        s = end;
        if (*s == ',') ++s;
    }
    return true;
}

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const double t_start = now_s();
    bool depth = false, occlusion = false;
    float depth_clear = 1.0f;
    if (argc == 3 && std::strcmp(argv[1], "--screen-integer") == 0) {
        const char* kind = integer_fetch_type(read_spv(argv[2]));
        std::puts(kind ? kind : "unsupported");
        return kind ? 0 : 77;
    }
    if (argc < 7) {
        std::fprintf(stderr, "usage: vk_ps_sweep vs.spv ps.spv out.raw width height render-targets [--occlusion] [--depth | --depth-zero]\n");
        return 2;
    }
    for (int i = 7; i < argc; ++i) {
        if (std::strcmp(argv[i], "--occlusion") == 0) occlusion = true;
        else if (std::strcmp(argv[i], "--depth") == 0) depth = true;
        else if (std::strcmp(argv[i], "--depth-zero") == 0) {
            depth = true;
            depth_clear = 0.0f;
        } else return 2;
    }
    // Count surviving samples even when their depth equals the attachment clear.
    occlusion |= depth;
    const uint32_t W = uint32_t(std::atoi(argv[4])), H = uint32_t(std::atoi(argv[5]));
    const uint32_t targets = uint32_t(std::atoi(argv[6]));
    if (!W || !H || targets < 1 || targets > 8) return 2;
    uint32_t pc[36] = {};
    if (const char* s = std::getenv("VKS_SGPRS"); s && !parse_words(s, pc, 32)) {
        std::fprintf(stderr, "VKS_SGPRS: expected up to 32 comma-separated words\n");
        return 2;
    }
    const char* heap_kb_env = std::getenv("VKS_HEAP_KB");
    const uint64_t heap_bytes = (heap_kb_env && std::atoi(heap_kb_env) > 0 ? uint64_t(std::atoi(heap_kb_env)) : 1024u) << 10;
    uint32_t fill[3] = {};   // offset, bytes, seed
    bool fill_heap = false;
    if (const char* s = std::getenv("VKS_HEAP_FILL"); s && *s) {
        if (!parse_words(s, fill, 3) || fill[0] % 4 || fill[1] % 4 || uint64_t(fill[0]) + fill[1] > heap_bytes) {
            std::fprintf(stderr, "VKS_HEAP_FILL: expected offset,bytes,seed inside the heap\n");
            return 2;
        }
        fill_heap = true;
    }
    const char* timeout_env = std::getenv("VKS_TIMEOUT_S");
    const uint64_t timeout_ns = (timeout_env && std::atoi(timeout_env) > 0 ? uint64_t(std::atoi(timeout_env)) : 20u) * 1000000000ull;

    auto fs_code = read_spv(argv[2]);
    const char* tex_mode = std::getenv("VKS_TEX");
    const bool integer_tex = tex_mode && (!std::strcmp(tex_mode, "sint") || !std::strcmp(tex_mode, "uint"));
    if (integer_tex) {
        const char* kind = integer_fetch_type(fs_code);
        if (!kind || std::strcmp(kind, tex_mode) || std::getenv("VKS_GS") || std::getenv("VKS_TEX_CONST")) {
            std::puts("unsupported: integer texture needs a matching bounded 2D fetch and fixed fixture inputs");
            return 77;
        }
    }
    gipa = load_vulkan();
    if (!gipa) return 1;
    VkInstance inst = VK_NULL_HANDLE;
#define LOADI(name) auto name = reinterpret_cast<PFN_##name>(gipa(inst, #name))
    LOADI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    LOADI(vkEnumeratePhysicalDevices); LOADI(vkGetPhysicalDeviceProperties); LOADI(vkGetPhysicalDeviceProperties2);
    LOADI(vkGetPhysicalDeviceFeatures2); LOADI(vkGetPhysicalDeviceQueueFamilyProperties);
    LOADI(vkGetPhysicalDeviceMemoryProperties); LOADI(vkCreateDevice); LOADI(vkGetDeviceProcAddr);
    LOADI(vkEnumerateDeviceExtensionProperties);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    std::vector<VkPhysicalDeviceProperties> candidates(pds.size());
    for (std::size_t i = 0; i < pds.size(); ++i) vkGetPhysicalDeviceProperties(pds[i], &candidates[i]);
    const std::size_t chosen = leonida::vk_test::pick_device(candidates);
    if (chosen >= pds.size()) {
        leonida::vk_test::report_no_device();
        return 1;
    }
    VkPhysicalDevice pd = pds[chosen];
    VkPhysicalDeviceVulkan11Properties props11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties props12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    props11.pNext = &props12;
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &props11};
    vkGetPhysicalDeviceProperties2(pd, &props);
    std::fprintf(stderr, "device: %s\n", props.properties.deviceName);
    constexpr uint32_t pc_bytes = 144;
    if (props.properties.limits.maxPushConstantsSize < pc_bytes) {
        std::printf("unsupported: push constants %u bytes\n", props.properties.limits.maxPushConstantsSize);
        return 77;
    }
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT)) qi++;
    if (qi == qn) return 1;

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, exts.data());
    auto has_ext = [&](const char* e) {
        for (auto& x : exts) if (!std::strcmp(x.extensionName, e)) return true;
        return false;
    };

    // What the pixel shader declares, against what the device has.
    const bool max_reconv = std::getenv("VKS_MAX_RECONV") != nullptr;
    if (max_reconv && fs_code.size() >= 6) add_maximal_reconvergence(fs_code);
    if (fs_code.size() < 6) {
        std::fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary_have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT lock_have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT};
    VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    have.pNext = &have12;
    have12.pNext = &lock_have;
    lock_have.pNext = &bary_have;
    vkGetPhysicalDeviceFeatures2(pd, &have);
    VkPhysicalDeviceSubgroupProperties sub{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 sub2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sub};
    vkGetPhysicalDeviceProperties2(pd, &sub2);

    std::string missing;
    auto need = [&](bool ok, const char* what) {
        if (ok) return;
        if (!missing.empty()) missing += ", ";
        missing += what;
    };
    VkPhysicalDeviceFeatures2 enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    if (occlusion) {
        need(have.features.occlusionQueryPrecise, "occlusionQueryPrecise");
        enable.features.occlusionQueryPrecise = VK_TRUE;
    }
    VkPhysicalDeviceVulkan12Features en12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT lock{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT};
    std::vector<const char*> enable_exts;
    uint32_t subgroup_ops = 0;
    for (const uint32_t cap : capabilities(fs_code)) {
        switch (cap) {
            case 10: need(have.features.shaderFloat64, "shaderFloat64"); enable.features.shaderFloat64 = VK_TRUE; break;
            case 11: need(have.features.shaderInt64, "shaderInt64"); enable.features.shaderInt64 = VK_TRUE; break;
            case 42: need(have.features.shaderResourceMinLod, "shaderResourceMinLod"); enable.features.shaderResourceMinLod = VK_TRUE; break;
            case 50: need(have.features.shaderImageGatherExtended, "shaderImageGatherExtended"); enable.features.shaderImageGatherExtended = VK_TRUE; break;
            case 55: need(have.features.shaderStorageImageReadWithoutFormat, "shaderStorageImageReadWithoutFormat"); enable.features.shaderStorageImageReadWithoutFormat = VK_TRUE; break;
            case 56: need(have.features.shaderStorageImageWriteWithoutFormat, "shaderStorageImageWriteWithoutFormat"); enable.features.shaderStorageImageWriteWithoutFormat = VK_TRUE; break;
            case 61: subgroup_ops |= VK_SUBGROUP_FEATURE_BASIC_BIT; break;
            case 62: subgroup_ops |= VK_SUBGROUP_FEATURE_VOTE_BIT; break;
            case 63: subgroup_ops |= VK_SUBGROUP_FEATURE_ARITHMETIC_BIT; break;
            case 64: subgroup_ops |= VK_SUBGROUP_FEATURE_BALLOT_BIT; break;
            case 65: subgroup_ops |= VK_SUBGROUP_FEATURE_SHUFFLE_BIT; break;
            case 66: subgroup_ops |= VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT; break;
            case 67: subgroup_ops |= VK_SUBGROUP_FEATURE_CLUSTERED_BIT; break;
            case 68: subgroup_ops |= VK_SUBGROUP_FEATURE_QUAD_BIT; break;
            case 5284:   // FragmentBarycentricKHR
                need(has_ext(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME) && bary_have.fragmentShaderBarycentric,
                     "fragmentShaderBarycentric");
                break;
            case 5378:   // FragmentShaderPixelInterlockEXT
                need(has_ext(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME) && lock_have.fragmentShaderPixelInterlock,
                     "fragmentShaderPixelInterlock");
                break;
        }
    }
    const char* gs_path = std::getenv("VKS_GS");
    std::vector<uint32_t> gs_code;
    if (gs_path && *gs_path) {
        gs_code = read_spv(gs_path);
        if (gs_code.size() < 6) {
            std::fprintf(stderr, "cannot read %s\n", gs_path);
            return 1;
        }
        need(have.features.geometryShader, "geometryShader");
        enable.features.geometryShader = VK_TRUE;
    }
    if (std::getenv("VKS_NEED_MAINTENANCE8")) need(has_ext("VK_KHR_maintenance8"), "maintenance8");
    if (subgroup_ops) {
        need((sub.supportedStages & VK_SHADER_STAGE_FRAGMENT_BIT) != 0, "subgroup operations in fragment shaders");
        need((sub.supportedOperations & subgroup_ops) == subgroup_ops, "a subgroup operation class");
    }
    leonida::vk_test::ShaderFma fma;
    fma.query(gipa, inst, pd);
    if (!fma.accepts(fs_code.data(), fs_code.size() * 4, argv[2])) return 77;
    need(have12.bufferDeviceAddress && have12.runtimeDescriptorArray, "bufferDeviceAddress or runtimeDescriptorArray");
    need(have.features.shaderInt64 && have.features.fragmentStoresAndAtomics &&
             have.features.shaderSampledImageArrayDynamicIndexing,
         "shaderInt64, fragmentStoresAndAtomics or shaderSampledImageArrayDynamicIndexing");
    if (!missing.empty()) {
        std::printf("unsupported: %s\n", missing.c_str());
        return 77;
    }
    enable.features.shaderInt64 = VK_TRUE;
    enable.features.fragmentStoresAndAtomics = VK_TRUE;
    enable.features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    enable.features.shaderStorageImageArrayDynamicIndexing = have.features.shaderStorageImageArrayDynamicIndexing;
    // Optional extras the sweep leaves on when present, so a shader sees the same device state a runtime would give it.
    if (has_ext(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME) && bary_have.fragmentShaderBarycentric) {
        enable_exts.push_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
        bary.fragmentShaderBarycentric = VK_TRUE;
    }
    if (has_ext(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME) && lock_have.fragmentShaderPixelInterlock) {
        enable_exts.push_back(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME);
        lock.fragmentShaderPixelInterlock = VK_TRUE;
    }
#ifdef VK_KHR_MAINTENANCE_8_EXTENSION_NAME
    VkPhysicalDeviceMaintenance8FeaturesKHR m8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
    const bool m8_on = has_ext(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
    if (m8_on) {
        enable_exts.push_back(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
        m8.maintenance8 = VK_TRUE;
    }
#else
    const bool m8_on = false;
#endif
    if (fma.supported()) enable_exts.push_back(fma.extension());
#ifdef VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME
    VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR mr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR};
    if (max_reconv) {
        if (!has_ext(VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME)) {
            std::printf("unsupported: shaderMaximalReconvergence\n");
            return 77;
        }
        enable_exts.push_back(VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME);
        mr.shaderMaximalReconvergence = VK_TRUE;
    }
#endif
    leonida::BindlessFeatures flags{
        have12.descriptorBindingSampledImageUpdateAfterBind && have12.descriptorBindingStorageImageUpdateAfterBind &&
            have12.descriptorBindingStorageBufferUpdateAfterBind,
        have12.descriptorBindingPartiallyBound == VK_TRUE};
    en12.descriptorIndexing = have12.descriptorIndexing;
    en12.descriptorBindingSampledImageUpdateAfterBind = flags.update_after_bind;
    en12.descriptorBindingStorageImageUpdateAfterBind = flags.update_after_bind;
    en12.descriptorBindingStorageBufferUpdateAfterBind = flags.update_after_bind;
    en12.descriptorBindingPartiallyBound = flags.partially_bound;
    en12.runtimeDescriptorArray = VK_TRUE;
    en12.bufferDeviceAddress = VK_TRUE;
    // pNext chain: features2 -> vulkan12 -> [barycentric] -> [interlock] -> [maintenance8] -> [fma]
    void** tail = &en12.pNext;
    if (bary.fragmentShaderBarycentric) { *tail = &bary; tail = &bary.pNext; }
    if (lock.fragmentShaderPixelInterlock) { *tail = &lock; tail = &lock.pNext; }
#ifdef VK_KHR_MAINTENANCE_8_EXTENSION_NAME
    if (m8_on) { *tail = &m8; tail = &m8.pNext; }
#endif
#ifdef VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME
    if (max_reconv) { *tail = &mr; tail = &mr.pNext; }
#endif
#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
    if (fma.supported()) { *tail = &fma.features; tail = &fma.features.pNext; }
#endif
    *tail = nullptr;
    enable.pNext = &en12;

    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &enable;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(enable_exts.size()); dci.ppEnabledExtensionNames = enable_exts.data();
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LOADD(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(dev, #name))
    LOADD(vkGetDeviceQueue); LOADD(vkCreateImage); LOADD(vkGetImageMemoryRequirements); LOADD(vkAllocateMemory);
    LOADD(vkBindImageMemory); LOADD(vkCreateImageView); LOADD(vkCreateRenderPass); LOADD(vkCreateFramebuffer);
    LOADD(vkCreateShaderModule); LOADD(vkCreatePipelineLayout); LOADD(vkCreateGraphicsPipelines);
    LOADD(vkCreateCommandPool); LOADD(vkAllocateCommandBuffers); LOADD(vkBeginCommandBuffer);
    LOADD(vkCmdBeginRenderPass); LOADD(vkCmdBindPipeline); LOADD(vkCmdDraw); LOADD(vkCmdEndRenderPass);
    LOADD(vkCmdCopyImageToBuffer); LOADD(vkEndCommandBuffer); LOADD(vkQueueSubmit); LOADD(vkCreateBuffer);
    LOADD(vkGetBufferMemoryRequirements); LOADD(vkBindBufferMemory); LOADD(vkMapMemory);
    LOADD(vkGetBufferDeviceAddress); LOADD(vkCmdBindDescriptorSets); LOADD(vkCmdPushConstants);
    LOADD(vkCreateFence); LOADD(vkWaitForFences);
    VkQueue q;
    vkGetDeviceQueue(dev, qi, 0, &q);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memtype = [&](uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0u;
    };

    const double t_device = now_s();
    // Set 0.
    constexpr uint32_t sampled_total = 4096 + 1024 + 256 + 256, sampler_total = 1024, storage_total = 1024 + 13 * 256;
    const auto fits = [&](const leonida::BindlessFeatures& f) {
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
    if (!fits(flags)) flags = {};
    if (!fits(flags)) {
        // Fall back to the sizes the runtime would pick for this device.
        flags.update_after_bind = en12.descriptorBindingSampledImageUpdateAfterBind;
        flags.partially_bound = en12.descriptorBindingPartiallyBound;
        leonida::bindless::size_bindings(flags, props.properties.limits, props11, props12);
    }
    std::optional<leonida::BindlessSet> set;
    set.emplace(vkGetDeviceProcAddr, dev, mp, flags);
    if (!set->ok() || !set->layout() || !set->set()) {
        std::printf("unsupported: cannot create the bindless set\n");
        return 77;
    }

    const double t_set = now_s();
    // Optional gradient textures behind the all-zero T# and S#.
    struct Tex { VkImage img = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; uint32_t w, h, d, layers; VkImageType type; VkImageViewType vt; uint32_t binding; };
    std::vector<Tex> texs;
    std::vector<VkDeviceSize> tex_offsets;
    VkBuffer tex_stage = VK_NULL_HANDLE;
    VkSampler tex_sampler = VK_NULL_HANDLE;
    if (std::getenv("VKS_TEX")) {
        const char* tc = std::getenv("VKS_TEX_CONST");   // all texels one RGBA8 value (hex, 0xAABBGGRR), for bisecting
        const bool tex_const = tc && *tc;
        const uint32_t tex_const_value = tex_const ? uint32_t(std::strtoul(tc, nullptr, 16)) : 0u;
        LOADD(vkCreateSampler); LOADD(vkUpdateDescriptorSets);
        texs = {{VK_NULL_HANDLE, VK_NULL_HANDLE, 8, 8, 1, 1, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 0},
                {VK_NULL_HANDLE, VK_NULL_HANDLE, 8, 8, 1, 6, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 3},
                {VK_NULL_HANDLE, VK_NULL_HANDLE, 8, 1, 1, 1, VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D, 5},
                {VK_NULL_HANDLE, VK_NULL_HANDLE, 8, 8, 4, 1, VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, 6}};
        if (integer_tex) texs.resize(1);
        std::vector<uint32_t> texels;
        std::vector<VkDeviceSize> offsets;
        for (auto& t : texs) {
            offsets.push_back(texels.size() * 4);
            for (uint32_t z = 0; z < t.d * t.layers; ++z)
                for (uint32_t y = 0; y < t.h; ++y)
                    for (uint32_t x = 0; x < t.w; ++x)
                        texels.push_back(integer_tex ? (128u + x) | ((31u + y) << 8) | ((250u - x) << 16) | ((80u - y) << 24)
                                                   : tex_const ? tex_const_value
                                                   : (x * 32u) | ((y * 32u) << 8) | ((64u + z * 40u) << 16) | (255u << 24));
        }
        void* sp = nullptr;
        VkDeviceMemory sm;
        VkBufferCreateInfo sbi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sbi.size = texels.size() * 4; sbi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        CHECK(vkCreateBuffer(dev, &sbi, nullptr, &tex_stage));
        VkMemoryRequirements sr;
        vkGetBufferMemoryRequirements(dev, tex_stage, &sr);
        VkMemoryAllocateInfo sai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        sai.allocationSize = sr.size;
        sai.memoryTypeIndex = memtype(sr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        CHECK(vkAllocateMemory(dev, &sai, nullptr, &sm));
        CHECK(vkBindBufferMemory(dev, tex_stage, sm, 0));
        CHECK(vkMapMemory(dev, sm, 0, sbi.size, 0, &sp));
        std::memcpy(sp, texels.data(), texels.size() * 4);
        for (std::size_t k = 0; k < texs.size(); ++k) {
            Tex& t = texs[k];
            VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ii.imageType = t.type; ii.format = integer_tex ? (!std::strcmp(tex_mode, "sint") ? VK_FORMAT_R8G8B8A8_SINT : VK_FORMAT_R8G8B8A8_UINT)
                                                         : VK_FORMAT_R8G8B8A8_UNORM; ii.extent = {t.w, t.h, t.d};
            ii.mipLevels = 1; ii.arrayLayers = t.layers; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            CHECK(vkCreateImage(dev, &ii, nullptr, &t.img));
            VkMemoryRequirements mr;
            vkGetImageMemoryRequirements(dev, t.img, &mr);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = mr.size;
            ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VkDeviceMemory m;
            CHECK(vkAllocateMemory(dev, &ai, nullptr, &m));
            CHECK(vkBindImageMemory(dev, t.img, m, 0));
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = t.img; vi.viewType = t.vt; vi.format = ii.format;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, t.layers};
            CHECK(vkCreateImageView(dev, &vi, nullptr, &t.view));
        }
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = integer_tex || std::getenv("VKS_TEX_NEAREST") ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        CHECK(vkCreateSampler(dev, &sci, nullptr, &tex_sampler));
        std::vector<VkDescriptorImageInfo> infos(texs.size() + 1);
        std::vector<VkWriteDescriptorSet> writes(texs.size() + 1);
        for (std::size_t k = 0; k < texs.size(); ++k) {
            infos[k] = {VK_NULL_HANDLE, texs[k].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            writes[k] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[k].dstSet = set->set(); writes[k].dstBinding = texs[k].binding; writes[k].dstArrayElement = 0;
            writes[k].descriptorCount = 1; writes[k].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; writes[k].pImageInfo = &infos[k];
        }
        infos[texs.size()] = {tex_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        VkWriteDescriptorSet& sw = writes[texs.size()];
        sw = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        sw.dstSet = set->set(); sw.dstBinding = 1; sw.dstArrayElement = 0; sw.descriptorCount = 1;
        sw.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; sw.pImageInfo = &infos[texs.size()];
        vkUpdateDescriptorSets(dev, uint32_t(writes.size()), writes.data(), 0, nullptr);
        tex_offsets = offsets;   // used by the upload commands below
    }
    // Heap.
    VkBufferCreateInfo hbi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    hbi.size = heap_bytes;
    hbi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBuffer heap;
    CHECK(vkCreateBuffer(dev, &hbi, nullptr, &heap));
    VkMemoryRequirements hr;
    vkGetBufferMemoryRequirements(dev, heap, &hr);
    VkMemoryAllocateFlagsInfo haf{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    haf.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo hai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    hai.pNext = &haf; hai.allocationSize = hr.size;
    hai.memoryTypeIndex = memtype(hr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory hmem;
    CHECK(vkAllocateMemory(dev, &hai, nullptr, &hmem));
    CHECK(vkBindBufferMemory(dev, heap, hmem, 0));
    void* hp;
    CHECK(vkMapMemory(dev, hmem, 0, heap_bytes, 0, &hp));
    std::memset(hp, 0, size_t(heap_bytes));
    if (fill_heap) {
        uint32_t x = fill[2] * 2654435761u + 0x9e3779b9u;
        for (uint32_t o = fill[0]; o < fill[0] + fill[1]; o += 4) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            const uint32_t bits = 0x3e800000u + (x % 0x01000000u);   // [0.25, 1)
            std::memcpy(static_cast<char*>(hp) + o, &bits, 4);
        }
    }
    VkBufferDeviceAddressInfo hda{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    hda.buffer = heap;
    const uint64_t base = vkGetBufferDeviceAddress(dev, &hda);
    pc[32] = uint32_t(base); pc[33] = uint32_t(base >> 32);

    // Targets.
    const VkFormat fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
    const uint32_t attachments = targets + uint32_t(depth);
    if (depth) {
        LOADI(vkGetPhysicalDeviceImageFormatProperties);
        VkImageFormatProperties image{};
        const auto result = vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_D32_SFLOAT, VK_IMAGE_TYPE_2D,
            VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0, &image);
        if (result == VK_ERROR_FORMAT_NOT_SUPPORTED || (result == VK_SUCCESS &&
            (image.maxExtent.width < W || image.maxExtent.height < H || !(image.sampleCounts & VK_SAMPLE_COUNT_1_BIT)))) {
            std::puts("unsupported: D32_SFLOAT depth attachment readback");
            return 77;
        }
        CHECK(result);
    }
    std::vector<VkImage> imgs(attachments);
    std::vector<VkImageView> views(attachments);
    for (uint32_t t = 0; t < attachments; ++t) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = t == targets ? VK_FORMAT_D32_SFLOAT : fmt; ii.extent = {W, H, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = (t == targets ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        CHECK(vkCreateImage(dev, &ii, nullptr, &imgs[t]));
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(dev, imgs[t], &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m;
        CHECK(vkAllocateMemory(dev, &ai, nullptr, &m));
        CHECK(vkBindImageMemory(dev, imgs[t], m, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = imgs[t]; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = ii.format;
        vi.subresourceRange = {VkImageAspectFlags(t == targets ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
        CHECK(vkCreateImageView(dev, &vi, nullptr, &views[t]));
    }
    std::vector<VkAttachmentDescription> ads(attachments);
    std::vector<VkAttachmentReference> refs(attachments);
    for (uint32_t t = 0; t < attachments; ++t) {
        ads[t] = {};
        ads[t].format = t == targets ? VK_FORMAT_D32_SFLOAT : fmt; ads[t].samples = VK_SAMPLE_COUNT_1_BIT;
        ads[t].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ads[t].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ads[t].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; ads[t].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ads[t].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; ads[t].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        refs[t] = {t, t == targets ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    }
    VkSubpassDescription sd{};
    sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; sd.colorAttachmentCount = targets; sd.pColorAttachments = refs.data();
    if (depth) sd.pDepthStencilAttachment = &refs[targets];
    VkSubpassDependency dep{0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    if (depth) {
        dep.srcStageMask |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = attachments; rpi.pAttachments = ads.data(); rpi.subpassCount = 1; rpi.pSubpasses = &sd;
    rpi.dependencyCount = 1; rpi.pDependencies = &dep;
    VkRenderPass rp;
    CHECK(vkCreateRenderPass(dev, &rpi, nullptr, &rp));
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = rp; fbi.attachmentCount = attachments; fbi.pAttachments = views.data(); fbi.width = W; fbi.height = H; fbi.layers = 1;
    VkFramebuffer fb;
    CHECK(vkCreateFramebuffer(dev, &fbi, nullptr, &fb));

    // Pipeline.
    auto module = [&](const char* p, VkShaderModule* m) {
        auto code = read_spv(p);
        if (code.empty()) {
            std::fprintf(stderr, "cannot read %s\n", p);
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size() * 4; smi.pCode = code.data();
        return vkCreateShaderModule(dev, &smi, nullptr, m);
    };
    VkShaderModule vs, fs;
    CHECK(module(argv[1], &vs));
    VkShaderModuleCreateInfo fsi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    fsi.codeSize = fs_code.size() * 4; fsi.pCode = fs_code.data();
    CHECK(vkCreateShaderModule(dev, &fsi, nullptr, &fs));
    VkPipelineShaderStageCreateInfo stages[3]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                              {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                              {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    uint32_t stage_count = 0;
    stages[stage_count].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[stage_count].module = vs;
    stages[stage_count++].pName = "main";
    if (!gs_code.empty()) {
        VkShaderModule gs;
        VkShaderModuleCreateInfo gsi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        gsi.codeSize = gs_code.size() * 4; gsi.pCode = gs_code.data();
        CHECK(vkCreateShaderModule(dev, &gsi, nullptr, &gs));
        stages[stage_count].stage = VK_SHADER_STAGE_GEOMETRY_BIT; stages[stage_count].module = gs;
        stages[stage_count++].pName = "main";
    }
    stages[stage_count].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[stage_count].module = fs;
    stages[stage_count++].pName = "main";
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp{0, 0, float(W), float(H), 0, 1};
    VkRect2D sc{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp; vps.scissorCount = 1; vps.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    std::vector<VkPipelineColorBlendAttachmentState> cbas(targets);
    for (auto& a : cbas) { a = {}; a.colorWriteMask = 0xF; }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = targets; cb.pAttachments = cbas.data();
    const VkShaderStageFlags pcs = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkPushConstantRange pcr{pcs, 0, pc_bytes};
    const VkDescriptorSetLayout set_layout = set->layout();
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &set_layout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl;
    CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = stage_count; gpi.pStages = stages; gpi.pVertexInputState = &vin; gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vps; gpi.pRasterizationState = &rs; gpi.pMultisampleState = &ms; gpi.pColorBlendState = &cb;
    VkPipelineDepthStencilStateCreateInfo ds_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    if (depth) {
        ds_state.depthTestEnable = ds_state.depthWriteEnable = VK_TRUE;
        ds_state.depthCompareOp = VK_COMPARE_OP_ALWAYS;
        gpi.pDepthStencilState = &ds_state;
    }
    gpi.layout = pl; gpi.renderPass = rp;
    VkPipeline pipe;
    {
        const VkResult r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipe);
        if (r != VK_SUCCESS) {
            std::fprintf(stderr, "PIPELINE %d\n", int(r));
            return r == VK_ERROR_DEVICE_LOST ? 4 : 5;
        }
    }

    const double t_pipe = now_s();
    // Readback buffer.
    const VkDeviceSize target_bytes = VkDeviceSize(W) * H * 16;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    const VkDeviceSize color_bytes = target_bytes * targets;
    bi.size = color_bytes + (depth ? target_bytes / 4 : 0); bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buf;
    CHECK(vkCreateBuffer(dev, &bi, nullptr, &buf));
    VkMemoryRequirements br;
    vkGetBufferMemoryRequirements(dev, buf, &br);
    VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    bai.allocationSize = br.size;
    bai.memoryTypeIndex = memtype(br.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory bmem;
    CHECK(vkAllocateMemory(dev, &bai, nullptr, &bmem));
    CHECK(vkBindBufferMemory(dev, buf, bmem, 0));

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qi;
    VkCommandPool pool;
    CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &cbi));
    LOADD(vkCmdPipelineBarrier); LOADD(vkCmdCopyBufferToImage);
    for (std::size_t k = 0; k < texs.size(); ++k) {
        const Tex& t = texs[k];
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib.image = t.img;
        ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, t.layers}; ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
        VkBufferImageCopy c{};
        c.bufferOffset = tex_offsets[k]; c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, t.layers}; c.imageExtent = {t.w, t.h, t.d};
        vkCmdCopyBufferToImage(cmd, tex_stage, t.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    }
    if (!set->flush(cmd)) {
        std::fprintf(stderr, "FAILED set->flush\n");
        return 1;
    }
    if (!set->fallbacks_ready()) set->record_fallback_init(cmd);
    std::vector<VkClearValue> clears(attachments);
    for (auto& c : clears) for (float& f : c.color.float32) f = -1;
    if (depth) clears[targets].depthStencil = {depth_clear, 0};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp; rbi.framebuffer = fb; rbi.renderArea = sc; rbi.clearValueCount = attachments; rbi.pClearValues = clears.data();
    LOADD(vkCreateQueryPool); LOADD(vkDestroyQueryPool); LOADD(vkGetQueryPoolResults);
    LOADD(vkCmdResetQueryPool); LOADD(vkCmdBeginQuery); LOADD(vkCmdEndQuery);
    VkQueryPool query = VK_NULL_HANDLE;
    if (occlusion) {
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_OCCLUSION;
        info.queryCount = 1;
        CHECK(vkCreateQueryPool(dev, &info, nullptr, &query));
        vkCmdResetQueryPool(cmd, query, 0, 1);
    }
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    const VkDescriptorSet ds = set->set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pl, pcs, 0, pc_bytes, pc);
    if (occlusion) vkCmdBeginQuery(cmd, query, 0, VK_QUERY_CONTROL_PRECISE_BIT);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    if (occlusion) vkCmdEndQuery(cmd, query, 0);
    vkCmdEndRenderPass(cmd);
    for (uint32_t t = 0; t < attachments; ++t) {
        VkBufferImageCopy reg{};
        reg.bufferOffset = target_bytes * t;
        reg.imageSubresource = {VkImageAspectFlags(t == targets ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
        reg.imageExtent = {W, H, 1};
        vkCmdCopyImageToBuffer(cmd, imgs[t], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &reg);
    }
    CHECK(vkEndCommandBuffer(cmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHECK(vkCreateFence(dev, &fci, nullptr, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(q, 1, &si, fence));
    const VkResult waited = vkWaitForFences(dev, 1, &fence, VK_TRUE, timeout_ns);
    if (waited == VK_TIMEOUT) {
        std::fprintf(stderr, "TIMEOUT\n");
        return 3;
    }
    CHECK(waited);
    if (occlusion) {
        uint64_t samples = 0;
        CHECK(vkGetQueryPoolResults(dev, query, 0, 1, sizeof(samples), &samples, sizeof(samples), VK_QUERY_RESULT_64_BIT));
        vkDestroyQueryPool(dev, query, nullptr);
        std::printf("samples_passed %llu\n", static_cast<unsigned long long>(samples));
    }
    const double t_draw = now_s();
    void* p;
    CHECK(vkMapMemory(dev, bmem, 0, bi.size, 0, &p));
    std::ofstream output(argv[3], std::ios::binary);
    output.write(static_cast<const char*>(p), std::streamsize(color_bytes));
    if (depth) {
        std::ofstream plane(std::string(argv[3]) + ".depth", std::ios::binary);
        plane.write(static_cast<const char*>(p) + color_bytes, std::streamsize(bi.size - color_bytes));
        plane.close();
        if (!plane) return 1;
    }
    output.close();
    if (!output) {
        std::fprintf(stderr, "failed to write %s\n", argv[3]);
        return 1;
    }
    std::fprintf(stderr, "ok %ux%u x%u -> %s (device %.2fs, set %.2fs, pipeline %.2fs, draw %.2fs)\n", W, H, targets, argv[3],
                 t_device - t_start, t_set - t_device, t_pipe - t_set, t_draw - t_pipe);
    return 0;
}
