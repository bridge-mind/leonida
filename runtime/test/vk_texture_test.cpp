// GPU test for translated texture-sampling pixel shaders, using the translator's runtime contract:
//   push constants: s0..s31 user data, then the 64-bit heap base (host BDA of the Xbox GPU VA window)
//   set 0: binding 0 = 2D images[], 1 = samplers[], 2 = descriptor hash table (storage buffer)
// The test puts a T# (8 dwords) at Xbox VA `tsharp_va` inside the heap, points user SGPR s[ptr_sgpr] at it,
// registers that T# -> image index 1 and the shader's literal S# -> sampler index 1, binds an 8x8 gradient at
// index 1 and a magenta fallback at index 0, draws a full-screen triangle and dumps the RGBA32F result.
//
//   vk_texture_test <vs.spv> <fs.spv> <out.raw> <ptr_sgpr> <s0> <s1> <s2> <s3> [--two-mip|--two-mip-sint|--two-mip-uint]   (S# words as hex)
//   --two-mip: expose a 128x128 quadrant texture and its solid cyan mip 1 to the 64x64 draw; the SINT and UINT
//   forms bind the same image as a signed or unsigned integer view instead of the default UNORM one.
//   vk_texture_test cube [out.raw]: ps_cube_sample against two 8x8 cubes, a 12-layer 2D array at binding 3 (the
//   translator's view of a cube array); every pixel must be the texel a cube sample of its direction reads.
//   vk_texture_test dref [out.raw]: ps_sample_c_lz against an 8x8 D32 texture at index 1 (texel x holds depth
//   (x + 0.5) / 8) and a LESS_OR_EQUAL compare sampler at index 1; both of its results, f32 and d16, must be the
//   1.0 or 0.0 the reference 0.5 gives against the texel each pixel reads.
//   vk_texture_test dref_a16 [out.raw]: the same with ps_sample_c_lz_a16 (16-bit coordinates).
//   vk_texture_test dref_c / dref_c_b [out.raw]: the same with ps_sample_c (implicit LOD) / ps_sample_c_b (LOD bias).
//   VKT_HEAP_MB=<n>: size of the heap in MiB (default 64 KiB)
//   VKT_NEED_MAINTENANCE8=1: the shader has a non-constant texel Offset outside gather (image_sample*_o, see
//   ps_texoff.asm); print "unsupported: ..." and exit 77 when VK_KHR_maintenance8 cannot be enabled.
// VK_KHR_maintenance8, shaderImageGatherExtended (a non-constant gather Offset) and shaderResourceMinLod (the MinLod
// clamp of image_sample_d_cl) are enabled when the device has them. A fragment shader that declares the MinLod
// capability on a device without shaderResourceMinLod prints "unsupported: ..." and exits 77.
#define VK_NO_PROTOTYPES
#include "vulkan_loader.h"
#include "vk_device_pick.h"
#include "vk_shader_fma.h"
#include "vk_pc_tail.h"
#include "spirv_capability.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

static PFN_vkGetInstanceProcAddr gipa;
#define CHECK(x)                                                                     \
    do {                                                                             \
        VkResult r_ = (x);                                                           \
        if (r_ != VK_SUCCESS) {                                                      \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, r_, __LINE__);   \
            return 1;                                                                \
        }                                                                            \
    } while (0)

static std::vector<uint32_t> read_spv(const char* p) {
    std::ifstream f(p, std::ios::binary);
    if (!f.good()) return {};
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), w.size() * 4);
    return w;
}

#include "../src/descriptor_table.h"
#include "../src/gfx10_descriptors.h"

static bool cube_texel_matches(const float* pixel, int tx, int ty, int face) {
    return std::fabs(pixel[0] - tx * 32 / 255.f) < 1e-3f &&
           std::fabs(pixel[1] - ty * 32 / 255.f) < 1e-3f &&
           std::fabs(pixel[2] - (6 + face) * 20 / 255.f) < 1e-3f &&
           std::fabs(pixel[3] - 1.0f) < 1e-3f;  // The cube upload stores opaque alpha.
}

static int cube_alpha_self_test() {
    float pixel[4] = {64.0f / 255.0f, 96.0f / 255.0f, 200.0f / 255.0f, 1.0f};
    if (!cube_texel_matches(pixel, 2, 3, 4)) {
        std::fprintf(stderr, "cube alpha self-test: rejected the expected RGBA texel\n");
        return 1;
    }
    for (const float alpha : {0.0f, std::numeric_limits<float>::quiet_NaN()}) {
        pixel[3] = alpha;
        if (cube_texel_matches(pixel, 2, 3, 4)) {
            std::fprintf(stderr, "cube alpha self-test: accepted incorrect alpha %g\n", alpha);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test-cube-alpha") == 0)
        return cube_alpha_self_test();
    const uint32_t W = 64, H = 64, TW = 8;
    const bool signed_mips = argc == 10 && std::strcmp(argv[9], "--two-mip-sint") == 0;
    const bool uint_mips = argc == 10 && std::strcmp(argv[9], "--two-mip-uint") == 0;
    const bool two_mip = signed_mips || uint_mips || (argc == 10 && std::strcmp(argv[9], "--two-mip") == 0);
    // With vs_ids spanning UV [0,1] over the viewport, adjacent fragments differ by two base texels.
    const uint32_t texture_side = two_mip ? 2 * W : TW;
    const uint32_t levels = two_mip ? 2 : 1;
    std::string vs_path, fs_path, out_path;
    int ptr_sgpr = 0;
    // Fixed shader only: fetch (0,0) is in bounds on both the fallback and the integer image.
    const bool fetch_uint = argc >= 2 && !std::strcmp(argv[1], "fetch-uint");
    const bool fetch_miss = argc >= 2 && !std::strcmp(argv[1], "fetch-miss");
    const bool fetch = fetch_uint || fetch_miss || (argc >= 2 && !std::strcmp(argv[1], "fetch-sint"));
    const bool cube = argc >= 2 && !std::strcmp(argv[1], "cube");
    // dref_c / dref_c_b: image_sample_c / image_sample_c_b (implicit LOD, bias), same texture and expected results.
    const char* dref_ps = nullptr;
    if (argc >= 2) {
        if (!std::strcmp(argv[1], "dref")) dref_ps = "ps_sample_c_lz.spv";
        else if (!std::strcmp(argv[1], "dref_a16")) dref_ps = "ps_sample_c_lz_a16.spv";
        else if (!std::strcmp(argv[1], "dref_c")) dref_ps = "ps_sample_c.spv";
        else if (!std::strcmp(argv[1], "dref_c_b")) dref_ps = "ps_sample_c_b.spv";
    }
    const bool dref = dref_ps != nullptr;
    bool autonomous = false;
    uint32_t ssharp[8] = {0x92, 0xc00000, 0x14500000, 0,
                          0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    if (!fetch && (argc == 9 || two_mip)) {
        vs_path = argv[1];
        fs_path = argv[2];
        out_path = argv[3];
        ptr_sgpr = std::atoi(argv[4]);
        for (int i = 0; i < 4; ++i) ssharp[i] = (uint32_t)strtoul(argv[5 + i], 0, 16);
    } else if (argc <= 2 || ((cube || dref || fetch) && argc == 3)) {
        // Autonomous mode using synthetic hand-written shaders (no game files required)
        autonomous = true;
        auto find_spv = [](const char* name) -> std::string {
            const char* candidates[] = {
                "runtime/build/shaders/spirv/",
                "runtime/test/shaders/spirv/",
                "build/shaders/spirv/",
                "shaders/spirv/",
                ""
            };
            for (const char* prefix : candidates) {
                std::string p = std::string(prefix) + name;
                std::ifstream f(p, std::ios::binary);
                if (f.good()) return p;
            }
            return std::string("runtime/build/shaders/spirv/") + name;
        };
        vs_path = find_spv("vs_ids.spv");
        fs_path = find_spv(fetch ? "ps_table_fetch.spv" : cube ? "ps_cube_sample.spv" : dref ? dref_ps : "ps_texture_sample.spv");
        const int out_arg = cube || dref || fetch ? 2 : 1;
        out_path = argc > out_arg ? argv[out_arg] : "runtime/build/vk_texture_test_out.raw";
        ptr_sgpr = 0;
        if (dref)
            ssharp[0] |= 3u << 12;              // the shader's S#: DEPTH_COMPARE_FUNC = LESSEQUAL
    } else {
        std::fprintf(stderr, "usage: vk_texture_test [cube|dref|dref_a16|dref_c|dref_c_b|fetch-uint|fetch-sint|fetch-miss] [out] | vs fs out ptr_sgpr s0 s1 s2 s3 [--two-mip|--two-mip-sint|--two-mip-uint]\n");
        return 2;
    }
    // Word 1 also selects the numeric type when the shader fetches through this key. Word 3 keeps BASE_LEVEL [15:12]
    // at 0, as the views here start at level 0: image_get_resinfo minifies a 2D array's layers from BASE_LEVEL.
    const VkFormat sampled_format = dref ? VK_FORMAT_D32_SFLOAT :
        (uint_mips || fetch_uint) ? VK_FORMAT_R8G8B8A8_UINT :
        (signed_mips || fetch) ? VK_FORMAT_R8G8B8A8_SINT : VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t tsharp[8] = {0x11111111, leonida::gfx10::format_from_vulkan(sampled_format) << 20, 0x33333333, 0x44440444,
                          0x55555555, 0x66666666, 0x77777777, 0x88888888};
    const uint32_t tsharp_va = 256;

    gipa = load_vulkan();
    if (!gipa) { std::fprintf(stderr, "the Vulkan loader was not found\n"); return 77; }
    VkInstance inst = VK_NULL_HANDLE;
#define LI(n) auto n = reinterpret_cast<PFN_##n>(gipa(inst, #n))
    LI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    const VkResult instance_result = vkCreateInstance(&ici, nullptr, &inst);
    if (instance_result == VK_ERROR_INCOMPATIBLE_DRIVER) return 77;  // no ICD, as on a GPU-less runner
    if (instance_result != VK_SUCCESS) {
        std::fprintf(stderr, "FAILED vkCreateInstance -> %d (line %d)\n", instance_result, __LINE__);
        return 1;
    }
    LI(vkEnumeratePhysicalDevices); LI(vkGetPhysicalDeviceProperties); LI(vkGetPhysicalDeviceQueueFamilyProperties);
    LI(vkGetPhysicalDeviceMemoryProperties); LI(vkCreateDevice); LI(vkGetDeviceProcAddr);
    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, nullptr));
    std::vector<VkPhysicalDevice> pds(n);
    if (n) CHECK(vkEnumeratePhysicalDevices(inst, &n, pds.data()));
    if (!n) {
        std::fprintf(stderr, "unsupported: no Vulkan physical device\n");
        return 77;  // a missing device is a skip in every texture-test mode
    }
    // LEONIDA_VK_DEVICE picks one GPU on a machine with several, as in the other test programs.
    std::vector<VkPhysicalDeviceProperties> candidates(n);
    for (uint32_t i = 0; i < n; ++i) vkGetPhysicalDeviceProperties(pds[i], &candidates[i]);
    const size_t chosen = leonida::vk_test::pick_device(candidates);
    if (chosen >= n) {
        leonida::vk_test::report_no_device();
        return 1;
    }
    VkPhysicalDevice pd = pds[chosen];
    VkPhysicalDeviceProperties properties = candidates[chosen];
    constexpr uint32_t push_words = 36;
    // LEONIDA_PC_TAIL_UBO=1 (rdna2spirv.py --pc-tail-ubo): s0..s31 only; the heap base is in a uniform block at set 1
    // (vk_pc_tail.h), so a 128-byte device runs these tests too.
    const bool pc_tail = leonida::vk_test::pc_tail_requested();
    const uint32_t push_bytes = pc_tail ? 128u : push_words * uint32_t(sizeof(uint32_t));
    if (properties.limits.maxPushConstantsSize < push_bytes) {
        std::fprintf(stderr, "unsupported: need %u push constant bytes, device has %u\n",
                     push_bytes, properties.limits.maxPushConstantsSize);
        return 77;   // a device limit, not a wrong result: every texture test skips (128-byte devices, D3D-14)
    }
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT)) ++qi;
    if (qi == qn) {
        std::fprintf(stderr, "unsupported: no graphics queue family\n");
        return 77;  // a missing queue is a skip in every texture-test mode
    }
    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    // Translated vertex shaders read BaseVertex / BaseInstance (DrawParameters) for the zero-based v0 / v3.
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.shaderDrawParameters = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.pNext = &f11;
    f12.bufferDeviceAddress = VK_TRUE; f12.runtimeDescriptorArray = VK_TRUE; f12.descriptorIndexing = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.pNext = &f12;
    // Texel offsets: the limits the shader's Offset operands must stay inside, and the two optional features.
    LI(vkEnumerateDeviceExtensionProperties); LI(vkGetPhysicalDeviceFeatures);
    VkPhysicalDeviceProperties pdp; vkGetPhysicalDeviceProperties(pd, &pdp);
    std::fprintf(stderr, "device: %s\ntexel offset limits: %d..%d, gather %d..%d\n", pdp.deviceName,
                 pdp.limits.minTexelOffset, int(pdp.limits.maxTexelOffset),
                 pdp.limits.minTexelGatherOffset, int(pdp.limits.maxTexelGatherOffset));
    VkPhysicalDeviceFeatures have; vkGetPhysicalDeviceFeatures(pd, &have);
    const auto fragment_code = read_spv(fs_path.c_str());
    using leonida::vk_test::spirv_declares_capability;
    const bool dynamic_sampled = spirv_declares_capability(fragment_code.data(), fragment_code.size() * 4,
        leonida::vk_test::spv_sampled_image_array_dynamic_indexing);
    if (dynamic_sampled && !have.shaderSampledImageArrayDynamicIndexing) {
        std::printf("unsupported: shaderSampledImageArrayDynamicIndexing\n");
        return 77;
    }
    f2.features.shaderSampledImageArrayDynamicIndexing = dynamic_sampled;
    f2.features.shaderImageGatherExtended = have.shaderImageGatherExtended;
    f2.features.shaderResourceMinLod = have.shaderResourceMinLod;
    // shaderInt64 only when a stage has 64-bit integers: SHADER-423's address pairs need none (TEST-468).
    const bool int64 = leonida::vk_test::needs_shader_int64(read_spv(vs_path.c_str())) ||
                       leonida::vk_test::needs_shader_int64(fragment_code);
    if (int64 && !have.shaderInt64) {
        std::printf("unsupported: shaderInt64 (a stage declares Int64)\n");
        return 77;
    }
    f2.features.shaderInt64 = int64 ? VK_TRUE : VK_FALSE;
    if (spirv_declares_capability(fragment_code.data(), fragment_code.size() * 4, 42) &&
        !have.shaderResourceMinLod) {   // 42 = MinLod
        std::printf("unsupported: shaderResourceMinLod (the shader clamps its LOD with MinLod)\n");
        return 77;
    }
    std::vector<const char*> enable;
    leonida::vk_test::ShaderFma fma;      // OpFmaKHR (rdna2spirv.py --fma khr), when the device has it
    fma.query(gipa, inst, pd);
    // A device without VK_KHR_shader_fma cannot create a pipeline from a stage that uses OpFmaKHR: skip the pair.
    if (!fma.accepts_file(vs_path.c_str()) || !fma.accepts_file(fs_path.c_str())) return 77;
    if (fma.supported()) { enable.push_back(fma.extension()); fma.link_after(&f2); }
    bool m8_on = false;
#ifdef VK_KHR_MAINTENANCE_8_EXTENSION_NAME   // the structure type is an enumerator: defined() never sees it
    // Older Vulkan headers (as on some CI images) do not declare maintenance8; the test is then unsupported.
    VkPhysicalDeviceMaintenance8FeaturesKHR m8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
    uint32_t en = 0; vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en); vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, exts.data());
    for (auto& x : exts) m8_on = m8_on || !std::strcmp(x.extensionName, VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
    if (m8_on) {
        enable.push_back(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
        m8.maintenance8 = VK_TRUE; f11.pNext = &m8;
        std::fprintf(stderr, "enabled %s\n", VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
    }
#endif
    if (std::getenv("VKT_NEED_MAINTENANCE8") && !m8_on) {
        std::printf("unsupported: VK_KHR_maintenance8 (non-constant texel Offset)\n");
        return 77;
    }
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(enable.size()); dci.ppEnabledExtensionNames = enable.data();
    VkDevice dev;
    VkResult created = vkCreateDevice(pd, &dci, nullptr, &dev);
    if (two_mip && created == VK_ERROR_FEATURE_NOT_PRESENT) {
        std::fprintf(stderr, "unsupported: device lacks a requested texture test feature\n");
        return 77;
    }
    CHECK(created);
#define LD(n) auto n = reinterpret_cast<PFN_##n>(vkGetDeviceProcAddr(dev, #n))
    LD(vkGetDeviceQueue); LD(vkCreateImage); LD(vkGetImageMemoryRequirements); LD(vkAllocateMemory);
    LD(vkBindImageMemory); LD(vkCreateImageView); LD(vkCreateRenderPass); LD(vkCreateFramebuffer);
    LD(vkCreateShaderModule); LD(vkCreatePipelineLayout); LD(vkCreateGraphicsPipelines); LD(vkCreateCommandPool);
    LD(vkAllocateCommandBuffers); LD(vkBeginCommandBuffer); LD(vkCmdBeginRenderPass); LD(vkCmdBindPipeline);
    LD(vkCmdDraw); LD(vkCmdEndRenderPass); LD(vkCmdCopyImageToBuffer); LD(vkCmdCopyBufferToImage);
    LD(vkEndCommandBuffer); LD(vkQueueSubmit); LD(vkQueueWaitIdle); LD(vkCreateBuffer);
    LD(vkGetBufferMemoryRequirements); LD(vkBindBufferMemory); LD(vkMapMemory); LD(vkGetBufferDeviceAddress);
    LD(vkCreateSampler); LD(vkCreateDescriptorSetLayout); LD(vkCreateDescriptorPool); LD(vkAllocateDescriptorSets);
    LD(vkUpdateDescriptorSets); LD(vkCmdBindDescriptorSets); LD(vkCmdPushConstants); LD(vkCmdPipelineBarrier);
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memtype = [&](uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0u;
    };
    auto make_buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* b, VkDeviceMemory* m, void** map) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = usage;
        VkResult r = vkCreateBuffer(dev, &bi, nullptr, b); if (r) return r;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, *b, &mr);
        VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.pNext = &fi; ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        r = vkAllocateMemory(dev, &ai, nullptr, m); if (r) return r;
        r = vkBindBufferMemory(dev, *b, *m, 0); if (r) return r;
        return vkMapMemory(dev, *m, 0, size, 0, map);
    };
    // Heap (the Xbox GPU VA window) with the T# at tsharp_va.
    VkBuffer heap; VkDeviceMemory heapm; void* heapp;
    // 64 KiB of zeros unless VKT_HEAP_MB says otherwise. Buffer loads are not bounds-checked against the V#, so a
    // shader that indexes with anything but the zeros it finds here reads past the end: a bigger heap keeps such
    // reads inside defined memory.
    const char* heap_mb = std::getenv("VKT_HEAP_MB");
    const VkDeviceSize heap_bytes = heap_mb && std::atoi(heap_mb) > 0 ? VkDeviceSize(std::atoi(heap_mb)) << 20 : 65536;
    CHECK(make_buffer(heap_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, &heap, &heapm, &heapp));
    std::memset(heapp, 0, size_t(heap_bytes));
    std::memcpy(static_cast<char*>(heapp) + tsharp_va, tsharp, 32);
    if (two_mip)  // The gather test loads its S# from the four words just after the T#.
        std::memcpy(static_cast<char*>(heapp) + tsharp_va + 32, ssharp, 16);
    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; bdai.buffer = heap;
    VkDeviceAddress heap_base = vkGetBufferDeviceAddress(dev, &bdai);
    // Descriptor hash table: 16 slots.
    leonida::DescriptorTable desc_table(16);
    // The heap already holds the original key. Change only the registered key for the miss control.
    if (fetch_miss) tsharp[7] ^= 1;
    desc_table.insert(tsharp, 1);
    desc_table.insert(ssharp, 1);
    const auto table = desc_table.build_gpu_buffer(true);
    VkBuffer tb; VkDeviceMemory tbm; void* tbp;
    CHECK(make_buffer(table.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tb, &tbm, &tbp));
    std::memcpy(tbp, table.data(), table.size() * 4);
    // Textures: index 0 = 1x1 magenta fallback, index 1 = 8x8 gradient (r = x/8, g = y/8, b = 0.5, a = 1), or in dref
    // mode an 8x8 D32 texture with depth (x + 0.5) / 8. Binding 3 (2D arrays) gets the same fallback and two 8x8 cubes
    // as 12 layers: layer l = (x/8, y/8, 20l/255, 1).
    auto aspect_of = [](VkFormat f) {
        return f == VK_FORMAT_D32_SFLOAT ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT)
                                         : VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT);
    };
    auto make_tex = [&](uint32_t w, uint32_t h, uint32_t mip_levels, uint32_t layers, VkImageViewType type,
                        VkFormat format, VkImage* img, VkImageView* view) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = format; ii.extent = {w, h, 1};
        ii.mipLevels = mip_levels; ii.arrayLayers = layers; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VkResult r = vkCreateImage(dev, &ii, nullptr, img); if (r) return r;
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, *img, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m; r = vkAllocateMemory(dev, &ai, nullptr, &m); if (r) return r;
        r = vkBindImageMemory(dev, *img, m, 0); if (r) return r;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = *img; vi.viewType = type; vi.format = ii.format;
        vi.subresourceRange = {aspect_of(format), 0, mip_levels, 0, layers};
        return vkCreateImageView(dev, &vi, nullptr, view);
    };
    const uint32_t magenta = 0xFFFF00FFu;
    // R8G8B8A8 in memory: red, green, blue, transparent yellow, then solid cyan in mip 1.
    const uint32_t quadrants[2][2] = {{0xFF0000FFu, 0xFF00FF00u}, {0xFFFF0000u, 0x0000FFFFu}};
    std::vector<uint32_t> texels(texture_side * texture_side + (two_mip ? (texture_side / 2) * (texture_side / 2) : 0), 0xFFFFFF00u);
    for (uint32_t y = 0; y < texture_side; y++) {
        for (uint32_t x = 0; x < texture_side; x++) {
            texels[y * texture_side + x] = two_mip ? quadrants[y >= texture_side / 2][x >= texture_side / 2]
                                         : (x * 32u) | ((y * 32u) << 8) | (128u << 16) | (255u << 24);
        }
    }
    if (fetch) texels[0] = 0x4df82282u;
    // The signed two-mip fixture uses 0 or -128 in each channel, including the alpha channel.
    if (signed_mips)
        for (auto& texel : texels) texel &= 0x80808080u;
    if (dref) {   // same size as the 8x8 gradient: one 32-bit word a texel
        for (uint32_t y = 0; y < TW; y++)
            for (uint32_t x = 0; x < TW; x++) {
                const float depth = (x + 0.5f) / TW;
                std::memcpy(&texels[y * TW + x], &depth, sizeof(depth));
            }
    }
    const uint32_t CUBE_LAYERS = 12;
    std::vector<uint32_t> cubes(CUBE_LAYERS * TW * TW);
    for (uint32_t y = 0; y < TW; y++)
        for (uint32_t x = 0; x < TW; x++) {
            for (uint32_t l = 0; l < CUBE_LAYERS; l++)
                cubes[(l * TW + y) * TW + x] = (x * 32u) | ((y * 32u) << 8) | ((l * 20u) << 16) | (255u << 24);
        }
    const VkFormat tfmt[4] = {VK_FORMAT_R8G8B8A8_UNORM, sampled_format,
                              VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    VkImage timg[4]; VkImageView tview[4];
    CHECK(make_tex(1, 1, 1, 1, VK_IMAGE_VIEW_TYPE_2D, tfmt[0], &timg[0], &tview[0]));
    CHECK(make_tex(texture_side, texture_side, levels, 1, VK_IMAGE_VIEW_TYPE_2D, tfmt[1], &timg[1], &tview[1]));
    CHECK(make_tex(1, 1, 1, 1, VK_IMAGE_VIEW_TYPE_2D_ARRAY, tfmt[2], &timg[2], &tview[2]));
    CHECK(make_tex(TW, TW, 1, CUBE_LAYERS, VK_IMAGE_VIEW_TYPE_2D_ARRAY, tfmt[3], &timg[3], &tview[3]));
    VkBuffer up; VkDeviceMemory upm; void* upp;
    const VkDeviceSize texture_offset = 256;
    const VkDeviceSize texture_bytes = texels.size() * sizeof(uint32_t);
    const VkDeviceSize cube_offset = texture_offset + texture_bytes;
    CHECK(make_buffer(cube_offset + cubes.size() * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &up, &upm, &upp));
    std::memcpy(upp, &magenta, sizeof(magenta));
    std::memcpy(static_cast<char*>(upp) + texture_offset, texels.data(), texture_bytes);
    std::memcpy(static_cast<char*>(upp) + cube_offset, cubes.data(), cubes.size() * sizeof(uint32_t));
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.maxLod = float(levels - 1);
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSamplerCreateInfo sci1 = sci;            // index 1: the S# the shader names
    if (two_mip) {
        const leonida::gfx10::SamplerWords words{ssharp[0], ssharp[1], ssharp[2], ssharp[3]};
        const auto sampler = leonida::gfx10::decode(words);
        sci1.minLod = sampler.min_lod;
        sci1.maxLod = sampler.max_lod;
    }
    if (dref) {
        sci1.compareEnable = VK_TRUE;
        sci1.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    }
    VkSampler smp[2]; CHECK(vkCreateSampler(dev, &sci, nullptr, &smp[0])); CHECK(vkCreateSampler(dev, &sci1, nullptr, &smp[1]));
    // Render target.
    const VkFormat fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkImage rt; VkImageView rtv;
    {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt; ii.extent = {W, H, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        CHECK(vkCreateImage(dev, &ii, nullptr, &rt));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, rt, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m; CHECK(vkAllocateMemory(dev, &ai, nullptr, &m)); vkBindImageMemory(dev, rt, m, 0);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = rt; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        CHECK(vkCreateImageView(dev, &vi, nullptr, &rtv));
    }
    VkAttachmentDescription ad{}; ad.format = fmt; ad.samples = VK_SAMPLE_COUNT_1_BIT;
    ad.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ad.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ad.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; ad.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    ad.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; ad.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference ar{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sd{}; sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; sd.colorAttachmentCount = 1; sd.pColorAttachments = &ar;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 1; rpi.pAttachments = &ad; rpi.subpassCount = 1; rpi.pSubpasses = &sd;
    VkSubpassDependency copy_dep{0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    rpi.dependencyCount = 1; rpi.pDependencies = &copy_dep;
    VkRenderPass rp; CHECK(vkCreateRenderPass(dev, &rpi, nullptr, &rp));
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = rp; fbi.attachmentCount = 1; fbi.pAttachments = &rtv; fbi.width = W; fbi.height = H; fbi.layers = 1;
    VkFramebuffer fb; CHECK(vkCreateFramebuffer(dev, &fbi, nullptr, &fb));
    // Descriptor set: binding 0 images[2], 1 samplers[2], 2 hash table, 3 2D-array images[2].
    VkDescriptorSetLayoutBinding b[4]{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER, 2, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = 4; dli.pBindings = b;
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
    VkDescriptorPoolSize ps[3]{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4}, {VK_DESCRIPTOR_TYPE_SAMPLER, 2}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 1; dpi.poolSizeCount = 3; dpi.pPoolSizes = ps;
    VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev, &dai, &ds));
    VkDescriptorImageInfo ii2[2]{{VK_NULL_HANDLE, tview[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                 {VK_NULL_HANDLE, tview[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    VkDescriptorImageInfo ai2[2]{{VK_NULL_HANDLE, tview[2], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                 {VK_NULL_HANDLE, tview[3], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    VkDescriptorImageInfo si2[2]{{smp[0]}, {smp[1]}};
    VkDescriptorBufferInfo tbi{tb, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr[4]{};
    for (auto& w : wr) { w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet = ds; }
    wr[0].dstBinding = 0; wr[0].descriptorCount = 2; wr[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; wr[0].pImageInfo = ii2;
    wr[1].dstBinding = 1; wr[1].descriptorCount = 2; wr[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; wr[1].pImageInfo = si2;
    wr[2].dstBinding = 2; wr[2].descriptorCount = 1; wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[2].pBufferInfo = &tbi;
    wr[3].dstBinding = 3; wr[3].descriptorCount = 2; wr[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; wr[3].pImageInfo = ai2;
    vkUpdateDescriptorSets(dev, 4, wr, 0, nullptr);
    // Pipeline.
    bool tail_block = false;   // a stage reads the --pc-tail-ubo block (declares set 1)
    auto module = [&](const char* p, VkShaderModule* m) {
        auto code = read_spv(p);
        if (code.empty()) {
            std::fprintf(stderr, "failed to read shader module: %s\n", p);
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        if (pc_tail && leonida::vk_test::spirv_declares_set(code.data(), code.size() * 4, leonida::vk_test::pc_tail_set))
            tail_block = true;
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = code.size() * 4; smi.pCode = code.data();
        return vkCreateShaderModule(dev, &smi, nullptr, m);
    };
    VkShaderModule vs, fs; CHECK(module(vs_path.c_str(), &vs)); CHECK(module(fs_path.c_str(), &fs));
    VkPipelineShaderStageCreateInfo st[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = vs; st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = fs; st[1].pName = "main";
    VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    leonida::vk_test::PcTail tail;
    if (tail_block && !tail.create(vkGetDeviceProcAddr, dev, mp, properties.limits.minUniformBufferOffsetAlignment,
                                   VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, 1)) {
        std::fprintf(stderr, "FAILED creating the --pc-tail-ubo block\n");
        return 1;
    }
    const VkDescriptorSetLayout set_layouts[] = {dsl, tail.set_layout};
    if (tail_block) { pli.setLayoutCount = 2; pli.pSetLayouts = set_layouts; }
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp{0, 0, float(W), float(H), 0, 1}; VkRect2D sc{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp; vps.scissorCount = 1; vps.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; cb.attachmentCount = 1; cb.pAttachments = &cba;
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = 2; gpi.pStages = st; gpi.pVertexInputState = &vin; gpi.pInputAssemblyState = &ia; gpi.pViewportState = &vps;
    gpi.pRasterizationState = &rs; gpi.pMultisampleState = &ms; gpi.pColorBlendState = &cb; gpi.layout = pl; gpi.renderPass = rp;
    VkPipeline pipe; CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipe));
    VkBuffer rb; VkDeviceMemory rbm; void* rbp;
    CHECK(make_buffer(VkDeviceSize(W) * H * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb, &rbm, &rbp));
    // Commands: upload textures, draw, read back.
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = qi;
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = pool; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CHECK(vkBeginCommandBuffer(cmd, &cbi));
    const VkDeviceSize up_offset[4] = {0, texture_offset, 0, cube_offset};
    const uint32_t up_size[4] = {1, texture_side, 1, TW}, up_layers[4] = {1, 1, 1, CUBE_LAYERS};
    for (int t = 0; t < 4; t++) {
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib.image = timg[t];
        const uint32_t mip_levels = t == 1 ? levels : 1;
        ib.subresourceRange = {aspect_of(tfmt[t]), 0, mip_levels, 0, up_layers[t]}; ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
        VkDeviceSize offset = up_offset[t];
        for (uint32_t level = 0; level < mip_levels; ++level) {
            const uint32_t side = up_size[t] >> level;
            VkBufferImageCopy c{}; c.bufferOffset = offset; c.imageSubresource = {aspect_of(tfmt[t]), level, 0, up_layers[t]};
            c.imageExtent = {side, side, 1};
            vkCmdCopyBufferToImage(cmd, up, timg[t], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
            offset += VkDeviceSize(side) * side * up_layers[t] * sizeof(uint32_t);
        }
        ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    }
    VkClearValue clear{}; for (int i = 0; i < 4; i++) clear.color.float32[i] = -1;
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp; rbi.framebuffer = fb; rbi.renderArea = sc; rbi.clearValueCount = 1; rbi.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, nullptr);
    uint32_t pc[push_words] = {};
    pc[ptr_sgpr] = tsharp_va;               // pointer to the T# (low dword; the shader forces the high dword to 0)
    pc[32] = uint32_t(heap_base); pc[33] = uint32_t(heap_base >> 32);
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, 0, push_bytes, pc);
    if (tail_block) {
        tail.write(0, pc, 4);
        const uint32_t offset = tail.offset(0);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, leonida::vk_test::pc_tail_set, 1, &tail.set,
                                1, &offset);
    }
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    VkBufferImageCopy rc{}; rc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; rc.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd, rt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &rc);
    VkBufferMemoryBarrier readback{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    readback.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    readback.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    readback.srcQueueFamilyIndex = readback.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readback.buffer = rb;
    readback.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &readback, 0, nullptr);
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CHECK(vkQueueWaitIdle(q));
    std::ofstream output(out_path, std::ios::binary);
    output.write(static_cast<const char*>(rbp), std::streamsize(W) * H * 16);
    output.close();
    if (!output) { std::fprintf(stderr, "failed to write %s\n", out_path.c_str()); return 1; }
    std::fprintf(stderr, "ok -> %s (heap base 0x%llx)\n", out_path.c_str(), (unsigned long long)heap_base);
    if (autonomous && cube) {
        // Expected: the texel a cube sample of the pixel centre's direction reads from the second cube (layer 6 + face).
        // Pixels on a face edge are skipped; the face chosen there depends on rounding.
        const float* pixels = static_cast<const float*>(rbp);
        int checked = 0, wrong = 0;
        for (uint32_t py = 0; py < H; py++)
            for (uint32_t px = 0; px < W; px++) {
                const float x = 2 * (px + 0.5f) / W - 1, y = 2 * (py + 0.5f) / H - 1, z = -0.6f;
                const float ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z);
                if (std::fabs(ax - ay) < 1e-3f || std::fabs(ax - az) < 1e-3f || std::fabs(ay - az) < 1e-3f) continue;
                int face;
                float sc, tc, ma;
                if (az >= ax && az >= ay) { face = z < 0 ? 5 : 4; sc = z < 0 ? -x : x; tc = -y; ma = az; }
                else if (ay >= ax) { face = y < 0 ? 3 : 2; sc = x; tc = y < 0 ? -z : z; ma = ay; }
                else { face = x < 0 ? 1 : 0; sc = x < 0 ? z : -z; tc = -y; ma = ax; }
                const float s = TW * 0.5f * (sc / ma + 1), t = TW * 0.5f * (tc / ma + 1);   // in texels
                const float* p = pixels + (py * W + px) * 4;
                bool ok = false;
                for (float ds : {-1e-3f, 1e-3f})
                    for (float dt : {-1e-3f, 1e-3f}) {
                        const int tx = std::min(int(TW) - 1, int(s + ds)), ty = std::min(int(TW) - 1, int(t + dt));
                        ok |= cube_texel_matches(p, tx, ty, face);
                    }
                checked++;
                wrong += !ok;
            }
        if (wrong) {
            std::fprintf(stderr, "vk_texture_test cube: %d of %d pixels FAILED\n", wrong, checked);
            return 1;
        }
        std::printf("vk_texture_test cube: PASSED (%d pixels, each the texel a cube sample of its direction reads)\n",
                    checked);
    }
    if (autonomous && dref) {
        // Expected: pixel column px reads texel x = floor(8u) at u = (px + 0.5) / 64 (never on a texel edge), whose
        // depth (x + 0.5) / 8 passes the reference 0.5 under LESS_OR_EQUAL from x = 4 on. Both results, the f32 one
        // and the d16 one widened back to f32, are exactly that 1.0 or 0.0.
        const float* pixels = static_cast<const float*>(rbp);
        int wrong_f32 = 0, wrong_d16 = 0;
        for (uint32_t py = 0; py < H; py++)
            for (uint32_t px = 0; px < W; px++) {
                const uint32_t tx = std::min(TW - 1, uint32_t((px + 0.5f) / W * TW));
                const float want = 0.5f <= (tx + 0.5f) / TW ? 1.0f : 0.0f;
                const float* p = pixels + (py * W + px) * 4;
                wrong_f32 += p[0] != want;
                wrong_d16 += p[1] != want;
            }
        if (wrong_f32 || wrong_d16) {
            std::fprintf(stderr, "vk_texture_test dref: FAILED, %d pixels wrong in the f32 result and %d in the d16 one "
                         "(of %u)\n", wrong_f32, wrong_d16, W * H);
            return 1;
        }
        std::printf("vk_texture_test dref: PASSED (%u pixels, f32 and d16 depth compares exact)\n", W * H);
    }
    if (fetch) {
        // The PS converts each raw VGPR word as a signed integer. On a miss magenta's
        // UNORM 1.0 channels therefore export float(0x3f800000).
        const float fallback_one = static_cast<float>(0x3f800000);
        const float expected[] = {
            fetch_miss ? fallback_one : fetch_uint ? 130.f : -126.f,
            fetch_miss ? 0.f : 34.f,
            fetch_miss ? fallback_one : fetch_uint ? 248.f : -8.f,
            fetch_miss ? fallback_one : 77.f
        };
        const auto* pixels = static_cast<const float*>(rbp);
        for (uint32_t i = 0; i < W * H * 4; ++i) {
            if (pixels[i] != expected[i % 4]) {
                std::fprintf(stderr, "%s pixel %u channel %u: got %g, expected %g\n",
                             argv[1], i / 4, i % 4, pixels[i], expected[i % 4]);
                return 1;
            }
        }
        std::printf("%s: %u exact pixels\n", argv[1], W * H);
    }
    if (autonomous && !cube && !dref && !fetch) {
        const float* pixels = static_cast<const float*>(rbp);
        // Sample center pixel (32, 32)
        int idx = (32 * W + 32) * 4;
        float r = pixels[idx + 0];
        float g = pixels[idx + 1];
        float b = pixels[idx + 2];
        float a = pixels[idx + 3];
        if (std::abs(b - (128.0f / 255.0f)) > 0.05f || std::abs(a - 1.0f) > 0.05f) {
            std::fprintf(stderr, "vk_texture_test: pixel check FAILED at (32,32): rgba=(%f, %f, %f, %f), expected gradient sample (~0.5, ~0.5, 0.5, 1.0)\n",
                         r, g, b, a);
            return 1;
        }
        std::printf("vk_texture_test: PASSED (verified center pixel rgba=(%.3f, %.3f, %.3f, %.3f) from gradient texture via descriptor table)\n",
                    r, g, b, a);
    }
    return 0;
}
