// VK_KHR_shader_fma for the test programs: translated shaders built with `rdna2spirv.py --fma khr` use OpFmaKHR
// (capability FMAKHR, SPV_KHR_fma), so the device must enable the extension and the shaderFmaFloat32 feature before
// the module can be created. A test program that creates a device calls query(), link_after() and extension() when
// supported(), and refuses a module that needs the capability on a device without it (exit 77, like every other
// missing device feature).
#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <utility>
#include <vector>

namespace leonida::vk_test {

#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
inline constexpr bool shader_fma_in_headers = true;
#else
inline constexpr bool shader_fma_in_headers = false;
#endif

inline constexpr std::uint32_t spv_sampled_image_array_dynamic_indexing = 29;
inline constexpr std::uint32_t spv_storage_image_array_dynamic_indexing = 31;

// Find OpCapability (17) in a binary module.
inline bool spirv_declares_capability(const void* data, std::size_t bytes, std::uint32_t capability) {
    if (bytes < 20 || bytes % 4) return false;
    std::vector<std::uint32_t> words(bytes / 4);
    std::memcpy(words.data(), data, bytes);
    for (std::size_t i = 5; i < words.size() && (words[i] >> 16);) {
        const std::uint32_t op = words[i] & 0xffff, count = words[i] >> 16;
        if (op == 17 && count == 2 && i + 1 < words.size() && words[i + 1] == capability) return true;
        i += count;
    }
    return false;
}

// The float widths that the module's OpFmaKHR (4427) results have: OpTypeFloat (22) gives each float type's width,
// OpTypeVector (23) carries its component's, and an OpFmaKHR's result type (word 1) is one of them. A module with
// Float64 but only an f32 OpFmaKHR needs only shaderFmaFloat32, so the width comes from the instruction itself.
inline void spirv_fma_widths(const void* data, std::size_t bytes, bool& f32, bool& f64) {
    f32 = f64 = false;
    if (bytes < 20 || bytes % 4) return;
    std::vector<std::uint32_t> words(bytes / 4);
    std::memcpy(words.data(), data, bytes);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> width;   // float type id -> width (a few entries)
    const auto width_of = [&](std::uint32_t id) -> std::uint32_t {
        for (const auto& [t, w] : width) if (t == id) return w;
        return 0;
    };
    for (std::size_t i = 5; i < words.size() && (words[i] >> 16);) {
        const std::uint32_t op = words[i] & 0xffff, count = words[i] >> 16;
        if (i + count > words.size()) break;
        if (op == 22 && count >= 3) width.emplace_back(words[i + 1], words[i + 2]);
        else if (op == 23 && count >= 4 && width_of(words[i + 2])) width.emplace_back(words[i + 1], width_of(words[i + 2]));
        else if (op == 4427 && count >= 3) (width_of(words[i + 1]) == 64 ? f64 : f32) = true;
        i += count;
    }
}

struct ShaderFma {
#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
    VkPhysicalDeviceShaderFmaFeaturesKHR features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR};
#endif
    bool float32 = false;  // the device lists the extension and reports shaderFmaFloat32
    bool float64 = false;  // ... and shaderFmaFloat64 (OpFmaKHR on doubles: rdna2spirv.py --fma khr, v_div_fmas_f64)

    bool supported() const { return float32 || float64; }

    // Asks the physical device. Enables exactly what it reports (shaderFmaFloat16/32/64), so features never
    // fail device creation.
    void query(PFN_vkGetInstanceProcAddr gipa, VkInstance inst, VkPhysicalDevice pd) {
#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
        const auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
            gipa(inst, "vkEnumerateDeviceExtensionProperties"));
        const auto features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
            gipa(inst, "vkGetPhysicalDeviceFeatures2"));
        if (!enumerate || !features2) return;
        std::uint32_t count = 0;
        enumerate(pd, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> extensions(count);
        enumerate(pd, nullptr, &count, extensions.data());
        bool listed = false;
        for (const auto& e : extensions)
            if (!std::strcmp(e.extensionName, VK_KHR_SHADER_FMA_EXTENSION_NAME)) listed = true;
        if (!listed) return;
        features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &features};
        features2(pd, &query);
        features.pNext = nullptr;
        float32 = features.shaderFmaFloat32 == VK_TRUE;
        float64 = features.shaderFmaFloat64 == VK_TRUE;
#else
        (void)gipa; (void)inst; (void)pd;
#endif
    }

    // Puts the features structure into the device pNext chain right after `base` (any structure in the chain).
    void link_after(void* base) {
#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
        if (!supported()) return;
        auto* b = static_cast<VkBaseOutStructure*>(base);
        features.pNext = b->pNext;
        b->pNext = reinterpret_cast<VkBaseOutStructure*>(&features);
#else
        (void)base;
#endif
    }

    // The extension name to add to ppEnabledExtensionNames when supported().
    static const char* extension() {
#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
        return VK_KHR_SHADER_FMA_EXTENSION_NAME;
#else
        return "VK_KHR_shader_fma";
#endif
    }

    // True when the module can be used: it does not need FMAKHR, or the device enables the shaderFmaFloat32 /
    // shaderFmaFloat64 that its OpFmaKHR result widths need. Otherwise says so and the caller returns 77.
    bool accepts(const void* code, std::size_t bytes, const char* name) const {
        if ((float32 && float64) || !spirv_declares_capability(code, bytes, 6030)) return true;  // FMAKHR
        bool need32 = false, need64 = false;
        spirv_fma_widths(code, bytes, need32, need64);
        if (!need32 && !need64) need32 = true;   // FMAKHR declared but no OpFmaKHR found: keep the old requirement
        const char* missing = need32 && !float32 ? "shaderFmaFloat32" : need64 && !float64 ? "shaderFmaFloat64" : nullptr;
        if (!missing) return true;
        std::fprintf(stderr, "unsupported: %s needs OpFmaKHR (VK_KHR_shader_fma, %s) and the device lacks it; "
                             "translate with --fma glsl\n", name, missing);
        return false;
    }

    // accepts() for every stage of a pipeline: a test that draws a translated pixel shader with a translated vertex
    // shader must refuse the pair when EITHER stage needs FMAKHR, because the device then fails pipeline creation
    // instead of skipping. A file that cannot be read is accepted here; the caller's own module loading reports it.
    bool accepts_file(const char* path) const {
        if ((float32 && float64) || !path || !*path) return true;
        std::ifstream in(path, std::ios::binary);
        if (!in) return true;
        const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return accepts(bytes.data(), bytes.size(), path);
    }
};

}  // namespace leonida::vk_test
