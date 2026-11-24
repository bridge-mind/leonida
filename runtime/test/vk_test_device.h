// Shared setup for the GPU tests of the D3D12 layer: a Vulkan 1.3 device (or 1.2 with VK_KHR_dynamic_rendering and
// VK_KHR_synchronization2) with dynamicRendering, synchronization2, timelineSemaphore and
// separateDepthStencilLayouts, images, buffers and one-shot submissions.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; Device::validation_errors() counts errors.
#pragma once

#include "vulkan_loader.h"
#include "../src/queue_families.h"
#include "../src/vulkan_device_filter.h"
#include "../src/vulkan13_features.h"
#include "vk_device_pick.h"
#include "../src/subgroup_size.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace vktest {

inline unsigned& validation_error_count() {
    static unsigned count = 0;
    return count;
}

inline unsigned& known_validation_error_count() {
    static unsigned count = 0;
    return count;
}

// The NVIDIA validation layer checks an sRGB swapchain as if it also had VK_IMAGE_USAGE_STORAGE_BIT, which the
// presenter never requests and sRGB swapchain formats cannot support on that driver. Its VUID 01778 covers the
// first message and the max-layers/max-extent follow-ons, so match the VUID rather than only the first text. The
// presenter's own usage is checked against supportedUsageFlags before creation, so this VUID is the known false
// positive here. It is counted separately so a test can skip only when it is the sole validation error.
inline bool known_srgb_swapchain_storage_error(const VkDebugUtilsMessengerCallbackDataEXT* data) {
    return data && data->pMessage && data->messageIdNumber == 0xc036022f;
}

// Only validation-type errors fail a test. General messages come from the loader and the layers around the
// application: a third-party implicit layer that fails to load (an overlay or capture tool, for example) reports a
// general error that says nothing about this code, so it is printed but not counted.
inline VKAPI_ATTR VkBool32 VKAPI_CALL on_validation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                    VkDebugUtilsMessageTypeFlagsEXT types,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    const bool validation = (types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) != 0;
    const bool known = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) && validation &&
                       known_srgb_swapchain_storage_error(data);
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) && validation) {
        ++validation_error_count();
        if (known) ++known_validation_error_count();
    }
    std::fprintf(stderr, "%s%s: %s\n", validation ? "validation" : "vulkan (not counted)",
                 known ? " (known sRGB swapchain check)" : "",
                 data && data->pMessage ? data->pMessage : "(no message)");
    return VK_FALSE;
}

enum class DeviceStatus { ready, unavailable, failed };

template<typename Entry, typename Enumerate>
VkResult enumerate_all(std::vector<Entry>& entries, Enumerate enumerate) {
    VkResult result;
    do {
        std::uint32_t count = 0;
        result = enumerate(&count, nullptr);
        if (result != VK_SUCCESS) return result;
        entries.resize(count);
        if (count == 0) return VK_SUCCESS;
        result = enumerate(&count, entries.data());
        if (result == VK_SUCCESS) entries.resize(count);
    } while (result == VK_INCOMPLETE);
    return result;
}

class Device {
public:
    PFN_vkGetInstanceProcAddr instance_proc = nullptr;
    PFN_vkGetDeviceProcAddr device_proc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    // Set before initialize(): how many queues of `family` to create (fewer if the family has fewer). `queues` holds the
    // handles; queues[0] == queue. A test that wants distinct VkQueues for its D3D12 queues passes them on as
    // VulkanDeviceInfo::command_queues when there is more than one.
    std::uint32_t queue_count_wanted = 1;
    std::uint32_t family_queue_count = 1;  // the family's queueCount, known after the device is chosen
    std::vector<VkQueue> queues;
    bool validation = false;
    bool bc = false;  // textureCompressionBC enabled
    bool all_queue_types = false;
    VkQueue compute_queue = VK_NULL_HANDLE, copy_queue = VK_NULL_HANDLE;
    std::uint32_t compute_family = 0, copy_family = 0;
    std::vector<VkQueueFamilyProperties> families;
    VkPhysicalDeviceFeatures enabled{};  // the core features enabled on `device`
    bool bindless = false;         // the features the bindless set 0 needs are enabled
    bool partially_bound = false;  // and its slots can stay empty
    bool image_view_min_lod = false;
    bool sampler_filter_minmax = false;
    bool vertex_attribute_divisor = false;
    bool storage_buffer_8bit = false; // storageBuffer8BitAccess enabled
    bool conditional_rendering = false;  // VK_EXT_conditional_rendering enabled, where available
    bool host_image_copy = false;  // VK_EXT_host_image_copy and hostImageCopy enabled
    // create(true) only: VK_KHR (else EXT) swapchain_maintenance1 and its feature enabled, with the surface half on the
    // instance, so a vsync swapchain can switch to IMMEDIATE or FIFO_RELAXED per present.
    bool swapchain_maintenance1 = false;
    bool shader_fma = false;       // VK_KHR_shader_fma and shaderFmaFloat32 enabled (OpFmaKHR, rdna2spirv.py --fma khr)
    bool occlusion_query_precise = false;    // occlusionQueryPrecise enabled
    bool pipeline_statistics_query = false;  // pipelineStatisticsQuery enabled
    bool request_debug_utils = false;  // set before create(): enable VK_EXT_debug_utils when the loader has it
    bool request_buffer_device_address = false;  // set before create(): enable bufferDeviceAddress where supported
    bool buffer_device_address = false;  // enabled: buffer() memory and buffers can be addressed (device_address())
    bool conservative_rasterization = false; // VK_EXT_conservative_rasterization enabled where available
    bool request_mesh_shader = false;  // set before create(): enable VK_EXT_mesh_shader (meshShader) where available
    bool mesh_shader = false;          // VK_EXT_mesh_shader and meshShader are enabled
    bool debug_utils = false;          // VK_EXT_debug_utils is enabled on the instance
    // The subgroup sizes of the device, with subgroupSizeControl and computeFullSubgroups enabled where it has them.
    leonida::SubgroupSizeCaps subgroup;
    bool custom_border_color = false;  // VK_EXT_custom_border_color and customBorderColors enabled
    bool custom_border_color_without_format = false;

    Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    ~Device() {
        if (device) {
            fn<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(device);
            const auto destroy_image = fn<PFN_vkDestroyImage>("vkDestroyImage");
            const auto destroy_buffer = fn<PFN_vkDestroyBuffer>("vkDestroyBuffer");
            const auto free_memory = fn<PFN_vkFreeMemory>("vkFreeMemory");
            for (auto [image, memory] : images_) {
                destroy_image(device, image, nullptr);
                free_memory(device, memory, nullptr);
            }
            for (auto [buffer, memory] : buffers_) {
                destroy_buffer(device, buffer, nullptr);
                free_memory(device, memory, nullptr);
            }
            if (pool_) fn<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(device, pool_, nullptr);
            fn<PFN_vkDestroyDevice>("vkDestroyDevice")(device, nullptr);
        }
        if (messenger_) {
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                instance_proc(instance, "vkDestroyDebugUtilsMessengerEXT"))(instance, messenger_, nullptr);
        }
        if (instance)
            reinterpret_cast<PFN_vkDestroyInstance>(instance_proc(instance, "vkDestroyInstance"))(instance, nullptr);
    }

    // A Vulkan 1.3 function falls back to its KHR name on a 1.2 device.
    template<typename F>
    F fn(const char* name) const {
        return reinterpret_cast<F>(leonida::device_proc_or_khr(device_proc, device, name));
    }

    template<typename F>
    F instance_fn(const char* name) const {
        return reinterpret_cast<F>(instance_proc(instance, name));
    }

    unsigned validation_errors() const { return validation_error_count(); }
    unsigned known_validation_errors() const { return known_validation_error_count(); }

    // presentation: also enable VK_KHR_surface + VK_KHR_win32_surface and VK_KHR_swapchain (Windows only).
    bool create(bool presentation = false) { return initialize(presentation) == DeviceStatus::ready; }

    DeviceStatus initialize(bool presentation = false) {
        const auto failure = [](const char* call, VkResult result) {
            std::fprintf(stderr, "%s failed (VkResult %d)\n", call, static_cast<int>(result));
            return DeviceStatus::failed;
        };
        instance_proc = load_vulkan();
        if (!instance_proc) return DeviceStatus::unavailable;
        const auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(instance_proc(nullptr, "vkCreateInstance"));
        const auto enumerate_layers = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
            instance_proc(nullptr, "vkEnumerateInstanceLayerProperties"));

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read once, at startup
#endif
        const char* env = std::getenv("LEONIDA_VK_VALIDATION");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        validation = env && std::strcmp(env, "0") != 0;
        const char* layer = "VK_LAYER_KHRONOS_validation";
        const char* debug_utils_name = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        VkDebugUtilsMessengerCreateInfoEXT messenger_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        messenger_info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        messenger_info.pfnUserCallback = on_validation;

        const VkValidationFeatureEnableEXT sync_validation =
            VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
        VkValidationFeaturesEXT validation_features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT, &messenger_info};
        validation_features.enabledValidationFeatureCount = 1;
        validation_features.pEnabledValidationFeatures = &sync_validation;

        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instance_info.pApplicationInfo = &app;
        std::vector<const char*> instance_extensions;
        bool surface_m1_khr = false, surface_m1_ext = false;
        if (presentation) {
            instance_extensions.push_back("VK_KHR_surface");
            instance_extensions.push_back("VK_KHR_win32_surface");
            const auto enumerate_instance = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
                instance_proc(nullptr, "vkEnumerateInstanceExtensionProperties"));
            std::uint32_t count = 0;
            enumerate_instance(nullptr, &count, nullptr);
            std::vector<VkExtensionProperties> available(count);
            enumerate_instance(nullptr, &count, available.data());
            bool caps2 = false;
            for (const auto& e : available)  // lists the HDR10 (ST.2084) surface formats, when the display has them
                if (std::strcmp(e.extensionName, "VK_EXT_swapchain_colorspace") == 0)
                    instance_extensions.push_back("VK_EXT_swapchain_colorspace");
            for (const auto& e : available) {
                caps2 = caps2 || std::strcmp(e.extensionName, "VK_KHR_get_surface_capabilities2") == 0;
                surface_m1_khr = surface_m1_khr || std::strcmp(e.extensionName, "VK_KHR_surface_maintenance1") == 0;
                surface_m1_ext = surface_m1_ext || std::strcmp(e.extensionName, "VK_EXT_surface_maintenance1") == 0;
            }
            if (caps2 && (surface_m1_khr || surface_m1_ext)) {
                instance_extensions.push_back("VK_KHR_get_surface_capabilities2");
                if (surface_m1_khr) instance_extensions.push_back("VK_KHR_surface_maintenance1");
                if (surface_m1_ext) instance_extensions.push_back("VK_EXT_surface_maintenance1");
            } else {
                surface_m1_khr = surface_m1_ext = false;
            }
        }
        if (validation) {
            std::vector<VkLayerProperties> layers;
            const VkResult layer_result = enumerate_all(layers, enumerate_layers);
            if (layer_result != VK_SUCCESS) return failure("vkEnumerateInstanceLayerProperties", layer_result);
            bool found = false;
            for (const auto& properties : layers) found = found || std::strcmp(properties.layerName, layer) == 0;
            if (!found) {
                std::fprintf(stderr, "LEONIDA_VK_VALIDATION is set but %s is not installed\n", layer);
                return DeviceStatus::failed;
            }
            instance_info.pNext = &validation_features;
            instance_info.enabledLayerCount = 1;
            instance_info.ppEnabledLayerNames = &layer;
            debug_utils = true;  // the validation layer provides it
        } else if (request_debug_utils) {
            const auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
                instance_proc(nullptr, "vkEnumerateInstanceExtensionProperties"));
            std::uint32_t count = 0;
            enumerate(nullptr, &count, nullptr);
            std::vector<VkExtensionProperties> extensions(count);
            enumerate(nullptr, &count, extensions.data());
            for (const auto& e : extensions)
                debug_utils = debug_utils || std::strcmp(e.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0;
        }
        if (debug_utils) {
            instance_extensions.push_back(debug_utils_name);
        }
        instance_info.enabledExtensionCount = std::uint32_t(instance_extensions.size());
        instance_info.ppEnabledExtensionNames = instance_extensions.data();
        VkResult result = create_instance(&instance_info, nullptr, &instance);
        if (result == VK_ERROR_INCOMPATIBLE_DRIVER) return DeviceStatus::unavailable;
        if (result != VK_SUCCESS) return failure("vkCreateInstance", result);
        if (validation) {
            const auto create_messenger = instance_fn<PFN_vkCreateDebugUtilsMessengerEXT>("vkCreateDebugUtilsMessengerEXT");
            if (!create_messenger) return failure("vkCreateDebugUtilsMessengerEXT", VK_ERROR_EXTENSION_NOT_PRESENT);
            result = create_messenger(instance, &messenger_info, nullptr, &messenger_);
            if (result != VK_SUCCESS) return failure("vkCreateDebugUtilsMessengerEXT", result);
        }

        const auto enumerate = instance_fn<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
        const auto properties_of = instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
        const auto families_of =
            instance_fn<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
        std::vector<VkPhysicalDevice> devices;
        result = enumerate_all(devices, [&](std::uint32_t* count, VkPhysicalDevice* values) {
            return enumerate(instance, count, values);
        });
        if (result != VK_SUCCESS) return failure("vkEnumeratePhysicalDevices", result);
        VkPhysicalDeviceProperties chosen{};
        int best_rank = -1;
        const auto features2_of = instance_fn<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2");
        const auto device_extensions_of =
            instance_fn<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
        leonida::Vulkan13Features candidate13;
        for (auto candidate : devices) {
            VkPhysicalDeviceProperties props{};
            properties_of(candidate, &props);
            if (!leonida::vulkan_device_wanted(props.deviceName)) continue;
            candidate13.query(candidate, props.apiVersion, features2_of, device_extensions_of);
            if (!candidate13.usable()) continue;
            std::uint32_t family_count = 0;
            families_of(candidate, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> candidate_families(family_count);
            families_of(candidate, &family_count, candidate_families.data());
            for (std::uint32_t i = 0; i < family_count; ++i) {
                if (!(candidate_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
                if (const int rank = leonida::vk_test::device_rank(props.deviceType, props.vendorID);
                    rank > best_rank) {
                    physical = candidate;
                    family = i;
                    family_queue_count = candidate_families[i].queueCount;
                    chosen = props;
                    best_rank = rank;
                    v13_.choose(candidate13);
                }
                break;
            }
        }
        if (!physical) {
            if (const std::string filter = leonida::vulkan_device_filter(); !filter.empty()) {
                // A filter that matches nothing fails the test instead of skipping it.
                std::fprintf(stderr,
                             "LEONIDA_VK_DEVICE=%s: no Vulkan 1.3 device (or 1.2 with VK_KHR_dynamic_rendering and "
                             "VK_KHR_synchronization2) with a graphics queue matches\n",
                             filter.c_str());
                return DeviceStatus::failed;
            }
            std::fprintf(stderr, "no Vulkan 1.3 device (or 1.2 with VK_KHR_dynamic_rendering and "
                                 "VK_KHR_synchronization2) with a graphics queue\n");
            return DeviceStatus::unavailable;
        }
        std::printf("device: %s%s%s\n", chosen.deviceName, v13_.core ? "" : " (Vulkan 1.2 + KHR extensions)",
                    validation ? " (validation on)" : "");

        VkPhysicalDeviceVulkan11Features features11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &features11};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &features12};
        features2_of(physical, &features);
        if (!features12.timelineSemaphore || !features12.separateDepthStencilLayouts ||
            !features12.bufferDeviceAddress) {
            std::fprintf(stderr, "the device lacks dynamicRendering, synchronization2, timelineSemaphore or "
                                 "separateDepthStencilLayouts\n");
            return DeviceStatus::unavailable;
        }
        // VK_EXT_conditional_rendering (SetPredication), where available.
        {
            const auto extensions_of =
                instance_fn<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
            std::uint32_t extension_count = 0;
            extensions_of(physical, nullptr, &extension_count, nullptr);
            std::vector<VkExtensionProperties> extensions(extension_count);
            extensions_of(physical, nullptr, &extension_count, extensions.data());
            for (const auto& e : extensions)
                if (!std::strcmp(e.extensionName, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME)) {
                    VkPhysicalDeviceConditionalRenderingFeaturesEXT supported{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
                    VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &supported};
                    instance_fn<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(physical, &query);
                    conditional_rendering = supported.conditionalRendering == VK_TRUE;
                }
        }
        VkPhysicalDeviceConditionalRenderingFeaturesEXT enable_conditional{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
        enable_conditional.conditionalRendering = VK_TRUE;

        leonida::query_subgroup_size_caps(instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2"),
                                          physical, v13_.subgroup_size_control, v13_.compute_full_subgroups,
                                          subgroup);
        std::printf("subgroup sizes: %u..%u (default %u), requirable in stages 0x%x, subgroupSizeControl %s, "
                    "computeFullSubgroups %s\n",
                    subgroup.min_size, subgroup.max_size, subgroup.default_size, unsigned(subgroup.required_stages),
                    subgroup.subgroup_size_control ? "on" : "off", subgroup.compute_full_subgroups ? "on" : "off");
        // shaderDrawParameters, where available: translated vertex shaders will read BaseVertex and BaseInstance once PR 436
        // (zero-based VS ids) is in; harmless before.
        VkPhysicalDeviceVulkan11Features enable11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                                                  v13_.enable(conditional_rendering ? &enable_conditional : nullptr)};
        enable11.shaderDrawParameters = features11.shaderDrawParameters;
        VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &enable11};
        enable12.timelineSemaphore = VK_TRUE;
        enable12.separateDepthStencilLayouts = VK_TRUE;
        buffer_device_address = request_buffer_device_address && features12.bufferDeviceAddress == VK_TRUE;
        enable12.bufferDeviceAddress = buffer_device_address;
        storage_buffer_8bit = features12.storageBuffer8BitAccess == VK_TRUE;
        enable12.storageBuffer8BitAccess = storage_buffer_8bit;
        // What the bindless set 0 of the translated shaders needs, where the device has it: runtime arrays of
        // images indexed dynamically, updated while the set is bound; and, to leave slots empty, partially bound
        // bindings.
        bindless = features12.runtimeDescriptorArray && features.features.shaderSampledImageArrayDynamicIndexing &&
                   features.features.shaderStorageImageArrayDynamicIndexing &&
                   features12.descriptorBindingSampledImageUpdateAfterBind &&
                   features12.descriptorBindingStorageImageUpdateAfterBind &&
                   features12.descriptorBindingStorageBufferUpdateAfterBind;
        partially_bound = bindless && features12.descriptorBindingPartiallyBound;
        enable12.runtimeDescriptorArray = bindless;
        enable12.descriptorBindingSampledImageUpdateAfterBind = bindless;
        enable12.descriptorBindingStorageImageUpdateAfterBind = bindless;
        enable12.descriptorBindingStorageBufferUpdateAfterBind = bindless;
        enable12.descriptorBindingPartiallyBound = partially_bound;
        sampler_filter_minmax = features12.samplerFilterMinmax == VK_TRUE;
        enable12.samplerFilterMinmax = sampler_filter_minmax;
        enable12.bufferDeviceAddress = VK_TRUE;

        const std::uint32_t graphics_count = std::max(1u, std::min(queue_count_wanted, family_queue_count));
        const std::vector<float> priority(graphics_count, 1.0f);
        {
            std::uint32_t family_count = 0;
            families_of(physical, &family_count, nullptr);
            families.resize(family_count);
            families_of(physical, &family_count, families.data());
        }
        compute_family = copy_family = family;
        if (all_queue_types) {
            const auto picked = leonida::pick_queue_families(families, family);
            compute_family = picked.compute;
            copy_family = picked.copy;
        }
        std::vector<VkDeviceQueueCreateInfo> queue_infos;
        for (std::uint32_t f : {family, compute_family, copy_family}) {
            bool seen = false;
            for (const auto& q : queue_infos) seen = seen || q.queueFamilyIndex == f;
            if (seen) continue;
            VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            q.queueFamilyIndex = f;
            q.queueCount = f == family ? graphics_count : 1;
            q.pQueuePriorities = priority.data();
            queue_infos.push_back(q);
            sharing_.push_back(f);
        }
        VkPhysicalDeviceFeatures enable{};
        enable.textureCompressionBC = features.features.textureCompressionBC;  // BC uploads, where available
        enable.shaderInt64 = features.features.shaderInt64;  // translated shaders declare Int64 (index features)
        bc = features.features.textureCompressionBC == VK_TRUE;
        enable.shaderSampledImageArrayDynamicIndexing = bindless;
        enable.shaderStorageImageArrayDynamicIndexing = bindless;
        enable.logicOp = features.features.logicOp;  // d3d12_draw_test's logic-op draw, where available
        enable.geometryShader = features.features.geometryShader;  // d3d12_geometry_test, where available
        enable.shaderInt64 = features.features.shaderInt64;  // translated shaders declare Int64 for their addresses
        const char* conditional_extension = VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME;

        // VK_EXT_host_image_copy where the device has it, for the D3D12 layer's CPU texture copies.
        const auto enumerate_extensions =
            instance_fn<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
        std::uint32_t extension_count = 0;
        enumerate_extensions(physical, nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> extensions(extension_count);
        enumerate_extensions(physical, nullptr, &extension_count, extensions.data());
        VkPhysicalDeviceImageViewMinLodFeaturesEXT min_lod{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT};
        if (std::any_of(extensions.begin(), extensions.end(), [](const auto& e) {
                return std::strcmp(e.extensionName, VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME) == 0;
            })) {
            VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &min_lod};
            instance_fn<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(physical, &query);
            image_view_min_lod = min_lod.minLod == VK_TRUE;
            if (image_view_min_lod) {
                min_lod.pNext = *v13_.next();
                *v13_.next() = &min_lod;
            }
        }
        const char* divisor_extension = nullptr;
        for (const char* name :
             {VK_KHR_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME, VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME}) {
            if (std::any_of(extensions.begin(), extensions.end(),
                            [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; })) {
                divisor_extension = name;
                break;
            }
        }
        VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR divisor_khr{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR};
        VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT divisor_ext{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT};
        if (divisor_extension) {
            const bool is_khr = std::strcmp(divisor_extension, VK_KHR_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME) == 0;
            VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                            is_khr ? static_cast<void*>(&divisor_khr) : static_cast<void*>(&divisor_ext)};
            instance_fn<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(physical, &query);
            vertex_attribute_divisor = (is_khr ? divisor_khr.vertexAttributeInstanceRateDivisor
                                               : divisor_ext.vertexAttributeInstanceRateDivisor) == VK_TRUE;
            if (vertex_attribute_divisor) {
                if (is_khr) {
                    divisor_khr.pNext = *v13_.next();
                    *v13_.next() = &divisor_khr;
                } else {
                    divisor_ext.pNext = *v13_.next();
                    *v13_.next() = &divisor_ext;
                }
            }
        }
        VkPhysicalDeviceHostImageCopyFeaturesEXT host_copy{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT};
        for (const auto& extension : extensions) {
            if (std::strcmp(extension.extensionName, VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME) != 0) continue;
            VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &host_copy};
            instance_fn<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(physical, &query);
            host_image_copy = host_copy.hostImageCopy == VK_TRUE;
        }
        const char* host_image_copy_name = VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME;
        if (host_image_copy) {
            host_copy.pNext = *v13_.next();  // after the conditional rendering features, if any
            *v13_.next() = &host_copy;
        }

#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
        // VK_KHR_shader_fma where the device has it: translated shaders built with --fma khr use OpFmaKHR.
        VkPhysicalDeviceShaderFmaFeaturesKHR fma_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR};
        if (std::any_of(extensions.begin(), extensions.end(), [](const auto& e) {
                return std::strcmp(e.extensionName, VK_KHR_SHADER_FMA_EXTENSION_NAME) == 0;
            })) {
            VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &fma_features};
            instance_fn<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(physical, &query);
            shader_fma = fma_features.shaderFmaFloat32 == VK_TRUE;
            if (shader_fma) {
                fma_features.pNext = *v13_.next();
                *v13_.next() = &fma_features;
            }
        }
#endif
        // VK_EXT_mesh_shader (meshShader only), for the tests that ask for it.
        VkPhysicalDeviceMeshShaderFeaturesEXT mesh_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        if (request_mesh_shader && std::any_of(extensions.begin(), extensions.end(), [](const auto& e) {
                return std::strcmp(e.extensionName, VK_EXT_MESH_SHADER_EXTENSION_NAME) == 0;
            })) {
            VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &mesh_features};
            instance_fn<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(physical, &query);
            mesh_shader = mesh_features.meshShader == VK_TRUE;
        }
        if (mesh_shader) {
            mesh_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, *v13_.next()};
            mesh_features.meshShader = VK_TRUE;
            *v13_.next() = &mesh_features;
        }
        enable.occlusionQueryPrecise = features.features.occlusionQueryPrecise;  // query heaps, where available
        enable.pipelineStatisticsQuery = features.features.pipelineStatisticsQuery;
        occlusion_query_precise = features.features.occlusionQueryPrecise == VK_TRUE;
        pipeline_statistics_query = features.features.pipelineStatisticsQuery == VK_TRUE;

        // VK_EXT_custom_border_color where the device has it.
        VkPhysicalDeviceCustomBorderColorFeaturesEXT custom_border{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
        for (const auto& extension : extensions) {
            if (std::strcmp(extension.extensionName, VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME) != 0) continue;
            VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &custom_border};
            instance_fn<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(physical, &query);
            custom_border_color = custom_border.customBorderColors == VK_TRUE;
            custom_border_color_without_format = custom_border.customBorderColorWithoutFormat == VK_TRUE;
        }
        const char* custom_border_color_name = VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME;
        if (custom_border_color) {
            custom_border.pNext = *v13_.next();   // where later feature structs go since #1706
            *v13_.next() = &custom_border;
        }

        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enable12};
        device_info.queueCreateInfoCount = std::uint32_t(queue_infos.size());
        device_info.pQueueCreateInfos = queue_infos.data();
        device_info.pEnabledFeatures = &enable;
        enabled = enable;
        std::vector<const char*> device_extensions;
        v13_.add_extensions(device_extensions);
        conservative_rasterization = std::any_of(extensions.begin(), extensions.end(), [](const auto& e) {
            return std::strcmp(e.extensionName, VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME) == 0;
        });
        if (conservative_rasterization) device_extensions.push_back(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
        if (image_view_min_lod) device_extensions.push_back(VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME);
        if (vertex_attribute_divisor) device_extensions.push_back(divisor_extension);
        if (conditional_rendering) device_extensions.push_back(conditional_extension);
        if (host_image_copy) device_extensions.push_back(host_image_copy_name);
        const char* swapchain = "VK_KHR_swapchain";
        if (presentation) device_extensions.push_back(swapchain);
#if defined(VK_KHR_swapchain_maintenance1)
        VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR swapchain_m1{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR};
        if (presentation) {
            const auto has = [&](const char* name) {
                return std::any_of(extensions.begin(), extensions.end(),
                                   [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
            };
            const char* name = surface_m1_khr && has("VK_KHR_swapchain_maintenance1") ? "VK_KHR_swapchain_maintenance1"
                               : surface_m1_ext && has("VK_EXT_swapchain_maintenance1") ? "VK_EXT_swapchain_maintenance1"
                                                                                       : nullptr;
            if (name) {
                VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &swapchain_m1};
                instance_fn<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(physical, &query);
                if (swapchain_m1.swapchainMaintenance1 == VK_TRUE) {
                    swapchain_m1.pNext = *v13_.next();
                    *v13_.next() = &swapchain_m1;
                    device_extensions.push_back(name);
                    swapchain_maintenance1 = true;
                }
            }
        }
#endif
#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
        if (shader_fma) device_extensions.push_back(VK_KHR_SHADER_FMA_EXTENSION_NAME);
#endif
        if (custom_border_color) device_extensions.push_back(custom_border_color_name);
        if (mesh_shader) device_extensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
        device_info.enabledExtensionCount = std::uint32_t(device_extensions.size());
        device_info.ppEnabledExtensionNames = device_extensions.data();
        result = instance_fn<PFN_vkCreateDevice>("vkCreateDevice")(physical, &device_info, nullptr, &device);
        if (result != VK_SUCCESS) return failure("vkCreateDevice", result);
        device_proc = instance_fn<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
        queues.resize(graphics_count);
        for (std::uint32_t i = 0; i < graphics_count; ++i)
            fn<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(device, family, i, &queues[i]);
        queue = queues[0];
        fn<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(device, compute_family, 0, &compute_queue);
        fn<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(device, copy_family, 0, &copy_queue);
        instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(physical,
                                                                                                    &memory_);
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = family;
        result = fn<PFN_vkCreateCommandPool>("vkCreateCommandPool")(device, &pool, nullptr, &pool_);
        return result == VK_SUCCESS ? DeviceStatus::ready : failure("vkCreateCommandPool", result);
    }

    bool depth_format_usable(VkFormat format) const {
        VkFormatProperties properties{};
        instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(physical, format,
                                                                                                    &properties);
        constexpr VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                                VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        return (properties.optimalTilingFeatures & needed) == needed;
    }

    // depth > 1: a 3D image. samples > 1: a multisampled one.
    VkImage image(VkFormat format, std::uint32_t width, std::uint32_t height, std::uint32_t layers,
                  VkImageUsageFlags usage, std::uint32_t depth = 1, std::uint32_t samples = 1) {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {width, height, depth};
        info.mipLevels = 1;
        info.arrayLayers = layers;
        info.samples = VkSampleCountFlagBits(samples);
        info.usage = usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        share(info);
        VkImage image = VK_NULL_HANDLE;
        if (fn<PFN_vkCreateImage>("vkCreateImage")(device, &info, nullptr, &image) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkMemoryRequirements requirements{};
        fn<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(device, image, &requirements);
        const VkDeviceMemory memory = allocate(requirements, 0);
        images_.push_back({image, memory});
        if (!memory || fn<PFN_vkBindImageMemory>("vkBindImageMemory")(device, image, memory, 0) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        return image;
    }

    // A host-visible, coherent buffer, mapped; with `mapped` null, a device-local one the host can't see.
    VkBuffer buffer(VkDeviceSize size, VkBufferUsageFlags usage, void** mapped) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage | (buffer_device_address ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0u);
        share(info);
        VkBuffer buffer = VK_NULL_HANDLE;
        if (fn<PFN_vkCreateBuffer>("vkCreateBuffer")(device, &info, nullptr, &buffer) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        VkMemoryRequirements requirements{};
        fn<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(device, buffer, &requirements);
        const VkDeviceMemory memory = allocate(requirements, mapped ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                                    : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        buffers_.push_back({buffer, memory});
        if (!memory || fn<PFN_vkBindBufferMemory>("vkBindBufferMemory")(device, buffer, memory, 0) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        if (!mapped) return buffer;
        if (fn<PFN_vkMapMemory>("vkMapMemory")(device, memory, 0, size, 0, mapped) != VK_SUCCESS) return VK_NULL_HANDLE;
        std::memset(*mapped, 0xcd, size_t(size));
        return buffer;
    }

    // The device address of a buffer from buffer(); 0 unless buffer_device_address is enabled.
    std::uint64_t device_address(VkBuffer buffer) const {
        if (!buffer_device_address) return 0;
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = buffer;
        return fn<PFN_vkGetBufferDeviceAddress>("vkGetBufferDeviceAddress")(device, &info);
    }

    // Records `body` into a one-shot command buffer, submits it and waits for the queue.
    template<typename Body>
    bool run(Body&& body) {
        VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        info.commandPool = pool_;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        if (fn<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(device, &info, &command_buffer) != VK_SUCCESS)
            return false;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        fn<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(command_buffer, &begin);
        const bool recorded = body(command_buffer);
        fn<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(command_buffer);
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;
        const bool ok = recorded && fn<PFN_vkQueueSubmit>("vkQueueSubmit")(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS &&
                        fn<PFN_vkQueueWaitIdle>("vkQueueWaitIdle")(queue) == VK_SUCCESS;
        fn<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(device, pool_, 1, &command_buffer);
        return ok;
    }

    // UNDEFINED -> layout for a whole image: what resource creation does for a D3D12 initial state.
    void initial_layout(VkCommandBuffer command_buffer, VkImage image, VkImageAspectFlags aspects,
                        std::uint32_t layers, VkImageLayout layout) const {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {aspects, 0, 1, 0, layers};
        fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                                                             nullptr, 1, &barrier);
    }

    // Copies one aspect of layers [0, layers) from an image in `layout` to buffer + offset, then makes it host-visible.
    void read_back(VkCommandBuffer command_buffer, VkImage image, VkImageLayout layout, VkImageAspectFlags aspect,
                   std::uint32_t width, std::uint32_t height, std::uint32_t layers, VkBuffer buffer,
                   VkDeviceSize offset) const {
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = {aspect, 0, 0, layers};
        copy.imageExtent = {width, height, 1};
        fn<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(command_buffer, image, layout, buffer, 1, &copy);
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0,
                                                             nullptr);
    }

private:
    std::vector<std::uint32_t> sharing_;

    template<typename Info>
    void share(Info& info) const {
        if (sharing_.size() < 2) return;
        info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = std::uint32_t(sharing_.size());
        info.pQueueFamilyIndices = sharing_.data();
    }

    VkDeviceMemory allocate(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags flags) {
        for (std::uint32_t i = 0; i < memory_.memoryTypeCount; ++i) {
            if (!(requirements.memoryTypeBits & (1u << i))) continue;
            if ((memory_.memoryTypes[i].propertyFlags & flags) != flags) continue;
            VkMemoryAllocateFlagsInfo address_flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            address_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            if (buffer_device_address) info.pNext = &address_flags;
            info.allocationSize = requirements.size;
            info.memoryTypeIndex = i;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            if (fn<PFN_vkAllocateMemory>("vkAllocateMemory")(device, &info, nullptr, &memory) == VK_SUCCESS)
                return memory;
            return VK_NULL_HANDLE;
        }
        return VK_NULL_HANDLE;
    }

    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_{};
    leonida::Vulkan13Features v13_;  // the chosen device's dynamicRendering/synchronization2 path
    std::vector<std::pair<VkImage, VkDeviceMemory>> images_;
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> buffers_;
};

} // namespace vktest
