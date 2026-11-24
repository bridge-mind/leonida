#pragma once

// dynamicRendering and synchronization2, the two Vulkan 1.3 features the D3D12 layer needs, on a Vulkan 1.3 device
// or on a Vulkan 1.2 device through VK_KHR_dynamic_rendering and VK_KHR_synchronization2. CrossOver's MoltenVK on
// macOS is such a 1.2 device (1.2.10: Vulkan 1.2 with both extensions); the extension structs, functions and
// behaviour are the core ones under another name, so the rest of the layer does not see the difference.
// Graphics pipelines also use the 1.3 core dynamic states (primitive topology, primitive restart, vertex binding
// stride); a 1.2 device gives them through VK_EXT_extended_dynamic_state and VK_EXT_extended_dynamic_state2, enabled
// where the device has both (extended_dynamic_state). Without them the device still works for compute.
//
//   leonida::Vulkan13Features v13;
//   v13.query(physical, props.apiVersion, get_features2, enumerate_extensions);   // per candidate
//   if (!v13.usable()) continue;
//   enable12.pNext = v13.enable(chain);       // chain: what came after VkPhysicalDeviceVulkan13Features
//   ... v13.next() is where later feature structs go (it used to be enable13.pNext)
//   v13.add_extensions(device_extensions);
//   fn = leonida::device_proc_or_khr(get_device_proc, device, "vkCmdPipelineBarrier2");

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace leonida {

class Vulkan13Features {
public:
    Vulkan13Features() = default;
    Vulkan13Features(const Vulkan13Features&) = delete;  // enable() links pointers into this object
    Vulkan13Features& operator=(const Vulkan13Features&) = delete;

    bool core = false;               // a Vulkan 1.3 device: the features are core
    bool dynamic_rendering = false;  // supported, as a core feature or through VK_KHR_dynamic_rendering
    bool synchronization2 = false;   // supported, as a core feature or through VK_KHR_synchronization2
    // Optional 1.3 core features, enabled where supported; on a 1.2 device they stay off (no required subgroup size).
    bool subgroup_size_control = false;
    bool compute_full_subgroups = false;
    bool extended_dynamic_state = false;  // core on 1.3; on 1.2 both VK_EXT_extended_dynamic_state(2) extensions

    bool usable() const { return dynamic_rendering && synchronization2; }

    // Fills the flags above for one physical device. A device older than 1.2 never qualifies.
    void query(VkPhysicalDevice physical, std::uint32_t api_version, PFN_vkGetPhysicalDeviceFeatures2 features2,
               PFN_vkEnumerateDeviceExtensionProperties enumerate_extensions) {
        core = api_version >= VK_API_VERSION_1_3;
        dynamic_rendering = synchronization2 = subgroup_size_control = compute_full_subgroups = false;
        extended_dynamic_state = false;
        if (!features2) return;
        if (core) {
            VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
            features2(physical, &query);
            dynamic_rendering = f13.dynamicRendering == VK_TRUE;
            synchronization2 = f13.synchronization2 == VK_TRUE;
            subgroup_size_control = f13.subgroupSizeControl == VK_TRUE;
            compute_full_subgroups = f13.computeFullSubgroups == VK_TRUE;
            extended_dynamic_state = true;  // core in 1.3, no feature bit
            return;
        }
        if (api_version < VK_API_VERSION_1_2 || !enumerate_extensions) return;
        std::uint32_t count = 0;
        enumerate_extensions(physical, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> extensions(count);
        enumerate_extensions(physical, nullptr, &count, extensions.data());
        bool has_dynamic_rendering = false, has_synchronization2 = false, has_eds = false, has_eds2 = false;
        for (const auto& e : extensions) {
            has_dynamic_rendering |= std::strcmp(e.extensionName, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) == 0;
            has_synchronization2 |= std::strcmp(e.extensionName, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0;
            has_eds |= std::strcmp(e.extensionName, VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME) == 0;
            has_eds2 |= std::strcmp(e.extensionName, VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME) == 0;
        }
        if (!has_dynamic_rendering || !has_synchronization2) return;
        VkPhysicalDeviceExtendedDynamicState2FeaturesEXT eds2{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT};
        VkPhysicalDeviceExtendedDynamicStateFeaturesEXT eds{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
                                                            has_eds2 ? &eds2 : nullptr};
        VkPhysicalDeviceSynchronization2FeaturesKHR sync2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
                                                          has_eds ? &eds : nullptr};
        VkPhysicalDeviceDynamicRenderingFeaturesKHR rendering{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR, &sync2};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &rendering};
        features2(physical, &query);
        dynamic_rendering = rendering.dynamicRendering == VK_TRUE;
        synchronization2 = sync2.synchronization2 == VK_TRUE;
        extended_dynamic_state = has_eds && has_eds2 && eds.extendedDynamicState == VK_TRUE &&
                                 eds2.extendedDynamicState2 == VK_TRUE;
    }

    // Copies the flags of the device query() chose (a later query() for another candidate overwrites them).
    void choose(const Vulkan13Features& from) {
        core = from.core;
        dynamic_rendering = from.dynamic_rendering;
        synchronization2 = from.synchronization2;
        subgroup_size_control = from.subgroup_size_control;
        compute_full_subgroups = from.compute_full_subgroups;
        extended_dynamic_state = from.extended_dynamic_state;
    }

    // The structs that enable both features, ahead of `chain`; returns their head for VkPhysicalDeviceVulkan12Features'
    // pNext. Call once, after query() or choose().
    void* enable(void* chain) {
        if (core) {
            enable13_.pNext = chain;
            enable13_.dynamicRendering = VK_TRUE;
            enable13_.synchronization2 = VK_TRUE;
            enable13_.subgroupSizeControl = subgroup_size_control ? VK_TRUE : VK_FALSE;
            enable13_.computeFullSubgroups = compute_full_subgroups ? VK_TRUE : VK_FALSE;
            return &enable13_;
        }
        sync2_.pNext = chain;
        sync2_.synchronization2 = VK_TRUE;
        if (extended_dynamic_state) {
            eds2_.pNext = chain;
            eds2_.extendedDynamicState2 = VK_TRUE;
            eds_.pNext = &eds2_;
            eds_.extendedDynamicState = VK_TRUE;
            sync2_.pNext = &eds_;
        }
        rendering_.pNext = &sync2_;
        rendering_.dynamicRendering = VK_TRUE;
        return &rendering_;
    }

    // Where a feature struct goes to follow these: insert with `s.pNext = *next(); *next() = &s;`.
    void** next() { return core ? &enable13_.pNext : extended_dynamic_state ? &eds2_.pNext : &sync2_.pNext; }

    // The device extensions a 1.2 device needs for both features; none on 1.3.
    void add_extensions(std::vector<const char*>& extensions) const {
        if (core) return;
        extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
        extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
        if (!extended_dynamic_state) return;
        extensions.push_back(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);
        extensions.push_back(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME);
    }

private:
    VkPhysicalDeviceVulkan13Features enable13_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceDynamicRenderingFeaturesKHR rendering_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
    VkPhysicalDeviceSynchronization2FeaturesKHR sync2_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR};
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT eds_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT};
    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT eds2_{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT};
};

// A device function by its core name, or else by its KHR or EXT name: a Vulkan 1.2 device returns null for Vulkan
// 1.3 core names (vkCmdPipelineBarrier2, vkCmdBeginRendering, vkCmdSetPrimitiveTopology, ...) and serves the enabled
// extension's vkCmdPipelineBarrier2KHR, vkCmdSetPrimitiveTopologyEXT and the like, which have the same signatures.
inline PFN_vkVoidFunction device_proc_or_khr(PFN_vkGetDeviceProcAddr get, VkDevice device, const char* name) {
    if (const PFN_vkVoidFunction function = get(device, name)) return function;
    if (const PFN_vkVoidFunction function = get(device, (std::string(name) + "KHR").c_str())) return function;
    return get(device, (std::string(name) + "EXT").c_str());
}

} // namespace leonida
