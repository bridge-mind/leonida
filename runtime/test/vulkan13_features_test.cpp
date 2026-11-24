// Unit tests for Vulkan13Features and device_proc_or_khr (runtime/src/vulkan13_features.h). CPU-only, no Vulkan
// instance and no GPU: synthetic callbacks exercise the 1.1 rejection, 1.2 KHR/EXT fallback, and 1.3 core paths.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan13_features.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using leonida::Vulkan13Features;
using leonida::device_proc_or_khr;

namespace {

struct FakeDevice {
    std::vector<const char*> extensions;
    VkBool32 dynamic_rendering = VK_FALSE;
    VkBool32 synchronization2 = VK_FALSE;
    VkBool32 subgroup_size_control = VK_FALSE;
    VkBool32 compute_full_subgroups = VK_FALSE;
    VkBool32 eds = VK_FALSE;
    VkBool32 eds2 = VK_FALSE;
    int features2_calls = 0;
    int enum_calls = 0;
    std::vector<VkStructureType> queried_types;
};

FakeDevice* g_device = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL fake_enumerate_extensions(
    VkPhysicalDevice, const char*, std::uint32_t* count, VkExtensionProperties* properties) {
    ++g_device->enum_calls;
    if (!properties) {
        *count = static_cast<std::uint32_t>(g_device->extensions.size());
        return VK_SUCCESS;
    }
    const std::uint32_t n = static_cast<std::uint32_t>(g_device->extensions.size());
    const std::uint32_t write = *count < n ? *count : n;
    for (std::uint32_t i = 0; i < write; ++i) {
        std::memset(&properties[i], 0, sizeof(properties[i]));
        const std::size_t len = std::min<std::size_t>(std::strlen(g_device->extensions[i]), VK_MAX_EXTENSION_NAME_SIZE - 1);
        std::memcpy(properties[i].extensionName, g_device->extensions[i], len);
    }
    *count = write;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_get_features2(VkPhysicalDevice, VkPhysicalDeviceFeatures2* features) {
    ++g_device->features2_calls;
    g_device->queried_types.clear();
    for (auto* node = static_cast<VkBaseOutStructure*>(features->pNext); node; node = node->pNext) {
        g_device->queried_types.push_back(node->sType);
        switch (node->sType) {
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: {
                auto* f = reinterpret_cast<VkPhysicalDeviceVulkan13Features*>(node);
                f->dynamicRendering = g_device->dynamic_rendering;
                f->synchronization2 = g_device->synchronization2;
                f->subgroupSizeControl = g_device->subgroup_size_control;
                f->computeFullSubgroups = g_device->compute_full_subgroups;
                break;
            }
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR:
                reinterpret_cast<VkPhysicalDeviceDynamicRenderingFeaturesKHR*>(node)->dynamicRendering =
                    g_device->dynamic_rendering;
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR:
                reinterpret_cast<VkPhysicalDeviceSynchronization2FeaturesKHR*>(node)->synchronization2 =
                    g_device->synchronization2;
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT:
                reinterpret_cast<VkPhysicalDeviceExtendedDynamicStateFeaturesEXT*>(node)->extendedDynamicState =
                    g_device->eds;
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT:
                reinterpret_cast<VkPhysicalDeviceExtendedDynamicState2FeaturesEXT*>(node)->extendedDynamicState2 =
                    g_device->eds2;
                break;
            default:
                break;
        }
    }
}

std::vector<VkStructureType> chain_types(const void* head) {
    std::vector<VkStructureType> types;
    for (const auto* node = static_cast<const VkBaseInStructure*>(head); node; node = node->pNext) {
        types.push_back(node->sType);
    }
    return types;
}

VKAPI_ATTR void VKAPI_CALL stub_core() {}
VKAPI_ATTR void VKAPI_CALL stub_khr() {}
VKAPI_ATTR void VKAPI_CALL stub_ext() {}

struct ProcLookup {
    const char* core_name = nullptr;
    const char* khr_name = nullptr;
    const char* ext_name = nullptr;
    std::vector<std::string> queries;
};

ProcLookup* g_proc = nullptr;

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fake_get_device_proc(VkDevice, const char* name) {
    g_proc->queries.emplace_back(name);
    if (g_proc->core_name && std::strcmp(name, g_proc->core_name) == 0) return stub_core;
    if (g_proc->khr_name && std::strcmp(name, g_proc->khr_name) == 0) return stub_khr;
    if (g_proc->ext_name && std::strcmp(name, g_proc->ext_name) == 0) return stub_ext;
    return nullptr;
}

}  // namespace

int main() {
    // Vulkan 1.3 core device: queries VkPhysicalDeviceVulkan13Features without enumerating extensions.
    {
        FakeDevice dev;
        dev.dynamic_rendering = VK_TRUE;
        dev.synchronization2 = VK_TRUE;
        dev.subgroup_size_control = VK_TRUE;
        dev.compute_full_subgroups = VK_FALSE;
        g_device = &dev;

        Vulkan13Features v13;
        v13.query(VK_NULL_HANDLE, VK_API_VERSION_1_3, fake_get_features2, nullptr);
        EXPECT(dev.features2_calls == 1 && dev.enum_calls == 0);
        EXPECT(dev.queried_types.size() == 1 &&
               dev.queried_types[0] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
        EXPECT(v13.core && v13.usable());
        EXPECT(v13.dynamic_rendering && v13.synchronization2);
        EXPECT(v13.subgroup_size_control && !v13.compute_full_subgroups);
        EXPECT(v13.extended_dynamic_state);

        VkBaseOutStructure tail{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, nullptr};
        void* head = v13.enable(&tail);
        const auto* enable13 = static_cast<const VkPhysicalDeviceVulkan13Features*>(head);
        EXPECT(enable13->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
        EXPECT(enable13->dynamicRendering == VK_TRUE && enable13->synchronization2 == VK_TRUE);
        EXPECT(enable13->subgroupSizeControl == VK_TRUE && enable13->computeFullSubgroups == VK_FALSE);
        EXPECT(enable13->pNext == &tail);

        // Inserting a later feature struct through next() places it between enable13 and tail.
        VkBaseOutStructure extra{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, nullptr};
        extra.pNext = static_cast<VkBaseOutStructure*>(*v13.next());
        *v13.next() = &extra;
        const auto types = chain_types(head);
        EXPECT(types.size() == 3);
        EXPECT(types[0] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
        EXPECT(types[1] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
        EXPECT(types[2] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);

        // Core 1.3 needs no extra extensions.
        std::vector<const char*> exts = {"VK_KHR_swapchain"};
        v13.add_extensions(exts);
        EXPECT(exts.size() == 1 && std::strcmp(exts[0], "VK_KHR_swapchain") == 0);
    }

    // Vulkan 1.3 missing either required feature is not usable; Vulkan 1.4 still takes the core path.
    {
        FakeDevice dev;
        dev.dynamic_rendering = VK_TRUE;
        dev.synchronization2 = VK_FALSE;
        g_device = &dev;

        Vulkan13Features v13;
        v13.query(VK_NULL_HANDLE, VK_MAKE_API_VERSION(0, 1, 4, 0), fake_get_features2, nullptr);
        EXPECT(v13.core && v13.dynamic_rendering && !v13.synchronization2 && !v13.usable());

        dev.dynamic_rendering = VK_FALSE;
        dev.synchronization2 = VK_TRUE;
        v13.query(VK_NULL_HANDLE, VK_API_VERSION_1_3, fake_get_features2, nullptr);
        EXPECT(v13.core && !v13.dynamic_rendering && v13.synchronization2 && !v13.usable());
    }

    // Vulkan 1.1 and null callbacks never qualify, and reset flags from an earlier query.
    {
        FakeDevice dev;
        dev.dynamic_rendering = VK_TRUE;
        dev.synchronization2 = VK_TRUE;
        dev.subgroup_size_control = VK_TRUE;
        g_device = &dev;

        Vulkan13Features v13;
        v13.query(VK_NULL_HANDLE, VK_API_VERSION_1_3, fake_get_features2, nullptr);
        EXPECT(v13.usable());

        dev.features2_calls = 0;
        v13.query(VK_NULL_HANDLE, VK_API_VERSION_1_1, fake_get_features2, fake_enumerate_extensions);
        EXPECT(!v13.core && !v13.usable() && !v13.extended_dynamic_state && !v13.subgroup_size_control);
        EXPECT(dev.features2_calls == 0 && dev.enum_calls == 0);

        v13.query(VK_NULL_HANDLE, VK_API_VERSION_1_3, nullptr, nullptr);
        EXPECT(v13.core && !v13.usable() && !v13.extended_dynamic_state);

        v13.query(VK_NULL_HANDLE, VK_API_VERSION_1_2, fake_get_features2, nullptr);
        EXPECT(!v13.core && !v13.usable() && dev.features2_calls == 0);
    }

    // Vulkan 1.2 missing a required KHR extension returns before querying features2.
    {
        FakeDevice dev;
        dev.extensions = {VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME};
        dev.dynamic_rendering = VK_TRUE;
        dev.synchronization2 = VK_TRUE;
        g_device = &dev;

        Vulkan13Features v13;
        v13.query(VK_NULL_HANDLE, VK_API_VERSION_1_2, fake_get_features2, fake_enumerate_extensions);
        EXPECT(!v13.core && !v13.usable());
        EXPECT(dev.enum_calls == 2 && dev.features2_calls == 0);
    }

    // Vulkan 1.2 with KHR dynamic_rendering + synchronization2, without extended_dynamic_state(2).
    {
        FakeDevice dev;
        dev.extensions = {VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME};
        dev.dynamic_rendering = VK_TRUE;
        dev.synchronization2 = VK_TRUE;
        g_device = &dev;

        Vulkan13Features v13;
        v13.query(VK_NULL_HANDLE, VK_API_VERSION_1_2, fake_get_features2, fake_enumerate_extensions);
        EXPECT(!v13.core && v13.usable() && !v13.extended_dynamic_state);
        EXPECT(!v13.subgroup_size_control && !v13.compute_full_subgroups);
        EXPECT(dev.queried_types.size() == 2);
        EXPECT(dev.queried_types[0] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR);
        EXPECT(dev.queried_types[1] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR);

        VkBaseOutStructure tail{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, nullptr};
        void* head = v13.enable(&tail);
        VkBaseOutStructure extra{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, nullptr};
        extra.pNext = static_cast<VkBaseOutStructure*>(*v13.next());
        *v13.next() = &extra;

        const auto types = chain_types(head);
        EXPECT(types.size() == 4);
        EXPECT(types[0] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR);
        EXPECT(types[1] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR);
        EXPECT(types[2] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
        EXPECT(types[3] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);

        const auto* rendering = static_cast<const VkPhysicalDeviceDynamicRenderingFeaturesKHR*>(head);
        const auto* sync2 = static_cast<const VkPhysicalDeviceSynchronization2FeaturesKHR*>(rendering->pNext);
        EXPECT(rendering->dynamicRendering == VK_TRUE);
        EXPECT(sync2->synchronization2 == VK_TRUE);

        std::vector<const char*> exts;
        v13.add_extensions(exts);
        EXPECT(exts.size() == 2);
        EXPECT(std::strcmp(exts[0], VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) == 0);
        EXPECT(std::strcmp(exts[1], VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0);

        // Extensions advertised, but each required feature bit cleared in turn after a successful query.
        dev.features2_calls = 0;
        dev.dynamic_rendering = VK_FALSE;
        dev.synchronization2 = VK_TRUE;
        v13.query(VK_NULL_HANDLE, VK_API_VERSION_1_2, fake_get_features2, fake_enumerate_extensions);
        EXPECT(dev.features2_calls == 1);
        EXPECT(!v13.core && !v13.dynamic_rendering && v13.synchronization2 && !v13.usable());

        dev.dynamic_rendering = VK_TRUE;
        dev.synchronization2 = VK_FALSE;
        v13.query(VK_NULL_HANDLE, VK_API_VERSION_1_2, fake_get_features2, fake_enumerate_extensions);
        EXPECT(dev.features2_calls == 2);
        EXPECT(!v13.core && v13.dynamic_rendering && !v13.synchronization2 && !v13.usable());
    }

    // Vulkan 1.2 with only EDS1 (missing EDS2), or both extensions with one feature bit false: EDS stays off.
    {
        FakeDevice dev;
        dev.extensions = {
            VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
            VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
            VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
        };
        dev.dynamic_rendering = VK_TRUE;
        dev.synchronization2 = VK_TRUE;
        dev.eds = VK_TRUE;
        dev.eds2 = VK_TRUE;
        g_device = &dev;

        Vulkan13Features v13;
        v13.query(VK_NULL_HANDLE, VK_API_VERSION_1_2, fake_get_features2, fake_enumerate_extensions);
        EXPECT(v13.usable() && !v13.extended_dynamic_state);
        EXPECT(dev.queried_types.size() == 3);
        EXPECT(dev.queried_types[2] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT);

        dev.extensions.push_back(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME);
        dev.eds2 = VK_FALSE;
        v13.query(VK_NULL_HANDLE, VK_API_VERSION_1_2, fake_get_features2, fake_enumerate_extensions);
        EXPECT(v13.usable() && !v13.extended_dynamic_state);
        EXPECT(dev.queried_types.size() == 4);
    }

    // Vulkan 1.2 with KHR + EDS1 + EDS2 all supported: enables all four structs and adds all four extensions.
    {
        FakeDevice dev;
        dev.extensions = {
            VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
            VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
            VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
            VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME,
        };
        dev.dynamic_rendering = VK_TRUE;
        dev.synchronization2 = VK_TRUE;
        dev.eds = VK_TRUE;
        dev.eds2 = VK_TRUE;
        g_device = &dev;

        Vulkan13Features candidate;
        candidate.query(VK_NULL_HANDLE, VK_API_VERSION_1_2, fake_get_features2, fake_enumerate_extensions);
        EXPECT(!candidate.core && candidate.usable() && candidate.extended_dynamic_state);

        // choose() preserves the winning device's flags even after candidate is overwritten.
        Vulkan13Features chosen;
        chosen.choose(candidate);
        candidate.query(VK_NULL_HANDLE, VK_API_VERSION_1_1, fake_get_features2, fake_enumerate_extensions);
        EXPECT(!candidate.usable());
        EXPECT(!chosen.core && chosen.usable() && chosen.extended_dynamic_state);

        VkBaseOutStructure tail{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, nullptr};
        void* head = chosen.enable(&tail);
        VkBaseOutStructure extra{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, nullptr};
        extra.pNext = static_cast<VkBaseOutStructure*>(*chosen.next());
        *chosen.next() = &extra;

        const auto types = chain_types(head);
        EXPECT(types.size() == 6);
        EXPECT(types[0] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR);
        EXPECT(types[1] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR);
        EXPECT(types[2] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT);
        EXPECT(types[3] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT);
        EXPECT(types[4] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
        EXPECT(types[5] == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);

        const auto* rendering = static_cast<const VkPhysicalDeviceDynamicRenderingFeaturesKHR*>(head);
        const auto* sync2 = static_cast<const VkPhysicalDeviceSynchronization2FeaturesKHR*>(rendering->pNext);
        const auto* eds = static_cast<const VkPhysicalDeviceExtendedDynamicStateFeaturesEXT*>(sync2->pNext);
        const auto* eds2 = static_cast<const VkPhysicalDeviceExtendedDynamicState2FeaturesEXT*>(eds->pNext);
        EXPECT(rendering->dynamicRendering == VK_TRUE);
        EXPECT(sync2->synchronization2 == VK_TRUE);
        EXPECT(eds->extendedDynamicState == VK_TRUE);
        EXPECT(eds2->extendedDynamicState2 == VK_TRUE);

        std::vector<const char*> exts;
        chosen.add_extensions(exts);
        EXPECT(exts.size() == 4);
        EXPECT(std::strcmp(exts[0], VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) == 0);
        EXPECT(std::strcmp(exts[1], VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0);
        EXPECT(std::strcmp(exts[2], VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME) == 0);
        EXPECT(std::strcmp(exts[3], VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME) == 0);
    }

    // choose() with a Vulkan 1.3 candidate preserves core=true and subgroup flags after source overwrite.
    {
        FakeDevice dev;
        dev.dynamic_rendering = VK_TRUE;
        dev.synchronization2 = VK_TRUE;
        dev.subgroup_size_control = VK_TRUE;
        dev.compute_full_subgroups = VK_TRUE;
        g_device = &dev;

        Vulkan13Features candidate;
        candidate.query(VK_NULL_HANDLE, VK_API_VERSION_1_3, fake_get_features2, nullptr);
        EXPECT(candidate.core && candidate.usable());

        Vulkan13Features chosen;
        chosen.choose(candidate);
        candidate.query(VK_NULL_HANDLE, VK_API_VERSION_1_1, fake_get_features2, fake_enumerate_extensions);
        EXPECT(!candidate.core && !candidate.usable());
        EXPECT(chosen.core && chosen.usable() && chosen.extended_dynamic_state);
        EXPECT(chosen.subgroup_size_control && chosen.compute_full_subgroups);

        VkBaseOutStructure tail{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, nullptr};
        void* head = chosen.enable(&tail);
        const auto* enable13 = static_cast<const VkPhysicalDeviceVulkan13Features*>(head);
        EXPECT(enable13->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
        EXPECT(enable13->dynamicRendering == VK_TRUE && enable13->synchronization2 == VK_TRUE);
        EXPECT(enable13->subgroupSizeControl == VK_TRUE && enable13->computeFullSubgroups == VK_TRUE);
        EXPECT(enable13->pNext == &tail);

        std::vector<const char*> exts;
        chosen.add_extensions(exts);
        EXPECT(exts.empty());
    }

    // device_proc_or_khr resolves core first, then KHR, then EXT, else null.
    {
        ProcLookup lookup;
        g_proc = &lookup;

        lookup.core_name = "vkCmdPipelineBarrier2";
        lookup.khr_name = "vkCmdPipelineBarrier2KHR";
        lookup.ext_name = "vkCmdPipelineBarrier2EXT";
        EXPECT(device_proc_or_khr(fake_get_device_proc, VK_NULL_HANDLE, "vkCmdPipelineBarrier2") == stub_core);
        EXPECT(lookup.queries.size() == 1 && lookup.queries[0] == "vkCmdPipelineBarrier2");

        lookup.queries.clear();
        lookup.core_name = nullptr;
        EXPECT(device_proc_or_khr(fake_get_device_proc, VK_NULL_HANDLE, "vkCmdPipelineBarrier2") == stub_khr);
        EXPECT(lookup.queries.size() == 2 && lookup.queries[1] == "vkCmdPipelineBarrier2KHR");

        lookup.queries.clear();
        lookup.khr_name = nullptr;
        lookup.ext_name = "vkCmdSetPrimitiveTopologyEXT";
        EXPECT(device_proc_or_khr(fake_get_device_proc, VK_NULL_HANDLE, "vkCmdSetPrimitiveTopology") == stub_ext);
        EXPECT(lookup.queries.size() == 3 && lookup.queries[2] == "vkCmdSetPrimitiveTopologyEXT");

        lookup.queries.clear();
        lookup.ext_name = nullptr;
        EXPECT(device_proc_or_khr(fake_get_device_proc, VK_NULL_HANDLE, "vkCmdBeginRendering") == nullptr);
        EXPECT(lookup.queries.size() == 3);
    }

    std::printf("vulkan13_features: all checks passed\n");
    return 0;
}
