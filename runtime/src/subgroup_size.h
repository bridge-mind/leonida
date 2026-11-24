#pragma once

// Subgroup size control for pipeline creation (docs/d3d12_vulkan.md, "Subgroup size"). A translated shader assumes
// one subgroup size: 64 when the game's shader was wave64, 32 otherwise (spirv/index.json `subgroup_size`). A GPU
// whose default differs runs it inexact (a wave64 shader reads lanes 32-63 of a 32-wide subgroup; a wave32 pixel
// shader on a 64-wide default reads the other half's lanes), so each stage asks for the size it was translated for
// where Vulkan allows (VK_EXT_subgroup_size_control, core in 1.3).
//
// The decision is a pure function of the shader's index entry and what the device offers, so it is tested without
// a GPU (subgroup_size_test). It depends only on the Vulkan headers.

#include "shader_library.h"

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <string_view>

namespace leonida {

// What the device offers. `subgroup_size_control` and `compute_full_subgroups` say the feature is ENABLED on the
// device (the validation layers reject the structure and the flag otherwise), not merely supported.
struct SubgroupSizeCaps {
    bool subgroup_size_control = false;
    bool compute_full_subgroups = false;
    std::uint32_t min_size = 0;
    std::uint32_t max_size = 0;
    std::uint32_t default_size = 0;  // VkPhysicalDeviceSubgroupProperties::subgroupSize
    std::uint32_t max_compute_workgroup_subgroups = 0;
    VkShaderStageFlags required_stages = 0;  // requiredSubgroupSizeStages
};

// Fills the limits of `caps` from vkGetPhysicalDeviceProperties2 and sets the two feature flags from what the
// caller enables. Returns false when the device reports no subgroup size at all.
template <class GetProperties2>
bool query_subgroup_size_caps(GetProperties2 get_properties2, VkPhysicalDevice physical, bool enable_size_control,
                              bool enable_full_subgroups, SubgroupSizeCaps& caps) {
    VkPhysicalDeviceSubgroupSizeControlProperties control{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, &control};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroup};
    get_properties2(physical, &properties);
    caps = {};
    caps.default_size = subgroup.subgroupSize;
    caps.min_size = control.minSubgroupSize;
    caps.max_size = control.maxSubgroupSize;
    caps.max_compute_workgroup_subgroups = control.maxComputeWorkgroupSubgroups;
    caps.required_stages = control.requiredSubgroupSizeStages;
    caps.subgroup_size_control = enable_size_control;
    caps.compute_full_subgroups = enable_full_subgroups;
    return caps.default_size != 0;
}

// The Vulkan stage of an index entry's `stage` ("cs", "ps", ...); 0 for one this layer has no pipeline for.
inline VkShaderStageFlagBits subgroup_stage_bit(std::string_view stage) {
    if (stage == "cs") return VK_SHADER_STAGE_COMPUTE_BIT;
    if (stage == "ps") return VK_SHADER_STAGE_FRAGMENT_BIT;
    if (stage == "vs") return VK_SHADER_STAGE_VERTEX_BIT;
    if (stage == "gs") return VK_SHADER_STAGE_GEOMETRY_BIT;
    if (stage == "ms") return VK_SHADER_STAGE_MESH_BIT_EXT;
    return VkShaderStageFlagBits(0);
}

enum class SubgroupOutcome {
    emulated,         // --wave64-emulate: exact on any size, nothing is required
    required,         // the stage asks for `size`
    device_fixed,     // the device has one size and it is the wanted one: nothing to ask for
    device_default,   // cannot be required, but the default size is the wanted one: works, not guaranteed stable
    unavailable,      // the wanted size cannot be had: the shader runs at another size and may be inexact
    unknown_stage,    // no Vulkan stage for this entry
    unknown_device,   // the device sizes were not queried (all zero): nothing is asked for
};

struct SubgroupRequest {
    SubgroupOutcome outcome = SubgroupOutcome::unknown_stage;
    VkShaderStageFlagBits stage = VkShaderStageFlagBits(0);
    std::uint32_t wanted = 0;  // the size the translation assumed
    std::uint32_t size = 0;    // requiredSubgroupSize to chain; 0: chain nothing
    VkPipelineShaderStageCreateFlags flags = 0;  // REQUIRE_FULL_SUBGROUPS, compute only
};

inline bool subgroup_is_power_of_two(std::uint32_t v) { return v && !(v & (v - 1)); }

// Overflow-safe check of local_size[0] * local_size[1] * local_size[2] <= max_invocations.
inline bool subgroup_workgroup_fits(const std::array<std::uint32_t, 3>& local_size, std::uint64_t max_invocations) {
    const std::uint64_t xy = std::uint64_t(local_size[0]) * local_size[1];
    if (local_size[2] == 0) return true;
    return xy <= max_invocations / local_size[2];
}

// What to ask for one stage of a pipeline made from `entry`.
//   - Emulated (`wave64_exchange` present): nothing.
//   - Otherwise the wanted size is the entry's `subgroup_size`, or 64 for a wave64 entry without one and 32 for the
//     rest. VkPipelineShaderStageRequiredSubgroupSizeCreateInfo is chained when the subgroupSizeControl feature is
//     enabled, the stage is in requiredSubgroupSizeStages, the size is a power of two in [min, max] and, for
//     compute and mesh, the workgroup fits in size * maxComputeWorkgroupSubgroups
//     (VUID-VkPipelineShaderStageCreateInfo-pNext-02755 and -02756,
//     VUID-VkPipelineShaderStageRequiredSubgroupSizeCreateInfo-requiredSubgroupSize-02760..02762).
//   - A compute stage also gets VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT when computeFullSubgroups
//     is enabled and the local size X is a multiple of the size (VUID-VkPipelineShaderStageCreateInfo-flags-02785:
//     the flag needs the feature; -flags-08988: a compute, task or mesh stage; with a required size, -pNext-02757
//     wants X to be a multiple of it).
//   - A device whose one and only subgroup size (min == max) is the wanted one needs nothing (`device_fixed`).
//   - Where the size cannot be required (stage not listed, feature off) but the device default is the wanted size,
//     the shader works as it is (`device_default`): nothing guarantees it, a driver may change its default.
//     Otherwise the outcome is `unavailable`.
inline SubgroupRequest decide_subgroup_size(const ShaderEntry& entry, const SubgroupSizeCaps& caps) {
    SubgroupRequest r;
    r.stage = subgroup_stage_bit(entry.stage);
    if (!r.stage) return r;
    if (entry.emulated()) {
        r.outcome = SubgroupOutcome::emulated;
        return r;
    }
    r.wanted = entry.subgroup_size ? entry.subgroup_size : (entry.wave64 ? 64 : 32);
    if (!caps.default_size && !caps.max_size) {
        r.outcome = SubgroupOutcome::unknown_device;
        return r;
    }

    const bool compute_like = r.stage == VK_SHADER_STAGE_COMPUTE_BIT || r.stage == VK_SHADER_STAGE_MESH_BIT_EXT;
    const std::uint64_t max_invocations = std::uint64_t(r.wanted) * caps.max_compute_workgroup_subgroups;
    const bool fixed = caps.min_size && caps.min_size == caps.max_size && caps.min_size == r.wanted;
    const bool requirable =
        caps.subgroup_size_control && (caps.required_stages & r.stage) && subgroup_is_power_of_two(r.wanted) &&
        r.wanted >= caps.min_size && r.wanted <= caps.max_size &&
        (!compute_like || subgroup_workgroup_fits(entry.local_size, max_invocations));

    if (fixed) {
        r.outcome = SubgroupOutcome::device_fixed;
    } else if (requirable) {
        r.outcome = SubgroupOutcome::required;
        r.size = r.wanted;
        if (r.stage == VK_SHADER_STAGE_COMPUTE_BIT && caps.compute_full_subgroups && entry.local_size[0] != 0 &&
            entry.local_size[0] % r.wanted == 0)
            r.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    } else if (caps.default_size == r.wanted) {
        r.outcome = SubgroupOutcome::device_default;
    } else {
        r.outcome = SubgroupOutcome::unavailable;
    }
    return r;
}

// Puts a request into a stage create-info. `storage` holds the chained structure and must outlive the
// vkCreate*Pipelines call. Chains in front of whatever pNext already holds.
inline void apply_subgroup_request(const SubgroupRequest& request,
                                   VkPipelineShaderStageRequiredSubgroupSizeCreateInfo& storage,
                                   VkPipelineShaderStageCreateInfo& stage) {
    stage.flags |= request.flags;
    if (!request.size) return;
    storage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    storage.requiredSubgroupSize = request.size;
    storage.pNext = const_cast<void*>(stage.pNext);
    stage.pNext = &storage;
}

}  // namespace leonida
