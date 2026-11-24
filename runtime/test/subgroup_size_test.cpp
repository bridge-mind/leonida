// The subgroup-size decision of pipeline creation (src/subgroup_size.h) without a GPU: synthetic shader index
// entries against synthetic device properties, and the VkPipelineShaderStageCreateInfo chain that results.
// The devices are the three shapes known so far:
//   - AMD, Windows driver (measured on a Radeon 880M, 26.11.2) and current Mesa RADV on GFX10+: sizes 32..64, the
//     stages FRAGMENT | COMPUTE | TASK | MESH can be required;
//   - NVIDIA: every stage can be required, but the only size is 32;
//   - an old RADV: sizes 32..64, COMPUTE only.
// No game data.
#include "../src/subgroup_size.h"

#include <cstdio>
#include <string>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

using namespace leonida;

namespace {

constexpr VkShaderStageFlags amd_stages = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT |
                                          VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
constexpr VkShaderStageFlags all_stages = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT |
                                          VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;

SubgroupSizeCaps amd() {
    SubgroupSizeCaps c;
    c.subgroup_size_control = c.compute_full_subgroups = true;
    c.min_size = 32;
    c.max_size = 64;
    c.default_size = 64;
    c.max_compute_workgroup_subgroups = 16;
    c.required_stages = amd_stages;
    return c;
}

SubgroupSizeCaps nvidia() {
    SubgroupSizeCaps c;
    c.subgroup_size_control = c.compute_full_subgroups = true;
    c.min_size = c.max_size = c.default_size = 32;
    c.max_compute_workgroup_subgroups = 32;
    c.required_stages = all_stages;
    return c;
}

SubgroupSizeCaps old_radv() {
    SubgroupSizeCaps c = amd();
    c.required_stages = VK_SHADER_STAGE_COMPUTE_BIT;
    return c;
}

// An index entry as the translator writes it: `wave64` shaders carry subgroup_size 64, the rest 32, and a shader
// translated with --wave64-emulate has no size and a wave64_exchange.
ShaderEntry entry(const char* stage, bool wave64, std::array<std::uint32_t, 3> local = {64, 1, 1},
                  bool emulated = false) {
    ShaderEntry e;
    e.name = std::string(stage) + (wave64 ? "_w64" : "_w32") + (emulated ? "_emu" : "");
    e.stage = stage;
    e.wave64 = wave64;
    e.subgroup_size = emulated ? 0 : (wave64 ? 64 : 32);
    if (emulated) e.wave64_exchange = "workgroup";
    e.local_size = local;
    return e;
}

}  // namespace

int main() {
    // --- AMD-like -------------------------------------------------------------------------------------------
    {
        const SubgroupSizeCaps caps = amd();
        // Native wave64 compute: 64, plus full subgroups when X is a multiple of 64.
        auto r = decide_subgroup_size(entry("cs", true), caps);
        CHECK(r.outcome == SubgroupOutcome::required && r.size == 64 && r.wanted == 64);
        CHECK(r.stage == VK_SHADER_STAGE_COMPUTE_BIT);
        CHECK(r.flags == VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT);
        r = decide_subgroup_size(entry("cs", true, {128, 2, 1}), caps);
        CHECK(r.size == 64 && (r.flags & VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT));
        // X not a multiple of 64 (32x2): 64 is still required, full subgroups are not (-pNext-02757).
        r = decide_subgroup_size(entry("cs", true, {32, 2, 1}), caps);
        CHECK(r.outcome == SubgroupOutcome::required && r.size == 64 && r.flags == 0);
        r = decide_subgroup_size(entry("cs", true, {96, 1, 1}), caps);
        CHECK(r.size == 64 && r.flags == 0);
        // A wave32 compute shader on a 32..64 device: 32 (the default is 64 here), full subgroups when X % 32 == 0.
        r = decide_subgroup_size(entry("cs", false, {96, 1, 1}), caps);
        CHECK(r.outcome == SubgroupOutcome::required && r.size == 32 &&
              r.flags == VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT);
        r = decide_subgroup_size(entry("cs", false, {48, 1, 1}), caps);
        CHECK(r.size == 32 && r.flags == 0);
        // Pixel shaders can be required on AMD: wave64 gets 64, wave32 gets 32, never a full-subgroups flag.
        r = decide_subgroup_size(entry("ps", true), caps);
        CHECK(r.outcome == SubgroupOutcome::required && r.size == 64 && r.flags == 0 &&
              r.stage == VK_SHADER_STAGE_FRAGMENT_BIT);
        r = decide_subgroup_size(entry("ps", false), caps);
        CHECK(r.outcome == SubgroupOutcome::required && r.size == 32 && r.flags == 0);
        // Mesh: wave64 by construction, 64 and no flag (the flag is only asked of compute).
        r = decide_subgroup_size(entry("ms", true, {128, 1, 1}), caps);
        CHECK(r.outcome == SubgroupOutcome::required && r.size == 64 && r.flags == 0 &&
              r.stage == VK_SHADER_STAGE_MESH_BIT_EXT);
        // Vertex and geometry stages are not in the list: a wave64 VS cannot be given 64, but the default is 64, so it
        // works as it is (device_default); there is no emulation for it.
        r = decide_subgroup_size(entry("vs", true), caps);
        CHECK(r.outcome == SubgroupOutcome::device_default && r.size == 0 && r.wanted == 64);
        r = decide_subgroup_size(entry("gs", true), caps);
        CHECK(r.outcome == SubgroupOutcome::device_default);
        // A wave32 VS on a 64-wide default is wrong too.
        r = decide_subgroup_size(entry("vs", false), caps);
        CHECK(r.outcome == SubgroupOutcome::unavailable && r.wanted == 32);
        // Emulated shaders ask for nothing, whatever the stage.
        r = decide_subgroup_size(entry("cs", true, {64, 1, 1}, true), caps);
        CHECK(r.outcome == SubgroupOutcome::emulated && r.size == 0 && r.flags == 0);
        r = decide_subgroup_size(entry("ms", true, {128, 1, 1}, true), caps);
        CHECK(r.outcome == SubgroupOutcome::emulated && r.size == 0 && r.flags == 0);
    }

    // --- NVIDIA-like: 32 only ----------------------------------------------------------------------------------
    {
        const SubgroupSizeCaps caps = nvidia();
        // Wave32 is what the device does: nothing to chain, in any stage.
        for (const char* stage : {"cs", "ps", "vs", "gs", "ms"}) {
            const auto r = decide_subgroup_size(entry(stage, false), caps);
            CHECK(r.outcome == SubgroupOutcome::device_fixed && r.size == 0 && r.flags == 0);
        }
        // Wave64 cannot be had (64 > max): native wave64 is reported in every stage, emulated is fine.
        for (const char* stage : {"cs", "ps", "vs", "gs", "ms"}) {
            const auto r = decide_subgroup_size(entry(stage, true), caps);
            CHECK(r.outcome == SubgroupOutcome::unavailable && r.size == 0 && r.flags == 0 && r.wanted == 64);
        }
        const auto emulated = decide_subgroup_size(entry("cs", true, {64, 1, 1}, true), caps);
        CHECK(emulated.outcome == SubgroupOutcome::emulated && emulated.size == 0);
    }

    // --- An old RADV: COMPUTE only ----------------------------------------------------------------------------------
    {
        const SubgroupSizeCaps caps = old_radv();
        auto r = decide_subgroup_size(entry("cs", true), caps);
        CHECK(r.outcome == SubgroupOutcome::required && r.size == 64 &&
              r.flags == VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT);
        // Pixel shaders cannot be required, and the default is 64: a wave32 PS is wrong, a wave64 PS works as it is
        // (device_default: nothing guarantees the default, min != max).
        r = decide_subgroup_size(entry("ps", false), caps);
        CHECK(r.outcome == SubgroupOutcome::unavailable && r.size == 0);
        r = decide_subgroup_size(entry("ps", true), caps);
        CHECK(r.outcome == SubgroupOutcome::device_default && r.size == 0);
        r = decide_subgroup_size(entry("ms", true, {128, 1, 1}), caps);
        CHECK(r.outcome == SubgroupOutcome::device_default);
    }

    // --- Feature, limit and index edge cases -------------------------------------------------------------------------
    {
        SubgroupSizeCaps caps = amd();
        // computeFullSubgroups off: the size is still required, the flag is not set.
        caps.compute_full_subgroups = false;
        auto r = decide_subgroup_size(entry("cs", true), caps);
        CHECK(r.outcome == SubgroupOutcome::required && r.size == 64 && r.flags == 0);
        // subgroupSizeControl not enabled: the chain would break VUID -pNext-02755, so it is not asked for.
        caps = amd();
        caps.subgroup_size_control = false;
        r = decide_subgroup_size(entry("cs", true), caps);
        CHECK(r.outcome == SubgroupOutcome::device_default && r.size == 0 && r.flags == 0);
        // The workgroup must fit size * maxComputeWorkgroupSubgroups (-pNext-02756): 1024 > 64 * 8.
        caps = amd();
        caps.max_compute_workgroup_subgroups = 8;
        r = decide_subgroup_size(entry("cs", true, {1024, 1, 1}), caps);
        CHECK(r.outcome == SubgroupOutcome::device_default);
        r = decide_subgroup_size(entry("cs", true, {512, 1, 1}), caps);
        CHECK(r.outcome == SubgroupOutcome::required && r.size == 64);
        r = decide_subgroup_size(entry("cs", true, {16, 8, 8}), caps);  // 1024 invocations too
        CHECK(r.outcome == SubgroupOutcome::device_default);
        // Large dimensions whose 3-way product wraps uint64_t must not bypass maxComputeWorkgroupSubgroups.
        r = decide_subgroup_size(entry("cs", true, {0x1000000u, 0x1000000u, 0x10000u}), amd());
        CHECK(r.outcome == SubgroupOutcome::device_default && r.size == 0 && r.flags == 0);
        // Zero local_size X (malformed/unknown index entry) must not enable REQUIRE_FULL_SUBGROUPS_BIT.
        r = decide_subgroup_size(entry("cs", true, {0, 1, 1}), amd());
        CHECK(r.flags == 0);
        // The limit is for compute and mesh; a pixel shader is not subject to it.
        r = decide_subgroup_size(entry("ps", true), caps);
        CHECK(r.outcome == SubgroupOutcome::required);
        // A size that is not a power of two, or outside [min, max], is not chained.
        caps = amd();
        ShaderEntry odd = entry("cs", true);
        odd.subgroup_size = 48;
        CHECK(decide_subgroup_size(odd, caps).outcome == SubgroupOutcome::unavailable);
        odd.subgroup_size = 128;
        CHECK(decide_subgroup_size(odd, caps).outcome == SubgroupOutcome::unavailable);
        odd.subgroup_size = 16;
        CHECK(decide_subgroup_size(odd, caps).outcome == SubgroupOutcome::unavailable);
        // An entry without subgroup_size (an index from an older translator) follows `wave64`.
        ShaderEntry bare = entry("cs", true);
        bare.subgroup_size = 0;
        CHECK(decide_subgroup_size(bare, amd()).size == 64);
        bare.wave64 = false;
        CHECK(decide_subgroup_size(bare, amd()).size == 32);
        // Device sizes that were never queried: nothing is asked for and nothing is reported.
        CHECK(decide_subgroup_size(entry("cs", true), SubgroupSizeCaps{}).outcome == SubgroupOutcome::unknown_device);
        // A stage this layer has no pipeline for.
        CHECK(decide_subgroup_size(entry("rt", true), amd()).outcome == SubgroupOutcome::unknown_stage);
        // A device that only has 64 runs a wave64 shader as it is.
        caps = amd();
        caps.min_size = caps.max_size = caps.default_size = 64;
        CHECK(decide_subgroup_size(entry("ps", true), caps).outcome == SubgroupOutcome::device_fixed);
        CHECK(decide_subgroup_size(entry("ps", false), caps).outcome == SubgroupOutcome::unavailable);
    }

    // --- The create-info chain -------------------------------------------------------------------------------------
    {
        const SubgroupSizeCaps caps = amd();
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo storage{};
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        apply_subgroup_request(decide_subgroup_size(entry("cs", true), caps), storage, stage);
        CHECK(stage.flags == VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT);
        CHECK(stage.pNext == &storage);
        CHECK(storage.sType == VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO);
        CHECK(storage.requiredSubgroupSize == 64 && storage.pNext == nullptr);

        // It goes in front of what the stage already chains.
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo other{};
        VkPipelineShaderStageCreateInfo chained{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        chained.pNext = &other;
        apply_subgroup_request(decide_subgroup_size(entry("ps", false), caps), storage, chained);
        CHECK(chained.pNext == &storage && storage.pNext == &other && storage.requiredSubgroupSize == 32);
        CHECK(chained.flags == 0);

        // A request for nothing leaves the stage alone, whatever the reason.
        for (const auto& e : {entry("cs", true, {64, 1, 1}, true), entry("vs", true)}) {
            VkPipelineShaderStageCreateInfo untouched{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            VkPipelineShaderStageRequiredSubgroupSizeCreateInfo unused{};
            apply_subgroup_request(decide_subgroup_size(e, caps), unused, untouched);
            CHECK(untouched.pNext == nullptr && untouched.flags == 0 && unused.sType == 0);
        }
    }

    // --- Querying the device ----------------------------------------------------------------------------------------
    {
        VkPhysicalDevice device = reinterpret_cast<VkPhysicalDevice>(std::uintptr_t(0x1000));
        VkPhysicalDevice seen = VK_NULL_HANDLE;
        const auto fake = [&](VkPhysicalDevice physical, VkPhysicalDeviceProperties2* properties) {
            seen = physical;
            auto* subgroup = static_cast<VkPhysicalDeviceSubgroupProperties*>(properties->pNext);
            auto* control = static_cast<VkPhysicalDeviceSubgroupSizeControlProperties*>(subgroup->pNext);
            subgroup->subgroupSize = 64;
            control->minSubgroupSize = 32;
            control->maxSubgroupSize = 64;
            control->maxComputeWorkgroupSubgroups = 16;
            control->requiredSubgroupSizeStages = amd_stages;
        };
        SubgroupSizeCaps caps;
        CHECK(query_subgroup_size_caps(fake, device, true, false, caps));
        CHECK(seen == device);
        CHECK(caps.default_size == 64 && caps.min_size == 32 && caps.max_size == 64 &&
              caps.max_compute_workgroup_subgroups == 16 && caps.required_stages == amd_stages);
        CHECK(caps.subgroup_size_control && !caps.compute_full_subgroups);
    }

    std::printf("subgroup size decision: AMD-like, NVIDIA-like, old RADV, limits, create-info chain and query OK\n");
    return 0;
}
