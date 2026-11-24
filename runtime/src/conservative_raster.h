#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <span>

namespace leonida::conservative_raster {
// Tier 1 bounds the outer uncertainty to half a pixel and requires a 16.8 rasterization grid. Leave room for
// half a grid step at that minimum precision in addition to Vulkan's fixed primitive overestimation. Do not
// infer Tier 3 from degenerate-triangle support: InnerCoverage also needs an explicit shader ABI opt-in.
inline bool tier1(bool extension_enabled, std::uint32_t subpixel_bits, float overestimate) {
    constexpr float maximum = 0.5f - 1.0f / 512.0f;
    return extension_enabled && subpixel_bits >= 8 && std::isfinite(overestimate) &&
           overestimate >= 0.0f && overestimate <= maximum;
}
inline std::uint32_t tier(bool extension_enabled, std::uint32_t subpixel_bits, float overestimate,
                          bool degenerate_triangles, bool fully_covered = false, bool inner_coverage_abi = false) {
    if (!tier1(extension_enabled, subpixel_bits, overestimate)) return 0;
    // Tier 2 allows 1/256 pixel total uncertainty and requires post-snap degenerates. Reserve half of the
    // device's actual raster grid step, conservatively capped at float's useful precision, for snapping.
    const float snap = std::ldexp(0.5f, -int(std::min(subpixel_bits, 24u)));
    if (!degenerate_triangles || overestimate > 1.0f / 256.0f - snap) return 1;
    // The physical property alone does not establish the guest shader-input ABI. Caller-owned libraries can
    // opt in once their InnerCoverage input is lowered to FullyCoveredEXT; the standalone device does not.
    return fully_covered && inner_coverage_abi ? 3u : 2u;
}

struct InnerCoverageRequirements {
    bool valid = false;
    bool fully_covered = false;
    bool post_depth = false;
};

// Inspect the actual module, not optional index metadata, before vkCreateShaderModule. Every valid use of
// FullyCoveredEXT requires FragmentFullyCoveredEXT; PostDepthCoverage has an additional device property.
inline InnerCoverageRequirements inner_coverage_requirements(std::span<const std::uint32_t> words) {
    InnerCoverageRequirements result;
    if (words.size() < 5 || words[0] != 0x07230203) return result;
    for (std::size_t i = 5; i < words.size();) {
        const auto count = words[i] >> 16, op = words[i] & 0xffff;
        if (!count || count > words.size() - i) return result;
        if (op == 17 && count >= 2 && words[i + 1] == 5265) result.fully_covered = true;
        if (op == 16 && count >= 3 && words[i + 2] == 4446) result.post_depth = true;
        i += count;
    }
    result.valid = true;
    return result;
}
} // namespace leonida::conservative_raster
