// D24_UNORM_S8_UINT without a native D24 (the layer stores D32_SFLOAT_S8_UINT): what the conversions are, as plain C++.
// vulkan_depth24.h runs them on the GPU around copies (depth24_convert.comp); the reference here is what the
// compute shader and the clear path must equal, and it is what the clears use.
//
// A D24 texel is a 24-bit UNORM: d / 16777215. The footprint of a copy holds it in bits 0-23 of a 32-bit word (bits
// 24-31 are unspecified; the PC runtime reads them as 0 and writes 0), the stencil plane is a separate plane of
// bytes. The layer's D32 holds the float nearest to d / 16777215, which the 24-bit quantisation maps back to d.
#pragma once

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>

namespace leonida {

inline constexpr std::uint32_t depth24_max = 0xffffff;

// What the D3D11.3 functional spec says (3.2.3.6, FLOAT -> UNORM, which D3D12 inherits): NaN is 0; above 1.0f, INF
// included, is 1.0f and below 0.0f is 0.0f; then c = c * (2^n - 1) "in float", c = c + 0.5f, and the fraction is
// dropped; the conversion has a tolerance of 0.6 ULP on the integer side ("exact (1/2 ULP) conversion precision is
// acknowledged to be too expensive") and must be non-decreasing with every output attainable. Read literally, a tie
// goes up (2.5 -> 3). The spec's general rule (3.2.2) names round-to-nearest for FLOAT -> UNORM as an exception to
// round-to-nearest-even, so the spec does not fix ties. What WARP and AMD's D3D12 driver do was measured: they
// round the float product to nearest even (2.5 -> 2, 0.5 -> 0), which the tolerance allows; NVIDIA's differ at
// ties. That is a measurement of those two implementations, not a rule of the spec.
//
// The conversion of a float to a 24-bit UNORM as WARP and AMD's D3D12 drivers do it (measured against both, 5215
// clear values, ties and their neighbours included, in d3d12_depth24_compare.h): NaN and values below 0 give 0, 1 and above the
// maximum, anything in between multiplies by 2^24 - 1 in float (a rounded product, so a value a hair over a tie can
// land on it) and then rounds to nearest, ties to even (std::nearbyint's default mode). NVIDIA's D3D12 driver breaks
// some of those ties the other way, and its Vulkan driver, which has a native D24, agrees with it; a texture the
// layer stores as D32 follows WARP and AMD, which is what the Xbox build's own hardware is.
inline std::uint32_t depth24_quantize(float value) {
    if (!(value > 0.0f)) return 0;
    if (value >= 1.0f) return depth24_max;
    const float product = value * 16777215.0f;  // a float multiply, not a double one
    return static_cast<std::uint32_t>(std::nearbyint(product));
}

// The float stored for a 24-bit UNORM: the nearest float to d / 16777215 (quantising it gives d back).
inline float depth24_to_float(std::uint32_t d) {
    return static_cast<float>(double(d & depth24_max) / double(depth24_max));
}

// A depth value as the layer's D32 must hold it for a D24 texture: what a clear writes, so that reading the texture
// back gives the quantisation D3D makes.
inline float depth24_clear_value(float value) { return depth24_to_float(depth24_quantize(value)); }

// One copy between a footprint (words of 24-bit depth) and the D32 aspect of a texture, as the compute pass sees it.
// The pass converts between the application's buffer and a tight scratch buffer of floats; a copy command moves the
// scratch to or from the image. vulkan_depth24.h fills the scratch and descriptor fields in `prepare`.
//
// The same pass, scratch and job repack 96-bit R32G32B32 texels (`rgb96`, D3D-62): the footprint then has three
// words a texel and the scratch four (the layer's R32G32B32A32 image), copied as they are. The row and slice
// distances stay in words; only the texel's width differs (app_texel_words, scratch_texel_words).
struct Depth24Job {
    bool to_float = false;           // an upload (D24 words to floats); false: a readback (floats to words)
    bool rgb96 = false;              // 96-bit texels: `to_float` is only the direction (true: the upload)
    std::uint32_t app_texel_words() const { return rgb96 ? 3 : 1; }
    std::uint32_t scratch_texel_words() const { return rgb96 ? 4 : 1; }
    VkBuffer app = VK_NULL_HANDLE;   // the application's buffer
    std::uint64_t app_offset = 0;    // bytes, a multiple of 4
    std::uint64_t app_size = 0;      // bytes in the buffer
    std::uint32_t width = 0, height = 0, slices = 0;
    std::uint32_t app_row_words = 0;    // words between rows and between slices of the footprint
    std::uint32_t app_slice_words = 0;
    // Set by prepare:
    VkBuffer scratch = VK_NULL_HANDLE;  // width * height * slices texels (floats, or four words), tightly packed
    VkDeviceSize scratch_offset = 0;
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::uint32_t app_base_words = 0;   // the region's first word in the descriptor's window
    std::uint32_t scratch_base_words = 0;  // and the scratch's
};

}  // namespace leonida
