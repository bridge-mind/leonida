#pragma once

// GFX10 resource descriptors: the T# (image, 8 dwords), S# (sampler, 4 dwords) and V# (buffer, 4 dwords) the
// console's shaders load from descriptor heaps. Translated shaders hash a T# or S# to find its bindless index and
// dereference a V# (docs/d3d12_vulkan.md sections 3.4, 5.1 and 5.3). This header writes them for Create*View and
// CreateSampler, reads them back when a heap is scanned, and maps what they say to Vulkan. It needs the Vulkan
// headers only.
//
// Where the layout comes from:
// - Field positions: Mesa's register database, src/amd/registers/gfx10-rsrc.json, the SQ_IMG_RSRC_WORD0..7,
//   SQ_IMG_SAMP_WORD0..3 and SQ_BUF_RSRC_WORD0..3 types with their gfx103 variants (RDNA2). The comments below
//   name those fields.
// - Enum values in the fields: Mesa's gfx6.json (SQ_RSRC_IMG_TYPE, SQ_SEL_XYZW01, SQ_TEX_CLAMP, SQ_TEX_XY_FILTER,
//   SQ_TEX_MIP_FILTER, SQ_TEX_DEPTH_COMPARE, SQ_TEX_BORDER_COLOR, SQ_IMG_FILTER_TYPE).
// - Which value goes into which field: Mesa's src/amd/common/ac_descriptors.c (ac_build_gfx10_texture_descriptor,
//   ac_build_sampler_descriptor, ac_build_buffer_descriptor) and radv's radv_image_view.c and radv_sampler.c.
// - Formats: the GFX10_FORMAT enum, which tools/gfx10_formats.json holds too. A format names its channel sizes
//   from the top bit down, and says nothing about which channel is red (src/amd/common/gfx10_format_table.py);
//   that is DST_SEL's job.
// Mesa is MIT-licensed; only positions and values are taken from it. What this repository measured agrees with
// it: tools/rdna2spirv.py reads the T# FORMAT at word 1 bits 28:20, the V# base as word 0 | (word 1 & 0xFFFF)
// << 32, its stride at word 1 bits 29:16, NUM_RECORDS in word 2 and FORMAT and DST_SEL at word 3 bits 18:12 and
// 11:0, and runtime/test/shaders/cs_imgstore.asm builds T#s with word 1 = FORMAT << 20.
//
// Only what a Vulkan view or sampler needs is encoded. Every other bit (tiling mode, compression metadata,
// performance hints, RESOURCE_LEVEL) is written as zero and ignored when decoding: no hardware reads these
// descriptors, and what the console's driver puts in those bits is not known.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

namespace leonida::gfx10 {

using ImageWords = std::array<std::uint32_t, 8>;    // a T#
using SamplerWords = std::array<std::uint32_t, 4>;  // an S#
using BufferWords = std::array<std::uint32_t, 4>;   // a V#
using Swizzle = std::array<std::uint8_t, 4>;        // DST_SEL_X..W

namespace detail {

constexpr std::uint32_t field(std::uint32_t value, unsigned low, unsigned width) {
    return (value & ((1u << width) - 1)) << low;
}

constexpr std::uint32_t get(std::uint32_t word, unsigned low, unsigned width) {
    return (word >> low) & ((1u << width) - 1);
}

// Mesa's util_unsigned_fixed and util_signed_fixed with 8 fraction bits, after its CLAMP. A NaN becomes `low`.
inline float clamped(float value, float low, float high) {
    return value > low ? (value < high ? value : high) : low;
}

inline std::uint32_t unsigned_fixed(float value, float low, float high) {
    return std::uint32_t(clamped(value, low, high) * 256.0f);
}

inline std::uint32_t signed_fixed(float value, float low, float high) {
    return std::uint32_t(std::int32_t(clamped(value, low, high) * 256.0f));
}

}  // namespace detail

// SQ_SEL_XYZW01: what one DST_SEL field selects. 2 and 3 are reserved.
inline constexpr std::uint8_t select_zero = 0, select_one = 1, select_x = 4, select_y = 5, select_z = 6, select_w = 7;
inline constexpr Swizzle identity_swizzle{select_x, select_y, select_z, select_w};

inline constexpr bool valid_swizzle(const Swizzle& swizzle) {
    return std::all_of(swizzle.begin(), swizzle.end(), [](std::uint8_t s) { return s <= 7 && s != 2 && s != 3; });
}

// ---- T#: SQ_IMG_RSRC_WORD0..7 ----

// SQ_RSRC_IMG_TYPE (WORD3.TYPE). 0..7 are reserved, which is how a 32-byte heap slot holding a V# is told apart.
enum class ImageType : std::uint32_t {
    image_1d = 8,
    image_2d = 9,
    image_3d = 10,
    cube = 11,
    image_1d_array = 12,
    image_2d_array = 13,
    image_2d_msaa = 14,
    image_2d_msaa_array = 15,
};

inline constexpr bool multisampled(ImageType type) {
    return type == ImageType::image_2d_msaa || type == ImageType::image_2d_msaa_array;
}

// What a T# says. decode() returns the normal form: depth is 1 unless the type is 3D, layer_count is 1 for 3D,
// samples is 1 unless the type is multisampled, and a multisampled image has one mip.
struct Image {
    std::uint64_t address = 0;        // console VA of the image; the T# keeps it in 256-byte units (48 bits)
    std::uint32_t format = 0;         // GFX10_FORMAT_* code, 9 bits
    ImageType type = ImageType::image_2d;
    std::uint32_t width = 1;          // of mip 0, 1..16384
    std::uint32_t height = 1;
    std::uint32_t depth = 1;          // of mip 0, for 3D: 1..8192
    std::uint32_t first_layer = 0;    // arrays and cubes (a cube's layers are its faces, six to a cube)
    std::uint32_t layer_count = 1;
    std::uint32_t first_mip = 0;      // the view's range
    std::uint32_t mip_count = 1;
    std::uint32_t resource_mips = 1;  // of the whole resource
    std::uint32_t samples = 1;        // a power of two
    Swizzle swizzle = identity_swizzle;
    float min_lod = 0.0f;             // the view's LOD clamp, 0..15 in steps of 1/256 (D3D12 ResourceMinLODClamp)
    bool operator==(const Image&) const = default;
};

// As ac_build_gfx10_texture_descriptor and ac_set_mutable_tex_desc_fields fill it:
//   WORD0  BASE_ADDRESS = address >> 8
//   WORD1  BASE_ADDRESS_HI [7:0] = address >> 40, MIN_LOD [19:8] (4.8 fixed), FORMAT [28:20], WIDTH_LO [31:30]
//   WORD2  WIDTH_HI [11:0], HEIGHT [27:14]                     (width - 1 and height - 1)
//   WORD3  DST_SEL_X/Y/Z/W [11:0], BASE_LEVEL [15:12], LAST_LEVEL [19:16], TYPE [31:28]
//   WORD4  DEPTH [12:0] = depth - 1 for 3D, else the last layer; BASE_ARRAY [28:16] = the first layer
//   WORD5  MAX_MIP [7:4] = resource mips - 1
// A multisampled image has BASE_LEVEL 0 and log2(samples) in LAST_LEVEL and MAX_MIP.
inline ImageWords encode(const Image& image) {
    using detail::field;
    const bool msaa = multisampled(image.type);
    const std::uint32_t sample_log2 = std::uint32_t(std::bit_width(std::max(image.samples, 1u)) - 1);
    const std::uint32_t base_level = msaa ? 0 : image.first_mip;
    const std::uint32_t last_level = msaa ? sample_log2 : image.first_mip + image.mip_count - 1;
    const std::uint32_t max_mip = msaa ? sample_log2 : image.resource_mips - 1;
    const std::uint32_t last =
        image.type == ImageType::image_3d ? image.depth - 1 : image.first_layer + image.layer_count - 1;
    const std::uint64_t units = image.address >> 8;
    const std::uint32_t width = image.width - 1, height = image.height - 1;
    ImageWords words{};
    words[0] = std::uint32_t(units);
    words[1] = field(std::uint32_t(units >> 32), 0, 8) | field(detail::unsigned_fixed(image.min_lod, 0, 15), 8, 12) |
               field(image.format, 20, 9) | field(width, 30, 2);
    words[2] = field(width >> 2, 0, 12) | field(height, 14, 14);
    words[3] = field(image.swizzle[0], 0, 3) | field(image.swizzle[1], 3, 3) | field(image.swizzle[2], 6, 3) |
               field(image.swizzle[3], 9, 3) | field(base_level, 12, 4) | field(last_level, 16, 4) |
               field(std::uint32_t(image.type), 28, 4);
    words[4] = field(last, 0, 13) | field(image.first_layer, 16, 13);
    words[5] = field(max_mip, 4, 4);
    return words;
}

// TYPE names an image, so these eight dwords are a T# and not a V# followed by something else.
inline constexpr bool is_image(const ImageWords& words) { return detail::get(words[3], 28, 4) >= 8; }

inline constexpr float image_min_lod(const ImageWords& words) {
    return float(detail::get(words[1], 8, 12)) / 256.0f;
}

// False when the dwords are not an image descriptor, or a field contradicts another: a reserved DST_SEL, a last
// level or layer before the first, or a cube whose layers are not whole cubes.
inline bool decode(const ImageWords& words, Image& out) {
    using detail::get;
    if (!is_image(words)) return false;
    Image image{};
    image.type = ImageType(get(words[3], 28, 4));
    image.address = ((std::uint64_t(get(words[1], 0, 8)) << 32) | words[0]) << 8;
    image.min_lod = image_min_lod(words);
    image.format = get(words[1], 20, 9);
    image.width = (get(words[1], 30, 2) | (get(words[2], 0, 12) << 2)) + 1;
    image.height = get(words[2], 14, 14) + 1;
    for (unsigned i = 0; i < 4; ++i) image.swizzle[i] = std::uint8_t(get(words[3], 3 * i, 3));
    if (!valid_swizzle(image.swizzle)) return false;

    const std::uint32_t base_level = get(words[3], 12, 4), last_level = get(words[3], 16, 4);
    if (multisampled(image.type)) {
        if (base_level) return false;
        image.samples = 1u << last_level;
    } else {
        if (last_level < base_level) return false;
        image.first_mip = base_level;
        image.mip_count = last_level - base_level + 1;
        image.resource_mips = get(words[5], 4, 4) + 1;
    }

    const std::uint32_t last = get(words[4], 0, 13);
    image.first_layer = get(words[4], 16, 13);
    if (image.type == ImageType::image_3d) {
        image.depth = last + 1;
    } else {
        if (last < image.first_layer) return false;
        image.layer_count = last - image.first_layer + 1;
        if (image.type == ImageType::cube && image.layer_count % 6) return false;
    }
    out = image;
    return true;
}

// The view type translated shaders sample a T# of this type through. A cube is a six-layer 2D array there: the
// shader has already chosen the face (tools/rdna2spirv.py, image_op; docs/design.md, "Images and samplers").
inline constexpr VkImageViewType vulkan_view_type(ImageType type) {
    switch (type) {
    case ImageType::image_1d: return VK_IMAGE_VIEW_TYPE_1D;
    case ImageType::image_1d_array: return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case ImageType::image_3d: return VK_IMAGE_VIEW_TYPE_3D;
    case ImageType::cube:
    case ImageType::image_2d_array:
    case ImageType::image_2d_msaa_array: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    default: return VK_IMAGE_VIEW_TYPE_2D;
    }
}

// ---- formats ----

// One row for each Vulkan format: its GFX10_FORMAT code, and the DST_SEL that reads its channels as R, G, B, A
// (0 and 1 where the format has no such channel). A format's channels are X, Y, Z, W from the lowest bit (the
// first byte) up: Mesa pairs the channels of a format in that order with the sizes in the code's name reversed
// (gfx10_format_table.py). So R8G8B8A8 is the identity and B8G8R8A8 is the same code read as Z, Y, X, W. The
// first row of a code is the format vulkan_format() returns for it: the one in the identity order, where Vulkan
// has one. Depth formats share the codes of the colour formats with the same bits.
struct FormatRow {
    std::uint32_t gfx10;
    VkFormat vulkan;
    Swizzle channels;
};

inline constexpr Swizzle channels_r{select_x, select_zero, select_zero, select_one};
inline constexpr Swizzle channels_rg{select_x, select_y, select_zero, select_one};
inline constexpr Swizzle channels_rgb{select_x, select_y, select_z, select_one};
inline constexpr Swizzle channels_rgba = identity_swizzle;
inline constexpr Swizzle channels_bgr{select_z, select_y, select_x, select_one};
inline constexpr Swizzle channels_bgra{select_z, select_y, select_x, select_w};
inline constexpr Swizzle channels_abgr{select_w, select_z, select_y, select_x};

inline constexpr FormatRow format_table[] = {
    {1, VK_FORMAT_R8_UNORM, channels_r},                          // GFX10_FORMAT_8_UNORM
    {2, VK_FORMAT_R8_SNORM, channels_r},                          // GFX10_FORMAT_8_SNORM
    {3, VK_FORMAT_R8_USCALED, channels_r},                        // GFX10_FORMAT_8_USCALED
    {4, VK_FORMAT_R8_SSCALED, channels_r},                        // GFX10_FORMAT_8_SSCALED
    {5, VK_FORMAT_R8_UINT, channels_r},                           // GFX10_FORMAT_8_UINT
    {5, VK_FORMAT_S8_UINT, channels_r},                           // GFX10_FORMAT_8_UINT
    {6, VK_FORMAT_R8_SINT, channels_r},                           // GFX10_FORMAT_8_SINT
    {7, VK_FORMAT_R16_UNORM, channels_r},                         // GFX10_FORMAT_16_UNORM
    {7, VK_FORMAT_D16_UNORM, channels_r},                         // GFX10_FORMAT_16_UNORM
    {8, VK_FORMAT_R16_SNORM, channels_r},                         // GFX10_FORMAT_16_SNORM
    {9, VK_FORMAT_R16_USCALED, channels_r},                       // GFX10_FORMAT_16_USCALED
    {10, VK_FORMAT_R16_SSCALED, channels_r},                      // GFX10_FORMAT_16_SSCALED
    {11, VK_FORMAT_R16_UINT, channels_r},                         // GFX10_FORMAT_16_UINT
    {12, VK_FORMAT_R16_SINT, channels_r},                         // GFX10_FORMAT_16_SINT
    {13, VK_FORMAT_R16_SFLOAT, channels_r},                       // GFX10_FORMAT_16_FLOAT
    {14, VK_FORMAT_R8G8_UNORM, channels_rg},                      // GFX10_FORMAT_8_8_UNORM
    {15, VK_FORMAT_R8G8_SNORM, channels_rg},                      // GFX10_FORMAT_8_8_SNORM
    {16, VK_FORMAT_R8G8_USCALED, channels_rg},                    // GFX10_FORMAT_8_8_USCALED
    {17, VK_FORMAT_R8G8_SSCALED, channels_rg},                    // GFX10_FORMAT_8_8_SSCALED
    {18, VK_FORMAT_R8G8_UINT, channels_rg},                       // GFX10_FORMAT_8_8_UINT
    {19, VK_FORMAT_R8G8_SINT, channels_rg},                       // GFX10_FORMAT_8_8_SINT
    {20, VK_FORMAT_R32_UINT, channels_r},                         // GFX10_FORMAT_32_UINT
    {21, VK_FORMAT_R32_SINT, channels_r},                         // GFX10_FORMAT_32_SINT
    {22, VK_FORMAT_R32_SFLOAT, channels_r},                       // GFX10_FORMAT_32_FLOAT
    {22, VK_FORMAT_D32_SFLOAT, channels_r},                       // GFX10_FORMAT_32_FLOAT
    {23, VK_FORMAT_R16G16_UNORM, channels_rg},                    // GFX10_FORMAT_16_16_UNORM
    {24, VK_FORMAT_R16G16_SNORM, channels_rg},                    // GFX10_FORMAT_16_16_SNORM
    {25, VK_FORMAT_R16G16_USCALED, channels_rg},                  // GFX10_FORMAT_16_16_USCALED
    {26, VK_FORMAT_R16G16_SSCALED, channels_rg},                  // GFX10_FORMAT_16_16_SSCALED
    {27, VK_FORMAT_R16G16_UINT, channels_rg},                     // GFX10_FORMAT_16_16_UINT
    {28, VK_FORMAT_R16G16_SINT, channels_rg},                     // GFX10_FORMAT_16_16_SINT
    {29, VK_FORMAT_R16G16_SFLOAT, channels_rg},                   // GFX10_FORMAT_16_16_FLOAT
    {36, VK_FORMAT_B10G11R11_UFLOAT_PACK32, channels_rgb},        // GFX10_FORMAT_10_11_11_FLOAT
    {50, VK_FORMAT_A2B10G10R10_UNORM_PACK32, channels_rgba},      // GFX10_FORMAT_2_10_10_10_UNORM
    {50, VK_FORMAT_A2R10G10B10_UNORM_PACK32, channels_bgra},      // GFX10_FORMAT_2_10_10_10_UNORM
    {51, VK_FORMAT_A2B10G10R10_SNORM_PACK32, channels_rgba},      // GFX10_FORMAT_2_10_10_10_SNORM
    {52, VK_FORMAT_A2B10G10R10_USCALED_PACK32, channels_rgba},    // GFX10_FORMAT_2_10_10_10_USCALED
    {53, VK_FORMAT_A2B10G10R10_SSCALED_PACK32, channels_rgba},    // GFX10_FORMAT_2_10_10_10_SSCALED
    {54, VK_FORMAT_A2B10G10R10_UINT_PACK32, channels_rgba},       // GFX10_FORMAT_2_10_10_10_UINT
    {54, VK_FORMAT_A2R10G10B10_UINT_PACK32, channels_bgra},       // GFX10_FORMAT_2_10_10_10_UINT
    {55, VK_FORMAT_A2B10G10R10_SINT_PACK32, channels_rgba},       // GFX10_FORMAT_2_10_10_10_SINT
    {56, VK_FORMAT_R8G8B8A8_UNORM, channels_rgba},                // GFX10_FORMAT_8_8_8_8_UNORM
    {56, VK_FORMAT_B8G8R8A8_UNORM, channels_bgra},                // GFX10_FORMAT_8_8_8_8_UNORM
    {57, VK_FORMAT_R8G8B8A8_SNORM, channels_rgba},                // GFX10_FORMAT_8_8_8_8_SNORM
    {58, VK_FORMAT_R8G8B8A8_USCALED, channels_rgba},              // GFX10_FORMAT_8_8_8_8_USCALED
    {59, VK_FORMAT_R8G8B8A8_SSCALED, channels_rgba},              // GFX10_FORMAT_8_8_8_8_SSCALED
    {60, VK_FORMAT_R8G8B8A8_UINT, channels_rgba},                 // GFX10_FORMAT_8_8_8_8_UINT
    {61, VK_FORMAT_R8G8B8A8_SINT, channels_rgba},                 // GFX10_FORMAT_8_8_8_8_SINT
    {62, VK_FORMAT_R32G32_UINT, channels_rg},                     // GFX10_FORMAT_32_32_UINT
    {63, VK_FORMAT_R32G32_SINT, channels_rg},                     // GFX10_FORMAT_32_32_SINT
    {64, VK_FORMAT_R32G32_SFLOAT, channels_rg},                   // GFX10_FORMAT_32_32_FLOAT
    {65, VK_FORMAT_R16G16B16A16_UNORM, channels_rgba},            // GFX10_FORMAT_16_16_16_16_UNORM
    {66, VK_FORMAT_R16G16B16A16_SNORM, channels_rgba},            // GFX10_FORMAT_16_16_16_16_SNORM
    {67, VK_FORMAT_R16G16B16A16_USCALED, channels_rgba},          // GFX10_FORMAT_16_16_16_16_USCALED
    {68, VK_FORMAT_R16G16B16A16_SSCALED, channels_rgba},          // GFX10_FORMAT_16_16_16_16_SSCALED
    {69, VK_FORMAT_R16G16B16A16_UINT, channels_rgba},             // GFX10_FORMAT_16_16_16_16_UINT
    {70, VK_FORMAT_R16G16B16A16_SINT, channels_rgba},             // GFX10_FORMAT_16_16_16_16_SINT
    {71, VK_FORMAT_R16G16B16A16_SFLOAT, channels_rgba},           // GFX10_FORMAT_16_16_16_16_FLOAT
    {72, VK_FORMAT_R32G32B32_UINT, channels_rgb},                 // GFX10_FORMAT_32_32_32_UINT
    {73, VK_FORMAT_R32G32B32_SINT, channels_rgb},                 // GFX10_FORMAT_32_32_32_SINT
    {74, VK_FORMAT_R32G32B32_SFLOAT, channels_rgb},               // GFX10_FORMAT_32_32_32_FLOAT
    {75, VK_FORMAT_R32G32B32A32_UINT, channels_rgba},             // GFX10_FORMAT_32_32_32_32_UINT
    {76, VK_FORMAT_R32G32B32A32_SINT, channels_rgba},             // GFX10_FORMAT_32_32_32_32_SINT
    {77, VK_FORMAT_R32G32B32A32_SFLOAT, channels_rgba},           // GFX10_FORMAT_32_32_32_32_FLOAT
    {128, VK_FORMAT_R8_SRGB, channels_r},                         // GFX10_FORMAT_8_SRGB
    {129, VK_FORMAT_R8G8_SRGB, channels_rg},                      // GFX10_FORMAT_8_8_SRGB
    {130, VK_FORMAT_R8G8B8A8_SRGB, channels_rgba},                // GFX10_FORMAT_8_8_8_8_SRGB
    {130, VK_FORMAT_B8G8R8A8_SRGB, channels_bgra},                // GFX10_FORMAT_8_8_8_8_SRGB
    {132, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, channels_rgb},        // GFX10_FORMAT_5_9_9_9_FLOAT
    {133, VK_FORMAT_B5G6R5_UNORM_PACK16, channels_rgb},           // GFX10_FORMAT_5_6_5_UNORM
    {133, VK_FORMAT_R5G6B5_UNORM_PACK16, channels_bgr},           // GFX10_FORMAT_5_6_5_UNORM
    {134, VK_FORMAT_A1R5G5B5_UNORM_PACK16, channels_bgra},        // GFX10_FORMAT_1_5_5_5_UNORM
    {136, VK_FORMAT_A4B4G4R4_UNORM_PACK16, channels_rgba},        // GFX10_FORMAT_4_4_4_4_UNORM
    {136, VK_FORMAT_A4R4G4B4_UNORM_PACK16, channels_bgra},        // GFX10_FORMAT_4_4_4_4_UNORM
    {136, VK_FORMAT_R4G4B4A4_UNORM_PACK16, channels_abgr},        // GFX10_FORMAT_4_4_4_4_UNORM
    {141, VK_FORMAT_D24_UNORM_S8_UINT, channels_r},               // GFX10_FORMAT_8_24_UNORM
    {141, VK_FORMAT_X8_D24_UNORM_PACK32, channels_r},             // GFX10_FORMAT_8_24_UNORM
    {146, VK_FORMAT_D32_SFLOAT_S8_UINT, channels_r},              // GFX10_FORMAT_X24_8_32_FLOAT
    {169, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, channels_rgba},         // GFX10_FORMAT_BC1_UNORM
    {169, VK_FORMAT_BC1_RGB_UNORM_BLOCK, channels_rgb},           // GFX10_FORMAT_BC1_UNORM
    {170, VK_FORMAT_BC1_RGBA_SRGB_BLOCK, channels_rgba},          // GFX10_FORMAT_BC1_SRGB
    {170, VK_FORMAT_BC1_RGB_SRGB_BLOCK, channels_rgb},            // GFX10_FORMAT_BC1_SRGB
    {171, VK_FORMAT_BC2_UNORM_BLOCK, channels_rgba},              // GFX10_FORMAT_BC2_UNORM
    {172, VK_FORMAT_BC2_SRGB_BLOCK, channels_rgba},               // GFX10_FORMAT_BC2_SRGB
    {173, VK_FORMAT_BC3_UNORM_BLOCK, channels_rgba},              // GFX10_FORMAT_BC3_UNORM
    {174, VK_FORMAT_BC3_SRGB_BLOCK, channels_rgba},               // GFX10_FORMAT_BC3_SRGB
    {175, VK_FORMAT_BC4_UNORM_BLOCK, channels_r},                 // GFX10_FORMAT_BC4_UNORM
    {176, VK_FORMAT_BC4_SNORM_BLOCK, channels_r},                 // GFX10_FORMAT_BC4_SNORM
    {177, VK_FORMAT_BC5_UNORM_BLOCK, channels_rg},                // GFX10_FORMAT_BC5_UNORM
    {178, VK_FORMAT_BC5_SNORM_BLOCK, channels_rg},                // GFX10_FORMAT_BC5_SNORM
    {179, VK_FORMAT_BC6H_UFLOAT_BLOCK, channels_rgb},             // GFX10_FORMAT_BC6_UFLOAT
    {180, VK_FORMAT_BC6H_SFLOAT_BLOCK, channels_rgb},             // GFX10_FORMAT_BC6_SFLOAT
    {181, VK_FORMAT_BC7_UNORM_BLOCK, channels_rgba},              // GFX10_FORMAT_BC7_UNORM
    {182, VK_FORMAT_BC7_SRGB_BLOCK, channels_rgba},               // GFX10_FORMAT_BC7_SRGB
};

// The format of the console's separate stencil surface: what a T# of a depth-stencil texture's stencil plane names.
inline constexpr std::uint32_t stencil_format = 5;  // GFX10_FORMAT_8_UINT

inline constexpr const FormatRow* format_row(VkFormat format) {
    for (const FormatRow& row : format_table)
        if (row.vulkan == format) return &row;
    return nullptr;
}

// The Vulkan format with the code's bits, in the hardware's channel order where Vulkan has such a format.
// VK_FORMAT_UNDEFINED: not in the table. A code that depth and colour formats share gives the colour format.
inline constexpr VkFormat vulkan_format(std::uint32_t gfx10_format) {
    for (const FormatRow& row : format_table)
        if (row.gfx10 == gfx10_format) return row.vulkan;
    return VK_FORMAT_UNDEFINED;
}

// 0 (GFX10_FORMAT_INVALID): not in the table.
inline constexpr std::uint32_t format_from_vulkan(VkFormat format) {
    const FormatRow* row = format_row(format);
    return row ? row->gfx10 : 0;
}

// The DST_SEL a T# of this format carries when the view applies no component mapping of its own: it reads the
// format's channels as R, G, B, A. The identity for a format that is not in the table.
inline constexpr Swizzle format_swizzle(VkFormat format) {
    const FormatRow* row = format_row(format);
    return row ? row->channels : identity_swizzle;
}

inline constexpr bool is_depth_stencil_format(VkFormat format) {
    switch (format) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_S8_UINT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

// The format to view an image of `image_format` in when a T# names `gfx10_format`: the image's own format when
// the codes agree, else the format of that code with the image's channel order (the sRGB view of a UNORM image,
// say). With `allow_channel_cast`, a listed same-size cross-family cast may use the code's first Vulkan format
// when no channel-order match exists (R8G8B8A8_UINT viewed as R32_UINT, say). Depth/stencil images are only
// compatible with their own format in Vulkan (VUID-VkImageViewCreateInfo-image-01762 / 09594), and colour
// images can never be viewed as depth/stencil. VK_FORMAT_UNDEFINED: no compatible format exists.
inline constexpr VkFormat vulkan_view_format(VkFormat image_format, std::uint32_t gfx10_format,
                                             bool allow_channel_cast = false) {
    const FormatRow* image = format_row(image_format);
    if (!image) return VK_FORMAT_UNDEFINED;
    if (image->gfx10 == gfx10_format) return image_format;
    if (is_depth_stencil_format(image_format)) return VK_FORMAT_UNDEFINED;
    const FormatRow* fallback = nullptr;
    for (const FormatRow& row : format_table) {
        if (row.gfx10 != gfx10_format || is_depth_stencil_format(row.vulkan)) continue;
        if (row.channels == image->channels) return row.vulkan;
        if (!fallback) fallback = &row;
    }
    return allow_channel_cast && fallback ? fallback->vulkan : VK_FORMAT_UNDEFINED;
}

// DST_SEL as the component mapping of a view whose format is `view_format`: X, Y, Z, W name the hardware's
// channels, and the format says which of R, G, B, A each one is. A channel the format does not have reads as zero.
inline constexpr VkComponentMapping vulkan_swizzle(const Swizzle& swizzle, VkFormat view_format) {
    const Swizzle channels = format_swizzle(view_format);
    const auto component = [&](std::uint8_t select) {
        if (select == select_one) return VK_COMPONENT_SWIZZLE_ONE;
        if (select >= select_x && select <= select_w)
            for (unsigned i = 0; i < 4; ++i)
                if (channels[i] == select) return VkComponentSwizzle(VK_COMPONENT_SWIZZLE_R + i);
        return VK_COMPONENT_SWIZZLE_ZERO;
    };
    return {component(swizzle[0]), component(swizzle[1]), component(swizzle[2]), component(swizzle[3])};
}

// The same for a format in the hardware's own order (X, Y, Z, W are R, G, B, A).
inline constexpr VkComponentMapping vulkan_swizzle(const Swizzle& swizzle) {
    return vulkan_swizzle(swizzle, VK_FORMAT_R8G8B8A8_UNORM);
}

// ---- S#: SQ_IMG_SAMP_WORD0..3 ----

enum class AddressMode : std::uint32_t {  // SQ_TEX_CLAMP
    wrap = 0,
    mirror = 1,
    clamp_last_texel = 2,
    mirror_once_last_texel = 3,
    clamp_half_border = 4,
    mirror_once_half_border = 5,
    clamp_border = 6,
    mirror_once_border = 7,
};

enum class Filter : std::uint32_t {  // SQ_TEX_XY_FILTER
    point = 0,
    bilinear = 1,
    aniso_point = 2,
    aniso_bilinear = 3,
};

enum class MipFilter : std::uint32_t {  // SQ_TEX_MIP_FILTER
    none = 0,
    point = 1,
    linear = 2,
};

enum class CompareFunction : std::uint32_t {  // SQ_TEX_DEPTH_COMPARE; the same values as VkCompareOp
    never = 0,
    less = 1,
    equal = 2,
    less_equal = 3,
    greater = 4,
    not_equal = 5,
    greater_equal = 6,
    always = 7,
};

enum class BorderColor : std::uint32_t {  // SQ_TEX_BORDER_COLOR
    transparent_black = 0,
    opaque_black = 1,
    opaque_white = 2,
    table = 3,  // SQ_TEX_BORDER_COLOR_REGISTER: the colour at border_color_index in the device's table
};

enum class FilterMode : std::uint32_t {  // SQ_IMG_FILTER_TYPE: D3D12's MINIMUM_ and MAXIMUM_ filters
    blend = 0,
    minimum = 1,
    maximum = 2,
};

// What an S# says. decode() returns the normal form: max_anisotropy is 1, 2, 4, 8 or 16.
struct Sampler {
    AddressMode address_u = AddressMode::wrap;
    AddressMode address_v = AddressMode::wrap;
    AddressMode address_w = AddressMode::wrap;
    Filter mag_filter = Filter::point;
    Filter min_filter = Filter::point;
    MipFilter mip_filter = MipFilter::none;
    std::uint32_t max_anisotropy = 1;
    CompareFunction compare = CompareFunction::never;  // never: not a comparison sampler, as radv writes it
    float min_lod = 0.0f;                              // 0..15 in steps of 1/256
    float max_lod = 0.0f;
    float lod_bias = 0.0f;                             // -32..31 in steps of 1/256
    BorderColor border_color = BorderColor::transparent_black;
    std::uint32_t border_color_index = 0;              // 12 bits
    bool unnormalized = false;
    FilterMode filter_mode = FilterMode::blend;
    // The layer's own: the border colour is read as integers (D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR). The hardware
    // picks integer or float by the image's format, but Vulkan needs VK_BORDER_COLOR_INT_* in the sampler, so the
    // layer keeps it in the S# words, which are the sampler's key: see kIntegerBorderPtr.
    bool integer_border = false;
    bool compare_never = false;  // D3D12 comparison filter with DEPTH_COMPARE_FUNC = NEVER
    bool operator==(const Sampler&) const = default;
};

// BORDER_COLOR_PTR of an S# with an integer border. The hardware reads the pointer only for BorderColor::table
// (Mesa writes it only for SQ_TEX_BORDER_COLOR_REGISTER), so with the three built-in types the field is free, and
// the layer marks an integer border there. An integer table colour is not marked: table colours are not kept yet.
inline constexpr std::uint32_t kIntegerBorderPtr = 0x800;

// With DERIV_ADJUST_EN clear, GFX10 S# WORD3 has BORDER_COLOR_PTR in bits 0..11 and BORDER_COLOR_TYPE
// in 30..31. Use an otherwise unused bit in layer-created samplers to distinguish a D3D12 NEVER comparison
// from an ordinary filter: both hardware DEPTH_COMPARE_FUNC fields are zero, but Vulkan needs distinct samplers.
inline constexpr std::uint32_t kCompareNever = 1u << 12;

// As ac_build_sampler_descriptor fills it for GFX10:
//   WORD0  CLAMP_X [2:0], CLAMP_Y [5:3], CLAMP_Z [8:6], MAX_ANISO_RATIO [11:9] = log2(anisotropy), at most 4,
//          DEPTH_COMPARE_FUNC [14:12], FORCE_UNNORMALIZED [15], FILTER_MODE [30:29]
//   WORD1  MIN_LOD [11:0], MAX_LOD [23:12]                       (4.8 fixed, clamped to 0..15)
//   WORD2  LOD_BIAS [13:0] (signed 5.8 fixed, clamped to -32..31), XY_MAG_FILTER [21:20], XY_MIN_FILTER [23:22],
//          MIP_FILTER [27:26]
//   WORD3  BORDER_COLOR_PTR [11:0], BORDER_COLOR_TYPE [31:30]
// Mesa also sets tuning fields from the anisotropy (ANISO_THRESHOLD, ANISO_BIAS, PERF_MIP, ANISO_OVERRIDE) and
// TRUNC_COORD for point filters. They are not written here, and decode() ignores them.
inline SamplerWords encode(const Sampler& sampler) {
    using detail::field;
    const std::uint32_t aniso_log2 =
        std::min(std::uint32_t(std::bit_width(std::max(sampler.max_anisotropy, 1u)) - 1), 4u);
    SamplerWords words{};
    words[0] = field(std::uint32_t(sampler.address_u), 0, 3) | field(std::uint32_t(sampler.address_v), 3, 3) |
               field(std::uint32_t(sampler.address_w), 6, 3) | field(aniso_log2, 9, 3) |
               field(std::uint32_t(sampler.compare), 12, 3) | field(sampler.unnormalized ? 1u : 0u, 15, 1) |
               field(std::uint32_t(sampler.filter_mode), 29, 2);
    words[1] = field(detail::unsigned_fixed(sampler.min_lod, 0, 15), 0, 12) |
               field(detail::unsigned_fixed(sampler.max_lod, 0, 15), 12, 12);
    words[2] = field(detail::signed_fixed(sampler.lod_bias, -32, 31), 0, 14) |
               field(std::uint32_t(sampler.mag_filter), 20, 2) | field(std::uint32_t(sampler.min_filter), 22, 2) |
               field(std::uint32_t(sampler.mip_filter), 26, 2);
    const bool marked = sampler.integer_border && sampler.border_color != BorderColor::table;
    words[3] = field(marked ? kIntegerBorderPtr : sampler.border_color_index, 0, 12) |
               field(std::uint32_t(sampler.border_color), 30, 2);
    if (sampler.compare == CompareFunction::never && sampler.compare_never) words[3] |= kCompareNever;
    return words;
}

// Every bit pattern is a sampler; values the enums do not name (MIP_FILTER 3, FILTER_MODE 3) are kept as they
// are, and vulkan_sampler() treats them as the default.
inline Sampler decode(const SamplerWords& words) {
    using detail::get;
    Sampler sampler{};
    sampler.address_u = AddressMode(get(words[0], 0, 3));
    sampler.address_v = AddressMode(get(words[0], 3, 3));
    sampler.address_w = AddressMode(get(words[0], 6, 3));
    sampler.max_anisotropy = 1u << std::min(get(words[0], 9, 3), 4u);
    sampler.compare = CompareFunction(get(words[0], 12, 3));
    // DERIV_ADJUST_EN repurposes WORD3, so its bits are not layer markers in that layout.
    const bool deriv_adjust = get(words[2], 31, 1) != 0;
    sampler.compare_never = !deriv_adjust && sampler.compare == CompareFunction::never && (words[3] & kCompareNever);
    sampler.unnormalized = get(words[0], 15, 1) != 0;
    sampler.filter_mode = FilterMode(get(words[0], 29, 2));
    sampler.min_lod = float(get(words[1], 0, 12)) / 256.0f;
    sampler.max_lod = float(get(words[1], 12, 12)) / 256.0f;
    const std::uint32_t bias = get(words[2], 0, 14);
    sampler.lod_bias = float(std::int32_t(bias ^ 0x2000u) - 0x2000) / 256.0f;  // sign-extend 14 bits
    sampler.mag_filter = Filter(get(words[2], 20, 2));
    sampler.min_filter = Filter(get(words[2], 22, 2));
    sampler.mip_filter = MipFilter(get(words[2], 26, 2));
    sampler.border_color_index = get(words[3], 0, 12);
    sampler.border_color = BorderColor(get(words[3], 30, 2));
    if (sampler.border_color != BorderColor::table && sampler.border_color_index == kIntegerBorderPtr) {
        sampler.integer_border = true;
        sampler.border_color_index = 0;
    }
    return sampler;
}

// The inverse of radv_tex_wrap. Vulkan has no half-border or mirror-once-border mode; the nearest one is used.
inline constexpr VkSamplerAddressMode vulkan_address_mode(AddressMode mode) {
    switch (mode) {
    case AddressMode::wrap: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case AddressMode::mirror: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case AddressMode::clamp_border: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case AddressMode::mirror_once_last_texel:
    case AddressMode::mirror_once_half_border:
    case AddressMode::mirror_once_border: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;  // clamp_last_texel, clamp_half_border
    }
}

inline constexpr VkSamplerReductionMode vulkan_reduction_mode(FilterMode mode) {
    switch (mode) {
    case FilterMode::minimum: return VK_SAMPLER_REDUCTION_MODE_MIN;
    case FilterMode::maximum: return VK_SAMPLER_REDUCTION_MODE_MAX;
    default: return VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE;
    }
}

// The inverse of radv_make_sampler_descriptor. Left to the caller, because they need more than this structure:
// - border_color == table: VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK is returned; the colour itself is in a table
//   the S# only indexes (VK_EXT_custom_border_color);
// - filter_mode != blend: chain a VkSamplerReductionModeCreateInfo with vulkan_reduction_mode();
// - mirror-once modes need samplerMirrorClampToEdge, anisotropy needs samplerAnisotropy.
// MIP_FILTER_NONE samples the base level only, which Vulkan expresses as nearest mip filtering with the LOD held
// in [0, 0.25] (the Vulkan specification's note on samplers without mipmapping). An unnormalised sampler gets
// what Vulkan requires of one: no mips, no anisotropy, no comparison.
// A decoded GFX10 bias has absolute value at most 32. When creating a device sampler, pass its smaller limit.
inline VkSamplerCreateInfo vulkan_sampler(const Sampler& sampler, float max_lod_bias = 32.0f) {
    static_assert(int(CompareFunction::never) == int(VK_COMPARE_OP_NEVER) &&
                  int(CompareFunction::less) == int(VK_COMPARE_OP_LESS) &&
                  int(CompareFunction::equal) == int(VK_COMPARE_OP_EQUAL) &&
                  int(CompareFunction::less_equal) == int(VK_COMPARE_OP_LESS_OR_EQUAL) &&
                  int(CompareFunction::greater) == int(VK_COMPARE_OP_GREATER) &&
                  int(CompareFunction::not_equal) == int(VK_COMPARE_OP_NOT_EQUAL) &&
                  int(CompareFunction::greater_equal) == int(VK_COMPARE_OP_GREATER_OR_EQUAL) &&
                  int(CompareFunction::always) == int(VK_COMPARE_OP_ALWAYS));
    const auto filter = [](Filter f) {
        return f == Filter::bilinear || f == Filter::aniso_bilinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    };
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = filter(sampler.mag_filter);
    info.minFilter = filter(sampler.min_filter);
    info.mipmapMode =
        sampler.mip_filter == MipFilter::linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = vulkan_address_mode(sampler.address_u);
    info.addressModeV = vulkan_address_mode(sampler.address_v);
    info.addressModeW = vulkan_address_mode(sampler.address_w);
    info.mipLodBias = std::clamp(sampler.lod_bias, -max_lod_bias, max_lod_bias);
    info.anisotropyEnable = sampler.max_anisotropy > 1 ? VK_TRUE : VK_FALSE;
    info.maxAnisotropy = float(std::max(sampler.max_anisotropy, 1u));
    info.compareEnable = (sampler.compare_never || sampler.compare != CompareFunction::never) ? VK_TRUE : VK_FALSE;
    info.compareOp = VkCompareOp(sampler.compare);
    info.minLod = sampler.min_lod;
    info.maxLod = sampler.max_lod;
    if (sampler.mip_filter == MipFilter::none) {
        info.minLod = 0.0f;
        info.maxLod = 0.25f;
    }
    // An integer image sampled with a FLOAT border colour reads it as 0 on NVIDIA (Vonsek, #538), so the type follows
    // integer_border. A table colour falls back to transparent black matching integer_border.
    const bool integer = sampler.integer_border;
    switch (sampler.border_color) {
    case BorderColor::opaque_black:
        info.borderColor = integer ? VK_BORDER_COLOR_INT_OPAQUE_BLACK : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        break;
    case BorderColor::opaque_white:
        info.borderColor = integer ? VK_BORDER_COLOR_INT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        break;
    default:
        info.borderColor = integer ? VK_BORDER_COLOR_INT_TRANSPARENT_BLACK : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        break;
    }
    if (sampler.unnormalized) {
        info.unnormalizedCoordinates = VK_TRUE;
        // VUID-VkSamplerCreateInfo-unnormalizedCoordinates-01075: U and V clamp to the edge or the border (a
        // repeating mode would wrap at a texel count the sampler cannot know); -01072: one filter for min and mag.
        // The S# allows any address mode with FORCE_UNNORMALIZED, and WRAP is its default (word 0 = 0).
        const auto clamp_or_border = [](VkSamplerAddressMode m) {
            return m == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ? m : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        };
        info.addressModeU = clamp_or_border(info.addressModeU);
        info.addressModeV = clamp_or_border(info.addressModeV);
        info.minFilter = info.magFilter;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.minLod = info.maxLod = 0.0f;
        info.anisotropyEnable = VK_FALSE;
        info.maxAnisotropy = 1.0f;
        info.compareEnable = VK_FALSE;
    }
    return info;
}

// ---- V#: SQ_BUF_RSRC_WORD0..3 ----

// SQ_BUF_RSRC_WORD3.OOB_SELECT: the bounds check the hardware makes, and the one the translator makes too
// (tools/rdna2spirv.py, buffer_address; docs/design.md, Buffer bounds checking): structured_with_offset is out of
// range at index >= num_records or offset + 4 > stride, structured at index >= num_records, disabled only at
// num_records == 0, raw at offset + 4 > num_records (docs/buffer_bounds.md). A V# built here for a shader to read must say which: the default
// below makes every stride-0 access out of range.
enum class OobSelect : std::uint32_t {
    structured_with_offset = 0,
    structured = 1,
    disabled = 2,
    raw = 3,
};

struct Buffer {
    std::uint64_t address = 0;      // console VA of the first byte, 48 bits
    std::uint32_t stride = 0;       // bytes a record, 14 bits; 0: a raw buffer
    std::uint32_t num_records = 0;  // records, or bytes when the stride is 0
    std::uint32_t format = 0;       // GFX10_FORMAT_* code below 128, for typed loads and stores
    Swizzle swizzle = identity_swizzle;
    OobSelect oob_select = OobSelect::structured_with_offset;
    bool operator==(const Buffer&) const = default;
};

// As ac_build_buffer_descriptor fills it:
//   WORD0  BASE_ADDRESS = address
//   WORD1  BASE_ADDRESS_HI [15:0] = address >> 32, STRIDE [29:16]
//   WORD2  NUM_RECORDS
//   WORD3  DST_SEL_X/Y/Z/W [11:0], FORMAT [18:12], OOB_SELECT [29:28], TYPE [31:30] = 0 (SQ_RSRC_BUF)
inline BufferWords encode(const Buffer& buffer) {
    using detail::field;
    BufferWords words{};
    words[0] = std::uint32_t(buffer.address);
    words[1] = field(std::uint32_t(buffer.address >> 32), 0, 16) | field(buffer.stride, 16, 14);
    words[2] = buffer.num_records;
    words[3] = field(buffer.swizzle[0], 0, 3) | field(buffer.swizzle[1], 3, 3) | field(buffer.swizzle[2], 6, 3) |
               field(buffer.swizzle[3], 9, 3) | field(buffer.format, 12, 7) |
               field(std::uint32_t(buffer.oob_select), 28, 2);
    return words;
}

// False when TYPE is not a buffer's (an image's TYPE has its top bit set) or a DST_SEL is reserved.
inline bool decode(const BufferWords& words, Buffer& out) {
    using detail::get;
    if (get(words[3], 30, 2) != 0) return false;
    Buffer buffer{};
    buffer.address = (std::uint64_t(get(words[1], 0, 16)) << 32) | words[0];
    buffer.stride = get(words[1], 16, 14);
    buffer.num_records = words[2];
    for (unsigned i = 0; i < 4; ++i) buffer.swizzle[i] = std::uint8_t(get(words[3], 3 * i, 3));
    if (!valid_swizzle(buffer.swizzle)) return false;
    buffer.format = get(words[3], 12, 7);
    buffer.oob_select = OobSelect(get(words[3], 28, 2));
    out = buffer;
    return true;
}

}  // namespace leonida::gfx10
