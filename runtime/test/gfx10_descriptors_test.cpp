// GFX10 T#, S# and V# descriptors (src/gfx10_descriptors.h): every field at the bit position Mesa's register
// database gives it, round trips, what the translator and the hand-written test shaders read, the format table and
// the Vulkan mappings. Needs the Vulkan headers, not a GPU.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/gfx10_descriptors.h"

#include <cstdio>
#include <set>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

int main() {
    using namespace leonida::gfx10;
    using detail::get;

    // ---- T# ----
    // The default image (2D, 1x1, one mip, identity DST_SEL) sets only DST_SEL (X=4, Y=5, Z=6, W=7 at bits 11:0)
    // and TYPE (SQ_RSRC_IMG_2D = 9 at bits 31:28).
    const ImageWords plain = encode(Image{});
    CHECK((plain == ImageWords{0, 0, 0, 0x90000FAC, 0, 0, 0, 0}));
    // One field at a time: the bits that change, word by word.
    const auto changed = [&](const Image& image) {
        ImageWords words = encode(image);
        for (unsigned i = 0; i < 8; ++i) words[i] ^= plain[i];
        return words;
    };
    Image image{};
    image.address = 0xAB1234567800;  // BASE_ADDRESS = address >> 8, BASE_ADDRESS_HI = address >> 40
    CHECK((changed(image) == ImageWords{0x12345678, 0xAB, 0, 0, 0, 0, 0, 0}));
    image = {};
    image.format = 0x1FF;  // FORMAT, word 1 bits 28:20
    CHECK((changed(image) == ImageWords{0, 0x1FF00000, 0, 0, 0, 0, 0, 0}));
    image = {};
    image.width = 16384;  // width - 1: WIDTH_LO word 1 bits 31:30, WIDTH_HI word 2 bits 11:0
    CHECK((changed(image) == ImageWords{0, 0xC0000000, 0xFFF, 0, 0, 0, 0, 0}));
    image.width = 2;
    CHECK((changed(image) == ImageWords{0, 0x40000000, 0, 0, 0, 0, 0, 0}));
    image.width = 5;
    CHECK((changed(image) == ImageWords{0, 0, 1, 0, 0, 0, 0, 0}));
    image = {};
    image.height = 16384;  // height - 1: HEIGHT word 2 bits 27:14
    CHECK((changed(image) == ImageWords{0, 0, 0x0FFFC000, 0, 0, 0, 0, 0}));
    image = {};
    image.min_lod = 2.5f;  // MIN_LOD word 1 bits 19:8, 4.8 fixed
    CHECK((changed(image) == ImageWords{0, 0x28000, 0, 0, 0, 0, 0, 0}));
    image.min_lod = 99.0f;  // clamped to 15, as Mesa does
    CHECK((changed(image) == ImageWords{0, 0xF0000, 0, 0, 0, 0, 0, 0}));
    image = {};
    image.swizzle = {select_one, select_zero, select_w, select_x};  // DST_SEL_X/Y/Z/W, three bits each
    CHECK(encode(image)[3] == (0x90000000u | 1u | (0u << 3) | (7u << 6) | (4u << 9)));
    image = {};
    image.first_mip = 3;  // BASE_LEVEL word 3 bits 15:12, LAST_LEVEL bits 19:16, MAX_MIP word 5 bits 7:4
    image.mip_count = 5;
    image.resource_mips = 12;
    CHECK((changed(image) == ImageWords{0, 0, 0, 0x73000, 0, 0xB0, 0, 0}));
    image = {};
    image.type = ImageType::image_2d_array;  // TYPE 13; DEPTH = the last layer, BASE_ARRAY word 4 bits 28:16
    image.first_layer = 5;
    image.layer_count = 7;
    CHECK((changed(image) == ImageWords{0, 0, 0, 0x90000000u ^ 0xD0000000u, 0x0005000B, 0, 0, 0}));
    image = {};
    image.type = ImageType::image_3d;  // TYPE 10; DEPTH = depth - 1, word 4 bits 12:0
    image.depth = 8192;
    CHECK((changed(image) == ImageWords{0, 0, 0, 0x90000000u ^ 0xA0000000u, 0x1FFF, 0, 0, 0}));
    image = {};
    image.type = ImageType::image_2d_msaa;  // TYPE 14; log2(samples) in LAST_LEVEL and MAX_MIP
    image.samples = 8;
    CHECK((changed(image) == ImageWords{0, 0, 0, (0x90000000u ^ 0xE0000000u) | 0x30000, 0, 0x30, 0, 0}));
    CHECK(get(encode(Image{.type = ImageType::image_1d})[3], 28, 4) == 8);
    CHECK(get(encode(Image{.type = ImageType::cube, .layer_count = 6})[3], 28, 4) == 11);
    CHECK(get(encode(Image{.type = ImageType::image_1d_array})[3], 28, 4) == 12);
    CHECK(get(encode(Image{.type = ImageType::image_2d_msaa_array})[3], 28, 4) == 15);

    // The translator reads one field of a T#: FORMAT, with OpBitFieldUExtract word1 20 9.
    image = {};
    image.format = format_from_vulkan(VK_FORMAT_R32_UINT);
    CHECK(get(encode(image)[1], 20, 9) == 20);
    // The T#s cs_imgstore.asm and vk_store_test build have word 1 = FORMAT << 20: 8_8_8_8_UNORM, 32_UINT, 32_SINT.
    CHECK(get(0x3800000, 20, 9) == format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM));
    CHECK(get(0x1400000, 20, 9) == format_from_vulkan(VK_FORMAT_R32_UINT));
    CHECK(get(0x1500000, 20, 9) == format_from_vulkan(VK_FORMAT_R32_SINT));
    // Their other dwords are fillers (0x11, 0x22, 0x33), so by TYPE they are not image descriptors.
    const ImageWords test_shader{0x1000, 0x3800000, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11};
    CHECK(!is_image(test_shader) && !decode(test_shader, image));

    // Round trips, in the normal form decode() returns.
    const Image images[] = {
        {.address = 0x123456700, .format = 56, .width = 1920, .height = 1080, .mip_count = 11, .resource_mips = 11},
        {.address = 0xFFFFFFFFFF00, .format = 0x1FF, .width = 16384, .height = 16384, .first_mip = 15,
         .resource_mips = 16},
        {.format = 130, .type = ImageType::image_2d_array, .width = 64, .height = 32, .first_layer = 3,
         .layer_count = 7, .swizzle = channels_bgra},
        {.format = 71, .type = ImageType::cube, .width = 256, .height = 256, .first_layer = 6, .layer_count = 12,
         .first_mip = 2, .mip_count = 3, .resource_mips = 9},
        {.format = 1, .type = ImageType::image_3d, .width = 128, .height = 64, .depth = 32, .mip_count = 6,
         .resource_mips = 6, .swizzle = channels_r},
        {.format = 13, .type = ImageType::image_1d, .width = 4096},
        {.format = 13, .type = ImageType::image_1d_array, .width = 17, .layer_count = 8192},
        {.format = 56, .type = ImageType::image_2d_msaa, .width = 3840, .height = 2160, .samples = 4},
        {.format = 22, .type = ImageType::image_2d_msaa_array, .width = 8, .height = 8, .first_layer = 1,
         .layer_count = 2, .samples = 8},
        {.format = 181, .width = 1024, .height = 1024, .mip_count = 4, .resource_mips = 11, .min_lod = 1.25f},
    };
    for (const Image& in : images) {
        Image out{};
        const ImageWords words = encode(in);
        CHECK(is_image(words) && decode(words, out) && out == in);
    }

    // Bits this header does not encode are ignored: SW_MODE and BC_SWIZZLE (word 3), PITCH_MSB (word 4), PERF_MOD
    // and the rest of word 5, the compression and metadata fields of words 6 and 7, RESOURCE_LEVEL (word 2 bit 31).
    ImageWords busy = encode(images[3]);
    busy[2] |= 0x80000000;
    busy[3] |= 0x0FF00000;
    busy[4] |= 0x00002000;
    busy[5] |= 0xFFFFFF0F;
    busy[6] = busy[7] = 0xFFFFFFFF;
    Image decoded{};
    CHECK(decode(busy, decoded) && decoded == images[3]);

    // Refusals. A failed decode leaves its output alone.
    const Image untouched = decoded;
    ImageWords bad = encode(images[0]);
    bad[3] &= 0x0FFFFFFF;  // TYPE 0: a buffer or nothing
    CHECK(!is_image(bad) && !decode(bad, decoded));
    bad[3] |= 0x70000000;  // TYPE 7: reserved
    CHECK(!is_image(bad) && !decode(bad, decoded));
    bad = encode(images[0]);
    bad[3] = (bad[3] & ~0x7u) | 2;  // DST_SEL_X = 2: reserved
    CHECK(!decode(bad, decoded));
    bad = encode(images[3]);
    bad[3] &= ~0xF0000u;  // LAST_LEVEL 0 below BASE_LEVEL 2
    CHECK(!decode(bad, decoded));
    bad = encode(images[2]);
    bad[4] &= ~0x1FFFu;  // the last layer (0) before the first (3)
    CHECK(!decode(bad, decoded));
    bad = encode(images[3]);
    bad[4] -= 1;  // eleven faces
    CHECK(!decode(bad, decoded));
    bad = encode(images[7]);
    bad[3] |= 0x1000;  // a multisampled image with a base level
    CHECK(!decode(bad, decoded));
    CHECK(decoded == untouched);

    // The view type translated shaders use; a cube is a 2D array there.
    CHECK(vulkan_view_type(ImageType::image_1d) == VK_IMAGE_VIEW_TYPE_1D);
    CHECK(vulkan_view_type(ImageType::image_1d_array) == VK_IMAGE_VIEW_TYPE_1D_ARRAY);
    CHECK(vulkan_view_type(ImageType::image_2d) == VK_IMAGE_VIEW_TYPE_2D);
    CHECK(vulkan_view_type(ImageType::image_2d_msaa) == VK_IMAGE_VIEW_TYPE_2D);
    CHECK(vulkan_view_type(ImageType::image_2d_array) == VK_IMAGE_VIEW_TYPE_2D_ARRAY);
    CHECK(vulkan_view_type(ImageType::image_2d_msaa_array) == VK_IMAGE_VIEW_TYPE_2D_ARRAY);
    CHECK(vulkan_view_type(ImageType::cube) == VK_IMAGE_VIEW_TYPE_2D_ARRAY);
    CHECK(vulkan_view_type(ImageType::image_3d) == VK_IMAGE_VIEW_TYPE_3D);

    // ---- formats ----
    // The table: ascending codes that fit FORMAT's nine bits, each Vulkan format once, the first row of a code in
    // the hardware's own channel order (X is red) wherever Vulkan has such a format.
    std::set<VkFormat> seen;
    std::uint32_t previous = 0;
    for (const FormatRow& row : format_table) {
        CHECK(row.gfx10 >= previous && row.gfx10 > 0 && row.gfx10 < 512);
        CHECK(seen.insert(row.vulkan).second);
        CHECK(valid_swizzle(row.channels) && row.channels[0] != select_zero && row.channels[0] != select_one);
        CHECK(format_from_vulkan(row.vulkan) == row.gfx10);
        CHECK(format_swizzle(row.vulkan) == row.channels);
        if (row.gfx10 != previous) {
            CHECK(vulkan_format(row.gfx10) == row.vulkan);
            CHECK(row.channels[0] == select_x || row.vulkan == VK_FORMAT_A1R5G5B5_UNORM_PACK16);
        }
        previous = row.gfx10;
    }
    CHECK(vulkan_format(56) == VK_FORMAT_R8G8B8A8_UNORM && vulkan_format(130) == VK_FORMAT_R8G8B8A8_SRGB);
    CHECK(vulkan_format(22) == VK_FORMAT_R32_SFLOAT && format_from_vulkan(VK_FORMAT_D32_SFLOAT) == 22);
    CHECK(vulkan_format(7) == VK_FORMAT_R16_UNORM && format_from_vulkan(VK_FORMAT_D16_UNORM) == 7);
    CHECK(vulkan_format(141) == VK_FORMAT_D24_UNORM_S8_UINT && vulkan_format(146) == VK_FORMAT_D32_SFLOAT_S8_UINT);
    CHECK(vulkan_format(169) == VK_FORMAT_BC1_RGBA_UNORM_BLOCK && vulkan_format(182) == VK_FORMAT_BC7_SRGB_BLOCK);
    CHECK(vulkan_format(0) == VK_FORMAT_UNDEFINED && vulkan_format(511) == VK_FORMAT_UNDEFINED);
    CHECK(vulkan_format(30) == VK_FORMAT_UNDEFINED);  // 10_11_11_UNORM: no Vulkan format
    CHECK(format_from_vulkan(VK_FORMAT_UNDEFINED) == 0 && format_from_vulkan(VK_FORMAT_R64_SFLOAT) == 0);
    // Channel order is DST_SEL's, not the format's: BGRA8 is 8_8_8_8 read as Z, Y, X, W.
    CHECK(format_from_vulkan(VK_FORMAT_B8G8R8A8_UNORM) == 56 && format_from_vulkan(VK_FORMAT_B8G8R8A8_SRGB) == 130);
    CHECK((format_swizzle(VK_FORMAT_B8G8R8A8_UNORM) == Swizzle{select_z, select_y, select_x, select_w}));
    CHECK((format_swizzle(VK_FORMAT_R8G8B8A8_UNORM) == identity_swizzle));
    CHECK((format_swizzle(VK_FORMAT_R8_UNORM) == Swizzle{select_x, select_zero, select_zero, select_one}));
    CHECK((format_swizzle(VK_FORMAT_R16G16_SFLOAT) == Swizzle{select_x, select_y, select_zero, select_one}));
    CHECK((format_swizzle(VK_FORMAT_B10G11R11_UFLOAT_PACK32) == Swizzle{select_x, select_y, select_z, select_one}));
    // Packed formats: the hardware's X is the lowest bits. R5G6B5_PACK16 has blue there, B5G6R5_PACK16 red.
    CHECK((format_swizzle(VK_FORMAT_R5G6B5_UNORM_PACK16) == Swizzle{select_z, select_y, select_x, select_one}));
    CHECK((format_swizzle(VK_FORMAT_B5G6R5_UNORM_PACK16) == Swizzle{select_x, select_y, select_z, select_one}));
    CHECK((format_swizzle(VK_FORMAT_A2R10G10B10_UNORM_PACK32) == channels_bgra));
    CHECK((format_swizzle(VK_FORMAT_A2B10G10R10_UNORM_PACK32) == channels_rgba));
    CHECK((format_swizzle(VK_FORMAT_UNDEFINED) == identity_swizzle));

    // The view format for a T#'s code on an image: the image's order is kept.
    CHECK(vulkan_view_format(VK_FORMAT_R8G8B8A8_UNORM, 56) == VK_FORMAT_R8G8B8A8_UNORM);
    CHECK(vulkan_view_format(VK_FORMAT_B8G8R8A8_UNORM, 130) == VK_FORMAT_B8G8R8A8_SRGB);
    CHECK(vulkan_view_format(VK_FORMAT_R8G8B8A8_UNORM, 60) == VK_FORMAT_R8G8B8A8_UINT);
    CHECK(vulkan_view_format(VK_FORMAT_D32_SFLOAT, 22) == VK_FORMAT_D32_SFLOAT);
    CHECK(vulkan_view_format(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 170) == VK_FORMAT_BC1_RGBA_SRGB_BLOCK);
    CHECK(vulkan_view_format(VK_FORMAT_R8G8B8A8_UNORM, 22) == VK_FORMAT_UNDEFINED);
    CHECK(vulkan_view_format(VK_FORMAT_R8G8B8A8_UNORM, 22, true) == VK_FORMAT_R32_SFLOAT);
    CHECK(vulkan_view_format(VK_FORMAT_R8G8B8A8_UINT, 20, true) == VK_FORMAT_R32_UINT);
    CHECK(vulkan_view_format(VK_FORMAT_UNDEFINED, 56) == VK_FORMAT_UNDEFINED);

    // Depth/stencil images are only compatible with their own format, even with allow_channel_cast = true,
    // and colour images never resolve to a depth/stencil view format.
    for (const VkFormat ds : {VK_FORMAT_D16_UNORM, VK_FORMAT_X8_D24_UNORM_PACK32, VK_FORMAT_D32_SFLOAT,
                              VK_FORMAT_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT,
                              VK_FORMAT_D32_SFLOAT_S8_UINT})
        CHECK(is_depth_stencil_format(ds));
    for (const VkFormat colour : {VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT,
                                  VK_FORMAT_R8_UINT, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED})
        CHECK(!is_depth_stencil_format(colour));
    for (const bool cast : {false, true}) {
        CHECK(vulkan_view_format(VK_FORMAT_D16_UNORM, 7, cast) == VK_FORMAT_D16_UNORM);
        for (const std::uint32_t bad_d16 : {8u, 11u, 12u, 13u, 1u, 20u, 22u, 56u, 141u, 146u})
            CHECK(vulkan_view_format(VK_FORMAT_D16_UNORM, bad_d16, cast) == VK_FORMAT_UNDEFINED);
        CHECK(vulkan_view_format(VK_FORMAT_D32_SFLOAT, 22, cast) == VK_FORMAT_D32_SFLOAT);
        for (const std::uint32_t bad_d32 : {20u, 21u, 7u, 13u, 56u, 141u, 146u})
            CHECK(vulkan_view_format(VK_FORMAT_D32_SFLOAT, bad_d32, cast) == VK_FORMAT_UNDEFINED);
        CHECK(vulkan_view_format(VK_FORMAT_D24_UNORM_S8_UINT, 141, cast) == VK_FORMAT_D24_UNORM_S8_UINT);
        CHECK(vulkan_view_format(VK_FORMAT_X8_D24_UNORM_PACK32, 141, cast) == VK_FORMAT_X8_D24_UNORM_PACK32);
        CHECK(vulkan_view_format(VK_FORMAT_D32_SFLOAT_S8_UINT, 146, cast) == VK_FORMAT_D32_SFLOAT_S8_UINT);
        CHECK(vulkan_view_format(VK_FORMAT_S8_UINT, 5, cast) == VK_FORMAT_S8_UINT);
        for (const std::uint32_t bad_ds : {1u, 5u, 7u, 13u, 20u, 21u, 22u, 56u}) {
            CHECK(vulkan_view_format(VK_FORMAT_D24_UNORM_S8_UINT, bad_ds, cast) == VK_FORMAT_UNDEFINED);
            CHECK(vulkan_view_format(VK_FORMAT_D32_SFLOAT_S8_UINT, bad_ds, cast) == VK_FORMAT_UNDEFINED);
        }
        CHECK(vulkan_view_format(VK_FORMAT_D24_UNORM_S8_UINT, 146, cast) == VK_FORMAT_UNDEFINED);
        CHECK(vulkan_view_format(VK_FORMAT_D32_SFLOAT_S8_UINT, 141, cast) == VK_FORMAT_UNDEFINED);
        // Colour images never resolve to a depth/stencil view format, while normal colour casts remain intact.
        CHECK(vulkan_view_format(VK_FORMAT_R32_SFLOAT, 141, cast) == VK_FORMAT_UNDEFINED);
        CHECK(vulkan_view_format(VK_FORMAT_R32_SFLOAT, 146, cast) == VK_FORMAT_UNDEFINED);
        CHECK(vulkan_view_format(VK_FORMAT_R8G8B8A8_UNORM, 141, cast) == VK_FORMAT_UNDEFINED);
        CHECK(vulkan_view_format(VK_FORMAT_R8G8B8A8_UNORM, 146, cast) == VK_FORMAT_UNDEFINED);
        CHECK(vulkan_view_format(VK_FORMAT_R16_UNORM, 13, cast) == VK_FORMAT_R16_SFLOAT);
        CHECK(vulkan_view_format(VK_FORMAT_R32_SFLOAT, 20, cast) == VK_FORMAT_R32_UINT);
    }
    // Exhaustive invariant across the whole format table: depth/stencil images only accept their own code,
    // and colour images never produce a depth/stencil view format.
    for (const FormatRow& img : format_table) {
        for (const FormatRow& view : format_table) {
            for (const bool cast : {false, true}) {
                const VkFormat resolved = vulkan_view_format(img.vulkan, view.gfx10, cast);
                if (is_depth_stencil_format(img.vulkan)) {
                    CHECK(resolved == (img.gfx10 == view.gfx10 ? img.vulkan : VK_FORMAT_UNDEFINED));
                } else {
                    CHECK(!is_depth_stencil_format(resolved));
                }
            }
        }
    }

    // DST_SEL as a component mapping.
    const auto same = [](const VkComponentMapping& m, VkComponentSwizzle r, VkComponentSwizzle g, VkComponentSwizzle b,
                         VkComponentSwizzle a) { return m.r == r && m.g == g && m.b == b && m.a == a; };
    constexpr auto R = VK_COMPONENT_SWIZZLE_R, G = VK_COMPONENT_SWIZZLE_G, B = VK_COMPONENT_SWIZZLE_B,
                   A = VK_COMPONENT_SWIZZLE_A, zero = VK_COMPONENT_SWIZZLE_ZERO, one = VK_COMPONENT_SWIZZLE_ONE;
    CHECK(same(vulkan_swizzle(identity_swizzle), R, G, B, A));
    CHECK(same(vulkan_swizzle({select_w, select_zero, select_one, select_x}), A, zero, one, R));
    // A BGRA view whose T# carries the format's own swizzle shows its channels as they are.
    CHECK(same(vulkan_swizzle(channels_bgra, VK_FORMAT_B8G8R8A8_UNORM), R, G, B, A));
    CHECK(same(vulkan_swizzle(identity_swizzle, VK_FORMAT_B8G8R8A8_UNORM), B, G, R, A));
    CHECK(same(vulkan_swizzle(channels_r, VK_FORMAT_R8_UNORM), R, zero, zero, one));
    CHECK(same(vulkan_swizzle({select_x, select_x, select_x, select_y}, VK_FORMAT_R8G8_UNORM), R, R, R, G));
    CHECK(same(vulkan_swizzle({select_y, select_w, select_x, select_one}, VK_FORMAT_R8_UNORM), zero, zero, R, one));

    // ---- S# ----
    CHECK((encode(Sampler{}) == SamplerWords{0, 0, 0, 0}));
    const auto one_field = [](const Sampler& s) { return encode(s); };
    CHECK((one_field({.address_u = AddressMode::clamp_border}) == SamplerWords{6, 0, 0, 0}));        // CLAMP_X
    CHECK((one_field({.address_v = AddressMode::mirror}) == SamplerWords{1u << 3, 0, 0, 0}));        // CLAMP_Y
    CHECK((one_field({.address_w = AddressMode::mirror_once_border}) == SamplerWords{7u << 6, 0, 0, 0}));
    CHECK((one_field({.max_anisotropy = 16}) == SamplerWords{4u << 9, 0, 0, 0}));  // MAX_ANISO_RATIO = log2
    CHECK((one_field({.max_anisotropy = 8}) == SamplerWords{3u << 9, 0, 0, 0}));
    CHECK((one_field({.max_anisotropy = 2}) == SamplerWords{1u << 9, 0, 0, 0}));
    CHECK((one_field({.max_anisotropy = 64}) == SamplerWords{4u << 9, 0, 0, 0}));  // at most 16x, as radv
    CHECK((one_field({.max_anisotropy = 0}) == SamplerWords{0, 0, 0, 0}));
    CHECK((one_field({.compare = CompareFunction::less_equal}) == SamplerWords{3u << 12, 0, 0, 0}));
    CHECK((one_field({.compare = CompareFunction::always}) == SamplerWords{7u << 12, 0, 0, 0}));
    CHECK((one_field({.compare_never = true}) == SamplerWords{0, 0, 0, 1u << 12}));
    CHECK(!decode(SamplerWords{0, 0, 1u << 31, 1u << 12}).compare_never);  // alternate WORD3 layout
    CHECK((one_field({.unnormalized = true}) == SamplerWords{1u << 15, 0, 0, 0}));  // FORCE_UNNORMALIZED
    CHECK((one_field({.filter_mode = FilterMode::maximum}) == SamplerWords{2u << 29, 0, 0, 0}));
    CHECK((one_field({.min_lod = 1.5f}) == SamplerWords{0, 0x180, 0, 0}));          // MIN_LOD, 4.8 fixed
    CHECK((one_field({.max_lod = 15.0f}) == SamplerWords{0, 0xF00u << 12, 0, 0}));  // MAX_LOD
    CHECK((one_field({.max_lod = 1000.0f}) == SamplerWords{0, 0xF00u << 12, 0, 0}));
    CHECK((one_field({.min_lod = -4.0f}) == SamplerWords{0, 0, 0, 0}));
    CHECK((one_field({.lod_bias = 1.0f}) == SamplerWords{0, 0, 0x100, 0}));  // LOD_BIAS, signed 5.8 fixed, 14 bits
    CHECK((one_field({.lod_bias = -1.5f}) == SamplerWords{0, 0, 0x3E80, 0}));
    CHECK((one_field({.lod_bias = 100.0f}) == SamplerWords{0, 0, 0x1F00, 0}));
    CHECK((one_field({.lod_bias = -100.0f}) == SamplerWords{0, 0, 0x2000, 0}));
    CHECK((one_field({.mag_filter = Filter::bilinear}) == SamplerWords{0, 0, 1u << 20, 0}));
    CHECK((one_field({.min_filter = Filter::aniso_bilinear}) == SamplerWords{0, 0, 3u << 22, 0}));
    CHECK((one_field({.mip_filter = MipFilter::linear}) == SamplerWords{0, 0, 2u << 26, 0}));
    CHECK((one_field({.border_color = BorderColor::opaque_white}) == SamplerWords{0, 0, 0, 2u << 30}));
    CHECK((one_field({.border_color = BorderColor::table, .border_color_index = 0xABC}) ==
           SamplerWords{0, 0, 0, (3u << 30) | 0xABC}));

    // The words radv writes for a trilinear, repeating, 16x anisotropic sampler with LOD 0..15
    // (radv_make_sampler_descriptor and ac_build_sampler_descriptor): the fields above, and tuning fields this
    // header neither writes nor reads: ANISO_THRESHOLD = 4 >> 1 (bits 18:16) and ANISO_BIAS = 4 (bits 26:21) in
    // word 0, PERF_MIP = 4 + 6 (bits 27:24) in word 1, ANISO_OVERRIDE (bit 29) in word 2.
    const Sampler trilinear{.mag_filter = Filter::aniso_bilinear, .min_filter = Filter::aniso_bilinear,
                            .mip_filter = MipFilter::linear, .max_anisotropy = 16, .max_lod = 15.0f};
    const SamplerWords ours = encode(trilinear);
    CHECK((ours == SamplerWords{4u << 9, 0xF00u << 12, (3u << 20) | (3u << 22) | (2u << 26), 0}));
    const SamplerWords radv{ours[0] | (2u << 16) | (4u << 21), ours[1] | (10u << 24), ours[2] | (1u << 29), 0};
    CHECK(decode(radv) == trilinear && decode(ours) == trilinear);
    // A shadow sampler: bilinear, clamp to an opaque white border, compare less-or-equal, no mips beyond the base.
    const Sampler shadow{.address_u = AddressMode::clamp_border, .address_v = AddressMode::clamp_border,
                         .address_w = AddressMode::clamp_border, .mag_filter = Filter::bilinear,
                         .min_filter = Filter::bilinear, .compare = CompareFunction::less_equal,
                         .border_color = BorderColor::opaque_white};
    CHECK((encode(shadow) == SamplerWords{0x31B6, 0, (1u << 20) | (1u << 22), 2u << 30}));
    // radv sets TRUNC_COORD (word 0 bit 27) on a point sampler; it is ignored too.
    const Sampler point{.address_u = AddressMode::clamp_last_texel, .address_v = AddressMode::clamp_last_texel,
                        .address_w = AddressMode::clamp_last_texel, .mip_filter = MipFilter::point, .min_lod = 0.5f,
                        .max_lod = 4.25f, .lod_bias = -0.75f, .unnormalized = false};
    SamplerWords truncating = encode(point);
    truncating[0] |= 1u << 27;
    CHECK(decode(truncating) == point);
    for (const Sampler& in : {Sampler{}, trilinear, shadow, point,
                              Sampler{.address_u = AddressMode::mirror_once_last_texel, .max_anisotropy = 4,
                                      .lod_bias = 31.0f, .border_color = BorderColor::table,
                                      .border_color_index = 4095, .unnormalized = true,
                                      .filter_mode = FilterMode::minimum}})
        CHECK(decode(encode(in)) == in);
    // MAX_ANISO_RATIO above 4 reads as 16x.
    CHECK(decode(SamplerWords{7u << 9, 0, 0, 0}).max_anisotropy == 16);
    CHECK(decode(SamplerWords{0, 0, 0x2000, 0}).lod_bias == -32.0f);

    // To Vulkan: the inverse of radv's radv_tex_wrap, radv_tex_filter, radv_tex_mipfilter and radv_tex_compare.
    VkSamplerCreateInfo info = vulkan_sampler(trilinear);
    CHECK(info.sType == VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO && info.pNext == nullptr);
    CHECK(info.magFilter == VK_FILTER_LINEAR && info.minFilter == VK_FILTER_LINEAR);
    CHECK(info.mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR && info.addressModeU == VK_SAMPLER_ADDRESS_MODE_REPEAT);
    CHECK(info.anisotropyEnable == VK_TRUE && info.maxAnisotropy == 16.0f && info.compareEnable == VK_FALSE);
    CHECK(info.minLod == 0.0f && info.maxLod == 15.0f && info.unnormalizedCoordinates == VK_FALSE);
    info = vulkan_sampler(shadow);
    CHECK(info.compareEnable == VK_TRUE && info.compareOp == VK_COMPARE_OP_LESS_OR_EQUAL);
    CHECK(info.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER &&
          info.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
    CHECK(info.borderColor == VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE && info.anisotropyEnable == VK_FALSE);
    // MIP_FILTER_NONE: the base level only.
    CHECK(info.mipmapMode == VK_SAMPLER_MIPMAP_MODE_NEAREST && info.minLod == 0.0f && info.maxLod == 0.25f);
    CHECK(vulkan_sampler({.lod_bias = -32.0f}, 15.0f).mipLodBias == -15.0f);
    CHECK(vulkan_sampler({.lod_bias = 31.0f}, 15.0f).mipLodBias == 15.0f);
    CHECK(vulkan_sampler({.lod_bias = 2.25f}, 15.0f).mipLodBias == 2.25f);
    info = vulkan_sampler(point);
    CHECK(info.magFilter == VK_FILTER_NEAREST && info.mipmapMode == VK_SAMPLER_MIPMAP_MODE_NEAREST);
    CHECK(info.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE && info.mipLodBias == -0.75f);
    CHECK(info.minLod == 0.5f && info.maxLod == 4.25f);
    info = vulkan_sampler({.mag_filter = Filter::aniso_point, .mip_filter = MipFilter::linear, .max_anisotropy = 8,
                           .compare = CompareFunction::greater, .max_lod = 9.0f, .unnormalized = true});
    CHECK(info.unnormalizedCoordinates == VK_TRUE && info.mipmapMode == VK_SAMPLER_MIPMAP_MODE_NEAREST);
    CHECK(info.maxLod == 0.0f && info.anisotropyEnable == VK_FALSE && info.compareEnable == VK_FALSE);
    // Unnormalised: U and V clamp (wrap, the default S# mode, would be invalid: VUID ...-01075), border stays border,
    // W is free; min and mag filter agree (VUID ...-01072).
    CHECK(info.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE &&
          info.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE &&
          info.addressModeW == VK_SAMPLER_ADDRESS_MODE_REPEAT && info.minFilter == info.magFilter);
    info = vulkan_sampler({.address_u = AddressMode::clamp_border, .address_v = AddressMode::mirror,
                           .min_filter = Filter::bilinear, .unnormalized = true});
    CHECK(info.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER &&
          info.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE && info.minFilter == VK_FILTER_NEAREST);
    CHECK(vulkan_address_mode(AddressMode::mirror) == VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);
    CHECK(vulkan_address_mode(AddressMode::mirror_once_last_texel) == VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE);
    CHECK(vulkan_address_mode(AddressMode::clamp_half_border) == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    info = vulkan_sampler({.border_color = BorderColor::opaque_black});
    CHECK(info.borderColor == VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK);
    info = vulkan_sampler({.border_color = BorderColor::table});  // the colour is not in the S#
    CHECK(info.borderColor == VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
    // An integer border: the layer's mark in BORDER_COLOR_PTR, which the hardware reads only for a table colour.
    for (const auto [type, want] : {std::pair{BorderColor::transparent_black, VK_BORDER_COLOR_INT_TRANSPARENT_BLACK},
                                    std::pair{BorderColor::opaque_black, VK_BORDER_COLOR_INT_OPAQUE_BLACK},
                                    std::pair{BorderColor::opaque_white, VK_BORDER_COLOR_INT_OPAQUE_WHITE}}) {
        const Sampler integer{.border_color = type, .integer_border = true};
        const SamplerWords words = encode(integer);
        CHECK(words[3] == ((std::uint32_t(type) << 30) | kIntegerBorderPtr));
        CHECK(decode(words) == integer && vulkan_sampler(integer).borderColor == want);
        CHECK(words != encode(Sampler{.border_color = type}));  // its own key, apart from the float border
    }
    // Not for a table colour, whose pointer is its index.
    CHECK((encode({.border_color = BorderColor::table, .border_color_index = 5, .integer_border = true}) ==
           SamplerWords{0, 0, 0, (3u << 30) | 5}));
    CHECK(!decode(SamplerWords{0, 0, 0, (3u << 30) | kIntegerBorderPtr}).integer_border);
    CHECK(vulkan_reduction_mode(FilterMode::blend) == VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE);
    CHECK(vulkan_reduction_mode(FilterMode::minimum) == VK_SAMPLER_REDUCTION_MODE_MIN);
    CHECK(vulkan_reduction_mode(FilterMode::maximum) == VK_SAMPLER_REDUCTION_MODE_MAX);

    // ---- V# ----
    CHECK((encode(Buffer{}) == BufferWords{0, 0, 0, 0xFAC}));
    Buffer buffer{.address = 0xABCD12345678, .stride = 0x3FFF, .num_records = 0xDEADBEEF, .format = 0x7F,
                  .swizzle = {select_y, select_one, select_zero, select_w}, .oob_select = OobSelect::raw};
    const BufferWords v = encode(buffer);
    CHECK(v[0] == 0x12345678);                     // BASE_ADDRESS
    CHECK(v[1] == (0xABCDu | (0x3FFFu << 16)));    // BASE_ADDRESS_HI [15:0], STRIDE [29:16]
    CHECK(v[2] == 0xDEADBEEF);                     // NUM_RECORDS
    CHECK(v[3] == ((5u | (1u << 3) | (0u << 6) | (7u << 9)) | (0x7Fu << 12) | (3u << 28)));  // DST_SEL, FORMAT, OOB
    // As tools/rdna2spirv.py reads it (buffer_address and the typed loads).
    CHECK(((std::uint64_t(v[1] & 0xFFFF) << 32) | v[0]) == buffer.address);
    CHECK(get(v[1], 16, 14) == buffer.stride && v[2] == buffer.num_records);
    CHECK(get(v[3], 12, 7) == buffer.format && (v[3] & 0xFFF) == (5u | (1u << 3) | (7u << 9)));
    Buffer read{};
    CHECK(decode(v, read) && read == buffer);
    buffer = {.address = 0x1000, .num_records = 256, .format = format_from_vulkan(VK_FORMAT_R32_SFLOAT)};
    CHECK(decode(encode(buffer), read) && read == buffer && read.stride == 0);
    // TYPE (bits 31:30) is 0 for a buffer; the first four dwords of a T# are refused, and so is a reserved DST_SEL.
    const Buffer kept = read;
    BufferWords wrong = v;
    wrong[3] |= 1u << 30;
    CHECK(!decode(wrong, read));
    const ImageWords texture = encode(images[0]);
    CHECK(!decode(BufferWords{texture[0], texture[1], texture[2], texture[3]}, read));
    wrong = v;
    wrong[3] = (wrong[3] & ~0x38u) | (3u << 3);
    CHECK(!decode(wrong, read) && read == kept);
    // And a V# in a 32-byte heap slot is not taken for an image.
    CHECK(!is_image(ImageWords{v[0], v[1], v[2], v[3], 0, 0, 0, 0}));

    std::printf("GFX10 descriptors: T#, S# and V# fields, round trips, %zu format rows and the Vulkan mappings OK\n",
                std::size(format_table));
    return 0;
}
