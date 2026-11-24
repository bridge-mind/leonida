"""T# (image resource descriptor) words per GPU family, and the raw kernel that reads a texture through one. No hardware
needed to build either; tools/image_fuzz.py --oracle hw runs the kernels.

The kernel carries its own texture: the texel rows sit in the kernel's code after s_endpgm, 256-byte aligned, and
the kernel finds them with s_getpc_b64, so the launcher (hw_oracle.c, hw_oracle_drm.c) needs no input buffer. The
texture is therefore in device memory the GPU already maps, linear (SW_MODE 0). The kernel writes the T# into
s[8:15] with the base address filled in at run time and runs
    image_load v[8:11], v[x:y], s[8:15] dmask:.. dim:SQ_RSRC_IMG_2D unorm
for one texel per lane (every texel is read once, 1024 lanes at most with a short loop beyond that) and stores the
four result registers of each texel as four dwords at texel_index * 16 in the output buffer.

Where the layout comes from (facts only; Mesa is MIT-licensed, the code here is this repository's own):
- field positions: Mesa src/amd/registers gfx10-rsrc.json (SQ_IMG_RSRC_WORD0..7 and the gfx103 variants of WORD4/5)
  and gfx11-rsrc.json;
- which value goes into which field: Mesa src/amd/common ac_descriptors.c (ac_build_gfx10_texture_descriptor,
  ac_set_mutable_tex_desc_fields: linear images on GFX10.3+ and GFX11 carry their pitch in WORD4 DEPTH and PITCH_MSB);
- format numbers: Mesa's GFX10_FORMAT enum (tools/gfx10_formats.json holds it too) and, for GFX11, the order of
  LLVM's AMDGPU::UfmtGFX11::UnifiedFormat (SIDefines.h), which Mesa's GFX11_FORMAT follows.
GFX11 renumbers the formats (8_8_8_8_UNORM is 56 on GFX10 and 42 on GFX11), narrows FORMAT to [27:20], moves MAX_MIP
into WORD1 and drops WORD2's RESOURCE_LEVEL bit (GFX10 needs it set). A GFX11 part (the Radeon 880M is gfx1150, RDNA3.5)
runs these descriptors; RDNA2 (gfx1030) takes the GFX10 ones; a gfx1036 (RDNA2) ran them for tools/texoff_probe.py (2D and 3D, R32_SFLOAT).
"""
import re

try:
    from . import asm
except ImportError:   # run as a script
    import asm

# Vulkan format name -> FORMAT field value, per family. GFX12 renumbers again and is not built here.
IMG_FORMATS = {
    "gfx10": {"R8G8B8A8_UNORM": 56, "R16G16B16A16_SFLOAT": 71, "R32G32B32A32_SFLOAT": 77, "R32_SFLOAT": 22,
              "R8G8B8A8_UINT": 60, "R8G8B8A8_SINT": 61},
    "gfx11": {"R8G8B8A8_UNORM": 42, "R16G16B16A16_SFLOAT": 57, "R32G32B32A32_SFLOAT": 63, "R32_SFLOAT": 22,
              "R8G8B8A8_UINT": 46, "R8G8B8A8_SINT": 47},
}
SQ_RSRC_IMG_2D = 9
SEL_0, SEL_1, SEL_X, SEL_Y, SEL_Z, SEL_W = 0, 1, 4, 5, 6, 7   # SQ_SEL_*: DST_SEL values
PERF_MOD = 4                                                   # what Mesa writes into WORD5 [22:20]
SENTINEL_BITS = (0x3F800000, 0x40000000, 0x40400000, 0x40800000)   # 1.0 2.0 3.0 4.0, as image_fuzz.SENTINELS
POISON = 0xBAADF00D                                            # what a kernel stores when its texture is misaligned


def check_arch(arch):
    """The family whose T# layout is built, or ValueError (GFX12 has its own, not written here)."""
    fam = asm.family(arch)
    if fam not in IMG_FORMATS:
        raise ValueError(f"no T# layout for {fam} ({arch}); this builder knows {', '.join(IMG_FORMATS)}")
    return fam


def has_custom_pitch(arch):
    """Whether WORD4 can carry a linear image's pitch: GFX10.3 (gfx1030..gfx1036) and GFX11. GFX10.1 has no such
    field and takes the default pitch (width rounded up to 256 bytes), which pitch_texels also produces."""
    fam = check_arch(arch)
    return fam == "gfx11" or re.fullmatch(r"gfx103[0-9a-f]", arch) is not None


def pitch_texels(width, bytes_per_texel):
    """Row pitch in texels: the row size rounded up to 256 bytes (a linear image's pitch must be a multiple of it)."""
    row = -(-width * bytes_per_texel // 256) * 256
    return row // bytes_per_texel


def image_words(arch, fmt, width, height, bytes_per_texel, swizzle=(SEL_X, SEL_Y, SEL_Z, SEL_W)):
    """The 8 words of a 2D, one-mip, linear T# for gfx10/gfx11, with the base address (WORD0 and WORD1 [7:0]) left
    zero: the kernel adds them once it knows where its texture landed."""
    fam = check_arch(arch)
    if not (1 <= width <= 16384 and 1 <= height <= 16384):
        raise ValueError(f"size {width}x{height} is outside 1..16384")
    code = IMG_FORMATS[fam][fmt]
    w, h = width - 1, height - 1
    word1 = (code << 20) | ((w & 3) << 30)
    word2 = ((w >> 2) & 0xFFF) | (h << 14)
    if fam == "gfx10":
        word2 |= 1 << 31                    # RESOURCE_LEVEL, "must be 1" on GFX10 and 0 on GFX11
    word3 = sum(s << (3 * i) for i, s in enumerate(swizzle)) | (SQ_RSRC_IMG_2D << 28)   # levels 0, SW_MODE 0 = linear
    word4 = 0
    if has_custom_pitch(arch):
        pitch = pitch_texels(width, bytes_per_texel) - 1
        word4 = (pitch & 0x1FFF) | (((pitch >> 13) & 1) << 13)    # DEPTH holds the low bits, PITCH_MSB bit 13
    word5 = PERF_MOD << 20
    return [0, word1, word2, word3, word4, word5, 0, 0]


def lay_out(texels, width, height, bytes_per_texel):
    """Tight row-major texel bytes -> rows padded to pitch_texels (the padding is a pattern that is wrong to read)."""
    row = width * bytes_per_texel
    stride = pitch_texels(width, bytes_per_texel) * bytes_per_texel
    out = bytearray()
    for y in range(height):
        out += texels[y * row:(y + 1) * row] + b"\xcd" * (stride - row)
    return bytes(out)


def lanes(width, height):
    """(threads, iterations): one lane per texel up to a 1024-lane group, then each lane takes several."""
    count = width * height
    threads = min(count, 1024)
    return threads, count // threads


def _longs(data):
    data += b"\0" * (-len(data) % 4)
    words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
    return [f"  .long {', '.join(f'0x{w:08x}' for w in words[i:i + 8])}" for i in range(0, len(words), 8)]


def kernel_lines(arch, uid, fmt, width, height, bytes_per_texel, dmask, texture, swizzle=(SEL_X, SEL_Y, SEL_Z, SEL_W)):
    """The body of a raw kernel (asm.Kernel(..., raw=True): s[2:3] is the output pointer, v0 the lane). uid keeps the
    labels of kernels in one module apart. `texture` is the padded texel bytes from lay_out. GFX11 spellings; asm.py
    puts the GFX10 ones back for a gfx10 target. A texture that did not land 256-byte aligned makes every word of the
    output POISON instead of reading from a wrong base."""
    t = image_words(arch, fmt, width, height, bytes_per_texel, swizzle)
    threads, iterations = lanes(width, height)
    n = dmask.bit_count()
    dst = "v8" if n == 1 else f"v[8:{7 + n}]"
    log2w = width.bit_length() - 1
    if width & (width - 1):
        raise ValueError("the width must be a power of two")
    out = ["s_getpc_b64 s[4:5]",
           f".Lpc{uid}:",
           f"s_add_u32 s4, s4, .Ltex{uid}-.Lpc{uid}",
           "s_addc_u32 s5, s5, 0",
           "s_and_b32 s6, s4, 255",
           "s_lshr_b64 s[4:5], s[4:5], 8",
           "s_mov_b32 s8, s4",
           "s_and_b32 s9, s5, 255",
           f"s_or_b32 s9, s9, 0x{t[1]:08x}"]
    out += [f"s_mov_b32 s{8 + i}, 0x{t[i]:08x}" for i in range(2, 8)]
    for k in range(iterations):
        out += [f"v_add_nc_u32 v1, {k * threads}, v0",
                f"v_and_b32 v2, {width - 1}, v1",
                f"v_lshrrev_b32 v3, {log2w}, v1"]
        out += [f"v_mov_b32 v{8 + i}, 0x{SENTINEL_BITS[i]:08x}" for i in range(4)]
        out += ["s_cmp_lg_u32 s6, 0",
                f"s_cbranch_scc0 .Lok{uid}_{k}"]
        out += [f"v_mov_b32 v{8 + i}, 0x{POISON:08x}" for i in range(4)]
        out += [f"s_branch .Lst{uid}_{k}",
                f".Lok{uid}_{k}:",
                f"image_load {dst}, v[2:3], s[8:15] dmask:0x{dmask:x} dim:SQ_RSRC_IMG_2D unorm",
                "s_waitcnt vmcnt(0)",
                f".Lst{uid}_{k}:",
                "v_lshlrev_b32 v4, 4, v1",
                "global_store_b128 v4, v[8:11], s[2:3]"]
    out += ["s_endpgm", ".p2align 8", f".Ltex{uid}:"]
    return out + _longs(texture)
