r"""SHADER-539: measure on an AMD GPU (tools/hw_oracle) which mip level image_sample_l and image_sample_lz read under the
S#'s MIN_LOD / MAX_LOD / LOD_BIAS / MIP_FILTER and the T#'s MIN_LOD and BASE_LEVEL.

    python -I tools/sampler_lod_probe.py                # each row's level under both rules below (no GPU)
    python -I tools/sampler_lod_probe.py --oracle hw    # native AMD GPU, wave32 and wave64, compared with the rule

docs/d3d12_vulkan.md ("vk_sampler_test") says the S# LOD fields apply to an explicit LOD as Vulkan's minLod, maxLod and
mipLodBias do, taken from radv and not measured on RDNA2, and that the T#'s MIN_LOD (ResourceMinLODClamp) is untested.
Here a raw kernel samples a linear 64x64 R32_SFLOAT texture with seven levels, through a point min/mag filter, at
the top-left texel. The texture memory is a ramp, every dword holding its own index, so each sample returns the
dword offset of the texel it read: the start of the level it came from. The level starts are not assumed; a
calibration row per level (image_sample_l at LOD n, no clamp) finds them, and every other row's result is named by
that table.

Two rules are computed for every row:
  - rdna2_level, the rule the hardware rows are checked against: the model the gfx1036 measurement fits (see its
    docstring; an exact half level rounds up, and the T#'s MIN_LOD clamps the fractional LOD before that rounding);
  - vulkan_level, for comparison: Vulkan's rule for the sampler and view the runtime makes of the same S# and T#
    (runtime/src/gfx10_descriptors.h): nearest is ceil(lambda + 0.5) - 1, and VK_EXT_image_view_min_lod raises the
    integer level to floor(minLod). Rows where it differs are marked "Vulkan reads n"; that is not a failure.
DIFFERS means the hardware disagrees with the RDNA2 model, not with Vulkan.
Standard library only; the texture is built here, no game files.

Exit codes: 0 every row matches the RDNA2 model (not "the hardware equals Vulkan"), 1 a row differs, 2 harness
error, 77 no hardware oracle.
Usage: `python -I tools/sampler_lod_probe.py --help` lists the options.
"""
import argparse
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hw_oracle"))

W = 64
LEVELS = 7                 # 64x64 .. 1x1
RAMP_DWORDS = 16384        # more than the chain needs under any 256-byte level alignment
CLAMP_LAST_TEXEL = 2
U = V = 0.25 / W           # inside texel (0, 0) at every level


# (name, instruction, LOD, S# (mip_filter, min_lod, max_lod, bias), T# (base_level, min_lod))
def rows():
    plain = (1, 0.0, 15.0, 0.0)
    out = [(f"calib_lod{n}", "l", float(n), plain, (0, 0.0)) for n in range(LEVELS)]
    out += [
        ("lz",                 "lz", 0.0, plain, (0, 0.0)),
        # S# MIN_LOD / MAX_LOD on an explicit LOD and on _lz
        ("s_min2_l0",          "l",  0.0, (1, 2.0, 15.0, 0.0), (0, 0.0)),
        ("s_min2_lz",          "lz", 0.0, (1, 2.0, 15.0, 0.0), (0, 0.0)),
        ("s_max1_l4",          "l",  4.0, (1, 0.0, 1.0, 0.0), (0, 0.0)),
        ("s_max0_l3",          "l",  3.0, (1, 0.0, 0.0, 0.0), (0, 0.0)),
        # S# LOD_BIAS on an explicit LOD and on _lz
        ("s_bias+2_l1",        "l",  1.0, (1, 0.0, 15.0, 2.0), (0, 0.0)),
        ("s_bias+2_lz",        "lz", 0.0, (1, 0.0, 15.0, 2.0), (0, 0.0)),
        ("s_bias-1_l3",        "l",  3.0, (1, 0.0, 15.0, -1.0), (0, 0.0)),
        ("s_bias+2_max3_l3",   "l",  3.0, (1, 0.0, 3.0, 2.0), (0, 0.0)),    # bias before the clamp
        ("s_bias+0.5_l1",      "l",  1.0, (1, 0.0, 15.0, 0.5), (0, 0.0)),   # 1.5: the nearest-level rounding
        # point mip filter rounding of a fractional LOD
        ("l1.25",              "l",  1.25, plain, (0, 0.0)),
        ("l1.5",               "l",  1.5, plain, (0, 0.0)),
        ("l1.75",              "l",  1.75, plain, (0, 0.0)),
        # MIP_FILTER none: the base level whatever the LOD and clamp
        ("mipnone_l3",         "l",  3.0, (0, 0.0, 15.0, 0.0), (0, 0.0)),
        ("mipnone_min2_l0",    "l",  0.0, (0, 2.0, 15.0, 0.0), (0, 0.0)),
        ("mipnone_t_min3_l0",  "l",  0.0, (0, 0.0, 15.0, 0.0), (0, 3.0)),   # the T#'s MIN_LOD still applies
        ("mipnone_t_min2.25",  "l",  0.0, (0, 0.0, 15.0, 0.0), (0, 2.25)),
        ("mipnone_t_min2.5",   "l",  0.0, (0, 0.0, 15.0, 0.0), (0, 2.5)),
        ("mipnone_base1_tmin3", "l", 0.0, (0, 0.0, 15.0, 0.0), (1, 3.0)),
        # T# MIN_LOD (ResourceMinLODClamp)
        ("t_min3_l1",          "l",  1.0, plain, (0, 3.0)),
        ("t_min3_lz",          "lz", 0.0, plain, (0, 3.0)),
        ("t_min3_l5",          "l",  5.0, plain, (0, 3.0)),                 # above the clamp: unchanged
        ("t_min2.5_l0",        "l",  0.0, plain, (0, 2.5)),                 # fractional clamp, point mip
        ("t_min2.5_l2.6",      "l",  2.6, plain, (0, 2.5)),
        ("t_min2.25_l0",       "l",  0.0, plain, (0, 2.25)),                # ceil(MIN_LOD), or clamp then round?
        ("t_min2.75_l0",       "l",  0.0, plain, (0, 2.75)),
        ("t_min2.25_lz",       "lz", 0.0, plain, (0, 2.25)),
        # more half-way points: is an exact .5 rounded up wherever it comes from?
        ("l0.5",               "l",  0.5, plain, (0, 0.0)),
        ("l2.5",               "l",  2.5, plain, (0, 0.0)),
        ("s_min1.5_l0",        "l",  0.0, (1, 1.5, 15.0, 0.0), (0, 0.0)),
        ("s_max1.5_l4",        "l",  4.0, (1, 0.0, 1.5, 0.0), (0, 0.0)),
        ("l1.49",              "l",  1.49, plain, (0, 0.0)),
        # BASE_LEVEL: S# fields count from the view's base, the T# MIN_LOD from the resource's level 0
        ("base1_l0",           "l",  0.0, plain, (1, 0.0)),
        ("base1_s_min1_l0",    "l",  0.0, (1, 1.0, 15.0, 0.0), (1, 0.0)),
        ("base1_t_min2_l0",    "l",  0.0, plain, (1, 2.0)),
        ("base1_t_min2_lz",    "lz", 0.0, plain, (1, 2.0)),
    ]
    return out


def vulkan_level(instr, lod, s, t):
    """Vulkan's level for the sampler and view the runtime makes of this S# and T#: nearest is ceil(lambda + 0.5) - 1,
    and VK_EXT_image_view_min_lod raises the integer level to floor(minLod)."""
    mip_filter, s_min, s_max, bias = s
    base, t_min = t
    last = LEVELS - 1
    if mip_filter == 0:
        level = base
    else:
        lam = (0.0 if instr == "lz" else lod) + bias
        lam = min(max(lam, s_min), s_max)
        d = max(0, math.ceil(lam + 0.5) - 1)
        level = min(base + d, last)
    return min(max(level, math.floor(t_min)), last)


def rdna2_level(instr, lod, s, t):
    """The model the gfx1036 measurement fits: lambda = LOD (0 for _lz) + LOD_BIAS, clamped to the S#'s
    [MIN_LOD, MAX_LOD]; with a point mip filter the T#'s MIN_LOD raises the still fractional LOD (counted from the
    resource's level 0) and the level is floor(lambda + 0.5): an exact .5 goes up. MIP_FILTER none ignores the S#'s
    LOD fields and reads the view's base level, raised to floor(T# MIN_LOD)."""
    mip_filter, s_min, s_max, bias = s
    base, t_min = t
    last = LEVELS - 1
    if mip_filter == 0:
        return min(max(base, math.floor(t_min)), last)
    lam = (0.0 if instr == "lz" else lod) + bias
    lam = base + min(max(lam, s_min), s_max)
    lam = max(lam, t_min)
    return min(max(math.floor(lam + 0.5), base), last)


rule_level = rdna2_level


def fixed(value, frac_bits, total_bits, signed=False):
    v = round(value * (1 << frac_bits))
    return v & ((1 << total_bits) - 1) if signed else max(0, min(v, (1 << total_bits) - 1))


def sampler_words(s):
    """S# positions as check.py's ssharp_words (Mesa gfx10-rsrc.json): WORD0 CLAMP_X/Y/Z [8:0], WORD1 MIN_LOD [11:0] and
    MAX_LOD [23:12] (4.8), WORD2 LOD_BIAS [13:0] (signed 5.8) and MIP_FILTER [27:26]. Point min/mag (0)."""
    mip_filter, s_min, s_max, bias = s
    clamp = CLAMP_LAST_TEXEL | CLAMP_LAST_TEXEL << 3 | CLAMP_LAST_TEXEL << 6
    return [clamp, fixed(s_min, 8, 12) | fixed(s_max, 8, 12) << 12, fixed(bias, 8, 14, True) | mip_filter << 26, 0]


def image_words(arch, t):
    """A linear 2D R32_SFLOAT T# with LEVELS levels. Mip fields as runtime/src/gfx10_descriptors.h has them: WORD1
    MIN_LOD [19:8] (4.8), WORD3 BASE_LEVEL [15:12] and LAST_LEVEL [19:16], WORD5 MAX_MIP [7:4]."""
    import tsharp
    if tsharp.check_arch(arch) != "gfx10":
        raise ValueError(f"the mip fields here are GFX10's; {arch} is not GFX10")
    base, t_min = t
    w = tsharp.image_words(arch, "R32_SFLOAT", W, W, 4)
    w[1] |= fixed(t_min, 8, 12) << 8
    w[3] |= base << 12 | (LEVELS - 1) << 16
    w[5] |= (LEVELS - 1) << 4
    return w


def f32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def kernel(arch, table):
    import tsharp
    out = ["s_getpc_b64 s[4:5]", ".Lpc:", "s_add_u32 s4, s4, .Ltex-.Lpc", "s_addc_u32 s5, s5, 0",
           "s_and_b32 s6, s4, 255", "s_lshr_b64 s[4:5], s[4:5], 8", "s_and_b32 s7, s5, 255"]
    for i, (_name, instr, lod, s, t) in enumerate(table):
        tw, sw = image_words(arch, t), sampler_words(s)
        out += ["s_mov_b32 s8, s4", f"s_or_b32 s9, s7, 0x{tw[1]:08x}"]
        out += [f"s_mov_b32 s{8 + k}, 0x{tw[k]:08x}" for k in range(2, 8)]
        out += [f"s_mov_b32 s{16 + k}, 0x{sw[k]:08x}" for k in range(4)]
        out += [f"v_mov_b32 v1, 0x{f32(U):08x}", f"v_mov_b32 v2, 0x{f32(V):08x}"]
        if instr == "l":
            out += [f"v_mov_b32 v3, 0x{f32(lod):08x}",
                    "image_sample_l v8, v[1:3], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D"]
        else:
            out += ["image_sample_lz v8, v[1:2], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D"]
        out += ["s_waitcnt vmcnt(0)", "s_cmp_lg_u32 s6, 0", f"s_cbranch_scc0 .Lok{i}",
                f"v_mov_b32 v8, 0x{tsharp.POISON:08x}", f".Lok{i}:",
                f"v_mov_b32 v5, {4 * i}", "global_store_b32 v5, v8, s[2:3]"]
    out += ["s_waitcnt vmcnt(0)", "s_endpgm", ".p2align 8", ".Ltex:"]
    ramp = b"".join(struct.pack("<f", float(i)) for i in range(RAMP_DWORDS))
    return out + tsharp._longs(ramp)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", choices=("rule", "hw"), default="rule")
    a = ap.parse_args()
    table = rows()
    if a.oracle == "rule":
        for name, instr, lod, s, t in table:
            r, v = rule_level(instr, lod, s, t), vulkan_level(instr, lod, s, t)
            print(f"{name:18} rule level {r}  vulkan {v}{'' if r == v else '  (Vulkan differs)'}")
        return 0
    import asm
    import runner
    import tsharp
    try:
        oracle = runner.open_oracle()
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77
    print(f"device: {oracle.device}")
    got = {}
    for wave in (32, 64):
        try:
            k = asm.Kernel(kernel(oracle.arch, table), wave=wave, threads=1, words=len(table), raw=True,
                           vgprs=16, sgprs=24)
        except ValueError as e:
            print(f"cannot build the kernel: {e}")
            return 2
        res = oracle.run([k])[0]
        if isinstance(res, str):
            print(f"wave{wave}: kernel rejected: {res}")
            return 2
        got[wave] = [struct.unpack("<f", struct.pack("<I", w))[0] if w != tsharp.POISON else None for w in res]
    if any(v is None for v in got[32] + got[64]):
        print("the texture did not land 256-byte aligned (POISON)")
        return 2
    # The calibration rows name the level starts. They need only be distinct (gfx1036 puts a linear chain smallest
    # level first, each row padded to 256 bytes); wave64 must agree.
    starts = [got[32][n] for n in range(LEVELS)]
    print("level starts (dwords): " + ", ".join(f"{n}:{s:.0f}" for n, s in enumerate(starts)))
    if len(set(starts)) != LEVELS or starts != [got[64][n] for n in range(LEVELS)]:
        print("calibration: the level starts are not distinct, or wave32 and wave64 disagree; cannot name levels")
        return 2
    level_of = {s: n for n, s in enumerate(starts)}
    bad = vk_diff = 0
    for i, (name, instr, lod, s, t) in enumerate(table):
        want, vk = rule_level(instr, lod, s, t), vulkan_level(instr, lod, s, t)
        g32, g64 = level_of.get(got[32][i]), level_of.get(got[64][i])
        ok = g32 == want and g64 == want
        bad += not ok
        vk_diff += g32 != vk
        shown = "?" if g32 is None else str(g32)
        extra = "" if g32 == g64 else f" (wave64 {g64})"
        note = "" if g32 == vk else f"  Vulkan reads {vk}"
        print(f"{name:18} hw level {shown:>2}{extra:12} rule {want}  {'ok' if ok else 'DIFFERS'}{note}")
    print(f"{len(table) - bad}/{len(table)} rows match the rule; Vulkan's level differs on {vk_diff}")
    return 1 if bad else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
