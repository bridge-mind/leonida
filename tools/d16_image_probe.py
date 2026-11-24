r"""How an RDNA2 GPU rounds d16 image results to f16 (point loads, point samples and bilinear samples). Standard library.

docs/design.md ("d16 and f16 format narrowing") narrows the f16 of every d16 image result from its f32 with integer
round to nearest even. For images that is inferred, not measured. BUILD-161 (tools/d16_format_probe.py) found that a
typed d16 buffer load rounds the exact UNORM/SNORM quotient once, straight to f16, and that the f32-first rule is
1 ulp high on four 16-bit values. This sweep asks the same of the image path. Every value of each format below sits
in a texture, and every texel is read in one kernel lane with:

  image_load       f32 and d16, at the texel (unnormalized integer coordinates)
  image_sample_lz  f32 and d16, point filter, at the texel centre (unnormalized coordinates)
  image_sample_lz  f32 and d16, bilinear filter, `--frac` texels to the right of the centre, so the result mixes the
                   texel with its right neighbour (the last texel of a row clamps to itself)

natively on the GPU through tools/hw_oracle, in wave32 and wave64. The loads and point samples are compared with
the candidate rules of d16_format_probe (rte: the exact quotient rounded once; f32_rte: the translator's f32 first,
then f16; their toward-zero and denormal-flushing variants). A bilinear d16 result has no exact quotient, so it is
compared with the hardware's own f32 result for the same sample: rounded to nearest even (what the translator does
with the f32 a Vulkan driver returns), toward zero, or neither.

The float32 set holds f32 bit patterns that sit exactly on f16 halfway points (0x3f7ff000 and 0x3f7fb000, the two
UNORM/SNORM quotients that land there, their negatives, and other ties), plus values next to them. It answers
whether a correction for the two quotients could be applied to every float-class image or only to normalized ones:
a 32_FLOAT texel on a halfway point is a real tie.

The texture is in the kernel's own code after s_endpgm, 256-byte aligned and linear, found with s_getpc_b64 (the
layout of tools/hw_oracle/tsharp.py, #1566). The T# is tsharp.image_words with the FORMAT field set to the format
here; the samplers are GFX10 S#s with FORCE_UNNORMALIZED, CLAMP_LAST_TEXEL on every axis and MIP_FILTER none:

  S# WORD0  CLAMP_X/Y/Z [8:0] = 2, FORCE_UNNORMALIZED [15] = 1
  S# WORD1  MIN_LOD = MAX_LOD = 0
  S# WORD2  XY_MAG_FILTER [21:20] and XY_MIN_FILTER [23:22]: 0 point, 1 bilinear; MIP_FILTER [27:26] = 0
  S# WORD3  BORDER_COLOR_TYPE 0 (never read: every coordinate is inside the image)

(Mesa src/amd/registers gfx10.json, SQ_IMG_SAMP_WORD0..3.) GFX10 only (RDNA1/RDNA2); another architecture exits 77.

Controls that must hold for the sweep to count: the f32 image_load of every UNORM/SNORM texel equals the correctly
rounded f32 quotient (what the RX 6700 returns for buffer loads, so a wrong T# or texture layout shows here), a point
sample equals the load, a 16_FLOAT texel comes back with its own bits, and no lane reports a misaligned texture.

Usage:
    python -I tools/d16_image_probe.py [--formats unorm8,unorm16] [--waves 32,64] [--frac 0.5] [--json out.json]
    python -I tools/d16_image_probe.py --self-test

Exit 0 when the sweep ran (which rule the hardware follows is a finding), 1 when wave32 and wave64 disagree, 2 on a
harness error (a control failed, a kernel without its words), 77 when the oracle is unavailable or the GPU is not
GFX10. Nothing here reads game files.
"""
import argparse
import json
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import d16_format_probe as fp  # noqa: E402

# name: (GFX10 FORMAT, kind, bits per texel, texture width). Width x rows = 1024 texels; rows are 256 bytes, so the
# 256-byte pitch of a linear image adds no padding.
FORMATS = {
    "unorm8": (1, "unorm", 8, 256),
    "snorm8": (2, "snorm", 8, 256),
    "unorm16": (7, "unorm", 16, 128),
    "snorm16": (8, "snorm", 16, 128),
    "float16": (13, "f16", 16, 128),          # control: its own bits
    "float32": (22, "f32", 32, 64),           # halfway points of f16, as f32 texels
}
TEXELS = 1024
WORDS_PER_LANE = 8
SEED = 0xBEEF1234                              # the d16 destinations start as this
POISON = 0xBAADF00D

# f32 patterns on an f16 halfway point (the low 13 mantissa bits are exactly 0x1000) and their neighbours
HALFWAY_UNORM = (0x3F7FF000, 0x3F7FB000)


def float32_texels():
    """1024 f32 bit patterns, in this order: the two UNORM/SNORM halfway quotients and their negatives; more f16
    halfway points, chosen so that for half of them the f16 below is the even one and for the other half the f16
    above is (the two readings of a tie are then told apart); the f32 just below and just above each of those; and
    a fixed random fill in [-1, 1]. HALFWAY_COUNT counts the exact ties, which come first."""
    ties = []
    for b in HALFWAY_UNORM:
        ties += [b, b | 0x80000000]
    for exp in (0x3F800000, 0x3F000000, 0x3E800000, 0x3C000000, 0x38800000, 0x3F700000):
        for step in range(1, 5):                      # mantissa 0x1000, 0x3000, 0x5000, 0x7000: even and odd below
            b = exp | (2 * step - 1) << 12
            ties += [b, b | 0x80000000]
    out = list(ties)
    for b in ties:
        out += [b - 1, b + 1]
    rng = random.Random(161)
    while len(out) < TEXELS:
        out.append(fp.f32_bits(rng.uniform(-1.0, 1.0)))
    return out[:TEXELS], len(ties)


def block_values(fmt, block):
    """The raw texel values (ints, the format's bits) of one 1024-texel texture."""
    _, kind, n, _ = FORMATS[fmt]
    if kind == "f32":
        return float32_texels()[0]
    return [(block * TEXELS + t) & ((1 << n) - 1) for t in range(TEXELS)]


def blocks(fmt):
    _, kind, n, _ = FORMATS[fmt]
    return 1 if kind == "f32" else max(1, (1 << n) // TEXELS)


def texture_bytes(fmt, values):
    n = FORMATS[fmt][2]
    return b"".join(v.to_bytes(n // 8, "little") for v in values)


def sampler_words(bilinear):
    word0 = 2 | 2 << 3 | 2 << 6 | 1 << 15
    word2 = (1 << 20 | 1 << 22) if bilinear else 0
    return [word0, 0, word2, 0]


def t_words(arch, fmt):
    """The T#: tsharp's 2D linear descriptor with this format's FORMAT ([28:20] on GFX10)."""
    from hw_oracle import tsharp
    code, _, n, width = FORMATS[fmt]
    words = tsharp.image_words(arch, "R32_SFLOAT", width, TEXELS // width, n // 8,
                               (tsharp.SEL_X, tsharp.SEL_0, tsharp.SEL_0, tsharp.SEL_1))
    words[1] = (words[1] & ~(0x1FF << 20)) | code << 20
    return words


def _f32(x):
    return f"0x{fp.f32_bits(x):08x}"


def kernel_lines(arch, uid, fmt, values, frac):
    """A raw kernel body (asm.Kernel(..., raw=True)): lane t reads texel t six ways and stores 8 words at 32 t."""
    from hw_oracle import tsharp
    _, _, n, width = FORMATS[fmt]
    t = t_words(arch, fmt)
    pt, bl = sampler_words(False), sampler_words(True)
    img = "dmask:0x1 dim:SQ_RSRC_IMG_2D unorm"
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
    out += [f"s_mov_b32 s{16 + i}, 0x{w:08x}" for i, w in enumerate(pt)]
    out += [f"s_mov_b32 s{20 + i}, 0x{w:08x}" for i, w in enumerate(bl)]
    out += [f"v_and_b32 v2, {width - 1}, v0",
            f"v_lshrrev_b32 v3, {width.bit_length() - 1}, v0",
            "v_cvt_f32_u32 v5, v2",
            "v_cvt_f32_u32 v6, v3",
            "v_add_f32 v5, 0.5, v5",
            "v_add_f32 v6, 0.5, v6",
            f"v_add_f32 v16, {_f32(frac)}, v5",
            "v_mov_b32 v17, v6"]
    out += [f"v_mov_b32 v{r}, 0x{SEED:08x}" for r in (9, 11, 13)]
    out += ["v_mov_b32 v14, v0", "v_mov_b32 v15, 0",
            "s_cmp_lg_u32 s6, 0",
            f"s_cbranch_scc1 .Lbad{uid}",
            f"image_load v8, v[2:3], s[8:15] {img}",
            f"image_load v9, v[2:3], s[8:15] {img} d16",
            f"image_sample_lz v10, v[5:6], s[8:15], s[16:19] {img}",
            f"image_sample_lz v11, v[5:6], s[8:15], s[16:19] {img} d16",
            f"image_sample_lz v12, v[16:17], s[8:15], s[20:23] {img}",
            f"image_sample_lz v13, v[16:17], s[8:15], s[20:23] {img} d16",
            "s_waitcnt vmcnt(0)",
            f"s_branch .Lst{uid}",
            f".Lbad{uid}:"]
    out += [f"v_mov_b32 v{8 + i}, 0x{POISON:08x}" for i in range(8)]
    out += [f".Lst{uid}:",
            "v_lshlrev_b32 v4, 5, v0",
            "global_store_b128 v4, v[8:11], s[2:3]",
            "global_store_b128 v4, v[12:15], s[2:3] offset:16",
            "s_endpgm", ".p2align 8", f".Ltex{uid}:"]
    return out + tsharp._longs(tsharp.lay_out(texture_bytes(fmt, values), width, TEXELS // width, n // 8))


def sweep(oracle, fmt, wave, frac):
    """{field: [word per texel value]} over every block of the format, or raise RuntimeError."""
    from hw_oracle import asm
    todo = [(b, block_values(fmt, b)) for b in range(blocks(fmt))]
    kernels = [asm.Kernel(kernel_lines(oracle.arch, f"{fmt}_{b}", fmt, vals, frac), wave, TEXELS,
                          TEXELS * WORDS_PER_LANE, raw=True, vgprs=24, sgprs=32) for b, vals in todo]
    out = {k: [] for k in ("value", "load", "load_d16", "point", "point_d16", "bilin", "bilin_d16", "right")}
    width = FORMATS[fmt][3]
    got = oracle.run(kernels, words=TEXELS * WORDS_PER_LANE)
    if len(got) != len(todo):                 # zip would drop the tail and the counts would quietly cover fewer texels
        raise RuntimeError(f"{fmt} wave{wave}: the hardware runner returned {len(got)} results for {len(todo)} kernels")
    for (b, vals), words in zip(todo, got):
        if isinstance(words, str) or len(words) != TEXELS * WORDS_PER_LANE:
            raise RuntimeError(f"{fmt} wave{wave} block {b}: "
                               f"{words if isinstance(words, str) else f'{len(words)} words'}")
        for lane in range(TEXELS):
            w = words[WORDS_PER_LANE * lane:WORDS_PER_LANE * (lane + 1)]
            if POISON in w[:6]:
                raise RuntimeError(f"{fmt} wave{wave} block {b}: the texture is not 256-byte aligned")
            if w[6] != lane:
                raise RuntimeError(f"{fmt} wave{wave} block {b}: lane {lane} stored {w[6]:#x} as its index")
            x = lane % width
            out["value"].append(vals[lane])
            out["right"].append(vals[lane + 1] if x + 1 < width else vals[lane])
            for k, v in zip(("load", "load_d16", "point", "point_d16", "bilin", "bilin_d16"), w):
                out[k].append(v)
    return out


def classify(fmt, hw, frac):
    """Counts per field and rule, the first inputs off the translator's rule, and the control failures."""
    _, kind, n, _ = FORMATS[fmt]
    res = {"inputs": len(hw["value"]), "fields": {}, "control": []}
    vals = hw["value"]

    def note(msg):
        res["control"].append(msg)

    if kind in ("unorm", "snorm"):
        want32 = [fp.models32(kind, n, v)["f32_rte"] for v in vals]
        exact = sum(g == w for g, w in zip(hw["load"], want32))
        res["f32_load_is_rounded_quotient"] = exact
        far = [i for i, (g, w) in enumerate(zip(hw["load"], want32)) if abs(_ord32(g) - _ord32(w)) > 1]
        if far:
            i = far[0]
            note(f"f32 image_load is more than 1 ulp from the quotient on {len(far)} texels "
                 f"(first {vals[i]:#x}: {hw['load'][i]:#010x}, want {want32[i]:#010x}): wrong T# or layout?")
    if kind == "f32":
        bad = [i for i, v in enumerate(vals) if hw["load"][i] != v and not _both_nan32(hw["load"][i], v)]
        res["f32_load_is_the_texel"] = len(vals) - len(bad)
        if bad:
            note(f"the f32 image_load of a 32_FLOAT texel is not the texel for {len(bad)} of them "
                 f"(first {vals[bad[0]]:#010x} -> {hw['load'][bad[0]]:#010x}): wrong T# or layout?")
    if kind == "f16":
        bad = [i for i, v in enumerate(vals) if (hw["load_d16"][i] & 0xFFFF) != v and not (
            (v & 0x7C00) == 0x7C00 and v & 0x3FF and (hw["load_d16"][i] & 0x7C00) == 0x7C00)]
        if bad:
            note(f"16_FLOAT d16 image_load changed {len(bad)} texels (first {vals[bad[0]]:#06x} -> "
                 f"{hw['load_d16'][bad[0]] & 0xFFFF:#06x})")
    bad = [i for i in range(len(vals)) if hw["point"][i] != hw["load"][i] and not _both_nan32(hw["point"][i],
                                                                                              hw["load"][i])]
    if bad:
        i = bad[0]
        note(f"point sample differs from image_load on {len(bad)} texels (first {vals[i]:#x}: "
             f"{hw['point'][i]:#010x} vs {hw['load'][i]:#010x}): wrong S# or coordinates?")

    hi = {f: sum(w >> 16 == SEED >> 16 for w in hw[f]) for f in ("load_d16", "point_d16", "bilin_d16")}
    res["d16_high_half_kept"] = hi

    if kind in ("unorm", "snorm"):
        rules = tuple(fp.RULES16) + tuple(r + "_flush" for r in fp.RULES16)
        for field in ("load_d16", "point_d16"):
            match = dict.fromkeys(rules, 0)
            off = []
            for v, w in zip(vals, hw[field]):
                m = fp.models16(kind, n, v)
                g = w & 0xFFFF
                for r in rules:
                    match[r] += g == m[r]
                if g != m["f32_rte"]:                    # the translator's image rule: f32 first (f16_bits)
                    off.append({"value": f"{v:#x}", "hw": f"{g:#06x}", "f32_rte": f"{m['f32_rte']:#06x}",
                                "fits": [r for r in rules if m[r] == g]})
            res["fields"][field] = {"match": match, "off_f32_rte": len(off), "first_off": off[:8]}
    if kind == "f32":
        # "either": the even neighbour is the one toward zero, so the two readings give the same word
        ties = {"inputs": 0, "to_even": 0, "toward_zero": 0, "either": 0, "other": 0}
        rest = {"inputs": 0, "rte": 0, "rtz": 0, "both": 0, "other": 0}
        words = []
        for v, w in zip(vals, hw["load_d16"]):
            g = w & 0xFFFF
            x = fp.Fraction(struct.unpack("<f", struct.pack("<I", v))[0])
            rte, rtz = fp.f16_bits(fp.to_f16(x)), fp.f16_bits(fp.to_f16(x, True))
            if (v & 0x1FFF) == 0x1000 and (v & 0x7F800000) >= 0x38800000:     # on a normal f16 halfway point
                ties["inputs"] += 1
                ties["other" if g not in (rte, rtz) else "either" if rte == rtz else
                     "to_even" if g == rte else "toward_zero"] += 1
                words.append({"f32": f"{v:#010x}", "hw": f"{g:#06x}", "to_even": f"{rte:#06x}"})
            else:
                rest["inputs"] += 1
                rest["both" if g == rte == rtz else "rte" if g == rte else "rtz" if g == rtz else "other"] += 1
        res["fields"]["load_d16"] = {"halfway": ties, "others": rest, "halfway_words": words}

    # bilinear: d16 against the hardware's own f32 for the same sample, and (UNORM/SNORM) against the exact mix
    rte_own = rtz_own = other = nan = rtz_any = 0
    first = []
    exact32 = once16 = 0
    for i, (w32, w16) in enumerate(zip(hw["bilin"], hw["bilin_d16"])):
        g = w16 & 0xFFFF
        if (w32 & 0x7F800000) == 0x7F800000:           # inf or NaN: compare the class only
            nan += (g & 0x7C00) == 0x7C00
            other += (g & 0x7C00) != 0x7C00
            continue
        f = fp.Fraction(struct.unpack("<f", struct.pack("<I", w32))[0])
        a, b = _f16_signed(f, w32, False), _f16_signed(f, w32, True)
        rtz_any += g == b
        if g == a:
            rte_own += 1
        elif g == b:
            rtz_own += 1
        else:
            other += 1
            if len(first) < 8:
                first.append({"value": f"{vals[i]:#x}", "f32": f"{w32:#010x}", "d16": f"{g:#06x}",
                              "rte_of_f32": f"{a:#06x}"})
        if kind in ("unorm", "snorm"):
            q = fp.quotient(kind, n, vals[i]) * (1 - fp.Fraction(frac)) + \
                fp.quotient(kind, n, hw["right"][i]) * fp.Fraction(frac)
            exact32 += w32 == fp.f32_bits(fp.to_f32(q))
            once16 += g == fp.f16_bits(fp.to_f16(q))
    res["fields"]["bilinear"] = {"frac": frac, "d16_is_rte_of_own_f32": rte_own, "d16_is_rtz_of_own_f32": rtz_own,
                                 "d16_equals_rtz_of_own_f32": rtz_any, "inf_or_nan": nan, "other": other, "first_other": first,
                                 "f32_is_rounded_exact_mix": exact32 if kind in ("unorm", "snorm") else None,
                                 "d16_is_exact_mix_rounded_once": once16 if kind in ("unorm", "snorm") else None}
    return res


def _f16_signed(x, w32, rtz):
    """f16 bits of the f32 word w32 (value x) rounded to nearest even or toward zero, keeping the sign of a zero."""
    h = fp.f16_bits(fp.to_f16(x, rtz))
    return h | 0x8000 if (w32 & 0x80000000) and (h & 0x7FFF) == 0 else h


def _ord32(w):
    """An f32 word as a signed integer that orders like the values (for ulp distances)."""
    return -(w & 0x7FFFFFFF) if w & 0x80000000 else w


def _both_nan32(a, b):
    return all((x & 0x7F800000) == 0x7F800000 and x & 0x7FFFFF for x in (a, b))


def self_test():
    v, ties = float32_texels()
    assert ties == 52 and v[:4] == [0x3F7FF000, 0xBF7FF000, 0x3F7FB000, 0xBF7FB000] and len(v) == TEXELS and len(set(v[:40])) == 40
    assert all((b & 0xFFF) == 0 and (b >> 12) & 1 for b in v[:ties])
    # half the ties have the f16 below as the even one (nearest-even and toward-zero agree there), half above
    pairs = [(fp.f16_bits(fp.to_f16(x)), fp.f16_bits(fp.to_f16(x, True)))
             for x in (fp.Fraction(struct.unpack("<f", struct.pack("<I", b))[0]) for b in v[:ties])]
    assert sum(a != b for a, b in pairs) >= 20 and sum(a == b for a, b in pairs) >= 20
    assert sampler_words(True)[2] == 0x500000 and sampler_words(False)[0] == 0x8092
    assert block_values("unorm16", 3)[:2] == [3072, 3073] and blocks("unorm16") == 64 and blocks("unorm8") == 1
    assert blocks("float32") == 1 and len(texture_bytes("float32", v)) == 4096
    assert fp.models16("unorm", 16, 0xFFEF)["f32_rte"] == 0x3C00 and fp.models16("unorm", 16, 0xFFEF)["rte"] == 0x3BFF
    print("self-test ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--waves", default="32,64")
    ap.add_argument("--frac", type=float, default=0.5, help="bilinear offset from the texel centre, in texels")
    ap.add_argument("--json")
    ap.add_argument("--hw-arch")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    fmts = [f for f in a.formats.split(",") if f]
    bad = [f for f in fmts if f not in FORMATS] + [w for w in a.waves.split(",") if w not in ("32", "64")]
    if bad or not fmts or not 0 < a.frac < 1:
        ap.error("unknown or out of range: " + ", ".join(bad or ["--frac must be in (0, 1)"]))
    waves = [int(w) for w in a.waves.split(",")]
    from hw_oracle import asm, runner
    try:
        oracle = runner.open_oracle(arch=a.hw_arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77
    rc = 0
    try:
        if asm.family(oracle.arch) != "gfx10":
            print(f"hardware oracle unavailable: {oracle.arch} is not GFX10; the T# and S# here are GFX10's")
            return 77
        print(f"hardware oracle: {oracle.device}")
        report = {"device": str(oracle.device), "arch": oracle.arch, "frac": a.frac, "formats": {}}
        for fmt in fmts:
            hw = {w: sweep(oracle, fmt, w, a.frac) for w in waves}
            first = hw[waves[0]]
            entry = classify(fmt, first, a.frac)
            entry["waves_differ"] = {k: sum(x != y for x, y in zip(first[k], hw[w][k]))
                                     for w in waves[1:] for k in first if k not in ("value", "right")}
            report["formats"][fmt] = entry
            for c in entry["control"]:
                print(f"harness error: {fmt}: {c}")
                rc = 2
            if any(entry["waves_differ"].values()) and rc != 2:
                rc = 1
            print(f"{fmt} ({entry['inputs']} texels): wave32/64 differ {entry['waves_differ'] or 'n/a'}; "
                  f"d16 high half kept {entry['d16_high_half_kept']}")
            for field, s in entry["fields"].items():
                print(f"  {field}: " + json.dumps({k: v for k, v in s.items() if not k.startswith("first")
                                                   and k != "halfway_words"}))
                for o in (s.get("first_off") or s.get("first_other") or [])[:3]:
                    print(f"    {o}")
                for o in s.get("halfway_words", [])[:6]:
                    print(f"    {o}")
    except RuntimeError as e:
        print(f"harness error: {e}")
        return 2
    finally:
        oracle.close()
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=1)
    return rc


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
