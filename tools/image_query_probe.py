r"""What `image_get_resinfo` and `image_get_lod` return on an RDNA2 GPU (SHADER-388). Standard library only.

docs/design.md ("Image atomics, size and LOD queries") left three answers unverified: resinfo's depth word for a 2D
image, the clamped LOD `image_get_lod` returns under a linear mip filter, and the order of u and v in the LOD query.
This probe runs both instructions natively (through tools/hw_oracle/) on hand-built descriptors and compares every
result word with the rules below. Neither instruction reads memory, so the T# has no texture behind it.

image_get_resinfo, dmask 0xf, address VGPR = mip level (unsigned), T# with BASE_LEVEL b and LAST_LEVEL l:
    n = l - b + 1 levels; past the last level (level >= n) width, height and depth are 0; otherwise
    width = max(1, WIDTH >> (b + level)), height likewise, and the depth word is
        2D:        0, whatever T# WORD4 DEPTH holds (a linear GFX10.3 T# keeps its pitch there)
        2D array:  max(1, (DEPTH + 1) >> (b + level)): the layer count is minified like a 3D depth
    and the fourth word is n.
image_get_lod, dmask 0x3, 2D, S# MIN_LOD lo, MAX_LOD hi, LOD_BIAS s, n levels; one 2x2 quad of lanes whose u steps
by du across x and v by dv across y (so the LOD is log2(max(du * width, dv * height))):
    VGPR 1 (raw) = that LOD + s
    VGPR 0 (clamped) = 0 with MIP_FILTER none; else c = min(max(min(raw, hi), lo), n - 1) (lo wins over hi when
    lo > hi, and the level count caps both), then floor(c + 0.5) with MIP_FILTER point (the level it would read,
    ties up) and c with MIP_FILTER linear.
That is the definition of OpImageQueryLod / textureQueryLod (the level read, or a fraction between the two read).

Every case runs in wave32 and wave64; every lane of a quad must agree.

Usage:
    python -I tools/image_query_probe.py [--waves 32,64] [--json out.json]
    python -I tools/image_query_probe.py --self-test        # the rules against values measured on gfx1036, no GPU

Exit 0 when every word matches the rules, 1 on a mismatch (a finding: the rules above are wrong on this GPU), 2 on a
harness error or a --json report that cannot be written (the summary line still gives the result), 77 when the
hardware oracle is unavailable or the GPU is not GFX10 (the T# layout is GFX10's).
Nothing here reads game files.
"""
import argparse
import json
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TYPE_2D, TYPE_2D_ARRAY = 9, 10                      # SQ_RSRC_IMG_* in T# WORD3 [31:28]
DIMS = {TYPE_2D: "SQ_RSRC_IMG_2D", TYPE_2D_ARRAY: "SQ_RSRC_IMG_2D_ARRAY"}
MIP_NONE, MIP_POINT, MIP_LINEAR = 0, 1, 2           # SQ_TEX_MIP_FILTER in S# WORD2 [27:26]
SENTINEL = 0xDEAD0000


def f2u(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def u2f(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


# ---- the rules ----

def resinfo_model(width, height, typ, depth_field, base, last, level):
    """[width, height, depth, levels] that image_get_resinfo dmask:0xf returns (see the module docstring)."""
    n = last - base + 1
    if level >= n:
        return [0, 0, 0, n]
    shift = base + level
    w, h = max(1, width >> shift), max(1, height >> shift)
    depth = 0 if typ == TYPE_2D else max(1, (depth_field + 1) >> shift)
    return [w, h, depth, n]


def lod_model(lod, mip, lo, hi, bias, levels):
    """(clamped, raw) that image_get_lod dmask:0x3 returns for a quad whose computed LOD is `lod`."""
    raw = lod + bias
    if mip == MIP_NONE:
        return 0.0, raw
    c = min(max(min(raw, hi), lo), levels - 1)
    return (math.floor(c + 0.5) if mip == MIP_POINT else c), raw


# ---- descriptors ----

def tsharp_words(arch, width, height, typ=TYPE_2D, depth_field=None, base=0, last=0):
    """A GFX10 T# (tools/hw_oracle/tsharp.py's linear 2D one) with the type, levels and WORD4 DEPTH set.
    depth_field None keeps what tsharp writes there (the pitch on GFX10.3)."""
    from hw_oracle import tsharp
    t = tsharp.image_words(arch, "R8G8B8A8_UNORM", width, height, 4)
    t[3] = (t[3] & ~(0xF << 28) & ~(0xFF << 12)) | (typ << 28) | (base << 12) | (last << 16)
    if depth_field is not None:
        t[4] = (t[4] & ~0x1FFF) | depth_field
    return t


def ssharp_words(mip, lo, hi, bias=0.0):
    """A GFX10 S#: wrap, bilinear min/mag, MIP_FILTER mip, MIN_LOD/MAX_LOD (unsigned 4.8), LOD_BIAS (signed 5.8)
    (runtime/src/gfx10_descriptors.h encode() has the same layout)."""
    word1 = int(lo * 256) | int(hi * 256) << 12
    word2 = (int(bias * 256) & 0x3FFF) | 1 << 20 | 1 << 22 | mip << 26
    return [0, word1, word2, 0]


# ---- kernels ----

def resinfo_kernel(t, level, typ, wave):
    from hw_oracle import asm
    lines = [f"s_mov_b32 s{8 + i}, 0x{w:08x}" for i, w in enumerate(t)]
    lines += [f"v_mov_b32 v{8 + i}, 0x{SENTINEL + i:08x}" for i in range(4)]
    lines += [f"v_mov_b32 v2, {level}",
              f"image_get_resinfo v[8:11], v2, s[8:15] dmask:0xf dim:{DIMS[typ]}",
              "s_waitcnt vmcnt(0)",
              "v_lshlrev_b32 v4, 4, v0",
              "global_store_b128 v4, v[8:11], s[2:3]",
              "s_endpgm"]
    return asm.Kernel(lines, wave=wave, threads=1, words=4, raw=True)


def lod_kernel(t, s, du, dv, wave):
    """One quad: lane i reads u = 0.5 + (i & 1) * du, v = 0.5 + (i >> 1) * dv. Each lane stores clamped, raw, u, v."""
    from hw_oracle import asm
    lines = [f"s_mov_b32 s{8 + i}, 0x{w:08x}" for i, w in enumerate(t)]
    lines += [f"s_mov_b32 s{16 + i}, 0x{w:08x}" for i, w in enumerate(s)]
    lines += ["v_and_b32 v5, 1, v0", "v_lshrrev_b32 v6, 1, v0", "v_cvt_f32_u32 v5, v5", "v_cvt_f32_u32 v6, v6",
              f"v_mul_f32 v2, 0x{f2u(du):08x}, v5", f"v_mul_f32 v3, 0x{f2u(dv):08x}, v6",
              "v_add_f32 v2, 0.5, v2", "v_add_f32 v3, 0.5, v3",
              f"v_mov_b32 v8, 0x{SENTINEL:08x}", f"v_mov_b32 v9, 0x{SENTINEL + 1:08x}",
              "image_get_lod v[8:9], v[2:3], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D",
              "s_waitcnt vmcnt(0)",
              "v_mov_b32 v10, v2", "v_mov_b32 v11, v3",
              "v_lshlrev_b32 v4, 4, v0",
              "global_store_b128 v4, v[8:11], s[2:3]",
              "s_endpgm"]
    return asm.Kernel(lines, wave=wave, threads=4, words=16, raw=True)


# ---- cases ----

def resinfo_cases():
    """(name, width, height, type, DEPTH field or None, base, last, level)."""
    out = []
    for w, h in ((16, 4), (64, 64)):
        for depth in (None, 0, 5, 0x1FFF):
            for level in (0, 1, 4, 5, 20):
                out.append((f"2D {w}x{h} depth={depth} level={level}", w, h, TYPE_2D, depth, 0, 4, level))
    for base, last in ((1, 4), (2, 3)):
        for level in (0, 1, 2, 3):
            out.append((f"2D 16x4 base={base} last={last} level={level}", 16, 4, TYPE_2D, 0, base, last, level))
    for layers in (1, 6, 7, 16, 100):
        for base in (0, 1):
            for level in (0, 1, 2, 3, 5, 9):
                out.append((f"2D array 256x256x{layers} base={base} level={level}",
                            256, 256, TYPE_2D_ARRAY, layers - 1, base, 8, level))
    return out


LODS = (-1.0, 0.0, 0.25, 0.5, 1.25, 2.75, 3.5, 7.0)
SAMPLERS = [(mip, lo, hi, bias) for mip in (MIP_NONE, MIP_POINT, MIP_LINEAR)
            for lo, hi, bias in ((0, 15, 0), (0, 0.25, 0), (0.5, 2.25, 0), (1, 1, 0), (5.5, 15, 0), (3, 2, 0),
                                 (0, 15, 1.0), (0, 15, -0.5))]


def lod_cases():
    """(name, width, height, last, du, dv, sampler, expected LOD before bias). A 16x16 five-level image steps both
    axes by the same texel count; the 64x4 seven-level image steps one axis only, which tells u from v."""
    out = []
    for samp in SAMPLERS:
        for lod in LODS:
            d = 2.0 ** lod / 16
            out.append((f"16x16 mip={samp[0]} lod=[{samp[1]},{samp[2]}] bias={samp[3]} L={lod}", 16, 16, 4, d, d,
                        samp, lod))
    for tx, ty, lod in ((1, 0, 0.0), (0, 1, 0.0), (4, 0, 2.0), (0, 4, 2.0), (4, 1, 2.0), (1, 4, 2.0), (8, 2, 3.0)):
        out.append((f"64x4 du={tx} texels dv={ty} texels", 64, 4, 6, tx / 64, ty / 4, (MIP_LINEAR, 0, 15, 0), lod))
    return out


# ---- comparison ----

def compare_resinfo(case, got):
    name, w, h, typ, depth, base, last, level = case
    want = resinfo_model(w, h, typ, depth if depth is not None else 0, base, last, level)
    if typ == TYPE_2D_ARRAY and depth is None:
        raise ValueError("a 2D array case needs its DEPTH field")
    return want == list(got[:4]), want


def compare_lod(case, got):
    name, w, h, last, du, dv, (mip, lo, hi, bias), lod = case
    want = lod_model(lod, mip, lo, hi, bias, last + 1)
    lanes = [(u2f(got[4 * i]), u2f(got[4 * i + 1])) for i in range(4)]
    return all(lane == want for lane in lanes), want, lanes


def run(oracle, waves):
    arch = oracle.arch
    rcases, lcases = resinfo_cases(), lod_cases()
    report = {"arch": arch, "resinfo": [], "lod": []}
    bad = 0
    for wave in waves:
        kernels = [resinfo_kernel(tsharp_words(arch, w, h, typ, d, b, l), lv, typ, wave)
                   for _, w, h, typ, d, b, l, lv in rcases]
        for case, got in zip(rcases, oracle.run(kernels)):
            if isinstance(got, str) or len(got) < 4:
                raise RuntimeError(f"{case[0]} wave{wave}: {got}")
            ok, want = compare_resinfo(case, got)
            bad += not ok
            report["resinfo"].append({"case": case[0], "wave": wave, "got": got[:4], "rule": want, "match": ok})
            if not ok:
                print(f"MISMATCH resinfo wave{wave} {case[0]}: got {got[:4]}, rule {want}")
        kernels = [lod_kernel(tsharp_words(arch, w, h, last=l), ssharp_words(*s), du, dv, wave)
                   for _, w, h, l, du, dv, s, _ in lcases]
        for case, got in zip(lcases, oracle.run(kernels, words=16)):
            if isinstance(got, str) or len(got) < 16:
                raise RuntimeError(f"{case[0]} wave{wave}: {got}")
            ok, want, lanes = compare_lod(case, got)
            bad += not ok
            report["lod"].append({"case": case[0], "wave": wave, "got": lanes, "rule": want, "match": ok})
            if not ok:
                print(f"MISMATCH get_lod wave{wave} {case[0]}: got {lanes}, rule {want}")
        print(f"wave{wave}: image_get_resinfo {len(rcases)} cases, image_get_lod {len(lcases)} cases")
    return bad, report


# ---- self-test: values read on gfx1036 (Radeon Graphics of a Ryzen 9 9950X3D, Windows driver 32.0.21045.16010) ----

MEASURED_RESINFO = [
    ((16, 4, TYPE_2D, 0x3F, 0, 4, 0), [16, 4, 0, 5]),
    ((16, 4, TYPE_2D, 0x1FFF, 0, 4, 1), [8, 2, 0, 5]),
    ((16, 4, TYPE_2D, 0, 0, 4, 5), [0, 0, 0, 5]),
    ((16, 4, TYPE_2D, 0, 1, 4, 0), [8, 2, 0, 4]),
    ((16, 4, TYPE_2D, 0, 2, 3, 1), [2, 1, 0, 2]),
    ((256, 256, TYPE_2D_ARRAY, 5, 0, 8, 1), [128, 128, 3, 9]),
    ((256, 256, TYPE_2D_ARRAY, 6, 0, 8, 2), [64, 64, 1, 9]),
    ((256, 256, TYPE_2D_ARRAY, 99, 1, 8, 3), [16, 16, 6, 8]),
    ((16, 4, TYPE_2D_ARRAY, 5, 0, 2, 3), [0, 0, 0, 3]),
]
MEASURED_LOD = [  # (lod, mip, lo, hi, bias, levels) -> (clamped, raw)
    ((0.25, MIP_NONE, 0, 15, 0, 5), (0.0, 0.25)),
    ((0.5, MIP_POINT, 0, 15, 0, 5), (1.0, 0.5)),
    ((2.75, MIP_POINT, 0.5, 2.25, 0, 5), (2.0, 2.75)),
    ((6.0, MIP_POINT, 0, 15, 0, 5), (4.0, 6.0)),
    ((0.5, MIP_LINEAR, 0, 0.25, 0, 5), (0.25, 0.5)),
    ((-1.0, MIP_LINEAR, 0.5, 2.25, 0, 5), (0.5, -1.0)),
    ((2.75, MIP_LINEAR, 0, 15, 0, 5), (2.75, 2.75)),
    ((0.0, MIP_LINEAR, 5.5, 15, 0, 5), (4.0, 0.0)),
    ((7.0, MIP_LINEAR, 3, 2, 0, 5), (3.0, 7.0)),
    ((1.25, MIP_LINEAR, 0, 15, 1.0, 5), (2.25, 2.25)),
    ((1.25, MIP_POINT, 0, 15, -0.5, 5), (1.0, 0.75)),
]


def self_test():
    bad = 0
    for args, want in MEASURED_RESINFO:
        got = resinfo_model(*args)
        if got != want:
            print(f"resinfo {args}: rule {got}, measured {want}")
            bad += 1
    for args, want in MEASURED_LOD:
        got = lod_model(*args)
        if got != want:
            print(f"get_lod {args}: rule {got}, measured {want}")
            bad += 1
    print(f"self-test: {len(MEASURED_RESINFO) + len(MEASURED_LOD)} measured values, {bad} disagree with the rules")
    return 1 if bad else 0


def _import_oracle():
    """The optional hardware oracle, imported only for a GPU run.

    --help and --self-test must not touch it: finding AMD clang, the HIP runtime and a host compiler is the
    expensive startup work this probe would otherwise pay before argparse can answer --help.
    """
    sys.dont_write_bytecode = True
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from hw_oracle import asm, runner
    return asm, runner


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--waves", default="32,64")
    p.add_argument("--json", help="write every case, its result and the rule's answer here")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    waves = [int(w) for w in a.waves.split(",")]
    if any(w not in (32, 64) for w in waves):
        p.error("--waves takes 32 and/or 64")
    asm, runner = _import_oracle()
    try:
        oracle = runner.open_oracle()
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77
    try:
        if asm.family(oracle.arch) != "gfx10":
            print(f"{oracle.arch} is not GFX10; this probe builds GFX10 descriptors")
            return 77
        print(f"hardware oracle: arch={oracle.arch}")
        try:
            bad, report = run(oracle, waves)
        except RuntimeError as e:
            print(f"harness error: {e}")
            return 2
    finally:
        oracle.close()
    total = len(report["resinfo"]) + len(report["lod"])
    print(f"{total - bad} of {total} results follow the rules, {bad} do not")
    if a.json:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(report, f, indent=1)
        except OSError as exc:
            print(f"cannot write {a.json}: {exc.strerror or exc}", file=sys.stderr)
            return 2  # not 1: that would read as a result off the rules; the summary above has the result
    return 1 if bad else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
