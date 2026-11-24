r"""How an RDNA2 GPU narrows 32-bit integer texels in d16 image loads, per image dimension (SHADER-387). Standard
library only.

docs/design.md ("Typed image loads") has native gfx1036 saturating 32-bit UINT/SINT texels to the 16-bit range in
2D d16 loads, and lists 1D, 2D-array and 3D d16 loads as unchecked. Mesa sets `integer_dest_saturates` for AMD,
the translator saturates (`saturate_int16`). This probe measures it natively (through tools/hw_oracle/) on every
dimension the translator handles: SQ_RSRC_IMG_1D, 2D, 2D_ARRAY and 3D, for the formats 32_UINT and 32_SINT (dmask
0x1) and 32_32_UINT and 32_32_SINT (dmask 0x3, two results packed into one VGPR), with `image_load` and
`image_load_mip` (mip 0).

Each lane reads one texel. Per texel the kernel stores four words:
  0  image_load without d16, x channel     the control: must be the staged word, or the texture is laid out wrong
  1  image_load d16                         the destination VGPR seeded with 0x5a5a5a5a first
  2  image_load_mip d16, mip 0              the same, through the other form
  3  image_load without d16, y channel     (two-channel formats; the control for y, else the seed)
Every d16 result is compared with three rules for each 32-bit input w:
  trunc   the low 16 bits of w (what the translator did before it saturated)
  sat_u   min(w, 0xffff), w unsigned
  sat_s   w as a signed int clamped to [-32768, 32767], its low 16 bits
and, for one-channel formats, the high half: kept (0x5a5a, as docs/design.md records for 2D) or something else.

The texture rides in the kernel's code after s_endpgm, as tools/hw_oracle/tsharp.py does for 2D, in a linear
layout whose rows are 256 bytes (64 R32 texels, 32 R32G32 texels), so the layout does not depend on how the T#
states the pitch. GFX10 only (RDNA1 and RDNA2: the T# layout); another family
exits 77.

Usage:
    python -I tools/image_d16_probe.py [--dims 1d,2d,2d_array,3d] [--formats r32_uint,...] [--json out.json]
    python -I tools/image_d16_probe.py --self-test        # the rules, T# words and kernels, no GPU

Exit 0 when the sweep ran (which rule the hardware follows is a finding, not a failure), 1 when image_load and
image_load_mip disagree, 2 on a harness, CLI, or report write error (a control word did not read back), 77 when the
hardware oracle is unavailable or the GPU is not GFX10. Nothing here reads game files.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED = 0x5A5A5A5A
SENTINEL = 0xDEAD0000
POISON = 0xBAADF00D
ROW_BYTES = 256

# Dimension -> (T# TYPE, MIMG dim name, coordinate count, texture extent as (height, depth or layers)).
DIMS = {
    "1d": (8, "SQ_RSRC_IMG_1D", 1, (1, 1)),
    "2d": (9, "SQ_RSRC_IMG_2D", 2, (2, 1)),
    "2d_array": (13, "SQ_RSRC_IMG_2D_ARRAY", 3, (2, 3)),
    "3d": (10, "SQ_RSRC_IMG_3D", 3, (2, 3)),
}
# Format -> (GFX10 FORMAT code, channels, signed). Codes from tools/gfx10_formats.json.
FORMATS = {
    "r32_uint": (20, 1, False),
    "r32_sint": (21, 1, True),
    "r32g32_uint": (62, 2, False),
    "r32g32_sint": (63, 2, True),
}
# Values that tell the rules apart, then a few that every rule maps alike.
EDGES = [0x00000000, 0x00000001, 0x00007FFE, 0x00007FFF, 0x00008000, 0x00008001, 0x0000FFFE, 0x0000FFFF,
         0x00010000, 0x00010001, 0x00012345, 0x0001FFFF, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFF7FFF,
         0xFFFF8000, 0xFFFF8001, 0xFFFFFFFE, 0xFFFFFFFF, 0x12348765, 0xDEAD5A5A, 0x00004321, 0xFFFFC000]


def rules(w):
    """The 16-bit result of each candidate rule for the 32-bit input w."""
    s = w - (1 << 32) if w & 0x80000000 else w
    return {"trunc": w & 0xFFFF, "sat_u": min(w, 0xFFFF), "sat_s": max(-32768, min(32767, s)) & 0xFFFF}


def texel_values(fmt, count):
    """The staged 32-bit words, `channels` per texel: EDGES first, then a fixed mixing sequence."""
    channels = FORMATS[fmt][1]
    out, x = [], 0x9E3779B9
    for i in range(count * channels):
        if i < len(EDGES):
            out.append(EDGES[i])
        else:
            x = (x * 1103515245 + 12345) & 0xFFFFFFFF
            out.append(x ^ (x >> 7) if i % 3 else EDGES[i % len(EDGES)])
    return [tuple(out[t * channels:(t + 1) * channels]) for t in range(count)]


def geometry(dim, fmt):
    """(width, height, depth or layers, texel count) of the probe's texture."""
    channels = FORMATS[fmt][1]
    width = ROW_BYTES // (4 * channels)
    height, depth = DIMS[dim][3]
    return width, height, depth, width * height * depth


def image_words(dim, fmt):
    """The 8 words of a linear, one-mip GFX10.3 T# with the base address left zero (the kernel adds it).
    Field positions as runtime/src/gfx10_descriptors.h lists them; a 1D or 2D image carries its pitch - 1 in WORD4
    DEPTH (GFX10.3 linear), an array its last layer and a 3D image its depth - 1."""
    kind, _, _, _ = DIMS[dim]
    code = FORMATS[fmt][0]
    width, height, depth, _ = geometry(dim, fmt)
    w, h = width - 1, height - 1
    word1 = (code << 20) | ((w & 3) << 30)
    word2 = ((w >> 2) & 0xFFF) | (h << 14) | (1 << 31)                 # RESOURCE_LEVEL, 1 on GFX10
    word3 = (4 | 5 << 3 | 6 << 6 | 7 << 9) | (kind << 28)              # X Y Z W, levels 0..0, linear
    word4 = (width - 1) if dim in ("1d", "2d") else (depth - 1)        # pitch is width: rows are exactly 256 bytes
    word5 = 4 << 20                                                    # PERF_MOD, as Mesa writes it
    return [0, word1, word2, word3, word4, word5, 0, 0]


def texture_bytes(dim, fmt):
    data = bytearray()
    for texel in texel_values(fmt, geometry(dim, fmt)[3]):
        for word in texel:
            data += word.to_bytes(4, "little")
    return bytes(data)


def _longs(data):
    words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
    return [f"  .long {', '.join(f'0x{w:08x}' for w in words[i:i + 8])}" for i in range(0, len(words), 8)]


def kernel_lines(dim, fmt, uid=0):
    """A raw kernel (asm.Kernel(..., raw=True): s[2:3] the output, v0 the lane), one texel per lane."""
    _, dim_name, ncoord, _ = DIMS[dim]
    width, height, _, count = geometry(dim, fmt)
    channels = FORMATS[fmt][1]
    t = image_words(dim, fmt)
    log2w, log2wh = width.bit_length() - 1, (width * height).bit_length() - 1
    addr = "v2" if ncoord == 1 else f"v[2:{1 + ncoord}]"
    mip = f"v[2:{2 + ncoord}]"
    d16_mask = "0x1" if channels == 1 else "0x3"
    out = ["s_getpc_b64 s[4:5]", f".Lpc{uid}:", f"s_add_u32 s4, s4, .Ltex{uid}-.Lpc{uid}", "s_addc_u32 s5, s5, 0",
           "s_and_b32 s6, s4, 255", "s_lshr_b64 s[4:5], s[4:5], 8", "s_mov_b32 s8, s4", "s_and_b32 s9, s5, 255",
           f"s_or_b32 s9, s9, 0x{t[1]:08x}"]
    out += [f"s_mov_b32 s{8 + i}, 0x{t[i]:08x}" for i in range(2, 8)]
    out += [f"v_and_b32 v2, {width - 1}, v0",
            f"v_lshrrev_b32 v3, {log2w}, v0",
            f"v_and_b32 v3, {height - 1}, v3",
            f"v_lshrrev_b32 v4, {log2wh}, v0",
            f"v_mov_b32 v{2 + ncoord}, 0"]                              # the mip of image_load_mip (or unused)
    if ncoord == 1:
        out.append("v_mov_b32 v4, 0")
    out += [f"v_mov_b32 v8, 0x{SENTINEL:08x}", f"v_mov_b32 v9, 0x{SEED:08x}", f"v_mov_b32 v10, 0x{SEED:08x}",
            f"v_mov_b32 v11, 0x{SEED:08x}",
            "s_cmp_lg_u32 s6, 0", f"s_cbranch_scc0 .Lok{uid}"]
    out += [f"v_mov_b32 v{8 + i}, 0x{POISON:08x}" for i in range(4)]
    out += [f"s_branch .Lst{uid}", f".Lok{uid}:",
            f"image_load v8, {addr}, s[8:15] dmask:0x1 dim:{dim_name} unorm",
            f"image_load v9, {addr}, s[8:15] dmask:{d16_mask} dim:{dim_name} unorm d16",
            f"image_load_mip v10, {mip}, s[8:15] dmask:{d16_mask} dim:{dim_name} unorm d16"]
    if channels == 2:
        out.append(f"image_load v11, {addr}, s[8:15] dmask:0x2 dim:{dim_name} unorm")
    out += ["s_waitcnt vmcnt(0)", f".Lst{uid}:", "v_lshlrev_b32 v6, 4, v0", "global_store_b128 v6, v[8:11], s[2:3]",
            "s_endpgm", ".p2align 8", f".Ltex{uid}:"]
    if count > 1024:
        raise ValueError("one lane per texel needs at most 1024 texels")
    return out + _longs(texture_bytes(dim, fmt))


def classify(dim, fmt, words):
    """Compare one kernel's output (4 words per texel) with the rules."""
    channels = FORMATS[fmt][1]
    staged = texel_values(fmt, geometry(dim, fmt)[3])
    entry = {"texels": len(staged), "control": [], "forms": {}}
    forms = {"image_load": 1, "image_load_mip": 2}
    agree = {f: {r: 0 for r in ("trunc", "sat_u", "sat_s")} for f in forms}
    high = {f: {} for f in forms}
    first_off = {f: [] for f in forms}
    for i, texel in enumerate(staged):
        got = words[4 * i:4 * i + 4]
        if got[0] != texel[0]:
            entry["control"].append(f"texel {i}: x read back 0x{got[0]:08x}, staged 0x{texel[0]:08x}")
        if channels == 2 and got[3] != texel[1]:
            entry["control"].append(f"texel {i}: y read back 0x{got[3]:08x}, staged 0x{texel[1]:08x}")
        for form, k in forms.items():
            lo, hi = got[k] & 0xFFFF, got[k] >> 16
            for r in agree[form]:
                want = rules(texel[0])[r]
                ok = lo == want
                if channels == 2:
                    ok = ok and hi == rules(texel[1])[r]
                agree[form][r] += ok
            if channels == 1:
                key = "kept" if hi == SEED >> 16 else f"0x{hi:04x}"
                high[form][key] = high[form].get(key, 0) + 1
            if len(first_off[form]) < 4 and not any(
                    (lo == rules(texel[0])[r]) for r in ("sat_u", "sat_s")):
                first_off[form].append({"in": [f"0x{v:08x}" for v in texel], "got": f"0x{got[k]:08x}"})
    for form in forms:
        entry["forms"][form] = {"match": agree[form], "high_half": high[form] or "n/a (two channels)",
                                "first_off": first_off[form]}
    entry["forms_differ"] = sum(words[4 * i + 1] != words[4 * i + 2] for i in range(len(staged)))
    return entry


def self_test():
    """The rules on the edge values, the T# fields and the kernels, without a GPU."""
    fails = []

    def check(cond, what):
        if not cond:
            fails.append(what)

    check(rules(0x00012345) == {"trunc": 0x2345, "sat_u": 0xFFFF, "sat_s": 0x7FFF}, "rules of 0x12345")
    check(rules(0xFFFF8000) == {"trunc": 0x8000, "sat_u": 0xFFFF, "sat_s": 0x8000}, "rules of -32768")
    check(rules(0x80000000) == {"trunc": 0x0000, "sat_u": 0xFFFF, "sat_s": 0x8000}, "rules of INT_MIN")
    check(rules(0x00007FFF) == {"trunc": 0x7FFF, "sat_u": 0x7FFF, "sat_s": 0x7FFF}, "rules of 0x7fff")
    # the edges tell every pair of rules apart, so a match count says which rule the hardware follows
    for a in ("trunc", "sat_u", "sat_s"):
        for b in ("trunc", "sat_u", "sat_s"):
            if a < b:
                check(any(rules(w)[a] != rules(w)[b] for w in EDGES), f"no edge separates {a} from {b}")
    for dim in DIMS:
        for fmt in FORMATS:
            width, height, depth, count = geometry(dim, fmt)
            check(width * 4 * FORMATS[fmt][1] == ROW_BYTES, f"{dim} {fmt}: a row is not 256 bytes")
            t = image_words(dim, fmt)
            check(t[3] >> 28 == DIMS[dim][0], f"{dim} {fmt}: TYPE")
            check((t[1] >> 20) & 0x1FF == FORMATS[fmt][0], f"{dim} {fmt}: FORMAT")
            check(((t[1] >> 30) | ((t[2] & 0xFFF) << 2)) + 1 == width, f"{dim} {fmt}: WIDTH")
            check(((t[2] >> 14) & 0x3FFF) + 1 == height, f"{dim} {fmt}: HEIGHT")
            lines = kernel_lines(dim, fmt)
            check(sum(ln.startswith("image_load") for ln in lines) == 2 + FORMATS[fmt][1], f"{dim} {fmt}: loads")
            check(all(DIMS[dim][1] in ln for ln in lines if ln.startswith("image_load")), f"{dim} {fmt}: dim")
            check(len(texture_bytes(dim, fmt)) == count * 4 * FORMATS[fmt][1], f"{dim} {fmt}: texture size")
            # a fake GPU that saturates: every texel matches sat_s or sat_u, and the controls hold
            signed = FORMATS[fmt][2]
            fake = []
            for texel in texel_values(fmt, count):
                r = [rules(v)["sat_s" if signed else "sat_u"] for v in texel]
                d16 = r[0] | ((r[1] << 16) if len(r) == 2 else SEED & 0xFFFF0000)
                fake += [texel[0], d16, d16, texel[1] if len(texel) == 2 else SEED]
            e = classify(dim, fmt, fake)
            rule = "sat_s" if signed else "sat_u"
            check(not e["control"] and e["forms"]["image_load"]["match"][rule] == count and e["forms_differ"] == 0,
                  f"{dim} {fmt}: the saturating fake does not match {rule}")
            check(e["forms"]["image_load"]["match"]["trunc"] < count, f"{dim} {fmt}: trunc not told apart")
    for f in fails:
        print("FAIL:", f)
    print(f"self-test: {'ok' if not fails else f'{len(fails)} failure(s)'}")
    return 1 if fails else 0


def validate_json_path(path):
    """Return an error string if `path` cannot be used as a --json output file, or None."""
    if not path or not path.strip():
        return "--json path must not be empty"
    if "\x00" in path:
        return "--json path contains a NUL byte"
    if path.endswith(os.sep) or (os.altsep and path.endswith(os.altsep)) or os.path.isdir(path):
        return f"--json path is a directory: {path}"
    cur = os.path.dirname(os.path.abspath(path))
    while cur:
        if os.path.isdir(cur):
            break
        if os.path.lexists(cur):
            return f"--json parent is not a directory: {cur}"
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return None


def write_json_report(path, report):
    """Create missing parent directories and write `report` as LF-terminated JSON, or return an error string."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as e:
            return f"cannot create directory for --json {path}: {e.strerror or e}"
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=1)
    except OSError as e:
        return f"cannot write --json {path}: {e.strerror or e}"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dims", default=",".join(DIMS))
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--json", help="write the per-case counts here")
    ap.add_argument("--hw-arch")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    dims = [d for d in a.dims.split(",") if d]
    fmts = [f for f in a.formats.split(",") if f]
    bad = [d for d in dims if d not in DIMS] + [f for f in fmts if f not in FORMATS]
    if bad or not dims or not fmts:
        ap.error("unknown or empty: " + ", ".join(bad or ["--dims/--formats"]))
    if a.json is not None:
        err = validate_json_path(a.json)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 2
    from hw_oracle import asm, runner
    try:
        oracle = runner.open_oracle(arch=a.hw_arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77
    rc = 0
    try:
        if asm.family(oracle.arch) != "gfx10":
            print(f"hardware oracle unavailable: {oracle.arch} is not GFX10; the T# here is GFX10's (RDNA2)")
            return 77
        print(f"hardware oracle: {oracle.device}")
        report = {"device": str(oracle.device), "arch": oracle.arch, "cases": {}}
        cases = [(d, f) for d in dims for f in fmts]
        kernels = []
        for uid, (d, f) in enumerate(cases):
            count = geometry(d, f)[3]
            kernels.append(asm.Kernel(kernel_lines(d, f, uid), 32, count, count * 4, raw=True))
        for (d, f), got in zip(cases, oracle.run(kernels)):
            name = f"{d} {f}"
            if isinstance(got, str):
                print(f"harness error: {name}: {got}")
                rc = 2
                continue
            e = classify(d, f, got)
            report["cases"][name] = e
            if e["control"]:
                rc = 2
                for c in e["control"][:4]:
                    print(f"harness error: {name}: {c}")
            if e["forms_differ"] and rc != 2:
                rc = 1
            for form, s in e["forms"].items():
                m = ", ".join(f"{r} {k}/{e['texels']}" for r, k in s["match"].items())
                hh = s["high_half"] if isinstance(s["high_half"], str) else \
                    ", ".join(f"{k} {v}" for k, v in s["high_half"].items())
                print(f"{name:<18} {form:<15} {m}; high half: {hh}")
            if e["forms_differ"]:
                print(f"{name}: image_load and image_load_mip differ on {e['forms_differ']} texels")
    except RuntimeError as e:
        print(f"harness error: {e}")
        return 2
    finally:
        oracle.close()
    if a.json is not None:
        err = write_json_report(a.json, report)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 2
    return rc


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
