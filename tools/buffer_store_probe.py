r"""Measure RDNA2 (GFX10) typed buffer store rounding and 2-component DST_SEL swizzling natively (SHADER-364).

Two questions left open by #2022 (measured on RDNA3.5 gfx1150, needing RDNA2 gfx103x data):
1. UNORM/SNORM store rounding: does hardware round the exact product clamp(x) * scale once (0.3 * 65535 -> 19661)
   with ties away from zero, or double-round through an intermediate f32 multiply (0.3 * 65535 -> 19660)?
2. 2-component buffer_store_format_xyzw channel mapping: why did gfx1150 write (x, w) instead of (x, y)?
   This probe sweeps all 12 2-component formats in FORMATS with XY01 (runtime RG) and XYZW, plus all 216 valid
   (DST_SEL_X, DST_SEL_Y, DST_SEL_Z) triples on 32_32_UINT, for both MUBUF (buffer_store_format_*) and MTBUF
   (tbuffer_store_format_*), in wave32 and wave64.

Usage:
    python -I tools/buffer_store_probe.py [--waves 32,64] [--json out.json]
    python -I tools/buffer_store_probe.py --self-test        # pure-Python model checks, no GPU

Exit 0 when the sweep ran and matched the model (or --self-test passed), 1 on a wave32/wave64, model, or
report-write error, 2 on a harness or CLI argument error, 77 when tools/hw_oracle is unavailable or the GPU is
not GFX10. Standard library only; nothing here reads game files.
"""
import argparse
import itertools
import json
import os
import struct
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# (name, GFX10 format code, kind, x channel bits)
NORM_FORMATS = (
    ("unorm16", 7, "unorm", 16),
    ("snorm16", 8, "snorm", 16),
    ("unorm8", 1, "unorm", 8),
    ("snorm8", 2, "snorm", 8),
    ("unorm10", 50, "unorm", 10),
)

# All 12 2-component formats in Gen.FORMATS: (name, code, kind, element_mask_or_None_for_64bit)
TWO_COMP_FORMATS = (
    ("8_8_UNORM", 14, "unorm", 0xFFFF),
    ("8_8_SNORM", 15, "snorm", 0xFFFF),
    ("8_8_UINT", 18, "uint", 0xFFFF),
    ("8_8_SINT", 19, "sint", 0xFFFF),
    ("16_16_UNORM", 23, "unorm", 0xFFFFFFFF),
    ("16_16_SNORM", 24, "snorm", 0xFFFFFFFF),
    ("16_16_UINT", 27, "uint", 0xFFFFFFFF),
    ("16_16_SINT", 28, "sint", 0xFFFFFFFF),
    ("16_16_FLOAT", 29, "f16", 0xFFFFFFFF),
    ("32_32_UINT", 62, "u32", None),
    ("32_32_SINT", 63, "i32", None),
    ("32_32_FLOAT", 64, "f32", None),
)

SEL_VALS = (0, 1, 4, 5, 6, 7)
SEL_XY01 = (4, 5, 0, 1)
SEL_XYZW = (4, 5, 6, 7)
SEL_X001 = (4, 0, 0, 1)


def f32_to_u32(val):
    return struct.unpack("<I", struct.pack("<f", val))[0]


def u32_to_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def vsharp_d3(code, sel=SEL_XYZW):
    sx, sy, sz, sw = sel
    return sx | (sy << 3) | (sz << 6) | (sw << 9) | (code << 12) | (1 << 24) | (3 << 28)


def round_half_away(frac):
    """Exact round to nearest integer, ties away from zero."""
    if frac >= 0:
        return int(frac + Fraction(1, 2))
    return -int(-frac + Fraction(1, 2))


def model_norm(u_bits, kind, bits):
    """Exact single-rounded UNORM/SNORM encoding (ties away from zero, NaN -> 0)."""
    f = u32_to_f32(u_bits)
    if f != f:
        return 0
    mask = (1 << bits) - 1
    if kind == "unorm":
        if f <= 0.0:
            return 0
        if f >= 1.0:
            return mask
        return round_half_away(Fraction(f) * mask)
    scale = (1 << (bits - 1)) - 1
    if f <= -1.0:
        return (-scale) & mask
    if f >= 1.0:
        return scale
    return round_half_away(Fraction(f) * scale) & mask


def model_norm_double(u_bits, kind, bits):
    """Double-rounded f32-multiply + RoundEven encoding (pre-SHADER-364 translator behaviour)."""
    f = u32_to_f32(u_bits)
    if f != f:
        return 0
    mask = (1 << bits) - 1
    if kind == "unorm":
        f = max(0.0, min(1.0, f))
        prod = u32_to_f32(f32_to_u32(f * float(mask)))
        return int(round(prod))
    scale = (1 << (bits - 1)) - 1
    f = max(-1.0, min(1.0, f))
    prod = u32_to_f32(f32_to_u32(f * float(scale)))
    return int(round(prod)) & mask


def model_2comp_channels(sel, comps):
    """Hardware 2-component MUBUF store COMP_SWAP rule from V# DST_SEL (sx, sy, sz, sw).

    comps is the 4-tuple (x, y, z, w) padded with 0 for absent store components.
    Returns (mem_ch0, mem_ch1).
    """
    sx, sy, sz, _sw = sel
    use_y = (sy >= 4) and (sz < 4)
    rev = (sx != 4) and ((sy >= 4) or (sz >= 4))
    sec = comps[1] if use_y else comps[3]
    return (sec, comps[0]) if rev else (comps[0], sec)


def norm_inputs(kind, bits):
    """Specials + inputs near (k + 0.5) / scale that expose double rounding."""
    scale = (1 << bits) - 1 if kind == "unorm" else (1 << (bits - 1)) - 1
    out = [
        0x00000000, 0x80000000, 0x00000001, 0x80000001,
        0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,
        f32_to_u32(0.5), f32_to_u32(-0.5),
        f32_to_u32(0.3), f32_to_u32(-0.3),
        f32_to_u32(1.0), f32_to_u32(-1.0),
        f32_to_u32(1.5), f32_to_u32(-1.5),
        0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
        0x7F800001, 0xFF800001,
    ]
    step = max(1, scale // 100)
    for k in range(0, scale, step):
        base = f32_to_u32((k + 0.5) / scale)
        for d in (-1, 0, 1):
            out.append(base + d)
            if kind == "snorm":
                out.append((base + d) | 0x80000000)
    return out


def _store_kernel(asm_mod, inst, code, sel, vals, wave):
    n_comp = len(inst.rsplit("_", 1)[1])
    vreg = "v4" if n_comp == 1 else f"v[4:{3 + n_comp}]"
    st = (f"{inst} {vreg}, v1, s[8:11], 0 offen glc" if inst.startswith("buffer_")
          else f"{inst} {vreg}, v1, s[8:11], 0 format:{code} offen glc")
    lines = [
        "v_mov_b32 v1, 0",
        "s_mov_b32 s8, s2",
        "s_and_b32 s9, s3, 0xffff",
        "s_mov_b32 s10, 0x100",
        f"s_mov_b32 s11, 0x{vsharp_d3(code, sel):x}",
    ]
    for i, v in enumerate(vals[:n_comp]):
        lines.append(f"v_mov_b32 v{4 + i}, 0x{v:x}")
    lines += [st, "s_waitcnt_vscnt null, 0x0", "s_endpgm"]
    return asm_mod.Kernel(lines, wave, 1, 2, raw=True, vgprs=12, sgprs=16)


def probe_norm(oracle, asm_mod, wave):
    """Run UNORM/SNORM rounding sweep + SNORM2 tie check in wave32 or wave64."""
    report = {}
    for name, code, kind, bits in NORM_FORMATS:
        inputs = norm_inputs(kind, bits)
        mask = (1 << bits) - 1
        sel = SEL_XYZW if bits == 10 else SEL_X001
        fmt_res = {}
        for inst in ("buffer_store_format_x", "tbuffer_store_format_x"):
            ks = [_store_kernel(asm_mod, inst, code, sel, (u,), wave) for u in inputs]
            got = oracle.run(ks, words=2)
            exact_ok = double_diffs = 0
            first_bad = None
            for u, words in zip(inputs, got):
                if isinstance(words, str) or len(words) < 2:
                    raise RuntimeError(f"{name} {inst} wave{wave}: {words}")
                hw = words[0] & mask
                want = model_norm(u, kind, bits)
                dbl = model_norm_double(u, kind, bits)
                if want != dbl:
                    double_diffs += 1
                if hw == want:
                    exact_ok += 1
                elif first_bad is None:
                    first_bad = {"f32": f"{u:#010x}", "hw": hw, "want": want, "double": dbl}
            fmt_res[inst] = {"inputs": len(inputs), "exact": exact_ok, "double_witnesses": double_diffs,
                             "first_mismatch": first_bad}
        report[name] = fmt_res
    # SNORM2 tie-breaking on code 45 (10_10_10_2_SNORM: 2-bit x at bits [1:0])
    tie_vals = (f32_to_u32(0.5), f32_to_u32(-0.5), f32_to_u32(0.5) - 1, f32_to_u32(-0.5) - 1)
    ks = [_store_kernel(asm_mod, "buffer_store_format_xyzw", 45, SEL_XYZW, (u, 0, 0, 0), wave) for u in tie_vals]
    got = oracle.run(ks, words=2)
    tie_ok = sum((w[0] & 3) == model_norm(u, "snorm", 2) for u, w in zip(tie_vals, got))
    report["snorm2_ties"] = {"inputs": len(tie_vals), "exact": tie_ok,
                             "half_pos": got[0][0] & 3, "half_neg": got[1][0] & 3}
    return report


def probe_two_comp(oracle, asm_mod, wave):
    """Run all 12 2-component formats (XY01 vs XYZW, MUBUF vs MTBUF) + all 216 (sx,sy,sz) triples on 32_32_UINT."""
    int_vals = (0x11, 0x22, 0x33, 0x44)
    flt_vals = (f32_to_u32(0.25), f32_to_u32(0.5), f32_to_u32(0.75), f32_to_u32(1.0))
    formats_ok = 0
    format_mismatches = []
    for fname, code, kind, mask in TWO_COMP_FORMATS:
        vals = flt_vals if kind in ("unorm", "snorm", "f16", "f32") else int_vals
        cases = (
            ("mubuf_xy_XY01", "buffer_store_format_xy", SEL_XY01),
            ("mubuf_xyzw_XY01", "buffer_store_format_xyzw", SEL_XY01),
            ("mubuf_xy_XYZW", "buffer_store_format_xy", SEL_XYZW),
            ("mubuf_xyzw_XYZW", "buffer_store_format_xyzw", SEL_XYZW),
            ("mtbuf_xyzw_XYZW", "tbuffer_store_format_xyzw", SEL_XYZW),
        )
        ks = [_store_kernel(asm_mod, inst, code, sel, vals, wave) for _, inst, sel in cases]
        got = oracle.run(ks, words=2)
        if any(isinstance(w, str) or len(w) < 2 for w in got):
            raise RuntimeError(f"{fname} wave{wave}: {got}")
        # Reference: mubuf_xy_XY01 == mubuf_xyzw_XY01 == mtbuf_xyzw_XYZW == (x, y),
        # mubuf_xy_XYZW == (x, 0), mubuf_xyzw_XYZW == (x, w) (distinct from (x, y) and (x, 0))
        def elem(w):
            return (w[0] & mask,) if mask is not None else (w[0], w[1])
        xy_xy01, xyzw_xy01, xy_xyzw, xyzw_xyzw, mt_xyzw = (elem(w) for w in got)
        if xy_xy01 == xyzw_xy01 == mt_xyzw and xy_xyzw != xy_xy01 and xyzw_xyzw != xy_xy01 and xyzw_xyzw != xy_xyzw:
            formats_ok += 1
        else:
            format_mismatches.append({"format": fname, "xy_xy01": xy_xy01, "xyzw_xy01": xyzw_xy01,
                                      "xy_xyzw": xy_xyzw, "xyzw_xyzw": xyzw_xyzw, "mt_xyzw": mt_xyzw})
    # Exhaustive 216 (sx, sy, sz) triples on 32_32_UINT (code 62)
    triples = list(itertools.product(SEL_VALS, repeat=3))
    ks = [_store_kernel(asm_mod, "buffer_store_format_xyzw", 62, (sx, sy, sz, 1), int_vals, wave)
          for sx, sy, sz in triples]
    got = oracle.run(ks, words=2)
    triples_ok = 0
    triple_bad = []
    for (sx, sy, sz), w in zip(triples, got):
        if isinstance(w, str) or len(w) < 2:
            raise RuntimeError(f"32_32_UINT ({sx},{sy},{sz}) wave{wave}: {w}")
        want = model_2comp_channels((sx, sy, sz, 1), int_vals)
        if (w[0], w[1]) == want:
            triples_ok += 1
        elif len(triple_bad) < 4:
            triple_bad.append({"sel": [sx, sy, sz, 1], "hw": [w[0], w[1]], "want": list(want)})
    return {
        "formats_total": len(TWO_COMP_FORMATS),
        "formats_ok": formats_ok,
        "format_mismatches": format_mismatches,
        "triples_total": len(triples),
        "triples_ok": triples_ok,
        "triple_mismatches": triple_bad,
    }


def self_test():
    # Double-rounding witness: UNORM16(0.3) -> exact 19661, double-rounded 19660
    u03 = f32_to_u32(0.3)
    assert model_norm(u03, "unorm", 16) == 19661
    assert model_norm_double(u03, "unorm", 16) == 19660
    # SNORM2 ties away from zero
    assert model_norm(f32_to_u32(0.5), "snorm", 2) == 1
    assert model_norm(f32_to_u32(-0.5), "snorm", 2) == 3
    assert model_norm(0x7FC00000, "unorm", 16) == 0
    assert model_norm(0xFFC00000, "snorm", 16) == 0
    # 2-component COMP_SWAP truth table
    c = (0x11, 0x22, 0x33, 0x44)
    assert model_2comp_channels(SEL_XY01, c) == (0x11, 0x22)
    assert model_2comp_channels((5, 4, 0, 1), c) == (0x22, 0x11)
    assert model_2comp_channels(SEL_XYZW, c) == (0x11, 0x44)
    assert model_2comp_channels((5, 4, 6, 7), c) == (0x44, 0x11)
    assert model_2comp_channels(SEL_XYZW, (0x11, 0x22, 0, 0)) == (0x11, 0)
    # Every norm format must have at least one double-rounding witness
    for name, _code, kind, bits in NORM_FORMATS:
        w = sum(model_norm(u, kind, bits) != model_norm_double(u, kind, bits) for u in norm_inputs(kind, bits))
        assert w > 0, f"{name} has no double-rounding witness"
    print("self-test OK")


def parse_waves(raw):
    """Parse --waves into a list of supported wave sizes (32, 64), or return (None, err_msg)."""
    tokens = [w.strip() for w in raw.split(",") if w.strip()]
    if not tokens:
        return None, "--waves must not be empty"
    waves = []
    for tok in tokens:
        try:
            w = int(tok)
        except ValueError:
            return None, f"invalid wave size {tok!r} (expected 32 or 64)"
        if w not in (32, 64):
            return None, f"unsupported wave size {w} (expected 32 or 64)"
        waves.append(w)
    return waves, None


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


def write_json_report(path, data):
    """Create missing parent directories and write `data` as LF-terminated JSON, or return an error string."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as e:
            return f"cannot create directory for --json {path}: {e.strerror or e}"
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        return f"cannot write --json {path}: {e.strerror or e}"
    return None


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--waves", default="32,64", help="wave sizes to run (default: 32,64)")
    ap.add_argument("--json", dest="json_out", help="write full JSON report to this path")
    ap.add_argument("--self-test", action="store_true", help="check the Python models without a GPU")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    waves, err = parse_waves(args.waves)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.json_out is not None:
        err = validate_json_path(args.json_out)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 2
    from hw_oracle import asm, runner
    try:
        oracle = runner.open_oracle()
    except runner.Unavailable as exc:
        print(f"SKIP: {exc}", file=sys.stderr)
        return 77
    if asm.family(oracle.arch) != "gfx10":
        print(f"SKIP: {oracle.arch} is not GFX10 (RDNA1/RDNA2)", file=sys.stderr)
        return 77
    out = {"arch": oracle.arch, "waves": {}}
    mismatches = 0
    for wave in waves:
        norm = probe_norm(oracle, asm, wave)
        two = probe_two_comp(oracle, asm, wave)
        out["waves"][str(wave)] = {"norm": norm, "two_comp": two}
        sweeps = [(f"{name} {inst}", info) for name, *_ in NORM_FORMATS for inst, info in norm[name].items()]
        bad_sweeps = [(label, info) for label, info in sweeps if info["exact"] != info["inputs"]]
        mismatches += len(bad_sweeps)
        ties = norm["snorm2_ties"]
        if ties["exact"] != ties["inputs"]:
            mismatches += 1
        if two["formats_ok"] != two["formats_total"] or two["triples_ok"] != two["triples_total"]:
            mismatches += 1
        print(f"wave{wave} ({oracle.arch}): norm {len(sweeps) - len(bad_sweeps)}/{len(sweeps)} sweeps exact across "
              f"{len(NORM_FORMATS)} formats, snorm2 ties {ties['exact']}/{ties['inputs']}, "
              f"2-comp {two['formats_ok']}/{two['formats_total']} formats, "
              f"{two['triples_ok']}/{two['triples_total']} DST_SEL triples")
        for label, info in bad_sweeps:
            first = info["first_mismatch"]
            print(f"  {label}: {info['exact']}/{info['inputs']} exact, first mismatch f32 {first['f32']} "
                  f"hw {first['hw']} want {first['want']} (double-rounded {first['double']})")
    if args.json_out is not None:
        err = write_json_report(args.json_out, out)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 1
    return 1 if mismatches else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
