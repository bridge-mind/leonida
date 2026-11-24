r"""Every f16 input through the f16 transcendentals on an AMD GPU (FUZZ-29). Standard library only.

v_exp_f16, v_log_f16, v_rcp_f16, v_rsq_f16, v_sqrt_f16, v_sin_f16 and v_cos_f16 run natively through the hardware
oracle (tools/hw_oracle/) on all 65,536 f16 bit patterns, in wave32 and wave64 and under several float MODEs, and
each result is compared with the rule the translator implements (tools/rdna2spirv.py: widen to f32, the GLSL.std.450
function, round to nearest even back to f16, f16 denormals kept):

  * v_exp_f16 = Exp2, v_log_f16 = Log2, v_rcp_f16 = 1 / x, v_rsq_f16 = InverseSqrt, v_sqrt_f16 = Sqrt. The
    reference is the exact value, correctly rounded to f16 (decimal arithmetic, 60 digits), which is the best a
    Vulkan driver can return for the translated SPIR-V.
  * v_sin_f16 / v_cos_f16 take turns: the reference is sin / cos of 2 pi x, reduced in turns, correctly rounded. A
    second table compares the hardware with Sin / Cos of the f32 product 6.2831855 * x, which is what the translator
    computed before reducing in turns (#594).
  * Where the rule's SPIR-V is undefined (Log2 / Sqrt / InverseSqrt of a negative, Log2 / InverseSqrt of 0, a
    division by 0), the hardware result is tabulated instead of judged.

Nothing is assumed about how close the hardware is: the tables count bit-exact results, 1 ulp, more than 1 ulp,
denormal operands read as zero, denormal results flushed, and NaN, and list the hardware word for each special
input. Each MODE is read back inside a kernel first, so a launcher that ignored the descriptor fails instead of
producing plausible tables.

Usage:
    python -I tools/f16_trans_sweep.py [--ops v_exp_f16,v_log_f16] [--modes keep16,flush16] [--json out.json]
    python -I tools/f16_trans_sweep.py --self-test

Exit 0 when the sweep ran (differences from the rule are findings, not failures), 1 when wave32 and wave64
disagree or --json cannot be written, 2 on a harness error (a MODE that did not apply, a kernel without its
words), 77 when the hardware oracle is unavailable. On Linux the
oracle needs LEONIDA_HW_DRM_NODE (docs/shader_validation.md). Nothing here reads game files.
"""
import argparse
import functools
import json
import os
import struct
import sys
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OPS = ("v_exp_f16", "v_log_f16", "v_rcp_f16", "v_rsq_f16", "v_sqrt_f16", "v_sin_f16", "v_cos_f16")

# f32 denormals flushed and IEEE off is the state docs/design.md infers for the game's shaders; the f16 denormal
# field is the open question, so both settings are run, then IEEE on, then f16 rounding toward zero.
MODES = {
    "keep16": {"ieee": 0, "dx10_clamp": 1, "denorm32": 0, "denorm16_64": 3},
    "flush16": {"ieee": 0, "dx10_clamp": 1, "denorm32": 0, "denorm16_64": 0},
    "ieee_keep16": {"ieee": 1, "dx10_clamp": 1, "denorm32": 0, "denorm16_64": 3},
    "ieee_flush16": {"ieee": 1, "dx10_clamp": 1, "denorm32": 0, "denorm16_64": 0},
    "keep16_rtz": {"ieee": 0, "dx10_clamp": 1, "denorm32": 0, "denorm16_64": 3, "round16_64": 3},
}

THREADS = 1024
PER_THREAD = 8                       # inputs per thread, packed two to a dword
PER_KERNEL = THREADS * PER_THREAD    # 8192 inputs, 4096 words per kernel: 8 kernels cover every input
QNAN = 0x7E00
TWO_PI_F32 = struct.unpack("<f", struct.pack("<f", 6.283185307179586))[0]

# Inputs whose hardware word is listed in the specials table.
SPECIALS = [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x3C00, 0xBC00, 0x3800, 0x4000, 0x7BFF,
            0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7D00, 0x7C01, 0x5C00, 0xDC00, 0x3400, 0x3A00]


def is_nan(h):
    return (h & 0x7C00) == 0x7C00 and (h & 0x3FF) != 0


def is_inf(h):
    return (h & 0x7FFF) == 0x7C00


def is_denorm(h):
    return (h & 0x7C00) == 0 and (h & 0x3FF) != 0


def is_zero(h):
    return (h & 0x7FFF) == 0


def neg(h):
    return bool(h & 0x8000)


def to_dec(h):
    """The exact value of a finite f16 as a Decimal."""
    return Decimal(struct.unpack("<e", struct.pack("<H", h))[0])


def from_dec(v, rtz=False):
    """f16 bits of the exact value v, rounded to nearest even (or toward zero), denormals kept, overflow to inf
    (toward zero: to 65504)."""
    if v == 0:
        return 0x8000 if v.is_signed() else 0
    sign = 0x8000 if v < 0 else 0
    a = abs(v)
    e = -14
    while a >= Decimal(2) ** (e + 1) and e < 15:
        e += 1
    ulp = Decimal(2) ** (e - 10)
    with localcontext() as c:
        c.prec = 80
        n = int((a / ulp).to_integral_value(rounding="ROUND_DOWN" if rtz else ROUND_HALF_EVEN))
    val = n * ulp
    if val > 65504:
        return sign | (0x7BFF if rtz else 0x7C00)
    return sign | struct.unpack("<H", struct.pack("<e", float(val)))[0]


@functools.lru_cache(maxsize=None)
def _pi_at(prec):
    with localcontext() as c:
        c.prec = prec
        c.prec += 2
        three = Decimal(3)
        lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
        while s != lasts:
            lasts = s
            n, na = n + na, na + 8
            d, da = d + da, da + 32
            t = (t * n) / d
            s += t
    return +s


def _pi():
    from decimal import getcontext
    return _pi_at(getcontext().prec)


def _sin(x):
    with localcontext() as c:
        c.prec += 2
        i, lasts, s, fact, num, sign = 1, 0, x, 1, x, 1
        while s != lasts:
            lasts = s
            i += 2
            fact *= i * (i - 1)
            num *= x * x
            sign *= -1
            s += num / fact * sign
    return +s


def sin_turns(f, cos=False):
    """sin(2 pi f) or cos(2 pi f) for an exact f, reduced in turns so quarter turns are exact."""
    f = f - f.to_integral_value(rounding="ROUND_FLOOR")       # [0, 1)
    if cos:
        f = f + Decimal("0.25")
        if f >= 1:
            f -= 1
    q = f * 4
    if q == q.to_integral_value():
        return [Decimal(0), Decimal(1), Decimal(0), Decimal(-1)][int(q)]
    if f > Decimal("0.5"):
        f -= 1                                                # (-0.5, 0.5]
    return _sin(2 * _pi() * f)


def exact(op, h):
    """The exact real result of the op on f16 h as a Decimal, or a str for a result that is not a finite real:
    'nan', '+inf', '-inf', '+0', '-0', or 'undef' where the translated SPIR-V is undefined."""
    if is_nan(h):
        return "nan"
    s = neg(h)
    if is_inf(h):
        return {"v_exp_f16": "+0" if s else "+inf", "v_log_f16": "undef" if s else "+inf",
                "v_rcp_f16": "-0" if s else "+0", "v_rsq_f16": "undef" if s else "+0",
                "v_sqrt_f16": "undef" if s else "+inf", "v_sin_f16": "undef", "v_cos_f16": "undef"}[op]
    if is_zero(h):
        return {"v_exp_f16": Decimal(1), "v_log_f16": "undef", "v_rcp_f16": "undef", "v_rsq_f16": "undef",
                "v_sqrt_f16": "-0" if s else "+0", "v_sin_f16": "-0" if s else "+0", "v_cos_f16": Decimal(1)}[op]
    x = to_dec(h)
    if s and op in ("v_log_f16", "v_rsq_f16", "v_sqrt_f16"):
        return "undef"
    with localcontext() as c:
        c.prec = 60
        if op == "v_exp_f16":
            return Decimal(2) ** x if x != x.to_integral_value() else (Decimal(2) ** int(x) if abs(x) < 64 else
                                                                       (Decimal(0) if x < 0 else Decimal(10) ** 30))
        if op == "v_log_f16":
            return x.ln() / Decimal(2).ln()
        if op == "v_rcp_f16":
            return 1 / x
        if op == "v_rsq_f16":
            return 1 / x.sqrt()
        if op == "v_sqrt_f16":
            return x.sqrt()
        return sin_turns(x, op == "v_cos_f16")


def product_turns(op, h):
    """sin / cos of the f32 product 6.2831855 * x for finite h (the translator before #594 was fixed)."""
    with localcontext() as c:
        c.prec = 60
        t = Decimal(struct.unpack("<f", struct.pack("<f", TWO_PI_F32 * float(to_dec(h))))[0]) / (2 * _pi())
        return sin_turns(t, op == "v_cos_f16")


def to_bits(r, rtz=False):
    if isinstance(r, str):
        return {"nan": QNAN, "+inf": 0x7C00, "-inf": 0xFC00, "+0": 0, "-0": 0x8000}.get(r)
    if r == 0:
        return 0
    return from_dec(r, rtz)


def reference(op, rtz=False):
    """65,536 entries: f16 bits per input, or None where the rule is undefined."""
    return [to_bits(exact(op, h), rtz) for h in range(65536)]


def order(h):
    return -(h & 0x7FFF) if neg(h) else h


def ulps(a, b):
    return abs(order(a) - order(b))


STORE128 = {"gfx10": "global_store_dwordx4", "gfx11": "global_store_b128", "gfx12": "global_store_b128"}
STORE32 = {"gfx10": "global_store_dword", "gfx11": "global_store_b32", "gfx12": "global_store_b32"}


def kernel_lines(op, base, arch="gfx1030"):
    """Thread t computes the op on base + 8t + k (k = 0..7) and stores them packed, low half first."""
    from hw_oracle import asm
    out = ["v_lshlrev_b32 v1, 3, v0", f"v_add_nc_u32 v1, 0x{base:x}, v1"]
    out += [f"v_add_nc_u32 v{2 + k}, {k}, v1" for k in range(PER_THREAD)]
    out += [f"{op} v{12 + k}, v{2 + k}" for k in range(PER_THREAD)]
    out.append("s_nop 7")
    for p in range(PER_THREAD // 2):
        lo, hi = 12 + 2 * p, 13 + 2 * p
        out += [f"v_and_b32 v{lo}, 0xffff, v{lo}", f"v_lshlrev_b32 v{hi}, 16, v{hi}", f"v_or_b32 v{8 + p}, v{lo}, v{hi}"]
    out += ["v_lshlrev_b32 v24, 4, v0", f"{STORE128[asm.family(arch)]} v24, v[8:11], s[2:3]", "s_endpgm"]
    return out


def mode_word(m):
    """MODE as the kernel descriptor sets it (FP_ROUND 3:0, FP_DENORM 7:4, DX10_CLAMP bit 8, IEEE bit 9)."""
    return (m.get("round32", 0) | m.get("round16_64", 0) << 2 | m.get("denorm32", 0) << 4 |
            m.get("denorm16_64", 3) << 6 | m.get("dx10_clamp", 1) << 8 | m.get("ieee", 1) << 9)


def check_modes(oracle, modes):
    from hw_oracle import asm
    body = ["s_getreg_b32 s4, hwreg(HW_REG_MODE, 0, 32)", "v_mov_b32 v1, s4", "v_mov_b32 v20, 0",
            f"{STORE32[asm.family(oracle.arch)]} v20, v1, s[2:3]", "s_endpgm"]
    kernels = [asm.Kernel(body, w, 1, 1, raw=True, mode=MODES[m]) for m in modes for w in (32, 64)]
    got = oracle.run(kernels, words=1)
    for k, (m, w) in enumerate((m, w) for m in modes for w in (32, 64)):
        if isinstance(got[k], str) or got[k][0] & 0x3FF != mode_word(MODES[m]):
            raise RuntimeError(f"MODE probe: {m} wave{w} reads back {got[k]!r}, expected "
                               f"0x{mode_word(MODES[m]):x}; the kernel descriptor's float mode was not applied")


def sweep(oracle, op, mode, wave):
    """The hardware's f16 result for every input, or raise."""
    from hw_oracle import asm
    kernels = [asm.Kernel(kernel_lines(op, j * PER_KERNEL, oracle.arch), wave, THREADS, PER_KERNEL // 2, raw=True,
                          mode=MODES[mode]) for j in range(65536 // PER_KERNEL)]
    got = oracle.run(kernels, words=PER_KERNEL // 2)
    out = []
    for j, words in enumerate(got):
        if isinstance(words, str) or len(words) != PER_KERNEL // 2:
            raise RuntimeError(f"{op} {mode} wave{wave} kernel {j}: {words if isinstance(words, str) else len(words)}")
        for w in words:
            out += [w & 0xFFFF, w >> 16]
    return out


def classify(op, hw, ref):
    """Counts for one op and mode. A denormal operand whose word equals the hardware's own word for a zero of the
    same sign counts as read as zero (the rule is undefined at 0 for log, rcp and rsq, so the rule can't say)."""
    zero_hw = {False: hw[0x0000], True: hw[0x8000]}
    c = {"inputs": 0, "exact": 0, "ulp1": 0, "ulp_more": 0, "max_ulp": 0, "nan_ok": 0, "zero_sign": 0,
         "denorm_in_as_zero": 0, "denorm_out_flushed": 0, "undefined": 0, "nan_payload_kept": 0, "snan_kept": 0,
         "snan_quieted": 0, "snan_other": 0, "other": 0}
    worst = []
    for h in range(65536):
        r, g = ref[h], hw[h]
        c["inputs"] += 1
        if r is None:
            c["undefined"] += 1
            continue
        if is_nan(r):
            if is_nan(g):
                c["nan_ok"] += 1
                if is_nan(h) and (g & 0x1FF) == (h & 0x1FF):
                    c["nan_payload_kept"] += 1
                if is_nan(h) and not h & 0x200:
                    c["snan_kept" if g == h else "snan_quieted" if g == h | 0x200 else "snan_other"] += 1
            else:
                c["other"] += 1
                worst.append((h, g, r))
            continue
        if g == r:
            c["exact"] += 1
        elif is_denorm(h) and g == zero_hw[neg(h)]:
            c["denorm_in_as_zero"] += 1
        elif is_denorm(r) and is_zero(g) and (g & 0x8000) == (r & 0x8000):
            c["denorm_out_flushed"] += 1
        elif is_zero(g) and is_zero(r):
            c["zero_sign"] += 1
        elif is_nan(g):
            c["other"] += 1
            worst.append((h, g, r))
        else:
            d = ulps(g, r)
            c["max_ulp"] = max(c["max_ulp"], d)
            if d == 1:
                c["ulp1"] += 1
            else:
                c["ulp_more"] += 1
                worst.append((h, g, r))
    worst.sort(key=lambda t: -ulps(t[1], t[2]) if not (is_nan(t[1]) or is_nan(t[2])) else -1 << 20)
    return c, worst[:8]


def product_reference(op, rtz=False):
    return [None if is_nan(h) or is_inf(h) else to_bits(product_turns(op, h), rtz) for h in range(65536)]


def range_stats(hw, turns):
    """sin / cos only: ulps against sin / cos of the f32 product, by |x| band."""
    bands = {"|x|<=1": (0, 1), "1<|x|<=256": (1, 256), "|x|>256": (256, 1e9)}
    out = {}
    for name, (lo, hi) in bands.items():
        n = ex = u1 = more = 0
        worst = 0
        for h in range(65536):
            if is_nan(h) or is_inf(h):
                continue
            a = abs(float(to_dec(h)))
            if not (lo < a <= hi or (lo == 0 and a == 0)):
                continue
            r = turns[h]
            g = hw[h]
            n += 1
            if is_nan(g):
                more += 1
                continue
            d = 0 if (g == r or (is_zero(g) and is_zero(r))) else ulps(g, r)
            worst = max(worst, d)
            ex += d == 0
            u1 += d == 1
            more += d > 1
        out[name] = {"inputs": n, "exact": ex, "ulp1": u1, "ulp_more": more, "max_ulp": worst}
    return out


def fmt(h):
    if h is None:
        return "undef"
    v = struct.unpack("<e", struct.pack("<H", h))[0]
    return f"0x{h:04x} ({v!r})"


def self_test():
    assert from_dec(Decimal(1)) == 0x3C00 and from_dec(Decimal(-2)) == 0xC000
    assert from_dec(Decimal(65519)) == 0x7BFF and from_dec(Decimal(65520)) == 0x7C00
    assert from_dec(Decimal(65520), rtz=True) == 0x7BFF
    assert from_dec(Decimal(2) ** -24) == 0x0001 and from_dec(Decimal(2) ** -26) == 0
    assert from_dec(Decimal(2) ** -25 * 3) == 0x0002        # 1.5 ulp of the smallest denormal: to even
    assert from_dec(Decimal(2) ** -25) == 0                 # exactly half: to even, 0
    assert to_bits(exact("v_exp_f16", 0x3C00)) == 0x4000
    assert to_bits(exact("v_exp_f16", 0xCE00)) == 0x0001    # 2^-24
    assert to_bits(exact("v_log_f16", 0x4400)) == 0x4000
    assert to_bits(exact("v_rcp_f16", 0x4000)) == 0x3800
    assert to_bits(exact("v_rsq_f16", 0x4400)) == 0x3800
    assert to_bits(exact("v_sqrt_f16", 0x4400)) == 0x4000
    assert exact("v_log_f16", 0) == "undef" and exact("v_sqrt_f16", 0xBC00) == "undef"
    assert exact("v_sqrt_f16", 0x8000) == "-0" and to_bits(exact("v_exp_f16", 0xFC00)) == 0
    assert to_bits(exact("v_sin_f16", 0x3400)) == 0x3C00              # sin(2 pi / 4) = 1
    assert to_bits(exact("v_cos_f16", 0x3800)) == 0xBC00              # cos(pi) = -1
    assert to_bits(exact("v_sin_f16", 0x3800)) == 0                   # sin(pi) = 0
    assert to_bits(exact("v_sin_f16", 0x8000)) == 0x8000
    assert ulps(0x3C00, 0x3C01) == 1 and ulps(0x0001, 0x8001) == 2
    lines = kernel_lines("v_exp_f16", 8192, "gfx1030")
    assert lines[1] == "v_add_nc_u32 v1, 0x2000, v1" and lines[-2].startswith("global_store_dwordx4")
    assert kernel_lines("v_exp_f16", 0, "gfx1150")[-2].startswith("global_store_b128")
    print("self-test ok")
    return 0


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
    ap.add_argument("--ops", default=",".join(OPS))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--waves", default="32,64")
    ap.add_argument("--json", help="write the counts, specials and worst cases here")
    ap.add_argument("--hw-arch")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    ops = [o.strip() for o in a.ops.split(",") if o.strip()]
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    try:
        waves = [int(w.strip()) for w in a.waves.split(",") if w.strip()]
    except ValueError:
        ap.error("--waves must contain only 32 and/or 64")
    for option, values in (("--ops", ops), ("--modes", modes), ("--waves", waves)):
        if not values:
            ap.error(f"{option} must select at least one value")
    bad = [o for o in ops if o not in OPS] + [m for m in modes if m not in MODES] + [str(w) for w in waves
                                                                                    if w not in (32, 64)]
    if bad:
        ap.error("unknown: " + ", ".join(bad))
    if a.json is not None:
        err = validate_json_path(a.json)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 1
    from hw_oracle import runner
    try:
        oracle = runner.open_oracle(arch=a.hw_arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77
    print(f"hardware oracle: {oracle.device}")
    report = {"device": str(oracle.device), "arch": oracle.arch, "modes": {m: MODES[m] for m in modes}, "ops": {}}
    rc = 0
    try:
        check_modes(oracle, modes)
        for op in ops:
            refs = {}
            report["ops"][op] = {}
            for mode in modes:
                rtz = MODES[mode].get("round16_64", 0) == 3
                if rtz not in refs:
                    ref = reference(op, rtz)
                    refs[rtz] = (ref, product_reference(op, rtz) if op in ("v_sin_f16", "v_cos_f16") else None)
                ref, turns = refs[rtz]
                hw = {w: sweep(oracle, op, mode, w) for w in waves}
                first = hw[waves[0]]
                differ = [h for w in waves[1:] for h in range(65536) if hw[w][h] != first[h]]
                counts, worst = classify(op, first, ref)
                if mode == modes[0]:
                    base = first
                entry = {"counts": counts, "waves_differ": len(differ),
                         f"differs_from_{modes[0]}": sum(first[h] != base[h] for h in range(65536)),
                         "specials": {f"0x{h:04x}": f"0x{first[h]:04x}" for h in SPECIALS},
                         "worst": [[f"0x{h:04x}", f"0x{g:04x}", f"0x{r:04x}"] for h, g, r in worst]}
                if op in ("v_sin_f16", "v_cos_f16"):
                    entry["vs_f32_product"] = range_stats(first, turns)
                report["ops"][op][mode] = entry
                if differ:
                    rc = 1
                print(f"{op} {mode}: " + ", ".join(f"{k} {v}" for k, v in counts.items()) +
                      (f"; wave32/64 differ on {len(differ)}" if differ else "; wave32 = wave64") +
                      (f"; differs from {modes[0]} on {entry[f'differs_from_{modes[0]}']}" if mode != modes[0] else ""))
                for h, g, r in worst[:3]:
                    print(f"    {fmt(h)}: hw {fmt(g)}, rule {fmt(r)}")
                if "vs_f32_product" in entry:
                    for band, s in entry["vs_f32_product"].items():
                        print(f"    vs Sin/Cos(f32 6.2831855 * x), {band}: " + ", ".join(f"{k} {v}" for k, v in s.items()))
            print(f"{op} specials (hw, mode {modes[0]}): " + ", ".join(
                f"{h} -> {v}" for h, v in report["ops"][op][modes[0]]["specials"].items()))
    except RuntimeError as e:
        print(f"harness error: {e}")
        return 2
    finally:
        oracle.close()
    if a.json is not None:
        err = write_json_report(a.json, report)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 1
    return rc


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
