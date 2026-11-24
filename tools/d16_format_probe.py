r"""How an RDNA2 GPU rounds UNORM/SNORM buffer loads to f16 with d16 (BUILD-161). Standard library only.

docs/design.md ("d16 and f16 format narrowing") narrows every d16 result with integer round to nearest even, but
for typed buffer loads that rounding was inferred (#489, #711), not measured. This sweep measures it: every value of
each format below goes through buffer_load_format_d16_x and tbuffer_load_format_d16_x natively on the GPU (through
tools/hw_oracle/), in wave32 and wave64. Every result is compared with several candidate rules computed from the
exact quotient (Fraction arithmetic):

  f32_rte    x / (2^n - 1) (SNORM: max(s / (2^(n-1) - 1), -1)) rounded to f32 (nearest even), then to f16
             nearest even: the translator's rule before BUILD-161 (the exact binary32 decode, then Gen.f16_bits)
  f32_rtz    rounded to f32, then to f16 toward zero (v_cvt_pkrtz_f16_f32 of the f32 value; PackHalf2x16 on AMD
             drivers, the #489 bug)
  rte        the exact quotient rounded once to f16, nearest even (no f32 step): what an RX 6700 does, and the
             translator's rule since BUILD-161 (Gen.d16_norm_f16)
  rtz        the exact quotient rounded once to f16, toward zero
  mul_rte    the f32 product v * f32(1 / (2^n - 1)) (a reciprocal multiply, not a division), then f16 nearest even

and each rule again with f16 denormal results flushed to zero (UNORM16 and SNORM16 have quotients under 2^-14).
Only the 16-bit formats tell the f32 step apart: rte and f32_rte differ on UNORM16 0xffaf and 0xffef and on SNORM16
0x7ff7 and 0x8009, and a flush shows on the smallest UNORM16 / SNORM16 values. On UNORM8, SNORM8 and UNORM10 every
rule but the rtz ones gives the same bits.
buffer_load_format_x (an f32 result, no d16) runs on the same values against the f32 rules, so the two stages can be
told apart. Controls that must hold for the sweep to count: every staged input word reads back as written, a
16_FLOAT d16 load returns its own bits, and d16_x leaves the high half of its VGPR as it was.

The kernels write their own inputs: thread t stores the format's bits for value base + t at dword t of the output
buffer, waits for the store, builds a GFX10 buffer descriptor (V#) over that buffer and loads its element with glc dlc
(past the L0 and GL1 caches), then stores the loaded VGPR at dword 1024 + t. GFX10 only (RDNA1 and RDNA2: the
V# layout, RESOURCE_LEVEL = 1), which is what BUILD-161 asks for; another architecture exits 77.

Usage:
    python -I tools/d16_format_probe.py [--formats unorm8,snorm16] [--waves 32,64] [--json out.json]
    python -I tools/d16_format_probe.py --self-test        # the models against known counts, no GPU

Exit 0 when the sweep ran (which rule the hardware follows is a finding, not a failure), 1 when wave32 and wave64
or MUBUF and MTBUF disagree, 2 on a harness error (a control failed, a kernel without its words), 77 when the
hardware oracle is unavailable or the GPU is not GFX10. Nothing here reads game files.
"""
import argparse
import json
import os
import struct
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# name: (GFX10 unified buffer format code, kind, bits of the x channel). The codes are the translator's FORMATS
# table (tools/rdna2spirv.py); 10_10_10_2 (50) holds x in its low 10 bits.
FORMATS = {
    "unorm8": (1, "unorm", 8),
    "snorm8": (2, "snorm", 8),
    "unorm16": (7, "unorm", 16),
    "snorm16": (8, "snorm", 16),
    "unorm10": (50, "unorm", 10),
    "float16": (13, "f16", 16),       # control: d16 of an f16 element is its own bits
}
INSTS = ("buffer_d16", "tbuffer_d16", "buffer_f32")
THREADS = 1024
WORDS = 2 * THREADS
HI_SEED = 0xBEEF0000                  # the d16_x destination starts as this | 0x1234; the high half must stay
LO_SEED = 0x1234
RULES16 = ("f32_rte", "f32_rtz", "rte", "rtz", "mul_rte")
RULES32 = ("f32_rte", "f32_rtz", "mul_f32")


# ---- exact arithmetic ----

def round_frac(x, prec, emin, rtz=False):
    """x rounded to a binary float with prec significand bits (implicit bit included) and minimum normal exponent
    emin, denormals kept; nearest even, or toward zero. No overflow: every value here is in [-1, 1].
    Exact, in integers: x = p/q, 2^e <= p/q < 2^(e+1), ulp = 2^k; the significand is p/q / 2^k rounded. Building
    Fractions for 2^e and the quotient instead gave the same bits about 4x slower (minutes for a 16-bit sweep)."""
    p, q = x.numerator, x.denominator
    if p == 0:
        return Fraction(0)
    sign, p = (-1, -p) if p < 0 else (1, p)
    e = p.bit_length() - q.bit_length()                   # floor(log2(p/q)) is e or e - 1
    if (p << -e if e < 0 else p) < (q << e if e > 0 else q):
        e -= 1
    k = max(e, emin) - prec + 1
    num, den = (p, q << k) if k >= 0 else (p << -k, q)
    n, r = divmod(num, den)
    if not rtz and (2 * r > den or (2 * r == den and n & 1)):   # half to even
        n += 1
    return Fraction(sign * n << k) if k >= 0 else Fraction(sign * n, 1 << -k)


def f16_bits(v):
    return struct.unpack("<H", struct.pack("<e", float(v)))[0]


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def to_f16(x, rtz=False):
    return round_frac(x, 11, -14, rtz)


def to_f32(x, rtz=False):
    return round_frac(x, 24, -126, rtz)


def quotient(kind, n, v):
    """The exact value of raw channel bits v (n bits) for a UNORM or SNORM channel."""
    if kind == "unorm":
        return Fraction(v, (1 << n) - 1)
    s = v - (1 << n) if v >> (n - 1) else v
    return max(Fraction(s, (1 << (n - 1)) - 1), Fraction(-1))


def is_denorm16(h):
    return (h & 0x7C00) == 0 and (h & 0x3FF) != 0


def models16(kind, n, v):
    """{rule: f16 bits} for one input, plus each rule with a denormal result flushed (rule + '_flush')."""
    x = quotient(kind, n, v)
    rcp = to_f32(Fraction(1, (1 << n) - 1) if kind == "unorm" else Fraction(1, (1 << (n - 1)) - 1))
    s = (v - (1 << n) if v >> (n - 1) else v) if kind == "snorm" else v
    prod = max(to_f32(s * rcp), Fraction(-1))
    out = {"rte": f16_bits(to_f16(x)), "rtz": f16_bits(to_f16(x, True)), "f32_rte": f16_bits(to_f16(to_f32(x))),
           "f32_rtz": f16_bits(to_f16(to_f32(x), True)), "mul_rte": f16_bits(to_f16(prod))}
    for r in RULES16:
        out[r + "_flush"] = (out[r] & 0x8000) if is_denorm16(out[r]) else out[r]
    return out


def models32(kind, n, v):
    x = quotient(kind, n, v)
    rcp = to_f32(Fraction(1, (1 << n) - 1) if kind == "unorm" else Fraction(1, (1 << (n - 1)) - 1))
    s = (v - (1 << n) if v >> (n - 1) else v) if kind == "snorm" else v
    return {"f32_rte": f32_bits(to_f32(x)), "f32_rtz": f32_bits(to_f32(x, True)),
            "mul_f32": f32_bits(max(to_f32(s * rcp), Fraction(-1)))}


# ---- kernels ----

def descriptor_word3(code):
    """GFX10 V# dword 3: DST_SEL x/y/z/w = X/Y/Z/W (4, 5, 6, 7), FORMAT [18:12], RESOURCE_LEVEL [24] = 1,
    OOB_SELECT [29:28] = 3 (raw: in bounds while offset < NUM_RECORDS), TYPE [31:30] = 0 (buffer)."""
    return 4 | 5 << 3 | 6 << 6 | 7 << 9 | code << 12 | 1 << 24 | 3 << 28


def kernel_lines(inst, fmt, base):
    """Thread t: stage format bits (base + t) & mask at dword t, load the element, store the VGPR at dword 1024 + t."""
    code, kind, n = FORMATS[fmt]
    mask = (1 << n) - 1
    load = {"buffer_d16": "buffer_load_format_d16_x v3, v2, s[8:11], 0 offen glc dlc",
            "tbuffer_d16": f"tbuffer_load_format_d16_x v3, v2, s[8:11], 0 format:{code} offen glc dlc",
            "buffer_f32": "buffer_load_format_x v3, v2, s[8:11], 0 offen glc dlc"}[inst]
    return [
        f"v_add_nc_u32 v1, 0x{base:x}, v0",
        f"v_and_b32 v1, 0x{mask:x}, v1",
        "v_lshlrev_b32 v2, 2, v0",
        "global_store_b32 v2, v1, s[2:3]",
        "s_waitcnt_vscnt null, 0x0",
        "s_mov_b32 s8, s2",
        "s_and_b32 s9, s3, 0xffff",                       # BASE_ADDRESS[47:32]; STRIDE 0, no swizzle
        f"s_mov_b32 s10, 0x{4 * THREADS:x}",               # NUM_RECORDS: the staged inputs, in bytes
        f"s_mov_b32 s11, 0x{descriptor_word3(code):x}",
        f"v_mov_b32 v3, 0x{HI_SEED | LO_SEED:x}",
        load,
        "s_waitcnt vmcnt(0)",
        f"v_add_nc_u32 v4, 0x{4 * THREADS:x}, v2",
        "global_store_b32 v4, v3, s[2:3]",
        "s_endpgm",
    ]


def jobs(fmt):
    """(inst, base) for every kernel a format needs: one per 1024 values."""
    n = FORMATS[fmt][2]
    insts = ("buffer_d16", "tbuffer_d16") if FORMATS[fmt][1] == "f16" else INSTS
    return [(inst, base) for inst in insts for base in range(0, 1 << n, THREADS)]


def sweep(oracle, fmt, wave):
    """{inst: [loaded VGPR word per value]} for every value of the format, or raise RuntimeError."""
    from hw_oracle import asm
    n = FORMATS[fmt][2]
    todo = jobs(fmt)
    kernels = [asm.Kernel(kernel_lines(inst, fmt, base), wave, THREADS, WORDS, raw=True, vgprs=8, sgprs=16)
               for inst, base in todo]
    got = oracle.run(kernels, words=WORDS)
    # Every kernel must answer: zip() over a shorter list would drop the values of the missing ones and still
    # succeed. hw_oracle's Oracle.run() returns one entry per kernel; this keeps that contract checked (BUILD-733).
    if len(got) != len(todo):
        raise RuntimeError(f"{fmt} wave{wave}: the oracle returned {len(got)} results for {len(todo)} kernels")
    out = {}
    for (inst, base), words in zip(todo, got):
        if words is None or isinstance(words, str) or len(words) != WORDS:
            why = "no result" if words is None else words if isinstance(words, str) else f"{len(words)} words"
            raise RuntimeError(f"{fmt} {inst} wave{wave} base {base}: {why}")
        count = min(THREADS, (1 << n) - base)
        staged = words[:count]
        want = [(base + t) & ((1 << n) - 1) for t in range(count)]
        if staged != want:
            bad = next(t for t in range(count) if staged[t] != want[t])
            raise RuntimeError(f"{fmt} {inst} wave{wave}: staged input {want[bad]:#x} read back as {staged[bad]:#x}")
        out.setdefault(inst, []).extend(words[THREADS:THREADS + count])
    return out


# ---- comparison ----

def classify(fmt, hw):
    """Counts of inputs each rule predicts exactly, the inputs where the translator's rule (rte for d16, f32_rte for f32) is wrong, and the
    control failures. hw: {inst: [VGPR word per value]}."""
    _, kind, n = FORMATS[fmt]
    res = {"inputs": 1 << n, "insts": {}, "control": []}
    for inst, words in hw.items():
        d16 = inst.endswith("d16")
        if d16:
            hi_bad = [v for v, w in enumerate(words) if w >> 16 != HI_SEED >> 16]
            if hi_bad:
                res["control"].append(f"{inst}: d16_x changed the high half for {len(hi_bad)} values "
                                      f"(first {hi_bad[0]:#x}: {words[hi_bad[0]]:#010x})")
        if kind == "f16":
            wrong = [v for v, w in enumerate(words) if (w & 0xFFFF) != v and not (
                (v & 0x7C00) == 0x7C00 and v & 0x3FF and (w & 0x7C00) == 0x7C00 and w & 0x3FF)]
            if wrong:
                res["control"].append(f"{inst}: 16_FLOAT came back changed for {len(wrong)} values "
                                      f"(first {wrong[0]:#06x} -> {words[wrong[0]] & 0xFFFF:#06x})")
            res["insts"][inst] = {"own_bits": (1 << n) - len(wrong)}
            continue
        rules = (tuple(RULES16) + tuple(r + "_flush" for r in RULES16)) if d16 else RULES32
        match = dict.fromkeys(rules, 0)
        off = []
        for v, w in enumerate(words):
            got = w & 0xFFFF if d16 else w
            m = models16(kind, n, v) if d16 else models32(kind, n, v)
            for r in rules:
                match[r] += got == m[r]
            rule = "rte" if d16 else "f32_rte"      # the translator's rule (Gen.d16_norm_f16, norm_bits)
            if got != m[rule]:
                off.append((v, got, m[rule], m))
        res["insts"][inst] = {"match": match, "off_rule": len(off),
                              "first_off": [{"value": f"{v:#x}", "hw": f"{g:#x}", "rule": f"{r:#x}",
                                             "fits": [k for k in rules if m[k] == g]} for v, g, r, m in off[:8]]}
    return res


def rte_rtz_disagreements(kind, n):
    """How many inputs the translator's rule and the #489 bug (f32 then RTZ) narrow differently (docs/design.md:
    123 of 256 UNORM8 and 110 of 256 SNORM8 loaded 1 ulp low)."""
    return sum(models16(kind, n, v)["f32_rte"] != models16(kind, n, v)["f32_rtz"] for v in range(1 << n))


def self_test():
    assert round_frac(Fraction(1, 3), 11, -14) == Fraction(1365, 4096)                     # 0x3555
    assert f16_bits(to_f16(Fraction(1, 255))) == 0x1C04                                   # 2^-8 * 1.00390625
    assert f16_bits(to_f16(Fraction(9, 255))) == 0x2885 and f16_bits(to_f16(Fraction(9, 255), True)) == 0x2884
    assert f16_bits(to_f16(Fraction(1, 65535))) == 0x0100 and is_denorm16(0x0100)        # under 2^-14
    assert f16_bits(to_f16(Fraction(1, 2) + Fraction(1, 4096))) == 0x3800                 # a tie: to even
    assert quotient("snorm", 8, 0x80) == -1 and quotient("snorm", 8, 0x81) == -1 and quotient("snorm", 8, 0x7F) == 1
    assert models16("unorm", 8, 255)["rte"] == 0x3C00 and models16("snorm", 16, 0x8000)["rte"] == 0xBC00
    assert models32("unorm", 8, 1)["f32_rte"] == 0x3B808081                               # 1/255 in f32
    assert models16("unorm", 16, 1)["f32_rte_flush"] == 0
    assert models16("unorm", 16, 0xFFEF)["rte"] == 0x3BFF and models16("unorm", 16, 0xFFEF)["f32_rte"] == 0x3C00
    assert models16("snorm", 16, 0x8009)["rte"] == 0xBBFF and models16("snorm", 16, 0x8009)["f32_rte"] == 0xBC00
    assert rte_rtz_disagreements("unorm", 8) == 123
    assert rte_rtz_disagreements("snorm", 8) == 110
    w3 = descriptor_word3(7)
    assert w3 & 0xFFF == 0xFAC and (w3 >> 12) & 0x7F == 7 and (w3 >> 24) & 1 and (w3 >> 28) & 3 == 3
    lines = kernel_lines("tbuffer_d16", "unorm16", 0x400)
    assert lines[0] == "v_add_nc_u32 v1, 0x400, v0" and "format:7 offen glc dlc" in lines[10]
    assert len(jobs("unorm16")) == 3 * 64 and len(jobs("unorm8")) == 3 and len(jobs("float16")) == 2 * 64
    print("self-test ok")
    return 0


def write_report(path, doc):
    """Write the JSON report, creating its folder; one ERROR line instead of a traceback on failure."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1)
    except OSError as e:
        print(f"ERROR: cannot write {path}: {e}", file=sys.stderr)
        return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--waves", default="32,64")
    ap.add_argument("--json", help="write the counts and the first differences here")
    ap.add_argument("--hw-arch")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    fmts = [f for f in a.formats.split(",") if f]
    waves = [int(w) for w in a.waves.split(",") if w.strip().isdigit()]
    bad = [f for f in fmts if f not in FORMATS] + [w for w in a.waves.split(",") if w and w not in ("32", "64")]
    if bad or not fmts or not waves:
        ap.error("unknown or empty: " + ", ".join(bad or ["--formats/--waves"]))
    if a.json and os.path.isdir(a.json):
        ap.error(f"--json {a.json} is a folder, not a file")
    from hw_oracle import asm, runner
    try:
        oracle = runner.open_oracle(arch=a.hw_arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77
    try:
        if asm.family(oracle.arch) != "gfx10":
            print(f"hardware oracle unavailable: {oracle.arch} is not GFX10; the descriptor here is GFX10's (RDNA2)")
            return 77
        print(f"hardware oracle: {oracle.device}")
        report = {"device": str(oracle.device), "arch": oracle.arch, "formats": {}}
        rc = 0
        for fmt in fmts:
            hw = {w: sweep(oracle, fmt, w) for w in waves}
            first = hw[waves[0]]
            entry = classify(fmt, first)
            entry["waves_differ"] = {inst: sum(a_ != b_ for a_, b_ in zip(first[inst], hw[w][inst]))
                                     for w in waves[1:] for inst in first}
            if "tbuffer_d16" in first:
                entry["mubuf_mtbuf_differ"] = sum((x ^ y) & 0xFFFF != 0
                                                  for x, y in zip(first["buffer_d16"], first["tbuffer_d16"]))
            report["formats"][fmt] = entry
            if entry["control"]:
                for c in entry["control"]:
                    print(f"harness error: {fmt} {c}")
                rc = 2
            if any(entry["waves_differ"].values()) or entry.get("mubuf_mtbuf_differ"):
                rc = max(rc, 1) if rc != 2 else 2
            print(f"{fmt} ({entry['inputs']} values): wave32/64 differ {entry['waves_differ'] or 'n/a'}; "
                  f"MUBUF/MTBUF d16 differ {entry.get('mubuf_mtbuf_differ', 'n/a')}")
            for inst, s in entry["insts"].items():
                if "own_bits" in s:
                    print(f"  {inst}: own bits {s['own_bits']}/{entry['inputs']}")
                    continue
                print(f"  {inst}: " + ", ".join(f"{r} {k}" for r, k in s["match"].items()) +
                      f"; off the rule {s['off_rule']}")
                for o in s["first_off"][:3]:
                    print(f"    value {o['value']}: hw {o['hw']}, rule {o['rule']}, fits {','.join(o['fits']) or '-'}")
    except RuntimeError as e:
        print(f"harness error: {e}")
        return 2
    finally:
        oracle.close()
    if a.json and not write_report(a.json, report):
        return 2
    return rc


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
