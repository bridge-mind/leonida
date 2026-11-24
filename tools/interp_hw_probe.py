r"""SHADER-430: what the RDNA2 v_interp ALU ops compute from the attribute words, measured on an AMD GPU.

v_interp_* run in a compute wave too: they read the attribute words from LDS at M0 (gfx1036, M0 = 0: attribute N
takes 12 dwords from byte 48*N; channel c's P0 and P10 words are dwords 2c and 2c+1, its P20 word is dword 8+c). This
tool writes random parameter words there (plain f32 values, and packed f16 pairs as a 16-bit varying leaves them),
runs every op through tools/hw_oracle and checks each result against exact models (fractions.Fraction, round to
nearest even):

    v_interp_mov_f32  p0          the raw P0 word
    v_interp_p1_f32               fma(P10, I, P0), one rounding to f32 (also for a packed channel: raw words as f32)
    v_interp_p2_f32               fma(P20, J, p1), one rounding to f32
    v_interp_p1ll_f16 [high]      fma(h10, I, h0) on the chosen f16 halves, exact widening, one rounding to f32
    v_interp_p2_f16   [high]      fma(h20, J, p1ll) rounded ONCE to f16 (not f32 then f16); the high half of vdst kept

Then a set of hand-built f16 rounding ties (p1ll lands exactly between two f16, P20*J is far below one f32 ulp) that
the two p2_f16 rounding models disagree on. What it does not cover: what the SPI writes into LDS for a real draw
(whether P10/P20 of a packed channel are f32 or per-half f16 differences under FP16_INTERP_MODE), v_interp_p1lv_f16,
f16 inf/NaN halves. Standard library only; no game files.

Usage:
    python -I tools/interp_hw_probe.py                          # gfx1036, seeds 1-4, 4 kernels each, f32 denormals kept
    python -I tools/interp_hw_probe.py --arch gfx1030 --seeds 1,2,3 --kernels 8
    python -I tools/interp_hw_probe.py --denorm32 0             # MODE flushing f32 denormals: denormal P0/P10/P20 read as 0

Exit codes: 0 every model matches, 1 a mismatch, 2 a harness error, 77 no hardware oracle on this machine.
"""
import argparse
import math
import os
import random
import struct
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_oracle import asm, runner  # noqa: E402

LANES = 32
ATTRS = 8
IJ_BASE = 1024


def f32v(x): return struct.unpack("<f", struct.pack("<I", x & 0xFFFFFFFF))[0]
def f32w(x): return struct.unpack("<I", struct.pack("<f", x))[0]
def f16v(h): return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]
def f16w(x): return struct.unpack("<H", struct.pack("<e", x))[0]


def rnd(q, mbits, emin, emax):
    """Round an exact rational to a binary format: nearest even, subnormals kept; a float, or inf on overflow."""
    if q == 0:
        return 0.0
    s = -1 if q < 0 else 1
    a = abs(q)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if F(2) ** e > a:
        e -= 1
    e = max(e, emin)
    ulp = F(2) ** (e - mbits)
    n = a / ulp
    fl = math.floor(n)
    r = n - fl
    if r > F(1, 2) or (r == F(1, 2) and fl % 2):
        fl += 1
    v = fl * ulp
    if v >= F(2) ** (emax + 1):
        return s * math.inf
    return s * float(v)


def r32(q): return rnd(q, 23, -126, 127)
def r16(q): return rnd(q, 10, -14, 15)


def same(got, want, bits=32):
    g = f32v(got) if bits == 32 else f16v(got)
    if want == 0:
        return g == 0 and math.copysign(1, g) == math.copysign(1, want)
    return g == want


def finite(*xs): return all(math.isfinite(x) for x in xs)


def flush32(x, flush):
    return math.copysign(0.0, x) if flush and x != 0 and abs(x) < 2.0 ** -126 else x


def lds_prologue(params, ij):
    """Lane 0 writes the attribute words, each lane its I/J pair; then M0 = 0 and v20/v21 = I/J."""
    L = ["s_mov_b32 s40, exec_lo", "s_mov_b32 exec_lo, 1"]

    def wr(addr, val):
        L.extend([f"v_mov_b32_e32 v31, {addr:#x}", f"v_mov_b32_e32 v32, {val:#x}", "ds_write_b32 v31, v32"])
    for (a, c), (p0, p10, p20) in params.items():
        wr(48 * a + 8 * c, p0)
        wr(48 * a + 8 * c + 4, p10)
        wr(48 * a + 32 + 4 * c, p20)
    for lane, (i, j) in enumerate(ij):
        L.append(f"s_mov_b32 exec_lo, {1 << lane:#x}")
        wr(IJ_BASE + 8 * lane, i)
        wr(IJ_BASE + 8 * lane + 4, j)
    L += ["s_mov_b32 exec_lo, s40", "s_waitcnt lgkmcnt(0)", "s_barrier",
          "v_lshlrev_b32_e32 v33, 3, v0", f"v_add_nc_u32_e32 v33, {IJ_BASE:#x}, v33",
          "ds_read_b32 v20, v33", "ds_read_b32 v21, v33 offset:4", "s_waitcnt lgkmcnt(0)",
          "s_mov_b32 m0, 0", "s_nop 7"]
    return L


# --- random kernels --------------------------------------------------------------------------------------------------

def gen_word(rng, packed):
    if packed:
        while True:
            lo, hi = rng.getrandbits(16), rng.getrandbits(16)
            if (lo & 0x7C00) != 0x7C00 and (hi & 0x7C00) != 0x7C00:   # no f16 inf/NaN halves
                return lo | hi << 16
    return f32w(rng.uniform(-8, 8) * 2.0 ** rng.randint(-6, 6))


NOUT_PER = 7


def random_kernel(rng, mode):
    params, kinds = {}, {}
    for a in range(ATTRS):
        packed = a % 2 == 1 or rng.random() < 0.5
        for c in range(4):
            params[a, c] = tuple(gen_word(rng, packed) for _ in range(3))
            kinds[a, c] = "packed" if packed else "f32"
    ij = [(f32w(rng.uniform(-0.25, 1.25)), f32w(rng.uniform(-0.25, 1.25))) for _ in range(LANES)]
    L = lds_prologue(params, ij)
    nout = ATTRS * 4 * NOUT_PER
    L.append(f"v_mul_u32_u24_e32 v34, {4 * nout}, v0")
    k = 0

    def st(reg):
        nonlocal k
        L.append(f"global_store_dword v34, v{reg}, s[2:3] offset:{4 * k}")
        k += 1
    for a in range(ATTRS):
        for c in "xyzw":
            at = f"attr{a}.{c}"
            L += [f"v_interp_p1_f32 v2, v20, {at}", "s_nop 3"]; st(2)
            L += [f"v_interp_p2_f32 v2, v21, {at}", "s_nop 3"]; st(2)
            L += [f"v_interp_p1ll_f16 v3, v20, {at}", "s_nop 3"]; st(3)
            L += ["v_mov_b32_e32 v5, 0xdeadbeef", f"v_interp_p2_f16 v5, v21, {at}, v3", "s_nop 3"]; st(5)
            L += [f"v_interp_p1ll_f16 v3, v20, {at} high", "s_nop 3"]; st(3)
            L += ["v_mov_b32_e32 v5, 0xdeadbeef", f"v_interp_p2_f16 v5, v21, {at}, v3 high", "s_nop 3"]; st(5)
            L += [f"v_interp_mov_f32 v6, p0, {at}", "s_nop 3"]; st(6)
    L += ["s_waitcnt vmcnt(0)", "s_endpgm"]
    kern = asm.Kernel(L, wave=32, threads=LANES, words=nout * LANES, raw=True, lds=2048, vgprs=40, sgprs=48, mode=mode)
    return kern, (params, kinds, ij, nout)


def check_random(res, info, flush, tally, bad):
    params, kinds, ij, nout = info
    for lane, (ib, jb) in enumerate(ij):
        I, J = F(f32v(ib)), F(f32v(jb))
        for (a, ci), (p0, p10, p20) in params.items():
            o = res[lane * nout + (a * 4 + ci) * NOUT_PER:][:NOUT_PER]
            kind = kinds[a, ci]

            def t(name, ok, detail):
                tally.setdefault((kind, name), [0, 0])[0 if ok else 1] += 1
                if not ok and len(bad) < 10:
                    bad.append(f"{kind} {name}: lane {lane} attr{a}.{'xyzw'[ci]} words {p0:#010x} {p10:#010x} "
                               f"{p20:#010x} I/J {ib:#010x} {jb:#010x} {detail}")
            t("mov p0 = raw word", o[6] == p0, f"got {o[6]:#010x}")
            P0, P10, P20 = f32v(p0), f32v(p10), f32v(p20)
            if finite(P0, P10, P20):
                w1 = flush32(r32(F(flush32(P10, flush)) * I + F(flush32(P0, flush))), flush)
                t("p1_f32 = fma(P10, I, P0)", same(o[0], w1), f"got {o[0]:#010x} want {f32w(w1):#010x}")
                g1 = f32v(o[0])
                if math.isfinite(g1):
                    w2 = flush32(r32(F(flush32(P20, flush)) * J + F(g1)), flush)
                    t("p2_f32 = fma(P20, J, p1)", same(o[1], w2), f"got {o[1]:#010x} want {f32w(w2):#010x}")
            for hi, (i1, i2) in ((0, (2, 3)), (1, (4, 5))):
                sh = 16 * hi
                h0, h10, h20 = f16v(p0 >> sh), f16v(p10 >> sh), f16v(p20 >> sh)
                tag = "high" if hi else "low"
                if not finite(h0, h10, h20):     # an f32 word's halves can be f16 inf/NaN: not modelled
                    continue
                w1 = r32(F(h10) * I + F(h0))
                t(f"p1ll_f16 {tag} = fma(h10, I, h0) in f32", same(o[i1], w1), f"got {o[i1]:#010x} want {f32w(w1):#010x}")
                g = f32v(o[i1])
                if math.isfinite(g):
                    w = r16(F(h20) * J + F(g))
                    if math.isfinite(w):
                        t(f"p2_f16 {tag} = fma(h20, J, p1ll) in f16", same(o[i2] & 0xFFFF, w, 16),
                          f"got {o[i2]:#010x} want {f16w(w):#06x}")
                    t(f"p2_f16 {tag} keeps vdst[31:16]", o[i2] >> 16 == 0xDEAD, f"got {o[i2]:#010x}")


# --- rounding ties for p2_f16 ----------------------------------------------------------------------------------------

def tie_kernel(mode):
    """Attribute a holds one (h0, h10, h20) triple in both halves; each lane gets I/J that put p1ll on an f16 tie."""
    groups = []
    for base in (1.0, 1.0 + 2 ** -10, -1.0, 0.5, 3.0):
        step = 2.0 ** (math.frexp(abs(base))[1] - 12)        # half an f16 ulp at |base|
        groups.append(((base, math.copysign(1.0, base), 2.0 ** -14),
                       [(k * step, J) for k in (1, 3) for J in (2.0 ** -20, -2.0 ** -20, 2.0 ** -30)]))
    params = {}
    for a, ((h0, h10, h20), _) in enumerate(groups):
        params[a, 0] = tuple(f16w(v) | f16w(v) << 16 for v in (h0, h10, h20))
    lanes = [(a, I, J) for a, (_, ij) in enumerate(groups) for I, J in ij][:LANES]
    ij = [(f32w(I), f32w(J)) for _, I, J in lanes]
    ij += [ij[-1]] * (LANES - len(ij))
    L = lds_prologue(params, ij)
    nout = 4 * len(groups)
    L.append(f"v_mul_u32_u24_e32 v34, {4 * nout}, v0")
    k = 0
    for a in range(len(groups)):
        for hi in ("", " high"):
            L += [f"v_interp_p1ll_f16 v3, v20, attr{a}.x{hi}", "s_nop 3", "v_mov_b32_e32 v5, 0",
                  f"v_interp_p2_f16 v5, v21, attr{a}.x, v3{hi}", "s_nop 3",
                  f"global_store_dword v34, v3, s[2:3] offset:{4 * k}",
                  f"global_store_dword v34, v5, s[2:3] offset:{4 * k + 4}"]
            k += 2
    L += ["s_waitcnt vmcnt(0)", "s_endpgm"]
    kern = asm.Kernel(L, wave=32, threads=LANES, words=nout * LANES, raw=True, lds=2048, vgprs=40, sgprs=48, mode=mode)
    return kern, (groups, lanes, nout)


def check_ties(res, info, tally, bad):
    groups, lanes, nout = info
    for lane, (a, I, J) in enumerate(lanes):
        h20 = groups[a][0][2]
        for hix in range(2):
            p1 = f32v(res[lane * nout + a * 4 + 2 * hix])
            got = res[lane * nout + a * 4 + 2 * hix + 1] & 0xFFFF
            q = F(h20) * F(J) + F(p1)
            once, twice = f16w(r16(q)), f16w(r16(F(r32(q))))
            if once == twice:
                continue
            ok = got == once
            tally.setdefault(("tie", "p2_f16 rounds once (not f32, then f16)"), [0, 0])[0 if ok else 1] += 1
            if not ok and len(bad) < 10:
                bad.append(f"tie: lane {lane} p1ll {p1!r} J {J!r} got {got:#06x} once {once:#06x} twice {twice:#06x}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="gfx1036", help="hw_oracle target (default gfx1036)")
    ap.add_argument("--seeds", default="1,2,3,4", help="comma-separated random seeds (default 1,2,3,4)")
    ap.add_argument("--kernels", type=int, default=4, help="random kernels per seed (default 4)")
    ap.add_argument("--denorm32", type=int, choices=(0, 3), default=3,
                    help="MODE f32 denormal field: 3 keeps them (default), 0 flushes (denormal P0/P10/P20 words then read as 0)")
    args = ap.parse_args(argv)
    try:
        seeds = [int(s) for s in args.seeds.split(",")]
    except ValueError:
        ap.error("--seeds takes integers separated by commas")
    mode = {"denorm32": args.denorm32}
    try:
        o = runner.open_oracle(arch=args.arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}", file=sys.stderr)
        return 77
    tally, bad = {}, []
    try:
        for seed in seeds:
            rng = random.Random(seed)
            ks = [random_kernel(rng, mode) for _ in range(args.kernels)]
            for r, (_, info) in zip(o.run([k for k, _ in ks]), ks):
                if isinstance(r, str):
                    print(f"kernel failed: {r}", file=sys.stderr)
                    return 2
                check_random(r, info, args.denorm32 == 0, tally, bad)
        kern, info = tie_kernel(mode)
        r = o.run([kern])[0]
        if isinstance(r, str):
            print(f"kernel failed: {r}", file=sys.stderr)
            return 2
        check_ties(r, info, tally, bad)
    finally:
        o.close()
    print(f"{args.arch}, MODE denorm32 {args.denorm32}, seeds {args.seeds} x {args.kernels} kernels")
    for (kind, name), (ok, nbad) in sorted(tally.items()):
        print(f"  {kind:7s} {name:44s} {ok:6d} match {nbad:5d} differ")
    for b in bad:
        print("  mismatch:", b)
    return 1 if any(n for _, n in tally.values()) else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
