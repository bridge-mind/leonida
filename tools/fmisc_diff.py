r"""Differential fuzzing of the 32-bit float VALU ops no other fuzzer generates, and of float compares. Standard
library only.

Generates seeded straight-line programs, one lane, runs them through

  * the reference below, written from the RDNA2 ISA guide's pseudocode for each opcode with exact rational
    arithmetic and one round-to-nearest-even per operation, and
  * tools/rdna2spirv.py, executing the SPIR-V with tests/semantic/spirv_mini_interp.py (lane 0 of a wave32),

and compares four exported words bit for bit (v8..v11), except that any NaN matches any NaN. The reference does
not share code with the translator, with tools/cpu_ref.py or with the other fuzzers. tools/f32_fuzz.py covers the
float arithmetic (add, mul, fma, floor, the transcendentals, v_cmp_class_f32 without modifiers, ...); this tool
covers the ops it does not generate.

What is modelled, per instruction (OPS, --list-ops):
  * v_fract_f32: S0 - floor(S0), at most 0x3f7fffff (a tiny negative S0 would round to 1.0); +-inf gives
    0xffc00000, a NaN comes back unchanged (a signalling one too); an integer gives +0.
  * v_frexp_mant_f32 / v_frexp_exp_i32_f32: S0 = mant * 2^exp, mant in [0.5, 1) with S0's sign. +-0 gives +-0 and
    0; +-inf and NaN give S0 (mant) and 0 (exp).
  * v_ldexp_f32 (VOP3 only): S0 * 2^S1, S1 a signed integer; overflow is +-inf, +-inf / +-0 / NaN pass through.
  * v_cvt_u32_f32 / v_cvt_i32_f32: truncate toward zero, saturate; NaN gives 0.
  * v_cvt_off_f32_i4: the signed 4-bit S0[3:0] / 16.
  * v_fmamk_f32: S0 * K + S1 with one rounding, K the literal.
  * v_cmp_<cond>_f32 for all 16 conditions (VOPC e32 into vcc_lo; e64 into s20, s21 or vcc_lo) and
    v_cmp_class_f32 e64, each followed by v_cndmask_b32_e64 on the mask (one lane: the mask is that lane's bit).
    The class bits are 0 signalling NaN, 1 quiet NaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal,
    8 +normal, 9 +inf.
  * VOP3 neg and abs on float sources (abs first), and on float results mul:2 / mul:4 / div:2 then clamp to [0, 1].
  * sources: VGPRs v0..v7 and v12..v19, SGPRs s4..s19, integer and float inline constants and literals, at most one
    literal and two scalar values per VOP3 instruction (the GFX10 constant bus); an e32 VOPC or VOP1 takes a literal
    only in src0.
Every rule was checked against an RDNA2 GPU (AMD Radeon iGPU, gfx1036, Windows driver 26.10.1) with --oracle hw.

Not judged (the program is skipped with the reason, counted): a denormal operand or result (whether f32 denormals
are flushed is the MODE question, SHADER-35), a NaN whose payload is later read as an integer (payloads are not
modelled), the clamp of a NaN and an output modifier whose result is a zero (tools/f32_fuzz.py measures both).

Known divergences (KNOWN_DIVERGENCES): by default the generator leaves out the forms where the translator is known to
differ from the hardware, so that a run shows new problems only; --known puts them back (a hardware-only run always
has them):
  * class-snan-mods: v_cmp_class_f32 with neg or abs on its float source. The hardware flips or clears the sign bit
    of a signalling NaN and still reports it as signalling (class bit 0). The translator applies the modifier with
    OpFNegate / FAbs, so the result depends on whether the float operation keeps the NaN signalling; in
    tests/semantic/spirv_mini_interp.py it does not, and the NaN tests as quiet (bit 1). Posted as a finding in the
    coordination issue (#2300).

With --oracle hw the same programs also run natively on an AMD GPU (tools/hw_oracle/), one lane of a wave32 compute
dispatch under the MODE inferred for the game's shaders (HW_MODE: IEEE off, f32 denormals flushed; with IEEE on the
hardware ignores omod), and the four words are compared with the reference: that validates the reference itself.
Without the oracle tools the run exits 77.

Usage:
    python -I tools/fmisc_diff.py [--programs N] [--seed S] [--max-ops N] [--ops v_ldexp_f32,cmp,class]
    python -I tools/fmisc_diff.py --no-literals         # sources without literals
    python -I tools/fmisc_diff.py --known               # also the forms in KNOWN_DIVERGENCES
    python -I tools/fmisc_diff.py --oracle hw           # hardware vs reference only
    python -I tools/fmisc_diff.py --oracle translator,hw
    python -I tools/fmisc_diff.py --list-ops
    python -I tools/fmisc_diff.py --self-test

Exit 0 when every program matches, 1 on a mismatch (the shortest failing program is printed after removing
instructions that do not matter), 2 on a harness error or when every program was skipped (nothing was compared), 77 when the requested hardware oracle is unavailable.
Skips are counted with their reasons.
"""
import argparse
import gc
import math
import os
import random
import re
import struct
import sys
import tempfile
import warnings
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
if SEM_DIR not in sys.path:
    sys.path.insert(0, SEM_DIR)

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
QNAN = 0x7FC00000
PINF, NINF = 0x7F800000, 0xFF800000

VREGS = tuple(range(0, 8)) + tuple(range(12, 20))   # data VGPRs; v8..v11 are the observation
SREGS = tuple(range(4, 20))                          # data SGPRs
INT_V, INT_S = (0, 1, 2, 3), (4, 5, 6, 7)            # seeded with integers; the rest with floats
FLT_V = tuple(v for v in VREGS if v not in INT_V)
FLT_S = tuple(s for s in SREGS if s not in INT_S)
MASKS = ("s20", "s21", "vcc_lo")
N_SEED = len(VREGS) + len(SREGS)
N_OBS = 4
INLINE_F = {"0.5": 0x3F000000, "-0.5": 0xBF000000, "1.0": 0x3F800000, "-1.0": 0xBF800000, "2.0": 0x40000000,
            "-2.0": 0xC0000000, "4.0": 0x40800000, "-4.0": 0xC0800000, "0.15915494": 0x3E22F983}
FVALS = (0, 0x80000000, 0x3F800000, 0xBF800000, 0x3F000000, 0x3FC00000, 0xBFC00000, 0x7F7FFFFF, 0xFF7FFFFF,
         PINF, NINF, QNAN, 0xFFC00000, 0x7FC00001, 0x7F800001, 0xFFA00000, 0x00800000, 0x80800000, 0x40200000,
         0xC0200000, 0x3F7FFFFF, 0x8DA24260, 0x47F1205C, 0x4F000000, 0x4F800000, 0x4F7FFFFF, 0xCF000000,
         0xCF000001, 0x4B000001, 0x4B7FFFFF, 0xBF7FFFFF, 0x3EFFFFFF, 0x3F000001, 0x40490FDB, 0xC1200000, 0x5F800000,
         0xCB800001, 0x34000000, 0xB3800000, 0x00FFFFFF, 0x80FFFFFF)
NVALS = (0, 1, 2, 7, 8, 15, -1, -8, 127, 128, -126, -127, -149, -150, 254, 255, -255, 0x7FFFFFFF, 0x80000000, 23,
         -23, 64, -16, 0x3FF, 0x200)
FCMP = ("f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru")


class Skip(Exception):
    pass


class Uncertain(Skip):
    """An input whose result this tool does not judge (see the module docstring); a counted skip."""


class RefError(Exception):
    pass


def s32(x):
    x &= M32
    return x - (1 << 32) if x >> 31 else x


# ---- f32 helpers (exact) --------------------------------------------------------------------------------------

def is_nan(w):
    return (w & 0x7F800000) == 0x7F800000 and (w & 0x7FFFFF) != 0


def is_inf(w):
    return (w & 0x7FFFFFFF) == PINF


def is_zero(w):
    return w & 0x7FFFFFFF == 0


def is_denorm(w):
    return (w & 0x7F800000) == 0 and (w & 0x7FFFFF) != 0


def to_frac(w):
    """The exact value of a finite f32 bit pattern."""
    e, m = (w >> 23) & 0xFF, w & 0x7FFFFF
    v = Fraction(m, 1 << 149) if e == 0 else Fraction(m | 0x800000) * Fraction(2) ** (e - 150)
    return -v if w >> 31 else v


def round_f32(fr, neg_zero=False):
    """An exact rational rounded to f32 (nearest even), overflow to inf. A result below 2^-150 in magnitude is a
    zero in every denormal mode; one between that and the smallest normal raises Uncertain."""
    if fr == 0:
        return 0x80000000 if neg_zero else 0
    sign = 0x80000000 if fr < 0 else 0
    a = abs(fr)
    if a < Fraction(1, 1 << 150):
        return sign
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    if e < -126:
        raise Uncertain("denormal result")
    scaled = a / Fraction(2) ** (e - 23)                  # in [2^23, 2^24)
    q, r = divmod(scaled.numerator, scaled.denominator)
    if 2 * r > scaled.denominator or (2 * r == scaled.denominator and q & 1):
        q += 1
    if q == 1 << 24:
        q, e = q >> 1, e + 1
    if e > 127:
        return sign | PINF
    return sign | ((e + 127) << 23) | (q & 0x7FFFFF)


def fract(w):
    # Measured on gfx1036 (docs/design.md, "Float min/max/med3 with NaN, v_fract"): a NaN comes back unchanged, a
    # signalling one too (v_cmp_class_f32 still reports class bit 0), and +-inf gives exactly 0xffc00000.
    if is_nan(w):
        return w
    if is_inf(w):
        return 0xFFC00000
    fr = to_frac(w)
    r = fr - math.floor(fr)
    if r == 0:
        return 0                                          # S0 + -floor(S0): x + -x is +0, so is -0 + +0
    return min(round_f32(r), 0x3F7FFFFF)


def frexp_mant(w):
    if is_nan(w) or is_inf(w) or is_zero(w):
        return w
    return (w & 0x807FFFFF) | (126 << 23)


def frexp_exp(w):
    if is_nan(w) or is_inf(w) or is_zero(w):
        return 0
    return (((w >> 23) & 0xFF) - 126) & M32


def ldexp(w, n):
    if is_nan(w) or is_inf(w) or is_zero(w):
        return w
    n = max(-400, min(400, s32(n)))                       # past +-400 every finite f32 overflows or vanishes
    return round_f32(to_frac(w) * Fraction(2) ** n, bool(w >> 31))


def cvt_u32(w):
    if is_nan(w):
        return 0
    if is_inf(w):
        return 0 if w >> 31 else M32
    return max(0, min(M32, math.trunc(to_frac(w))))


def cvt_i32(w):
    if is_nan(w):
        return 0
    if is_inf(w):
        return 0x80000000 if w >> 31 else 0x7FFFFFFF
    return max(-(1 << 31), min((1 << 31) - 1, math.trunc(to_frac(w)))) & M32


def cvt_off(w):
    n = w & 0xF
    return round_f32(Fraction(n - 16 if n & 8 else n, 16))


def fmamk(a, k, b):
    if is_nan(a) or is_nan(k) or is_nan(b):
        return QNAN
    if is_inf(a) or is_inf(k):
        if is_zero(a) or is_zero(k):
            return QNAN
        p = PINF | ((a ^ k) & 0x80000000)
        if is_inf(b) and b != p:
            return QNAN
        return p
    if is_inf(b):
        return b
    prod = to_frac(a) * to_frac(k)
    fr = prod + to_frac(b)
    neg_zero = prod == 0 and bool((a ^ k) >> 31) and bool(b >> 31)    # -0 + -0; any other exact zero is +0
    return round_f32(fr, neg_zero)


def class_bits(w):
    if is_nan(w):
        return 1 << (1 if (w >> 22) & 1 else 0)
    neg = w >> 31
    if is_inf(w):
        return 1 << (2 if neg else 9)
    if is_zero(w):
        return 1 << (5 if neg else 6)
    if is_denorm(w):
        return 1 << (4 if neg else 7)
    return 1 << (3 if neg else 8)


def fcmp(c, a, b):
    if is_nan(a) or is_nan(b):
        lt = eq = gt = False
        un = True
    else:
        fa = math.inf if a == PINF else -math.inf if a == NINF else to_frac(a)
        fb = math.inf if b == PINF else -math.inf if b == NINF else to_frac(b)
        lt, eq, gt, un = fa < fb, fa == fb, fa > fb, False
    return {"f": False, "lt": lt, "eq": eq, "le": lt or eq, "gt": gt, "lg": lt or gt, "ge": gt or eq,
            "o": not un, "u": un, "nge": not (gt or eq), "nlg": not (lt or gt), "ngt": not gt,
            "nle": not (lt or eq), "neq": not eq, "nlt": not lt, "tru": True}[c]


# name -> (form, float source mask, float result, fn). form: vop1 (e32 and e64), vop3 (VOP3 only), fmamk.
OPS = {
    "v_fract_f32": ("vop1", (1,), True, fract),
    "v_frexp_mant_f32": ("vop1", (1,), True, frexp_mant),
    "v_frexp_exp_i32_f32": ("vop1", (1,), False, frexp_exp),
    "v_cvt_u32_f32": ("vop1", (1,), False, cvt_u32),
    "v_cvt_i32_f32": ("vop1", (1,), False, cvt_i32),
    "v_cvt_off_f32_i4": ("vop1", (0,), True, cvt_off),
    "v_ldexp_f32": ("vop3", (1, 0), True, ldexp),
    "v_fmamk_f32": ("fmamk", (1, 1, 1), True, fmamk),
}
ALL_OPS = sorted(OPS) + ["class", "cmp"]   # cmp / class: a VOPC compare into a mask, then v_cndmask_b32_e64 on it

VK_CONSTANTS = False   # --vk-constants: leave the seeds as literals in the vk programs
KNOWN = False   # also generate the forms in KNOWN_DIVERGENCES (--known; always on for a hardware-only run)
LITERALS = True
KNOWN_DIVERGENCES = {
    "class-snan-mods": "v_cmp_class_f32 with neg/abs on a signalling NaN: the hardware reports it as signalling, the "
                       "translator's OpFNegate / FAbs can quiet it (it does in spirv_mini_interp)",
}


# ---- program generation -------------------------------------------------------------------------------------

def fval(rng):
    r = rng.random()
    if r < 0.55:
        return rng.choice(FVALS)
    if r < 0.8:
        return round_f32(Fraction(rng.randint(-2000, 2000), rng.choice((1, 2, 3, 4, 7, 8, 1024))))
    return (rng.getrandbits(1) << 31) | (rng.randint(1, 254) << 23) | rng.getrandbits(23)


def ival(rng):
    return (rng.choice(NVALS) if rng.random() < 0.7 else rng.randint(-300, 300)) & M32


def fconst(rng):
    """An inline constant for a float source: a float one, or an integer one whose bits are not a denormal."""
    return rng.choice(tuple(INLINE_F)) if rng.random() < 0.7 else str(rng.choice((0, -1, -2, -16)))


class Gen:
    def __init__(self, rng):
        self.r = rng

    def vreg(self, flt=True):
        """A VGPR; mostly one seeded with the kind of value the source reads (an integer read as a float is mostly
        a denormal, which the reference does not judge)."""
        return f"v{self.r.choice(VREGS if self.r.random() < 0.15 else FLT_V if flt else INT_V)}"

    def src(self, st, flt, vop3, lit_ok=True, mods=True):
        """One source; st = [scalar values used, literal used]. VOP3 float sources may get neg / abs (mods)."""
        r = self.r
        k = r.random()
        if k < 0.5:
            t = self.vreg(flt)
        elif k < 0.75:
            s = f"s{r.choice(SREGS if r.random() < 0.15 else FLT_S if flt else INT_S)}"
            if s not in st[0] and len(st[0]) >= (2 if vop3 else 1):
                t = self.vreg(flt)
            else:
                st[0].add(s)
                t = s
        elif k < 0.9 or not (LITERALS and lit_ok and not st[1] and len(st[0]) < (2 if vop3 else 1)):
            t = fconst(r) if flt else str(r.randint(-16, 64))
        else:
            v = fval(r) if flt else ival(r)
            if -16 <= s32(v) <= 64 or v in INLINE_F.values():
                v = 0x7FC00001 if flt else 0x12345
            t = f"0x{v:x}"
            st[0].add("lit")
            st[1] = True
        if vop3 and flt and mods and t[0] in "vs":
            if r.random() < 0.3:
                t = f"|{t}|"
            if r.random() < 0.3:
                t = f"-{t}"
        return t

    def outmods(self):
        r = self.r
        m = " clamp" if r.random() < 0.2 else ""          # GFX10's assembler wants clamp before the omod
        if r.random() < 0.25:
            m += " " + r.choice(("mul:2", "mul:4", "div:2"))
        return m

    def op(self, ops):
        r = self.r
        name = r.choice(ops)
        d = r.choice(INT_V if name in OPS and not OPS[name][2] else FLT_V)
        st = [set(), False]
        if name in ("cmp", "class"):
            m = r.choice(MASKS)
            sel = f"v_cndmask_b32_e64 v{d}, 0, 1.0, {m}"
            if name == "class":
                a = self.src(st, True, True, mods=KNOWN)
                mask = self.src(st, False, True)
                return [f"v_cmp_class_f32_e64 {m}, {a}, {mask}", sel]
            c = r.choice(FCMP)
            if r.random() < 0.4:
                a = self.src(st, True, False)
                return [f"v_cmp_{c}_f32_e32 vcc_lo, {a}, {self.vreg()}", sel.replace(f", {m}", ", vcc_lo")]
            a, b = self.src(st, True, True), self.src(st, True, True)
            return [f"v_cmp_{c}_f32_e64 {m}, {a}, {b}", sel]
        form, fin, fout, _ = OPS[name]
        if form == "vop1":
            if r.random() < 0.4:
                return [f"{name}_e32 v{d}, {self.src(st, fin[0], False)}"]
            return [f"{name}_e64 v{d}, {self.src(st, fin[0], True)}{self.outmods() if fout else ''}"]
        if form == "vop3":
            a = self.src(st, True, True)
            return [f"{name} v{d}, {a}, {self.src(st, False, True)}{self.outmods()}"]
        # fmamk: src0 any (an SGPR or inline constant next to the literal K), K, then a VGPR
        st = [set(), True]                                    # K is the literal: one more scalar value fits
        return [f"v_fmamk_f32 v{d}, {self.src(st, True, False)}, 0x{fval(r):x}, {self.vreg()}"]


def seeds(rng):
    return ([f"v_mov_b32_e32 v{v}, 0x{(ival if v in INT_V else fval)(rng):x}" for v in VREGS]
            + [f"s_mov_b32 s{s}, 0x{(ival if s in INT_S else fval)(rng):x}" for s in SREGS])


def observe(regs):
    return [f"v_mov_b32_e32 v{8 + i}, v{r}" for i, r in enumerate(regs)]


def generate(rng, n_ops, ops):
    gen = Gen(rng)
    body = []
    for _ in range(n_ops):
        body += gen.op(ops)
    written = []
    for line in reversed(body):
        if line.startswith("v_cmp"):
            continue
        dst = int(line.split()[1].rstrip(",")[1:])
        if dst not in written:
            written.append(dst)
    regs = written[:4]
    regs += [rng.choice(VREGS) for _ in range(4 - len(regs))]
    return seeds(rng) + body + observe(regs)


# ---- the reference interpreter --------------------------------------------------------------------------------

_OMOD = {"mul:2": Fraction(2), "mul:4": Fraction(4), "div:2": Fraction(1, 2)}


def parse(rest):
    mods = {}
    keep = []
    for t in rest.split(" "):
        if t in _OMOD:
            mods["omod"] = t
        elif t == "clamp":
            mods["clamp"] = True
        else:
            keep.append(t)
    return [a.strip() for a in " ".join(keep).split(",")], mods


def raw(tok, st):
    if re.fullmatch(r"v\d+", tok):
        return st["v"].get(int(tok[1:]), 0)
    if re.fullmatch(r"s\d+|vcc_lo", tok):
        if tok not in st["s"]:
            raise RefError(f"{tok} read before it is written")
        return st["s"][tok]
    if tok in INLINE_F:
        return INLINE_F[tok]
    return int(tok, 0) & M32


def ivalue(tok, st):
    if re.fullmatch(r"v\d+", tok) and int(tok[1:]) in st["nan"]:
        raise Uncertain("a computed NaN read as an integer")
    return raw(tok, st)


def fvalue(tok, st):
    neg = tok.startswith("-") and not tok[1:2].isdigit()
    if neg:
        tok = tok[1:]
    ab = tok.startswith("|")
    w = raw(tok.strip("|"), st)
    if ab:
        w &= 0x7FFFFFFF
    if neg:
        w ^= 0x80000000
    if is_denorm(w):
        raise Uncertain("denormal operand")
    return w


def out_mods(w, mods):
    om = mods.get("omod")
    if om and not (is_nan(w) or is_inf(w)):
        v = to_frac(w) * _OMOD[om]
        if v == 0:
            raise Uncertain("output modifier on a zero")
        w = round_f32(v)
    if mods.get("clamp"):
        if is_nan(w):
            raise Uncertain("clamp of a NaN")
        w = 0 if w >> 31 else min(w, 0x3F800000)
    return w


def reference(lines):
    """Run the program; returns the four exported words. Raises Uncertain for what it does not judge."""
    st = {"v": {}, "s": {}, "nan": set()}   # nan: VGPRs holding a computed NaN (payload not modelled)
    for line in lines:
        op, _, rest = line.partition(" ")
        args, mods = parse(rest)
        if op == "s_mov_b32":
            st["s"][args[0]] = raw(args[1], st)
            continue
        if op.startswith("v_cmp_"):
            a = fvalue(args[1], st)
            if op.startswith("v_cmp_class_f32"):
                bit = class_bits(a) & ivalue(args[2], st)
            else:
                bit = fcmp(op[len("v_cmp_"):].split("_f32")[0], a, fvalue(args[2], st))
            st["s"][args[0]] = int(bool(bit))
            continue
        d = int(args[0][1:])
        if op in ("v_mov_b32_e32", "v_cndmask_b32_e64"):
            src = args[1] if op == "v_mov_b32_e32" else args[2 if raw(args[3], st) & 1 else 1]
            st["v"][d] = raw(src, st)
            nan = re.fullmatch(r"v\d+", src) and int(src[1:]) in st["nan"]
            (st["nan"].add if nan else st["nan"].discard)(d)
            continue
        base = op[:-4] if op.endswith(("_e32", "_e64")) else op
        if base not in OPS:
            raise RefError(f"{op} is not modelled")
        form, fin, fout, fn = OPS[base]
        if form == "fmamk":
            vals = [fvalue(args[1], st), fvalue(args[2], st), fvalue(args[3], st)]
        else:
            vals = [fvalue(t, st) if fin[i] else ivalue(t, st) for i, t in enumerate(args[1:])]
        w = fn(*vals) & M32
        if fout:
            if is_denorm(w):
                raise Uncertain("denormal result")
            w = out_mods(w, mods)
        (st["nan"].add if fout and is_nan(w) else st["nan"].discard)(d)
        st["v"][d] = w
    return [st["v"].get(8 + i, 0) for i in range(4)]


# ---- the translator side -----------------------------------------------------------------------------------

def translated(lines, name="ps_fmisc"):
    """Translate and execute; returns the four exported words, or raises Skip."""
    import rdna2spirv
    from spirv_mini_interp import run_fragment
    with tempfile.TemporaryDirectory(prefix="fmisc-diff-") as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        path = os.path.join(dis, name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}  fmisc_diff case\n")
            for line in lines:
                f.write("\t" + line + "\n")
            f.write("\texp mrt0 v8, v9, v10, v11 done vm\n\ts_endpgm\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spvasm = rdna2spirv.translate(path)
                gc.collect()
        except rdna2spirv.Unsupported as e:
            raise Skip(f"translator unsupported: {e}") from None
        try:
            out = run_fragment(spvasm)
        except Exception as e:  # noqa: BLE001 - an interpreter gap is a counted skip, never a pass
            raise Skip(f"mini_interp: {type(e).__name__}: {str(e)[:80]}") from None
    return [w & M32 for w in out["%out_mrt0"]]


# The float MODE the hardware kernels start with: the one docs/design.md infers for the game's shaders (IEEE off,
# DX10_CLAMP on, f32 denormals flushed; tools/f32_fuzz.py's "flush"). The assembler's compute default has IEEE on,
# under which the hardware ignores omod (measured: v_fract_f32_e64 v4, -v16 mul:2 with v16 = 0.5 gives 0.5).
HW_MODE = {"ieee": 0, "dx10_clamp": 1, "denorm32": 0}


def hw_kernel(lines, wave=32):
    """A raw kernel for tools/hw_oracle: the program on one lane, then v8..v11 stored at the output pointer. wave 64
    runs it in a wave64 wave with the lane masks as 64-bit pairs."""
    from hw_oracle import asm
    if wave == 64:
        import vk_fuzz_run
        lines = [vk_fuzz_run.wave64_text(ln) for ln in lines]
    return asm.Kernel(list(lines) + ["v_mov_b32 v0, 0", "global_store_b128 v0, v[8:11], s[2:3]", "s_endpgm"],
                      wave=wave, threads=1, words=4, raw=True, mode=HW_MODE)


# ---- comparison, shrinking, the run --------------------------------------------------------------------------

def _diff(lines, want, got, label):
    bad = [k for k in range(4) if want[k] != got[k] and not (is_nan(want[k]) and is_nan(got[k]))]
    if not bad:
        return None
    i = bad[0]
    return f"word {i} ({lines[-N_OBS + i].split(', ')[1]}): reference 0x{want[i]:08x}, {label} 0x{got[i]:08x}"


def check(lines, backends=("translator",), hw=None):
    """None on a match, else a description. Raises Skip (every backend skipped, or Uncertain) / RefError.
    hw is the hardware result for this program (four words or a str reason), or a callable lines -> that."""
    want = reference(lines)
    problems, skips = [], []
    if "translator" in backends:
        try:
            problems.append(_diff(lines, want, translated(lines), "translated"))
        except Skip as e:
            skips.append(e)
    if "hw" in backends:
        got = hw(lines) if callable(hw) else hw
        if isinstance(got, str):
            skips.append(Skip(f"hardware: {got}"))
        else:
            problems.append(_diff(lines, want, got, "hardware"))
    if skips and len(skips) == len(backends):
        raise skips[0]
    return "; ".join(p for p in problems if p) or None


def shrink(lines, check_fn=check):
    """Observe one register if that still fails, drop instructions (a compare and its v_cndmask together) while the
    program still mismatches, then the seeds of registers nothing reads."""
    def fails(trial):
        try:
            return bool(check_fn(trial))
        except (Skip, RefError):
            return False
    seed, body, obs = lines[:N_SEED], lines[N_SEED:-N_OBS], lines[-N_OBS:]
    for line in obs:
        one = [line.replace(line.split()[1], f"v{8 + i},") for i in range(4)]
        if fails(seed + body + one):
            obs = one
            break
    changed = True
    while changed:
        changed = False
        i = len(body) - 1
        while i >= 0:
            n = 2 if body[i].startswith("v_cndmask") and i and body[i - 1].startswith("v_cmp") else 1
            trial = body[:i + 1 - n] + body[i + 1:]
            if fails(seed + trial + obs):
                body, changed = trial, True
            i -= n
    used = set()
    for line in body + obs:
        used |= set(re.findall(r"\b[vs]\d+\b", line.partition(" ")[2]))
    seed = [line for line in seed if line.split()[1].rstrip(",") in used]
    return seed + body + obs


def omod_writer(lines, k):
    """Whether the last instruction that wrote the register observed as word k carries an output modifier."""
    reg = lines[-N_OBS + k].split(", ")[1]
    for line in reversed(lines[:-N_OBS]):
        op, _, rest = line.partition(" ")
        if rest.split(",")[0].strip() == reg and not op.startswith("v_cmp"):
            return "mul:" in line or "div:" in line
    return False


def vk_verdict(lines, hw, got):
    """(class, detail) for a program translated and run on a Vulkan GPU against the hardware's four words:
    match (a NaN matches any NaN), denormal (the words differ and the reference declines the program for a denormal:
    RDNA2 in the game's MODE flushes f32 denormals, the translator leaves them to the driver), omod_neg_zero (an output
    modifier turned the hardware's -0 into +0 and the translated -0 stayed: the translator does not model it, #1281)
    or mismatch."""
    bad = [k for k in range(4) if hw[k] != got[k] and not (is_nan(hw[k]) and is_nan(got[k]))]
    if not bad:
        return "match", ""
    try:
        reference(lines)
    except Skip as e:
        if "denormal" in str(e):
            return "denormal", str(e)
    except RefError:
        pass
    if all(hw[k] == 0 and got[k] == 0x80000000 and omod_writer(lines, k) for k in bad):
        return "omod_neg_zero", "omod made the hardware's -0 a +0"
    i = bad[0]
    return "mismatch", f"word {i} ({lines[-N_OBS + i].split(', ')[1]}): hardware 0x{hw[i]:08x}, translated on the GPU 0x{got[i]:08x}"


def run_suite(programs, seed, max_ops, ops, backends=("translator",), oracle=None, wave=32, build=None, work=None):
    counts = {"pass": 0, "mismatch": 0, "skip": {}, "vk": {}}
    failures = []
    rng = random.Random(seed)
    progs = [generate(rng, rng.randrange(1, max_ops + 1), ops) for _ in range(programs)]
    hw = oracle.run([hw_kernel(p, wave) for p in progs]) if "hw" in backends else [None] * programs
    vk_failures = []
    if "vk" in backends:
        import vk_fuzz_run
        vk = vk_fuzz_run.run_programs(progs, 4, wave, build, work, opaque=not VK_CONSTANTS)
        for i, lines in enumerate(progs):
            if isinstance(hw[i], str) or isinstance(vk[i], str):
                why = ("hardware: " + hw[i]) if isinstance(hw[i], str) else vk[i]
                counts["vk"]["skip: " + why] = counts["vk"].get("skip: " + why, 0) + 1
                continue
            klass, detail = vk_verdict(lines, hw[i], vk[i])
            counts["vk"][klass] = counts["vk"].get(klass, 0) + 1
            if klass == "mismatch":
                vk_failures.append((i, detail, lines))
    counts["vk_failures"] = vk_failures
    backends = tuple(b for b in backends if b != "vk")
    for i, lines in enumerate(progs):
        try:
            problem = check(lines, backends, hw[i])
        except Skip as e:
            reason = ("uncertain: " if isinstance(e, Uncertain) else "") + str(e)
            counts["skip"][reason] = counts["skip"].get(reason, 0) + 1
            continue
        if problem is None:
            counts["pass"] += 1
            continue
        counts["mismatch"] += 1
        failures.append((i, problem, lines))
    return counts, failures


def self_test():
    """Harness canaries: a hand program matches, the reference gives hand-worked values, a bad form is a skip."""
    base = ([f"v_mov_b32_e32 v{v}, 0x{0x3F800000 + (v << 20):x}" for v in VREGS]      # v1 1.125, v2 1.25, ...
            + [f"s_mov_b32 s{s}, {s}" for s in SREGS])
    body = ["v_ldexp_f32 v1, -v2, s4",                                 # -1.25 * 16
            "v_fract_f32_e64 v3, v3 mul:2",                            # 1.375 -> 0.375 -> 0.75
            "v_cmp_nge_f32_e64 s20, |v1|, 0x7fc00001",                  # unordered: true
            "v_cndmask_b32_e64 v4, 0, 1, s20"]
    lines = base + body + observe([1, 3, 4, 2])
    ok = check(lines) is None
    worked = reference(lines) == [0xC1A00000, 0x3F400000, 1, 0x3FA00000]
    try:
        translated(base + ["v_fract_f32_sdwa_dpp v1, v2"] + observe([1, 2, 3, 4]))
        skipped = False
    except Skip:
        skipped = True
    print(f"self-test: match {ok}, hand values {worked}, unsupported form skipped {skipped}")
    return ok and worked and skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description="f32 misc ops and float compares: differential fuzzing.")
    ap.add_argument("--programs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=5)
    ap.add_argument("--ops", default=None, help="comma-separated opcodes (--list-ops; cmp and class = compare + "
                                                "v_cndmask)")
    ap.add_argument("--no-literals", action="store_true", help="no literal sources (v_fmamk_f32 keeps its K)")
    ap.add_argument("--list-ops", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--all-failures", action="store_true", help="print every failing program, not just the first")
    ap.add_argument("--known", action="store_true",
                    help="also generate the forms where the translator is known to differ (KNOWN_DIVERGENCES; "
                         "always on with --oracle hw alone)")
    ap.add_argument("--oracle", default="translator",
                    help="comma-separated comparison targets: translator (default), hw (natively on an AMD GPU; see "
                         "tools/hw_oracle/) and vk (with hw: the translated programs on a Vulkan GPU through "
                         "vk_compute_test, judged against the hardware's words). 'hw' alone compares the hardware "
                         "with the reference only")
    ap.add_argument("--hw-arch", default=None, help="GPU architecture to assemble for (default: detected)")
    ap.add_argument("--wave", type=int, choices=(32, 64), default=32,
                    help="the wave size of the hardware kernels and of the vk programs (64 needs native 64-wide "
                         "subgroups; lane 0 is judged)")
    ap.add_argument("--build", default=os.path.join(os.path.dirname(HERE), "runtime", "build"),
                    help="with vk: the folder with vk_compute_test")
    ap.add_argument("--vk-constants", action="store_true",
                    help="with vk: keep the seeds as literal v_mov_b32 constants (the driver may fold them)")
    ap.add_argument("--work", default=None, help="with vk: a folder for the temporary files (VK_FUZZ_KEEP=1 keeps them)")
    a = ap.parse_args(argv)
    if a.programs < 1 or a.max_ops < 1:
        ap.error("--programs and --max-ops must be at least 1")
    backends = tuple(dict.fromkeys(t.strip() for t in a.oracle.split(",")))
    if not backends or any(t not in ("translator", "hw", "vk") for t in backends):
        ap.error("--oracle takes translator, hw, vk (with hw) or a comma-separated mix")
    if "vk" in backends and "hw" not in backends:
        ap.error("--oracle vk judges the translated programs against the hardware's words: use --oracle hw,vk")
    global KNOWN, LITERALS, VK_CONSTANTS
    VK_CONSTANTS = a.vk_constants
    KNOWN = a.known or backends == ("hw",)
    LITERALS = not a.no_literals
    if a.list_ops:
        for name in ALL_OPS:
            if name == "cmp":
                print(f"{'cmp':20} v_cmp_<cond>_f32 e32 / e64 + v_cndmask_b32_e64: {', '.join(FCMP)}")
            elif name == "class":
                print(f"{'class':20} v_cmp_class_f32_e64 + v_cndmask_b32_e64")
            else:
                form, _, fout, _ = OPS[name]
                kind = {"vop1": "VOP1 e32/e64", "vop3": "VOP3 only", "fmamk": "VOP2 with literal K"}[form]
                print(f"{name:20} {kind}{', omod/clamp' if fout and form != 'fmamk' else ''}")
        return 0
    if a.self_test:
        return 0 if self_test() else 1
    if a.ops is not None:
        tokens = [t.strip() for t in a.ops.split(",")]
        if not any(tokens) or any(not t for t in tokens):
            ap.error("--ops must be a comma-separated list of opcodes")
        ops = sorted(set(tokens))
    else:
        ops = list(ALL_OPS)
    unknown = [o for o in ops if o not in ALL_OPS]
    if unknown:
        ap.error("unknown ops: " + ", ".join(unknown))
    oracle = None
    if "hw" in backends:
        from hw_oracle import runner
        try:
            oracle = runner.open_oracle(arch=a.hw_arch)
        except runner.Unavailable as e:
            print(f"hardware oracle unavailable: {e}")
            return 77
        print(f"hardware oracle: {oracle.device}")
    try:
        counts, failures = run_suite(a.programs, a.seed, a.max_ops, ops, backends, oracle, a.wave, a.build, a.work)
        vk_bad = counts.pop("vk_failures")
        skips = sum(counts["skip"].values())
        print(f"{a.programs} programs (seed {a.seed}, {'+'.join(backends)}{', known' if KNOWN else ''}): "
              f"{counts['pass']} pass, {counts['mismatch']} mismatch, {skips} skipped")
        for reason, n in sorted(counts["skip"].items(), key=lambda kv: -kv[1]):
            print(f"  skip x{n}: {reason}")

        want_vk = "vk" in backends
        if want_vk:
            print(f"translated on the Vulkan GPU (wave{a.wave}) against the hardware: "
                  + ", ".join(f"{k} {n}" for k, n in sorted(counts["vk"].items(), key=lambda kv: kv[0].startswith("skip"))))
            for i, detail, lines in vk_bad[: None if a.all_failures else 3]:
                print(f"VK MISMATCH program {i}: {detail}")
                print("\n".join("\t" + line for line in lines))
        backends = tuple(b for b in backends if b != "vk")

        def check_fn(lines):
            return check(lines, backends, (lambda l: oracle.run([hw_kernel(l, a.wave)])[0]) if oracle else None)
        for i, problem, lines in failures[: None if a.all_failures else 1]:
            small = shrink(lines, check_fn)
            print(f"MISMATCH program {i}: {check_fn(small) or problem}")
            print("\n".join("\t" + line for line in small))
    except (RefError, RuntimeError) as e:
        print(f"harness error: {e}")
        return 2
    finally:
        if oracle:
            oracle.close()
    if failures or vk_bad:
        return 1
    if not counts["pass"]:
        print("no program ran: every one was skipped")
        return 2
    if want_vk:
        import vk_fuzz_run
        if vk_fuzz_run.all_skipped(counts["vk"]):
            print("no Vulkan program ran: every one was skipped")
            return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
