r"""Differential fuzzing of SDWA (sub-dword addressing) VALU forms. Standard library only.

Generates seeded straight-line programs of `_sdwa` VOP1/VOP2/VOPC instructions, one lane, runs them through

  * the reference below, written from the RDNA2 ISA guide's SDWA description and each opcode's pseudocode, and
  * tools/rdna2spirv.py, executing the SPIR-V with tests/semantic/spirv_mini_interp.py,

and compares four exported words bit for bit: three VGPRs and VCC (through v_cndmask_b32_e64). The reference does
not share code with the translator or with tools/cpu_ref.py.

What is modelled, per instruction:
  * sources: VGPRs v1..v7, SGPRs s4..s7 (seeded with s_mov_b32, so the translator knows them as constants), integer
    inline constants -16..64 and float inline constants (0.5, 1.0, 2.0, 4.0 and their negatives). A float constant
    in an f16 op (v_cmp_class_f16's mask included) is the f16 value over a zero high half, in any other op
    (v_cvt_f16_{u16,i16} included) its f32 bits. A source is narrowed to src0_sel/src1_sel (BYTE_0..3, WORD_0/1,
    DWORD), constants too, zero-extended, or sign-extended under sext() (integer ops); then neg/abs (f16 ops, on
    the sign bit of the f16 word) and the op.
  * dst_sel and dst_unused: the result's low bits go to the selected field; UNUSED_PAD zeroes the other bits,
    UNUSED_SEXT fills the bits above the field with its top bit (bits below stay zero), UNUSED_PRESERVE keeps them.
    A 16-bit result (f16 ops, v_cvt_{u16,i16}_f16, v_cvt_f16_{u16,i16}) with dst_sel:DWORD zeroes the high half,
    whatever dst_unused says: unlike the non-SDWA forms (docs/design.md "16-bit results"), which keep it.
  * clamp: unsigned saturation on v_add/sub/subrev_nc_u32, [0, 1] on f16 results (clamp(-0) = +0) and on
    v_cvt_f32_f16.
  * f16 arithmetic rounds to nearest even with denormals kept (the MODE the translator assumes); min/max order -0
    below +0; v_fract_f16(-0) = +0 and the result stops at 0x3bff.
  * VOPC writes VCC (one bit, this lane); v_cndmask_b32 reads it.
Every one of these rules was checked against an RDNA2 GPU (AMD Radeon iGPU, gfx1036, Windows driver 26.10.1) with
--oracle hw, the f16 constant, dst_sel:DWORD and v_fract_f16(-0) rules included.

The int family is the 32-bit integer VOP2 ops, v_cndmask_b32, the integer VOP1 ops and v_cvt_f32_{u32,i32,ubyteN},
and the 32- and 16-bit integer compares. The f16 family is v_add/sub/subrev/mul/min/max_f16, the f16 rounding ops
and v_fract_f16, the f16 <-> 16-bit integer and f16 -> f32 conversions, the f16 compares and v_cmp_class_f16.
--list-ops prints them. Not generated: omod, v_cmpx, an SGPR destination for VOPC, DPP, literals (SDWA has none),
transcendentals, f32 arithmetic.

Skips (counted with their reasons): an f16 NaN operand of an op whose NaN result or IEEE-mode NaN rule is not
modelled (the compares and v_cmp_class_f16 take NaNs), v_fract_f16 of an infinity, and anything the translator or
the interpreter does not take.

Known divergences (KNOWN_DIVERGENCES): by default the generator leaves out the forms where the translator is known to
differ from the hardware, so that a run shows new problems only; --known puts them back (a hardware-only run always
has them):
  * a constant shift amount (inline constant or seeded SGPR) for v_lshlrev_b32 / v_lshrrev_b32 / v_ashrrev_i32 with
    src0_sel other than DWORD or sext(): the translator's constant fold shifted by the whole constant (#2005).
Two forms that used to be listed here are generated in every run again now that main translates them: a 16-bit
result with dst_sel:DWORD (the high half zeroed, #2187) and a float inline constant with a src_sel other than DWORD
in an f16 op (the selection made from the f16 value, #2209).

With --oracle hw the same programs also run natively on an AMD GPU (tools/hw_oracle/) and the hardware's four words
are compared with the reference: that validates the reference itself. SDWA exists on GFX9 and GFX10 only, so the
hardware oracle needs an RDNA1/RDNA2 GPU; anything else exits 77, like a missing tool. The kernel runs on one lane
of a wave32 compute dispatch, so IEEE mode is on there (it only changes NaN rules, which are skipped).

Usage:
    python -I tools/sdwa_diff.py [--programs N] [--seed S] [--max-ops N] [--ops v_add_f16,v_cmp_lt_u16]
    python -I tools/sdwa_diff.py --family f16          # int, f16 or all (default)
    python -I tools/sdwa_diff.py --known               # also the forms in KNOWN_DIVERGENCES
    python -I tools/sdwa_diff.py --oracle hw           # hardware vs reference only
    python -I tools/sdwa_diff.py --oracle translator,hw
    python -I tools/sdwa_diff.py --list-ops
    python -I tools/sdwa_diff.py --self-test

Exit 0 when every program matches, 1 on a mismatch (the shortest failing program is printed after removing
instructions that do not matter), 2 on a harness error, 77 when the requested hardware oracle is unavailable.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
if SEM_DIR not in sys.path:
    sys.path.insert(0, SEM_DIR)

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

VREGS = tuple(range(1, 8))    # data VGPRs v1..v7 (v0 is the hardware kernel's store offset)
SREGS = (4, 5, 6, 7)          # data SGPRs s4..s7
N_SEED = len(VREGS) + len(SREGS) + 1   # the seed lines: VGPRs, SGPRs, VCC
N_OBS = 4                     # the observation lines: v8..v10 copies, v11 = VCC

SEL = {"BYTE_0": (0, 8), "BYTE_1": (8, 8), "BYTE_2": (16, 8), "BYTE_3": (24, 8),
       "WORD_0": (0, 16), "WORD_1": (16, 16), "DWORD": (0, 32)}
UNUSED = ("UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE")
FLOAT_CONSTS = {"0.5": 0x3F000000, "-0.5": 0xBF000000, "1.0": 0x3F800000, "-1.0": 0xBF800000,
                "2.0": 0x40000000, "-2.0": 0xC0000000, "4.0": 0x40800000, "-4.0": 0xC0800000}
FLOAT_CONSTS_F16 = {"0.5": 0x3800, "-0.5": 0xB800, "1.0": 0x3C00, "-1.0": 0xBC00,
                    "2.0": 0x4000, "-2.0": 0xC000, "4.0": 0x4400, "-4.0": 0xC400}

INTERESTING = (0, 1, 2, 0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000, M32,
               0x00FF00FF, 0x80808080, 0x8000FFFF, 0x12345678, 0xDEADBEEF, 31, 32, 33)
F16_INTERESTING = (0x0000, 0x8000, 0x3C00, 0xBC00, 0x3800, 0x3E00, 0xC100, 0x4200, 0x4400, 0x5640, 0xB400,
                   0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x0001, 0x83FF, 0x0400, 0x3BFF, 0x3C01, 0xC500, 0x4900)


class Skip(Exception):
    pass


class RefError(Exception):
    pass


def s32(x):
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def sx(x, w):
    x &= (1 << w) - 1
    return x - (1 << w) if x >> (w - 1) else x


# ---- f16 helpers (bit patterns in, bit patterns out) ------------------------------------------------------

def h2f(h):
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x):
    """Round a Python float (exact here: every f16 sum or product fits a double) to f16, nearest even."""
    if math.isnan(x):
        raise Skip("reference: f16 NaN result (payload not modelled)")
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return 0x7C00 if x > 0 else 0xFC00


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def is_nan16(h):
    return (h & 0x7C00) == 0x7C00 and (h & 0x3FF) != 0


def _no_nan(*hs):
    if any(is_nan16(h) for h in hs):
        raise Skip("reference: f16 NaN operand (IEEE-mode NaN rules not modelled)")


def f16_bin(fn):
    def op(a, b):
        a, b = a & 0xFFFF, b & 0xFFFF
        _no_nan(a, b)
        return f2h(fn(h2f(a), h2f(b)))
    return op


def f16_minmax(want_max):
    def op(a, b):
        a, b = a & 0xFFFF, b & 0xFFFF
        _no_nan(a, b)
        # -0 orders below +0; otherwise the numeric order
        ka, kb = (h2f(a), a >> 15 == 0), (h2f(b), b >> 15 == 0)
        if want_max:
            return a if ka > kb else b
        return a if ka < kb else b
    return op


def f16_round(fn):
    def op(a, b=0):
        a &= 0xFFFF
        _no_nan(a)
        x = h2f(a)
        if math.isinf(x):
            return a
        return f2h(math.copysign(float(fn(x)), x))
    return op


def f16_fract(a, b=0):
    a &= 0xFFFF
    _no_nan(a)
    x = h2f(a)
    if math.isinf(x):
        raise Skip("reference: v_fract_f16 of an infinity (NaN result)")
    # x + -floor(x): fract(-0) = -0 + +0 = +0 (gfx1036); the result stops at the largest f16 below 1.0
    fl = math.copysign(float(math.floor(x)), x) if math.floor(x) == 0 else float(math.floor(x))   # floor(-0) = -0
    return min(f2h(x + -fl), 0x3BFF)


def cvt_u16_f16(a, b=0):
    a &= 0xFFFF
    _no_nan(a)
    x = h2f(a)
    return 0 if x <= 0 else 0xFFFF if x >= 65535 else int(x)


def cvt_i16_f16(a, b=0):
    a &= 0xFFFF
    _no_nan(a)
    x = h2f(a)
    if x >= 32767:
        return 0x7FFF
    if x <= -32768:
        return 0x8000
    return int(x) & 0xFFFF


def cmp_class_f16(a, b):
    a &= 0xFFFF
    e, m, neg = (a >> 10) & 0x1F, a & 0x3FF, a >> 15
    if e == 0x1F:
        bit = (0 if not m >> 9 else 1) if m else (2 if neg else 9)
    elif e == 0:
        bit = (4 if neg else 7) if m else (5 if neg else 6)
    else:
        bit = 3 if neg else 8
    return (b >> bit) & 1


def ffbh_i32(a):
    a &= M32
    if a in (0, M32):
        return M32
    x = a ^ M32 if a >> 31 else a
    return 32 - x.bit_length()


# ---- the reference: one entry per opcode -------------------------------------------------------------
# (family, form, width, fn). family "int" / "f16" decides the modifiers (sext vs neg/abs) and how a float inline
# constant reads. form: "vop2" d, a, b; "vop1" d, a; "vopc" vcc, a, b; "cndmask" d, a, b, vcc.
# width: bits of the result (16 or 32); a 16-bit result has a zero high half under dst_sel:DWORD.
# fn(a, b) gets the selected 32-bit sources and returns the result (a bit for "vopc").

def _ops():
    ops = {}

    def add(name, family, form, fn, width=32, clamp=None):
        ops[name] = (family, form, width, fn, clamp)

    usat = "usat"
    add("v_add_nc_u32", "int", "vop2", lambda a, b: a + b, clamp=usat)
    add("v_sub_nc_u32", "int", "vop2", lambda a, b: a - b, clamp=usat)
    add("v_subrev_nc_u32", "int", "vop2", lambda a, b: b - a, clamp=usat)
    add("v_and_b32", "int", "vop2", lambda a, b: a & b)
    add("v_or_b32", "int", "vop2", lambda a, b: a | b)
    add("v_xor_b32", "int", "vop2", lambda a, b: a ^ b)
    add("v_xnor_b32", "int", "vop2", lambda a, b: ~(a ^ b))
    add("v_lshlrev_b32", "int", "vop2", lambda a, b: b << (a & 31))
    add("v_lshrrev_b32", "int", "vop2", lambda a, b: (b & M32) >> (a & 31))
    add("v_ashrrev_i32", "int", "vop2", lambda a, b: s32(b) >> (a & 31))
    add("v_mul_u32_u24", "int", "vop2", lambda a, b: (a & 0xFFFFFF) * (b & 0xFFFFFF))
    add("v_mul_hi_u32_u24", "int", "vop2", lambda a, b: ((a & 0xFFFFFF) * (b & 0xFFFFFF)) >> 32)
    add("v_mul_i32_i24", "int", "vop2", lambda a, b: sx(a, 24) * sx(b, 24))
    add("v_mul_hi_i32_i24", "int", "vop2", lambda a, b: (sx(a, 24) * sx(b, 24)) >> 32)
    add("v_min_u32", "int", "vop2", lambda a, b: min(a & M32, b & M32))
    add("v_max_u32", "int", "vop2", lambda a, b: max(a & M32, b & M32))
    add("v_min_i32", "int", "vop2", lambda a, b: min(s32(a), s32(b)))
    add("v_max_i32", "int", "vop2", lambda a, b: max(s32(a), s32(b)))
    add("v_cndmask_b32", "int", "cndmask", None)
    add("v_mov_b32", "int", "vop1", lambda a, b: a)
    add("v_not_b32", "int", "vop1", lambda a, b: ~a)
    add("v_bfrev_b32", "int", "vop1", lambda a, b: int(f"{a & M32:032b}"[::-1], 2))
    add("v_ffbh_u32", "int", "vop1", lambda a, b: M32 if not a & M32 else 32 - (a & M32).bit_length())
    add("v_ffbl_b32", "int", "vop1", lambda a, b: M32 if not a & M32 else ((a & -a) & M32).bit_length() - 1)
    add("v_ffbh_i32", "int", "vop1", lambda a, b: ffbh_i32(a))
    add("v_cvt_f32_u32", "int", "vop1", lambda a, b: f2u(float(a & M32)))
    add("v_cvt_f32_i32", "int", "vop1", lambda a, b: f2u(float(s32(a))))
    for n in range(4):
        add(f"v_cvt_f32_ubyte{n}", "int", "vop1", lambda a, b, n=n: f2u(float((a >> (8 * n)) & 0xFF)))
    cmps = {"f": lambda a, b: False, "lt": lambda a, b: a < b, "eq": lambda a, b: a == b,
            "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "ne": lambda a, b: a != b,
            "ge": lambda a, b: a >= b, "t": lambda a, b: True}
    for c, f in cmps.items():
        add(f"v_cmp_{c}_u32", "int", "vopc", lambda a, b, f=f: f(a & M32, b & M32))
        add(f"v_cmp_{c}_i32", "int", "vopc", lambda a, b, f=f: f(s32(a), s32(b)))
        if c not in ("f", "t"):            # GFX10 has no v_cmp_f/t_{i16,u16}
            add(f"v_cmp_{c}_u16", "int", "vopc", lambda a, b, f=f: f(a & 0xFFFF, b & 0xFFFF))
            add(f"v_cmp_{c}_i16", "int", "vopc", lambda a, b, f=f: f(sx(a, 16), sx(b, 16)))

    fclamp = "f16"
    add("v_add_f16", "f16", "vop2", f16_bin(lambda x, y: x + y), 16, fclamp)
    add("v_sub_f16", "f16", "vop2", f16_bin(lambda x, y: x - y), 16, fclamp)
    add("v_subrev_f16", "f16", "vop2", f16_bin(lambda x, y: y - x), 16, fclamp)
    add("v_mul_f16", "f16", "vop2", f16_bin(lambda x, y: x * y), 16, fclamp)
    add("v_max_f16", "f16", "vop2", f16_minmax(True), 16, fclamp)
    add("v_min_f16", "f16", "vop2", f16_minmax(False), 16, fclamp)
    add("v_floor_f16", "f16", "vop1", f16_round(math.floor), 16, fclamp)
    add("v_ceil_f16", "f16", "vop1", f16_round(math.ceil), 16, fclamp)
    add("v_trunc_f16", "f16", "vop1", f16_round(math.trunc), 16, fclamp)
    add("v_rndne_f16", "f16", "vop1", f16_round(round), 16, fclamp)
    add("v_fract_f16", "f16", "vop1", f16_fract, 16, fclamp)
    add("v_cvt_f32_f16", "f16", "vop1", lambda a, b: f2u(h2f(a)) if not is_nan16(a) else _no_nan(a), 32, "f32")
    add("v_cvt_u16_f16", "f16", "vop1", cvt_u16_f16, 16)
    add("v_cvt_i16_f16", "f16", "vop1", cvt_i16_f16, 16)
    add("v_cvt_f16_u16", "int16", "vop1", lambda a, b: f2h(float(a & 0xFFFF)), 16, fclamp)
    add("v_cvt_f16_i16", "int16", "vop1", lambda a, b: f2h(float(sx(a, 16))), 16, fclamp)

    def fcmp(pred):
        def op(a, b):
            a, b = a & 0xFFFF, b & 0xFFFF
            x, y = h2f(a), h2f(b)
            unordered = math.isnan(x) or math.isnan(y)
            return pred(x, y, unordered)
        return op
    fcmps = {"f": lambda x, y, u: False, "lt": lambda x, y, u: x < y, "eq": lambda x, y, u: x == y,
             "le": lambda x, y, u: x <= y, "gt": lambda x, y, u: x > y, "lg": lambda x, y, u: not u and x != y,
             "ge": lambda x, y, u: x >= y, "o": lambda x, y, u: not u, "u": lambda x, y, u: u,
             "nge": lambda x, y, u: not x >= y, "nlg": lambda x, y, u: u or x == y, "ngt": lambda x, y, u: not x > y,
             "nle": lambda x, y, u: not x <= y, "neq": lambda x, y, u: not x == y, "nlt": lambda x, y, u: not x < y,
             "tru": lambda x, y, u: True}
    for c, p in fcmps.items():
        add(f"v_cmp_{c}_f16", "f16", "vopc", fcmp(p))
    add("v_cmp_class_f16", "f16", "vopc", lambda a, b: cmp_class_f16(a, b & 0x3FF))
    return ops


OPS = _ops()
CLAMP_OPS = tuple(sorted(o for o, v in OPS.items() if v[4]))


# ---- program text: parsing ----------------------------------------------------------------------------

_SRC = re.compile(r"^(?P<neg>-)?(?P<abs>\|)?(?:sext\((?P<sx>[^)]+)\)|(?P<tok>[^|]+))\|?$")


def parse_src(tok):
    """{'tok': register or constant text, 'neg', 'abs', 'sext'}."""
    tok = tok.strip()
    if tok in FLOAT_CONSTS or re.fullmatch(r"-?\d+", tok):
        return {"tok": tok, "neg": False, "abs": False, "sext": False}   # -16 is a constant, not neg(16)
    m = _SRC.match(tok)
    if not m:
        raise RefError(f"source {tok!r}")
    t = m.group("sx") or m.group("tok")
    return {"tok": t.strip(), "neg": bool(m.group("neg")), "abs": bool(m.group("abs")), "sext": bool(m.group("sx"))}


def parse(line):
    """(op without _sdwa, [operand texts], {modifier: value}, clamp) for one SDWA line."""
    op, _, rest = line.partition(" ")
    if not op.endswith("_sdwa"):
        raise RefError(f"not an SDWA line: {line!r}")
    mods = dict(m.split(":", 1) for m in re.findall(r"\b(?:src0_sel|src1_sel|dst_sel|dst_unused):\w+", rest))
    clamp = bool(re.search(r"\bclamp\b", rest))
    body = re.split(r"\s+(?:clamp|src0_sel|src1_sel|dst_sel|dst_unused)\b", rest, maxsplit=1)[0]
    return op[:-len("_sdwa")], [a.strip() for a in body.split(",")], mods, clamp


# ---- the reference interpreter ------------------------------------------------------------------------

class State:
    def __init__(self):
        self.v, self.s, self.vcc = {}, {}, 0


def read_raw(st, tok, f16):
    if tok.startswith("v") and tok[1:].isdigit():
        return st.v.get(int(tok[1:]), 0)
    if tok.startswith("s") and tok[1:].isdigit():
        return st.s.get(int(tok[1:]), 0)
    if tok in FLOAT_CONSTS:
        return FLOAT_CONSTS_F16[tok] if f16 else FLOAT_CONSTS[tok]
    return int(tok, 0) & M32


def select(value, sel, sext):
    off, w = SEL[sel]
    v = (value >> off) & ((1 << w) - 1)
    if sext and w < 32:
        v = sx(v, w) & M32
    return v


def place(old, res, width, sel, unused):
    off, w = SEL[sel]
    res &= M32
    if w == 32:
        return res & 0xFFFF if width == 16 else res
    field = res & ((1 << w) - 1)
    if unused == "UNUSED_PRESERVE":
        mask = ((1 << w) - 1) << off
        return (old & ~mask & M32) | (field << off)
    if unused == "UNUSED_SEXT":
        return (sx(field, w) << off) & M32
    return field << off


def _clamp(kind, r):
    if kind == "usat":
        return max(0, min(M32, r))
    if kind == "f16":
        if r & 0x8000:            # negative, -0 included
            return 0
        return min(r, 0x3C00)     # +inf and anything above 1.0 (NaN results are skipped)
    if kind == "f32":
        if r & 0x80000000:
            return 0
        return min(r, 0x3F800000)
    return r


def step(st, line):
    op, args, mods, clamp = parse(line)
    if op not in OPS:
        raise RefError(f"{op} is not modelled")
    family, form, width, fn, clamp_kind = OPS[op]
    if clamp and not clamp_kind:
        raise RefError(f"{op}: clamp not modelled")
    f16 = family == "f16"
    srcs = args[1:3] if form in ("vop2", "vopc", "cndmask") else args[1:2]
    vals = []
    for i, text in enumerate(srcs):
        s = parse_src(text)
        v = select(read_raw(st, s["tok"], f16),
                   mods.get(f"src{i}_sel", "DWORD"), s["sext"])
        if s["abs"]:
            v &= ~0x8000 if f16 else ~0x80000000
        if s["neg"]:
            v ^= 0x8000 if f16 else 0x80000000
        vals.append(v & M32)
    if form == "vopc":
        st.vcc = int(bool(fn(vals[0], vals[1])))
        return
    if form == "cndmask":
        r = vals[1] if st.vcc else vals[0]
    else:
        r = fn(vals[0], vals[1] if len(vals) > 1 else 0)
    r = _clamp(clamp_kind, r) if clamp else r
    d = int(args[0][1:])
    st.v[d] = place(st.v.get(d, 0), r, width, mods.get("dst_sel", "DWORD"), mods.get("dst_unused", "UNUSED_PAD"))


def reference(lines):
    """Run the program; returns the four exported words. Raises Skip for what the reference does not model."""
    st = State()
    for line in lines:
        op, _, rest = line.partition(" ")
        args = [a.strip() for a in rest.split(",")]
        if op == "v_mov_b32":
            st.v[int(args[0][1:])] = read_raw(st, args[1], False)
        elif op == "s_mov_b32":
            if args[0] == "vcc_lo":
                st.vcc = int(args[1], 0) & 1
            else:
                st.s[int(args[0][1:])] = int(args[1], 0) & M32
        elif op == "v_cndmask_b32_e64":
            st.v[int(args[0][1:])] = int(args[2], 0) if st.vcc else int(args[1], 0)
        else:
            step(st, line)
    return [st.v.get(r, 0) for r in (8, 9, 10, 11)]


# ---- program generation -------------------------------------------------------------------------------

def _word(rng):
    r = rng.random()
    if r < 0.35:
        return rng.choice(INTERESTING)
    if r < 0.7:
        return _half(rng) | _half(rng) << 16
    return rng.getrandbits(32)


def _half(rng):
    r = rng.random()
    if r < 0.5:
        return rng.choice(F16_INTERESTING)
    if r < 0.65:
        return rng.randrange(0, 0x400) | rng.choice((0, 0x8000))     # denormal
    h = rng.getrandbits(16)
    while is_nan16(h):
        h = rng.getrandbits(16)
    return h


def _src(rng, family, allow_const=True):
    """An operand text for a source of an op of this family."""
    r = rng.random()
    if r < 0.68 or not allow_const:
        tok = f"v{rng.choice(VREGS)}" if r < 0.55 or not allow_const else f"s{rng.choice(SREGS)}"
    elif r < 0.9:
        tok = str(rng.choice((-16, -1, 0, 1, 2, 7, 15, 16, 24, 31, 32, 63, 64, rng.randrange(-16, 65))))
    else:
        return rng.choice(tuple(FLOAT_CONSTS))
    if family == "cndmask":
        return tok                        # llvm-mc takes no sext() on v_cndmask_b32_sdwa
    if family in ("int", "int16"):
        return f"sext({tok})" if rng.random() < 0.3 else tok
    if not tok[0] in "vs":
        return tok
    if rng.random() < 0.2:
        tok = f"|{tok}|"
    if rng.random() < 0.25:
        tok = "-" + tok
    return tok


KNOWN = False   # also generate the forms in KNOWN_DIVERGENCES (--known; always on for a hardware-only run)
SHIFTS = ("v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32")
KNOWN_DIVERGENCES = {
    "shift-const-sel": "a constant shift amount (inline or seeded SGPR) with src0_sel or sext(): until #2005 the "
                       "translator shifts by the whole constant",
}


def emit(rng, name):
    family, form, width, fn, clamp_kind = OPS[name]
    sels = tuple(SEL)
    s0, s1 = rng.choice(sels), rng.choice(sels)
    if form == "vopc":
        src1_family = "int" if name == "v_cmp_class_f16" else family
        a, b = _src(rng, family), _src(rng, src1_family)
    else:
        d = f"v{rng.choice(VREGS)}"
        clamp = " clamp" if clamp_kind and rng.random() < 0.25 else ""
        dsel = rng.choice(sels)
        unused = rng.choice(UNUSED)
        a = _src(rng, "cndmask" if form == "cndmask" else family)
        b = _src(rng, "cndmask" if form == "cndmask" else family) if form != "vop1" else None
        if form == "cndmask" and a[0] == b[0] == "s" and a != b:
            b = f"v{rng.choice(VREGS)}"       # VCC is on the constant bus too: at most one other SGPR (gfx1030)
    if not KNOWN:
        inner = a[5:-1] if a.startswith("sext(") else a
        if name in SHIFTS and inner[0] != "v":
            s0, a = "DWORD", inner
    if form == "vopc":
        return f"{name}_sdwa vcc_lo, {a}, {b} src0_sel:{s0} src1_sel:{s1}"
    tail = f"{clamp} dst_sel:{dsel} dst_unused:{unused} src0_sel:{s0}"
    if form == "vop1":
        return f"{name}_sdwa {d}, {a}{tail}"
    vcc = ", vcc_lo" if form == "cndmask" else ""
    return f"{name}_sdwa {d}, {a}, {b}{vcc}{tail} src1_sel:{s1}"


def seeds(rng):
    return ([f"v_mov_b32 v{r}, 0x{_word(rng):x}" for r in VREGS] + [f"s_mov_b32 s{r}, 0x{_word(rng):x}" for r in SREGS]
            + [f"s_mov_b32 vcc_lo, {rng.randrange(2)}"])


def observe(regs):
    return [f"v_mov_b32 v8, v{regs[0]}", f"v_mov_b32 v9, v{regs[1]}", f"v_mov_b32 v10, v{regs[2]}",
            "v_cndmask_b32_e64 v11, 0, 1, vcc_lo"]


def generate(rng, n_ops, ops):
    body = [emit(rng, rng.choice(ops)) for _ in range(n_ops)]
    written = []
    for line in reversed(body):
        d = line.split()[1].rstrip(",")
        if d[0] == "v" and d[1:].isdigit() and d not in written:
            written.append(d)
    regs = [int(d[1:]) for d in written[:3]]
    regs += rng.sample([r for r in VREGS if r not in regs], 3 - len(regs))
    return seeds(rng) + body + observe(regs)


# ---- the translator side ------------------------------------------------------------------------------

def translated(lines, name="ps_sdwa"):
    """Translate and execute; returns the four exported words, or raises Skip."""
    import rdna2spirv
    from spirv_mini_interp import run_fragment
    with tempfile.TemporaryDirectory(prefix="sdwa-diff-") as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        path = os.path.join(dis, name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}  sdwa_diff case\n")
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


def _diff(lines, want, got, label):
    bad = [k for k in range(4) if want[k] != got[k]]
    if not bad:
        return None
    i = bad[0]
    what = "VCC" if i == 3 else lines[-N_OBS + i].split(", ")[1]
    return f"word {i} ({what}): reference 0x{want[i]:08x}, {label} 0x{got[i]:08x}"


def check(lines, backends=("translator",), hw=None):
    """None on a match, else a description. Raises Skip (the reference or every backend skipped) / RefError.
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
    """Drop instructions (not the seeds or the observation) while the program still mismatches."""
    body = lines[N_SEED:-N_OBS]
    changed = True
    while changed:
        changed = False
        for i in range(len(body)):
            trial = lines[:N_SEED] + body[:i] + body[i + 1:] + lines[-N_OBS:]
            try:
                if check_fn(trial):
                    body = body[:i] + body[i + 1:]
                    changed = True
                    break
            except (Skip, RefError):
                continue
    return lines[:N_SEED] + body + lines[-N_OBS:]


def vk_verdict(lines, hw, got):
    """(class, detail) for a program translated and run on a Vulkan GPU against the hardware's four words: match, nan
    (the words differ and the reference declines the program for an f16 NaN: its payload and the IEEE-mode NaN rules are
    not modelled, and the hardware kernels run with IEEE on where the translator assumes the game's MODE) or mismatch."""
    for k in range(4):
        if hw[k] != got[k]:
            try:
                reference(lines)
            except Skip as e:
                if "NaN" in str(e):
                    return "nan", str(e)
            except RefError:
                pass
            what = "VCC" if k == 3 else lines[-N_OBS + k].split(", ")[1]
            return "mismatch", f"word {k} ({what}): hardware 0x{hw[k]:08x}, translated on the GPU 0x{got[k]:08x}"
    return "match", ""


def hw_prepare(oracle, programs, wave=32):
    """Run every program on the hardware in batches. Returns per program the four words, or a str reason. wave 64 runs
    each in a wave64 wave with VCC as a 64-bit lane mask."""
    if wave == 64:
        import vk_fuzz_run
        from hw_oracle import asm
        programs = [asm.Kernel([vk_fuzz_run.wave64_text(ln) for ln in p], wave=64, threads=1, words=4)
                    for p in programs]
    return oracle.run(programs)


def run_suite(programs, seed, max_ops, ops, backends=("translator",), oracle=None, wave=32, build=None, work=None,
              vk_constants=False):
    counts = {"pass": 0, "mismatch": 0, "skip": {}, "vk": {}, "vk_failures": []}
    failures = []
    rng = random.Random(seed)
    progs = [generate(rng, rng.randrange(1, max_ops + 1), ops) for _ in range(programs)]
    hw_words = hw_prepare(oracle, progs, wave) if "hw" in backends else [None] * programs
    if "vk" in backends:
        import vk_fuzz_run
        counts["vk"], counts["vk_failures"] = vk_fuzz_run.judge(progs, hw_words, vk_verdict, wave=wave,
                                                                build=build, work_parent=work, opaque=not vk_constants,
                                                                uniform=False)
    backends = tuple(b for b in backends if b != "vk")
    for i, lines in enumerate(progs):
        try:
            problem = check(lines, backends, hw_words[i])
        except Skip as e:
            counts["skip"][str(e)] = counts["skip"].get(str(e), 0) + 1
            continue
        if problem is None:
            counts["pass"] += 1
            continue
        counts["mismatch"] += 1
        failures.append((i, problem, lines))
    return counts, failures


def self_test():
    base = [f"v_mov_b32 v{r}, 0x{0x11111111 * r:x}" for r in VREGS] + [f"s_mov_b32 s{r}, {r}" for r in SREGS]
    base.append("s_mov_b32 vcc_lo, 0")
    ok = check(base + ["v_add_nc_u32_sdwa v1, v2, v3 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE "
                       "src0_sel:BYTE_1 src1_sel:WORD_0"] + observe([1, 2, 3])) is None
    # 0x2222 + 0x33 = 0x2255 into the high word of v1 (0x11111111): 0x22551111
    placed = reference(base + ["v_add_nc_u32_sdwa v1, v2, v3 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE "
                               "src0_sel:WORD_0 src1_sel:BYTE_0"] + observe([1, 2, 3]))[0] == 0x22551111
    try:
        translated(base + ["v_add_nc_u32_dpp v1, v2, v3 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf"
                           .replace("_dpp", "_sdwa_dpp")] + observe([1, 2, 3]))
        skipped = False
    except Skip:
        skipped = True
    print(f"self-test: match {ok}, placement modelled {placed}, unsupported form skipped {skipped}")
    return ok and placed and skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description="SDWA VOP1/VOP2/VOPC differential fuzzing.")
    ap.add_argument("--programs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=4)
    ap.add_argument("--ops", default=None, help="comma-separated opcodes without _sdwa (default: the --family ones)")
    ap.add_argument("--family", default="all", choices=("int", "f16", "all"))
    ap.add_argument("--list-ops", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--all-failures", action="store_true", help="print every failing program, not just the first")
    ap.add_argument("--oracle", default="translator",
                    help="comma-separated comparison targets: translator (default), hw (natively on an RDNA1/RDNA2 "
                         "GPU; see tools/hw_oracle/). 'hw' alone compares the hardware with the reference only")
    ap.add_argument("--hw-arch", default=None, help="GPU architecture to assemble for (default: detected)")
    ap.add_argument("--wave", type=int, choices=(32, 64), default=32,
                    help="the wave size of the hardware kernels and of the vk programs (64 needs native 64-wide "
                         "subgroups; lane 0 is judged)")
    ap.add_argument("--build", default=os.path.join(os.path.dirname(HERE), "runtime", "build"),
                    help="with vk: the folder with vk_compute_test")
    ap.add_argument("--vk-constants", action="store_true",
                    help="with vk: keep the seeds as literal v_mov_b32 constants (the driver may fold them)")
    ap.add_argument("--work", default=None, help="with vk: a folder for the temporary files (VK_FUZZ_KEEP=1 keeps them)")
    ap.add_argument("--known", action="store_true",
                    help="also generate the forms where the translator is known to differ (KNOWN_DIVERGENCES; "
                         "always on with --oracle hw alone)")
    a = ap.parse_args(argv)
    if a.programs < 1 or a.max_ops < 1:
        ap.error("--programs and --max-ops must be at least 1")
    backends = tuple(dict.fromkeys(t.strip() for t in a.oracle.split(",")))
    if not backends or any(t not in ("translator", "hw", "vk") for t in backends):
        ap.error("--oracle takes translator, hw, vk (with hw) or a comma-separated mix")
    if "vk" in backends and "hw" not in backends:
        ap.error("--oracle vk judges the translated programs against the hardware's words: use --oracle hw,vk")
    global KNOWN
    KNOWN = a.known or backends == ("hw",)
    if a.list_ops:
        for name in sorted(OPS):
            family, form, width, _, clamp = OPS[name]
            print(f"{name:20} {'f16' if family == 'f16' else 'int'} {form}{' clamp' if clamp else ''}")
        return 0
    if a.self_test:
        return 0 if self_test() else 1
    if a.ops is not None:
        tokens = [t.strip() for t in a.ops.split(",")]
        if not any(tokens) or any(not t for t in tokens):
            ap.error("--ops must be a comma-separated list of opcodes")
        ops = sorted(set(tokens))
    else:
        ops = sorted(o for o, v in OPS.items()
                     if a.family == "all" or (a.family == "f16") == (v[0] == "f16"))
    unknown = [o for o in ops if o not in OPS]
    if unknown:
        ap.error("unknown ops: " + ", ".join(unknown))
    oracle = None
    if "hw" in backends:
        from hw_oracle import asm, runner
        try:
            oracle = runner.open_oracle(arch=a.hw_arch)
        except runner.Unavailable as e:
            print(f"hardware oracle unavailable: {e}")
            return 77
        if asm.family(oracle.arch) != "gfx10":
            print(f"hardware oracle unavailable: {oracle.arch} has no SDWA (GFX9 and GFX10 only)")
            oracle.close()
            return 77
        print(f"hardware oracle: {oracle.device}")
    try:
        counts, failures = run_suite(a.programs, a.seed, a.max_ops, ops, backends, oracle, a.wave, a.build, a.work,
                                     a.vk_constants)
        vk_bad = counts.pop("vk_failures")
        skips = sum(counts["skip"].values())
        print(f"{a.programs} programs (seed {a.seed}, {'+'.join(backends)}): {counts['pass']} pass, "
              f"{counts['mismatch']} mismatch, {skips} skipped")
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
            return check(lines, backends, (lambda l: hw_prepare(oracle, [l], a.wave)[0]) if oracle else None)
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
    if not counts["pass"]:   # nothing was compared: every program was skipped
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
