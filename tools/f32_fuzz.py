r"""Differential fuzzer for the translator's 32-bit float VALU (FUZZ-3; standard library only).

Seeded programs of VOP1 / VOP2 / VOP3 float instructions (with neg/abs source modifiers, clamp and omod) run
through two independent sides and are compared bit-exact:

  * F32Model below: the RDNA2 ISA definitions ("RDNA 2" Instruction Set Architecture reference guide, chapter
    6 "Vector ALU Operations" and the VOP1/VOP2/VOP3/VOPC opcode tables), with exact rational arithmetic and one
    round-to-nearest-even per operation. It shares no code with tools/rdna2spirv.py.
  * tools/rdna2spirv.py -> tests/semantic/spirv_mini_interp.py (the translated SPIR-V, executed).

What a comparison can and cannot say:
  * NaN results match any NaN: payloads are not modelled (the interpreter passes floats through Python).
  * Denormal f32 operands or results are skipped with a reason: whether the game's shaders flush f32
    denormals is the MODE register question (SHADER-35), not a property of these instructions.
  * v_rcp/v_rsq/v_sqrt/v_exp/v_log are only fed inputs whose results are exact (0, inf, NaN, negatives, powers
    of two); the hardware's 1-ULP approximations are out of scope.
  * An input for which the translated SPIR-V is undefined (GLSL.std.450 FMax of NaN, OpFDiv by zero, Log2 of
    0, ...) while the ISA defines the result is reported as its own class, "undefined in SPIR-V": the
    interpreter refuses to guess, and so does this tool.

With --oracle hw (or translator,hw) the same programs also run natively on an AMD GPU (tools/hw_oracle/, wave32) and
the hardware is judged against F32Model: match / mismatch where the model has an answer, "known deviation" where the
hardware equals an analysed alternative (omod turns -0 into +0), and informational tables where the model declines
(denormals under four MODE settings, NaN payloads, clamp of NaN, omod under IEEE/denormal modes). GFX10.1-only opcodes
(v_mad_f32, v_mac_f32, v_madak_f32, v_madmk_f32, v_mad_legacy_f32) are skipped on gfx1030/GFX11 with a reason, never substituted.
v_min/max/min3/max3/med3_f32 are hardware-only ops. Exit 77 when the hardware tools are missing. See
docs/shader_validation.md ("32-bit float VALU on the GPU").

Usage:
    python -I tools/f32_fuzz.py [--programs N] [--seed S] [--ops v_add_f32,...] [--show N]
    python -I tools/f32_fuzz.py --oracle hw [--probes N] [--negative-control | --perturb NAME] [--no-info]
    python -I tools/f32_fuzz.py --list-ops
    python -I tools/f32_fuzz.py --self-test

Exit code: 0 when every program matches, 1 when any mismatches, 2 on a harness error. Nothing here reads game
files.
"""
import argparse
import gc
import math
import os
import random
import struct
import sys
import tempfile
import warnings
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or HERE
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
if SEM_DIR not in sys.path:
    sys.path.insert(0, SEM_DIR)

import rdna2spirv  # noqa: E402
from spirv_mini_interp import run_fragment  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
QNAN = 0x7FC00000
PINF, NINF = 0x7F800000, 0xFF800000


class NotModelled(Exception):
    """An input whose hardware result this tool does not claim to know; counted as a skip with the reason."""


class Denormal(NotModelled):
    """An f32 denormal reached an operand or a result: outside what this tool judges (see the module doc)."""


# ---- f32 bit helpers (exact) ----

def is_nan(b):
    return (b >> 23) & 0xFF == 0xFF and b & 0x7FFFFF != 0


def is_inf(b):
    return b & 0x7FFFFFFF == PINF


def is_zero(b):
    return b & 0x7FFFFFFF == 0


def is_denorm(b):
    return (b >> 23) & 0xFF == 0 and b & 0x7FFFFF != 0


def sign(b):
    return b >> 31


def to_frac(b):
    """Exact value of a finite f32."""
    e, m = (b >> 23) & 0xFF, b & 0x7FFFFF
    v = Fraction(m, 1 << 23) * Fraction(2) ** -126 if e == 0 else Fraction((1 << 23) | m, 1 << 23) * Fraction(2) ** (e - 127)
    return -v if sign(b) else v


def round_f32(v, neg_zero=False):
    """Fraction -> f32 bits, round to nearest even, overflow to inf. `neg_zero`: the sign of an exact zero."""
    if v == 0:
        return 0x80000000 if neg_zero else 0
    s = 0x80000000 if v < 0 else 0
    a = -v if v < 0 else v
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if e < -126:
        q = a / (Fraction(2) ** -149)                 # denormal: units of 2^-149
        n = _rne(q)
        if n == 0:
            return s
        if n < (1 << 23):
            return s | n
        return s | (1 << 23)                           # rounded up to the smallest normal
    q = a / (Fraction(2) ** (e - 23))                  # [2^23, 2^24)
    n = _rne(q)
    if n == 1 << 24:
        n >>= 1
        e += 1
    if e > 127:
        return s | PINF
    return s | ((e + 127) << 23) | (n & 0x7FFFFF)


def _rne(q):
    n = q.numerator // q.denominator
    rem = q - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n & 1):
        n += 1
    return n


def round_rtz_f16(b):
    """f32 bits -> f16 bits, round toward zero; finite overflow gives +-65504, inf stays inf, NaN stays NaN."""
    s = (b >> 16) & 0x8000
    if is_nan(b):
        return 0x7E00 | s
    if is_inf(b):
        return 0x7C00 | s
    a = abs(to_frac(b))
    if a >= 65504:
        return s | 0x7BFF
    if a < Fraction(2) ** -14:
        return s | int(a / Fraction(2) ** -24)        # f16 denormal (truncated)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    n = int(a / Fraction(2) ** (e - 10))
    return s | ((e + 15) << 10) | (n & 0x3FF)


def round_rne_f16(b):
    s = (b >> 16) & 0x8000
    if is_nan(b):
        return 0x7E00 | s
    if is_inf(b):
        return 0x7C00 | s
    a = abs(to_frac(b))
    if a == 0:
        return s
    if a < Fraction(2) ** -14:
        n = _rne(a / Fraction(2) ** -24)
        return s | n                                   # n == 0x400 is the smallest normal, same encoding
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    n = _rne(a / Fraction(2) ** (e - 10))
    if n == 1 << 11:
        n >>= 1
        e += 1
    if e > 15:
        return s | 0x7C00
    return s | ((e + 15) << 10) | (n & 0x3FF)


def f16_to_f32(h):
    s = (h & 0x8000) << 16
    e, m = (h >> 10) & 0x1F, h & 0x3FF
    if e == 0x1F:
        return s | PINF | (m << 13)
    if e == 0:
        return round_f32(Fraction(m, 1 << 24) * (-1 if s else 1), neg_zero=bool(s))
    return s | ((e - 15 + 127) << 23) | (m << 13)


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def neg(b):
    return b ^ 0x80000000


def fabs(b):
    return b & 0x7FFFFFFF


# ---- the ISA model ----

def add(x, y):
    if is_nan(x) or is_nan(y):
        return QNAN
    if is_inf(x) or is_inf(y):
        if is_inf(x) and is_inf(y) and sign(x) != sign(y):
            return QNAN
        return x if is_inf(x) else y
    v = to_frac(x) + to_frac(y)
    return round_f32(v, neg_zero=v == 0 and sign(x) and sign(y))


def mul(x, y):
    if is_nan(x) or is_nan(y):
        return QNAN
    s = sign(x) ^ sign(y)
    if is_inf(x) or is_inf(y):
        if is_zero(x) or is_zero(y):
            return QNAN
        return (s << 31) | PINF
    return round_f32(to_frac(x) * to_frac(y), neg_zero=bool(s))


def fma(x, y, z):
    if is_nan(x) or is_nan(y) or is_nan(z):
        return QNAN
    ps = sign(x) ^ sign(y)
    if is_inf(x) or is_inf(y):
        if is_zero(x) or is_zero(y):
            return QNAN
        if is_inf(z) and sign(z) != ps:
            return QNAN
        return (ps << 31) | PINF
    if is_inf(z):
        return z
    p = to_frac(x) * to_frac(y)
    v = p + to_frac(z)
    if v == 0:
        pz = p == 0
        return round_f32(v, neg_zero=bool(pz and ps and sign(z)))
    return round_f32(v)


def legacy_mul(x, y):
    """DX9 rules: +-0 times anything, inf and NaN included, is +0 (LLVM AMDGPUInstCombineIntrinsic, fmul_legacy)."""
    if is_zero(x) or is_zero(y):
        return 0
    return mul(x, y)


def cvt_int_to_f32(v):
    return round_f32(Fraction(v))


def rounding(x, how):
    if is_nan(x):
        return QNAN
    if is_inf(x) or is_zero(x):
        return x
    v = to_frac(x)
    r = {"floor": math.floor, "ceil": math.ceil, "trunc": math.trunc}.get(how)
    n = r(v) if r else _rne_signed(v)
    return round_f32(Fraction(n), neg_zero=bool(sign(x)))


def _rne_signed(v):
    return _rne(v) if v >= 0 else -_rne(-v)


def cvt_pk_u8(x, byte, base):
    """flt32_to_uint8: round to nearest even, clamp to [0, 255], NaN -> 0."""
    if is_nan(x):
        u = 0
    elif sign(x) or is_zero(x):
        u = 0
    elif is_inf(x):
        u = 255
    else:
        u = min(255, _rne(to_frac(x)))
    sh = 8 * (byte & 3)
    return (base & ~(0xFF << sh) & M32) | (u << sh)


def cube(op, x, y, z):
    """V_CUBEID/SC/TC/MA_F32 (ISA pseudocode; comparisons are float compares, so -0 < 0 is false)."""
    if is_nan(x) or is_nan(y) or is_nan(z):
        raise NotModelled("cube of NaN")
    ax, ay, az = (abs(to_frac(v)) if not is_inf(v) else math.inf for v in (x, y, z))
    lt0 = lambda v: sign(v) and not is_zero(v)        # noqa: E731
    if az >= ax and az >= ay:
        major = "z"
    elif ay >= ax:
        major = "y"
    else:
        major = "x"
    if op == "v_cubeid_f32":
        face = {"z": 5 if lt0(z) else 4, "y": 3 if lt0(y) else 2, "x": 1 if lt0(x) else 0}[major]
        return bits(float(face))
    if op == "v_cubesc_f32":
        return {"z": neg(x) if lt0(z) else x, "y": x, "x": z if lt0(x) else neg(z)}[major]
    if op == "v_cubetc_f32":
        return {"z": neg(y), "y": neg(z) if lt0(y) else z, "x": neg(y)}[major]
    m = {"z": z, "y": y, "x": x}[major]
    return 0 if is_zero(m) else mul(bits(2.0), m)


def cmp_class(x, mask):
    if is_nan(x):
        c = 1 if x & 0x400000 else 0
    elif is_inf(x):
        c = 2 if sign(x) else 9
    elif is_zero(x):
        c = 5 if sign(x) else 6
    elif is_denorm(x):
        c = 4 if sign(x) else 7
    else:
        c = 3 if sign(x) else 8
    return (mask >> c) & 1


def transcendental(op, x):
    """Exact-result inputs only (see gen): special values and powers of two."""
    if is_nan(x):
        return QNAN
    if op == "v_rcp_f32":
        if is_zero(x):
            return (x & 0x80000000) | PINF
        if is_inf(x):
            return x & 0x80000000
        return round_f32(1 / to_frac(x))
    if op == "v_sqrt_f32":
        if is_zero(x):
            return x
        if sign(x):
            return QNAN
        if is_inf(x):
            return x
        return _exact_pow2_op(x, lambda e: Fraction(2) ** (e // 2))
    if op == "v_rsq_f32":
        if is_zero(x):
            return (x & 0x80000000) | PINF
        if sign(x):
            return QNAN
        if is_inf(x):
            return 0
        return _exact_pow2_op(x, lambda e: Fraction(2) ** (-(e // 2)))
    if op == "v_exp_f32":
        if is_inf(x):
            return 0 if sign(x) else PINF
        v = to_frac(x)
        if v.denominator != 1:
            raise ValueError("v_exp_f32 of a non-integer")
        n = int(v)
        if n >= 128:
            return PINF
        if n < -126:
            raise Denormal("v_exp_f32 result")
        return round_f32(Fraction(2) ** n)
    if op == "v_log_f32":
        if is_zero(x):
            return NINF
        if sign(x):
            return QNAN
        if is_inf(x):
            return PINF
        return _exact_pow2_op(x, lambda e: Fraction(e))
    raise ValueError(op)


def _exact_pow2_op(x, f):
    v = to_frac(x)
    e = v.numerator.bit_length() - 1 if v.denominator == 1 else -(v.denominator.bit_length() - 1)
    if Fraction(2) ** e != v:
        raise ValueError("not a power of two")
    return round_f32(f(e))


OMOD = {"mul:2": 2, "mul:4": 4, "div:2": Fraction(1, 2)}


def apply_out_mods(r, mods):
    """VOP3 output modifiers: omod scales the rounded result, then clamp to [0, 1] (NaN -> 0)."""
    for m in mods:
        if m in OMOD:
            if not is_nan(r):
                r = mul(r, round_f32(Fraction(OMOD[m])))
    if "clamp" in mods:
        if is_nan(r) or r == 0x80000000:
            raise NotModelled("clamp of NaN or -0 (@shelbykb2's claim)")
        if sign(r):
            r = 0
        elif not is_zero(r) and (is_inf(r) or to_frac(r) > 1):
            r = bits(1.0)
    return r


# ---- op table: name -> (sources, kind, model) ----
# kinds: "f" float source (takes neg/abs), "u" integer source, "k" literal float constant (fmaak/fmamk/madak/madmk).

def _op(srcs, fn, mods=True, sel="any", hw_only=False):
    return {"srcs": srcs, "fn": fn, "mods": mods, "sel": sel, "hw_only": hw_only}


def _fval(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def fmin(x, y):
    """RDNA2 v_min_f32 (docs/design.md: measured on gfx1030): one NaN gives the other operand, both NaN give
    src0, -0 < +0."""
    if is_nan(x):
        return x if is_nan(y) else y
    if is_nan(y):
        return x
    if is_zero(x) and is_zero(y):
        return x if sign(x) else y
    return x if _fval(x) < _fval(y) else y


def fmax(x, y):
    if is_nan(x):
        return x if is_nan(y) else y
    if is_nan(y):
        return x
    if is_zero(x) and is_zero(y):
        return y if sign(x) else x
    return x if _fval(x) > _fval(y) else y


def fmed3(x, y, z):
    """With any NaN operand: min3(s0, s1, s2); otherwise the median, -0 < +0."""
    if is_nan(x) or is_nan(y) or is_nan(z):
        return fmin(fmin(x, y), z)
    return sorted((x, y, z), key=lambda b: (_fval(b), 0 if sign(b) else 1))[1]


ALL_OPS = {
    "v_add_f32": _op("ff", lambda x, y: add(x, y)),
    "v_sub_f32": _op("ff", lambda x, y: add(x, neg(y))),
    "v_subrev_f32": _op("ff", lambda x, y: add(y, neg(x))),
    "v_mul_f32": _op("ff", mul),
    "v_mul_legacy_f32": _op("ff", legacy_mul),
    "v_mad_legacy_f32": _op("fff", lambda x, y, z: add(legacy_mul(x, y), z)),
    "v_mad_f32": _op("fff", lambda x, y, z: add(mul(x, y), z)),
    "v_fma_f32": _op("fff", fma),
    "v_fmac_f32": _op("ffd", fma),                       # D = S0 * S1 + D
    "v_mac_f32": _op("ffd", lambda x, y, z: add(mul(x, y), z)),
    "v_fmaak_f32": _op("ffk", fma, mods=False),
    "v_fmamk_f32": _op("fkf", fma, mods=False),
    "v_madak_f32": _op("ffk", lambda x, y, z: add(mul(x, y), z), mods=False),
    "v_madmk_f32": _op("fkf", lambda x, y, z: add(mul(x, y), z), mods=False),
    "v_cvt_f32_i32": _op("u", lambda x: cvt_int_to_f32(x - (1 << 32) if x >> 31 else x)),
    "v_cvt_f32_u32": _op("u", lambda x: cvt_int_to_f32(x)),
    "v_cvt_f32_ubyte0": _op("u", lambda x: cvt_int_to_f32(x & 0xFF)),
    "v_cvt_f32_ubyte1": _op("u", lambda x: cvt_int_to_f32((x >> 8) & 0xFF)),
    "v_cvt_f32_ubyte2": _op("u", lambda x: cvt_int_to_f32((x >> 16) & 0xFF)),
    "v_cvt_f32_ubyte3": _op("u", lambda x: cvt_int_to_f32(x >> 24)),
    "v_cvt_f32_f16": _op("h", lambda x: f16_to_f32(x & 0xFFFF)),
    "v_cvt_f16_f32": _op("f", None, mods=False),         # partial write, handled in step()
    "v_cvt_pkrtz_f16_f32": _op("ff", lambda x, y: round_rtz_f16(x) | (round_rtz_f16(y) << 16), mods=False),
    "v_cvt_pk_u8_f32": _op("fuu", cvt_pk_u8, mods=False),
    "v_floor_f32": _op("f", lambda x: rounding(x, "floor")),
    "v_ceil_f32": _op("f", lambda x: rounding(x, "ceil")),
    "v_trunc_f32": _op("f", lambda x: rounding(x, "trunc")),
    "v_rndne_f32": _op("f", lambda x: rounding(x, "rne")),
    "v_rcp_f32": _op("t", lambda x: transcendental("v_rcp_f32", x)),
    # v_rcp_iflag_f32 is v_rcp_f32 without the FP exception flags: bit-identical on gfx1030 (design.md, "Denormal
    # flush and special results"), so it shares the model; clamp and omod are what is new here (SHADER-136).
    "v_rcp_iflag_f32": _op("t", lambda x: transcendental("v_rcp_f32", x)),
    "v_rsq_f32": _op("t", lambda x: transcendental("v_rsq_f32", x)),
    "v_sqrt_f32": _op("t", lambda x: transcendental("v_sqrt_f32", x)),
    "v_exp_f32": _op("e", lambda x: transcendental("v_exp_f32", x)),
    "v_log_f32": _op("t", lambda x: transcendental("v_log_f32", x)),
    "v_cubeid_f32": _op("fff", lambda x, y, z: cube("v_cubeid_f32", x, y, z)),
    "v_cubesc_f32": _op("fff", lambda x, y, z: cube("v_cubesc_f32", x, y, z)),
    "v_cubetc_f32": _op("fff", lambda x, y, z: cube("v_cubetc_f32", x, y, z)),
    "v_cubema_f32": _op("fff", lambda x, y, z: cube("v_cubema_f32", x, y, z)),
    "v_cmp_class_f32": _op("fu", None, mods=False),      # writes VCC, handled in step()
    # Hardware-only: the RDNA2 NaN / signed-zero rules measured by #634 (docs/design.md), which the translator
    # emits as NMin / NMax. They are in the table so --oracle hw can say whether gfx1150 follows the same rules; the
    # translator comparison leaves them out (the SPIR-V interpreter has no defined result for every case).
    "v_min_f32": _op("ff", fmin, hw_only=True),
    "v_max_f32": _op("ff", fmax, hw_only=True),
    "v_min3_f32": _op("fff", lambda x, y, z: fmin(fmin(x, y), z), hw_only=True),
    "v_max3_f32": _op("fff", lambda x, y, z: fmax(fmax(x, y), z), hw_only=True),
    "v_med3_f32": _op("fff", fmed3, hw_only=True),
}
VOP3_ONLY = {"v_mad_legacy_f32", "v_mad_f32", "v_fma_f32", "v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32",
             "v_cubema_f32", "v_cvt_pk_u8_f32", "v_cvt_pkrtz_f16_f32", "v_min3_f32", "v_max3_f32", "v_med3_f32"}
OPS = {k: v for k, v in ALL_OPS.items() if not v["hw_only"]}   # what the translator comparison covers
HW_ONLY_OPS = tuple(k for k, v in ALL_OPS.items() if v["hw_only"])
LITERAL_CONST_OPS = {"v_fmaak_f32", "v_fmamk_f32", "v_madak_f32", "v_madmk_f32"}

# Ops the hardware gives an output modifier in their VOP3 form but whose float output is an integer or packed
# value: no omod/clamp generated for them.
NO_OUT_MODS = {"v_cvt_f16_f32", "v_cvt_pkrtz_f16_f32", "v_cvt_pk_u8_f32", "v_cmp_class_f32"} | LITERAL_CONST_OPS

# ---- operand values ----

SPECIAL = [0, 0x80000000, bits(1.0), bits(-1.0), bits(0.5), bits(-0.5), bits(2.0), bits(-2.0), PINF, NINF,
           QNAN, 0xFFC00000, 0x7F800001, bits(1.5), bits(2.5), bits(-2.5), bits(0.49999997), bits(254.5),
           bits(255.5), bits(256.0), bits(-1e30), bits(1e30), 0x7F7FFFFF, 0xFF7FFFFF, 0x00800000, 0x80800000,
           bits(65504.0), bits(65519.996), bits(65520.0), bits(3.0), bits(1.0000001), bits(0.75), bits(-0.75),
           bits(16777216.0), bits(8388608.5), bits(1e-30)]
INTS = [0, 1, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 0x01000001, 0x00FFFFFF, 0xFFFFFF7F, 255, 0x12345678,
        16777217, 0xFFFFFFFE, 0x40000001]
TRANS = [0, 0x80000000, PINF, NINF, QNAN, bits(1.0), bits(4.0), bits(0.25), bits(16.0), bits(2.0 ** -20),
         bits(2.0 ** 40), bits(-4.0), bits(-1.0), bits(-(2.0 ** -14)), bits(2.0 ** 126), bits(2.0 ** -126)]
EXPS = [NINF, PINF, QNAN, 0, 0x80000000, bits(1.0), bits(3.0), bits(-3.0), bits(127.0), bits(128.0),
        bits(-126.0), bits(-20.0), bits(10.0)]


def rand_float(rng):
    r = rng.random()
    if r < 0.4:
        return rng.choice(SPECIAL)
    if r < 0.7:                      # a normal number with a modest exponent: arithmetic stays normal
        e = rng.randrange(127 - 20, 127 + 21)
        return (rng.getrandbits(1) << 31) | (e << 23) | rng.getrandbits(23)
    if r < 0.85:                     # small integers and halves: exact cases, ties
        return bits(rng.randrange(-300, 301) / rng.choice((1, 2, 4)))
    e = rng.randrange(1, 255)
    return (rng.getrandbits(1) << 31) | (e << 23) | rng.getrandbits(23)


# ---- programs ----
# A program: seeds (v0..v7 = values, by v_mov_b32 of a literal) and one instruction writing v8 (v8 is seeded
# too, so accumulating ops read it). Observed: v8, the first two sources' registers, and v9 (VCC as 0/1).

SEED_REGS = tuple(range(0, 9))


def fmt_src(reg, m):
    s = f"v{reg}"
    if m.get("abs"):
        s = f"|{s}|"
    if m.get("neg"):
        s = "-" + s
    return s


def gen(rng, op):
    spec = ALL_OPS[op]
    vals = [rand_float(rng) for _ in SEED_REGS]
    srcs, mods_src, text_srcs = [], [], []
    e64 = op in VOP3_ONLY or (spec["mods"] and rng.random() < 0.5)
    k_lit = None
    regs = list(range(8))
    rng.shuffle(regs)
    for i, kind in enumerate(spec["srcs"]):
        if kind == "d":
            continue
        r = regs[i]
        if kind == "u":
            vals[r] = rng.choice(INTS) if rng.random() < 0.6 else rng.getrandbits(32)
            if op == "v_cvt_pk_u8_f32" and i == 1:
                vals[r] = rng.randrange(0, 8)
        elif kind == "h":
            vals[r] = rng.getrandbits(32) if rng.random() < 0.5 else rng.choice(
                (0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E00, 0x0001, 0x8001, 0x03FF, 0x7BFF, 0x0400, 0x8000))
        elif kind == "t":
            vals[r] = rng.choice(TRANS)
        elif kind == "e":
            vals[r] = rng.choice(EXPS)
        elif kind == "k":
            k_lit = rand_float(rng)
            srcs.append(("k", k_lit))
            continue
        m = {}
        if e64 and kind in "fte" and spec["mods"]:
            m = {"neg": rng.random() < 0.25, "abs": rng.random() < 0.2}
        srcs.append((r, m))
        text_srcs.append(fmt_src(r, m))
    out_mods = []
    if e64 and spec["mods"] and op not in NO_OUT_MODS:
        if rng.random() < 0.3:
            out_mods.append(rng.choice(tuple(OMOD)))
        if rng.random() < 0.3:
            out_mods.append("clamp")
    return vals, srcs, e64, out_mods


def source_value(vals, s):
    r, m = s
    v = vals[r]
    if m.get("abs"):
        v = fabs(v)
    if m.get("neg"):
        v = neg(v)
    return v


def flush(b):
    """A denormal read or written as a zero of its sign (what a flush-to-zero f32 unit does)."""
    return b & 0x80000000 if is_denorm(b) else b


# model(..., denorm=): what happens to f32 denormals.
#   "skip"    raise Denormal (the default: this tool does not claim to know)
#   "keep"    exact IEEE behaviour, denormals read and written as they are
#   "ftz"     inputs read as signed zero and denormal results written as signed zero
#   "ftz_in", "ftz_out"   only one of the two
DENORM_MODES = ("skip", "keep", "ftz", "ftz_in", "ftz_out")


def model(op, vals, srcs, out_mods, denorm="skip", translator=True):
    """Words of v8 (result) and v9 (VCC as 0/1). `translator`: refuse inputs the SPIR-V interpreter cannot carry
    (they are fine on a GPU)."""
    spec = ALL_OPS[op]
    ins = []
    for s in srcs:
        ins.append(s[1] if s[0] == "k" else source_value(vals, s))
    if "d" in spec["srcs"]:
        ins.append(vals[8])
    kinds = spec["srcs"].replace("d", "f")
    fpos = [i for i, k in enumerate(kinds) if k in "fktde"]
    if denorm == "skip":
        if any(is_denorm(ins[i]) for i in fpos):
            raise Denormal("operand")
    elif denorm in ("ftz", "ftz_in"):
        for i in fpos:
            ins[i] = flush(ins[i])
    out_flush = denorm in ("ftz", "ftz_out")
    vcc = 0
    if op == "v_cvt_f16_f32":
        r = (vals[8] & 0xFFFF0000) | round_rne_f16(ins[0])
    elif op == "v_cmp_class_f32":
        if translator and is_nan(ins[0]) and not ins[0] & 0x400000:
            # The translator reads the class from the bits after an OpBitcast to float and back, exact on a
            # GPU; the interpreter holds floats as Python floats, which quiets a signalling NaN on the way.
            raise NotModelled("signalling NaN through the interpreter's float bitcast")
        r, vcc = vals[8], cmp_class(ins[0], ins[1])
    else:
        r = spec["fn"](*ins)
        floaty = spec["mods"] or op in LITERAL_CONST_OPS
        if floaty and is_denorm(r):
            if denorm == "skip":
                raise Denormal("result")
            if out_flush:
                r = flush(r)
        if op in ("v_mad_f32", "v_mac_f32", "v_madak_f32", "v_madmk_f32", "v_mad_legacy_f32"):
            prod = legacy_mul(ins[0], ins[1]) if op == "v_mad_legacy_f32" else mul(ins[0], ins[1])
            if is_denorm(prod) and denorm == "skip":
                raise Denormal("intermediate product")
        r = apply_out_mods(r, out_mods)
        if is_denorm(r):
            if denorm == "skip":
                raise Denormal("result")
            if out_flush:
                r = flush(r)
    return [r & M32, vcc]


def line(op, vals, srcs, e64, out_mods):
    spec = ALL_OPS[op]
    parts = []
    for s in srcs:
        parts.append(f"0x{s[1]:x}" if s[0] == "k" else fmt_src(*s))
    name = op
    if op == "v_cmp_class_f32":
        name += "_e64" if e64 else "_e32"
        return f"{name} vcc_lo, " + ", ".join(parts) if e64 else f"{name} vcc_lo, " + ", ".join(parts)
    if op not in VOP3_ONLY and op not in LITERAL_CONST_OPS:
        name += "_e64" if e64 else "_e32"
    if not e64 and op not in LITERAL_CONST_OPS:
        # VOP1/VOP2: src1 must be a VGPR, already true; no modifiers
        pass
    del spec
    return f"{name} v8, " + ", ".join(parts) + ("" if not out_mods else " " + " ".join(out_mods))


def listing(op, vals, srcs, e64, out_mods):
    out = ["; f32_fuzz case  code 0x0..0x1000"]
    for r in SEED_REGS:
        out.append(f"\tv_mov_b32_e32 v{r}, 0x{vals[r]:x}")
    out.append("\ts_mov_b32 vcc_lo, 0")
    out.append("\t" + line(op, vals, srcs, e64, out_mods))
    out.append("\tv_cndmask_b32_e64 v9, 0, 1, vcc_lo")
    obs = [s[0] for s in srcs if s[0] != "k"][:2] + [0, 1]
    out.append(f"\texp mrt0 v8, v9, v{obs[0]}, v{obs[1]} done vm")
    out.append("\ts_endpgm")
    return "\n".join(out) + "\n", obs[:2]


def run_translated(text, name="ps_f32_fuzz"):
    with tempfile.TemporaryDirectory(prefix="f32-fuzz-") as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        path = os.path.join(dis, name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(path)
                gc.collect()
        except rdna2spirv.Unsupported as e:
            return "skip", f"translator unsupported: {e}"
    try:
        out = run_fragment(spv)
    except Exception as e:  # noqa: BLE001 - the interpreter is half of the comparison
        msg = str(e)
        if "undefined" in msg:
            return "undefined", msg[:90]
        return "skip", f"interpreter: {type(e).__name__}: {msg[:90]}"
    if list(out) != ["%out_mrt0"]:
        return "skip", f"outputs {sorted(out)}"
    return "ok", [w & M32 for w in out["%out_mrt0"]]


def same(want, got):
    if is_nan(want) and is_nan(got):
        return True
    return want == got


def check(op, vals, srcs, e64, out_mods):
    try:
        want = model(op, vals, srcs, out_mods)
    except Denormal as e:
        return "skip", f"denormal {e}"
    except NotModelled as e:
        return "skip", str(e)
    text, obs = listing(op, vals, srcs, e64, out_mods)
    status, got = run_translated(text)
    if status != "ok":
        return status, got
    want_words = [want[0], want[1], vals[obs[0]], vals[obs[1]]]
    if all(same(w, g) or (k >= 2 and same(w | 0x400000 if is_nan(w) else w, g)) for k, (w, g) in
           enumerate(zip(want_words, got))):
        return "match", None
    return "mismatch", (want_words, got, text)


def fdesc(b):
    if is_nan(b):
        return f"0x{b:08x}(NaN)"
    try:
        return f"0x{b:08x}({struct.unpack('<f', struct.pack('<I', b))[0]!r})"
    except Exception:  # noqa: BLE001
        return f"0x{b:08x}"


def run_suite(programs, seed, ops, show=2, quiet=False):
    return translate_cases(gen_cases(programs, seed, ops), seed, show, quiet)


def translate_cases(cases, seed, show=2, quiet=False):
    """Judge each case against the translated SPIR-V. Cases of hardware-only ops are not sent to the translator."""
    counts = {"match": 0, "mismatch": 0, "undefined": 0, "skip": {}}
    by_op, undef_by_op, shown = {}, {}, {}
    programs = 0
    for case in cases:
        if case.op in HW_ONLY_OPS or case.group != "random":
            continue
        programs += 1
        i = case.index
        op, vals, srcs, e64, out_mods = case.args()
        status, detail = check(op, vals, srcs, e64, out_mods)
        if status == "skip":
            counts["skip"][detail] = counts["skip"].get(detail, 0) + 1
            continue
        counts[status] += 1
        if status == "undefined":
            key = (op, detail)
            undef_by_op[key] = undef_by_op.get(key, 0) + 1
            continue
        if status == "mismatch":
            by_op[op] = by_op.get(op, 0) + 1
            if shown.get(op, 0) < show and not quiet:
                shown[op] = shown.get(op, 0) + 1
                want, got, text = detail
                src_line = [ln for ln in text.splitlines() if ln.startswith("\t" + op)][0].strip()
                print(f"--- program {i}: {src_line}")
                used = sorted({s[0] for s in srcs if s[0] != "k"} | ({8} if "d" in ALL_OPS[op]["srcs"] or op in (
                    "v_cvt_f16_f32",) else set()))
                print("    inputs: " + " ".join(f"v{r}={fdesc(vals[r])}" for r in used))
                print("    model v8/vcc:      " + " ".join(fdesc(w) for w in want[:2]))
                print("    translated v8/vcc: " + " ".join(fdesc(w) for w in got[:2]))
    if not quiet:
        print(f"{programs} programs (seed {seed}): {counts['match']} match, {counts['mismatch']} mismatch, "
              f"{counts['undefined']} undefined in SPIR-V, {sum(counts['skip'].values())} skipped")
        for reason, n in sorted(counts["skip"].items(), key=lambda t: -t[1]):
            print(f"  skip x{n}: {reason}")
        for (op, reason), n in sorted(undef_by_op.items(), key=lambda t: -t[1]):
            print(f"  undefined x{n}: {op}: {reason}")
        for op, n in sorted(by_op.items(), key=lambda t: -t[1]):
            print(f"  mismatch x{n}: {op}")
    return counts, by_op, undef_by_op


def translator_verdict(case, want, got):
    """For a hardware mismatch: what the translated SPIR-V does with the same program."""
    text, _ = listing(*case.args())
    status, tr = run_translated(text)
    if status != "ok":
        return f"{status}: {tr}"
    mine = tr[:2]
    if hw_same(mine, got, case.op):
        return "matches the hardware (the model is the one that is off)"
    if hw_same(mine, want, case.op):
        return "matches the model, so it shares the model's answer (" + " ".join(fdesc(w) for w in mine) + ")"
    return "matches neither: " + " ".join(fdesc(w) for w in mine)


# ---- the hardware oracle (--oracle hw) ----
# The same programs, natively on the AMD GPU (tools/hw_oracle/), compared against F32Model. One wave32 thread per
# program; the kernel seeds v0..v8, runs the instruction, and stores v8 and VCC (as 0/1) through the output
# pointer. Where the model declines (denormals, NaN under clamp, ...) the program still runs, and what the
# hardware did is recorded, not judged.

# Float MODE a kernel starts with. "flush" is the state inferred for the game's pixel shaders (docs/design.md: the
# corpus writes only MODE.FP_ROUND and carries omod on 102,365 float instructions, so IEEE is off; f32 denormals
# flushed). The other three change one thing each, to answer what the hardware does there.
HW_MODES = {
    "flush": {"ieee": 0, "dx10_clamp": 1, "denorm32": 0},     # MODE 0x1c0
    "keep": {"ieee": 0, "dx10_clamp": 1, "denorm32": 3},      # 0x1f0: f32 denormals kept
    "no_dx10": {"ieee": 0, "dx10_clamp": 0, "denorm32": 0},   # 0x0c0: DX10_CLAMP off
    "ieee": {"ieee": 1, "dx10_clamp": 1, "denorm32": 0},      # 0x3c0: the assembler default for a compute kernel
}
HW_PRIMARY = "flush"


# Ops whose omod does not depend on the f32 denormal mode: the transcendental unit and the cube ops flush or select
# whatever MODE says, and gfx1150 and gfx1031 (RDNA2) apply their omod with denormals kept (measured, 'omod' table).
OMOD_IGNORES_DENORM_MODE = ("v_rcp_f32", "v_rcp_iflag_f32", "v_rsq_f32", "v_sqrt_f32", "v_exp_f32", "v_log_f32",
                            "v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32", "v_cubema_f32")


def omod_active(mode, op=None):
    """Whether mul:2 / mul:4 / div:2 act in this mode (measured on gfx1150 and gfx1031): never with MODE.IEEE = 1
    (LLVM's SIFoldOperands.cpp: "omod is ignored by hardware if IEEE bit is enabled" and, for f32, if output denormals
    are enabled), and for the ALU ops only with f32 denormals flushed."""
    m = HW_MODES[mode]
    return not m["ieee"] and (op in OMOD_IGNORES_DENORM_MODE or not m["denorm32"])


def strip_omod(case):
    return Case(case.op, case.vals, case.srcs, case.e64, [m for m in case.out_mods if m not in OMOD], case.group,
                case.index)
HW_STORE = {"gfx11": "global_store_b64", "gfx10": "global_store_dwordx2"}
HW_ZERO_REG = 20          # v20 = 0, the store's address register (v0..v8 are the program's)


def mode_word(m):
    """The MODE register value a kernel descriptor with these fields yields (FP_ROUND 3:0, FP_DENORM 7:4, DX10_CLAMP
    bit 8, IEEE bit 9); unset fields are the assembler defaults: round to nearest, f16/f64 denormals kept."""
    return (m.get("round32", 0) | m.get("round16_64", 0) << 2 | m.get("denorm32", 0) << 4 |
            m.get("denorm16_64", 3) << 6 | m.get("dx10_clamp", 1) << 8 | m.get("ieee", 1) << 9)


class Case:
    """One generated program: the op, seed values, sources, encoding, output modifiers and where it came from."""
    __slots__ = ("op", "vals", "srcs", "e64", "out_mods", "group", "index")

    def __init__(self, op, vals, srcs, e64, out_mods, group="random", index=0):
        self.op, self.vals, self.srcs, self.e64, self.out_mods = op, vals, srcs, e64, out_mods
        self.group, self.index = group, index

    def args(self):
        return self.op, self.vals, self.srcs, self.e64, self.out_mods


def hw_lines(case, arch="gfx1150", guard=True):
    """The native kernel body: seeds, the instruction, VCC as 0/1 in v9, store v8 and v9. `guard` puts s_nop 7
    after the instruction (the transcendental unit and SGPR/VCC writes have wait-state rules the model lacks)."""
    from hw_oracle import asm
    op, vals, srcs, e64, out_mods = case.args()
    out = [f"v_mov_b32 v{r}, 0x{vals[r]:x}" for r in SEED_REGS]
    out.append("s_mov_b32 vcc_lo, 0")
    # GFX11's assembler takes `clamp` before the omod (GFX10's takes either order): the same bits either way.
    out.append(line(op, vals, srcs, e64, sorted(out_mods, key=lambda m: m != "clamp")))
    if guard:
        out.append("s_nop 7")
    out += ["v_cndmask_b32_e64 v9, 0, 1, vcc_lo", f"v_mov_b32 v{HW_ZERO_REG}, 0",
            f"{HW_STORE[asm.family(arch)]} v{HW_ZERO_REG}, v[8:9], s[2:3]", "s_endpgm"]
    return out


def hw_run(oracle, cases, mode=HW_PRIMARY, guard=True):
    """[v8, vcc] per case, or the assembler's complaint as a str."""
    from hw_oracle import asm
    kernels = [asm.Kernel(hw_lines(c, oracle.arch, guard), 32, 1, 2, raw=True, mode=HW_MODES[mode]) for c in cases]
    return oracle.run(kernels, words=2)


def hw_probe_modes(oracle):
    """Read MODE back inside a kernel for every mode and compare with what the directives should give, so a
    runtime that ignored them fails here instead of producing plausible-looking data. Returns {mode: word}."""
    from hw_oracle import asm
    body = ["s_getreg_b32 s4, hwreg(HW_REG_MODE, 0, 32)", "v_mov_b32 v1, s4", "v_mov_b32 v20, 0",
            "global_store_b32 v20, v1, s[2:3]" if asm.family(oracle.arch) == "gfx11" else
            "global_store_dword v20, v1, s[2:3]", "s_endpgm"]
    names = list(HW_MODES)
    got = oracle.run([asm.Kernel(body, 32, 1, 1, raw=True, mode=HW_MODES[n]) for n in names], words=1)
    out = {}
    for n, w in zip(names, got):
        if isinstance(w, str) or w[0] != mode_word(HW_MODES[n]):
            raise RuntimeError(f"MODE probe: mode {n} reads back {w!r}, expected 0x{mode_word(HW_MODES[n]):x}; the "
                               "kernel descriptor's float mode was not applied")
        out[n] = w[0]
    return out


F16_HALVES = {"v_cvt_f16_f32": (0,), "v_cvt_pkrtz_f16_f32": (0, 1)}      # f16 results in v8: which halves


def is_nan16(h):
    return h & 0x7C00 == 0x7C00 and h & 0x3FF != 0


def hw_same(want, got, op=None):
    """Compare hardware v8 and VCC, requiring byte equality for packed u8 results."""
    if want[1] != got[1]:
        return False
    if op == "v_cvt_pk_u8_f32":
        return want[0] == got[0]
    halves = F16_HALVES.get(op)
    if halves is None:
        return same(want[0], got[0])
    for h in (0, 1):
        w, g = (want[0] >> 16 * h) & 0xFFFF, (got[0] >> 16 * h) & 0xFFFF
        if not (w == g or (h in halves and is_nan16(w) and is_nan16(g))):
            return False
    return True


# ---- known deviations: hardware behaviour measured on gfx1150 that the ISA-text model does not have ----
# Each maps (case, want) to the words the hardware gives instead, or None when it does not apply. A hardware result
# that equals the deviation's words is filed under "known deviation", not "match"; anything else stays a mismatch.

def dev_omod_neg_zero(case, want, mode=None):
    """VOP3 omod (mul:2, mul:4, div:2) turns a -0 result into +0, on gfx1150 and on RDNA2 (gfx1031, gfx1036). LLVM
    notes that omod mishandles signed zeros (SIFoldOperands.cpp) and D30209 describes -0 being forced to +0 before the
    modifier. Only where the omod was active: not with MODE.IEEE = 1, not for ALU ops with f32 denormals kept
    (omod_active)."""
    if mode is None:
        mode = HW_PRIMARY
    if want[0] == 0x80000000 and any(m in OMOD for m in case.out_mods) and omod_active(mode, case.op):
        return [0, want[1]]
    return None


KNOWN_DEVIATIONS = {"omod_neg_zero": dev_omod_neg_zero}


def hw_verdict(case, got, model_fn=model):
    """('match' | 'mismatch', want) when the model has an answer, ('skip', reason) when it declines."""
    try:
        want = model_fn(case.op, case.vals, case.srcs, case.out_mods, translator=False)
    except Denormal as e:
        return "skip", f"denormal {e}"
    except NotModelled as e:
        return "skip", str(e)
    if hw_same(want, got, case.op):
        return "match", want
    for name, fn in KNOWN_DEVIATIONS.items():
        alt = fn(case, want)
        if alt is not None and hw_same(alt, got, case.op):
            return "deviation:" + name, want
    return "mismatch", want


# What the model gets wrong on purpose, to prove the comparison can fail (--perturb, --negative-control). Each
# entry: op -> replacement for ALL_OPS[op]["fn"]; none of them shares code with the real model's result path.
def _mul_rtz(x, y):
    r = mul(x, y)
    if is_nan(r) or is_inf(r) or is_zero(r):
        return r
    exact = to_frac(x) * to_frac(y)
    return r - 1 if abs(to_frac(r)) > abs(exact) else r


PERTURBATIONS = {
    "add_lsb": ("v_add_f32", lambda x, y: add(x, y) ^ 1),
    "mul_rtz": ("v_mul_f32", _mul_rtz),
    "fma_unfused": ("v_fma_f32", lambda x, y, z: add(mul(x, y), z)),
    "legacy_is_ieee": ("v_mul_legacy_f32", mul),
    "rndne_floor": ("v_rndne_f32", lambda x: rounding(x, "floor")),
    "min_ieee_nan": ("v_min_f32", lambda x, y: QNAN if is_nan(x) or is_nan(y) else fmin(x, y)),
}


class perturbed:
    """Context manager: swap one op's model function."""

    def __init__(self, name):
        self.op, self.fn = PERTURBATIONS[name]
        self.old = None

    def __enter__(self):
        self.old = ALL_OPS[self.op]["fn"]
        ALL_OPS[self.op]["fn"] = self.fn
        return self

    def __exit__(self, *exc):
        ALL_OPS[self.op]["fn"] = self.old


# ---- informational classification: what the hardware did where the model declines ----

DENORM_LABELS = {"keep": "kept", "ftz": "flushed in+out", "ftz_in": "inputs flushed only", "ftz_out": "results flushed only"}


def denorm_label(case, got):
    """Which denormal variants of the model the hardware's words equal: 'where' ('operand' / 'result') and the
    matching variant names ('none' when no variant does)."""
    op, vals, srcs, e64, out_mods = case.args()
    spec = ALL_OPS[op]
    ins = [s[1] if s[0] == "k" else source_value(vals, s) for s in srcs] + ([vals[8]] if "d" in spec["srcs"] else [])
    kinds = spec["srcs"].replace("d", "f")
    operand = any(is_denorm(ins[i]) for i, k in enumerate(kinds) if k in "fktde")
    hit = []
    for d in ("keep", "ftz", "ftz_in", "ftz_out"):
        try:
            w = model(op, vals, srcs, out_mods, denorm=d, translator=False)
        except (NotModelled, ValueError):
            continue
        alt = dev_omod_neg_zero(case, w)               # a flushed zero is -0 too, and omod turns that into +0
        if hw_same(w, got, op) or (alt is not None and hw_same(alt, got, op)):
            hit.append(d)
    return ("operand" if operand else "result"), "|".join(hit) or "none"


def has_denorm_in(case):
    op, vals, srcs, e64, out_mods = case.args()
    spec = ALL_OPS[op]
    ins = [s[1] if s[0] == "k" else source_value(vals, s) for s in srcs] + ([vals[8]] if "d" in spec["srcs"] else [])
    kinds = spec["srcs"].replace("d", "f")
    return any(is_denorm(ins[i]) for i, k in enumerate(kinds) if k in "fktde")


def float_inputs(case):
    """[(position, value after neg/abs)] for the float sources, in instruction order."""
    spec = ALL_OPS[case.op]
    ins = [s[1] if s[0] == "k" else source_value(case.vals, s) for s in case.srcs]
    if "d" in spec["srcs"]:
        ins.append(case.vals[8])
    kinds = spec["srcs"].replace("d", "f")
    return [(i, v) for i, (v, k) in enumerate(zip(ins, kinds)) if k in "fktde"]


CUBE_OPS = ("v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32", "v_cubema_f32")
NAN_NOT_F32 = ("v_cmp_class_f32", "v_cvt_pk_u8_f32", "v_cvt_f16_f32", "v_cvt_pkrtz_f16_f32")


def nan_facts(case, result):
    """What the hardware did with NaNs, as short strings (one case can give several). Source n is src n of the
    instruction after neg/abs. 'quieted' = the same payload with bit 22 set. Not for ops whose result is not an f32."""
    if case.op in NAN_NOT_F32:
        return nan_facts_f16(case, result) if case.op.startswith("v_cvt_") and "f16" in case.op else []
    nans = [(i, v) for i, v in float_inputs(case) if is_nan(v)]
    facts = []
    if not is_nan(result):
        return ["NaN in, no NaN out"] if nans else []
    if not nans:
        return [f"no NaN in, NaN out: 0x{result:08x}"]
    first_i, first = nans[0]
    if case.op in CUBE_OPS:
        pass                                            # the result is a selected source: no 'first' rule
    elif result in (first, first | 0x400000, first ^ 0x80000000, (first ^ 0x80000000) | 0x400000):
        facts.append("result = first NaN source" + (" (a signalling one)" if not first & 0x400000 else ""))
    elif any(result in (v, v | 0x400000) for _, v in nans[1:]):
        facts.append("result = a later NaN source")
    elif result in (0xFFC00000, 0x7FC00000):
        facts.append(f"result = default NaN 0x{result:08x}")
    else:
        facts.append(f"result = other 0x{result:08x}")
    for i, v in nans:
        if not v & 0x400000 and result == v:
            facts.append("signalling NaN passes unchanged")
            break
        if not v & 0x400000 and result in (v | 0x400000, (v ^ 0x80000000) | 0x400000):
            facts.append("signalling NaN quieted")
            break
    if any(result == (v ^ 0x80000000) | 0x400000 or result == v ^ 0x80000000 for _, v in nans):
        facts.append("sign of the NaN flipped")
    return facts


def nan_facts_f16(case, result):
    """v_cvt_f16_f32 (low half) and v_cvt_pkrtz_f16_f32 (both halves): the f16 NaN against the f32 NaN it came from."""
    ins = [v for _, v in float_inputs(case)]
    facts = []
    for h in F16_HALVES[case.op]:
        x, half = ins[h], (result >> 16 * h) & 0xFFFF
        if not is_nan(x):
            continue
        trunc = ((x >> 16) & 0x8000) | 0x7C00 | ((x >> 13) & 0x3FF) | 0x200
        facts.append("f16 NaN = f32 payload truncated to 10 bits, quiet bit set" if half == trunc else
                     "f16 NaN = canonical 0x7e00" if half & 0x7FFF == 0x7E00 else f"f16 NaN other 0x{half:04x}")
    return facts


def omod_label(case, got):
    """What an omod (mul:2, mul:4, div:2) did, judged where it changes the result and nothing is denormal."""
    if not any(m in OMOD for m in case.out_mods):
        return None
    try:
        with_ = model(case.op, case.vals, case.srcs, case.out_mods, translator=False)
        without = model(case.op, case.vals, case.srcs, [m for m in case.out_mods if m not in OMOD],
                        translator=False)
    except (NotModelled, ValueError):
        return None
    if hw_same(with_, without, case.op) and with_[0] != 0x80000000:
        return None
    if hw_same(with_, without, case.op) and hw_same(with_, got, case.op):
        return "-0 result kept (applied or ignored, same word)"     # neither applied nor ignored is shown
    if hw_same(with_, got, case.op):
        return "applied"
    alt = dev_omod_neg_zero(case, with_)
    if alt is not None and hw_same(alt, got, case.op):
        return "applied, but a -0 result becomes +0"
    if hw_same(without, got, case.op):
        return "ignored"
    return "other"


def clamp_label(case, got):
    """What a clamp bit did to a NaN or -0 result (the model declines): measured against the unclamped result."""
    op, vals, srcs, e64, out_mods = case.args()
    try:
        raw = model(op, vals, srcs, [m for m in out_mods if m != "clamp"], denorm="keep", translator=False)[0]
    except (NotModelled, ValueError):
        return "n/a"
    r = got[0]
    if is_nan(raw):
        if r == 0:
            return "NaN -> +0"
        if is_nan(r):
            return "NaN passes through" + (" (quieted)" if raw & 0x400000 == 0 and r & 0x400000 else "")
        return f"NaN -> 0x{r:08x}"
    if raw == 0x80000000:
        return {0: "-0 -> +0", 0x80000000: "-0 stays -0"}.get(r, f"-0 -> 0x{r:08x}")
    return "other"


FAMILIES = [
    ("add / sub / mul", ("v_add_f32", "v_sub_f32", "v_subrev_f32", "v_mul_f32")),
    ("DX9 legacy mul / mad", ("v_mul_legacy_f32", "v_mad_legacy_f32")),
    ("mad / mac / madak / madmk", ("v_mad_f32", "v_mac_f32", "v_madak_f32", "v_madmk_f32")),
    ("fma / fmac / fmaak / fmamk", ("v_fma_f32", "v_fmac_f32", "v_fmaak_f32", "v_fmamk_f32")),
    ("int -> f32 convert", ("v_cvt_f32_i32", "v_cvt_f32_u32", "v_cvt_f32_ubyte0", "v_cvt_f32_ubyte1",
                            "v_cvt_f32_ubyte2", "v_cvt_f32_ubyte3")),
    ("f16 convert", ("v_cvt_f32_f16", "v_cvt_f16_f32", "v_cvt_pkrtz_f16_f32")),
    ("v_cvt_pk_u8_f32", ("v_cvt_pk_u8_f32",)),
    ("floor / ceil / trunc / rndne", ("v_floor_f32", "v_ceil_f32", "v_trunc_f32", "v_rndne_f32")),
    ("rcp / rsq / sqrt / exp / log (exact inputs)", ("v_rcp_f32", "v_rcp_iflag_f32", "v_rsq_f32", "v_sqrt_f32",
                                                      "v_exp_f32", "v_log_f32")),
    ("cube", ("v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32", "v_cubema_f32")),
    ("v_cmp_class_f32", ("v_cmp_class_f32",)),
    ("min / max / med3 (hardware only)", HW_ONLY_OPS),
]


def family_of(op):
    return next(name for name, ops in FAMILIES if op in ops)


# ---- generated probes: structured inputs the random generator never produces ----

DENORMS = [0x00000001, 0x00000002, 0x007FFFFF, 0x00400000, 0x00000100, 0x00123456, 0x80000001, 0x807FFFFF,
           0x80400000]
POW2_DENORMS = [0x00400000, 0x00200000, 0x00000001, 0x00080000]        # 2^-127, 2^-128, 2^-149, 2^-130
NANS = [0x7FC00000, 0xFFC00000, 0x7FC00001, 0xFFC12345, 0x7FA00000, 0xFFA00000, 0x7F800001, 0xFF800001,
        0x7FFFFFFF, 0xFFFFFFFF, 0x7FC0DEAD, 0x7FBFFFFF]
NEAR_MIN = [bits(2.0 ** -126), bits(1.5 * 2.0 ** -126), bits(2.0 ** -125), bits(-(2.0 ** -126)),
            bits(-1.5 * 2.0 ** -126), bits(2.0 ** -126 * 1.0000001)]
SMALLS = [bits(0.5), bits(0.25), bits(-0.5), bits(1.0), bits(-1.0), bits(1.5), bits(0.75), bits(2.0 ** -10)]


def _float_slots(case):
    """Registers (or the literal) holding float operands: [(srcs index, kind char)]."""
    kinds = ALL_OPS[case.op]["srcs"].replace("d", "")
    return [(i, k) for i, k in enumerate(kinds) if k in "fkte"]


def _set(case, i, v):
    s = case.srcs[i]
    if s[0] == "k":
        case.srcs[i] = ("k", v)
    else:
        case.vals[s[0]] = v


def gen_probe(rng, op, kind):
    """A Case of the given probe kind, or None when the op has no such probe."""
    vals, srcs, e64, out_mods = gen(rng, op)
    case = Case(op, vals, list(srcs), e64, out_mods, group=kind)
    slots = _float_slots(case)
    spec = ALL_OPS[op]
    if kind == "denorm_in":
        pos = [(i, k) for i, k in slots if k in "ftk"]
        if not pos:
            return None
        for i, k in rng.sample(pos, rng.randrange(1, min(2, len(pos)) + 1)):
            _set(case, i, rng.choice(POW2_DENORMS if k == "t" else DENORMS))
        if "d" in spec["srcs"] and rng.random() < 0.3:
            case.vals[8] = rng.choice(DENORMS)
        return case
    if kind == "denorm_out":
        for _ in range(80):
            for i, k in slots:
                if k == "t":
                    _set(case, i, rng.choice(TRANS + [bits(2.0 ** 127)]))
                elif k == "e":
                    _set(case, i, rng.choice(EXPS + [bits(-127.0), bits(-140.0), bits(-149.0), bits(-150.0)]))
                elif k == "f":
                    _set(case, i, rng.choice(NEAR_MIN + SMALLS) if rng.random() < 0.7 else rand_float(rng))
                else:
                    _set(case, i, rng.choice(NEAR_MIN + SMALLS))
            if "d" in spec["srcs"]:
                case.vals[8] = rng.choice(NEAR_MIN + [0] + SMALLS)
            try:
                if not is_denorm(model(op, case.vals, case.srcs, case.out_mods, denorm="keep",
                                       translator=False)[0]):
                    continue
            except (NotModelled, ValueError):
                continue
            return case
        return None
    if kind == "nan":
        pos = [(i, k) for i, k in slots if k in "fkt"]
        if not pos:
            return None
        picked = [p for p in pos if rng.random() < 0.6] or [rng.choice(pos)]
        for i, k in picked:
            _set(case, i, rng.choice(NANS))
        return case
    if kind == "zero_inf":
        # the DX9 rule: a zero times anything (inf, NaN included) is +0; small samples need these to see a plain multiply
        if op not in ("v_mul_legacy_f32", "v_mad_legacy_f32"):
            return None
        _set(case, 0, rng.choice([0, 0x80000000]))
        _set(case, 1, rng.choice([PINF, NINF, QNAN, 0x7F800001, bits(3.0)]))
        if rng.random() < 0.5:
            x, y = case.srcs[0], case.srcs[1]
            case.srcs[0], case.srcs[1] = y, x
        return case
    if kind == "cancel":
        # z = -round(x * y): a fused multiply-add keeps the rounding residue of the product, an unfused one gives 0
        if op not in ("v_fma_f32", "v_fmac_f32", "v_fmaak_f32", "v_fmamk_f32"):
            return None
        x = (rng.getrandbits(1) << 31) | (rng.randrange(120, 135) << 23) | rng.getrandbits(23)
        y = (rng.getrandbits(1) << 31) | (rng.randrange(120, 135) << 23) | rng.getrandbits(23)
        z = neg(mul(x, y))
        _set(case, 0, x)
        _set(case, 1, y)
        if op == "v_fmac_f32":
            case.vals[8] = z
        else:
            _set(case, 2, z)
        return case
    if kind == "invalid":
        pool = [0, 0x80000000, PINF, NINF, bits(1.0), bits(-1.0), bits(2.0), bits(0.5), bits(-4.0)]
        for _ in range(80):
            for i, k in slots:
                _set(case, i, rng.choice(pool))
            if "d" in spec["srcs"]:
                case.vals[8] = rng.choice(pool)
            try:
                r = model(op, case.vals, case.srcs, case.out_mods, denorm="keep", translator=False)[0]
            except (NotModelled, ValueError):
                continue
            if is_nan(r) and not any(is_nan(v) for _, v in float_inputs(case)):
                return case
        return None
    if kind == "clamp":
        if not spec["mods"] or op in NO_OUT_MODS:
            return None
        case.e64 = True
        case.out_mods = ["clamp"] + ([rng.choice(tuple(OMOD))] if rng.random() < 0.3 else [])
        for _ in range(80):
            for i, k in slots:
                if k in "ft":
                    _set(case, i, rng.choice([QNAN, 0xFFC00000, 0x80000000, 0, bits(-1.0), bits(2.0), PINF, NINF,
                                              bits(0.5), bits(-0.0)]))
            if "d" in spec["srcs"]:
                case.vals[8] = rng.choice([0, 0x80000000, QNAN])
            try:
                model(op, case.vals, case.srcs, case.out_mods, denorm="keep", translator=False)
            except NotModelled:
                return case
            except ValueError:
                continue
        return None
    raise ValueError(kind)


PROBE_KINDS = ("denorm_in", "denorm_out", "nan", "clamp", "invalid", "cancel", "zero_inf")


def gen_cases(programs, seed, ops):
    """The seeded random programs, in the order run_suite draws them."""
    rng = random.Random(seed)
    cases = []
    for i in range(programs):
        op = rng.choice(ops)
        vals, srcs, e64, out_mods = gen(rng, op)
        cases.append(Case(op, vals, srcs, e64, out_mods, "random", i))
    return cases


def gen_probes(per_kind, seed, ops):
    rng = random.Random(seed * 7919 + 13)
    cases = []
    for op in ops:
        for kind in PROBE_KINDS:
            for j in range(per_kind):
                c = gen_probe(rng, op, kind)
                if c is not None:
                    c.index = j
                    cases.append(c)
    return cases


class HwReport:
    """Everything one hardware run learned."""

    def __init__(self):
        self.per_op = {}                 # op -> counters
        self.mismatches = []             # (case, want, got)
        self.denorm = {}                 # (mode, family, where, label) -> n
        self.nan = {}                    # (mode, op, label) -> n
        self.clamp = {}                  # (mode, label) -> n
        self.skips = {}                  # reason -> n
        self.minmax = {}                 # (mode, label) -> n
        self.results = []                # (case, primary-mode words) for every case that ran
        self.groups = {}                 # group -> counters
        self.deviations = []             # (name, case, want, got): hardware = the known deviation's words
        self.omod = {}                   # (mode, label) -> n

    def row(self, op):
        return self.per_op.setdefault(op, {"programs": 0, "match": 0, "mismatch": 0, "dev": 0, "gfx11": 0,
                                           "denorm": 0, "other": 0, "asm": 0})

    def bump(self, table, key):
        table[key] = table.get(key, 0) + 1


def hw_suite(oracle, cases, guard=True, info_modes=("keep", "no_dx10", "ieee"), translator=False, model_fn=model):
    """Run every case in the primary mode and compare with the model; then rerun the cases the model declines (and
    those with NaNs) in the other modes for the informational tables. Returns an HwReport."""
    from hw_oracle import asm
    rep = HwReport()
    runnable = []
    for c in cases:
        row = rep.row(c.op)
        row["programs"] += 1
        why = asm.valu_skip_reason(c.op, oracle.arch)
        if why:
            row["gfx11"] += 1
            rep.bump(rep.skips, why)
            continue
        runnable.append(c)
    primary = hw_run(oracle, runnable, HW_PRIMARY, guard)
    info = []
    for c, got in zip(runnable, primary):
        row = rep.row(c.op)
        if isinstance(got, str):
            row["asm"] += 1
            rep.bump(rep.skips, got)
            continue
        rep.results.append((c, got))
        verdict, want = hw_verdict(c, got, model_fn)
        grp = rep.groups.setdefault(c.group, {"programs": 0, "match": 0, "mismatch": 0, "dev": 0,
                                                             "skip": 0})
        grp["programs"] += 1
        if verdict == "match":
            row["match"] += 1
            grp["match"] += 1
        elif verdict.startswith("deviation:"):
            row["dev"] += 1
            grp["dev"] += 1
            rep.deviations.append((verdict.split(":", 1)[1], c, want, got))
        elif verdict == "mismatch":
            row["mismatch"] += 1
            grp["mismatch"] += 1
            rep.mismatches.append((c, want, got))
        else:
            row["denorm" if want.startswith("denormal") else "other"] += 1
            grp["skip"] += 1
            rep.bump(rep.skips, want)
        interesting = verdict == "skip" or has_denorm_in(c) or any(is_nan(v) for _, v in float_inputs(c)) \
            or is_nan(got[0]) or c.op in HW_ONLY_OPS or "clamp" in c.out_mods or any(m in OMOD for m in c.out_mods)
        if interesting:
            info.append(c)
        classify(rep, HW_PRIMARY, c, got, verdict, want)
    for mode in info_modes:
        for c, got in zip(info, hw_run(oracle, info, mode, guard)):
            if isinstance(got, str):
                continue
            ce = c if omod_active(mode, c.op) else strip_omod(c)     # the model has no omod to judge there
            verdict, want = hw_verdict(ce, got, model_fn)
            classify(rep, mode, ce, got, verdict, want, orig=c)
    return rep


def classify(rep, mode, c, got, verdict, want, orig=None):
    fam = family_of(c.op)
    if has_denorm_in(c) or (verdict == "skip" and want.startswith("denormal")):
        rep.bump(rep.denorm, (mode, fam, *denorm_label(c, got)))
    src = orig or c
    if "clamp" not in src.out_mods and not any(m in OMOD for m in src.out_mods):
        for fact in nan_facts(c, got[0]):
            rep.bump(rep.nan, (mode, c.op, fact))
    if "clamp" in c.out_mods and verdict == "skip" and not want.startswith("denormal"):
        rep.bump(rep.clamp, (mode, fam, clamp_label(c, got)))
    if c.op in HW_ONLY_OPS and not any(m in OMOD for m in c.out_mods):
        rep.bump(rep.minmax, (mode, "model matches" if verdict in ("match",) or verdict.startswith("deviation") else
                              "model differs" if verdict == "mismatch" else f"model declines ({want})"))
    lab = omod_label(orig or c, got)
    if lab:
        rep.bump(rep.omod, (mode, lab))


def negative_control(rep):
    """Re-judge the hardware words already collected with each deliberately wrong model. Returns {name: (cases of
    that op, mismatches the perturbation caused)}; every entry with cases must show mismatches."""
    out = {}
    for name, (op, _) in PERTURBATIONS.items():
        n = bad = 0
        with perturbed(name):
            for c, got in rep.results:
                if c.op != op:
                    continue
                n += 1
                if hw_verdict(c, got)[0] == "mismatch":
                    bad += 1
        out[name] = (n, bad)
    return out


def hw_print(rep, show=2, translator_check=None, quiet=False):
    """Tables and mismatch details. Returns the number of mismatches."""
    n_mis = len(rep.mismatches)
    if quiet:
        return n_mis
    for g, r in sorted(rep.groups.items()):
        print(f"  group {g:10s} {r['programs']:6d} ran: {r['match']} match, {r['mismatch']} mismatch, "
              f"{r['dev']} known deviation, {r['skip']} not judged")
    fam_rows = {}
    for op, r in rep.per_op.items():
        f = fam_rows.setdefault(family_of(op), dict.fromkeys(r, 0))
        for k, v in r.items():
            f[k] += v
    head = (f"{'family':46s} {'programs':>8s} {'match':>7s} {'mismatch':>8s} {'known dev':>9s} {'GFX11 skip':>10s} "
            f"{'denormal':>8s} {'other':>6s} {'asm rej':>7s}")
    print(head)
    tot = dict.fromkeys(("programs", "match", "mismatch", "dev", "gfx11", "denorm", "other", "asm"), 0)
    for name, _ in FAMILIES:
        r = fam_rows.get(name)
        if not r:
            continue
        print(f"{name:46s} {r['programs']:8d} {r['match']:7d} {r['mismatch']:8d} {r['dev']:9d} {r['gfx11']:10d} "
              f"{r['denorm']:8d} {r['other']:6d} {r['asm']:7d}")
        for k in tot:
            tot[k] += r[k]
    print(f"{'total':46s} {tot['programs']:8d} {tot['match']:7d} {tot['mismatch']:8d} {tot['dev']:9d} "
          f"{tot['gfx11']:10d} {tot['denorm']:8d} {tot['other']:6d} {tot['asm']:7d}")
    if tot["asm"]:
        print(f"WARNING: the assembler rejected {tot['asm']} program(s) the GFX11 table did not list; this is a harness "
              "problem, not a skip, and the run fails (exit 2)")
        for reason, n in sorted(rep.skips.items(), key=lambda t: -t[1]):
            if reason.startswith("assembler rejected"):
                print(f"  {n} x {reason}")
    by_dev = {}
    for name, c, _, _ in rep.deviations:
        by_dev.setdefault(name, {}).setdefault(c.op, 0)
        by_dev[name][c.op] += 1
    for name, ops in sorted(by_dev.items()):
        print(f"  known deviation {name} x{sum(ops.values())}: " + ", ".join(f"{o} {n}" for o, n in sorted(ops.items())))
    for reason, n in sorted(rep.skips.items(), key=lambda t: -t[1]):
        print(f"  skip x{n}: {reason}")
    shown = {}
    for c, want, got in rep.mismatches:
        if shown.get(c.op, 0) >= show:
            continue
        shown[c.op] = shown.get(c.op, 0) + 1
        print(f"--- {c.group} program {c.index}: {line(*c.args())}")
        used = sorted({s[0] for s in c.srcs if s[0] != "k"} | ({8} if "d" in ALL_OPS[c.op]["srcs"] else set()))
        print("    inputs: " + " ".join(f"v{r}={fdesc(c.vals[r])}" for r in used) +
              "".join(f" literal={fdesc(s[1])}" for s in c.srcs if s[0] == "k"))
        print("    model v8/vcc:    " + " ".join(fdesc(w) for w in want))
        print("    hardware v8/vcc: " + " ".join(fdesc(w) for w in got))
        if translator_check:
            print("    translator: " + translator_check(c, want, got))
    return n_mis


def info_print(rep):
    """The informational tables: what the hardware does where the model does not judge."""
    print("\nhardware behaviour (informational; not counted as match or mismatch)")
    modes = list(HW_MODES)
    if rep.denorm:
        print("\n  f32 denormals (label = which model variants the hardware equals: kept = exact denormals, "
              "flushed in+out = operands read as zero and denormal results written as zero)")
        tab = {}
        for (mode, fam, where, label), n in rep.denorm.items():
            tab.setdefault((fam, where, label), {})[mode] = n
        for (fam, where, label), by in sorted(tab.items()):
            print(f"    {fam:44s} {where:8s} {label:28s} " + "  ".join(f"{m}={by.get(m, 0)}" for m in modes))
    if rep.clamp:
        print("\n  clamp bit on a NaN or -0 result (DX10_CLAMP on = 'flush' and the other modes, off = 'no_dx10')")
        tab = {}
        for (mode, fam, label), n in rep.clamp.items():
            tab.setdefault(label, {})[mode] = tab.get(label, {}).get(mode, 0) + n
        for label, by in sorted(tab.items()):
            print(f"    {label:34s} " + "  ".join(f"{m}={by.get(m, 0)}" for m in modes))
    if rep.omod:
        print("\n  output modifier (mul:2, mul:4, div:2) where it changes the result; IEEE on = 'ieee'")
        tab = {}
        for (mode, label), n in rep.omod.items():
            tab.setdefault(label, {})[mode] = n
        for label, by in sorted(tab.items()):
            print(f"    {label:47s} " + "  ".join(f"{m}={by.get(m, 0)}" for m in modes))
    if rep.nan:
        print("\n  NaN handling (programs without clamp or omod; same program counts in every mode, so a column that "
              "differs is a mode effect)")
        tab = {}
        for (mode, op, fact), n in rep.nan.items():
            tab.setdefault((op, fact), {})[mode] = n
        for (op, fact), by in sorted(tab.items()):
            print(f"    {op:20s} {fact:62s} " + "  ".join(f"{m}={by.get(m, 0)}" for m in modes))
    if rep.minmax:
        print("\n  min / max / med3 against the RDNA2 rules measured in #634")
        tab = {}
        for (mode, label), n in rep.minmax.items():
            tab.setdefault(label, {})[mode] = n
        for label, by in sorted(tab.items()):
            print(f"    {label:60s} " + "  ".join(f"{m}={by.get(m, 0)}" for m in modes))


def self_test():
    """The model and the harness must both be able to pass and to fail."""
    ok = True
    # Model spot checks with known answers.
    checks = [
        (add(bits(1.0), bits(2.0)), bits(3.0)),
        (add(0x80000000, 0x80000000), 0x80000000),
        (add(0x80000000, 0), 0),
        (mul(bits(-2.0), 0), 0x80000000),
        (legacy_mul(PINF, 0x80000000), 0),
        (fma(bits(1.0 + 2 ** -23), bits(1.0 - 2 ** -23), bits(-1.0)), bits(-(2.0 ** -46))),
        (round_rtz_f16(bits(70000.0)), 0x7BFF),
        (round_rtz_f16(bits(1.0 + 2 ** -11)), 0x3C00),
        (round_rne_f16(bits(65520.0)), 0x7C00),
        (rounding(bits(-0.5), "rne"), 0x80000000),
        (rounding(bits(2.5), "rne"), bits(2.0)),
        (cube("v_cubeid_f32", bits(1.0), bits(-3.0), bits(2.0)), bits(3.0)),
        (cmp_class(0xFFC00000, 1 << 1), 1),
        (apply_out_mods(bits(0.75), ["mul:2", "clamp"]), bits(1.0)),
        (round_f32(Fraction(1, 3)), 0x3EAAAAAB),
    ]
    for i, (got, want) in enumerate(checks):
        if got != want:
            print(f"model check {i} FAILED: 0x{got:08x} != 0x{want:08x}")
            ok = False
    vals = [bits(1.0), bits(2.0)] + [0] * 7
    status, d = check("v_add_f32", vals, [(0, {}), (1, {})], False, [])
    if status != "match":
        print(f"canary FAILED: v_add_f32 1+2 gave {status}: {d}")
        ok = False
    real = OPS["v_add_f32"]["fn"]
    OPS["v_add_f32"]["fn"] = lambda x, y: real(x, y) ^ 1
    try:
        status, _ = check("v_add_f32", vals, [(0, {}), (1, {})], False, [])
    finally:
        OPS["v_add_f32"]["fn"] = real
    if status != "mismatch":
        print("canary FAILED: a corrupted model result was not flagged")
        ok = False
    ok &= hw_self_test()
    if ok:
        print("canaries: model checks ok, match ok, mismatch detected")
    return ok


def hw_self_test():
    """The hardware side without hardware: kernel text, the verdicts, the classifiers and the negative control."""
    ok = True
    cases = gen_cases(300, 5, tuple(ALL_OPS))
    for op, _ in PERTURBATIONS.values():
        cases += gen_cases(150, 6, (op,))
    fake = []
    for c in cases:
        try:
            w = model(c.op, c.vals, c.srcs, c.out_mods, denorm="ftz", translator=False)
        except (NotModelled, ValueError):
            continue
        fake.append((c, w))
    rep = HwReport()
    rep.results = list(fake)
    bad = [c.op for c, w in fake if hw_verdict(c, w)[0] == "mismatch"]
    if bad or len(fake) < 100:
        print(f"canary FAILED: the model disagrees with itself on {bad[:3]} ({len(fake)} cases)")
        ok = False
    nc = negative_control(rep)
    silent = [n for n, (cases_n, bad_n) in nc.items() if cases_n >= 5 and not bad_n]
    if silent:
        print(f"canary FAILED: perturbations not detected: {silent}")
        ok = False
    for c, _ in fake[:40]:
        text = "\n".join(hw_lines(c, "gfx1150"))
        if "global_store_b64 v20, v[8:9], s[2:3]" not in text or not text.endswith("s_endpgm") or "@" in text:
            print(f"canary FAILED: malformed kernel body for {c.op}")
            ok = False
            break
    a = Case("v_add_f32", [bits(1.0), bits(2.0)] + [0] * 7, [(0, {}), (1, {})], False, [])
    if hw_verdict(a, [bits(3.0), 0])[0] != "match" or hw_verdict(a, [bits(3.0) ^ 1, 0])[0] != "mismatch":
        print("canary FAILED: hw_verdict")
        ok = False
    if hw_verdict(a, [QNAN, 0])[0] != "mismatch" or hw_verdict(a, [bits(3.0), 1])[0] != "mismatch":
        print("canary FAILED: hw_verdict value/vcc")
        ok = False
    d = Case("v_add_f32", [0x00000001, bits(1.0)] + [0] * 7, [(0, {}), (1, {})], False, [])
    if hw_verdict(d, [bits(1.0), 0])[0] != "skip":
        print("canary FAILED: a denormal operand is not skipped")
        ok = False
    m = Case("v_mul_f32", [0x00000001, bits(1048576.0)] + [0] * 7, [(0, {}), (1, {})], False, [])
    if denorm_label(m, [0, 0]) != ("operand", "ftz|ftz_in|ftz_out") or \
            denorm_label(m, [bits(2.0 ** -129), 0]) != ("operand", "keep"):
        print("canary FAILED: denorm_label")
        ok = False
    n = Case("v_add_f32", [0x7FA00000, bits(1.0)] + [0] * 7, [(0, {}), (1, {})], False, [])
    if nan_facts(n, 0x7FE00000) != ["result = first NaN source (a signalling one)", "signalling NaN quieted"] or \
            nan_facts(n, 0x7FA00000)[1:] != ["signalling NaN passes unchanged"]:
        print("canary FAILED: nan_facts")
        ok = False
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="32-bit float VALU differential fuzzer: the model against the translator "
                                             "and/or the AMD GPU.")
    ap.add_argument("--programs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ops", default=None, help="comma-separated op names (default: all the oracle covers)")
    ap.add_argument("--show", type=int, default=2, help="failures to print per op")
    ap.add_argument("--list-ops", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--oracle", default="translator",
                    help="comma-separated: translator (default; needs llvm-mc for gfx1013) or hw (natively on the AMD "
                         "GPU, wave32; tools/hw_oracle/), or translator,hw. Exit 77 when the hardware tools are missing")
    ap.add_argument("--hw-arch", default=None, help="GPU architecture to assemble for (default: detected)")
    ap.add_argument("--probes", type=int, default=12,
                    help="with hw: structured probes per op and kind (denormal operand, denormal result, NaN payloads, "
                         "clamp of NaN); they are judged where the model has an answer and recorded where it has not")
    ap.add_argument("--no-hw-guard", action="store_true", help="no s_nop 7 after the instruction in the native kernels")
    ap.add_argument("--no-info", action="store_true", help="leave out the informational hardware tables")
    ap.add_argument("--perturb", choices=sorted(PERTURBATIONS), help="judge the hardware against a deliberately wrong "
                    "model (mismatches must appear)")
    ap.add_argument("--negative-control", action="store_true",
                    help="after the hardware run, re-judge its words with every perturbed model; exit 1 if one is not caught")
    ap.add_argument("--fma", choices=("glsl", "khr"), default="glsl",
                    help="with the translator oracle: emit the fused multiply-adds as GLSL Fma (default) or OpFmaKHR")
    ap.add_argument("--keep", help="write the kernel body of every hardware mismatch to this folder")
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        ap.error("no arguments given; pass --help for the available options")
    a = ap.parse_args(argv)
    rdna2spirv.FMA_KHR = a.fma == "khr"
    if a.programs < 1:
        ap.error("--programs must be at least 1")
    if a.probes < 0:
        ap.error("--probes cannot be negative")
    if a.list_ops:
        print("\n".join(ALL_OPS))
        return 0
    if a.self_test:
        return 0 if self_test() else 1
    backends = tuple(dict.fromkeys(t.strip() for t in a.oracle.split(",")))
    if not backends or any(t not in ("translator", "hw") for t in backends):
        ap.error("--oracle takes translator, hw or translator,hw")
    if (a.perturb or a.negative_control or a.keep) and "hw" not in backends:
        ap.error("--perturb, --negative-control and --keep need --oracle hw")
    ops = tuple(ALL_OPS if "hw" in backends else OPS)
    if a.ops:
        ops = tuple(o.strip() for o in a.ops.split(",") if o.strip())
        unknown = [o for o in ops if o not in ALL_OPS]
        if unknown:
            ap.error(f"unknown ops: {', '.join(unknown)} (see --list-ops)")
        if "hw" not in backends and any(o in HW_ONLY_OPS for o in ops):
            ap.error(f"{', '.join(o for o in ops if o in HW_ONLY_OPS)} can only be run with --oracle hw")
    oracle = None
    if "hw" in backends:
        from hw_oracle import runner
        try:
            oracle = runner.open_oracle(arch=a.hw_arch)
        except runner.Unavailable as e:
            print(f"hardware oracle unavailable: {e}")
            return 77
        print(f"hardware oracle: {oracle.device}")
    rc = 0
    harness = False
    try:
        cases = gen_cases(a.programs, a.seed, ops)
        if "translator" in backends:
            counts, _, _ = translate_cases(cases, a.seed, show=a.show)
            rc |= bool(counts["mismatch"])
        if oracle:
            words = hw_probe_modes(oracle)
            print("MODE read back in the kernels: " + ", ".join(f"{n} 0x{w:x}" for n, w in words.items()))
            if a.probes:
                cases += gen_probes(a.probes, a.seed, ops)
            model_ctx = perturbed(a.perturb) if a.perturb else None
            if model_ctx:
                model_ctx.__enter__()
            try:
                rep = hw_suite(oracle, cases, guard=not a.no_hw_guard,
                               info_modes=() if a.no_info else tuple(m for m in HW_MODES if m != HW_PRIMARY))
            finally:
                if model_ctx:
                    model_ctx.__exit__(None, None, None)
            print(f"\nhardware vs F32Model{' (perturbed: ' + a.perturb + ')' if a.perturb else ''}, "
                  f"{len(cases)} programs, mode {HW_PRIMARY} (MODE 0x{words[HW_PRIMARY]:x}), seed {a.seed}:")
            tr = translator_verdict if "translator" in backends else None
            n_mis = hw_print(rep, show=a.show, translator_check=tr)
            if any(r["asm"] for r in rep.per_op.values()):
                harness = True
            if tr and rep.deviations:
                tally = {}
                for name, c, want, got in rep.deviations:
                    if c.op in HW_ONLY_OPS:
                        continue
                    verdict = tr(c, want, got).split(" (")[0]
                    tally[(name, verdict)] = tally.get((name, verdict), 0) + 1
                for (name, verdict), n in sorted(tally.items()):
                    print(f"  translator on known deviation {name} x{n}: {verdict}")
            if not a.no_info:
                info_print(rep)
            if a.keep and rep.mismatches:
                os.makedirs(a.keep, exist_ok=True)
                for k, (c, _, _) in enumerate(rep.mismatches):
                    with open(os.path.join(a.keep, f"{c.group}_{c.index}_{c.op}_{k}.asm"), "w", encoding="utf-8", newline="\n") as f:
                        f.write("\n".join(hw_lines(c, oracle.arch, not a.no_hw_guard)) + "\n")
            if a.perturb:
                print(f"perturbed model '{a.perturb}': {n_mis} mismatches" + ("" if n_mis else " -- NOT DETECTED"))
                rc |= not n_mis
                return 2 if harness else int(bool(rc))
            if a.negative_control:
                print("\nnegative control (wrong models judged against the same hardware words):")
                for name, (n, bad) in negative_control(rep).items():
                    status = "caught" if bad else ("no programs of that op ran" if not n else "NOT CAUGHT")
                    print(f"  {name:16s} {n:5d} programs of {PERTURBATIONS[name][0]}: {bad} mismatches ({status})")
                    rc |= bool(n and not bad)
            rc |= bool(n_mis)
    finally:
        if oracle:
            oracle.close()
    return 2 if harness else int(bool(rc))


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
