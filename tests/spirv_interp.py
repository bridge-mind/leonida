"""A tiny interpreter for the SPIR-V assembly that tools/rdna2spirv.py emits, so tests can check computed values
without a GPU.

It only covers the straight-line, one-lane subset the translator produces for branch-free shaders: scalar/vector
arithmetic, bit ops, bitcasts, the GLSL.std.450 functions used so far, and Function/Output variables. Anything else
raises NotImplementedError (a test then fails loudly instead of passing on a guess). Floats are rounded to binary32
after every operation. An integer is always held as its unsigned 32-bit pattern, whatever the signedness of its
type; the instructions that read it as signed (OpS*, SMin, SMax) convert with s32() themselves, as SPIR-V defines
them on the bits and not on the type. Texture, buffer and control-flow instructions are out of scope.
"""
import math
import re
import struct

M32 = 0xFFFFFFFF


def f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


class _F32(float):
    """A binary32 value made from a bit pattern (OpBitcast) that keeps that exact pattern.

    struct's binary32 -> double conversion quiets a signalling NaN (sets bit 22, issue #509), so the float alone
    cannot give the word back. Copies, selects, composites, loads and stores pass the value on unchanged and keep
    `bits`; arithmetic on it returns a plain float, which drops them (SPIR-V leaves the bits of a NaN result
    open). The same fix as spirv_mini_interp (#533)."""

    __slots__ = ("bits",)

    def __new__(cls, bits):
        self = super().__new__(cls, struct.unpack("<f", struct.pack("<I", bits))[0])
        self.bits = bits
        return self


def f2u(x):
    if isinstance(x, _F32):
        return x.bits
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(u):
    return _F32(u & M32)


def s32(u):
    u &= M32
    return u - (1 << 32) if u >> 31 else u


def half_to_float(h):
    # Bit-level f16 -> f32; every non-NaN result is exact. struct's '<e' is not used because its NaN result
    # depends on the Python version: before 3.14 it drops the payload and returns the canonical quiet NaN, from
    # 3.14 it keeps sign and payload (CPython gh-130317). Pinned rule, the one CI's 3.12 sees: a NaN half expands
    # to the canonical quiet NaN with its sign, payload dropped.
    h &= 0xFFFF
    sign = (h >> 15) << 31
    exp = (h >> 10) & 0x1F
    frac = h & 0x3FF
    if exp == 0x1F:
        bits = sign | (0x7FC00000 if frac else 0x7F800000)     # NaN: canonical quiet, payload dropped; infinity
    elif exp == 0 and frac:
        b = frac.bit_length()
        bits = sign | ((b + 102) << 23) | ((frac << (24 - b)) & 0x7FFFFF)   # subnormal half -> normal f32
    elif exp == 0:
        bits = sign                                            # +-0
    else:
        bits = sign | ((exp + 112) << 23) | (frac << 13)       # normal half -> f32, exact
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def float_to_half(x):
    # Bit-level double -> f16, round to nearest even; overflow gives infinity (struct's '<e' raised OverflowError
    # here, which the old except mapped to the same words). '<e' is not used for the same version dependence as
    # half_to_float (gh-130317); pinned rule: any NaN becomes the canonical quiet NaN with the operand's sign,
    # payload dropped. Rounding runs on the operand's own 64-bit pattern, so normal, subnormal and tie results
    # cannot change with the Python version, and the double is rounded once, as struct's '<e' rounded it.
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    sign = (bits >> 63) << 15
    exp = (bits >> 52) & 0x7FF
    mant = bits & ((1 << 52) - 1)
    if exp == 0x7FF:
        return sign | (0x7E00 if mant else 0x7C00)             # NaN: canonical quiet, sign kept; infinity
    if exp == 0:
        return sign                                            # +-0 and double subnormals sit far below the half grid
    e = exp - 1023
    full = (1 << 52) | mant
    if e < -14:
        drop = 28 - e                                          # subnormal grid: multiples of 2**-24
        n = full >> drop
        rem = full & ((1 << drop) - 1)
        if rem > (1 << (drop - 1)) or (rem == (1 << (drop - 1)) and n & 1):
            n += 1
        if n > 0x3FF:                                          # rounds up to the smallest normal
            return sign | 0x0400
        return sign | n
    n = full >> 42                                             # keep the top 11 significand bits, RNE the rest
    rem = full & ((1 << 42) - 1)
    if rem > (1 << 41) or (rem == (1 << 41) and n & 1):
        n += 1
    if n > 0x7FF:                                              # significand carry bumps the exponent
        e += 1
        n = 0x400
    if e > 15:                                                 # overflow, incl. the 65520 RNE tie
        return sign | 0x7C00
    return sign | ((e + 15) << 10) | (n & 0x3FF)


class Result:
    def __init__(self, vgpr, sgpr, outputs):
        self.vgpr, self.sgpr, self.outputs = vgpr, sgpr, outputs


def _ints(args):
    return [a for a in args]


def run(spvasm):
    types, consts, mem = {}, {}, {}
    body = []

    def declare(res, args):
        mem[res] = {"Output": [(0.0, 0.0, 0.0, 0.0)], "PushConstant": "pc", "Input": [0]}.get(args[1])
        if args[1] == "Function":
            init = consts[args[2]] if len(args) > 2 else 0
            mem[res] = list(init) if isinstance(init, list) else [init]

    in_func = False
    for raw in spvasm.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"(%\S+)\s*=\s*(Op\w+)\s*(.*)$", line)
        if m:
            res, op, rest = m.groups()
        else:
            res, (op, _, rest) = None, line.partition(" ")
        args = rest.split()
        if op == "OpFunction":
            in_func = True
            continue
        if in_func:
            body.append((res, op, args))
            continue
        if op.startswith("OpType"):
            types[res] = (op[6:],) + tuple(args)
        elif op == "OpConstant":
            t = types[args[0]][0]
            if t == "Float":
                consts[res] = float(args[1]) if types[args[0]][1] == "64" else f32(float(args[1]))
            else:
                consts[res] = int(args[1]) & M32
        elif op == "OpConstantTrue":
            consts[res] = True
        elif op == "OpConstantFalse":
            consts[res] = False
        elif op == "OpConstantNull":
            t = types[args[0]]
            consts[res] = [0] * int(consts[t[2]]) if t[0] == "Array" else 0
        elif op == "OpVariable":
            declare(res, args)

    val = dict(consts)

    def get(a):
        return val[a]

    def ptr(a):
        return val[a] if a in val and isinstance(val[a], tuple) and len(val[a]) == 2 and isinstance(val[a][0], list) else (mem[a], None)

    def load(p):
        box, idx = p
        return box[0] if idx is None else box[idx]

    def store(p, v):
        box, idx = p
        if idx is None:
            box[0] = v
        else:
            box[idx] = v

    ARITH = {
        "OpIAdd": lambda a, b: (a + b) & M32, "OpISub": lambda a, b: (a - b) & M32, "OpIMul": lambda a, b: (a * b) & M32,
        "OpBitwiseAnd": lambda a, b: a & b, "OpBitwiseOr": lambda a, b: a | b, "OpBitwiseXor": lambda a, b: a ^ b,
        "OpShiftLeftLogical": lambda a, b: (a << (b & 31)) & M32 if b < 32 else 0,
        "OpShiftRightLogical": lambda a, b: a >> b if b < 32 else 0,
        "OpFAdd": lambda a, b: f32(a + b), "OpFSub": lambda a, b: f32(a - b), "OpFMul": lambda a, b: f32(a * b),
        # x / +-0: infinity with the sign of x times the sign of the zero; 0 / 0 and NaN / 0 are NaN.
        "OpFDiv": lambda a, b: f32(a / b) if b else
        math.copysign(math.inf, math.copysign(1.0, a) * math.copysign(1.0, b)) if a and not math.isnan(a) else math.nan,
        "OpFOrdGreaterThan": lambda a, b: a > b, "OpFOrdLessThan": lambda a, b: a < b,
        "OpFOrdEqual": lambda a, b: a == b,
        "OpFOrdLessThanEqual": lambda a, b: a <= b, "OpFOrdGreaterThanEqual": lambda a, b: a >= b,   # False on a NaN
        "OpLogicalOr": lambda a, b: a or b, "OpLogicalAnd": lambda a, b: a and b,
        "OpSLessThanEqual": lambda a, b: s32(a) <= s32(b), "OpSGreaterThanEqual": lambda a, b: s32(a) >= s32(b),
        "OpULessThan": lambda a, b: a < b, "OpUGreaterThan": lambda a, b: a > b,
        "OpULessThanEqual": lambda a, b: a <= b, "OpUGreaterThanEqual": lambda a, b: a >= b,
        "OpFUnordNotEqual": lambda a, b: a != b,          # true when either is NaN, as Python's != is
        "OpIEqual": lambda a, b: a == b, "OpINotEqual": lambda a, b: a != b,
    }

    F64 = {"OpFAdd": lambda a, b: a + b, "OpFSub": lambda a, b: a - b, "OpFMul": lambda a, b: a * b}

    def float_bits(rt):
        t = types.get(rt)
        if not t:
            return 32
        if t[0] == "Vector":
            t = types.get(t[1], ())
        return int(t[1]) if t and t[0] == "Float" else 32

    def glsl(fn, a, bits=32):
        round_out = (lambda v: float(v)) if bits == 64 else f32
        if fn in ("Fma", "FMin", "FMax", "FClamp", "FAbs", "Sqrt", "InverseSqrt", "Exp2", "Log2", "Sin", "Cos", "Fract") and isinstance(a[0], tuple):      # vector: per component
            return tuple(glsl(fn, list(c), bits) for c in zip(*a))
        if fn == "UnpackHalf2x16":
            return (f32(half_to_float(a[0])), f32(half_to_float(a[0] >> 16)))
        if fn == "PackHalf2x16":
            return float_to_half(a[0][0]) | float_to_half(a[0][1]) << 16
        if fn == "Fma":
            from fractions import Fraction
            if any(math.isnan(x) or math.isinf(x) for x in a):
                return round_out(a[0] * a[1] + a[2])
            return round_out(float(Fraction(a[0]) * Fraction(a[1]) + Fraction(a[2])))
        rounding = {"Floor": math.floor, "Ceil": math.ceil, "Trunc": math.trunc, "RoundEven": round}
        if fn in rounding:      # a zero result keeps the sign of the operand (IEEE 754; RDNA2 and Vulkan GPUs do)
            return a[0] if math.isnan(a[0]) or math.isinf(a[0]) else math.copysign(float(rounding[fn](a[0])), a[0])
        simple = {"FMin": min, "FMax": max, "FAbs": abs, "Sin": math.sin, "Cos": math.cos,
                  "Sqrt": math.sqrt, "InverseSqrt": lambda x: 1 / math.sqrt(x), "Exp2": lambda x: 2.0 ** x,
                  "Log2": math.log2}
        if fn in ("NMin", "NMax"):
            # A NaN operand gives the other operand (both NaN: a NaN); zeros ordered -0 < +0 as on RDNA2.
            def pick(x, y):
                if math.isnan(x) or math.isnan(y):
                    return y if math.isnan(x) else x
                kx, ky = (x, math.copysign(1.0, x)), (y, math.copysign(1.0, y))
                return y if (ky < kx if fn == "NMin" else ky > kx) else x
            if isinstance(a[0], tuple):
                return tuple(pick(x, y) for x, y in zip(a[0], a[1]))
            return pick(a[0], a[1])
        if fn in simple:
            return round_out(float(simple[fn](*a)))
        if fn == "Fract":
            return round_out(a[0] - math.floor(a[0]))
        if fn == "FClamp":
            return round_out(min(max(a[0], a[1]), a[2]))
        if fn in ("SMax", "SMin"):
            return (max if fn == "SMax" else min)(*a, key=s32)
        if fn in ("UMax", "UMin"):
            return (max if fn == "UMax" else min)(*a)
        raise NotImplementedError("GLSL." + fn)

    for res, op, args in body:
        if op in ("OpLabel", "OpReturn", "OpFunctionEnd"):
            continue
        if op == "OpVariable":
            declare(res, args)
            continue
        if op == "OpAccessChain":
            base = args[1]
            if mem.get(base) == "pc":
                val[res] = ([0], 0)       # user data / push constants read as zero
            else:
                val[res] = (mem[base], int(get(args[2])))
        elif op == "OpLoad":
            val[res] = load(ptr(args[1]))
        elif op == "OpStore":
            store(ptr(args[0]), get(args[1]))
        elif op == "OpCopyObject":
            val[res] = get(args[1])
        elif op == "OpBitcast":
            t, v = types[args[0]][0], get(args[1])
            if types[args[0]] == ("Float", "64"):          # v2uint (lo, hi) -> double
                val[res] = struct.unpack("<d", struct.pack("<II", v[0] & M32, v[1] & M32))[0]
            elif t == "Vector" and isinstance(v, float):  # double -> v2uint (lo, hi)
                val[res] = struct.unpack("<II", struct.pack("<d", v))
            elif t == "Float":
                val[res] = u2f(v) if isinstance(v, int) else v
            else:
                val[res] = f2u(v) if isinstance(v, float) else v & M32
        elif op == "OpSelect":
            val[res] = get(args[2]) if get(args[1]) else get(args[3])
        elif op in ARITH:
            t, x, y = types.get(args[0], ("",)), get(args[1]), get(args[2])
            if t == ("Float", "64") and op in F64:
                val[res] = F64[op](x, y)                   # binary64: Python floats, no f32 rounding
            elif isinstance(x, tuple) and isinstance(y, tuple):     # vector operands: component by component
                val[res] = tuple(ARITH[op](p, q) for p, q in zip(x, y))
            else:
                val[res] = ARITH[op](x, y)
        elif op == "OpNot":
            val[res] = ~get(args[1]) & M32
        elif op == "OpLogicalNot":
            val[res] = not get(args[1])
        elif op == "OpFNegate":
            val[res] = -get(args[1])
        elif op == "OpIsNan":
            val[res] = math.isnan(get(args[1]))
        elif op == "OpIsInf":
            val[res] = math.isinf(get(args[1]))
        elif op == "OpCompositeConstruct":
            val[res] = tuple(get(a) for a in args[1:])
        elif op == "OpCompositeExtract":
            val[res] = get(args[1])[int(args[2])]
        elif op == "OpExtInst":
            fn = args[2]
            val[res] = glsl(fn, [get(a) for a in args[3:]], float_bits(args[0]))
        elif op == "OpBitFieldSExtract":
            v, off, cnt = s32(get(args[1])), get(args[2]), get(args[3])
            x = (v >> off) & ((1 << cnt) - 1)
            val[res] = (x - (1 << cnt) if cnt and x >> (cnt - 1) else x) & M32
        elif op == "OpBitFieldUExtract":
            val[res] = (get(args[1]) >> get(args[2])) & ((1 << get(args[3])) - 1)
        elif op == "OpBitCount":
            val[res] = bin(get(args[1]) & M32).count("1")
        elif op == "OpConvertUToF":
            val[res] = f32(float(get(args[1])))
        elif op == "OpConvertSToF":
            val[res] = f32(float(s32(get(args[1]))))
        elif op == "OpConvertFToU":
            x = get(args[1]); val[res] = 0 if math.isnan(x) or x <= 0 else M32 if x >= 2**32 else int(x)
        elif op == "OpConvertFToS":
            x = get(args[1])
            val[res] = 0 if math.isnan(x) else (-2**31 if x <= -2**31 else 2**31 - 1 if x >= 2**31 else int(x)) & M32
        elif op == "OpIAddCarry":
            t = get(args[1]) + get(args[2]); val[res] = (t & M32, t >> 32)
        elif op == "OpISubBorrow":
            a, b = get(args[1]), get(args[2]); val[res] = ((a - b) & M32, int(a < b))
        elif op == "OpUMulExtended":
            t = get(args[1]) * get(args[2]); val[res] = (t & M32, t >> 32)
        elif op == "OpSMulExtended":
            t = s32(get(args[1])) * s32(get(args[2])); val[res] = (t & M32, (t >> 32) & M32)
        elif op in ("OpShiftRightArithmetic",):
            val[res] = (s32(get(args[1])) >> (get(args[2]) & 31)) & M32
        else:
            raise NotImplementedError(op)
    vg = next(v for k, v in mem.items() if k == "%vgpr")
    sg = next(v for k, v in mem.items() if k == "%sgpr")
    outs = {k[1:]: v[0] for k, v in mem.items() if k.startswith("%out_")}
    return Result([x & M32 for x in vg], list(sg), outs)
