"""Plain-Python binary16 reference for the semantic f16 tests (standard library only).

Finite results are computed exactly with fractions and rounded once, to nearest even, into binary16 (subnormals and
overflow to infinity included); infinities and NaNs follow IEEE 754. A NaN result is compared by class, not payload:
use same_f16().
"""
import math
import struct
from fractions import Fraction

OLD = 0xDEADBEEF        # destination seed: whichever half an instruction does not write must survive


def f16_value(h):
    """The float a binary16 word stands for (exact)."""
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def is_nan(h):
    return (h & 0x7C00) == 0x7C00 and (h & 0x03FF) != 0


def same_f16(a, b):
    """Equal words, or both NaN."""
    return a == b or (is_nan(a) and is_nan(b))


def round_f16(x):
    """binary16 word nearest to the Fraction x (ties to even, overflow to infinity); +0 for an exact zero."""
    sign = 0x8000 if x < 0 else 0
    x = abs(x)
    if x == 0:
        return 0
    # Quantum: 2^-24 below the smallest normal (2^-14), else 2^(floor(log2 x) - 10).
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** e > x:
        e -= 1
    q = Fraction(2) ** (max(e, -14) - 10)
    m, rem = divmod(x, q)
    m = int(m)
    if rem > q / 2 or (rem == q / 2 and m & 1):
        m += 1
    # m counts quanta; at or above 2^11 quanta the exponent steps up (the same encoding handles the carry).
    if m < 0x400:                       # subnormal
        return sign | m
    exp = max(e, -14)
    while m >= 0x800:
        m >>= 1
        exp += 1
    bits = ((exp + 15) << 10) | (m - 0x400)
    return sign | (0x7C00 if bits >= 0x7C00 else bits)


def to_f16(v):
    """binary16 word for a float or Fraction result (inf and NaN pass through)."""
    if isinstance(v, float):
        if math.isnan(v):
            return 0x7E00
        if math.isinf(v):
            return 0x7C00 if v > 0 else 0xFC00
        v = Fraction(v)
    return round_f16(v)


def _special(*hs):
    return any((h & 0x7C00) == 0x7C00 for h in hs)


def ref_add(a, b):
    if _special(a, b):
        return to_f16(f16_value(a) + f16_value(b))
    if a == 0x8000 and b == 0x8000:
        return 0x8000
    return round_f16(Fraction(f16_value(a)) + Fraction(f16_value(b)))


def ref_sub(a, b):
    return ref_add(a, b ^ 0x8000) if not is_nan(b) else ref_add(a, b)


def ref_mul(a, b):
    if _special(a, b):
        return to_f16(f16_value(a) * f16_value(b))
    r = round_f16(Fraction(f16_value(a)) * Fraction(f16_value(b)))
    return r | (0x8000 if (a ^ b) & 0x8000 and r == 0 else 0)      # a zero product keeps the sign rule


def ref_fma(a, b, c):
    if _special(a, b, c):
        return to_f16(f16_value(a) * f16_value(b) + f16_value(c))
    p = Fraction(f16_value(a)) * Fraction(f16_value(b))
    if p == 0 and (c & 0x7FFF) == 0:
        # +-0 + +-0: negative only when both are negative
        return 0x8000 if ((a ^ b) & 0x8000) and (c & 0x8000) else 0
    return round_f16(p + Fraction(f16_value(c)))


def neg(h):
    return h ^ 0x8000


def fabs(h):
    return h & 0x7FFF


def scale(h, factor):
    """h * factor for a power-of-two output modifier (inputs here stay in the normal range)."""
    return to_f16(f16_value(h) * factor) if not is_nan(h) else h


def clamp01(h):
    """clamp to [0, 1]; a NaN becomes +0."""
    if is_nan(h):
        return 0
    v = f16_value(h)
    return 0x3C00 if v > 1.0 else (0 if v < 0.0 or h == 0x8000 else h)
