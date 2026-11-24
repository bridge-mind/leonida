"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_f32_f64 and v_cvt_f64_f32.

Models IEEE 754-2008 / IEEE 754-2019 binary32 <-> binary64 floating-point conversions and AMD RDNA2 / GFX10
Vector Arithmetic Logic Unit (VALU) architectural semantics:
  - v_cvt_f32_f64: D.f = (float)S0.d   (double-precision binary64 to single-precision binary32)
  - v_cvt_f64_f32: D.d = (double)S0.f  (single-precision binary32 to double-precision binary64)

RDNA2 Floating-Point Conversion ISA Rules:
  1. Register Widths and Alignment:
     - v_cvt_f32_f64: Destination is a single 32-bit VGPR (v[d]). Source is an aligned 64-bit pair
       (v[s:s+1] or s[s:s+1]) or 64-bit immediate.
     - v_cvt_f64_f32: Destination is an aligned 64-bit pair (v[d:d+1]). Source is a single 32-bit
       VGPR (v[s]) or SGPR (s[s]) or 32-bit immediate.
  2. Rounding Mode:
     - Round-to-nearest-even (IEEE 754 roundTiesToEven) is standard.
     - v_cvt_f64_f32 is exact for all finite binary32 values without loss of precision.
     - v_cvt_f32_f64 rounds 53 mantissa bits to 24 bits with tie-breaking to even LSB.
  3. Boundary Extremes & Special Values:
     - Signed Zeros: +0.0 and -0.0 are preserved across conversions (+0.0 <-> +0.0, -0.0 <-> -0.0).
     - Infinities: +inf and -inf convert to corresponding target infinities (+inf <-> +inf, -inf <-> -inf).
     - Overflow: In v_cvt_f32_f64, magnitudes >= 2^128 - 2^104 round to +-infinity.
     - Subnormals:
       * f32 subnormals convert exactly to normalized f64 numbers without precision loss.
       * f64 subnormals (magnitudes < 2^-1022) underflow and round to +-0.0 in f32.
       * f64 numbers in range [2^-149, 2^-126) convert to f32 subnormals under roundTiesToEven.
  4. NaN Propagation:
     - Input NaN produces a canonical quiet NaN in the target format (f32: 0x7FC00000, f64: 0x7FF8000000000000).
     - Signaling NaN inputs trigger the IEEE invalid operation status flag.
  5. VOP1 / VOP3 Encoding and Modifiers:
     - VOP1 (e32): Compact encoding without modifiers.
     - VOP3 (e64): Extended encoding supporting input modifiers:
       * abs (|x|): Clears sign bit before conversion.
       * neg (-x): Inverts sign bit before conversion.
       * neg(abs(x)): Inverts sign after abs (forcing negative).
       * clamp: Clamps converted result to [0.0, 1.0]. Clamping negative values or -0.0 yields +0.0.
         Clamping NaN returns NaN unclamped.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import struct
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary32 Bit Patterns
SIGN_MASK_F32: int = 0x80000000
EXP_MASK_F32: int = 0x7F800000
MANT_MASK_F32: int = 0x007FFFFF
QUIET_NAN_BIT_F32: int = 0x00400000

POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_ONE_F32: int = 0x3F800000          # 1.0f
NEG_ONE_F32: int = 0xBF800000          # -1.0f
POS_TWO_F32: int = 0x40000000          # 2.0f
NEG_TWO_F32: int = 0xC0000000          # -2.0f
HALF_F32: int = 0x3F000000             # 0.5f
NEG_HALF_F32: int = 0xBF000000         # -0.5f
POS_TEN_F32: int = 0x41200000          # 10.0f
NEG_TEN_F32: int = 0xC1200000          # -10.0f

FLT_MIN: int = 0x00800000              # Smallest positive normal f32: 2^-126
NEG_FLT_MIN: int = 0x80800000          # -2^-126
FLT_MAX: int = 0x7F7FFFFF              # Largest positive normal f32: (2 - 2^-23) * 2^127
NEG_FLT_MAX: int = 0xFF7FFFFF
POS_INF_F32: int = 0x7F800000          # +infinity f32
NEG_INF_F32: int = 0xFF800000          # -infinity f32
QNAN_F32: int = 0x7FC00000             # Canonical quiet NaN f32
NEG_QNAN_F32: int = 0xFFC00000         # Negative quiet NaN f32
QNAN_PAYLOAD_A_F32: int = 0x7FC00001
QNAN_PAYLOAD_B_F32: int = 0xFFC10002
SNAN_F32: int = 0x7F800001             # Signaling NaN (MSB of mantissa is 0, payload != 0)
SUBNORM_MIN_F32: int = 0x00000001      # Smallest positive subnormal f32: 2^-149
SUBNORM_MAX_F32: int = 0x007FFFFF      # Largest positive subnormal f32: (1 - 2^-23) * 2^-126
NEG_SUBNORM_MIN_F32: int = 0x80000001
NEG_SUBNORM_MAX_F32: int = 0x807FFFFF

# IEEE 754 binary64 Bit Patterns
SIGN_MASK_F64: int = 0x8000000000000000
EXP_MASK_F64: int = 0x7FF0000000000000
MANT_MASK_F64: int = 0x000FFFFFFFFFFFFF
QUIET_NAN_BIT_F64: int = 0x0008000000000000

POS_ZERO_F64: int = 0x0000000000000000
NEG_ZERO_F64: int = 0x8000000000000000
POS_ONE_F64: int = 0x3FF0000000000000          # 1.0
NEG_ONE_F64: int = 0xBFF0000000000000          # -1.0
POS_TWO_F64: int = 0x4000000000000000          # 2.0
NEG_TWO_F64: int = 0xC000000000000000          # -2.0
HALF_F64: int = 0x3FE0000000000000             # 0.5
NEG_HALF_F64: int = 0xBFE0000000000000         # -0.5
POS_TEN_F64: int = 0x4024000000000000          # 10.0
NEG_TEN_F64: int = 0xC024000000000000          # -10.0

DBL_MIN: int = 0x0010000000000000              # Smallest positive normal f64: 2^-1022
NEG_DBL_MIN: int = 0x8010000000000000          # -2^-1022
DBL_MAX: int = 0x7FEFFFFFFFFFFFFF              # Largest positive normal f64
NEG_DBL_MAX: int = 0xFFEFFFFFFFFFFFFF
POS_INF_F64: int = 0x7FF0000000000000          # +infinity f64
NEG_INF_F64: int = 0xFFF0000000000000          # -infinity f64
QNAN_F64: int = 0x7FF8000000000000             # Canonical quiet NaN f64
NEG_QNAN_F64: int = 0xFFF8000000000000         # Negative quiet NaN f64
QNAN_PAYLOAD_A_F64: int = 0x7FF8000000000001
QNAN_PAYLOAD_B_F64: int = 0xFFF9000000000002
SNAN_F64: int = 0x7FF0000000000001             # Signaling NaN
SUBNORM_MIN_F64: int = 0x0000000000000001      # Smallest positive subnormal f64: 2^-1074
SUBNORM_MAX_F64: int = 0x000FFFFFFFFFFFFF      # Largest positive subnormal f64
NEG_SUBNORM_MIN_F64: int = 0x8000000000000001
NEG_SUBNORM_MAX_F64: int = 0x800FFFFFFFFFFFFF

# Boundary overflow threshold: 2^128 - 2^104 (doubles >= this magnitude round to f32 inf in roundTiesToEven)
OVERFLOW_THRESHOLD_F64: float = 3.4028235677973366e+38


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned word as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & SIGN_MASK_F64) else val


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF_F32 if val > 0 else NEG_INF_F32


def bits_to_f64(bits: int) -> float:
    """Decode 64-bit unsigned integer to IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    """Encode double float to IEEE 754 double-precision 64-bit unsigned integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return POS_INF_F64 if val > 0 else NEG_INF_F64


def is_nan_f32(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 single-precision NaN."""
    bits &= MASK32
    return (bits & EXP_MASK_F32) == EXP_MASK_F32 and (bits & MANT_MASK_F32) != 0


def is_qnan_f32(bits: int) -> bool:
    """True if bits represent a quiet single-precision NaN (bit 22 is 1)."""
    bits &= MASK32
    return is_nan_f32(bits) and bool(bits & QUIET_NAN_BIT_F32)


def is_snan_f32(bits: int) -> bool:
    """True if bits represent a signaling single-precision NaN (bit 22 is 0)."""
    bits &= MASK32
    return is_nan_f32(bits) and not bool(bits & QUIET_NAN_BIT_F32)


def is_subnormal_f32(bits: int) -> bool:
    """True if bits represent an IEEE 754 single-precision subnormal number."""
    bits &= MASK32
    return (bits & EXP_MASK_F32) == 0 and (bits & MANT_MASK_F32) != 0


def is_nan_f64(bits: int) -> bool:
    """True if 64-bit integer represents an IEEE 754 double-precision NaN."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == EXP_MASK_F64 and (bits & MANT_MASK_F64) != 0


def is_qnan_f64(bits: int) -> bool:
    """True if bits represent a quiet double-precision NaN (bit 51 is 1)."""
    bits &= MASK64
    return is_nan_f64(bits) and bool(bits & QUIET_NAN_BIT_F64)


def is_snan_f64(bits: int) -> bool:
    """True if bits represent a signaling double-precision NaN (bit 51 is 0)."""
    bits &= MASK64
    return is_nan_f64(bits) and not bool(bits & QUIET_NAN_BIT_F64)


def is_subnormal_f64(bits: int) -> bool:
    """True if bits represent an IEEE 754 double-precision subnormal number."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == 0 and (bits & MANT_MASK_F64) != 0


def apply_f64_modifiers(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    """Applies VOP3 input modifiers to 64-bit float: abs (|x|) first, then neg (-x)."""
    bits &= MASK64
    if abs_val:
        bits &= ~SIGN_MASK_F64
    if neg:
        bits ^= SIGN_MASK_F64
    return bits & MASK64


def apply_f32_modifiers(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    """Applies VOP3 input modifiers to 32-bit float: abs (|x|) first, then neg (-x)."""
    bits &= MASK32
    if abs_val:
        bits &= ~SIGN_MASK_F32
    if neg:
        bits ^= SIGN_MASK_F32
    return bits & MASK32


def apply_f32_clamp(bits: int) -> int:
    """Applies VOP3 clamp modifier to 32-bit float: clamps result to [0.0, 1.0]."""
    bits &= MASK32
    if is_nan_f32(bits):
        return bits
    val = bits_to_f32(bits)
    if val < 0.0 or (val == 0.0 and (bits & SIGN_MASK_F32)):
        return POS_ZERO_F32
    if val > 1.0:
        return POS_ONE_F32
    return bits


def apply_f64_clamp(bits: int) -> int:
    """Applies VOP3 clamp modifier to 64-bit float: clamps result to [0.0, 1.0]."""
    bits &= MASK64
    if is_nan_f64(bits):
        return bits
    val = bits_to_f64(bits)
    if val < 0.0 or (val == 0.0 and (bits & SIGN_MASK_F64)):
        return POS_ZERO_F64
    if val > 1.0:
        return POS_ONE_F64
    return bits


def ref_cvt_f32_f64(
    src64: int,
    clamp: bool = False,
    neg: bool = False,
    abs_val: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_f32_f64.

    Converts 64-bit double-precision IEEE 754 value to 32-bit single-precision IEEE 754 value.
    """
    s64_mod = apply_f64_modifiers(src64, neg=neg, abs_val=abs_val)

    # 1. NaN handling: canonical quiet NaN
    if is_nan_f64(s64_mod):
        res = QNAN_F32
        return apply_f32_clamp(res) if clamp else res

    # 2. Signed zeros
    if (s64_mod & ~SIGN_MASK_F64) == 0:
        res = NEG_ZERO_F32 if (s64_mod & SIGN_MASK_F64) else POS_ZERO_F32
        return apply_f32_clamp(res) if clamp else res

    # 3. Infinities
    if (s64_mod & ~SIGN_MASK_F64) == EXP_MASK_F64:
        res = NEG_INF_F32 if (s64_mod & SIGN_MASK_F64) else POS_INF_F32
        return apply_f32_clamp(res) if clamp else res

    # 4. Finite double-precision value conversion with roundTiesToEven
    d_val = bits_to_f64(s64_mod)
    try:
        res = struct.unpack("<I", struct.pack("<f", d_val))[0]
    except OverflowError:
        res = POS_INF_F32 if d_val > 0 else NEG_INF_F32

    return apply_f32_clamp(res) if clamp else res


def ref_cvt_f64_f32(
    src32: int,
    clamp: bool = False,
    neg: bool = False,
    abs_val: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_f64_f32.

    Converts 32-bit single-precision IEEE 754 value to 64-bit double-precision IEEE 754 value.
    This conversion is exact for all finite single-precision values.
    """
    s32_mod = apply_f32_modifiers(src32, neg=neg, abs_val=abs_val)

    # 1. NaN handling: canonical quiet NaN
    if is_nan_f32(s32_mod):
        res = QNAN_F64
        return apply_f64_clamp(res) if clamp else res

    # 2. Signed zeros
    if (s32_mod & ~SIGN_MASK_F32) == 0:
        res = NEG_ZERO_F64 if (s32_mod & SIGN_MASK_F32) else POS_ZERO_F64
        return apply_f64_clamp(res) if clamp else res

    # 3. Infinities
    if (s32_mod & ~SIGN_MASK_F32) == EXP_MASK_F32:
        res = NEG_INF_F64 if (s32_mod & SIGN_MASK_F32) else POS_INF_F64
        return apply_f64_clamp(res) if clamp else res

    # 4. Finite single-precision value conversion (exact)
    f_val = bits_to_f32(s32_mod)
    res = struct.unpack("<Q", struct.pack("<d", f_val))[0]

    return apply_f64_clamp(res) if clamp else res


class TestValuCvtF32F64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_basic_conversions(self) -> None:
        # 1.0f <-> 1.0
        self.assertEqual(ref_cvt_f32_f64(POS_ONE_F64), POS_ONE_F32)
        self.assertEqual(ref_cvt_f64_f32(POS_ONE_F32), POS_ONE_F64)

        # -1.0f <-> -1.0
        self.assertEqual(ref_cvt_f32_f64(NEG_ONE_F64), NEG_ONE_F32)
        self.assertEqual(ref_cvt_f64_f32(NEG_ONE_F32), NEG_ONE_F64)

        # 2.0f <-> 2.0
        self.assertEqual(ref_cvt_f32_f64(POS_TWO_F64), POS_TWO_F32)
        self.assertEqual(ref_cvt_f64_f32(POS_TWO_F32), POS_TWO_F64)

        # 0.5f <-> 0.5
        self.assertEqual(ref_cvt_f32_f64(HALF_F64), HALF_F32)
        self.assertEqual(ref_cvt_f64_f32(HALF_F32), HALF_F64)

        # 10.0f <-> 10.0
        self.assertEqual(ref_cvt_f32_f64(POS_TEN_F64), POS_TEN_F32)
        self.assertEqual(ref_cvt_f64_f32(POS_TEN_F32), POS_TEN_F64)

    def test_signed_zeros(self) -> None:
        self.assertEqual(ref_cvt_f32_f64(POS_ZERO_F64), POS_ZERO_F32)
        self.assertEqual(ref_cvt_f32_f64(NEG_ZERO_F64), NEG_ZERO_F32)
        self.assertEqual(ref_cvt_f64_f32(POS_ZERO_F32), POS_ZERO_F64)
        self.assertEqual(ref_cvt_f64_f32(NEG_ZERO_F32), NEG_ZERO_F64)

    def test_infinities(self) -> None:
        self.assertEqual(ref_cvt_f32_f64(POS_INF_F64), POS_INF_F32)
        self.assertEqual(ref_cvt_f32_f64(NEG_INF_F64), NEG_INF_F32)
        self.assertEqual(ref_cvt_f64_f32(POS_INF_F32), POS_INF_F64)
        self.assertEqual(ref_cvt_f64_f32(NEG_INF_F32), NEG_INF_F64)

    def test_nan_conversions(self) -> None:
        self.assertEqual(ref_cvt_f32_f64(QNAN_F64), QNAN_F32)
        self.assertEqual(ref_cvt_f32_f64(NEG_QNAN_F64), QNAN_F32)
        self.assertEqual(ref_cvt_f32_f64(SNAN_F64), QNAN_F32)
        self.assertEqual(ref_cvt_f64_f32(QNAN_F32), QNAN_F64)
        self.assertEqual(ref_cvt_f64_f32(NEG_QNAN_F32), QNAN_F64)
        self.assertEqual(ref_cvt_f64_f32(SNAN_F32), QNAN_F64)

    def test_modifiers(self) -> None:
        # abs modifier clears sign bit
        self.assertEqual(ref_cvt_f32_f64(NEG_ONE_F64, abs_val=True), POS_ONE_F32)
        self.assertEqual(ref_cvt_f64_f32(NEG_ONE_F32, abs_val=True), POS_ONE_F64)

        # neg modifier flips sign bit
        self.assertEqual(ref_cvt_f32_f64(POS_ONE_F64, neg=True), NEG_ONE_F32)
        self.assertEqual(ref_cvt_f64_f32(POS_ONE_F32, neg=True), NEG_ONE_F64)

        # neg(abs) forces negative
        self.assertEqual(ref_cvt_f32_f64(POS_ONE_F64, neg=True, abs_val=True), NEG_ONE_F32)
        self.assertEqual(ref_cvt_f64_f32(NEG_ONE_F32, neg=True, abs_val=True), NEG_ONE_F64)

        # clamp modifier
        self.assertEqual(ref_cvt_f32_f64(POS_TWO_F64, clamp=True), POS_ONE_F32)
        self.assertEqual(ref_cvt_f32_f64(NEG_TWO_F64, clamp=True), POS_ZERO_F32)
        self.assertEqual(ref_cvt_f64_f32(POS_TWO_F32, clamp=True), POS_ONE_F64)
        self.assertEqual(ref_cvt_f64_f32(NEG_TWO_F32, clamp=True), POS_ZERO_F64)

    def test_dword_utilities(self) -> None:
        lo, hi = split_dwords(POS_ONE_F64)
        self.assertEqual(lo, 0x00000000)
        self.assertEqual(hi, 0x3FF00000)
        self.assertEqual(join_dwords(lo, hi), POS_ONE_F64)


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    """Register file domain."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register operand representing a 32-bit register or 64-bit pair."""
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.index < 0 or self.index >= 256:
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")
        if self.count not in (1, 2):
            raise ValueError(f"Unsupported register count (must be 1 or 2): {self.count}")
        if self.count == 2 and self.index + 1 >= 256:
            raise ValueError(f"64-bit register pair exceeds register file boundary: {self.index}")

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == RegKind.SGPR

    def format_asm(self) -> str:
        if self.count == 2:
            return f"{self.kind.value}[{self.index}:{self.index + 1}]"
        return f"{self.kind.value}{self.index}"


@dataclasses.dataclass(frozen=True)
class Immediate:
    """Symbolic 32-bit or 64-bit immediate operand."""
    bits: int
    is_64bit: bool = False

    def __init__(self, val: int | float, is_64bit: bool = False) -> None:
        if isinstance(val, float):
            b = f64_to_bits(val) if is_64bit else f32_to_bits(val)
        else:
            b = (val & MASK64) if is_64bit else (val & MASK32)
        object.__setattr__(self, "bits", b)
        object.__setattr__(self, "is_64bit", is_64bit)

    @property
    def as_float(self) -> float:
        return bits_to_f64(self.bits) if self.is_64bit else bits_to_f32(self.bits)

    def format_asm(self) -> str:
        if self.is_64bit:
            return f"{self.bits:#018x}"
        return f"{self.bits:#010x}"


@dataclasses.dataclass(frozen=True)
class OperandModifiers:
    """Input modifiers for VALU floating-point instructions (abs and neg)."""
    neg: bool = False
    abs_val: bool = False

    def apply(self, bits: int, is_64bit: bool) -> int:
        if is_64bit:
            return apply_f64_modifiers(bits, neg=self.neg, abs_val=self.abs_val)
        return apply_f32_modifiers(bits, neg=self.neg, abs_val=self.abs_val)

    def format_asm(self, base_str: str) -> str:
        res = base_str
        if self.abs_val:
            res = f"|{res}|"
        if self.neg:
            res = f"-{res}"
        return res


@dataclasses.dataclass(frozen=True)
class InstructionModifiers:
    """Output modifiers for VALU instructions (clamp)."""
    clamp: bool = False

    def apply_f32(self, bits: int) -> int:
        return apply_f32_clamp(bits) if self.clamp else (bits & MASK32)

    def apply_f64(self, bits: int) -> int:
        return apply_f64_clamp(bits) if self.clamp else (bits & MASK64)

    def format_asm(self) -> str:
        return " clamp" if self.clamp else ""


class Opcode(enum.Enum):
    """VALU floating-point conversion opcodes."""
    V_CVT_F32_F64 = "v_cvt_f32_f64"
    V_CVT_F64_F32 = "v_cvt_f64_f32"


class Encoding(enum.Enum):
    """Instruction encoding format."""
    VOP1 = "e32"
    VOP3 = "e64"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a floating-point conversion VALU instruction."""
    opcode: Opcode
    dst: Register
    src0: Register | Immediate
    src0_mods: OperandModifiers = dataclasses.field(default_factory=OperandModifiers)
    inst_mods: InstructionModifiers = dataclasses.field(default_factory=InstructionModifiers)
    encoding: Encoding = Encoding.VOP3

    def __post_init__(self) -> None:
        if not self.dst.is_vgpr:
            raise ValueError(f"VALU instruction destination must be a VGPR: {self.dst}")

        if self.opcode == Opcode.V_CVT_F32_F64:
            if self.dst.count != 1:
                raise ValueError(f"v_cvt_f32_f64 destination must be 32-bit single VGPR (count=1): {self.dst}")
            if isinstance(self.src0, Register) and self.src0.count != 2:
                raise ValueError(f"v_cvt_f32_f64 source register must be 64-bit pair (count=2): {self.src0}")
        elif self.opcode == Opcode.V_CVT_F64_F32:
            if self.dst.count != 2:
                raise ValueError(f"v_cvt_f64_f32 destination must be 64-bit pair VGPR (count=2): {self.dst}")
            if isinstance(self.src0, Register) and self.src0.count != 1:
                raise ValueError(f"v_cvt_f64_f32 source register must be 32-bit single register (count=1): {self.src0}")

        has_mods = self.src0_mods.neg or self.src0_mods.abs_val or self.inst_mods.clamp
        if self.encoding == Encoding.VOP1 and has_mods:
            raise ValueError("VOP1 (e32) encoding does not support input modifiers or clamp (requires VOP3/e64)")

    def to_assembly(self) -> str:
        s0_str = self.src0.format_asm()
        s0_fmt = self.src0_mods.format_asm(s0_str)
        dst_str = self.dst.format_asm()
        clamp_str = self.inst_mods.format_asm()
        enc_suffix = f"_{self.encoding.value}" if self.encoding == Encoding.VOP1 else ""
        return f"{self.opcode.value}{enc_suffix} {dst_str}, {s0_fmt}{clamp_str}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of VALU instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


class TestValuCvtF32F64AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_formatting(self) -> None:
        v0 = Register(RegKind.VGPR, 0, count=1)
        v_pair = Register(RegKind.VGPR, 2, count=2)
        s_single = Register(RegKind.SGPR, 4, count=1)
        s_pair = Register(RegKind.SGPR, 6, count=2)

        self.assertEqual(v0.format_asm(), "v0")
        self.assertEqual(v_pair.format_asm(), "v[2:3]")
        self.assertEqual(s_single.format_asm(), "s4")
        self.assertEqual(s_pair.format_asm(), "s[6:7]")

    def test_immediate_formatting(self) -> None:
        imm32 = Immediate(1.0, is_64bit=False)
        imm64 = Immediate(1.0, is_64bit=True)
        self.assertEqual(imm32.format_asm(), f"{POS_ONE_F32:#010x}")
        self.assertEqual(imm64.format_asm(), f"{POS_ONE_F64:#018x}")

    def test_operand_modifiers_formatting(self) -> None:
        mods_abs = OperandModifiers(abs_val=True)
        mods_neg = OperandModifiers(neg=True)
        mods_both = OperandModifiers(neg=True, abs_val=True)

        self.assertEqual(mods_abs.format_asm("v[2:3]"), "|v[2:3]|")
        self.assertEqual(mods_neg.format_asm("v[2:3]"), "-v[2:3]")
        self.assertEqual(mods_both.format_asm("v[2:3]"), "-|v[2:3]|")

    def test_instruction_formatting(self) -> None:
        # v_cvt_f32_f64 VOP3
        inst1 = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
            src0_mods=OperandModifiers(neg=True, abs_val=True),
            inst_mods=InstructionModifiers(clamp=True),
        )
        self.assertEqual(inst1.to_assembly(), "v_cvt_f32_f64 v0, -|v[2:3]| clamp")

        # v_cvt_f32_f64 VOP1
        inst1_e32 = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
            encoding=Encoding.VOP1,
        )
        self.assertEqual(inst1_e32.to_assembly(), "v_cvt_f32_f64_e32 v0, v[2:3]")

        # v_cvt_f64_f32 VOP3
        inst2 = Instruction(
            opcode=Opcode.V_CVT_F64_F32,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2, count=1),
            inst_mods=InstructionModifiers(clamp=True),
        )
        self.assertEqual(inst2.to_assembly(), "v_cvt_f64_f32 v[0:1], v2 clamp")

        # v_cvt_f64_f32 VOP1
        inst2_e32 = Instruction(
            opcode=Opcode.V_CVT_F64_F32,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2, count=1),
            encoding=Encoding.VOP1,
        )
        self.assertEqual(inst2_e32.to_assembly(), "v_cvt_f64_f32_e32 v[0:1], v2")

    def test_instruction_validation(self) -> None:
        # Destination must be VGPR
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_F32_F64,
                dst=Register(RegKind.SGPR, 0, count=1),
                src0=Register(RegKind.VGPR, 2, count=2),
            )

        # v_cvt_f32_f64 dst must be count=1
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_F32_F64,
                dst=Register(RegKind.VGPR, 0, count=2),
                src0=Register(RegKind.VGPR, 2, count=2),
            )

        # v_cvt_f32_f64 src0 must be count=2
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_F32_F64,
                dst=Register(RegKind.VGPR, 0, count=1),
                src0=Register(RegKind.VGPR, 2, count=1),
            )

        # v_cvt_f64_f32 dst must be count=2
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_F64_F32,
                dst=Register(RegKind.VGPR, 0, count=1),
                src0=Register(RegKind.VGPR, 2, count=1),
            )

        # v_cvt_f64_f32 src0 must be count=1
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_F64_F32,
                dst=Register(RegKind.VGPR, 0, count=2),
                src0=Register(RegKind.VGPR, 2, count=2),
            )

        # VOP1 rejects modifiers
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_F32_F64,
                dst=Register(RegKind.VGPR, 0, count=1),
                src0=Register(RegKind.VGPR, 2, count=2),
                src0_mods=OperandModifiers(neg=True),
                encoding=Encoding.VOP1,
            )

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
        ))
        prog.append(Instruction(
            opcode=Opcode.V_CVT_F64_F32,
            dst=Register(RegKind.VGPR, 4, count=2),
            src0=Register(RegKind.VGPR, 0, count=1),
            inst_mods=InstructionModifiers(clamp=True),
        ))
        expected = (
            "v_cvt_f32_f64 v0, v[2:3]\n"
            "v_cvt_f64_f32 v[4:5], v0 clamp"
        )
        self.assertEqual(prog.to_assembly(), expected)


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarVALUInterpreter:
    """Deterministic scalar interpreter for GFX10/RDNA2 VALU conversion instructions."""

    def __init__(self) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.vcc: int = 0
        self.trace: list[str] = []

    def read_reg_u32(self, kind: RegKind, idx: int) -> int:
        if idx < 0 or (kind == RegKind.VGPR and idx >= 256) or (kind == RegKind.SGPR and idx >= 106):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        return (self.vgprs[idx] if kind == RegKind.VGPR else self.sgprs[idx]) & MASK32

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
        if idx < 0 or (kind == RegKind.VGPR and idx >= 256) or (kind == RegKind.SGPR and idx >= 106):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[idx] = val32
        else:
            self.sgprs[idx] = val32

    def read_reg_u64(self, kind: RegKind, idx: int) -> int:
        lo = self.read_reg_u32(kind, idx)
        hi = self.read_reg_u32(kind, idx + 1)
        return join_dwords(lo, hi)

    def write_reg_u64(self, kind: RegKind, idx: int, val: int) -> None:
        lo, hi = split_dwords(val)
        self.write_reg_u32(kind, idx, lo)
        self.write_reg_u32(kind, idx + 1, hi)

    def read_operand(self, op: Register | Immediate, mods: OperandModifiers, is_64bit: bool) -> int:
        if isinstance(op, Immediate):
            raw = op.bits
        elif isinstance(op, Register):
            raw = self.read_reg_u64(op.kind, op.index) if is_64bit else self.read_reg_u32(op.kind, op.index)
        else:
            raise TypeError(f"Unsupported operand type: {type(op)}")
        return mods.apply(raw, is_64bit)

    def step(self, inst: Instruction) -> int:
        """Executes a single VALU conversion instruction deterministically."""
        # Read-before-write semantics: latch operand
        if inst.opcode == Opcode.V_CVT_F32_F64:
            s0 = self.read_operand(inst.src0, inst.src0_mods, is_64bit=True)
            res = ref_cvt_f32_f64(s0)
            final_res = inst.inst_mods.apply_f32(res)
            self.write_reg_u32(inst.dst.kind, inst.dst.index, final_res)
            self.trace.append(f"{inst.to_assembly()} => {final_res:#010x}")
            return final_res
        elif inst.opcode == Opcode.V_CVT_F64_F32:
            s0 = self.read_operand(inst.src0, inst.src0_mods, is_64bit=False)
            res = ref_cvt_f64_f32(s0)
            final_res = inst.inst_mods.apply_f64(res)
            self.write_reg_u64(inst.dst.kind, inst.dst.index, final_res)
            self.trace.append(f"{inst.to_assembly()} => {final_res:#018x}")
            return final_res
        else:
            raise NotImplementedError(f"Unsupported opcode in scalar interpreter: {inst.opcode}")

    def run_program(self, prog: Program) -> list[int]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


class TestValuCvtF32F64ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar VALU instruction interpreter."""

    def test_scalar_register_read_write(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 0, POS_ONE_F32)
        interp.write_reg_u64(RegKind.VGPR, 2, POS_TWO_F64)
        interp.write_reg_u64(RegKind.SGPR, 4, NEG_ONE_F64)

        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), POS_ONE_F32)
        self.assertEqual(interp.read_reg_u64(RegKind.VGPR, 2), POS_TWO_F64)
        self.assertEqual(interp.read_reg_u64(RegKind.SGPR, 4), NEG_ONE_F64)

    def test_scalar_cvt_f32_f64_vgpr(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u64(RegKind.VGPR, 2, POS_TEN_F64)

        inst = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
        )
        res = interp.step(inst)
        self.assertEqual(res, POS_TEN_F32)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), POS_TEN_F32)

    def test_scalar_cvt_f32_f64_sgpr(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u64(RegKind.SGPR, 4, NEG_HALF_F64)

        inst = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.SGPR, 4, count=2),
        )
        res = interp.step(inst)
        self.assertEqual(res, NEG_HALF_F32)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), NEG_HALF_F32)

    def test_scalar_cvt_f64_f32_vgpr(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 2, POS_TEN_F32)

        inst = Instruction(
            opcode=Opcode.V_CVT_F64_F32,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2, count=1),
        )
        res = interp.step(inst)
        self.assertEqual(res, POS_TEN_F64)
        self.assertEqual(interp.read_reg_u64(RegKind.VGPR, 0), POS_TEN_F64)

    def test_scalar_cvt_f64_f32_sgpr(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.SGPR, 4, NEG_HALF_F32)

        inst = Instruction(
            opcode=Opcode.V_CVT_F64_F32,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.SGPR, 4, count=1),
        )
        res = interp.step(inst)
        self.assertEqual(res, NEG_HALF_F64)
        self.assertEqual(interp.read_reg_u64(RegKind.VGPR, 0), NEG_HALF_F64)

    def test_scalar_immediate_operands(self) -> None:
        interp = ScalarVALUInterpreter()

        # Immediate f64 -> f32
        inst1 = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Immediate(2.0, is_64bit=True),
        )
        res1 = interp.step(inst1)
        self.assertEqual(res1, POS_TWO_F32)

        # Immediate f32 -> f64
        inst2 = Instruction(
            opcode=Opcode.V_CVT_F64_F32,
            dst=Register(RegKind.VGPR, 2, count=2),
            src0=Immediate(2.0, is_64bit=False),
        )
        res2 = interp.step(inst2)
        self.assertEqual(res2, POS_TWO_F64)

    def test_scalar_modifiers_abs_neg_clamp(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u64(RegKind.VGPR, 2, NEG_TWO_F64)

        # v_cvt_f32_f64 with abs and clamp: |-2.0| = 2.0 clamped to [0,1] = 1.0f
        inst1 = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
            src0_mods=OperandModifiers(abs_val=True),
            inst_mods=InstructionModifiers(clamp=True),
        )
        res1 = interp.step(inst1)
        self.assertEqual(res1, POS_ONE_F32)

        # v_cvt_f64_f32 with neg and clamp: -2.0f clamped to [0,1] = +0.0
        interp.write_reg_u32(RegKind.VGPR, 4, POS_TWO_F32)
        inst2 = Instruction(
            opcode=Opcode.V_CVT_F64_F32,
            dst=Register(RegKind.VGPR, 6, count=2),
            src0=Register(RegKind.VGPR, 4, count=1),
            src0_mods=OperandModifiers(neg=True),
            inst_mods=InstructionModifiers(clamp=True),
        )
        res2 = interp.step(inst2)
        self.assertEqual(res2, POS_ZERO_F64)

    def test_scalar_program_round_trip(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 2, POS_TEN_F32)

        prog = Program()
        # v[4:5] = (double)v2
        prog.append(Instruction(
            opcode=Opcode.V_CVT_F64_F32,
            dst=Register(RegKind.VGPR, 4, count=2),
            src0=Register(RegKind.VGPR, 2, count=1),
        ))
        # v6 = (float)v[4:5]
        prog.append(Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 6, count=1),
            src0=Register(RegKind.VGPR, 4, count=2),
        ))

        results = interp.run_program(prog)
        self.assertEqual(results, [POS_TEN_F64, POS_TEN_F32])
        self.assertEqual(interp.read_reg_u64(RegKind.VGPR, 4), POS_TEN_F64)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 6), POS_TEN_F32)
        self.assertEqual(len(interp.trace), 2)


# ---------------------------------------------------------------------------
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ---------------------------------------------------------------------------

class SIMDWavefrontInterpreter:
    """Multi-lane SIMD interpreter modeling Wave32 and Wave64 execution on RDNA2."""

    def __init__(self, wave_size: int = 64) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size} (must be 32 or 64)")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        # Per-lane VGPR files: [wave_size][256] 32-bit registers
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        # Wavefront-shared SGPR file: [106] 32-bit registers
        self.sgprs: list[int] = [0] * 106
        self.vcc: int = 0

    def set_exec_mask(self, mask: int) -> None:
        valid_mask = (1 << self.wave_size) - 1
        self.exec_mask = mask & valid_mask

    def read_lane_reg_u32(self, lane: int, kind: RegKind, idx: int) -> int:
        if kind == RegKind.VGPR:
            return self.vgprs[lane][idx] & MASK32
        return self.sgprs[idx] & MASK32

    def write_lane_reg_u32(self, lane: int, kind: RegKind, idx: int, val: int) -> None:
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[lane][idx] = val32
        else:
            self.sgprs[idx] = val32

    def read_lane_reg_u64(self, lane: int, kind: RegKind, idx: int) -> int:
        lo = self.read_lane_reg_u32(lane, kind, idx)
        hi = self.read_lane_reg_u32(lane, kind, idx + 1)
        return join_dwords(lo, hi)

    def write_lane_reg_u64(self, lane: int, kind: RegKind, idx: int, val: int) -> None:
        lo, hi = split_dwords(val)
        self.write_lane_reg_u32(lane, kind, idx, lo)
        self.write_lane_reg_u32(lane, kind, idx + 1, hi)

    def read_lane_operand(self, lane: int, op: Register | Immediate, mods: OperandModifiers, is_64bit: bool) -> int:
        if isinstance(op, Immediate):
            raw = op.bits
        elif isinstance(op, Register):
            raw = self.read_lane_reg_u64(lane, op.kind, op.index) if is_64bit else self.read_lane_reg_u32(lane, op.kind, op.index)
        else:
            raise TypeError(f"Unsupported operand type: {type(op)}")
        return mods.apply(raw, is_64bit)

    def step(self, inst: Instruction) -> list[int | None]:
        """Executes a conversion instruction across all wavefront lanes under EXEC mask."""
        is_64 = (inst.opcode == Opcode.V_CVT_F32_F64)

        # 1. Latch all inputs before write-back (read-before-write across wavefront)
        latched_inputs: list[int] = []
        for lane in range(self.wave_size):
            s0 = self.read_lane_operand(lane, inst.src0, inst.src0_mods, is_64bit=is_64)
            latched_inputs.append(s0)

        # 2. Compute and write back strictly for active lanes
        lane_results: list[int | None] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append(None)
                continue

            s0 = latched_inputs[lane]
            if inst.opcode == Opcode.V_CVT_F32_F64:
                res = ref_cvt_f32_f64(s0)
                final_res = inst.inst_mods.apply_f32(res)
                self.write_lane_reg_u32(lane, inst.dst.kind, inst.dst.index, final_res)
            elif inst.opcode == Opcode.V_CVT_F64_F32:
                res = ref_cvt_f64_f32(s0)
                final_res = inst.inst_mods.apply_f64(res)
                self.write_lane_reg_u64(lane, inst.dst.kind, inst.dst.index, final_res)
            else:
                raise NotImplementedError(f"Unsupported opcode in SIMD interpreter: {inst.opcode}")

            lane_results.append(final_res)

        return lane_results


class TestValuCvtF32F64SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_execution(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        inst = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
        )
        for lane in range(32):
            simd.write_lane_reg_u64(lane, RegKind.VGPR, 2, POS_TWO_F64)

        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertEqual(results[lane], POS_TWO_F32)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), POS_TWO_F32)

    def test_wave64_multi_lane_divergence(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=64)
        inst_f64_f32 = Instruction(
            opcode=Opcode.V_CVT_F64_F32,
            dst=Register(RegKind.VGPR, 4, count=2),
            src0=Register(RegKind.VGPR, 2, count=1),
        )
        expected: list[int] = []
        for lane in range(64):
            f_val = f32_to_bits(float(lane) * 1.5)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, f_val)
            expected.append(ref_cvt_f64_f32(f_val))

        results = simd.step(inst_f64_f32)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            self.assertEqual(results[lane], expected[lane])
            self.assertEqual(simd.read_lane_reg_u64(lane, RegKind.VGPR, 4), expected[lane])

    def test_wave32_exec_masking_partial(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        # Checkerboard EXEC mask (even lanes active)
        exec_mask = 0x55555555
        simd.set_exec_mask(exec_mask)

        canary_u32 = 0xDEADBEEF
        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, canary_u32)
            simd.write_lane_reg_u64(lane, RegKind.VGPR, 2, POS_TEN_F64)

        inst = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
        )
        results = simd.step(inst)

        for lane in range(32):
            if (exec_mask >> lane) & 1:
                self.assertEqual(results[lane], POS_TEN_F32)
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), POS_TEN_F32)
            else:
                self.assertIsNone(results[lane])
                # Inactive lanes strictly retain prior canary value
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), canary_u32)

    def test_wave64_exec_masking_all_inactive(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=64)
        simd.set_exec_mask(0)

        canary_u64 = 0xBAADF00DCAFE1234
        for lane in range(64):
            simd.write_lane_reg_u64(lane, RegKind.VGPR, 0, canary_u64)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, POS_TWO_F32)

        inst = Instruction(
            opcode=Opcode.V_CVT_F64_F32,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2, count=1),
        )
        results = simd.step(inst)
        self.assertEqual(results, [None] * 64)
        for lane in range(64):
            self.assertEqual(simd.read_lane_reg_u64(lane, RegKind.VGPR, 0), canary_u64)

    def test_simd_with_shared_sgpr_scalar_broadcast(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        # Shared SGPR pair s[4:5] holds double 10.0
        simd.write_lane_reg_u64(0, RegKind.SGPR, 4, POS_TEN_F64)

        inst = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.SGPR, 4, count=2),
        )
        results = simd.step(inst)
        for lane in range(32):
            self.assertEqual(results[lane], POS_TEN_F32)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), POS_TEN_F32)


# ---------------------------------------------------------------------------
# Register Aliasing and In-Place Operand Preservation
# ---------------------------------------------------------------------------

class TestValuCvtF32F64RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_aliasing_cvt_f32_f64_dst_equals_src_low_dword(self) -> None:
        interp = ScalarVALUInterpreter()
        # v[0:1] = 2.0 (lo=0, hi=0x40000000)
        interp.write_reg_u64(RegKind.VGPR, 0, POS_TWO_F64)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), 0x40000000)

        # v0 = (float)v[0:1]
        inst = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 0, count=2),
        )
        res = interp.step(inst)
        self.assertEqual(res, POS_TWO_F32)
        # Low dword v0 now holds f32 2.0f
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), POS_TWO_F32)
        # High dword v1 remains undisturbed
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), 0x40000000)

    def test_aliasing_cvt_f32_f64_dst_equals_src_high_dword(self) -> None:
        interp = ScalarVALUInterpreter()
        # v[0:1] = 10.0 (lo=0, hi=0x40240000)
        interp.write_reg_u64(RegKind.VGPR, 0, POS_TEN_F64)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), 0x40240000)

        # v1 = (float)v[0:1]
        inst = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 1, count=1),
            src0=Register(RegKind.VGPR, 0, count=2),
        )
        res = interp.step(inst)
        self.assertEqual(res, POS_TEN_F32)
        # High dword v1 now holds f32 10.0f
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), POS_TEN_F32)
        # Low dword v0 remains undisturbed
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0)

    def test_aliasing_cvt_f64_f32_dst_low_equals_src(self) -> None:
        interp = ScalarVALUInterpreter()
        # v0 = 2.0f, v1 = canary
        interp.write_reg_u32(RegKind.VGPR, 0, POS_TWO_F32)
        interp.write_reg_u32(RegKind.VGPR, 1, 0xAAAAAAAA)

        # v[0:1] = (double)v0
        inst = Instruction(
            opcode=Opcode.V_CVT_F64_F32,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 0, count=1),
        )
        res = interp.step(inst)
        self.assertEqual(res, POS_TWO_F64)
        self.assertEqual(interp.read_reg_u64(RegKind.VGPR, 0), POS_TWO_F64)

    def test_aliasing_cvt_f64_f32_dst_high_equals_src(self) -> None:
        interp = ScalarVALUInterpreter()
        # v1 = 1.0f, v0 = canary
        interp.write_reg_u32(RegKind.VGPR, 1, POS_ONE_F32)
        interp.write_reg_u32(RegKind.VGPR, 0, 0xBBBBBBBB)

        # v[0:1] = (double)v1
        inst = Instruction(
            opcode=Opcode.V_CVT_F64_F32,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 1, count=1),
        )
        res = interp.step(inst)
        self.assertEqual(res, POS_ONE_F64)
        self.assertEqual(interp.read_reg_u64(RegKind.VGPR, 0), POS_ONE_F64)

    def test_unmodified_registers_preserved(self) -> None:
        interp = ScalarVALUInterpreter()
        for idx in range(10, 20):
            interp.write_reg_u32(RegKind.VGPR, idx, 0x22220000 + idx)

        inst = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Immediate(1.0, is_64bit=True),
        )
        interp.step(inst)

        for idx in range(10, 20):
            self.assertEqual(interp.read_reg_u32(RegKind.VGPR, idx), 0x22220000 + idx)

    def test_simd_aliasing_with_divergent_exec_mask(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        exec_mask = 0x0000000F  # First 4 lanes active
        simd.set_exec_mask(exec_mask)

        # All lanes: v[0:1] = 10.0
        for lane in range(32):
            simd.write_lane_reg_u64(lane, RegKind.VGPR, 0, POS_TEN_F64)

        # v0 = (float)v[0:1]
        inst = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 0, count=2),
        )
        simd.step(inst)

        for lane in range(32):
            if lane < 4:
                # Active lane wrote f32 10.0f to v0
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), POS_TEN_F32)
            else:
                # Inactive lane preserved low dword of POS_TEN_F64 (which is 0)
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 0)
            # High dword v1 remains undisturbed in all lanes
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 1), 0x40240000)


# ---------------------------------------------------------------------------
# Condition Codes, Status Flags, and Carry/Borrow Isolation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ConditionCodes:
    """Architectural condition codes for floating-point conversion operations."""
    equal: bool
    less: bool
    greater: bool
    unordered: bool
    zero: bool
    negative: bool


@dataclasses.dataclass(frozen=True)
class FPStatusFlags:
    """IEEE 754 floating-point status flags for conversion operations."""
    invalid: bool       # e.g., sNaN operand
    denormal: bool      # e.g., subnormal operand or result
    inexact: bool       # precision loss in conversion
    overflow: bool      # exponent overflow to infinity
    underflow: bool     # exponent underflow


def compute_condition_codes_f32(result: int) -> ConditionCodes:
    """Computes condition codes evaluating attributes of a 32-bit floating-point result."""
    result &= MASK32
    if is_nan_f32(result):
        return ConditionCodes(
            equal=False,
            less=False,
            greater=False,
            unordered=True,
            zero=False,
            negative=False,
        )
    val = bits_to_f32(result)
    is_zero = (result in (POS_ZERO_F32, NEG_ZERO_F32))
    is_neg = bool(result & SIGN_MASK_F32)
    return ConditionCodes(
        equal=(val == 0.0),
        less=(val < 0.0),
        greater=(val > 0.0),
        unordered=False,
        zero=is_zero,
        negative=is_neg,
    )


def compute_condition_codes_f64(result: int) -> ConditionCodes:
    """Computes condition codes evaluating attributes of a 64-bit floating-point result."""
    result &= MASK64
    if is_nan_f64(result):
        return ConditionCodes(
            equal=False,
            less=False,
            greater=False,
            unordered=True,
            zero=False,
            negative=False,
        )
    val = bits_to_f64(result)
    is_zero = (result in (POS_ZERO_F64, NEG_ZERO_F64))
    is_neg = bool(result & SIGN_MASK_F64)
    return ConditionCodes(
        equal=(val == 0.0),
        less=(val < 0.0),
        greater=(val > 0.0),
        unordered=False,
        zero=is_zero,
        negative=is_neg,
    )


def compute_fp_status_flags_cvt_f32_f64(s64: int, result32: int) -> FPStatusFlags:
    """Computes IEEE 754 status flags for double to single float conversion."""
    s64 &= MASK64
    result32 &= MASK32

    invalid = is_snan_f64(s64)
    denormal = is_subnormal_f64(s64) or is_subnormal_f32(result32)

    is_s64_inf = (s64 & ~SIGN_MASK_F64) == EXP_MASK_F64
    is_res_inf = (result32 & ~SIGN_MASK_F32) == EXP_MASK_F32
    overflow = not is_nan_f64(s64) and not is_s64_inf and is_res_inf

    is_s64_zero = (s64 & ~SIGN_MASK_F64) == 0
    is_res_zero = (result32 & ~SIGN_MASK_F32) == 0
    underflow = not is_nan_f64(s64) and not is_s64_zero and (is_subnormal_f32(result32) or is_res_zero)

    if is_nan_f64(s64):
        inexact = False
    elif overflow or underflow:
        inexact = True
    else:
        d_val = bits_to_f64(s64)
        f_val = float(bits_to_f32(result32))
        inexact = (d_val != f_val)

    return FPStatusFlags(
        invalid=invalid,
        denormal=denormal,
        inexact=inexact,
        overflow=overflow,
        underflow=underflow,
    )


def compute_fp_status_flags_cvt_f64_f32(s32: int, result64: int) -> FPStatusFlags:
    """Computes IEEE 754 status flags for single to double float conversion."""
    s32 &= MASK32
    result64 &= MASK64

    invalid = is_snan_f32(s32)
    # f32 subnormal converted to f64 is a denormal input operand
    denormal = is_subnormal_f32(s32)

    return FPStatusFlags(
        invalid=invalid,
        denormal=denormal,
        inexact=False,
        overflow=False,
        underflow=False,
    )


class TestValuCvtF32F64ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow isolation."""

    def test_condition_codes_f32_and_f64(self) -> None:
        cc_pos = compute_condition_codes_f32(POS_ONE_F32)
        self.assertTrue(cc_pos.greater)
        self.assertFalse(cc_pos.less)
        self.assertFalse(cc_pos.zero)
        self.assertFalse(cc_pos.negative)
        self.assertFalse(cc_pos.unordered)

        cc_neg = compute_condition_codes_f64(NEG_TWO_F64)
        self.assertTrue(cc_neg.less)
        self.assertFalse(cc_neg.greater)
        self.assertTrue(cc_neg.negative)
        self.assertFalse(cc_neg.unordered)

        cc_zero = compute_condition_codes_f32(NEG_ZERO_F32)
        self.assertTrue(cc_zero.zero)
        self.assertTrue(cc_zero.equal)
        self.assertTrue(cc_zero.negative)
        self.assertFalse(cc_zero.unordered)

        cc_nan = compute_condition_codes_f64(QNAN_F64)
        self.assertTrue(cc_nan.unordered)
        self.assertFalse(cc_nan.zero)
        self.assertFalse(cc_nan.equal)

    def test_status_flags_snan_invalid(self) -> None:
        flags_f32_f64 = compute_fp_status_flags_cvt_f32_f64(SNAN_F64, QNAN_F32)
        self.assertTrue(flags_f32_f64.invalid)

        flags_f64_f32 = compute_fp_status_flags_cvt_f64_f32(SNAN_F32, QNAN_F64)
        self.assertTrue(flags_f64_f32.invalid)

        # Quiet NaN does not trigger invalid flag
        flags_qnan = compute_fp_status_flags_cvt_f32_f64(QNAN_F64, QNAN_F32)
        self.assertFalse(flags_qnan.invalid)

    def test_status_flags_denormal(self) -> None:
        # f32 subnormal converted to f64 normal flags denormal operand
        flags = compute_fp_status_flags_cvt_f64_f32(SUBNORM_MIN_F32, 0x36A0000000000000)
        self.assertTrue(flags.denormal)
        self.assertFalse(flags.invalid)
        self.assertFalse(flags.inexact)

    def test_status_flags_overflow_inexact(self) -> None:
        # 1e39 exceeds float32 max (~3.4e38), overflowing to +infinity
        d_large = f64_to_bits(1e39)
        res32 = ref_cvt_f32_f64(d_large)
        self.assertEqual(res32, POS_INF_F32)

        flags = compute_fp_status_flags_cvt_f32_f64(d_large, res32)
        self.assertTrue(flags.overflow)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.invalid)

    def test_status_flags_f32_to_f64_exact(self) -> None:
        # Any finite float32 converts to float64 with zero precision loss
        res64 = ref_cvt_f64_f32(POS_ONE_F32)
        flags = compute_fp_status_flags_cvt_f64_f32(POS_ONE_F32, res64)
        self.assertFalse(flags.inexact)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.underflow)

    def test_carry_borrow_vcc_isolation(self) -> None:
        # In RDNA2 ISA, VALU floating-point conversion instructions never write or disturb VCC
        interp = ScalarVALUInterpreter()
        canary_vcc = 0xAAAAAAAAAAAAAAAA
        interp.vcc = canary_vcc

        inst = Instruction(
            opcode=Opcode.V_CVT_F32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Immediate(1.0, is_64bit=True),
        )
        interp.step(inst)
        self.assertEqual(interp.vcc, canary_vcc)

        simd = SIMDWavefrontInterpreter(wave_size=64)
        simd.vcc = canary_vcc
        simd.step(inst)
        self.assertEqual(simd.vcc, canary_vcc)


# ---------------------------------------------------------------------------
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ---------------------------------------------------------------------------

class TestValuCvtF32F64IEEE754Boundaries(unittest.TestCase):
    """Exhaustive unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_signed_zero_matrix(self) -> None:
        # v_cvt_f32_f64 signed zeros
        self.assertEqual(ref_cvt_f32_f64(POS_ZERO_F64), POS_ZERO_F32)
        self.assertEqual(ref_cvt_f32_f64(NEG_ZERO_F64), NEG_ZERO_F32)
        self.assertEqual(ref_cvt_f32_f64(POS_ZERO_F64) & SIGN_MASK_F32, 0)
        self.assertEqual(ref_cvt_f32_f64(NEG_ZERO_F64) & SIGN_MASK_F32, SIGN_MASK_F32)

        # v_cvt_f64_f32 signed zeros
        self.assertEqual(ref_cvt_f64_f32(POS_ZERO_F32), POS_ZERO_F64)
        self.assertEqual(ref_cvt_f64_f32(NEG_ZERO_F32), NEG_ZERO_F64)
        self.assertEqual(ref_cvt_f64_f32(POS_ZERO_F32) & SIGN_MASK_F64, 0)
        self.assertEqual(ref_cvt_f64_f32(NEG_ZERO_F32) & SIGN_MASK_F64, SIGN_MASK_F64)

    def test_infinities_matrix(self) -> None:
        self.assertEqual(ref_cvt_f32_f64(POS_INF_F64), POS_INF_F32)
        self.assertEqual(ref_cvt_f32_f64(NEG_INF_F64), NEG_INF_F32)
        self.assertEqual(ref_cvt_f64_f32(POS_INF_F32), POS_INF_F64)
        self.assertEqual(ref_cvt_f64_f32(NEG_INF_F32), NEG_INF_F64)

    def test_subnormals_f32_to_f64_exact(self) -> None:
        # All subnormals in float32 convert to exact normalized floats in float64
        subnormals_f32 = [
            SUBNORM_MIN_F32,
            SUBNORM_MAX_F32,
            NEG_SUBNORM_MIN_F32,
            NEG_SUBNORM_MAX_F32,
            0x00000042,
            0x80000042,
        ]
        for sub in subnormals_f32:
            with self.subTest(sub=f"{sub:#010x}"):
                self.assertTrue(is_subnormal_f32(sub))
                res64 = ref_cvt_f64_f32(sub)
                # In float64, these exponent bits are non-zero (normalized!)
                self.assertFalse(is_subnormal_f64(res64))
                self.assertEqual(bits_to_f32(sub), bits_to_f64(res64))

    def test_subnormals_f64_to_f32(self) -> None:
        # A double representing 2^-140 converts to a subnormal float32
        d_val = 2.0 ** -140
        d_bits = f64_to_bits(d_val)
        res32 = ref_cvt_f32_f64(d_bits)
        self.assertTrue(is_subnormal_f32(res32))

        # A double subnormal (magnitude < 2^-1022) is far smaller than f32 min subnormal (2^-149)
        # and therefore underflows to signed zero under IEEE 754 roundTiesToEven
        self.assertEqual(ref_cvt_f32_f64(SUBNORM_MIN_F64), POS_ZERO_F32)
        self.assertEqual(ref_cvt_f32_f64(SUBNORM_MAX_F64), POS_ZERO_F32)
        self.assertEqual(ref_cvt_f32_f64(NEG_SUBNORM_MIN_F64), NEG_ZERO_F32)
        self.assertEqual(ref_cvt_f32_f64(NEG_SUBNORM_MAX_F64), NEG_ZERO_F32)

    def test_overflow_boundaries_f64_to_f32(self) -> None:
        # FLT_MAX as double
        flt_max_as_double = f64_to_bits(bits_to_f32(FLT_MAX))
        self.assertEqual(ref_cvt_f32_f64(flt_max_as_double), FLT_MAX)

        # Value just below overflow threshold rounds to FLT_MAX
        val_below_thresh = f64_to_bits(3.4028234e+38)
        self.assertEqual(ref_cvt_f32_f64(val_below_thresh), FLT_MAX)

        # Value >= overflow threshold rounds to +inf
        val_at_thresh = f64_to_bits(OVERFLOW_THRESHOLD_F64)
        self.assertEqual(ref_cvt_f32_f64(val_at_thresh), POS_INF_F32)

        # Negative counterpart rounds to -inf
        val_neg_thresh = f64_to_bits(-OVERFLOW_THRESHOLD_F64)
        self.assertEqual(ref_cvt_f32_f64(val_neg_thresh), NEG_INF_F32)

    def test_nan_suppression_and_propagation(self) -> None:
        f64_nans = [QNAN_F64, NEG_QNAN_F64, QNAN_PAYLOAD_A_F64, QNAN_PAYLOAD_B_F64, SNAN_F64]
        for nan in f64_nans:
            with self.subTest(nan=f"{nan:#018x}"):
                res32 = ref_cvt_f32_f64(nan)
                self.assertTrue(is_nan_f32(res32))
                self.assertTrue(is_qnan_f32(res32))

        f32_nans = [QNAN_F32, NEG_QNAN_F32, QNAN_PAYLOAD_A_F32, QNAN_PAYLOAD_B_F32, SNAN_F32]
        for nan in f32_nans:
            with self.subTest(nan=f"{nan:#010x}"):
                res64 = ref_cvt_f64_f32(nan)
                self.assertTrue(is_nan_f64(res64))
                self.assertTrue(is_qnan_f64(res64))

    def test_modifiers_on_boundary_values(self) -> None:
        # abs on negative infinity and negative zero
        self.assertEqual(ref_cvt_f32_f64(NEG_INF_F64, abs_val=True), POS_INF_F32)
        self.assertEqual(ref_cvt_f32_f64(NEG_ZERO_F64, abs_val=True), POS_ZERO_F32)
        self.assertEqual(ref_cvt_f64_f32(NEG_INF_F32, abs_val=True), POS_INF_F64)
        self.assertEqual(ref_cvt_f64_f32(NEG_ZERO_F32, abs_val=True), POS_ZERO_F64)

        # neg on positive infinity and positive zero
        self.assertEqual(ref_cvt_f32_f64(POS_INF_F64, neg=True), NEG_INF_F32)
        self.assertEqual(ref_cvt_f32_f64(POS_ZERO_F64, neg=True), NEG_ZERO_F32)
        self.assertEqual(ref_cvt_f64_f32(POS_INF_F32, neg=True), NEG_INF_F64)
        self.assertEqual(ref_cvt_f64_f32(POS_ZERO_F32, neg=True), NEG_ZERO_F64)

        # clamp on overflow and boundaries
        self.assertEqual(ref_cvt_f32_f64(POS_INF_F64, clamp=True), POS_ONE_F32)
        self.assertEqual(ref_cvt_f32_f64(NEG_INF_F64, clamp=True), POS_ZERO_F32)
        self.assertEqual(ref_cvt_f32_f64(NEG_ONE_F64, clamp=True), POS_ZERO_F32)
        self.assertEqual(ref_cvt_f32_f64(QNAN_F64, clamp=True), QNAN_F32)

        self.assertEqual(ref_cvt_f64_f32(POS_INF_F32, clamp=True), POS_ONE_F64)
        self.assertEqual(ref_cvt_f64_f32(NEG_INF_F32, clamp=True), POS_ZERO_F64)
        self.assertEqual(ref_cvt_f64_f32(NEG_ONE_F32, clamp=True), POS_ZERO_F64)
        self.assertEqual(ref_cvt_f64_f32(QNAN_F32, clamp=True), QNAN_F64)


# ---------------------------------------------------------------------------
# Mutated Oracles and Negative Controls / Syntax Rejection
# ---------------------------------------------------------------------------

def mutant_drop_sign(src: int, is_f64_to_f32: bool) -> int:
    """Mutant A: incorrectly clears sign bit regardless of input sign."""
    if is_f64_to_f32:
        return ref_cvt_f32_f64(src) & ~SIGN_MASK_F32
    return ref_cvt_f64_f32(src) & ~SIGN_MASK_F64


def mutant_nan_flush_zero(src: int, is_f64_to_f32: bool) -> int:
    """Mutant B: improperly flushes NaNs to zero instead of returning NaN."""
    if is_f64_to_f32:
        if is_nan_f64(src):
            return POS_ZERO_F32
        return ref_cvt_f32_f64(src)
    if is_nan_f32(src):
        return POS_ZERO_F64
    return ref_cvt_f64_f32(src)


def mutant_no_clamp(src: int, is_f64_to_f32: bool) -> int:
    """Mutant C: ignores clamp modifier and returns unclamped value."""
    if is_f64_to_f32:
        return ref_cvt_f32_f64(src, clamp=False)
    return ref_cvt_f64_f32(src, clamp=False)


def mutant_f64_to_f32_truncation(src64: int) -> int:
    """Mutant D: uses truncation towards zero instead of roundTiesToEven."""
    s64_mod = src64 & MASK64
    if is_nan_f64(s64_mod):
        return QNAN_F32
    d_val = bits_to_f64(s64_mod)
    sign = (s64_mod >> 63) & 1
    abs_d = abs(d_val)
    if abs_d >= 2.0 ** 128:
        return NEG_INF_F32 if sign else POS_INF_F32
    if abs_d < 2.0 ** -149:
        return NEG_ZERO_F32 if sign else POS_ZERO_F32
    m, e = math.frexp(abs_d)  # abs_d = m * 2^e, 0.5 <= m < 1.0
    scale = 2.0 ** (e - 24)
    int_mant = int(math.floor(abs_d / scale))
    bits = (int_mant << (1 - 1)) if e < -125 else int_mant
    return ((sign << 31) | (bits & 0x7FFFFFFF)) & MASK32


def mutant_f32_to_f64_precision_truncate(src32: int) -> int:
    """Mutant E: corrupts low mantissa bits during f32 to f64 conversion."""
    res = ref_cvt_f64_f32(src32)
    # Artificially drop low 8 bits of the 23-bit converted mantissa (bits 36..29)
    return res & 0xFFFFFFFE00000000


def mutant_leak_vcc(interp: ScalarVALUInterpreter, inst: Instruction) -> None:
    """Mutant F: improperly modifies VCC register during VALU FP execution."""
    interp.step(inst)
    interp.vcc ^= 0x1


def mutant_exec_leak(sim: SIMDWavefrontInterpreter, inst: Instruction) -> None:
    """Mutant G: improperly writes inactive lanes under EXEC mask."""
    sim.step(inst)
    # Overwrite lane 0 regardless of EXEC
    if inst.opcode == Opcode.V_CVT_F32_F64:
        sim.write_lane_reg_u32(0, inst.dst.kind, inst.dst.index, 0x12345678)
    else:
        sim.write_lane_reg_u64(0, inst.dst.kind, inst.dst.index, 0x1234567890ABCDEF)


def parse_register(tok: str) -> Register:
    """Parses a register string like 'v0', 'v[2:3]', 's4', 's[4:5]'."""
    tok = tok.strip()
    if tok.startswith("v[") or tok.startswith("s["):
        kind = RegKind.VGPR if tok.startswith("v") else RegKind.SGPR
        inside = tok[2:-1]
        lo_str, hi_str = inside.split(":")
        lo, hi = int(lo_str), int(hi_str)
        if hi != lo + 1:
            raise ValueError(f"Non-contiguous register pair: {tok}")
        return Register(kind=kind, index=lo, count=2)
    elif tok.startswith("v") or tok.startswith("s"):
        kind = RegKind.VGPR if tok.startswith("v") else RegKind.SGPR
        idx = int(tok[1:])
        return Register(kind=kind, index=idx, count=1)
    else:
        raise ValueError(f"Unrecognized register token: {tok}")


def parse_vop_cvt(asm_str: str) -> Instruction:
    """Parses assembly text for v_cvt_f32_f64 or v_cvt_f64_f32."""
    asm_str = asm_str.strip()
    tokens = asm_str.split(None, 1)
    if not tokens:
        raise ValueError("Empty assembly string")
    op_str = tokens[0]
    rest = tokens[1] if len(tokens) > 1 else ""

    encoding = Encoding.VOP3
    if op_str.endswith("_e32"):
        encoding = Encoding.VOP1
        base_op = op_str[:-4]
    else:
        base_op = op_str

    if base_op == "v_cvt_f32_f64":
        opcode = Opcode.V_CVT_F32_F64
    elif base_op == "v_cvt_f64_f32":
        opcode = Opcode.V_CVT_F64_F32
    else:
        raise ValueError(f"Unsupported opcode: {op_str}")

    clamp = False
    if rest.endswith(" clamp"):
        clamp = True
        rest = rest[:-6].strip()

    args = [a.strip() for a in rest.split(",")]
    if len(args) != 2:
        raise ValueError(f"Instruction requires exactly 2 operands, got {len(args)}: {rest}")

    dst = parse_register(args[0])

    s0_str = args[1]
    neg = False
    abs_val = False
    if s0_str.startswith("-"):
        neg = True
        s0_str = s0_str[1:].strip()
    if s0_str.startswith("|") and s0_str.endswith("|"):
        abs_val = True
        s0_str = s0_str[1:-1].strip()

    if s0_str.startswith("0x") or s0_str.startswith("-0x"):
        is_64 = (opcode == Opcode.V_CVT_F32_F64)
        src0 = Immediate(int(s0_str, 0), is_64bit=is_64)
    else:
        src0 = parse_register(s0_str)

    return Instruction(
        opcode=opcode,
        dst=dst,
        src0=src0,
        src0_mods=OperandModifiers(neg=neg, abs_val=abs_val),
        inst_mods=InstructionModifiers(clamp=clamp),
        encoding=encoding,
    )


class TestValuCvtF32F64NegativeControlsAndMutations(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection for v_cvt_f32_f64 and v_cvt_f64_f32."""

    def test_syntax_rejection_invalid_opcode(self) -> None:
        with self.assertRaises(ValueError):
            parse_vop_cvt("v_add_f64 v0, v[2:3]")

    def test_syntax_rejection_insufficient_operands(self) -> None:
        with self.assertRaises(ValueError):
            parse_vop_cvt("v_cvt_f32_f64 v0")

    def test_syntax_rejection_dst_type_mismatch(self) -> None:
        # v_cvt_f32_f64 destination must be 32-bit single VGPR (count=1), rejecting pair v[0:1]
        with self.assertRaises(ValueError):
            parse_vop_cvt("v_cvt_f32_f64 v[0:1], v[2:3]")

        # v_cvt_f64_f32 destination must be 64-bit pair VGPR (count=2), rejecting single v0
        with self.assertRaises(ValueError):
            parse_vop_cvt("v_cvt_f64_f32 v0, v2")

    def test_syntax_rejection_src_type_mismatch(self) -> None:
        # v_cvt_f32_f64 source must be 64-bit pair (count=2), rejecting single v2
        with self.assertRaises(ValueError):
            parse_vop_cvt("v_cvt_f32_f64 v0, v2")

        # v_cvt_f64_f32 source must be 32-bit single (count=1), rejecting pair v[2:3]
        with self.assertRaises(ValueError):
            parse_vop_cvt("v_cvt_f64_f32 v[0:1], v[2:3]")

    def test_syntax_rejection_sgpr_destination(self) -> None:
        # Destination of VALU instruction cannot be SGPR
        with self.assertRaises(ValueError):
            parse_vop_cvt("v_cvt_f32_f64 s0, v[2:3]")
        with self.assertRaises(ValueError):
            parse_vop_cvt("v_cvt_f64_f32 s[0:1], v2")

    def test_syntax_rejection_vop1_modifiers(self) -> None:
        # VOP1 encoding (_e32) does not support clamp or modifiers
        with self.assertRaises(ValueError):
            parse_vop_cvt("v_cvt_f32_f64_e32 v0, v[2:3] clamp")
        with self.assertRaises(ValueError):
            parse_vop_cvt("v_cvt_f32_f64_e32 v0, -v[2:3]")

    def test_syntax_rejection_register_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            parse_vop_cvt("v_cvt_f32_f64 v256, v[2:3]")
        with self.assertRaises(ValueError):
            parse_vop_cvt("v_cvt_f32_f64 v0, v[255:256]")

    def test_mutation_detection_drop_sign(self) -> None:
        # Negative float must retain negative sign bit
        correct = ref_cvt_f32_f64(NEG_TWO_F64)
        mutant = mutant_drop_sign(NEG_TWO_F64, is_f64_to_f32=True)
        self.assertEqual(correct, NEG_TWO_F32)
        self.assertEqual(mutant, POS_TWO_F32)
        self.assertNotEqual(correct, mutant)

    def test_mutation_detection_nan_flush_zero(self) -> None:
        # NaN must propagate, not flush to zero
        correct = ref_cvt_f32_f64(QNAN_F64)
        mutant = mutant_nan_flush_zero(QNAN_F64, is_f64_to_f32=True)
        self.assertTrue(is_nan_f32(correct))
        self.assertEqual(mutant, POS_ZERO_F32)
        self.assertNotEqual(correct, mutant)

    def test_mutation_detection_no_clamp(self) -> None:
        correct = ref_cvt_f32_f64(POS_TWO_F64, clamp=True)
        mutant = mutant_no_clamp(POS_TWO_F64, is_f64_to_f32=True)
        self.assertEqual(correct, POS_ONE_F32)
        self.assertEqual(mutant, POS_TWO_F32)
        self.assertNotEqual(correct, mutant)

    def test_mutation_detection_truncation_vs_ties_to_even(self) -> None:
        # 1.0 + 1.5 * 2^-23 in double precision:
        # Ties to even will round up to 1.0 + 2^-22 in f32
        d_val = f64_to_bits(1.0 + 1.5 * (2.0 ** -23))
        correct = ref_cvt_f32_f64(d_val)
        mutant = mutant_f64_to_f32_truncation(d_val)
        self.assertNotEqual(correct, mutant)

    def test_mutation_detection_precision_truncate(self) -> None:
        # 1.0f + 2^-20 has low mantissa bits in f64 that mutant zeroes out
        f_val = f32_to_bits(1.0 + (2.0 ** -20))
        correct = ref_cvt_f64_f32(f_val)
        mutant = mutant_f32_to_f64_precision_truncate(f_val)
        self.assertEqual(bits_to_f64(correct), 1.0 + (2.0 ** -20))
        self.assertNotEqual(correct, mutant)

    def test_mutation_detection_corrupt_vcc(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.vcc = 0x5555555555555555
        interp.write_reg_u64(RegKind.VGPR, 2, POS_ONE_F64)
        inst = parse_vop_cvt("v_cvt_f32_f64 v0, v[2:3]")

        interp.step(inst)
        self.assertEqual(interp.vcc, 0x5555555555555555)

        mutant_leak_vcc(interp, inst)
        self.assertNotEqual(interp.vcc, 0x5555555555555555)

    def test_mutation_detection_exec_leak(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        sim.set_exec_mask(0)  # all lanes disabled
        sim.write_lane_reg_u32(0, RegKind.VGPR, 0, 0xAAAAAAAA)
        inst = parse_vop_cvt("v_cvt_f32_f64 v0, v[2:3]")

        sim.step(inst)
        self.assertEqual(sim.read_lane_reg_u32(0, RegKind.VGPR, 0), 0xAAAAAAAA)

        mutant_exec_leak(sim, inst)
        self.assertNotEqual(sim.read_lane_reg_u32(0, RegKind.VGPR, 0), 0xAAAAAAAA)


if __name__ == "__main__":
    unittest.main()
