from __future__ import annotations

"""End-to-end semantic value tests for RDNA2 v_fract_f32 and v_fract_f16.

Covers v_fract_f32 (single-precision) and v_fract_f16 (half-precision) executed
end-to-end through tools/rdna2spirv.py and tests/semantic/spirv_mini_interp.py
against independent Fraction-based IEEE-754 `x - floor(x)` reference oracles:
  - Formula: D = S0 - floor(S0), returning values in [0.0, 1.0).
  - Non-negative property: for any finite input x, fract(x) >= 0.0 and fract(x) < 1.0.
  - Negative number fractional part: fract(-0.25) == 0.75, fract(-1.75) == 0.25,
    fract(-0.5) == 0.5, fract(-0.75) == 0.25.
  - Exact integers: fract(N.0) == +0.0 (0.0, 1.0, -1.0, 2.0, -2.0, 42.0, etc.).
  - Large numbers with no fractional bits in mantissa:
    * Single-precision (|x| >= 2^23 = 8388608.0 for f32) yields +0.0, while
      2^23 - 0.5 (8388607.5) preserves fractional part 0.5.
    * Half-precision (|x| >= 2^10 = 1024.0 for f16) yields +0.0, while
      1023.5 preserves fractional part 0.5.
  - Special IEEE-754 values:
    * +0.0 -> +0.0 and -0.0 -> +0.0 (sign bit strictly 0).
    * +-inf -> NaN and NaN -> NaN.
  - Subnormal bounds and rounding transitions:
    * Positive subnormals (x in (0, 2^-126) for f32, (0, 2^-14) for f16) have
      floor(x) == 0, so fract(x) == x exact.
    * Negative subnormals and tiny negatives: x - floor(x) = 1.0 - |x|, which
      rounds to 1.0 in IEEE-754 nearest-even. The RDNA2 hardware contract and
      translator clamp any result that rounded up to 1.0 to 0x3F7FFFFF (f32,
      largest float < 1.0) and 0x3BFF (f16, largest half < 1.0).
  - Forms: _e32, _e64, and bare mnemonics.
  - Half-precision op_sel combinations ([0,0], [1,0], [0,1], [1,1]) preserving
    untouched halves of destination VGPRs.
  - VOP3 source modifiers (-v1, |v1|, -|v1|) and output modifiers
    (clamp, mul:2, mul:4, div:2, mul:2 clamp, mul:4 clamp).
  - Inline constants (0.5, -0.5, 1.0, -1.0), SGPR sources (s4), destination
    aliasing (v0, v0), and runtime memory inputs via s_buffer_load_dwordx4.
  - Negative controls: mutating SPIR-V Fract to Trunc/Floor or removing the
    rounding clamp fails test assertions.
  - Multi-lane SIMD divergence under EXEC mask across 8-thread, 16-thread, and
    32-thread compute wavefronts.

Zero game data; hand-written RDNA2 assembly only; standard library Python only.
"""

from fractions import Fraction
import gc
import itertools
import math
import os
import re
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xBEEF1234
PUSH_ADDR = [0x1000, 0, 32, 0] + [0] * 28
VSHARP_CS = [0, 0, 0xFFFFFFFF, 0x31016FAC]
HEAP_FILL = 0xEEEEEEEE

# Representative IEEE-754 binary32 bit encodings.
F_POS_ZERO = 0x00000000      # +0.0
F_NEG_ZERO = 0x80000000      # -0.0
F_SUB_MIN = 0x00000001       # +2^-149 (smallest positive subnormal)
F_NEG_SUB_MIN = 0x80000001   # -2^-149
F_SUB_1E40 = 0x000116C2      # ~+1.0003e-40 (subnormal)
F_NEG_SUB_1E40 = 0x800116C2  # ~-1.0003e-40
F_SUB_MAX = 0x007FFFFF       # largest positive subnormal (2^-126 - 2^-149)
F_NEG_SUB_MAX = 0x807FFFFF   # largest negative subnormal magnitude
F_NORM_MIN = 0x00800000      # +2^-126 (smallest positive normal)
F_NEG_NORM_MIN = 0x80800000  # -2^-126
F_NEG_1E10 = 0xAEDBE6FF      # -1e-10
F_NEG_2PM26 = 0xB2800000     # -2^-26
F_NEG_2PM25 = 0xB3000000     # -2^-25 (exact rounding midpoint to 1.0)
F_NEG_2PM24 = 0xB3800000     # -2^-24: 1 - 2^-24 = 0x3F7FFFFF exact
F_NEG_2PM23 = 0xB4000000     # -2^-23: 1 - 2^-23 = 0x3F7FFFFE exact
F_0P125 = 0x3E000000         # 0.125
F_0P25 = 0x3E800000          # 0.25
F_NEG_0P25 = 0xBE800000      # -0.25
F_0P375 = 0x3EC00000         # 0.375
F_0P5 = 0x3F000000           # 0.5
F_NEG_0P5 = 0xBF000000       # -0.5
F_0P75 = 0x3F400000          # 0.75
F_NEG_0P75 = 0xBF400000      # -0.75
F_NEAR_ONE = 0x3F7FFFFF      # 1.0 - 2^-24 (largest f32 < 1.0)
F_1P0 = 0x3F800000           # 1.0
F_NEG_1P0 = 0xBF800000       # -1.0
F_1P25 = 0x3FA00000          # 1.25
F_NEG_1P25 = 0xBFA00000      # -1.25
F_1P5 = 0x3FC00000           # 1.5
F_NEG_1P5 = 0xBFC00000       # -1.5
F_1P75 = 0x3FE00000          # 1.75
F_NEG_1P75 = 0xBFE00000      # -1.75
F_2P0 = 0x40000000           # 2.0
F_NEG_2P0 = 0xC0000000       # -2.0
F_2P5 = 0x40200000           # 2.5
F_NEG_2P5 = 0xC0200000       # -2.5
F_3P0 = 0x40400000           # 3.0
F_NEG_3P0 = 0xC0400000       # -3.0
F_3P75 = 0x40700000          # 3.75
F_NEG_3P75 = 0xC0700000      # -3.75
F_42P0 = 0x42280000          # 42.0
F_NEG_42P0 = 0xC2280000      # -42.0
F_1023P5 = 0x447FE000        # 1023.5 in f32
F_1024P0 = 0x44800000        # 1024.0 in f32
F_1024P25 = 0x44800800       # 1024.25 in f32 (exact in f32, rounds in f16)
F_2P23_M0P5 = 0x4AFFFFFF     # 2^23 - 0.5 = 8388607.5
F_2P23 = 0x4B000000          # 2^23 = 8388608.0 (mantissa bits exhausted)
F_NEG_2P23 = 0xCB000000      # -2^23 = -8388608.0
F_2P24 = 0x4B800000          # 2^24 = 16777216.0
F_NEG_2P24 = 0xCB800000      # -2^24 = -16777216.0
F_POS_INF = 0x7F800000       # +inf
F_NEG_INF = 0xFF800000       # -inf
F_QNAN = 0x7FC00000          # +qNaN
F_NEG_QNAN = 0xFFC00000      # -qNaN
F_SNAN = 0x7F800001          # +sNaN
F_NEG_SNAN = 0xFF800001      # -sNaN

# Representative IEEE-754 binary16 bit encodings.
H_POS_ZERO = 0x0000          # +0.0
H_NEG_ZERO = 0x8000          # -0.0
H_SUB_MIN = 0x0001           # +2^-24
H_NEG_SUB_MIN = 0x8001       # -2^-24
H_SUB_MAX = 0x03FF           # largest positive subnormal
H_NEG_SUB_MAX = 0x83FF       # largest negative subnormal
H_NORM_MIN = 0x0400          # +2^-14
H_NEG_NORM_MIN = 0x8400      # -2^-14
H_0P125 = 0x3000             # 0.125
H_0P25 = 0x3400              # 0.25
H_NEG_0P25 = 0xB400          # -0.25
H_0P375 = 0x3600             # 0.375
H_0P5 = 0x3800               # 0.5
H_NEG_0P5 = 0xB800           # -0.5
H_0P75 = 0x3A00              # 0.75
H_NEG_0P75 = 0xBA00          # -0.75
H_NEAR_ONE = 0x3BFF          # 1.0 - 2^-11 (largest f16 < 1.0)
H_1P0 = 0x3C00               # 1.0
H_NEG_1P0 = 0xBC00           # -1.0
H_1P25 = 0x3D00              # 1.25
H_NEG_1P25 = 0xBD00          # -1.25
H_1P5 = 0x3E00               # 1.5
H_NEG_1P5 = 0xBE00           # -1.5
H_1P75 = 0x3F00              # 1.75
H_NEG_1P75 = 0xBF00          # -1.75
H_2P0 = 0x4000               # 2.0
H_NEG_2P0 = 0xC000           # -2.0
H_2P5 = 0x4100               # 2.5
H_NEG_2P5 = 0xC100           # -2.5
H_3P0 = 0x4200               # 3.0
H_NEG_3P0 = 0xC200           # -3.0
H_3P75 = 0x4380              # 3.75
H_NEG_3P75 = 0xC380          # -3.75
H_1023P5 = 0x63FF            # 1023.5 (f16 fractional part 0.5)
H_1024 = 0x6400              # 2^10 = 1024.0 (f16 mantissa exhausted -> 0.0)
H_NEG_1024 = 0xE400          # -1024.0
H_2048 = 0x6800              # 2048.0
H_MAX = 0x7BFF               # 65504.0
H_NEG_MAX = 0xFBFF           # -65504.0
H_POS_INF = 0x7C00           # +inf
H_NEG_INF = 0xFC00           # -inf
H_QNAN = 0x7E01              # +qNaN
H_NEG_QNAN = 0xFE01          # -qNaN
H_SNAN = 0x7C01              # +sNaN
H_NEG_SNAN = 0xFC01          # -sNaN


# -----------------------------------------------------------------------------
# Independent Fraction-Based IEEE-754 Binary32 Oracle
# -----------------------------------------------------------------------------

def f32_is_finite(w: int) -> bool:
    return ((w >> 23) & 0xFF) != 0xFF


def f32_is_nan(w: int) -> bool:
    return ((w >> 23) & 0xFF) == 0xFF and (w & 0x7FFFFF) != 0


def f32_is_inf(w: int) -> bool:
    return (w & 0x7FFFFFFF) == 0x7F800000


def f32_to_fraction(w: int) -> Fraction:
    """Decode finite IEEE-754 binary32 bit pattern ``w`` into an exact Fraction."""
    w &= 0xFFFFFFFF
    exp = (w >> 23) & 0xFF
    if exp == 0xFF:
        raise ValueError(f"non-finite binary32 {w:#010x}")
    sign = -1 if (w >> 31) else 1
    mant = w & 0x7FFFFF
    if exp == 0:
        return Fraction(0) if mant == 0 else Fraction(sign * mant, 1 << 149)
    return Fraction(sign * ((1 << 23) | mant)) * (Fraction(2) ** (exp - 150))


def fraction_to_f32(q: Fraction) -> int:
    """Round an exact Fraction to IEEE-754 binary32 (round-to-nearest, ties-to-even)."""
    if q == 0:
        return F_POS_ZERO
    sgn = 0x80000000 if q < 0 else 0
    a = abs(q)
    e = max(a.numerator.bit_length() - a.denominator.bit_length(), -126)
    if e > -126 and Fraction(2) ** e > a:
        e -= 1
    m = a / (Fraction(2) ** (e - 23))
    n = m.numerator // m.denominator
    rem = m - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
        n += 1
    if e == -126 and n < (1 << 23):
        return sgn | n
    if n >= (1 << 24):
        n >>= 1
        e += 1
    if e > 127:
        return sgn | F_POS_INF
    return sgn | ((e + 127) << 23) | (n - (1 << 23))


def exact_fract_f32_fraction(w: int) -> Fraction | None:
    """Exact ``x - floor(x)`` as a Fraction in [0, 1), or None for +-inf / NaN."""
    if not f32_is_finite(w):
        return None
    q = f32_to_fraction(w)
    return q - math.floor(q)


def oracle_fract_f32(
    w: int,
    *,
    abs_: bool = False,
    neg: bool = False,
    mul: int = 1,
    div: int = 1,
    clamp: bool = False,
) -> int | None:
    """Independent binary32 reference for v_fract_f32, returning 32-bit int or None for NaN.

    Models RDNA2 source modifiers (abs before neg), exact ``x - floor(x)``,
    the hardware/translator 0x3f800000 -> 0x3f7fffff clamp when no output multiplier is
    active, and VOP3 output scaling (mul/div) + [0, 1] clamp.
    """
    w &= 0xFFFFFFFF
    if abs_:
        w &= 0x7FFFFFFF
    if neg:
        w ^= 0x80000000
    if w & 0x7F800000 == 0:
        w = 0  # a denormal (or -0) operand reads as +0 with f32 denormals flushed, as on RDNA2 in the game's MODE
    r = exact_fract_f32_fraction(w)
    if r is None:
        return None
    if mul == 1 and div == 1:
        res = fraction_to_f32(r)
        return F_NEAR_ONE if res == F_1P0 else res
    scaled = r * Fraction(mul, div)
    if clamp:
        scaled = min(max(scaled, Fraction(0)), Fraction(1))
    return fraction_to_f32(scaled)


# -----------------------------------------------------------------------------
# Independent Fraction-Based IEEE-754 Binary16 Oracle
# -----------------------------------------------------------------------------

def h16_is_finite(h: int) -> bool:
    return ((h >> 10) & 0x1F) != 0x1F


def h16_is_nan(h: int) -> bool:
    return (h & 0x7FFF) > 0x7C00


def h16_is_inf(h: int) -> bool:
    return (h & 0x7FFF) == 0x7C00


def h16_to_fraction(h: int) -> Fraction:
    """Decode finite IEEE-754 binary16 bit pattern ``h`` into an exact Fraction."""
    h &= 0xFFFF
    exp = (h >> 10) & 0x1F
    if exp == 0x1F:
        raise ValueError(f"non-finite binary16 {h:#06x}")
    sign = -1 if (h >> 15) else 1
    mant = h & 0x3FF
    if exp == 0:
        return Fraction(0) if mant == 0 else Fraction(sign * mant, 1 << 24)
    return Fraction(sign * ((1 << 10) | mant)) * (Fraction(2) ** (exp - 25))


def fraction_to_h16(q: Fraction) -> int:
    """Round an exact Fraction to IEEE-754 binary16 (nearest, ties to even)."""
    if q == 0:
        return H_POS_ZERO
    sgn = 0x8000 if q < 0 else 0
    a = abs(q)
    e = max(a.numerator.bit_length() - a.denominator.bit_length(), -14)
    if e > -14 and Fraction(2) ** e > a:
        e -= 1
    m = a / (Fraction(2) ** (e - 10))
    n = m.numerator // m.denominator
    rem = m - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
        n += 1
    if e == -14 and n < 1024:
        return sgn | n
    if n >= 2048:
        n >>= 1
        e += 1
    if e > 15:
        return sgn | H_POS_INF
    return sgn | ((e + 15) << 10) | (n - 1024)


def exact_fract_f16_fraction(h: int) -> Fraction | None:
    """Exact ``x - floor(x)`` as a Fraction in [0, 1), or None for +-inf / NaN."""
    if not h16_is_finite(h):
        return None
    q = h16_to_fraction(h)
    return q - math.floor(q)


def oracle_fract_f16(
    h: int,
    *,
    abs_: bool = False,
    neg: bool = False,
    mul: int = 1,
    div: int = 1,
    clamp: bool = False,
) -> int | None:
    """Independent binary16 reference for v_fract_f16, returning 16-bit int or None for NaN."""
    h &= 0xFFFF
    if abs_:
        h &= 0x7FFF
    if neg:
        h ^= 0x8000
    r = exact_fract_f16_fraction(h)
    if r is None:
        return None
    if mul == 1 and div == 1:
        res = fraction_to_h16(r)
        return H_NEAR_ONE if res == H_1P0 else res
    scaled = r * Fraction(mul, div)
    if clamp:
        scaled = min(max(scaled, Fraction(0)), Fraction(1))
    return fraction_to_h16(scaled)


def merge_half(old: int, half: int, dst_hi: int) -> int:
    h = half & 0xFFFF
    return ((h << 16) | (old & 0xFFFF)) if dst_hi else ((old & 0xFFFF0000) | h)


# -----------------------------------------------------------------------------
# Translation & Execution Helpers
# -----------------------------------------------------------------------------

_TMP_DIR = None
_PS_PATH = None
_CS_PATH = None
_CS_FSSM = None


def _make_fssm(rsrc2: int, group: tuple[int, int, int]) -> bytes:
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


def setUpModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH, _CS_FSSM
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    fssm_dir = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)
    _PS_PATH = os.path.join(disasm_dir, "ps_fract_f32.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_fract_f32.s")
    _CS_FSSM = os.path.join(fssm_dir, "cs_fract_f32.fssm")
    with open(os.path.join(_TMP_DIR.name, "ps_fract_f32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH, _CS_FSSM
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _PS_PATH = None
        _CS_PATH = None
        _CS_FSSM = None


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_valu_fract_f32\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    )
    with open(_PS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
        gc.collect()
    return spv


def run_ps(lines: list[str], *, push_constants=None, memory=None) -> int:
    out = si.run_fragment(translate_ps(lines), push_constants=push_constants, memory=memory)
    return out["%out_mrt0"][0]


def run_op(line: str, v1: int = 0, *, old: int = OLD, s4: int = 0) -> int:
    return run_ps([
        f"s_mov_b32 s4, {s4:#x}",
        f"v_mov_b32 v1, {v1:#x}",
        f"v_mov_b32 v0, {old:#x}",
        line,
    ])


def run_cs(lines: list[str], group_size: int, memory: dict[int, int]) -> None:
    if _TMP_DIR is None:
        setUpModule()
    # rsrc2 = 0x0088: USER_SGPR = 4 (s[0:3] V#), TGID_X_EN = 1 (s4), TIDIG_COMP_CNT = 0 (v0 = local_id.x).
    with open(_CS_FSSM, "wb") as f:
        f.write(_make_fssm(0x0088, (group_size, 1, 1)))
    asm = "; test_cs_valu_fract_f32\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_CS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_CS_PATH)
        gc.collect()
    si.run_workgroup(
        spv,
        group_size,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
        push_constants=VSHARP_CS + [0] * 28,
        memory=memory,
    )


class _Heap(dict):
    def get(self, key, default=None):
        return super().get(key, HEAP_FILL)


# -----------------------------------------------------------------------------
# Oracle Unit Tests
# -----------------------------------------------------------------------------

class OracleF32Tests(unittest.TestCase):
    """Verifies the independent Fraction-based binary32 oracle against IEEE-754 semantics."""

    def test_fraction_decode_and_roundtrip(self):
        samples = [
            F_POS_ZERO, F_NEG_ZERO,
            F_SUB_MIN, F_NEG_SUB_MIN, F_SUB_1E40, F_NEG_SUB_1E40,
            F_SUB_MAX, F_NEG_SUB_MAX, F_NORM_MIN, F_NEG_NORM_MIN,
            F_NEG_1E10, F_NEG_2PM26, F_NEG_2PM25, F_NEG_2PM24, F_NEG_2PM23,
            F_0P125, F_0P25, F_NEG_0P25, F_0P375,
            F_0P5, F_NEG_0P5, F_0P75, F_NEG_0P75,
            F_NEAR_ONE, F_1P0, F_NEG_1P0,
            F_1P25, F_NEG_1P25, F_1P5, F_NEG_1P5, F_1P75, F_NEG_1P75,
            F_2P0, F_NEG_2P0, F_2P5, F_NEG_2P5, F_3P0, F_NEG_3P0,
            F_3P75, F_NEG_3P75, F_42P0, F_NEG_42P0,
            F_1023P5, F_1024P0, F_1024P25,
            F_2P23_M0P5, F_2P23, F_NEG_2P23, F_2P24, F_NEG_2P24,
        ]
        for w in samples:
            with self.subTest(w=hex(w)):
                self.assertTrue(f32_is_finite(w))
                q = f32_to_fraction(w)
                expected_float = struct.unpack("<f", struct.pack("<I", w))[0]
                self.assertEqual(q, Fraction(expected_float))
                want = F_POS_ZERO if w == F_NEG_ZERO else w
                self.assertEqual(fraction_to_f32(q), want)

    def test_hand_worked_fract_f32_rows(self):
        rows = [
            (F_POS_ZERO, F_POS_ZERO),
            (F_NEG_ZERO, F_POS_ZERO),
            (F_SUB_MIN, F_POS_ZERO),       # denormal operands read as zero (f32 denormals flushed)
            (F_SUB_1E40, F_POS_ZERO),
            (F_SUB_MAX, F_POS_ZERO),
            (F_NORM_MIN, F_NORM_MIN),
            (F_NEG_SUB_MIN, F_POS_ZERO),   # denormal operands read as zero: fract(0) = +0
            (F_NEG_SUB_1E40, F_POS_ZERO),
            (F_NEG_SUB_MAX, F_POS_ZERO),
            (F_NEG_NORM_MIN, F_NEAR_ONE),
            (F_NEG_1E10, F_NEAR_ONE),
            (F_NEG_2PM26, F_NEAR_ONE),
            (F_NEG_2PM25, F_NEAR_ONE),
            (F_NEG_2PM24, F_NEAR_ONE),
            (F_NEG_2PM23, 0x3F7FFFFE),
            (F_0P25, F_0P25),
            (F_NEG_0P25, F_0P75),
            (F_0P5, F_0P5),
            (F_NEG_0P5, F_0P5),
            (F_0P75, F_0P75),
            (F_NEG_0P75, F_0P25),
            (F_1P0, F_POS_ZERO),
            (F_NEG_1P0, F_POS_ZERO),
            (F_1P25, F_0P25),
            (F_NEG_1P25, F_0P75),
            (F_1P75, F_0P75),
            (F_NEG_1P75, F_0P25),
            (F_2P0, F_POS_ZERO),
            (F_NEG_2P0, F_POS_ZERO),
            (F_2P5, F_0P5),
            (F_NEG_2P5, F_0P5),
            (F_3P75, F_0P75),
            (F_NEG_3P75, F_0P25),
            (F_42P0, F_POS_ZERO),
            (F_NEG_42P0, F_POS_ZERO),
            (F_1023P5, F_0P5),
            (F_1024P0, F_POS_ZERO),
            (F_1024P25, F_0P25),
            (F_2P23_M0P5, F_0P5),
            (F_2P23, F_POS_ZERO),
            (F_NEG_2P23, F_POS_ZERO),
            (F_2P24, F_POS_ZERO),
            (F_NEG_2P24, F_POS_ZERO),
            (F_NEAR_ONE, F_NEAR_ONE),
        ]
        for w, want in rows:
            with self.subTest(w=hex(w)):
                self.assertEqual(oracle_fract_f32(w), want)

    def test_non_negative_property(self):
        # fract(x) >= 0.0 and fract(x) < 1.0 for all finite numbers
        test_inputs = [
            -100.5, -42.25, -10.75, -2.5, -1.75, -1.25, -0.75, -0.5, -0.25, -0.0,
            0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 42.125, 100.875,
        ]
        for val in test_inputs:
            bits = struct.unpack("<I", struct.pack("<f", val))[0]
            res_w = oracle_fract_f32(bits)
            res_val = struct.unpack("<f", struct.pack("<I", res_w))[0]
            with self.subTest(val=val):
                self.assertGreaterEqual(res_val, 0.0)
                self.assertLess(res_val, 1.0)

    def test_large_numbers_mantissa_exhaustion(self):
        # |x| >= 2^23 has no fractional bits in IEEE binary32 mantissa -> fract(x) == 0.0
        for exp in range(23, 31):
            val_pos = float(1 << exp)
            val_neg = -val_pos
            w_pos = struct.unpack("<I", struct.pack("<f", val_pos))[0]
            w_neg = struct.unpack("<I", struct.pack("<f", val_neg))[0]
            with self.subTest(exp=exp):
                self.assertEqual(oracle_fract_f32(w_pos), F_POS_ZERO)
                self.assertEqual(oracle_fract_f32(w_neg), F_POS_ZERO)

    def test_special_ieee_oracle(self):
        self.assertEqual(oracle_fract_f32(F_POS_ZERO), F_POS_ZERO)
        self.assertEqual(oracle_fract_f32(F_NEG_ZERO), F_POS_ZERO)
        self.assertIsNone(oracle_fract_f32(F_POS_INF))
        self.assertIsNone(oracle_fract_f32(F_NEG_INF))
        self.assertIsNone(oracle_fract_f32(F_QNAN))
        self.assertIsNone(oracle_fract_f32(F_NEG_QNAN))
        self.assertIsNone(oracle_fract_f32(F_SNAN))
        self.assertIsNone(oracle_fract_f32(F_NEG_SNAN))


class OracleF16Tests(unittest.TestCase):
    """Verifies the independent Fraction-based binary16 oracle against IEEE-754 semantics."""

    def test_fraction_roundtrip_f16(self):
        samples = [
            H_POS_ZERO, H_NEG_ZERO,
            H_SUB_MIN, H_NEG_SUB_MIN, H_SUB_MAX, H_NEG_SUB_MAX,
            H_NORM_MIN, H_NEG_NORM_MIN,
            H_0P125, H_0P25, H_NEG_0P25, H_0P375,
            H_0P5, H_NEG_0P5, H_0P75, H_NEG_0P75,
            H_NEAR_ONE, H_1P0, H_NEG_1P0,
            H_1P25, H_NEG_1P25, H_1P5, H_NEG_1P5, H_1P75, H_NEG_1P75,
            H_2P0, H_NEG_2P0, H_2P5, H_NEG_2P5, H_3P0, H_NEG_3P0,
            H_3P75, H_NEG_3P75, H_1023P5, H_1024, H_NEG_1024,
            H_2048, H_MAX, H_NEG_MAX,
        ]
        for h in samples:
            with self.subTest(h=hex(h)):
                self.assertTrue(h16_is_finite(h))
                q = h16_to_fraction(h)
                expected_float = struct.unpack("<e", struct.pack("<H", h))[0]
                self.assertEqual(q, Fraction(expected_float))
                want = H_POS_ZERO if h == H_NEG_ZERO else h
                self.assertEqual(fraction_to_h16(q), want)

    def test_hand_worked_fract_f16_rows(self):
        rows = [
            (H_POS_ZERO, H_POS_ZERO),
            (H_NEG_ZERO, H_POS_ZERO),
            (H_SUB_MIN, H_SUB_MIN),
            (H_NEG_SUB_MIN, H_NEAR_ONE),
            (H_SUB_MAX, H_SUB_MAX),
            (H_NEG_SUB_MAX, H_NEAR_ONE),
            (H_NORM_MIN, H_NORM_MIN),
            (H_NEG_NORM_MIN, H_NEAR_ONE),
            (H_0P25, H_0P25),
            (H_NEG_0P25, H_0P75),
            (H_0P5, H_0P5),
            (H_NEG_0P5, H_0P5),
            (H_0P75, H_0P75),
            (H_NEG_0P75, H_0P25),
            (H_1P0, H_POS_ZERO),
            (H_NEG_1P0, H_POS_ZERO),
            (H_1P25, H_0P25),
            (H_NEG_1P25, H_0P75),
            (H_1P75, H_0P75),
            (H_NEG_1P75, H_0P25),
            (H_2P0, H_POS_ZERO),
            (H_NEG_2P0, H_POS_ZERO),
            (H_2P5, H_0P5),
            (H_NEG_2P5, H_0P5),
            (H_3P75, H_0P75),
            (H_NEG_3P75, H_0P25),
            (H_1023P5, H_0P5),
            (H_1024, H_POS_ZERO),
            (H_NEG_1024, H_POS_ZERO),
            (H_2048, H_POS_ZERO),
            (H_MAX, H_POS_ZERO),
            (H_NEG_MAX, H_POS_ZERO),
            (H_NEAR_ONE, H_NEAR_ONE),
        ]
        for h, want in rows:
            with self.subTest(h=hex(h)):
                self.assertEqual(oracle_fract_f16(h), want)

    def test_non_negative_property_f16(self):
        for h in (H_NEG_0P75, H_NEG_0P5, H_NEG_0P25, H_0P25, H_0P5, H_0P75, H_1023P5):
            res_h = oracle_fract_f16(h)
            res_val = struct.unpack("<e", struct.pack("<H", res_h))[0]
            with self.subTest(h=hex(h)):
                self.assertGreaterEqual(res_val, 0.0)
                self.assertLess(res_val, 1.0)

    def test_large_numbers_mantissa_exhaustion_f16(self):
        # In binary16, mantissa has 10 bits. Any |x| >= 2^10 (1024.0) has no fractional bits.
        for h in (H_1024, H_NEG_1024, H_2048, H_MAX, H_NEG_MAX):
            with self.subTest(h=hex(h)):
                self.assertEqual(oracle_fract_f16(h), H_POS_ZERO)

    def test_special_ieee_oracle_f16(self):
        self.assertEqual(oracle_fract_f16(H_POS_ZERO), H_POS_ZERO)
        self.assertEqual(oracle_fract_f16(H_NEG_ZERO), H_POS_ZERO)
        self.assertIsNone(oracle_fract_f16(H_POS_INF))
        self.assertIsNone(oracle_fract_f16(H_NEG_INF))
        self.assertIsNone(oracle_fract_f16(H_QNAN))
        self.assertIsNone(oracle_fract_f16(H_NEG_QNAN))
        self.assertIsNone(oracle_fract_f16(H_SNAN))
        self.assertIsNone(oracle_fract_f16(H_NEG_SNAN))


class PrecisionF32VsF16Tests(unittest.TestCase):
    """Contrasts precision limits and verifies mathematical agreement between f32 and f16."""

    def test_common_exact_fractions_identical_mathematics(self):
        pairs = [
            (F_0P25, H_0P25),
            (F_NEG_0P25, H_NEG_0P25),
            (F_0P5, H_0P5),
            (F_NEG_0P5, H_NEG_0P5),
            (F_0P75, H_0P75),
            (F_NEG_0P75, H_NEG_0P75),
            (F_1P25, H_1P25),
            (F_NEG_1P25, H_NEG_1P25),
            (F_1P75, H_1P75),
            (F_NEG_1P75, H_NEG_1P75),
            (F_3P75, H_3P75),
            (F_NEG_3P75, H_NEG_3P75),
            (F_1023P5, H_1023P5),
        ]
        for f_w, h_w in pairs:
            q32 = f32_to_fraction(oracle_fract_f32(f_w))
            q16 = h16_to_fraction(oracle_fract_f16(h_w))
            with self.subTest(f_w=hex(f_w), h_w=hex(h_w)):
                self.assertEqual(q32, q16)

    def test_mantissa_exhaustion_divergence_between_precisions(self):
        # At 1024.25:
        # In f32, 1024.25 is exact -> fract is 0.25 (0x3E800000).
        self.assertEqual(oracle_fract_f32(F_1024P25), F_0P25)
        # In f16, 1024.25 cannot be represented: it rounds to 1024.0 -> fract is +0.0 (0x0000).
        f16_rounded_1024p25 = fraction_to_h16(Fraction(102425, 100))
        self.assertEqual(f16_rounded_1024p25, H_1024)
        self.assertEqual(oracle_fract_f16(f16_rounded_1024p25), H_POS_ZERO)

        # At 2^23 - 0.5 (8388607.5):
        # In f32, 8388607.5 is exact -> fract is 0.5 (0x3F000000).
        self.assertEqual(oracle_fract_f32(F_2P23_M0P5), F_0P5)
        # At 2^23 (8388608.0):
        # In f32, mantissa bits are exhausted -> fract is +0.0 (0x00000000).
        self.assertEqual(oracle_fract_f32(F_2P23), F_POS_ZERO)


# -----------------------------------------------------------------------------
# End-to-End Execution Tests
# -----------------------------------------------------------------------------

class FractF32ExecutionTests(unittest.TestCase):
    """End-to-end execution of RDNA2 v_fract_f32 through rdna2spirv and spirv_mini_interp."""

    def test_finite_values_across_e32_e64_and_bare(self):
        samples = [
            F_POS_ZERO, F_NEG_ZERO,
            F_0P125, F_0P25, F_NEG_0P25, F_0P375,
            F_0P5, F_NEG_0P5, F_0P75, F_NEG_0P75,
            F_NEAR_ONE, F_1P0, F_NEG_1P0,
            F_1P25, F_NEG_1P25, F_1P5, F_NEG_1P5, F_1P75, F_NEG_1P75,
            F_2P0, F_NEG_2P0, F_2P5, F_NEG_2P5, F_3P0, F_NEG_3P0,
            F_3P75, F_NEG_3P75, F_42P0, F_NEG_42P0,
            F_1023P5, F_1024P0, F_1024P25,
            F_2P23_M0P5, F_2P23, F_NEG_2P23, F_2P24, F_NEG_2P24,
        ]
        for suffix in ("_e32", "_e64", ""):
            for w in samples:
                want = oracle_fract_f32(w)
                with self.subTest(form="v_fract_f32" + suffix, w=hex(w)):
                    self.assertEqual(run_op(f"v_fract_f32{suffix} v0, v1", w), want)

    def test_negative_number_fractional_part(self):
        # Specific assertion: fract(-0.25) == 0.75, fract(-1.75) == 0.25
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_0P25), F_0P75)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_1P75), F_0P25)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_0P5), F_0P5)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_0P75), F_0P25)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_1P25), F_0P75)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_2P5), F_0P5)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_3P75), F_0P25)

    def test_exact_integers_and_signed_zero(self):
        # Integers and +-0.0 must yield +0.0 (sign bit must be 0)
        cases = [
            F_POS_ZERO, F_NEG_ZERO,
            F_1P0, F_NEG_1P0,
            F_2P0, F_NEG_2P0,
            F_3P0, F_NEG_3P0,
            F_42P0, F_NEG_42P0,
            F_1024P0,
            F_2P23, F_NEG_2P23,
            F_2P24, F_NEG_2P24,
        ]
        for w in cases:
            with self.subTest(w=hex(w)):
                got = run_op("v_fract_f32_e32 v0, v1", w)
                self.assertEqual(got, F_POS_ZERO)
                self.assertNotEqual(got, F_NEG_ZERO)

    def test_large_numbers_no_fractional_bits(self):
        # 2^23 has no fractional bits -> 0.0
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_2P23), F_POS_ZERO)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_2P23), F_POS_ZERO)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_2P24), F_POS_ZERO)
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_NEG_2P24), F_POS_ZERO)
        # 2^23 - 0.5 has fractional bit 0.5
        self.assertEqual(run_op("v_fract_f32_e32 v0, v1", F_2P23_M0P5), F_0P5)

    def test_special_ieee_values_execution(self):
        for w in (F_POS_INF, F_NEG_INF, F_QNAN, F_NEG_QNAN, F_SNAN, F_NEG_SNAN):
            for suffix in ("_e32", "_e64", ""):
                with self.subTest(w=hex(w), suffix=suffix):
                    got = run_op(f"v_fract_f32{suffix} v0, v1", w)
                    self.assertTrue(f32_is_nan(got), f"expected NaN, got {got:#010x}")

    def test_vop3_source_modifiers(self):
        cases = [
            ("v_fract_f32_e64 v0, -v1", F_0P25, False, True),       # -(0.25) -> 0.75
            ("v_fract_f32_e64 v0, -v1", F_NEG_0P25, False, True),   # -(-0.25) -> 0.25
            ("v_fract_f32_e64 v0, -v1", F_1P75, False, True),       # -(1.75) -> 0.25
            ("v_fract_f32_e64 v0, -v1", F_POS_ZERO, False, True),   # -(0.0) -> +0.0
            ("v_fract_f32_e64 v0, -v1", F_1P0, False, True),        # -(1.0) -> +0.0
            ("v_fract_f32_e64 v0, |v1|", F_NEG_0P25, True, False),  # |-0.25| = 0.25 -> 0.25
            ("v_fract_f32_e64 v0, |v1|", F_NEG_1P75, True, False),  # |-1.75| = 1.75 -> 0.75
            ("v_fract_f32_e64 v0, -|v1|", F_0P25, True, True),      # -|0.25| = -0.25 -> 0.75
            ("v_fract_f32_e64 v0, -|v1|", F_NEG_0P25, True, True),  # -|-0.25| = -0.25 -> 0.75
            ("v_fract_f32_e64 v0, -|v1|", F_1P75, True, True),      # -|1.75| = -1.75 -> 0.25
        ]
        for line, w, abs_, neg in cases:
            with self.subTest(line=line, w=hex(w)):
                want = oracle_fract_f32(w, abs_=abs_, neg=neg)
                self.assertEqual(run_op(line, w), want)

    def test_vop3_output_modifiers(self):
        cases = [
            ("clamp", F_0P75, dict(clamp=True)),
            ("clamp", F_NEG_0P25, dict(clamp=True)),
            ("clamp", F_NEG_ZERO, dict(clamp=True)),
            ("mul:2", F_0P5, dict(mul=2)),
            ("mul:2", F_0P75, dict(mul=2)),
            ("mul:2", F_NEG_0P25, dict(mul=2)),
            ("mul:4", F_0P25, dict(mul=4)),
            ("mul:4", F_0P5, dict(mul=4)),
            ("mul:4", F_NEG_1P25, dict(mul=4)),
            ("div:2", F_0P5, dict(div=2)),
            ("div:2", F_NEG_1P25, dict(div=2)),
            ("div:2", F_0P25, dict(div=2)),
            ("mul:2 clamp", F_0P25, dict(mul=2, clamp=True)),
            ("mul:2 clamp", F_0P5, dict(mul=2, clamp=True)),
            ("mul:2 clamp", F_0P75, dict(mul=2, clamp=True)),
            ("mul:2 clamp", F_NEG_0P25, dict(mul=2, clamp=True)),
            ("mul:4 clamp", F_0P5, dict(mul=4, clamp=True)),
        ]
        for suffix, w, kw in cases:
            line = f"v_fract_f32_e64 v0, v1 {suffix}"
            want = oracle_fract_f32(w, **kw)
            with self.subTest(line=line, w=hex(w)):
                self.assertEqual(run_op(line, w), want)

        # Explicit verification of key omod scaling values
        self.assertEqual(run_op("v_fract_f32_e64 v0, v1 mul:2", F_0P5), F_1P0)
        self.assertEqual(run_op("v_fract_f32_e64 v0, v1 mul:2 clamp", F_0P75), F_1P0)
        self.assertEqual(run_op("v_fract_f32_e64 v0, v1 mul:4", F_0P5), F_2P0)
        self.assertEqual(run_op("v_fract_f32_e64 v0, v1 div:2", F_0P5), F_0P25)

    def test_inline_float_constants(self):
        for lit, w in (("0.5", F_0P5), ("-0.5", F_NEG_0P5), ("1.0", F_1P0), ("-1.0", F_NEG_1P0)):
            want = oracle_fract_f32(w)
            for form in ("_e32", "_e64", ""):
                with self.subTest(lit=lit, form=form or "bare"):
                    self.assertEqual(run_op(f"v_fract_f32{form} v0, {lit}"), want)

        self.assertEqual(run_op("v_fract_f32_e64 v0, -0.5 mul:2"), F_1P0)
        self.assertEqual(run_op("v_fract_f32_e64 v0, -0.5 div:2"), F_0P25)

    def test_sgpr_sources_and_destination_aliasing(self):
        s4 = F_NEG_0P25
        self.assertEqual(run_op("v_fract_f32_e32 v0, s4", s4=s4), F_0P75)
        self.assertEqual(run_op("v_fract_f32_e64 v0, s4", s4=s4), F_0P75)
        self.assertEqual(run_op("v_fract_f32_e64 v0, -s4", s4=s4), F_0P25)
        self.assertEqual(run_op("v_fract_f32_e64 v0, |s4|", s4=s4), F_0P25)

        # Destination aliasing: dst == src (v0, v0)
        self.assertEqual(
            run_ps([f"v_mov_b32 v0, {F_NEG_0P25:#x}", "v_fract_f32_e32 v0, v0"]),
            F_0P75,
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v0, {F_3P75:#x}", "v_fract_f32 v0, v0"]),
            F_0P75,
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v0, {F_NEG_1P75:#x}", "v_fract_f32_e64 v0, v0"]),
            F_0P25,
        )

    def test_runtime_memory_inputs_via_s_buffer_load(self):
        lanes = [F_NEG_0P25, F_3P75, F_NEG_1P75, F_POS_ZERO]
        mem = {0x1000 + 4 * i: w for i, w in enumerate(lanes)}
        for idx, w in enumerate(lanes):
            reg = f"v{idx + 1}"
            lines = [
                "s_buffer_load_dwordx4 s[4:7], s[0:3], 0",
                "s_waitcnt lgkmcnt(0)",
                "v_mov_b32_e32 v1, s4",
                "v_mov_b32_e32 v2, s5",
                "v_mov_b32_e32 v3, s6",
                "v_mov_b32_e32 v4, s7",
                f"v_mov_b32_e32 v0, {OLD:#x}",
                f"v_fract_f32_e32 v0, {reg}",
            ]
            got = run_ps(lines, push_constants=PUSH_ADDR, memory=mem)
            self.assertEqual(got, oracle_fract_f32(w))


class FractF16AndEdgeCaseExecutionTests(unittest.TestCase):
    """Execution tests for v_fract_f16, op_sel, and joint IEEE-754 edge cases."""

    def test_f16_basic_execution_and_opsel(self):
        samples = [
            H_POS_ZERO, H_NEG_ZERO,
            H_0P125, H_0P25, H_NEG_0P25, H_0P375,
            H_0P5, H_NEG_0P5, H_0P75, H_NEG_0P75,
            H_NEAR_ONE, H_1P0, H_NEG_1P0,
            H_1P25, H_NEG_1P25, H_1P5, H_NEG_1P5, H_1P75, H_NEG_1P75,
            H_2P0, H_NEG_2P0, H_2P5, H_NEG_2P5, H_3P0, H_NEG_3P0,
            H_3P75, H_NEG_3P75, H_1023P5, H_1024, H_NEG_1024,
            H_MAX, H_NEG_MAX,
        ]
        for h in samples:
            want = oracle_fract_f16(h)
            v1_lo = 0x55550000 | h
            self.assertEqual(run_op("v_fract_f16_e32 v0, v1", v1_lo), merge_half(OLD, want, 0))
            self.assertEqual(run_op("v_fract_f16 v0, v1", v1_lo), merge_half(OLD, want, 0))

        # Test op_sel across all 4 combinations
        v1 = (H_NEG_1P75 << 16) | H_NEG_0P25  # hi: -1.75 -> 0.25; lo: -0.25 -> 0.75
        for src_hi, dst_hi in itertools.product((0, 1), repeat=2):
            h_in = H_NEG_1P75 if src_hi else H_NEG_0P25
            want_h = oracle_fract_f16(h_in)
            line = f"v_fract_f16_e64 v0, v1 op_sel:[{src_hi},{dst_hi}]"
            with self.subTest(src_hi=src_hi, dst_hi=dst_hi):
                self.assertEqual(run_op(line, v1), merge_half(OLD, want_h, dst_hi))

    def test_f16_negative_number_fractional_part(self):
        # Specific assertions for half-precision: fract(-0.25) == 0.75, fract(-1.75) == 0.25
        v_lo_0p25 = 0xAAAA0000 | H_NEG_0P25
        v_lo_1p75 = 0xAAAA0000 | H_NEG_1P75
        self.assertEqual(run_op("v_fract_f16_e32 v0, v1", v_lo_0p25), merge_half(OLD, H_0P75, 0))
        self.assertEqual(run_op("v_fract_f16_e32 v0, v1", v_lo_1p75), merge_half(OLD, H_0P25, 0))

    def test_f16_mantissa_exhaustion_execution(self):
        # Numbers >= 1024.0 in f16 yield 0.0, while 1023.5 yields 0.5
        self.assertEqual(run_op("v_fract_f16_e32 v0, v1", H_1024), merge_half(OLD, H_POS_ZERO, 0))
        self.assertEqual(run_op("v_fract_f16_e32 v0, v1", H_NEG_1024), merge_half(OLD, H_POS_ZERO, 0))
        self.assertEqual(run_op("v_fract_f16_e32 v0, v1", H_MAX), merge_half(OLD, H_POS_ZERO, 0))
        self.assertEqual(run_op("v_fract_f16_e32 v0, v1", H_NEG_MAX), merge_half(OLD, H_POS_ZERO, 0))
        self.assertEqual(run_op("v_fract_f16_e32 v0, v1", H_1023P5), merge_half(OLD, H_0P5, 0))

    def test_f16_vop3_modifiers(self):
        # Source modifiers
        v1 = (H_0P25 << 16) | H_NEG_0P25
        self.assertEqual(
            run_op("v_fract_f16_e64 v0, -v1 op_sel:[0,0]", v1),
            merge_half(OLD, H_0P25, 0),  # -(-0.25) = 0.25 -> 0.25
        )
        self.assertEqual(
            run_op("v_fract_f16_e64 v0, |v1| op_sel:[0,0]", v1),
            merge_half(OLD, H_0P25, 0),  # |-0.25| = 0.25 -> 0.25
        )
        # Output modifiers: on RDNA2 with denorms kept, omod is ignored on f16.
        self.assertEqual(
            run_op("v_fract_f16_e64 v0, v1 mul:2", H_0P5),
            merge_half(OLD, H_0P5, 0),
        )
        self.assertEqual(
            run_op("v_fract_f16_e64 v0, v1 div:2", H_0P5),
            merge_half(OLD, H_0P5, 0),
        )

    def test_ieee754_zero_sign_clearing_joint(self):
        # Both f32 and f16 with -0.0 must return +0.0 (sign bit cleared)
        got_f32 = run_op("v_fract_f32_e32 v0, v1", F_NEG_ZERO)
        self.assertEqual(got_f32, F_POS_ZERO)
        self.assertEqual(got_f32 & 0x80000000, 0)

        got_f16 = run_op("v_fract_f16_e32 v0, v1", H_NEG_ZERO)
        self.assertEqual(got_f16 & 0xFFFF, H_POS_ZERO)
        self.assertEqual((got_f16 & 0x8000), 0)

    def test_ieee754_inf_nan_joint(self):
        # Both f32 and f16 with +-inf or NaN must return NaN
        for w in (F_POS_INF, F_NEG_INF, F_QNAN, F_SNAN):
            got = run_op("v_fract_f32_e32 v0, v1", w)
            self.assertTrue(f32_is_nan(got), f"expected f32 NaN for {w:#010x}")

        for h in (H_POS_INF, H_NEG_INF, H_QNAN, H_SNAN):
            got = run_op("v_fract_f16_e32 v0, v1", h)
            self.assertTrue(h16_is_nan(got & 0xFFFF), f"expected f16 NaN for {h:#06x}")

    def test_joint_f32_f16_shader_pipeline(self):
        lines = [
            f"v_mov_b32 v1, {F_NEG_0P25:#x}",
            f"v_mov_b32 v2, {H_NEG_1P75:#x}",
            "v_fract_f32_e32 v0, v1",
            "v_fract_f16_e32 v3, v2",
            "exp mrt0 v0, v3, v0, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_ps(lines)
        out = si.run_fragment(spv)["%out_mrt0"]
        self.assertEqual(out[0], F_0P75)
        self.assertEqual(out[1] & 0xFFFF, H_0P25)
        self.assertEqual(out[2], F_0P75)
        self.assertEqual(out[3] & 0xFFFF, H_0P25)


# -----------------------------------------------------------------------------
# Subnormal Bounds, Negative Controls, and SIMD Tests
# -----------------------------------------------------------------------------

class SubnormalAndRoundingBoundaryTests(unittest.TestCase):
    """Audits subnormal numbers, boundary rounding transitions, and clamp behavior."""

    def test_f32_positive_subnormals(self):
        # A subnormal operand reads as zero (f32 denormals flushed, docs/design.md), so fract gives +0; the smallest
        # normal is unaffected and gives itself.
        cases = [
            F_SUB_MIN,
            F_SUB_1E40,
            F_SUB_MAX,
            F_NORM_MIN,
        ]
        for w in cases:
            want = w if w == F_NORM_MIN else 0
            with self.subTest(w=hex(w)):
                self.assertEqual(oracle_fract_f32(w), want)
                self.assertEqual(run_op("v_fract_f32_e32 v0, v1", w), want)
                self.assertEqual(run_op("v_fract_f32 v0, v1", w), want)

    def test_f32_tiny_negatives_and_rounding_transition(self):
        # For x in (-2^-24, 0), exact x - floor(x) rounds to 1.0 in binary32.
        # Under RDNA2 hardware semantics and fract_below_one, this must return 0x3F7FFFFF (never 1.0).
        cases = [
            (F_NEG_SUB_MIN, F_POS_ZERO),   # denormal operands read as zero: fract(0) = +0
            (F_NEG_SUB_1E40, F_POS_ZERO),
            (F_NEG_SUB_MAX, F_POS_ZERO),
            (F_NEG_NORM_MIN, F_NEAR_ONE),
            (F_NEG_1E10, F_NEAR_ONE),
            (F_NEG_2PM26, F_NEAR_ONE),
            (F_NEG_2PM25, F_NEAR_ONE),     # exact tie -> rounds to 1.0 -> clamped to 0x3F7FFFFF
            (F_NEG_2PM24, F_NEAR_ONE),     # 1 - 2^-24 = 0x3F7FFFFF exact
            (F_NEG_2PM23, 0x3F7FFFFE),     # 1 - 2^-23 = 0x3F7FFFFE exact
        ]
        for inp, want in cases:
            with self.subTest(inp=hex(inp)):
                self.assertEqual(oracle_fract_f32(inp), want)
                self.assertEqual(run_op("v_fract_f32_e32 v0, v1", inp), want)
                self.assertEqual(run_op("v_fract_f32_e64 v0, v1", inp), want)

    def test_f16_subnormals_and_boundary(self):
        # Positive subnormals retain exact bits; negative subnormals clamp to 0x3BFF.
        self.assertEqual(run_op("v_fract_f16_e32 v0, v1", H_SUB_MIN), merge_half(OLD, H_SUB_MIN, 0))
        self.assertEqual(run_op("v_fract_f16_e32 v0, v1", H_SUB_MAX), merge_half(OLD, H_SUB_MAX, 0))
        self.assertEqual(run_op("v_fract_f16_e32 v0, v1", H_NEG_SUB_MIN), merge_half(OLD, H_NEAR_ONE, 0))
        self.assertEqual(run_op("v_fract_f16_e32 v0, v1", H_NEG_SUB_MAX), merge_half(OLD, H_NEAR_ONE, 0))


class NegativeControlSabotageTests(unittest.TestCase):
    """Validates that test assertions detect invalid SPIR-V translations or omitted clamps."""

    def test_sabotage_fract_to_trunc(self):
        # fract(-1.25) == 0.75; trunc(-1.25) == -1.0 (0xBF800000)
        spv = translate_ps([
            f"v_mov_b32 v1, {F_NEG_1P25:#x}",
            "v_fract_f32_e32 v0, v1",
        ])
        expected = F_0P75
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], expected)

        sabotaged, count = re.subn(r"\bFract\b", "Trunc", spv, count=1)
        self.assertEqual(count, 1)
        got = si.run_fragment(sabotaged)["%out_mrt0"][0]
        self.assertNotEqual(got, expected)
        self.assertEqual(got, F_NEG_1P0)

    def test_sabotage_fract_to_floor(self):
        # fract(-1.25) == 0.75; floor(-1.25) == -2.0 (0xC0000000)
        spv = translate_ps([
            f"v_mov_b32 v1, {F_NEG_1P25:#x}",
            "v_fract_f32_e32 v0, v1",
        ])
        expected = F_0P75
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], expected)

        sabotaged, count = re.subn(r"\bFract\b", "Floor", spv, count=1)
        self.assertEqual(count, 1)
        got = si.run_fragment(sabotaged)["%out_mrt0"][0]
        self.assertNotEqual(got, expected)
        self.assertEqual(got, F_NEG_2P0)

    def test_sabotage_clamp_selection(self):
        # For x = -1e-10, GLSL.std.450 Fract rounds to 1.0 (0x3F800000).
        # Without OpSelect clamp to 0x3F7FFFFF, result incorrectly violates [0.0, 1.0).
        spv = translate_ps([
            f"v_mov_b32 v1, {F_NEG_1E10:#x}",
            "v_fract_f32_e32 v0, v1",
        ])
        expected = F_NEAR_ONE
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], expected)

        # Bypass the OpSelect clamp by replacing OpSelect %float %cond %clamped %raw with OpCopyObject %float %raw
        sabotaged, count = re.subn(r"OpSelect %float %\w+ %\w+ (%\w+)", r"OpCopyObject %float \1", spv, count=1)
        self.assertEqual(count, 1)
        got = si.run_fragment(sabotaged)["%out_mrt0"][0]
        self.assertNotEqual(got, expected)
        self.assertEqual(got, F_1P0)


class SimdDivergenceAndMaskingTests(unittest.TestCase):
    """Audits multi-lane SIMD divergence, thread masking, and inactive lane preservation."""

    def test_single_lane_exec_masking(self):
        # exec_lo = 0 masks off execution -> pre-existing OLD preserved
        self.assertEqual(run_ps([
            f"v_mov_b32 v1, {F_1P5:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "s_mov_b32 exec_lo, 0",
            "v_fract_f32_e32 v0, v1",
            "s_mov_b32 exec_lo, 1",
        ]), OLD)

        # re-enabling exec_lo executes normally
        self.assertEqual(run_ps([
            f"v_mov_b32 v1, {F_NEG_1P75:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "s_mov_b32 exec_lo, 0",
            "v_fract_f32_e32 v0, v1",
            "s_mov_b32 exec_lo, 1",
            "v_fract_f32_e32 v0, v1",
        ]), F_0P25)

    def test_multi_lane_divergence_8_threads(self):
        # 8 compute threads: even threads active, odd threads masked off.
        inputs = [
            F_NEG_0P25,    # lane 0 (active): -0.25 -> 0.75
            F_NEG_1P75,    # lane 1 (masked off)
            F_NEG_1E10,    # lane 2 (active): -1e-10 -> 0x3F7FFFFF
            F_3P75,        # lane 3 (masked off)
            F_42P0,        # lane 4 (active): 42.0 -> +0.0
            F_2P23,        # lane 5 (masked off)
            F_2P23_M0P5,   # lane 6 (active): 2^23 - 0.5 -> 0.5
            F_NEG_ZERO,    # lane 7 (masked off)
        ]
        lane_olds = [0xD0000000 + 0x01010101 * i for i in range(8)]
        mem = _Heap(
            {0x100 + 4 * i: w for i, w in enumerate(inputs)}
            | {0x200 + 4 * i: w for i, w in enumerate(lane_olds)}
        )

        run_cs([
            "v_lshlrev_b32_e32 v5, 2, v0",
            "v_add_nc_u32_e32 v6, 0x100, v5",
            "v_add_nc_u32_e32 v7, 0x200, v5",
            "buffer_load_dword v1, v6, s[0:3], 0 offen",
            "buffer_load_dword v2, v7, s[0:3], 0 offen",
            "v_and_b32_e32 v3, 1, v0",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v3",
            "s_and_saveexec_b32 s6, vcc_lo",
            "v_fract_f32_e32 v2, v1",
            "s_mov_b32 exec_lo, s6",
            "buffer_store_dword v2, v5, s[0:3], 0 offen",
        ], group_size=8, memory=mem)

        got = [mem.get(4 * i) for i in range(8)]
        want = []
        for lane, (inp, old_val) in enumerate(zip(inputs, lane_olds)):
            if lane % 2 == 1:
                want.append(old_val)
            else:
                want.append(oracle_fract_f32(inp))
        self.assertEqual(got, want)

    def test_wavefront_32_threads_divergence(self):
        inputs = [
            F_NEG_0P25, F_NEG_1P75, F_0P25, F_0P75,
            F_1P25, F_NEG_1P25, F_2P5, F_NEG_2P5,
            F_3P75, F_NEG_3P75, F_42P0, F_NEG_42P0,
            F_POS_ZERO, F_NEG_ZERO, F_1P0, F_NEG_1P0,
            F_2P0, F_NEG_2P0, F_3P0, F_NEG_3P0,
            F_1023P5, F_1024P0, F_1024P25, F_2P23_M0P5,
            F_2P23, F_NEG_2P23, F_2P24, F_NEG_2P24,
            F_NEAR_ONE, 0x3F7FFFFE, F_SUB_MIN, F_NEG_SUB_MIN,
        ]
        olds = [0xA0000000 + i for i in range(32)]
        mem = _Heap(
            {0x100 + 4 * i: inputs[i] for i in range(32)}
            | {0x200 + 4 * i: olds[i] for i in range(32)}
        )

        run_cs([
            "v_lshlrev_b32_e32 v5, 2, v0",
            "v_add_nc_u32_e32 v6, 0x100, v5",
            "v_add_nc_u32_e32 v7, 0x200, v5",
            "buffer_load_dword v1, v6, s[0:3], 0 offen",
            "buffer_load_dword v2, v7, s[0:3], 0 offen",
            "v_and_b32_e32 v3, 1, v0",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v3",
            "s_and_saveexec_b32 s6, vcc_lo",
            "v_fract_f32_e32 v2, v1",
            "s_mov_b32 exec_lo, s6",
            "buffer_store_dword v2, v5, s[0:3], 0 offen",
        ], group_size=32, memory=mem)

        got = [mem.get(4 * i) for i in range(32)]
        want = []
        for i in range(32):
            if i % 2 == 1:
                want.append(olds[i])
            else:
                want.append(oracle_fract_f32(inputs[i]))
        self.assertEqual(got, want)

    def test_joint_f32_f16_simd_divergence(self):
        # 16 threads executing both v_fract_f32 and v_fract_f16 under divergence
        f32_inputs = [F_NEG_0P25, F_1P75, F_3P75, F_NEG_1P25] * 4
        f16_inputs = [H_NEG_1P75, H_0P25, H_NEG_0P25, H_1023P5] * 4
        lane_olds = [0x55000000 + i for i in range(16)]

        mem = _Heap(
            {0x100 + 4 * i: f32_inputs[i] for i in range(16)}
            | {0x200 + 4 * i: f16_inputs[i] for i in range(16)}
            | {0x300 + 4 * i: lane_olds[i] for i in range(16)}
        )

        run_cs([
            "v_lshlrev_b32_e32 v5, 2, v0",
            "v_add_nc_u32_e32 v6, 0x100, v5",
            "v_add_nc_u32_e32 v7, 0x200, v5",
            "v_add_nc_u32_e32 v8, 0x300, v5",
            "buffer_load_dword v1, v6, s[0:3], 0 offen",
            "buffer_load_dword v2, v7, s[0:3], 0 offen",
            "buffer_load_dword v3, v8, s[0:3], 0 offen",
            "v_and_b32_e32 v4, 1, v0",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v4",
            "s_and_saveexec_b32 s6, vcc_lo",
            "v_fract_f32_e32 v1, v1",
            "v_fract_f16_e32 v2, v2",
            "v_and_b32_e32 v2, 0xffff, v2",
            "v_add_nc_u32_e32 v3, v1, v2",
            "s_mov_b32 exec_lo, s6",
            "buffer_store_dword v3, v5, s[0:3], 0 offen",
        ], group_size=16, memory=mem)

        got = [mem.get(4 * i) for i in range(16)]
        want = []
        for i in range(16):
            if i % 2 == 1:
                want.append(lane_olds[i])
            else:
                f32_res = oracle_fract_f32(f32_inputs[i])
                f16_res = oracle_fract_f16(f16_inputs[i])
                want.append((f32_res + f16_res) & 0xFFFFFFFF)
        self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
