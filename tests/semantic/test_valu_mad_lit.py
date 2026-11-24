"""Semantic value tests for RDNA2 literal-operand f32 multiply-add VALU ops:
  v_madak_f32  vdst, src0, vsrc1, K   ->  D = f32(f32(src0 * vsrc1) + K)   (unfused, NoContraction)
  v_madmk_f32  vdst, src0, K, vsrc1   ->  D = f32(f32(src0 * K) + vsrc1)   (unfused, NoContraction)

Both are GFX10.1 VOP2-only instructions that carry an inline 32-bit literal K (no VOP3 source/output modifiers).
In LLVM AMDGPU disassembly print order, both compute a[1] * a[2] + a[3], with the literal K occupying the addend
slot in v_madak_f32 and the second multiplicand slot in v_madmk_f32.

Each test builds a hand-written RDNA2 pixel or compute shader listing (stdlib only, zero game data), translates it
with tools/rdna2spirv.py, executes the SPIR-V in tests/semantic/spirv_mini_interp.py, and checks the output words
against an independent plain-Python IEEE-754 binary32 Fraction reference oracle.
"""
from __future__ import annotations

from fractions import Fraction
import gc
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

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

# Hand-built compute header (RSRC2: 4 user SGPRs, tid=x, group 4x1x1) matching test_valu_f32_min3_max3_mad.py.
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 4, 1, 1)
CS_FSSM_BYTES = bytes(CS_FSSM)

# Canonical binary32 bit patterns used across tests.
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
FLT_MIN = 0x00800000           # 2^-126
NEG_FLT_MIN = 0x80800000       # -2^-126
QUARTER = 0x3E800000           # 0.25
HALF = 0x3F000000              # 0.5
NEG_HALF = 0xBF000000          # -0.5
ONE = 0x3F800000               # 1.0
NEG_ONE = 0xBF800000           # -1.0
ONE_HALF = 0x3FC00000          # 1.5
NEG_ONE_HALF = 0xBFC00000      # -1.5
TWO = 0x40000000               # 2.0
NEG_TWO = 0xC0000000           # -2.0
TWO_HALF = 0x40200000          # 2.5
THREE = 0x40400000             # 3.0
NEG_THREE = 0xC0400000         # -3.0
FOUR = 0x40800000              # 4.0
FIVE = 0x40A00000              # 5.0
SIX = 0x40C00000               # 6.0
FLT_MAX = 0x7F7FFFFF           # (2 - 2^-23) * 2^127
NEG_FLT_MAX = 0xFF7FFFFF       # -(2 - 2^-23) * 2^127
POS_INF = 0x7F800000           # +inf
NEG_INF = 0xFF800000           # -inf
QNAN = 0x7FC00000              # canonical positive qNaN
NEG_QNAN = 0xFFC00000          # negative qNaN
SNAN = 0x7F800001              # signalling NaN
INV_TWO_PI = 0x3E22F983        # 0.15915494 (RDNA inline constant 1/(2*pi))


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def is_nan_bits(bits: int) -> bool:
    bits &= MASK32
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def exact_f32_to_fraction(bits: int) -> Fraction:
    """Decode a finite binary32 bit pattern into an exact rational Fraction."""
    bits &= MASK32
    sign = -1 if (bits >> 31) else 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    if exp == 0xFF:
        raise ValueError(f"non-finite bits {bits:#010x}")
    if exp == 0:
        return Fraction(sign * frac, 1 << 149)
    mant = (1 << 23) | frac
    shift = exp - 150
    if shift >= 0:
        return Fraction(sign * (mant << shift), 1)
    return Fraction(sign * mant, 1 << (-shift))


def round_fraction_to_f32_bits(val: Fraction, zero_sign: int = 0) -> int:
    """Round an exact Fraction to IEEE-754 binary32 (round-to-nearest-even)."""
    if val == 0:
        return NEG_ZERO if zero_sign else POS_ZERO
    sign = 1 if val < 0 else 0
    aval = -val if sign else val
    num, den = aval.numerator, aval.denominator
    e = num.bit_length() - den.bit_length()
    if (num << max(0, -e)) < (den << max(0, e)):
        e -= 1

    if e >= 128:
        return (sign << 31) | POS_INF

    target_shift = 149 if e < -126 else 23 - e
    if target_shift >= 0:
        scaled_num = num << target_shift
    else:
        scaled_num = num
        den = den << (-target_shift)

    q, rem = divmod(scaled_num, den)
    twice_rem = rem << 1
    if twice_rem > den or (twice_rem == den and (q & 1)):
        q += 1

    if e < -126:
        if q >= (1 << 23):
            return (sign << 31) | (1 << 23)
        return (sign << 31) | q

    if q >= (1 << 24):
        q >>= 1
        e += 1
        if e >= 128:
            return (sign << 31) | POS_INF

    return (sign << 31) | ((e + 127) << 23) | (q & 0x7FFFFF)


def _zero_sign_bit(bits: int) -> int:
    return 1 if (bits & MASK32) == NEG_ZERO else 0


def _mul_bits_f32(xb: int, yb: int) -> int:
    x, y = bits_to_f32(xb), bits_to_f32(yb)
    if math.isnan(x) or math.isnan(y):
        return QNAN
    if (x == 0.0 and math.isinf(y)) or (math.isinf(x) and y == 0.0):
        return QNAN
    sign = ((xb ^ yb) >> 31) & 1
    if math.isinf(x) or math.isinf(y):
        return (sign << 31) | POS_INF
    if x == 0.0 or y == 0.0:
        return NEG_ZERO if sign else POS_ZERO
    return round_fraction_to_f32_bits(exact_f32_to_fraction(xb) * exact_f32_to_fraction(yb), zero_sign=sign)


def _add_bits_f32(ab: int, bb: int) -> int:
    a, b = bits_to_f32(ab), bits_to_f32(bb)
    if math.isnan(a) or math.isnan(b):
        return QNAN
    if math.isinf(a) and math.isinf(b):
        if (ab >> 31) != (bb >> 31):
            return QNAN
        return ab & MASK32
    if math.isinf(a):
        return ab & MASK32
    if math.isinf(b):
        return bb & MASK32
    if a == 0.0 and b == 0.0:
        return NEG_ZERO if (_zero_sign_bit(ab) and _zero_sign_bit(bb)) else POS_ZERO
    total = exact_f32_to_fraction(ab) + exact_f32_to_fraction(bb)
    return round_fraction_to_f32_bits(total, zero_sign=0)


def ref_madak_f32(src0_bits: int, src1_bits: int, k_bits: int) -> int:
    """Reference for v_madak_f32 vdst, src0, vsrc1, K -> f32(f32(src0 * vsrc1) + K)."""
    return _add_bits_f32(_mul_bits_f32(src0_bits, src1_bits), k_bits)


def ref_madmk_f32(src0_bits: int, k_bits: int, src1_bits: int) -> int:
    """Reference for v_madmk_f32 vdst, src0, K, vsrc1 -> f32(f32(src0 * K) + vsrc1)."""
    return _add_bits_f32(_mul_bits_f32(src0_bits, k_bits), src1_bits)


def ref_fma_f32(xb: int, yb: int, zb: int) -> int:
    """True IEEE-754 binary32 fused multiply-add: fma(x, y, z) (used to verify unfused divergence)."""
    x, y, z = bits_to_f32(xb), bits_to_f32(yb), bits_to_f32(zb)
    if math.isnan(x) or math.isnan(y) or math.isnan(z):
        return QNAN
    if (x == 0.0 and math.isinf(y)) or (math.isinf(x) and y == 0.0):
        return QNAN
    prod_sign = ((xb ^ yb) >> 31) & 1
    if math.isinf(x) or math.isinf(y):
        return _add_bits_f32((prod_sign << 31) | POS_INF, zb)
    if math.isinf(z):
        return zb & MASK32
    total = exact_f32_to_fraction(xb) * exact_f32_to_fraction(yb) + exact_f32_to_fraction(zb)
    if total == 0:
        if (x == 0.0 or y == 0.0) and z == 0.0 and prod_sign == 1 and _zero_sign_bit(zb) == 1:
            return NEG_ZERO
        return POS_ZERO
    return round_fraction_to_f32_bits(total, zero_sign=0)


_TMP_DIR = None
_PS_PATH = None
_CS_PATH = None


def setUpModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    fssm_dir = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)
    _PS_PATH = os.path.join(disasm_dir, "ps_mad_lit.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_mad_lit.s")
    with open(os.path.join(_TMP_DIR.name, "ps_mad_lit.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(os.path.join(fssm_dir, "cs_mad_lit.fssm"), "wb") as f:
        f.write(CS_FSSM_BYTES)


def tearDownModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _PS_PATH = None
        _CS_PATH = None


def translate_ps(lines: list[str], export_regs: tuple[str, str, str, str] = ("v0", "v0", "v0", "v0")) -> str:
    if _TMP_DIR is None:
        setUpModule()
    assert _PS_PATH is not None
    asm = (
        "; test_valu_mad_lit\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + f"\texp mrt0 {', '.join(export_regs)} done vm\n"
        + "\ts_endpgm\n"
    )
    with open(_PS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
        gc.collect()
    return spv


def run_ps_mrt0(
    lines: list[str],
    export_regs: tuple[str, str, str, str] = ("v0", "v0", "v0", "v0"),
    **run_options,
) -> list[int]:
    spv = translate_ps(lines, export_regs=export_regs)
    return list(si.run_fragment(spv, **run_options)["%out_mrt0"])


def run_madak(src0_bits: int, src1_bits: int, k_bits: int, op: str = "v_madak_f32") -> int:
    return run_ps_mrt0([
        f"v_mov_b32 v1, {src0_bits & MASK32:#x}",
        f"v_mov_b32 v2, {src1_bits & MASK32:#x}",
        f"{op} v0, v1, v2, {k_bits & MASK32:#x}",
    ])[0]


def run_madmk(src0_bits: int, k_bits: int, src1_bits: int, op: str = "v_madmk_f32") -> int:
    return run_ps_mrt0([
        f"v_mov_b32 v1, {src0_bits & MASK32:#x}",
        f"v_mov_b32 v2, {src1_bits & MASK32:#x}",
        f"{op} v0, v1, {k_bits & MASK32:#x}, v2",
    ])[0]


def run_cs_workgroup_exec(lines: list[str], lanes: int = 4) -> dict[int, int]:
    """Translate and run a compute shader storing `v1` per lane at byte address `4 * lane`."""
    if _TMP_DIR is None:
        setUpModule()
    assert _CS_PATH is not None
    asm = "; test_valu_mad_lit_cs\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_CS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_CS_PATH)
        gc.collect()
    mem: dict[int, int] = {}
    si.run_workgroup(
        spv,
        lanes,
        32,
        inputs_for=lambda i: {"%local_id": (i, 0, 0), "%wg_id": (0, 0, 0)},
        push_constants=[0, 0, 64, 0x31016FAC] + [0] * 28,
        memory=mem,
    )
    return mem


class ReferenceOracleTests(unittest.TestCase):
    """Cross-check the plain-Python IEEE-754 Fraction oracle against hand-derived binary32 constants."""

    def test_hand_worked_constants(self):
        # Asymmetric operands prove operand order: madak = 2*3 + 5 = 11, madmk = 2*5 + 3 = 13.
        self.assertEqual(ref_madak_f32(TWO, THREE, FIVE), f32_to_bits(11.0))
        self.assertEqual(ref_madmk_f32(TWO, FIVE, THREE), f32_to_bits(13.0))
        self.assertNotEqual(ref_madak_f32(TWO, THREE, FIVE), ref_madmk_f32(TWO, FIVE, THREE))

        # Negative and fractional values: (-1.5) * 2.5 + (-0.5) = -4.25; (-1.5) * (-0.5) + 2.5 = 3.25.
        self.assertEqual(ref_madak_f32(NEG_ONE_HALF, TWO_HALF, NEG_HALF), f32_to_bits(-4.25))
        self.assertEqual(ref_madmk_f32(NEG_ONE_HALF, NEG_HALF, TWO_HALF), f32_to_bits(3.25))

        # Signed zeros: (-0)*2 + (-0) = -0; (+0)*2 + (-0) = +0; exact cancellation -> +0.
        self.assertEqual(ref_madak_f32(NEG_ZERO, TWO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(ref_madmk_f32(NEG_ZERO, TWO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(ref_madak_f32(POS_ZERO, TWO, NEG_ZERO), POS_ZERO)
        self.assertEqual(ref_madmk_f32(POS_ZERO, TWO, NEG_ZERO), POS_ZERO)
        self.assertEqual(ref_madak_f32(NEG_TWO, THREE, SIX), POS_ZERO)
        self.assertEqual(ref_madmk_f32(NEG_TWO, THREE, SIX), POS_ZERO)

        # Unfused vs fused divergence: (1 + 2^-20)*(1 - 2^-20) - 1 = +0.0 (unfused) vs -2^-40 (fused).
        a, b = 0x3F800008, 0x3F7FFFF0
        self.assertEqual(ref_madak_f32(a, b, NEG_ONE), POS_ZERO)
        self.assertEqual(ref_madmk_f32(a, b, NEG_ONE), POS_ZERO)
        self.assertEqual(ref_fma_f32(a, b, NEG_ONE), 0xAB800000)

        # Intermediate overflow: 2^100 * 2^28 - 2^127 = +inf (unfused) vs 2^127 (fused).
        p100, p28, n127 = 0x71800000, 0x4D800000, 0xFF000000
        self.assertEqual(ref_madak_f32(p100, p28, n127), POS_INF)
        self.assertEqual(ref_madmk_f32(p100, p28, n127), POS_INF)
        self.assertEqual(ref_fma_f32(p100, p28, n127), 0x7F000000)


class MadLitBasicAndOrderingTests(unittest.TestCase):
    """Normal floats, fractions, zeros, _e32 spellings, and operand ordering distinction."""

    def test_operand_ordering_distinction(self):
        # For the exact same (src0, vsrc1, K), madak computes src0 * vsrc1 + K while madmk computes src0 * K + vsrc1.
        cases = [
            # (src0, vsrc1, K)
            (TWO, THREE, FIVE),                     # madak = 2*3+5 = 11.0;    madmk = 2*5+3 = 13.0
            (FOUR, FIVE, f32_to_bits(1.5)),         # madak = 4*5+1.5 = 21.5;  madmk = 4*1.5+5 = 11.0
            (NEG_ONE_HALF, TWO_HALF, NEG_HALF),     # madak = -4.25;           madmk = 3.25
            (THREE, POS_ZERO, FOUR),                # madak = 0+4 = 4.0;       madmk = 12+0 = 12.0
            (THREE, FOUR, POS_ZERO),                # madak = 12+0 = 12.0;     madmk = 0+4 = 4.0
            (NEG_TWO, QUARTER, ONE_HALF),           # madak = -0.5+1.5 = 1.0;  madmk = -3.0+0.25 = -2.75
            (f32_to_bits(0.1), f32_to_bits(7.0), f32_to_bits(-3.25)),
        ]
        for s0, s1, k in cases:
            want_ak = ref_madak_f32(s0, s1, k)
            want_mk = ref_madmk_f32(s0, k, s1)
            self.assertNotEqual(want_ak, want_mk, "test triple must distinguish madak from madmk")
            with self.subTest(s0=hex(s0), s1=hex(s1), k=hex(k)):
                words = run_ps_mrt0([
                    f"v_mov_b32 v1, {s0:#x}",
                    f"v_mov_b32 v2, {s1:#x}",
                    f"v_madak_f32 v10, v1, v2, {k:#x}",
                    f"v_madmk_f32 v11, v1, {k:#x}, v2",
                    f"v_madak_f32_e32 v12, v1, v2, {k:#x}",
                    f"v_madmk_f32_e32 v13, v1, {k:#x}, v2",
                ], ("v10", "v11", "v12", "v13"))
                self.assertEqual(words, [want_ak, want_mk, want_ak, want_mk])

        # Special boundary: when src0 == 0 and one of (vsrc1, K) is +inf while the other is finite,
        # the op that multiplies by +inf produces NaN while the op that adds +inf produces +inf.
        words = run_ps_mrt0([
            f"v_mov_b32 v1, {POS_ZERO:#x}",
            f"v_mov_b32 v2, {TWO:#x}",
            f"v_madak_f32 v10, v1, v2, {POS_INF:#x}",   # 0 * 2 + inf = +inf
            f"v_madmk_f32 v11, v1, {POS_INF:#x}, v2",   # 0 * inf + 2 = NaN
            f"v_mov_b32 v3, {POS_INF:#x}",
            f"v_madak_f32 v12, v1, v3, {TWO:#x}",       # 0 * inf + 2 = NaN
            f"v_madmk_f32 v13, v1, {TWO:#x}, v3",       # 0 * 2 + inf = +inf
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(words[0], POS_INF)
        self.assertTrue(is_nan_bits(words[1]), hex(words[1]))
        self.assertTrue(is_nan_bits(words[2]), hex(words[2]))
        self.assertEqual(words[3], POS_INF)

    def test_normal_fractional_and_zero_sweep_batched(self):
        # 5 * 5 * 4 = 100 triples per instruction covering positives, negatives, fractions, and zeros.
        src0_vals = (NEG_THREE, NEG_HALF, POS_ZERO, QUARTER, TWO_HALF)
        src1_vals = (NEG_ONE_HALF, NEG_ZERO, ONE, f32_to_bits(1.0 / 3.0), SIX)
        k_vals = (NEG_TWO, POS_ZERO, f32_to_bits(0.1), f32_to_bits(3.75))
        triples = [(a, b, k) for a in src0_vals for b in src1_vals for k in k_vals]

        for op in ("v_madak_f32", "v_madmk_f32"):
            for start in range(0, len(triples), 4):
                chunk = triples[start:start + 4]
                lines, expected = [], []
                for slot, (a, b, k) in enumerate(chunk):
                    base = 10 + 3 * slot
                    lines.append(f"v_mov_b32 v{base + 1}, {a:#x}")
                    lines.append(f"v_mov_b32 v{base + 2}, {b:#x}")
                    if op == "v_madak_f32":
                        lines.append(f"{op} v{base}, v{base + 1}, v{base + 2}, {k:#x}")
                        expected.append(ref_madak_f32(a, b, k))
                    else:
                        lines.append(f"{op} v{base}, v{base + 1}, {k:#x}, v{base + 2}")
                        expected.append(ref_madmk_f32(a, k, b))
                regs = [f"v{10 + 3 * slot}" for slot in range(len(chunk))]
                while len(regs) < 4:
                    regs.append(regs[0])
                    expected.append(expected[0])
                with self.subTest(op=op, batch=start // 4):
                    self.assertEqual(run_ps_mrt0(lines, tuple(regs)), expected)


class LiteralAndOperandEncodingTests(unittest.TestCase):
    """Hex literals, decimal float literals, inline constants, SGPR src0, and destination aliasing."""

    def test_hex_and_float_literal_encodings(self):
        # Mix hex 32-bit float encodings and decimal float literal spellings in K.
        words = run_ps_mrt0([
            f"v_mov_b32 v1, {TWO:#x}",
            f"v_mov_b32 v2, {THREE:#x}",
            "v_madak_f32 v10, v1, v2, 0x3f800000",   # 2.0 * 3.0 + 1.0 = 7.0
            "v_madak_f32 v11, v1, v2, 2.5",          # 2.0 * 3.0 + 2.5 = 8.5
            "v_madmk_f32 v12, v1, 0xbf000000, v2",   # 2.0 * (-0.5) + 3.0 = 2.0
            "v_madmk_f32 v13, v1, -0.5, v2",         # 2.0 * (-0.5) + 3.0 = 2.0
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(words, [f32_to_bits(7.0), f32_to_bits(8.5), TWO, TWO])

        # Scientific-notation float literals and negative hex / float literals.
        words2 = run_ps_mrt0([
            f"v_mov_b32 v1, {FOUR:#x}",
            f"v_mov_b32 v2, {ONE_HALF:#x}",
            "v_madak_f32 v10, v1, v2, -0.5",         # 4.0 * 1.5 - 0.5 = 5.5
            "v_madak_f32 v11, v1, v2, 1.25e1",       # 4.0 * 1.5 + 12.5 = 18.5
            "v_madmk_f32 v12, v1, 2.5, v2",          # 4.0 * 2.5 + 1.5 = 11.5
            "v_madmk_f32 v13, v1, 2.5e-1, v2",       # 4.0 * 0.25 + 1.5 = 2.5
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(words2, [f32_to_bits(v) for v in (5.5, 18.5, 11.5, 2.5)])

    def test_inline_constants_as_src0(self):
        # VOP2 src0 accepts RDNA inline constants (0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 4.0, -4.0, 0.15915494).
        words_ak = run_ps_mrt0([
            f"v_mov_b32 v2, {THREE:#x}",
            "v_madak_f32 v10, 0, v2, 2.5",           # 0.0 * 3.0 + 2.5 = 2.5
            "v_madak_f32 v11, 1.0, v2, -0.5",        # 1.0 * 3.0 + (-0.5) = 2.5
            "v_madak_f32 v12, -1.0, v2, 0x40a00000", # (-1.0) * 3.0 + 5.0 = 2.0
            "v_madak_f32 v13, 0.15915494, v2, 0x3f800000",
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(words_ak, [
            TWO_HALF,
            TWO_HALF,
            TWO,
            ref_madak_f32(INV_TWO_PI, THREE, ONE),
        ])

        words_mk = run_ps_mrt0([
            f"v_mov_b32 v2, {THREE:#x}",
            "v_madmk_f32 v10, 0, 2.5, v2",           # 0.0 * 2.5 + 3.0 = 3.0
            "v_madmk_f32 v11, 1.0, -0.5, v2",        # 1.0 * (-0.5) + 3.0 = 2.5
            "v_madmk_f32 v12, -1.0, 0x40a00000, v2", # (-1.0) * 5.0 + 3.0 = -2.0
            "v_madmk_f32 v13, 0.5, 0x40400000, v2",  # 0.5 * 3.0 + 3.0 = 4.5
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(words_mk, [THREE, TWO_HALF, NEG_TWO, f32_to_bits(4.5)])

    def test_sgpr_as_src0(self):
        # VOP2 src0 may be an SGPR (either loaded by s_mov_b32 or supplied via push constants in s0..s3).
        words = run_ps_mrt0([
            f"s_mov_b32 s4, {TWO_HALF:#x}",
            f"s_mov_b32 s5, {NEG_TWO:#x}",
            f"v_mov_b32 v2, {FOUR:#x}",
            "v_madak_f32 v10, s4, v2, 0x3fc00000",   # 2.5 * 4.0 + 1.5 = 11.5
            "v_madmk_f32 v11, s4, 0x3fc00000, v2",   # 2.5 * 1.5 + 4.0 = 7.75
            "v_madak_f32 v12, s5, v2, 2.5",          # (-2.0) * 4.0 + 2.5 = -5.5
            "v_madmk_f32 v13, s0, -0.5, v2",         # s0(=6.0) * (-0.5) + 4.0 = 1.0
        ], ("v10", "v11", "v12", "v13"), push_constants=[SIX] + [0] * 31)
        self.assertEqual(words, [f32_to_bits(11.5), f32_to_bits(7.75), f32_to_bits(-5.5), ONE])

    def test_destination_register_aliasing(self):
        # vdst aliasing src0, vsrc1, or both must read pre-write operand values cleanly.
        cases = [
            ("v_madak_f32 v0, v0, v1, 5.0", ref_madak_f32(TWO, THREE, FIVE)),   # 2*3 + 5 = 11.0
            ("v_madak_f32 v0, v1, v0, 5.0", ref_madak_f32(THREE, TWO, FIVE)),   # 3*2 + 5 = 11.0
            ("v_madak_f32 v0, v0, v0, 5.0", ref_madak_f32(TWO, TWO, FIVE)),     # 2*2 + 5 = 9.0
            ("v_madmk_f32 v0, v0, 5.0, v1", ref_madmk_f32(TWO, FIVE, THREE)),   # 2*5 + 3 = 13.0
            ("v_madmk_f32 v0, v1, 5.0, v0", ref_madmk_f32(THREE, FIVE, TWO)),   # 3*5 + 2 = 17.0
            ("v_madmk_f32 v0, v0, 5.0, v0", ref_madmk_f32(TWO, FIVE, TWO)),     # 2*5 + 2 = 12.0
        ]
        for insn, want in cases:
            with self.subTest(insn=insn):
                got = run_ps_mrt0([
                    f"v_mov_b32 v0, {TWO:#x}",
                    f"v_mov_b32 v1, {THREE:#x}",
                    insn,
                ])[0]
                self.assertEqual(got, want)


class EdgeCaseAndRoundingTests(unittest.TestCase):
    """Unfused rounding, NoContraction, signed zeros, infinities, NaN propagation, and chains."""

    def test_unfused_rounding_and_nocontraction(self):
        # Cases where unfused binary32 mul-then-add differs from fused FMA (v_fmaak_f32 / v_fmamk_f32).
        cases = [
            # (a, b, c, want_unfused, want_fused)
            (0x3F800008, 0x3F7FFFF0, NEG_ONE, POS_ZERO, 0xAB800000),      # (1+2^-20)*(1-2^-20) - 1
            (0x3F800001, 0x3F800001, 0xBF800002, POS_ZERO, 0x28800000),   # (1+2^-23)^2 - (1+2^-22)
            (0x3FC00001, 0x3FBFFFFF, 0xC0100000, POS_ZERO, 0xA8800000),   # (1.5+2^-23)*(1.5-2^-23) - 2.25
            (0x71800000, 0x4D800000, 0xFF000000, POS_INF, 0x7F000000),    # 2^100*2^28 - 2^127 (mul ovf)
            (0xF1800000, 0x4D800000, 0x7F000000, NEG_INF, 0xFF000000),    # -2^100*2^28 + 2^127 (mul -ovf)
        ]
        for a, b, c, want_mad, want_fma in cases:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                self.assertEqual(ref_madak_f32(a, b, c), want_mad)
                self.assertEqual(ref_madmk_f32(a, b, c), want_mad)
                self.assertEqual(ref_fma_f32(a, b, c), want_fma)
                self.assertNotEqual(want_mad, want_fma)
                self.assertEqual(run_madak(a, b, c), want_mad)
                self.assertEqual(run_madmk(a, b, c), want_mad)
                # Cross-check that the fused literal counterparts really diverge on the same inputs.
                self.assertEqual(run_madak(a, b, c, op="v_fmaak_f32"), want_fma)
                self.assertEqual(run_madmk(a, b, c, op="v_fmamk_f32"), want_fma)

        # Verify both ops emit unfused OpFMul + OpFAdd decorated with NoContraction (never Fma).
        for insn in (
            f"v_madak_f32 v0, v1, v2, {THREE:#x}",
            f"v_madmk_f32 v0, v1, {TWO:#x}, v2",
        ):
            with self.subTest(insn=insn):
                spv = translate_ps([
                    f"v_mov_b32 v1, {ONE:#x}",
                    f"v_mov_b32 v2, {TWO:#x}",
                    insn,
                ])
                fmul_ids = re.findall(r"(%\w+)\s*=\s*OpFMul %float", spv)
                fadd_ids = re.findall(r"(%\w+)\s*=\s*OpFAdd %float", spv)
                self.assertEqual(len(fmul_ids), 1)
                self.assertEqual(len(fadd_ids), 1)
                self.assertIn(f"OpDecorate {fmul_ids[0]} NoContraction", spv)
                self.assertIn(f"OpDecorate {fadd_ids[0]} NoContraction", spv)
                self.assertNotIn("Fma", spv)

    def test_signed_zeros(self):
        # IEEE-754 RTN signed-zero rules for (a * b) + c with both VGPR and literal operands.
        cases = [
            (NEG_ZERO, TWO, NEG_ZERO, NEG_ZERO),       # (-0.0)*2.0 + (-0.0) = -0.0
            (TWO, NEG_ZERO, NEG_ZERO, NEG_ZERO),       # 2.0*(-0.0) + (-0.0) = -0.0
            (POS_ZERO, TWO, NEG_ZERO, POS_ZERO),       # (+0.0)*2.0 + (-0.0) = +0.0
            (NEG_ZERO, TWO, POS_ZERO, POS_ZERO),       # (-0.0)*2.0 + (+0.0) = +0.0
            (NEG_ZERO, NEG_ONE, NEG_ZERO, POS_ZERO),   # (-0.0)*(-1.0) + (-0.0) = (+0.0)+(-0.0) = +0.0
            (NEG_TWO, THREE, SIX, POS_ZERO),           # -6.0 + 6.0 = +0.0 (exact cancellation)
            (TWO, NEG_THREE, SIX, POS_ZERO),           # -6.0 + 6.0 = +0.0
        ]
        for a, b, c, want in cases:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                self.assertEqual(ref_madak_f32(a, b, c), want)
                self.assertEqual(ref_madmk_f32(a, b, c), want)
                self.assertEqual(run_madak(a, b, c), want)
                self.assertEqual(run_madmk(a, b, c), want)

        # Decimal `-0.0` literal spelling also preserves the negative zero sign bit.
        words = run_ps_mrt0([
            f"v_mov_b32 v1, {NEG_ZERO:#x}",
            f"v_mov_b32 v2, {TWO:#x}",
            f"v_mov_b32 v3, {NEG_ZERO:#x}",
            "v_madak_f32 v10, v1, v2, -0.0",           # (-0.0)*2.0 + (-0.0) = -0.0
            "v_madmk_f32 v11, v2, -0.0, v3",           # 2.0*(-0.0) + (-0.0) = -0.0
            "v_madak_f32 v12, v1, v2, 0.0",            # (-0.0)*2.0 + (+0.0) = +0.0
            "v_madmk_f32 v13, v2, 0.0, v3",            # 2.0*(+0.0) + (-0.0) = +0.0
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(words, [NEG_ZERO, NEG_ZERO, POS_ZERO, POS_ZERO])

    def test_infinities_and_overflow(self):
        exact_cases = [
            (POS_INF, TWO, ONE, POS_INF),
            (NEG_INF, TWO, ONE, NEG_INF),
            (NEG_INF, NEG_TWO, NEG_ONE, POS_INF),
            (TWO, POS_INF, NEG_THREE, POS_INF),
            (TWO, THREE, NEG_INF, NEG_INF),
            (POS_INF, TWO, POS_INF, POS_INF),
            (NEG_INF, TWO, NEG_INF, NEG_INF),
            (FLT_MAX, TWO, NEG_FLT_MAX, POS_INF),      # mul overflows to +inf
            (NEG_FLT_MAX, TWO, FLT_MAX, NEG_INF),      # mul overflows to -inf
            (FLT_MAX, ONE, FLT_MAX, POS_INF),          # add overflows to +inf
            (NEG_FLT_MAX, ONE, NEG_FLT_MAX, NEG_INF),  # add overflows to -inf
        ]
        for a, b, c, want in exact_cases:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                self.assertEqual(ref_madak_f32(a, b, c), want)
                self.assertEqual(ref_madmk_f32(a, b, c), want)
                self.assertEqual(run_madak(a, b, c), want)
                self.assertEqual(run_madmk(a, b, c), want)

    def test_nan_propagation_and_invalid_operations(self):
        nan_cases = [
            # NaN in src0, second multiplicand, or addend (including literal K for both ops).
            (QNAN, ONE, TWO),
            (ONE, QNAN, TWO),
            (ONE, TWO, QNAN),
            (NEG_QNAN, TWO, THREE),
            (TWO, NEG_QNAN, THREE),
            (TWO, THREE, NEG_QNAN),
            (SNAN, ONE, TWO),
            (ONE, SNAN, TWO),
            (ONE, TWO, SNAN),
            # 0 * inf -> NaN
            (POS_ZERO, POS_INF, ONE),
            (POS_INF, POS_ZERO, ONE),
            (NEG_ZERO, NEG_INF, TWO),
            (NEG_INF, NEG_ZERO, TWO),
            # (+inf) + (-inf) -> NaN
            (POS_INF, ONE, NEG_INF),
            (NEG_INF, ONE, POS_INF),
            (ONE, POS_INF, NEG_INF),
            (NEG_ONE, POS_INF, POS_INF),
        ]
        for a, b, c in nan_cases:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                self.assertTrue(is_nan_bits(ref_madak_f32(a, b, c)))
                self.assertTrue(is_nan_bits(ref_madmk_f32(a, b, c)))
                self.assertTrue(is_nan_bits(run_madak(a, b, c)))
                self.assertTrue(is_nan_bits(run_madmk(a, b, c)))

    def test_chained_madak_and_madmk(self):
        # 1. Horner evaluation of p(x) = ((2.5*x - 1.5)*x + 0.5)*x + 3.0 at x = 2.0 using v_madak_f32:
        #    step1: 2.5 * 2.0 + (-1.5) = 3.5
        #    step2: 3.5 * 2.0 + 0.5    = 7.5
        #    step3: 7.5 * 2.0 + 3.0    = 18.0
        # 2. Chained v_madmk_f32 scale-and-accumulate:
        #    v2 = 1.5, v3 = 0.25 -> v4 = v2 * 2.0 + v3 = 3.25
        #    -> v4 = v4 * (-0.5) + v2 = -1.625 + 1.5 = -0.125
        #    -> v4 = v4 * 4.0 + v3 = -0.5 + 0.25 = -0.25
        # 3. Interleaved v_madmk_f32 -> v_madak_f32 -> v_madmk_f32:
        #    v5 = 3.0, v6 = 0.5 -> v7 = v5 * 2.5 + v6 = 8.0
        #    -> v7 = v7 * v6 + (-1.0) = 3.0
        #    -> v7 = v7 * 0xbf000000 + v5 = 3.0*(-0.5) + 3.0 = 1.5
        # 4. Non-power-of-two fractional chain checked against exact reference oracle.
        x0 = f32_to_bits(0.3)
        y0 = f32_to_bits(1.7)
        k1 = f32_to_bits(-0.7)
        k2 = f32_to_bits(2.3)
        k3 = f32_to_bits(0.1)
        want_chain = ref_madak_f32(ref_madmk_f32(ref_madak_f32(x0, y0, k1), k2, x0), y0, k3)

        words = run_ps_mrt0([
            f"v_mov_b32 v1, {TWO:#x}",
            f"v_mov_b32 v10, {TWO_HALF:#x}",
            "v_madak_f32 v10, v10, v1, -1.5",
            "v_madak_f32 v10, v10, v1, 0.5",
            "v_madak_f32 v10, v10, v1, 3.0",
            f"v_mov_b32 v2, {ONE_HALF:#x}",
            f"v_mov_b32 v3, {QUARTER:#x}",
            "v_madmk_f32 v11, v2, 2.0, v3",
            "v_madmk_f32 v11, v11, -0.5, v2",
            "v_madmk_f32 v11, v11, 4.0, v3",
            f"v_mov_b32 v5, {THREE:#x}",
            f"v_mov_b32 v6, {HALF:#x}",
            "v_madmk_f32 v12, v5, 2.5, v6",
            "v_madak_f32 v12, v12, v6, -1.0",
            "v_madmk_f32 v12, v12, 0xbf000000, v5",
            f"v_mov_b32 v8, {x0:#x}",
            f"v_mov_b32 v9, {y0:#x}",
            f"v_madak_f32 v13, v8, v9, {k1:#x}",
            f"v_madmk_f32 v13, v13, {k2:#x}, v8",
            f"v_madak_f32 v13, v13, v9, {k3:#x}",
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(words, [f32_to_bits(18.0), f32_to_bits(-0.25), ONE_HALF, want_chain])


class ExecMaskingTests(unittest.TestCase):
    """Inactive lanes under EXEC must leave destination registers untouched for v_madak_f32 and v_madmk_f32."""

    def test_pixel_shader_inactive_exec_preserves_destination(self):
        sentinel = 0x3E4CCCCD  # 0.2f
        insns = [
            f"v_madak_f32 v0, v1, v2, {FIVE:#x}",
            f"v_madak_f32_e32 v0, v1, v2, 2.5",
            f"v_madmk_f32 v0, v1, {FIVE:#x}, v2",
            f"v_madmk_f32_e32 v0, v1, -0.5, v2",
        ]
        for disable_exec in ("s_mov_b32 exec_lo, 0", "v_cmpx_eq_u32_e32 vcc_lo, 0, 1"):
            for insn in insns:
                with self.subTest(disable=disable_exec, insn=insn):
                    got = run_ps_mrt0([
                        f"v_mov_b32 v0, {sentinel:#x}",
                        f"v_mov_b32 v1, {TWO:#x}",
                        f"v_mov_b32 v2, {THREE:#x}",
                        disable_exec,
                        insn,
                        "s_mov_b32 exec_lo, 1",
                    ])[0]
                    self.assertEqual(got, sentinel)

    def test_compute_workgroup_partial_exec_mask(self):
        sentinel = ONE
        cases = [
            (f"v_madak_f32 v1, v2, v4, {FIVE:#x}", f32_to_bits(3.0 * (-2.0) + 5.0)),   # -1.0
            (f"v_madmk_f32 v1, v2, {FIVE:#x}, v4", f32_to_bits(3.0 * 5.0 + (-2.0))),   # 13.0
        ]
        for insn, active_val in cases:
            with self.subTest(insn=insn):
                mem = run_cs_workgroup_exec([
                    "v_lshlrev_b32 v3, 2, v0",
                    f"v_mov_b32 v1, {sentinel:#x}",
                    f"v_mov_b32 v2, {THREE:#x}",
                    f"v_mov_b32 v4, {NEG_TWO:#x}",
                    "s_mov_b32 exec_lo, 5",      # lanes 0 and 2 active; lanes 1 and 3 inactive
                    insn,
                    "s_mov_b32 exec_lo, 0xf",
                    "buffer_store_dword v1, v3, s[0:3], 0 offen",
                ])
                self.assertEqual(mem, {0x0: active_val, 0x4: sentinel, 0x8: active_val, 0xC: sentinel})


class NegativeControlsTests(unittest.TestCase):
    """Sabotage translated SPIR-V output and confirm value assertions catch each mutation."""

    def test_sabotaged_spirv_is_detected(self):
        # 1. v_madak_f32 OpFAdd mutated to OpFSub.
        spv_ak = translate_ps([
            f"v_mov_b32 v1, {TWO:#x}",
            f"v_mov_b32 v2, {THREE:#x}",
            f"v_madak_f32 v0, v1, v2, {FIVE:#x}",
        ])
        want_ak = f32_to_bits(11.0)
        self.assertEqual(si.run_fragment(spv_ak)["%out_mrt0"][0], want_ak)
        self.assertNotEqual(si.run_fragment(spv_ak.replace("OpFAdd %float", "OpFSub %float"))["%out_mrt0"][0], want_ak)

        # 2. v_madmk_f32 multiplicand/addend swap (reproducing the pre-2026-11-21 S0*S1+K bug).
        spv_mk = translate_ps([
            f"v_mov_b32 v1, {TWO:#x}",
            f"v_mov_b32 v2, {THREE:#x}",
            f"v_madmk_f32 v0, v1, {FIVE:#x}, v2",
        ])
        want_mk = f32_to_bits(13.0)
        self.assertEqual(si.run_fragment(spv_mk)["%out_mrt0"][0], want_mk)
        bad_mk, count = re.subn(
            r"(%\w+) = OpFMul %float (%\w+) (%\w+)\n(\s*)(%\w+) = OpFAdd %float \1 (%\w+)",
            r"\1 = OpFMul %float \2 \6\n\4\5 = OpFAdd %float \1 \3",
            spv_mk,
        )
        self.assertEqual(count, 1)
        self.assertEqual(si.run_fragment(bad_mk)["%out_mrt0"][0], want_ak)
        self.assertNotEqual(si.run_fragment(bad_mk)["%out_mrt0"][0], want_mk)

        # 3. Replacing unfused OpFMul + OpFAdd with fused Fma breaks the cancellation case.
        spv_unfused = translate_ps([
            "v_mov_b32 v1, 0x3f800008",
            "v_mov_b32 v2, 0x3f7ffff0",
            f"v_madak_f32 v0, v1, v2, {NEG_ONE:#x}",
        ])
        self.assertEqual(si.run_fragment(spv_unfused)["%out_mrt0"][0], POS_ZERO)
        bad_fused, fcnt = re.subn(
            r"(%\w+) = OpFMul %float (%\w+) (%\w+)\n(\s*)(%\w+) = OpFAdd %float \1 (%\w+)",
            r"\5 = OpExtInst %float %glsl Fma \2 \3 \6",
            spv_unfused,
        )
        self.assertEqual(fcnt, 1)
        # Drop any dangling OpDecorate on the removed OpFMul id so spirv_mini_interp parses cleanly.
        fmul_id = re.search(r"(%\w+) = OpFMul %float", spv_unfused).group(1)
        bad_fused = re.sub(rf"^\s*OpDecorate {re.escape(fmul_id)} NoContraction\n", "", bad_fused, flags=re.M)
        self.assertNotEqual(si.run_fragment(bad_fused)["%out_mrt0"][0], POS_ZERO)


if __name__ == "__main__":
    unittest.main()
