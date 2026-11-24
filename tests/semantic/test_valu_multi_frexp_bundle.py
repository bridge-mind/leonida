"""Semantic reference tests and translator-backed validation for AMD RDNA2 VALU Multi-Precision Float Decomposition Suite.

Subsystem Feature Bundle 48 covers 6 multi-precision floating-point frexp decomposition instructions (f16, f32, f64):
  1. v_frexp_exp_i16_f16 - Extracts 16-bit integer exponent from half-precision float (S0 = mant * 2^exp, |mant| in [0.5, 1))
  2. v_frexp_mant_f16    - Extracts 16-bit half-precision mantissa in [0.5, 1) with sign preservation
  3. v_frexp_exp_i32_f32 - Extracts 32-bit signed integer exponent from single-precision float (biased exponent - 126)
  4. v_frexp_mant_f32    - Extracts 32-bit single-precision mantissa in [0.5, 1) with sign preservation
  5. v_frexp_exp_i32_f64 - Extracts 32-bit signed integer exponent from double-precision float (biased exponent - 1022)
  6. v_frexp_mant_f64    - Extracts 64-bit double-precision mantissa in [0.5, 1) with sign preservation

RDNA2 Floating-Point Decomposition Architecture:
  - Supports f16 (IEEE 754 half-precision binary16), f32 (binary32), and f64 (binary64).
  - Unbiased exponent and normalized significand extraction.
  - Zero, infinity, NaN, and denormal/subnormal edge-case handling adhering to GFX1030+ hardware semantics.
  - End-to-end SPIR-V assembly translation through tools/rdna2spirv.py.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


# --- Half-precision helpers ---
def h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & MASK16))[0]


def f2h(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def is_f16_finite_nonzero(h: int) -> bool:
    return (h >> 10) & 0x1F != 31 and (h & 0x7FFF) != 0


def oracle_frexp_exp_i16_f16(h: int) -> int:
    """Reference mathematical oracle for v_frexp_exp_i16_f16."""
    if not is_f16_finite_nonzero(h):
        return 0
    val = h2f(h)
    _, exp = math.frexp(val)
    return exp & MASK16


def oracle_frexp_mant_f16(h: int) -> int:
    """Reference mathematical oracle for v_frexp_mant_f16."""
    if not is_f16_finite_nonzero(h):
        return h & MASK16
    val = h2f(h)
    mant, _ = math.frexp(val)
    return f2h(mant)


# --- Single-precision helpers ---
def f32_to_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def is_f32_normal(b: int) -> bool:
    e = (b >> 23) & 0xFF
    return 0 < e < 255


def oracle_frexp_exp_i32_f32(b: int) -> int:
    """Reference mathematical oracle for v_frexp_exp_i32_f32."""
    b &= MASK32
    if not is_f32_normal(b):
        return 0
    e = (b >> 23) & 0xFF
    return (e - 126) & MASK32


def oracle_frexp_mant_f32(b: int) -> int:
    """Reference mathematical oracle for v_frexp_mant_f32."""
    b &= MASK32
    e = (b >> 23) & 0xFF
    sgn = b & 0x80000000
    if e == 0:
        return sgn
    if e == 255:
        return b
    return sgn | 0x3F000000 | (b & 0x007FFFFF)


# --- Double-precision helpers ---
def d2u(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def u2d(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u & MASK64))[0]


def is_f64_finite_nonzero(u: int) -> bool:
    exp_f = (u >> 52) & 0x7FF
    frac = u & ((1 << 52) - 1)
    return exp_f < 2047 and (exp_f > 0 or frac != 0)


def oracle_frexp_exp_i32_f64(u: int) -> int:
    """Reference mathematical oracle for v_frexp_exp_i32_f64."""
    u &= MASK64
    if not is_f64_finite_nonzero(u):
        return 0
    val = u2d(u)
    _, exp = math.frexp(val)
    return exp & MASK32


def oracle_frexp_mant_f64(u: int) -> int:
    """Reference mathematical oracle for v_frexp_mant_f64."""
    u &= MASK64
    if not is_f64_finite_nonzero(u):
        return u
    val = u2d(u)
    mant, _ = math.frexp(val)
    return d2u(mant)

# -----------------------------------------------------------------------------
# Opcode 1: v_frexp_exp_i16_f16
# -----------------------------------------------------------------------------
class VFrexpExpI16F16OracleTests(unittest.TestCase):
    def test_frexp_exp_f16_powers_of_two(self):
        # 1.0 (0x3C00) = 0.5 * 2^1 -> exp = 1
        self.assertEqual(oracle_frexp_exp_i16_f16(0x3C00), 1)
        # 0.5 (0x3800) = 0.5 * 2^0 -> exp = 0
        self.assertEqual(oracle_frexp_exp_i16_f16(0x3800), 0)
        # 2.0 (0x4000) = 0.5 * 2^2 -> exp = 2
        self.assertEqual(oracle_frexp_exp_i16_f16(0x4000), 2)
        # -4.0 (0xC400) = -0.5 * 2^3 -> exp = 3
        self.assertEqual(oracle_frexp_exp_i16_f16(0xC400), 3)

    def test_frexp_exp_f16_special_values(self):
        # +/-0.0 -> 0
        self.assertEqual(oracle_frexp_exp_i16_f16(0x0000), 0)
        self.assertEqual(oracle_frexp_exp_i16_f16(0x8000), 0)
        # +/-INF -> 0
        self.assertEqual(oracle_frexp_exp_i16_f16(0x7C00), 0)
        self.assertEqual(oracle_frexp_exp_i16_f16(0xFC00), 0)
        # NaN -> 0
        self.assertEqual(oracle_frexp_exp_i16_f16(0x7E00), 0)

class RegKind(enum.Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"
    IMM = "imm"


@dataclass(frozen=True)
class Operand:
    kind: RegKind
    index: int = 0
    raw_val: int = 0


@dataclass(frozen=True)
class ValuInstructionAST:
    mnemonic: str
    dst: Operand
    src0: Operand
    src1: Operand | None = None
    src2: Operand | None = None


class WavefrontSimdState:
    def __init__(self, lane_count: int = 32):
        self.lane_count = lane_count
        self.exec_mask: int = (1 << lane_count) - 1
        self.vgprs = [[0] * 64 for _ in range(lane_count)]

    def set_vgpr(self, lane: int, reg_idx: int, val: int):
        self.vgprs[lane][reg_idx] = val & MASK32

    def get_vgpr(self, lane: int, reg_idx: int) -> int:
        return self.vgprs[lane][reg_idx] & MASK32

class VFrexpExpI16F16AstInterpreterTests(unittest.TestCase):
    def test_frexp_exp_f16_ast_construction(self):
        ast = ValuInstructionAST("v_frexp_exp_i16_f16", Operand(RegKind.VGPR, 0), Operand(RegKind.VGPR, 1))
        self.assertEqual(ast.mnemonic, "v_frexp_exp_i16_f16")
        self.assertEqual(ast.dst.index, 0)
        self.assertEqual(ast.src0.index, 1)

    def test_frexp_exp_f16_scalar_execution(self):
        # -1.0 (0xBC00) -> exp = 1
        res = oracle_frexp_exp_i16_f16(0xBC00)
        self.assertEqual(res, 1)

class VFrexpExpI16F16SimdDivergenceTests(unittest.TestCase):
    def test_frexp_exp_f16_simd_exec_masking(self):
        wave = WavefrontSimdState(lane_count=32)
        wave.exec_mask = 0x55555555  # Even lanes
        for i in range(32):
            wave.set_vgpr(i, 0, 0xDEADBEEF)
            if (wave.exec_mask >> i) & 1:
                wave.set_vgpr(i, 0, oracle_frexp_exp_i16_f16(0x3C00))
        for i in range(32):
            if (0x55555555 >> i) & 1:
                self.assertEqual(wave.get_vgpr(i, 0), 1)
            else:
                self.assertEqual(wave.get_vgpr(i, 0), 0xDEADBEEF)

    def test_frexp_exp_f16_subnormal_normalization(self):
        # Subnormal 0x0001 (2^-24) -> exp = -23 (0xFFE9)
        self.assertEqual(oracle_frexp_exp_i16_f16(0x0001), 0xFFE9)

class VFrexpExpI16F16TranslatorTests(unittest.TestCase):
    def test_frexp_exp_f16_translation_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x3C00",
            "v_frexp_exp_i16_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_frexp_exp_f16_exec", lines)
        val = mrt0_words(spv)[0]
        self.assertEqual(val, 1)

    def test_frexp_exp_f16_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_frexp_exp_f16", ["v_frexp_exp_i16_f16"])

# -----------------------------------------------------------------------------
# Opcode 2: v_frexp_mant_f16
# -----------------------------------------------------------------------------
class VFrexpMantF16OracleTests(unittest.TestCase):
    def test_frexp_mant_f16_basic(self):
        # 1.0 (0x3C00) -> 0.5 (0x3800)
        self.assertEqual(oracle_frexp_mant_f16(0x3C00), 0x3800)
        # -1.0 (0xBC00) -> -0.5 (0xB800)
        self.assertEqual(oracle_frexp_mant_f16(0xBC00), 0xB800)

    def test_frexp_mant_f16_special_values(self):
        # +/-0.0 preserved
        self.assertEqual(oracle_frexp_mant_f16(0x0000), 0x0000)
        self.assertEqual(oracle_frexp_mant_f16(0x8000), 0x8000)
        # +/-INF preserved
        self.assertEqual(oracle_frexp_mant_f16(0x7C00), 0x7C00)
        self.assertEqual(oracle_frexp_mant_f16(0xFC00), 0xFC00)
        # NaN preserved
        self.assertEqual(oracle_frexp_mant_f16(0x7E00), 0x7E00)

class VFrexpMantF16AstInterpreterTests(unittest.TestCase):
    def test_frexp_mant_f16_ast_construction(self):
        ast = ValuInstructionAST("v_frexp_mant_f16", Operand(RegKind.VGPR, 0), Operand(RegKind.VGPR, 1))
        self.assertEqual(ast.mnemonic, "v_frexp_mant_f16")

    def test_frexp_mant_f16_scalar_execution(self):
        # 2.0 (0x4000) -> 0.5 (0x3800)
        res = oracle_frexp_mant_f16(0x4000)
        self.assertEqual(res, 0x3800)

class VFrexpMantF16SimdDivergenceTests(unittest.TestCase):
    def test_frexp_mant_f16_simd_exec_masking(self):
        wave = WavefrontSimdState(lane_count=32)
        wave.exec_mask = 0x0000FFFF
        for i in range(32):
            wave.set_vgpr(i, 0, 0x12345678)
            if (wave.exec_mask >> i) & 1:
                inp = 0x3C00 if (i % 2 == 0) else 0xBC00
                wave.set_vgpr(i, 0, oracle_frexp_mant_f16(inp))
        for i in range(16):
            expected = 0x3800 if (i % 2 == 0) else 0xB800
            self.assertEqual(wave.get_vgpr(i, 0), expected)
        for i in range(16, 32):
            self.assertEqual(wave.get_vgpr(i, 0), 0x12345678)

class VFrexpMantF16TranslatorTests(unittest.TestCase):
    def test_frexp_mant_f16_translation_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x3C00",
            "v_frexp_mant_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_frexp_mant_f16_exec", lines)
        val = mrt0_words(spv)[0]
        self.assertEqual(val, 0x3800)

    def test_frexp_mant_f16_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_frexp_mant_f16", ["v_frexp_mant_f16"])

# -----------------------------------------------------------------------------
# Opcode 3: v_frexp_exp_i32_f32
# -----------------------------------------------------------------------------
class VFrexpExpI32F32OracleTests(unittest.TestCase):
    def test_frexp_exp_f32_powers_of_two(self):
        # 1.0f (0x3F800000) -> 1
        self.assertEqual(oracle_frexp_exp_i32_f32(0x3F800000), 1)
        # 0.5f (0x3F000000) -> 0
        self.assertEqual(oracle_frexp_exp_i32_f32(0x3F000000), 0)
        # 4.0f (0x40800000) -> 3
        self.assertEqual(oracle_frexp_exp_i32_f32(0x40800000), 3)
        # -8.0f (0xC1000000) -> 4
        self.assertEqual(oracle_frexp_exp_i32_f32(0xC1000000), 4)

    def test_frexp_exp_f32_special_values(self):
        # 0.0f -> 0
        self.assertEqual(oracle_frexp_exp_i32_f32(0), 0)
        # INF / NaN -> 0
        self.assertEqual(oracle_frexp_exp_i32_f32(0x7F800000), 0)
        self.assertEqual(oracle_frexp_exp_i32_f32(0x7FC00000), 0)

class VFrexpExpI32F32AstInterpreterTests(unittest.TestCase):
    def test_frexp_exp_f32_ast_construction(self):
        ast = ValuInstructionAST("v_frexp_exp_i32_f32", Operand(RegKind.VGPR, 0), Operand(RegKind.VGPR, 1))
        self.assertEqual(ast.mnemonic, "v_frexp_exp_i32_f32")

    def test_frexp_exp_f32_scalar_execution(self):
        # 16.0f (0x41800000) -> exp = 5
        res = oracle_frexp_exp_i32_f32(0x41800000)
        self.assertEqual(res, 5)

class VFrexpExpI32F32SimdDivergenceTests(unittest.TestCase):
    def test_frexp_exp_f32_simd_exec_masking(self):
        wave = WavefrontSimdState(lane_count=32)
        wave.exec_mask = 0xAAAAAAAA
        for i in range(32):
            wave.set_vgpr(i, 0, 0x99999999)
            if (wave.exec_mask >> i) & 1:
                # 2.0^i in f32
                f_val = f32_to_bits(math.pow(2.0, i))
                wave.set_vgpr(i, 0, oracle_frexp_exp_i32_f32(f_val))
        for i in range(32):
            if (0xAAAAAAAA >> i) & 1:
                self.assertEqual(wave.get_vgpr(i, 0), i + 1)
            else:
                self.assertEqual(wave.get_vgpr(i, 0), 0x99999999)

class VFrexpExpI32F32TranslatorTests(unittest.TestCase):
    def test_frexp_exp_f32_translation_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x3F800000",
            "v_frexp_exp_i32_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_frexp_exp_f32_exec", lines)
        val = mrt0_words(spv)[0]
        self.assertEqual(val, 1)

    def test_frexp_exp_f32_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_frexp_exp_f32", ["v_frexp_exp_i32_f32"])

# -----------------------------------------------------------------------------
# Opcode 4: v_frexp_mant_f32
# -----------------------------------------------------------------------------
class VFrexpMantF32OracleTests(unittest.TestCase):
    def test_frexp_mant_f32_basic(self):
        # 1.0f (0x3F800000) -> 0.5f (0x3F000000)
        self.assertEqual(oracle_frexp_mant_f32(0x3F800000), 0x3F000000)
        # -1.0f (0xBF800000) -> -0.5f (0xBF000000)
        self.assertEqual(oracle_frexp_mant_f32(0xBF800000), 0xBF000000)

    def test_frexp_mant_f32_special_values(self):
        # 0.0f -> 0.0f
        self.assertEqual(oracle_frexp_mant_f32(0x00000000), 0x00000000)
        self.assertEqual(oracle_frexp_mant_f32(0x80000000), 0x80000000)
        # INF / NaN preserved
        self.assertEqual(oracle_frexp_mant_f32(0x7F800000), 0x7F800000)
        self.assertEqual(oracle_frexp_mant_f32(0x7FC00000), 0x7FC00000)

class VFrexpMantF32AstInterpreterTests(unittest.TestCase):
    def test_frexp_mant_f32_ast_construction(self):
        ast = ValuInstructionAST("v_frexp_mant_f32", Operand(RegKind.VGPR, 0), Operand(RegKind.VGPR, 1))
        self.assertEqual(ast.mnemonic, "v_frexp_mant_f32")

    def test_frexp_mant_f32_scalar_execution(self):
        # 4.0f (0x40800000) -> 0.5f (0x3F000000)
        res = oracle_frexp_mant_f32(0x40800000)
        self.assertEqual(res, 0x3F000000)

class VFrexpMantF32SimdDivergenceTests(unittest.TestCase):
    def test_frexp_mant_f32_simd_exec_masking(self):
        wave = WavefrontSimdState(lane_count=32)
        wave.exec_mask = 0xF0F0F0F0
        for i in range(32):
            wave.set_vgpr(i, 0, 0x55555555)
            if (wave.exec_mask >> i) & 1:
                inp = f32_to_bits(1.5 * math.pow(2.0, i))
                wave.set_vgpr(i, 0, oracle_frexp_mant_f32(inp))
        for i in range(32):
            if (0xF0F0F0F0 >> i) & 1:
                # 1.5 * 2^i = 0.75 * 2^(i+1) -> mantissa is 0.75f (0x3F400000)
                self.assertEqual(wave.get_vgpr(i, 0), 0x3F400000)
            else:
                self.assertEqual(wave.get_vgpr(i, 0), 0x55555555)

class VFrexpMantF32TranslatorTests(unittest.TestCase):
    def test_frexp_mant_f32_translation_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x3F800000",
            "v_frexp_mant_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_frexp_mant_f32_exec", lines)
        val = mrt0_words(spv)[0]
        self.assertEqual(val, 0x3F000000)

    def test_frexp_mant_f32_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_frexp_mant_f32", ["v_frexp_mant_f32"])

# -----------------------------------------------------------------------------
# Opcode 5: v_frexp_exp_i32_f64
# -----------------------------------------------------------------------------
class VFrexpExpI32F64OracleTests(unittest.TestCase):
    def test_frexp_exp_f64_powers_of_two(self):
        # 1.0d (0x3FF0000000000000) -> 1
        self.assertEqual(oracle_frexp_exp_i32_f64(0x3FF0000000000000), 1)
        # 0.5d (0x3FE0000000000000) -> 0
        self.assertEqual(oracle_frexp_exp_i32_f64(0x3FE0000000000000), 0)
        # 2.0d (0x4000000000000000) -> 2
        self.assertEqual(oracle_frexp_exp_i32_f64(0x4000000000000000), 2)
        # -4.0d (0xC010000000000000) -> 3
        self.assertEqual(oracle_frexp_exp_i32_f64(0xC010000000000000), 3)

    def test_frexp_exp_f64_special_values(self):
        # 0.0d -> 0
        self.assertEqual(oracle_frexp_exp_i32_f64(0), 0)
        # INF / NaN -> 0
        self.assertEqual(oracle_frexp_exp_i32_f64(0x7FF0000000000000), 0)
        self.assertEqual(oracle_frexp_exp_i32_f64(0x7FF8000000000000), 0)

class VFrexpExpI32F64AstInterpreterTests(unittest.TestCase):
    def test_frexp_exp_f64_ast_construction(self):
        ast = ValuInstructionAST("v_frexp_exp_i32_f64", Operand(RegKind.VGPR, 2), Operand(RegKind.VGPR, 0))
        self.assertEqual(ast.mnemonic, "v_frexp_exp_i32_f64")

    def test_frexp_exp_f64_scalar_execution(self):
        # 8.0d (0x4020000000000000) -> 4
        res = oracle_frexp_exp_i32_f64(0x4020000000000000)
        self.assertEqual(res, 4)

class VFrexpExpI32F64SimdDivergenceTests(unittest.TestCase):
    def test_frexp_exp_f64_simd_exec_masking(self):
        wave = WavefrontSimdState(lane_count=32)
        wave.exec_mask = 0x00FF00FF
        for i in range(32):
            wave.set_vgpr(i, 2, 0x77777777)
            if (wave.exec_mask >> i) & 1:
                val = d2u(math.pow(2.0, i))
                wave.set_vgpr(i, 2, oracle_frexp_exp_i32_f64(val))
        for i in range(32):
            if (0x00FF00FF >> i) & 1:
                self.assertEqual(wave.get_vgpr(i, 2), i + 1)
            else:
                self.assertEqual(wave.get_vgpr(i, 2), 0x77777777)

    def test_frexp_exp_f64_subnormal_normalization(self):
        # Smallest subnormal 0x0000000000000001 (2^-1074) -> exp = -1073 (0xFFFFFBCF in u32)
        self.assertEqual(oracle_frexp_exp_i32_f64(1), 0xFFFFFBCF)

class VFrexpExpI32F64TranslatorTests(unittest.TestCase):
    def test_frexp_exp_f64_translation_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x3FF00000",
            "v_frexp_exp_i32_f64 v2, v[0:1]",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_frexp_exp_f64_exec", lines)
        val = mrt0_words(spv)[0]
        self.assertEqual(val, 1)

    def test_frexp_exp_f64_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_frexp_exp_f64", ["v_frexp_exp_i32_f64 v0, v[0:1] clamp"])

# -----------------------------------------------------------------------------
# Opcode 6: v_frexp_mant_f64
# -----------------------------------------------------------------------------
class VFrexpMantF64OracleTests(unittest.TestCase):
    def test_frexp_mant_f64_basic(self):
        # 1.0d (0x3FF0000000000000) -> 0.5d (0x3FE0000000000000)
        self.assertEqual(oracle_frexp_mant_f64(0x3FF0000000000000), 0x3FE0000000000000)
        # -1.0d (0xBFF0000000000000) -> -0.5d (0xBFE0000000000000)
        self.assertEqual(oracle_frexp_mant_f64(0xBFF0000000000000), 0xBFE0000000000000)

    def test_frexp_mant_f64_special_values(self):
        # 0.0d -> 0.0d
        self.assertEqual(oracle_frexp_mant_f64(0), 0)
        # INF / NaN preserved
        self.assertEqual(oracle_frexp_mant_f64(0x7FF0000000000000), 0x7FF0000000000000)
        self.assertEqual(oracle_frexp_mant_f64(0x7FF8000000000000), 0x7FF8000000000000)

class VFrexpMantF64AstInterpreterTests(unittest.TestCase):
    def test_frexp_mant_f64_ast_construction(self):
        ast = ValuInstructionAST("v_frexp_mant_f64", Operand(RegKind.VGPR, 2), Operand(RegKind.VGPR, 0))
        self.assertEqual(ast.mnemonic, "v_frexp_mant_f64")

    def test_frexp_mant_f64_scalar_execution(self):
        # 2.0d (0x4000000000000000) -> 0.5d (0x3FE0000000000000)
        res = oracle_frexp_mant_f64(0x4000000000000000)
        self.assertEqual(res, 0x3FE0000000000000)

class VFrexpMantF64SimdDivergenceTests(unittest.TestCase):
    def test_frexp_mant_f64_simd_exec_masking(self):
        wave = WavefrontSimdState(lane_count=32)
        wave.exec_mask = 0x33333333
        for i in range(32):
            wave.set_vgpr(i, 2, 0xAAAAAAAA)
            wave.set_vgpr(i, 3, 0xBBBBBBBB)
            if (wave.exec_mask >> i) & 1:
                # 3.0 * 2^i
                d_val = d2u(3.0 * math.pow(2.0, i))
                mant = oracle_frexp_mant_f64(d_val)
                wave.set_vgpr(i, 2, mant & MASK32)
                wave.set_vgpr(i, 3, (mant >> 32) & MASK32)
        # 3.0 = 0.75 * 4 -> mantissa is 0.75d (0x3FE8000000000000)
        expected_lo = 0x00000000
        expected_hi = 0x3FE80000
        for i in range(32):
            if (0x33333333 >> i) & 1:
                self.assertEqual(wave.get_vgpr(i, 2), expected_lo)
                self.assertEqual(wave.get_vgpr(i, 3), expected_hi)
            else:
                self.assertEqual(wave.get_vgpr(i, 2), 0xAAAAAAAA)
                self.assertEqual(wave.get_vgpr(i, 3), 0xBBBBBBBB)

class VFrexpMantF64TranslatorTests(unittest.TestCase):
    def test_frexp_mant_f64_translation_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x3FF00000",
            "v_frexp_mant_f64 v[2:3], v[0:1]",
            "exp mrt0 v2, v3, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_frexp_mant_f64_exec", lines)
        res = mrt0_words(spv)
        self.assertEqual(res[0], 0x00000000)
        self.assertEqual(res[1], 0x3FE00000)

    def test_frexp_mant_f64_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_frexp_mant_f64", ["v_frexp_mant_f64 v0, v[0:1]"])


if __name__ == "__main__":
    unittest.main()
