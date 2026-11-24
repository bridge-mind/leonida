"""Semantic interpreter tests for RDNA2 float32 <-> float16 conversions: v_cvt_f16_f32 and v_cvt_f32_f16.

v_cvt_f16_f32 converts a binary32 float in S0 to a binary16 float in D[15:0] with IEEE-754 round to nearest, ties to
even (RNE). On GFX10, this is a 16-bit operation that preserves the high 16 bits D[31:16] of the destination VGPR.
v_cvt_f32_f16 converts a binary16 float in S0[15:0] to a binary32 float in D[31:0], completely overwriting the 32-bit
destination and ignoring S0[31:16].

Each test case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py, and compares the exported dwords with a plain-Python reference.
The suite verifies:
  - Normal binary32 <-> binary16 conversions and exact representation.
  - Partial destination preservation of high 16 bits for v_cvt_f16_f32.
  - Source high-half masking for v_cvt_f32_f16.
  - Signed zero representations (+0.0, -0.0).
  - Signed infinities (+inf, -inf).
  - Quiet and signaling NaNs.
  - Subnormal/denormal conversions in both directions and underflow to signed zero.
  - Rounding modes: IEEE-754 round-to-nearest-even tie breaking at normal and subnormal midpoints.
  - Overflow threshold to infinity (65504 max finite vs 65520 tie boundary to inf).
  - VOP3 source modifiers (negation, absolute value, negated abs).
  - SGPR operands and inline constants.
  - Lossless roundtrip property: for any finite f16 h, v_cvt_f16_f32(v_cvt_f32_f16(h)) == h.
"""
from __future__ import annotations

import math
import os
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
MASK16 = 0xFFFF
OLD_HIGH = 0x55AA0000
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def f32_to_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f16_to_bits(val: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return 0x7C00 if val > 0 else 0xFC00


def bits_to_f16(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def ref_cvt_f16_f32(f32_bits: int) -> int:
    """Pure-Python reference for binary32 -> binary16 conversion."""
    f32_bits &= MASK32
    val = bits_to_f32(f32_bits)
    if math.isnan(val):
        sign = 0x8000 if (f32_bits & 0x80000000) else 0x0000
        return sign | 0x7E00
    return f16_to_bits(val)


def ref_cvt_f32_f16(f16_bits: int) -> int:
    """Pure-Python reference for binary16 -> binary32 conversion."""
    f16_bits &= MASK16
    val = bits_to_f16(f16_bits)
    if math.isnan(val):
        sign = 0x80000000 if (f16_bits & 0x8000) else 0x00000000
        return sign | 0x7FC00000
    return f32_to_bits(val)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cvt_f16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cvt_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines: list[str]) -> tuple[int, ...]:
    """Assemble and run synthetic pixel shader lines, returning out_mrt0 registers."""
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_alu_cvt_f16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return tuple(si.run_fragment(spv)["%out_mrt0"])


def run_cvt_f16_f32(src_bits: int, old_dst: int = OLD_HIGH, form: str = "e32") -> int:
    """Run v_cvt_f16_f32 and return the 32-bit v0 destination word."""
    lines = [
        f"v_mov_b32 v1, {src_bits:#x}",
        f"v_mov_b32 v0, {old_dst:#x}",
        f"v_cvt_f16_f32_{form} v0, v1",
    ]
    return run_lines(lines)[0]


def run_cvt_f32_f16(src_word: int, form: str = "e32") -> int:
    """Run v_cvt_f32_f16 and return the 32-bit v0 destination word."""
    lines = [
        f"v_mov_b32 v1, {src_word:#x}",
        f"v_cvt_f32_f16_{form} v0, v1",
    ]
    return run_lines(lines)[0]


class ReferenceTests(unittest.TestCase):
    def test_reference_f16_f32(self):
        self.assertEqual(ref_cvt_f16_f32(0x3F800000), 0x3C00)   # 1.0
        self.assertEqual(ref_cvt_f16_f32(0xBF800000), 0xBC00)   # -1.0
        self.assertEqual(ref_cvt_f16_f32(0x00000000), 0x0000)   # +0.0
        self.assertEqual(ref_cvt_f16_f32(0x80000000), 0x8000)   # -0.0
        self.assertEqual(ref_cvt_f16_f32(0x7F800000), 0x7C00)   # +inf
        self.assertEqual(ref_cvt_f16_f32(0xFF800000), 0xFC00)   # -inf
        self.assertEqual(ref_cvt_f16_f32(0x7FC00000), 0x7E00)   # quiet NaN
        self.assertEqual(ref_cvt_f16_f32(0xFFC00000), 0xFE00)   # negative quiet NaN
        self.assertEqual(ref_cvt_f16_f32(0x33800000), 0x0001)   # smallest subnormal 2^-24

    def test_reference_f32_f16(self):
        self.assertEqual(ref_cvt_f32_f16(0x3C00), 0x3F800000)   # 1.0
        self.assertEqual(ref_cvt_f32_f16(0xBC00), 0xBF800000)   # -1.0
        self.assertEqual(ref_cvt_f32_f16(0x0000), 0x00000000)   # +0.0
        self.assertEqual(ref_cvt_f32_f16(0x8000), 0x80000000)   # -0.0
        self.assertEqual(ref_cvt_f32_f16(0x7C00), 0x7F800000)   # +inf
        self.assertEqual(ref_cvt_f32_f16(0xFC00), 0xFF800000)   # -inf
        self.assertEqual(ref_cvt_f32_f16(0x7E00), 0x7FC00000)   # quiet NaN
        self.assertEqual(ref_cvt_f32_f16(0xFE00), 0xFFC00000)   # negative quiet NaN
        self.assertEqual(ref_cvt_f32_f16(0x0001), 0x33800000)   # smallest subnormal 2^-24


class CvtF16F32Tests(unittest.TestCase):
    def test_normal_values(self):
        cases = [
            (0x3F800000, 0x3C00),   # 1.0
            (0xBF800000, 0xBC00),   # -1.0
            (0x3F000000, 0x3800),   # 0.5
            (0xBF000000, 0xB800),   # -0.5
            (0x40000000, 0x4000),   # 2.0
            (0xC0000000, 0xC000),   # -2.0
            (0x3FC00000, 0x3E00),   # 1.5
            (0xBFC00000, 0xBE00),   # -1.5
            (0x40200000, 0x4100),   # 2.5
            (0x3E000000, 0x3000),   # 0.125
            (0x42280000, 0x5140),   # 42.0
            (0x477FE000, 0x7BFF),   # 65504.0 (max finite f16)
            (0xC77FE000, 0xFBFF),   # -65504.0
        ]
        for f32_bits, exp_h in cases:
            with self.subTest(f32=hex(f32_bits), expected=hex(exp_h)):
                got = run_cvt_f16_f32(f32_bits, old_dst=OLD_HIGH)
                self.assertEqual(got & 0xFFFF, exp_h)
                self.assertEqual(got & 0xFFFF0000, OLD_HIGH)

    def test_preserves_dst_high_half(self):
        dst_patterns = [0x00000000, 0x12340000, 0xDEAD0000, 0xFFFF0000, 0x80000000, 0x55AA0000]
        for old in dst_patterns:
            with self.subTest(old=hex(old)):
                got = run_cvt_f16_f32(0x3F800000, old_dst=old)
                self.assertEqual(got, old | 0x3C00)

    def test_signed_zeros(self):
        # +0.0
        got_pos = run_cvt_f16_f32(0x00000000, old_dst=0x11110000)
        self.assertEqual(got_pos, 0x11110000)

        # -0.0
        got_neg = run_cvt_f16_f32(0x80000000, old_dst=0x11110000)
        self.assertEqual(got_neg, 0x11118000)

    def test_infinities(self):
        # +inf
        got_pos = run_cvt_f16_f32(0x7F800000, old_dst=0x22220000)
        self.assertEqual(got_pos, 0x22227C00)

        # -inf
        got_neg = run_cvt_f16_f32(0xFF800000, old_dst=0x22220000)
        self.assertEqual(got_neg, 0x2222FC00)

    def test_nans(self):
        # Quiet NaNs
        got_qnan = run_cvt_f16_f32(0x7FC00000, old_dst=OLD_HIGH)
        self.assertEqual(got_qnan & 0xFFFF, 0x7E00)
        self.assertEqual(got_qnan & 0xFFFF0000, OLD_HIGH)

        got_nqnan = run_cvt_f16_f32(0xFFC00000, old_dst=OLD_HIGH)
        self.assertEqual(got_nqnan & 0xFFFF, 0xFE00)
        self.assertEqual(got_nqnan & 0xFFFF0000, OLD_HIGH)

        # Payload / signaling NaN encodings must produce f16 NaN (exponent 0x1F, non-zero mantissa)
        for nan_bits in (0x7F800001, 0x7FA00000, 0xFF800001):
            with self.subTest(nan=hex(nan_bits)):
                got = run_cvt_f16_f32(nan_bits, old_dst=OLD_HIGH)
                h = got & 0xFFFF
                self.assertEqual(h & 0x7C00, 0x7C00)
                self.assertNotEqual(h & 0x03FF, 0)
                self.assertEqual(got & 0xFFFF0000, OLD_HIGH)

    def test_denormals_and_underflow(self):
        cases = [
            (0x33800000, 0x0001),   # 2^-24 -> smallest positive subnormal
            (0xB3800000, 0x8001),   # -2^-24 -> smallest negative subnormal
            (0x34000000, 0x0002),   # 2^-23 -> 2 * 2^-24
            (0x38000000, 0x0200),   # 2^-15 -> 512 * 2^-24
            (0x387FC000, 0x03FF),   # 1023 * 2^-24 -> largest subnormal
            (0xB87FC000, 0x83FF),   # negative largest subnormal
            (0x32800000, 0x0000),   # 2^-26 -> underflow to +0
            (0xB2800000, 0x8000),   # -2^-26 -> underflow to -0
            (0x00000001, 0x0000),   # smallest f32 denormal -> underflow to +0
            (0x80000001, 0x8000),   # smallest negative f32 denormal -> underflow to -0
        ]
        for f32_bits, exp_h in cases:
            with self.subTest(f32=hex(f32_bits), expected=hex(exp_h)):
                got = run_cvt_f16_f32(f32_bits, old_dst=OLD_HIGH)
                self.assertEqual(got & 0xFFFF, exp_h)
                self.assertEqual(got & 0xFFFF0000, OLD_HIGH)

    def test_rounding_modes_ties_to_even(self):
        # 1. Normal range midpoints:
        # 1.0 (0x3C00) next is 0x3C01 (step = 2^-10).
        # Midpoint: 1.0 + 2^-11 = 1.00048828125 (0x3F801000 in f32).
        # Tie-to-even: mantissa 0 is even -> rounds down to 0x3C00.
        got_tie0 = run_cvt_f16_f32(f32_to_bits(1.0 + 2**-11))
        self.assertEqual(got_tie0 & 0xFFFF, 0x3C00)

        # Strictly above midpoint: rounds up to 0x3C01.
        got_up = run_cvt_f16_f32(f32_to_bits(1.0 + 2**-11 + 2**-20))
        self.assertEqual(got_up & 0xFFFF, 0x3C01)

        # Strictly below midpoint: rounds down to 0x3C00.
        got_down = run_cvt_f16_f32(f32_to_bits(1.0 + 2**-11 - 2**-20))
        self.assertEqual(got_down & 0xFFFF, 0x3C00)

        # Midpoint between 0x3C01 and 0x3C02: 1.0 + 1.5 * 2^-10 (0x3F803000 in f32).
        # Tie-to-even: mantissa 2 is even -> rounds up to 0x3C02.
        got_tie1 = run_cvt_f16_f32(f32_to_bits(1.0 + 1.5 * 2**-10))
        self.assertEqual(got_tie1 & 0xFFFF, 0x3C02)

        # 2. Denormal range midpoints:
        # Midpoint between 0 and 2^-24: 2^-25.
        # Tie-to-even: 0 is even -> rounds down to 0x0000.
        got_denorm_tie0 = run_cvt_f16_f32(f32_to_bits(2**-25))
        self.assertEqual(got_denorm_tie0 & 0xFFFF, 0x0000)

        # Midpoint between 1 * 2^-24 and 2 * 2^-24: 1.5 * 2^-24.
        # Tie-to-even: 2 is even -> rounds up to 0x0002.
        got_denorm_tie1 = run_cvt_f16_f32(f32_to_bits(1.5 * 2**-24))
        self.assertEqual(got_denorm_tie1 & 0xFFFF, 0x0002)

        # Midpoint between 2 * 2^-24 and 3 * 2^-24: 2.5 * 2^-24.
        # Tie-to-even: 2 is even -> rounds down to 0x0002.
        got_denorm_tie2 = run_cvt_f16_f32(f32_to_bits(2.5 * 2**-24))
        self.assertEqual(got_denorm_tie2 & 0xFFFF, 0x0002)

        # Midpoint between 3 * 2^-24 and 4 * 2^-24: 3.5 * 2^-24.
        # Tie-to-even: 4 is even -> rounds up to 0x0004.
        got_denorm_tie3 = run_cvt_f16_f32(f32_to_bits(3.5 * 2**-24))
        self.assertEqual(got_denorm_tie3 & 0xFFFF, 0x0004)

    def test_overflow_boundary(self):
        # 65504.0 is max finite f16 (0x7BFF).
        got_max = run_cvt_f16_f32(f32_to_bits(65504.0))
        self.assertEqual(got_max & 0xFFFF, 0x7BFF)

        # Values < 65520 round to 65504.0.
        got_near = run_cvt_f16_f32(f32_to_bits(65519.0))
        self.assertEqual(got_near & 0xFFFF, 0x7BFF)

        # 65520.0 is the tie-to-even overflow boundary to +inf (0x7C00).
        got_over = run_cvt_f16_f32(f32_to_bits(65520.0))
        self.assertEqual(got_over & 0xFFFF, 0x7C00)

        # -65520.0 overflows to -inf (0xFC00).
        got_nover = run_cvt_f16_f32(f32_to_bits(-65520.0))
        self.assertEqual(got_nover & 0xFFFF, 0xFC00)

        # Massive floats overflow to inf.
        for big in (65536.0, 100000.0, 1e20):
            self.assertEqual(run_cvt_f16_f32(f32_to_bits(big)) & 0xFFFF, 0x7C00)
            self.assertEqual(run_cvt_f16_f32(f32_to_bits(-big)) & 0xFFFF, 0xFC00)

    def test_modifiers(self):
        # Negation: -v1 negates input before conversion
        lines_neg = [
            f"v_mov_b32 v1, {f32_to_bits(1.5):#x}",
            f"v_mov_b32 v0, {OLD_HIGH:#x}",
            "v_cvt_f16_f32_e64 v0, -v1",
        ]
        self.assertEqual(run_lines(lines_neg)[0], OLD_HIGH | 0xBE00)   # -1.5 in f16

        # Double negation
        lines_neg_neg = [
            f"v_mov_b32 v1, {f32_to_bits(-2.0):#x}",
            f"v_mov_b32 v0, {OLD_HIGH:#x}",
            "v_cvt_f16_f32_e64 v0, -v1",
        ]
        self.assertEqual(run_lines(lines_neg_neg)[0], OLD_HIGH | 0x4000)   # 2.0 in f16

        # Absolute value: |v1|
        lines_abs = [
            f"v_mov_b32 v1, {f32_to_bits(-1.5):#x}",
            f"v_mov_b32 v0, {OLD_HIGH:#x}",
            "v_cvt_f16_f32_e64 v0, |v1|",
        ]
        self.assertEqual(run_lines(lines_abs)[0], OLD_HIGH | 0x3E00)   # 1.5 in f16

        # Negated absolute value: -|v1|
        lines_neg_abs = [
            f"v_mov_b32 v1, {f32_to_bits(1.5):#x}",
            f"v_mov_b32 v0, {OLD_HIGH:#x}",
            "v_cvt_f16_f32_e64 v0, -|v1|",
        ]
        self.assertEqual(run_lines(lines_neg_abs)[0], OLD_HIGH | 0xBE00)   # -1.5 in f16

    def test_operands_inline_and_sgpr(self):
        # Inline float literal 1.0
        lines_fimm = [
            f"v_mov_b32 v0, {OLD_HIGH:#x}",
            "v_cvt_f16_f32_e32 v0, 1.0",
        ]
        self.assertEqual(run_lines(lines_fimm)[0], OLD_HIGH | 0x3C00)

        # Inline integer 0
        lines_imm0 = [
            f"v_mov_b32 v0, {OLD_HIGH:#x}",
            "v_cvt_f16_f32_e32 v0, 0",
        ]
        self.assertEqual(run_lines(lines_imm0)[0], OLD_HIGH | 0x0000)

        # Inline float literal 0.5
        lines_fimm_half = [
            f"v_mov_b32 v0, {OLD_HIGH:#x}",
            "v_cvt_f16_f32_e32 v0, 0.5",
        ]
        self.assertEqual(run_lines(lines_fimm_half)[0], OLD_HIGH | 0x3800)

        # Inline float literal -1.0
        lines_fimm_neg = [
            f"v_mov_b32 v0, {OLD_HIGH:#x}",
            "v_cvt_f16_f32_e32 v0, -1.0",
        ]
        self.assertEqual(run_lines(lines_fimm_neg)[0], OLD_HIGH | 0xBC00)

        # SGPR operand
        lines_sgpr = [
            f"s_mov_b32 s2, {f32_to_bits(2.5):#x}",
            f"v_mov_b32 v0, {OLD_HIGH:#x}",
            "v_cvt_f16_f32_e32 v0, s2",
        ]
        self.assertEqual(run_lines(lines_sgpr)[0], OLD_HIGH | 0x4100)


class CvtF32F16Tests(unittest.TestCase):
    def test_normal_values(self):
        cases = [
            (0x3C00, 0x3F800000),   # 1.0
            (0xBC00, 0xBF800000),   # -1.0
            (0x3800, 0x3F000000),   # 0.5
            (0xB800, 0xBF000000),   # -0.5
            (0x4000, 0x40000000),   # 2.0
            (0xC000, 0xC0000000),   # -2.0
            (0x3E00, 0x3FC00000),   # 1.5
            (0xBE00, 0xBFC00000),   # -1.5
            (0x4100, 0x40200000),   # 2.5
            (0x3000, 0x3E000000),   # 0.125
            (0x5140, 0x42280000),   # 42.0
            (0x7BFF, 0x477FE000),   # 65504.0
            (0xFBFF, 0xC77FE000),   # -65504.0
        ]
        for f16_bits, exp_f32 in cases:
            with self.subTest(f16=hex(f16_bits), expected=hex(exp_f32)):
                got = run_cvt_f32_f16(f16_bits)
                self.assertEqual(got, exp_f32)

    def test_ignores_src_high_half(self):
        noise_high = [0x12340000, 0xDEAD0000, 0xFFFF0000, 0x80000000, 0x55AA0000]
        for noise in noise_high:
            with self.subTest(noise=hex(noise)):
                got = run_cvt_f32_f16(noise | 0x3C00)
                self.assertEqual(got, 0x3F800000)

                got_neg = run_cvt_f32_f16(noise | 0xBC00)
                self.assertEqual(got_neg, 0xBF800000)

    def test_signed_zeros(self):
        # +0.0
        self.assertEqual(run_cvt_f32_f16(0x0000), 0x00000000)
        self.assertEqual(run_cvt_f32_f16(0x99990000), 0x00000000)

        # -0.0
        self.assertEqual(run_cvt_f32_f16(0x8000), 0x80000000)
        self.assertEqual(run_cvt_f32_f16(0x99998000), 0x80000000)

    def test_infinities(self):
        # +inf
        self.assertEqual(run_cvt_f32_f16(0x7C00), 0x7F800000)
        self.assertEqual(run_cvt_f32_f16(0x12347C00), 0x7F800000)

        # -inf
        self.assertEqual(run_cvt_f32_f16(0xFC00), 0xFF800000)
        self.assertEqual(run_cvt_f32_f16(0x1234FC00), 0xFF800000)

    def test_nans(self):
        # Quiet NaNs
        self.assertEqual(run_cvt_f32_f16(0x7E00), 0x7FC00000)
        self.assertEqual(run_cvt_f32_f16(0xFE00), 0xFFC00000)

        # Non-quiet NaN payloads expand into f32 NaNs
        for nan_h in (0x7C01, 0x7E01, 0xFC01, 0xFE01):
            with self.subTest(nan_h=hex(nan_h)):
                got = run_cvt_f32_f16(nan_h)
                self.assertTrue(math.isnan(bits_to_f32(got)))
                self.assertEqual(got & 0x7F800000, 0x7F800000)
                self.assertNotEqual(got & 0x007FFFFF, 0)

    def test_denormals_exact_conversion(self):
        cases = [
            (0x0001, 0x33800000),   # 2^-24
            (0x8001, 0xB3800000),   # -2^-24
            (0x0002, 0x34000000),   # 2^-23
            (0x8002, 0xB4000000),   # -2^-23
            (0x0200, 0x38000000),   # 2^-15
            (0x8200, 0xB8000000),   # -2^-15
            (0x03FF, 0x387FC000),   # 1023 * 2^-24
            (0x83FF, 0xB87FC000),   # -1023 * 2^-24
        ]
        for f16_bits, exp_f32 in cases:
            with self.subTest(f16=hex(f16_bits), expected=hex(exp_f32)):
                got = run_cvt_f32_f16(f16_bits)
                self.assertEqual(got, exp_f32)
                # Verify exact value recovery
                expected_val = bits_to_f16(f16_bits)
                actual_val = bits_to_f32(got)
                self.assertEqual(actual_val, expected_val)

    def test_modifiers(self):
        # Negation: -v1
        lines_neg = [
            "v_mov_b32 v1, 0x3C00",   # 1.0
            "v_cvt_f32_f16_e64 v0, -v1",
        ]
        self.assertEqual(run_lines(lines_neg)[0], 0xBF800000)   # -1.0

        # Absolute value: |v1|
        lines_abs = [
            "v_mov_b32 v1, 0xBC00",   # -1.0
            "v_cvt_f32_f16_e64 v0, |v1|",
        ]
        self.assertEqual(run_lines(lines_abs)[0], 0x3F800000)   # 1.0

        # Negated absolute value: -|v1|
        lines_neg_abs = [
            "v_mov_b32 v1, 0x3C00",   # 1.0
            "v_cvt_f32_f16_e64 v0, -|v1|",
        ]
        self.assertEqual(run_lines(lines_neg_abs)[0], 0xBF800000)   # -1.0

    def test_operands_inline_and_sgpr(self):
        # Inline hex constant 0x3C00 (1.0)
        lines_imm = [
            "v_cvt_f32_f16_e32 v0, 0x3C00",
        ]
        self.assertEqual(run_lines(lines_imm)[0], 0x3F800000)

        # Inline zero 0
        lines_imm0 = [
            "v_cvt_f32_f16_e32 v0, 0",
        ]
        self.assertEqual(run_lines(lines_imm0)[0], 0x00000000)

        # SGPR operand
        lines_sgpr = [
            "s_mov_b32 s2, 0x4000",   # 2.0
            "v_cvt_f32_f16_e32 v0, s2",
        ]
        self.assertEqual(run_lines(lines_sgpr)[0], 0x40000000)


class RoundtripTests(unittest.TestCase):
    def test_lossless_f16_roundtrip(self):
        # All finite f16 numbers must roundtrip losslessly through f32:
        # v_cvt_f16_f32(v_cvt_f32_f16(h)) == h
        samples = [
            0x0000, 0x8000,           # +/-0.0
            0x0001, 0x8001,           # min subnormals
            0x0002, 0x8002,           # subnormals
            0x0100, 0x8100,
            0x03FF, 0x83FF,           # max subnormals
            0x0400, 0x8400,           # min normal floats
            0x3C00, 0xBC00,           # +/-1.0
            0x3800, 0xB800,           # +/-0.5
            0x4000, 0xC000,           # +/-2.0
            0x3E00, 0xBE00,           # +/-1.5
            0x7BFF, 0xFBFF,           # max finite floats
            0x7C00, 0xFC00,           # +/-inf
        ]
        for h in samples:
            with self.subTest(h=hex(h)):
                lines = [
                    f"v_mov_b32 v1, {h:#x}",
                    "v_cvt_f32_f16_e32 v2, v1",
                    f"v_mov_b32 v0, {OLD_HIGH:#x}",
                    "v_cvt_f16_f32_e32 v0, v2",
                ]
                got = run_lines(lines)[0]
                self.assertEqual(got & 0xFFFF, h)
                self.assertEqual(got & 0xFFFF0000, OLD_HIGH)


class HalfNanRuleTests(unittest.TestCase):
    """si.half_to_float / si.float_to_half pin one NaN rule on every Python version.

    struct's '<e' format drops NaN payloads before Python 3.14 and keeps them from 3.14 (CPython gh-130317), so
    the old struct-based helpers returned version-dependent words for NaN operands while the expected words here
    and the docs rows are pinned to the pre-3.14 behaviour (CI runs 3.12). The helpers are bit-level now with the
    chosen rule: a NaN keeps its sign and loses its payload to the canonical quiet NaN (f16 0x7E00/0xFE00, f32
    0x7FC00000/0xFFC00000). These tests hold that rule on any Python; the non-NaN spot checks pin the exact
    conversion the helpers must keep providing alongside it.
    """

    def test_half_to_float_nan_canonicalizes_payload_keeps_sign(self):
        cases = [(0x7E00, 0x7FC00000), (0xFE00, 0xFFC00000), (0x7C01, 0x7FC00000), (0x7FFF, 0x7FC00000),
                 (0x7F01, 0x7FC00000), (0xFE01, 0xFFC00000), (0xFF01, 0xFFC00000), (0xFFFF, 0xFFC00000)]
        for h, want in cases:
            with self.subTest(h=hex(h)):
                self.assertEqual(f32_to_bits(si.half_to_float(h)), want)

    def test_float_to_half_nan_canonicalizes_payload_keeps_sign(self):
        cases = [(0x7FC00000, 0x7E00), (0x7F800001, 0x7E00), (0x7FA00000, 0x7E00), (0x7FFFFFFF, 0x7E00),
                 (0xFFC00000, 0xFE00), (0xFF800001, 0xFE00), (0xFFA00000, 0xFE00), (0xFFFFFFFF, 0xFE00)]
        for bits, want in cases:
            with self.subTest(bits=hex(bits)):
                self.assertEqual(si.float_to_half(bits_to_f32(bits)), want)

    def test_non_nan_values_are_unchanged(self):
        self.assertEqual(f32_to_bits(si.half_to_float(0x7C00)), 0x7F800000)
        self.assertEqual(f32_to_bits(si.half_to_float(0xFC00)), 0xFF800000)
        self.assertEqual(f32_to_bits(si.half_to_float(0x0001)), 0x33800000)
        self.assertEqual(f32_to_bits(si.half_to_float(0x03FF)), 0x387FC000)
        self.assertEqual(f32_to_bits(si.half_to_float(0x8000)), 0x80000000)
        self.assertEqual(si.float_to_half(bits_to_f32(0x7F800000)), 0x7C00)
        self.assertEqual(si.float_to_half(bits_to_f32(0xFF800000)), 0xFC00)
        self.assertEqual(si.float_to_half(bits_to_f32(0x33800000)), 0x0001)
        self.assertEqual(si.float_to_half(bits_to_f32(0x387FC000)), 0x03FF)
        self.assertEqual(si.float_to_half(bits_to_f32(0x477FE000)), 0x7BFF)      # 65504, max finite
        self.assertEqual(si.float_to_half(bits_to_f32(0x477FF000)), 0x7C00)      # 65520 RNE tie -> inf

    def test_float_to_half_rounds_ties_to_even(self):
        # float_to_half rounds by hand, so its tie rule needs direct cases. The v_cvt_f16_f32 tie tests above do
        # not reach it: the translator lowers that opcode with its own integer RNE. The helper backs PackHalf2x16
        # and OpQuantizeToF16. Round-half-up in either path (normal or subnormal) fails at least one row here.
        cases = [
            # Normal path: the dropped 13 bits are exactly 0x1000.
            (0x3F801000, 0x3C00),   # 1 + 2^-11, between 0x3C00 and 0x3C01 -> even 0x3C00
            (0xBF801000, 0xBC00),   # negative of the above
            (0x3F803000, 0x3C02),   # 1 + 3 * 2^-11, between 0x3C01 and 0x3C02 -> even 0x3C02
            (0x3F800FFF, 0x3C00),   # just below the first tie
            (0x3F801001, 0x3C01),   # just above the first tie
            (0x3FFFF000, 0x4000),   # tie above 0x3FFF -> even, mantissa carry into the exponent (2.0)
            (0x477FD000, 0x7BFE),   # 65488, between 65472 and 65504 -> even 0x7BFE
            (0x477FEFFF, 0x7BFF),   # largest f32 below 65520 stays at max finite 65504
            (0xC77FF000, 0xFC00),   # -65520 tie -> -inf
            # Subnormal path: n = value / 2^-24 has a fraction of exactly one half.
            (0x33000000, 0x0000),   # 0.5 -> even 0
            (0xB3000000, 0x8000),   # -0.5 -> -0
            (0x33C00000, 0x0002),   # 1.5 -> even 2
            (0x34200000, 0x0002),   # 2.5 -> even 2
            (0x387FE000, 0x0400),   # 1023.5 -> even 1024, the smallest normal
        ]
        for bits, want in cases:
            with self.subTest(bits=hex(bits)):
                self.assertEqual(si.float_to_half(bits_to_f32(bits)), want)


if __name__ == "__main__":
    unittest.main()
