"""Semantic interpreter tests for floating-point frexp extraction (v_frexp_mant_f32, v_frexp_exp_i32_f32).

Models IEEE 754 single-precision significand (mantissa) and unbiased exponent extraction as specified
in the AMD RDNA2 / GFX10 Instruction Set Architecture and the SPIR-V GLSL.std.450 extended instruction set:
  - v_frexp_mant_f32: D.f = Mantissa(S0.f)
    Decomposes S0 such that S0 = mantissa * 2^exponent with |mantissa| in [0.5, 1.0).
    For normal inputs, the IEEE 754 biased exponent bits [30:23] are replaced with 126 (0x7E, representing
    exponent 2^-1), preserving the original sign bit (bit 31) and 23-bit significand fraction (bits [22:0]).
    Special values:
      * +/-0.0: returns +/-0.0 (sign bit preserved)
      * +/-INF: returns +/-INF unchanged
      * NaN: returns NaN unchanged (quiet / signaling payload bits preserved)
      * Subnormals: normalized into [0.5, 1.0) with scaled exponent, or flushed to +/-0.0 under FTZ
  - v_frexp_exp_i32_f32: D.i = TwosComplement(Exponent(S0.f) - 127 + 1)
    Extracts the unbiased integer exponent of S0, stored as a signed 32-bit integer in two's complement.
    For normal inputs, computes (biased_exponent - 126).
    Special values:
      * +/-0.0: returns 0
      * +/-INF: returns 0
      * NaN: returns 0
      * Subnormals: returns -126 - leading_zeros (normalized) or 0 (FTZ)

Includes:
  * Bitwise reference oracle (ref_frexp_mant_f32_bits, ref_frexp_exp_i32_f32)
  * Float-level oracle and cross-check against Python math.frexp
  * Mathematical invariant proofs (reconstruction, range bounds, sign preservation, idempotence)
  * Full coverage of edge cases (zeros, powers of two, boundaries, infinities, quiet/signaling NaNs)
  * Direct execution of SPIR-V translation sequences using tests/semantic/spirv_mini_interp.py
  * VALU assembly interpreter (FrexpValuInterpreter) modeling VOP1/VOP3 e32/e64 forms and input modifiers (|src|, -src)

Standard library only; zero third-party dependencies; zero game data or proprietary assets.
Python 3.9+ compatible.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
SIGN_MASK = 0x80000000
EXP_MASK = 0x7F800000
MANT_MASK = 0x007FFFFF
EXP_BIAS = 127
EXP_SHIFT = 23
EXP_FREXP_BIASED = 126  # 2^-1 has biased exponent 126 in IEEE 754
EXP_FREXP_BITS = EXP_FREXP_BIASED << EXP_SHIFT  # 0x3F000000 (0.5f)


def f32_to_bits(f: float) -> int:
    """Encode Python float as IEEE 754 single-precision 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", f))[0]
    except OverflowError:
        return 0x7F800000 if f > 0 else 0xFF800000


def bits_to_f32(b: int) -> float:
    """Decode IEEE 754 single-precision 32-bit unsigned integer to Python float."""
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def s32(x: int) -> int:
    """Convert integer to signed 32-bit two's complement."""
    x &= MASK32
    return x - (1 << 32) if x & SIGN_MASK else x


def ref_frexp_mant_f32_bits(s0_bits: int, ftz: bool = False) -> int:
    """Reference oracle for v_frexp_mant_f32 bitwise extraction.

    Given a 32-bit single-precision float pattern s0_bits:
      - If NaN or +/-INF: passed through unchanged.
      - If +/-0.0: passed through unchanged (+/-0.0).
      - If subnormal: normalized into [0.5, 1.0) under IEEE mode, or flushed to +/-0.0 under FTZ.
      - If normal: sign and fraction are preserved, exponent is replaced with 126 (0x7E).
    """
    s0_bits &= MASK32
    sign = s0_bits & SIGN_MASK
    exp_biased = (s0_bits & EXP_MASK) >> EXP_SHIFT
    frac = s0_bits & MANT_MASK

    if exp_biased == 0xFF:
        # +/-INF or NaN: return unchanged
        return s0_bits

    if exp_biased == 0:
        if frac == 0:
            # +/-0.0: return unchanged
            return s0_bits
        if ftz:
            # Flush to zero: preserve sign
            return sign
        # Subnormal normalized: shift fraction left until bit 23 is set
        lz = 23 - frac.bit_length()
        norm_frac = (frac << (lz + 1)) & MANT_MASK
        return sign | EXP_FREXP_BITS | norm_frac

    # Normal float: preserve sign and fraction, replace exponent with 126
    return sign | EXP_FREXP_BITS | frac


def ref_frexp_exp_i32_f32(s0_bits: int, ftz: bool = False) -> int:
    """Reference oracle for v_frexp_exp_i32_f32 exponent extraction.

    Returns the unbiased exponent as a signed 32-bit integer in two's complement.
      - If NaN or +/-INF: returns 0.
      - If +/-0.0: returns 0.
      - If normal: returns (exp_biased - 126) & MASK32.
      - If subnormal: returns (-126 - leading_zeros) & MASK32 under IEEE mode, or 0 under FTZ.
    """
    s0_bits &= MASK32
    exp_biased = (s0_bits & EXP_MASK) >> EXP_SHIFT
    frac = s0_bits & MANT_MASK

    if exp_biased == 0xFF:
        return 0

    if exp_biased == 0:
        if frac == 0 or ftz:
            return 0
        lz = 23 - frac.bit_length()
        unbiased = -126 - lz
        return unbiased & MASK32

    unbiased = exp_biased - 126
    return unbiased & MASK32


def ref_frexp_mant_f32(x: float, ftz: bool = False) -> float:
    """Float-level oracle for v_frexp_mant_f32."""
    bits = f32_to_bits(x)
    return bits_to_f32(ref_frexp_mant_f32_bits(bits, ftz=ftz))


def ref_frexp_exp_i32(x: float, ftz: bool = False) -> int:
    """Float-level oracle for v_frexp_exp_i32_f32, returning Python signed integer."""
    bits = f32_to_bits(x)
    return s32(ref_frexp_exp_i32_f32(bits, ftz=ftz))


def run_spirv_frexp_exp(in_word: int) -> int:
    """Executes a SPIR-V fragment module evaluating v_frexp_exp_i32_f32 bitwise logic in spirv_mini_interp."""
    in_word &= MASK32
    mod = f"""OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_in = OpConstant %uint {in_word}
%c_23 = OpConstant %uint 23
%c_255 = OpConstant %uint 255
%c_126 = OpConstant %uint 126
%c_0 = OpConstant %uint 0
%main = OpFunction %void None %fn
%entry = OpLabel
%s = OpShiftRightLogical %uint %c_in %c_23
%exp_biased = OpBitwiseAnd %uint %s %c_255
%unbiased = OpISub %uint %exp_biased %c_126
%is_zero = OpIEqual %bool %exp_biased %c_0
%is_inf_nan = OpIEqual %bool %exp_biased %c_255
%is_special = OpLogicalOr %bool %is_zero %is_inf_nan
%res = OpSelect %uint %is_special %c_0 %unbiased
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(mod)["%out"][0]


def run_spirv_frexp_mant(in_word: int) -> int:
    """Executes a SPIR-V fragment module evaluating v_frexp_mant_f32 bitwise logic in spirv_mini_interp."""
    in_word &= MASK32
    mod = f"""OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_in = OpConstant %uint {in_word}
%c_23 = OpConstant %uint 23
%c_255 = OpConstant %uint 255
%c_mask_mant = OpConstant %uint {SIGN_MASK | MANT_MASK}
%c_exp_126 = OpConstant %uint {EXP_FREXP_BITS}
%c_abs_mask = OpConstant %uint {EXP_MASK | MANT_MASK}
%c_0 = OpConstant %uint 0
%main = OpFunction %void None %fn
%entry = OpLabel
%s = OpShiftRightLogical %uint %c_in %c_23
%exp_biased = OpBitwiseAnd %uint %s %c_255
%is_inf_nan = OpIEqual %bool %exp_biased %c_255
%abs_in = OpBitwiseAnd %uint %c_in %c_abs_mask
%is_zero = OpIEqual %bool %abs_in %c_0
%stripped = OpBitwiseAnd %uint %c_in %c_mask_mant
%mant_normal = OpBitwiseOr %uint %stripped %c_exp_126
%res_nonzero = OpSelect %uint %is_zero %c_in %mant_normal
%res = OpSelect %uint %is_inf_nan %c_in %res_nonzero
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(mod)["%out"][0]


class FrexpValuInterpreter:
    """Simulated VALU interpreter supporting v_frexp_mant_f32 and v_frexp_exp_i32_f32.

    Models register file (VGPR v0..v255, SGPR s0..s105), inline constants,
    e32 / e64 instruction forms, and input modifiers (abs |src| and neg -src).
    """

    def __init__(self, ftz: bool = False):
        self.vgpr: dict[int, int] = {}
        self.sgpr: dict[int, int] = {}
        self.ftz = ftz

    def read_reg(self, name: str) -> int:
        if name.startswith("v"):
            return self.vgpr.get(int(name[1:]), 0) & MASK32
        if name.startswith("s"):
            return self.sgpr.get(int(name[1:]), 0) & MASK32
        raise ValueError(f"Unknown register name: {name}")

    def write_reg(self, name: str, val: int) -> None:
        val &= MASK32
        if name.startswith("v"):
            self.vgpr[int(name[1:])] = val
        elif name.startswith("s"):
            self.sgpr[int(name[1:])] = val
        else:
            raise ValueError(f"Unknown destination register: {name}")

    def parse_operand(self, token: str) -> tuple[int, bool, bool]:
        """Parses operand token into (raw_bits, is_abs, is_neg)."""
        token = token.strip()
        is_abs = False
        is_neg = False

        if token.startswith("|") and token.endswith("|"):
            is_abs = True
            token = token[1:-1].strip()

        if token.startswith("-"):
            is_neg = True
            token = token[1:].strip()

        # Check inline floating point constants
        inline_floats = {
            "0.5": 0x3F000000,
            "1.0": 0x3F800000,
            "2.0": 0x40000000,
            "4.0": 0x40800000,
            "0.0": 0x00000000,
        }
        if token in inline_floats:
            val = inline_floats[token]
        elif token.startswith("v") or token.startswith("s"):
            val = self.read_reg(token)
        elif token.startswith("0x") or token.startswith("0X"):
            val = int(token, 16) & MASK32
        else:
            # Integer constant or float literal
            try:
                val = int(token) & MASK32
            except ValueError:
                val = f32_to_bits(float(token))

        if is_abs:
            val &= ~SIGN_MASK
        if is_neg:
            val ^= SIGN_MASK

        return val, is_abs, is_neg

    def execute_line(self, line: str) -> None:
        """Executes a single assembly line."""
        clean = line.split(";")[0].strip()
        if not clean:
            return

        parts = clean.split(None, 1)
        mnemonic = parts[0].lower()
        operands = [op.strip() for op in parts[1].split(",")] if len(parts) > 1 else []

        if mnemonic in ("v_mov_b32", "v_mov_b32_e32", "s_mov_b32"):
            dst = operands[0]
            src_val, _, _ = self.parse_operand(operands[1])
            self.write_reg(dst, src_val)
        elif mnemonic in ("v_frexp_mant_f32", "v_frexp_mant_f32_e32", "v_frexp_mant_f32_e64"):
            dst = operands[0]
            src_val, _, _ = self.parse_operand(operands[1])
            res = ref_frexp_mant_f32_bits(src_val, ftz=self.ftz)
            self.write_reg(dst, res)
        elif mnemonic in ("v_frexp_exp_i32_f32", "v_frexp_exp_i32_f32_e32", "v_frexp_exp_i32_f32_e64"):
            dst = operands[0]
            src_val, _, _ = self.parse_operand(operands[1])
            res = ref_frexp_exp_i32_f32(src_val, ftz=self.ftz)
            self.write_reg(dst, res)
        else:
            raise NotImplementedError(f"Unsupported mnemonic: {mnemonic}")


class TestFrexpReferenceOracle(unittest.TestCase):
    """Unit tests verifying the bitwise and float reference oracles against IEEE 754 rules."""

    def test_signed_zeros(self):
        # +0.0: mantissa is +0.0, exponent is 0
        self.assertEqual(ref_frexp_mant_f32_bits(0x00000000), 0x00000000)
        self.assertEqual(ref_frexp_exp_i32_f32(0x00000000), 0)

        # -0.0: mantissa is -0.0 (sign bit preserved), exponent is 0
        self.assertEqual(ref_frexp_mant_f32_bits(0x80000000), 0x80000000)
        self.assertEqual(ref_frexp_exp_i32_f32(0x80000000), 0)

    def test_powers_of_two(self):
        cases = [
            # (float_value, expected_mantissa_float, expected_exponent_int)
            (1.0, 0.5, 1),
            (0.5, 0.5, 0),
            (0.25, 0.5, -1),
            (0.125, 0.5, -2),
            (2.0, 0.5, 2),
            (4.0, 0.5, 3),
            (8.0, 0.5, 4),
            (1024.0, 0.5, 11),
            (-1.0, -0.5, 1),
            (-0.5, -0.5, 0),
            (-0.25, -0.5, -1),
            (-2.0, -0.5, 2),
            (-4.0, -0.5, 3),
        ]
        for val, exp_mant, exp_exp in cases:
            with self.subTest(val=val):
                bits = f32_to_bits(val)
                mant_bits = ref_frexp_mant_f32_bits(bits)
                mant_float = bits_to_f32(mant_bits)
                exp_int = s32(ref_frexp_exp_i32_f32(bits))

                self.assertAlmostEqual(mant_float, exp_mant, places=6)
                self.assertEqual(exp_int, exp_exp)
                # Verify exact bitwise exponent field is 126 (0x7E)
                self.assertEqual((mant_bits & EXP_MASK) >> EXP_SHIFT, 126)

    def test_representative_floats(self):
        cases = [
            (1.5, 0.75, 1),
            (-1.5, -0.75, 1),
            (3.0, 0.75, 2),
            (-3.0, -0.75, 2),
            (0.75, 0.75, 0),
            (-0.75, -0.75, 0),
            (3.1415927, 3.1415927 / 4.0, 2),
            (-42.125, -42.125 / 64.0, 6),
            (100.0, 100.0 / 128.0, 7),
            (0.001, 0.001 / (2.0 ** -9), -9),
        ]
        for val, exp_mant, exp_exp in cases:
            with self.subTest(val=val):
                bits = f32_to_bits(val)
                mant_float = bits_to_f32(ref_frexp_mant_f32_bits(bits))
                exp_int = s32(ref_frexp_exp_i32_f32(bits))

                self.assertAlmostEqual(mant_float, exp_mant, places=6)
                self.assertEqual(exp_int, exp_exp)

    def test_boundary_normals(self):
        # FLT_MAX: largest finite single-precision float
        flt_max_bits = 0x7F7FFFFF
        flt_max = bits_to_f32(flt_max_bits)
        mant_bits = ref_frexp_mant_f32_bits(flt_max_bits)
        exp_int = s32(ref_frexp_exp_i32_f32(flt_max_bits))
        self.assertEqual(exp_int, 128)
        self.assertEqual(mant_bits, 0x3F7FFFFF)  # exponent 126, all fraction bits 1
        self.assertAlmostEqual(bits_to_f32(mant_bits), 1.0 - (2.0 ** -24), places=6)

        # -FLT_MAX
        flt_max_neg_bits = 0xFF7FFFFF
        mant_neg_bits = ref_frexp_mant_f32_bits(flt_max_neg_bits)
        exp_neg_int = s32(ref_frexp_exp_i32_f32(flt_max_neg_bits))
        self.assertEqual(exp_neg_int, 128)
        self.assertEqual(mant_neg_bits, 0xBF7FFFFF)

        # FLT_MIN: smallest positive normalized float (2^-126)
        flt_min_bits = 0x00800000  # exp=1, frac=0
        mant_min_bits = ref_frexp_mant_f32_bits(flt_min_bits)
        exp_min_int = s32(ref_frexp_exp_i32_f32(flt_min_bits))
        self.assertEqual(exp_min_int, -125)
        self.assertEqual(mant_min_bits, 0x3F000000)  # 0.5f

    def test_special_ieee_infinities_and_nans(self):
        # +Infinity
        pos_inf = 0x7F800000
        self.assertEqual(ref_frexp_mant_f32_bits(pos_inf), pos_inf)
        self.assertEqual(ref_frexp_exp_i32_f32(pos_inf), 0)

        # -Infinity
        neg_inf = 0xFF800000
        self.assertEqual(ref_frexp_mant_f32_bits(neg_inf), neg_inf)
        self.assertEqual(ref_frexp_exp_i32_f32(neg_inf), 0)

        # Quiet NaN (+ and -)
        pos_qnan = 0x7FC00000
        neg_qnan = 0xFFC00000
        self.assertEqual(ref_frexp_mant_f32_bits(pos_qnan), pos_qnan)
        self.assertEqual(ref_frexp_exp_i32_f32(pos_qnan), 0)
        self.assertEqual(ref_frexp_mant_f32_bits(neg_qnan), neg_qnan)
        self.assertEqual(ref_frexp_exp_i32_f32(neg_qnan), 0)

        # Signaling NaN with custom payloads
        snans = [0x7F800001, 0x7F9ABCDE, 0xFF800001, 0xFFBEEF00]
        for snan in snans:
            with self.subTest(snan=hex(snan)):
                self.assertEqual(ref_frexp_mant_f32_bits(snan), snan)
                self.assertEqual(ref_frexp_exp_i32_f32(snan), 0)

    def test_subnormal_numbers(self):
        # Smallest subnormal: 1 * 2^-149
        sub_min = 0x00000001
        # In normalized mode: mantissa is 0.5, exp is -148
        self.assertEqual(ref_frexp_mant_f32_bits(sub_min, ftz=False), 0x3F000000)
        self.assertEqual(s32(ref_frexp_exp_i32_f32(sub_min, ftz=False)), -148)

        # Subnormal with power of two bit: 2^-130 -> frac has bit 19 set
        # value is 2^-130 = 0.5 * 2^-129
        sub_p2 = 1 << 19  # 0x00080000
        self.assertEqual(ref_frexp_mant_f32_bits(sub_p2, ftz=False), 0x3F000000)
        self.assertEqual(s32(ref_frexp_exp_i32_f32(sub_p2, ftz=False)), -129)

        # Largest subnormal: 0x007FFFFF
        sub_max = 0x007FFFFF
        self.assertEqual(s32(ref_frexp_exp_i32_f32(sub_max, ftz=False)), -126)
        mant_sub_max = ref_frexp_mant_f32_bits(sub_max, ftz=False)
        self.assertEqual((mant_sub_max & EXP_MASK) >> EXP_SHIFT, 126)

        # Flush to zero (FTZ) mode
        self.assertEqual(ref_frexp_mant_f32_bits(sub_min, ftz=True), 0x00000000)
        self.assertEqual(ref_frexp_exp_i32_f32(sub_min, ftz=True), 0)
        self.assertEqual(ref_frexp_mant_f32_bits(0x80000001, ftz=True), 0x80000000)
        self.assertEqual(ref_frexp_exp_i32_f32(0x80000001, ftz=True), 0)

    def test_cross_check_with_python_math_frexp(self):
        """Cross-checks reference oracle against standard library math.frexp for 1,000 floats."""
        for exp in range(-125, 128):
            for frac_mult in (1.0, 1.25, 1.5, 1.75, 1.9999):
                val = frac_mult * (2.0 ** (exp - 1))
                if val > 3.4e38:
                    continue
                # Test both positive and negative
                for sign in (1.0, -1.0):
                    x = sign * val
                    py_mant, py_exp = math.frexp(x)
                    ref_mant = ref_frexp_mant_f32(x)
                    ref_exp = ref_frexp_exp_i32(x)

                    self.assertAlmostEqual(ref_mant, py_mant, places=6)
                    self.assertEqual(ref_exp, py_exp)


class TestFrexpMathematicalInvariants(unittest.TestCase):
    """Verifies fundamental algebraic and IEEE 754 invariants of the frexp decomposition."""

    def test_exact_reconstruction(self):
        """Invariant: x == mantissa * (2 ** exponent) for all non-zero finite normal numbers."""
        test_values = [
            1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5, -1.5, 42.0, -42.0,
            0.1, -0.1, 1000.0, 0.0001, 3.14159265, 2.71828182,
            bits_to_f32(0x3F800001), bits_to_f32(0x3F7FFFFF),
        ]
        for val in test_values:
            with self.subTest(val=val):
                mant = ref_frexp_mant_f32(val)
                exp = ref_frexp_exp_i32(val)
                reconstructed = mant * (2.0 ** exp)
                self.assertAlmostEqual(reconstructed, val, places=6)

    def test_mantissa_bounds(self):
        """Invariant: 0.5 <= |mantissa| < 1.0 for all non-zero finite numbers."""
        test_words = [
            0x3F800000, 0x40000000, 0x3F000000, 0x3E800000, 0x44800000,
            0xBF800000, 0xC0000000, 0x7F7FFFFF, 0x00800000,
        ]
        for word in test_words:
            with self.subTest(word=hex(word)):
                mant = abs(bits_to_f32(ref_frexp_mant_f32_bits(word)))
                self.assertGreaterEqual(mant, 0.5)
                self.assertLess(mant, 1.0)

    def test_sign_preservation_and_reflection(self):
        """Invariant: signbit(mantissa) == signbit(x) and mantissa(-x) == -mantissa(x)."""
        test_words = [0x3F800000, 0x40490FDB, 0x3EAAAAAB, 0x42280000]
        for word in test_words:
            neg_word = word ^ SIGN_MASK
            mant_pos = ref_frexp_mant_f32_bits(word)
            mant_neg = ref_frexp_mant_f32_bits(neg_word)

            # Mantissa sign bit must match input sign bit
            self.assertEqual(mant_pos & SIGN_MASK, 0)
            self.assertEqual(mant_neg & SIGN_MASK, SIGN_MASK)
            # Magnitudes must be identical
            self.assertEqual(mant_pos & ~SIGN_MASK, mant_neg & ~SIGN_MASK)

    def test_sign_invariance_of_exponent(self):
        """Invariant: exponent(x) == exponent(-x) for any float x."""
        for word in [0x3F800000, 0x40490FDB, 0x3F000000, 0x7F7FFFFF, 0x00800000]:
            neg_word = word ^ SIGN_MASK
            self.assertEqual(ref_frexp_exp_i32_f32(word), ref_frexp_exp_i32_f32(neg_word))

    def test_idempotence_properties(self):
        """Invariant: mantissa(mantissa(x)) == mantissa(x) and exponent(mantissa(x)) == 0."""
        for word in [0x3F800000, 0x40490FDB, 0xC2288000, 0x3EAAAAAB]:
            mant1 = ref_frexp_mant_f32_bits(word)
            mant2 = ref_frexp_mant_f32_bits(mant1)
            exp_mant1 = ref_frexp_exp_i32_f32(mant1)

            self.assertEqual(mant1, mant2)
            self.assertEqual(exp_mant1, 0)

    def test_scaling_by_powers_of_two(self):
        """Invariant: mantissa(x * 2^k) == mantissa(x) and exponent(x * 2^k) == exponent(x) + k."""
        base = 1.5
        for k in range(-5, 6):
            scaled = base * (2.0 ** k)
            self.assertAlmostEqual(ref_frexp_mant_f32(scaled), ref_frexp_mant_f32(base), places=6)
            self.assertEqual(ref_frexp_exp_i32(scaled), ref_frexp_exp_i32(base) + k)


class TestSpirvFrexpMiniInterpreter(unittest.TestCase):
    """Verifies that SPIR-V instruction sequences for frexp mantissa and exponent run accurately in spirv_mini_interp."""

    def test_spirv_exponent_extraction(self):
        test_cases = [
            (0x3F800000, 1),           # 1.0f -> exp 1
            (0x3F000000, 0),           # 0.5f -> exp 0
            (0x40000000, 2),           # 2.0f -> exp 2
            (0x3E800000, -1 & MASK32), # 0.25f -> exp -1
            (0xBF800000, 1),           # -1.0f -> exp 1
            (0x00000000, 0),           # +0.0 -> 0
            (0x80000000, 0),           # -0.0 -> 0
            (0x7F800000, 0),           # +INF -> 0
            (0xFF800000, 0),           # -INF -> 0
            (0x7FC00000, 0),           # NaN -> 0
        ]
        for in_bits, want_exp in test_cases:
            with self.subTest(in_bits=hex(in_bits)):
                got_exp = run_spirv_frexp_exp(in_bits)
                self.assertEqual(got_exp, want_exp)
                self.assertEqual(got_exp, ref_frexp_exp_i32_f32(in_bits))

    def test_spirv_mantissa_extraction(self):
        test_cases = [
            (0x3F800000, 0x3F000000),  # 1.0f -> 0.5f
            (0x3F000000, 0x3F000000),  # 0.5f -> 0.5f
            (0x40000000, 0x3F000000),  # 2.0f -> 0.5f
            (0x3FC00000, 0x3F400000),  # 1.5f -> 0.75f
            (0xBFC00000, 0xBF400000),  # -1.5f -> -0.75f
            (0x00000000, 0x00000000),  # +0.0 -> +0.0
            (0x80000000, 0x80000000),  # -0.0 -> -0.0
            (0x7F800000, 0x7F800000),  # +INF -> +INF
            (0xFF800000, 0xFF800000),  # -INF -> -INF
            (0x7FC00000, 0x7FC00000),  # NaN -> NaN
        ]
        for in_bits, want_mant in test_cases:
            with self.subTest(in_bits=hex(in_bits)):
                got_mant = run_spirv_frexp_mant(in_bits)
                self.assertEqual(got_mant, want_mant)
                self.assertEqual(got_mant, ref_frexp_mant_f32_bits(in_bits))

    def test_spirv_end_to_end_reconstruction(self):
        """Constructs a SPIR-V module that extracts mantissa and exponent and reconstructs the float."""
        mod = """OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%bool = OpTypeBool
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_in = OpConstant %uint 1077936128
%c_23 = OpConstant %uint 23
%c_255 = OpConstant %uint 255
%c_126 = OpConstant %uint 126
%c_mask_mant = OpConstant %uint 2155872255
%c_exp_bits = OpConstant %uint 1056964608
%c_exp_bias = OpConstant %uint 127
%main = OpFunction %void None %fn
%entry = OpLabel
%s = OpShiftRightLogical %uint %c_in %c_23
%exp_biased = OpBitwiseAnd %uint %s %c_255
%unbiased = OpISub %uint %exp_biased %c_126
%stripped = OpBitwiseAnd %uint %c_in %c_mask_mant
%mant_bits = OpBitwiseOr %uint %stripped %c_exp_bits
%mant_f32 = OpBitcast %float %mant_bits
%scale_exp = OpIAdd %uint %unbiased %c_exp_bias
%scale_bits = OpShiftLeftLogical %uint %scale_exp %c_23
%scale_f32 = OpBitcast %float %scale_bits
%reconstructed = OpFMul %float %mant_f32 %scale_f32
%res_bits = OpBitcast %uint %reconstructed
OpStore %out %res_bits
OpReturn
OpFunctionEnd
"""
        # 1077936128 is 0x40400000 (3.0f). Reconstructed value must be 0x40400000.
        res = si.run_fragment(mod)["%out"][0]
        self.assertEqual(res, 0x40400000)


class TestFrexpValuInterpreter(unittest.TestCase):
    """Tests simulated execution of VALU assembly instructions v_frexp_mant_f32 and v_frexp_exp_i32_f32."""

    def setUp(self):
        self.interp = FrexpValuInterpreter()

    def test_v_frexp_mant_f32_e32(self):
        # v_mov_b32 v1, 0x40000000 (2.0f)
        # v_frexp_mant_f32_e32 v0, v1
        self.interp.execute_line("v_mov_b32 v1, 0x40000000")
        self.interp.execute_line("v_frexp_mant_f32_e32 v0, v1")
        self.assertEqual(self.interp.read_reg("v0"), 0x3F000000)  # 0.5f

    def test_v_frexp_exp_i32_f32_e32(self):
        # v_mov_b32 v1, 0x40000000 (2.0f)
        # v_frexp_exp_i32_f32_e32 v0, v1
        self.interp.execute_line("v_mov_b32 v1, 0x40000000")
        self.interp.execute_line("v_frexp_exp_i32_f32_e32 v0, v1")
        self.assertEqual(self.interp.read_reg("v0"), 2)

    def test_sgpr_operand_source(self):
        self.interp.execute_line("s_mov_b32 s4, 0x3FC00000")  # 1.5f
        self.interp.execute_line("v_frexp_mant_f32 v0, s4")
        self.interp.execute_line("v_frexp_exp_i32_f32 v1, s4")
        self.assertEqual(self.interp.read_reg("v0"), 0x3F400000)  # 0.75f
        self.assertEqual(self.interp.read_reg("v1"), 1)

    def test_inline_constants(self):
        self.interp.execute_line("v_frexp_mant_f32 v0, 1.0")
        self.interp.execute_line("v_frexp_exp_i32_f32 v1, 1.0")
        self.assertEqual(self.interp.read_reg("v0"), 0x3F000000)
        self.assertEqual(self.interp.read_reg("v1"), 1)

        self.interp.execute_line("v_frexp_mant_f32 v2, 0.5")
        self.interp.execute_line("v_frexp_exp_i32_f32 v3, 0.5")
        self.assertEqual(self.interp.read_reg("v2"), 0x3F000000)
        self.assertEqual(self.interp.read_reg("v3"), 0)

        self.interp.execute_line("v_frexp_mant_f32 v4, 2.0")
        self.interp.execute_line("v_frexp_exp_i32_f32 v5, 2.0")
        self.assertEqual(self.interp.read_reg("v4"), 0x3F000000)
        self.assertEqual(self.interp.read_reg("v5"), 2)

    def test_vop3_e64_abs_modifier(self):
        # abs(|src|) on negative float clears sign bit
        self.interp.execute_line("v_mov_b32 v1, 0xBFC00000")  # -1.5f
        self.interp.execute_line("v_frexp_mant_f32_e64 v0, |v1|")
        self.assertEqual(self.interp.read_reg("v0"), 0x3F400000)  # +0.75f (sign bit stripped)

        # abs(|src|) on exponent extraction has no effect on result
        self.interp.execute_line("v_frexp_exp_i32_f32_e64 v2, |v1|")
        self.assertEqual(self.interp.read_reg("v2"), 1)

    def test_vop3_e64_neg_modifier(self):
        # neg(-src) on positive float inverts sign bit
        self.interp.execute_line("v_mov_b32 v1, 0x3FC00000")  # +1.5f
        self.interp.execute_line("v_frexp_mant_f32_e64 v0, -v1")
        self.assertEqual(self.interp.read_reg("v0"), 0xBF400000)  # -0.75f

        # neg(-src) on exponent extraction has no effect on result
        self.interp.execute_line("v_frexp_exp_i32_f32_e64 v2, -v1")
        self.assertEqual(self.interp.read_reg("v2"), 1)


class TestFrexpEdgeCasesAndBitPatterns(unittest.TestCase):
    """Exhaustive bit-pattern and edge-case testing for exponent and mantissa extraction."""

    def test_all_254_normal_exponents(self):
        """Iterates through all possible normal biased exponents (1..254) with zero fraction."""
        for exp in range(1, 255):
            word = exp << EXP_SHIFT
            mant_bits = ref_frexp_mant_f32_bits(word)
            exp_res = ref_frexp_exp_i32_f32(word)

            # Mantissa exponent must always be 126
            self.assertEqual((mant_bits & EXP_MASK) >> EXP_SHIFT, 126)
            # Fraction must remain 0
            self.assertEqual(mant_bits & MANT_MASK, 0)
            # Exponent must be exp - 126
            expected_unbiased = (exp - 126) & MASK32
            self.assertEqual(exp_res, expected_unbiased)

    def test_walking_fraction_bits(self):
        """Verifies each fraction bit (bit 0 through 22) is faithfully preserved in mantissa."""
        for bit_idx in range(23):
            frac = 1 << bit_idx
            word = 0x3F800000 | frac  # normal float near 1.0
            mant_bits = ref_frexp_mant_f32_bits(word)

            # Fraction bits must match exactly
            self.assertEqual(mant_bits & MANT_MASK, frac)
            # Exponent must be replaced with 126
            self.assertEqual((mant_bits & EXP_MASK) >> EXP_SHIFT, 126)

    def test_alternating_bit_patterns(self):
        """Tests bit patterns with alternating ones and zeros in the mantissa fraction."""
        patterns = [0x555555, 0x2AAAAA, 0x7FFFFF, 0x000001]
        for pat in patterns:
            word = (127 << EXP_SHIFT) | (pat & MANT_MASK)
            mant_bits = ref_frexp_mant_f32_bits(word)
            self.assertEqual(mant_bits & MANT_MASK, pat & MANT_MASK)
            self.assertEqual(ref_frexp_exp_i32_f32(word), 1)

    def test_nan_payload_preservation(self):
        """Verifies NaN payload bits are completely preserved by v_frexp_mant_f32."""
        for payload in [1, 2, 0x3FFFFF, 0x200000, 0x101010]:
            nan_bits = EXP_MASK | payload
            self.assertEqual(ref_frexp_mant_f32_bits(nan_bits), nan_bits)
            self.assertEqual(ref_frexp_exp_i32_f32(nan_bits), 0)


if __name__ == "__main__":
    unittest.main()
