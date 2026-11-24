"""Semantic interpreter tests for AMD RDNA2 v_subrev_f32, v_madak_f32, and v_madmk_f32.

Instructions covered:
  - v_subrev_f32: D.f32 = S1.f32 - S0.f32 (VOP2 0x33 / VOP3 0x2bb on GFX10), the reversed form of v_sub_f32.
    Supports VOP3 clamp to [0, 1] and input modifiers (neg, abs) on either or both operands.
  - v_madak_f32: D.f32 = S0.f32 * S1.f32 + K.f32 (VOP2 0x3b on GFX10), where K is a 32-bit inline literal constant.
    Lowered to OpFMul followed by OpFAdd.
  - v_madmk_f32: D.f32 = S0.f32 * K.f32 + S1.f32 (VOP2 0x3c on GFX10), where K is a 32-bit inline literal constant.
    In assembly print order `vdst, src0, K, vsrc1`, S0 is multiplied by K and S1 is added.

Standard library only: zero game data, no GPU hardware required.
"""
from __future__ import annotations

import gc
import math
import os
import random
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
from support import mrt0_words, translate_lines  # noqa: E402

EXIT = ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def f2u(x: float) -> int:
    """Float -> uint32 bit pattern."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(u: int) -> float:
    """Uint32 bit pattern -> float."""
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def ref_subrev_f32(s0: float, s1: float) -> float:
    """Reference single-precision reversed subtract: S1 - S0."""
    return u2f(f2u(s1 - s0))


def ref_subrev_f32_clamped(s0: float, s1: float) -> float:
    """Reference reversed subtract with VOP3 clamp to [0.0, 1.0]."""
    diff = s1 - s0
    if math.isnan(diff):
        return 0.0
    return max(0.0, min(1.0, diff))


def ref_madak_f32(s0: float, s1: float, k: float) -> float:
    """Reference non-fused multiply-add for v_madak_f32: S0 * S1 + K."""
    prod = u2f(f2u(s0 * s1))
    return u2f(f2u(prod + k))


def ref_madmk_f32(s0: float, k: float, s1: float) -> float:
    """Reference non-fused multiply-add for v_madmk_f32: S0 * K + S1."""
    prod = u2f(f2u(s0 * k))
    return u2f(f2u(prod + s1))


class SubrevF32Tests(unittest.TestCase):
    """Semantic tests for v_subrev_f32."""

    def _run_subrev(self, s0_val: float, s1_val: float, opcode: str = "v_subrev_f32") -> float:
        lines = [
            f"v_mov_b32 v1, 0x{f2u(s0_val):08x}",
            f"v_mov_b32 v2, 0x{f2u(s1_val):08x}",
            f"{opcode} v0, v1, v2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        spv = translate_lines("ps_subrev", lines)
        res = mrt0_words(spv)
        return u2f(res[0])

    def test_operand_order_reversed_basic(self):
        # 3.5 - 1.0 = 2.5
        res = self._run_subrev(1.0, 3.5)
        self.assertAlmostEqual(res, 2.5, places=6)

        # 1.0 - 3.5 = -2.5
        res_neg = self._run_subrev(3.5, 1.0)
        self.assertAlmostEqual(res_neg, -2.5, places=6)

    def test_contrasting_with_v_sub_f32(self):
        # Guarantee that v_subrev_f32(a, b) == -v_sub_f32(a, b)
        a, b = 2.75, 8.125
        res_subrev = self._run_subrev(a, b)

        lines_sub = [
            f"v_mov_b32 v1, 0x{f2u(a):08x}",
            f"v_mov_b32 v2, 0x{f2u(b):08x}",
            "v_sub_f32 v0, v1, v2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        spv_sub = translate_lines("ps_sub", lines_sub)
        res_sub = u2f(mrt0_words(spv_sub)[0])

        self.assertAlmostEqual(res_subrev, b - a, places=6)
        self.assertAlmostEqual(res_sub, a - b, places=6)
        self.assertAlmostEqual(res_subrev, -res_sub, places=6)

    def test_identities_and_zeros(self):
        # x - 0.0 = x
        self.assertEqual(self._run_subrev(0.0, 42.0), 42.0)
        # 0.0 - x = -x
        self.assertEqual(self._run_subrev(42.0, 0.0), -42.0)
        # x - x = 0.0
        self.assertEqual(self._run_subrev(13.5, 13.5), 0.0)

    def test_negative_values(self):
        # (-2.0) - 5.0 = -7.0 -> subrev(5.0, -2.0) = -2.0 - 5.0 = -7.0
        self.assertEqual(self._run_subrev(5.0, -2.0), -7.0)
        # 5.0 - (-2.0) = 7.0 -> subrev(-2.0, 5.0) = 5.0 - (-2.0) = 7.0
        self.assertEqual(self._run_subrev(-2.0, 5.0), 7.0)

    def test_vop3_clamp(self):
        # Below zero clamps to 0.0
        lines_under = [
            f"v_mov_b32 v1, 0x{f2u(4.0):08x}",
            f"v_mov_b32 v2, 0x{f2u(1.0):08x}",
            "v_subrev_f32_e64 v0, v1, v2 clamp",  # 1.0 - 4.0 = -3.0 -> clamp 0.0
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        res_under = u2f(mrt0_words(translate_lines("ps_subrev_clamp1", lines_under))[0])
        self.assertEqual(res_under, 0.0)

        # Above 1.0 clamps to 1.0
        lines_over = [
            f"v_mov_b32 v1, 0x{f2u(2.0):08x}",
            f"v_mov_b32 v2, 0x{f2u(5.0):08x}",
            "v_subrev_f32_e64 v0, v1, v2 clamp",  # 5.0 - 2.0 = 3.0 -> clamp 1.0
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        res_over = u2f(mrt0_words(translate_lines("ps_subrev_clamp2", lines_over))[0])
        self.assertEqual(res_over, 1.0)

        # In-range preserves value
        lines_mid = [
            f"v_mov_b32 v1, 0x{f2u(0.25):08x}",
            f"v_mov_b32 v2, 0x{f2u(0.75):08x}",
            "v_subrev_f32_e64 v0, v1, v2 clamp",  # 0.75 - 0.25 = 0.5
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        res_mid = u2f(mrt0_words(translate_lines("ps_subrev_clamp3", lines_mid))[0])
        self.assertEqual(res_mid, 0.5)

    def test_vop3_input_modifiers(self):
        # v2 - (-v1) = v2 + v1
        lines_neg0 = [
            f"v_mov_b32 v1, 0x{f2u(2.0):08x}",
            f"v_mov_b32 v2, 0x{f2u(3.0):08x}",
            "v_subrev_f32_e64 v0, -v1, v2",  # 3.0 - (-2.0) = 5.0
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        self.assertEqual(u2f(mrt0_words(translate_lines("ps_neg0", lines_neg0))[0]), 5.0)

        # (-v2) - v1 = -(v2 + v1)
        lines_neg1 = [
            f"v_mov_b32 v1, 0x{f2u(2.0):08x}",
            f"v_mov_b32 v2, 0x{f2u(3.0):08x}",
            "v_subrev_f32_e64 v0, v1, -v2",  # (-3.0) - 2.0 = -5.0
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        self.assertEqual(u2f(mrt0_words(translate_lines("ps_neg1", lines_neg1))[0]), -5.0)

        # v2 - |v1|
        lines_abs0 = [
            f"v_mov_b32 v1, 0x{f2u(-4.0):08x}",
            f"v_mov_b32 v2, 0x{f2u(10.0):08x}",
            "v_subrev_f32_e64 v0, |v1|, v2",  # 10.0 - |-4.0| = 6.0
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        self.assertEqual(u2f(mrt0_words(translate_lines("ps_abs0", lines_abs0))[0]), 6.0)

        # |v2| - v1
        lines_abs1 = [
            f"v_mov_b32 v1, 0x{f2u(3.0):08x}",
            f"v_mov_b32 v2, 0x{f2u(-7.0):08x}",
            "v_subrev_f32_e64 v0, v1, |v2|",  # |-7.0| - 3.0 = 4.0
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        self.assertEqual(u2f(mrt0_words(translate_lines("ps_abs1", lines_abs1))[0]), 4.0)

    def test_inline_constants_and_sgprs(self):
        # v1 - 1.0
        lines_const = [
            f"v_mov_b32 v1, 0x{f2u(3.5):08x}",
            "v_subrev_f32 v0, 1.0, v1",  # 3.5 - 1.0 = 2.5
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        self.assertEqual(u2f(mrt0_words(translate_lines("ps_const", lines_const))[0]), 2.5)

        # v1 - s0
        lines_sgpr = [
            f"s_mov_b32 s0, 0x{f2u(1.25):08x}",
            f"v_mov_b32 v1, 0x{f2u(6.25):08x}",
            "v_subrev_f32 v0, s0, v1",  # 6.25 - 1.25 = 5.0
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        self.assertEqual(u2f(mrt0_words(translate_lines("ps_sgpr", lines_sgpr))[0]), 5.0)

    def test_spirv_lowering_instruction(self):
        lines = [
            "v_mov_b32 v1, 0x3f800000",
            "v_mov_b32 v2, 0x40800000",
            "v_subrev_f32 v0, v1, v2",
        ] + EXIT
        spv = translate_lines("ps_spv_check", lines)
        # S1 is first argument to OpFSub, S0 is second argument
        self.assertIn("OpFSub %float", spv)


class MadakF32Tests(unittest.TestCase):
    """Semantic tests for v_madak_f32: D = S0 * S1 + K."""

    def _run_madak(self, s0: float, s1: float, k: float) -> float:
        lines = [
            f"v_mov_b32 v1, 0x{f2u(s0):08x}",
            f"v_mov_b32 v2, 0x{f2u(s1):08x}",
            f"v_madak_f32 v0, v1, v2, 0x{f2u(k):08x}",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        spv = translate_lines("ps_madak", lines)
        res = mrt0_words(spv)
        return u2f(res[0])

    def test_asymmetric_operands(self):
        # 2.0 * 4.0 + 3.0 = 11.0
        self.assertEqual(self._run_madak(2.0, 4.0, 3.0), 11.0)
        # 3.0 * 5.0 + (-2.0) = 13.0
        self.assertEqual(self._run_madak(3.0, 5.0, -2.0), 13.0)
        # -1.5 * 4.0 + 10.0 = 4.0
        self.assertEqual(self._run_madak(-1.5, 4.0, 10.0), 4.0)

    def test_zero_operands(self):
        # 0.0 * 100.0 + 7.5 = 7.5
        self.assertEqual(self._run_madak(0.0, 100.0, 7.5), 7.5)
        # 100.0 * 0.0 + 7.5 = 7.5
        self.assertEqual(self._run_madak(100.0, 0.0, 7.5), 7.5)
        # 2.0 * 3.0 + 0.0 = 6.0
        self.assertEqual(self._run_madak(2.0, 3.0, 0.0), 6.0)

    def test_sgpr_and_inline_constant_source(self):
        # S0 from SGPR: s0 * v1 + K
        lines_sgpr = [
            f"s_mov_b32 s0, 0x{f2u(3.0):08x}",
            f"v_mov_b32 v1, 0x{f2u(4.0):08x}",
            f"v_madak_f32 v0, s0, v1, 0x{f2u(5.0):08x}",  # 3.0 * 4.0 + 5.0 = 17.0
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        res_sgpr = u2f(mrt0_words(translate_lines("ps_madak_sgpr", lines_sgpr))[0])
        self.assertEqual(res_sgpr, 17.0)

        # S0 from inline constant 1.0: 1.0 * v1 + K
        lines_const = [
            f"v_mov_b32 v1, 0x{f2u(4.0):08x}",
            f"v_madak_f32 v0, 1.0, v1, 0x{f2u(3.0):08x}",  # 1.0 * 4.0 + 3.0 = 7.0
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        res_const = u2f(mrt0_words(translate_lines("ps_madak_const", lines_const))[0])
        self.assertEqual(res_const, 7.0)

    def test_spirv_lowering_unfused(self):
        lines = [
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x40800000",
            "v_madak_f32 v0, v1, v2, 0x40400000",
        ] + EXIT
        spv = translate_lines("ps_madak_spv", lines)
        # Must emit OpFMul then OpFAdd (not fused Fma)
        self.assertIn("OpFMul %float", spv)
        self.assertIn("OpFAdd %float", spv)
        self.assertNotIn("%glsl Fma", spv)


class MadmkF32Tests(unittest.TestCase):
    """Semantic tests for v_madmk_f32: D = S0 * K + S1."""

    def _run_madmk(self, s0: float, k: float, s1: float) -> float:
        lines = [
            f"v_mov_b32 v1, 0x{f2u(s0):08x}",
            f"v_mov_b32 v2, 0x{f2u(s1):08x}",
            f"v_madmk_f32 v0, v1, 0x{f2u(k):08x}, v2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        spv = translate_lines("ps_madmk", lines)
        res = mrt0_words(spv)
        return u2f(res[0])

    def test_asymmetric_operands_and_factor_order(self):
        # 2.0 * 3.0 + 4.0 = 10.0
        # If buggy factor order (multiplying S0 * S1 + K): 2.0 * 4.0 + 3.0 = 11.0 != 10.0
        res = self._run_madmk(2.0, 3.0, 4.0)
        self.assertEqual(res, 10.0)

        # 5.0 * 0.5 + 1.5 = 4.0
        self.assertEqual(self._run_madmk(5.0, 0.5, 1.5), 4.0)

        # -4.0 * 2.5 + 15.0 = 5.0
        self.assertEqual(self._run_madmk(-4.0, 2.5, 15.0), 5.0)

    def test_zero_operands(self):
        # 0.0 * 3.0 + 4.0 = 4.0
        self.assertEqual(self._run_madmk(0.0, 3.0, 4.0), 4.0)
        # 2.0 * 0.0 + 4.0 = 4.0
        self.assertEqual(self._run_madmk(2.0, 0.0, 4.0), 4.0)
        # 2.0 * 3.0 + 0.0 = 6.0
        self.assertEqual(self._run_madmk(2.0, 3.0, 0.0), 6.0)

    def test_sgpr_and_inline_constant_source(self):
        # S0 from SGPR: s0 * K + v1
        lines_sgpr = [
            f"s_mov_b32 s0, 0x{f2u(3.0):08x}",
            f"v_mov_b32 v1, 0x{f2u(4.0):08x}",
            f"v_madmk_f32 v0, s0, 0x{f2u(5.0):08x}, v1",  # 3.0 * 5.0 + 4.0 = 19.0
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        res_sgpr = u2f(mrt0_words(translate_lines("ps_madmk_sgpr", lines_sgpr))[0])
        self.assertEqual(res_sgpr, 19.0)

        # S0 from inline constant 1.0: 1.0 * K + v1
        lines_const = [
            f"v_mov_b32 v1, 0x{f2u(4.0):08x}",
            f"v_madmk_f32 v0, 1.0, 0x{f2u(3.0):08x}, v1",  # 1.0 * 3.0 + 4.0 = 7.0
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        res_const = u2f(mrt0_words(translate_lines("ps_madmk_const", lines_const))[0])
        self.assertEqual(res_const, 7.0)

    def test_spirv_lowering_unfused(self):
        lines = [
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x40800000",
            "v_madmk_f32 v0, v1, 0x40400000, v2",
        ] + EXIT
        spv = translate_lines("ps_madmk_spv", lines)
        self.assertIn("OpFMul %float", spv)
        self.assertIn("OpFAdd %float", spv)
        self.assertNotIn("%glsl Fma", spv)


class ComparativeAndSweepTests(unittest.TestCase):
    """Comparative tests and seeded input sweeps."""

    def test_madak_vs_madmk_equivalence(self):
        # When factors and addends match:
        # madak(a, b, c) -> a * b + c
        # madmk(a, b, c) -> a * b + c
        a, b, c = 3.25, 2.5, 7.125
        lines = [
            f"v_mov_b32 v1, 0x{f2u(a):08x}",
            f"v_mov_b32 v2, 0x{f2u(b):08x}",
            f"v_mov_b32 v3, 0x{f2u(c):08x}",
            f"v_madak_f32 v0, v1, v2, 0x{f2u(c):08x}",  # a * b + c
            f"v_madmk_f32 v1, v1, 0x{f2u(b):08x}, v3",  # a * b + c
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ] + EXIT
        spv = translate_lines("ps_equiv", lines)
        words = mrt0_words(spv)
        res_madak = u2f(words[0])
        res_madmk = u2f(words[1])
        self.assertEqual(res_madak, res_madmk)
        self.assertEqual(res_madak, ref_madak_f32(a, b, c))

    def test_seeded_sweep(self):
        rng = random.Random(0x1E041DA)
        for _ in range(50):
            s0 = rng.uniform(-100.0, 100.0)
            s1 = rng.uniform(-100.0, 100.0)
            k = rng.uniform(-100.0, 100.0)

            # Round to single precision float
            s0 = u2f(f2u(s0))
            s1 = u2f(f2u(s1))
            k = u2f(f2u(k))

            # Test subrev
            expected_subrev = ref_subrev_f32(s0, s1)
            lines_subrev = [
                f"v_mov_b32 v1, 0x{f2u(s0):08x}",
                f"v_mov_b32 v2, 0x{f2u(s1):08x}",
                "v_subrev_f32 v0, v1, v2",
                "v_mov_b32 v1, 0",
                "v_mov_b32 v2, 0",
                "v_mov_b32 v3, 0",
            ] + EXIT
            actual_subrev = u2f(mrt0_words(translate_lines("ps_sweep_subrev", lines_subrev))[0])
            self.assertEqual(f2u(actual_subrev), f2u(expected_subrev))

            # Test madak
            expected_madak = ref_madak_f32(s0, s1, k)
            lines_madak = [
                f"v_mov_b32 v1, 0x{f2u(s0):08x}",
                f"v_mov_b32 v2, 0x{f2u(s1):08x}",
                f"v_madak_f32 v0, v1, v2, 0x{f2u(k):08x}",
                "v_mov_b32 v1, 0",
                "v_mov_b32 v2, 0",
                "v_mov_b32 v3, 0",
            ] + EXIT
            actual_madak = u2f(mrt0_words(translate_lines("ps_sweep_madak", lines_madak))[0])
            self.assertEqual(f2u(actual_madak), f2u(expected_madak))

            # Test madmk
            expected_madmk = ref_madmk_f32(s0, k, s1)
            lines_madmk = [
                f"v_mov_b32 v1, 0x{f2u(s0):08x}",
                f"v_mov_b32 v2, 0x{f2u(s1):08x}",
                f"v_madmk_f32 v0, v1, 0x{f2u(k):08x}, v2",
                "v_mov_b32 v1, 0",
                "v_mov_b32 v2, 0",
                "v_mov_b32 v3, 0",
            ] + EXIT
            actual_madmk = u2f(mrt0_words(translate_lines("ps_sweep_madmk", lines_madmk))[0])
            self.assertEqual(f2u(actual_madmk), f2u(expected_madmk))


if __name__ == "__main__":
    unittest.main()
