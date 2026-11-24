"""End-to-end semantic tests for RDNA2 v_sqrt_f16 (D.f16 = sqrt(S0.f16)).

Covers v_sqrt_f16 executed end-to-end through tools/rdna2spirv.py (via
support.translate_lines) and tests/semantic/spirv_mini_interp.py:
  - Exact binary16 square roots (0.0, -0.0, 0.25, 1.0, 4.0, 9.0, powers of 2,
    and exact subnormal squares) plus fractional/non-exact and max finite
    values checked against a Python struct("<e", ...) reference.
  - IEEE-754 binary16 special cases: +inf (0x7C00 -> 0x7C00), NaNs (0x7E00,
    0x7E01, 0x7C01 -> NaN), and negative inputs (-1.0 = 0xBC00, -inf = 0xFC00),
    which give 0xFE00 (a negative quiet NaN, the hardware's result; the
    translator keeps the undefined GLSL.std.450 Sqrt(x < 0) out of it).
  - Preservation of the untouched 16-bit half of the destination VGPR and
    independence from the unused 16-bit half of the source VGPR.
  - _e32, _e64, and bare mnemonics, op_sel:[src,dst] ([0,0], [1,0], [0,1],
    [1,1]), VOP3 source/output modifiers (|v1|, clamp, mul:2, div:2), inline
    constants, SGPR sources, destination aliasing (v0, v0), and EXEC masking.
  - In-memory SPIR-V negative controls (mutating GLSL.std.450 Sqrt and
    altering the f16 half-extraction / destination-masking logic).

Standard library only; hand-written RDNA2 listings only; zero game data.
"""
from __future__ import annotations

import itertools
import math
import os
import re
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
from support import translate_lines  # noqa: E402

OLD = 0xBEEF1234
MASK32 = 0xFFFFFFFF

# Representative binary16 encodings.
H_POS_ZERO = 0x0000     # +0.0
H_NEG_ZERO = 0x8000     # -0.0
H_SUB_MIN = 0x0001      # 2^-24 (sqrt = 2^-12 = 0x0C00)
H_SUB_4 = 0x0004        # 4 * 2^-24 = 2^-22 (sqrt = 2^-11 = 0x1000)
H_SUB_2M16 = 0x0100     # 2^-16 (sqrt = 2^-8 = 0x1C00)
H_SUB_MAX = 0x03FF      # 1023 * 2^-24 (largest +subnormal)
H_NORM_MIN = 0x0400     # 2^-14 (sqrt = 2^-7 = 0x2000)
H_0P0625 = 0x2C00       # 0.0625 (sqrt = 0.25 = 0x3400)
H_0P25 = 0x3400         # 0.25 (sqrt = 0.5 = 0x3800)
H_0P5 = 0x3800          # 0.5
H_1P0 = 0x3C00          # 1.0 (sqrt = 1.0)
H_1P5 = 0x3E00          # 1.5
H_2P0 = 0x4000          # 2.0
H_3P0 = 0x4200          # 3.0
H_4P0 = 0x4400          # 4.0 (sqrt = 2.0)
H_9P0 = 0x4880          # 9.0 (sqrt = 3.0)
H_MAX = 0x7BFF          # 65504.0
H_NEG_1P0 = 0xBC00      # -1.0
H_NEG_4P0 = 0xC400      # -4.0
H_POS_INF = 0x7C00      # +inf
H_NEG_INF = 0xFC00      # -inf
H_QNAN = 0x7E00         # +qNaN
H_QNAN_PAYLOAD = 0x7E01  # +qNaN with payload
H_SNAN = 0x7C01         # +sNaN


def h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return H_POS_INF if x > 0 else H_NEG_INF


def is_f16_nan(h: int) -> bool:
    h &= 0xFFFF
    return (h & 0x7C00) == 0x7C00 and (h & 0x03FF) != 0


def f16_ulp_diff(a: int, b: int) -> int:
    """ULP distance between two finite non-NaN binary16 values."""
    def ordered(h: int) -> int:
        h &= 0xFFFF
        return -((h & 0x7FFF)) if (h & 0x8000) else h

    return abs(ordered(a) - ordered(b))


def merge_half(old_word: int, half: int, hi: int) -> int:
    if hi:
        return (old_word & 0x0000FFFF) | ((half & 0xFFFF) << 16)
    return (old_word & 0xFFFF0000) | (half & 0xFFFF)


def ref_sqrt_f16(h: int) -> int:
    x = h2f(h)
    if math.isnan(x):
        return H_QNAN
    if math.isinf(x):
        return H_POS_INF if x > 0 else H_QNAN
    if x == 0.0:
        return h & 0x8000
    return f2h(math.sqrt(x))


def run_ps(name: str, body_lines: list[str], **run_options):
    lines = list(body_lines) + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    spv = translate_lines(name, lines)
    out = si.run_fragment(spv, **run_options)
    return out["%out_mrt0"][0], spv


def run_op(op_line: str, v1_word: int, v0_init: int = OLD, s4: int | None = None) -> int:
    lines = [f"v_mov_b32 v0, {v0_init:#x}", f"v_mov_b32 v1, {v1_word:#x}"]
    if s4 is not None:
        lines.append(f"s_mov_b32 s4, {s4:#x}")
    lines.append(op_line)
    word, _ = run_ps("ps_sqrt_f16", lines)
    return word


class SqrtF16ExactAndFractionalTests(unittest.TestCase):
    def test_exact_square_roots_across_mnemonics(self):
        cases = (
            ("pos_zero", H_POS_ZERO, H_POS_ZERO),
            ("neg_zero", H_NEG_ZERO, H_NEG_ZERO),
            ("subnormal_2m24", H_SUB_MIN, 0x0C00),   # sqrt(2^-24) = 2^-12
            ("subnormal_2m22", H_SUB_4, 0x1000),     # sqrt(2^-22) = 2^-11
            ("subnormal_2m16", H_SUB_2M16, 0x1C00),  # sqrt(2^-16) = 2^-8
            ("norm_min_2m14", H_NORM_MIN, 0x2000),   # sqrt(2^-14) = 2^-7
            ("0.0625", H_0P0625, H_0P25),
            ("0.25", H_0P25, H_0P5),
            ("1.0", H_1P0, H_1P0),
            ("4.0", H_4P0, H_2P0),
            ("9.0", H_9P0, H_3P0),
        )
        for suffix in ("_e32", "_e64", ""):
            for name, h_in, want_half in cases:
                with self.subTest(suffix=suffix, case=name):
                    # High bits [31:16] of v1 must not affect the low-half result.
                    got = run_op(f"v_sqrt_f16{suffix} v0, v1", 0xCAFE0000 | h_in)
                    self.assertEqual(got, merge_half(OLD, want_half, 0))

    def test_fractional_and_subnormal_values_within_1ulp(self):
        cases = (
            H_SUB_MAX,
            0x0200,     # 2^-15 (subnormal)
            0x1234,     # small normal
            H_0P5,
            H_1P5,
            H_2P0,
            H_3P0,
            0x4500,     # 5.0
            0x4900,     # 10.0
            0x5640,     # 100.0
            H_MAX,      # 65504.0
        )
        for h_in in cases:
            want = ref_sqrt_f16(h_in)
            with self.subTest(h_in=hex(h_in), want=hex(want)):
                got = run_op("v_sqrt_f16_e32 v0, v1", 0xDEAD0000 | h_in)
                self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)
                self.assertLessEqual(f16_ulp_diff(got & 0xFFFF, want), 1)


class SqrtF16SpecialCasesTests(unittest.TestCase):
    def test_pos_inf_and_nans(self):
        for suffix in ("_e32", "_e64"):
            got_inf = run_op(f"v_sqrt_f16{suffix} v0, v1", 0xCAFE0000 | H_POS_INF)
            self.assertEqual(got_inf, merge_half(OLD, H_POS_INF, 0))

            for h_nan in (H_QNAN, H_QNAN_PAYLOAD, H_SNAN):
                with self.subTest(suffix=suffix, h_nan=hex(h_nan)):
                    got_nan = run_op(f"v_sqrt_f16{suffix} v0, v1", 0xCAFE0000 | h_nan)
                    self.assertEqual(got_nan & 0xFFFF0000, OLD & 0xFFFF0000)
                    self.assertTrue(is_f16_nan(got_nan & 0xFFFF), f"expected NaN, got {got_nan:#010x}")

    def test_negative_inputs_give_negative_quiet_nan(self):
        # GLSL.std.450 leaves Sqrt(x < 0) undefined, so the translator selects the hardware's 0xFE00 itself.
        for line, v1_word, dst_hi in (
            ("v_sqrt_f16_e32 v0, v1", (H_1P0 << 16) | H_NEG_1P0, 0),
            ("v_sqrt_f16_e32 v0, v1", (H_1P0 << 16) | H_NEG_INF, 0),
            ("v_sqrt_f16_e64 v0, v1 op_sel:[1,0]", (H_NEG_1P0 << 16) | H_1P0, 0),
            ("v_sqrt_f16_e64 v0, v1 op_sel:[1,1]", (H_NEG_INF << 16) | H_1P0, 1),
        ):
            with self.subTest(line=line, v1=hex(v1_word)):
                self.assertEqual(run_op(line, v1_word), merge_half(OLD, 0xFE00, dst_hi))


class SqrtF16ModifiersAndOperandsTests(unittest.TestCase):
    def test_op_sel_all_combinations(self):
        # v1 = (hi: 9.0, lo: 4.0) -> sqrt(hi) = 3.0, sqrt(lo) = 2.0.
        v1 = (H_9P0 << 16) | H_4P0
        for src_hi, dst_hi in itertools.product((0, 1), repeat=2):
            want_half = H_3P0 if src_hi else H_2P0
            line = f"v_sqrt_f16_e64 v0, v1 op_sel:[{src_hi},{dst_hi}]"
            with self.subTest(line=line):
                self.assertEqual(run_op(line, v1), merge_half(OLD, want_half, dst_hi))

    def test_vop3_abs_neg_and_output_modifiers(self):
        v1 = (H_NEG_4P0 << 16) | H_NEG_1P0
        self.assertEqual(run_op("v_sqrt_f16_e64 v0, |v1| op_sel:[0,0]", v1), merge_half(OLD, H_1P0, 0))
        self.assertEqual(run_op("v_sqrt_f16_e64 v0, |v1| op_sel:[1,1]", v1), merge_half(OLD, H_2P0, 1))
        # -v1 on a negative value turns it positive before sqrt.
        self.assertEqual(run_op("v_sqrt_f16_e64 v0, -v1 op_sel:[1,0]", v1), merge_half(OLD, H_2P0, 0))
        # -|v1| is negative: 0xFE00.
        self.assertEqual(run_op("v_sqrt_f16_e64 v0, -|v1| op_sel:[0,0]", v1), merge_half(OLD, 0xFE00, 0))

        # Output modifiers: mul:2 and div:2 ignored on RDNA2 with denorms kept; clamp applies.
        v1_pos = (H_9P0 << 16) | H_4P0
        self.assertEqual(run_op("v_sqrt_f16_e64 v0, v1 mul:2", v1_pos), merge_half(OLD, H_2P0, 0))
        self.assertEqual(run_op("v_sqrt_f16_e64 v0, v1 div:2", v1_pos), merge_half(OLD, H_2P0, 0))
        self.assertEqual(run_op("v_sqrt_f16_e64 v0, v1 clamp", v1_pos), merge_half(OLD, H_1P0, 0))
        self.assertEqual(run_op("v_sqrt_f16_e64 v0, v1 op_sel:[1,1] clamp", v1_pos), merge_half(OLD, H_1P0, 1))

    def test_inline_constants_sgpr_aliasing_and_exec_mask(self):
        self.assertEqual(run_op("v_sqrt_f16_e32 v0, 4.0", 0), merge_half(OLD, H_2P0, 0))
        self.assertEqual(run_op("v_sqrt_f16_e32 v0, 0.25", 0), merge_half(OLD, H_0P5, 0))
        self.assertEqual(run_op("v_sqrt_f16_e64 v0, |0xBC00|", 0), merge_half(OLD, H_1P0, 0))

        s4 = (H_9P0 << 16) | H_4P0
        self.assertEqual(run_op("v_sqrt_f16_e32 v0, s4", 0, s4=s4), merge_half(OLD, H_2P0, 0))
        self.assertEqual(run_op("v_sqrt_f16_e64 v0, s4 op_sel:[1,1]", 0, s4=s4), merge_half(OLD, H_3P0, 1))

        # Destination aliasing: v0 as both source and destination.
        word_alias, _ = run_ps("ps_sqrt_alias", [
            f"v_mov_b32 v0, {(H_9P0 << 16) | H_4P0:#x}",
            "v_sqrt_f16_e64 v0, v0 op_sel:[1,0]",
        ])
        self.assertEqual(word_alias, (H_9P0 << 16) | H_3P0)

        # EXEC masking: when lane 0 is masked off (exec = 0x2), v0 keeps OLD.
        word_masked, _ = run_ps("ps_sqrt_exec_off", [
            f"v_mov_b32 v0, {OLD:#x}",
            f"v_mov_b32 v1, {H_4P0:#x}",
            "s_mov_b64 exec, 0x2",
            "v_sqrt_f16_e32 v0, v1",
            "s_mov_b64 exec, -1",
        ])
        self.assertEqual(word_masked, OLD)

        # EXEC masking: when lane 0 is active (exec = 0x1), v0 is updated.
        word_active, _ = run_ps("ps_sqrt_exec_on", [
            f"v_mov_b32 v0, {OLD:#x}",
            f"v_mov_b32 v1, {H_4P0:#x}",
            "s_mov_b64 exec, 0x1",
            "v_sqrt_f16_e32 v0, v1",
            "s_mov_b64 exec, -1",
        ])
        self.assertEqual(word_active, merge_half(OLD, H_2P0, 0))


class SqrtF16NegativeControlsTests(unittest.TestCase):
    def test_sabotaged_spirv_changes_output(self):
        want = merge_half(OLD, H_2P0, 0)
        _, spv = run_ps("ps_sqrt_neg_ctrl", [
            f"v_mov_b32 v0, {OLD:#x}",
            f"v_mov_b32 v1, {(H_9P0 << 16) | H_4P0:#x}",
            "v_sqrt_f16_e32 v0, v1",
        ])
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], want)

        # Negative control 1: replace GLSL.std.450 Sqrt with InverseSqrt (sqrt(4)=2.0 -> 1/sqrt(4)=0.5).
        bad_op, n1 = re.subn(r"OpExtInst %float %glsl Sqrt\b", "OpExtInst %float %glsl InverseSqrt", spv, count=1)
        self.assertEqual(n1, 1)
        got_bad_op = si.run_fragment(bad_op)["%out_mrt0"][0]
        self.assertNotEqual(got_bad_op, want)
        self.assertEqual(got_bad_op, merge_half(OLD, H_0P5, 0))

        # Negative control 2: alter UnpackHalf2x16 lane extraction (index 0 -> 1) so hi half 9.0 is read instead of 4.0.
        bad_extract, n2 = re.subn(
            r"(OpExtInst %v2float %glsl UnpackHalf2x16 %\w+\n\s*%\w+ = OpCompositeExtract %float %\w+ )0\b",
            r"\g<1>1",
            spv,
            count=1,
        )
        self.assertEqual(n2, 1)
        got_bad_extract = si.run_fragment(bad_extract)["%out_mrt0"][0]
        self.assertNotEqual(got_bad_extract, want)
        self.assertEqual(got_bad_extract, merge_half(OLD, H_3P0, 0))

        # Negative control 3: corrupt destination high-half preservation mask (0xffff0000 -> 0x00000000).
        bad_mask, n3 = re.subn(
            r"(%\w+ = OpConstant %uint )(?:0x[fF]{4}0000|4294901760)\b",
            r"\g<1>0",
            spv,
            count=1,
        )
        self.assertEqual(n3, 1)
        got_bad_mask = si.run_fragment(bad_mask)["%out_mrt0"][0]
        self.assertNotEqual(got_bad_mask, want)
        self.assertEqual(got_bad_mask, H_2P0)


if __name__ == "__main__":
    unittest.main()
