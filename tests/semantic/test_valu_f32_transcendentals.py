"""Translator-to-interpreter semantic tests for float32 transcendental VALU instructions:

  v_exp_f32   D = 2^S0
  v_log_f32   D = log2(S0)
  v_sqrt_f32  D = sqrt(S0)
  v_rsq_f32   D = 1 / sqrt(S0)

Every test translates synthetic hand-written RDNA2 disassembly via tools/rdna2spirv.py and executes the emitted
SPIR-V in tests/semantic/spirv_mini_interp.py. Undefined domains (log2/rsq of x <= 0, sqrt of x < 0) are verified
to raise UnsupportedConstruct in spirv_mini_interp.py. Standard library only; no game data.
"""
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
from support import mrt0_words, translate_lines  # noqa: E402

POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
POS_INF = 0x7F800000
NEG_INF = 0xFF800000
QNAN = 0x7FC00000


def f32_to_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_to_f32(w):
    return struct.unpack("<f", struct.pack("<I", w & 0xFFFFFFFF))[0]


def ref_f32(x):
    return f32_to_bits(bits_to_f32(f32_to_bits(x)))


def ref_exp2(w):
    x = bits_to_f32(w)
    if math.isnan(x):
        return w
    if math.isinf(x):
        return POS_INF if x > 0 else POS_ZERO
    return f32_to_bits(math.pow(2.0, x))


def ref_log2(w):
    x = bits_to_f32(w)
    if math.isnan(x):
        return w
    if math.isinf(x):
        return POS_INF
    return f32_to_bits(math.log2(x))


def ref_sqrt(w):
    x = bits_to_f32(w)
    if math.isnan(x):
        return w
    if x == 0.0:
        return w
    if math.isinf(x):
        return POS_INF
    return f32_to_bits(math.sqrt(x))


def ref_rsq(w):
    x = bits_to_f32(w)
    if math.isnan(x):
        return w
    if math.isinf(x):
        return POS_ZERO
    return f32_to_bits(1.0 / math.sqrt(x))


def run_ps(name, body_lines):
    lines = list(body_lines) + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    spv = translate_lines(name, lines)
    words = mrt0_words(spv)
    return words[0], spv


class ExactPowersAndSquaresTests(unittest.TestCase):
    def test_v_exp_f32_powers_of_two(self):
        for exp_val in (-126.0, -10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0, 127.0):
            w = f32_to_bits(exp_val)
            with self.subTest(exp_val=exp_val):
                got, _ = run_ps("ps_exp_pow2", [f"v_mov_b32 v1, {w:#x}", "v_exp_f32 v0, v1"])
                self.assertEqual(got, ref_exp2(w))

    def test_v_log_f32_powers_of_two(self):
        for k in (-126, -4, -2, -1, 0, 1, 2, 10, 127):
            w = f32_to_bits(math.ldexp(1.0, k))
            with self.subTest(k=k):
                got, _ = run_ps("ps_log_pow2", [f"v_mov_b32 v1, {w:#x}", "v_log_f32 v0, v1"])
                self.assertEqual(got, f32_to_bits(float(k)))

    def test_v_sqrt_f32_perfect_squares(self):
        for root in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 25.0, 256.0):
            w = f32_to_bits(root * root)
            with self.subTest(root=root):
                got, _ = run_ps("ps_sqrt_sq", [f"v_mov_b32 v1, {w:#x}", "v_sqrt_f32 v0, v1"])
                self.assertEqual(got, f32_to_bits(root))

    def test_v_rsq_f32_reciprocal_squares(self):
        cases = [(0.0625, 4.0), (0.25, 2.0), (1.0, 1.0), (4.0, 0.5), (16.0, 0.25), (64.0, 0.125)]
        for x, expected in cases:
            w = f32_to_bits(x)
            with self.subTest(x=x):
                got, _ = run_ps("ps_rsq_sq", [f"v_mov_b32 v1, {w:#x}", "v_rsq_f32 v0, v1"])
                self.assertEqual(got, f32_to_bits(expected))


class NonPowerFiniteTests(unittest.TestCase):
    def test_non_power_of_two_values(self):
        cases = [
            ("v_exp_f32", (0.5, -0.5, 1.5, 3.25), ref_exp2),
            ("v_log_f32", (3.0, 5.5, 10.0, 0.3), ref_log2),
            ("v_sqrt_f32", (2.0, 3.0, 5.0, 0.75), ref_sqrt),
            ("v_rsq_f32", (2.0, 3.0, 5.0, 0.75), ref_rsq),
        ]
        for op, inputs, ref_fn in cases:
            for val in inputs:
                w = f32_to_bits(val)
                with self.subTest(op=op, val=val):
                    got, _ = run_ps("ps_trans_finite", [f"v_mov_b32 v1, {w:#x}", f"{op} v0, v1"])
                    self.assertEqual(got, ref_fn(w))


class SpecialValuesAndDomainsTests(unittest.TestCase):
    def test_special_valid_values(self):
        cases = [
            ("v_sqrt_f32", POS_ZERO, POS_ZERO),
            ("v_sqrt_f32", NEG_ZERO, NEG_ZERO),
            ("v_exp_f32", POS_INF, POS_INF),
            ("v_exp_f32", NEG_INF, POS_ZERO),
            ("v_log_f32", POS_INF, POS_INF),
            ("v_sqrt_f32", POS_INF, POS_INF),
            ("v_rsq_f32", POS_INF, POS_ZERO),
        ]
        for op, src_bits, expected_bits in cases:
            with self.subTest(op=op, src=hex(src_bits)):
                got, _ = run_ps("ps_trans_spec", [f"v_mov_b32 v1, {src_bits:#x}", f"{op} v0, v1"])
                self.assertEqual(got, expected_bits)

    def test_nan_propagation(self):
        for op in ("v_exp_f32", "v_log_f32", "v_sqrt_f32", "v_rsq_f32"):
            with self.subTest(op=op):
                got, _ = run_ps("ps_trans_nan", [f"v_mov_b32 v1, {QNAN:#x}", f"{op} v0, v1"])
                self.assertTrue(math.isnan(bits_to_f32(got)), f"{op} did not propagate NaN: {got:#x}")

    def test_undefined_domains_raise_in_interpreter(self):
        cases = [
            ("v_log_f32", POS_ZERO),
            ("v_log_f32", f32_to_bits(-1.0)),
            ("v_sqrt_f32", f32_to_bits(-1.0)),
            ("v_rsq_f32", POS_ZERO),
            ("v_rsq_f32", f32_to_bits(-1.0)),
        ]
        for op, src_bits in cases:
            with self.subTest(op=op, src=hex(src_bits)):
                with self.assertRaises(si.UnsupportedConstruct):
                    run_ps("ps_trans_undef", [f"v_mov_b32 v1, {src_bits:#x}", f"{op} v0, v1"])


class ModifiersAndOperandFormsTests(unittest.TestCase):
    def test_vop3_src_and_output_modifiers(self):
        w4 = f32_to_bits(4.0)
        wn4 = f32_to_bits(-4.0)
        w2 = f32_to_bits(2.0)

        got_abs, _ = run_ps("ps_sqrt_abs", [f"v_mov_b32 v1, {wn4:#x}", "v_sqrt_f32_e64 v0, |v1|"])
        self.assertEqual(got_abs, w2)

        got_neg, _ = run_ps("ps_exp_neg", [f"v_mov_b32 v1, {w2:#x}", "v_exp_f32_e64 v0, -v1"])
        self.assertEqual(got_neg, f32_to_bits(0.25))

        got_clamp, _ = run_ps("ps_exp_clamp", [f"v_mov_b32 v1, {w2:#x}", "v_exp_f32_e64 v0, v1 clamp"])
        self.assertEqual(got_clamp, f32_to_bits(1.0))

        got_mul2, _ = run_ps("ps_sqrt_mul2", [f"v_mov_b32 v1, {w4:#x}", "v_sqrt_f32_e64 v0, v1 mul:2"])
        self.assertEqual(got_mul2, w4)

        got_div2, _ = run_ps("ps_rsq_div2", [f"v_mov_b32 v1, {w4:#x}", "v_rsq_f32_e64 v0, v1 div:2"])
        self.assertEqual(got_div2, f32_to_bits(0.25))

    def test_e32_e64_sgpr_inline_and_literal_operands(self):
        w4 = f32_to_bits(4.0)
        w16 = f32_to_bits(16.0)
        for op, inline_s, lit_w, exp_inline, exp_lit in (
            ("v_exp_f32", "2.0", f32_to_bits(3.0), 4.0, 8.0),
            ("v_log_f32", "4.0", w16, 2.0, 4.0),
            ("v_sqrt_f32", "4.0", f32_to_bits(9.0), 2.0, 3.0),
            ("v_rsq_f32", "4.0", w16, 0.5, 0.25),
        ):
            for suffix in ("", "_e32", "_e64"):
                with self.subTest(op=op + suffix, form="sgpr"):
                    got, _ = run_ps("ps_trans_sgpr", [f"s_mov_b32 s0, {w4:#x}", f"{op}{suffix} v0, s0"])
                    self.assertEqual(got, {
                        "v_exp_f32": f32_to_bits(16.0),
                        "v_log_f32": f32_to_bits(2.0),
                        "v_sqrt_f32": f32_to_bits(2.0),
                        "v_rsq_f32": f32_to_bits(0.5),
                    }[op])
                with self.subTest(op=op + suffix, form="inline"):
                    got, _ = run_ps("ps_trans_inline", [f"{op}{suffix} v0, {inline_s}"])
                    self.assertEqual(got, f32_to_bits(exp_inline))
                with self.subTest(op=op + suffix, form="literal"):
                    got, _ = run_ps("ps_trans_lit", [f"{op}{suffix} v0, {lit_w:#x}"])
                    self.assertEqual(got, f32_to_bits(exp_lit))


class SpirvStructureTests(unittest.TestCase):
    def test_emitted_glsl_ext_insts(self):
        for op, glsl_name in (
            ("v_exp_f32", "Exp2"),
            ("v_log_f32", "Log2"),
            ("v_sqrt_f32", "Sqrt"),
            ("v_rsq_f32", "InverseSqrt"),
        ):
            with self.subTest(op=op):
                _, spv = run_ps("ps_trans_struct", ["v_mov_b32 v1, 1.0", f"{op} v0, v1"])
                self.assertIn(f"= OpExtInst %float %glsl {glsl_name}", spv)


if __name__ == "__main__":
    unittest.main()
