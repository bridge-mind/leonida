"""Value tests for f16 add/sub/min/max VALU instructions:

  v_add_f16   D.f16 = S0.f16 + S1.f16
  v_sub_f16   D.f16 = S0.f16 - S1.f16
  v_min_f16   D.f16 = min(S0.f16, S1.f16)   (-0.0 < +0.0, NaN -> other)
  v_max_f16   D.f16 = max(S0.f16, S1.f16)   (-0.0 < +0.0, NaN -> other)

Each case translates a hand-written pixel-shader listing with tools/rdna2spirv.py
and executes the resulting SPIR-V via tests/semantic/spirv_mini_interp.py.
GFX10 partial-register preservation (the untouched 16-bit half of vdst keeps its
prior sentinel bits) and VOP3 op_sel/neg/abs/clamp modifiers are verified directly.
Standard library only.
"""
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, translate_lines  # noqa: E402  (support puts tools/ on sys.path)

MASK32 = 0xFFFFFFFF
POS_ZERO = 0x0000
NEG_ZERO = 0x8000
POS_INF = 0x7C00
NEG_INF = 0xFC00
QNAN = 0x7E00


def f16_to_bits(x):
    """Pack a Python float into its IEEE-754 binary16 16-bit unsigned word."""
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return POS_INF if x > 0 else NEG_INF


def bits_to_f16(h):
    """Unpack a 16-bit unsigned word as an IEEE-754 binary16 Python float."""
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def run_f16(op, a_word, b_word, *, v0_init=0xBEEFDEAD, setup=None, operands="v1, v2"):
    """Translate and run `op v0, <operands>` with v0 preloaded; return the full 32-bit v0 word."""
    if setup is None:
        setup = [
            f"v_mov_b32 v1, {a_word & MASK32:#010x}",
            f"v_mov_b32 v2, {b_word & MASK32:#010x}",
        ]
    lines = [f"v_mov_b32 v0, {v0_init & MASK32:#010x}"] + list(setup) + [
        f"{op} v0, {operands}",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_{op.split()[0]}", lines)
    words = mrt0_words(spv)
    if len(set(words)) != 1:
        raise AssertionError(f"{op}: exported mrt0 words differ: {[hex(w) for w in words]}")
    return words[0]


class AddSubF16Tests(unittest.TestCase):
    def test_exact_cancellation_and_overflow(self):
        cases = [
            ("v_add_f16", 1.5, 2.25, 3.75),
            ("v_add_f16", -2.5, 0.75, -1.75),
            ("v_add_f16", 1.5, -1.5, 0.0),         # cancellation -> +0.0
            ("v_add_f16", 65504.0, 32.0, math.inf),
            ("v_add_f16", -65504.0, -32.0, -math.inf),
            ("v_sub_f16", 5.0, 1.25, 3.75),
            ("v_sub_f16", 1.25, 5.0, -3.75),       # non-commutative
            ("v_sub_f16", 1.5, 1.5, 0.0),          # cancellation -> +0.0
            ("v_sub_f16", -65504.0, 32.0, -math.inf),
        ]
        for op_base, a, b, exp in cases:
            ab, bb, eb = f16_to_bits(a), f16_to_bits(b), f16_to_bits(exp)
            for suffix in ("", "_e32", "_e64"):
                op = op_base + suffix
                with self.subTest(op=op, a=a, b=b):
                    got = run_f16(op, ab, bb)
                    self.assertEqual(got & 0xFFFF, eb)
                    self.assertEqual(got >> 16, 0xBEEF)  # upper half preserved

    def test_signed_zeros(self):
        cases = [
            ("v_add_f16", POS_ZERO, NEG_ZERO, POS_ZERO),
            ("v_add_f16", NEG_ZERO, NEG_ZERO, NEG_ZERO),
            ("v_sub_f16", NEG_ZERO, POS_ZERO, NEG_ZERO),
            ("v_sub_f16", POS_ZERO, NEG_ZERO, POS_ZERO),
            ("v_sub_f16", NEG_ZERO, NEG_ZERO, POS_ZERO),
        ]
        for op, ab, bb, eb in cases:
            with self.subTest(op=op, a=hex(ab), b=hex(bb)):
                got = run_f16(op, ab, bb)
                self.assertEqual(got & 0xFFFF, eb)


class MinMaxF16Tests(unittest.TestCase):
    def test_signed_zero_ordering_and_nan(self):
        # Hardware NMin/NMax orders -0.0 < +0.0 and returns the non-NaN operand.
        self.assertEqual(run_f16("v_min_f16", NEG_ZERO, POS_ZERO) & 0xFFFF, NEG_ZERO)
        self.assertEqual(run_f16("v_min_f16", POS_ZERO, NEG_ZERO) & 0xFFFF, NEG_ZERO)
        self.assertEqual(run_f16("v_max_f16", NEG_ZERO, POS_ZERO) & 0xFFFF, POS_ZERO)
        self.assertEqual(run_f16("v_max_f16", POS_ZERO, NEG_ZERO) & 0xFFFF, POS_ZERO)

        one = f16_to_bits(1.0)
        self.assertEqual(run_f16("v_min_f16", QNAN, one) & 0xFFFF, one)
        self.assertEqual(run_f16("v_min_f16", one, QNAN) & 0xFFFF, one)
        self.assertEqual(run_f16("v_max_f16", QNAN, one) & 0xFFFF, one)
        self.assertEqual(run_f16("v_max_f16", one, QNAN) & 0xFFFF, one)

    def test_boundary_pairs_and_forms(self):
        vals = (NEG_INF, f16_to_bits(-65504.0), f16_to_bits(-1.0), NEG_ZERO, POS_ZERO, f16_to_bits(1.0),
                f16_to_bits(65504.0), POS_INF)
        for suffix in ("", "_e32", "_e64"):
            for i, ab in enumerate(vals):
                for bb in vals[i:]:
                    with self.subTest(suffix=suffix, a=hex(ab), b=hex(bb)):
                        self.assertEqual(run_f16("v_min_f16" + suffix, ab, bb) & 0xFFFF, ab)
                        self.assertEqual(run_f16("v_max_f16" + suffix, ab, bb) & 0xFFFF, bb)


class Gfx10PartialRegOpselAndModifiersTests(unittest.TestCase):
    def test_partial_register_preservation_and_opsel(self):
        # Pack distinct f16 values into lo/hi halves of v1 and v2
        v1_packed = (f16_to_bits(10.0) << 16) | f16_to_bits(1.5)
        v2_packed = (f16_to_bits(20.0) << 16) | f16_to_bits(2.5)

        # dst=lo (0): preserves 0xBEEF in bits 31:16
        got = run_f16("v_add_f16_e64", v1_packed, v2_packed, operands="v1, v2 op_sel:[1,0,0]")
        self.assertEqual(got, (0xBEEF << 16) | f16_to_bits(12.5))

        got = run_f16("v_sub_f16_e64", v1_packed, v2_packed, operands="v1, v2 op_sel:[0,1,0]")
        self.assertEqual(got, (0xBEEF << 16) | f16_to_bits(-18.5))

        # dst=hi (1): preserves 0xDEAD in bits 15:0
        got = run_f16("v_add_f16_e64", v1_packed, v2_packed, operands="v1, v2 op_sel:[1,1,1]")
        self.assertEqual(got, (f16_to_bits(30.0) << 16) | 0xDEAD)

        got = run_f16("v_min_f16_e64", v1_packed, v2_packed, operands="v1, v2 op_sel:[1,0,1]")
        self.assertEqual(got, (f16_to_bits(2.5) << 16) | 0xDEAD)

        got = run_f16("v_max_f16_e64", v1_packed, v2_packed, operands="v1, v2 op_sel:[0,1,1]")
        self.assertEqual(got, (f16_to_bits(20.0) << 16) | 0xDEAD)

    def test_vop3_modifiers_and_operand_kinds(self):
        a, b = f16_to_bits(-1.5), f16_to_bits(0.75)
        cases = [
            ("v_add_f16_e64", "-v1, v2", 2.25),
            ("v_add_f16_e64", "|v1|, v2", 2.25),
            ("v_sub_f16_e64", "v1, -v2", -0.75),
            ("v_add_f16_e64", "|v1|, v2 clamp", 1.0),
            ("v_sub_f16_e64", "v1, v2 clamp", 0.0),
            ("v_min_f16_e64", "-v1, v2", 0.75),
            ("v_max_f16_e64", "v1, -v2 clamp", 0.0),
        ]
        for op, operands, exp in cases:
            with self.subTest(op=op, operands=operands):
                got = run_f16(op, a, b, operands=operands)
                self.assertEqual(got & 0xFFFF, f16_to_bits(exp))
                self.assertEqual(got >> 16, 0xBEEF)

        # SGPR, literal, and inline-float operands
        sgpr_got = run_f16("v_add_f16_e32", 0, 0,
                           setup=[f"s_mov_b32 s0, {f16_to_bits(2.5):#010x}",
                                  f"v_mov_b32 v2, {f16_to_bits(1.25):#010x}"],
                           operands="s0, v2")
        self.assertEqual(sgpr_got & 0xFFFF, f16_to_bits(3.75))

        lit_got = run_f16("v_sub_f16_e32", 0, 0,
                          setup=[f"v_mov_b32 v2, {f16_to_bits(1.25):#010x}"],
                          operands=f"{f16_to_bits(4.5):#010x}, v2")
        self.assertEqual(lit_got & 0xFFFF, f16_to_bits(3.25))

        imm_got = run_f16("v_max_f16_e64", f16_to_bits(0.25), 0, operands="v1, 0.5")
        self.assertEqual(imm_got & 0xFFFF, f16_to_bits(0.5))

        imm_neg_got = run_f16("v_add_f16_e32", 0, f16_to_bits(2.5), operands="-1.0, v2")
        self.assertEqual(imm_neg_got & 0xFFFF, f16_to_bits(1.5))


if __name__ == "__main__":
    unittest.main()
