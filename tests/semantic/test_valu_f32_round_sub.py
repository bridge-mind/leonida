"""Value and structural tests for f32 rounding and subtraction VALU instructions:

  v_floor_f32    D = floor(S0)
  v_ceil_f32     D = ceil(S0)
  v_trunc_f32    D = trunc(S0)
  v_sub_f32      D = S0 - S1
  v_subrev_f32   D = S1 - S0

Each case translates a hand-written pixel-shader listing with tools/rdna2spirv.py
and executes the resulting SPIR-V via tests/semantic/spirv_mini_interp.py.
Results are compared as raw 32-bit IEEE-754 binary32 words so +0.0 (0x00000000)
and -0.0 (0x80000000) are strictly distinguished. Standard library only.
"""
import math
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, translate_lines  # noqa: E402  (support puts tools/ on sys.path)

POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
POS_INF = 0x7F800000
NEG_INF = 0xFF800000


def f32_to_bits(x):
    """Pack a Python float into its IEEE-754 binary32 32-bit unsigned word."""
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return POS_INF if x > 0 else NEG_INF


def bits_to_f32(bits):
    """Unpack a 32-bit unsigned word as an IEEE-754 binary32 Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def ref_round_op(fn, bits):
    """Reference for Floor/Ceil/Trunc matching spirv_mini_interp.py (IEEE 754 roundToIntegral sign)."""
    v = bits_to_f32(bits)
    if not math.isfinite(v):
        return bits & 0xFFFFFFFF
    return f32_to_bits(math.copysign(float(fn(v)), v))


def ref_sub_f32(a_bits, b_bits):
    """Reference for v_sub_f32 (S0 - S1) in binary32."""
    return f32_to_bits(bits_to_f32(a_bits) - bits_to_f32(b_bits))


def ref_subrev_f32(a_bits, b_bits):
    """Reference for v_subrev_f32 (S1 - S0) in binary32."""
    return f32_to_bits(bits_to_f32(b_bits) - bits_to_f32(a_bits))


def run_ps(lines, name="ps_valu_f32"):
    """Translate `lines` as a pixel shader, run it, and return the single uniform mrt0 word."""
    spv = translate_lines(name, lines)
    words = mrt0_words(spv)
    if len(set(words)) != 1:
        raise AssertionError(f"{name}: exported mrt0 words differ: {[hex(w) for w in words]}")
    return words[0], spv


def run_unary(op, src_bits, *, src_expr="v1"):
    """Load src_bits into v1, execute `op v0, <src_expr>`, and return (mrt0_word, spv)."""
    lines = [
        f"v_mov_b32 v1, {src_bits & 0xFFFFFFFF:#010x}",
        f"{op} v0, {src_expr}",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    return run_ps(lines, name=f"ps_{op.split()[0]}")


def run_binary(op, a_bits, b_bits, *, setup=None, operands="v1, v2"):
    """Execute `op v0, <operands>` with default v1/v2 setup and return (mrt0_word, spv)."""
    if setup is None:
        setup = [
            f"v_mov_b32 v1, {a_bits & 0xFFFFFFFF:#010x}",
            f"v_mov_b32 v2, {b_bits & 0xFFFFFFFF:#010x}",
        ]
    lines = list(setup) + [
        f"{op} v0, {operands}",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    return run_ps(lines, name=f"ps_{op.split()[0]}")


ROUND_OPS = {
    "v_floor_f32": math.floor,
    "v_ceil_f32": math.ceil,
    "v_trunc_f32": math.trunc,
}

# Hand-verified (input_float, expected_floor, expected_ceil, expected_trunc)
ROUND_HAND_CONSTANTS = [
    (-1.75, -2.0, -1.0, -1.0),
    (-1.5, -2.0, -1.0, -1.0),
    (-1.25, -2.0, -1.0, -1.0),
    (-0.5, -1.0, -0.0, -0.0),
    (-0.0, -0.0, -0.0, -0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.5, 0.0, 1.0, 0.0),
    (1.25, 1.0, 2.0, 1.0),
    (1.5, 1.0, 2.0, 1.0),
    (1.75, 1.0, 2.0, 1.0),
    (-2.0, -2.0, -2.0, -2.0),
    (-1.0, -1.0, -1.0, -1.0),
    (1.0, 1.0, 1.0, 1.0),
    (2.0, 2.0, 2.0, 2.0),
    (float(2**22) + 0.5, float(2**22), float(2**22 + 1), float(2**22)),
    (float(2**23), float(2**23), float(2**23), float(2**23)),
    (float(-(2**23)), float(-(2**23)), float(-(2**23)), float(-(2**23))),
    (float(2**24), float(2**24), float(2**24), float(2**24)),
    (float(-(2**24)), float(-(2**24)), float(-(2**24)), float(-(2**24))),
]


class ReferenceConstantsTests(unittest.TestCase):
    def test_rounding_hand_constants(self):
        for x, exp_fl, exp_ce, exp_tr in ROUND_HAND_CONSTANTS:
            xb = f32_to_bits(x)
            with self.subTest(x=x, xb=hex(xb)):
                self.assertEqual(ref_round_op(math.floor, xb), f32_to_bits(exp_fl))
                self.assertEqual(ref_round_op(math.ceil, xb), f32_to_bits(exp_ce))
                self.assertEqual(ref_round_op(math.trunc, xb), f32_to_bits(exp_tr))

    def test_sub_hand_constants(self):
        pairs = [
            (5.0, 2.0, 3.0, -3.0),
            (1.5, 4.25, -2.75, 2.75),
            (0.0, 0.0, 0.0, 0.0),
            (-0.0, 0.0, -0.0, 0.0),
            (0.0, -0.0, 0.0, -0.0),
            (-0.0, -0.0, 0.0, 0.0),
        ]
        for a, b, exp_sub, exp_subrev in pairs:
            ab, bb = f32_to_bits(a), f32_to_bits(b)
            with self.subTest(a=a, b=b):
                self.assertEqual(ref_sub_f32(ab, bb), f32_to_bits(exp_sub))
                self.assertEqual(ref_subrev_f32(ab, bb), f32_to_bits(exp_subrev))


class RoundingOpsTests(unittest.TestCase):
    def test_fractional_integer_and_large_values(self):
        for op_base, py_fn in ROUND_OPS.items():
            for suffix in ("", "_e32", "_e64"):
                op = op_base + suffix
                for x, *_ in ROUND_HAND_CONSTANTS:
                    xb = f32_to_bits(x)
                    expected = ref_round_op(py_fn, xb)
                    with self.subTest(op=op, x=x, xb=hex(xb)):
                        got, _ = run_unary(op, xb)
                        self.assertEqual(
                            got, expected,
                            f"{op}({x} / {xb:#010x}): got {got:#010x}, expected {expected:#010x}",
                        )

    def test_infinities(self):
        for op_base in ROUND_OPS:
            for inf_bits in (POS_INF, NEG_INF):
                with self.subTest(op=op_base, inf=hex(inf_bits)):
                    got, _ = run_unary(op_base, inf_bits)
                    self.assertEqual(got, inf_bits)

    def test_vop3_modifiers(self):
        # neg (-v1), abs (|v1|), neg-abs (-|v1|), clamp, mul:2, div:2, and omod-before-clamp
        cases = [
            ("v_floor_f32_e64", -1.25, "-v1", 1.0),       # floor(-(-1.25)) = floor(1.25) = 1.0
            ("v_floor_f32_e64", -1.25, "|v1|", 1.0),      # floor(|-1.25|) = 1.0
            ("v_floor_f32_e64", 1.25, "-|v1|", -2.0),     # floor(-|1.25|) = floor(-1.25) = -2.0
            ("v_ceil_f32_e64", 1.25, "-v1", -1.0),        # ceil(-1.25) = -1.0
            ("v_ceil_f32_e64", -1.25, "|v1|", 2.0),       # ceil(1.25) = 2.0
            ("v_trunc_f32_e64", -0.5, "|v1|", 0.0),       # trunc(0.5) = +0.0 (bit-exact)
            ("v_trunc_f32_e64", 0.5, "-v1", -0.0),        # trunc(-0.5) = -0.0 (bit-exact)
            ("v_floor_f32_e64", -0.5, "v1 clamp", 0.0),   # clamp(-1.0) = +0.0
            ("v_ceil_f32_e64", 1.25, "v1 clamp", 1.0),    # clamp(2.0) = 1.0
            ("v_trunc_f32_e64", 0.5, "v1 clamp", 0.0),    # clamp(+0.0) = +0.0
            ("v_floor_f32_e64", 1.75, "v1 mul:2", 2.0),   # floor(1.75) * 2 = 2.0
            ("v_ceil_f32_e64", 1.25, "v1 div:2", 1.0),    # ceil(1.25) / 2 = 1.0
            ("v_trunc_f32_e64", 1.75, "v1 mul:2 clamp", 1.0),  # (1.0 * 2) clamped = 1.0
        ]
        for op, x, expr, exp in cases:
            xb = f32_to_bits(x)
            expected = f32_to_bits(exp)
            with self.subTest(op=op, x=x, expr=expr):
                got, _ = run_unary(op, xb, src_expr=expr)
                self.assertEqual(
                    got, expected,
                    f"{op} v0, {expr} (v1={x}): got {got:#010x}, expected {expected:#010x}",
                )


class SubAndSubrevTests(unittest.TestCase):
    def test_non_commutativity_and_forms(self):
        pairs = [
            (5.0, 2.0),
            (1.5, 4.25),
            (-3.5, 1.25),
            (-7.0, -2.5),
            (0.125, -0.375),
        ]
        for a, b in pairs:
            ab, bb = f32_to_bits(a), f32_to_bits(b)
            for suffix in ("", "_e32", "_e64"):
                with self.subTest(a=a, b=b, suffix=suffix):
                    got_sub, _ = run_binary("v_sub_f32" + suffix, ab, bb)
                    got_rev, _ = run_binary("v_subrev_f32" + suffix, ab, bb)
                    self.assertEqual(got_sub, ref_sub_f32(ab, bb))
                    self.assertEqual(got_rev, ref_subrev_f32(ab, bb))
                    self.assertNotEqual(got_sub, got_rev)

    def test_signed_zeros(self):
        zero_cases = [
            (POS_ZERO, POS_ZERO, POS_ZERO, POS_ZERO),
            (NEG_ZERO, POS_ZERO, NEG_ZERO, POS_ZERO),
            (POS_ZERO, NEG_ZERO, POS_ZERO, NEG_ZERO),
            (NEG_ZERO, NEG_ZERO, POS_ZERO, POS_ZERO),
            (f32_to_bits(1.0), f32_to_bits(1.0), POS_ZERO, POS_ZERO),
            (f32_to_bits(-1.0), f32_to_bits(-1.0), POS_ZERO, POS_ZERO),
        ]
        for ab, bb, exp_sub, exp_rev in zero_cases:
            with self.subTest(a=hex(ab), b=hex(bb)):
                got_sub, _ = run_binary("v_sub_f32", ab, bb)
                got_rev, _ = run_binary("v_subrev_f32", ab, bb)
                self.assertEqual(got_sub, exp_sub)
                self.assertEqual(got_rev, exp_rev)

    def test_infinities(self):
        one = f32_to_bits(1.0)
        inf_cases = [
            (POS_INF, one, POS_INF, NEG_INF),
            (NEG_INF, one, NEG_INF, POS_INF),
            (POS_INF, NEG_INF, POS_INF, NEG_INF),
        ]
        for ab, bb, exp_sub, exp_rev in inf_cases:
            with self.subTest(a=hex(ab), b=hex(bb)):
                got_sub, _ = run_binary("v_sub_f32", ab, bb)
                got_rev, _ = run_binary("v_subrev_f32", ab, bb)
                self.assertEqual(got_sub, exp_sub)
                self.assertEqual(got_rev, exp_rev)

        # inf - inf = NaN
        for op in ("v_sub_f32", "v_subrev_f32"):
            with self.subTest(op=op):
                got, _ = run_binary(op, POS_INF, POS_INF)
                self.assertTrue(math.isnan(bits_to_f32(got)), f"expected NaN, got {got:#010x}")

    def test_operand_kinds(self):
        # SGPR, literal, and inline-float operands in _e32 and _e64 forms
        cases = [
            ("v_sub_f32_e32", ["s_mov_b32 s0, 0x40a00000", "v_mov_b32 v2, 0x40000000"], "s0, v2", 3.0),
            ("v_subrev_f32_e32", ["s_mov_b32 s0, 0x40a00000", "v_mov_b32 v2, 0x40000000"], "s0, v2", -3.0),
            ("v_sub_f32_e32", ["v_mov_b32 v2, 0x3fc00000"], "0x40400000, v2", 1.5),
            ("v_subrev_f32_e32", ["v_mov_b32 v2, 0x3fc00000"], "0x40400000, v2", -1.5),
            ("v_sub_f32_e32", ["v_mov_b32 v2, 0x3e800000"], "0.5, v2", 0.25),
            ("v_subrev_f32_e32", ["v_mov_b32 v2, 0x3e800000"], "0.5, v2", -0.25),
            ("v_sub_f32_e64", ["v_mov_b32 v1, 0x40800000", "s_mov_b32 s0, 0x3fc00000"], "v1, s0", 2.5),
            ("v_subrev_f32_e64", ["v_mov_b32 v1, 0x40800000", "s_mov_b32 s0, 0x3fc00000"], "v1, s0", -2.5),
            ("v_sub_f32_e64", ["v_mov_b32 v1, 0x3f000000"], "v1, -1.0", 1.5),
            ("v_subrev_f32_e64", ["v_mov_b32 v1, 0x3f000000"], "v1, -1.0", -1.5),
        ]
        for op, setup, operands, exp in cases:
            with self.subTest(op=op, operands=operands):
                got, _ = run_binary(op, 0, 0, setup=setup, operands=operands)
                self.assertEqual(got, f32_to_bits(exp))

    def test_vop3_modifiers(self):
        # v1 = 1.5, v2 = -2.25
        a, b = f32_to_bits(1.5), f32_to_bits(-2.25)
        cases = [
            # neg / abs on sources
            ("v_sub_f32_e64", "-v1, v2", -1.5 - (-2.25)),          # 0.75
            ("v_sub_f32_e64", "v1, -v2", 1.5 - 2.25),             # -0.75
            ("v_sub_f32_e64", "|v1|, |v2|", 1.5 - 2.25),          # -0.75
            ("v_sub_f32_e64", "-|v1|, -|v2|", -1.5 - (-2.25)),    # 0.75
            ("v_subrev_f32_e64", "-v1, v2", -2.25 - (-1.5)),      # -0.75
            ("v_subrev_f32_e64", "v1, |v2|", 2.25 - 1.5),         # 0.75
            # clamp, mul:2, div:2
            ("v_sub_f32_e64", "v1, v2 clamp", 1.0),               # 3.75 -> 1.0
            ("v_subrev_f32_e64", "v1, v2 clamp", 0.0),            # -3.75 -> 0.0
            ("v_sub_f32_e64", "v1, v2 mul:2", 7.5),               # 3.75 * 2
            ("v_sub_f32_e64", "v1, v2 div:2", 1.875),             # 3.75 / 2
            ("v_subrev_f32_e64", "v1, v2 mul:2", -7.5),           # -3.75 * 2
            ("v_subrev_f32_e64", "v1, v2 div:2", -1.875),         # -3.75 / 2
        ]
        for op, operands, exp in cases:
            with self.subTest(op=op, operands=operands):
                got, _ = run_binary(op, a, b, operands=operands)
                self.assertEqual(got, f32_to_bits(exp))

        # Verify omod applies before clamp: (1.5 - 0.0) = 1.5 -> div:2 = 0.75 -> clamp = 0.75
        # (if clamp ran before div:2, result would be 0.5).
        got, _ = run_binary("v_sub_f32_e64", f32_to_bits(1.5), POS_ZERO, operands="v1, v2 div:2 clamp")
        self.assertEqual(got, f32_to_bits(0.75))


class StructuralEmissionTests(unittest.TestCase):
    def test_subrev_swaps_opfsub_operands(self):
        a, b = f32_to_bits(7.0), f32_to_bits(2.0)
        _, spv_sub = run_binary("v_sub_f32", a, b)
        _, spv_rev = run_binary("v_subrev_f32", a, b)

        m_sub = re.search(r"%\w+\s*=\s*OpFSub\s+%float\s+(%\w+)\s+(%\w+)", spv_sub)
        m_rev = re.search(r"%\w+\s*=\s*OpFSub\s+%float\s+(%\w+)\s+(%\w+)", spv_rev)
        self.assertIsNotNone(m_sub, "expected OpFSub in v_sub_f32 SPIR-V output")
        self.assertIsNotNone(m_rev, "expected OpFSub in v_subrev_f32 SPIR-V output")

        # Trace each OpFSub operand back to its bitcasted constant word (%c_<hex> or similar).
        def operand_const(spv, ssa):
            m = re.search(rf"{re.escape(ssa)}\s*=\s*OpBitcast\s+%float\s+(%\w+)", spv)
            self.assertIsNotNone(m, f"expected OpBitcast for {ssa}")
            return m.group(1)

        sub_lhs = operand_const(spv_sub, m_sub.group(1))
        sub_rhs = operand_const(spv_sub, m_sub.group(2))
        rev_lhs = operand_const(spv_rev, m_rev.group(1))
        rev_rhs = operand_const(spv_rev, m_rev.group(2))
        self.assertNotEqual(sub_lhs, sub_rhs)
        self.assertEqual((rev_lhs, rev_rhs), (sub_rhs, sub_lhs))

    def test_rounding_ops_emit_glsl_extinst(self):
        expected_inst = {
            "v_floor_f32": "Floor",
            "v_ceil_f32": "Ceil",
            "v_trunc_f32": "Trunc",
        }
        for op, glsl_name in expected_inst.items():
            with self.subTest(op=op):
                _, spv = run_unary(op, f32_to_bits(1.5))
                self.assertRegex(spv, rf"%\w+\s*=\s*OpExtInst\s+%float\s+%glsl\s+{glsl_name}\s+%\w+")


if __name__ == "__main__":
    unittest.main()
