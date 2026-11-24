"""All 16 binary32 vector compares, v_cmp_*_f32 and v_cmpx_*_f32, including the negated ones, on NaN, infinity and -0.

RDNA2 ISA (VOPC), per lane, with unordered = either source is a NaN (quiet or signaling):
  F = 0, TRU = 1, O = ordered, U = unordered;
  LT, EQ, LE, GT, GE and LG (S0 < S1 or S0 > S1) are false when unordered, and -0 equals +0;
  NGE, NLG, NGT, NLE, NEQ, NLT are the logical negations of GE, LG, GT, LE, EQ, LT, so they are TRUE when unordered.
v_cmp writes the result to VCC (e32) or an SGPR pair (e64) and v_cmpx writes EXEC; v_cmpx is checked by the lanes it
leaves active. VOP3 abs and neg act on the sources before the compare (so -|NaN| is still a NaN).
Expected values come from the Python predicates below, not from the translator. Each case translates a one-lane pixel
shader with tools/rdna2spirv.py and runs it in tests/semantic/spirv_mini_interp.py.
"""
import itertools
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

PREDS = {
    "f": lambda a, b: False,
    "tru": lambda a, b: True,
    "o": lambda a, b: not (math.isnan(a) or math.isnan(b)),
    "u": lambda a, b: math.isnan(a) or math.isnan(b),
    "lt": lambda a, b: a < b,
    "eq": lambda a, b: a == b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "lg": lambda a, b: a < b or a > b,
    "nlt": lambda a, b: not a < b,
    "neq": lambda a, b: not a == b,
    "nle": lambda a, b: not a <= b,
    "ngt": lambda a, b: not a > b,
    "nge": lambda a, b: not a >= b,
    "nlg": lambda a, b: not (a < b or a > b),
}
# Bit patterns: -inf, -1, -0, +0, +1, 1+ulp, +inf, quiet NaN (both signs), signaling NaN.
VALUES = [0xFF800000, 0xBF800000, 0x80000000, 0x00000000, 0x3F800000, 0x3F800001, 0x7F800000,
          0x7FC00000, 0xFFC00000, 0x7F800001]


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def run(lines):
    spv = support.translate_lines("ps_vcmp_f32", lines + ["exp mrt0 v10, v11, v12, v13 done vm", "s_endpgm"])
    return si.run_fragment(spv)["%out_mrt0"]


def load(a, b):
    return [f"s_mov_b32 s2, {a:#x}", "v_mov_b32 v2, s2", f"s_mov_b32 s3, {b:#x}", "v_mov_b32 v3, s3",
            "v_mov_b32 v10, 0", "v_mov_b32 v11, 0", "v_mov_b32 v12, 0", "v_mov_b32 v13, 0", "v_mov_b32 v14, 1"]


class VcmpF32(unittest.TestCase):
    def test_vcc_and_sgpr_forms(self):
        for pred, fn in PREDS.items():
            for a, b in itertools.product(VALUES, VALUES):
                lines = load(a, b) + [f"v_cmp_{pred}_f32_e32 vcc_lo, v2, v3", "v_cndmask_b32_e64 v10, v10, v14, vcc_lo",
                                      f"v_cmp_{pred}_f32_e64 s20, v2, v3", "v_cndmask_b32_e64 v11, v11, v14, s20"]
                want = int(fn(f32(a), f32(b)))
                with self.subTest(pred=pred, a=hex(a), b=hex(b)):
                    self.assertEqual(list(run(lines)[:2]), [want, want])

    def test_cmpx_narrows_exec(self):
        for pred, fn in PREDS.items():
            for a, b in itertools.product(VALUES, VALUES):
                # EXEC is the single lane; v_cmpx clears it when the compare is false, so the write after it is skipped.
                # EXEC is restored afterwards so the shader can still export.
                lines = load(a, b) + ["s_mov_b32 s10, exec_lo", f"v_cmpx_{pred}_f32_e32 v2, v3", "v_mov_b32 v10, 7",
                                      "s_mov_b32 exec_lo, s10", f"v_cmpx_{pred}_f32_e64 v2, v3", "v_mov_b32 v11, 9",
                                      "s_mov_b32 exec_lo, s10"]
                want = int(fn(f32(a), f32(b)))
                with self.subTest(pred=pred, a=hex(a), b=hex(b)):
                    self.assertEqual(list(run(lines)[:2]), [7 * want, 9 * want])

    def test_abs_neg_modifiers(self):
        nan = 0x7FC00000
        cases = [
            ("v_cmp_nge_f32_e64 s20, -|v2|, v3", -abs(f32(0x3F800000)), f32(0x3F800000), "nge"),
            ("v_cmp_neq_f32_e64 s20, |v2|, v3", abs(f32(0xBF800000)), f32(0x3F800000), "neq"),
            ("v_cmp_nlt_f32_e64 s20, -v2, v3", -f32(0x00000000), f32(0x80000000), "nlt"),
            ("v_cmp_nge_f32_e64 s20, -|v2|, v3", -abs(f32(nan)), f32(0x3F800000), "nge"),
            ("v_cmp_neq_f32_e64 s20, v2, |v3|", f32(nan), abs(f32(nan)), "neq"),
        ]
        srcs = [(0x3F800000, 0x3F800000), (0xBF800000, 0x3F800000), (0x00000000, 0x80000000), (nan, 0x3F800000),
                (nan, 0xFFC00000)]
        for (text, ea, eb, pred), (a, b) in zip(cases, srcs):
            lines = load(a, b) + [text, "v_cndmask_b32_e64 v10, v10, v14, s20"]
            with self.subTest(text=text):
                self.assertEqual(run(lines)[0], int(PREDS[pred](ea, eb)))

    def test_lg_is_ordered_without_ord_not_equal(self):
        # LG is less-or-greater, false on a NaN. OpFOrdNotEqual means the same in SPIR-V, but SPIRV-Cross (MoltenVK
        # 1.4.2) writes it as a plain `!=`, which Metal makes true on a NaN: ps_vcmp_f32_nan, ps_vcmp_f16 and the cmpx
        # forms failed on a Mac in exactly the LG bits of their NaN pairs. The values are checked above; this pins
        # the form, which the CPU interpreter can't tell apart.
        for text in ("v_cmp_lg_f32_e32 vcc_lo, v2, v3", "v_cmp_lg_f32_e64 s20, v2, v3", "v_cmpx_lg_f32_e32 v2, v3",
                     "v_cmp_lg_f16_e32 vcc_lo, v2, v3", "v_cmpx_lg_f16_e64 v2, v3"):
            with self.subTest(text=text):
                spv = support.translate_lines("ps_vcmp_lg", load(0x7FC00000, 0x3F800000) + [
                    text, "exp mrt0 v10, v11, v12, v13 done vm", "s_endpgm"])
                self.assertNotIn("OpFOrdNotEqual", spv)
                self.assertIn("OpFOrdLessThan", spv)
                self.assertIn("OpFOrdGreaterThan", spv)


if __name__ == "__main__":
    unittest.main()
