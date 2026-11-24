"""Semantic interpreter tests for RDNA2 v_mul_f16, v_exp_f16, v_log_f16, and v_rsq_f16.

Covers:
  - v_mul_f16: S0.f16 * S1.f16 (powers of two, fractions, negatives, +-0.0, +-inf, overflow, subnormals/underflow, NaN)
  - v_exp_f16: 2 ** S0.f16
  - v_log_f16: log2(S0.f16) (for S0 > 0; spirv_mini_interp rejects Log2(x <= 0) as undefined in GLSL.std.450)
  - v_rsq_f16: 1 / sqrt(S0.f16) (for S0 > 0; spirv_mini_interp rejects InverseSqrt(x <= 0) as undefined)

Per docs/design.md ("16-bit results"), GFX10+ 16-bit VALU ops write one 16-bit half of the destination VGPR
while preserving the other 16-bit half: _e32 writes bits 15:0 (preserving bits 31:16), and _e64 with
op_sel:[...,1] writes bits 31:16 (preserving bits 15:0).
"""
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

OLD = 0xBEEF1234

# Representative binary16 encodings.
H_POS_ZERO = 0x0000
H_NEG_ZERO = 0x8000
H_SUB_MIN = 0x0001       # 2^-24
H_NORM_MIN = 0x0400      # 2^-14
H_0P0625 = 0x2C00        # 0.0625 (2^-4)
H_0P25 = 0x3400          # 0.25
H_0P3 = 0x34CD           # ~0.3
H_0P5 = 0x3800           # 0.5
H_NEG_0P5 = 0xB800       # -0.5
H_1P0 = 0x3C00           # 1.0
H_NEG_1P0 = 0xBC00       # -1.0
H_1P25 = 0x3D00          # 1.25
H_1P5 = 0x3E00           # 1.5
H_NEG_1P5 = 0xBE00       # -1.5
H_1P75 = 0x3F00          # 1.75
H_2P0 = 0x4000           # 2.0
H_NEG_2P0 = 0xC000       # -2.0
H_3P0 = 0x4200           # 3.0
H_NEG_3P0 = 0xC200       # -3.0
H_4P0 = 0x4400           # 4.0
H_NEG_4P0 = 0xC400       # -4.0
H_5P5 = 0x4580           # 5.5
H_10P0 = 0x4900          # 10.0
H_15P0 = 0x4B80          # 15.0
H_16P0 = 0x4C00          # 16.0
H_NEG_25P0 = 0xCE40      # -25.0
H_1024 = 0x6400          # 1024.0
H_MAX = 0x7BFF           # 65504.0
H_NEG_MAX = 0xFBFF       # -65504.0
H_POS_INF = 0x7C00       # +inf
H_NEG_INF = 0xFC00       # -inf
H_QNAN = 0x7E01          # +qNaN


def h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f32(x: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.inf if x > 0 else -math.inf


def f2h(x: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return H_POS_INF if x > 0 else H_NEG_INF


def is_f16_nan(h: int) -> bool:
    return (h & 0x7FFF) > 0x7C00


def pack_expected(h: int, old: int = OLD, dst_hi: bool = False) -> int:
    return ((h << 16) | (old & 0xFFFF)) if dst_hi else ((old & 0xFFFF0000) | h)


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def ref_mul_f16(a: int, b: int, clamp: bool = False) -> int:
    r = f32(h2f(a) * h2f(b))
    if clamp and not math.isnan(r):
        r = clamp01(r)
    return f2h(r)


def ref_exp_f16(a: int, clamp: bool = False) -> int:
    x = h2f(a)
    if math.isnan(x):
        return H_QNAN
    r = (math.inf if x > 0 else 0.0) if math.isinf(x) else (f32(2.0 ** x) if x < 1024.0 else math.inf)
    if clamp:
        r = clamp01(r)
    return f2h(r)


def ref_log_f16(a: int, clamp: bool = False) -> int:
    x = h2f(a)
    if math.isnan(x):
        return H_QNAN
    r = math.inf if math.isinf(x) else f32(math.log2(x))
    if clamp:
        r = clamp01(r)
    return f2h(r)


def ref_rsq_f16(a: int, clamp: bool = False) -> int:
    x = h2f(a)
    if math.isnan(x):
        return H_QNAN
    r = 0.0 if math.isinf(x) else f32(1.0 / math.sqrt(x))
    if clamp:
        r = clamp01(r)
    return f2h(r)


def run_ps(setup_lines, op_line: str, old: int = OLD) -> int:
    lines = list(setup_lines) + [
        f"v_mov_b32 v0, {old:#x}",
        op_line,
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_f16_ops", lines)
    return si.run_fragment(spv)["%out_mrt0"][0]


class MulF16Tests(unittest.TestCase):
    EXACT_CASES = [
        (H_1P0, H_2P0, "1.0 * 2.0"),
        (H_0P5, H_4P0, "0.5 * 4.0"),
        (H_0P25, H_0P5, "0.25 * 0.5"),
        (H_1P5, H_1P5, "1.5 * 1.5"),
        (H_1P25, H_1P75, "1.25 * 1.75"),
        (H_NEG_1P5, H_2P0, "-1.5 * 2.0"),
        (H_NEG_2P0, H_NEG_3P0, "-2.0 * -3.0"),
        (H_POS_ZERO, H_1P0, "+0.0 * 1.0"),
        (H_NEG_ZERO, H_1P0, "-0.0 * 1.0"),
        (H_NEG_ZERO, H_NEG_1P0, "-0.0 * -1.0"),
        (H_POS_INF, H_2P0, "+inf * 2.0"),
        (H_NEG_INF, H_2P0, "-inf * 2.0"),
        (H_POS_INF, H_NEG_0P5, "+inf * -0.5"),
        (H_MAX, H_2P0, "overflow to +inf"),
        (H_NEG_MAX, H_2P0, "overflow to -inf"),
        (H_NORM_MIN, 0x1400, "2^-14 * 2^-10 = 2^-24 (min subnormal)"),
        (H_NORM_MIN, 0x1000, "2^-14 * 2^-11 = underflow to +0.0"),
    ]

    def test_exact_cases_e32_and_e64(self):
        for a, b, desc in self.EXACT_CASES:
            want = ref_mul_f16(a, b)
            for form, op_line in (
                ("e32", "v_mul_f16_e32 v0, v1, v2"),
                ("e64", "v_mul_f16_e64 v0, v1, v2"),
            ):
                with self.subTest(case=desc, form=form):
                    got = run_ps([f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}"], op_line)
                    self.assertEqual(got, pack_expected(want))

    def test_nan_propagation(self):
        got = run_ps([f"v_mov_b32 v1, {H_QNAN:#x}", f"v_mov_b32 v2, {H_1P0:#x}"], "v_mul_f16_e32 v0, v1, v2")
        self.assertEqual(got >> 16, OLD >> 16)
        self.assertTrue(is_f16_nan(got & 0xFFFF))

    def test_opsel_and_half_preservation(self):
        v1 = (H_1P5 << 16) | H_2P0
        v2 = (H_4P0 << 16) | H_0P5
        cases = [
            ("op_sel:[0,0,0]", False, ref_mul_f16(H_2P0, H_0P5)),
            ("op_sel:[1,0,0]", False, ref_mul_f16(H_1P5, H_0P5)),
            ("op_sel:[0,1,0]", False, ref_mul_f16(H_2P0, H_4P0)),
            ("op_sel:[1,1,1]", True, ref_mul_f16(H_1P5, H_4P0)),
            ("op_sel:[0,0,1]", True, ref_mul_f16(H_2P0, H_0P5)),
        ]
        for opsel, dst_hi, want in cases:
            with self.subTest(opsel=opsel):
                got = run_ps([f"v_mov_b32 v1, {v1:#x}", f"v_mov_b32 v2, {v2:#x}"],
                             f"v_mul_f16_e64 v0, v1, v2 {opsel}")
                self.assertEqual(got, pack_expected(want, OLD, dst_hi=dst_hi))

    def test_modifiers_and_operand_kinds(self):
        # neg, abs, neg-abs, clamp
        self.assertEqual(
            run_ps([f"v_mov_b32 v1, {H_1P5:#x}", f"v_mov_b32 v2, {H_2P0:#x}"], "v_mul_f16_e64 v0, -v1, v2"),
            pack_expected(ref_mul_f16(H_NEG_1P5, H_2P0)),
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v1, {H_NEG_1P5:#x}", f"v_mov_b32 v2, {H_2P0:#x}"], "v_mul_f16_e64 v0, |v1|, v2"),
            pack_expected(ref_mul_f16(H_1P5, H_2P0)),
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v1, {H_NEG_1P5:#x}", f"v_mov_b32 v2, {H_2P0:#x}"], "v_mul_f16_e64 v0, -|v1|, v2"),
            pack_expected(ref_mul_f16(H_NEG_1P5, H_2P0)),
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v1, {H_2P0:#x}", f"v_mov_b32 v2, {H_2P0:#x}"], "v_mul_f16_e64 v0, v1, v2 clamp"),
            pack_expected(H_1P0),
        )
        # SGPR, inline float constant, and literal
        self.assertEqual(
            run_ps([f"s_mov_b32 s4, {H_1P5:#x}", f"v_mov_b32 v2, {H_2P0:#x}"], "v_mul_f16_e32 v0, s4, v2"),
            pack_expected(ref_mul_f16(H_1P5, H_2P0)),
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v2, {H_4P0:#x}"], "v_mul_f16_e32 v0, 0.5, v2"),
            pack_expected(H_2P0),
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v2, {H_2P0:#x}"], f"v_mul_f16_e32 v0, {H_1P5:#x}, v2"),
            pack_expected(ref_mul_f16(H_1P5, H_2P0)),
        )


class UnaryTranscendentalF16Tests(unittest.TestCase):
    EXP_CASES = [
        H_NEG_25P0, H_NEG_4P0, H_NEG_2P0, H_NEG_1P0, H_NEG_ZERO, H_POS_ZERO,
        H_0P5, H_1P0, H_1P5, H_2P0, H_3P0, H_15P0, H_16P0, H_POS_INF, H_NEG_INF,
    ]
    LOG_CASES = [
        H_SUB_MIN, H_NORM_MIN, H_0P0625, H_0P25, H_0P3, H_0P5,
        H_1P0, H_1P5, H_2P0, H_3P0, H_4P0, H_5P5, H_10P0, H_1024, H_POS_INF,
    ]
    RSQ_CASES = [
        H_SUB_MIN, H_NORM_MIN, H_0P0625, H_0P25, H_0P3, H_0P5,
        H_1P0, H_1P5, H_4P0, H_5P5, H_10P0, H_16P0, H_1024, H_POS_INF,
    ]

    def test_exp_log_rsq_values(self):
        table = [
            ("v_exp_f16", self.EXP_CASES, ref_exp_f16),
            ("v_log_f16", self.LOG_CASES, ref_log_f16),
            ("v_rsq_f16", self.RSQ_CASES, ref_rsq_f16),
        ]
        for op, cases, ref_fn in table:
            for a in cases:
                want = ref_fn(a)
                for suffix in ("_e32", "_e64"):
                    with self.subTest(op=op + suffix, a=hex(a)):
                        got = run_ps([f"v_mov_b32 v1, {a:#x}"], f"{op}{suffix} v0, v1")
                        self.assertEqual(got, pack_expected(want))

    def test_nan_propagation(self):
        for op in ("v_exp_f16", "v_log_f16", "v_rsq_f16"):
            with self.subTest(op=op):
                got = run_ps([f"v_mov_b32 v1, {H_QNAN:#x}"], f"{op}_e32 v0, v1")
                self.assertEqual(got >> 16, OLD >> 16)
                self.assertTrue(is_f16_nan(got & 0xFFFF))

    def test_opsel_and_half_preservation(self):
        v1 = (H_4P0 << 16) | H_0P25
        for op, ref_fn in (("v_exp_f16", ref_exp_f16), ("v_log_f16", ref_log_f16), ("v_rsq_f16", ref_rsq_f16)):
            for src_hi, dst_hi, src_h in ((0, 0, H_0P25), (1, 0, H_4P0), (0, 1, H_0P25), (1, 1, H_4P0)):
                with self.subTest(op=op, src_hi=src_hi, dst_hi=dst_hi):
                    got = run_ps([f"v_mov_b32 v1, {v1:#x}"], f"{op}_e64 v0, v1 op_sel:[{src_hi},{dst_hi}]")
                    self.assertEqual(got, pack_expected(ref_fn(src_h), OLD, dst_hi=bool(dst_hi)))

    def test_modifiers_and_operand_kinds(self):
        # VOP3 neg/abs/clamp
        self.assertEqual(
            run_ps([f"v_mov_b32 v1, {H_2P0:#x}"], "v_exp_f16_e64 v0, -v1"),
            pack_expected(H_0P25),
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v1, {H_NEG_2P0:#x}"], "v_exp_f16_e64 v0, |v1|"),
            pack_expected(H_4P0),
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v1, {H_2P0:#x}"], "v_exp_f16_e64 v0, v1 clamp"),
            pack_expected(H_1P0),
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v1, {H_NEG_4P0:#x}"], "v_log_f16_e64 v0, |v1|"),
            pack_expected(H_2P0),
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v1, {H_0P5:#x}"], "v_log_f16_e64 v0, v1 clamp"),
            pack_expected(H_POS_ZERO),
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v1, {H_NEG_4P0:#x}"], "v_rsq_f16_e64 v0, |v1|"),
            pack_expected(H_0P5),
        )
        self.assertEqual(
            run_ps([f"v_mov_b32 v1, {H_0P25:#x}"], "v_rsq_f16_e64 v0, v1 clamp"),
            pack_expected(H_1P0),
        )
        # SGPR, inline constant, and literal operands
        self.assertEqual(run_ps([f"s_mov_b32 s4, {H_2P0:#x}"], "v_exp_f16_e32 v0, s4"), pack_expected(H_4P0))
        self.assertEqual(run_ps([], "v_log_f16_e32 v0, 4.0"), pack_expected(H_2P0))
        self.assertEqual(run_ps([], f"v_rsq_f16_e32 v0, {H_0P25:#x}"), pack_expected(H_2P0))


if __name__ == "__main__":
    unittest.main()
