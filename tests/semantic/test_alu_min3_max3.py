"""Semantic interpreter tests for RDNA2 3-way ALU min/max instructions:
- v_min3_f32: 3-way IEEE binary32 floating-point minimum
- v_max3_f32: 3-way IEEE binary32 floating-point maximum
- v_min3_i32: 3-way signed 32-bit two's complement integer minimum
- v_max3_u32: 3-way unsigned 32-bit integer maximum
(also covers v_min3_u32 and v_max3_i32 for completeness).

Each test case translates a synthetic pixel shader listing using tools/rdna2spirv.py,
executes the resulting SPIR-V in tests/semantic/spirv_mini_interp.py, and compares the
exported mrt0 color output words against independent plain-Python reference models.

Float min/max and IEEE 754 / RDNA2 semantics:
- Lowering to SPIR-V in tools/rdna2spirv.py maps v_min3_f32 and v_max3_f32 to GLSL.std.450
  NMin and NMax (OpExtInst %float %glsl NMin/NMax).
- NMin and NMax implement IEEE 754-2008 minNum / maxNum NaN skipping: if any operand is NaN,
  the non-NaN operand is chosen; if all operands are NaN, NaN is returned.
- Signed zeros are ordered with -0.0 below +0.0 (the hardware order measured on AMD RDNA2,
  docs/design.md): min(+0.0, -0.0) = -0.0 and max(+0.0, -0.0) = +0.0 in either operand order.
- VOP3 input modifiers: absolute value (|src|), negation (-src), and neg-abs (-|src|).
- VOP3 output modifier: clamp to [0.0, 1.0]. If all operands are NaN, clamp raises
  UnsupportedConstruct because GLSL.std.450 FClamp on NaN is undefined.

Integer min/max:
- v_min3_i32 and v_max3_i32 evaluate operands as signed 32-bit integers in two's complement.
- v_min3_u32 and v_max3_u32 evaluate operands as unsigned 32-bit integers.
- Boundary conditions where signed and unsigned order disagree: 0x80000000 is INT_MIN as i32
  but 2^31 as u32; 0xFFFFFFFF is -1 as i32 but UINT32_MAX as u32.

Zero game data, standard library Python only, Python 3.9+ compatible.
"""
from __future__ import annotations

import itertools
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
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

INT_OPS = ("v_min3_i32", "v_min3_u32", "v_max3_i32", "v_max3_u32")
FLOAT_OPS = ("v_min3_f32", "v_max3_f32")

# IEEE 754 binary32 bit patterns
POS_ZERO_BITS = 0x00000000
NEG_ZERO_BITS = 0x80000000
POS_INF_BITS = 0x7F800000
NEG_INF_BITS = 0xFF800000
QNAN_BITS = 0x7FC00000
SNAN_BITS = 0x7F800001
NEG_QNAN_BITS = 0xFFC00000
SUBNORM_MIN_BITS = 0x00000001  # 2^-149
SUBNORM_MAX_BITS = 0x007FFFFF  # (2^23 - 1) * 2^-149

# Triples where signed and unsigned order disagree, plus ties and boundary values.
INT_TRIPLES = (
    (5, 0x80000000, 7),
    (0xFFFFFFFF, 0, 1),
    (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
    (0x80000001, 0x80000000, 0x7FFFFFFE),
    (1, 0xFFFFFFFE, 0xFFFFFFFF),
    (9, 9, 0x80000000),
    (0xFFFFFFFF, 0xFFFFFFFF, 0),
    (3, 3, 3),
    (10, 0xFFFFFFF0, 5),
    (0x7FFFFFFF, 0x7FFFFFFF, 0x80000000),
)

# Float triples covering positive, negative, zero, infinities, subnormals, fractions and scales.
FLOAT_TRIPLES = (
    (1.5, -2.5, 3.5),
    (-10.0, 0.0, 10.0),
    (-0.5, -2.5, -1.5),
    (0.125, 0.5, 0.25),
    (100.0, 100.0, 100.0),
    (-1.0, -1.0, 0.0),
    (0.0, 0.0, 0.0),
    (-float("inf"), 0.0, 10.0),
    (10.0, 20.0, float("inf")),
    (-float("inf"), 5.0, float("inf")),
)


def s32(x: int) -> int:
    x &= MASK32
    return x - (1 << 32) if x & 0x80000000 else x


def u32(x: int) -> int:
    return x & MASK32


def u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f32_to_u32(f: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", f))[0]
    except OverflowError:
        return POS_INF_BITS if f > 0 else NEG_INF_BITS


def ref_nmin_nmax(name: str, x: float, y: float) -> float:
    """Reference GLSL.std.450 NMin / NMax: NaN operand gives the other operand; both NaN give NaN.
    Zeros are ordered -0.0 < +0.0 (measured RDNA2 hardware semantics)."""
    if math.isnan(x):
        return y if not math.isnan(y) else math.nan
    if math.isnan(y):
        return x
    kx, ky = (x, math.copysign(1.0, x)), (y, math.copysign(1.0, y))
    if name == "NMin":
        return y if ky < kx else x
    return y if ky > kx else x


def ref_int(op: str, *xs: int) -> int:
    key = s32 if op.endswith("_i32") else u32
    pick = min if "min3" in op else max
    return pick(xs, key=key) & MASK32


def ref_float(op: str, a: float, b: float, c: float, *, clamp: bool = False) -> float:
    fn = "NMin" if "min3" in op else "NMax"
    r = ref_nmin_nmax(fn, ref_nmin_nmax(fn, a, b), c)
    if clamp:
        if math.isnan(r):
            raise si.UnsupportedConstruct("clamp on NaN is undefined")
        r = max(0.0, min(1.0, r))
    return r


_TMP_DIR: tempfile.TemporaryDirectory[str] | None = None
_SRC_PATH: str = ""


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_min3_max3.s")
    with open(os.path.join(_TMP_DIR.name, "ps_min3_max3.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_alu(op: str, *xs: int, inline: tuple[int, ...] = (), clamp: bool = False) -> int:
    """Translate and run `op v0, a, b, c` where operands xs are raw 32-bit words."""
    if _TMP_DIR is None:
        setUpModule()
    lines, operands = [], []
    for i, x in enumerate(xs):
        if i in inline:
            operands.append(f"{x & MASK32:#x}")
        else:
            lines.append(f"v_mov_b32 v{i + 1}, {x & MASK32:#x}")
            operands.append(f"v{i + 1}")
    clamp_str = " clamp" if clamp else ""
    lines.append(f"{op} v0, {', '.join(operands)}{clamp_str}")
    lines.append("exp mrt0 v0, v0, v0, v0 done vm")
    lines.append("s_endpgm")
    asm = "; test_alu_min3_max3\n" + "".join(f"\t{ln}\n" for ln in lines)
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


def run_composite(asm_lines: list[str]) -> tuple[int, ...]:
    """Run arbitrary fragment shader lines ending with mrt0 export and s_endpgm."""
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_alu_composite\n" + "".join(f"\t{ln}\n" for ln in asm_lines)
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"]


class ReferenceTests(unittest.TestCase):
    """Plain-Python reference tests against hand-worked and IEEE 754 edge cases."""

    def test_integer_reference(self) -> None:
        self.assertEqual(ref_int("v_min3_i32", 5, 0x80000000, 7), 0x80000000)
        self.assertEqual(ref_int("v_min3_u32", 5, 0x80000000, 7), 5)
        self.assertEqual(ref_int("v_max3_i32", 5, 0x80000000, 7), 7)
        self.assertEqual(ref_int("v_max3_u32", 5, 0x80000000, 7), 0x80000000)
        self.assertEqual(ref_int("v_min3_i32", 0xFFFFFFFF, 0, 1), 0xFFFFFFFF)
        self.assertEqual(ref_int("v_max3_u32", 0xFFFFFFFF, 0, 1), 0xFFFFFFFF)
        self.assertEqual(ref_int("v_max3_i32", 0xFFFFFFFF, 0, 1), 1)
        self.assertEqual(ref_int("v_min3_u32", 0xFFFFFFFF, 0, 1), 0)

    def test_ieee754_float_reference(self) -> None:
        # Normal ordered values
        self.assertEqual(ref_float("v_min3_f32", 1.0, 2.0, 3.0), 1.0)
        self.assertEqual(ref_float("v_max3_f32", 1.0, 2.0, 3.0), 3.0)
        self.assertEqual(ref_float("v_min3_f32", -10.0, 0.0, 10.0), -10.0)
        self.assertEqual(ref_float("v_max3_f32", -10.0, 0.0, 10.0), 10.0)

        # IEEE 754 minNum / maxNum: non-NaN numbers are chosen over NaN
        nan = float("nan")
        self.assertEqual(ref_float("v_min3_f32", nan, 2.0, 3.0), 2.0)
        self.assertEqual(ref_float("v_min3_f32", 1.0, nan, 3.0), 1.0)
        self.assertEqual(ref_float("v_min3_f32", 1.0, 2.0, nan), 1.0)
        self.assertEqual(ref_float("v_max3_f32", nan, 2.0, 3.0), 3.0)
        self.assertEqual(ref_float("v_max3_f32", 1.0, nan, 3.0), 3.0)
        self.assertEqual(ref_float("v_max3_f32", 1.0, 2.0, nan), 2.0)

        # Two NaNs: remaining non-NaN is chosen
        self.assertEqual(ref_float("v_min3_f32", nan, nan, 4.0), 4.0)
        self.assertEqual(ref_float("v_min3_f32", nan, -5.0, nan), -5.0)
        self.assertEqual(ref_float("v_max3_f32", nan, nan, 4.0), 4.0)
        self.assertEqual(ref_float("v_max3_f32", nan, -5.0, nan), -5.0)

        # Three NaNs: result must be NaN
        self.assertTrue(math.isnan(ref_float("v_min3_f32", nan, nan, nan)))
        self.assertTrue(math.isnan(ref_float("v_max3_f32", nan, nan, nan)))

        # Signed zero handling (-0.0 < +0.0)
        self.assertEqual(f32_to_u32(ref_float("v_min3_f32", 0.0, -0.0, 1.0)), NEG_ZERO_BITS)
        self.assertEqual(f32_to_u32(ref_float("v_max3_f32", 0.0, -0.0, -1.0)), POS_ZERO_BITS)

        # Infinities
        inf = float("inf")
        self.assertEqual(ref_float("v_min3_f32", -inf, 0.0, inf), -inf)
        self.assertEqual(ref_float("v_max3_f32", -inf, 0.0, inf), inf)


class FloatMin3Max3Tests(unittest.TestCase):
    """v_min3_f32 and v_max3_f32 tests across scales, permutations, clamping, and inline constants."""

    def test_float_every_order(self) -> None:
        for op in FLOAT_OPS:
            for fs in FLOAT_TRIPLES:
                for perm in sorted(set(itertools.permutations(fs))):
                    with self.subTest(op=op, fs=perm):
                        words = [f32_to_u32(f) for f in perm]
                        want = f32_to_u32(ref_float(op, *perm))
                        self.assertEqual(run_alu(op, *words), want)

    def test_float_signed_zeros(self) -> None:
        # RDNA2 orders -0.0 below +0.0:
        # min3(+0, -0, 1.0) = -0.0
        # max3(+0, -0, -1.0) = +0.0
        zeros_triples_min = [
            (POS_ZERO_BITS, NEG_ZERO_BITS, f32_to_u32(1.0)),
            (NEG_ZERO_BITS, POS_ZERO_BITS, f32_to_u32(1.0)),
            (f32_to_u32(1.0), POS_ZERO_BITS, NEG_ZERO_BITS),
        ]
        for xs in zeros_triples_min:
            with self.subTest(case="signed_zeros_min3", xs=[hex(x) for x in xs]):
                self.assertEqual(run_alu("v_min3_f32", *xs), NEG_ZERO_BITS)

        zeros_triples_max = [
            (POS_ZERO_BITS, NEG_ZERO_BITS, f32_to_u32(-1.0)),
            (NEG_ZERO_BITS, POS_ZERO_BITS, f32_to_u32(-1.0)),
            (f32_to_u32(-1.0), POS_ZERO_BITS, NEG_ZERO_BITS),
        ]
        for xs in zeros_triples_max:
            with self.subTest(case="signed_zeros_max3", xs=[hex(x) for x in xs]):
                self.assertEqual(run_alu("v_max3_f32", *xs), POS_ZERO_BITS)

    def test_float_subnormals(self) -> None:
        # Subnormals handled under IEEE gradual ordering
        words = [SUBNORM_MIN_BITS, SUBNORM_MAX_BITS, f32_to_u32(1.0)]
        self.assertEqual(run_alu("v_min3_f32", *words), SUBNORM_MIN_BITS)
        self.assertEqual(run_alu("v_max3_f32", *words), f32_to_u32(1.0))

    def test_float_clamping(self) -> None:
        cases = [
            ("v_min3_f32", (2.5, 3.5, 4.5), 1.0),
            ("v_min3_f32", (-0.5, 0.5, 0.8), 0.0),
            ("v_min3_f32", (0.25, 0.75, 0.5), 0.25),
            ("v_max3_f32", (-2.5, -1.0, -0.5), 0.0),
            ("v_max3_f32", (0.2, 1.5, 0.8), 1.0),
            ("v_max3_f32", (0.2, 0.7, 0.5), 0.7),
        ]
        for op, fs, want in cases:
            with self.subTest(op=op, fs=fs):
                words = [f32_to_u32(f) for f in fs]
                got = run_alu(op, *words, clamp=True)
                want_bits = f32_to_u32(want)
                self.assertEqual(got, want_bits)
                self.assertEqual(want_bits, f32_to_u32(ref_float(op, *fs, clamp=True)))

    def test_float_inline_constants(self) -> None:
        cases = [
            ("v_min3_f32", (1.5, -2.5, 3.5), (1, 2)),
            ("v_max3_f32", (1.5, -2.5, 3.5), (0, 2)),
            ("v_min3_f32", (0.2, 0.4, 0.8), (0, 1)),
            ("v_max3_f32", (-10.0, 0.0, 10.0), (1, 2)),
        ]
        for op, fs, inline in cases:
            with self.subTest(op=op, fs=fs, inline=inline):
                words = [f32_to_u32(f) for f in fs]
                want = f32_to_u32(ref_float(op, *fs))
                got = run_alu(op, *words, inline=inline)
                self.assertEqual(got, want)


class FloatNanSkippingTests(unittest.TestCase):
    """GLSL.std.450 NMin and NMax implement IEEE 754-2008 NaN-skipping:
    - If one or two operands are NaN, the non-NaN operand is chosen.
    - If all three operands are NaN, NaN is returned.
    - If clamp modifier is specified on all-NaN, UnsupportedConstruct is raised because
      FClamp on NaN is undefined.
    """

    def test_single_nan_operand_skipped(self) -> None:
        nans = [QNAN_BITS, SNAN_BITS, NEG_QNAN_BITS]
        a_f, b_f = 1.0, 2.0
        a_bits, b_bits = f32_to_u32(a_f), f32_to_u32(b_f)
        for op in FLOAT_OPS:
            expected_f = 1.0 if "min3" in op else 2.0
            expected_bits = f32_to_u32(expected_f)
            for nan_bits in nans:
                triples = [
                    (nan_bits, a_bits, b_bits),
                    (a_bits, nan_bits, b_bits),
                    (a_bits, b_bits, nan_bits),
                ]
                for xs in triples:
                    with self.subTest(op=op, nan=hex(nan_bits), xs=[hex(x) for x in xs]):
                        self.assertEqual(run_alu(op, *xs), expected_bits)

    def test_two_nan_operands_skipped(self) -> None:
        c_f = 5.0
        c_bits = f32_to_u32(c_f)
        for op in FLOAT_OPS:
            triples = [
                (QNAN_BITS, SNAN_BITS, c_bits),
                (QNAN_BITS, c_bits, NEG_QNAN_BITS),
                (c_bits, QNAN_BITS, SNAN_BITS),
            ]
            for xs in triples:
                with self.subTest(op=op, xs=[hex(x) for x in xs]):
                    self.assertEqual(run_alu(op, *xs), c_bits)

    def test_all_three_nans_produce_nan(self) -> None:
        for op in FLOAT_OPS:
            with self.subTest(op=op):
                res = run_alu(op, QNAN_BITS, SNAN_BITS, NEG_QNAN_BITS)
                self.assertTrue(math.isnan(u32_to_f32(res)))

    def test_clamp_on_all_nan_rejected(self) -> None:
        for op in FLOAT_OPS:
            with self.subTest(op=op):
                with self.assertRaises(si.UnsupportedConstruct):
                    run_alu(op, QNAN_BITS, SNAN_BITS, NEG_QNAN_BITS, clamp=True)


class FloatModifiersTests(unittest.TestCase):
    """VOP3 input modifiers: abs (|src|), neg (-src), and neg-abs (-|src|)."""

    def test_abs_and_neg_input_modifiers(self) -> None:
        lines = [
            "v_mov_b32 v1, 2.5",
            "v_mov_b32 v2, -3.5",
            "v_mov_b32 v3, 4.0",
            "v_min3_f32 v0, -v1, |v2|, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        out = run_composite(lines)
        # -v1 is -2.5, |v2| is 3.5, v3 is 4.0. min is -2.5.
        self.assertEqual(out[0], f32_to_u32(-2.5))

    def test_max3_with_neg_abs_modifiers(self) -> None:
        lines = [
            "v_mov_b32 v1, 2.5",
            "v_mov_b32 v2, -3.5",
            "v_mov_b32 v3, -1.0",
            "v_max3_f32 v0, -v1, |v2|, -|v3|",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        out = run_composite(lines)
        # -v1 is -2.5, |v2| is 3.5, -|v3| is -1.0. max is 3.5.
        self.assertEqual(out[0], f32_to_u32(3.5))


class IntegerMin3Max3Tests(unittest.TestCase):
    """v_min3_i32 and v_max3_u32 (plus v_min3_u32 and v_max3_i32) order, sweep, and inline constants."""

    def test_every_order(self) -> None:
        for op in INT_OPS:
            for xs in INT_TRIPLES:
                for perm in sorted(set(itertools.permutations(xs))):
                    with self.subTest(op=op, xs=[hex(v) for v in perm]):
                        self.assertEqual(run_alu(op, *perm), ref_int(op, *perm))

    def test_sweep(self) -> None:
        values = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
        for op in INT_OPS:
            for xs in itertools.product(values, repeat=3):
                with self.subTest(op=op, xs=[hex(v) for v in xs]):
                    self.assertEqual(run_alu(op, *xs), ref_int(op, *xs))

    def test_inline_constants(self) -> None:
        cases = [
            ("v_min3_i32", (0x7FFFFFFF, 0xFFFFFFF0, 64), 0xFFFFFFF0),
            ("v_min3_u32", (0x7FFFFFFF, 0xFFFFFFF0, 64), 64),
            ("v_max3_i32", (0x80000000, 0xFFFFFFF0, 64), 64),
            ("v_max3_u32", (0x80000000, 0xFFFFFFF0, 64), 0xFFFFFFF0),
        ]
        for op, xs, want in cases:
            with self.subTest(op=op):
                self.assertEqual(want, ref_int(op, *xs))
                self.assertEqual(run_alu(op, *xs, inline=(1, 2)), want)


class CompositeMrt0ShaderTests(unittest.TestCase):
    """Combined pixel shader exporting v_min3_f32, v_max3_f32, v_min3_i32, and v_max3_u32 to mrt0."""

    def test_composite_shader_all_four_ops(self) -> None:
        lines = [
            "v_mov_b32 v4, 1.5",
            "v_mov_b32 v5, -2.5",
            "v_mov_b32 v6, 3.5",
            "v_min3_f32 v0, v4, v5, v6",
            "v_max3_f32 v1, v4, v5, v6",
            "v_mov_b32 v7, 10",
            "v_mov_b32 v8, 0xfffffff0",
            "v_mov_b32 v9, 5",
            "v_min3_i32 v2, v7, v8, v9",
            "v_max3_u32 v3, v7, v8, v9",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        out = run_composite(lines)
        self.assertEqual(len(out), 4)

        want_min_f = f32_to_u32(ref_float("v_min3_f32", 1.5, -2.5, 3.5))
        want_max_f = f32_to_u32(ref_float("v_max3_f32", 1.5, -2.5, 3.5))
        want_min_i = ref_int("v_min3_i32", 10, 0xFFFFFFF0, 5)
        want_max_u = ref_int("v_max3_u32", 10, 0xFFFFFFF0, 5)

        self.assertEqual(out[0], want_min_f, "mrt0.x must match v_min3_f32")
        self.assertEqual(out[1], want_max_f, "mrt0.y must match v_max3_f32")
        self.assertEqual(out[2], want_min_i, "mrt0.z must match v_min3_i32")
        self.assertEqual(out[3], want_max_u, "mrt0.w must match v_max3_u32")


class NegativeSensitivityTests(unittest.TestCase):
    """Negative controls ensuring min/max or signed/unsigned confusions fail assertions."""

    def test_signed_vs_unsigned_sabotage_is_caught(self) -> None:
        s0, s1, s2 = 10, 0xFFFFFFF0, 5
        self.assertNotEqual(ref_int("v_min3_i32", s0, s1, s2), ref_int("v_min3_u32", s0, s1, s2))
        self.assertNotEqual(ref_int("v_max3_i32", s0, s1, s2), ref_int("v_max3_u32", s0, s1, s2))

    def test_min_vs_max_sabotage_is_caught(self) -> None:
        f0, f1, f2 = 1.0, 2.0, 3.0
        self.assertNotEqual(ref_float("v_min3_f32", f0, f1, f2), ref_float("v_max3_f32", f0, f1, f2))
        i0, i1, i2 = 10, 20, 30
        self.assertNotEqual(ref_int("v_min3_i32", i0, i1, i2), ref_int("v_max3_i32", i0, i1, i2))

    def test_signed_zero_sabotage_is_caught(self) -> None:
        # If signed zeros are incorrectly considered identical (+0.0 == -0.0 in Python comparison)
        # without ordering key, min(+0.0, -0.0) would return +0.0 instead of -0.0 when +0 is first.
        bad_min = min(0.0, -0.0)
        self.assertEqual(f32_to_u32(bad_min), POS_ZERO_BITS)
        self.assertEqual(f32_to_u32(ref_nmin_nmax("NMin", 0.0, -0.0)), NEG_ZERO_BITS)


if __name__ == "__main__":
    unittest.main()
