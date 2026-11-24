"""Semantic interpreter tests for single-precision floating point min/max (v_min_f32, v_max_f32, v_min3_f32, v_max3_f32, v_med3_f32).

Models IEEE 754-2008 / 2019 binary32 minNum/maxNum and AMD RDNA2 / GFX10 VALU semantics:
  - v_min_f32: D.f = min(S0.f, S1.f)
  - v_max_f32: D.f = max(S0.f, S1.f)
  - v_min3_f32: D.f = min(min(S0.f, S1.f), S2.f)
  - v_max3_f32: D.f = max(max(S0.f, S1.f), S2.f)
  - v_med3_f32: D.f = median(S0.f, S1.f, S2.f) with fallback to min3 if any operand is NaN.

RDNA2 Floating-Point Rules:
  1. NaN suppression (IEEE 754-2008 minNum/maxNum):
     If one operand is NaN and the other is a valid floating-point number, the non-NaN operand
     is returned with its bit pattern intact. If both operands are NaN, a quiet NaN is returned.
  2. Signed zeros (-0.0 < +0.0):
     RDNA2 orders -0.0 strictly below +0.0 (measured on hardware; docs/design.md).
     min(+0.0, -0.0) = -0.0, min(-0.0, +0.0) = -0.0
     max(+0.0, -0.0) = +0.0, max(-0.0, +0.0) = +0.0
  3. Infinities and Extreme Floats:
     -INF < -FLT_MAX < ... < -FLT_MIN < -0.0 < +0.0 < +FLT_MIN < ... < +FLT_MAX < +INF.
     Subnormals are strictly preserved (not flushed to zero in IEEE mode).
  4. VOP2 (e32) and VOP3 (e64) Encoding Forms & Modifiers:
     Input modifiers: neg (-src), abs (|src|), and neg-abs (-|src|).
     Output modifier: clamp to [0.0, 1.0].
  5. Translation lowering:
     tools/rdna2spirv.py maps v_min_f32, v_max_f32, v_min3_f32, and v_max3_f32 to GLSL.std.450
     NMin and NMax (OpExtInst %float %glsl NMin/NMax), which enforce identical NaN-skipping and
     signed-zero rules. v_med3_f32 lowers to FClamp with an OpIsNan check selecting NMin(NMin(x, y), z).
  6. Wavefront SIMD execution:
     Evaluates lockstep execution across 32 and 64 lanes with divergent float inputs.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
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
SIGN_MASK = 0x80000000
EXP_MASK = 0x7F800000
MANT_MASK = 0x007FFFFF

# Canonical IEEE 754 binary32 bit patterns
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
HALF = 0x3F000000          # 0.5
NEG_HALF = 0xBF000000      # -0.5
ONE = 0x3F800000           # 1.0
NEG_ONE = 0xBF800000       # -1.0
ONE_HALF = 0x3FC00000      # 1.5
TWO = 0x40000000           # 2.0
NEG_TWO = 0xC0000000       # -2.0
THREE = 0x40400000         # 3.0
FOUR = 0x40800000          # 4.0
FIVE = 0x40A00000          # 5.0
TEN = 0x41200000           # 10.0
NEG_TEN = 0xC1200000       # -10.0
FLT_MIN = 0x00800000       # Smallest positive normal: 2^-126
NEG_FLT_MIN = 0x80800000   # -2^-126
FLT_MAX = 0x7F7FFFFF       # (2 - 2^-23) * 2^127
NEG_FLT_MAX = 0xFF7FFFFF
POS_INF = 0x7F800000       # +infinity
NEG_INF = 0xFF800000       # -infinity
QNAN = 0x7FC00000          # Canonical quiet NaN
NEG_QNAN = 0xFFC00000      # Negative quiet NaN
QNAN_PAYLOAD = 0x7FC10001  # Quiet NaN with payload A
QNAN_PAYLOAD_B = 0xFFC20002  # Quiet NaN with payload B
SNAN = 0x7F800001          # Signaling NaN
SUBNORM_MIN = 0x00000001   # Smallest subnormal: 2^-149
SUBNORM_MAX = 0x007FFFFF   # Largest positive subnormal
NEG_SUBNORM_MIN = 0x80000001
NEG_SUBNORM_MAX = 0x807FFFFF

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def is_nan_bits(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 NaN."""
    bits &= MASK32
    return (bits & EXP_MASK) == EXP_MASK and (bits & MANT_MASK) != 0


def is_qnan_bits(bits: int) -> bool:
    """True if bits represent a quiet NaN (MSB of mantissa is 1)."""
    bits &= MASK32
    return is_nan_bits(bits) and bool(bits & 0x00400000)


def is_snan_bits(bits: int) -> bool:
    """True if bits represent a signaling NaN (MSB of mantissa is 0)."""
    bits &= MASK32
    return is_nan_bits(bits) and not bool(bits & 0x00400000)


def is_subnormal_bits(bits: int) -> bool:
    """True if bits represent an IEEE 754 subnormal (denormal) number."""
    bits &= MASK32
    return (bits & EXP_MASK) == 0 and (bits & MANT_MASK) != 0


def zero_ordered_key(v: float) -> tuple[float, float]:
    """Sort key ordering -0.0 strictly below +0.0."""
    return (v, math.copysign(1.0, v))


def ref_min_f32(s0: int, s1: int) -> int:
    """Pure bitwise reference oracle for RDNA2 v_min_f32.

    Rules:
      - If one operand is NaN, return the other operand unchanged.
      - If both operands are NaN, return canonical quiet NaN (0x7FC00000).
      - If values are non-NaN, order -0.0 < +0.0.
    """
    s0 &= MASK32
    s1 &= MASK32
    nan0 = is_nan_bits(s0)
    nan1 = is_nan_bits(s1)
    if nan0 and nan1:
        return QNAN
    if nan0:
        return s1
    if nan1:
        return s0

    f0 = bits_to_f32(s0)
    f1 = bits_to_f32(s1)
    k0 = zero_ordered_key(f0)
    k1 = zero_ordered_key(f1)
    return s0 if k0 <= k1 else s1


def ref_max_f32(s0: int, s1: int) -> int:
    """Pure bitwise reference oracle for RDNA2 v_max_f32.

    Rules:
      - If one operand is NaN, return the other operand unchanged.
      - If both operands are NaN, return canonical quiet NaN (0x7FC00000).
      - If values are non-NaN, order -0.0 < +0.0.
    """
    s0 &= MASK32
    s1 &= MASK32
    nan0 = is_nan_bits(s0)
    nan1 = is_nan_bits(s1)
    if nan0 and nan1:
        return QNAN
    if nan0:
        return s1
    if nan1:
        return s0

    f0 = bits_to_f32(s0)
    f1 = bits_to_f32(s1)
    k0 = zero_ordered_key(f0)
    k1 = zero_ordered_key(f1)
    return s0 if k0 >= k1 else s1


def ref_min3_f32(s0: int, s1: int, s2: int) -> int:
    """Pure bitwise reference oracle for RDNA2 v_min3_f32: min(min(s0, s1), s2)."""
    return ref_min_f32(ref_min_f32(s0, s1), s2)


def ref_max3_f32(s0: int, s1: int, s2: int) -> int:
    """Pure bitwise reference oracle for RDNA2 v_max3_f32: max(max(s0, s1), s2)."""
    return ref_max_f32(ref_max_f32(s0, s1), s2)


def ref_med3_f32(s0: int, s1: int, s2: int) -> int:
    """Pure bitwise reference oracle for RDNA2 v_med3_f32.

    Rules:
      - If any operand is NaN, hardware evaluates min3(s0, s1, s2).
      - Otherwise, returns the median with -0.0 < +0.0 tie-breaking.
    """
    if is_nan_bits(s0) or is_nan_bits(s1) or is_nan_bits(s2):
        return ref_min3_f32(s0, s1, s2)
    triplet = [(bits_to_f32(s), zero_ordered_key(bits_to_f32(s)), s) for s in (s0, s1, s2)]
    triplet.sort(key=lambda t: t[1])
    return triplet[1][2]


def apply_f32_modifiers(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    """Applies VOP3 input modifiers: abs (|x|) first, then neg (-x)."""
    bits &= MASK32
    if abs_val:
        bits &= ~SIGN_MASK
    if neg:
        bits ^= SIGN_MASK
    return bits & MASK32


def apply_f32_clamp(bits: int) -> int:
    """Applies VOP3 clamp modifier: clamps result to [0.0, 1.0]."""
    bits &= MASK32
    if is_nan_bits(bits):
        return bits
    val = bits_to_f32(bits)
    if val < 0.0 or (val == 0.0 and (bits & SIGN_MASK)):
        return POS_ZERO
    if val > 1.0:
        return ONE
    return bits


_TMP_DIR: tempfile.TemporaryDirectory[str] | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_f32_min_max.s")
    with open(os.path.join(_TMP_DIR.name, "ps_f32_min_max.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_f32_2op(
    op: str,
    s0: int,
    s1: int,
    e64: bool = False,
    neg0: bool = False,
    abs0: bool = False,
    neg1: bool = False,
    abs1: bool = False,
    clamp: bool = False,
) -> int:
    """Translates and runs a 2-operand float min/max instruction through rdna2spirv and spirv_mini_interp."""
    if _TMP_DIR is None or _SRC_PATH is None:
        setUpModule()
        assert _SRC_PATH is not None

    def fmt_src(v_name: str, neg: bool, abs_val: bool) -> str:
        s = v_name
        if abs_val:
            s = f"|{s}|"
        if neg:
            s = f"-{s}"
        return s

    instr_name = f"{op}_e64" if e64 else (f"{op}_e32" if not (neg0 or abs0 or neg1 or abs1 or clamp) else f"{op}_e64")
    src0_str = fmt_src("v1", neg0, abs0)
    src1_str = fmt_src("v2", neg1, abs1)
    clamp_str = " clamp" if clamp else ""

    lines = [
        f"\tv_mov_b32 v1, {s0 & MASK32:#x}",
        f"\tv_mov_b32 v2, {s1 & MASK32:#x}",
        f"\t{instr_name} v0, {src0_str}, {src1_str}{clamp_str}",
        "\texp mrt0 v0, v0, v0, v0 done vm",
        "\ts_endpgm",
    ]
    asm = "; ps_f32_min_max\n" + "\n".join(lines) + "\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


def run_f32_3op(
    op: str,
    s0: int,
    s1: int,
    s2: int,
    neg0: bool = False,
    abs0: bool = False,
    neg1: bool = False,
    abs1: bool = False,
    neg2: bool = False,
    abs2: bool = False,
    clamp: bool = False,
) -> int:
    """Translates and runs a 3-operand float min/max/med instruction through rdna2spirv and spirv_mini_interp."""
    if _TMP_DIR is None or _SRC_PATH is None:
        setUpModule()
        assert _SRC_PATH is not None

    def fmt_src(v_name: str, neg: bool, abs_val: bool) -> str:
        s = v_name
        if abs_val:
            s = f"|{s}|"
        if neg:
            s = f"-{s}"
        return s

    clamp_str = " clamp" if clamp else ""
    lines = [
        f"\tv_mov_b32 v1, {s0 & MASK32:#x}",
        f"\tv_mov_b32 v2, {s1 & MASK32:#x}",
        f"\tv_mov_b32 v3, {s2 & MASK32:#x}",
        f"\t{op} v0, {fmt_src('v1', neg0, abs0)}, {fmt_src('v2', neg1, abs1)}, {fmt_src('v3', neg2, abs2)}{clamp_str}",
        "\texp mrt0 v0, v0, v0, v0 done vm",
        "\ts_endpgm",
    ]
    asm = "; ps_f32_min_max3\n" + "\n".join(lines) + "\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class TestF32ReferenceOracle2Op(unittest.TestCase):
    """Verifies the pure-Python reference oracle ref_min_f32 and ref_max_f32."""

    def test_ordinary_comparisons(self) -> None:
        self.assertEqual(ref_min_f32(ONE, TWO), ONE)
        self.assertEqual(ref_min_f32(TWO, ONE), ONE)
        self.assertEqual(ref_max_f32(ONE, TWO), TWO)
        self.assertEqual(ref_max_f32(TWO, ONE), TWO)
        self.assertEqual(ref_min_f32(NEG_ONE, NEG_TWO), NEG_TWO)
        self.assertEqual(ref_max_f32(NEG_ONE, NEG_TWO), NEG_ONE)
        self.assertEqual(ref_min_f32(NEG_ONE, ONE), NEG_ONE)
        self.assertEqual(ref_max_f32(NEG_ONE, ONE), ONE)

    def test_signed_zeros(self) -> None:
        self.assertEqual(ref_min_f32(POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(ref_min_f32(NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(ref_min_f32(POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_min_f32(NEG_ZERO, NEG_ZERO), NEG_ZERO)

        self.assertEqual(ref_max_f32(POS_ZERO, NEG_ZERO), POS_ZERO)
        self.assertEqual(ref_max_f32(NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_max_f32(POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_max_f32(NEG_ZERO, NEG_ZERO), NEG_ZERO)

    def test_infinities(self) -> None:
        self.assertEqual(ref_min_f32(NEG_INF, POS_INF), NEG_INF)
        self.assertEqual(ref_max_f32(NEG_INF, POS_INF), POS_INF)
        self.assertEqual(ref_min_f32(POS_INF, FLT_MAX), FLT_MAX)
        self.assertEqual(ref_max_f32(POS_INF, FLT_MAX), POS_INF)
        self.assertEqual(ref_min_f32(NEG_INF, NEG_FLT_MAX), NEG_INF)
        self.assertEqual(ref_max_f32(NEG_INF, NEG_FLT_MAX), NEG_FLT_MAX)

    def test_normals_and_subnormals(self) -> None:
        self.assertEqual(ref_min_f32(FLT_MIN, SUBNORM_MAX), SUBNORM_MAX)
        self.assertEqual(ref_max_f32(FLT_MIN, SUBNORM_MAX), FLT_MIN)
        self.assertEqual(ref_min_f32(SUBNORM_MIN, SUBNORM_MAX), SUBNORM_MIN)
        self.assertEqual(ref_max_f32(SUBNORM_MIN, SUBNORM_MAX), SUBNORM_MAX)
        self.assertEqual(ref_min_f32(NEG_FLT_MIN, NEG_SUBNORM_MAX), NEG_FLT_MIN)
        self.assertEqual(ref_max_f32(NEG_FLT_MIN, NEG_SUBNORM_MAX), NEG_SUBNORM_MAX)

    def test_nan_suppression(self) -> None:
        self.assertEqual(ref_min_f32(QNAN, ONE), ONE)
        self.assertEqual(ref_min_f32(ONE, QNAN), ONE)
        self.assertEqual(ref_max_f32(QNAN, ONE), ONE)
        self.assertEqual(ref_max_f32(ONE, QNAN), ONE)

        self.assertEqual(ref_min_f32(SNAN, TWO), TWO)
        self.assertEqual(ref_max_f32(TWO, SNAN), TWO)

        self.assertTrue(is_nan_bits(ref_min_f32(QNAN, SNAN)))
        self.assertTrue(is_nan_bits(ref_max_f32(QNAN, SNAN)))

    def test_nan_payload_preservation(self) -> None:
        custom_float = 0x3EAAAAAB  # 0.33333334
        self.assertEqual(ref_min_f32(QNAN_PAYLOAD, custom_float), custom_float)
        self.assertEqual(ref_max_f32(custom_float, QNAN_PAYLOAD), custom_float)


class TestF32ReferenceOracle3Op(unittest.TestCase):
    """Verifies reference oracles for 3-way instructions: min3, max3, med3."""

    def test_min3_permutations(self) -> None:
        triples = [
            (ONE, TWO, THREE),
            (NEG_TEN, POS_ZERO, TEN),
            (NEG_TWO, NEG_ONE, f32_to_bits(-3.0)),
        ]
        for a, b, c in triples:
            expected = ref_min_f32(ref_min_f32(a, b), c)
            for p0, p1, p2 in itertools.permutations([a, b, c]):
                self.assertEqual(ref_min3_f32(p0, p1, p2), expected)

    def test_max3_permutations(self) -> None:
        triples = [
            (ONE, TWO, THREE),
            (NEG_TEN, POS_ZERO, TEN),
            (NEG_TWO, NEG_ONE, f32_to_bits(-3.0)),
        ]
        for a, b, c in triples:
            expected = ref_max_f32(ref_max_f32(a, b), c)
            for p0, p1, p2 in itertools.permutations([a, b, c]):
                self.assertEqual(ref_max3_f32(p0, p1, p2), expected)

    def test_med3_permutations(self) -> None:
        for p in itertools.permutations([ONE, TWO, THREE]):
            self.assertEqual(ref_med3_f32(*p), TWO)

        for p in itertools.permutations([NEG_TEN, POS_ZERO, TEN]):
            self.assertEqual(ref_med3_f32(*p), POS_ZERO)

    def test_signed_zeros_3op(self) -> None:
        self.assertEqual(ref_min3_f32(POS_ZERO, NEG_ZERO, ONE), NEG_ZERO)
        self.assertEqual(ref_max3_f32(POS_ZERO, NEG_ZERO, NEG_ONE), POS_ZERO)
        self.assertEqual(ref_med3_f32(POS_ZERO, NEG_ZERO, ONE), POS_ZERO)
        self.assertEqual(ref_med3_f32(POS_ZERO, NEG_ZERO, NEG_ONE), NEG_ZERO)

    def test_med3_nan_fallback_to_min3(self) -> None:
        self.assertEqual(ref_med3_f32(QNAN, TWO, THREE), TWO)
        self.assertEqual(ref_med3_f32(ONE, QNAN, THREE), ONE)
        self.assertEqual(ref_med3_f32(ONE, TWO, QNAN), ONE)
        self.assertEqual(ref_med3_f32(QNAN, QNAN, FOUR), FOUR)

    def test_extreme_floats_3op(self) -> None:
        self.assertEqual(ref_min3_f32(NEG_INF, ONE, POS_INF), NEG_INF)
        self.assertEqual(ref_max3_f32(NEG_INF, ONE, POS_INF), POS_INF)
        self.assertEqual(ref_med3_f32(NEG_INF, ONE, POS_INF), ONE)


class TestVMinVMaxF32Execution(unittest.TestCase):
    """Executes synthetic shaders testing v_min_f32 and v_max_f32 in spirv_mini_interp."""

    TEST_PAIRS = [
        (ONE, TWO),
        (TWO, ONE),
        (NEG_ONE, NEG_TWO),
        (NEG_TWO, NEG_ONE),
        (HALF, ONE_HALF),
        (POS_INF, NEG_INF),
        (FLT_MAX, ONE),
        (NEG_FLT_MAX, NEG_ONE),
        (FLT_MIN, SUBNORM_MAX),
        (SUBNORM_MIN, FLT_MIN),
        (QNAN, THREE),
        (FOUR, QNAN),
    ]

    def test_v_min_f32_e32_and_e64(self) -> None:
        for s0, s1 in self.TEST_PAIRS:
            expected = ref_min_f32(s0, s1)
            with self.subTest(s0=hex(s0), s1=hex(s1), form="e32"):
                got_e32 = run_f32_2op("v_min_f32", s0, s1, e64=False)
                self.assertEqual(got_e32, expected)
            with self.subTest(s0=hex(s0), s1=hex(s1), form="e64"):
                got_e64 = run_f32_2op("v_min_f32", s0, s1, e64=True)
                self.assertEqual(got_e64, expected)

    def test_v_max_f32_e32_and_e64(self) -> None:
        for s0, s1 in self.TEST_PAIRS:
            expected = ref_max_f32(s0, s1)
            with self.subTest(s0=hex(s0), s1=hex(s1), form="e32"):
                got_e32 = run_f32_2op("v_max_f32", s0, s1, e64=False)
                self.assertEqual(got_e32, expected)
            with self.subTest(s0=hex(s0), s1=hex(s1), form="e64"):
                got_e64 = run_f32_2op("v_max_f32", s0, s1, e64=True)
                self.assertEqual(got_e64, expected)


class TestVMin3VMax3VMed3Execution(unittest.TestCase):
    """Executes synthetic shaders testing v_min3_f32, v_max3_f32, and v_med3_f32."""

    TEST_TRIPLES = [
        (ONE, TWO, THREE),
        (THREE, ONE, TWO),
        (TWO, THREE, ONE),
        (NEG_ONE, NEG_TWO, f32_to_bits(-3.0)),
        (NEG_TEN, POS_ZERO, TEN),
        (HALF, ONE_HALF, TWO),
    ]

    def test_v_min3_f32_execution(self) -> None:
        for s0, s1, s2 in self.TEST_TRIPLES:
            expected = ref_min3_f32(s0, s1, s2)
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                got = run_f32_3op("v_min3_f32", s0, s1, s2)
                self.assertEqual(got, expected)

    def test_v_max3_f32_execution(self) -> None:
        for s0, s1, s2 in self.TEST_TRIPLES:
            expected = ref_max3_f32(s0, s1, s2)
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                got = run_f32_3op("v_max3_f32", s0, s1, s2)
                self.assertEqual(got, expected)

    def test_v_med3_f32_execution(self) -> None:
        for s0, s1, s2 in self.TEST_TRIPLES:
            expected = ref_med3_f32(s0, s1, s2)
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                got = run_f32_3op("v_med3_f32", s0, s1, s2)
                self.assertEqual(got, expected)

    def test_signed_zeros_3op_execution(self) -> None:
        self.assertEqual(run_f32_3op("v_min3_f32", POS_ZERO, NEG_ZERO, ONE), NEG_ZERO)
        self.assertEqual(run_f32_3op("v_max3_f32", POS_ZERO, NEG_ZERO, NEG_ONE), POS_ZERO)
        self.assertEqual(run_f32_3op("v_med3_f32", POS_ZERO, NEG_ZERO, ONE), POS_ZERO)
        self.assertEqual(run_f32_3op("v_med3_f32", POS_ZERO, NEG_ZERO, NEG_ONE), NEG_ZERO)

    def test_extreme_values_3op_execution(self) -> None:
        self.assertEqual(run_f32_3op("v_min3_f32", NEG_INF, FLT_MIN, POS_INF), NEG_INF)
        self.assertEqual(run_f32_3op("v_max3_f32", NEG_INF, FLT_MIN, POS_INF), POS_INF)
        self.assertEqual(run_f32_3op("v_med3_f32", NEG_INF, FLT_MIN, POS_INF), FLT_MIN)


class TestSignedZerosTieBreak(unittest.TestCase):
    """Verifies that RDNA2 -0.0 < +0.0 ordering is strictly preserved by translator and interpreter."""

    def test_min_zero_combinations(self) -> None:
        self.assertEqual(run_f32_2op("v_min_f32", POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(run_f32_2op("v_min_f32", NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(run_f32_2op("v_min_f32", POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(run_f32_2op("v_min_f32", NEG_ZERO, NEG_ZERO), NEG_ZERO)

    def test_max_zero_combinations(self) -> None:
        self.assertEqual(run_f32_2op("v_max_f32", POS_ZERO, NEG_ZERO), POS_ZERO)
        self.assertEqual(run_f32_2op("v_max_f32", NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(run_f32_2op("v_max_f32", POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(run_f32_2op("v_max_f32", NEG_ZERO, NEG_ZERO), NEG_ZERO)


class TestVALUModifiers2Op(unittest.TestCase):
    """Verifies VOP3 input modifiers (neg, abs) and output clamp for 2-op instructions."""

    def test_neg_modifier(self) -> None:
        got = run_f32_2op("v_min_f32", ONE, HALF, e64=True, neg0=True)
        self.assertEqual(got, NEG_ONE)
        got = run_f32_2op("v_max_f32", ONE, HALF, e64=True, neg0=True)
        self.assertEqual(got, HALF)

    def test_abs_modifier(self) -> None:
        got = run_f32_2op("v_min_f32", NEG_TWO, ONE_HALF, e64=True, abs0=True)
        self.assertEqual(got, ONE_HALF)
        got = run_f32_2op("v_max_f32", NEG_TWO, f32_to_bits(-3.0), e64=True, abs0=True, abs1=True)
        self.assertEqual(got, THREE)

    def test_neg_abs_combination(self) -> None:
        got_min = run_f32_2op("v_min_f32", TWO, ONE, e64=True, neg0=True, abs0=True)
        self.assertEqual(got_min, NEG_TWO)
        got_max = run_f32_2op("v_max_f32", NEG_TWO, ONE, e64=True, neg0=True, abs0=True)
        self.assertEqual(got_max, ONE)

    def test_clamp_modifier(self) -> None:
        got = run_f32_2op("v_max_f32", TWO, THREE, e64=True, clamp=True)
        self.assertEqual(got, ONE)
        got = run_f32_2op("v_min_f32", NEG_ONE, NEG_TWO, e64=True, clamp=True)
        self.assertEqual(got, POS_ZERO)


class TestVALUModifiers3Op(unittest.TestCase):
    """Verifies VOP3 input modifiers (neg, abs) and output clamp for 3-op instructions."""

    def test_min3_modifiers(self) -> None:
        got = run_f32_3op("v_min3_f32", TWO, THREE, FOUR, neg0=True, abs0=True)
        self.assertEqual(got, NEG_TWO)

    def test_max3_modifiers(self) -> None:
        got = run_f32_3op("v_max3_f32", NEG_TWO, f32_to_bits(-3.0), f32_to_bits(-4.0), abs2=True)
        self.assertEqual(got, FOUR)

    def test_med3_modifiers(self) -> None:
        got = run_f32_3op("v_med3_f32", NEG_ONE, NEG_TWO, THREE, abs0=True)
        self.assertEqual(got, ONE)

    def test_clamp_modifier_3op(self) -> None:
        got = run_f32_3op("v_max3_f32", TWO, THREE, FOUR, clamp=True)
        self.assertEqual(got, ONE)
        got = run_f32_3op("v_min3_f32", NEG_ONE, NEG_TWO, f32_to_bits(-3.0), clamp=True)
        self.assertEqual(got, POS_ZERO)


class TestNaNPropagation(unittest.TestCase):
    """Deep verification of NaN propagation, payloads, and NaN suppression across 2-op and 3-op."""

    def test_nan_vs_finite_payload_preserved(self) -> None:
        # Check diverse finite floats against NaNs with different signs and mantissas
        finite_samples = [0x3E200000, 0x40490FDB, 0xC0000000, 0x00800000, 0x7F7FFFFF]
        nan_samples = [QNAN, NEG_QNAN, QNAN_PAYLOAD, QNAN_PAYLOAD_B, SNAN]

        for fin in finite_samples:
            for nan in nan_samples:
                with self.subTest(fin=hex(fin), nan=hex(nan)):
                    # 2-operand
                    self.assertEqual(run_f32_2op("v_min_f32", nan, fin), fin)
                    self.assertEqual(run_f32_2op("v_min_f32", fin, nan), fin)
                    self.assertEqual(run_f32_2op("v_max_f32", nan, fin), fin)
                    self.assertEqual(run_f32_2op("v_max_f32", fin, nan), fin)

    def test_both_nan_returns_quiet_nan(self) -> None:
        got_min = run_f32_2op("v_min_f32", QNAN_PAYLOAD, QNAN_PAYLOAD_B)
        self.assertTrue(is_nan_bits(got_min))
        got_max = run_f32_2op("v_max_f32", QNAN_PAYLOAD, QNAN_PAYLOAD_B)
        self.assertTrue(is_nan_bits(got_max))

    def test_3way_nan_combinations(self) -> None:
        # 1 NaN in 3-way: remaining 2 operands are evaluated
        self.assertEqual(run_f32_3op("v_min3_f32", QNAN, ONE, TWO), ONE)
        self.assertEqual(run_f32_3op("v_min3_f32", ONE, QNAN, TWO), ONE)
        self.assertEqual(run_f32_3op("v_min3_f32", ONE, TWO, QNAN), ONE)
        self.assertEqual(run_f32_3op("v_max3_f32", QNAN, ONE, TWO), TWO)
        self.assertEqual(run_f32_3op("v_max3_f32", ONE, QNAN, TWO), TWO)
        self.assertEqual(run_f32_3op("v_max3_f32", ONE, TWO, QNAN), TWO)
        self.assertEqual(run_f32_3op("v_med3_f32", QNAN, ONE, TWO), ONE)  # med3 with NaN falls back to min3
        self.assertEqual(run_f32_3op("v_med3_f32", ONE, QNAN, TWO), ONE)

        # 2 NaNs in 3-way: the single non-NaN operand is returned
        self.assertEqual(run_f32_3op("v_min3_f32", QNAN, QNAN_PAYLOAD, THREE), THREE)
        self.assertEqual(run_f32_3op("v_max3_f32", QNAN, THREE, QNAN_PAYLOAD), THREE)

        # 3 NaNs in 3-way: result is a NaN
        self.assertTrue(is_nan_bits(run_f32_3op("v_min3_f32", QNAN, QNAN_PAYLOAD, NEG_QNAN)))
        self.assertTrue(is_nan_bits(run_f32_3op("v_max3_f32", QNAN, QNAN_PAYLOAD, NEG_QNAN)))

    def test_nan_with_modifiers(self) -> None:
        # Negation or absolute value on NaN keeps NaN property, non-NaN operand still wins
        got_neg = run_f32_2op("v_min_f32", QNAN, ONE, e64=True, neg0=True)
        self.assertEqual(got_neg, ONE)
        got_abs = run_f32_2op("v_max_f32", QNAN, TWO, e64=True, abs0=True)
        self.assertEqual(got_abs, TWO)


class TestSubnormalBehavior(unittest.TestCase):
    """Verifies subnormal (denormal) floating point ordering, comparisons, and preservation."""

    def test_subnormal_vs_signed_zeros(self) -> None:
        # Positive subnormal is strictly greater than +0.0 and -0.0
        self.assertEqual(run_f32_2op("v_min_f32", SUBNORM_MIN, POS_ZERO), POS_ZERO)
        self.assertEqual(run_f32_2op("v_min_f32", SUBNORM_MIN, NEG_ZERO), NEG_ZERO)
        self.assertEqual(run_f32_2op("v_max_f32", SUBNORM_MIN, POS_ZERO), SUBNORM_MIN)
        self.assertEqual(run_f32_2op("v_max_f32", SUBNORM_MIN, NEG_ZERO), SUBNORM_MIN)

        # Negative subnormal is strictly less than -0.0 and +0.0
        self.assertEqual(run_f32_2op("v_min_f32", NEG_SUBNORM_MIN, NEG_ZERO), NEG_SUBNORM_MIN)
        self.assertEqual(run_f32_2op("v_min_f32", NEG_SUBNORM_MIN, POS_ZERO), NEG_SUBNORM_MIN)
        self.assertEqual(run_f32_2op("v_max_f32", NEG_SUBNORM_MIN, NEG_ZERO), NEG_ZERO)
        self.assertEqual(run_f32_2op("v_max_f32", NEG_SUBNORM_MIN, POS_ZERO), POS_ZERO)

    def test_subnormal_vs_flt_min(self) -> None:
        # Smallest positive normal FLT_MIN (2^-126) > largest subnormal SUBNORM_MAX
        self.assertEqual(run_f32_2op("v_min_f32", FLT_MIN, SUBNORM_MAX), SUBNORM_MAX)
        self.assertEqual(run_f32_2op("v_max_f32", FLT_MIN, SUBNORM_MAX), FLT_MIN)
        self.assertEqual(run_f32_2op("v_min_f32", NEG_FLT_MIN, NEG_SUBNORM_MAX), NEG_FLT_MIN)
        self.assertEqual(run_f32_2op("v_max_f32", NEG_FLT_MIN, NEG_SUBNORM_MAX), NEG_SUBNORM_MAX)

    def test_subnormal_scale_ordering(self) -> None:
        chain = [
            NEG_FLT_MIN,
            NEG_SUBNORM_MAX,
            NEG_SUBNORM_MIN,
            NEG_ZERO,
            POS_ZERO,
            SUBNORM_MIN,
            SUBNORM_MAX,
            FLT_MIN,
        ]
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            self.assertEqual(ref_min_f32(a, b), a)
            self.assertEqual(ref_max_f32(a, b), b)
            self.assertEqual(run_f32_2op("v_min_f32", a, b), a)
            self.assertEqual(run_f32_2op("v_max_f32", a, b), b)


class TestWavefrontSIMDDivergence(unittest.TestCase):
    """Executes lockstep multi-lane SIMD wavefronts to verify no cross-lane data corruption."""

    @staticmethod
    def _run_cs_divergent(wave_size: int, op: str) -> dict[int, int]:
        fssm = bytearray(0x148)
        struct.pack_into("<4I", fssm, 0x138, 0x8088, wave_size, 1, 1)

        # Diverse operands per lane:
        # lane % 4 == 0: (POS_ZERO, NEG_ZERO)
        # lane % 4 == 1: (QNAN, 1.0 + lane)
        # lane % 4 == 2: (NEG_INF, 2.0 + lane)
        # lane % 4 == 3: (SUBNORM_MIN, FLT_MIN)
        lines = [
            "v_and_b32 v5, 3, v0",  # lane % 4
            # Case 0:
            "v_mov_b32 v1, 0x00000000",
            "v_mov_b32 v2, 0x80000000",
            # Case 1:
            "v_cmp_eq_u32_e32 vcc_lo, 1, v5",
            "v_cndmask_b32_e32 v1, v1, 0x7FC00000, vcc_lo",
            "v_cvt_f32_u32 v6, v0",
            "v_add_f32_e32 v6, 0x3F800000, v6",  # 1.0 + lane
            "v_cndmask_b32_e32 v2, v2, v6, vcc_lo",
            # Case 2:
            "v_cmp_eq_u32_e32 vcc_lo, 2, v5",
            "v_cndmask_b32_e32 v1, v1, 0xFF800000, vcc_lo",
            "v_add_f32_e32 v6, 0x40000000, v6",  # 2.0 + lane
            "v_cndmask_b32_e32 v2, v2, v6, vcc_lo",
            # Case 3:
            "v_cmp_eq_u32_e32 vcc_lo, 3, v5",
            "v_cndmask_b32_e32 v1, v1, 0x00000001, vcc_lo",
            "v_cndmask_b32_e32 v2, v2, 0x00800000, vcc_lo",
            # ALU operation
            f"{op} v3, v1, v2",
            "v_lshlrev_b32 v4, 2, v0",
            "buffer_store_dword v3, v4, s[0:3], 0 offen",
            "s_endpgm",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            src = os.path.join(tmp, "disasm", "cs_wave_divergent.s")
            with open(src, "w", encoding="utf-8") as f:
                f.write("; cs_wave_divergent\n" + "\n".join(f"\t{ln}" for ln in lines) + "\n")
            with open(os.path.join(tmp, "cs_wave_divergent.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            os.mkdir(os.path.join(tmp, "fssm"))
            with open(os.path.join(tmp, "fssm", "cs_wave_divergent.fssm"), "wb") as f:
                f.write(fssm)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)

        mem: dict[int, int] = {}
        push = [0, 0, 0x1000, 0x31016FAC] + [0] * 28
        si.run_workgroup(
            spv,
            wave_size,
            32,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=push,
            memory=mem,
        )
        return mem

    def test_wave32_divergence_min(self) -> None:
        mem = self._run_cs_divergent(32, "v_min_f32")
        for i in range(32):
            mod = i % 4
            if mod == 0:
                want = ref_min_f32(POS_ZERO, NEG_ZERO)  # -0.0
            elif mod == 1:
                want = f32_to_bits(1.0 + float(i))
            elif mod == 2:
                want = NEG_INF
            else:
                want = SUBNORM_MIN
            self.assertEqual(mem.get(4 * i), want, f"Lane {i} mismatch")

    def test_wave64_divergence_max(self) -> None:
        mem = self._run_cs_divergent(64, "v_max_f32")
        for i in range(64):
            mod = i % 4
            if mod == 0:
                want = ref_max_f32(POS_ZERO, NEG_ZERO)  # +0.0
            elif mod == 1:
                want = f32_to_bits(1.0 + float(i))
            elif mod == 2:
                want = f32_to_bits(3.0 + float(i))
            else:
                want = FLT_MIN
            self.assertEqual(mem.get(4 * i), want, f"Lane {i} mismatch")


class TestNegativeControls(unittest.TestCase):
    """Negative controls ensuring non-compliant floating point behavior is rejected."""

    def test_python_default_min_zero_divergence(self) -> None:
        # Standard Python min(0.0, -0.0) yields +0.0 because 0.0 <= -0.0 evaluates to True.
        # This confirms that naive Python float comparisons violate AMD RDNA2 semantics.
        naive_python_min = min(0.0, -0.0)
        self.assertEqual(f32_to_bits(naive_python_min), POS_ZERO)
        # The reference oracle correctly yields -0.0 (0x80000000).
        self.assertEqual(ref_min_f32(POS_ZERO, NEG_ZERO), NEG_ZERO)

    def test_python_default_nan_divergence(self) -> None:
        # Standard Python min(nan, 2.0) propagates NaN.
        # IEEE 754-2008 and RDNA2 minNum instead suppress NaN in favor of the number.
        naive_python_min = min(float("nan"), 2.0)
        self.assertTrue(math.isnan(naive_python_min))
        # The reference oracle correctly returns 2.0.
        self.assertEqual(ref_min_f32(QNAN, TWO), TWO)

    def test_clamp_on_all_nan_raises_unsupported(self) -> None:
        # Clamping an all-NaN result is undefined in GLSL.std.450 / SPIR-V and raises UnsupportedConstruct.
        with self.assertRaises(si.UnsupportedConstruct):
            run_f32_2op("v_min_f32", QNAN, QNAN, e64=True, clamp=True)

    def test_med3_third_operand_nan_is_min3(self) -> None:
        # Since #2029 a NaN operand gives min3 of the operands (measured on gfx1030), not an eager FClamp on NaN.
        self.assertEqual(run_f32_3op("v_med3_f32", ONE, TWO, QNAN), ONE)

    def test_bit_exactness_rejects_sign_mismatch(self) -> None:
        # Negative control: verify that +0.0 and -0.0 are distinct 32-bit words.
        self.assertNotEqual(POS_ZERO, NEG_ZERO)
        self.assertEqual(POS_ZERO & SIGN_MASK, 0)
        self.assertEqual(NEG_ZERO & SIGN_MASK, SIGN_MASK)


if __name__ == "__main__":
    unittest.main()
