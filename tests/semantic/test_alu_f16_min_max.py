"""Semantic interpreter tests for RDNA2 half-precision floating-point min/max:
v_min_f16 and v_max_f16 (VOP2/VOP3 scalar forms),
v_pk_min_f16 and v_pk_max_f16 (VOP3P packed SIMD forms).

Features and specifications:
- IEEE 754-2008 / RDNA rules: NaN suppression (return non-NaN operand; return quiet NaN if both are NaN).
- Signed zero ordering: -0.0 < +0.0 tie-breaking.
- Packed 16-bit lane execution (v_pk_min_f16 / v_pk_max_f16 across lo and hi halves).
- Subnormal representation and gradual underflow (denormals preserved).
- Output clamp to [0.0, 1.0].
- Destination register half preservation on scalar f16 ops.
- VOP3 op_sel and input modifiers (negation, absolute value).
- Multi-lane SIMD divergence under EXEC mask.
- Deterministic seeded sweep and negative control sabotage tests.
- Standard library Python only; zero game data.
"""
from __future__ import annotations

import gc
import math
import os
import random
import re
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
from support import STUB_DXBC, mrt0_words  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
OLD = 0xA55A5AA5
VSHARP_CS = [0, 0, 0xFFFFFFFF, 0x31016FAC]

# IEEE 754-2008 binary16 bit constants
POS_ZERO = 0x0000          # +0.0
NEG_ZERO = 0x8000          # -0.0
POS_ONE = 0x3C00           # +1.0
NEG_ONE = 0xBC00           # -1.0
POS_HALF = 0x3800          # +0.5
NEG_HALF = 0xB800          # -0.5
POS_TWO = 0x4000           # +2.0
NEG_TWO = 0xC000           # -2.0
POS_THREE = 0x4200         # +3.0
NEG_THREE = 0xC200         # -3.0
POS_INF = 0x7C00           # +inf
NEG_INF = 0xFC00           # -inf
QNAN_CANONICAL = 0x7E00    # canonical quiet NaN
QNAN_ALT = 0x7E11          # quiet NaN with payload
SNAN = 0x7D00              # signaling NaN
NEG_QNAN = 0xFE22          # negative quiet NaN

# Subnormal (denormal) binary16 constants
POS_MIN_SUBNORM = 0x0001   # +2^-24 (~5.96e-8), smallest positive subnormal
POS_MAX_SUBNORM = 0x03FF   # largest positive subnormal (~6.097e-5)
NEG_MIN_SUBNORM = 0x8001   # -2^-24, smallest negative subnormal
NEG_MAX_SUBNORM = 0x83FF   # largest negative subnormal

# Normal binary16 boundaries
POS_MIN_NORMAL = 0x0400    # +2^-14 (~6.104e-5), smallest positive normal
POS_MAX_NORMAL = 0x7BFF    # +65504.0, largest finite binary16
NEG_MAX_NORMAL = 0xFBFF    # -65504.0


def f16_to_float(bits: int) -> float:
    """Decode 16-bit integer as IEEE 754 half-precision float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def float_to_f16_bits(f: float) -> int:
    """Encode float into IEEE 754 half-precision bits."""
    if math.isnan(f):
        return QNAN_CANONICAL
    try:
        return struct.unpack("<H", struct.pack("<e", f))[0]
    except OverflowError:
        return POS_INF if f > 0 else NEG_INF


def is_nan16(bits: int) -> bool:
    """Check if 16-bit integer represents a binary16 NaN."""
    return (bits & 0x7C00) == 0x7C00 and (bits & 0x03FF) != 0


def is_subnormal16(bits: int) -> bool:
    """Check if 16-bit integer represents a binary16 subnormal."""
    return (bits & 0x7C00) == 0 and (bits & 0x03FF) != 0


def half_bits(word: int, hi: int) -> int:
    """Extract low (hi=0) or high (hi=1) 16 bits of a 32-bit dword."""
    return ((word >> 16) if hi else word) & MASK16


def merge_half(res16: int, dst_hi: int, old: int = OLD) -> int:
    """Merge 16-bit result into low or high half of 32-bit word, preserving the other half."""
    res16 &= MASK16
    return ((res16 << 16) | (old & MASK16)) if dst_hi else ((old & 0xFFFF0000) | res16)


def pack2(lo: int, hi: int) -> int:
    """Pack two 16-bit half words into a single 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def ref_min_f16(a_bits: int, b_bits: int, *, clamp: bool = False) -> int:
    """Reference oracle for RDNA2 v_min_f16 conforming to IEEE 754-2008 / GLSL NMin.

    Rules:
    - NaN suppression: if one operand is NaN, return the other operand.
    - If both are NaN, return a quiet NaN.
    - Signed zero ordering: -0.0 < +0.0.
    - Clamping: if clamp=True, output clamped to [0.0, 1.0] (negative / -0.0 maps to +0.0).
    """
    a_nan = is_nan16(a_bits)
    b_nan = is_nan16(b_bits)
    if a_nan and b_nan:
        chosen_bits = QNAN_CANONICAL
    elif a_nan:
        chosen_bits = b_bits & MASK16
    elif b_nan:
        chosen_bits = a_bits & MASK16
    else:
        is_a_neg_zero = (a_bits & MASK16) == NEG_ZERO
        is_a_pos_zero = (a_bits & MASK16) == POS_ZERO
        is_b_neg_zero = (b_bits & MASK16) == NEG_ZERO
        is_b_pos_zero = (b_bits & MASK16) == POS_ZERO

        if (is_a_neg_zero and is_b_pos_zero) or (is_a_pos_zero and is_b_neg_zero):
            chosen_bits = NEG_ZERO
        elif is_a_neg_zero and is_b_neg_zero:
            chosen_bits = NEG_ZERO
        elif is_a_pos_zero and is_b_pos_zero:
            chosen_bits = POS_ZERO
        else:
            fa, fb = f16_to_float(a_bits), f16_to_float(b_bits)
            key_a = (fa, math.copysign(1.0, fa))
            key_b = (fb, math.copysign(1.0, fb))
            chosen_bits = (a_bits & MASK16) if key_a <= key_b else (b_bits & MASK16)

    if clamp:
        if is_nan16(chosen_bits):
            return POS_ZERO
        cf = f16_to_float(chosen_bits)
        if cf <= 0.0 or (chosen_bits & MASK16) == NEG_ZERO:
            return POS_ZERO
        if cf >= 1.0:
            return POS_ONE
        return float_to_f16_bits(cf)

    return chosen_bits & MASK16


def ref_max_f16(a_bits: int, b_bits: int, *, clamp: bool = False) -> int:
    """Reference oracle for RDNA2 v_max_f16 conforming to IEEE 754-2008 / GLSL NMax.

    Rules:
    - NaN suppression: if one operand is NaN, return the other operand.
    - If both are NaN, return a quiet NaN.
    - Signed zero ordering: -0.0 < +0.0.
    - Clamping: if clamp=True, output clamped to [0.0, 1.0].
    """
    a_nan = is_nan16(a_bits)
    b_nan = is_nan16(b_bits)
    if a_nan and b_nan:
        chosen_bits = QNAN_CANONICAL
    elif a_nan:
        chosen_bits = b_bits & MASK16
    elif b_nan:
        chosen_bits = a_bits & MASK16
    else:
        is_a_neg_zero = (a_bits & MASK16) == NEG_ZERO
        is_a_pos_zero = (a_bits & MASK16) == POS_ZERO
        is_b_neg_zero = (b_bits & MASK16) == NEG_ZERO
        is_b_pos_zero = (b_bits & MASK16) == POS_ZERO

        if (is_a_neg_zero and is_b_pos_zero) or (is_a_pos_zero and is_b_neg_zero):
            chosen_bits = POS_ZERO
        elif is_a_neg_zero and is_b_neg_zero:
            chosen_bits = NEG_ZERO
        elif is_a_pos_zero and is_b_pos_zero:
            chosen_bits = POS_ZERO
        else:
            fa, fb = f16_to_float(a_bits), f16_to_float(b_bits)
            key_a = (fa, math.copysign(1.0, fa))
            key_b = (fb, math.copysign(1.0, fb))
            chosen_bits = (a_bits & MASK16) if key_a >= key_b else (b_bits & MASK16)

    if clamp:
        if is_nan16(chosen_bits):
            return POS_ZERO
        cf = f16_to_float(chosen_bits)
        if cf <= 0.0 or (chosen_bits & MASK16) == NEG_ZERO:
            return POS_ZERO
        if cf >= 1.0:
            return POS_ONE
        return float_to_f16_bits(cf)

    return chosen_bits & MASK16


def ref_pk_f16(
    op: str,
    a_word: int,
    b_word: int,
    *,
    op_sel: tuple[int, int] = (0, 0),
    op_sel_hi: tuple[int, int] = (1, 1),
    neg_lo: tuple[int, int] = (0, 0),
    neg_hi: tuple[int, int] = (0, 0),
    clamp: bool = False,
) -> int:
    """Reference oracle for RDNA2 packed v_pk_min_f16 / v_pk_max_f16."""
    fn = ref_min_f16 if "min" in op else ref_max_f16

    a_lo = half_bits(a_word, op_sel[0])
    b_lo = half_bits(b_word, op_sel[1])
    if neg_lo[0]:
        a_lo ^= 0x8000
    if neg_lo[1]:
        b_lo ^= 0x8000

    a_hi = half_bits(a_word, op_sel_hi[0])
    b_hi = half_bits(b_word, op_sel_hi[1])
    if neg_hi[0]:
        a_hi ^= 0x8000
    if neg_hi[1]:
        b_hi ^= 0x8000

    res_lo = fn(a_lo, b_lo, clamp=clamp)
    res_hi = fn(a_hi, b_hi, clamp=clamp)
    return ((res_hi & MASK16) << 16) | (res_lo & MASK16)


# -----------------------------------------------------------------------------
# Translation and Execution Harness
# -----------------------------------------------------------------------------

_TMP_DIR = None
_PS_PATH = None
_CS_PATH = None
_CS_FSSM = None


def _make_fssm(rsrc2: int, group: tuple[int, int, int]) -> bytes:
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


def setUpModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH, _CS_FSSM
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    fssm_dir = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)
    _PS_PATH = os.path.join(disasm_dir, "ps_alu_f16.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_alu_f16.s")
    _CS_FSSM = os.path.join(fssm_dir, "cs_alu_f16.fssm")
    with open(os.path.join(_TMP_DIR.name, "ps_alu_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(os.path.join(_TMP_DIR.name, "cs_alu_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH, _CS_FSSM
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _PS_PATH = None
        _CS_PATH = None
        _CS_FSSM = None


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_alu_f16_min_max\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_PS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
        gc.collect()
        return spv


def run_ps(lines: list[str]) -> int:
    spv = translate_ps(lines)
    return si.run_fragment(spv)["%out_mrt0"][0]


def run_op(op_line: str, v1: int = 0, v2: int = 0, *, old: int = OLD, s4: int = 0, s5: int = 0) -> int:
    lines = [
        f"v_mov_b32 v0, {old:#x}",
        f"v_mov_b32 v1, {v1:#x}",
        f"v_mov_b32 v2, {v2:#x}",
        f"s_mov_b32 s4, {s4:#x}",
        f"s_mov_b32 s5, {s5:#x}",
        op_line,
    ]
    return run_ps(lines)


def run_cs(lines: list[str], group_size: int, memory: dict[int, int]) -> None:
    if _TMP_DIR is None:
        setUpModule()
    with open(_CS_FSSM, "wb") as f:
        f.write(_make_fssm(0x0088, (group_size, 1, 1)))
    asm = "; test_cs_alu_f16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_CS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_CS_PATH)
        gc.collect()
    si.run_workgroup(
        spv,
        group_size,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
        push_constants=VSHARP_CS + [0] * 28,
        memory=memory,
    )


# -----------------------------------------------------------------------------
# Commit 1 Tests: Reference Oracle and Scalar v_min_f16 / v_max_f16
# -----------------------------------------------------------------------------

class ReferenceOracleTests(unittest.TestCase):
    """Test IEEE 754 half-precision reference oracle rules."""

    def test_signed_zero_ordering(self):
        self.assertEqual(ref_min_f16(NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(ref_min_f16(POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(ref_max_f16(NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_max_f16(POS_ZERO, NEG_ZERO), POS_ZERO)
        self.assertEqual(ref_min_f16(NEG_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(ref_max_f16(POS_ZERO, POS_ZERO), POS_ZERO)

    def test_subnormal_ordering_and_gradual_underflow(self):
        self.assertEqual(ref_min_f16(POS_MIN_SUBNORM, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_max_f16(POS_MIN_SUBNORM, POS_ZERO), POS_MIN_SUBNORM)
        self.assertEqual(ref_min_f16(POS_MAX_SUBNORM, POS_MIN_NORMAL), POS_MAX_SUBNORM)
        self.assertEqual(ref_max_f16(POS_MAX_SUBNORM, POS_MIN_NORMAL), POS_MIN_NORMAL)
        self.assertEqual(ref_min_f16(NEG_MIN_SUBNORM, NEG_ZERO), NEG_MIN_SUBNORM)
        self.assertEqual(ref_max_f16(NEG_MIN_SUBNORM, NEG_ZERO), NEG_ZERO)

    def test_infinities(self):
        self.assertEqual(ref_min_f16(POS_INF, POS_MAX_NORMAL), POS_MAX_NORMAL)
        self.assertEqual(ref_max_f16(POS_INF, POS_MAX_NORMAL), POS_INF)
        self.assertEqual(ref_min_f16(NEG_INF, NEG_MAX_NORMAL), NEG_INF)
        self.assertEqual(ref_max_f16(NEG_INF, NEG_MAX_NORMAL), NEG_MAX_NORMAL)
        self.assertEqual(ref_min_f16(NEG_INF, POS_INF), NEG_INF)
        self.assertEqual(ref_max_f16(NEG_INF, POS_INF), POS_INF)

    def test_clamp_oracle(self):
        self.assertEqual(ref_min_f16(POS_TWO, POS_THREE, clamp=True), POS_ONE)
        self.assertEqual(ref_min_f16(NEG_TWO, NEG_ONE, clamp=True), POS_ZERO)
        self.assertEqual(ref_max_f16(NEG_HALF, NEG_ONE, clamp=True), POS_ZERO)
        self.assertEqual(ref_max_f16(POS_HALF, POS_HALF, clamp=True), POS_HALF)
        self.assertEqual(ref_min_f16(POS_HALF, POS_ONE, clamp=True), POS_HALF)

    def test_roundtrip_binary16(self):
        test_patterns = [
            POS_ZERO, NEG_ZERO, POS_ONE, NEG_ONE, POS_HALF, NEG_HALF,
            POS_TWO, NEG_TWO, POS_INF, NEG_INF, POS_MIN_SUBNORM,
            POS_MAX_SUBNORM, NEG_MIN_SUBNORM, POS_MIN_NORMAL, POS_MAX_NORMAL,
        ]
        for bits in test_patterns:
            flt = f16_to_float(bits)
            enc = float_to_f16_bits(flt)
            self.assertEqual(enc, bits, f"Pattern {bits:#06x} failed roundtrip, got {enc:#06x}")


class ScalarMinMaxF16Tests(unittest.TestCase):
    """End-to-end interpreter tests for scalar v_min_f16 and v_max_f16."""

    def test_scalar_min_f16_basic(self):
        got = run_op("v_min_f16_e32 v0, v1, v2", v1=POS_ONE, v2=POS_TWO)
        self.assertEqual(got, merge_half(POS_ONE, dst_hi=0))

        got = run_op("v_min_f16_e32 v0, v1, v2", v1=POS_TWO, v2=NEG_THREE)
        self.assertEqual(got, merge_half(NEG_THREE, dst_hi=0))

    def test_scalar_max_f16_basic(self):
        got = run_op("v_max_f16_e32 v0, v1, v2", v1=POS_ONE, v2=POS_TWO)
        self.assertEqual(got, merge_half(POS_TWO, dst_hi=0))

        got = run_op("v_max_f16_e32 v0, v1, v2", v1=NEG_ONE, v2=NEG_TWO)
        self.assertEqual(got, merge_half(NEG_ONE, dst_hi=0))

    def test_destination_half_preservation(self):
        got_lo = run_op("v_min_f16_e64 v0, v1, v2 op_sel:[0,0,0]", v1=POS_HALF, v2=POS_ONE, old=OLD)
        self.assertEqual(got_lo, merge_half(POS_HALF, dst_hi=0, old=OLD))
        self.assertEqual(got_lo & 0xFFFF0000, OLD & 0xFFFF0000)

        got_hi = run_op("v_max_f16_e64 v0, v1, v2 op_sel:[0,0,1]", v1=POS_HALF, v2=POS_ONE, old=OLD)
        self.assertEqual(got_hi, merge_half(POS_ONE, dst_hi=1, old=OLD))
        self.assertEqual(got_hi & 0x0000FFFF, OLD & 0x0000FFFF)

    def test_subnormal_handling_scalar(self):
        got_min = run_op("v_min_f16_e32 v0, v1, v2", v1=POS_MIN_SUBNORM, v2=POS_ZERO)
        self.assertEqual(got_min, merge_half(POS_ZERO, dst_hi=0))

        got_max = run_op("v_max_f16_e32 v0, v1, v2", v1=POS_MIN_SUBNORM, v2=POS_ZERO)
        self.assertEqual(got_max, merge_half(POS_MIN_SUBNORM, dst_hi=0))

        got_min2 = run_op("v_min_f16_e32 v0, v1, v2", v1=POS_MAX_SUBNORM, v2=POS_MIN_NORMAL)
        self.assertEqual(got_min2, merge_half(POS_MAX_SUBNORM, dst_hi=0))

        got_max2 = run_op("v_max_f16_e32 v0, v1, v2", v1=POS_MAX_SUBNORM, v2=POS_MIN_NORMAL)
        self.assertEqual(got_max2, merge_half(POS_MIN_NORMAL, dst_hi=0))

    def test_output_clamp_scalar(self):
        got = run_op("v_min_f16_e64 v0, v1, v2 clamp", v1=POS_TWO, v2=POS_THREE)
        self.assertEqual(got, merge_half(POS_ONE, dst_hi=0))

        got2 = run_op("v_max_f16_e64 v0, v1, v2 clamp", v1=NEG_TWO, v2=NEG_ONE)
        self.assertEqual(got2, merge_half(POS_ZERO, dst_hi=0))

        got3 = run_op("v_min_f16_e64 v0, v1, v2 clamp", v1=POS_HALF, v2=POS_ONE)
        self.assertEqual(got3, merge_half(POS_HALF, dst_hi=0))

    def test_source_modifiers_neg_abs(self):
        got = run_op("v_min_f16_e64 v0, -v1, v2", v1=NEG_TWO, v2=POS_ONE)
        self.assertEqual(got, merge_half(POS_ONE, dst_hi=0))

        got2 = run_op("v_min_f16_e64 v0, |v1|, v2", v1=NEG_TWO, v2=POS_HALF)
        self.assertEqual(got2, merge_half(POS_HALF, dst_hi=0))

        got3 = run_op("v_max_f16_e64 v0, -|v1|, v2", v1=POS_ONE, v2=NEG_TWO)
        self.assertEqual(got3, merge_half(NEG_ONE, dst_hi=0))

    def test_inline_constants_and_sgprs(self):
        got = run_op("v_min_f16_e64 v0, 1.0, v2", v2=POS_TWO)
        self.assertEqual(got, merge_half(POS_ONE, dst_hi=0))

        got2 = run_op("v_max_f16_e64 v0, 0.0, v2", v2=NEG_ONE)
        self.assertEqual(got2, merge_half(POS_ZERO, dst_hi=0))

        got3 = run_op("v_min_f16_e64 v0, s4, v2", s4=POS_TWO, v2=POS_HALF)
        self.assertEqual(got3, merge_half(POS_HALF, dst_hi=0))

        got4 = run_op("v_max_f16_e64 v0, v1, s5", v1=POS_ONE, s5=NEG_HALF)
        self.assertEqual(got4, merge_half(POS_ONE, dst_hi=0))


# -----------------------------------------------------------------------------
# Commit 2 Tests: Packed Half-Precision v_pk_min_f16 / v_pk_max_f16 & SIMD EXEC
# -----------------------------------------------------------------------------

class PackedReferenceOracleTests(unittest.TestCase):
    """Test packed reference oracle logic."""

    def test_packed_reference_basic(self):
        v1 = pack2(POS_ONE, POS_HALF)
        v2 = pack2(POS_TWO, POS_THREE)
        self.assertEqual(ref_pk_f16("v_pk_min_f16", v1, v2), pack2(POS_ONE, POS_HALF))
        self.assertEqual(ref_pk_f16("v_pk_max_f16", v1, v2), pack2(POS_TWO, POS_THREE))

    def test_packed_reference_modifiers(self):
        v1 = pack2(POS_ONE, POS_HALF)
        v2 = pack2(POS_TWO, POS_THREE)
        got = ref_pk_f16("v_pk_min_f16", v1, v2, op_sel=(0, 1), op_sel_hi=(1, 0))
        self.assertEqual(got, pack2(POS_ONE, POS_HALF))

        got_neg = ref_pk_f16("v_pk_min_f16", v1, v2, neg_lo=(1, 0))
        self.assertEqual(got_neg, pack2(NEG_ONE, POS_HALF))


class PackedMinMaxF16Tests(unittest.TestCase):
    """End-to-end tests for packed v_pk_min_f16 and v_pk_max_f16 instructions."""

    def test_packed_min_max_basic(self):
        v1 = pack2(POS_TWO, POS_ONE)
        v2 = pack2(POS_HALF, POS_THREE)

        got_min = run_op("v_pk_min_f16 v0, v1, v2", v1=v1, v2=v2)
        self.assertEqual(got_min, pack2(POS_HALF, POS_ONE))

        got_max = run_op("v_pk_max_f16 v0, v1, v2", v1=v1, v2=v2)
        self.assertEqual(got_max, pack2(POS_TWO, POS_THREE))

    def test_packed_subnormal_lanes(self):
        v1 = pack2(POS_MIN_SUBNORM, POS_MAX_SUBNORM)
        v2 = pack2(POS_ZERO, POS_MIN_NORMAL)

        got_min = run_op("v_pk_min_f16 v0, v1, v2", v1=v1, v2=v2)
        self.assertEqual(got_min, pack2(POS_ZERO, POS_MAX_SUBNORM))

        got_max = run_op("v_pk_max_f16 v0, v1, v2", v1=v1, v2=v2)
        self.assertEqual(got_max, pack2(POS_MIN_SUBNORM, POS_MIN_NORMAL))

    def test_packed_clamp(self):
        v1 = pack2(NEG_TWO, POS_THREE)
        v2 = pack2(NEG_HALF, POS_TWO)
        got = run_op("v_pk_min_f16 v0, v1, v2 clamp", v1=v1, v2=v2)
        self.assertEqual(got, pack2(POS_ZERO, POS_ONE))

        got2 = run_op("v_pk_max_f16 v0, v1, v2 clamp", v1=pack2(NEG_TWO, POS_HALF), v2=pack2(NEG_ONE, POS_ZERO))
        self.assertEqual(got2, pack2(POS_ZERO, POS_HALF))

    def test_packed_opsel_swizzle(self):
        v1 = pack2(POS_TWO, POS_ONE)
        v2 = pack2(POS_HALF, POS_THREE)
        got = run_op("v_pk_min_f16 v0, v1, v2 op_sel:[0,1] op_sel_hi:[1,0]", v1=v1, v2=v2)
        self.assertEqual(got, pack2(POS_TWO, POS_HALF))

    def test_packed_negation_modifiers(self):
        v1 = pack2(POS_ONE, POS_TWO)
        v2 = pack2(POS_HALF, POS_THREE)
        got = run_op("v_pk_min_f16 v0, v1, v2 neg_lo:[1,0] neg_hi:[0,1]", v1=v1, v2=v2)
        self.assertEqual(got, pack2(NEG_ONE, NEG_THREE))

    def test_packed_sgpr_operands(self):
        v1 = pack2(POS_TWO, POS_ONE)
        s4 = pack2(POS_HALF, POS_THREE)
        got = run_op("v_pk_min_f16 v0, v1, s4", v1=v1, s4=s4)
        self.assertEqual(got, pack2(POS_HALF, POS_ONE))


class SimdExecMaskDivergenceTests(unittest.TestCase):
    """Test multi-lane SIMD divergence and execution under partial EXEC masks."""

    def test_single_lane_exec_mask_toggle(self):
        got_masked = run_ps([
            f"v_mov_b32 v0, {OLD:#x}",
            f"v_mov_b32 v1, {POS_ONE:#x}",
            f"v_mov_b32 v2, {POS_TWO:#x}",
            "s_mov_b32 exec_lo, 0",
            "v_min_f16_e32 v0, v1, v2",
            "s_mov_b32 exec_lo, 1",
        ])
        self.assertEqual(got_masked, OLD)

        got_pk_masked = run_ps([
            f"v_mov_b32 v0, {OLD:#x}",
            f"v_mov_b32 v1, {pack2(POS_ONE, POS_TWO):#x}",
            f"v_mov_b32 v2, {pack2(POS_THREE, POS_HALF):#x}",
            "s_mov_b32 exec_lo, 0",
            "v_pk_min_f16 v0, v1, v2",
            "s_mov_b32 exec_lo, 1",
        ])
        self.assertEqual(got_pk_masked, OLD)

    def test_multilane_workgroup_divergence(self):
        lane_v1 = [
            pack2(POS_TWO, POS_ONE),
            pack2(POS_THREE, POS_TWO),
            pack2(POS_HALF, POS_TWO),
            pack2(NEG_ONE, POS_THREE),
            pack2(POS_MIN_SUBNORM, POS_ONE),
            pack2(POS_ONE, POS_MAX_NORMAL),
            pack2(NEG_TWO, NEG_THREE),
            pack2(POS_HALF, POS_HALF),
        ]
        lane_v2 = [
            pack2(POS_HALF, POS_THREE),
            pack2(POS_ONE, POS_ONE),
            pack2(POS_ONE, POS_HALF),
            pack2(NEG_TWO, POS_ONE),
            pack2(POS_ZERO, POS_TWO),
            pack2(POS_TWO, POS_INF),
            pack2(NEG_ONE, NEG_ONE),
            pack2(POS_THREE, POS_ONE),
        ]
        lane_olds = [0xD00D0000 + i for i in range(8)]

        mem = {
            0x100 + 4 * i: lane_v1[i] for i in range(8)
        }
        mem.update({0x200 + 4 * i: lane_v2[i] for i in range(8)})
        mem.update({0x300 + 4 * i: lane_olds[i] for i in range(8)})

        run_cs([
            "v_lshlrev_b32_e32 v5, 2, v0",
            "v_add_nc_u32_e32 v6, 0x100, v5",
            "v_add_nc_u32_e32 v7, 0x200, v5",
            "v_add_nc_u32_e32 v8, 0x300, v5",
            "buffer_load_dword v1, v6, s[0:3], 0 offen",
            "buffer_load_dword v2, v7, s[0:3], 0 offen",
            "buffer_load_dword v3, v8, s[0:3], 0 offen",
            "v_and_b32_e32 v4, 1, v0",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v4",
            "s_and_saveexec_b32 s6, vcc_lo",
            "v_pk_min_f16 v3, v1, v2",
            "s_mov_b32 exec_lo, s6",
            "buffer_store_dword v3, v5, s[0:3], 0 offen",
        ], group_size=8, memory=mem)

        for lane in range(8):
            got = mem.get(4 * lane)
            if lane % 2 == 1:
                self.assertEqual(got, lane_olds[lane], f"Lane {lane} was masked out but was overwritten!")
            else:
                want = ref_pk_f16("v_pk_min_f16", lane_v1[lane], lane_v2[lane])
                self.assertEqual(got, want, f"Lane {lane} active result mismatch: got {got:#x}, want {want:#x}")


# -----------------------------------------------------------------------------
# Commit 3 Tests: NaN Suppression, Signed Zero Tie-Breaking, Negative Controls
# -----------------------------------------------------------------------------

class NanSuppressionTests(unittest.TestCase):
    """Test IEEE 754-2008 / RDNA NaN suppression rules."""

    def test_scalar_nan_suppression_single_nan(self):
        # min(NaN, x) = x and min(x, NaN) = x
        for nan_val in (QNAN_CANONICAL, QNAN_ALT, SNAN, NEG_QNAN):
            for num_val in (POS_ONE, NEG_TWO, POS_ZERO, NEG_ZERO, POS_MIN_SUBNORM, POS_INF):
                with self.subTest(nan=hex(nan_val), num=hex(num_val)):
                    # min
                    got_min_1 = run_op("v_min_f16_e32 v0, v1, v2", v1=nan_val, v2=num_val)
                    self.assertEqual(got_min_1, merge_half(num_val, dst_hi=0))
                    got_min_2 = run_op("v_min_f16_e32 v0, v1, v2", v1=num_val, v2=nan_val)
                    self.assertEqual(got_min_2, merge_half(num_val, dst_hi=0))

                    # max
                    got_max_1 = run_op("v_max_f16_e32 v0, v1, v2", v1=nan_val, v2=num_val)
                    self.assertEqual(got_max_1, merge_half(num_val, dst_hi=0))
                    got_max_2 = run_op("v_max_f16_e32 v0, v1, v2", v1=num_val, v2=nan_val)
                    self.assertEqual(got_max_2, merge_half(num_val, dst_hi=0))

    def test_scalar_both_nan_returns_quiet_nan(self):
        got_min = run_op("v_min_f16_e32 v0, v1, v2", v1=QNAN_CANONICAL, v2=QNAN_ALT)
        self.assertTrue(is_nan16(got_min & 0xFFFF))

        got_max = run_op("v_max_f16_e32 v0, v1, v2", v1=QNAN_ALT, v2=NEG_QNAN)
        self.assertTrue(is_nan16(got_max & 0xFFFF))

    def test_packed_nan_suppression(self):
        # Mixed lane NaN:
        # lo: min(QNAN_ALT, 1.0) = 1.0
        # hi: min(2.0, QNAN_CANONICAL) = 2.0
        v1 = pack2(QNAN_ALT, POS_TWO)
        v2 = pack2(POS_ONE, QNAN_CANONICAL)
        got = run_op("v_pk_min_f16 v0, v1, v2", v1=v1, v2=v2)
        self.assertEqual(got, pack2(POS_ONE, POS_TWO))

        # Packed max with NaNs:
        # lo: max(-1.0, NEG_QNAN) = -1.0
        # hi: max(QNAN_ALT, 0.5) = 0.5
        v3 = pack2(NEG_ONE, QNAN_ALT)
        v4 = pack2(NEG_QNAN, POS_HALF)
        got2 = run_op("v_pk_max_f16 v0, v1, v2", v1=v3, v2=v4)
        self.assertEqual(got2, pack2(NEG_ONE, POS_HALF))


class SignedZeroTieBreakingTests(unittest.TestCase):
    """Test IEEE 754-2008 / RDNA signed zero tie-breaking: -0.0 < +0.0."""

    def test_scalar_signed_zero_min(self):
        # min(-0.0, +0.0) == -0.0 (0x8000)
        got1 = run_op("v_min_f16_e32 v0, v1, v2", v1=NEG_ZERO, v2=POS_ZERO)
        self.assertEqual(got1, merge_half(NEG_ZERO, dst_hi=0))

        # min(+0.0, -0.0) == -0.0 (0x8000)
        got2 = run_op("v_min_f16_e32 v0, v1, v2", v1=POS_ZERO, v2=NEG_ZERO)
        self.assertEqual(got2, merge_half(NEG_ZERO, dst_hi=0))

        # min(-0.0, -0.0) == -0.0
        got3 = run_op("v_min_f16_e32 v0, v1, v2", v1=NEG_ZERO, v2=NEG_ZERO)
        self.assertEqual(got3, merge_half(NEG_ZERO, dst_hi=0))

    def test_scalar_signed_zero_max(self):
        # max(-0.0, +0.0) == +0.0 (0x0000)
        got1 = run_op("v_max_f16_e32 v0, v1, v2", v1=NEG_ZERO, v2=POS_ZERO)
        self.assertEqual(got1, merge_half(POS_ZERO, dst_hi=0))

        # max(+0.0, -0.0) == +0.0 (0x0000)
        got2 = run_op("v_max_f16_e32 v0, v1, v2", v1=POS_ZERO, v2=NEG_ZERO)
        self.assertEqual(got2, merge_half(POS_ZERO, dst_hi=0))

        # max(+0.0, +0.0) == +0.0
        got3 = run_op("v_max_f16_e32 v0, v1, v2", v1=POS_ZERO, v2=POS_ZERO)
        self.assertEqual(got3, merge_half(POS_ZERO, dst_hi=0))

    def test_packed_signed_zero_tie_breaking(self):
        # lo: min(-0.0, +0.0) = -0.0 (0x8000)
        # hi: max(-0.0, +0.0) = +0.0 (0x0000)
        # result: (0x0000 << 16) | 0x8000 = 0x00008000
        v1 = pack2(NEG_ZERO, NEG_ZERO)
        v2 = pack2(POS_ZERO, POS_ZERO)
        got = run_op("v_pk_min_f16 v0, v1, v2", v1=v1, v2=v2)
        # For v_pk_min_f16: both lanes compute min, so both give -0.0 -> 0x80008000
        self.assertEqual(got, pack2(NEG_ZERO, NEG_ZERO))

        # For v_pk_max_f16: both lanes compute max, so both give +0.0 -> 0x00000000
        got_max = run_op("v_pk_max_f16 v0, v1, v2", v1=v1, v2=v2)
        self.assertEqual(got_max, pack2(POS_ZERO, POS_ZERO))


class DeterministicSweepTests(unittest.TestCase):
    """Deterministic pseudorandom sweep comparing simulated execution against oracle."""

    def test_seeded_sweep(self):
        rng = random.Random(0x7F16)
        pool = [
            POS_ZERO, NEG_ZERO, POS_ONE, NEG_ONE, POS_HALF, NEG_HALF,
            POS_TWO, NEG_TWO, POS_THREE, NEG_THREE, POS_INF, NEG_INF,
            POS_MIN_SUBNORM, POS_MAX_SUBNORM, NEG_MIN_SUBNORM,
            POS_MIN_NORMAL, POS_MAX_NORMAL, NEG_MAX_NORMAL,
            QNAN_CANONICAL, QNAN_ALT,
        ]

        for _ in range(30):
            a = rng.choice(pool)
            b = rng.choice(pool)

            # Scalar min test
            got_min = run_op("v_min_f16_e32 v0, v1, v2", v1=a, v2=b) & 0xFFFF
            want_min = ref_min_f16(a, b)
            if is_nan16(want_min):
                self.assertTrue(is_nan16(got_min))
            else:
                self.assertEqual(got_min, want_min, f"min({a:#x}, {b:#x}) mismatch")

            # Scalar max test
            got_max = run_op("v_max_f16_e32 v0, v1, v2", v1=a, v2=b) & 0xFFFF
            want_max = ref_max_f16(a, b)
            if is_nan16(want_max):
                self.assertTrue(is_nan16(got_max))
            else:
                self.assertEqual(got_max, want_max, f"max({a:#x}, {b:#x}) mismatch")


class NegativeControlsAndSabotageTests(unittest.TestCase):
    """Negative controls and mutation detection tests verifying test suite vigilance."""

    def test_opcode_sabotage_fails(self):
        # Invert NMin to NMax in translated SPIR-V: must detect divergence!
        spv = translate_ps([
            f"v_mov_b32 v1, {POS_ONE:#x}",
            f"v_mov_b32 v2, {POS_TWO:#x}",
            "v_min_f16_e32 v0, v1, v2",
        ])
        self.assertIn("NMin", spv)
        bad_spv, count = re.subn(r"NMin", "NMax", spv)
        self.assertGreaterEqual(count, 1)

        out_good = si.run_fragment(spv)["%out_mrt0"][0] & 0xFFFF
        out_bad = si.run_fragment(bad_spv)["%out_mrt0"][0] & 0xFFFF

        self.assertEqual(out_good, POS_ONE)
        self.assertEqual(out_bad, POS_TWO)
        self.assertNotEqual(out_good, out_bad)

    def test_clamp_omission_sabotage_fails(self):
        # Removing FClamp from clamped v_min_f16 translation must fail clamp expectation
        spv = translate_ps([
            f"v_mov_b32 v1, {POS_TWO:#x}",
            f"v_mov_b32 v2, {POS_THREE:#x}",
            "v_min_f16_e64 v0, v1, v2 clamp",
        ])
        self.assertIn("FClamp", spv)
        # Sabotage FClamp by bypassing it with OpCopyObject of its first argument
        bad_spv, count = re.subn(r"OpExtInst %float %glsl FClamp (%\w+) (%\w+) (%\w+)", r"OpCopyObject %float \1", spv)
        self.assertGreaterEqual(count, 1)

        out_good = si.run_fragment(spv)["%out_mrt0"][0] & 0xFFFF
        out_bad = si.run_fragment(bad_spv)["%out_mrt0"][0] & 0xFFFF

        self.assertEqual(out_good, POS_ONE)
        self.assertEqual(out_bad, POS_TWO)
        self.assertNotEqual(out_good, out_bad)

    def test_signed_zero_tie_break_sabotage_fails(self):
        # Tampering with signed zero ordering in oracle or hardware asserts
        # min(-0.0, +0.0) must be -0.0 (0x8000), not +0.0 (0x0000)
        got = run_op("v_min_f16_e32 v0, v1, v2", v1=NEG_ZERO, v2=POS_ZERO) & 0xFFFF
        self.assertEqual(got, NEG_ZERO)
        self.assertNotEqual(got, POS_ZERO)


if __name__ == "__main__":
    unittest.main()
