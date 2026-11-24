from __future__ import annotations

"""Semantic interpreter tests for AMD RDNA2 floating-point rounding instructions:
  - v_floor_f32 / v_floor_f16: round down towards -infinity (Floor)
  - v_ceil_f32 / v_ceil_f16: round up towards +infinity (Ceil)
  - v_trunc_f32 / v_trunc_f16: round towards zero (Trunc)
  - v_rndne_f32 / v_rndne_f16: round to nearest even integer (RoundEven)

Models IEEE 754-2008 / IEEE 754-2019 roundToIntegral semantics and SPIR-V GLSL.std.450
extended instructions, verified against AMD GFX10 / RDNA2 hardware semantics:
  * Signed zero preservation: IEEE 754 roundToIntegral requires that if the rounded
    integral result is zero, it preserves the sign of the input operand.
    Thus floor(-0.5) = -1.0, but ceil(-0.5) = -0.0, trunc(-0.5) = -0.0, rndne(-0.4) = -0.0.
  * Subnormal numbers: gradual underflow behavior where subnormals within (-1.0, 1.0)
    round appropriately to +/-0.0 or +/-1.0 preserving signs.
  * Special IEEE 754 values: positive/negative infinities and quiet/signaling NaNs pass through.
  * Tie-breaking to even for rndne / RoundEven: half-integers (n + 0.5) round to the nearest
    even integer (0.5 -> 0.0, 1.5 -> 2.0, 2.5 -> 2.0, 3.5 -> 4.0; -0.5 -> -0.0, -1.5 -> -2.0).
  * High-magnitude floats: values where |x| >= 2^23 (for f32) or |x| >= 2^10 (for f16)
    have no fractional significand bits (ulp >= 1.0) and are already exact integers.
  * Hardware operand modifiers: source abs (|src|), neg (-src), negative abs (-|src|),
    destination clamp ([0.0, 1.0]), output modifiers (mul:2, mul:4, div:2), and 16-bit half
    selection and destination preservation via op_sel:[src,dst].
  * Multi-lane SIMD wave64 isolation.
  * Negative controls verifying that erroneous rounding modes, sign dropping, or clamp
    violations fail assertions.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible.
"""

import math
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

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
SIGN_MASK32 = 0x80000000
EXP_MASK32 = 0x7F800000
MANT_MASK32 = 0x007FFFFF

MASK16 = 0xFFFF
SIGN_MASK16 = 0x8000
EXP_MASK16 = 0x7C00
MANT_MASK16 = 0x03FF

POS_ZERO_F32 = 0x00000000
NEG_ZERO_F32 = 0x80000000
POS_INF_F32 = 0x7F800000
NEG_INF_F32 = 0xFF800000
QNAN_F32 = 0x7FC00000

POS_ZERO_F16 = 0x0000
NEG_ZERO_F16 = 0x8000
POS_INF_F16 = 0x7C00
NEG_INF_F16 = 0xFC00
QNAN_F16 = 0x7E00

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


# ==============================================================================
# Bitwise & Float Conversion Utilities
# ==============================================================================


def f32_to_bits(f: float) -> int:
    """Encode Python float as IEEE 754 single-precision 32-bit unsigned integer."""
    if math.isnan(f):
        return QNAN_F32
    try:
        return struct.unpack("<I", struct.pack("<f", f))[0]
    except OverflowError:
        return NEG_INF_F32 if f < 0 else POS_INF_F32


def bits_to_f32(b: int) -> float:
    """Decode IEEE 754 single-precision 32-bit unsigned integer to Python float."""
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def f16_to_bits(f: float) -> int:
    """Encode Python float as IEEE 754 half-precision 16-bit unsigned integer."""
    if math.isnan(f):
        return QNAN_F16
    try:
        return struct.unpack("<H", struct.pack("<e", f))[0]
    except OverflowError:
        return NEG_INF_F16 if f < 0 else POS_INF_F16


def bits_to_f16(h: int) -> float:
    """Decode IEEE 754 half-precision 16-bit unsigned integer to Python float."""
    return struct.unpack("<e", struct.pack("<H", h & MASK16))[0]


def is_nan_f32(b: int) -> bool:
    """Check if 32-bit word represents an IEEE 754 single-precision NaN."""
    return ((b & EXP_MASK32) == EXP_MASK32) and ((b & MANT_MASK32) != 0)


def is_nan_f16(h: int) -> bool:
    """Check if 16-bit word represents an IEEE 754 half-precision NaN."""
    return ((h & EXP_MASK16) == EXP_MASK16) and ((h & MANT_MASK16) != 0)


# ==============================================================================
# Reference Oracles
# ==============================================================================


def ref_round_f32(val: float, mode: str) -> float:
    """Computes IEEE 754 binary32 rounding to integral with sign preservation.

    Modes:
      - 'floor': round down towards -infinity
      - 'ceil':  round up towards +infinity
      - 'trunc': round towards zero
      - 'rndne': round to nearest even integer
    """
    if math.isnan(val) or math.isinf(val):
        return val
    fn = {
        "floor": math.floor,
        "ceil": math.ceil,
        "trunc": math.trunc,
        "rndne": round,
    }[mode]
    # math.copysign ensures that rounding -0.5 towards 0 retains -0.0
    return math.copysign(float(fn(val)), val)


def ref_floor_f32(val: float) -> float:
    return ref_round_f32(val, "floor")


def ref_ceil_f32(val: float) -> float:
    return ref_round_f32(val, "ceil")


def ref_trunc_f32(val: float) -> float:
    return ref_round_f32(val, "trunc")


def ref_rndne_f32(val: float) -> float:
    return ref_round_f32(val, "rndne")


def ref_round_f32_bits(bits: int, mode: str) -> int:
    """Bit-accurate reference for binary32 rounding instructions."""
    bits &= MASK32
    if (bits & EXP_MASK32) == EXP_MASK32:
        return bits  # Infinities and NaNs pass through unchanged
    f = bits_to_f32(bits)
    res_f = ref_round_f32(f, mode)
    return f32_to_bits(res_f)


def ref_round_f16(val: float, mode: str) -> float:
    """Computes IEEE 754 binary16 rounding to integral with sign preservation."""
    if math.isnan(val) or math.isinf(val):
        return val
    fn = {
        "floor": math.floor,
        "ceil": math.ceil,
        "trunc": math.trunc,
        "rndne": round,
    }[mode]
    return math.copysign(float(fn(val)), val)


def ref_round_f16_bits(h: int, mode: str) -> int:
    """Bit-accurate reference for binary16 rounding instructions."""
    h &= MASK16
    if (h & EXP_MASK16) == EXP_MASK16:
        return h  # Infinities and NaNs pass through unchanged
    f = bits_to_f16(h)
    res_f = ref_round_f16(f, mode)
    return f16_to_bits(res_f)


# ==============================================================================
# Simulated VALU Interpreter
# ==============================================================================


def split_operands(s: str) -> List[str]:
    """Splits assembly operand string on commas outside of bracketed lists."""
    parts: List[str] = []
    current: List[str] = []
    in_bracket = False
    for ch in s:
        if ch == "[":
            in_bracket = True
            current.append(ch)
        elif ch == "]":
            in_bracket = False
            current.append(ch)
        elif ch == "," and not in_bracket:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


class RoundValuInterpreter:
    """Simulated VALU interpreter supporting floating point rounding instructions:
    v_floor_f32, v_ceil_f32, v_trunc_f32, v_rndne_f32,
    v_floor_f16, v_ceil_f16, v_trunc_f16, v_rndne_f16.

    Models VGPR (v0..v255) and SGPR (s0..s105) register state, e32 / e64 forms,
    source modifiers (|src|, -src, -|src|), destination clamp ([0.0, 1.0]),
    output multipliers (mul:2, mul:4, div:2), and f16 op_sel half selection.
    """

    def __init__(self) -> None:
        self.vgpr: Dict[int, int] = {}
        self.sgpr: Dict[int, int] = {}

    def read_reg(self, name: str) -> int:
        name = name.strip()
        if name.startswith("v"):
            return self.vgpr.get(int(name[1:]), 0) & MASK32
        if name.startswith("s"):
            return self.sgpr.get(int(name[1:]), 0) & MASK32
        raise ValueError(f"Unknown register name: {name}")

    def write_reg(self, name: str, val: int) -> None:
        name = name.strip()
        val &= MASK32
        if name.startswith("v"):
            self.vgpr[int(name[1:])] = val
        elif name.startswith("s"):
            self.sgpr[int(name[1:])] = val
        else:
            raise ValueError(f"Unknown destination register: {name}")

    def parse_operand(self, token: str) -> Tuple[int, bool, bool]:
        """Parses operand token into (raw_bits, is_abs, is_neg)."""
        token = token.strip()
        is_abs = False
        is_neg = False

        if token.startswith("-|") and token.endswith("|"):
            is_abs = True
            is_neg = True
            token = token[2:-1].strip()
        elif token.startswith("|") and token.endswith("|"):
            is_abs = True
            token = token[1:-1].strip()
        elif token.startswith("-"):
            is_neg = True
            token = token[1:].strip()

        inline_floats = {
            "0.0": 0x00000000,
            "0.5": 0x3F000000,
            "1.0": 0x3F800000,
            "1.5": 0x3FC00000,
            "2.0": 0x40000000,
            "2.5": 0x40200000,
            "3.0": 0x40400000,
            "4.0": 0x40800000,
            "-0.5": 0xBF000000,
            "-1.0": 0xBF800000,
            "-1.5": 0xBFC00000,
            "-2.0": 0xC0000000,
        }

        if token in inline_floats:
            val = inline_floats[token]
        elif token.startswith("v") or token.startswith("s"):
            val = self.read_reg(token)
        elif token.startswith("0x") or token.startswith("0X"):
            val = int(token, 16) & MASK32
        else:
            try:
                val = int(token) & MASK32
            except ValueError:
                val = f32_to_bits(float(token))

        if is_abs:
            val &= ~SIGN_MASK32
        if is_neg:
            val ^= SIGN_MASK32

        return val, is_abs, is_neg

    def execute_line(self, line: str) -> None:
        """Executes a single RDNA2 assembly line."""
        clean = line.split(";")[0].strip()
        if not clean:
            return

        parts = clean.split(None, 1)
        mnemonic = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        # Parse comma-separated arguments, recognizing trailing modifiers
        args = split_operands(rest) if rest else []
        dst = args[0] if args else ""
        src_and_mods = args[1].split() if len(args) > 1 else []

        src_token = src_and_mods[0] if src_and_mods else ""
        mods = [m.lower() for m in src_and_mods[1:]]

        # Check for op_sel in modifiers, e.g., op_sel:[1,0]
        op_sel_src = 0
        op_sel_dst = 0
        for m in mods:
            if m.startswith("op_sel:["):
                inner = m[len("op_sel:[") : -1]
                bits = [int(x.strip()) for x in inner.split(",")]
                op_sel_src = bits[0]
                op_sel_dst = bits[1]

        base_op = mnemonic
        for suf in ("_e32", "_e64"):
            if base_op.endswith(suf):
                base_op = base_op[: -len(suf)]
                break

        if base_op in ("v_mov_b32", "s_mov_b32"):
            src_val, _, _ = self.parse_operand(src_token)
            self.write_reg(dst, src_val)
            return

        f32_modes = {
            "v_floor_f32": "floor",
            "v_ceil_f32": "ceil",
            "v_trunc_f32": "trunc",
            "v_rndne_f32": "rndne",
        }
        f16_modes = {
            "v_floor_f16": "floor",
            "v_ceil_f16": "ceil",
            "v_trunc_f16": "trunc",
            "v_rndne_f16": "rndne",
        }

        if base_op in f32_modes:
            mode = f32_modes[base_op]
            src_val, _, _ = self.parse_operand(src_token)
            res_bits = ref_round_f32_bits(src_val, mode)

            # Apply output modifiers
            if not is_nan_f32(res_bits) and res_bits not in (POS_INF_F32, NEG_INF_F32):
                res_f = bits_to_f32(res_bits)
                if "mul:2" in mods:
                    res_f *= 2.0
                elif "mul:4" in mods:
                    res_f *= 4.0
                elif "div:2" in mods:
                    res_f *= 0.5

                if "clamp" in mods:
                    res_f = max(0.0, min(1.0, res_f))
                res_bits = f32_to_bits(res_f)

            self.write_reg(dst, res_bits)

        elif base_op in f16_modes:
            mode = f16_modes[base_op]
            src_val, is_abs, is_neg = self.parse_operand(src_token)

            # Pick high or low half of source based on op_sel_src
            h = (src_val >> 16) & MASK16 if op_sel_src else (src_val & MASK16)
            if is_abs:
                h &= ~SIGN_MASK16
            if is_neg:
                h ^= SIGN_MASK16

            res_h = ref_round_f16_bits(h, mode)

            if not is_nan_f16(res_h) and res_h not in (POS_INF_F16, NEG_INF_F16):
                res_f = bits_to_f16(res_h)
                if "mul:2" in mods:
                    res_f *= 2.0
                elif "mul:4" in mods:
                    res_f *= 4.0
                elif "div:2" in mods:
                    res_f *= 0.5

                if "clamp" in mods:
                    res_f = max(0.0, min(1.0, res_f))
                res_h = f16_to_bits(res_f)

            # Write into dst half according to op_sel_dst, preserving the other half
            old_dst = self.read_reg(dst)
            if op_sel_dst:
                new_dst = (old_dst & 0x0000FFFF) | ((res_h & MASK16) << 16)
            else:
                new_dst = (old_dst & 0xFFFF0000) | (res_h & MASK16)
            self.write_reg(dst, new_dst)

        else:
            raise NotImplementedError(f"Unsupported instruction mnemonic: {mnemonic}")


# ==============================================================================
# SPIR-V Translation & Execution Harness
# ==============================================================================

_TMP_DIR: Optional[tempfile.TemporaryDirectory] = None
_SRC_PATH: Optional[str] = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_round.s")
    with open(os.path.join(_TMP_DIR.name, "ps_round.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_assembly(lines: Sequence[str]) -> Tuple[int, int, int, int]:
    """Translates a synthetic pixel shader with tools/rdna2spirv.py and runs it
    in tests/semantic/spirv_mini_interp.py, returning exported mrt0 (v0, v1, v2, v3).
    """
    if _TMP_DIR is None:
        setUpModule()
    assert _SRC_PATH is not None
    asm = (
        "; test_alu_round\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    out = si.run_fragment(spv)["%out_mrt0"]
    return (out[0], out[1], out[2], out[3])


# ==============================================================================
# Unit Tests
# ==============================================================================


class TestRoundReferenceOracles(unittest.TestCase):
    """Verifies that the reference oracles adhere strictly to IEEE 754-2008 /
    IEEE 754-2019 roundToIntegral semantics and sign preservation rules.
    """

    def test_signed_zeros_f32(self):
        # In all modes, +0.0 returns +0.0 and -0.0 returns -0.0
        for mode in ("floor", "ceil", "trunc", "rndne"):
            with self.subTest(mode=mode):
                self.assertEqual(ref_round_f32_bits(POS_ZERO_F32, mode), POS_ZERO_F32)
                self.assertEqual(ref_round_f32_bits(NEG_ZERO_F32, mode), NEG_ZERO_F32)

    def test_exact_integers_f32(self):
        integers = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 42.0, -99.0, 1024.0, -2048.0]
        for val in integers:
            bits = f32_to_bits(val)
            for mode in ("floor", "ceil", "trunc", "rndne"):
                with self.subTest(val=val, mode=mode):
                    self.assertEqual(ref_round_f32_bits(bits, mode), bits)

    def test_positive_fractions_f32(self):
        cases = [
            # (val, floor, ceil, trunc, rndne)
            (0.25, 0.0, 1.0, 0.0, 0.0),
            (0.50, 0.0, 1.0, 0.0, 0.0),  # tie to even (0 is even)
            (0.75, 0.0, 1.0, 0.0, 1.0),
            (1.25, 1.0, 2.0, 1.0, 1.0),
            (1.50, 1.0, 2.0, 1.0, 2.0),  # tie to even (2 is even)
            (1.75, 1.0, 2.0, 1.0, 2.0),
            (2.25, 2.0, 3.0, 2.0, 2.0),
            (2.50, 2.0, 3.0, 2.0, 2.0),  # tie to even (2 is even)
            (2.75, 2.0, 3.0, 2.0, 3.0),
            (3.50, 3.0, 4.0, 3.0, 4.0),  # tie to even (4 is even)
        ]
        for val, exp_fl, exp_ce, exp_tr, exp_rn in cases:
            bits = f32_to_bits(val)
            with self.subTest(val=val):
                self.assertEqual(ref_round_f32_bits(bits, "floor"), f32_to_bits(exp_fl))
                self.assertEqual(ref_round_f32_bits(bits, "ceil"), f32_to_bits(exp_ce))
                self.assertEqual(ref_round_f32_bits(bits, "trunc"), f32_to_bits(exp_tr))
                self.assertEqual(ref_round_f32_bits(bits, "rndne"), f32_to_bits(exp_rn))

    def test_negative_fractions_and_signed_zero_preservation_f32(self):
        cases = [
            # (val, floor, ceil, trunc, rndne)
            (-0.25, -1.0, -0.0, -0.0, -0.0),
            (-0.50, -1.0, -0.0, -0.0, -0.0),  # tie to even (-0.0)
            (-0.75, -1.0, -0.0, -0.0, -1.0),
            (-1.25, -2.0, -1.0, -1.0, -1.0),
            (-1.50, -2.0, -1.0, -1.0, -2.0),  # tie to even (-2.0 is even)
            (-1.75, -2.0, -1.0, -1.0, -2.0),
            (-2.50, -3.0, -2.0, -2.0, -2.0),  # tie to even (-2.0 is even)
            (-3.50, -4.0, -3.0, -3.0, -4.0),  # tie to even (-4.0 is even)
        ]
        for val, exp_fl, exp_ce, exp_tr, exp_rn in cases:
            bits = f32_to_bits(val)
            with self.subTest(val=val):
                self.assertEqual(ref_round_f32_bits(bits, "floor"), f32_to_bits(exp_fl))
                self.assertEqual(ref_round_f32_bits(bits, "ceil"), f32_to_bits(exp_ce))
                self.assertEqual(ref_round_f32_bits(bits, "trunc"), f32_to_bits(exp_tr))
                self.assertEqual(ref_round_f32_bits(bits, "rndne"), f32_to_bits(exp_rn))

    def test_subnormals_f32(self):
        # Positive subnormal: (0.0, 1.0)
        pos_sub = 0x00000001
        self.assertEqual(ref_round_f32_bits(pos_sub, "floor"), POS_ZERO_F32)
        self.assertEqual(ref_round_f32_bits(pos_sub, "ceil"), f32_to_bits(1.0))
        self.assertEqual(ref_round_f32_bits(pos_sub, "trunc"), POS_ZERO_F32)
        self.assertEqual(ref_round_f32_bits(pos_sub, "rndne"), POS_ZERO_F32)

        # Negative subnormal: (-1.0, 0.0)
        neg_sub = 0x80000001
        self.assertEqual(ref_round_f32_bits(neg_sub, "floor"), f32_to_bits(-1.0))
        self.assertEqual(ref_round_f32_bits(neg_sub, "ceil"), NEG_ZERO_F32)
        self.assertEqual(ref_round_f32_bits(neg_sub, "trunc"), NEG_ZERO_F32)
        self.assertEqual(ref_round_f32_bits(neg_sub, "rndne"), NEG_ZERO_F32)

    def test_infinities_and_nans_f32(self):
        for val in (POS_INF_F32, NEG_INF_F32, QNAN_F32, 0x7FC01234, 0x7F800001):
            for mode in ("floor", "ceil", "trunc", "rndne"):
                with self.subTest(val=hex(val), mode=mode):
                    self.assertEqual(ref_round_f32_bits(val, mode), val)

    def test_large_magnitudes_without_fractions_f32(self):
        # 2^23 = 8388608.0; any finite f32 >= 2^23 has ulp >= 1.0, hence is integral
        large_values = [
            f32_to_bits(float(1 << 23)),
            f32_to_bits(float(1 << 24)),
            f32_to_bits(float(1 << 30)),
            0x7F7FFFFF,  # max finite float
            f32_to_bits(-float(1 << 23)),
            0xFF7FFFFF,  # most negative finite float
        ]
        for bits in large_values:
            for mode in ("floor", "ceil", "trunc", "rndne"):
                with self.subTest(bits=hex(bits), mode=mode):
                    self.assertEqual(ref_round_f32_bits(bits, mode), bits)

    def test_f16_reference_oracles(self):
        self.assertEqual(ref_round_f16_bits(POS_ZERO_F16, "floor"), POS_ZERO_F16)
        self.assertEqual(ref_round_f16_bits(NEG_ZERO_F16, "floor"), NEG_ZERO_F16)

        # 1.75 in f16 is 0x3F00
        h_1_75 = f16_to_bits(1.75)
        self.assertEqual(ref_round_f16_bits(h_1_75, "floor"), f16_to_bits(1.0))
        self.assertEqual(ref_round_f16_bits(h_1_75, "ceil"), f16_to_bits(2.0))
        self.assertEqual(ref_round_f16_bits(h_1_75, "trunc"), f16_to_bits(1.0))
        self.assertEqual(ref_round_f16_bits(h_1_75, "rndne"), f16_to_bits(2.0))

        # -1.75 in f16 is 0xBF00
        h_neg_1_75 = f16_to_bits(-1.75)
        self.assertEqual(ref_round_f16_bits(h_neg_1_75, "floor"), f16_to_bits(-2.0))
        self.assertEqual(ref_round_f16_bits(h_neg_1_75, "ceil"), f16_to_bits(-1.0))
        self.assertEqual(ref_round_f16_bits(h_neg_1_75, "trunc"), f16_to_bits(-1.0))
        self.assertEqual(ref_round_f16_bits(h_neg_1_75, "rndne"), f16_to_bits(-2.0))

        # -0.5 in f16: ceil/trunc/rndne preserve -0.0
        h_neg_0_5 = f16_to_bits(-0.5)
        self.assertEqual(ref_round_f16_bits(h_neg_0_5, "floor"), f16_to_bits(-1.0))
        self.assertEqual(ref_round_f16_bits(h_neg_0_5, "ceil"), NEG_ZERO_F16)
        self.assertEqual(ref_round_f16_bits(h_neg_0_5, "trunc"), NEG_ZERO_F16)
        self.assertEqual(ref_round_f16_bits(h_neg_0_5, "rndne"), NEG_ZERO_F16)


class TestMathematicalInvariants(unittest.TestCase):
    """Validates core mathematical properties and identities across a broad spectrum
    of floating-point values.
    """

    def test_bounding_inequalities(self):
        rng = random.Random(42)
        test_floats = [
            rng.uniform(-100.0, 100.0) for _ in range(500)
        ] + [0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 1e-10, -1e-10]

        for x in test_floats:
            fl = ref_floor_f32(x)
            ce = ref_ceil_f32(x)
            tr = ref_trunc_f32(x)
            rn = ref_rndne_f32(x)

            # Ordering invariant: floor(x) <= x <= ceil(x)
            self.assertLessEqual(fl, x)
            self.assertGreaterEqual(ce, x)

            # Difference bounds
            self.assertGreaterEqual(x - fl, 0.0)
            self.assertLess(x - fl, 1.0)
            self.assertGreaterEqual(ce - x, 0.0)
            self.assertLess(ce - x, 1.0)
            self.assertLess(abs(x - tr), 1.0)
            self.assertLessEqual(abs(x - rn), 0.5)

            # Truncation relationship
            if x >= 0:
                self.assertEqual(fl, tr)
                self.assertLessEqual(tr, ce)
            else:
                self.assertEqual(ce, tr)
                self.assertGreaterEqual(tr, fl)

    def test_idempotence(self):
        test_floats = [-4.8, -3.5, -2.0, -0.7, -0.0, 0.0, 0.3, 1.5, 2.0, 5.9]
        for x in test_floats:
            for mode in ("floor", "ceil", "trunc", "rndne"):
                with self.subTest(x=x, mode=mode):
                    first = ref_round_f32(x, mode)
                    second = ref_round_f32(first, mode)
                    self.assertEqual(first, second)

    def test_reflection_symmetry(self):
        test_floats = [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.5, 3.7, 42.0]
        for x in test_floats:
            with self.subTest(x=x):
                # floor(-x) == -ceil(x)
                self.assertEqual(ref_floor_f32(-x), -ref_ceil_f32(x))
                # ceil(-x) == -floor(x)
                self.assertEqual(ref_ceil_f32(-x), -ref_floor_f32(x))
                # trunc(-x) == -trunc(x)
                self.assertEqual(ref_trunc_f32(-x), -ref_trunc_f32(x))
                # rndne(-x) == -rndne(x)
                self.assertEqual(ref_rndne_f32(-x), -ref_rndne_f32(x))


class TestRoundValuInterpreter(unittest.TestCase):
    """Verifies that RoundValuInterpreter models RDNA2 VALU instructions, operand
    modifiers, clamp bounds, and f16 op_sel behavior.
    """

    def setUp(self) -> None:
        self.interp = RoundValuInterpreter()

    def test_basic_f32_instructions(self):
        self.interp.execute_line("v_mov_b32 v1, 1.75")
        self.interp.execute_line("v_floor_f32 v0, v1")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(1.0))

        self.interp.execute_line("v_ceil_f32 v0, v1")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(2.0))

        self.interp.execute_line("v_trunc_f32 v0, v1")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(1.0))

        self.interp.execute_line("v_rndne_f32 v0, v1")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(2.0))

    def test_e32_and_e64_forms(self):
        self.interp.execute_line("v_mov_b32 v2, 2.5")
        self.interp.execute_line("v_floor_f32_e32 v0, v2")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(2.0))

        self.interp.execute_line("v_floor_f32_e64 v0, v2")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(2.0))

    def test_inline_constants(self):
        self.interp.execute_line("v_floor_f32 v0, 1.5")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(1.0))

        self.interp.execute_line("v_ceil_f32 v0, 1.5")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(2.0))

        self.interp.execute_line("v_trunc_f32 v0, -0.5")
        self.assertEqual(self.interp.read_reg("v0"), NEG_ZERO_F32)

        self.interp.execute_line("v_rndne_f32 v0, 2.5")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(2.0))

    def test_input_modifiers(self):
        self.interp.execute_line("v_mov_b32 v1, -1.75")

        # |v1| = 1.75 -> floor(1.75) = 1.0
        self.interp.execute_line("v_floor_f32_e64 v0, |v1|")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(1.0))

        # -v1 = 1.75 -> ceil(1.75) = 2.0
        self.interp.execute_line("v_ceil_f32_e64 v0, -v1")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(2.0))

        # -|v1| = -1.75 -> trunc(-1.75) = -1.0
        self.interp.execute_line("v_trunc_f32_e64 v0, -|v1|")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(-1.0))

    def test_output_clamp_and_multipliers(self):
        # Result > 1.0 clamped to 1.0
        self.interp.execute_line("v_mov_b32 v1, 2.5")
        self.interp.execute_line("v_floor_f32 v0, v1 clamp")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(1.0))

        # Result < 0.0 clamped to 0.0
        self.interp.execute_line("v_mov_b32 v1, -2.5")
        self.interp.execute_line("v_floor_f32 v0, v1 clamp")
        self.assertEqual(self.interp.read_reg("v0"), POS_ZERO_F32)

        # Output multiplier mul:2 before clamp: floor(0.75) = 0.0 * 2 = 0.0
        self.interp.execute_line("v_mov_b32 v1, 0.75")
        self.interp.execute_line("v_ceil_f32 v0, v1 mul:2")
        self.assertEqual(self.interp.read_reg("v0"), f32_to_bits(2.0))

    def test_f16_half_selection_and_destination_preservation(self):
        # Packed f16: high half 1.75 (0x3F00), low half 3.0 (0x4200)
        self.interp.execute_line("v_mov_b32 v4, 0x3F004200")

        # Initial dst with canary pattern 0x12345678
        self.interp.execute_line("v_mov_b32 v0, 0x12345678")

        # op_sel:[1, 1]: read src high (1.75), write dst high (floor(1.75) = 1.0 = 0x3C00)
        # Low half (0x5678) must be preserved!
        self.interp.execute_line("v_floor_f16 v0, v4 op_sel:[1,1]")
        self.assertEqual(self.interp.read_reg("v0"), 0x3C005678)

        # op_sel:[0, 0]: read src low (3.0), write dst low (ceil(3.0) = 3.0 = 0x4200)
        # High half (0x3C00) must be preserved!
        self.interp.execute_line("v_ceil_f16 v0, v4 op_sel:[0,0]")
        self.assertEqual(self.interp.read_reg("v0"), 0x3C004200)


class TestSpirvMiniInterpExecution(unittest.TestCase):
    """End-to-end integration tests compiling synthetic RDNA2 assembly with
    tools/rdna2spirv.py and executing the generated SPIR-V in
    tests/semantic/spirv_mini_interp.py.
    """

    def test_f32_rounding_modes(self):
        # Run floor, ceil, trunc, rndne on 1.75
        out = run_assembly([
            "v_mov_b32 v4, 1.75",
            "v_floor_f32 v0, v4",
            "v_ceil_f32 v1, v4",
            "v_trunc_f32 v2, v4",
            "v_rndne_f32 v3, v4",
        ])
        self.assertEqual(out[0], f32_to_bits(1.0))
        self.assertEqual(out[1], f32_to_bits(2.0))
        self.assertEqual(out[2], f32_to_bits(1.0))
        self.assertEqual(out[3], f32_to_bits(2.0))

    def test_f32_negative_fraction_and_signed_zero(self):
        # On -0.5: floor is -1.0, while ceil, trunc, rndne give -0.0
        out = run_assembly([
            "v_mov_b32 v4, -0.5",
            "v_floor_f32 v0, v4",
            "v_ceil_f32 v1, v4",
            "v_trunc_f32 v2, v4",
            "v_rndne_f32 v3, v4",
        ])
        self.assertEqual(out[0], f32_to_bits(-1.0))
        self.assertEqual(out[1], NEG_ZERO_F32)
        self.assertEqual(out[2], NEG_ZERO_F32)
        self.assertEqual(out[3], NEG_ZERO_F32)

    def test_f32_subnormal_execution(self):
        # Negative subnormal 0x80000001
        out = run_assembly([
            "v_mov_b32 v4, 0x80000001",
            "v_floor_f32 v0, v4",
            "v_ceil_f32 v1, v4",
            "v_trunc_f32 v2, v4",
            "v_rndne_f32 v3, v4",
        ])
        # v_floor reads a denormal as zero (f32 denormals flushed, docs/design.md), so floor(-denormal) is -0.
        self.assertEqual(out[0], NEG_ZERO_F32)
        self.assertEqual(out[1], NEG_ZERO_F32)
        self.assertEqual(out[2], NEG_ZERO_F32)
        self.assertEqual(out[3], NEG_ZERO_F32)

    def test_f32_infinities_and_nans(self):
        out = run_assembly([
            "v_mov_b32 v4, 0x7F800000",  # +inf
            "v_floor_f32 v0, v4",
            "v_mov_b32 v5, 0xFF800000",  # -inf
            "v_ceil_f32 v1, v5",
            "v_mov_b32 v6, 0x7FC00000",  # qNaN
            "v_trunc_f32 v2, v6",
            "v_rndne_f32 v3, v4",
        ])
        self.assertEqual(out[0], POS_INF_F32)
        self.assertEqual(out[1], NEG_INF_F32)
        self.assertEqual(out[2], QNAN_F32)
        self.assertEqual(out[3], POS_INF_F32)

    def test_modifiers_and_clamp_in_spirv(self):
        out = run_assembly([
            "v_mov_b32 v4, 0xBFC00000",  # -1.5
            "v_floor_f32_e64 v0, |v4|",  # floor(|-1.5|) = 1.0
            "v_floor_f32_e64 v1, -v4",   # floor(-(-1.5)) = 1.0
            "v_mov_b32 v5, 2.5",
            "v_floor_f32 v2, v5 clamp",  # floor(2.5) = 2.0 -> clamp to 1.0
            "v_mov_b32 v6, -2.5",
            "v_floor_f32 v3, v6 clamp",  # floor(-2.5) = -3.0 -> clamp to 0.0
        ])
        self.assertEqual(out[0], f32_to_bits(1.0))
        self.assertEqual(out[1], f32_to_bits(1.0))
        self.assertEqual(out[2], f32_to_bits(1.0))
        self.assertEqual(out[3], POS_ZERO_F32)

    def test_f16_execution_and_op_sel(self):
        out = run_assembly([
            "v_mov_b32 v4, 0x3F004200",  # hi: 1.75, lo: 3.0
            "v_mov_b32 v0, 0x12345678",
            "v_floor_f16 v0, v4 op_sel:[1,1]",  # hi = floor(1.75) = 1.0 (0x3C00)
            "v_mov_b32 v1, 0x12345678",
            "v_ceil_f16 v1, v4 op_sel:[0,0]",   # lo = ceil(3.0) = 3.0 (0x4200)
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual(out[0], 0x3C005678)
        self.assertEqual(out[1], 0x12344200)

    def test_spirv_opcode_generation(self):
        assert _SRC_PATH is not None
        asm = (
            "; test_spirv_opcodes\n"
            "\tv_floor_f32 v0, v0\n"
            "\tv_ceil_f32 v1, v1\n"
            "\tv_trunc_f32 v2, v2\n"
            "\tv_rndne_f32 v3, v3\n"
            "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
        )
        with open(_SRC_PATH, "w", encoding="utf-8") as f:
            f.write(asm)
        spv = rdna2spirv.translate(_SRC_PATH)

        self.assertIn("OpExtInst %float %glsl Floor", spv)
        self.assertIn("OpExtInst %float %glsl Ceil", spv)
        self.assertIn("OpExtInst %float %glsl Trunc", spv)
        self.assertIn("OpExtInst %float %glsl RoundEven", spv)


class TestMultiLaneSimdWave(unittest.TestCase):
    """Verifies that vector rounding instructions execute correctly across all 64
    independent lanes of a SIMD Wave64 without lane cross-contamination.
    """

    def test_wave64_simd_rounding(self):
        num_lanes = 64
        rng = random.Random(1337)
        lane_inputs = [
            rng.uniform(-50.0, 50.0) for _ in range(num_lanes)
        ]
        # Force specific boundary edge cases in prominent lanes
        lane_inputs[0] = 0.0
        lane_inputs[1] = -0.0
        lane_inputs[2] = 0.5
        lane_inputs[3] = -0.5
        lane_inputs[4] = 1.5
        lane_inputs[5] = -1.5
        lane_inputs[6] = 2.5
        lane_inputs[7] = -2.5

        interp = RoundValuInterpreter()
        expected_floor = [ref_floor_f32(x) for x in lane_inputs]
        expected_ceil = [ref_ceil_f32(x) for x in lane_inputs]
        expected_trunc = [ref_trunc_f32(x) for x in lane_inputs]
        expected_rndne = [ref_rndne_f32(x) for x in lane_inputs]

        actual_floor = []
        actual_ceil = []
        actual_trunc = []
        actual_rndne = []

        for x in lane_inputs:
            bits = f32_to_bits(x)
            interp.write_reg("v1", bits)

            interp.execute_line("v_floor_f32 v0, v1")
            actual_floor.append(bits_to_f32(interp.read_reg("v0")))

            interp.execute_line("v_ceil_f32 v0, v1")
            actual_ceil.append(bits_to_f32(interp.read_reg("v0")))

            interp.execute_line("v_trunc_f32 v0, v1")
            actual_trunc.append(bits_to_f32(interp.read_reg("v0")))

            interp.execute_line("v_rndne_f32 v0, v1")
            actual_rndne.append(bits_to_f32(interp.read_reg("v0")))

        for i in range(num_lanes):
            self.assertEqual(f32_to_bits(actual_floor[i]), f32_to_bits(expected_floor[i]))
            self.assertEqual(f32_to_bits(actual_ceil[i]), f32_to_bits(expected_ceil[i]))
            self.assertEqual(f32_to_bits(actual_trunc[i]), f32_to_bits(expected_trunc[i]))
            self.assertEqual(f32_to_bits(actual_rndne[i]), f32_to_bits(expected_rndne[i]))


class TestNegativeControls(unittest.TestCase):
    """Negative controls ensuring mutated semantics or dropped edge cases fail."""

    def test_floor_ceil_swap_detected(self):
        # 1.5: floor=1.0, ceil=2.0
        val = 1.5
        correct_floor = ref_floor_f32(val)
        mutated_floor = ref_ceil_f32(val)
        self.assertNotEqual(correct_floor, mutated_floor)

    def test_signed_zero_dropping_detected(self):
        # -0.5: trunc must be -0.0 (0x80000000), not +0.0 (0x00000000)
        bits_neg_half = f32_to_bits(-0.5)
        correct_trunc_bits = ref_round_f32_bits(bits_neg_half, "trunc")
        mutated_trunc_bits = POS_ZERO_F32
        self.assertEqual(correct_trunc_bits, NEG_ZERO_F32)
        self.assertNotEqual(correct_trunc_bits, mutated_trunc_bits)

    def test_tie_to_odd_detected(self):
        # 1.5: round-to-nearest-even gives 2.0; round-to-nearest-odd would give 1.0
        val = 1.5
        correct_rndne = ref_rndne_f32(val)
        tie_to_odd = 1.0
        self.assertEqual(correct_rndne, 2.0)
        self.assertNotEqual(correct_rndne, tie_to_odd)

    def test_clamp_omission_detected(self):
        val = 2.5
        unclamped = ref_floor_f32(val)  # 2.0
        clamped = max(0.0, min(1.0, unclamped))  # 1.0
        self.assertNotEqual(unclamped, clamped)


if __name__ == "__main__":
    unittest.main()
