"""Semantic value tests for RDNA2 single-precision transcendental VALU operations:
  - v_cos_f32:   D.f32 = cos(S0.f32 * 2*pi)    (operand in turns / revolutions)
  - v_sin_f32:   D.f32 = sin(S0.f32 * 2*pi)    (operand in turns / revolutions)
  - v_rcp_f32:   D.f32 = 1.0 / S0.f32          (single-precision reciprocal)
  - v_rsq_f32:   D.f32 = 1.0 / sqrt(S0.f32)    (single-precision inverse square root)
  - v_sqrt_f32:  D.f32 = sqrt(S0.f32)          (single-precision square root)

RDNA2 takes trigonometric inputs in units of revolutions (turns), reducing to [-0.5, 0.5]
before multiplying by 2*pi. Reciprocal and inverse square root provide 1-ULP approximations
satisfying core identities:
  - Pythagorean:    sin^2(x) + cos^2(x) ~= 1.0
  - Reciprocal:     x * rcp(x) ~= 1.0
  - Inverse Sqrt:   x * (rsq(x))^2 ~= 1.0
  - Sqrt / Rsq:     rsq(x) * sqrt(x) ~= 1.0

This test suite verifies plain-Python IEEE-754 / RDNA2 reference oracles, mathematical
invariants, an in-test RDNA2 SIMD wave interpreter modeling multi-lane divergence under
EXEC masking, and end-to-end SPIR-V shader translation and execution via spirv_mini_interp.

Standard library Python only; synthetic hand-written listings only (no game data).
"""
from __future__ import annotations

import gc
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
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
TWO_PI = 6.283185307179586476925286766559
SENTINEL_FILL = 0xDEADBEEF

# Canonical binary32 IEEE-754 bit patterns.
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
SUBNORM_MIN = 0x00000001       # 2^-149 (smallest positive subnormal)
NEG_SUBNORM_MIN = 0x80000001   # -2^-149
SUBNORM_MAX = 0x007FFFFF       # (1 - 2^-23) * 2^-126
FLT_MIN = 0x00800000           # 2^-126 (smallest positive normal)
NEG_FLT_MIN = 0x80800000       # -2^-126
HALF = 0x3F000000              # 0.5
NEG_HALF = 0xBF000000          # -0.5
ONE = 0x3F800000               # 1.0
NEG_ONE = 0xBF800000           # -1.0
TWO = 0x40000000               # 2.0
NEG_TWO = 0xC0000000           # -2.0
FOUR = 0x40800000              # 4.0
SIXTEEN = 0x41800000           # 16.0
POS_INF = 0x7F800000           # +inf
NEG_INF = 0xFF800000           # -inf
QNAN = 0x7FC00000              # canonical positive quiet NaN
NEG_QNAN = 0xFFC00000          # canonical negative quiet NaN
SNAN = 0x7F800001              # signaling NaN


def f32_to_bits(val: float) -> int:
    """Pack Python float to 32-bit unsigned integer IEEE-754 representation."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def bits_to_f32(bits: int) -> float:
    """Unpack 32-bit unsigned integer to Python single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_f32_nan_bits(bits: int) -> bool:
    """Check if 32-bit word encodes an IEEE-754 NaN."""
    b = bits & MASK32
    return (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF) != 0


def round_even(x: float) -> float:
    """Round to nearest integer, breaking ties to even (IEEE-754 roundTiesToEven)."""
    return float(round(x))


def ref_turn_reduce(turns: float) -> float:
    """Reduce turn count to [-0.5, 0.5] as performed by RDNA2 hardware and rdna2spirv.

    For finite inputs, turns - round_even(turns) reduces exactly in binary32.
    Signed zeros (+0.0 and -0.0) are preserved.
    """
    if math.isnan(turns) or math.isinf(turns):
        return float("nan")
    if turns == 0.0:
        return turns
    return turns - round_even(turns)


def ref_cos_f32(turns_bits: int) -> int:
    """RDNA2 v_cos_f32 reference oracle: D.f32 = cos(turns * 2*pi)."""
    if is_f32_nan_bits(turns_bits):
        return QNAN
    val = bits_to_f32(turns_bits)
    if math.isinf(val):
        return QNAN
    if val == 0.0:
        return ONE
    t = ref_turn_reduce(val)
    rad = t * TWO_PI
    res = math.cos(rad)
    return f32_to_bits(res)


def ref_sin_f32(turns_bits: int) -> int:
    """RDNA2 v_sin_f32 reference oracle: D.f32 = sin(turns * 2*pi)."""
    if is_f32_nan_bits(turns_bits):
        return QNAN
    val = bits_to_f32(turns_bits)
    if math.isinf(val):
        return QNAN
    if turns_bits == NEG_ZERO:
        return NEG_ZERO
    if turns_bits == POS_ZERO:
        return POS_ZERO
    t = ref_turn_reduce(val)
    rad = t * TWO_PI
    res = math.sin(rad)
    if res == 0.0 and math.copysign(1.0, val) < 0:
        return NEG_ZERO
    return f32_to_bits(res)


def ref_rcp_f32(x_bits: int) -> int:
    """RDNA2 v_rcp_f32 reference oracle: D.f32 = 1.0 / S0.f32."""
    if is_f32_nan_bits(x_bits):
        return QNAN
    val = bits_to_f32(x_bits)
    if val == 0.0:
        return NEG_INF if (x_bits & 0x80000000) else POS_INF
    if math.isinf(val):
        return NEG_ZERO if val < 0 else POS_ZERO
    res = 1.0 / val
    return f32_to_bits(res)


def ref_rsq_f32(x_bits: int) -> int:
    """RDNA2 v_rsq_f32 reference oracle: D.f32 = 1.0 / sqrt(S0.f32)."""
    if is_f32_nan_bits(x_bits):
        return QNAN
    val = bits_to_f32(x_bits)
    if val < 0.0:
        return QNAN
    if val == 0.0:
        return POS_INF
    if math.isinf(val):
        return POS_ZERO
    res = 1.0 / math.sqrt(val)
    return f32_to_bits(res)


def ref_sqrt_f32(x_bits: int) -> int:
    """RDNA2 v_sqrt_f32 reference oracle: D.f32 = sqrt(S0.f32)."""
    if is_f32_nan_bits(x_bits):
        return QNAN
    val = bits_to_f32(x_bits)
    if val < 0.0:
        return QNAN
    if val == 0.0:
        return x_bits  # preserves signed -0.0
    if math.isinf(val):
        return POS_INF
    res = math.sqrt(val)
    return f32_to_bits(res)


class Rdna2ValuSimdInterpreter:
    """Software model of RDNA2 Vector ALU SIMD execution with EXEC mask divergence.

    Simulates a wavefront (32 lanes for Wave32, 64 lanes for Wave64), tracking:
      - Vector registers v0..v63 per lane
      - Scalar registers s0..s15
      - 64-bit EXEC mask (exec_lo, exec_hi)
      - Per-lane divergence: vector instructions execute only on active lanes where
        ((exec >> lane) & 1) != 0. Inactive lanes strictly retain their previous register state.
    """

    def __init__(self, wave_size: int = 32):
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave_size {wave_size}; must be 32 or 64")
        self.wave_size = wave_size
        self.exec_mask = (1 << wave_size) - 1
        self.vgprs = [[SENTINEL_FILL] * 64 for _ in range(wave_size)]
        self.sgprs = [0] * 16

    def set_exec(self, mask: int):
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def parse_reg(self, token: str) -> tuple[str, int]:
        token = token.strip().rstrip(",")
        if token.startswith("v") and token[1:].isdigit():
            return ("v", int(token[1:]))
        if token.startswith("s") and token[1:].isdigit():
            return ("s", int(token[1:]))
        if token == "exec":
            return ("exec", 0)
        if token == "exec_lo":
            return ("exec_lo", 0)
        if token == "exec_hi":
            return ("exec_hi", 0)
        return ("imm", int(token, 0))

    def get_src_lane(self, token: str, lane: int) -> int:
        kind, idx = self.parse_reg(token)
        if kind == "v":
            return self.vgprs[lane][idx]
        if kind == "s":
            return self.sgprs[idx]
        if kind == "imm":
            return idx & MASK32
        raise ValueError(f"Unknown src token: {token}")

    def execute_line(self, line: str):
        line = line.split(";")[0].strip()
        if not line or line.startswith("//"):
            return
        parts = line.replace(",", " ").split()
        op = parts[0]
        args = parts[1:]

        if op == "s_mov_b32":
            dst_kind, dst_idx = self.parse_reg(args[0])
            val = self.get_src_lane(args[1], 0)
            if dst_kind == "s":
                self.sgprs[dst_idx] = val & MASK32
            elif dst_kind == "exec_lo":
                self.exec_mask = (self.exec_mask & 0xFFFFFFFF00000000) | (val & MASK32)
            elif dst_kind == "exec_hi":
                self.exec_mask = (self.exec_mask & 0x00000000FFFFFFFF) | ((val & MASK32) << 32)
        elif op == "s_mov_b64":
            dst_kind, dst_idx = self.parse_reg(args[0])
            src_kind, src_idx = self.parse_reg(args[1])
            if src_kind == "imm":
                val = src_idx & MASK64
            elif src_kind == "s":
                val = (self.sgprs[src_idx] & MASK32) | ((self.sgprs[src_idx + 1] & MASK32) << 32)
            elif src_kind == "exec":
                val = self.exec_mask
            else:
                val = 0
            if dst_kind == "exec":
                self.set_exec(val)
            elif dst_kind == "s":
                self.sgprs[dst_idx] = val & MASK32
                self.sgprs[dst_idx + 1] = (val >> 32) & MASK32
        elif op == "s_not_b64":
            dst_kind, dst_idx = self.parse_reg(args[0])
            src_kind, _ = self.parse_reg(args[1])
            val = ~self.exec_mask if src_kind == "exec" else ~0
            if dst_kind == "exec":
                self.set_exec(val)
        elif op == "s_and_b64":
            dst_kind, _ = self.parse_reg(args[0])
            src0_kind, src0_val = self.parse_reg(args[1])
            src1_kind, src1_val = self.parse_reg(args[2])
            v0 = self.exec_mask if src0_kind == "exec" else src0_val
            v1 = self.exec_mask if src1_kind == "exec" else src1_val
            res = v0 & v1
            if dst_kind == "exec":
                self.set_exec(res)
        elif op in ("v_mov_b32", "v_cos_f32", "v_sin_f32", "v_rcp_f32", "v_rsq_f32", "v_sqrt_f32"):
            dst_kind, dst_idx = self.parse_reg(args[0])
            if dst_kind != "v":
                raise ValueError(f"Vector instruction destination must be a VGPR: {line}")
            for lane in range(self.wave_size):
                if not ((self.exec_mask >> lane) & 1):
                    continue
                src_val = self.get_src_lane(args[1], lane)
                if op == "v_mov_b32":
                    out_val = src_val
                elif op == "v_cos_f32":
                    out_val = ref_cos_f32(src_val)
                elif op == "v_sin_f32":
                    out_val = ref_sin_f32(src_val)
                elif op == "v_rcp_f32":
                    out_val = ref_rcp_f32(src_val)
                elif op == "v_rsq_f32":
                    out_val = ref_rsq_f32(src_val)
                elif op == "v_sqrt_f32":
                    out_val = ref_sqrt_f32(src_val)
                self.vgprs[lane][dst_idx] = out_val & MASK32

    def run_program(self, lines: list[str]):
        for line in lines:
            self.execute_line(line)


class TestTranscendentalOracles(unittest.TestCase):
    """Test suite validating the RDNA2 transcendental reference oracles."""

    def test_cos_cardinal_angles(self):
        # 0 turns = 0 rad -> cos = 1.0
        self.assertEqual(ref_cos_f32(POS_ZERO), ONE)
        self.assertEqual(ref_cos_f32(NEG_ZERO), ONE)
        # 0.25 turns = pi/2 rad -> cos = 0.0
        c025 = bits_to_f32(ref_cos_f32(f32_to_bits(0.25)))
        self.assertAlmostEqual(c025, 0.0, places=6)
        # 0.5 turns = pi rad -> cos = -1.0
        c05 = bits_to_f32(ref_cos_f32(f32_to_bits(0.5)))
        self.assertAlmostEqual(c05, -1.0, places=6)
        # 0.75 turns = 3pi/2 rad -> cos = 0.0
        c075 = bits_to_f32(ref_cos_f32(f32_to_bits(0.75)))
        self.assertAlmostEqual(c075, 0.0, places=6)
        # 1.0 turns = 2pi rad -> cos = 1.0
        c10 = bits_to_f32(ref_cos_f32(f32_to_bits(1.0)))
        self.assertAlmostEqual(c10, 1.0, places=6)

    def test_sin_cardinal_angles(self):
        # 0.0 turns -> sin = 0.0; -0.0 turns -> sin = -0.0
        self.assertEqual(ref_sin_f32(POS_ZERO), POS_ZERO)
        self.assertEqual(ref_sin_f32(NEG_ZERO), NEG_ZERO)
        # 0.25 turns = pi/2 rad -> sin = 1.0
        s025 = bits_to_f32(ref_sin_f32(f32_to_bits(0.25)))
        self.assertAlmostEqual(s025, 1.0, places=6)
        # 0.5 turns = pi rad -> sin = 0.0
        s05 = bits_to_f32(ref_sin_f32(f32_to_bits(0.5)))
        self.assertAlmostEqual(s05, 0.0, places=6)
        # 0.75 turns = 3pi/2 rad -> sin = -1.0
        s075 = bits_to_f32(ref_sin_f32(f32_to_bits(0.75)))
        self.assertAlmostEqual(s075, -1.0, places=6)
        # 1.0 turns = 2pi rad -> sin = 0.0
        s10 = bits_to_f32(ref_sin_f32(f32_to_bits(1.0)))
        self.assertAlmostEqual(s10, 0.0, places=6)

    def test_pythagorean_identity(self):
        # sin^2(x) + cos^2(x) == 1.0 across continuous spectrum of angles
        for i in range(-50, 51):
            turns = i * 0.125 + 0.03125
            t_bits = f32_to_bits(turns)
            s = bits_to_f32(ref_sin_f32(t_bits))
            c = bits_to_f32(ref_cos_f32(t_bits))
            pyth = s * s + c * c
            self.assertAlmostEqual(pyth, 1.0, delta=1e-5, msg=f"Pythagorean identity failed at turns={turns}")

    def test_turn_periodicity_and_symmetry(self):
        # Invariance under whole integer turn offsets: f(x + k) == f(x)
        dyadic_angles = [0.125, 0.25, 0.375, 0.625, 0.875]
        for a in dyadic_angles:
            for k in (1, 2, 5, 10, 50, 100):
                base_c = bits_to_f32(ref_cos_f32(f32_to_bits(a)))
                shift_c = bits_to_f32(ref_cos_f32(f32_to_bits(a + float(k))))
                self.assertAlmostEqual(base_c, shift_c, delta=1e-5)

                base_s = bits_to_f32(ref_sin_f32(f32_to_bits(a)))
                shift_s = bits_to_f32(ref_sin_f32(f32_to_bits(a + float(k))))
                self.assertAlmostEqual(base_s, shift_s, delta=1e-5)

        # Symmetry: cos(-x) == cos(x), sin(-x) == -sin(x)
        angles = [0.1, 0.23, 0.45, 0.67, 0.89]
        for a in angles:
            pos_bits = f32_to_bits(a)
            neg_bits = f32_to_bits(-a)
            self.assertAlmostEqual(bits_to_f32(ref_cos_f32(pos_bits)), bits_to_f32(ref_cos_f32(neg_bits)), delta=1e-6)
            self.assertAlmostEqual(bits_to_f32(ref_sin_f32(pos_bits)), -bits_to_f32(ref_sin_f32(neg_bits)), delta=1e-6)


class TestReciprocalAndRsqOracles(unittest.TestCase):
    """Test suite validating v_rcp_f32, v_rsq_f32, and v_sqrt_f32 reference oracles."""

    def test_rcp_exact_powers_of_two(self):
        cases = [
            (1.0, 1.0),
            (2.0, 0.5),
            (4.0, 0.25),
            (8.0, 0.125),
            (0.5, 2.0),
            (0.25, 4.0),
            (0.125, 8.0),
            (-2.0, -0.5),
            (-0.25, -4.0),
        ]
        for inp, expected in cases:
            got_bits = ref_rcp_f32(f32_to_bits(inp))
            self.assertEqual(got_bits, f32_to_bits(expected))

    def test_rsq_exact_values(self):
        cases = [
            (1.0, 1.0),
            (4.0, 0.5),
            (16.0, 0.25),
            (64.0, 0.125),
            (0.25, 2.0),
            (0.0625, 4.0),
        ]
        for inp, expected in cases:
            got_bits = ref_rsq_f32(f32_to_bits(inp))
            self.assertEqual(got_bits, f32_to_bits(expected))

    def test_sqrt_exact_values(self):
        cases = [
            (1.0, 1.0),
            (4.0, 2.0),
            (9.0, 3.0),
            (16.0, 4.0),
            (0.25, 0.5),
            (0.0, 0.0),
        ]
        for inp, expected in cases:
            got_bits = ref_sqrt_f32(f32_to_bits(inp))
            self.assertEqual(got_bits, f32_to_bits(expected))

    def test_reciprocal_identity(self):
        # x * rcp(x) ~= 1.0 for positive normal floats
        for i in range(1, 100):
            val = i * 0.375 + 0.125
            r = bits_to_f32(ref_rcp_f32(f32_to_bits(val)))
            product = val * r
            self.assertAlmostEqual(product, 1.0, delta=1e-5, msg=f"Reciprocal identity failed at {val}")

    def test_inverse_sqrt_identity(self):
        # x * (rsq(x))^2 ~= 1.0 for positive normal floats
        for i in range(1, 100):
            val = i * 0.5 + 0.25
            rsq = bits_to_f32(ref_rsq_f32(f32_to_bits(val)))
            identity = val * (rsq * rsq)
            self.assertAlmostEqual(identity, 1.0, delta=1e-5, msg=f"Inverse sqrt identity failed at {val}")

    def test_sqrt_rsq_duality(self):
        # rsq(x) * sqrt(x) ~= 1.0
        for i in range(1, 100):
            val = i * 1.25 + 0.5
            rsq = bits_to_f32(ref_rsq_f32(f32_to_bits(val)))
            sq = bits_to_f32(ref_sqrt_f32(f32_to_bits(val)))
            product = rsq * sq
            self.assertAlmostEqual(product, 1.0, delta=1e-5, msg=f"Sqrt/Rsq duality failed at {val}")


class TestSimdWaveInterpreterDivergence(unittest.TestCase):
    """Test suite validating multi-lane SIMD divergence under EXEC mask."""

    def test_wave32_checkerboard_divergence_rcp_rsq(self):
        # 32-lane Wave32 with checkerboard active masks
        interp = Rdna2ValuSimdInterpreter(wave_size=32)

        # Initialize v0 per lane with distinct inputs: (lane + 1) * 2.0
        for lane in range(32):
            val = float((lane + 1) * 2)
            interp.vgprs[lane][0] = f32_to_bits(val)
            interp.vgprs[lane][1] = SENTINEL_FILL

        # Phase 1: Activate even lanes only (0x55555555) and compute v_rcp_f32
        interp.run_program([
            "s_mov_b32 exec_lo, 0x55555555",
            "v_rcp_f32 v1, v0",
        ])

        for lane in range(32):
            if lane % 2 == 0:
                expected = ref_rcp_f32(interp.vgprs[lane][0])
                self.assertEqual(interp.vgprs[lane][1], expected, f"Even lane {lane} failed rcp")
            else:
                self.assertEqual(interp.vgprs[lane][1], SENTINEL_FILL, f"Odd lane {lane} was improperly modified")

        # Phase 2: Invert mask to odd lanes only (0xAAAAAAAA) and compute v_rsq_f32
        interp.run_program([
            "s_mov_b32 exec_lo, 0xAAAAAAAA",
            "v_rsq_f32 v1, v0",
        ])

        for lane in range(32):
            if lane % 2 == 1:
                expected = ref_rsq_f32(interp.vgprs[lane][0])
                self.assertEqual(interp.vgprs[lane][1], expected, f"Odd lane {lane} failed rsq")
            else:
                # Even lanes should still hold their previous rcp result!
                expected = ref_rcp_f32(interp.vgprs[lane][0])
                self.assertEqual(interp.vgprs[lane][1], expected, f"Even lane {lane} lost previous value")

    def test_wave64_halves_divergence_rcp_sqrt(self):
        # 64-lane Wave64 split into low half (0..31) and high half (32..63)
        interp = Rdna2ValuSimdInterpreter(wave_size=64)

        for lane in range(64):
            val = float(lane + 4)
            interp.vgprs[lane][0] = f32_to_bits(val)
            interp.vgprs[lane][2] = SENTINEL_FILL

        # Activate lower half (0x00000000FFFFFFFF): compute v_rcp_f32 into v2
        interp.run_program([
            "s_mov_b64 exec, 0x00000000FFFFFFFF",
            "v_rcp_f32 v2, v0",
        ])

        for lane in range(32):
            expected = ref_rcp_f32(interp.vgprs[lane][0])
            self.assertEqual(interp.vgprs[lane][2], expected, f"Lower lane {lane} failed rcp")
        for lane in range(32, 64):
            self.assertEqual(interp.vgprs[lane][2], SENTINEL_FILL, f"Upper lane {lane} was modified")

        # Activate upper half (0xFFFFFFFF00000000): compute v_sqrt_f32 into v2
        interp.run_program([
            "s_mov_b64 exec, 0xFFFFFFFF00000000",
            "v_sqrt_f32 v2, v0",
        ])

        for lane in range(32):
            expected = ref_rcp_f32(interp.vgprs[lane][0])
            self.assertEqual(interp.vgprs[lane][2], expected, f"Lower lane {lane} was overwritten")
        for lane in range(32, 64):
            expected = ref_sqrt_f32(interp.vgprs[lane][0])
            self.assertEqual(interp.vgprs[lane][2], expected, f"Upper lane {lane} failed sqrt")


class TestTrigTranslationAndExecution(unittest.TestCase):
    """End-to-end translation and execution of v_cos_f32 and v_sin_f32."""

    def test_v_cos_and_v_sin_pixel_shader_execution(self):
        test_turns = [0.0, 0.25, 0.5, 0.75]
        for t in test_turns:
            tb = f32_to_bits(t)
            lines = [
                f"v_mov_b32 v0, 0x{tb:08x}",
                "v_cos_f32 v1, v0",
                "v_sin_f32 v2, v0",
                "exp mrt0 v1, v2, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_trig_{tb:08x}", lines)
            words = support.mrt0_words(spv)
            got_cos = bits_to_f32(words[0])
            got_sin = bits_to_f32(words[1])
            expected_cos = bits_to_f32(ref_cos_f32(tb))
            expected_sin = bits_to_f32(ref_sin_f32(tb))
            self.assertAlmostEqual(got_cos, expected_cos, delta=1e-5)
            self.assertAlmostEqual(got_sin, expected_sin, delta=1e-5)

    def test_large_turns_reduction_in_shader(self):
        for large_turn in (100.3, 254.51):
            tb = f32_to_bits(large_turn)
            lines = [
                f"v_mov_b32 v0, 0x{tb:08x}",
                "v_sin_f32 v1, v0",
                "v_cos_f32 v2, v0",
                "exp mrt0 v1, v2, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_large_trig_{tb:08x}", lines)
            words = support.mrt0_words(spv)
            got_sin = bits_to_f32(words[0])
            got_cos = bits_to_f32(words[1])
            expected_sin = bits_to_f32(ref_sin_f32(tb))
            expected_cos = bits_to_f32(ref_cos_f32(tb))
            self.assertAlmostEqual(got_sin, expected_sin, delta=1e-5)
            self.assertAlmostEqual(got_cos, expected_cos, delta=1e-5)


class TestRcpRsqShaderExecution(unittest.TestCase):
    """End-to-end pixel shader execution for reciprocal and inverse square root."""

    def test_rcp_and_rsq_powers_of_two_shader(self):
        test_values = [1.0, 2.0, 4.0, 16.0, 0.25]
        for val in test_values:
            vb = f32_to_bits(val)
            lines = [
                f"v_mov_b32 v0, 0x{vb:08x}",
                "v_rcp_f32 v1, v0",
                "v_rsq_f32 v2, v0",
                "v_sqrt_f32 v3, v0",
                "exp mrt0 v1, v2, v3, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_rcp_rsq_{vb:08x}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], ref_rcp_f32(vb))
            self.assertEqual(words[1], ref_rsq_f32(vb))
            self.assertEqual(words[2], ref_sqrt_f32(vb))


class TestSpecialFloatBoundaries(unittest.TestCase):
    """Test suite validating special float values: zeros, infinities, subnormals, and NaNs."""

    def test_rcp_special_values(self):
        # rcp(+-0.0) = +-inf
        self.assertEqual(ref_rcp_f32(POS_ZERO), POS_INF)
        self.assertEqual(ref_rcp_f32(NEG_ZERO), NEG_INF)
        # rcp(+-inf) = +-0.0
        self.assertEqual(ref_rcp_f32(POS_INF), POS_ZERO)
        self.assertEqual(ref_rcp_f32(NEG_INF), NEG_ZERO)

    def test_rsq_special_values(self):
        # rsq(+0.0) = +inf
        self.assertEqual(ref_rsq_f32(POS_ZERO), POS_INF)
        # rsq(-x) = qNaN for x > 0
        self.assertTrue(is_f32_nan_bits(ref_rsq_f32(NEG_ONE)))
        self.assertTrue(is_f32_nan_bits(ref_rsq_f32(NEG_TWO)))
        # rsq(+inf) = +0.0
        self.assertEqual(ref_rsq_f32(POS_INF), POS_ZERO)
        # rsq(-inf) = qNaN
        self.assertTrue(is_f32_nan_bits(ref_rsq_f32(NEG_INF)))

    def test_sqrt_special_values(self):
        # sqrt(+0.0) = +0.0, sqrt(-0.0) = -0.0
        self.assertEqual(ref_sqrt_f32(POS_ZERO), POS_ZERO)
        self.assertEqual(ref_sqrt_f32(NEG_ZERO), NEG_ZERO)
        # sqrt(-x) = qNaN for x > 0
        self.assertTrue(is_f32_nan_bits(ref_sqrt_f32(NEG_ONE)))
        self.assertTrue(is_f32_nan_bits(ref_sqrt_f32(NEG_TWO)))
        # sqrt(+inf) = +inf, sqrt(-inf) = qNaN
        self.assertEqual(ref_sqrt_f32(POS_INF), POS_INF)
        self.assertTrue(is_f32_nan_bits(ref_sqrt_f32(NEG_INF)))

    def test_trig_special_values(self):
        # cos(+-0.0) = 1.0
        self.assertEqual(ref_cos_f32(POS_ZERO), ONE)
        self.assertEqual(ref_cos_f32(NEG_ZERO), ONE)
        # sin(+0.0) = +0.0, sin(-0.0) = -0.0
        self.assertEqual(ref_sin_f32(POS_ZERO), POS_ZERO)
        self.assertEqual(ref_sin_f32(NEG_ZERO), NEG_ZERO)
        # cos(+-inf) = qNaN, sin(+-inf) = qNaN
        self.assertTrue(is_f32_nan_bits(ref_cos_f32(POS_INF)))
        self.assertTrue(is_f32_nan_bits(ref_cos_f32(NEG_INF)))
        self.assertTrue(is_f32_nan_bits(ref_sin_f32(POS_INF)))
        self.assertTrue(is_f32_nan_bits(ref_sin_f32(NEG_INF)))

    def test_nan_propagation(self):
        # Both qNaN and sNaN must produce a NaN result across all operations
        for nan_input in (QNAN, NEG_QNAN, SNAN):
            self.assertTrue(is_f32_nan_bits(ref_cos_f32(nan_input)), f"cos({nan_input:#x}) not NaN")
            self.assertTrue(is_f32_nan_bits(ref_sin_f32(nan_input)), f"sin({nan_input:#x}) not NaN")
            self.assertTrue(is_f32_nan_bits(ref_rcp_f32(nan_input)), f"rcp({nan_input:#x}) not NaN")
            self.assertTrue(is_f32_nan_bits(ref_rsq_f32(nan_input)), f"rsq({nan_input:#x}) not NaN")
            self.assertTrue(is_f32_nan_bits(ref_sqrt_f32(nan_input)), f"sqrt({nan_input:#x}) not NaN")

    def test_subnormal_boundaries(self):
        # Reciprocal of tiny subnormals overflows to infinity (2^149 > 2^128)
        self.assertEqual(ref_rcp_f32(SUBNORM_MIN), POS_INF)
        self.assertEqual(ref_rcp_f32(NEG_SUBNORM_MIN), NEG_INF)
        # Inverse sqrt of subnormal min: 1 / sqrt(2^-149) = 2^74.5 fits in f32 normal range
        rsq_subnorm = ref_rsq_f32(SUBNORM_MIN)
        self.assertFalse(is_f32_nan_bits(rsq_subnorm))
        self.assertNotEqual(rsq_subnorm, POS_INF)
        self.assertAlmostEqual(bits_to_f32(rsq_subnorm), 1.0 / math.sqrt(bits_to_f32(SUBNORM_MIN)), delta=1e16)
        # Smallest normal float: 1 / 2^-126 = 2^126 (fits in f32)
        rcp_flt_min = ref_rcp_f32(FLT_MIN)
        self.assertFalse(is_f32_nan_bits(rcp_flt_min))
        self.assertNotEqual(rcp_flt_min, POS_INF)
        self.assertAlmostEqual(bits_to_f32(rcp_flt_min), 2.0 ** 126, delta=1e20)


class TestNegativeControlsAndFaults(unittest.TestCase):
    """Test suite validating negative controls, fault handling, and unsupported constructs."""

    def test_spirv_unsupported_rcp_zero(self):
        # In SPIR-V / spirv_mini_interp, OpFDiv by zero raises UnsupportedConstruct
        for zero_word in (POS_ZERO, NEG_ZERO):
            lines = [
                f"v_mov_b32 v0, 0x{zero_word:08x}",
                "v_rcp_f32 v1, v0",
                "exp mrt0 v1, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_rcp_zero_{zero_word:08x}", lines)
            with self.assertRaises(si.UnsupportedConstruct):
                support.mrt0_words(spv)

    def test_spirv_unsupported_rsq_zero_or_negative(self):
        # In GLSL.std.450 / spirv_mini_interp, InverseSqrt of <= 0 raises UnsupportedConstruct
        for bad_val in (POS_ZERO, NEG_ZERO, NEG_ONE):
            lines = [
                f"v_mov_b32 v0, 0x{bad_val:08x}",
                "v_rsq_f32 v1, v0",
                "exp mrt0 v1, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_rsq_bad_{bad_val:08x}", lines)
            with self.assertRaises(si.UnsupportedConstruct):
                support.mrt0_words(spv)

    def test_simd_interpreter_fault_rejection(self):
        # Disallow unsupported wave sizes
        with self.assertRaises(ValueError):
            Rdna2ValuSimdInterpreter(wave_size=16)
        with self.assertRaises(ValueError):
            Rdna2ValuSimdInterpreter(wave_size=128)

        # Disallow vector destination to a scalar register
        interp = Rdna2ValuSimdInterpreter(wave_size=32)
        with self.assertRaises(ValueError):
            interp.execute_line("v_rcp_f32 s0, v0")

        # Disallow unknown token syntax
        with self.assertRaises(ValueError):
            interp.execute_line("v_mov_b32 v0, invalid_token_xyz")

    def test_negative_control_oracle_assertions(self):
        # Sanity check: assert that intentionally mutated values fail
        with self.assertRaises(AssertionError):
            self.assertEqual(ref_rcp_f32(f32_to_bits(2.0)), f32_to_bits(0.25))

        with self.assertRaises(AssertionError):
            self.assertEqual(ref_rsq_f32(f32_to_bits(4.0)), f32_to_bits(2.0))

        with self.assertRaises(AssertionError):
            self.assertEqual(ref_cos_f32(POS_ZERO), NEG_ONE)


if __name__ == "__main__":
    unittest.main()
