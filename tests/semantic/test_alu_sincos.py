"""Semantic interpreter tests for RDNA2 v_sin_f32 / v_cos_f32 / v_sin_f16 / v_cos_f16 (SHADER-88).

The operand is an angle in TURNS (normalized revolution where 1.0 turn = 2*pi radians).
RDNA2 hardware range-reduces in turns to [-0.5, 0.5] before scaling by 2*pi. Multiplying
by 2*pi before reducing rounds the fractional turn away in binary32 (up to ~1e-4 rad near
+-255 turns), so the translator reduces in turns first.

Each case translates a synthetic shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py, and validates the result against an independent
reference oracle and analytical mathematical identities (including IEEE 754 half-precision
precision bounds and the Pythagorean identity sin^2(x) + cos^2(x) ~= 1.0).

Standard library only: zero game data, no GPU hardware required.
"""
from __future__ import annotations

import gc
import math
import os
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xAAAA5555
TOL32 = 1e-6
TOL16 = 1e-3
TWO_PI = 6.283185307179586

INPUTS_F32 = [0.25, 0.5, 1.0, 100.3, 254.25, 254.51, 255.9, 0.1, 0.0, -0.0, -0.25, -100.3, -254.51, -255.5]

# Representative binary16 encodings
H_POS_ZERO = 0x0000
H_NEG_ZERO = 0x8000
H_POS_INF = 0x7C00
H_NEG_INF = 0xFC00
H_QNAN = 0x7E00
H_SNAN = 0x7C01
H_NEG_QNAN = 0xFE00
H_NEG_SNAN = 0xFC01

H_ONE = 0x3C00
H_NEG_ONE = 0xBC00
H_HALF = 0x3800
H_QUARTER = 0x3400
H_THREE_QUARTERS = 0x3A00
H_EIGHTH = 0x3000


def f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f16_is_nan(h: int) -> bool:
    return (h & 0x7FFF) > 0x7C00


def f16_is_inf(h: int) -> bool:
    return (h & 0x7FFF) == 0x7C00


def f16_is_finite(h: int) -> bool:
    return ((h >> 10) & 0x1F) != 0x1F


def ref_f32(op: str, x: float) -> float:
    return math.sin(2 * math.pi * x) if "sin" in op else math.cos(2 * math.pi * x)


def reduce_turn_angle(x: float) -> float:
    """Range-reduce turn angle to [-0.5, 0.5] turns and scale by 2*pi in float32.

    Matches RDNA2 hardware semantics and rdna2spirv.turn_angle lowering:
    preserves signed zero when x == 0.0, uses RoundEven for symmetric tie-breaking,
    and multiplies by 2*pi in IEEE 754 binary32.
    """
    xf = f32(x)
    if xf == 0.0:
        t = xf
    else:
        rnd = f32(round(xf))
        t = f32(xf - rnd)
    return f32(t * TWO_PI)


def oracle_sincos_f16(
    op: str,
    h: int,
    *,
    abs_: bool = False,
    neg_: bool = False,
    clamp_: bool = False,
) -> int | None:
    """Independent reference oracle for RDNA2 v_sin_f16 and v_cos_f16.

    Takes raw 16-bit integer representation of IEEE 754 half-precision float operand.
    Applies VOP3 source modifiers (abs before neg), turn angle reduction, transcendental
    evaluation, VOP3 clamp, and IEEE 754 round-to-nearest-even halfword packing.
    Returns 16-bit unsigned integer bit pattern or None for non-finite values.
    """
    h &= 0xFFFF
    if abs_:
        h &= 0x7FFF
    if neg_:
        h ^= 0x8000
    if not f16_is_finite(h):
        return None
    val_f = h2f(h)
    # The hardware gives an exact +0 for sin at an odd half turn and cos at an odd quarter turn, where Sin/Cos of a
    # rounded 2*pi give a 1-ulp denormal (SHADER-134, docs/design.md).
    if val_f != 0.0 and abs(val_f - round(val_f)) == (0.5 if "sin" in op else 0.25):
        return 0
    ang = reduce_turn_angle(val_f)
    res_f = f32(math.sin(ang) if "sin" in op else math.cos(ang))
    if clamp_:
        res_f = max(0.0, min(1.0, res_f))
    return f2h(res_f)


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None
_CS_PATH: str | None = None
_CS_FSSM: str | None = None
VSHARP_CS = [0, 0, 0xFFFFFFFF, 0x31016FAC]


class _Heap(dict):
    """Word-addressable byte heap for compute shader buffer load/store emulation."""

    def read(self, addr: int, size: int) -> bytes:
        data = bytearray(size)
        for offset in range(0, size, 4):
            word = self.get((addr + offset) & 0xFFFFFFFF, 0)
            struct.pack_into("<I", data, offset, word & 0xFFFFFFFF)
        return bytes(data)

    def write(self, addr: int, data: bytes) -> None:
        for offset in range(0, len(data), 4):
            chunk = data[offset : offset + 4]
            if len(chunk) < 4:
                chunk = chunk.ljust(4, b"\x00")
            word = struct.unpack_into("<I", chunk)[0]
            self[(addr + offset) & 0xFFFFFFFF] = word


def _make_fssm(rsrc2: int, group: tuple[int, int, int]) -> bytes:
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH, _CS_PATH, _CS_FSSM
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    fssm_dir = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_sincos.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_sincos.s")
    _CS_FSSM = os.path.join(fssm_dir, "cs_sincos.fssm")
    with open(os.path.join(_TMP_DIR.name, "ps_sincos.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH, _CS_PATH, _CS_FSSM
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None
        _CS_PATH = None
        _CS_FSSM = None


def run_cs(lines: list[str], group_size: int, memory: dict[int, int]) -> None:
    if _TMP_DIR is None:
        setUpModule()
    with open(_CS_FSSM, "wb") as f:
        f.write(_make_fssm(0x0088, (group_size, 1, 1)))
    asm = "; cs_sincos\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_CS_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_CS_PATH)
    si.run_workgroup(
        spv,
        group_size,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
        push_constants=VSHARP_CS + [0] * 28,
        memory=memory,
    )


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_sincos\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return spv


def run_ps(lines: list[str]) -> list[int]:
    spv = translate_ps(lines)
    return list(si.run_fragment(spv)["%out_mrt0"])


def run(line: str, v1: int, old: int = 0) -> int:
    return run_ps([
        f"v_mov_b32 v1, {v1:#x}",
        f"v_mov_b32 v0, {old:#x}",
        line,
    ])[0]


class SinCosF32Tests(unittest.TestCase):
    """Original single-precision tests preserved for regression defense."""

    def test_values_against_double_reference(self):
        for op in ("v_sin_f32", "v_cos_f32"):
            for x in INPUTS_F32:
                x32 = u2f(f2u(x))
                with self.subTest(op=op, x=x):
                    got = u2f(run(f"{op}_e32 v0, v1", f2u(x)))
                    self.assertLessEqual(abs(got - ref_f32(op, x32)), TOL32)

    def test_exact_quarter_turns(self):
        self.assertLessEqual(abs(u2f(run("v_sin_f32_e32 v0, v1", f2u(0.25))) - 1.0), TOL32)
        self.assertLessEqual(abs(u2f(run("v_cos_f32_e32 v0, v1", f2u(0.5))) + 1.0), TOL32)
        self.assertLessEqual(abs(u2f(run("v_sin_f32_e32 v0, v1", f2u(254.25))) - 1.0), TOL32)
        self.assertLessEqual(abs(u2f(run("v_cos_f32_e32 v0, v1", f2u(-254.5))) + 1.0), TOL32)

    def test_worst_case_of_the_issue(self):
        x32 = u2f(f2u(254.51))
        got = u2f(run("v_sin_f32_e32 v0, v1", f2u(254.51)))
        self.assertLessEqual(abs(got - math.sin(2 * math.pi * x32)), TOL32)

    def test_signed_zero_and_special_values(self):
        self.assertEqual(run("v_sin_f32_e32 v0, v1", 0x00000000), 0x00000000)
        self.assertEqual(run("v_sin_f32_e32 v0, v1", 0x80000000), 0x80000000)
        self.assertEqual(u2f(run("v_cos_f32_e32 v0, v1", 0x80000000)), 1.0)
        for bits in (0x7F800000, 0xFF800000, 0x7FC00000):
            for op in ("v_sin_f32", "v_cos_f32"):
                with self.subTest(op=op, x=hex(bits)):
                    self.assertTrue(math.isnan(u2f(run(f"{op}_e32 v0, v1", bits))))

    def test_modifiers(self):
        got = u2f(run("v_sin_f32_e64 v0, -v1", f2u(100.3)))
        self.assertLessEqual(abs(got - math.sin(-2 * math.pi * u2f(f2u(100.3)))), TOL32)
        got = u2f(run("v_cos_f32_e64 v0, |v1|", f2u(-254.51)))
        self.assertLessEqual(abs(got - math.cos(2 * math.pi * u2f(f2u(254.51)))), TOL32)
        got = u2f(run("v_sin_f32_e64 v0, v1 clamp", f2u(-0.25)))
        self.assertEqual(got, 0.0)
        got = u2f(run("v_sin_f32_e64 v0, v1 mul:2", f2u(100.3)))
        self.assertLessEqual(abs(got - 2 * math.sin(2 * math.pi * u2f(f2u(100.3)))), 2 * TOL32)

    def test_whole_turns_give_plus_zero_for_either_sign(self):
        for x in (-1.0, -2.0, -100.0, 1.0, 2.0, 100.0):
            with self.subTest(x=x):
                self.assertEqual(run("v_sin_f32_e32 v0, v1", f2u(x)), 0x00000000)
                self.assertEqual(run("v_cos_f32_e32 v0, v1", f2u(x)), f2u(1.0))

    def test_tiny_arguments_keep_their_sign_and_size(self):
        for x in (-2.0 ** -26, 2.0 ** -26, -2.0 ** -20, 2.0 ** -20):
            with self.subTest(x=x):
                got = u2f(run("v_sin_f32_e32 v0, v1", f2u(x)))
                self.assertEqual(math.copysign(1.0, got), math.copysign(1.0, x))
                self.assertNotEqual(got, 0.0)
                self.assertLessEqual(abs(got - math.sin(2 * math.pi * x)), 2e-9)
                self.assertLessEqual(abs(u2f(run("v_cos_f32_e32 v0, v1", f2u(x))) - 1.0), 1e-6)


class SinCosF16ReferenceOracleTests(unittest.TestCase):
    """Tests the reference oracle against mathematical truth and validates the translator against the oracle."""

    def test_oracle_exact_cardinal_turns(self):
        # 0.0 turns (0 rad) -> sin = 0.0, cos = 1.0
        self.assertEqual(oracle_sincos_f16("v_sin_f16", H_POS_ZERO), H_POS_ZERO)
        self.assertEqual(oracle_sincos_f16("v_cos_f16", H_POS_ZERO), H_ONE)

        # 0.25 turns (pi/2 rad) -> sin = 1.0, cos ~= 0.0 (subnormal negative due to f32 2*pi approximation)
        self.assertEqual(oracle_sincos_f16("v_sin_f16", H_QUARTER), H_ONE)
        self.assertLessEqual(abs(h2f(oracle_sincos_f16("v_cos_f16", H_QUARTER))), TOL16)

        # 0.5 turns (pi rad) -> sin ~= 0.0, cos = -1.0
        self.assertLessEqual(abs(h2f(oracle_sincos_f16("v_sin_f16", H_HALF))), TOL16)
        self.assertEqual(oracle_sincos_f16("v_cos_f16", H_HALF), H_NEG_ONE)

        # 0.75 turns (3pi/2 rad) -> sin = -1.0, cos ~= 0.0
        self.assertEqual(oracle_sincos_f16("v_sin_f16", H_THREE_QUARTERS), H_NEG_ONE)
        self.assertLessEqual(abs(h2f(oracle_sincos_f16("v_cos_f16", H_THREE_QUARTERS))), TOL16)

        # 1.0 turns (2pi rad) -> sin = 0.0, cos = 1.0
        self.assertEqual(oracle_sincos_f16("v_sin_f16", H_ONE), H_POS_ZERO)
        self.assertEqual(oracle_sincos_f16("v_cos_f16", H_ONE), H_ONE)

    def test_oracle_symmetry_and_periodicity(self):
        # Fractional turns strictly inside (-0.5, 0.5) modulo 1.0 (avoiding round-even tie at +-0.5)
        turns = [0.125, 0.25, 0.375, 0.625, 0.75, 0.875]
        for t in turns:
            h = f2h(t)
            h_plus_1 = f2h(t + 1.0)
            h_plus_10 = f2h(t + 10.0)

            # Periodicity: f(x + 1 turn) == f(x)
            self.assertEqual(oracle_sincos_f16("v_sin_f16", h), oracle_sincos_f16("v_sin_f16", h_plus_1))
            self.assertEqual(oracle_sincos_f16("v_cos_f16", h), oracle_sincos_f16("v_cos_f16", h_plus_1))
            self.assertEqual(oracle_sincos_f16("v_sin_f16", h), oracle_sincos_f16("v_sin_f16", h_plus_10))
            self.assertEqual(oracle_sincos_f16("v_cos_f16", h), oracle_sincos_f16("v_cos_f16", h_plus_10))

            # Odd / Even symmetries: sin(-x) == -sin(x), cos(-x) == cos(x)
            s_pos = h2f(oracle_sincos_f16("v_sin_f16", h))
            s_neg = h2f(oracle_sincos_f16("v_sin_f16", f2h(-t)))
            self.assertLessEqual(abs(s_pos + s_neg), TOL16)

            c_pos = h2f(oracle_sincos_f16("v_cos_f16", h))
            c_neg = h2f(oracle_sincos_f16("v_cos_f16", f2h(-t)))
            self.assertLessEqual(abs(c_pos - c_neg), TOL16)

    def test_normalized_turns_against_interpreter(self):
        # Sweep normalized turns x in [0.0, 1.0) with step 1/32
        steps = 32
        for i in range(steps):
            turn = i / steps
            h_in = f2h(turn)
            for op in ("v_sin_f16", "v_cos_f16"):
                want = oracle_sincos_f16(op, h_in)
                got_word = run(f"{op} v0, v1", (OLD & 0xFFFF0000) | h_in, old=OLD)
                got_h = got_word & 0xFFFF
                with self.subTest(op=op, turn=turn, h_in=hex(h_in)):
                    self.assertEqual(got_h, want)
                    # Verify high half preservation
                    self.assertEqual(got_word & 0xFFFF0000, OLD & 0xFFFF0000)

    def test_half_precision_precision_limits_and_ulp(self):
        # Verify precision against double-precision math.sin/math.cos within f16 bounds
        test_turns = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        for t in test_turns:
            h_in = f2h(t)
            actual_t = h2f(h_in)  # The shader executes on the binary16 operand
            for op in ("v_sin_f16", "v_cos_f16"):
                got_val = h2f(run(f"{op} v0, v1", h_in))
                exact_math = math.sin(2 * math.pi * actual_t) if "sin" in op else math.cos(2 * math.pi * actual_t)
                with self.subTest(op=op, turn=t):
                    self.assertLessEqual(abs(got_val - exact_math), TOL16)

    def test_pythagorean_identity_f16(self):
        # sin^2(x) + cos^2(x) ~= 1.0 across [0.0, 1.0)
        for i in range(33):
            turn = i / 32.0
            h_in = f2h(turn)
            sin_val = h2f(run("v_sin_f16 v0, v1", h_in))
            cos_val = h2f(run("v_cos_f16 v0, v1", h_in))
            pyth = sin_val * sin_val + cos_val * cos_val
            with self.subTest(turn=turn):
                self.assertLessEqual(abs(pyth - 1.0), 2e-3)

    def test_vop3_modifiers(self):
        x = 0.3
        h_in = f2h(x)
        old_val = 0xBEEF1234

        # Negate source modifier: sin(-x)
        got = run("v_sin_f16_e64 v0, -v1", h_in, old=old_val)
        want = oracle_sincos_f16("v_sin_f16", h_in, neg_=True)
        self.assertEqual(got & 0xFFFF, want)
        self.assertEqual(got & 0xFFFF0000, old_val & 0xFFFF0000)

        # Absolute value modifier: cos(|-x|) == cos(x)
        got = run("v_cos_f16_e64 v0, |v1|", f2h(-x), old=old_val)
        want = oracle_sincos_f16("v_cos_f16", h_in)
        self.assertEqual(got & 0xFFFF, want)
        self.assertEqual(got & 0xFFFF0000, old_val & 0xFFFF0000)

        # Negated absolute value: sin(-|x|)
        got = run("v_sin_f16_e64 v0, -|v1|", h_in, old=old_val)
        want = oracle_sincos_f16("v_sin_f16", h_in, abs_=True, neg_=True)
        self.assertEqual(got & 0xFFFF, want)

        # Clamp output modifier: sin(0.75 turns) = -1.0 -> clamp to 0.0
        got = run("v_sin_f16_e64 v0, v1 clamp", H_THREE_QUARTERS, old=old_val)
        want = oracle_sincos_f16("v_sin_f16", H_THREE_QUARTERS, clamp_=True)
        self.assertEqual(got & 0xFFFF, want)
        self.assertEqual(h2f(got & 0xFFFF), 0.0)

    def test_instruction_forms(self):
        h_in = f2h(0.125)
        for form in ("", "_e32", "_e64"):
            with self.subTest(op="v_sin_f16", form=form or "bare"):
                got = run(f"v_sin_f16{form} v0, v1", h_in, old=OLD)
                self.assertEqual(got & 0xFFFF, oracle_sincos_f16("v_sin_f16", h_in))
                self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)

            with self.subTest(op="v_cos_f16", form=form or "bare"):
                got = run(f"v_cos_f16{form} v0, v1", h_in, old=OLD)
                self.assertEqual(got & 0xFFFF, oracle_sincos_f16("v_cos_f16", h_in))
                self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)


class SinCosSIMDAndPackedHalfwordTests(unittest.TestCase):
    """Semantic tests for packed 16-bit register execution and SIMD multi-lane divergence."""

    def test_packed_dual_channel_repacking(self):
        # Two independent angles packed in v1: hi = 0.25 (pi/2), lo = 0.0
        v1_val = (H_QUARTER << 16) | H_POS_ZERO
        lines = [
            f"v_mov_b32 v1, {v1_val:#x}",
            "v_sin_f16 v2, v1",           # sin(lo=0.0) -> 0.0 in v2.lo
            "v_lshrrev_b32 v3, 16, v1",   # hi=0.25 in v3.lo
            "v_cos_f16 v4, v3",           # cos(hi=0.25) -> +0 in v4.lo (SHADER-134)
            "v_pack_b32_f16 v0, v2, v4",  # packed [cos(hi), sin(lo)]
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_ps(lines)
        word = si.run_fragment(spv)["%out_mrt0"][0]
        lo_got = word & 0xFFFF
        hi_got = (word >> 16) & 0xFFFF
        self.assertEqual(lo_got, oracle_sincos_f16("v_sin_f16", H_POS_ZERO))
        self.assertEqual(hi_got, oracle_sincos_f16("v_cos_f16", H_QUARTER))

        # Test pair 2: hi = 1.0 (2pi), lo = 0.5 (pi) -> [cos(1.0)=1.0, sin(0.5)=~0.0]
        v1_val2 = (H_ONE << 16) | H_HALF
        lines2 = [
            f"v_mov_b32 v1, {v1_val2:#x}",
            "v_sin_f16 v2, v1",
            "v_lshrrev_b32 v3, 16, v1",
            "v_cos_f16 v4, v3",
            "v_pack_b32_f16 v0, v2, v4",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv2 = translate_ps(lines2)
        word2 = si.run_fragment(spv2)["%out_mrt0"][0]
        self.assertEqual(word2 & 0xFFFF, oracle_sincos_f16("v_sin_f16", H_HALF))
        self.assertEqual((word2 >> 16) & 0xFFFF, oracle_sincos_f16("v_cos_f16", H_ONE))

    def test_packed_destination_high_half_preservation(self):
        sentinel_high = 0xCAFE0000
        for op in ("v_sin_f16", "v_cos_f16"):
            for turn in (0.0, 0.25, 0.5, 0.75):
                h_in = f2h(turn)
                got = run(f"{op} v0, v1", h_in, old=sentinel_high | 0x1234)
                with self.subTest(op=op, turn=turn):
                    self.assertEqual(got & 0xFFFF0000, sentinel_high)
                    self.assertEqual(got & 0xFFFF, oracle_sincos_f16(op, h_in))

    def test_packed_destination_aliasing(self):
        # In-place destination aliasing: dst == src (v0, v0)
        # v0 holds (0xFEED << 16) | H_QUARTER
        v0_init = (0xFEED << 16) | H_QUARTER
        lines = [
            f"v_mov_b32 v0, {v0_init:#x}",
            "v_sin_f16 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        got = si.run_fragment(translate_ps(lines))["%out_mrt0"][0]
        self.assertEqual(got & 0xFFFF0000, 0xFEED0000)
        self.assertEqual(got & 0xFFFF, H_ONE)

        # Re-evaluating with cos: cos(1.0 turns) == 1.0 (0x3c00)
        lines_cos = [
            f"v_mov_b32 v0, {got:#x}",
            "v_cos_f16 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        got_cos = si.run_fragment(translate_ps(lines_cos))["%out_mrt0"][0]
        self.assertEqual(got_cos & 0xFFFF0000, 0xFEED0000)
        self.assertEqual(got_cos & 0xFFFF, H_ONE)

    def test_packed_quadrature_carrier(self):
        # Angle 45 deg (0.125 turns): sin(45) == cos(45) == sqrt(2)/2 ~= 0.7071
        lines = [
            f"v_mov_b32 v1, {H_EIGHTH:#x}",
            "v_sin_f16 v2, v1",
            "v_cos_f16 v3, v1",
            "v_pack_b32_f16 v0, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        word = si.run_fragment(translate_ps(lines))["%out_mrt0"][0]
        s_val = h2f(word & 0xFFFF)
        c_val = h2f((word >> 16) & 0xFFFF)
        self.assertLessEqual(abs(s_val - c_val), TOL16)
        self.assertLessEqual(abs(s_val - (math.sqrt(2.0) / 2.0)), TOL16)

    def test_exec_masking_single_lane(self):
        # When exec_lo == 0, VALU instruction does not execute and destination register keeps old value
        lines_inactive = [
            f"v_mov_b32 v1, {H_QUARTER:#x}",
            "v_mov_b32 v0, 0xCAFEBABE",
            "s_mov_b32 exec_lo, 0",
            "v_sin_f16 v0, v1",
            "s_mov_b32 exec_lo, 1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        got_inactive = si.run_fragment(translate_ps(lines_inactive))["%out_mrt0"][0]
        self.assertEqual(got_inactive, 0xCAFEBABE)

        # When exec_lo == 1, instruction executes normally
        lines_active = [
            f"v_mov_b32 v1, {H_QUARTER:#x}",
            "v_mov_b32 v0, 0xCAFEBABE",
            "s_mov_b32 exec_lo, 1",
            "v_sin_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        got_active = si.run_fragment(translate_ps(lines_active))["%out_mrt0"][0]
        self.assertEqual(got_active & 0xFFFF0000, 0xCAFE0000)
        self.assertEqual(got_active & 0xFFFF, H_ONE)

    def test_exec_masking_multi_lane_divergence(self):
        # 8-thread workgroup: even lanes active (sin of 0.25 turns -> 1.0), odd lanes masked off
        group_size = 8
        lane_inputs = [H_QUARTER] * group_size
        lane_olds = [0x55550000 + i for i in range(group_size)]

        mem = _Heap(
            {0x100 + 4 * i: inp for i, inp in enumerate(lane_inputs)}
            | {0x200 + 4 * i: old for i, old in enumerate(lane_olds)}
        )

        lines = [
            "v_lshlrev_b32_e32 v5, 2, v0",
            "v_add_nc_u32_e32 v6, 0x100, v5",
            "v_add_nc_u32_e32 v7, 0x200, v5",
            "buffer_load_dword v1, v6, s[0:3], 0 offen",
            "buffer_load_dword v2, v7, s[0:3], 0 offen",
            "v_and_b32_e32 v3, 1, v0",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v3",
            "s_and_saveexec_b32 s6, vcc_lo",
            "v_sin_f16 v2, v1",
            "s_mov_b32 exec_lo, s6",
            "buffer_store_dword v2, v5, s[0:3], 0 offen",
        ]
        run_cs(lines, group_size=group_size, memory=mem)

        got = [mem.get(4 * i) for i in range(group_size)]
        want = []
        for i in range(group_size):
            if i % 2 == 0:
                want.append((lane_olds[i] & 0xFFFF0000) | H_ONE)
            else:
                want.append(lane_olds[i])
        self.assertEqual(got, want)

    def test_exec_masking_multi_lane_partitioning(self):
        # 8-thread workgroup: lanes 0..3 compute sin(0.25)=1.0, lanes 4..7 compute cos(0.25)=~0.0
        group_size = 8
        lane_inputs = [H_QUARTER] * group_size
        mem = _Heap({0x100 + 4 * i: inp for i, inp in enumerate(lane_inputs)})

        lines = [
            "v_lshlrev_b32_e32 v5, 2, v0",
            "v_add_nc_u32_e32 v6, 0x100, v5",
            "buffer_load_dword v1, v6, s[0:3], 0 offen",
            "v_mov_b32 v2, 0x33330000",
            "v_cmp_lt_u32_e32 vcc_lo, v0, 4",
            "s_and_saveexec_b32 s6, vcc_lo",
            "v_sin_f16 v2, v1",
            "s_not_b32 exec_lo, exec_lo",
            "s_and_b32 exec_lo, exec_lo, s6",
            "v_cos_f16 v2, v1",
            "s_mov_b32 exec_lo, s6",
            "buffer_store_dword v2, v5, s[0:3], 0 offen",
        ]
        run_cs(lines, group_size=group_size, memory=mem)

        got = [mem.get(4 * i) for i in range(group_size)]
        want = []
        for i in range(group_size):
            if i < 4:
                want.append(0x33330000 | H_ONE)
            else:
                want.append(0x33330000 | oracle_sincos_f16("v_cos_f16", H_QUARTER))
        self.assertEqual(got, want)


class SinCosSpecialFloatBoundariesAndNegativeControlsTests(unittest.TestCase):
    """Semantic tests for special float boundaries (signed zeros, inf, nan) and negative controls."""

    def test_signed_zeros(self):
        # sin(+0.0) == +0.0 (0x0000), sin(-0.0) == -0.0 (0x8000)
        # cos(+0.0) == 1.0 (0x3C00), cos(-0.0) == 1.0 (0x3C00)
        # Upper 16 bits of destination register must be preserved.
        got_sin_pos = run("v_sin_f16 v0, v1", H_POS_ZERO, old=OLD)
        self.assertEqual(got_sin_pos & 0xFFFF, H_POS_ZERO)
        self.assertEqual(got_sin_pos & 0xFFFF0000, OLD & 0xFFFF0000)

        got_sin_neg = run("v_sin_f16 v0, v1", H_NEG_ZERO, old=OLD)
        self.assertEqual(got_sin_neg & 0xFFFF, H_NEG_ZERO)
        self.assertEqual(got_sin_neg & 0xFFFF0000, OLD & 0xFFFF0000)

        got_cos_pos = run("v_cos_f16 v0, v1", H_POS_ZERO, old=OLD)
        self.assertEqual(got_cos_pos & 0xFFFF, H_ONE)
        self.assertEqual(got_cos_pos & 0xFFFF0000, OLD & 0xFFFF0000)

        got_cos_neg = run("v_cos_f16 v0, v1", H_NEG_ZERO, old=OLD)
        self.assertEqual(got_cos_neg & 0xFFFF, H_ONE)
        self.assertEqual(got_cos_neg & 0xFFFF0000, OLD & 0xFFFF0000)

    def test_infinities_produce_nan(self):
        # cos(+-inf) == NaN, sin(+-inf) == NaN
        for op in ("v_sin_f16", "v_cos_f16"):
            for inf_val in (H_POS_INF, H_NEG_INF):
                with self.subTest(op=op, inf=hex(inf_val)):
                    got = run(f"{op} v0, v1", inf_val, old=OLD)
                    self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)
                    self.assertTrue(f16_is_nan(got & 0xFFFF))
                    self.assertTrue(math.isnan(h2f(got & 0xFFFF)))

    def test_nan_propagation(self):
        # Quiet and signaling NaNs propagate as NaN
        nan_inputs = [H_QNAN, H_SNAN, H_NEG_QNAN, H_NEG_SNAN, 0x7E55, 0xFEAA]
        for op in ("v_sin_f16", "v_cos_f16"):
            for nan_val in nan_inputs:
                with self.subTest(op=op, nan=hex(nan_val)):
                    got = run(f"{op} v0, v1", nan_val, old=OLD)
                    self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)
                    self.assertTrue(f16_is_nan(got & 0xFFFF))

    def test_subnormal_angles_preserve_sign(self):
        # Smallest subnormal +2^-24 (0x0001) and -2^-24 (0x8001)
        # sin(+2^-24) produces positive subnormal 0x0006, cos(+2^-24) produces 1.0 (0x3C00)
        got_sin_sub = run("v_sin_f16 v0, v1", 0x0001, old=OLD)
        self.assertEqual(got_sin_sub & 0xFFFF, 0x0006)
        self.assertEqual(got_sin_sub & 0xFFFF0000, OLD & 0xFFFF0000)

        got_sin_sub_neg = run("v_sin_f16 v0, v1", 0x8001, old=OLD)
        self.assertEqual(got_sin_sub_neg & 0xFFFF, 0x8006)
        self.assertEqual(got_sin_sub_neg & 0xFFFF0000, OLD & 0xFFFF0000)

        got_cos_sub = run("v_cos_f16 v0, v1", 0x0001, old=OLD)
        self.assertEqual(got_cos_sub & 0xFFFF, H_ONE)

        got_cos_sub_neg = run("v_cos_f16 v0, v1", 0x8001, old=OLD)
        self.assertEqual(got_cos_sub_neg & 0xFFFF, H_ONE)

    def test_whole_integer_turns(self):
        # Non-zero integer turns N in [-100, 100] produce +0.0 (0x0000) for sin, 1.0 (0x3C00) for cos
        whole_turns = [-100.0, -10.0, -2.0, -1.0, 1.0, 2.0, 10.0, 100.0]
        for n in whole_turns:
            h_in = f2h(n)
            with self.subTest(n=n, h_in=hex(h_in)):
                got_sin = run("v_sin_f16 v0, v1", h_in, old=OLD)
                self.assertEqual(got_sin & 0xFFFF, H_POS_ZERO)
                self.assertEqual(got_sin & 0xFFFF0000, OLD & 0xFFFF0000)

                got_cos = run("v_cos_f16 v0, v1", h_in, old=OLD)
                self.assertEqual(got_cos & 0xFFFF, H_ONE)
                self.assertEqual(got_cos & 0xFFFF0000, OLD & 0xFFFF0000)

    def test_negative_control_spirv_opcode_sabotage(self):
        # Sabotaging OpExtInst %glsl Sin with Cos or vice-versa causes detectable divergence
        spv_sin = translate_ps([
            f"v_mov_b32 v1, {H_QUARTER:#x}",
            "v_sin_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        expected_sin = H_ONE
        self.assertEqual(si.run_fragment(spv_sin)["%out_mrt0"][0] & 0xFFFF, expected_sin)

        sabotaged_sin, n = re.subn(r"\bSin\b", "Cos", spv_sin, count=1)
        self.assertEqual(n, 1)
        sabotaged_out = si.run_fragment(sabotaged_sin)["%out_mrt0"][0] & 0xFFFF
        self.assertNotEqual(sabotaged_out, expected_sin)

        spv_cos = translate_ps([
            f"v_mov_b32 v1, {H_POS_ZERO:#x}",
            "v_cos_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        expected_cos = H_ONE
        self.assertEqual(si.run_fragment(spv_cos)["%out_mrt0"][0] & 0xFFFF, expected_cos)

        sabotaged_cos, n = re.subn(r"\bCos\b", "Sin", spv_cos, count=1)
        self.assertEqual(n, 1)
        sabotaged_cos_out = si.run_fragment(sabotaged_cos)["%out_mrt0"][0] & 0xFFFF
        self.assertNotEqual(sabotaged_cos_out, expected_cos)

    def test_negative_control_turn_scaling_sabotage(self):
        # Sabotaging the 2*pi scaling factor (1086918619 -> 1.0 = 1065353216) fails the check
        spv = translate_ps([
            f"v_mov_b32 v1, {H_QUARTER:#x}",
            "v_sin_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertIn("1086918619", spv)
        expected = H_ONE
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0] & 0xFFFF, expected)

        sabotaged = spv.replace("1086918619", "1065353216")
        sabotaged_out = si.run_fragment(sabotaged)["%out_mrt0"][0] & 0xFFFF
        self.assertNotEqual(sabotaged_out, expected)

    def test_negative_control_high_half_preservation_failure(self):
        # A hypothetical buggy translator that clobbers the upper half would be detected
        spv = translate_ps([
            f"v_mov_b32 v1, {H_QUARTER:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "v_sin_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        actual_out = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(actual_out & 0xFFFF0000, OLD & 0xFFFF0000)

        corrupted_out = actual_out & 0x0000FFFF
        self.assertNotEqual(corrupted_out & 0xFFFF0000, OLD & 0xFFFF0000)

    def test_negative_control_nan_assertion(self):
        # Ensure our NaN detection logic correctly distinguishes non-NaN values
        self.assertTrue(f16_is_nan(H_QNAN))
        self.assertTrue(f16_is_nan(H_SNAN))
        self.assertFalse(f16_is_nan(H_ONE))
        self.assertFalse(f16_is_nan(H_POS_ZERO))
        self.assertFalse(f16_is_nan(H_POS_INF))


if __name__ == "__main__":
    unittest.main()
