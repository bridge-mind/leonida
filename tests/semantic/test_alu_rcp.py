"""Semantic interpreter tests for RDNA2 v_rcp_f32 and v_rsq_f32 approximations.

The tests model reciprocal and reciprocal square root floating-point approximations:
  - v_rcp_f32: D.f32 = 1.0 / S0.f32, emitted in SPIR-V as OpFDiv %float %f1 %x.
  - v_rsq_f32: D.f32 = 1.0 / sqrt(S0.f32), emitted in SPIR-V as OpExtInst %float %glsl InverseSqrt %x.

Approximation bounds and ISA semantics (AMD RDNA2 ISA / GLSL.std.450 / SPIR-V):
  - Bit-exactness on power-of-two denominators:
      For v_rcp_f32, exact powers of 2 (such as 1.0, 2.0, 4.0, 0.5, 0.25, 0.125, etc.) have bit-exact
      binary32 reciprocals without rounding loss.
      For v_rsq_f32, exact powers of 4 (such as 1.0, 4.0, 16.0, 64.0, 0.25, 0.0625, etc.) have bit-exact
      binary32 reciprocal square roots.
  - General finite approximation envelope:
      For general non-power-of-two inputs, results are compared against reference double precision
      computations (1.0 / x and 1.0 / math.sqrt(x)) within single-precision tolerance (relative error
      <= 1.5e-7, corresponding to <= 1 ULP round-to-nearest float32 precision).
  - Special floating-point values:
      v_rcp_f32: +inf -> +0.0, -inf -> -0.0, NaN -> NaN.
      v_rsq_f32: +inf -> 0.0, NaN -> NaN.
  - Domain exceptions:
      SPIR-V leaves OpFDiv by zero (+-0.0) undefined; spirv_mini_interp raises
      si.UnsupportedConstruct("OpFDiv by zero (undefined in SPIR-V)").
      GLSL.std.450 leaves InverseSqrt of x <= 0 (+-0.0, negative values, -inf) undefined;
      spirv_mini_interp raises si.UnsupportedConstruct.
  - Instruction encodings and modifiers:
      Both VOP1 (_e32) and VOP3 (_e64) encodings are tested with VGPR, SGPR, literal, inline constants,
      input modifiers (-S0, |S0|, -|S0|), output clamp, and output scale modifiers (mul:2, mul:4, div:2).
      In particular, applying |S0| or -S0 to a negative operand enables v_rsq_f32 to execute validly.
  - Execution masking:
      Inactive lanes (EXEC=0) preserve their destination register value; multilane compute execution
      verifies masked lanes across the wave.
  - Negative control tests:
      Sabotaged SPIR-V instructions are asserted to fail the checks.

Standard library Python only; hand-written RDNA2 listings only. Zero game data.
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

OLD = 0xBEEF1234
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]

# Common float32 test constants
POS_ZERO_BITS = 0x00000000
NEG_ZERO_BITS = 0x80000000
POS_INF_BITS = 0x7F800000
NEG_INF_BITS = 0xFF800000
QNAN_BITS = 0x7FC00000
SNAN_BITS = 0x7F800001
NEG_QNAN_BITS = 0xFFC00000

# Relative tolerance envelope for float32 approximations (1 ULP envelope: ~1.192e-7)
TOL_F32_REL = 1.5e-7

_TMP_DIR: tempfile.TemporaryDirectory | None = None
_PS_PATH: str | None = None
_CS_PATH: str | None = None
_CS_FSSM: str | None = None


def f2u(x: float) -> int:
    """Float to unsigned 32-bit integer bits."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(u: int) -> float:
    """Unsigned 32-bit integer bits to float."""
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def is_f32_nan(bits: int) -> bool:
    """Check if binary32 bits represent a NaN."""
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def _make_fssm(rsrc2: int, group: tuple[int, int, int]) -> bytes:
    """Construct minimal compute shader FSSM header with workgroup dimensions."""
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


def setUpModule() -> None:
    global _TMP_DIR, _PS_PATH, _CS_PATH, _CS_FSSM
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    fssm_dir = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)
    _PS_PATH = os.path.join(disasm_dir, "ps_alu_rcp.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_alu_rcp.s")
    _CS_FSSM = os.path.join(fssm_dir, "cs_alu_rcp.fssm")
    with open(os.path.join(_TMP_DIR.name, "ps_alu_rcp.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _PS_PATH, _CS_PATH, _CS_FSSM
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _PS_PATH = None
        _CS_PATH = None
        _CS_FSSM = None


def translate_ps(lines: list[str]) -> str:
    """Translate hand-written pixel shader instruction lines to SPIR-V assembly text."""
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_ps_alu_rcp\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    assert _PS_PATH is not None
    with open(_PS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
        gc.collect()
    return spv


def run_ps_v0(lines: list[str]) -> int:
    """Run translated pixel shader lines and return the dword written to v0 (mrt0)."""
    return si.run_fragment(translate_ps(lines))["%out_mrt0"][0]


def run_op(line: str, v1: int, old: int = OLD, s4: int = 0) -> int:
    """Execute line with v1 seeded, destination v0 initialized to old, and s4 seeded."""
    return run_ps_v0([
        f"s_mov_b32 s4, {s4:#x}",
        f"v_mov_b32 v1, {v1:#x}",
        f"v_mov_b32 v0, {old:#x}",
        line,
    ])


def run_cs(lines: list[str], group_size: int, subgroup_size: int, memory: dict[int, int]) -> dict[int, int]:
    """Execute compute shader lines across a workgroup and return buffer memory writes."""
    if _TMP_DIR is None:
        setUpModule()
    assert _CS_FSSM is not None and _CS_PATH is not None
    with open(_CS_FSSM, "wb") as f:
        f.write(_make_fssm(0x0088, (group_size, 1, 1)))
    asm = "; test_cs_alu_rcp\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_CS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_CS_PATH)
        gc.collect()
    return si.run_workgroup(
        spv,
        group_size,
        subgroup_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
        push_constants=VSHARP + [0] * 28,
        memory=memory,
    )


# -----------------------------------------------------------------------------
# 1. v_rcp_f32 Semantic & Approximation Tests
# -----------------------------------------------------------------------------

class AluRcpF32Tests(unittest.TestCase):
    def test_rcp_exact_powers_of_two(self) -> None:
        """Powers of 2 have bit-exact reciprocals in binary32."""
        cases = [
            (1.0, 1.0),
            (2.0, 0.5),
            (4.0, 0.25),
            (8.0, 0.125),
            (16.0, 0.0625),
            (32.0, 0.03125),
            (64.0, 0.015625),
            (128.0, 0.0078125),
            (256.0, 0.00390625),
            (1024.0, 1.0 / 1024.0),
            (0.5, 2.0),
            (0.25, 4.0),
            (0.125, 8.0),
            (0.0625, 16.0),
            (0.03125, 32.0),
            (0.015625, 64.0),
            (-1.0, -1.0),
            (-2.0, -0.5),
            (-4.0, -0.25),
            (-8.0, -0.125),
            (-0.5, -2.0),
            (-0.25, -4.0),
            (-0.125, -8.0),
            (1048576.0, 1.0 / 1048576.0),          # 2^20 -> 2^-20
            (1.0 / 1048576.0, 1048576.0),          # 2^-20 -> 2^20
        ]
        for x, want in cases:
            with self.subTest(x=x, want=want):
                got_bits = run_op("v_rcp_f32_e32 v0, v1", f2u(x))
                self.assertEqual(got_bits, f2u(want))

    def test_rcp_general_finite_approximation_envelope(self) -> None:
        """General non-power-of-two inputs are within the 1-ULP float32 approximation envelope."""
        inputs = [
            0.1, 0.3, 0.7, 1.25, 2.5, 3.0, 5.0, 7.0, 9.0, 10.0, 15.0, 33.33, 100.0, 12345.678,
            -0.1, -0.3, -1.25, -3.0, -7.0, -10.0, -100.0, -12345.678,
        ]
        for x in inputs:
            with self.subTest(x=x):
                got_bits = run_op("v_rcp_f32_e32 v0, v1", f2u(x))
                got_flt = u2f(got_bits)
                ref_flt = 1.0 / x
                rel_err = abs((got_flt - ref_flt) / ref_flt)
                self.assertLessEqual(rel_err, TOL_F32_REL)

    def test_rcp_infinities_and_nans(self) -> None:
        """v_rcp_f32 maps +-inf to +-0.0, and NaN inputs return a NaN."""
        # 1.0 / (+inf) = +0.0
        self.assertEqual(run_op("v_rcp_f32_e32 v0, v1", POS_INF_BITS), POS_ZERO_BITS)
        # 1.0 / (-inf) = -0.0
        self.assertEqual(run_op("v_rcp_f32_e32 v0, v1", NEG_INF_BITS), NEG_ZERO_BITS)

        # NaN inputs return a NaN
        for nan_bits in (QNAN_BITS, SNAN_BITS, NEG_QNAN_BITS):
            with self.subTest(nan_bits=hex(nan_bits)):
                res = run_op("v_rcp_f32_e32 v0, v1", nan_bits)
                self.assertTrue(is_f32_nan(res))
                self.assertTrue(math.isnan(u2f(res)))

    def test_rcp_division_by_zero_raises(self) -> None:
        """OpFDiv by +-0.0 is undefined in SPIR-V and raises si.UnsupportedConstruct."""
        with self.assertRaises(si.UnsupportedConstruct) as ctx:
            run_op("v_rcp_f32_e32 v0, v1", POS_ZERO_BITS)
        self.assertIn("OpFDiv by zero", str(ctx.exception))

        with self.assertRaises(si.UnsupportedConstruct) as ctx_neg:
            run_op("v_rcp_f32_e32 v0, v1", NEG_ZERO_BITS)
        self.assertIn("OpFDiv by zero", str(ctx_neg.exception))

    def test_rcp_e64_input_modifiers(self) -> None:
        """VOP3 input modifiers: negation (-v1), absolute value (|v1|), negative abs (-|v1|)."""
        # Negation: -v1 inverts the sign of the reciprocal
        self.assertEqual(run_op("v_rcp_f32_e64 v0, -v1", f2u(4.0)), f2u(-0.25))
        self.assertEqual(run_op("v_rcp_f32_e64 v0, -v1", f2u(-2.0)), f2u(0.5))

        # Absolute value: |v1| forces positive input
        self.assertEqual(run_op("v_rcp_f32_e64 v0, |v1|", f2u(-4.0)), f2u(0.25))
        self.assertEqual(run_op("v_rcp_f32_e64 v0, |v1|", f2u(2.0)), f2u(0.5))

        # Negative absolute value: -|v1| forces negative input
        self.assertEqual(run_op("v_rcp_f32_e64 v0, -|v1|", f2u(4.0)), f2u(-0.25))
        self.assertEqual(run_op("v_rcp_f32_e64 v0, -|v1|", f2u(-4.0)), f2u(-0.25))

    def test_rcp_e64_output_modifiers(self) -> None:
        """VOP3 output modifiers: clamp, mul:2, mul:4, div:2."""
        # clamp: clamps reciprocal result to [0.0, 1.0]
        # 1.0 / 0.5 = 2.0 -> clamped to 1.0
        self.assertEqual(run_op("v_rcp_f32_e64 v0, v1 clamp", f2u(0.5)), f2u(1.0))
        # 1.0 / (-2.0) = -0.5 -> clamped to 0.0
        self.assertEqual(run_op("v_rcp_f32_e64 v0, v1 clamp", f2u(-2.0)), POS_ZERO_BITS)
        # 1.0 / 4.0 = 0.25 in [0, 1] -> unchanged 0.25
        self.assertEqual(run_op("v_rcp_f32_e64 v0, v1 clamp", f2u(4.0)), f2u(0.25))

        # Output scaling
        # (1.0 / 4.0) * 2 = 0.5
        self.assertEqual(run_op("v_rcp_f32_e64 v0, v1 mul:2", f2u(4.0)), f2u(0.5))
        # (1.0 / 4.0) * 4 = 1.0
        self.assertEqual(run_op("v_rcp_f32_e64 v0, v1 mul:4", f2u(4.0)), f2u(1.0))
        # (1.0 / 2.0) / 2 = 0.25
        self.assertEqual(run_op("v_rcp_f32_e64 v0, v1 div:2", f2u(2.0)), f2u(0.25))

        # Combining input and output modifiers: -|v1| with clamp -> clamped to 0.0
        self.assertEqual(run_op("v_rcp_f32_e64 v0, -|v1| clamp", f2u(4.0)), POS_ZERO_BITS)

    def test_rcp_operand_forms(self) -> None:
        """Operand forms: SGPR, inline float constants, literal floats, hex dwords."""
        # SGPR operand
        self.assertEqual(run_op("v_rcp_f32_e32 v0, s4", 0, s4=f2u(4.0)), f2u(0.25))
        # Inline float constants
        self.assertEqual(run_op("v_rcp_f32_e32 v0, 0.5", 0), f2u(2.0))
        self.assertEqual(run_op("v_rcp_f32_e32 v0, 1.0", 0), f2u(1.0))
        self.assertEqual(run_op("v_rcp_f32_e32 v0, 2.0", 0), f2u(0.5))
        self.assertEqual(run_op("v_rcp_f32_e32 v0, -1.0", 0), f2u(-1.0))
        # Literal float constant
        self.assertEqual(run_op("v_rcp_f32_e32 v0, 4.0", 0), f2u(0.25))
        # Hex float immediate (4.0 = 0x40800000)
        self.assertEqual(run_op("v_rcp_f32_e32 v0, 0x40800000", 0), f2u(0.25))


# -----------------------------------------------------------------------------
# 2. v_rsq_f32 Semantic & Approximation Tests
# -----------------------------------------------------------------------------

class AluRsqF32Tests(unittest.TestCase):
    def test_rsq_exact_powers_of_four(self) -> None:
        """Exact square powers of 2 (powers of 4) have bit-exact reciprocal square roots in binary32."""
        cases = [
            (1.0, 1.0),
            (4.0, 0.5),
            (16.0, 0.25),
            (64.0, 0.125),
            (256.0, 0.0625),
            (1024.0, 0.03125),
            (4096.0, 0.015625),
            (0.25, 2.0),
            (0.0625, 4.0),
            (0.015625, 8.0),
            (0.00390625, 16.0),
        ]
        for x, want in cases:
            with self.subTest(x=x, want=want):
                got_bits = run_op("v_rsq_f32_e32 v0, v1", f2u(x))
                self.assertEqual(got_bits, f2u(want))

    def test_rsq_general_positive_approximation_envelope(self) -> None:
        """General positive inputs are within the 1-ULP float32 approximation envelope."""
        inputs = [
            0.01, 0.05, 0.1, 0.5, 2.0, 3.0, 5.0, 7.0, 9.0, 10.0, 25.0, 50.0, 100.0, 500.0, 10000.0,
        ]
        for x in inputs:
            with self.subTest(x=x):
                got_bits = run_op("v_rsq_f32_e32 v0, v1", f2u(x))
                got_flt = u2f(got_bits)
                ref_flt = 1.0 / math.sqrt(x)
                rel_err = abs((got_flt - ref_flt) / ref_flt)
                self.assertLessEqual(rel_err, TOL_F32_REL)

    def test_rsq_infinities_and_nans(self) -> None:
        """v_rsq_f32 on +inf returns 0.0; NaN inputs return a NaN."""
        # 1.0 / sqrt(+inf) = 0.0
        self.assertEqual(run_op("v_rsq_f32_e32 v0, v1", POS_INF_BITS), POS_ZERO_BITS)

        # NaN inputs return a NaN
        for nan_bits in (QNAN_BITS, SNAN_BITS):
            with self.subTest(nan_bits=hex(nan_bits)):
                res = run_op("v_rsq_f32_e32 v0, v1", nan_bits)
                self.assertTrue(is_f32_nan(res))
                self.assertTrue(math.isnan(u2f(res)))

    def test_rsq_non_positive_inputs_raise(self) -> None:
        """InverseSqrt of x <= 0 is undefined in GLSL.std.450 and raises si.UnsupportedConstruct."""
        non_positive_cases = [
            ("pos_zero", POS_ZERO_BITS),
            ("neg_zero", NEG_ZERO_BITS),
            ("neg_one", f2u(-1.0)),
            ("neg_four", f2u(-4.0)),
            ("neg_inf", NEG_INF_BITS),
        ]
        for name, bits in non_positive_cases:
            with self.subTest(name=name):
                with self.assertRaises(si.UnsupportedConstruct) as ctx:
                    run_op("v_rsq_f32_e32 v0, v1", bits)
                self.assertIn("InverseSqrt", str(ctx.exception))

    def test_rsq_e64_input_modifiers(self) -> None:
        """VOP3 input modifiers: absolute value (|v1|) and negation (-v1) rescue negative inputs."""
        # Absolute value: |-4.0| = 4.0, so 1.0 / sqrt(|-4.0|) = 0.5
        self.assertEqual(run_op("v_rsq_f32_e64 v0, |v1|", f2u(-4.0)), f2u(0.5))
        self.assertEqual(run_op("v_rsq_f32_e64 v0, |v1|", f2u(-16.0)), f2u(0.25))

        # Negation: -(-4.0) = 4.0, so 1.0 / sqrt(-v1) = 0.5
        self.assertEqual(run_op("v_rsq_f32_e64 v0, -v1", f2u(-4.0)), f2u(0.5))
        self.assertEqual(run_op("v_rsq_f32_e64 v0, -v1", f2u(-16.0)), f2u(0.25))

    def test_rsq_e64_output_modifiers(self) -> None:
        """VOP3 output modifiers: clamp, mul:2, mul:4, div:2."""
        # clamp: clamps result to [0.0, 1.0]
        # 1.0 / sqrt(0.0625) = 4.0 -> clamped to 1.0
        self.assertEqual(run_op("v_rsq_f32_e64 v0, v1 clamp", f2u(0.0625)), f2u(1.0))
        # 1.0 / sqrt(4.0) = 0.5 in [0, 1] -> unchanged 0.5
        self.assertEqual(run_op("v_rsq_f32_e64 v0, v1 clamp", f2u(4.0)), f2u(0.5))

        # Output scaling
        # (1.0 / sqrt(16.0)) * 2 = 0.25 * 2 = 0.5
        self.assertEqual(run_op("v_rsq_f32_e64 v0, v1 mul:2", f2u(16.0)), f2u(0.5))
        # (1.0 / sqrt(16.0)) * 4 = 0.25 * 4 = 1.0
        self.assertEqual(run_op("v_rsq_f32_e64 v0, v1 mul:4", f2u(16.0)), f2u(1.0))
        # (1.0 / sqrt(4.0)) / 2 = 0.5 / 2 = 0.25
        self.assertEqual(run_op("v_rsq_f32_e64 v0, v1 div:2", f2u(4.0)), f2u(0.25))

    def test_rsq_operand_forms(self) -> None:
        """Operand forms: SGPR, inline float constants, literal floats, hex dwords."""
        # SGPR operand
        self.assertEqual(run_op("v_rsq_f32_e32 v0, s4", 0, s4=f2u(16.0)), f2u(0.25))
        # Inline float constants
        self.assertEqual(run_op("v_rsq_f32_e32 v0, 1.0", 0), f2u(1.0))
        self.assertEqual(run_op("v_rsq_f32_e32 v0, 4.0", 0), f2u(0.5))
        # Hex float immediate (16.0 = 0x41800000)
        self.assertEqual(run_op("v_rsq_f32_e32 v0, 0x41800000", 0), f2u(0.25))


# -----------------------------------------------------------------------------
# 3. Execution Masking & Multilane Wave Tests
# -----------------------------------------------------------------------------

class AluRcpRsqExecMaskingTests(unittest.TestCase):
    def test_single_lane_exec_masking(self) -> None:
        """With EXEC disabled (exec_lo = 0), v0 remains unchanged."""
        for line in ("v_rcp_f32_e32 v0, v1", "v_rsq_f32_e32 v0, v1", "v_rcp_f32_e64 v0, v1 mul:2"):
            with self.subTest(line=line):
                got = run_ps_v0([
                    f"v_mov_b32 v1, {f2u(4.0):#x}",
                    f"v_mov_b32 v0, {OLD:#x}",
                    "s_mov_b32 exec_lo, 0",
                    line,
                    "s_mov_b32 exec_lo, 1",
                ])
                self.assertEqual(got, OLD)

    def test_multilane_compute_exec_masking(self) -> None:
        """Multilane compute shader with even lanes active and odd lanes disabled."""
        cases = [
            ("v_rcp_f32_e32 v3, v2", f2u(4.0), f2u(0.25)),
            ("v_rsq_f32_e32 v3, v2", f2u(16.0), f2u(0.25)),
            ("v_rcp_f32_e64 v3, v2 mul:2", f2u(4.0), f2u(0.5)),
            ("v_rsq_f32_e64 v3, v2 mul:4", f2u(16.0), f2u(1.0)),
        ]
        for insn, src_word, active_want in cases:
            with self.subTest(insn=insn):
                mem: dict[int, int] = {}
                lines = [
                    "v_lshlrev_b32 v4, 2, v0",          # byte offset = lane_id * 4
                    f"v_mov_b32 v2, {src_word:#x}",
                    f"v_mov_b32 v3, {OLD:#x}",
                    "v_and_b32 v5, v0, 1",
                    "v_cmpx_eq_u32 v5, 0",              # keep even lanes active
                    insn,
                    "s_mov_b32 exec_lo, -1",
                    "buffer_store_dword v3, v4, s[0:3], 0 offen",
                ]
                run_cs(lines, group_size=4, subgroup_size=32, memory=mem)
                self.assertEqual(
                    [mem[4 * i] for i in range(4)],
                    [active_want, OLD, active_want, OLD],
                )


# -----------------------------------------------------------------------------
# 4. Negative Control Tests
# -----------------------------------------------------------------------------

class NegativeControlTests(unittest.TestCase):
    def test_sabotaged_rcp_spirv_is_detected(self) -> None:
        """Mutating OpFDiv to OpFMul in translated v_rcp_f32 fails the assertion."""
        spv = translate_ps([
            f"v_mov_b32 v0, {OLD:#x}",
            f"v_mov_b32 v1, {f2u(4.0):#x}",
            "v_rcp_f32_e32 v0, v1",
        ])
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], f2u(0.25))

        bad_spv, count = re.subn(r"OpFDiv %float %f1\b", "OpFMul %float %f1", spv, count=1)
        self.assertEqual(count, 1)
        # 1.0 * 4.0 = 4.0 != 0.25
        self.assertNotEqual(si.run_fragment(bad_spv)["%out_mrt0"][0], f2u(0.25))

    def test_sabotaged_rsq_spirv_is_detected(self) -> None:
        """Mutating InverseSqrt to Sqrt in translated v_rsq_f32 fails the assertion."""
        spv = translate_ps([
            f"v_mov_b32 v0, {OLD:#x}",
            f"v_mov_b32 v1, {f2u(4.0):#x}",
            "v_rsq_f32_e32 v0, v1",
        ])
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], f2u(0.5))

        bad_spv, count = re.subn(r"InverseSqrt", "Sqrt", spv, count=1)
        self.assertEqual(count, 1)
        # sqrt(4.0) = 2.0 != 0.5
        self.assertNotEqual(si.run_fragment(bad_spv)["%out_mrt0"][0], f2u(0.5))


if __name__ == "__main__":
    unittest.main()
