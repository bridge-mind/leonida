"""Semantic interpreter tests for AMD RDNA2 vector 32-bit integer multiplication instructions:
  - v_mul_lo_u32: D = (S0 * S1) & 0xFFFFFFFF
  - v_mul_hi_u32: D = ((u32(S0) * u32(S1)) >> 32) & 0xFFFFFFFF
  - v_mul_hi_i32: D = ((i32(S0) * i32(S1)) >> 32) & 0xFFFFFFFF

Standard library only. Tests:
  * Reference semantic model for unsigned and signed 32-bit integer multiplication
  * Mathematical identities: zero multiplier, unity multiplier, commutativity
  * 64-bit product decomposition and full integer reconstruction from (lo, hi) halves
  * Signed vs unsigned divergence on negative operands (e.g., -1 * -1, INT32_MIN * 2)
  * Extreme boundary values: 0, 1, INT32_MAX, INT32_MIN, -1, -2, large unsigned words
  * End-to-end SPIR-V translation via tools/rdna2spirv.py and execution via spirv_mini_interp
  * Inline constant operands and opcode suffix aliases (_e32, _e64)
  * Negative controls verifying detection of signedness, truncation, or arithmetic mutations
"""
from __future__ import annotations

import gc
import itertools
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
SIGN32 = 0x80000000
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

OPS = ("v_mul_lo_u32", "v_mul_hi_u32", "v_mul_hi_i32")


def s32(x: int) -> int:
    """Convert a 32-bit word into a signed Python integer (two's complement)."""
    x &= MASK32
    return x - (1 << 32) if (x & SIGN32) else x


def u32(x: int) -> int:
    """Mask a Python integer to 32 bits unsigned."""
    return x & MASK32


def ref_mul_lo_u32(a: int, b: int) -> int:
    """Low 32 bits of product. Identical for signed and unsigned integers modulo 2^32."""
    return (u32(a) * u32(b)) & MASK32


def ref_mul_hi_u32(a: int, b: int) -> int:
    """High 32 bits of unsigned 32-bit product."""
    return ((u32(a) * u32(b)) >> 32) & MASK32


def ref_mul_hi_i32(a: int, b: int) -> int:
    """High 32 bits of signed 32-bit product (two's complement representation)."""
    return ((s32(a) * s32(b)) >> 32) & MASK32


def ref_mul(op: str, a: int, b: int) -> int:
    """Oracle lookup by instruction name."""
    base = op.split("_e")[0] if "_e" in op else op
    if base == "v_mul_lo_u32":
        return ref_mul_lo_u32(a, b)
    elif base == "v_mul_hi_u32":
        return ref_mul_hi_u32(a, b)
    elif base == "v_mul_hi_i32":
        return ref_mul_hi_i32(a, b)
    raise ValueError(f"Unknown instruction {op}")


def ref_mul_64_u(a: int, b: int) -> tuple[int, int]:
    """Returns (lo, hi_u) for 64-bit unsigned product."""
    return ref_mul_lo_u32(a, b), ref_mul_hi_u32(a, b)


def ref_mul_64_i(a: int, b: int) -> tuple[int, int]:
    """Returns (lo, hi_i) for 64-bit signed product."""
    return ref_mul_lo_u32(a, b), ref_mul_hi_i32(a, b)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_alu_mul.s")
    with open(os.path.join(_TMP_DIR.name, "ps_alu_mul.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_mul_single(op: str, a: int, b: int, inline: tuple[int, ...] = ()) -> int:
    """Run `op v0, src0, src1` and return result written to mrt0[0].

    `inline` specifies which operand indices (0 for src0, 1 for src1) are passed
    as inline/literal constants directly in the instruction text rather than preloaded into VGPRs.
    """
    if _TMP_DIR is None:
        setUpModule()
    lines = []
    operands = []
    for i, val in enumerate((a, b)):
        if i in inline:
            operands.append(str(val))
        else:
            reg = f"v{i + 1}"
            lines.append(f"\tv_mov_b32 {reg}, {u32(val):#x}")
            operands.append(reg)

    lines.append(f"\t{op} v0, {operands[0]}, {operands[1]}")
    asm = "; test_alu_mul\n" + "\n".join(lines) + "\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
        gc.collect()
    return si.run_fragment(spv)["%out_mrt0"][0]


def run_mul_pair(op_hi: str, a: int, b: int) -> tuple[int, int]:
    """Run `v_mul_lo_u32 v0, v2, v3` and `{op_hi} v1, v2, v3` to test full 64-bit product export.

    Returns (v0, v1) corresponding to (lo, hi).
    """
    if _TMP_DIR is None:
        setUpModule()
    lines = [
        f"\tv_mov_b32 v2, {u32(a):#x}",
        f"\tv_mov_b32 v3, {u32(b):#x}",
        "\tv_mul_lo_u32 v0, v2, v3",
        f"\t{op_hi} v1, v2, v3",
    ]
    asm = "; test_alu_mul_pair\n" + "\n".join(lines) + "\n\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
        gc.collect()
    out = si.run_fragment(spv)["%out_mrt0"]
    return out[0], out[1]


SAMPLE_VALUES = (
    0,
    1,
    2,
    3,
    7,
    10,
    16,
    64,
    100,
    0x10000,
    0x7FFFFFFF,
    0x80000000,
    0x80000001,
    0x7FFFFFFE,
    0xFFFFFFFF,
    0xFFFFFFFE,
    0x12345678,
    0x9ABCDEF0,
)

SAMPLE_PAIRS = (
    (0, 0),
    (0, 1),
    (0, 0xFFFFFFFF),
    (1, 1),
    (1, 0xFFFFFFFF),
    (2, 3),
    (0x10000, 0x10000),
    (0x7FFFFFFF, 0x7FFFFFFF),
    (0x80000000, 0x80000000),
    (0x80000000, 2),
    (0x80000000, 0xFFFFFFFF),
    (0xFFFFFFFF, 0xFFFFFFFF),
    (0xFFFFFFFF, 5),
    (0x12345678, 0x9ABCDEF0),
    (0xAAAA5555, 0x5555AAAA),
    (0x80000001, 0x7FFFFFFE),
    (-16, 64),
    (-100, 200),
)


class ReferenceModelTests(unittest.TestCase):
    """Mathematical validation of reference models and hardware identities."""

    def test_zero_property(self):
        for val in SAMPLE_VALUES:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_mul_lo_u32(val, 0), 0)
                self.assertEqual(ref_mul_lo_u32(0, val), 0)
                self.assertEqual(ref_mul_hi_u32(val, 0), 0)
                self.assertEqual(ref_mul_hi_u32(0, val), 0)
                self.assertEqual(ref_mul_hi_i32(val, 0), 0)
                self.assertEqual(ref_mul_hi_i32(0, val), 0)

    def test_unity_property(self):
        for val in SAMPLE_VALUES:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_mul_lo_u32(val, 1), u32(val))
                self.assertEqual(ref_mul_lo_u32(1, val), u32(val))
                self.assertEqual(ref_mul_hi_u32(val, 1), 0)
                self.assertEqual(ref_mul_hi_u32(1, val), 0)
                expected_hi_i = 0xFFFFFFFF if s32(val) < 0 else 0
                self.assertEqual(ref_mul_hi_i32(val, 1), expected_hi_i)
                self.assertEqual(ref_mul_hi_i32(1, val), expected_hi_i)

    def test_negation_identity(self):
        for val in SAMPLE_VALUES:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_mul_lo_u32(val, -1), (-u32(val)) & MASK32)
                self.assertEqual(ref_mul_lo_u32(val, 0xFFFFFFFF), (-u32(val)) & MASK32)

    def test_commutativity(self):
        for a, b in SAMPLE_PAIRS:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(ref_mul_lo_u32(a, b), ref_mul_lo_u32(b, a))
                self.assertEqual(ref_mul_hi_u32(a, b), ref_mul_hi_u32(b, a))
                self.assertEqual(ref_mul_hi_i32(a, b), ref_mul_hi_i32(b, a))

    def test_64bit_reconstruction(self):
        for a, b in SAMPLE_PAIRS:
            with self.subTest(a=hex(a), b=hex(b)):
                # Unsigned reconstruction
                lo_u, hi_u = ref_mul_64_u(a, b)
                reconstructed_u = (hi_u << 32) | lo_u
                self.assertEqual(reconstructed_u, u32(a) * u32(b))

                # Signed reconstruction: sign-extend 32-bit hi word into 64-bit int
                lo_i, hi_i = ref_mul_64_i(a, b)
                hi_signed = s32(hi_i)
                reconstructed_i = (hi_signed << 32) | lo_i
                self.assertEqual(reconstructed_i, s32(a) * s32(b))

    def test_hand_calculated_constants(self):
        # 0xFFFFFFFF * 0xFFFFFFFF:
        # unsigned: (2^32 - 1)^2 = 2^64 - 2*2^32 + 1 = (2^32 - 2) * 2^32 + 1
        # signed: (-1) * (-1) = 1
        self.assertEqual(ref_mul_lo_u32(0xFFFFFFFF, 0xFFFFFFFF), 1)
        self.assertEqual(ref_mul_hi_u32(0xFFFFFFFF, 0xFFFFFFFF), 0xFFFFFFFE)
        self.assertEqual(ref_mul_hi_i32(0xFFFFFFFF, 0xFFFFFFFF), 0)

        # 0x7FFFFFFF * 0x7FFFFFFF (INT32_MAX^2):
        # (2^31 - 1)^2 = 2^62 - 2^32 + 1 = (2^30 - 1) * 2^32 + 1
        # 2^30 - 1 = 0x3FFFFFFF
        self.assertEqual(ref_mul_lo_u32(0x7FFFFFFF, 0x7FFFFFFF), 1)
        self.assertEqual(ref_mul_hi_u32(0x7FFFFFFF, 0x7FFFFFFF), 0x3FFFFFFF)
        self.assertEqual(ref_mul_hi_i32(0x7FFFFFFF, 0x7FFFFFFF), 0x3FFFFFFF)

        # 0x80000000 * 0x80000000 (INT32_MIN^2):
        # signed: (-2^31)^2 = 2^62 = 0x40000000 * 2^32 + 0
        # unsigned: (2^31)^2 = 2^62 = 0x40000000 * 2^32 + 0
        self.assertEqual(ref_mul_lo_u32(0x80000000, 0x80000000), 0)
        self.assertEqual(ref_mul_hi_u32(0x80000000, 0x80000000), 0x40000000)
        self.assertEqual(ref_mul_hi_i32(0x80000000, 0x80000000), 0x40000000)

        # 0x80000000 * 0xFFFFFFFF (INT32_MIN * -1):
        # signed: (-2^31) * (-1) = 2^31 = 0x80000000 (hi = 0)
        # unsigned: (2^31) * (2^32 - 1) = 2^63 - 2^31 = 0x7FFFFFFF * 2^32 + 0x80000000
        self.assertEqual(ref_mul_lo_u32(0x80000000, 0xFFFFFFFF), 0x80000000)
        self.assertEqual(ref_mul_hi_u32(0x80000000, 0xFFFFFFFF), 0x7FFFFFFF)
        self.assertEqual(ref_mul_hi_i32(0x80000000, 0xFFFFFFFF), 0)

        # 0x80000000 * 2 (INT32_MIN * 2):
        # signed: (-2^31) * 2 = -2^32 = -1 * 2^32 + 0 (hi = -1 = 0xFFFFFFFF)
        # unsigned: (2^31) * 2 = 2^32 = 1 * 2^32 + 0 (hi = 1)
        self.assertEqual(ref_mul_lo_u32(0x80000000, 2), 0)
        self.assertEqual(ref_mul_hi_u32(0x80000000, 2), 1)
        self.assertEqual(ref_mul_hi_i32(0x80000000, 2), 0xFFFFFFFF)

        # 100 * -1:
        # signed: -100 = 0xFFFFFFFF_FFFFFF9C
        # unsigned: 100 * (2^32 - 1) = 99 * 2^32 + (2^32 - 100) = 0x00000063 * 2^32 + 0xFFFFFF9C
        self.assertEqual(ref_mul_lo_u32(100, -1), 0xFFFFFF9C)
        self.assertEqual(ref_mul_hi_u32(100, -1), 99)
        self.assertEqual(ref_mul_hi_i32(100, -1), 0xFFFFFFFF)

        # 0x10000 * 0x10000 = 0x1_00000000:
        self.assertEqual(ref_mul_lo_u32(0x10000, 0x10000), 0)
        self.assertEqual(ref_mul_hi_u32(0x10000, 0x10000), 1)
        self.assertEqual(ref_mul_hi_i32(0x10000, 0x10000), 1)


class AluMulTranslatorTests(unittest.TestCase):
    """End-to-end interpreter tests of translated RDNA2 multiplication instructions."""

    def test_sample_pairs_execution(self):
        for op in OPS:
            for a, b in SAMPLE_PAIRS:
                with self.subTest(op=op, a=hex(a), b=hex(b)):
                    expected = ref_mul(op, a, b)
                    self.assertEqual(run_mul_single(op, a, b), expected)
                    # Commutativity verification on hardware translation
                    self.assertEqual(run_mul_single(op, b, a), expected)

    def test_inline_constants(self):
        test_cases = [
            ("v_mul_lo_u32", 100, -1, 0xFFFFFF9C),
            ("v_mul_hi_u32", 100, -1, 99),
            ("v_mul_hi_i32", 100, -1, 0xFFFFFFFF),
            ("v_mul_lo_u32", 0x7FFFFFFF, 2, 0xFFFFFFFE),
            ("v_mul_hi_u32", 0x7FFFFFFF, 2, 0),
            ("v_mul_hi_i32", 0x7FFFFFFF, 2, 0),
            ("v_mul_lo_u32", -16, 64, (-1024) & MASK32),
            ("v_mul_hi_u32", -16, 64, 63),
            ("v_mul_hi_i32", -16, 64, 0xFFFFFFFF),
        ]
        for op, a, b, expected in test_cases:
            with self.subTest(op=op, a=a, b=b, position="src1"):
                # b passed as inline constant
                self.assertEqual(run_mul_single(op, a, b, inline=(1,)), expected)
            with self.subTest(op=op, a=a, b=b, position="src0"):
                # a passed as inline constant
                self.assertEqual(run_mul_single(op, a, b, inline=(0,)), expected)

    def test_instruction_suffix_aliases(self):
        aliases = [
            ("v_mul_lo_u32_e32", "v_mul_lo_u32"),
            ("v_mul_lo_u32_e64", "v_mul_lo_u32"),
            ("v_mul_hi_u32_e32", "v_mul_hi_u32"),
            ("v_mul_hi_i32_e32", "v_mul_hi_i32"),
        ]
        test_val_a = 0x12345678
        test_val_b = 0x00000002
        for alias, base_op in aliases:
            with self.subTest(alias=alias):
                expected = ref_mul(base_op, test_val_a, test_val_b)
                self.assertEqual(run_mul_single(alias, test_val_a, test_val_b), expected)

    def test_combined_pair_reconstruction(self):
        # Executes v_mul_lo and v_mul_hi in the same shader and compares exported words
        pairs = [
            (0x12345678, 0x9ABCDEF0),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x80000000, 2),
            (100, -1),
        ]
        for a, b in pairs:
            with self.subTest(pair="unsigned", a=hex(a), b=hex(b)):
                got = run_mul_pair("v_mul_hi_u32", a, b)
                expected = ref_mul_64_u(a, b)
                self.assertEqual(got, expected)

            with self.subTest(pair="signed", a=hex(a), b=hex(b)):
                got = run_mul_pair("v_mul_hi_i32", a, b)
                expected = ref_mul_64_i(a, b)
                self.assertEqual(got, expected)


class NegativeControlsTests(unittest.TestCase):
    """Negative controls ensuring mutated arithmetic or signedness confusion are caught."""

    def test_signed_vs_unsigned_hi_divergence_detected(self):
        divergent_pairs = [
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFF, 5),
            (0x80000000, 2),
            (100, -1),
            (-16, 64),
        ]
        for a, b in divergent_pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                hi_u = ref_mul_hi_u32(a, b)
                hi_i = ref_mul_hi_i32(a, b)
                self.assertNotEqual(hi_u, hi_i, f"Expected divergence between unsigned and signed hi for ({a}, {b})")

    def test_low_vs_high_word_divergence_detected(self):
        divergent_pairs = [
            (0x10000, 0x10000),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x80000000, 0x80000000),
        ]
        for a, b in divergent_pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                lo = ref_mul_lo_u32(a, b)
                hi_u = ref_mul_hi_u32(a, b)
                self.assertNotEqual(lo, hi_u, f"Expected divergence between lo and hi for ({a}, {b})")

    def test_sabotaged_shift_count_detected(self):
        a = 0x12345678
        b = 0x9ABCDEF0
        correct_hi_u = ref_mul_hi_u32(a, b)
        sabotaged_hi_u = ((u32(a) * u32(b)) >> 31) & MASK32
        self.assertNotEqual(correct_hi_u, sabotaged_hi_u)

    def test_dropped_sign_extension_detected(self):
        a = -100
        b = 200
        correct_hi_i = ref_mul_hi_i32(a, b)
        # Dropped sign extension: treating signed product as unsigned before shift
        sabotaged_hi_i = ((u32(a) * u32(b)) >> 32) & MASK32
        self.assertNotEqual(correct_hi_i, sabotaged_hi_i)


if __name__ == "__main__":
    unittest.main()
