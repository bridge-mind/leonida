from __future__ import annotations

"""Semantic ALU interpreter tests for RDNA2 v_cvt_pk_u8_f32.

RDNA2 ISA definition for `v_cvt_pk_u8_f32 vD, src0, src1, src2` (VOP3):
  - Converts binary32 `src0` to uint8 using round-to-nearest-even (IEEE-754 roundTiesToEven),
    clamped to [0, 255] (negative values -> 0; values >= 255.0 -> 255; NaN -> 0).
  - Inserts that 8-bit unsigned integer into byte slot `src1 & 3` (bits [7:0], [15:8],
    [23:16], or [31:24]) of `src2`, preserving the remaining 3 bytes of `src2` in `vD`.
  - Used by shader compilers to pack a 4-component float32 vector (e.g. RGBA color or
    XYZW attribute) into a single packed 32-bit VGPR through 4 sequential conversions:
      v_cvt_pk_u8_f32 vD, v_x, 0, v_base
      v_cvt_pk_u8_f32 vD, v_y, 1, vD
      v_cvt_pk_u8_f32 vD, v_z, 2, vD
      v_cvt_pk_u8_f32 vD, v_w, 3, vD
"""

import math
import os
import random
import struct
import sys
import unittest
from fractions import Fraction

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
CANARY_BASE = 0xDEADBEEF


def f32_to_bits(x: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def bits_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def is_nan_bits(b: int) -> bool:
    b &= MASK32
    return (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF) != 0


def ref_cvt_u8_byte(src: float | int) -> int:
    """RDNA2 flt32_to_uint8: round-to-nearest-even, clamped to [0, 255] (NaN -> 0)."""
    if isinstance(src, int):
        b = src & MASK32
        if is_nan_bits(b):
            return 0
        if (b >> 31) != 0 or (b & 0x7FFFFFFF) == 0:
            return 0
        if b == 0x7F800000:
            return 255
        x = bits_to_f32(b)
    else:
        if math.isnan(src):
            return 0
        if math.isinf(src):
            return 255 if src > 0 else 0
        x = float(src)

    if x <= 0.0:
        return 0
    if x >= 255.0:
        return 255

    q = Fraction(x)
    n = q.numerator // q.denominator
    rem = q - n
    half = Fraction(1, 2)
    if rem > half or (rem == half and (n & 1)):
        n += 1
    return min(255, max(0, n))


def ref_cvt_pk_u8_f32(src0: float | int, slot: int, base: int = 0) -> int:
    """Insert converted byte at byte slot (slot & 3) of base dword."""
    byte_val = ref_cvt_u8_byte(src0)
    sh = 8 * (slot & 3)
    return ((base & MASK32) & ~(0xFF << sh)) | (byte_val << sh)


def ref_pack_4x_u8_f32(
    c0: float | int,
    c1: float | int,
    c2: float | int,
    c3: float | int,
    base: int = 0,
) -> int:
    """Model 4-component vector float32 packing into a single 32-bit register."""
    d0 = ref_cvt_pk_u8_f32(c0, 0, base)
    d1 = ref_cvt_pk_u8_f32(c1, 1, d0)
    d2 = ref_cvt_pk_u8_f32(c2, 2, d1)
    d3 = ref_cvt_pk_u8_f32(c3, 3, d2)
    return d3


def run_ps_pack(
    lines: list[str],
    export_regs: tuple[str, str, str, str] = ("v0", "v0", "v0", "v0"),
) -> list[int]:
    full_lines = list(lines) + [f"exp mrt0 {', '.join(export_regs)} done vm", "s_endpgm"]
    spv = support.translate_lines("ps_alu_cvt_pk_u8_f32", full_lines)
    return support.mrt0_words(spv)


class TestAluCvtPkU8F32ReferenceOracle(unittest.TestCase):
    """Reference oracle self-tests and baseline execution for v_cvt_pk_u8_f32."""

    def test_exact_integers(self) -> None:
        for k in range(256):
            expected = k
            self.assertEqual(ref_cvt_u8_byte(float(k)), expected, f"int {k}")
            self.assertEqual(ref_cvt_u8_byte(f32_to_bits(float(k))), expected)

    def test_round_to_nearest_even_ties(self) -> None:
        ties = [
            (0.5, 0),
            (1.5, 2),
            (2.5, 2),
            (3.5, 4),
            (4.5, 4),
            (126.5, 126),
            (127.5, 128),
            (128.5, 128),
            (252.5, 252),
            (253.5, 254),
            (254.5, 254),
        ]
        for val, want in ties:
            with self.subTest(tie=val):
                self.assertEqual(ref_cvt_u8_byte(val), want)
                self.assertEqual(ref_cvt_u8_byte(f32_to_bits(val)), want)

        eps = 1e-5
        for val, _ in ties:
            below = val - eps
            above = val + eps
            with self.subTest(delta=val):
                self.assertEqual(ref_cvt_u8_byte(below), math.floor(val))
                self.assertEqual(ref_cvt_u8_byte(above), math.ceil(val))

    def test_byte_slot_selection_and_masking(self) -> None:
        # Each byte slot (0, 1, 2, 3) must write only bits [8*s+7 : 8*s] and preserve the rest.
        test_val = float(0x42)
        self.assertEqual(ref_cvt_pk_u8_f32(test_val, 0, CANARY_BASE), 0xDEADBE42)
        self.assertEqual(ref_cvt_pk_u8_f32(test_val, 1, CANARY_BASE), 0xDEAD42EF)
        self.assertEqual(ref_cvt_pk_u8_f32(test_val, 2, CANARY_BASE), 0xDE42BEEF)
        self.assertEqual(ref_cvt_pk_u8_f32(test_val, 3, CANARY_BASE), 0x42ADBEEF)

        # Slot index masking (slot & 3): slots wrapping beyond 3.
        self.assertEqual(ref_cvt_pk_u8_f32(test_val, 4, CANARY_BASE), 0xDEADBE42)
        self.assertEqual(ref_cvt_pk_u8_f32(test_val, 5, CANARY_BASE), 0xDEAD42EF)
        self.assertEqual(ref_cvt_pk_u8_f32(test_val, 7, CANARY_BASE), 0x42ADBEEF)
        self.assertEqual(ref_cvt_pk_u8_f32(test_val, -1, CANARY_BASE), 0x42ADBEEF)
        self.assertEqual(ref_cvt_pk_u8_f32(test_val, 0xFFFFFFFF, CANARY_BASE), 0x42ADBEEF)

    def test_vec4_packing_oracle_rgba_vectors(self) -> None:
        vectors = [
            (255.0, 128.0, 64.0, 32.0, 0, 0x204080FF),
            (0.0, 0.0, 0.0, 0.0, 0, 0x00000000),
            (18.0, 52.0, 86.0, 120.0, 0, 0x78563412),
            (254.5, 127.5, 1.5, 0.5, 0, 0x000280FE),
            (170.0, 85.0, 240.0, 15.0, 0, 0x0FF055AA),
        ]
        for r, g, b, a, base, want in vectors:
            with self.subTest(r=r, g=g, b=b, a=a):
                self.assertEqual(ref_pack_4x_u8_f32(r, g, b, a, base), want)

    def test_ps_shader_execution_single_and_vec4(self) -> None:
        # 1. Single byte insertion into slot 2 of CANARY_BASE.
        out_single = run_ps_pack([
            f"v_mov_b32 v1, {f32_to_bits(86.0):#x}",
            f"v_mov_b32 v3, {CANARY_BASE:#x}",
            "v_cvt_pk_u8_f32 v0, v1, 2, v3",
        ])[0]
        self.assertEqual(out_single, ref_cvt_pk_u8_f32(86.0, 2, CANARY_BASE))
        self.assertEqual(out_single, 0xDE56BEEF)

        # 2. Sequential 4-component vector pack into v0.
        out_vec4 = run_ps_pack([
            f"v_mov_b32 v1, {f32_to_bits(100.0):#x}",
            f"v_mov_b32 v2, {f32_to_bits(200.0):#x}",
            f"v_mov_b32 v3, {f32_to_bits(50.0):#x}",
            f"v_mov_b32 v4, {f32_to_bits(25.0):#x}",
            "v_cvt_pk_u8_f32 v0, v1, 0, 0",
            "v_cvt_pk_u8_f32 v0, v2, 1, v0",
            "v_cvt_pk_u8_f32 v0, v3, 2, v0",
            "v_cvt_pk_u8_f32 v0, v4, 3, v0",
        ])[0]
        want_vec4 = ref_pack_4x_u8_f32(100.0, 200.0, 50.0, 25.0, 0)
        self.assertEqual(out_vec4, want_vec4)
        self.assertEqual(out_vec4, 0x1932C864)

    def test_inline_constants_and_sgpr_operands(self) -> None:
        # Float source as inline constants (1.0, 2.0, 4.0, 0.5) and SGPRs.
        out = run_ps_pack([
            "v_cvt_pk_u8_f32 v0, 1.0, 0, 0",
            "v_cvt_pk_u8_f32 v0, 2.0, 1, v0",
            "v_cvt_pk_u8_f32 v0, 4.0, 2, v0",
            "v_cvt_pk_u8_f32 v0, 0.5, 3, v0",
        ])[0]
        self.assertEqual(out, (1) | (2 << 8) | (4 << 16) | (0 << 24))

        # Slot and base from SGPRs.
        out_sgpr = run_ps_pack([
            f"v_mov_b32 v1, {f32_to_bits(65.0):#x}",
            "s_mov_b32 s4, 1",
            f"s_mov_b32 s5, {CANARY_BASE:#x}",
            "v_cvt_pk_u8_f32_e64 v0, v1, s4, s5",
        ])[0]
        self.assertEqual(out_sgpr, ref_cvt_pk_u8_f32(65.0, 1, CANARY_BASE))


if __name__ == "__main__":
    unittest.main()
