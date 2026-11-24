"""Semantic interpreter tests for RDNA2 s_sext_i32_i8, s_sext_i32_i16, and s_flbit_i32.

RDNA2 scalar ALU operations covered:
  - s_sext_i32_i8 D, S0:
      Sign-extends 8-bit signed integer in S0[7:0] to 32-bit signed integer in D.
      Upper source bits S0[31:8] are completely ignored.
      Destination D is in range [-128, 127] (0xFFFFFF80 .. 0x0000007F).
      Scalar Condition Code (SCC) is preserved (SCC is not written).
  - s_sext_i32_i16 D, S0:
      Sign-extends 16-bit signed integer in S0[15:0] to 32-bit signed integer in D.
      Upper source bits S0[31:16] are completely ignored.
      Destination D is in range [-32768, 32767] (0xFFFF8000 .. 0x00007FFF).
      Scalar Condition Code (SCC) is preserved (SCC is not written).
  - s_flbit_i32 D, S0:
      Find first bit from MSB down to LSB before the first bit that differs
      from the sign bit (scalar equivalent of v_ffbh_i32 / FindSMsb).
      Returns bit count from MSB (31 - bit_index) or -1 (0xFFFFFFFF) if all bits
      match the sign bit (0 or -1).
      Scalar Condition Code (SCC) is preserved (SCC is not written).
  - s_flbit_i32_b32 D, S0:
      Count leading zeros from MSB down to LSB (unsigned bit scan).
      Returns bit count from MSB (31 - highest_set_bit) or -1 (0xFFFFFFFF) on zero.
      Scalar Condition Code (SCC) is preserved.

Condition Code (SCC) semantics:
  - By RDNA2 ISA specification, s_sext_i32_i8, s_sext_i32_i16, and s_flbit_i32 do NOT
    modify the SCC condition code.
  - In SALU operations where SCC is updated based on result D (e.g. s_and_b32, s_or_b32,
    s_not_b32, s_bfe_u32/i32): SCC = (D != 0).

Zero game data; standard library Python only; Python 3.9+ compatible.
"""
from __future__ import annotations

import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def u32(x: int) -> int:
    """Normalize integer to unsigned 32-bit word [0, 0xFFFFFFFF]."""
    return x & M32


def s32(x: int) -> int:
    """Interpret 32-bit integer as signed [-2^31, 2^31 - 1]."""
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def ref_s_sext_i32_i8(s0: int) -> int:
    """RDNA2 s_sext_i32_i8 reference: sign-extend S0[7:0] to 32-bit signed integer.

    Bits [31:8] of S0 are ignored. Returns unsigned 32-bit word representation.
    """
    b = s0 & 0xFF
    signed_val = b - 0x100 if (b & 0x80) else b
    return signed_val & M32


def ref_s_sext_i32_i16(s0: int) -> int:
    """RDNA2 s_sext_i32_i16 reference: sign-extend S0[15:0] to 32-bit signed integer.

    Bits [31:16] of S0 are ignored. Returns unsigned 32-bit word representation.
    """
    h = s0 & 0xFFFF
    signed_val = h - 0x10000 if (h & 0x8000) else h
    return signed_val & M32


def ref_s_flbit_i32(s0: int) -> int:
    """RDNA2 s_flbit_i32 reference: count bits from MSB before first differing bit.

    Signed MSB scan (FindSMsb):
      - Returns M32 (0xFFFFFFFF, representing -1) if all bits equal the sign bit (0 or -1).
      - For positive values (sign=0): counts bits from bit 31 before the first 1-bit.
      - For negative values (sign=1): counts bits from bit 31 before the first 0-bit.
    """
    s0 &= M32
    if s0 == 0 or s0 == M32:
        return M32
    sign = (s0 >> 31) & 1
    for n in range(1, 32):
        if ((s0 >> (31 - n)) & 1) != sign:
            return n
    return M32


def ref_s_flbit_i32_b32(s0: int) -> int:
    """RDNA2 s_flbit_i32_b32 reference: count leading zeros from MSB down to LSB.

    Unsigned MSB scan (FindUMsb):
      - Returns M32 (0xFFFFFFFF, representing -1) if s0 == 0.
      - Otherwise returns 31 - highest_set_bit_index (32 - s0.bit_length()).
    """
    s0 &= M32
    if s0 == 0:
        return M32
    return 32 - s0.bit_length()


def ref_scc_ne_zero(d: int) -> int:
    """Standard RDNA2 SALU condition code rule: SCC = (D != 0)."""
    return 1 if (d & M32) != 0 else 0


flbit_i32 = ref_s_flbit_i32

OPS = {
    "s_sext_i32_i8": ref_s_sext_i32_i8,
    "s_sext_i32_i16": ref_s_sext_i32_i16,
    "s_flbit_i32": ref_s_flbit_i32,
}

VALUES = [
    0, 1, 0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFF,
    0x12345678, 0x87654321, 0x7FFFFFFF, 0x80000000,
    0xFFFFFFFF, 0xFFFFFFFE, 0x40000000, 0xC0000000, 0x00010000,
]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_sext.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_sext.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_op(op: str, x: int, scc_in: int) -> tuple[int, int]:
    """Execute shader: SCC = scc_in, then s0 = op(s2); return (s0, SCC after)."""
    if _TMP_DIR is None:
        setUpModule()
    lines = [
        f"s_mov_b32 s2, {x:#x}",
        f"s_mov_b32 s4, {scc_in}",
        "s_cmp_eq_u32 s4, 1",
        f"{op} s0, s2",
        "s_cselect_b32 s1, 0x40000001, 0x40000000",
        "s_and_b32 s5, s0, 0xffff",
        "s_lshr_b32 s6, s0, 16",
        "s_or_b32 s5, s5, 0x40000000",
        "s_or_b32 s6, s6, 0x40000000",
        "v_mov_b32 v0, s5",
        "v_mov_b32 v1, s6",
        "v_mov_b32 v2, s1",
    ]
    asm = (
        "; test_salu_sext\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v2, v2 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi, scc = si.run_fragment(spv)["%out_mrt0"][:3]
    assert lo >> 16 == hi >> 16 == 0x4000 and scc >> 1 == 0x20000000, (hex(lo), hex(hi), hex(scc))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16, scc & 1


def run_program(lines: list[str]) -> tuple[int, int]:
    """Execute arbitrary assembly lines writing result to s0 and SCC to s1."""
    if _TMP_DIR is None:
        setUpModule()
    prologue = [
        "s_cselect_b32 s1, 0x40000001, 0x40000000",
        "s_and_b32 s5, s0, 0xffff",
        "s_lshr_b32 s6, s0, 16",
        "s_or_b32 s5, s5, 0x40000000",
        "s_or_b32 s6, s6, 0x40000000",
        "v_mov_b32 v0, s5",
        "v_mov_b32 v1, s6",
        "v_mov_b32 v2, s1",
    ]
    all_lines = list(lines) + prologue
    asm = (
        "; test_salu_program\n"
        + "".join(f"\t{ln}\n" for ln in all_lines)
        + "\texp mrt0 v0, v1, v2, v2 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi, scc = si.run_fragment(spv)["%out_mrt0"][:3]
    assert lo >> 16 == hi >> 16 == 0x4000 and scc >> 1 == 0x20000000, (hex(lo), hex(hi), hex(scc))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16, scc & 1


class ScalarAluInterpreter:
    """Deterministic semantic interpreter for RDNA2 scalar ALU instructions.

    Models an array of 106 32-bit SGPRs (s0..s105) and a 1-bit Scalar Condition Code (SCC).
    Enforces register aliasing rules, unaligned/independent register preservation,
    and exact condition code preservation/updates.
    """

    def __init__(self, num_sgprs: int = 106, initial_scc: int = 0) -> None:
        self.num_sgprs = num_sgprs
        self.sgprs = [0] * num_sgprs
        self.scc = initial_scc & 1

    def reset(self, initial_scc: int = 0) -> None:
        self.sgprs = [0] * self.num_sgprs
        self.scc = initial_scc & 1

    def read_sgpr(self, idx: int) -> int:
        assert 0 <= idx < self.num_sgprs, f"SGPR index {idx} out of range"
        return self.sgprs[idx] & M32

    def write_sgpr(self, idx: int, val: int) -> None:
        assert 0 <= idx < self.num_sgprs, f"SGPR index {idx} out of range"
        self.sgprs[idx] = val & M32

    def resolve_src(self, src: int | str) -> int:
        if isinstance(src, int):
            return src & M32
        if isinstance(src, str) and src.startswith("s") and src[1:].isdigit():
            return self.read_sgpr(int(src[1:]))
        raise ValueError(f"Unrecognized scalar source: {src}")

    def exec_s_mov_b32(self, dst: int, src: int | str) -> None:
        val = self.resolve_src(src)
        self.write_sgpr(dst, val)

    def exec_s_sext_i32_i8(self, dst: int, src: int | str) -> None:
        val = self.resolve_src(src)
        res = ref_s_sext_i32_i8(val)
        self.write_sgpr(dst, res)

    def exec_s_sext_i32_i16(self, dst: int, src: int | str) -> None:
        val = self.resolve_src(src)
        res = ref_s_sext_i32_i16(val)
        self.write_sgpr(dst, res)

    def exec_s_flbit_i32(self, dst: int, src: int | str) -> None:
        val = self.resolve_src(src)
        res = ref_s_flbit_i32(val)
        self.write_sgpr(dst, res)

    def exec_s_flbit_i32_b32(self, dst: int, src: int | str) -> None:
        val = self.resolve_src(src)
        res = ref_s_flbit_i32_b32(val)
        self.write_sgpr(dst, res)

    def exec_s_and_b32(self, dst: int, s0: int | str, s1: int | str) -> None:
        v0 = self.resolve_src(s0)
        v1 = self.resolve_src(s1)
        res = (v0 & v1) & M32
        self.write_sgpr(dst, res)
        self.scc = ref_scc_ne_zero(res)

    def exec_s_cmp_eq_u32(self, s0: int | str, s1: int | str) -> None:
        v0 = self.resolve_src(s0)
        v1 = self.resolve_src(s1)
        self.scc = 1 if (v0 == v1) else 0

    def exec_s_cmp_lg_u32(self, s0: int | str, s1: int | str) -> None:
        v0 = self.resolve_src(s0)
        v1 = self.resolve_src(s1)
        self.scc = 1 if (v0 != v1) else 0


class ReferenceSignExtensionTests(unittest.TestCase):
    """Exhaustive and property-based verification of scalar sign extension oracles."""

    def test_ref_s_sext_i32_i8_boundaries(self) -> None:
        self.assertEqual(ref_s_sext_i32_i8(0x80), 0xFFFFFF80)
        self.assertEqual(s32(ref_s_sext_i32_i8(0x80)), -128)

        self.assertEqual(ref_s_sext_i32_i8(0x7F), 0x0000007F)
        self.assertEqual(s32(ref_s_sext_i32_i8(0x7F)), 127)

        self.assertEqual(ref_s_sext_i32_i8(0x00), 0x00000000)
        self.assertEqual(ref_s_sext_i32_i8(0x01), 0x00000001)

        self.assertEqual(ref_s_sext_i32_i8(0xFF), 0xFFFFFFFF)
        self.assertEqual(s32(ref_s_sext_i32_i8(0xFF)), -1)

        self.assertEqual(ref_s_sext_i32_i8(0xFE), 0xFFFFFFFE)
        self.assertEqual(s32(ref_s_sext_i32_i8(0xFE)), -2)

    def test_ref_s_sext_i32_i16_boundaries(self) -> None:
        self.assertEqual(ref_s_sext_i32_i16(0x8000), 0xFFFF8000)
        self.assertEqual(s32(ref_s_sext_i32_i16(0x8000)), -32768)

        self.assertEqual(ref_s_sext_i32_i16(0x7FFF), 0x00007FFF)
        self.assertEqual(s32(ref_s_sext_i32_i16(0x7FFF)), 32767)

        self.assertEqual(ref_s_sext_i32_i16(0x0000), 0x00000000)
        self.assertEqual(ref_s_sext_i32_i16(0x0001), 0x00000001)

        self.assertEqual(ref_s_sext_i32_i16(0xFFFF), 0xFFFFFFFF)
        self.assertEqual(s32(ref_s_sext_i32_i16(0xFFFF)), -1)

        self.assertEqual(ref_s_sext_i32_i16(0xFFFE), 0xFFFFFFFE)
        self.assertEqual(s32(ref_s_sext_i32_i16(0xFFFE)), -2)

    def test_high_bits_ignored(self) -> None:
        patterns_8 = [0xDEADBE00, 0x12345600, 0x7FFFFF00, 0x80000000, 0xFFFFFF00]
        for p in patterns_8:
            self.assertEqual(ref_s_sext_i32_i8(p | 0x80), 0xFFFFFF80)
            self.assertEqual(ref_s_sext_i32_i8(p | 0x7F), 0x0000007F)
            self.assertEqual(ref_s_sext_i32_i8(p | 0x00), 0x00000000)
            self.assertEqual(ref_s_sext_i32_i8(p | 0xFF), 0xFFFFFFFF)

        patterns_16 = [0xDEAD0000, 0x12340000, 0x7FFF0000, 0x80000000, 0xFFFF0000]
        for p in patterns_16:
            self.assertEqual(ref_s_sext_i32_i16(p | 0x8000), 0xFFFF8000)
            self.assertEqual(ref_s_sext_i32_i16(p | 0x7FFF), 0x00007FFF)
            self.assertEqual(ref_s_sext_i32_i16(p | 0x0000), 0x00000000)
            self.assertEqual(ref_s_sext_i32_i16(p | 0xFFFF), 0xFFFFFFFF)

    def test_exhaustive_i8_against_struct_pack(self) -> None:
        for byte_val in range(256):
            expected_signed = struct.unpack("<b", bytes([byte_val]))[0]
            expected_u32 = expected_signed & M32
            self.assertEqual(ref_s_sext_i32_i8(byte_val), expected_u32)
            noisy_input = (byte_val | 0xCAFE00) & M32
            self.assertEqual(ref_s_sext_i32_i8(noisy_input), expected_u32)

    def test_exhaustive_i16_samples_against_struct_pack(self) -> None:
        for word_val in list(range(0, 512)) + list(range(0x7F00, 0x8100)) + list(range(0xFF00, 0x10000, 3)):
            expected_signed = struct.unpack("<h", struct.pack("<H", word_val))[0]
            expected_u32 = expected_signed & M32
            self.assertEqual(ref_s_sext_i32_i16(word_val), expected_u32)
            noisy_input = (word_val | 0x55AA0000) & M32
            self.assertEqual(ref_s_sext_i32_i16(noisy_input), expected_u32)

    def test_algebraic_invariants(self) -> None:
        for v in [0, 1, 0x7F, 0x80, 0xFF, 0x12345678, 0x80000000, 0xFFFFFFFF]:
            res_i8 = ref_s_sext_i32_i8(v)
            self.assertEqual(ref_s_sext_i32_i8(res_i8), res_i8)
            res_i16 = ref_s_sext_i32_i16(v)
            self.assertEqual(ref_s_sext_i32_i16(res_i16), res_i16)

        for v in range(256):
            ext8 = ref_s_sext_i32_i8(v)
            self.assertEqual(ref_s_sext_i32_i16(ext8), ext8)


class ReferenceMsbScanTests(unittest.TestCase):
    """Reference oracle tests for scalar MSB scanning instructions s_flbit_i32 and s_flbit_i32_b32."""

    def test_s_flbit_i32_powers_of_two(self) -> None:
        self.assertEqual(ref_s_flbit_i32(0x40000000), 1)
        self.assertEqual(ref_s_flbit_i32(0x20000000), 2)
        self.assertEqual(ref_s_flbit_i32(0x10000000), 3)
        self.assertEqual(ref_s_flbit_i32(0x00000002), 30)
        self.assertEqual(ref_s_flbit_i32(0x00000001), 31)

        for bit in range(31):
            val = 1 << bit
            expected = 31 - bit
            self.assertEqual(ref_s_flbit_i32(val), expected)

    def test_s_flbit_i32_negative_patterns(self) -> None:
        self.assertEqual(ref_s_flbit_i32(0x80000000), 1)
        self.assertEqual(ref_s_flbit_i32(0xC0000000), 2)
        self.assertEqual(ref_s_flbit_i32(0xE0000000), 3)
        self.assertEqual(ref_s_flbit_i32(0xFFFFFFFE), 31)

    def test_s_flbit_i32_uniform_cases(self) -> None:
        self.assertEqual(ref_s_flbit_i32(0x00000000), M32)
        self.assertEqual(ref_s_flbit_i32(0xFFFFFFFF), M32)

    def test_s_flbit_i32_bitwise_not_symmetry(self) -> None:
        test_values = [
            1, 2, 7, 0x40000000, 0x12345678, 0x7FFFFFFF, 0x55555555,
            0x80000000, 0xC0000000, 0xFFFFFFFE, 0x87654321, 0xAAAAAAAA,
        ]
        for v in test_values:
            not_v = (~v) & M32
            self.assertEqual(ref_s_flbit_i32(v), ref_s_flbit_i32(not_v))

    def test_s_flbit_i32_b32_unsigned_scan(self) -> None:
        self.assertEqual(ref_s_flbit_i32_b32(0x80000000), 0)
        self.assertEqual(ref_s_flbit_i32_b32(0x40000000), 1)
        self.assertEqual(ref_s_flbit_i32_b32(0x00000001), 31)
        self.assertEqual(ref_s_flbit_i32_b32(0x00000000), M32)

        for bit in range(32):
            val = 1 << bit
            self.assertEqual(ref_s_flbit_i32_b32(val), 31 - bit)


class ScalarAluInterpreterTests(unittest.TestCase):
    """Interpreter test suite for register aliasing, unaffected registers, and state isolation."""

    def setUp(self) -> None:
        self.interp = ScalarAluInterpreter(num_sgprs=106, initial_scc=0)

    def test_exec_s_flbit_i32_immediate_and_reg(self) -> None:
        self.interp.exec_s_flbit_i32(10, 0x40000000)
        self.assertEqual(self.interp.read_sgpr(10), 1)

        self.interp.exec_s_mov_b32(11, 0xFFFFFFFE)
        self.interp.exec_s_flbit_i32(12, "s11")
        self.assertEqual(self.interp.read_sgpr(12), 31)

    def test_sgpr_aliasing_in_place(self) -> None:
        self.interp.write_sgpr(4, 0x20000000)
        self.interp.exec_s_flbit_i32(4, "s4")
        self.assertEqual(self.interp.read_sgpr(4), 2)

        self.interp.write_sgpr(5, 0x80000000)
        self.interp.exec_s_flbit_i32(5, "s5")
        self.assertEqual(self.interp.read_sgpr(5), 1)

        self.interp.write_sgpr(6, 0x123480)
        self.interp.exec_s_sext_i32_i8(6, "s6")
        self.assertEqual(self.interp.read_sgpr(6), 0xFFFFFF80)

        self.interp.write_sgpr(7, 0xABCD7FFF)
        self.interp.exec_s_sext_i32_i16(7, "s7")
        self.assertEqual(self.interp.read_sgpr(7), 0x00007FFF)

    def test_sgpr_isolation_and_unaffected_registers(self) -> None:
        for i in range(20):
            self.interp.write_sgpr(i, 0x1000 + i)

        self.interp.exec_s_flbit_i32(10, 0x40000000)
        self.assertEqual(self.interp.read_sgpr(10), 1)

        for i in range(20):
            if i != 10:
                self.assertEqual(self.interp.read_sgpr(i), 0x1000 + i)

    def test_scc_preserved_during_interpreter_execution(self) -> None:
        for init_scc in (0, 1):
            self.interp.reset(initial_scc=init_scc)
            self.interp.exec_s_sext_i32_i8(1, 0x80)
            self.assertEqual(self.interp.scc, init_scc)

            self.interp.exec_s_sext_i32_i16(2, 0x7FFF)
            self.assertEqual(self.interp.scc, init_scc)

            self.interp.exec_s_flbit_i32(3, 0x40000000)
            self.assertEqual(self.interp.scc, init_scc)

            self.interp.exec_s_flbit_i32(4, 0)
            self.assertEqual(self.interp.scc, init_scc)

    def test_compound_sign_extend_then_flbit(self) -> None:
        self.interp.write_sgpr(0, 0x80)
        self.interp.exec_s_sext_i32_i8(1, "s0")
        self.assertEqual(self.interp.read_sgpr(1), 0xFFFFFF80)
        self.interp.exec_s_flbit_i32(2, "s1")
        self.assertEqual(self.interp.read_sgpr(2), 25)

        self.interp.write_sgpr(3, 0x7F)
        self.interp.exec_s_sext_i32_i8(4, "s3")
        self.assertEqual(self.interp.read_sgpr(4), 0x0000007F)
        self.interp.exec_s_flbit_i32(5, "s4")
        self.assertEqual(self.interp.read_sgpr(5), 25)


class ConditionCodeAndEdgePatternTests(unittest.TestCase):
    """Tests for SCC condition code updates, extreme boundaries, and edge bit patterns."""

    def test_scc_ne_zero_rule(self) -> None:
        # Zero yields SCC = 0
        self.assertEqual(ref_scc_ne_zero(0), 0)
        self.assertEqual(ref_scc_ne_zero(0x00000000), 0)

        # Non-zero yields SCC = 1
        for val in [1, -1, 0x7F, 0x80, 0x7FFF, 0x8000, 0x80000000, 0xFFFFFFFF]:
            self.assertEqual(ref_scc_ne_zero(val), 1)

    def test_extreme_boundaries_in_shaders(self) -> None:
        # Explicit required boundaries: -128, 127, -32768, 32767, 0, -1
        boundaries = [-128, 127, -32768, 32767, 0, -1]
        for val in boundaries:
            uval = val & M32
            # Check s_sext_i32_i8
            res8, scc8 = run_op("s_sext_i32_i8", uval, 0)
            self.assertEqual(res8, ref_s_sext_i32_i8(uval))
            self.assertEqual(scc8, 0)

            # Check s_sext_i32_i16
            res16, scc16 = run_op("s_sext_i32_i16", uval, 1)
            self.assertEqual(res16, ref_s_sext_i32_i16(uval))
            self.assertEqual(scc16, 1)

            # Check s_flbit_i32
            res_fl, scc_fl = run_op("s_flbit_i32", uval, 0)
            self.assertEqual(res_fl, ref_s_flbit_i32(uval))
            self.assertEqual(scc_fl, 0)

    def test_alternating_bit_patterns(self) -> None:
        # 0x55555555: 0101 0101...
        val5 = 0x55555555
        self.assertEqual(ref_s_sext_i32_i8(val5), 0x00000055)
        self.assertEqual(ref_s_sext_i32_i16(val5), 0x00005555)
        self.assertEqual(ref_s_flbit_i32(val5), 1)

        # 0xAAAAAAAA: 1010 1010...
        valA = 0xAAAAAAAA
        self.assertEqual(ref_s_sext_i32_i8(valA), 0xFFFFFFAA)
        self.assertEqual(ref_s_sext_i32_i16(valA), 0xFFFFAAAA)
        self.assertEqual(ref_s_flbit_i32(valA), 1)

        # Run both in synthetic shaders
        for pat in (val5, valA):
            self.assertEqual(run_op("s_sext_i32_i8", pat, 0)[0], ref_s_sext_i32_i8(pat))
            self.assertEqual(run_op("s_sext_i32_i16", pat, 0)[0], ref_s_sext_i32_i16(pat))
            self.assertEqual(run_op("s_flbit_i32", pat, 0)[0], ref_s_flbit_i32(pat))

    def test_shader_aliasing_and_chaining(self) -> None:
        # Shader doing in-place aliasing s0 = s_sext_i32_i8(s0) then s0 = s_flbit_i32(s0)
        prog = [
            "s_mov_b32 s0, 0x80",
            "s_sext_i32_i8 s0, s0",
            "s_flbit_i32 s0, s0",
            "s_cmp_eq_u32 s0, 25",
        ]
        res, scc = run_program(prog)
        self.assertEqual(res, 25)
        self.assertEqual(scc, 1)


class NegativeControlTests(unittest.TestCase):
    """Sensitivity tests ensuring mutants in sign extension, scan, or condition codes fail."""

    def test_mutant_zero_extension_caught(self) -> None:
        # A buggy implementation that zero-extends instead of sign-extends
        def mutant_sext_i8(x: int) -> int:
            return x & 0xFF

        def mutant_sext_i16(x: int) -> int:
            return x & 0xFFFF

        # Negative inputs must fail mutant check
        negative_i8 = 0x80
        self.assertEqual(ref_s_sext_i32_i8(negative_i8), 0xFFFFFF80)
        self.assertEqual(mutant_sext_i8(negative_i8), 0x00000080)
        self.assertNotEqual(ref_s_sext_i32_i8(negative_i8), mutant_sext_i8(negative_i8))

        negative_i16 = 0x8000
        self.assertEqual(ref_s_sext_i32_i16(negative_i16), 0xFFFF8000)
        self.assertEqual(mutant_sext_i16(negative_i16), 0x00008000)
        self.assertNotEqual(ref_s_sext_i32_i16(negative_i16), mutant_sext_i16(negative_i16))

    def test_mutant_bit_width_caught(self) -> None:
        # A buggy implementation that uses 7-bit instead of 8-bit sign extension
        def mutant_sext_7bit(x: int) -> int:
            b = x & 0x7F
            return (b - 0x80 if (b & 0x40) else b) & M32

        # Value 0x40 (+64 in 8-bit, but -64 in 7-bit)
        self.assertEqual(ref_s_sext_i32_i8(0x40), 0x00000040)
        self.assertEqual(mutant_sext_7bit(0x40), 0xFFFFFFC0)
        self.assertNotEqual(ref_s_sext_i32_i8(0x40), mutant_sext_7bit(0x40))

    def test_mutant_unmasked_high_bits_caught(self) -> None:
        # A buggy implementation that fails to mask high bits before sign-extending
        def mutant_sext_unmasked(x: int) -> int:
            return ((x << 24) >> 24) & M32  # Python shifts preserve arbitrarily large integers

        dirty_input = 0x123480
        clean_res = ref_s_sext_i32_i8(dirty_input)
        self.assertEqual(clean_res, 0xFFFFFF80)

    def test_mutant_lsb_scan_caught(self) -> None:
        # A buggy implementation that scans from LSB instead of MSB
        def mutant_flbit_lsb(x: int) -> int:
            x &= M32
            if x == 0 or x == M32:
                return M32
            # Find lowest set bit
            for i in range(32):
                if (x >> i) & 1:
                    return i
            return M32

        val = 0x40000000
        self.assertEqual(ref_s_flbit_i32(val), 1)
        self.assertEqual(mutant_flbit_lsb(val), 30)
        self.assertNotEqual(ref_s_flbit_i32(val), mutant_flbit_lsb(val))

    def test_mutant_zero_handling_caught(self) -> None:
        # A buggy implementation returning 0 instead of -1 (0xFFFFFFFF) on zero
        def mutant_flbit_zero_as_0(x: int) -> int:
            return 0 if x == 0 else ref_s_flbit_i32(x)

        self.assertEqual(ref_s_flbit_i32(0), M32)
        self.assertEqual(mutant_flbit_zero_as_0(0), 0)
        self.assertNotEqual(ref_s_flbit_i32(0), mutant_flbit_zero_as_0(0))

    def test_mutant_inverted_scc_caught(self) -> None:
        # Inverted SCC condition: SCC = (D == 0) instead of (D != 0)
        def mutant_scc(d: int) -> int:
            return 1 if (d & M32) == 0 else 0

        self.assertEqual(ref_scc_ne_zero(0), 0)
        self.assertEqual(mutant_scc(0), 1)
        self.assertNotEqual(ref_scc_ne_zero(0), mutant_scc(0))

        self.assertEqual(ref_scc_ne_zero(42), 1)
        self.assertEqual(mutant_scc(42), 0)
        self.assertNotEqual(ref_scc_ne_zero(42), mutant_scc(42))

    def test_mutant_corrupted_neighbor_register_caught(self) -> None:
        # A buggy interpreter that clobbers adjacent register
        interp = ScalarAluInterpreter(num_sgprs=10, initial_scc=0)
        interp.write_sgpr(4, 0x1111)
        interp.write_sgpr(5, 0x2222)
        interp.write_sgpr(6, 0x3333)

        # Standard execution preserves neighbors
        interp.exec_s_sext_i32_i8(5, 0x7F)
        self.assertEqual(interp.read_sgpr(4), 0x1111)
        self.assertEqual(interp.read_sgpr(5), 0x7F)
        self.assertEqual(interp.read_sgpr(6), 0x3333)


class SextFlbitLegacyTests(unittest.TestCase):
    """Integration tests executing synthetic shaders through rdna2spirv and spirv_mini_interp."""

    def test_legacy_reference(self) -> None:
        self.assertEqual(ref_s_sext_i32_i8(0x80), 0xFFFFFF80)
        self.assertEqual(ref_s_sext_i32_i8(0x1234807F), 0x7F)
        self.assertEqual(ref_s_sext_i32_i16(0xABCD8000), 0xFFFF8000)
        self.assertEqual(flbit_i32(1), 31)
        self.assertEqual(flbit_i32(0x40000000), 1)
        self.assertEqual(flbit_i32(0xFFFFFFFE), 31)
        self.assertEqual(flbit_i32(0x80000000), 1)
        self.assertEqual(flbit_i32(0), M32)
        self.assertEqual(flbit_i32(M32), M32)

    def test_values_and_scc_kept(self) -> None:
        for op, fn in OPS.items():
            for x in VALUES:
                for scc_in in (0, 1):
                    with self.subTest(op=op, x=hex(x), scc=scc_in):
                        self.assertEqual(run_op(op, x, scc_in), (fn(x), scc_in))

    def test_seeded(self) -> None:
        rng = random.Random(0x5E)
        for _ in range(30):
            op, x = rng.choice(list(OPS)), rng.getrandbits(32) >> rng.randrange(32)
            if rng.getrandbits(1):
                x = ~x & M32
            with self.subTest(op=op, x=hex(x)):
                self.assertEqual(run_op(op, x, 0)[0], OPS[op](x))


if __name__ == "__main__":
    unittest.main()
