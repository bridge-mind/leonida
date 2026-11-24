"""Semantic interpreter tests for RDNA2 scalar absolute value and bitwise NOT negation.

Covers:
  - s_abs_i32: D.i = abs(S0.i), SCC = (D != 0).
      Two's complement 32-bit signed integer absolute value.
      Boundary wrap: 0x80000000 (-2147483648) maps to 0x80000000, SCC = 1.
      Zero maps to 0, SCC = 0.
  - s_not_b32: D = ~S0 (bitwise NOT 32-bit), SCC = (D != 0).
      Inverts all 32 bits.
      Boundary: 0xFFFFFFFF maps to 0, SCC = 0. All other inputs produce SCC = 1.
  - s_not_b64: D = ~S0 (bitwise NOT 64-bit), SCC = (D != 0).
      Inverts all 64 bits across SGPR pairs or special registers (EXEC, VCC).
      Boundary: 0xFFFFFFFFFFFFFFFF maps to 0, SCC = 0. All other inputs produce SCC = 1.
  - Involution: ~(~x) == x for both 32-bit and 64-bit negation.
  - Two's complement relationship: -x == ~x + 1 (mod 2^32), abs(x) == (x < 0 ? (~x + 1) : x).
  - De Morgan's laws: ~(A & B) == (~A) | (~B), ~(A | B) == (~A) & (~B).
  - Interaction with SCC condition code across sequences and conditional selects.
  - SPIR-V translation validation via rdna2spirv.Gen.
  - Synthetic fragment shader execution via spirv_mini_interp.

Run directly:
    python -I tests/semantic/test_scalar_alu_abs_neg.py
Or via unittest discovery:
    python -m unittest discover -s tests/semantic -p 'test_scalar_alu_abs_neg.py'
"""
from __future__ import annotations

import gc
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

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
SIGN32 = 0x80000000
SIGN64 = 0x8000000000000000
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def u32(v: int) -> int:
    """Normalize integer to 32-bit unsigned."""
    return v & MASK32


def s32(v: int) -> int:
    """Interpret 32-bit unsigned integer as signed two's complement."""
    v &= MASK32
    return v - (1 << 32) if (v & SIGN32) else v


def u64(v: int) -> int:
    """Normalize integer to 64-bit unsigned."""
    return v & MASK64


def s64(v: int) -> int:
    """Interpret 64-bit unsigned integer as signed two's complement."""
    v &= MASK64
    return v - (1 << 64) if (v & SIGN64) else v


def split_dwords(v64: int) -> tuple[int, int]:
    """Split 64-bit integer into (low_u32, high_u32)."""
    v = v64 & MASK64
    return v & MASK32, (v >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_u32, high_u32) into a 64-bit unsigned integer."""
    return (((hi & MASK32) << 32) | (lo & MASK32)) & MASK64


def ref_s_abs_i32(s0: int) -> tuple[int, int]:
    """Reference implementation of s_abs_i32.

    D = |(signed)S0| mod 2^32.
    SCC = 1 if D != 0 else 0.
    In two's complement, 0x80000000 (-2147483648) wraps to 0x80000000, with SCC = 1.
    """
    v = s0 & MASK32
    signed_val = v - (1 << 32) if (v & SIGN32) else v
    abs_val = abs(signed_val) & MASK32
    scc = 1 if abs_val != 0 else 0
    return abs_val, scc


def ref_s_not_b32(s0: int) -> tuple[int, int]:
    """Reference implementation of s_not_b32.

    D = ~S0 mod 2^32 (bitwise NOT).
    SCC = 1 if D != 0 else 0.
    """
    res = (~s0) & MASK32
    scc = 1 if res != 0 else 0
    return res, scc


def ref_s_not_b64(s0: int) -> tuple[int, int]:
    """Reference implementation of s_not_b64.

    D = ~S0 mod 2^64 (bitwise NOT).
    SCC = 1 if D != 0 else 0.
    """
    res = (~s0) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


class ScalarAbsNegInterpreter:
    """Deterministic semantic interpreter for RDNA2 scalar absolute value and negation.

    Tracks 32-bit SGPRs (paired for 64-bit operations), 64-bit EXEC, 64-bit VCC,
    and the 1-bit SCC (Scalar Condition Code) register.
    """

    def __init__(self, num_sgprs: int = 106) -> None:
        self.num_sgprs = num_sgprs
        self.sgpr = [0] * num_sgprs
        self.exec_mask = MASK64
        self.vcc = 0
        self.scc = 0

    def reset(self) -> None:
        self.sgpr = [0] * self.num_sgprs
        self.exec_mask = MASK64
        self.vcc = 0
        self.scc = 0

    def get_sgpr_u32(self, r: int) -> int:
        return self.sgpr[r] & MASK32

    def set_sgpr_u32(self, r: int, val: int) -> None:
        self.sgpr[r] = val & MASK32

    def get_sgpr_u64(self, r: int) -> int:
        lo = self.get_sgpr_u32(r)
        hi = self.get_sgpr_u32(r + 1)
        return join_dwords(lo, hi)

    def set_sgpr_u64(self, r: int, val64: int) -> None:
        lo, hi = split_dwords(val64)
        self.set_sgpr_u32(r, lo)
        self.set_sgpr_u32(r + 1, hi)

    def _resolve_src32(self, operand: int | dict | str) -> int:
        """Resolve a 32-bit source operand (SGPR, immediate, or special register half)."""
        if isinstance(operand, int):
            return operand & MASK32
        if isinstance(operand, dict):
            k = operand.get("k")
            if k == "imm":
                return operand["v"] & MASK32
            if k == "sgpr":
                return self.get_sgpr_u32(operand["r"])
            if k == "special":
                t = operand.get("t", "")
                if t in ("exec", "exec_lo"):
                    return self.exec_mask & MASK32
                if t == "exec_hi":
                    return (self.exec_mask >> 32) & MASK32
                if t in ("vcc", "vcc_lo"):
                    return self.vcc & MASK32
                if t == "vcc_hi":
                    return (self.vcc >> 32) & MASK32
                if t == "scc":
                    return self.scc
            raise ValueError(f"Unknown 32-bit operand dictionary: {operand}")
        if isinstance(operand, str):
            op_lower = operand.lower()
            if op_lower in ("exec", "exec_lo"):
                return self.exec_mask & MASK32
            if op_lower == "exec_hi":
                return (self.exec_mask >> 32) & MASK32
            if op_lower in ("vcc", "vcc_lo"):
                return self.vcc & MASK32
            if op_lower == "vcc_hi":
                return (self.vcc >> 32) & MASK32
            if op_lower == "scc":
                return self.scc
            if op_lower.startswith("s"):
                return self.get_sgpr_u32(int(op_lower[1:]))
        raise TypeError(f"Unsupported 32-bit operand type: {type(operand)} ({operand})")

    def _resolve_src64(self, operand: int | dict | str) -> int:
        """Resolve a 64-bit source operand (SGPR pair, immediate, or special register)."""
        if isinstance(operand, int):
            return operand & MASK64
        if isinstance(operand, dict):
            k = operand.get("k")
            if k == "imm":
                return operand["v"] & MASK64
            if k == "sgpr":
                return self.get_sgpr_u64(operand["r"])
            if k == "special":
                t = operand.get("t", "")
                if t in ("exec", "exec_lo", "exec_hi"):
                    return self.exec_mask
                if t in ("vcc", "vcc_lo", "vcc_hi"):
                    return self.vcc
            raise ValueError(f"Unknown 64-bit operand dictionary: {operand}")
        if isinstance(operand, str):
            op_lower = operand.lower()
            if op_lower in ("exec", "exec_lo", "exec_hi"):
                return self.exec_mask
            if op_lower in ("vcc", "vcc_lo", "vcc_hi"):
                return self.vcc
            if op_lower.startswith("s[") and ":" in op_lower:
                lo_idx = int(op_lower[2:-1].split(":")[0])
                return self.get_sgpr_u64(lo_idx)
            if op_lower.startswith("s"):
                return self.get_sgpr_u64(int(op_lower[1:]))
        raise TypeError(f"Unsupported 64-bit operand type: {type(operand)} ({operand})")

    def _write_dst32(self, dst: int | dict | str, val: int) -> None:
        """Write 32-bit result into target SGPR or special register."""
        val = val & MASK32
        if isinstance(dst, int):
            self.set_sgpr_u32(dst, val)
        elif isinstance(dst, dict):
            k = dst.get("k")
            if k == "sgpr":
                self.set_sgpr_u32(dst["r"], val)
            elif k == "special":
                t = dst.get("t", "")
                if t in ("exec", "exec_lo"):
                    self.exec_mask = (self.exec_mask & (MASK32 << 32)) | val
                elif t == "exec_hi":
                    self.exec_mask = (self.exec_mask & MASK32) | (val << 32)
                elif t in ("vcc", "vcc_lo"):
                    self.vcc = (self.vcc & (MASK32 << 32)) | val
                elif t == "vcc_hi":
                    self.vcc = (self.vcc & MASK32) | (val << 32)
                elif t == "scc":
                    self.scc = 1 if val != 0 else 0
        elif isinstance(dst, str):
            d_lower = dst.lower()
            if d_lower.startswith("s"):
                self.set_sgpr_u32(int(d_lower[1:]), val)
            elif d_lower in ("exec", "exec_lo"):
                self.exec_mask = (self.exec_mask & (MASK32 << 32)) | val
            elif d_lower == "exec_hi":
                self.exec_mask = (self.exec_mask & MASK32) | (val << 32)
            elif d_lower in ("vcc", "vcc_lo"):
                self.vcc = (self.vcc & (MASK32 << 32)) | val
            elif d_lower == "vcc_hi":
                self.vcc = (self.vcc & MASK32) | (val << 32)
            elif d_lower == "scc":
                self.scc = 1 if val != 0 else 0

    def _write_dst64(self, dst: int | dict | str, val: int) -> None:
        """Write 64-bit result into target SGPR pair or special register."""
        val = val & MASK64
        if isinstance(dst, int):
            self.set_sgpr_u64(dst, val)
        elif isinstance(dst, dict):
            k = dst.get("k")
            if k == "sgpr":
                self.set_sgpr_u64(dst["r"], val)
            elif k == "special":
                t = dst.get("t", "")
                if t in ("exec", "exec_lo", "exec_hi"):
                    self.exec_mask = val
                elif t in ("vcc", "vcc_lo", "vcc_hi"):
                    self.vcc = val
        elif isinstance(dst, str):
            d_lower = dst.lower()
            if d_lower in ("exec", "exec_lo", "exec_hi"):
                self.exec_mask = val
            elif d_lower in ("vcc", "vcc_lo", "vcc_hi"):
                self.vcc = val
            elif d_lower.startswith("s[") and ":" in d_lower:
                lo_idx = int(d_lower[2:-1].split(":")[0])
                self.set_sgpr_u64(lo_idx, val)
            elif d_lower.startswith("s"):
                self.set_sgpr_u64(int(d_lower[1:]), val)

    def step(self, op: str, dst: int | dict | str, s0: int | dict | str,
             s1: int | dict | str | None = None) -> tuple[int, int]:
        """Execute a single scalar ALU instruction and return (result, scc)."""
        if op == "s_abs_i32":
            val = self._resolve_src32(s0)
            res, scc = ref_s_abs_i32(val)
            self._write_dst32(dst, res)
            self.scc = scc
            return res, self.scc

        if op == "s_not_b32":
            val = self._resolve_src32(s0)
            res, scc = ref_s_not_b32(val)
            self._write_dst32(dst, res)
            self.scc = scc
            return res, self.scc

        if op == "s_not_b64":
            val = self._resolve_src64(s0)
            res, scc = ref_s_not_b64(val)
            self._write_dst64(dst, res)
            self.scc = scc
            return res, self.scc

        if op == "s_mov_b32":
            val = self._resolve_src32(s0)
            self._write_dst32(dst, val)
            return val, self.scc

        if op == "s_mov_b64":
            val = self._resolve_src64(s0)
            self._write_dst64(dst, val)
            return val, self.scc

        if op == "s_add_u32":
            v0 = self._resolve_src32(s0)
            v1 = self._resolve_src32(s1 if s1 is not None else 0)
            full = v0 + v1
            res = full & MASK32
            self.scc = 1 if full >= (1 << 32) else 0
            self._write_dst32(dst, res)
            return res, self.scc

        if op == "s_and_b32":
            v0 = self._resolve_src32(s0)
            v1 = self._resolve_src32(s1 if s1 is not None else 0)
            res = (v0 & v1) & MASK32
            self.scc = 1 if res != 0 else 0
            self._write_dst32(dst, res)
            return res, self.scc

        if op == "s_and_b64":
            v0 = self._resolve_src64(s0)
            v1 = self._resolve_src64(s1 if s1 is not None else 0)
            res = (v0 & v1) & MASK64
            self.scc = 1 if res != 0 else 0
            self._write_dst64(dst, res)
            return res, self.scc

        if op == "s_or_b32":
            v0 = self._resolve_src32(s0)
            v1 = self._resolve_src32(s1 if s1 is not None else 0)
            res = (v0 | v1) & MASK32
            self.scc = 1 if res != 0 else 0
            self._write_dst32(dst, res)
            return res, self.scc

        if op == "s_or_b64":
            v0 = self._resolve_src64(s0)
            v1 = self._resolve_src64(s1 if s1 is not None else 0)
            res = (v0 | v1) & MASK64
            self.scc = 1 if res != 0 else 0
            self._write_dst64(dst, res)
            return res, self.scc

        if op == "s_xor_b32":
            v0 = self._resolve_src32(s0)
            v1 = self._resolve_src32(s1 if s1 is not None else 0)
            res = (v0 ^ v1) & MASK32
            self.scc = 1 if res != 0 else 0
            self._write_dst32(dst, res)
            return res, self.scc

        if op == "s_xor_b64":
            v0 = self._resolve_src64(s0)
            v1 = self._resolve_src64(s1 if s1 is not None else 0)
            res = (v0 ^ v1) & MASK64
            self.scc = 1 if res != 0 else 0
            self._write_dst64(dst, res)
            return res, self.scc

        if op == "s_cselect_b32":
            v0 = self._resolve_src32(s0)
            v1 = self._resolve_src32(s1 if s1 is not None else 0)
            res = v0 if self.scc != 0 else v1
            self._write_dst32(dst, res)
            return res, self.scc

        if op == "s_cselect_b64":
            v0 = self._resolve_src64(s0)
            v1 = self._resolve_src64(s1 if s1 is not None else 0)
            res = v0 if self.scc != 0 else v1
            self._write_dst64(dst, res)
            return res, self.scc

        raise NotImplementedError(f"Unsupported scalar opcode: {op}")

    def execute_ir(self, insn: dict) -> tuple[int, int]:
        """Execute an instruction dictionary matching the rdna2spirv parser IR format."""
        op = insn["op"]
        args = insn["args"]
        dst = args[0]
        s0 = args[1]
        s1 = args[2] if len(args) > 2 else None
        return self.step(op, dst, s0, s1)

    def run_sequence(self, instructions: list[dict | tuple]) -> tuple[int, int]:
        """Execute an ordered sequence of instructions and return the final (result, scc)."""
        last_res, last_scc = 0, self.scc
        for item in instructions:
            if isinstance(item, dict):
                last_res, last_scc = self.execute_ir(item)
            else:
                op = item[0]
                dst = item[1]
                s0 = item[2]
                s1 = item[3] if len(item) > 3 else None
                last_res, last_scc = self.step(op, dst, s0, s1)
        return last_res, last_scc


class TestScalarAbsI32Reference(unittest.TestCase):
    """Direct mathematical unit tests for 32-bit scalar integer absolute value (s_abs_i32)."""

    def test_abs_zero(self):
        res, scc = ref_s_abs_i32(0)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_abs_positive_values(self):
        positives = [1, 2, 42, 100, 1000, 0x12345678, 0x7FFFFFFF]
        for val in positives:
            with self.subTest(val=hex(val)):
                res, scc = ref_s_abs_i32(val)
                self.assertEqual(res, val)
                self.assertEqual(scc, 1)

    def test_abs_negative_values(self):
        cases = [
            (-1, 1),
            (-2, 2),
            (-42, 42),
            (-1000, 1000),
            (-0x7FFFFFFF, 0x7FFFFFFF),
            (0xFFFFFFFF, 1),
            (0xFFFFFFFE, 2),
            (0x80000001, 0x7FFFFFFF),
        ]
        for inp, want in cases:
            with self.subTest(inp=hex(inp & MASK32)):
                res, scc = ref_s_abs_i32(inp)
                self.assertEqual(res, want)
                self.assertEqual(scc, 1)

    def test_abs_int_min_overflow_wrap(self):
        # 0x80000000 (-2147483648) wraps back to 0x80000000 in two's complement.
        # Hardware SCC = (D != 0) is 1.
        res, scc = ref_s_abs_i32(0x80000000)
        self.assertEqual(res, 0x80000000)
        self.assertEqual(scc, 1)

    def test_abs_idempotence(self):
        samples = [0, 1, -1, 42, -42, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF, 0xDEADBEEF]
        for val in samples:
            with self.subTest(val=hex(val & MASK32)):
                first_res, first_scc = ref_s_abs_i32(val)
                second_res, second_scc = ref_s_abs_i32(first_res)
                self.assertEqual(second_res, first_res)
                self.assertEqual(second_scc, first_scc)

    def test_abs_non_negative_range(self):
        for val in range(-250, 251):
            res, scc = ref_s_abs_i32(val)
            self.assertGreaterEqual(res, 0)
            self.assertLessEqual(res, 0x7FFFFFFF)
            self.assertEqual(scc, 0 if val == 0 else 1)


class TestScalarNotB32Reference(unittest.TestCase):
    """Direct mathematical unit tests for 32-bit scalar bitwise NOT (s_not_b32)."""

    def test_not_zero_to_all_ones(self):
        res, scc = ref_s_not_b32(0)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_not_all_ones_to_zero(self):
        # 0xFFFFFFFF is the only 32-bit input that yields 0 and clears SCC.
        res, scc = ref_s_not_b32(0xFFFFFFFF)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_not_alternating_bits(self):
        self.assertEqual(ref_s_not_b32(0x55555555), (0xAAAAAAAA, 1))
        self.assertEqual(ref_s_not_b32(0xAAAAAAAA), (0x55555555, 1))
        self.assertEqual(ref_s_not_b32(0x33333333), (0xCCCCCCCC, 1))
        self.assertEqual(ref_s_not_b32(0x0F0F0F0F), (0xF0F0F0F0, 1))

    def test_not_walking_ones(self):
        for bit in range(32):
            pattern = 1 << bit
            res, scc = ref_s_not_b32(pattern)
            expected = (~pattern) & MASK32
            self.assertEqual(res, expected)
            self.assertEqual(scc, 1)

    def test_not_walking_zeroes(self):
        for bit in range(32):
            pattern = (~(1 << bit)) & MASK32
            res, scc = ref_s_not_b32(pattern)
            expected = 1 << bit
            self.assertEqual(res, expected)
            self.assertEqual(scc, 1)

    def test_not_involution(self):
        samples = [0, 1, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 0x12345678, 0xCAFEBABE]
        for val in samples:
            with self.subTest(val=hex(val)):
                r1, _ = ref_s_not_b32(val)
                r2, _ = ref_s_not_b32(r1)
                self.assertEqual(r2, val & MASK32)

    def test_not_xor_equivalence(self):
        rng = random.Random(42)
        for _ in range(100):
            val = rng.getrandbits(32)
            res, _ = ref_s_not_b32(val)
            self.assertEqual(res, val ^ MASK32)


class TestScalarNotB64Reference(unittest.TestCase):
    """Direct mathematical unit tests for 64-bit scalar bitwise NOT (s_not_b64)."""

    def test_not_b64_zero_to_all_ones(self):
        res, scc = ref_s_not_b64(0)
        self.assertEqual(res, MASK64)
        self.assertEqual(scc, 1)

    def test_not_b64_all_ones_to_zero(self):
        # MASK64 is the only 64-bit input that yields 0 and clears SCC.
        res, scc = ref_s_not_b64(MASK64)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_not_b64_half_zeros(self):
        res1, scc1 = ref_s_not_b64(0x00000000FFFFFFFF)
        self.assertEqual(res1, 0xFFFFFFFF00000000)
        self.assertEqual(scc1, 1)

        res2, scc2 = ref_s_not_b64(0xFFFFFFFF00000000)
        self.assertEqual(res2, 0x00000000FFFFFFFF)
        self.assertEqual(scc2, 1)

    def test_not_b64_alternating_bits(self):
        res1, scc1 = ref_s_not_b64(0x5555555555555555)
        self.assertEqual(res1, 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(scc1, 1)

        res2, scc2 = ref_s_not_b64(0xAAAAAAAAAAAAAAAA)
        self.assertEqual(res2, 0x5555555555555555)
        self.assertEqual(scc2, 1)

    def test_not_b64_walking_bits(self):
        for bit in range(64):
            pattern = 1 << bit
            res, scc = ref_s_not_b64(pattern)
            expected = (~pattern) & MASK64
            self.assertEqual(res, expected)
            self.assertEqual(scc, 1)

    def test_not_b64_involution(self):
        samples = [
            0,
            1,
            MASK64,
            0x7FFFFFFFFFFFFFFF,
            0x8000000000000000,
            0x123456789ABCDEF0,
            0xDEADBEEFCAFEBABE,
        ]
        for val in samples:
            with self.subTest(val=hex(val)):
                r1, _ = ref_s_not_b64(val)
                r2, _ = ref_s_not_b64(r1)
                self.assertEqual(r2, val & MASK64)


class TestScalarAbsNegInterpreter(unittest.TestCase):
    """Unit tests for ScalarAbsNegInterpreter state tracking and instruction execution."""

    def setUp(self):
        self.interp = ScalarAbsNegInterpreter()

    def test_interpreter_abs_i32_state(self):
        # s_abs_i32 with negative input: -42 -> 42, SCC = 1
        self.interp.step("s_mov_b32", "s0", -42)
        res, scc = self.interp.step("s_abs_i32", "s1", "s0")
        self.assertEqual(res, 42)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.get_sgpr_u32(1), 42)
        self.assertEqual(self.interp.scc, 1)

        # s_abs_i32 with zero: 0 -> 0, SCC = 0
        self.interp.step("s_mov_b32", "s0", 0)
        res, scc = self.interp.step("s_abs_i32", "s1", "s0")
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)
        self.assertEqual(self.interp.get_sgpr_u32(1), 0)
        self.assertEqual(self.interp.scc, 0)

        # s_abs_i32 with 0x80000000: wraps to 0x80000000, SCC = 1
        self.interp.step("s_mov_b32", "s0", 0x80000000)
        res, scc = self.interp.step("s_abs_i32", "s1", "s0")
        self.assertEqual(res, 0x80000000)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.get_sgpr_u32(1), 0x80000000)
        self.assertEqual(self.interp.scc, 1)

    def test_interpreter_not_b32_state(self):
        # Inverting 0 produces 0xFFFFFFFF with SCC = 1
        self.interp.step("s_mov_b32", "s0", 0)
        res, scc = self.interp.step("s_not_b32", "s1", "s0")
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.get_sgpr_u32(1), 0xFFFFFFFF)
        self.assertEqual(self.interp.scc, 1)

        # Inverting 0xFFFFFFFF produces 0 with SCC = 0
        res, scc = self.interp.step("s_not_b32", "s2", "s1")
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)
        self.assertEqual(self.interp.get_sgpr_u32(2), 0)
        self.assertEqual(self.interp.scc, 0)

    def test_interpreter_not_b64_state(self):
        # Write 64-bit value to s[0:1], negate into s[2:3]
        val64 = 0x12345678_9ABCDEF0
        self.interp.set_sgpr_u64(0, val64)
        res, scc = self.interp.step("s_not_b64", "s2", "s0")
        expected = (~val64) & MASK64
        self.assertEqual(res, expected)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.get_sgpr_u64(2), expected)

        # Inverting all ones produces 0 and clears SCC
        self.interp.set_sgpr_u64(0, MASK64)
        res, scc = self.interp.step("s_not_b64", "s4", "s0")
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)
        self.assertEqual(self.interp.get_sgpr_u64(4), 0)
        self.assertEqual(self.interp.scc, 0)

    def test_adjacent_registers_remain_untouched(self):
        self.interp.set_sgpr_u32(0, 0xAAAAAAAA)
        self.interp.set_sgpr_u32(1, 0xBBBBBBBB)
        self.interp.set_sgpr_u32(4, 0xCCCCCCCC)
        self.interp.set_sgpr_u32(5, 0xDDDDDDDD)

        # Write 64-bit result into s[2:3]
        self.interp.step("s_not_b64", 2, 0x0000000000000000)
        self.assertEqual(self.interp.get_sgpr_u32(0), 0xAAAAAAAA)
        self.assertEqual(self.interp.get_sgpr_u32(1), 0xBBBBBBBB)
        self.assertEqual(self.interp.get_sgpr_u32(2), 0xFFFFFFFF)
        self.assertEqual(self.interp.get_sgpr_u32(3), 0xFFFFFFFF)
        self.assertEqual(self.interp.get_sgpr_u32(4), 0xCCCCCCCC)
        self.assertEqual(self.interp.get_sgpr_u32(5), 0xDDDDDDDD)

    def test_special_registers_exec_and_vcc(self):
        # Invert EXEC lane mask: 0xFFFFFFFF_00000000 -> 0x00000000_FFFFFFFF
        self.interp.exec_mask = 0xFFFFFFFF_00000000
        res, scc = self.interp.step("s_not_b64", "exec", "exec")
        self.assertEqual(res, 0x00000000_FFFFFFFF)
        self.assertEqual(self.interp.exec_mask, 0x00000000_FFFFFFFF)
        self.assertEqual(scc, 1)

        # Invert VCC mask: 0 -> MASK64
        self.interp.vcc = 0
        res, scc = self.interp.step("s_not_b64", "vcc", "vcc")
        self.assertEqual(res, MASK64)
        self.assertEqual(self.interp.vcc, MASK64)
        self.assertEqual(scc, 1)

        # Lane partition properties: (exec & ~exec) == 0, (exec | ~exec) == MASK64
        self.interp.exec_mask = 0xAAAAAAAAAAAAAAAA
        self.interp.step("s_not_b64", "s0", "exec")
        self.assertEqual(self.interp.get_sgpr_u64(0), 0x5555555555555555)
        self.assertEqual(self.interp.exec_mask & self.interp.get_sgpr_u64(0), 0)
        self.assertEqual(self.interp.exec_mask | self.interp.get_sgpr_u64(0), MASK64)

    def test_two_complement_negation_via_not_and_add(self):
        # Mathematical identity: -x == ~x + 1
        test_values = [1, 2, 42, 100, 0x12345678, 0x7FFFFFFF, 0x80000000]
        for val in test_values:
            with self.subTest(val=hex(val)):
                self.interp.step("s_mov_b32", "s0", val)
                self.interp.step("s_not_b32", "s1", "s0")
                neg_res, _ = self.interp.step("s_add_u32", "s2", "s1", 1)

                expected_neg = ((-val) & MASK32)
                self.assertEqual(neg_res, expected_neg)

                # For negative inputs, s_abs_i32 matches two's complement negation
                abs_res, _ = self.interp.step("s_abs_i32", "s3", neg_res)
                self.assertEqual(abs_res, val if val != 0x80000000 else 0x80000000)

    def test_de_morgans_laws_32(self):
        # ~(A & B) == (~A) | (~B)
        # ~(A | B) == (~A) & (~B)
        pairs = [
            (0x12345678, 0x9ABCDEF0),
            (0x55555555, 0xAAAAAAAA),
            (0x0000FFFF, 0xFFFF0000),
            (0xFFFFFFFF, 0x12345678),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                # Left side: ~(A & B)
                self.interp.step("s_and_b32", "s0", a, b)
                lhs1, _ = self.interp.step("s_not_b32", "s1", "s0")

                # Right side: (~A) | (~B)
                self.interp.step("s_not_b32", "s2", a)
                self.interp.step("s_not_b32", "s3", b)
                rhs1, _ = self.interp.step("s_or_b32", "s4", "s2", "s3")
                self.assertEqual(lhs1, rhs1)

                # Left side: ~(A | B)
                self.interp.step("s_or_b32", "s0", a, b)
                lhs2, _ = self.interp.step("s_not_b32", "s1", "s0")

                # Right side: (~A) & (~B)
                rhs2, _ = self.interp.step("s_and_b32", "s5", "s2", "s3")
                self.assertEqual(lhs2, rhs2)

    def test_de_morgans_laws_64(self):
        pairs = [
            (0x12345678_9ABCDEF0, 0x0FEDCBA9_87654321),
            (0x55555555_55555555, 0xAAAAAAA_AAAAAAAA),
            (0x00000000_FFFFFFFF, 0xFFFFFFFF_00000000),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                # Left side: ~(A & B)
                self.interp.step("s_and_b64", "s0", a, b)
                lhs1, _ = self.interp.step("s_not_b64", "s2", "s0")

                # Right side: (~A) | (~B)
                self.interp.step("s_not_b64", "s4", a)
                self.interp.step("s_not_b64", "s6", b)
                rhs1, _ = self.interp.step("s_or_b64", "s8", "s4", "s6")
                self.assertEqual(lhs1, rhs1)

                # Left side: ~(A | B)
                self.interp.step("s_or_b64", "s0", a, b)
                lhs2, _ = self.interp.step("s_not_b64", "s2", "s0")

                # Right side: (~A) & (~B)
                rhs2, _ = self.interp.step("s_and_b64", "s10", "s4", "s6")
                self.assertEqual(lhs2, rhs2)

    def test_cselect_branching_on_scc(self):
        # When s_abs_i32 result is zero, SCC is 0 -> s_cselect_b32 chooses S1
        self.interp.step("s_abs_i32", "s0", 0)
        self.interp.step("s_cselect_b32", "s1", 100, 200)
        self.assertEqual(self.interp.get_sgpr_u32(1), 200)

        # When s_abs_i32 result is non-zero, SCC is 1 -> s_cselect_b32 chooses S0
        self.interp.step("s_abs_i32", "s0", -5)
        self.interp.step("s_cselect_b32", "s1", 100, 200)
        self.assertEqual(self.interp.get_sgpr_u32(1), 100)

        # When s_not_b32 result is zero (input 0xFFFFFFFF), SCC is 0 -> S1 chosen
        self.interp.step("s_not_b32", "s0", 0xFFFFFFFF)
        self.interp.step("s_cselect_b32", "s2", 300, 400)
        self.assertEqual(self.interp.get_sgpr_u32(2), 400)


class TestSpirvTranslatorIntegration(unittest.TestCase):
    """Validation of rdna2spirv instruction lowering for s_abs_i32, s_not_b32, and s_not_b64."""

    def test_rdna2spirv_abs_i32_emission(self):
        gen = rdna2spirv.Gen("ps")
        insn = {
            "op": "s_abs_i32",
            "args": [{"k": "sgpr", "r": 0}, {"k": "sgpr", "r": 1}],
            "mods": [],
        }
        gen._insn(insn)
        body = "\n".join(gen.body)

        self.assertIn("OpBitcast %int", body)
        self.assertIn("OpExtInst %int %glsl SAbs", body)
        self.assertIn("OpBitcast %uint", body)
        self.assertIn("OpINotEqual %bool", body)
        self.assertIn("OpSelect %uint", body)
        self.assertIn("OpStore %scc", body)

    def test_rdna2spirv_not_b32_emission(self):
        gen = rdna2spirv.Gen("ps")
        insn = {
            "op": "s_not_b32",
            "args": [{"k": "sgpr", "r": 2}, {"k": "sgpr", "r": 3}],
            "mods": [],
        }
        gen._insn(insn)
        body = "\n".join(gen.body)

        self.assertIn("OpNot %uint", body)
        self.assertIn("OpINotEqual %bool", body)
        self.assertIn("OpSelect %uint", body)
        self.assertIn("OpStore %scc", body)

    def test_rdna2spirv_not_b64_emission(self):
        # 1. Scalar SGPR pair negation
        gen = rdna2spirv.Gen("ps")
        insn = {
            "op": "s_not_b64",
            "args": [{"k": "sgpr", "r": 4, "n": 2}, {"k": "sgpr", "r": 6, "n": 2}],
            "mods": [],
        }
        gen._insn(insn)
        body = "\n".join(gen.body)

        self.assertIn("OpNot %uint", body)
        self.assertIn("OpBitwiseOr %uint", body)
        self.assertIn("OpStore %scc", body)

        # 2. Mask register (EXEC) negation emits wave-vote reduction
        gen_exec = rdna2spirv.Gen("ps")
        insn_exec = {
            "op": "s_not_b64",
            "args": [{"k": "special", "t": "exec"}, {"k": "special", "t": "exec"}],
            "mods": [],
        }
        gen_exec._insn(insn_exec)
        body_exec = "\n".join(gen_exec.body)

        self.assertIn("OpNot %uint", body_exec)
        self.assertIn("OpGroupNonUniformBallot", body_exec)
        self.assertIn("%c20", body_exec)
        self.assertIn("OpStore %scc", body_exec)

    def test_spirv_math_model_equivalence(self):
        rng = random.Random(1337)
        for _ in range(300):
            val = rng.getrandbits(32)

            # 1. SPIR-V s_abs_i32 math emulation:
            # OpBitcast %int -> OpExtInst %glsl SAbs -> OpBitcast %uint -> OpINotEqual -> OpSelect
            signed_i = val - (1 << 32) if (val & SIGN32) else val
            spv_abs = abs(signed_i) & MASK32
            spv_scc_abs = 1 if spv_abs != 0 else 0

            ref_abs_val, ref_abs_scc = ref_s_abs_i32(val)
            self.assertEqual(spv_abs, ref_abs_val)
            self.assertEqual(spv_scc_abs, ref_abs_scc)

            # 2. SPIR-V s_not_b32 math emulation:
            # OpNot %uint -> OpINotEqual -> OpSelect
            spv_not = (~val) & MASK32
            spv_scc_not = 1 if spv_not != 0 else 0

            ref_not_val, ref_not_scc = ref_s_not_b32(val)
            self.assertEqual(spv_not, ref_not_val)
            self.assertEqual(spv_scc_not, ref_not_scc)


class TestFragmentShaderExecution(unittest.TestCase):
    """End-to-end execution of synthetic shaders exercising s_abs_i32 and s_not_b32 via spirv_mini_interp."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.disasm_dir = os.path.join(cls.tmp_dir.name, "disasm")
        os.mkdir(cls.disasm_dir)
        cls.src_path = os.path.join(cls.disasm_dir, "ps_abs_neg_test.s")
        with open(os.path.join(cls.tmp_dir.name, "ps_abs_neg_test.dxbc"), "wb") as f:
            f.write(STUB_DXBC)

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        if cls.tmp_dir is not None:
            cls.tmp_dir.cleanup()

    def _run_ps_lines(self, lines: list[str]) -> tuple[int, int]:
        """Translate assembly snippet into SPIR-V, execute in mini-interp, and return (mrt0[0], mrt0[1])."""
        asm = "; synthetic test\n" + "".join(f"\t{line}\n" for line in lines) + \
              "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
        with open(self.src_path, "w", encoding="utf-8") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(self.src_path)
        out = si.run_fragment(spv)["%out_mrt0"]
        return out[0], out[1]

    def test_synthetic_shader_s_abs_i32(self):
        values = [0, 1, 42, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE, 0x12345678]
        for x in values:
            with self.subTest(x=hex(x)):
                want_val, want_scc = ref_s_abs_i32(x)
                lines = [
                    f"s_mov_b32 s2, {x:#x}",
                    "s_abs_i32 s0, s2",
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                ]
                got_val, got_scc = self._run_ps_lines(lines)
                self.assertEqual(got_val, want_val)
                self.assertEqual(got_scc, want_scc)

    def test_synthetic_shader_s_not_b32(self):
        values = [0, 1, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA, 0x12345678, 0x0000FFFF]
        for x in values:
            with self.subTest(x=hex(x)):
                want_val, want_scc = ref_s_not_b32(x)
                lines = [
                    f"s_mov_b32 s2, {x:#x}",
                    "s_not_b32 s0, s2",
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                ]
                got_val, got_scc = self._run_ps_lines(lines)
                self.assertEqual(got_val, want_val)
                self.assertEqual(got_scc, want_scc)

    def test_synthetic_shader_s_not_b64(self):
        cases = [
            (0x12345678_9ABCDEF0, 0x12345678, 0x9ABCDEF0),
            (0x00000000_00000000, 0x00000000, 0x00000000),
            (0xFFFFFFFF_FFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0x55555555_AAAAAAAA, 0x55555555, 0xAAAAAAAA),
        ]
        for val64, hi, lo in cases:
            with self.subTest(val=hex(val64)):
                want_lo, _ = split_dwords((~val64) & MASK64)
                want_scc = 1 if ((~val64) & MASK64) != 0 else 0
                lines = [
                    f"s_mov_b32 s4, {lo:#x}",
                    f"s_mov_b32 s5, {hi:#x}",
                    "s_not_b64 s[2:3], s[4:5]",
                    "s_cselect_b32 s10, 1, 0",
                    "v_mov_b32 v0, s2",
                    "v_mov_b32 v1, s10",
                ]
                got_val, got_scc = self._run_ps_lines(lines)
                self.assertEqual(got_val, want_lo)
                self.assertEqual(got_scc, want_scc)

    def test_synthetic_shader_negation_chain(self):
        # In a shader, negate an integer by bitwise NOT followed by adding 1:
        # -42 -> 42
        lines = [
            "s_mov_b32 s2, -42",
            "s_not_b32 s0, s2",
            "s_add_u32 s0, s0, 1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
        ]
        got_val, _ = self._run_ps_lines(lines)
        self.assertEqual(got_val, 42)


if __name__ == "__main__":
    unittest.main()
