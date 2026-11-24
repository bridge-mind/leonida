from __future__ import annotations

"""Semantic interpreter tests for RDNA2 64-bit scalar bitwise ALU operations.

Covers:
  * s_and_b64   (D = S0 & S1, SCC = (D != 0))
  * s_or_b64    (D = S0 | S1, SCC = (D != 0))
  * s_xor_b64   (D = S0 ^ S1, SCC = (D != 0))
  * s_andn2_b64 (D = S0 & ~S1, SCC = (D != 0))
  * s_orn2_b64  (D = S0 | ~S1, SCC = (D != 0))
  * s_nand_b64  (D = ~(S0 & S1), SCC = (D != 0))
  * s_nor_b64   (D = ~(S0 | S1), SCC = (D != 0))
  * s_xnor_b64  (D = ~(S0 ^ S1), SCC = (D != 0))
  * 64-bit scalar register pair packing (s[r:r+1]) and special registers (EXEC, VCC)
  * SCC (Scalar Condition Code) flag transitions across 64-bit boundaries (zero vs non-zero)
  * Boolean algebra identities (commutativity, associativity, distributivity, De Morgan's laws, involution)
  * Real-world wave execution mask manipulation (active lane partitioning, saving/restoring EXEC, VCC accumulation)
  * Lowering equivalence to SPIR-V bitwise instructions (OpBitwiseAnd, OpBitwiseOr, OpBitwiseXor, OpNot)

Run directly:
    python -I tests/semantic/test_scalar_alu_logic.py
Or via unittest discovery:
    python -m unittest discover -s tests/semantic -p test_scalar_alu_logic.py -v
"""

import gc
import os
import random
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


class ScalarLogicInterpreter:
    """Deterministic semantic interpreter for RDNA2 64-bit scalar bitwise logic.

    Tracks 32-bit SGPRs (paired for 64-bit operations), 64-bit EXEC, 64-bit VCC,
    and the 1-bit SCC (Scalar Condition Code) register.
    """

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgpr = [0] * num_sgprs
        self.exec_mask = MASK64
        self.vcc = 0
        self.scc = 0

    def reset(self) -> None:
        self.sgpr = [0] * len(self.sgpr)
        self.exec_mask = MASK64
        self.vcc = 0
        self.scc = 0

    @staticmethod
    def eval_s_and_b64(s0: int, s1: int) -> tuple[int, int]:
        """Execute s_and_b64: D = S0 & S1; SCC = (D != 0)."""
        d = (s0 & s1) & MASK64
        scc = 1 if d != 0 else 0
        return d, scc

    @staticmethod
    def eval_s_or_b64(s0: int, s1: int) -> tuple[int, int]:
        """Execute s_or_b64: D = S0 | S1; SCC = (D != 0)."""
        d = (s0 | s1) & MASK64
        scc = 1 if d != 0 else 0
        return d, scc

    @staticmethod
    def eval_s_xor_b64(s0: int, s1: int) -> tuple[int, int]:
        """Execute s_xor_b64: D = S0 ^ S1; SCC = (D != 0)."""
        d = (s0 ^ s1) & MASK64
        scc = 1 if d != 0 else 0
        return d, scc

    @staticmethod
    def eval_s_andn2_b64(s0: int, s1: int) -> tuple[int, int]:
        """Execute s_andn2_b64: D = S0 & ~S1; SCC = (D != 0)."""
        d = (s0 & (~s1 & MASK64)) & MASK64
        scc = 1 if d != 0 else 0
        return d, scc

    @staticmethod
    def eval_s_orn2_b64(s0: int, s1: int) -> tuple[int, int]:
        """Execute s_orn2_b64: D = S0 | ~S1; SCC = (D != 0)."""
        d = (s0 | (~s1 & MASK64)) & MASK64
        scc = 1 if d != 0 else 0
        return d, scc

    @staticmethod
    def eval_s_nand_b64(s0: int, s1: int) -> tuple[int, int]:
        """Execute s_nand_b64: D = ~(S0 & S1); SCC = (D != 0)."""
        d = (~(s0 & s1)) & MASK64
        scc = 1 if d != 0 else 0
        return d, scc

    @staticmethod
    def eval_s_nor_b64(s0: int, s1: int) -> tuple[int, int]:
        """Execute s_nor_b64: D = ~(S0 | S1); SCC = (D != 0)."""
        d = (~(s0 | s1)) & MASK64
        scc = 1 if d != 0 else 0
        return d, scc

    @staticmethod
    def eval_s_xnor_b64(s0: int, s1: int) -> tuple[int, int]:
        """Execute s_xnor_b64: D = ~(S0 ^ S1); SCC = (D != 0)."""
        d = (~(s0 ^ s1)) & MASK64
        scc = 1 if d != 0 else 0
        return d, scc

    def _resolve_src64(self, operand: int | dict | str) -> int:
        """Resolve a 64-bit source operand (SGPR pair, special register, or immediate)."""
        if isinstance(operand, int):
            return operand & MASK64
        if isinstance(operand, dict):
            k = operand.get("k")
            if k == "imm":
                return operand["v"] & MASK64
            if k == "sgpr":
                r = operand["r"]
                lo = self.sgpr[r] & MASK32
                hi = self.sgpr[r + 1] & MASK32
                return ((hi << 32) | lo) & MASK64
            if k == "special":
                t = operand.get("t", "")
                if t in ("exec", "exec_lo", "exec_hi"):
                    return self.exec_mask
                if t in ("vcc", "vcc_lo", "vcc_hi"):
                    return self.vcc
                if t == "scc":
                    return self.scc
        if isinstance(operand, str):
            op_lower = operand.lower()
            if op_lower in ("exec", "exec_lo", "exec_hi"):
                return self.exec_mask
            if op_lower in ("vcc", "vcc_lo", "vcc_hi"):
                return self.vcc
            if op_lower == "scc":
                return self.scc
            if op_lower.startswith("s[") and ":" in op_lower:
                parts = op_lower[2:-1].split(":")
                lo_idx = int(parts[0])
                hi_idx = int(parts[1])
                lo = self.sgpr[lo_idx] & MASK32
                hi = self.sgpr[hi_idx] & MASK32
                return ((hi << 32) | lo) & MASK64
            if op_lower.startswith("s"):
                r = int(op_lower[1:])
                lo = self.sgpr[r] & MASK32
                hi = self.sgpr[r + 1] & MASK32
                return ((hi << 32) | lo) & MASK64
        raise ValueError(f"Unsupported 64-bit source operand: {operand}")

    def _write_dst64(self, dst: dict | str | tuple[int, int], val: int, scc: int) -> None:
        """Write back 64-bit destination and update SCC."""
        val = val & MASK64
        if isinstance(dst, dict):
            k = dst.get("k")
            if k == "sgpr":
                r = dst["r"]
                self.sgpr[r] = val & MASK32
                self.sgpr[r + 1] = (val >> 32) & MASK32
            elif k == "special":
                t = dst.get("t", "")
                if t in ("exec", "exec_lo", "exec_hi"):
                    self.exec_mask = val
                elif t in ("vcc", "vcc_lo", "vcc_hi"):
                    self.vcc = val
        elif isinstance(dst, str):
            dst_lower = dst.lower()
            if dst_lower in ("exec", "exec_lo", "exec_hi"):
                self.exec_mask = val
            elif dst_lower in ("vcc", "vcc_lo", "vcc_hi"):
                self.vcc = val
            elif dst_lower.startswith("s[") and ":" in dst_lower:
                parts = dst_lower[2:-1].split(":")
                lo_idx = int(parts[0])
                hi_idx = int(parts[1])
                self.sgpr[lo_idx] = val & MASK32
                self.sgpr[hi_idx] = (val >> 32) & MASK32
            elif dst_lower.startswith("s"):
                r = int(dst_lower[1:])
                self.sgpr[r] = val & MASK32
                self.sgpr[r + 1] = (val >> 32) & MASK32
        elif isinstance(dst, tuple) and len(dst) == 2:
            self.sgpr[dst[0]] = val & MASK32
            self.sgpr[dst[1]] = (val >> 32) & MASK32
        else:
            raise ValueError(f"Unsupported 64-bit destination operand: {dst}")

        self.scc = scc

    def step(self, insn: dict) -> tuple[int, int]:
        """Execute one scalar instruction dictionary."""
        op = insn["op"]
        dst = insn["dst"]
        s0 = self._resolve_src64(insn["s0"])
        s1 = self._resolve_src64(insn["s1"])

        if op == "s_and_b64":
            res, scc = self.eval_s_and_b64(s0, s1)
        elif op == "s_or_b64":
            res, scc = self.eval_s_or_b64(s0, s1)
        elif op == "s_xor_b64":
            res, scc = self.eval_s_xor_b64(s0, s1)
        elif op == "s_andn2_b64":
            res, scc = self.eval_s_andn2_b64(s0, s1)
        elif op == "s_orn2_b64":
            res, scc = self.eval_s_orn2_b64(s0, s1)
        elif op == "s_nand_b64":
            res, scc = self.eval_s_nand_b64(s0, s1)
        elif op == "s_nor_b64":
            res, scc = self.eval_s_nor_b64(s0, s1)
        elif op == "s_xnor_b64":
            res, scc = self.eval_s_xnor_b64(s0, s1)
        elif op == "s_mov_b64":
            res, scc = s0, self.scc
        else:
            raise NotImplementedError(f"Unsupported scalar op: {op}")

        self._write_dst64(dst, res, scc)
        return res, self.scc

    def run_sequence(self, instructions: list[dict]) -> int:
        """Execute a list of instruction dictionaries sequentially."""
        last_res = 0
        for insn in instructions:
            last_res, _ = self.step(insn)
        return last_res


def spirv_lowering_model_logic(op: str, s0: int, s1: int) -> tuple[int, int]:
    """Mathematical emulation of the SPIR-V lowering emitted by tools/rdna2spirv.py.

    Emulates the mask lowering path in rdna2spirv.py:
      - Inverts operand y via OpNot if the mnemonic ends with n2_b64.
      - Applies OpBitwiseAnd, OpBitwiseOr, or OpBitwiseXor.
      - Sets SCC to 1 if result is non-zero, else 0 (OpGroupNonUniformAny vote).
    """
    s0 = s0 & MASK64
    s1 = s1 & MASK64

    y = (~s1 & MASK64) if op.endswith("n2_b64") else s1

    base = op[2:].rsplit("_", 1)[0]
    if base in ("and", "andn2"):
        r = (s0 & y) & MASK64
    elif base in ("or", "orn2"):
        r = (s0 | y) & MASK64
    elif base == "xor":
        r = (s0 ^ y) & MASK64
    else:
        raise ValueError(f"Unknown logic opcode: {op}")

    scc = 1 if r != 0 else 0
    return r, scc


class TestScalarLogicTruthTables(unittest.TestCase):
    """Verification of fundamental bitwise truth tables and 64-bit mask transformations."""

    def test_s_and_b64_truth_table(self) -> None:
        cases = [
            (0, 0, 0, 0),
            (0, 1, 0, 0),
            (1, 0, 0, 0),
            (1, 1, 1, 1),
            (0x0F0F0F0F_0F0F0F0F, 0x33333333_33333333, 0x03030303_03030303, 1),
            (0x55555555_55555555, 0xAAAAAAAA_AAAAAAAA, 0, 0),
            (MASK64, MASK64, MASK64, 1),
        ]
        for s0, s1, want_d, want_scc in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, scc = ScalarLogicInterpreter.eval_s_and_b64(s0, s1)
                self.assertEqual(d, want_d)
                self.assertEqual(scc, want_scc)

    def test_s_or_b64_truth_table(self) -> None:
        cases = [
            (0, 0, 0, 0),
            (0, 1, 1, 1),
            (1, 0, 1, 1),
            (1, 1, 1, 1),
            (0x0F0F0F0F_0F0F0F0F, 0x33333333_33333333, 0x3F3F3F3F_3F3F3F3F, 1),
            (0x55555555_55555555, 0xAAAAAAAA_AAAAAAAA, MASK64, 1),
            (0, MASK64, MASK64, 1),
        ]
        for s0, s1, want_d, want_scc in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, scc = ScalarLogicInterpreter.eval_s_or_b64(s0, s1)
                self.assertEqual(d, want_d)
                self.assertEqual(scc, want_scc)

    def test_s_xor_b64_truth_table(self) -> None:
        cases = [
            (0, 0, 0, 0),
            (0, 1, 1, 1),
            (1, 0, 1, 1),
            (1, 1, 0, 0),
            (0x0F0F0F0F_0F0F0F0F, 0x33333333_33333333, 0x3C3C3C3C_3C3C3C3C, 1),
            (0x55555555_55555555, 0xAAAAAAAA_AAAAAAAA, MASK64, 1),
            (MASK64, MASK64, 0, 0),
        ]
        for s0, s1, want_d, want_scc in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, scc = ScalarLogicInterpreter.eval_s_xor_b64(s0, s1)
                self.assertEqual(d, want_d)
                self.assertEqual(scc, want_scc)

    def test_s_andn2_b64_truth_table(self) -> None:
        # D = S0 & ~S1
        cases = [
            (0, 0, 0, 0),
            (0, 1, 0, 0),
            (1, 0, 1, 1),
            (1, 1, 0, 0),
            (0x0F0F0F0F_0F0F0F0F, 0x33333333_33333333, 0x0C0C0C0C_0C0C0C0C, 1),
            (MASK64, 0x55555555_55555555, 0xAAAAAAAA_AAAAAAAA, 1),
            (MASK64, MASK64, 0, 0),
            (0x12345678_9ABCDEF0, 0, 0x12345678_9ABCDEF0, 1),
        ]
        for s0, s1, want_d, want_scc in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, scc = ScalarLogicInterpreter.eval_s_andn2_b64(s0, s1)
                self.assertEqual(d, want_d)
                self.assertEqual(scc, want_scc)

    def test_s_orn2_b64_truth_table(self) -> None:
        # D = S0 | ~S1
        cases = [
            (0, 0, MASK64, 1),
            (0, 1, MASK64 - 1, 1),
            (1, 0, MASK64, 1),
            (1, 1, MASK64, 1),
            (0, MASK64, 0, 0),
            (0x0F0F0F0F_0F0F0F0F, 0x33333333_33333333, (0x0F0F0F0F_0F0F0F0F | (~0x33333333_33333333 & MASK64)), 1),
            (MASK64, 0, MASK64, 1),
            (MASK64, MASK64, MASK64, 1),
        ]
        for s0, s1, want_d, want_scc in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, scc = ScalarLogicInterpreter.eval_s_orn2_b64(s0, s1)
                self.assertEqual(d, want_d)
                self.assertEqual(scc, want_scc)

    def test_s_nand_b64_truth_table(self) -> None:
        # D = ~(S0 & S1)
        cases = [
            (0, 0, MASK64, 1),
            (0, 1, MASK64, 1),
            (1, 0, MASK64, 1),
            (1, 1, MASK64 - 1, 1),
            (MASK64, MASK64, 0, 0),  # All ones produces 0, clearing SCC!
            (0x55555555_55555555, 0xAAAAAAAA_AAAAAAAA, MASK64, 1),
            (0x0F0F0F0F_0F0F0F0F, 0x33333333_33333333, (~0x03030303_03030303) & MASK64, 1),
        ]
        for s0, s1, want_d, want_scc in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, scc = ScalarLogicInterpreter.eval_s_nand_b64(s0, s1)
                self.assertEqual(d, want_d)
                self.assertEqual(scc, want_scc)

    def test_s_nor_b64_truth_table(self) -> None:
        # D = ~(S0 | S1)
        cases = [
            (0, 0, MASK64, 1),
            (0, 1, MASK64 - 1, 1),
            (1, 0, MASK64 - 1, 1),
            (1, 1, MASK64 - 1, 1),
            (0, MASK64, 0, 0),      # S0=0, S1=~0 produces 0, clearing SCC!
            (MASK64, 0, 0, 0),      # S0=~0, S1=0 produces 0, clearing SCC!
            (MASK64, MASK64, 0, 0), # S0=~0, S1=~0 produces 0, clearing SCC!
            (0x55555555_55555555, 0xAAAAAAAA_AAAAAAAA, 0, 0),
            (0x0F0F0F0F_0F0F0F0F, 0x33333333_33333333, (~0x3F3F3F3F_3F3F3F3F) & MASK64, 1),
        ]
        for s0, s1, want_d, want_scc in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, scc = ScalarLogicInterpreter.eval_s_nor_b64(s0, s1)
                self.assertEqual(d, want_d)
                self.assertEqual(scc, want_scc)

    def test_s_xnor_b64_truth_table(self) -> None:
        # D = ~(S0 ^ S1)
        cases = [
            (0, 0, MASK64, 1),
            (0, 1, MASK64 - 1, 1),
            (1, 0, MASK64 - 1, 1),
            (1, 1, MASK64, 1),
            (MASK64, 0, 0, 0),      # ~(~0 ^ 0) = ~0xFFFFFFFFFFFFFFFF = 0!
            (0, MASK64, 0, 0),
            (MASK64, MASK64, MASK64, 1),
            (0x55555555_55555555, 0xAAAAAAAA_AAAAAAAA, 0, 0),
            (0x0F0F0F0F_0F0F0F0F, 0x33333333_33333333, (~0x3C3C3C3C_3C3C3C3C) & MASK64, 1),
        ]
        for s0, s1, want_d, want_scc in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, scc = ScalarLogicInterpreter.eval_s_xnor_b64(s0, s1)
                self.assertEqual(d, want_d)
                self.assertEqual(scc, want_scc)


class TestScalarLogicBoundaries(unittest.TestCase):
    """Boundary conditions covering 64-bit integer limits, MSB/LSB bits, and dword isolation."""

    def test_individual_bit_positions(self) -> None:
        for bit in range(64):
            val = 1 << bit
            with self.subTest(bit=bit):
                d, scc = ScalarLogicInterpreter.eval_s_and_b64(val, val)
                self.assertEqual(d, val)
                self.assertEqual(scc, 1)

                d, scc = ScalarLogicInterpreter.eval_s_xor_b64(val, val)
                self.assertEqual(d, 0)
                self.assertEqual(scc, 0)

                d, scc = ScalarLogicInterpreter.eval_s_andn2_b64(val, val)
                self.assertEqual(d, 0)
                self.assertEqual(scc, 0)

                d, scc = ScalarLogicInterpreter.eval_s_orn2_b64(val, val)
                self.assertEqual(d, MASK64)
                self.assertEqual(scc, 1)

                d, scc = ScalarLogicInterpreter.eval_s_nand_b64(val, val)
                self.assertEqual(d, (~val) & MASK64)
                self.assertEqual(scc, 1)

                d, scc = ScalarLogicInterpreter.eval_s_nor_b64(val, val)
                self.assertEqual(d, (~val) & MASK64)
                self.assertEqual(scc, 1)

                d, scc = ScalarLogicInterpreter.eval_s_xnor_b64(val, val)
                self.assertEqual(d, MASK64)
                self.assertEqual(scc, 1)

    def test_high_dword_isolation(self) -> None:
        hi_val = 0x80000000_00000000
        d, scc = ScalarLogicInterpreter.eval_s_and_b64(hi_val, MASK64)
        self.assertEqual(d, hi_val)
        self.assertEqual(scc, 1)

        d, scc = ScalarLogicInterpreter.eval_s_and_b64(hi_val, 0x00000000_FFFFFFFF)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 0)

        d, scc = ScalarLogicInterpreter.eval_s_or_b64(hi_val, 0x00000000_00000001)
        self.assertEqual(d, 0x80000000_00000001)
        self.assertEqual(scc, 1)

        d, scc = ScalarLogicInterpreter.eval_s_nor_b64(hi_val, 0xFFFFFFFF_00000000)
        self.assertEqual(d, 0x00000000_FFFFFFFF)
        self.assertEqual(scc, 1)

    def test_low_dword_isolation(self) -> None:
        lo_val = 0x00000000_80000000
        d, scc = ScalarLogicInterpreter.eval_s_and_b64(lo_val, 0xFFFFFFFF_00000000)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 0)

        d, scc = ScalarLogicInterpreter.eval_s_xor_b64(lo_val, 0x00000000_FFFFFFFF)
        self.assertEqual(d, 0x00000000_7FFFFFFF)
        self.assertEqual(scc, 1)

        d, scc = ScalarLogicInterpreter.eval_s_nand_b64(lo_val, 0x00000000_FFFFFFFF)
        self.assertEqual(d, 0xFFFFFFFF_7FFFFFFF)
        self.assertEqual(scc, 1)


class TestScalarConditionCode(unittest.TestCase):
    """Exhaustive testing of SCC state transitions and zero detection."""

    def setUp(self) -> None:
        self.interp = ScalarLogicInterpreter()

    def test_scc_set_on_nonzero_low_only(self) -> None:
        self.interp.scc = 0
        self.interp.step({"op": "s_and_b64", "dst": "s[0:1]", "s0": 1, "s1": 1})
        self.assertEqual(self.interp.scc, 1)

    def test_scc_set_on_nonzero_high_only(self) -> None:
        self.interp.scc = 0
        self.interp.step({"op": "s_and_b64", "dst": "s[0:1]", "s0": 1 << 63, "s1": 1 << 63})
        self.assertEqual(self.interp.scc, 1)

    def test_scc_cleared_on_zero_result(self) -> None:
        self.interp.scc = 1
        self.interp.step({"op": "s_and_b64", "dst": "s[0:1]", "s0": 0x1234, "s1": 0})
        self.assertEqual(self.interp.scc, 0)

    def test_scc_cleared_on_nand_all_ones(self) -> None:
        self.interp.scc = 1
        self.interp.step({"op": "s_nand_b64", "dst": "s[0:1]", "s0": MASK64, "s1": MASK64})
        self.assertEqual(self.interp.scc, 0)

    def test_scc_cleared_on_nor_all_ones(self) -> None:
        self.interp.scc = 1
        self.interp.step({"op": "s_nor_b64", "dst": "s[0:1]", "s0": MASK64, "s1": 0})
        self.assertEqual(self.interp.scc, 0)

    def test_scc_transition_chains(self) -> None:
        transitions = [
            ("s_and_b64", 0, 0, 0),
            ("s_or_b64", 0, 1, 1),
            ("s_xor_b64", 1, 1, 0),
            ("s_orn2_b64", 0, 0, 1),
            ("s_andn2_b64", 0xFF, 0xFF, 0),
            ("s_andn2_b64", 0xFF, 0x0F, 1),
            ("s_nand_b64", MASK64, MASK64, 0),
            ("s_nor_b64", 0, 0, 1),
            ("s_nor_b64", MASK64, MASK64, 0),
            ("s_xnor_b64", MASK64, 0, 0),
            ("s_xnor_b64", MASK64, MASK64, 1),
        ]
        for op, s0, s1, expected_scc in transitions:
            with self.subTest(op=op, s0=hex(s0), s1=hex(s1)):
                self.interp.step({"op": op, "dst": "s[0:1]", "s0": s0, "s1": s1})
                self.assertEqual(self.interp.scc, expected_scc)


class TestBooleanAlgebraIdentities(unittest.TestCase):
    """Algebraic invariant checks across random 64-bit inputs."""

    def test_commutativity(self) -> None:
        rng = random.Random(42)
        for _ in range(50):
            a = rng.getrandbits(64)
            b = rng.getrandbits(64)

            # s_and_b64(A, B) == s_and_b64(B, A)
            self.assertEqual(
                ScalarLogicInterpreter.eval_s_and_b64(a, b),
                ScalarLogicInterpreter.eval_s_and_b64(b, a),
            )
            # s_or_b64(A, B) == s_or_b64(B, A)
            self.assertEqual(
                ScalarLogicInterpreter.eval_s_or_b64(a, b),
                ScalarLogicInterpreter.eval_s_or_b64(b, a),
            )
            # s_xor_b64(A, B) == s_xor_b64(B, A)
            self.assertEqual(
                ScalarLogicInterpreter.eval_s_xor_b64(a, b),
                ScalarLogicInterpreter.eval_s_xor_b64(b, a),
            )
            # s_nand_b64(A, B) == s_nand_b64(B, A)
            self.assertEqual(
                ScalarLogicInterpreter.eval_s_nand_b64(a, b),
                ScalarLogicInterpreter.eval_s_nand_b64(b, a),
            )
            # s_nor_b64(A, B) == s_nor_b64(B, A)
            self.assertEqual(
                ScalarLogicInterpreter.eval_s_nor_b64(a, b),
                ScalarLogicInterpreter.eval_s_nor_b64(b, a),
            )
            # s_xnor_b64(A, B) == s_xnor_b64(B, A)
            self.assertEqual(
                ScalarLogicInterpreter.eval_s_xnor_b64(a, b),
                ScalarLogicInterpreter.eval_s_xnor_b64(b, a),
            )

    def test_asymmetry_of_not2_variants(self) -> None:
        a = 0xFF00FF00_AA55AA55
        b = 0x00FF00FF_55AA55AA
        # S0 & ~S1 != S1 & ~S0
        self.assertNotEqual(
            ScalarLogicInterpreter.eval_s_andn2_b64(a, b),
            ScalarLogicInterpreter.eval_s_andn2_b64(b, a),
        )
        # S0 | ~S1 != S1 | ~S0
        self.assertNotEqual(
            ScalarLogicInterpreter.eval_s_orn2_b64(a, b),
            ScalarLogicInterpreter.eval_s_orn2_b64(b, a),
        )

    def test_associativity(self) -> None:
        rng = random.Random(1337)
        for _ in range(30):
            a = rng.getrandbits(64)
            b = rng.getrandbits(64)
            c = rng.getrandbits(64)

            # (A & B) & C == A & (B & C)
            ab, _ = ScalarLogicInterpreter.eval_s_and_b64(a, b)
            bc, _ = ScalarLogicInterpreter.eval_s_and_b64(b, c)
            self.assertEqual(
                ScalarLogicInterpreter.eval_s_and_b64(ab, c)[0],
                ScalarLogicInterpreter.eval_s_and_b64(a, bc)[0],
            )

            # (A | B) | C == A | (B | C)
            ab, _ = ScalarLogicInterpreter.eval_s_or_b64(a, b)
            bc, _ = ScalarLogicInterpreter.eval_s_or_b64(b, c)
            self.assertEqual(
                ScalarLogicInterpreter.eval_s_or_b64(ab, c)[0],
                ScalarLogicInterpreter.eval_s_or_b64(a, bc)[0],
            )

            # (A ^ B) ^ C == A ^ (B ^ C)
            ab, _ = ScalarLogicInterpreter.eval_s_xor_b64(a, b)
            bc, _ = ScalarLogicInterpreter.eval_s_xor_b64(b, c)
            self.assertEqual(
                ScalarLogicInterpreter.eval_s_xor_b64(ab, c)[0],
                ScalarLogicInterpreter.eval_s_xor_b64(a, bc)[0],
            )

    def test_distributivity(self) -> None:
        rng = random.Random(999)
        for _ in range(30):
            a = rng.getrandbits(64)
            b = rng.getrandbits(64)
            c = rng.getrandbits(64)

            # A & (B | C) == (A & B) | (A & C)
            b_or_c, _ = ScalarLogicInterpreter.eval_s_or_b64(b, c)
            lhs, _ = ScalarLogicInterpreter.eval_s_and_b64(a, b_or_c)

            a_and_b, _ = ScalarLogicInterpreter.eval_s_and_b64(a, b)
            a_and_c, _ = ScalarLogicInterpreter.eval_s_and_b64(a, c)
            rhs, _ = ScalarLogicInterpreter.eval_s_or_b64(a_and_b, a_and_c)

            self.assertEqual(lhs, rhs)

            # A | (B & C) == (A | B) & (A | C)
            b_and_c, _ = ScalarLogicInterpreter.eval_s_and_b64(b, c)
            lhs2, _ = ScalarLogicInterpreter.eval_s_or_b64(a, b_and_c)

            a_or_b, _ = ScalarLogicInterpreter.eval_s_or_b64(a, b)
            a_or_c, _ = ScalarLogicInterpreter.eval_s_or_b64(a, c)
            rhs2, _ = ScalarLogicInterpreter.eval_s_and_b64(a_or_b, a_or_c)

            self.assertEqual(lhs2, rhs2)

    def test_de_morgans_laws(self) -> None:
        rng = random.Random(2026)
        for _ in range(30):
            a = rng.getrandbits(64)
            b = rng.getrandbits(64)

            # ~(A & B) == ~A | ~B
            # Directly test s_nand_b64 vs (~A | ~B)
            nand_res, _ = ScalarLogicInterpreter.eval_s_nand_b64(a, b)
            not_a, _ = ScalarLogicInterpreter.eval_s_xor_b64(a, MASK64)
            not_b, _ = ScalarLogicInterpreter.eval_s_xor_b64(b, MASK64)
            not_a_or_not_b, _ = ScalarLogicInterpreter.eval_s_or_b64(not_a, not_b)

            self.assertEqual(nand_res, not_a_or_not_b)

            # ~(A | B) == ~A & ~B
            # Directly test s_nor_b64 vs (~A & ~B)
            nor_res, _ = ScalarLogicInterpreter.eval_s_nor_b64(a, b)
            not_a_and_not_b, _ = ScalarLogicInterpreter.eval_s_and_b64(not_a, not_b)

            self.assertEqual(nor_res, not_a_and_not_b)

    def test_involution_and_self_cancellation(self) -> None:
        rng = random.Random(777)
        for _ in range(30):
            a = rng.getrandbits(64)
            b = rng.getrandbits(64)

            # (A ^ B) ^ B == A
            x, _ = ScalarLogicInterpreter.eval_s_xor_b64(a, b)
            res, _ = ScalarLogicInterpreter.eval_s_xor_b64(x, b)
            self.assertEqual(res, a)

            # A & ~A == 0
            zero, scc = ScalarLogicInterpreter.eval_s_andn2_b64(a, a)
            self.assertEqual(zero, 0)
            self.assertEqual(scc, 0)

            # A | ~A == ~0
            all_ones, scc = ScalarLogicInterpreter.eval_s_orn2_b64(a, a)
            self.assertEqual(all_ones, MASK64)
            self.assertEqual(scc, 1)

            # ~(A ^ ~A) == 0 (XNOR with complement is 0)
            not_a, _ = ScalarLogicInterpreter.eval_s_xor_b64(a, MASK64)
            xnor_zero, scc_xnor = ScalarLogicInterpreter.eval_s_xnor_b64(a, not_a)
            self.assertEqual(xnor_zero, 0)
            self.assertEqual(scc_xnor, 0)


class TestRegisterPairAliasingAndState(unittest.TestCase):
    """Register file packing, preservation, and overlapping source/destination semantics."""

    def setUp(self) -> None:
        self.interp = ScalarLogicInterpreter()

    def test_sgpr_packing_low_high(self) -> None:
        val = 0x12345678_9ABCDEF0
        self.interp.step({"op": "s_mov_b64", "dst": "s[4:5]", "s0": val, "s1": 0})
        self.assertEqual(self.interp.sgpr[4], 0x9ABCDEF0)
        self.assertEqual(self.interp.sgpr[5], 0x12345678)

    def test_unwritten_registers_preserved(self) -> None:
        for i in range(10):
            self.interp.sgpr[i] = (i + 1) * 0x11111111

        self.interp.step({
            "op": "s_or_b64",
            "dst": "s[4:5]",
            "s0": 0xFF00FF00_00FF00FF,
            "s1": 0x00FF00FF_FF00FF00,
        })

        for i in range(10):
            if i in (4, 5):
                self.assertEqual(self.interp.sgpr[i], 0xFFFFFFFF)
            else:
                self.assertEqual(self.interp.sgpr[i], (i + 1) * 0x11111111)

    def test_in_place_destination_aliasing(self) -> None:
        # D = D & S1
        self.interp.step({"op": "s_mov_b64", "dst": "s[2:3]", "s0": 0xFF00FF00_FF00FF00, "s1": 0})
        self.interp.step({"op": "s_and_b64", "dst": "s[2:3]", "s0": "s[2:3]", "s1": 0x0F0F0F0F_0F0F0F0F})
        res = ((self.interp.sgpr[3] << 32) | self.interp.sgpr[2]) & MASK64
        self.assertEqual(res, 0x0F000F00_0F000F00)

        # D = S0 ^ D
        self.interp.step({"op": "s_xor_b64", "dst": "s[2:3]", "s0": 0x0F000F00_0F000F00, "s1": "s[2:3]"})
        res = ((self.interp.sgpr[3] << 32) | self.interp.sgpr[2]) & MASK64
        self.assertEqual(res, 0)
        self.assertEqual(self.interp.scc, 0)


class TestWaveMaskUseCases(unittest.TestCase):
    """Simulates real RDNA2 shader wave control flow and mask operations."""

    def setUp(self) -> None:
        self.interp = ScalarLogicInterpreter()

    def test_wave64_branch_partition(self) -> None:
        exec_mask = 0xFFFFFFFFFFFFFFFF
        cond_vcc = 0xAAAAAAAAAAAAAAAA

        self.interp.exec_mask = exec_mask
        self.interp.vcc = cond_vcc

        # then_lanes = cond & exec
        self.interp.step({"op": "s_and_b64", "dst": "s[0:1]", "s0": "vcc", "s1": "exec"})
        then_lanes = ((self.interp.sgpr[1] << 32) | self.interp.sgpr[0]) & MASK64
        self.assertEqual(then_lanes, cond_vcc)
        self.assertEqual(self.interp.scc, 1)

        # else_lanes = exec & ~cond
        self.interp.step({"op": "s_andn2_b64", "dst": "s[2:3]", "s0": "exec", "s1": "vcc"})
        else_lanes = ((self.interp.sgpr[3] << 32) | self.interp.sgpr[2]) & MASK64
        self.assertEqual(else_lanes, 0x5555555555555555)
        self.assertEqual(self.interp.scc, 1)

        # Check disjointness: then_lanes & else_lanes == 0
        self.interp.step({"op": "s_and_b64", "dst": "s[4:5]", "s0": "s[0:1]", "s1": "s[2:3]"})
        overlap = ((self.interp.sgpr[5] << 32) | self.interp.sgpr[4]) & MASK64
        self.assertEqual(overlap, 0)
        self.assertEqual(self.interp.scc, 0)

        # Check completeness: then_lanes | else_lanes == exec
        self.interp.step({"op": "s_or_b64", "dst": "s[6:7]", "s0": "s[0:1]", "s1": "s[2:3]"})
        recombined = ((self.interp.sgpr[7] << 32) | self.interp.sgpr[6]) & MASK64
        self.assertEqual(recombined, exec_mask)
        self.assertEqual(self.interp.scc, 1)

    def test_wave32_exec_mask_operation(self) -> None:
        exec_wave32 = 0x00000000_FFFFFFFF
        cond = 0x00000000_0000FFFF
        self.interp.exec_mask = exec_wave32

        # Active lanes not matching cond
        self.interp.step({"op": "s_andn2_b64", "dst": "s[0:1]", "s0": "exec", "s1": cond})
        active_unmatched = ((self.interp.sgpr[1] << 32) | self.interp.sgpr[0]) & MASK64
        self.assertEqual(active_unmatched, 0x00000000_FFFF0000)
        self.assertEqual(self.interp.scc, 1)

    def test_vcc_accumulation_in_loop(self) -> None:
        self.interp.vcc = 0
        iterations = [
            0x00000000_00000001,
            0x00000000_00000010,
            0x00000001_00000000,
            0x00010000_00000000,
        ]
        accum = 0
        for step_mask in iterations:
            accum |= step_mask
            self.interp.step({"op": "s_or_b64", "dst": "vcc", "s0": "vcc", "s1": step_mask})
            self.assertEqual(self.interp.vcc, accum)
            self.assertEqual(self.interp.scc, 1)


class TestSpirvLoweringEquivalence(unittest.TestCase):
    """Equivalence between the ISA semantics and the SPIR-V lowering in tools/rdna2spirv.py."""

    def test_spirv_lowering_model_equivalence(self) -> None:
        rng = random.Random(888)
        ops = ["s_and_b64", "s_or_b64", "s_xor_b64", "s_andn2_b64", "s_orn2_b64"]

        for op in ops:
            for _ in range(25):
                s0 = rng.getrandbits(64)
                s1 = rng.getrandbits(64)

                isa_eval = getattr(ScalarLogicInterpreter, f"eval_{op}")
                isa_res, isa_scc = isa_eval(s0, s1)
                spv_res, spv_scc = spirv_lowering_model_logic(op, s0, s1)

                self.assertEqual(isa_res, spv_res)
                self.assertEqual(isa_scc, spv_scc)

    def test_rdna2spirv_gen_emission(self) -> None:
        """Verify direct Gen emission of 64-bit scalar logical instructions."""
        op_expectations = {
            "s_and_b64": ("OpBitwiseAnd", False),
            "s_or_b64": ("OpBitwiseOr", False),
            "s_xor_b64": ("OpBitwiseXor", False),
            "s_andn2_b64": ("OpBitwiseAnd", True),
            "s_orn2_b64": ("OpBitwiseOr", True),
        }

        for op, (expected_spv_op, expects_not) in op_expectations.items():
            with self.subTest(op=op):
                g = rdna2spirv.Gen("ps")
                ins = {
                    "op": op,
                    "args": [
                        {"k": "sgpr", "r": 4, "n": 2},
                        {"k": "sgpr", "r": 0, "n": 2},
                        {"k": "sgpr", "r": 2, "n": 2},
                    ],
                    "mods": [],
                }
                g.insn(ins)
                emitted = "\n".join(g.body)
                self.assertIn(expected_spv_op, emitted)
                if expects_not:
                    self.assertIn("OpNot", emitted)
                self.assertIn("OpStore %scc", emitted)

    def test_rdna2spirv_translation_output(self) -> None:
        """Translate synthetic pixel shader snippets and verify emitted SPIR-V instructions."""
        asm_template = """\
\tv_mov_b32 v0, 0
\t{op} s[4:5], s[0:1], s[2:3]
\tv_mov_b32 v1, s4
\texp mrt0 v0, v1, v0, v0 done vm
\ts_endpgm
"""
        op_expectations = {
            "s_and_b64": ("OpBitwiseAnd", False),
            "s_or_b64": ("OpBitwiseOr", False),
            "s_xor_b64": ("OpBitwiseXor", False),
            "s_andn2_b64": ("OpBitwiseAnd", True),
            "s_orn2_b64": ("OpBitwiseOr", True),
        }

        for op, (expected_spv_op, expects_not) in op_expectations.items():
            with self.subTest(op=op):
                code = asm_template.format(op=op)
                td = tempfile.TemporaryDirectory()
                self.addCleanup(td.cleanup)
                disasm_dir = os.path.join(td.name, "disasm")
                os.mkdir(disasm_dir)
                s_path = os.path.join(disasm_dir, "ps_logic.s")
                dxbc_path = os.path.join(td.name, "ps_logic.dxbc")

                with open(s_path, "w", encoding="utf-8") as f:
                    f.write(code)
                with open(dxbc_path, "wb") as f:
                    f.write(b"DXBC" + bytes(28))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    spv_asm = rdna2spirv.translate(s_path)
                    gc.collect()

                self.assertIn(expected_spv_op, spv_asm)
                if expects_not:
                    self.assertIn("OpNot", spv_asm)
                self.assertIn("OpStore %scc", spv_asm)


if __name__ == "__main__":
    unittest.main()
