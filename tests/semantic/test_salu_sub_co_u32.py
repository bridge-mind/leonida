from __future__ import annotations

"""Semantic interpreter tests for RDNA2 scalar subtract with carry out (s_sub_co_u32 / s_sub_u32).

Covers:
  * Reference mathematical oracle for 32-bit unsigned scalar subtraction with borrow:
      D = (S0 - S1) mod 2^32
      SCC = 1 if (S0 < S1) else 0 (unsigned borrow-out / carry-out)
  * Deterministic semantic interpreter for SALU operations tracking SGPRs and SCC state.
  * Cross-validation of shader assembly execution via rdna2spirv lowering and spirv_mini_interp.
  * Multi-precision 64-bit subtraction borrow chain oracle (s_sub_u32 + s_subb_u32).
"""

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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_sub_co.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_sub_co.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def ref_sub_co_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference implementation of scalar subtract with carry-out (borrow).

    Returns (difference, borrow_out) where:
      difference = (s0 - s1) & 0xFFFFFFFF
      borrow_out = 1 if (s0 & 0xFFFFFFFF) < (s1 & 0xFFFFFFFF) else 0
    """
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    diff = (u0 - u1) & MASK32
    borrow = 1 if u0 < u1 else 0
    return diff, borrow


def ref_subb_co_u32(s0: int, s1: int, borrow_in: int) -> tuple[int, int]:
    """Reference implementation of scalar subtract with borrow-in and borrow-out."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    b_in = 1 if borrow_in else 0
    diff = (u0 - u1 - b_in) & MASK32
    borrow = 1 if u0 < (u1 + b_in) else 0
    return diff, borrow


def ref_sub64(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> tuple[int, int, int]:
    """Reference 64-bit subtraction decomposed into 32-bit halves with borrow."""
    a64 = ((a_hi & MASK32) << 32) | (a_lo & MASK32)
    b64 = ((b_hi & MASK32) << 32) | (b_lo & MASK32)
    diff64 = (a64 - b64) & MASK64
    borrow64 = 1 if a64 < b64 else 0
    return diff64 & MASK32, (diff64 >> 32) & MASK32, borrow64


class ScalarALUInterpreter:
    """Deterministic semantic interpreter for RDNA2 scalar ALU operations."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgpr = [0] * num_sgprs
        self.scc = 0

    def reset(self) -> None:
        self.sgpr = [0] * len(self.sgpr)
        self.scc = 0

    @staticmethod
    def eval_sub_co_u32(s0: int, s1: int) -> tuple[int, int]:
        return ref_sub_co_u32(s0, s1)

    @staticmethod
    def eval_subb_co_u32(s0: int, s1: int, borrow_in: int) -> tuple[int, int]:
        return ref_subb_co_u32(s0, s1, borrow_in)

    def _resolve_val(self, operand: int | str | dict) -> int:
        if isinstance(operand, int):
            return operand & MASK32
        if isinstance(operand, dict):
            if operand.get("k") == "imm":
                return operand["v"] & MASK32
            if operand.get("k") == "sgpr":
                return self.sgpr[operand["r"]] & MASK32
        if isinstance(operand, str) and operand.startswith("s"):
            return self.sgpr[int(operand[1:])] & MASK32
        raise ValueError(f"Unknown operand: {operand}")

    def _resolve_dst(self, operand: int | str | dict) -> int | None:
        if isinstance(operand, dict) and operand.get("k") == "sgpr":
            return operand["r"]
        if isinstance(operand, str) and operand.startswith("s"):
            return int(operand[1:])
        if isinstance(operand, int):
            return operand
        return None

    def step(self, insn: dict) -> tuple[int, int]:
        op = insn["op"]
        dst_idx = self._resolve_dst(insn["dst"])
        s0 = self._resolve_val(insn["s0"])
        s1 = self._resolve_val(insn.get("s1", 0))

        if op in ("s_sub_u32", "s_sub_co_u32"):
            res, scc = self.eval_sub_co_u32(s0, s1)
            self.scc = scc
        elif op == "s_subb_u32":
            res, scc = self.eval_subb_co_u32(s0, s1, self.scc)
            self.scc = scc
        elif op == "s_mov_b32":
            res = s0 & MASK32
        elif op == "s_cselect_b32":
            res = s0 & MASK32 if self.scc else s1 & MASK32
        else:
            raise NotImplementedError(f"Unsupported op: {op}")

        if dst_idx is not None:
            self.sgpr[dst_idx] = res
        return res, self.scc

    def run_sequence(self, instructions: list[dict]) -> int:
        last_res = 0
        for insn in instructions:
            last_res, _ = self.step(insn)
        return last_res


def run_assembly_lines(lines: list[str]) -> tuple[int, int]:
    """Assemble a shader snippet, translate via rdna2spirv, and execute in spirv_mini_interp."""
    if _TMP_DIR is None or _SRC_PATH is None:
        setUpModule()
    asm = "; test_salu_sub_co_u32\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    words = si.run_fragment(spv)["%out_mrt0"]
    return int(words[0]), int(words[1])


def run_salu_sub(s0: int, s1: int) -> tuple[int, int]:
    """Execute s_sub_u32 in a pixel shader and return (result, SCC)."""
    lines = [
        f"s_mov_b32 s2, {s0:#x}",
        f"s_mov_b32 s3, {s1:#x}",
        "s_sub_u32 s0, s2, s3",
        "s_cselect_b32 s1, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
    ]
    return run_assembly_lines(lines)


class TestScalarSubCoU32ReferenceOracle(unittest.TestCase):
    """Validation of reference mathematical oracle and interpreter semantics."""

    def setUp(self) -> None:
        self.interp = ScalarALUInterpreter()

    def test_zero_identity(self) -> None:
        for val in (0, 1, 42, 0x12345678, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):
            with self.subTest(val=hex(val)):
                diff, borrow = ref_sub_co_u32(val, 0)
                self.assertEqual(diff, val)
                self.assertEqual(borrow, 0)

    def test_identical_operands_yield_zero_no_borrow(self) -> None:
        for val in (0, 1, 0x1000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):
            with self.subTest(val=hex(val)):
                diff, borrow = ref_sub_co_u32(val, val)
                self.assertEqual(diff, 0)
                self.assertEqual(borrow, 0)

    def test_basic_subtraction_no_borrow(self) -> None:
        pairs = [
            (10, 3, 7, 0),
            (100, 100, 0, 0),
            (0xFFFFFFFF, 1, 0xFFFFFFFE, 0),
            (0x80000000, 0x7FFFFFFF, 1, 0),
            (0x10000000, 0x0FFFFFFF, 1, 0),
        ]
        for s0, s1, exp_diff, exp_borrow in pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                diff, borrow = ref_sub_co_u32(s0, s1)
                self.assertEqual(diff, exp_diff)
                self.assertEqual(borrow, exp_borrow)

    def test_borrow_boundary_detection(self) -> None:
        cases = [
            (0, 1, 0xFFFFFFFF, 1),
            (0, 0xFFFFFFFF, 1, 1),
            (5, 10, 0xFFFFFFFB, 1),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 1),
            (0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 1),
        ]
        for s0, s1, exp_diff, exp_borrow in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                diff, borrow = ref_sub_co_u32(s0, s1)
                self.assertEqual(diff, exp_diff)
                self.assertEqual(borrow, exp_borrow)

    def test_interpreter_step_and_state(self) -> None:
        self.interp.step({"op": "s_mov_b32", "dst": "s0", "s0": 100})
        self.interp.step({"op": "s_mov_b32", "dst": "s1", "s0": 40})
        res, scc = self.interp.step({"op": "s_sub_co_u32", "dst": "s2", "s0": "s0", "s1": "s1"})
        self.assertEqual(res, 60)
        self.assertEqual(scc, 0)
        self.assertEqual(self.interp.sgpr[2], 60)
        self.assertEqual(self.interp.scc, 0)

        # Underflow step
        res, scc = self.interp.step({"op": "s_sub_co_u32", "dst": "s3", "s0": 10, "s1": 25})
        self.assertEqual(res, (10 - 25) & MASK32)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.sgpr[3], (10 - 25) & MASK32)
        self.assertEqual(self.interp.scc, 1)

    def test_spirv_lowering_matches_reference_oracle(self) -> None:
        cases = [
            (0, 0),
            (1, 0),
            (0, 1),
            (100, 35),
            (35, 100),
            (0xFFFFFFFF, 0),
            (0, 0xFFFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x80000000, 1),
            (0x7FFFFFFF, 0x80000000),
        ]
        for s0, s1 in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                expected_diff, expected_scc = ref_sub_co_u32(s0, s1)
                got_diff, got_scc = run_salu_sub(s0, s1)
                self.assertEqual((got_diff, got_scc), (expected_diff, expected_scc))

    def test_spirv_lowering_random_sweep(self) -> None:
        rng = random.Random(0x50B0)
        for _ in range(25):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            expected = ref_sub_co_u32(s0, s1)
            got = run_salu_sub(s0, s1)
            self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
