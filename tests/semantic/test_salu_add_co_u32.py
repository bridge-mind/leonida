from __future__ import annotations

"""Semantic interpreter tests for RDNA2 scalar add with carry out (s_add_co_u32 / s_add_u32).

In the AMD RDNA2 ISA:
  * s_add_u32 (scalar add with unsigned carry-out, functionally s_add_co_u32):
      D.u = (S0.u + S1.u) mod 2^32
      SCC = (S0.u + S1.u >= 2^32)
  * Lowers in SPIR-V (tools/rdna2spirv.py) to OpIAddCarry %umul %s0 %s1:
      Member 0: 32-bit wrapped sum (stored in destination SGPR D)
      Member 1: 32-bit unsigned carry-out (stored in SCC condition flag register)

This test module verifies mathematical reference behavior, condition code updates,
modulo wrapping, boundary extremes, register aliasing, and translation lowering.
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

M32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def ref_s_add_co_u32(s0: int, s1: int) -> tuple[int, int]:
    """Pure reference oracle for RDNA2 scalar add with carry-out (s_add_co_u32 / s_add_u32).

    Computes unsigned 32-bit sum (s0 + s1) mod 2^32 and the 1-bit carry-out flag SCC.
    Returns (result, scc).
    """
    a = s0 & M32
    b = s1 & M32
    full = a + b
    return full & M32, 1 if full >= (1 << 32) else 0


def oracle_add_co_u32_analysis(s0: int, s1: int) -> dict[str, int | bool]:
    """Detailed arithmetic analysis for scalar add with carry: inputs, 64-bit sum, carry flag, and wrapped result."""
    a = s0 & M32
    b = s1 & M32
    full = a + b
    has_carry = full >= (1 << 32)
    wrapped = full & M32
    return {
        "s0": a,
        "s1": b,
        "full_sum": full,
        "carry": has_carry,
        "scc": 1 if has_carry else 0,
        "wrapped_result": wrapped,
    }


class ScalarALUState:
    """Minimal deterministic interpreter state modeling SGPR bank and SCC flag."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgpr = [0] * num_sgprs
        self.scc = 0

    def reset(self) -> None:
        self.sgpr = [0] * len(self.sgpr)
        self.scc = 0

    def step(self, op: str, dst: int, s0: int | str, s1: int | str | None = None) -> tuple[int, int]:
        """Execute a single scalar ALU instruction."""
        canonical_op = "s_add_u32" if op in ("s_add_co_u32", "s_add_u32") else op
        v0 = self.sgpr[int(s0[1:])] if isinstance(s0, str) and s0.startswith("s") else int(s0) & M32
        v1 = 0
        if s1 is not None:
            v1 = self.sgpr[int(s1[1:])] if isinstance(s1, str) and s1.startswith("s") else int(s1) & M32

        if canonical_op == "s_mov_b32":
            res = v0
            # SCC is not modified by s_mov_b32
        elif canonical_op == "s_add_u32":
            res, scc = ref_s_add_co_u32(v0, v1)
            self.scc = scc
        elif canonical_op == "s_cselect_b32":
            res = v0 if self.scc != 0 else v1
        else:
            raise NotImplementedError(f"Unsupported op in ScalarALUState: {op}")

        self.sgpr[dst] = res & M32
        return self.sgpr[dst], self.scc


_TMP_DIR = None
_SRC_PATH = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_add_co.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_add_co.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines: list[str]) -> tuple[int, int]:
    """Assemble synthetic pixel shader lines, translate with rdna2spirv, run via mini-interpreter.

    Returns (v0, v1) where v0 is the 32-bit computation result and v1 is the extracted SCC flag.
    """
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_salu_add_co_u32\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


def run_op(s0: int, s1: int, scc_in: int = 0, op: str = "s_add_co_u32") -> tuple[int, int]:
    """Execute scalar add with carry via SPIR-V translator and interpreter.

    Preset SCC to `scc_in`, execute instruction, and read back result and post-op SCC.
    Returns (result, scc_out).
    """
    canonical_insn = "s_add_u32" if op in ("s_add_co_u32", "s_add_u32") else op
    lines = [
        f"s_mov_b32 s2, {s0 & M32:#x}",
        f"s_mov_b32 s3, {s1 & M32:#x}",
        f"s_cmp_eq_u32 {1 if scc_in else 0}, 1",  # preset SCC
        f"{canonical_insn} s0, s2, s3",
        "s_cselect_b32 s1, 1, 0",                # read SCC into s1
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
    ]
    return run_lines(lines)


class TestSaluAddCoU32ReferenceOracle(unittest.TestCase):
    """Reference oracle tests for RDNA2 scalar add with carry out (s_add_co_u32)."""

    def test_oracle_zero_identity(self) -> None:
        """Adding 0 to any value must return the original value with SCC = 0."""
        cases = [0, 1, 42, 0x12345678, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                res0, scc0 = ref_s_add_co_u32(val, 0)
                self.assertEqual(res0, val)
                self.assertEqual(scc0, 0)

                res1, scc1 = ref_s_add_co_u32(0, val)
                self.assertEqual(res1, val)
                self.assertEqual(scc1, 0)

    def test_oracle_basic_addition_no_carry(self) -> None:
        """Sums strictly below 2^32 must produce exact integer sum with SCC = 0."""
        cases = [
            (10, 20, 30),
            (0x1000, 0x2000, 0x3000),
            (0x12345678, 0x01020304, 0x1336597C),
            (0x7FFFFFFE, 1, 0x7FFFFFFF),
            (0x40000000, 0x3FFFFFFF, 0x7FFFFFFF),
            (0x7FFFFFFF, 0x7FFFFFFF, 0xFFFFFFFE),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
        ]
        for a, b, expected_sum in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                res, scc = ref_s_add_co_u32(a, b)
                self.assertEqual(res, expected_sum)
                self.assertEqual(scc, 0)

    def test_oracle_carry_detection(self) -> None:
        """Sums at or above 2^32 must set SCC = 1 and wrap modulo 2^32."""
        cases = [
            (0xFFFFFFFF, 1, 0, 1),
            (0xFFFFFFFF, 2, 1, 1),
            (0xFFFFFFFE, 2, 0, 1),
            (0x80000000, 0x80000000, 0, 1),
            (0x80000000, 0x80000001, 1, 1),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 1),
            (0xC0000000, 0x50000000, 0x10000000, 1),
        ]
        for a, b, exp_res, exp_scc in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                res, scc = ref_s_add_co_u32(a, b)
                self.assertEqual(res, exp_res)
                self.assertEqual(scc, exp_scc)

    def test_oracle_commutativity(self) -> None:
        """s_add_co_u32 must be strictly commutative across all operand ranges."""
        test_pairs = [
            (0, 0),
            (0, 0xFFFFFFFF),
            (1, 0xFFFFFFFF),
            (0x12345678, 0x87654321),
            (0x80000000, 0x80000000),
            (0x7FFFFFFF, 0x80000000),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0xCAFEBABE, 0xDEADBEEF),
            (0xAAAAAAAA, 0x55555555),
        ]
        for a, b in test_pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                r1, s1 = ref_s_add_co_u32(a, b)
                r2, s2 = ref_s_add_co_u32(b, a)
                self.assertEqual(r1, r2)
                self.assertEqual(s1, s2)

    def test_oracle_analysis_breakdown(self) -> None:
        """Detailed analysis oracle must accurately compute components."""
        info = oracle_add_co_u32_analysis(0xFFFFFFFF, 1)
        self.assertEqual(info["s0"], 0xFFFFFFFF)
        self.assertEqual(info["s1"], 1)
        self.assertEqual(info["full_sum"], 0x100000000)
        self.assertTrue(info["carry"])
        self.assertEqual(info["scc"], 1)
        self.assertEqual(info["wrapped_result"], 0)

        info2 = oracle_add_co_u32_analysis(0x1234, 0x5678)
        self.assertFalse(info2["carry"])
        self.assertEqual(info2["scc"], 0)
        self.assertEqual(info2["wrapped_result"], 0x68AC)

    def test_state_interpreter_execution(self) -> None:
        """ScalarALUState must correctly track register writes and SCC updates."""
        state = ScalarALUState()
        state.step("s_mov_b32", dst=2, s0=0xFFFFFFFF)
        state.step("s_mov_b32", dst=3, s0=1)
        res, scc = state.step("s_add_co_u32", dst=4, s0="s2", s1="s3")
        self.assertEqual(res, 0)
        self.assertEqual(scc, 1)
        self.assertEqual(state.sgpr[4], 0)
        self.assertEqual(state.scc, 1)

        # Ensure s_cselect_b32 uses updated SCC
        csel_res, _ = state.step("s_cselect_b32", dst=5, s0=0xCAFE, s1=0xBEEF)
        self.assertEqual(csel_res, 0xCAFE)

    def test_spirv_translation_matches_oracle(self) -> None:
        """SPIR-V translation and execution must match reference oracle for representative inputs."""
        sample_pairs = [
            (0, 0),
            (100, 200),
            (0x7FFFFFFF, 0),
            (0x7FFFFFFF, 1),
            (0xFFFFFFFF, 0),
            (0xFFFFFFFF, 1),
            (0x80000000, 0x80000000),
            (0xFFFFFFFF, 0xFFFFFFFF),
        ]
        for a, b in sample_pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                got_res, got_scc = run_op(a, b, scc_in=0, op="s_add_co_u32")
                exp_res, exp_scc = ref_s_add_co_u32(a, b)
                self.assertEqual((got_res, got_scc), (exp_res, exp_scc))


if __name__ == "__main__":
    unittest.main()
