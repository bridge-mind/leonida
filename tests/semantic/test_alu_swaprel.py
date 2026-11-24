"""Reference oracle and value tests for v_swaprel_b32 (GFX10 VOP1 opcode 104).

RDNA2 instruction semantics:
  D_off = (M0 >> 16) & 0x3FF
  S0_off = M0 & 0x3FF
  tmp = VGPR[D + D_off]
  VGPR[D + D_off] = VGPR[S0 + S0_off]
  VGPR[S0 + S0_off] = tmp

Both registers are read before either is written. When both operands address
the same physical VGPR, its value is preserved. M0 bits [15:10] and [31:26]
are ignored.

Standard library only; zero game data, zero external dependencies.
"""
from __future__ import annotations

import os
import struct
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from support import STUB_DXBC, translate_lines  # noqa: E402

MASK10 = 0x3FF
MASK32 = 0xFFFFFFFF
NUM_VGPRS = 256


def sentinel(n: int) -> int:
    """Deterministic normal f32 bit pattern per VGPR index."""
    return 0x40000000 + (n & 0x00FFFFFF)


class SwaprelOracle:
    """Architectural reference oracle for RDNA2 v_swaprel_b32."""

    def __init__(self, size: int = NUM_VGPRS, init_map: dict[int, int] | None = None) -> None:
        self.size = size
        self.regs = [sentinel(i) for i in range(size)]
        if init_map:
            for r, v in init_map.items():
                if 0 <= r < size:
                    self.regs[r] = v & MASK32

    @staticmethod
    def decode_offsets(m0: int) -> tuple[int, int]:
        """Decode relative offsets: (D_offset, S0_offset)."""
        d_off = (m0 >> 16) & MASK10
        s0_off = m0 & MASK10
        return d_off, s0_off

    def step(self, d: int, s0: int, m0: int) -> tuple[int, int]:
        """Execute v_swaprel_b32 in-place; return effective (d_eff, s0_eff)."""
        d_off, s0_off = self.decode_offsets(m0)
        d_eff = d + d_off
        s0_eff = s0 + s0_off
        if 0 <= d_eff < self.size and 0 <= s0_eff < self.size:
            val_d = self.regs[d_eff]
            val_s0 = self.regs[s0_eff]
            self.regs[d_eff] = val_s0
            self.regs[s0_eff] = val_d
        return d_eff, s0_eff

    def export(self, *indices: int) -> list[int]:
        """Read export registers."""
        return [self.regs[i] for i in indices]


def run_ps_swaprel(d: int, s0: int, m0: int, export: tuple[int, int, int, int], extra: tuple[str, ...] = ()) -> list[int]:
    """Assemble, translate, and execute a fragment shader with v_swaprel_b32."""
    max_reg = max(max(export), d, s0, 30) + 1
    lines = [f"v_mov_b32 v{n}, {sentinel(n):#x}" for n in range(max_reg)]
    lines.append(f"s_mov_b32 m0, {m0:#x}")
    lines.extend(extra)
    lines.append(f"v_swaprel_b32 v{d}, v{s0}")
    lines.append(f"exp mrt0 {', '.join(f'v{r}' for r in export)} done vm")
    lines.append("s_endpgm")

    spv = translate_lines("ps_swaprel", lines)
    out = si.run_fragment(spv)
    return list(out["%out_mrt0"])


class TestSwaprelOracleInvariants(unittest.TestCase):
    """Mathematical and architectural invariant tests for SwaprelOracle."""

    def test_oracle_decode_offsets(self) -> None:
        self.assertEqual(SwaprelOracle.decode_offsets(0), (0, 0))
        self.assertEqual(SwaprelOracle.decode_offsets(0x0005_0002), (5, 2))
        self.assertEqual(SwaprelOracle.decode_offsets(0x03FF_03FF), (1023, 1023))
        # Verify ignored bits [15:10] and [31:26]
        self.assertEqual(SwaprelOracle.decode_offsets(0xFC05_FC02), (5, 2))

    def test_oracle_self_swap_identity(self) -> None:
        oracle = SwaprelOracle()
        orig = list(oracle.regs)
        # Target same register: d=4 + 3 = 7, s0=6 + 1 = 7
        d_eff, s0_eff = oracle.step(4, 6, 0x0003_0001)
        self.assertEqual(d_eff, 7)
        self.assertEqual(s0_eff, 7)
        self.assertEqual(oracle.regs, orig)

    def test_oracle_involution_double_swap(self) -> None:
        oracle = SwaprelOracle()
        orig = list(oracle.regs)
        oracle.step(2, 8, 0x0004_0003)
        self.assertNotEqual(oracle.regs, orig)
        oracle.step(2, 8, 0x0004_0003)
        self.assertEqual(oracle.regs, orig)

    def test_oracle_unaffected_registers_isolated(self) -> None:
        oracle = SwaprelOracle()
        val3 = oracle.regs[3]
        val11 = oracle.regs[11]
        oracle.step(1, 5, 0x0002_0006)  # swaps 1+2=3 and 5+6=11
        self.assertEqual(oracle.regs[3], val11)
        self.assertEqual(oracle.regs[11], val3)
        # All other registers unchanged
        for r in range(NUM_VGPRS):
            if r not in (3, 11):
                self.assertEqual(oracle.regs[r], sentinel(r))

    def test_oracle_commutativity(self) -> None:
        o1 = SwaprelOracle()
        o2 = SwaprelOracle()
        # Swapping (A, B) vs (B, A) with swapped M0 offsets
        o1.step(2, 9, 0x0003_0007)  # 2+3=5, 9+7=16
        o2.step(9, 2, 0x0007_0003)  # 9+7=16, 2+3=5
        self.assertEqual(o1.regs, o2.regs)


class TestSwaprelInterpreterBasic(unittest.TestCase):
    """Interpreter value tests for v_swaprel_b32 matching the reference oracle."""

    def _verify_case(self, d: int, s0: int, m0: int) -> None:
        oracle = SwaprelOracle()
        d_eff, s0_eff = oracle.step(d, s0, m0)
        export = (d_eff, s0_eff, d_eff + 1, s0_eff + 1 if s0_eff + 1 != d_eff else s0_eff - 1)
        expected = oracle.export(*export)
        actual = run_ps_swaprel(d, s0, m0, export)
        self.assertEqual(actual, expected)

    def test_zero_offset_direct_swap(self) -> None:
        self._verify_case(1, 2, 0)
        self._verify_case(5, 12, 0)

    def test_distinct_offsets(self) -> None:
        for d, s0, m0 in (
            (3, 10, 0x0005_0002),
            (20, 1, 0x0002_0007),
            (5, 5, 0x0001_0000),
            (2, 8, 0x000A_0004),
        ):
            with self.subTest(d=d, s0=s0, m0=hex(m0)):
                self._verify_case(d, s0, m0)

    def test_ignored_m0_bits(self) -> None:
        # Bits 15:10 and 31:26 set: must produce exact same result as clean M0
        self._verify_case(2, 6, 0xFC03_FC04)
        self._verify_case(4, 9, 0xA802_5405)

    def test_self_swap(self) -> None:
        # Both addresses point to same VGPR (d=4+3=7, s0=6+1=7)
        self._verify_case(4, 6, 0x0003_0001)




class TestSwaprelChainedSequencesAndPreservation(unittest.TestCase):
    """Multi-step chained swap permutations and register state preservation."""

    def test_involution_two_swaps_restore_identity(self) -> None:
        oracle = SwaprelOracle()
        orig = list(oracle.regs)
        # Swap 3 and 7 twice
        oracle.step(1, 2, 0x0002_0005)
        self.assertNotEqual(oracle.regs, orig)
        oracle.step(1, 2, 0x0002_0005)
        self.assertEqual(oracle.regs, orig)

    def test_three_way_cyclic_permutation(self) -> None:
        # A=v2, B=v5, C=v8
        oracle = SwaprelOracle()
        vA, vB, vC = oracle.regs[2], oracle.regs[5], oracle.regs[8]
        # Step 1: swap A and B -> A has vB, B has vA
        oracle.step(2, 5, 0)
        # Step 2: swap B and C -> B has vC, C has vA
        oracle.step(5, 8, 0)
        self.assertEqual(oracle.regs[2], vB)
        self.assertEqual(oracle.regs[5], vC)
        self.assertEqual(oracle.regs[8], vA)

    def test_disjoint_parallel_swaps(self) -> None:
        oracle = SwaprelOracle()
        oracle.step(1, 2, 0)
        oracle.step(3, 4, 0)
        oracle.step(5, 6, 0)
        self.assertEqual(oracle.regs[1], sentinel(2))
        self.assertEqual(oracle.regs[2], sentinel(1))
        self.assertEqual(oracle.regs[3], sentinel(4))
        self.assertEqual(oracle.regs[4], sentinel(3))
        self.assertEqual(oracle.regs[5], sentinel(6))
        self.assertEqual(oracle.regs[6], sentinel(5))




class TestSwaprelNegativeAndEdgeControls(unittest.TestCase):
    """Negative controls, syntax rejection, and invalid operand validation."""

    def test_reject_scalar_destination_operand(self) -> None:
        lines = [
            "s_mov_b32 m0, 0",
            "v_swaprel_b32 s0, v1",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines("ps_bad_dest", lines)

    def test_reject_scalar_source_operand(self) -> None:
        lines = [
            "s_mov_b32 m0, 0",
            "v_swaprel_b32 v1, s0",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines("ps_bad_src", lines)

    def test_reject_literal_operand(self) -> None:
        lines = [
            "s_mov_b32 m0, 0",
            "v_swaprel_b32 v1, 1.0",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines("ps_bad_lit", lines)


if __name__ == "__main__":
    unittest.main()
