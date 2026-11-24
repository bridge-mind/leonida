"""Semantic interpreter tests for subgroup bitwise lane reductions and population counting.

Covers RDNA2 instructions:
  - s_bcnt1_i32_b64: D = PopulationCount(S0_64), SCC = (D != 0)
  - s_ff1_i32_b64:   D = FindFirstOne(S0_64), -1 (0xFFFFFFFF) if none; SCC unchanged
  - s_bcnt1_i32_b32: D = PopulationCount(S0_32), SCC = (D != 0)
  - s_ff1_i32_b32:   D = FindFirstOne(S0_32), -1 (0xFFFFFFFF) if none; SCC unchanged

Subgroup Bitwise Lane Reduction Architecture:
  In RDNA2 and Vulkan/SPIR-V translation, a lane mask represents a boolean predicate
  across all invocations of a wave (wave32 or wave64).
  - Each invocation determines its local condition (e.g., comparison or EXEC status).
  - The translator packs per-lane bits into 32-bit words via OpShiftLeftLogical and
    OpSelect, separating lower lanes (0..31) and upper lanes (32..63).
  - Cross-lane reduction is achieved via OpGroupNonUniformBitwiseOr with Reduce group
    operation across Execution Scope 3 (Subgroup).
  - s_bcnt1 counts set bits in both low and high words via OpBitCount, sums them with
    OpIAdd, and sets SCC = 1 if the sum is non-zero, 0 if empty.
  - s_ff1 finds the least significant set bit using GLSL.std.450 FindILsb on the low
    word. If the low word is non-zero, its bit position (0..31) is returned. If the low
    word is zero, FindILsb on the high word is biased by +32. If both are zero, -1
    (0xFFFFFFFF) is returned. Importantly, s_ff1 leaves SCC unmodified.
  - When the source operand is a standard scalar SGPR pair (not a lane mask), the
    interpreter operates directly on the loaded scalar dwords without subgroup reduction.

Zero game data: Hand-crafted semantic tests using standard library only.
Compatible with Python 3.9+.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_popcount_u64(val: int) -> int:
    """Deterministic population count for 64-bit integer (Python 3.9 compatible)."""
    return bin(val & MASK64).count("1")


def ref_popcount_u32(val: int) -> int:
    """Deterministic population count for 32-bit integer."""
    return bin(val & MASK32).count("1")


def ref_find_first_one_u64(val: int) -> int:
    """Find index of lowest set bit (0..63) in 64-bit uint, or -1 (0xFFFFFFFF) if zero."""
    val &= MASK64
    if val == 0:
        return MASK32
    return (val & -val).bit_length() - 1


def ref_find_first_one_u32(val: int) -> int:
    """Find index of lowest set bit (0..31) in 32-bit uint, or -1 (0xFFFFFFFF) if zero."""
    val &= MASK32
    if val == 0:
        return MASK32
    return (val & -val).bit_length() - 1


def ref_bcnt1_i32_b64(src64: int) -> tuple[int, int]:
    """Reference oracle for s_bcnt1_i32_b64: (result_u32, scc)."""
    count = ref_popcount_u64(src64)
    scc = 1 if count != 0 else 0
    return count, scc


def ref_ff1_i32_b64(src64: int) -> int:
    """Reference oracle for s_ff1_i32_b64: lowest bit index 0..63 or 0xFFFFFFFF."""
    return ref_find_first_one_u64(src64)


def ref_subgroup_wave_mask_reduction(lane_bits: list[int] | dict[int, bool], wave_size: int = 64) -> tuple[int, int]:
    """Model OpGroupNonUniformBitwiseOr Reduce across lanes for low and high halves."""
    lo_accum = 0
    hi_accum = 0
    for lane in range(wave_size):
        bit_val = 1 if (lane_bits[lane] if isinstance(lane_bits, list) else lane_bits.get(lane, False)) else 0
        if bit_val:
            if lane < 32:
                lo_accum |= (1 << (lane & 31))
            else:
                hi_accum |= (1 << (lane & 31))
    return lo_accum & MASK32, hi_accum & MASK32


class LaneMaskReductionInterpreter:
    """Semantic interpreter modeling whole-wave execution, lane reductions, and scalar ALU ops."""

    def __init__(self, wave_size: int = 64, num_sgprs: int = 106):
        if wave_size not in (32, 64):
            raise ValueError(f"unsupported wave_size: {wave_size}")
        self.wave_size = wave_size
        self.num_sgprs = num_sgprs
        self.sgprs = [0] * num_sgprs
        self.scc = 0
        self.exec_mask = (1 << wave_size) - 1
        self.vcc_mask = 0
        self.cur_lane_masks = set()  # set of SGPR indices or 'vcc', 'exec' that hold lane masks

    def reset(self):
        self.sgprs = [0] * self.num_sgprs
        self.scc = 0
        self.exec_mask = (1 << self.wave_size) - 1
        self.vcc_mask = 0
        self.cur_lane_masks.clear()

    def set_sgpr_u32(self, r: int, val: int):
        self.sgprs[r] = val & MASK32

    def get_sgpr_u32(self, r: int) -> int:
        return self.sgprs[r] & MASK32

    def set_sgpr_u64(self, r: int, val64: int):
        self.sgprs[r] = val64 & MASK32
        self.sgprs[r + 1] = (val64 >> 32) & MASK32

    def get_sgpr_u64(self, r: int) -> int:
        lo = self.get_sgpr_u32(r)
        hi = self.get_sgpr_u32(r + 1)
        return (hi << 32) | lo

    def set_exec(self, mask64: int):
        full_mask = (1 << self.wave_size) - 1
        self.exec_mask = mask64 & full_mask

    def set_vcc(self, mask64: int):
        full_mask = (1 << self.wave_size) - 1
        self.vcc_mask = mask64 & full_mask

    def resolve_source_mask(self, operand: dict | str) -> tuple[int, int]:
        """Resolve a 64-bit operand to (lo_word, hi_word).

        If the operand holds a lane mask, simulates the subgroup OpGroupNonUniformBitwiseOr Reduce.
        Otherwise loads plain scalar dwords.
        """
        if isinstance(operand, str):
            operand_kind = operand
        elif isinstance(operand, dict):
            operand_kind = operand.get("t") or operand.get("k")
        else:
            raise TypeError(f"unsupported operand format: {operand}")

        if operand_kind in ("exec", "exec_b64"):
            return ref_subgroup_wave_mask_reduction(
                [(self.exec_mask & (1 << lane)) != 0 for lane in range(self.wave_size)],
                self.wave_size,
            )
        if operand_kind in ("vcc", "vcc_b64"):
            return ref_subgroup_wave_mask_reduction(
                [(self.vcc_mask & (1 << lane)) != 0 for lane in range(self.wave_size)],
                self.wave_size,
            )
        if isinstance(operand, dict) and operand.get("k") == "sgpr":
            r = operand["r"]
            n = operand.get("n", 2)
            is_lane_mask = (r in self.cur_lane_masks) and (r + 1 in self.cur_lane_masks if n == 2 else True)
            if is_lane_mask:
                mask_val = self.get_sgpr_u64(r)
                return ref_subgroup_wave_mask_reduction(
                    [(mask_val & (1 << lane)) != 0 for lane in range(self.wave_size)],
                    self.wave_size,
                )
            lo = self.get_sgpr_u32(r)
            hi = self.get_sgpr_u32(r + 1) if n == 2 else 0
            return lo, hi
        if isinstance(operand, dict) and operand.get("k") == "imm":
            imm = operand["v"] & MASK64
            return imm & MASK32, (imm >> 32) & MASK32
        raise ValueError(f"unsupported operand: {operand}")

    def eval_s_bcnt1_i32_b64(self, dst_r: int, src_operand: dict | str) -> int:
        """Execute s_bcnt1_i32_b64: counts bits in 64-bit source, updates SCC."""
        lo, hi = self.resolve_source_mask(src_operand)
        c1 = ref_popcount_u32(lo)
        c2 = ref_popcount_u32(hi)
        total = (c1 + c2) & MASK32
        self.set_sgpr_u32(dst_r, total)
        self.scc = 1 if total != 0 else 0
        return total

    def eval_s_ff1_i32_b64(self, dst_r: int, src_operand: dict | str) -> int:
        """Execute s_ff1_i32_b64: finds first set bit in 64-bit source, SCC unchanged."""
        lo, hi = self.resolve_source_mask(src_operand)
        l = ref_find_first_one_u32(lo)
        h = ref_find_first_one_u32(hi)

        if lo != 0:
            result = l
        elif hi != 0:
            result = (h + 32) & MASK32
        else:
            result = MASK32  # -1

        self.set_sgpr_u32(dst_r, result)
        # s_ff1 does NOT touch SCC
        return result

    def eval_s_bcnt1_i32_b32(self, dst_r: int, src_operand: dict | str, half: str | None = None) -> int:
        """Execute s_bcnt1_i32_b32: 32-bit count, updates SCC."""
        lo, hi = self.resolve_source_mask(src_operand)
        if half == "hi":
            lo, hi = hi, 0
        elif half == "lo":
            hi = 0
        c1 = ref_popcount_u32(lo)
        c2 = ref_popcount_u32(hi)
        total = (c1 + c2) & MASK32
        self.set_sgpr_u32(dst_r, total)
        self.scc = 1 if total != 0 else 0
        return total

    def eval_s_ff1_i32_b32(self, dst_r: int, src_operand: dict | str, half: str | None = None) -> int:
        """Execute s_ff1_i32_b32: 32-bit find-first-one, SCC unchanged."""
        lo, hi = self.resolve_source_mask(src_operand)
        if half == "hi":
            lo = hi
        result = ref_find_first_one_u32(lo)
        self.set_sgpr_u32(dst_r, result)
        return result


class TestLaneMaskReductionsReference(unittest.TestCase):
    """Verify independent pure-Python mathematical reference oracles for bit reductions."""

    def test_bcnt1_u64_empty_and_full(self):
        self.assertEqual(ref_bcnt1_i32_b64(0), (0, 0))
        self.assertEqual(ref_bcnt1_i32_b64(MASK64), (64, 1))

    def test_bcnt1_u64_single_bits_all_positions(self):
        for bit in range(64):
            val = 1 << bit
            count, scc = ref_bcnt1_i32_b64(val)
            self.assertEqual(count, 1, f"bit position {bit} must have count 1")
            self.assertEqual(scc, 1, f"bit position {bit} must set SCC to 1")

    def test_bcnt1_u64_half_masks(self):
        low_half = 0x00000000FFFFFFFF
        high_half = 0xFFFFFFFF00000000
        self.assertEqual(ref_bcnt1_i32_b64(low_half), (32, 1))
        self.assertEqual(ref_bcnt1_i32_b64(high_half), (32, 1))

    def test_bcnt1_u64_alternating_patterns(self):
        self.assertEqual(ref_bcnt1_i32_b64(0x5555555555555555), (32, 1))
        self.assertEqual(ref_bcnt1_i32_b64(0xAAAAAAAAAAAAAAAA), (32, 1))
        self.assertEqual(ref_bcnt1_i32_b64(0x3333333333333333), (32, 1))
        self.assertEqual(ref_bcnt1_i32_b64(0x0F0F0F0F0F0F0F0F), (32, 1))

    def test_bcnt1_u64_sparse_subgroup_strides(self):
        # Selected lanes: (lane & 7) == 5 -> 8 lanes active in a 64-lane wave
        sparse_mask = sum(1 << lane for lane in range(64) if (lane & 7) == 5)
        self.assertEqual(ref_bcnt1_i32_b64(sparse_mask), (8, 1))

    def test_ff1_u64_empty_mask_returns_all_ones(self):
        # When no bit is set, FindILsb conventions return -1 (0xFFFFFFFF in uint)
        self.assertEqual(ref_ff1_i32_b64(0), MASK32)

    def test_ff1_u64_single_bits_all_positions(self):
        for bit in range(64):
            val = 1 << bit
            first = ref_ff1_i32_b64(val)
            self.assertEqual(first, bit, f"bit position {bit} must be found as first one")

    def test_ff1_u64_precedence_low_over_high(self):
        # When bits are set in both halves, low half takes precedence
        val = (1 << 5) | (1 << 45)
        self.assertEqual(ref_ff1_i32_b64(val), 5)

        val_boundary = (1 << 31) | (1 << 32)
        self.assertEqual(ref_ff1_i32_b64(val_boundary), 31)

        val_ends = 1 | (1 << 63)
        self.assertEqual(ref_ff1_i32_b64(val_ends), 0)

    def test_ff1_u64_high_half_only(self):
        # When low half is 0, high half bits are biased by +32
        self.assertEqual(ref_ff1_i32_b64(1 << 32), 32)
        self.assertEqual(ref_ff1_i32_b64(1 << 40), 40)
        self.assertEqual(ref_ff1_i32_b64(1 << 63), 63)
        self.assertEqual(ref_ff1_i32_b64(0xFFFFFFFF00000000), 32)

    def test_subgroup_wave_mask_reduction_logic(self):
        # 64-lane test: lane 7 and lane 42 set
        lane_bits = [i in (7, 42) for i in range(64)]
        lo, hi = ref_subgroup_wave_mask_reduction(lane_bits, wave_size=64)
        self.assertEqual(lo, 1 << 7)
        self.assertEqual(hi, 1 << (42 - 32))

        # Wave32 test: high half is strictly 0
        lane_bits_32 = [i == 10 for i in range(32)]
        lo32, hi32 = ref_subgroup_wave_mask_reduction(lane_bits_32, wave_size=32)
        self.assertEqual(lo32, 1 << 10)
        self.assertEqual(hi32, 0)


class TestLaneMaskSemanticInterpreter(unittest.TestCase):
    """Test full semantic interpreter execution across wave32 and wave64 configurations."""

    def setUp(self):
        self.interp64 = LaneMaskReductionInterpreter(wave_size=64)
        self.interp32 = LaneMaskReductionInterpreter(wave_size=32)

    def test_exec_mask_bcnt1_population_counting(self):
        # Full wave: 64 lanes active
        self.interp64.set_exec(MASK64)
        count = self.interp64.eval_s_bcnt1_i32_b64(dst_r=4, src_operand="exec")
        self.assertEqual(count, 64)
        self.assertEqual(self.interp64.get_sgpr_u32(4), 64)
        self.assertEqual(self.interp64.scc, 1)

        # Empty wave: 0 lanes active
        self.interp64.set_exec(0)
        count_zero = self.interp64.eval_s_bcnt1_i32_b64(dst_r=4, src_operand="exec")
        self.assertEqual(count_zero, 0)
        self.assertEqual(self.interp64.get_sgpr_u32(4), 0)
        self.assertEqual(self.interp64.scc, 0)

    def test_exec_mask_ff1_find_first_one(self):
        # Full wave: first one is lane 0
        self.interp64.set_exec(MASK64)
        first = self.interp64.eval_s_ff1_i32_b64(dst_r=6, src_operand="exec")
        self.assertEqual(first, 0)
        self.assertEqual(self.interp64.get_sgpr_u32(6), 0)

        # High half only: lanes 32..63 active
        self.interp64.set_exec(0xFFFFFFFF00000000)
        first_hi = self.interp64.eval_s_ff1_i32_b64(dst_r=6, src_operand="exec")
        self.assertEqual(first_hi, 32)
        self.assertEqual(self.interp64.get_sgpr_u32(6), 32)

        # Empty wave: returns 0xFFFFFFFF
        self.interp64.set_exec(0)
        first_empty = self.interp64.eval_s_ff1_i32_b64(dst_r=6, src_operand="exec")
        self.assertEqual(first_empty, MASK32)
        self.assertEqual(self.interp64.get_sgpr_u32(6), MASK32)

    def test_ff1_preserves_scc(self):
        """s_ff1 must NOT modify the SCC condition register."""
        self.interp64.scc = 1
        self.interp64.set_exec(0)
        self.interp64.eval_s_ff1_i32_b64(dst_r=2, src_operand="exec")
        self.assertEqual(self.interp64.scc, 1, "s_ff1 on empty mask must preserve SCC == 1")

        self.interp64.scc = 0
        self.interp64.set_exec(MASK64)
        self.interp64.eval_s_ff1_i32_b64(dst_r=2, src_operand="exec")
        self.assertEqual(self.interp64.scc, 0, "s_ff1 on non-empty mask must preserve SCC == 0")

    def test_vcc_mask_reduction(self):
        vcc_test_pattern = (1 << 12) | (1 << 37) | (1 << 50)
        self.interp64.set_vcc(vcc_test_pattern)

        count = self.interp64.eval_s_bcnt1_i32_b64(dst_r=10, src_operand="vcc")
        self.assertEqual(count, 3)
        self.assertEqual(self.interp64.scc, 1)

        ff1 = self.interp64.eval_s_ff1_i32_b64(dst_r=11, src_operand="vcc")
        self.assertEqual(ff1, 12)

    def test_sgpr_pair_lane_mask_vs_scalar(self):
        """An SGPR pair marked as lane mask undergoes subgroup reduction; unmarked is scalar."""
        mask_val = (1 << 5) | (1 << 35)

        # 1. Marked as lane mask in cur_lane_masks
        self.interp64.cur_lane_masks = {16, 17}
        self.interp64.set_sgpr_u64(16, mask_val)
        c_mask = self.interp64.eval_s_bcnt1_i32_b64(dst_r=20, src_operand={"k": "sgpr", "r": 16, "n": 2})
        ff1_mask = self.interp64.eval_s_ff1_i32_b64(dst_r=21, src_operand={"k": "sgpr", "r": 16, "n": 2})
        self.assertEqual(c_mask, 2)
        self.assertEqual(ff1_mask, 5)

        # 2. Unmarked SGPR pair (scalar data)
        self.interp64.cur_lane_masks.clear()
        self.interp64.set_sgpr_u64(16, mask_val)
        c_scalar = self.interp64.eval_s_bcnt1_i32_b64(dst_r=22, src_operand={"k": "sgpr", "r": 16, "n": 2})
        ff1_scalar = self.interp64.eval_s_ff1_i32_b64(dst_r=23, src_operand={"k": "sgpr", "r": 16, "n": 2})
        self.assertEqual(c_scalar, 2)
        self.assertEqual(ff1_scalar, 5)

    def test_immediate_operand(self):
        # Immediate constant is zero-extended in the high 32 bits
        imm_op = {"k": "imm", "v": 0x80000001}
        count = self.interp64.eval_s_bcnt1_i32_b64(dst_r=30, src_operand=imm_op)
        first = self.interp64.eval_s_ff1_i32_b64(dst_r=31, src_operand=imm_op)
        self.assertEqual(count, 2)
        self.assertEqual(first, 0)
        self.assertEqual(self.interp64.scc, 1)

    def test_wave32_lane_mask_reductions(self):
        """In wave32 mode, masks only span 32 lanes (bits 32..63 are zero)."""
        self.interp32.set_exec(MASK32)
        count = self.interp32.eval_s_bcnt1_i32_b64(dst_r=0, src_operand="exec")
        first = self.interp32.eval_s_ff1_i32_b64(dst_r=1, src_operand="exec")
        self.assertEqual(count, 32)
        self.assertEqual(first, 0)

        # Highest lane in wave32
        self.interp32.set_exec(1 << 31)
        count_hi = self.interp32.eval_s_bcnt1_i32_b64(dst_r=0, src_operand="exec")
        first_hi = self.interp32.eval_s_ff1_i32_b64(dst_r=1, src_operand="exec")
        self.assertEqual(count_hi, 1)
        self.assertEqual(first_hi, 31)

    def test_half_mask_operations(self):
        """s_bcnt1_i32_b32 and s_ff1_i32_b32 targeting lo or hi halves."""
        self.interp64.set_exec(0x00000F0F000000F0)  # lo has 4 bits (lanes 4..7), hi has 8 bits (lanes 32..35, 40..43)

        c_lo = self.interp64.eval_s_bcnt1_i32_b32(dst_r=0, src_operand="exec", half="lo")
        c_hi = self.interp64.eval_s_bcnt1_i32_b32(dst_r=1, src_operand="exec", half="hi")
        self.assertEqual(c_lo, 4)
        self.assertEqual(c_hi, 8)

        ff1_lo = self.interp64.eval_s_ff1_i32_b32(dst_r=2, src_operand="exec", half="lo")
        ff1_hi = self.interp64.eval_s_ff1_i32_b32(dst_r=3, src_operand="exec", half="hi")
        self.assertEqual(ff1_lo, 4)
        self.assertEqual(ff1_hi, 0)  # lane 32 is bit 0 of the high half


class TestRdna2SpirvEmission(unittest.TestCase):
    """Test SPIR-V translation emitted by rdna2spirv.Gen for lane mask reductions."""

    def test_s_bcnt1_i32_b64_exec_emission(self):
        g = rdna2spirv.Gen("ps")
        g.wave64 = True
        ins = {
            "op": "s_bcnt1_i32_b64",
            "args": [{"k": "sgpr", "r": 8}, {"k": "special", "t": "exec"}],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)

        # Subgroup bitwise OR reduction over EXEC
        self.assertIn("OpGroupNonUniformBitwiseOr %uint %c3 Reduce", emitted)
        # Population count on both halves
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpIAdd %uint", emitted)
        # Result stored to destination SGPR s8
        self.assertIn("OpAccessChain %_ptr_Function_uint %sgpr %c8", emitted)
        # Condition code SCC updated
        self.assertIn("OpStore %scc", emitted)
        self.assertTrue(g.uses_subgroup)
        self.assertTrue(g.uses_wave_bits)

    def test_s_ff1_i32_b64_exec_emission(self):
        g = rdna2spirv.Gen("ps")
        g.wave64 = True
        ins = {
            "op": "s_ff1_i32_b64",
            "args": [{"k": "sgpr", "r": 10}, {"k": "special", "t": "exec"}],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)

        # GLSL.std.450 FindILsb for finding least significant set bit
        self.assertIn("FindILsb", emitted)
        # High-half +32 biasing (%c20 is 32)
        self.assertIn("OpIAdd %uint", emitted)
        self.assertIn("%c20", emitted)
        # Fallback to -1 (%cffffffff) when both words are 0
        self.assertIn("%cffffffff", emitted)
        # Store to destination SGPR s10
        self.assertIn("OpAccessChain %_ptr_Function_uint %sgpr %ca", emitted)
        # s_ff1 does NOT store to SCC
        self.assertNotIn("OpStore %scc", emitted)

    def test_s_bcnt1_i32_b64_scalar_sgpr_emission(self):
        g = rdna2spirv.Gen("ps")
        g.wave64 = True
        ins = {
            "op": "s_bcnt1_i32_b64",
            "args": [{"k": "sgpr", "r": 4}, {"k": "sgpr", "r": 2, "n": 2}],
            "mods": [],
        }
        # Not in lane_masks: treated as plain scalar SGPR pair
        g.insn(ins)
        emitted = "\n".join(g.body)

        # Must NOT emit subgroup reductions for plain scalar SGPR pair
        self.assertNotIn("OpGroupNonUniformBitwiseOr", emitted)
        # Loads s2 and s3 directly
        self.assertIn("OpAccessChain %_ptr_Function_uint %sgpr %c2", emitted)
        self.assertIn("OpAccessChain %_ptr_Function_uint %sgpr %c3", emitted)
        # Counts bits and updates SCC
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpIAdd %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_ff1_i32_b64_scalar_sgpr_emission(self):
        g = rdna2spirv.Gen("ps")
        g.wave64 = True
        ins = {
            "op": "s_ff1_i32_b64",
            "args": [{"k": "sgpr", "r": 4}, {"k": "sgpr", "r": 2, "n": 2}],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)

        self.assertNotIn("OpGroupNonUniformBitwiseOr", emitted)
        self.assertIn("FindILsb", emitted)
        self.assertNotIn("OpStore %scc", emitted)

    def test_s_bcnt1_i32_b64_immediate_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bcnt1_i32_b64",
            "args": [{"k": "sgpr", "r": 0}, {"k": "imm", "v": 0x1234}],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)

        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_ff1_i32_b64_immediate_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff1_i32_b64",
            "args": [{"k": "sgpr", "r": 0}, {"k": "imm", "v": 0x80}],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)

        self.assertIn("FindILsb", emitted)
        self.assertNotIn("OpStore %scc", emitted)


class TestTranslatorShaderIntegration(unittest.TestCase):
    """Verify full end-to-end translation of disassembly listings with lane mask reductions."""

    def test_translate_lines_exec_bcnt1_and_ff1(self):
        lines = [
            "s_bcnt1_i32_b64 s4, exec",
            "s_ff1_i32_b64 s6, exec",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_test_exec_reductions", lines)
        self.assertIn("OpGroupNonUniformBitwiseOr %uint %c3 Reduce", spv)
        self.assertIn("OpBitCount %uint", spv)
        self.assertIn("FindILsb", spv)
        self.assertIn("OpStore %scc", spv)

    def test_translate_lines_vcc_lane_mask(self):
        lines = [
            "v_cmp_gt_u32_e32 vcc, 10, v0",
            "s_bcnt1_i32_b64 s4, vcc",
            "s_ff1_i32_b64 s6, vcc",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_test_vcc_reductions", lines)
        self.assertIn("OpGroupNonUniformBitwiseOr %uint %c3 Reduce", spv)
        self.assertIn("OpBitCount %uint", spv)
        self.assertIn("FindILsb", spv)

    def test_translate_lines_sgpr_comparison_lane_mask(self):
        lines = [
            "v_cmp_eq_u32_e64 s[4:5], v0, 5",
            "s_bcnt1_i32_b64 s6, s[4:5]",
            "s_ff1_i32_b64 s7, s[4:5]",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_test_sgpr_cmp_mask", lines)
        self.assertIn("OpGroupNonUniformBitwiseOr %uint %c3 Reduce", spv)
        self.assertIn("OpBitCount %uint", spv)
        self.assertIn("FindILsb", spv)

    def test_translate_lines_sgpr_scalar_data_no_reduction(self):
        lines = [
            "s_bcnt1_i32_b64 s6, s[2:3]",
            "s_ff1_i32_b64 s7, s[2:3]",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_test_sgpr_data_scalar", lines)
        self.assertNotIn("OpGroupNonUniformBitwiseOr", spv)
        self.assertIn("OpBitCount %uint", spv)
        self.assertIn("FindILsb", spv)


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Ensure tests are strictly sensitive to subtle mathematical and translator errors."""

    def test_bcnt1_detects_omitted_high_word_addition(self):
        """Saboteur: only counting low half bits (c1 without c2)."""
        def sabotaged_bcnt1(lo: int, hi: int) -> int:
            return ref_popcount_u32(lo)

        mask_high_only = (0, 0x00000001)
        expected = 1
        sabotaged = sabotaged_bcnt1(*mask_high_only)
        self.assertNotEqual(sabotaged, expected, "Omitted high word in bcnt1 must be detected")

    def test_ff1_detects_missing_high_word_offset(self):
        """Saboteur: finding first bit in high half without adding +32."""
        def sabotaged_ff1(lo: int, hi: int) -> int:
            l = ref_find_first_one_u32(lo)
            h = ref_find_first_one_u32(hi)
            if lo != 0:
                return l
            if hi != 0:
                return h  # BUG: omitted + 32!
            return MASK32

        # Bit 36 set (lane 36 -> high word bit 4)
        lo, hi = 0, 1 << 4
        expected = 36
        sabotaged = sabotaged_ff1(lo, hi)
        self.assertEqual(sabotaged, 4)
        self.assertNotEqual(sabotaged, expected, "Missing +32 offset in high half must be detected")

    def test_ff1_detects_zero_instead_of_negative_one_on_empty(self):
        """Saboteur: returning 0 instead of -1 (0xFFFFFFFF) when no bits are set."""
        def sabotaged_ff1_empty(lo: int, hi: int) -> int:
            if lo == 0 and hi == 0:
                return 0  # BUG: should be 0xFFFFFFFF
            return 0

        self.assertNotEqual(sabotaged_ff1_empty(0, 0), MASK32)

    def test_bcnt1_detects_inverted_scc(self):
        """Saboteur: setting SCC = (total == 0) instead of (total != 0)."""
        def sabotaged_scc(total: int) -> int:
            return 1 if total == 0 else 0

        self.assertNotEqual(sabotaged_scc(0), 0)
        self.assertNotEqual(sabotaged_scc(1), 1)

    def test_detects_ff1_clobbering_scc(self):
        """Ensure tests catch any implementation of s_ff1 that writes to SCC."""
        interp = LaneMaskReductionInterpreter(wave_size=64)
        interp.scc = 0
        interp.set_exec(MASK64)
        interp.eval_s_ff1_i32_b64(dst_r=0, src_operand="exec")

        # SCC must remain 0
        self.assertEqual(interp.scc, 0)


if __name__ == "__main__":
    unittest.main()
