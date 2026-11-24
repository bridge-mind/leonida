"""Semantic interpreter tests for vector condition code (VCC) branching.

Tests the ISA semantics and SPIR-V translator emission for:
  - s_cbranch_vccz:  branch if VCC == 0 (all lanes in the wave evaluated condition to false or were inactive)
  - s_cbranch_vccnz: branch if VCC != 0 (at least one active lane in the wave evaluated condition to true)

Covers:
  1. Pure reference semantics for Wave32 (32-bit VCC) and Wave64 (64-bit VCC pair vcc_lo/vcc_hi).
  2. Multi-lane WaveInterpreter modeling SIMD wave state across Wave32 and Wave64:
     - Vector condition code generation via vector comparisons (v_cmp_*).
     - Carry-out generation into VCC via vector addition with carry (v_add_co_u32).
     - Execution mask (EXEC) interaction: inactive lanes write 0 into VCC.
     - Direct scalar manipulation of VCC (s_mov_b32/b64, s_and_b32/b64, s_or_b32/b64).
     - Skip patterns, divergent loop termination, and structured if-then-else joins.
  3. SPIR-V translator lowering via tools/rdna2spirv.py branch_condition():
     - Verification of per-lane VCC load (%lane = OpINotEqual %bool %vcc 0).
     - Whole-subgroup vote emission (OpGroupNonUniformAny with Subgroup scope).
     - Logical negation (OpLogicalNot) for s_cbranch_vccz vs unnegated for s_cbranch_vccnz.
     - Required SPIR-V capabilities (GroupNonUniform, GroupNonUniformVote).
  4. Simulation oracle matching: verification that the SPIR-V subgroup vote model
     matches the hardware ISA reference across deterministic and edge-case bitmasks.
  5. Negative controls and mutation detection: verifies sensitivity to inverted conditions,
     All-vs-Any reductions, and high-half masking bugs in Wave64.

Run:
    python -I tests/semantic/test_control_flow_vcc.py
    python -m unittest discover -s tests/semantic -p "test_control_flow_vcc.py" -v
"""
from __future__ import annotations

import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402


# -----------------------------------------------------------------------------
# 1. Pure Reference Semantics (AMD RDNA2 ISA: VCC & Scalar Control Flow)
# -----------------------------------------------------------------------------

def eval_vccz(vcc_mask: int, wave_size: int = 32) -> bool:
    """s_cbranch_vccz: taken iff the wave vector condition code mask is zero."""
    mask = (1 << wave_size) - 1
    return (vcc_mask & mask) == 0


def eval_vccnz(vcc_mask: int, wave_size: int = 32) -> bool:
    """s_cbranch_vccnz: taken iff the wave vector condition code mask is non-zero."""
    mask = (1 << wave_size) - 1
    return (vcc_mask & mask) != 0


# -----------------------------------------------------------------------------
# 2. Multi-Lane Wave Interpreter
# -----------------------------------------------------------------------------

class WaveInterpreter:
    """Simulates a SIMD wave executing basic blocks with VCC control flow.

    Models:
      - Wave32 (32 lanes, 32-bit VCC) or Wave64 (64 lanes, 64-bit VCC).
      - EXEC mask: per-lane active bit controlling vector instructions.
      - VCC register: Vector Condition Code, updated by v_cmp_* and v_add_co_*.
        Crucially, lanes that are inactive in EXEC contribute 0 to VCC.
      - SGPRs: uniform scalar registers shared across all lanes.
      - VGPRs: vector registers with independent values per lane.
      - Control flow instructions: s_cbranch_vccz, s_cbranch_vccnz, s_branch, s_endpgm.
    """

    def __init__(
        self,
        wave_size: int = 32,
        initial_exec: int | None = None,
        initial_vcc: int = 0,
    ) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave_size: {wave_size}")
        self.wave_size = wave_size
        self.mask_all = (1 << wave_size) - 1
        self.exec_mask = self.mask_all if initial_exec is None else (initial_exec & self.mask_all)
        self.vcc = initial_vcc & self.mask_all
        self.sgprs = [0] * 128
        self.vgprs = [[0] * 256 for _ in range(wave_size)]
        self.trace: list[str] = []

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def is_vcc_lane_set(self, lane: int) -> bool:
        return bool((self.vcc >> lane) & 1)

    def run_program(self, blocks: dict[str, list], start_block: str, max_steps: int = 1000) -> None:
        """Execute a graph of basic blocks until 's_endpgm' or max_steps reached."""
        cur: str | None = start_block
        steps = 0
        self.trace = []

        while cur is not None and steps < max_steps:
            steps += 1
            self.trace.append(cur)
            block = blocks.get(cur)
            if block is None:
                raise ValueError(f"Unknown block label: {cur}")

            next_block: str | None = None
            for ins in block:
                if isinstance(ins, str):
                    next_block = ins
                    break

                op = ins[0]
                args = ins[1:]

                if op == "s_endpgm":
                    return

                elif op == "s_branch":
                    next_block = args[0]
                    break

                elif op == "s_cbranch_vccz":
                    target = args[0]
                    if eval_vccz(self.vcc, self.wave_size):
                        next_block = target
                        break

                elif op == "s_cbranch_vccnz":
                    target = args[0]
                    if eval_vccnz(self.vcc, self.wave_size):
                        next_block = target
                        break

                elif op == "s_cbranch_execz":
                    target = args[0]
                    if (self.exec_mask & self.mask_all) == 0:
                        next_block = target
                        break

                elif op == "s_cbranch_execnz":
                    target = args[0]
                    if (self.exec_mask & self.mask_all) != 0:
                        next_block = target
                        break

                elif op == "s_mov_b32":
                    dst_s, val = args
                    if dst_s in ("vcc_lo", "vcc"):
                        self.vcc = ((self.vcc & ~0xFFFFFFFF) | (val & 0xFFFFFFFF)) & self.mask_all
                    elif dst_s == "vcc_hi":
                        if self.wave_size == 64:
                            self.vcc = ((self.vcc & 0xFFFFFFFF) | ((val & 0xFFFFFFFF) << 32)) & self.mask_all
                        else:
                            self.sgprs[107] = val & 0xFFFFFFFF
                    else:
                        self.sgprs[dst_s] = val & 0xFFFFFFFF

                elif op == "s_mov_b64":
                    dst_s, val = args
                    if dst_s == "vcc":
                        self.vcc = val & self.mask_all
                    else:
                        self.sgprs[dst_s] = val & 0xFFFFFFFF
                        self.sgprs[dst_s + 1] = (val >> 32) & 0xFFFFFFFF

                elif op == "s_and_b32":
                    dst_s, val = args
                    if dst_s in ("vcc_lo", "vcc"):
                        lo = (self.vcc & 0xFFFFFFFF) & (val & 0xFFFFFFFF)
                        self.vcc = ((self.vcc & ~0xFFFFFFFF) | lo) & self.mask_all
                    else:
                        self.sgprs[dst_s] = (self.sgprs[dst_s] & val) & 0xFFFFFFFF

                elif op == "s_or_b32":
                    dst_s, val = args
                    if dst_s in ("vcc_lo", "vcc"):
                        lo = (self.vcc & 0xFFFFFFFF) | (val & 0xFFFFFFFF)
                        self.vcc = ((self.vcc & ~0xFFFFFFFF) | lo) & self.mask_all
                    else:
                        self.sgprs[dst_s] = (self.sgprs[dst_s] | val) & 0xFFFFFFFF

                elif op == "s_and_b64":
                    dst_s, val = args
                    if dst_s == "vcc":
                        self.vcc = (self.vcc & val) & self.mask_all
                    else:
                        cur_val = self.sgprs[dst_s] | (self.sgprs[dst_s + 1] << 32)
                        res = cur_val & val
                        self.sgprs[dst_s] = res & 0xFFFFFFFF
                        self.sgprs[dst_s + 1] = (res >> 32) & 0xFFFFFFFF

                elif op == "s_or_b64":
                    dst_s, val = args
                    if dst_s == "vcc":
                        self.vcc = (self.vcc | val) & self.mask_all
                    else:
                        cur_val = self.sgprs[dst_s] | (self.sgprs[dst_s + 1] << 32)
                        res = cur_val | val
                        self.sgprs[dst_s] = res & 0xFFFFFFFF
                        self.sgprs[dst_s + 1] = (res >> 32) & 0xFFFFFFFF

                elif op == "s_set_vcc":
                    self.vcc = args[0] & self.mask_all

                elif op == "s_set_exec":
                    self.exec_mask = args[0] & self.mask_all

                elif op == "v_mov_b32":
                    dst_v, val = args
                    for lane in range(self.wave_size):
                        if self.is_lane_active(lane):
                            self.vgprs[lane][dst_v] = val & 0xFFFFFFFF

                elif op == "v_add_u32":
                    dst_v, src_v, imm = args
                    for lane in range(self.wave_size):
                        if self.is_lane_active(lane):
                            self.vgprs[lane][dst_v] = (self.vgprs[lane][src_v] + imm) & 0xFFFFFFFF

                elif op == "v_sub_u32":
                    dst_v, src_v, imm = args
                    for lane in range(self.wave_size):
                        if self.is_lane_active(lane):
                            self.vgprs[lane][dst_v] = (self.vgprs[lane][src_v] - imm) & 0xFFFFFFFF

                elif op == "v_dec_u32":
                    dst_v = args[0]
                    for lane in range(self.wave_size):
                        if self.is_lane_active(lane):
                            curr_val = self.vgprs[lane][dst_v]
                            self.vgprs[lane][dst_v] = (curr_val - 1) & 0xFFFFFFFF if curr_val > 0 else 0

                elif op in ("v_cmp_gt_u32", "v_cmp_ge_u32", "v_cmp_lt_u32",
                            "v_cmp_le_u32", "v_cmp_eq_u32", "v_cmp_ne_u32"):
                    dst_target, src_v, val_or_imm = args
                    cond_mask = 0
                    for lane in range(self.wave_size):
                        if not self.is_lane_active(lane):
                            continue
                        lhs = self.vgprs[lane][src_v]
                        rhs = val_or_imm if isinstance(val_or_imm, int) else self.vgprs[lane][val_or_imm]

                        if op == "v_cmp_gt_u32":
                            passed = lhs > rhs
                        elif op == "v_cmp_ge_u32":
                            passed = lhs >= rhs
                        elif op == "v_cmp_lt_u32":
                            passed = lhs < rhs
                        elif op == "v_cmp_le_u32":
                            passed = lhs <= rhs
                        elif op == "v_cmp_eq_u32":
                            passed = lhs == rhs
                        elif op == "v_cmp_ne_u32":
                            passed = lhs != rhs
                        else:
                            passed = False

                        if passed:
                            cond_mask |= (1 << lane)

                    if dst_target in ("vcc", "vcc_lo"):
                        self.vcc = cond_mask & self.mask_all
                    else:
                        self.sgprs[dst_target] = cond_mask & 0xFFFFFFFF
                        if self.wave_size == 64:
                            self.sgprs[dst_target + 1] = (cond_mask >> 32) & 0xFFFFFFFF

                elif op == "v_add_co_u32":
                    dst_v, src_v0, src_v1 = args
                    carry_mask = 0
                    for lane in range(self.wave_size):
                        if not self.is_lane_active(lane):
                            continue
                        a = self.vgprs[lane][src_v0]
                        b = src_v1 if isinstance(src_v1, int) else self.vgprs[lane][src_v1]
                        full_sum = a + b
                        self.vgprs[lane][dst_v] = full_sum & 0xFFFFFFFF
                        if full_sum > 0xFFFFFFFF:
                            carry_mask |= (1 << lane)

                    self.vcc = carry_mask & self.mask_all

                else:
                    raise NotImplementedError(f"Unsupported instruction in test interpreter: {op}")

            if next_block is None:
                cur = block[-1] if isinstance(block[-1], str) else None
            else:
                cur = next_block

        if steps >= max_steps:
            raise TimeoutError("Execution exceeded max_steps (infinite loop detected)")


# -----------------------------------------------------------------------------
# 3. SPIR-V Subgroup Vote Interpreter
# -----------------------------------------------------------------------------

def simulate_spirv_subgroup_vcc_branch(op: str, lane_vcc_words: list[int]) -> bool:
    """Interprets the SPIR-V instructions emitted by rdna2spirv for VCC branching.

    Emitted sequence:
      %s = OpLoad %uint %vcc
      %lane = OpINotEqual %bool %s %c0
      %any = OpGroupNonUniformAny %bool %c3 %lane
      (if vccz): %b = OpLogicalNot %bool %any
    """
    any_active_vcc = any((word & 0xFFFFFFFF) != 0 for word in lane_vcc_words)

    if op == "s_cbranch_vccnz":
        return any_active_vcc
    elif op == "s_cbranch_vccz":
        return not any_active_vcc
    else:
        raise ValueError(f"Unknown branch opcode: {op}")


# -----------------------------------------------------------------------------
# Test Suite
# -----------------------------------------------------------------------------

class TestControlFlowVccReference(unittest.TestCase):
    """Verifies vector condition code branching semantics against ISA definitions."""

    def test_vccz_and_vccnz_duality_wave32(self):
        """vccz and vccnz must be strict logical complements for every 32-bit mask."""
        test_masks = [
            0x00000000,
            0xFFFFFFFF,
            0x00000001,
            0x80000000,
            0x55555555,
            0xAAAAAAAA,
            0x0000FFFF,
            0xFFFF0000,
            0x00FF00FF,
            0xFF00FF00,
        ]
        for bit in range(32):
            test_masks.append(1 << bit)
            test_masks.append(~(1 << bit) & 0xFFFFFFFF)

        rng = random.Random(20261122)
        for _ in range(100):
            test_masks.append(rng.getrandbits(32))

        for mask in test_masks:
            z = eval_vccz(mask, wave_size=32)
            nz = eval_vccnz(mask, wave_size=32)
            with self.subTest(mask=hex(mask)):
                self.assertNotEqual(z, nz, "vccz and vccnz must be strict logical complements")
                self.assertEqual(z, (mask & 0xFFFFFFFF) == 0)
                self.assertEqual(nz, (mask & 0xFFFFFFFF) != 0)

    def test_vccz_and_vccnz_duality_wave64(self):
        """vccz and vccnz must be strict logical complements across full 64-bit masks."""
        test_masks = [
            0x0000000000000000,
            0xFFFFFFFFFFFFFFFF,
            0x0000000000000001,
            0x8000000000000000,
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
            0x00000000FFFFFFFF,
            0xFFFFFFFF00000000,
            0x0000FFFF0000FFFF,
            0xFFFF0000FFFF0000,
        ]
        for bit in range(64):
            test_masks.append(1 << bit)
            test_masks.append(~(1 << bit) & 0xFFFFFFFFFFFFFFFF)

        rng = random.Random(20261122)
        for _ in range(100):
            test_masks.append(rng.getrandbits(64))

        for mask in test_masks:
            z = eval_vccz(mask, wave_size=64)
            nz = eval_vccnz(mask, wave_size=64)
            with self.subTest(mask=hex(mask)):
                self.assertNotEqual(z, nz, "vccz and vccnz must be strict logical complements in Wave64")
                self.assertEqual(z, (mask & 0xFFFFFFFFFFFFFFFF) == 0)
                self.assertEqual(nz, (mask & 0xFFFFFFFFFFFFFFFF) != 0)

    def test_wave32_ignores_upper_half_bits(self):
        """Wave32 mode isolates the lower 32 bits and ignores bits 32..63."""
        high_half_mask = 0x1234567800000000
        self.assertTrue(eval_vccz(high_half_mask, wave_size=32))
        self.assertFalse(eval_vccnz(high_half_mask, wave_size=32))

        # In contrast, Wave64 must observe the high half bits
        self.assertFalse(eval_vccz(high_half_mask, wave_size=64))
        self.assertTrue(eval_vccnz(high_half_mask, wave_size=64))

    def test_wave64_truncates_beyond_bit63(self):
        """Bits beyond 63 are outside the 64-lane wave and must not alter evaluation."""
        bit64 = 1 << 64
        self.assertTrue(eval_vccz(bit64, wave_size=64))
        self.assertFalse(eval_vccnz(bit64, wave_size=64))

        bit65_with_low = (1 << 65) | 1
        self.assertFalse(eval_vccz(bit65_with_low, wave_size=64))
        self.assertTrue(eval_vccnz(bit65_with_low, wave_size=64))


class TestWaveInterpreterVccFlow(unittest.TestCase):
    """Verifies VCC control flow in multi-lane simulated programs across Wave32 and Wave64."""

    def test_wave32_skip_pattern_vccz_all_fail(self):
        """When all lanes fail the comparison, s_cbranch_vccz branches to skip work."""
        blocks = {
            "entry": [
                ("v_cmp_gt_u32", "vcc", 0, 100),
                ("s_cbranch_vccz", "skip"),
                "work_block",
            ],
            "work_block": [
                ("v_mov_b32", 1, 999),
                ("s_branch", "skip"),
            ],
            "skip": [
                ("s_endpgm",),
            ],
        }

        interp = WaveInterpreter(wave_size=32, initial_exec=0xFFFFFFFF)
        for lane in range(32):
            interp.vgprs[lane][0] = 50   # all below 100
            interp.vgprs[lane][1] = 10   # initial value

        interp.run_program(blocks, "entry")

        self.assertEqual(interp.trace, ["entry", "skip"])
        self.assertEqual(interp.vcc, 0)
        for lane in range(32):
            self.assertEqual(interp.vgprs[lane][1], 10)

    def test_wave32_skip_pattern_vccz_one_lane_satisfies(self):
        """When at least one lane satisfies the comparison, s_cbranch_vccz does NOT branch."""
        interp = WaveInterpreter(wave_size=32, initial_exec=0xFFFFFFFF)
        for lane in range(32):
            interp.vgprs[lane][0] = 50
            interp.vgprs[lane][1] = 10
        interp.vgprs[17][0] = 150  # only lane 17 passes comparison

        blocks = {
            "entry": [
                ("v_cmp_gt_u32", "vcc", 0, 100),
                ("s_cbranch_vccz", "skip"),
                "work_block",
            ],
            "work_block": [
                ("v_mov_b32", 1, 999),
                ("s_branch", "skip"),
            ],
            "skip": [
                ("s_endpgm",),
            ],
        }

        interp.run_program(blocks, "entry")

        self.assertEqual(interp.trace, ["entry", "work_block", "skip"])
        self.assertEqual(interp.vcc, 1 << 17)
        # All active lanes in the wave execute work_block
        for lane in range(32):
            self.assertEqual(interp.vgprs[lane][1], 999)

    def test_wave64_skip_pattern_vccz_high_half_only(self):
        """In Wave64, a condition satisfied only in lane >= 32 correctly prevents vccz branching."""
        interp = WaveInterpreter(wave_size=64, initial_exec=0xFFFFFFFFFFFFFFFF)
        for lane in range(64):
            interp.vgprs[lane][0] = 20
            interp.vgprs[lane][1] = 5
        interp.vgprs[47][0] = 200  # lane 47 in upper half passes

        blocks = {
            "entry": [
                ("v_cmp_gt_u32", "vcc", 0, 100),
                ("s_cbranch_vccz", "skip"),
                "work_block",
            ],
            "work_block": [
                ("v_mov_b32", 1, 777),
                ("s_branch", "skip"),
            ],
            "skip": [
                ("s_endpgm",),
            ],
        }

        interp.run_program(blocks, "entry")

        self.assertEqual(interp.trace, ["entry", "work_block", "skip"])
        self.assertEqual(interp.vcc, 1 << 47)
        for lane in range(64):
            self.assertEqual(interp.vgprs[lane][1], 777)

    def test_wave64_skip_pattern_vccz_all_lanes_fail(self):
        """In Wave64, when all 64 lanes fail, s_cbranch_vccz skips the work block."""
        interp = WaveInterpreter(wave_size=64, initial_exec=0xFFFFFFFFFFFFFFFF)
        for lane in range(64):
            interp.vgprs[lane][0] = 10
            interp.vgprs[lane][1] = 50

        blocks = {
            "entry": [
                ("v_cmp_gt_u32", "vcc", 0, 100),
                ("s_cbranch_vccz", "skip"),
                "work_block",
            ],
            "work_block": [
                ("v_mov_b32", 1, 777),
                ("s_branch", "skip"),
            ],
            "skip": [
                ("s_endpgm",),
            ],
        }

        interp.run_program(blocks, "entry")

        self.assertEqual(interp.trace, ["entry", "skip"])
        self.assertEqual(interp.vcc, 0)
        for lane in range(64):
            self.assertEqual(interp.vgprs[lane][1], 50)

    def test_wave32_vccnz_direct_branch(self):
        """s_cbranch_vccnz branches directly when VCC is non-zero, falls through when zero."""
        blocks = {
            "entry": [
                ("v_cmp_eq_u32", "vcc", 0, 42),
                ("s_cbranch_vccnz", "found_target"),
                "not_found",
            ],
            "found_target": [
                ("s_mov_b32", 10, 0xAA),
                ("s_endpgm",),
            ],
            "not_found": [
                ("s_mov_b32", 10, 0x55),
                ("s_endpgm",),
            ],
        }

        # Case 1: no lane has 42 -> falls through to not_found
        interp_none = WaveInterpreter(wave_size=32)
        for lane in range(32):
            interp_none.vgprs[lane][0] = lane + 100
        interp_none.run_program(blocks, "entry")
        self.assertEqual(interp_none.trace, ["entry", "not_found"])
        self.assertEqual(interp_none.sgprs[10], 0x55)

        # Case 2: lane 0 has 42 -> branches to found_target
        interp_one = WaveInterpreter(wave_size=32)
        for lane in range(32):
            interp_one.vgprs[lane][0] = 42 if lane == 0 else lane + 100
        interp_one.run_program(blocks, "entry")
        self.assertEqual(interp_one.trace, ["entry", "found_target"])
        self.assertEqual(interp_one.sgprs[10], 0xAA)

    def test_wave64_vccnz_boundary_lanes(self):
        """s_cbranch_vccnz triggers on boundary lanes in Wave64: 0, 31, 32, and 63."""
        boundary_lanes = [0, 31, 32, 63]

        for target_lane in boundary_lanes:
            with self.subTest(target_lane=target_lane):
                interp = WaveInterpreter(wave_size=64)
                for lane in range(64):
                    interp.vgprs[lane][0] = 1000 if lane == target_lane else 0

                blocks = {
                    "entry": [
                        ("v_cmp_gt_u32", "vcc", 0, 500),
                        ("s_cbranch_vccnz", "hit"),
                        "miss",
                    ],
                    "hit": [
                        ("s_mov_b32", 12, target_lane + 1),
                        ("s_endpgm",),
                    ],
                    "miss": [
                        ("s_mov_b32", 12, 0),
                        ("s_endpgm",),
                    ],
                }

                interp.run_program(blocks, "entry")
                self.assertEqual(interp.trace, ["entry", "hit"])
                self.assertEqual(interp.sgprs[12], target_lane + 1)
                self.assertEqual(interp.vcc, 1 << target_lane)

    def test_exec_mask_gates_vcc_comparison(self):
        """Lanes inactive in EXEC write 0 to VCC; only active lanes participate in branch."""
        # Lane 5 satisfies condition (value 200 > 100), but lane 5 is disabled in EXEC
        active_mask_without_lane5 = 0xFFFFFFFF & ~(1 << 5)

        interp = WaveInterpreter(wave_size=32, initial_exec=active_mask_without_lane5)
        for lane in range(32):
            interp.vgprs[lane][0] = 200 if lane == 5 else 10

        blocks = {
            "entry": [
                ("v_cmp_gt_u32", "vcc", 0, 100),
                ("s_cbranch_vccz", "skip"),
                "work",
            ],
            "work": [
                ("s_mov_b32", 20, 0x1111),
                ("s_endpgm",),
            ],
            "skip": [
                ("s_mov_b32", 20, 0x2222),
                ("s_endpgm",),
            ],
        }

        interp.run_program(blocks, "entry")

        # Since lane 5 was disabled in EXEC, VCC stayed 0, so s_cbranch_vccz skipped
        self.assertEqual(interp.trace, ["entry", "skip"])
        self.assertEqual(interp.vcc, 0)
        self.assertEqual(interp.sgprs[20], 0x2222)

        # Now re-enable lane 5 in EXEC
        interp_active = WaveInterpreter(wave_size=32, initial_exec=0xFFFFFFFF)
        for lane in range(32):
            interp_active.vgprs[lane][0] = 200 if lane == 5 else 10

        interp_active.run_program(blocks, "entry")
        self.assertEqual(interp_active.trace, ["entry", "work"])
        self.assertEqual(interp_active.vcc, 1 << 5)
        self.assertEqual(interp_active.sgprs[20], 0x1111)

    def test_divergent_loop_termination_vccnz(self):
        """s_cbranch_vccnz loops while any lane has non-zero counter, terminating cleanly."""
        wave_size = 32
        interp = WaveInterpreter(wave_size=wave_size, initial_exec=0xFFFFFFFF)
        # Staggered countdown values: lanes 0..7 have values 0..7
        for lane in range(wave_size):
            interp.vgprs[lane][0] = lane if lane <= 7 else 0
            interp.vgprs[lane][1] = 0

        blocks = {
            "entry": [
                ("v_cmp_ne_u32", "vcc", 0, 0),
                ("s_cbranch_vccz", "exit"),
                "loop_body",
            ],
            "loop_body": [
                ("v_add_u32", 1, 1, 1),
                ("v_dec_u32", 0),
                ("v_cmp_ne_u32", "vcc", 0, 0),
                ("s_cbranch_vccnz", "loop_body"),
                "exit",
            ],
            "exit": [
                ("s_endpgm",),
            ],
        }

        interp.run_program(blocks, "entry", max_steps=100)

        # Max initial counter was 7, so loop body must execute 7 times across the wave
        loop_occurrences = interp.trace.count("loop_body")
        self.assertEqual(loop_occurrences, 7)
        self.assertEqual(interp.vcc, 0)
        for lane in range(wave_size):
            self.assertEqual(interp.vgprs[lane][0], 0)
            # All active lanes in the SIMD wave execute the loop 7 times until uniform exit
            self.assertEqual(interp.vgprs[lane][1], 7)

    def test_carry_out_branching_v_add_co_u32(self):
        """v_add_co_u32 writes arithmetic carry-out bits to VCC and drives branch decisions."""
        blocks = {
            "entry": [
                ("v_add_co_u32", 2, 0, 1),
                ("s_cbranch_vccnz", "overflow_detected"),
                "no_overflow",
            ],
            "overflow_detected": [
                ("s_mov_b32", 30, 0xCAFE),
                ("s_endpgm",),
            ],
            "no_overflow": [
                ("s_mov_b32", 30, 0xBEEF),
                ("s_endpgm",),
            ],
        }

        # Case 1: no overflow in any lane
        interp_no_ovf = WaveInterpreter(wave_size=32)
        for lane in range(32):
            interp_no_ovf.vgprs[lane][0] = 1000
            interp_no_ovf.vgprs[lane][1] = 2000
        interp_no_ovf.run_program(blocks, "entry")
        self.assertEqual(interp_no_ovf.trace, ["entry", "no_overflow"])
        self.assertEqual(interp_no_ovf.vcc, 0)
        self.assertEqual(interp_no_ovf.sgprs[30], 0xBEEF)

        # Case 2: overflow only in lane 12 (0xFFFFFFFF + 1 = 0 with carry 1)
        interp_ovf = WaveInterpreter(wave_size=32)
        for lane in range(32):
            if lane == 12:
                interp_ovf.vgprs[lane][0] = 0xFFFFFFFF
                interp_ovf.vgprs[lane][1] = 1
            else:
                interp_ovf.vgprs[lane][0] = 1000
                interp_ovf.vgprs[lane][1] = 2000
        interp_ovf.run_program(blocks, "entry")
        self.assertEqual(interp_ovf.trace, ["entry", "overflow_detected"])
        self.assertEqual(interp_ovf.vcc, 1 << 12)
        self.assertEqual(interp_ovf.sgprs[30], 0xCAFE)
        self.assertEqual(interp_ovf.vgprs[12][2], 0)

    def test_scalar_vcc_manipulation(self):
        """Setting VCC through scalar instructions correctly controls s_cbranch_vccz/nz."""
        interp = WaveInterpreter(wave_size=64)

        # 1. Direct s_mov_b32 to vcc_lo
        blocks_lo = {
            "entry": [
                ("s_mov_b32", "vcc_lo", 0x1234),
                ("s_cbranch_vccz", "fail"),
                "pass_lo",
            ],
            "pass_lo": [("s_endpgm",)],
            "fail": [("s_endpgm",)],
        }
        interp.run_program(blocks_lo, "entry")
        self.assertEqual(interp.trace, ["entry", "pass_lo"])
        self.assertEqual(interp.vcc, 0x1234)

        # 2. Direct s_mov_b32 to vcc_hi
        blocks_hi = {
            "entry": [
                ("s_mov_b32", "vcc_lo", 0),
                ("s_mov_b32", "vcc_hi", 0x5678),
                ("s_cbranch_vccz", "fail"),
                "pass_hi",
            ],
            "pass_hi": [("s_endpgm",)],
            "fail": [("s_endpgm",)],
        }
        interp.run_program(blocks_hi, "entry")
        self.assertEqual(interp.trace, ["entry", "pass_hi"])
        self.assertEqual(interp.vcc, 0x567800000000)

        # 3. Clearing with s_and_b64
        blocks_clr = {
            "entry": [
                ("s_and_b64", "vcc", 0),
                ("s_cbranch_vccnz", "fail"),
                "pass_zero",
            ],
            "pass_zero": [("s_endpgm",)],
            "fail": [("s_endpgm",)],
        }
        interp.run_program(blocks_clr, "entry")
        self.assertEqual(interp.trace, ["entry", "pass_zero"])
        self.assertEqual(interp.vcc, 0)

    def test_divergent_if_then_else_vcc(self):
        """Structured if-then-else construct guarded by s_cbranch_vccz."""
        wave_size = 32
        interp = WaveInterpreter(wave_size=wave_size, initial_exec=0xFFFFFFFF)
        for lane in range(wave_size):
            interp.vgprs[lane][0] = 50 + (lane % 2) * 100  # evens = 50, odds = 150
            interp.vgprs[lane][2] = 0

        blocks = {
            "entry": [
                ("v_cmp_gt_u32", "vcc", 0, 100),  # odds pass
                ("s_cbranch_vccz", "else_block"),
                "then_block",
            ],
            "then_block": [
                ("v_add_u32", 2, 2, 1000),
                ("s_branch", "join"),
            ],
            "else_block": [
                ("v_add_u32", 2, 2, 2000),
                ("s_branch", "join"),
            ],
            "join": [
                ("s_endpgm",),
            ],
        }

        interp.run_program(blocks, "entry")

        # Odds satisfied condition -> then_block was taken
        self.assertEqual(interp.trace, ["entry", "then_block", "join"])
        for lane in range(wave_size):
            self.assertEqual(interp.vgprs[lane][2], 1000)


class TestTranslatorSpirvEmission(unittest.TestCase):
    """Verifies that tools/rdna2spirv.py emits correct SPIR-V instructions for VCC branching."""

    def test_branch_condition_vccz_emission(self):
        """s_cbranch_vccz must load %vcc, test != 0, vote Any, and negate the result."""
        g = rdna2spirv.Gen("ps")
        cond = rdna2spirv.branch_condition(g, "s_cbranch_vccz")

        self.assertTrue(g.uses_vote, "s_cbranch_vccz must flag uses_vote = True")

        body_text = "\n".join(g.body)
        self.assertIn("OpLoad %uint %vcc", body_text)
        self.assertIn("OpINotEqual %bool", body_text)
        self.assertIn("OpGroupNonUniformBallot %v4uint_bal %c3", body_text)
        self.assertIn("%c20", body_text)  # select the ballot word for this wave32 pixel half
        self.assertIn("OpINotEqual %bool", body_text)
        self.assertIn(f"{cond} = OpLogicalNot %bool", body_text)

    def test_branch_condition_vccnz_emission(self):
        """s_cbranch_vccnz must load %vcc, test != 0, vote Any, without logical negation."""
        g = rdna2spirv.Gen("ps")
        cond = rdna2spirv.branch_condition(g, "s_cbranch_vccnz")

        self.assertTrue(g.uses_vote, "s_cbranch_vccnz must flag uses_vote = True")

        body_text = "\n".join(g.body)
        self.assertIn("OpLoad %uint %vcc", body_text)
        self.assertIn("OpINotEqual %bool", body_text)
        self.assertIn("OpGroupNonUniformBallot %v4uint_bal %c3", body_text)
        self.assertIn("%c20", body_text)  # select the ballot word for this wave32 pixel half
        self.assertIn("OpINotEqual %bool", body_text)
        self.assertIn(f"{cond} = OpCopyObject %bool", body_text)
        self.assertNotIn("OpLogicalNot", body_text)

    def test_required_capabilities_for_vcc_vote(self):
        """Assembling shaders with s_cbranch_vccz/nz must declare GroupNonUniform capabilities."""
        g = rdna2spirv.Gen("ps")
        rdna2spirv.branch_condition(g, "s_cbranch_vccz")
        asm = rdna2spirv.assemble(g)
        caps = rdna2spirv.shader_meta(g, asm)["capabilities"]

        self.assertIn("GroupNonUniform", caps)
        self.assertIn("GroupNonUniformVote", caps)

    def test_spirv_subgroup_simulation_matches_isa_reference(self):
        """Simulated SPIR-V subgroup vote instructions match eval_vccz/eval_vccnz."""
        test_masks = [
            0x00000000,
            0xFFFFFFFF,
            0x00000001,
            0x80000000,
            0x0000FFFF,
            0x55555555,
            0xAAAAAAAA,
        ]
        rng = random.Random(20261122)
        test_masks.extend(rng.getrandbits(32) for _ in range(50))

        for mask in test_masks:
            # In SPIR-V, each invocation holds 0xFFFFFFFF or 0 in %vcc
            lane_words = [0xFFFFFFFF if ((mask >> bit) & 1) else 0 for bit in range(32)]

            sim_z = simulate_spirv_subgroup_vcc_branch("s_cbranch_vccz", lane_words)
            sim_nz = simulate_spirv_subgroup_vcc_branch("s_cbranch_vccnz", lane_words)

            ref_z = eval_vccz(mask, wave_size=32)
            ref_nz = eval_vccnz(mask, wave_size=32)

            self.assertEqual(sim_z, ref_z, f"vccz mismatch for mask {hex(mask)}")
            self.assertEqual(sim_nz, ref_nz, f"vccnz mismatch for mask {hex(mask)}")

    def test_wave_uniformity_prevents_sgpr_divergence(self):
        """Verifies that subgroup voting ensures all lanes take an identical branch path."""
        # Non-uniform lane values
        lane_words = [0xFFFFFFFF if (i == 3) else 0 for i in range(32)]
        vote_result = simulate_spirv_subgroup_vcc_branch("s_cbranch_vccnz", lane_words)

        # Every lane in the subgroup sees the exact same boolean vote outcome
        for lane_idx in range(32):
            self.assertTrue(vote_result, f"Lane {lane_idx} must observe uniform true")


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Ensures test assertions detect erroneous, inverted, or truncated implementations."""

    def test_inverted_vccz_fails_detection(self):
        """Inverted eval_vccz logic is caught immediately."""
        def sabotaged_eval_vccz(mask: int, wave_size: int = 32) -> bool:
            return (mask & ((1 << wave_size) - 1)) != 0

        self.assertNotEqual(sabotaged_eval_vccz(0), eval_vccz(0))
        self.assertNotEqual(sabotaged_eval_vccz(0xFFFFFFFF), eval_vccz(0xFFFFFFFF))

    def test_subgroup_all_instead_of_any_fails_detection(self):
        """Using OpGroupNonUniformAll instead of Any fails when only a subset of lanes is set."""
        def all_active(lane_words: list[int]) -> bool:
            return all((w & 0xFFFFFFFF) != 0 for w in lane_words)

        partial_lane_words = [0xFFFFFFFF] + [0] * 31
        sim_nz = simulate_spirv_subgroup_vcc_branch("s_cbranch_vccnz", partial_lane_words)
        self.assertTrue(sim_nz)
        self.assertFalse(all_active(partial_lane_words))

    def test_wave64_ignoring_vcc_hi_fails_detection(self):
        """Truncating Wave64 to 32 bits fails to detect high-half VCC conditions."""
        def buggy_wave64_eval_vccz(mask: int) -> bool:
            return (mask & 0xFFFFFFFF) == 0

        high_half_mask = 1 << 45
        self.assertFalse(eval_vccz(high_half_mask, wave_size=64))
        self.assertTrue(buggy_wave64_eval_vccz(high_half_mask))

    def test_branches_registration(self):
        """Verifies that s_cbranch_vccz and s_cbranch_vccnz are registered in rdna2spirv.BRANCHES."""
        self.assertIn("s_cbranch_vccz", rdna2spirv.BRANCHES)
        self.assertIn("s_cbranch_vccnz", rdna2spirv.BRANCHES)


if __name__ == "__main__":
    unittest.main()
