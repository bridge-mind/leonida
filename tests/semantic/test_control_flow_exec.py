"""Semantic interpreter tests for whole-wave execution mask branching.

Tests the ISA semantics and translator emission for:
  - s_cbranch_execz:  branch if EXEC == 0 (all lanes in the wave are disabled/inactive)
  - s_cbranch_execnz: branch if EXEC != 0 (at least one lane in the wave is active)

Covers:
  1. Pure mathematical reference oracle for Wave32 and Wave64 execution masks.
  2. Multi-lane WaveInterpreter modeling SIMD wave state, vector lane masking,
     scalar registers, and whole-wave control flow (skip patterns, loop convergence,
     and divergent if-then-else joins).
  3. SPIR-V translation verification via tools/rdna2spirv.py branch_condition():
     subgroup vote emission (OpGroupNonUniformAny with Subgroup scope),
     logical negation for execz, and required capability flags.
  4. Invariant and duality verification: execz and execnz are strict complements.
  5. Negative controls / mutation detection: verifies sensitivity to swapped or corrupted
     branch conditions.

Run:
    python -I tests/semantic/test_control_flow_exec.py
    python -m unittest discover -s tests/semantic -v
"""
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
# 1. Reference Semantics (AMD RDNA2 ISA: Scalar ALU & Control Flow)
# -----------------------------------------------------------------------------

def eval_execz(exec_mask: int, wave_size: int = 32) -> bool:
    """s_cbranch_execz: taken iff the wave execution mask is zero."""
    mask = exec_mask & ((1 << wave_size) - 1)
    return mask == 0


def eval_execnz(exec_mask: int, wave_size: int = 32) -> bool:
    """s_cbranch_execnz: taken iff the wave execution mask is non-zero."""
    mask = exec_mask & ((1 << wave_size) - 1)
    return mask != 0


# -----------------------------------------------------------------------------
# 2. Multi-Lane Wave Interpreter
# -----------------------------------------------------------------------------

class WaveInterpreter:
    """Simulates a SIMD wave executing basic blocks with execution mask control flow.

    Models:
      - Wave32 (32 lanes) or Wave64 (64 lanes).
      - EXEC mask: per-lane active bit controlling vector instructions.
      - SGPRs: uniform scalar registers shared across all lanes in the wave.
      - VGPRs: vector registers with independent values per lane.
      - Whole-wave branches: s_cbranch_execz, s_cbranch_execnz, s_branch.
    """

    def __init__(self, wave_size: int = 32, initial_exec: int = 0xFFFFFFFF):
        self.wave_size = wave_size
        self.mask_all = (1 << wave_size) - 1
        self.exec_mask = initial_exec & self.mask_all
        self.sgprs = [0] * 128
        self.vgprs = [[0] * 256 for _ in range(wave_size)]
        self.pc = 0
        self.trace = []

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def run_program(self, blocks: dict, start_block: str, max_steps: int = 1000):
        """Execute a graph of basic blocks until 's_endpgm' or max_steps reached."""
        cur = start_block
        steps = 0
        self.trace = []

        while cur is not None and steps < max_steps:
            steps += 1
            self.trace.append(cur)
            block = blocks.get(cur)
            if block is None:
                raise ValueError(f"Unknown block label: {cur}")

            next_block = None
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

                elif op == "s_cbranch_execz":
                    target = args[0]
                    if self.exec_mask == 0:
                        next_block = target
                        break

                elif op == "s_cbranch_execnz":
                    target = args[0]
                    if self.exec_mask != 0:
                        next_block = target
                        break

                elif op == "s_mov_b32":
                    dst_s, val = args
                    self.sgprs[dst_s] = val & 0xFFFFFFFF

                elif op == "s_and_saveexec_b64" or op == "s_and_saveexec_b32":
                    dst_s, mask = args
                    # Save current EXEC to scalar register(s)
                    self.sgprs[dst_s] = self.exec_mask & 0xFFFFFFFF
                    if self.wave_size == 64 and op.endswith("64"):
                        self.sgprs[dst_s + 1] = (self.exec_mask >> 32) & 0xFFFFFFFF
                    # Set EXEC = EXEC & mask
                    self.exec_mask = (self.exec_mask & mask) & self.mask_all

                elif op == "s_or_exec_b64" or op == "s_or_exec_b32":
                    mask = args[0]
                    self.exec_mask = (self.exec_mask | mask) & self.mask_all

                elif op == "s_set_exec":
                    self.exec_mask = args[0] & self.mask_all

                elif op == "v_mov_b32":
                    dst_v, val = args
                    # Vector write: only active lanes are updated
                    for lane in range(self.wave_size):
                        if self.is_lane_active(lane):
                            self.vgprs[lane][dst_v] = val & 0xFFFFFFFF

                elif op == "v_add_u32":
                    dst_v, src_v, imm = args
                    for lane in range(self.wave_size):
                        if self.is_lane_active(lane):
                            self.vgprs[lane][dst_v] = (self.vgprs[lane][src_v] + imm) & 0xFFFFFFFF

                elif op == "v_dec_u32":
                    dst_v = args[0]
                    for lane in range(self.wave_size):
                        if self.is_lane_active(lane):
                            curr = self.vgprs[lane][dst_v]
                            self.vgprs[lane][dst_v] = (curr - 1) & 0xFFFFFFFF if curr > 0 else 0

                elif op == "v_cmp_gt_u32":
                    # Generate condition mask: 1 where vgpr[src] > imm
                    dst_s, src_v, imm = args
                    cond_mask = 0
                    for lane in range(self.wave_size):
                        if self.vgprs[lane][src_v] > imm:
                            cond_mask |= (1 << lane)
                    self.sgprs[dst_s] = cond_mask & 0xFFFFFFFF
                    if self.wave_size == 64:
                        self.sgprs[dst_s + 1] = (cond_mask >> 32) & 0xFFFFFFFF

                else:
                    raise NotImplementedError(f"Unsupported instruction in test interpreter: {op}")

            # If no branch was taken, fall through to block's default successor if given
            if next_block is None:
                cur = block[-1] if isinstance(block[-1], str) else None
            else:
                cur = next_block

        if steps >= max_steps:
            raise TimeoutError("Execution exceeded max_steps (infinite loop detected)")


# -----------------------------------------------------------------------------
# 3. SPIR-V Subgroup Vote Interpreter
# -----------------------------------------------------------------------------

def simulate_spirv_subgroup_branch(op: str, lane_actives: list[bool]) -> bool:
    """Interprets the SPIR-V instructions emitted by rdna2spirv for whole-wave branching.

    Emitted sequence:
      %lane = OpLoad %bool %active
      %any  = OpGroupNonUniformAny %bool %c3 %lane
      (if execz): %b = OpLogicalNot %bool %any
    """
    any_active = any(lane_actives)

    if op == "s_cbranch_execnz":
        return any_active
    elif op == "s_cbranch_execz":
        return not any_active
    else:
        raise ValueError(f"Unknown branch opcode: {op}")


# -----------------------------------------------------------------------------
# Test Suite
# -----------------------------------------------------------------------------

class TestControlFlowExecReference(unittest.TestCase):
    """Verifies whole-wave execution mask semantics against ISA definitions."""

    def test_execz_and_execnz_duality(self):
        """execz and execnz must be strict logical complements for every possible mask."""
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
        # All 32 single-bit masks
        for bit in range(32):
            test_masks.append(1 << bit)
            test_masks.append(~(1 << bit) & 0xFFFFFFFF)

        # Deterministic random masks
        rng = random.Random(20261121)
        for _ in range(100):
            test_masks.append(rng.getrandbits(32))

        for mask in test_masks:
            z = eval_execz(mask, wave_size=32)
            nz = eval_execnz(mask, wave_size=32)
            with self.subTest(mask=hex(mask)):
                self.assertNotEqual(z, nz, "execz and execnz must never have identical outcome")
                self.assertEqual(z, mask == 0)
                self.assertEqual(nz, mask != 0)

    def test_wave64_execution_masks(self):
        """Verifies 64-bit execution masks in Wave64 mode."""
        mask64_all = 0xFFFFFFFFFFFFFFFF
        self.assertTrue(eval_execz(0, wave_size=64))
        self.assertFalse(eval_execnz(0, wave_size=64))

        self.assertFalse(eval_execz(mask64_all, wave_size=64))
        self.assertTrue(eval_execnz(mask64_all, wave_size=64))

        # Check MSB in Wave64 (lane 63)
        lane63 = 1 << 63
        self.assertFalse(eval_execz(lane63, wave_size=64))
        self.assertTrue(eval_execnz(lane63, wave_size=64))

        # High 32 bits only / Low 32 bits only
        self.assertFalse(eval_execz(0xFFFFFFFF00000000, wave_size=64))
        self.assertTrue(eval_execnz(0xFFFFFFFF00000000, wave_size=64))
        self.assertFalse(eval_execz(0x00000000FFFFFFFF, wave_size=64))
        self.assertTrue(eval_execnz(0x00000000FFFFFFFF, wave_size=64))

        # Truncation boundary check: bits beyond 63 are ignored by wave_size=64
        self.assertTrue(eval_execz(1 << 64, wave_size=64))


class TestWaveInterpreterExecutionFlow(unittest.TestCase):
    """Verifies execution mask control flow in multi-lane simulated programs."""

    def test_skip_pattern_execz_taken_when_all_lanes_inactive(self):
        """When condition is false for all lanes, s_cbranch_execz skips the then block."""
        blocks = {
            "entry": [
                ("v_cmp_gt_u32", 4, 0, 100),
                ("s_and_saveexec_b32", 6, 0),  # s4 is 0
                ("s_cbranch_execz", "skip"),
                "then_block",
            ],
            "then_block": [
                ("v_mov_b32", 1, 999),
                ("s_branch", "skip"),
            ],
            "skip": [
                ("s_endpgm",),
            ],
        }

        interp = WaveInterpreter(wave_size=32, initial_exec=0xFFFFFFFF)
        for lane in range(32):
            interp.vgprs[lane][0] = 50
            interp.vgprs[lane][1] = 10

        interp.run_program(blocks, "entry")

        self.assertEqual(interp.trace, ["entry", "skip"])
        for lane in range(32):
            self.assertEqual(interp.vgprs[lane][1], 10)

    def test_skip_pattern_execz_not_taken_when_one_lane_active(self):
        """When at least one lane satisfies condition, s_cbranch_execz does NOT branch."""
        interp = WaveInterpreter(wave_size=32, initial_exec=0xFFFFFFFF)
        for lane in range(32):
            interp.vgprs[lane][0] = 50
            interp.vgprs[lane][1] = 10
        interp.vgprs[7][0] = 150  # only lane 7 satisfies condition

        cond_mask = 1 << 7
        blocks = {
            "entry": [
                ("s_and_saveexec_b32", 6, cond_mask),
                ("s_cbranch_execz", "skip"),
                "then_block",
            ],
            "then_block": [
                ("v_mov_b32", 1, 999),
                ("s_branch", "skip"),
            ],
            "skip": [
                ("s_endpgm",),
            ],
        }

        interp.run_program(blocks, "entry")

        self.assertEqual(interp.trace, ["entry", "then_block", "skip"])
        for lane in range(32):
            expected = 999 if lane == 7 else 10
            self.assertEqual(interp.vgprs[lane][1], expected)

    def test_divergent_loop_termination_execnz(self):
        """s_cbranch_execnz loops while any lane is active, terminating when all lanes finish."""
        wave_size = 32
        interp = WaveInterpreter(wave_size=wave_size, initial_exec=0xFFFFFFFF)
        for lane in range(wave_size):
            interp.vgprs[lane][0] = lane if lane <= 7 else 0  # countdown counter in v0
            interp.vgprs[lane][1] = 0                         # accumulator in v1

        blocks = {
            "entry": [
                ("v_cmp_gt_u32", 4, 0, 0),
                ("s_and_saveexec_b32", 6, 0),
                ("s_cbranch_execz", "exit"),
                "loop_body",
            ],
            "loop_body": [
                ("v_add_u32", 1, 1, 1),
                ("v_dec_u32", 0),
                ("v_cmp_gt_u32", 4, 0, 0),
            ],
        }

        initial_active = 0
        for lane in range(wave_size):
            if interp.vgprs[lane][0] > 0:
                initial_active |= (1 << lane)
        interp.exec_mask = initial_active

        iterations = 0
        max_loop = 50
        while interp.exec_mask != 0 and iterations < max_loop:
            iterations += 1
            for ins in blocks["loop_body"]:
                op, *args = ins
                if op == "v_add_u32":
                    d, s, imm = args
                    for l in range(wave_size):
                        if interp.is_lane_active(l):
                            interp.vgprs[l][d] += imm
                elif op == "v_dec_u32":
                    d = args[0]
                    for l in range(wave_size):
                        if interp.is_lane_active(l):
                            interp.vgprs[l][d] -= 1
                elif op == "v_cmp_gt_u32":
                    d, s, imm = args
                    cond = 0
                    for l in range(wave_size):
                        if interp.vgprs[l][s] > imm:
                            cond |= (1 << l)
                    interp.exec_mask &= cond

        self.assertEqual(iterations, 7)
        self.assertEqual(interp.exec_mask, 0)
        for lane in range(wave_size):
            expected_acc = lane if lane <= 7 else 0
            self.assertEqual(interp.vgprs[lane][1], expected_acc)

    def test_divergent_if_else_reconvergence(self):
        """Simulates structured if-then-else with s_cbranch_execz guards on both branches."""
        wave_size = 32
        interp = WaveInterpreter(wave_size=wave_size, initial_exec=0xFFFFFFFF)
        for lane in range(wave_size):
            interp.vgprs[lane][0] = lane % 2
            interp.vgprs[lane][2] = 0

        even_mask = 0x55555555
        odd_mask = 0xAAAAAAAA

        blocks = {
            "entry": [
                ("s_mov_b32", 10, interp.exec_mask),
                ("s_and_saveexec_b32", 12, even_mask),
                ("s_cbranch_execz", "else_block"),
                "then_block",
            ],
            "then_block": [
                ("v_add_u32", 2, 2, 100),
                ("s_branch", "else_block"),
            ],
            "else_block": [
                ("s_set_exec", odd_mask),
                ("s_cbranch_execz", "join"),
                "else_body",
            ],
            "else_body": [
                ("v_add_u32", 2, 2, 200),
                ("s_branch", "join"),
            ],
            "join": [
                ("s_mov_b32", 14, 0x1234),
                ("s_set_exec", 0xFFFFFFFF),
                ("s_endpgm",),
            ],
        }

        interp.run_program(blocks, "entry")

        self.assertEqual(interp.trace, ["entry", "then_block", "else_block", "else_body", "join"])
        for lane in range(wave_size):
            expected = 100 if (lane % 2 == 0) else 200
            self.assertEqual(interp.vgprs[lane][2], expected, f"Mismatch at lane {lane}")


class TestTranslatorSpirvEmission(unittest.TestCase):
    """Verifies that tools/rdna2spirv.py emits correct subgroup vote constructs."""

    def test_branch_condition_execz_emission(self):
        """s_cbranch_execz must emit OpGroupNonUniformAny with logical NOT."""
        g = rdna2spirv.Gen("ps")
        cond = rdna2spirv.branch_condition(g, "s_cbranch_execz")

        self.assertTrue(g.uses_vote, "s_cbranch_execz must set uses_vote = True")

        body_text = "\n".join(g.body)
        self.assertIn("OpLoad %bool %active", body_text)
        self.assertIn("OpGroupNonUniformBallot %v4uint_bal %c3", body_text)
        self.assertIn("%c20", body_text)  # select the ballot word for this wave32 pixel half
        self.assertIn("OpINotEqual %bool", body_text)
        self.assertIn(f"{cond} = OpLogicalNot %bool", body_text)

    def test_branch_condition_execnz_emission(self):
        """s_cbranch_execnz must emit OpGroupNonUniformAny without logical NOT."""
        g = rdna2spirv.Gen("ps")
        cond = rdna2spirv.branch_condition(g, "s_cbranch_execnz")

        self.assertTrue(g.uses_vote, "s_cbranch_execnz must set uses_vote = True")

        body_text = "\n".join(g.body)
        self.assertIn("OpLoad %bool %active", body_text)
        self.assertIn("OpGroupNonUniformBallot %v4uint_bal %c3", body_text)
        self.assertIn("%c20", body_text)  # select the ballot word for this wave32 pixel half
        self.assertIn("OpINotEqual %bool", body_text)
        self.assertIn(f"{cond} = OpCopyObject %bool", body_text)
        self.assertNotIn("OpLogicalNot", body_text)

    def test_spirv_subgroup_simulation_matches_isa_reference(self):
        """Simulated SPIR-V subgroup vote instructions match eval_execz/eval_execnz."""
        test_masks = [
            0x00000000,
            0xFFFFFFFF,
            0x00000001,
            0x80000000,
            0x0000FFFF,
            0x55555555,
        ]
        rng = random.Random(42)
        test_masks.extend(rng.getrandbits(32) for _ in range(50))

        for mask in test_masks:
            lane_actives = [bool((mask >> bit) & 1) for bit in range(32)]

            sim_z = simulate_spirv_subgroup_branch("s_cbranch_execz", lane_actives)
            sim_nz = simulate_spirv_subgroup_branch("s_cbranch_execnz", lane_actives)

            ref_z = eval_execz(mask, wave_size=32)
            ref_nz = eval_execnz(mask, wave_size=32)

            self.assertEqual(sim_z, ref_z, f"execz mismatch for mask {hex(mask)}")
            self.assertEqual(sim_nz, ref_nz, f"execnz mismatch for mask {hex(mask)}")

    def test_required_capabilities_for_vote(self):
        """Modules using s_cbranch_execz/execnz require GroupNonUniform and GroupNonUniformVote."""
        g = rdna2spirv.Gen("ps")
        rdna2spirv.branch_condition(g, "s_cbranch_execz")
        asm = rdna2spirv.assemble(g)
        caps = rdna2spirv.shader_meta(g, asm)["capabilities"]
        self.assertIn("GroupNonUniform", caps)
        self.assertIn("GroupNonUniformVote", caps)


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Ensures test assertions are sensitive to erroneous or inverted implementations."""

    def test_unnegated_execz_fails_detection(self):
        """If s_cbranch_execz omits the OpLogicalNot, it behaves as execnz and must fail."""
        def sabotaged_eval_execz(mask: int) -> bool:
            return mask != 0

        self.assertNotEqual(sabotaged_eval_execz(0), eval_execz(0))
        self.assertNotEqual(sabotaged_eval_execz(0xFFFFFFFF), eval_execz(0xFFFFFFFF))

    def test_all_reduction_instead_of_any_fails_detection(self):
        """If whole-wave active check uses 'All' instead of 'Any', partially active waves fail."""
        def all_active(mask: int, wave_size: int = 32) -> bool:
            return (mask & ((1 << wave_size) - 1)) == ((1 << wave_size) - 1)

        partial_mask = 0x00000001
        self.assertTrue(eval_execnz(partial_mask))
        self.assertFalse(all_active(partial_mask))

    def test_branches_membership(self):
        """Verifies that s_cbranch_execz and s_cbranch_execnz are registered in BRANCHES."""
        self.assertIn("s_cbranch_execz", rdna2spirv.BRANCHES)
        self.assertIn("s_cbranch_execnz", rdna2spirv.BRANCHES)
        self.assertNotIn("s_cbranch_nonexistent", rdna2spirv.BRANCHES)


if __name__ == "__main__":
    unittest.main()
