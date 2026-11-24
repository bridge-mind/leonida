"""Unit-level checks for tools/wave64_emulate_opt.py: the CFG analyses behind --wave64-emulate.

This complements the end-to-end workgroup suite at tests/semantic/test_wave64_emulate_opt.py, which
executes whole translated modules and caught nothing about the per-opcode read/write tables; these
are hand-built block graphs of instruction dicts shaped the way `rdna2spirv` passes them in
(`{"op": ..., "args": [...]}`, operands as `{"k": "sgpr"/"special", ...}`). The invariants under
test come from the module's own docstrings:

- `scc_live_outs` is conservative: an SCC write stays in the answer when a read can reach it before
  an unconditional overwrite, and drops when the value is killed first;
- `_scalar_liveness` is exact conservative backward liveness per block;
- `_postdominators` follows Cooper-Harvey-Kennedy on the reversed CFG with a virtual exit;
- the pc-guard unwrapping only inlines guards whose `%pc` set is exactly `{g}`, and loop bodies
  prove `%pc != h` on exit;
- `_region_has_cycle` and `_tree_has_loop` see cycles and loop nodes the way their callers need.

`find_local_exec_folds` and `plan_structured_emulation` consume lane-mask structures the
translator builds; they are exercised end to end by the semantic suite running under
`--wave64-emulate` (tests/semantic/test_wave64_halves.py), not duplicated here.

    python -I tests/test_wave64_emulate_opt_cfg.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import wave64_emulate_opt as w  # noqa: E402


def named(blocks, live):
    """(block index, instruction index) of every member — keeps the failure output readable."""
    return sorted((bi, j) for bi, b in enumerate(blocks) for j, i in enumerate(b) if id(i) in live)


def sgpr(r, n=1):
    return {"k": "sgpr", "r": r, "n": n}


def special(name):
    return {"k": "special", "t": name}


def ins(op, *args):
    return {"op": op, "args": list(args)}


def blocks_of(*block_lists):
    return [list(b) for b in block_lists]


class OperandHelpersTest(unittest.TestCase):
    def test_base_op_strips_only_the_known_encodings(self):
        self.assertEqual(w.base_op("s_and_b64_e64"), "s_and_b64")
        self.assertEqual(w.base_op("v_add_f32_sdwa"), "v_add_f32")
        self.assertEqual(w.base_op("v_cndmask_b32_dpp"), "v_cndmask_b32")
        self.assertEqual(w.base_op("s_and_b64"), "s_and_b64")
        self.assertEqual(w.base_op("s_load_dwordx2_soff"), "s_load_dwordx2_soff")

    def test_operand_scalar_mask_names_bits(self):
        self.assertEqual(w._operand_scalar_mask(sgpr(4, 3)), 0b1110000)
        self.assertEqual(w._operand_scalar_mask(sgpr(126, 4)), 3 << 126)    # s126, s127; past-128 names dropped
        self.assertEqual(w._operand_scalar_mask(special("vcc")), 1 << w._BIT_VCC)
        self.assertEqual(w._operand_scalar_mask(special("scc")), 1 << w._BIT_SCC)
        self.assertEqual(w._operand_scalar_mask(special("m0")), 1 << w._BIT_M0)
        self.assertEqual(w._operand_scalar_mask({"k": "vgpr", "r": 0}), 0)
        self.assertEqual(w._operand_scalar_mask({"k": "imm", "v": 5}), 0)


class SccClassificationTest(unittest.TestCase):
    def test_reads_scc(self):
        self.assertTrue(w.reads_scc(ins("s_cbranch_scc1", sgpr(0))))
        self.assertTrue(w.reads_scc(ins("s_cselect_b32", sgpr(0), sgpr(1), special("scc"))))
        self.assertTrue(w.reads_scc(ins("s_addc_u32", sgpr(0), sgpr(1), sgpr(2))))
        self.assertTrue(w.reads_scc(ins("s_mov_b32", sgpr(0), special("scc"))))
        self.assertFalse(w.reads_scc(ins("s_and_b64", special("exec"), special("vcc"), sgpr(1, 2))))
        self.assertFalse(w.reads_scc(ins("s_mov_b32", sgpr(0), sgpr(2))))

    def test_writes_scc(self):
        self.assertTrue(w.writes_scc(ins("s_and_b64", sgpr(0, 2), sgpr(2, 2), sgpr(4, 2))))
        self.assertTrue(w.writes_scc(ins("s_and_saveexec_b64", sgpr(0, 2), sgpr(2, 2))))  # saveexec writes scc
        self.assertTrue(w.writes_scc(ins("s_add_u32", sgpr(0), sgpr(1), {"k": "imm", "v": 5})))              # carries write scc
        self.assertTrue(w.writes_scc(ins("s_add_u32_e64", sgpr(0), sgpr(1), 5)))
        self.assertFalse(w.writes_scc(ins("s_mov_b32", sgpr(0), sgpr(1))))
        self.assertFalse(w.writes_scc(ins("v_cmp_lt_i32", special("vcc"), sgpr(0), sgpr(1))))


class ActiveInsnsTest(unittest.TestCase):
    def test_stops_at_the_first_terminator(self):
        blk = [ins("s_mov_b32", sgpr(0), sgpr(1)),
               ins("s_cbranch_execz", 7),
               ins("s_mov_b32", sgpr(2), sgpr(3)),   # unreachable side of the branch in the same dict list
               ins("s_endpgm")]
        active = w._active_insns(blk)
        self.assertEqual([i["op"] for i in active], ["s_mov_b32", "s_cbranch_execz"])

    def test_without_a_terminator_everything_is_active(self):
        blk = [ins("s_mov_b32", sgpr(0), sgpr(1)), ins("s_nop")]
        self.assertEqual(len(w._active_insns(blk)), 2)


class SccLiveOutsTest(unittest.TestCase):
    # The answer is the set of instructions whose SCC result is still needed AFTER they run
    # (a read's own demand travels to the writer that survives until a consumer). Blocks are cut
    # at their branch terminator, so instructions after the branch in the listing are out of scope.

    def test_the_surviving_write_of_a_linear_chain_is_live_out(self):
        # B0's write is unconditionally overwritten by B1 before B2 reads it: only B1's write is
        # in the answer. The read itself adds nothing (nothing needs SCC after B2's branch).
        blocks = blocks_of(
            [ins("s_and_b64", sgpr(0, 2), sgpr(1, 2), sgpr(2, 2))],
            [ins("s_or_b64", sgpr(0, 2), sgpr(1, 2), sgpr(2, 2))],
            [ins("s_cbranch_scc1", 3)],
        )
        live = w.scc_live_outs(blocks, succ=[[1], [2], []])
        self.assertEqual(named(blocks, live), [(1, 0)])

    def test_a_terminator_reading_this_blocks_write_needs_no_upstream(self):
        # B1 writes SCC (s_not) then branches on it: the read consumes THIS block's value, so no
        # upstream writer is demanded. The block-leading write is live-out.
        blocks = blocks_of(
            [ins("s_and_b64", sgpr(0, 2), sgpr(1, 2), sgpr(2, 2))],
            [ins("s_not_b64", sgpr(0, 2), sgpr(1, 2)), ins("s_cbranch_scc1", 2)],
        )
        live = w.scc_live_outs(blocks, succ=[[1], [1]])
        self.assertEqual(named(blocks, live), [(1, 0)])

    def test_a_self_loop_branch_keeps_its_block_read_live_out(self):
        # B1 branches on SCC inside a self loop and nobody writes it: the write before it (B0)
        # and the branch itself (its result is needed at the loop exit) are live out.
        blocks = blocks_of(
            [ins("s_and_b64", sgpr(0, 2), sgpr(1, 2), sgpr(2, 2))],
            [ins("s_cbranch_scc1", 2)],
        )
        live = w.scc_live_outs(blocks, succ=[[1], [1]])
        self.assertEqual(named(blocks, live), [(0, 0), (1, 0)])

    def test_a_write_following_the_branch_in_the_listing_is_not_in_scope(self):
        # The s_or after the branch belongs to the not-taken path and _active_insns cut it off.
        blocks = blocks_of(
            [ins("s_and_b64", sgpr(0, 2), sgpr(1, 2), sgpr(2, 2))],
            [ins("s_cbranch_scc1", 2),
             ins("s_or_b64", sgpr(0, 2), sgpr(1, 2), sgpr(2, 2))],
        )
        live = w.scc_live_outs(blocks, succ=[[1], [1]])
        self.assertEqual(named(blocks, live), [(0, 0), (1, 0)])

    def test_no_blocks_at_all(self):
        self.assertEqual(w.scc_live_outs([], succ=[]), set())


class ScalarLivenessTest(unittest.TestCase):
    def test_a_dead_write_is_not_live_in(self):
        # B0: s10 = s11 then s10 = 0; B1: reads s11. Live-in at B0 is s11, not s10.
        blocks = blocks_of(
            [ins("s_mov_b32", sgpr(10), sgpr(11)), ins("s_mov_b32", sgpr(10), {"k": "imm", "v": 0})],
            [ins("s_mov_b32", sgpr(0), sgpr(11))],
        )
        live_in = w._scalar_liveness([w._active_insns(b) for b in blocks], succ=[[1], []])
        bit11 = 1 << 11
        self.assertEqual(live_in[0] & bit11, bit11)
        self.assertEqual(live_in[0] & (1 << 10), 0)

    def test_a_read_reaching_a_loop_head_stays_live_across_the_back_edge(self):
        blocks = blocks_of(
            [ins("s_mov_b32", sgpr(20), {"k": "imm", "v": 3})],            # write only
            [ins("s_mov_b32", sgpr(0), sgpr(20))],      # loop body reads s20
        )
        live_in = w._scalar_liveness([w._active_insns(b) for b in blocks], succ=[[1], [1]])
        self.assertEqual(live_in[0] & (1 << 20), 0)     # B0's own write satisfies the read

    def test_scc_live_through_an_unwritten_successor_block(self):
        # B0 does not touch SCC; B1 only branches on it (nothing ever writes it): the read demand
        # flows across both blocks.
        blocks = blocks_of(
            [ins("s_mov_b32", sgpr(0), sgpr(1))],
            [ins("s_cbranch_scc1", 1)],
        )
        live_in = w._scalar_liveness([w._active_insns(b) for b in blocks], succ=[[1], [1]])
        self.assertEqual(live_in[0] & (1 << w._BIT_SCC), 1 << w._BIT_SCC)
        self.assertEqual(live_in[0] & (1 << w._BIT_VCC), 0)


class PostdominatorsTest(unittest.TestCase):
    def test_diamond(self):
        #  0 -> {1, 2}; 1, 2 -> 3; 3 -> exit. ipdom(1) == ipdom(2) == 3, ipdom(0) == 3.
        succ = [[1, 2], [3], [3], []]
        ipdom = w._postdominators(succ)
        self.assertEqual(ipdom[0], 3)
        self.assertEqual(ipdom[1], 3)
        self.assertEqual(ipdom[2], 3)
        self.assertEqual(ipdom[3], 4)                    # the virtual exit block is n

    def test_a_straight_line_chains_to_the_end(self):
        succ = [[1], [2], []]
        ipdom = w._postdominators(succ)
        self.assertEqual([ipdom[0], ipdom[1], ipdom[2]], [1, 2, 3])

    def test_a_loop_head_postdominates_nothing_about_the_exit_side(self):
        # 0 -> {1, 2}; 1 -> {1, 2} (back edge); 2 -> exit.
        ipdom = w._postdominators([[1, 2], [1, 2], []])
        self.assertEqual(ipdom[2], 3)
        self.assertEqual(ipdom[1], 2)


class TreeAndRegionGuardTest(unittest.TestCase):
    def test_loop_inside_a_guard_is_found(self):
        # the recursion must descend into guard bodies: no loop at the top level here.
        self.assertTrue(w._tree_has_loop([("guard", 3, [("loop", 4, [("block", 0)])])]))
        self.assertFalse(w._tree_has_loop([("guard", 3, [("block", 0)])]))

    def test_guard_over_an_ambiguous_pc_containing_the_target_stays_wrapped(self):
        # the seen set holds both 3 and another pc: 3 in it is not a proof of 3 alone, so the
        # guard stays (an in-set membership check instead of equality would wrongly inline it).
        tree = [("block", 0), ("guard", 3, [("block", 1), ("block", 3)])]
        out_pcs = {0: {1, 3}, 1: {3}, 3: {3}}
        self.assertEqual(w._unwrap_known_pc_guards(tree, out_pcs), tree)

    def test_tree_has_loop_finds_loop_nodes_through_guards(self):
        self.assertFalse(w._tree_has_loop([]))
        self.assertFalse(w._tree_has_loop([("block", 0)]))
        tree = [("guard", 3, [("block", 1)]), ("loop", 4, [("block", 2)])]
        self.assertTrue(w._tree_has_loop(tree))
        self.assertFalse(w._tree_has_loop([("guard", 3, [("block", 1)])]))

    def test_region_cycle_check(self):
        self.assertFalse(w._region_has_cycle({0, 1, 2}, [[1], [2], []]))
        self.assertTrue(w._region_has_cycle({0, 1, 2}, [[1], [2], [0]]))
        # an edge leaving the region must not create a phantom cycle
        self.assertFalse(w._region_has_cycle({0, 1}, [[2], [0]]))

    def test_guard_unwraps_only_when_pc_is_known_to_be_g(self):
        # entry proves %pc == 3 before the guard, so the guard dissolves into its body.
        tree = [("block", 0), ("guard", 3, [("block", 1), ("block", 3)])]
        out_pcs = {0: {3}, 1: {3}, 3: {3}}
        out = w._unwrap_known_pc_guards(tree, out_pcs)
        self.assertEqual(out, [("block", 0), ("block", 1), ("block", 3)])

    def test_guard_over_an_unknown_pc_is_kept(self):
        # entry_pc 0 is not provably g != skip, so the guard stays and the body keeps its own frame.
        tree = [("guard", 3, [("block", 1), ("block", 3)])]
        out_pcs = {1: {1, 3}, 3: {3}}
        out = w._unwrap_known_pc_guards(tree, out_pcs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "guard")
        _, g, body = out[0]
        self.assertEqual(g, 3)
        self.assertEqual(body, [("block", 1), ("block", 3)])

    def test_guard_unwraps_when_entry_pc_proves_it(self):
        tree = [("guard", 3, [("block", 1), ("block", 3)])]
        out_pcs = {1: {3}, 3: {3}}
        out = w._unwrap_known_pc_guards(tree, out_pcs, entry_pc=3)
        self.assertEqual(out, [("block", 1), ("block", 3)])

    def test_guard_after_a_loop_stays_wrapped(self):
        # After the loop body's block sets %pc, the unwrap's seen-set here has already dropped the
        # header value, and a guard with no proof block in front of it is not inlined.
        tree = [("block", 0), ("loop", 2, [("block", 1)]), ("guard", 2, [("block", 2)])]
        out_pcs = {0: {0}, 1: {2}, 2: {2}}
        out = w._unwrap_known_pc_guards(tree, out_pcs)
        self.assertEqual(out, [("block", 0), ("loop", 2, [("block", 1)]), ("guard", 2, [("block", 2)])])

    def test_loop_shape_is_kept_and_a_guard_needs_proof_in_front_of_it(self):
        # The tree survives unwrapping unchanged when no guard's %pc set is exactly {g}: the
        # trailing guard here sees an empty seen-set (the loop body block reset it), so it stays.
        tree = [("block", 0), ("loop", 2, [("block", 1)]), ("block", 2)]
        out_pcs = {0: {0}, 1: {2}, 2: {2}}
        out = w._unwrap_known_pc_guards(tree, out_pcs)
        kinds = [n[0] for n in out]
        self.assertEqual(kinds, ["block", "loop", "block"])
        guard_tree = [("guard", 2, [("block", 1)])]
        out_pcs = {1: {2}, 2: {2}}          # %pc at entry_pc 0 is not provably 2, so the guard stays
        out = w._unwrap_known_pc_guards(guard_tree, out_pcs)
        self.assertEqual(out, [("guard", 2, [("block", 1)])])


if __name__ == "__main__":
    unittest.main()
