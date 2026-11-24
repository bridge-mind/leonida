"""SHADER-10: wave64 vertex/pixel shaders on 32-wide subgroups. Tier classification and partition invariance.

tools/wave64_halves.py decides from the code whether a wave64 VS/PS/GS gives the same results when its wave is cut
into two 32-lane halves (the criterion of docs/design.md, "Wave64 graphics: partition invariance"). Two kinds of
check, both on hand-written listings (no llvm-mc, GPU or game data):

  * Classification: each program is translated, and the tier in its metadata (wave64_graphics) and the reasons
    are asserted. The programs are the idioms of the research note (skip regions, quad masks, waterfall loops,
    wave-aggregated atomics, readfirstlane/readlane, lane counts, lane ids, ...), including the near misses that
    must be tier C.
  * Execution: vertex shaders that use integer arithmetic only run on the strict wave interpreter
    (spirv_wave_interp.py) as one wave of 64 lanes and as two waves of 32 lanes, with vertex index v for global lane
    v (so a lane's data does not depend on how the wave is cut). Tier A and B programs must export the same words
    both ways and the words the header derives; tier C programs must differ in the words the test names, which
    shows the detector is right to refuse them.

The GPU twin of the execution half is runtime/test/shaders/w64_graphics.py (pixel shaders, one subgroup size forced
to 64 and to 32 on the same GPU).
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (sets the tools path)
from support import translate_lines, translate_listing  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402

LITERAL_OPS = ("exp", "buffer_", "global_", "ds_", "image_")
VOP3_ONLY = ("v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32", "v_readlane_b32", "v_writelane_b32", "v_lshl_add_u32",
             "v_cmp_", "v_cndmask_b32_e64", "v_add_nc_u32_e64")


def size(text):
    """Byte size of an instruction for the fake encoding the translator's branch offsets are placed with."""
    op = text.split()[0]
    if op.startswith(LITERAL_OPS) or op.endswith("_e64") or op in VOP3_ONLY[:5] or "0x" in text:
        return 8
    if op.startswith("v_cmp") and not op.endswith("_e32") and "vcc" not in text:
        return 8
    return 4


def listing(name, lines):
    out = f"; {name} synthetic\n"
    for text in lines:
        out += f"\t{text} ; encoding: [{','.join(['0x00'] * size(text))}]\n"
    return out.encode()


def meta_of(name, lines):
    """(tier, reasons, idioms) of a translated wave64 program, from the metadata the translator writes."""
    asm = translate_listing(name, listing(name, lines))
    g = rdna2spirv.translate.last_gen
    w = g.w64g
    return (w["tier"], w["reasons"], w["idioms"]) if w else (None, [], []), asm


PRO = ["s_mov_b64 s[10:11], exec"]           # the program is wave64 (a 64-bit op on EXEC)
LANE_ID = ["v_mbcnt_lo_u32_b32 v10, -1, 0", "v_mbcnt_hi_u32_b32 v10, -1, v10"]   # the lane id, 0..63 in the wave
EPI_VS = ["v_mov_b32_e32 v5, 0", "exp pos0 v5, v5, v5, v5 done", "exp param0 v6, v7, v5, v5", "s_endpgm"]
EPI_PS = ["exp mrt0 v6, v7, v5, v5 done vm", "s_endpgm"]


def vs(body, prologue=PRO):
    return prologue + ["v_mov_b32_e32 v6, 0", "v_mov_b32_e32 v7, 0"] + body + EPI_VS


def ps(body):
    return PRO + ["v_mov_b32_e32 v6, 0", "v_mov_b32_e32 v7, 0", "v_mov_b32_e32 v5, 0"] + body + EPI_PS[:-1] + \
        ["s_endpgm"]


# ---- execution of vertex shaders: 64 lanes against two halves of 32 --------------------------------------------
def run_cut(asm, wave, push=None):
    """Outputs of every lane of a 64-lane program run as 64/wave waves of `wave` lanes (vertex index = global lane)."""
    words = []
    for half in range(64 // wave):
        res = wi.run_wave(asm, wave=wave, push_constants=push,
                          builtins={"VertexIndex": lambda lane, base=half * wave: base + lane,
                                     "InstanceIndex": lambda lane: 0, "BaseVertex": lambda lane: 0,
                                     "BaseInstance": lambda lane: 0})
        words += [o and tuple(sorted(o.items())) for o in res["outputs"]]
    return words


def out_param(words, k=0):
    """Per lane, the words of param `k`'s output as a tuple (None for a lane that wrote nothing)."""
    result = []
    for w in words:
        d = dict(w) if w else {}
        result.append(next((v for name, v in d.items() if name.endswith(f"param{k}")), None))
    return result


# ---- the early exit of a pixel kill ------------------------------------------------------------------------------
KILL = ["s_mov_b64 s[20:21], exec", "v_cmp_gt_u32_e32 vcc, 40, v0", "s_andn2_b64 s[20:21], s[20:21], vcc"]
RUN_WQM = ["s_wqm_b64 s[22:23], s[20:21]", "s_and_b64 exec, exec, s[22:23]"]
EXIT_PS = ["s_mov_b64 exec, s[20:21]", "exp null off, off, off, off done vm", "s_endpgm"]
EXIT_VS = ["s_mov_b64 exec, s[20:21]", "exp pos0 v5, v5, v5, v5 done", "s_endpgm"]


def kill(head, run, exit_, stage="ps", vote="s_cbranch_scc0"):
    """head, a branch to the exit when the live mask is empty, the run side and the epilogue, then the exit."""
    epi = EPI_PS if stage == "ps" else EPI_VS
    tail = run + epi
    hop = sum(size(t) for t in tail) // 4
    return PRO + ["v_mov_b32_e32 v6, 0", "v_mov_b32_e32 v7, 0", "v_mov_b32_e32 v5, 0"] + head + \
        [f"{vote} {hop}"] + tail + exit_


class Classification(unittest.TestCase):
    def tier(self, name, lines, tier, reasons=None, idioms=None, contains=None):
        (t, r, i), _ = meta_of(name, lines)
        self.assertEqual(t, tier, f"{name}: tier {t}, reasons {r}, idioms {i}")
        if reasons is not None:
            self.assertEqual(sorted(r), sorted(reasons), name)
        if contains is not None:
            self.assertTrue(set(contains) <= set(r), f"{name}: reasons {r} lack {contains}")
        if idioms is not None:
            self.assertEqual(sorted(i), sorted(idioms), name)

    # -- tier A
    def test_a_skip_wqm_quad(self):
        self.tier("ps_a_skip", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4",
            "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execz 1", "v_mul_u32_u24_e32 v6, 7, v0",
            "s_or_b64 exec, exec, s[12:13]", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_wqm_b64 s[14:15], vcc",
            "v_cndmask_b32_e64 v7, 0, 1, s[14:15]"]), "A", [], [])

    def test_a_vccz_narrowed(self):
        self.tier("ps_a_vccz", ps([
            "v_cmp_eq_u32_e32 vcc, 2, v0", "s_cbranch_vccz 3", "s_and_saveexec_b64 s[16:17], vcc",
            "v_add_nc_u32_e32 v6, 9, v6", "s_or_b64 exec, exec, s[16:17]"]), "A", [], [])

    def test_a_loop_execnz_no_scalar_state(self):
        self.tier("ps_a_loop", ps([
            "v_and_b32_e32 v4, 3, v0", "v_add_nc_u32_e32 v10, 1, v4", "v_mov_b32_e32 v9, 0",
            "v_add_nc_u32_e32 v9, 1, v9", "v_cmp_lt_u32_e32 vcc, v9, v10", "s_and_b64 exec, exec, vcc",
            "s_cbranch_execnz -4", "s_mov_b64 exec, s[10:11]", "v_mov_b32_e32 v6, v9"]), "A", [], [])

    def test_a_bpermute_swizzle_in_half(self):
        self.tier("ps_a_bperm", ps(["v_lshlrev_b32_e32 v4, 2, v0", "ds_bpermute_b32 v6, v4, v0"]), "A", [], [])

    def test_a_bpermute_lane_id_address(self):
        # the lane id is 0..63 in the wave and 0..31 in a half: equal modulo 32 lanes, which is all a wave64
        # ds_bpermute uses (address bits 2..6, per half), so xor/and/or/add on it and a shift by 2 stay exact
        self.tier("ps_a_bperm_id", ps([*LANE_ID, "v_xor_b32_e32 v11, 1, v10", "v_lshlrev_b32_e32 v11, 2, v11",
                                       "ds_bpermute_b32 v6, v11, v0", "v_and_b32_e32 v12, -4, v10",
                                       "v_or_b32_e32 v12, 2, v12", "v_lshl_add_u32 v12, v12, 2, 0",
                                       "ds_bpermute_b32 v7, v12, v0", "s_waitcnt lgkmcnt(0)"]), "A", [], [])

    def test_a_permute_lane_id_address(self):
        self.tier("ps_a_perm_id", ps([*LANE_ID, "v_add_nc_u32_e32 v11, 5, v10", "v_lshlrev_b32_e32 v11, 2, v11",
                                      "ds_permute_b32 v6, v11, v0", "s_waitcnt lgkmcnt(0)"]), "A", [], [])

    # -- tier B
    def test_b_waterfall(self):
        self.tier("ps_b_wf", ps([
            "v_and_b32_e32 v4, 7, v0", "v_readfirstlane_b32 s20, v4", "v_cmp_eq_u32_e32 vcc, s20, v4",
            "s_and_saveexec_b64 s[12:13], vcc", "s_mul_i32 s21, s20, 17", "v_add_nc_u32_e32 v6, s21, v0",
            "s_xor_b64 exec, exec, s[12:13]", "s_cbranch_execnz -7", "s_mov_b64 exec, s[10:11]"]), "B", [],
            ["waterfall"])

    def test_b_readfirstlane_of_uniform(self):
        self.tier("ps_b_rfl", ps(["v_mov_b32_e32 v5, s4", "v_readfirstlane_b32 s20, v5",
                                  "v_add_nc_u32_e32 v6, s20, v0"]), "B", [], ["readfirstlane-uniform"])

    def test_b_readlane_of_uniform_any_lane(self):
        self.tier("ps_b_rl", ps(["v_mov_b32_e32 v5, s4", "v_readlane_b32 s20, v5, 40",
                                 "v_add_nc_u32_e32 v6, s20, v0"]), "B", [], ["readlane-uniform"])

    def test_b_aggregated_atomic(self):
        self.tier("ps_b_atomic", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_and_saveexec_b64 s[12:13], vcc",
            "s_bcnt1_i32_b64 s6, exec", "v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10",
            "v_cmp_eq_u32_e32 vcc, 0, v10", "s_and_saveexec_b64 s[14:15], vcc", "v_mov_b32_e32 v11, s6",
            "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc", "s_or_b64 exec, exec, s[14:15]",
            "v_readfirstlane_b32 s7, v11", "v_add_nc_u32_e32 v13, s7, v10", "v_lshlrev_b32_e32 v13, 2, v13",
            "buffer_store_dword v0, v13, s[0:3], 0 offen offset:256", "s_mov_b64 exec, s[10:11]"]), "B", [],
            ["aggregated-atomic", "elected-lane"])

    def test_b_spill_and_reload_in_half(self):
        self.tier("ps_b_spill", ps(["v_writelane_b32 v20, s4, 3", "v_readlane_b32 s21, v20, 3",
                                    "v_add_nc_u32_e32 v6, s21, v0"]), "B", [], ["lane-spill"])

    def test_b_spill_and_reload_upper_half_lane_40(self):
        # Pure scalar spill into lanes 32..63 (v20 is never read as a vector operand): lowered to a parked scalar
        # slot so SGPR spills past lane 31 are exact on 32-wide subgroups (tier B).
        self.tier("ps_b_spill_hi", ps([
            "v_writelane_b32 v20, s4, 40", "v_writelane_b32 v20, s5, 63",
            "v_readlane_b32 s21, v20, 40", "v_readlane_b32 s22, v20, 63",
            "s_add_u32 s23, s21, s22", "v_add_nc_u32_e32 v6, s23, v0"]), "B", [], ["lane-spill"])

    def test_c_spill_in_divergent_vote_branch_is_stale_scalar(self):
        # Spilling to lane 40 behind a wave-wide vote and reading outside must stay tier C (stale-scalar).
        # v_writelane_b32 is an 8-byte VOP3 instruction (2 dwords).
        self.tier("ps_c_spill_stale", ps([
            "v_writelane_b32 v20, s4, 40", "v_cmp_eq_u32_e32 vcc, 0, v0", "s_cbranch_vccz 2",
            "v_writelane_b32 v20, s5, 40", "v_readlane_b32 s21, v20, 40",
            "v_mov_b32_e32 v6, s21"]), "C", contains=["stale-scalar"])

    def test_c_spill_of_tainted_count_stays_tier_c(self):
        # Spilling a tainted scalar (lane-count) into lane 40 and reloading must still report lane-count.
        self.tier("ps_c_spill_cnt", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_bcnt1_i32_b64 s6, vcc",
            "v_writelane_b32 v20, s6, 40", "v_readlane_b32 s21, v20, 40",
            "v_mov_b32_e32 v6, s21"]), "C", ["lane-count"])

    def test_b_readfirstlane_uniform_inside_narrowed_exec(self):
        # v5 written from a scalar inside a narrowed EXEC block is uniform across all active lanes,
        # so v_readfirstlane_b32 inside that block is exact (tier B).
        self.tier("ps_b_rfl_narrow", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4",
            "s_and_saveexec_b64 s[12:13], vcc", "v_mov_b32_e32 v5, s4",
            "v_readfirstlane_b32 s20, v5", "v_add_nc_u32_e32 v6, s20, v0",
            "s_or_b64 exec, exec, s[12:13]"]), "B", [], ["readfirstlane-uniform"])

    def test_c_readfirstlane_after_exec_restore_not_uniform(self):
        # Writing v5 from a scalar under narrowed EXEC and reading after EXEC restore is NOT uniform (tier C).
        self.tier("ps_c_rfl_after_restore", ps([
            "v_mov_b32_e32 v5, v0", "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4",
            "s_and_saveexec_b64 s[12:13], vcc", "v_mov_b32_e32 v5, s4",
            "s_or_b64 exec, exec, s[12:13]", "v_readfirstlane_b32 s20, v5",
            "v_mov_b32_e32 v6, s20"]), "C")

    def test_c_readfirstlane_active_uniform_scalar_escapes_after_restore(self):
        # Reading v5 inside narrowed EXEC produces lane-0's old value in a half where EXEC == 0;
        # using that scalar after EXEC restore escapes the mask and must stay tier C (PR #2469 review).
        self.tier("ps_c_rfl_escape", ps([
            "v_mov_b32_e32 v5, 7", "v_cmp_eq_u32_e32 vcc, 33, v0",
            "s_and_saveexec_b64 s[12:13], vcc", "v_mov_b32_e32 v5, 50",
            "v_readfirstlane_b32 s20, v5", "s_mov_b64 exec, s[10:11]",
            "v_mov_b32_e32 v6, s20"]), "C", contains=["cross-lane"])

    def test_c_nonlinear_count_scaling_refused(self):
        # Count as shift distance (1 << count) and 24-bit truncating multiply (v_mul_u32_u24) are nonlinear
        # and must NOT qualify for wave-aggregated atomics (PR #2469 review).
        self.tier("ps_c_atomic_shift_dist", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_and_saveexec_b64 s[12:13], vcc",
            "s_bcnt1_i32_b64 s6, exec", "s_lshl_b32 s6, 1, s6",
            "v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10",
            "v_cmp_eq_u32_e32 vcc, 0, v10", "s_and_saveexec_b64 s[14:15], vcc", "v_mov_b32_e32 v11, s6",
            "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc", "s_or_b64 exec, exec, s[14:15]",
            "v_readfirstlane_b32 s7, v11", "v_add_nc_u32_e32 v13, s7, v10", "v_lshlrev_b32_e32 v13, 2, v13",
            "buffer_store_dword v0, v13, s[0:3], 0 offen offset:256", "s_mov_b64 exec, s[10:11]"]), "C",
            contains=["lane-count"])
        self.tier("ps_c_atomic_mul_u24", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_and_saveexec_b64 s[12:13], vcc",
            "s_bcnt1_i32_b64 s6, exec", "s_mul_i32 s6, s6, 0x100000",
            "v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10",
            "v_cmp_eq_u32_e32 vcc, 0, v10", "s_and_saveexec_b64 s[14:15], vcc", "v_mov_b32_e32 v15, s6",
            "v_mul_u32_u24_e32 v11, 1, v15",
            "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc", "s_or_b64 exec, exec, s[14:15]",
            "v_readfirstlane_b32 s7, v11", "v_add_nc_u32_e32 v13, s7, v10", "v_lshlrev_b32_e32 v13, 2, v13",
            "buffer_store_dword v0, v13, s[0:3], 0 offen offset:256", "s_mov_b64 exec, s[10:11]"]), "C",
            contains=["lane-count"])

    def test_b_scaled_aggregated_atomic_and_global_atomic_add(self):
        # Stride-scaled s_bcnt1 (s_lshl_b32 by 2 = 4 bytes/lane) in buffer_atomic_add and global_atomic_add.
        self.tier("ps_b_atomic_scaled", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_and_saveexec_b64 s[12:13], vcc",
            "s_bcnt1_i32_b64 s6, exec", "s_lshl_b32 s6, s6, 2",
            "v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10",
            "v_cmp_eq_u32_e32 vcc, 0, v10", "s_and_saveexec_b64 s[14:15], vcc", "v_mov_b32_e32 v11, s6",
            "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc", "s_or_b64 exec, exec, s[14:15]",
            "v_readfirstlane_b32 s7, v11", "v_lshlrev_b32_e32 v13, 2, v10", "v_add_nc_u32_e32 v13, s7, v13",
            "buffer_store_dword v0, v13, s[0:3], 0 offen offset:256", "s_mov_b64 exec, s[10:11]"]), "B", [],
            ["aggregated-atomic", "elected-lane"])
        # global_atomic_add with glc has operand order (vdst, vaddr, vdata, saddr).
        self.tier("ps_b_global_atomic", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_and_saveexec_b64 s[12:13], vcc",
            "s_bcnt1_i32_b64 s6, exec", "v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10",
            "v_cmp_eq_u32_e32 vcc, 0, v10", "s_and_saveexec_b64 s[14:15], vcc", "v_mov_b32_e32 v11, s6",
            "v_mov_b32_e32 v12, 0", "global_atomic_add v14, v12, v11, s[4:5] glc", "s_or_b64 exec, exec, s[14:15]",
            "v_readfirstlane_b32 s7, v14", "v_add_nc_u32_e32 v13, s7, v10", "v_lshlrev_b32_e32 v13, 2, v13",
            "global_store_dword v13, v0, s[4:5]", "s_mov_b64 exec, s[10:11]"]), "B", [],
            ["aggregated-atomic", "elected-lane"])

    # -- tier C: each near miss of the idioms above
    def test_c_lane_count_exported(self):
        self.tier("ps_c_cnt", ps(["v_cmp_eq_u32_e32 vcc, 1, v0", "s_bcnt1_i32_b64 s6, vcc", "v_mov_b32_e32 v6, s6"]),
                  "C", ["lane-count"])

    def test_c_rank_exported(self):
        self.tier("ps_c_rank", ps(["v_mbcnt_lo_u32_b32 v6, exec_lo, 0", "v_mbcnt_hi_u32_b32 v6, exec_hi, v6"]), "C",
                  ["lane-rank"])

    def test_c_lane_id_exported(self):
        self.tier("ps_c_id", ps(["v_mbcnt_lo_u32_b32 v6, -1, 0", "v_mbcnt_hi_u32_b32 v6, -1, v6"]), "C", ["lane-id"])

    def test_c_bpermute_lane_id_address_near_misses(self):
        # a shift by 1: address bit 7 is not reached, lane bit 5 lands in bit 6, which the permute uses
        self.tier("ps_c_bperm_shl1", ps([*LANE_ID, "v_lshlrev_b32_e32 v11, 1, v10", "ds_bpermute_b32 v6, v11, v0",
                                         "s_waitcnt lgkmcnt(0)"]), "C", ["lane-id"])
        # the low count alone is 32 for every upper-half lane of the wave
        self.tier("ps_c_bperm_lo", ps(["v_mbcnt_lo_u32_b32 v10, -1, 0", "v_lshlrev_b32_e32 v11, 2, v10",
                                       "ds_bpermute_b32 v6, v11, v0", "s_waitcnt lgkmcnt(0)"]), "C", ["lane-id"])
        # a right shift moves lane bit 5 down into the bits the permute uses
        self.tier("ps_c_bperm_shr", ps([*LANE_ID, "v_lshrrev_b32_e32 v11, 1, v10", "v_lshlrev_b32_e32 v11, 3, v11",
                                        "ds_bpermute_b32 v6, v11, v0", "s_waitcnt lgkmcnt(0)"]), "C", ["lane-id"])
        # combined with a lane mask: the mask is not the same in the wave and in a half
        self.tier("ps_c_bperm_mask", ps([*LANE_ID, "v_and_b32_e32 v11, s10, v10", "v_lshlrev_b32_e32 v11, 2, v11",
                                         "ds_bpermute_b32 v6, v11, v0", "s_waitcnt lgkmcnt(0)"]), "C")
        # SDWA selects and integer clamp change the value: a byte of id << 7, a saturating id - 32
        self.tier("ps_c_bperm_sdwa", ps([*LANE_ID, "v_lshlrev_b32_e32 v11, 7, v10",
                                         "v_mov_b32_sdwa v12, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1",
                                         "v_lshlrev_b32_e32 v12, 2, v12", "ds_bpermute_b32 v6, v12, v0",
                                         "s_waitcnt lgkmcnt(0)"]), "C", ["lane-id"])
        self.tier("ps_c_bperm_clamp", ps([*LANE_ID, "v_add_nc_u32_e64 v11, v10, -32 clamp",
                                          "v_lshlrev_b32_e32 v11, 2, v11", "ds_bpermute_b32 v6, v11, v0",
                                          "s_waitcnt lgkmcnt(0)"]), "C", ["lane-id"])
        # the lane id still taints the data it permutes, and the lane id itself
        self.tier("ps_c_bperm_data", ps([*LANE_ID, "v_lshlrev_b32_e32 v11, 2, v10", "ds_bpermute_b32 v6, v11, v10",
                                         "s_waitcnt lgkmcnt(0)"]), "C", ["lane-id"])

    def test_c_first_lane_index(self):
        self.tier("ps_c_ff1", ps(["v_cmp_eq_u32_e32 vcc, 1, v0", "s_ff1_i32_b64 s6, vcc", "v_mov_b32_e32 v6, s6"]),
                  "C", ["first-lane"])

    def test_c_readlane_of_varying(self):
        self.tier("ps_c_rl", ps(["v_readlane_b32 s6, v0, 40", "v_mov_b32_e32 v6, s6"]), "C", ["cross-lane"])
        self.tier("ps_c_rl7", ps(["v_readlane_b32 s6, v0, 7", "v_mov_b32_e32 v6, s6"]), "C", ["cross-lane"])

    def test_c_writelane_of_varying_lane(self):
        self.tier("ps_c_wl", ps(["v_writelane_b32 v6, s4, 40"]), "C", ["cross-lane"])

    def test_c_readfirstlane_of_varying_exported(self):
        self.tier("ps_c_rfl", ps(["v_readfirstlane_b32 s6, v0", "v_mov_b32_e32 v6, s6"]), "C")

    def test_c_waterfall_scalar_escapes(self):
        self.tier("ps_c_wf_out", ps([
            "v_and_b32_e32 v4, 7, v0", "v_readfirstlane_b32 s20, v4", "v_cmp_eq_u32_e32 vcc, s20, v4",
            "s_and_saveexec_b64 s[12:13], vcc", "v_add_nc_u32_e32 v6, s20, v0",
            "s_xor_b64 exec, exec, s[12:13]", "s_cbranch_execnz -5", "s_mov_b64 exec, s[10:11]",
            "v_mov_b32_e32 v7, s20"]), "C")

    def test_c_waterfall_trip_count(self):
        self.tier("ps_c_wf_count", ps([
            "s_mov_b32 s22, 0", "v_and_b32_e32 v4, 7, v0", "v_readfirstlane_b32 s20, v4",
            "v_cmp_eq_u32_e32 vcc, s20, v4", "s_and_saveexec_b64 s[12:13], vcc", "s_add_u32 s22, s22, 1",
            "s_xor_b64 exec, exec, s[12:13]", "s_cbranch_execnz -6", "s_mov_b64 exec, s[10:11]",
            "v_mov_b32_e32 v7, s22"]), "C", ["stale-scalar"])

    def test_c_vote_defines_scalar_read_later(self):
        self.tier("ps_c_stale", ps([
            "v_cmp_eq_u32_e32 vcc, 0, v0", "s_mov_b32 s5, 7", "s_cbranch_vccz 1", "s_add_u32 s5, s5, 1",
            "v_mov_b32_e32 v6, s5"]), "C")

    def test_c_exec_skip_defines_scalar_read_later(self):
        self.tier("ps_c_stale_exec", ps([
            "v_cmp_eq_u32_e32 vcc, 0, v0", "s_and_saveexec_b64 s[12:13], vcc", "s_mov_b32 s5, 7",
            "s_cbranch_execz 1", "s_add_u32 s5, s5, 1", "s_or_b64 exec, exec, s[12:13]", "v_mov_b32_e32 v6, s5"]),
            "C", ["stale-scalar"])

    def test_slist_taint_and_stale_tracking(self):
        # LLVM VCCTuple scalar lists (slist, e.g. [s104,s105,vcc_lo,vcc_hi]) must propagate taint and stale defs/uses.
        self.tier("ps_a_slist_clean", ps([
            "buffer_store_dword v0, v1, [s104,s105,vcc_lo,vcc_hi], 0 offen"]), "A", [], [])
        self.tier("ps_c_slist_src_taint", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_bcnt1_i32_b64 s104, vcc",
            "buffer_store_dword v0, v1, [s104,s105,vcc_lo,vcc_hi], 0 offen"]), "C", ["lane-count"])
        self.tier("ps_c_slist_dst_taint", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_bcnt1_i32_b64 s4, vcc",
            "s_load_dwordx4 [s104,s105,vcc_lo,vcc_hi], s[4:5], 0", "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v6, s104"]), "C", ["lane-count"])
        self.tier("ps_c_slist_stale_def", ps([
            "v_cmp_eq_u32_e32 vcc, 0, v0", "s_and_saveexec_b64 s[12:13], vcc", "s_mov_b32 s104, 7",
            "s_cbranch_execz 1", "s_load_dwordx4 [s104,s105,vcc_lo,vcc_hi], s[0:1], 0",
            "s_or_b64 exec, exec, s[12:13]", "v_mov_b32_e32 v6, s104"]), "C", ["stale-scalar"])
        self.tier("ps_c_slist_stale_use", ps([
            "v_cmp_eq_u32_e32 vcc, 0, v0", "s_and_saveexec_b64 s[12:13], vcc", "s_mov_b32 s104, 7",
            "s_cbranch_execz 1", "s_add_u32 s104, s104, 1", "s_or_b64 exec, exec, s[12:13]",
            "buffer_store_dword v0, v1, [s104,s105,vcc_lo,vcc_hi], 0 offen"]), "C", ["stale-scalar"])
        # VOP3b scalar carry-out destination written as an slist must be classified as a destination, not a source.
        import wave64_halves
        ins = {"op": "v_add_co_u32_e64", "args": [
            {"k": "vgpr", "r": 6, "n": 1},
            {"k": "slist", "r": [105, "vcc_lo"], "n": 2},
            {"k": "vgpr", "r": 0, "n": 1},
            {"k": "vgpr", "r": 1, "n": 1},
        ], "mods": []}
        dests, srcs, _, _ = wave64_halves.operands(ins)
        self.assertEqual([wave64_halves._regs(d) for d in dests], [[("v", 6)], [("s", 105), "vcc"]])
        self.assertEqual([wave64_halves._regs(s) for s in srcs], [[("v", 0)], [("v", 1)]])

    def test_c_branch_on_count(self):
        self.tier("ps_c_brcnt", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_bcnt1_i32_b64 s6, vcc", "s_cmp_eq_u32 s6, 5", "s_cbranch_scc1 1",
            "v_mov_b32_e32 v6, 1"]), "C")

    def test_c_vccz_without_narrowing(self):
        self.tier("ps_c_vccz", ps(["v_cmp_eq_u32_e32 vcc, 2, v0", "s_cbranch_vccz 1", "v_add_nc_u32_e32 v6, 9, v6"]),
                  "C", ["unsafe-vote-branch"])

    def test_c_half_of_a_mask(self):
        self.tier("ps_c_half", ps(["s_mov_b32 exec_hi, 0"]), "C", ["mask-half"])
        self.tier("ps_c_half2", ps(["v_cmp_eq_u32_e32 vcc, 1, v0", "s_mov_b32 s6, vcc_hi", "v_mov_b32_e32 v6, s6"]),
                  "C", contains=["mask-half"])

    def test_c_vcmpx_on_the_lane_id(self):
        # v_cmpx writes EXEC like v_cmp + s_and_b64 exec: on 32-wide waves lanes 32..63 see ids 0..31 and stay on.
        for form in ("e32", "e64"):
            self.tier(f"vs_c_vcmpx_{form}", vs([*LANE_ID, "v_mov_b32_e32 v6, 1", f"v_cmpx_gt_u32_{form} 32, v10",
                                                "v_mov_b32_e32 v6, 2"]), "C", ["tainted-exec"])

    def test_a_vcmpx_on_a_uniform_value(self):
        self.tier("vs_a_vcmpx_uniform", vs(["s_mov_b32 s20, 5", "v_mov_b32_e32 v6, 1", "v_cmpx_gt_u32_e32 s20, 3",
                                             "v_mov_b32_e32 v6, 2"]), "A", [])

    def test_c_mask_stored_as_data(self):
        self.tier("ps_c_maskdata", ps(["v_cmp_eq_u32_e32 vcc, 1, v0", "s_mov_b64 s[6:7], vcc",
                                       "v_mov_b32_e32 v6, s6"]), "C")

    def test_c_elected_lane_side_effect_not_a_count(self):
        self.tier("ps_c_elect", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_and_saveexec_b64 s[12:13], vcc",
            "v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10",
            "v_cmp_eq_u32_e32 vcc, 0, v10", "s_and_saveexec_b64 s[14:15], vcc", "v_mov_b32_e32 v11, 1",
            "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc",
            "s_or_b64 exec, exec, s[14:15]", "s_mov_b64 exec, s[10:11]"]), "C", ["elected-side-effect"])

    def test_c_rank_into_export_even_with_base(self):
        self.tier("ps_c_slot_exported", ps([
            "v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10", "v_add_nc_u32_e32 v6, 1, v10"]),
            "C", ["lane-rank"])

    # -- non-wave64 and other stages are not analysed
    def test_wave32_is_not_analysed(self):
        asm = translate_listing("ps_w32", listing("ps_w32", ["v_mov_b32_e32 v6, 0", "exp mrt0 v6, v6, v6, v6 done vm",
                                                           "s_endpgm"]))
        self.assertIsNone(rdna2spirv.translate.last_gen.w64g)
        self.assertNotEqual(asm, "")


class Execution(unittest.TestCase):
    """Vertex shaders: the words of lane k with all 64 lanes in one wave and with the lanes cut into two waves."""

    def run_both(self, name, lines, push=None):
        (tier, reasons, idioms), asm = meta_of(name, lines)
        whole, halves = run_cut(asm, 64, push), run_cut(asm, 32, push)
        return tier, out_param(whole), out_param(halves)

    def test_a_vertex_invariant(self):
        tier, whole, halves = self.run_both("vs_a_skip", vs([
            "v_and_b32_e32 v4, 7, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "v_mov_b32_e32 v6, 0x3e8",
            "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execz 1", "v_mul_u32_u24_e32 v6, 7, v0",
            "s_or_b64 exec, exec, s[12:13]", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_wqm_b64 s[14:15], vcc",
            "v_cndmask_b32_e64 v7, 0, 1, s[14:15]"]))
        self.assertEqual(tier, "A")
        self.assertEqual(whole, halves)
        # lanes with k % 8 == 1 run the skip region; whole-quad mode lights the 4 consecutive lanes around them
        want = [(7 * k if k % 8 == 1 else 1000, int(k % 8 < 4)) for k in range(64)]
        self.assertEqual([w[:2] for w in whole], want)

    def test_b_waterfall_invariant(self):
        tier, whole, halves = self.run_both("vs_b_wf", vs([
            "v_and_b32_e32 v4, 7, v0", "v_readfirstlane_b32 s20, v4", "v_cmp_eq_u32_e32 vcc, s20, v4",
            "s_and_saveexec_b64 s[12:13], vcc", "s_mul_i32 s21, s20, 17", "v_add_nc_u32_e32 v6, s21, v0",
            "s_xor_b64 exec, exec, s[12:13]", "s_cbranch_execnz -7", "s_mov_b64 exec, s[10:11]"]))
        self.assertEqual(tier, "B")
        self.assertEqual(whole, halves)
        self.assertEqual([w[0] for w in whole], [17 * (k & 7) + k for k in range(64)])

    def test_a_bpermute_lane_id_address_invariant(self):
        tier, whole, halves = self.run_both("vs_a_bperm_id", vs([
            *LANE_ID, "v_xor_b32_e32 v11, 1, v10", "v_lshlrev_b32_e32 v11, 2, v11", "ds_bpermute_b32 v6, v11, v0",
            "s_waitcnt lgkmcnt(0)"]))
        self.assertEqual(tier, "A")
        self.assertEqual(whole, halves)
        self.assertEqual([w[0] for w in whole], [k ^ 1 for k in range(64)])

    def test_a_permlane_swizzle_invariant(self):
        # vertex k is lane k of the wave (and lane k % 32 of its half): every read stays in the lane's half
        tier, whole, halves = self.run_both("vs_a_perm", vs([
            "s_mov_b32 s20, 0x67452301", "s_mov_b32 s21, 0xefcdab89", "v_permlane16_b32 v6, v0, s20, s21",
            "s_mov_b32 s22, 0x76543210", "s_mov_b32 s23, 0xfedcba98", "v_permlanex16_b32 v7, v0, s22, s23"]))
        self.assertEqual(tier, "A")
        self.assertEqual(whole, halves)
        self.assertEqual([w[:2] for w in whole], [(k ^ 1, k ^ 16) for k in range(64)])
        tier, whole, halves = self.run_both("vs_a_swz", vs([
            "ds_swizzle_b32 v6, v0 offset:swizzle(QUAD_PERM,3,2,1,0)", "ds_swizzle_b32 v7, v0 offset:swizzle(SWAP,16)",
            "s_waitcnt lgkmcnt(0)"]))
        self.assertEqual(tier, "A")
        self.assertEqual(whole, halves)
        self.assertEqual([w[:2] for w in whole], [(k ^ 3, k ^ 16) for k in range(64)])

    def test_c_bpermute_shift_by_one_differs(self):
        # the negative control of the rule: lane bit 5 reaches an address bit the permute uses
        tier, whole, halves = self.run_both("vs_c_bperm_shl1", vs([
            *LANE_ID, "v_lshlrev_b32_e32 v11, 1, v10", "ds_bpermute_b32 v6, v11, v0", "s_waitcnt lgkmcnt(0)"]))
        self.assertEqual(tier, "C")
        self.assertNotEqual([w[0] for w in whole], [w[0] for w in halves])

    def test_b_readlane_of_uniform_value_at_lane_40(self):
        # lane 40 does not exist in a 32-lane wave: without the rewrite the shuffle reads a lane that is not running
        tier, whole, halves = self.run_both("vs_b_rl", vs([
            "v_mov_b32_e32 v5, s4", "v_readlane_b32 s20, v5, 40", "v_add_nc_u32_e32 v6, s20, v0"]),
            push=[0, 0, 0, 0, 77])
        self.assertEqual(tier, "B")
        self.assertEqual(whole, halves)
        self.assertEqual([w[0] for w in whole], [77 + k for k in range(64)])

    def test_b_readfirstlane_of_uniform(self):
        tier, whole, halves = self.run_both("vs_b_rfl", vs([
            "v_mov_b32_e32 v5, s4", "v_readfirstlane_b32 s20, v5", "v_add_nc_u32_e32 v6, s20, v0"]),
            push=[0, 0, 0, 0, 9])
        self.assertEqual(tier, "B")
        self.assertEqual(whole, halves)
        self.assertEqual([w[0] for w in whole], [9 + k for k in range(64)])

    def test_b_spill_upper_half_lane_40_and_63_invariant(self):
        # Lanes 40 and 63 do not exist on a 32-lane subgroup: parked scalar slots keep 64-lane and 2x32-lane exact.
        tier, whole, halves = self.run_both("vs_b_spill_hi", vs([
            "v_writelane_b32 v20, s4, 40", "v_writelane_b32 v20, s5, 63",
            "v_readlane_b32 s21, v20, 40", "v_readlane_b32 s22, v20, 63",
            "s_add_u32 s23, s21, s22", "v_add_nc_u32_e32 v6, s23, v0"]),
            push=[0, 0, 0, 0, 100, 23])
        self.assertEqual(tier, "B")
        self.assertEqual(whole, halves)
        self.assertEqual([w[0] for w in whole], [123 + k for k in range(64)])

    def test_b_readfirstlane_uniform_under_narrowed_exec_invariant(self):
        # Also test with only lane 33 active (lower half has empty EXEC): using s20 ONLY inside the
        # narrowed block keeps 64-lane and 2x32-lane outputs identical.
        tier, whole, halves = self.run_both("vs_b_rfl_narrow", vs([
            "v_mov_b32_e32 v5, 7", "v_cmp_eq_u32_e32 vcc, 33, v0",
            "s_and_saveexec_b64 s[12:13], vcc", "v_mov_b32_e32 v5, s4",
            "v_readfirstlane_b32 s20, v5", "v_add_nc_u32_e32 v6, s20, v0",
            "s_mov_b64 exec, s[10:11]"]),
            push=[0, 0, 0, 0, 50])
        self.assertEqual(tier, "B")
        self.assertEqual(whole, halves)
        self.assertEqual([w[0] for w in whole], [83 if k == 33 else 0 for k in range(64)])

    def test_c_readfirstlane_escape_after_restore_differs(self):
        # Same narrowed-EXEC readfirstlane, but s20 escapes past EXEC restore: lower half gets lane-0
        # fallback (7) while upper half gets 50, so it MUST classify C.
        tier, whole, halves = self.run_both("vs_c_rfl_escape", vs([
            "v_mov_b32_e32 v5, 7", "v_cmp_eq_u32_e32 vcc, 33, v0",
            "s_and_saveexec_b64 s[12:13], vcc", "v_mov_b32_e32 v5, 50",
            "v_readfirstlane_b32 s20, v5", "s_mov_b64 exec, s[10:11]",
            "v_mov_b32_e32 v6, s20"]))
        self.assertEqual(tier, "C")
        self.assertNotEqual(whole, halves)
        self.assertEqual([w[0] for w in whole], [50] * 64)
        self.assertEqual([w[0] for w in halves], [7] * 32 + [50] * 32)

    def test_c_lane_count_differs(self):
        tier, whole, halves = self.run_both("vs_c_cnt", vs([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_bcnt1_i32_b64 s6, vcc",
            "v_mov_b32_e32 v6, s6", "v_mov_b32_e32 v7, v4"]))
        self.assertEqual(tier, "C")
        self.assertEqual([w[1] for w in whole], [w[1] for w in halves])     # per-lane data is the same
        self.assertEqual({w[0] for w in whole}, {16})
        self.assertEqual({w[0] for w in halves}, {8})

    def test_c_lane_id_differs(self):
        tier, whole, halves = self.run_both("vs_c_id", vs([
            "v_mbcnt_lo_u32_b32 v6, -1, 0", "v_mbcnt_hi_u32_b32 v6, -1, v6"]))
        self.assertEqual(tier, "C")
        self.assertEqual([w[0] for w in whole], list(range(64)))
        self.assertEqual([w[0] for w in halves], list(range(32)) * 2)

    def test_c_stale_scalar_differs(self):
        tier, whole, halves = self.run_both("vs_c_stale", vs([
            "v_cmp_eq_u32_e32 vcc, 0, v0", "s_mov_b32 s5, 7", "s_cbranch_vccz 1", "s_add_u32 s5, s5, 1",
            "v_mov_b32_e32 v6, s5"]))
        self.assertEqual(tier, "C")
        self.assertEqual({w[0] for w in whole}, {8})                      # vertex 0 is on, the whole wave adds
        self.assertEqual({w[0] for w in halves[:32]}, {8})
        self.assertEqual({w[0] for w in halves[32:]}, {7})                # the other half voted no


AGG_HEAD = ["v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_and_saveexec_b64 s[12:13], vcc"]
AGG_RANK = ["v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10"]
AGG_ELECT = ["v_cmp_eq_u32_e32 vcc, 0, v10", "s_and_saveexec_b64 s[14:15], vcc", "v_mov_b32_e32 v11, s6",
             "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc", "s_or_b64 exec, exec, s[14:15]",
             "v_readfirstlane_b32 s7, v11"]


class ReviewRegressions(unittest.TestCase):
    """Programs an independent review used to get a false tier A or B (silent wrong rendering). Each must be C,
    or, as vertex shaders, the 64-lane and the two-half runs must agree."""
    tier = Classification.tier

    def test_scc_of_a_vote_into_data(self):
        for name, body in (
                ("cselect", ["v_cmp_eq_u32_e32 vcc, 1, v0", "s_cmp_lg_u64 vcc, 0", "s_cselect_b32 s5, 1, 0",
                             "v_mov_b32_e32 v6, s5"]),
                ("saveexec_cselect", ["v_cmp_eq_u32_e32 vcc, 1, v0", "s_and_saveexec_b64 s[12:13], vcc",
                                      "s_cselect_b32 s5, 1, 0", "s_or_b64 exec, exec, s[12:13]",
                                      "v_mov_b32_e32 v6, s5"]),
                ("addc", ["v_cmp_eq_u32_e32 vcc, 1, v0", "s_cmp_lg_u64 vcc, 0", "s_addc_u32 s5, s5, 0",
                          "v_mov_b32_e32 v6, s5"]),
                ("cmov", ["v_cmp_eq_u32_e32 vcc, 1, v0", "s_cmp_lg_u64 vcc, 0", "s_mov_b32 s5, 0",
                          "s_cmov_b32 s5, 9", "v_mov_b32_e32 v6, s5"]),
                ("bcnt", ["s_bcnt1_i32_b64 s6, exec", "s_cselect_b32 s5, 1, 0", "v_mov_b32_e32 v6, s5"])):
            self.tier("ps_scc_" + name, ps(body), "C", contains=["cross-lane"])

    def test_scc_vote_still_branches(self):
        # the one allowed consumer of a vote's SCC: s_cbranch_scc*, skipping lane-masked work
        self.tier("ps_scc_branch", ps([
            "v_cmp_eq_u32_e32 vcc, 2, v0", "s_and_saveexec_b64 s[16:17], vcc", "s_cbranch_scc0 1",
            "v_add_nc_u32_e32 v6, 9, v6", "s_or_b64 exec, exec, s[16:17]"]), "A", [], [])

    def test_rank_alone_is_not_a_slot(self):
        self.tier("ps_rank_addr", ps([
            "v_mbcnt_lo_u32_b32 v10, exec_lo, 0", "v_mbcnt_hi_u32_b32 v10, exec_hi, v10",
            "v_lshlrev_b32_e32 v13, 2, v10", "buffer_store_dword v0, v13, s[0:3], 0 offen offset:256"]),
            "C", ["lane-rank"])

    def test_base_alone_is_not_a_slot(self):
        self.tier("ps_base_addr", ps(AGG_HEAD + ["s_bcnt1_i32_b64 s6, exec"] + AGG_RANK + AGG_ELECT + [
            "v_mov_b32_e32 v13, s7", "buffer_store_dword v0, v13, s[0:3], 0 offen offset:256",
            "s_mov_b64 exec, s[10:11]"]), "C", ["allocation-order"])

    def test_atomic_add_of_a_count_by_every_lane(self):
        for name, mods in (("plain", ""), ("glc", " glc")):
            self.tier("ps_allcount_" + name, ps([
                "s_bcnt1_i32_b64 s6, exec", "v_mov_b32_e32 v11, s6", "v_mov_b32_e32 v12, 0",
                "buffer_atomic_add v11, v12, s[0:3], 0 offen" + mods]), "C", ["lane-count"])

    def test_count_of_another_mask_in_the_elected_atomic(self):
        self.tier("ps_cnt_other", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_bcnt1_i32_b64 s6, vcc",
            "v_cmp_eq_u32_e32 vcc, 2, v4", "s_and_saveexec_b64 s[12:13], vcc"] + AGG_RANK + AGG_ELECT[:2] + [
            "v_mov_b32_e32 v11, s6", "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc",
            "s_or_b64 exec, exec, s[14:15]", "s_mov_b64 exec, s[10:11]"]), "C", ["lane-count"])

    def test_count_of_exec_before_it_narrowed(self):
        self.tier("ps_cnt_wider", ps([
            "s_bcnt1_i32_b64 s6, exec", "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc, 1, v4",
            "s_and_saveexec_b64 s[12:13], vcc"] + AGG_RANK + AGG_ELECT[:2] + [
            "v_mov_b32_e32 v11, s6", "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc",
            "s_or_b64 exec, exec, s[14:15]", "s_mov_b64 exec, s[10:11]"]), "C", ["lane-count"])

    def test_vote_whose_other_side_has_code(self):
        self.tier("ps_vccnz_else", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_cbranch_vccnz 2", "v_mov_b32_e32 v6, 5", "s_branch 3",
            "s_and_saveexec_b64 s[16:17], vcc", "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[16:17]"]),
            "C", contains=["unsafe-vote-branch"])
        self.tier("ps_vccnz_else2", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_cbranch_vccnz 2", "v_mov_b32_e32 v6, 5", "s_branch 3",
            "s_and_saveexec_b64 s[16:17], vcc", "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[16:17]",
            "v_mov_b32_e32 v7, 1"]), "C", contains=["unsafe-vote-branch"])

    def test_vccnz_with_a_jump_only_fall_through_is_a_skip(self):
        self.tier("ps_vccnz_jump", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_cbranch_vccnz 1", "s_branch 3", "s_and_saveexec_b64 s[16:17], vcc",
            "v_add_nc_u32_e32 v6, 13, v6", "s_or_b64 exec, exec, s[16:17]"]), "A", [], [])

    def test_exec_zero_side_with_code_of_its_own(self):
        # execz: the taken side (no lane active in the wave) writes an SGPR that is exported afterwards
        self.tier("ps_execz_sgpr", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_mov_b32 s5, 7", "s_and_saveexec_b64 s[12:13], vcc",
            "s_cbranch_execz 2", "v_mov_b32_e32 v8, 100", "s_branch 1", "s_mov_b32 s5, 9",
            "s_or_b64 exec, exec, s[12:13]", "v_mov_b32_e32 v6, s5"]), "C", contains=["unsafe-execz-branch"])
        # execnz: the fall-through is the exec-zero side
        self.tier("ps_execnz_else", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execnz 2",
            "s_mov_b32 s5, 9", "s_branch 1", "v_mov_b32_e32 v8, 100", "s_or_b64 exec, exec, s[12:13]",
            "v_mov_b32_e32 v6, s5"]), "C", contains=["unsafe-execz-branch"])
        # a VALU-only exec-zero side is refused too (kept simple: the side must be the join)
        self.tier("ps_execz_valu", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execz 2",
            "v_mov_b32_e32 v6, 7", "s_branch 1", "v_mov_b32_e32 v6, 5", "s_or_b64 exec, exec, s[12:13]"]),
            "C", contains=["unsafe-execz-branch"])

    def test_exec_zero_side_that_is_the_join_or_a_loop_exit_is_a_skip(self):
        self.tier("ps_execz_join", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execz 1",
            "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[12:13]"]), "A", [], [])
        self.tier("ps_execz_jump_join", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execnz 1",
            "s_branch 1", "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[12:13]"]), "A", [], [])

    # ---- second review: a rank / count / mask that is not bound to the mask it belongs to
    ATOMIC = ["v_mov_b32_e32 v11, s6", "v_mov_b32_e32 v12, 0", "buffer_atomic_add v11, v12, s[0:3], 0 offen glc",
              "s_or_b64 exec, exec, s[14:15]"]
    SLOT = ["v_readfirstlane_b32 s7, v11", "v_add_nc_u32_e32 v13, s7, {rank}", "v_lshlrev_b32_e32 v13, 2, v13",
            "buffer_store_dword v0, v13, s[0:3], 0 offen offset:256"]
    ELECT_NARROWED = ["v_cmp_eq_u32_e32 vcc, 0, v10", "s_and_saveexec_b64 s[14:15], vcc"]
    ENTRY_RANK = ["v_mbcnt_lo_u32_b32 v14, exec_lo, 0", "v_mbcnt_hi_u32_b32 v14, exec_hi, v14"]

    def test_slot_rank_of_the_entry_exec_is_not_the_counted_mask(self):
        body = (self.ENTRY_RANK + AGG_HEAD + ["s_bcnt1_i32_b64 s6, exec"] + AGG_RANK + self.ELECT_NARROWED
                + self.ATOMIC + [x.format(rank="v14") for x in self.SLOT] + ["s_mov_b64 exec, s[10:11]"])
        self.tier("ps_rank_entry_exec", ps(body), "C", contains=["lane-rank"])

    def test_slot_rank_from_the_saved_exec(self):
        body = (AGG_HEAD + ["s_bcnt1_i32_b64 s6, exec"] + AGG_RANK + self.ELECT_NARROWED + self.ATOMIC + [
            "v_mbcnt_lo_u32_b32 v14, s12, 0", "v_mbcnt_hi_u32_b32 v14, s13, v14"]
            + [x.format(rank="v14") for x in self.SLOT] + ["s_mov_b64 exec, s[10:11]"])
        self.tier("ps_rank_saved_exec", ps(body), "C", contains=["lane-rank"])

    def test_slot_of_the_right_mask_is_still_b(self):
        body = (AGG_HEAD + ["s_bcnt1_i32_b64 s6, exec"] + AGG_RANK + self.ELECT_NARROWED + self.ATOMIC
                + [x.format(rank="v10") for x in self.SLOT] + ["s_mov_b64 exec, s[10:11]"])
        self.tier("ps_rank_right_mask", ps(body), "B", [], ["aggregated-atomic", "elected-lane"])

    def test_lane_mask_combined_with_a_register_is_data(self):
        for name, body in (
                ("and_user_bcnt", ["s_and_b64 s[14:15], exec, s[4:5]", "s_bcnt1_i32_b64 s6, s[14:15]",
                                   "v_mov_b32_e32 v6, s6"]),
                ("and_user_narrowed", ["v_cmp_lt_u32_e32 vcc, 40, v0", "s_and_b64 exec, exec, vcc",
                                       "s_and_b64 s[14:15], exec, s[4:5]", "s_mov_b64 exec, s[10:11]",
                                       "s_bcnt1_i32_b64 s6, s[14:15]", "v_mov_b32_e32 v6, s6"]),
                ("vcc_and_user_cmp", ["v_cmp_lt_u32_e32 vcc, 40, v0", "s_and_b64 s[14:15], vcc, s[4:5]",
                                      "s_cmp_eq_u64 s[14:15], 0", "s_cselect_b32 s6, 1, 2", "v_mov_b32_e32 v6, s6"]),
                ("vcc_and_user_branch", ["v_cmp_lt_u32_e32 vcc, 40, v0", "s_and_b64 s[14:15], vcc, s[4:5]",
                                         "s_cmp_eq_u64 s[14:15], 0", "s_cbranch_scc1 1", "v_mov_b32_e32 v6, 4"]),
                ("or_user_ff1", ["v_cmp_lt_u32_e32 vcc, 40, v0", "s_or_b64 s[14:15], vcc, s[4:5]",
                                 "s_ff1_i32_b64 s6, s[14:15]", "v_mov_b32_e32 v6, s6"]),
                ("andn2_user_bcnt", ["v_cmp_lt_u32_e32 vcc, 40, v0", "s_andn2_b64 s[14:15], s[4:5], vcc",
                                     "s_bcnt1_i32_b64 s6, s[14:15]", "v_mov_b32_e32 v6, s6"]),
                ("and_user_bcnt0", ["v_cmp_lt_u32_e32 vcc, 40, v0", "s_and_b64 s[14:15], vcc, s[4:5]",
                                    "s_bcnt0_i32_b64 s6, s[14:15]", "v_mov_b32_e32 v6, s6"])):
            self.tier("ps_maskdata_" + name, ps(body), "C")

    def test_mask_and_constants_stay_masks(self):
        self.tier("ps_mask_const", ps([
            "v_cmp_lt_u32_e32 vcc, 40, v0", "s_and_b64 s[14:15], vcc, -1", "s_and_saveexec_b64 s[12:13], s[14:15]",
            "v_add_nc_u32_e32 v6, 9, v6", "s_or_b64 exec, exec, s[12:13]"]), "A", [], [])

    def test_waterfall_vgpr_rewritten_before_the_compare(self):
        self.tier("ps_wf_rewritten", ps([
            "v_and_b32_e32 v4, 7, v0", "v_readfirstlane_b32 s20, v4", "v_add_nc_u32_e32 v4, 1, v4",
            "v_cmp_eq_u32_e32 vcc, s20, v4", "s_and_saveexec_b64 s[12:13], vcc", "v_add_nc_u32_e32 v6, s20, v0",
            "s_xor_b64 exec, exec, s[12:13]", "s_cbranch_execnz -7", "s_mov_b64 exec, s[10:11]"]), "C")

    def test_analysis_failure_is_tier_c_and_not_an_error(self):
        import wave64_halves
        old = wave64_halves.analyse

        def boom(*a, **k):
            raise RuntimeError("boom")
        wave64_halves.analyse = boom
        try:
            self.tier("ps_boom", ps(["v_mov_b32_e32 v6, 1"]), "C", ["analysis-error"])
        finally:
            wave64_halves.analyse = old


class ReviewExecution(unittest.TestCase):
    run_both = Execution.run_both

    def lanes(self, name, body):
        tier, whole, halves = self.run_both(name, vs(body))
        return tier, sorted({w[0] if w else None for w in whole}), sorted({w[0] if w else None for w in halves})

    def test_cselect_of_a_vote_differs_and_is_c(self):
        tier, whole, halves = self.lanes("vs_cselect", [
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_cmp_lg_u64 vcc, 0", "s_cselect_b32 s5, 1, 0", "v_mov_b32_e32 v6, s5"])
        self.assertEqual((tier, whole, halves), ("C", [1], [0, 1]))     # vertex 1 is in the first half only

    def test_vote_else_side_differs_and_is_c(self):
        tier, whole, halves = self.lanes("vs_vccnz_else", [
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_cbranch_vccnz 2", "v_mov_b32_e32 v6, 5", "s_branch 3",
            "s_and_saveexec_b64 s[16:17], vcc", "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[16:17]"])
        self.assertEqual(tier, "C")
        self.assertNotEqual(whole, halves)

    def test_vote_with_empty_other_side_is_invariant(self):
        tier, whole, halves = self.lanes("vs_vccz_join", [
            "v_cmp_eq_u32_e32 vcc, 33, v0", "v_mov_b32_e32 v6, 4", "s_cbranch_vccz 3",
            "s_and_saveexec_b64 s[16:17], vcc", "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[16:17]"])
        self.assertEqual(tier, "A")
        self.assertEqual(whole, halves)
        self.assertEqual(whole, [4, 7])                                  # only vertex 33, in the second half


class SccVotePolarity(unittest.TestCase):
    """`s_cmp_eq_u64 m, 0` sets SCC when the mask is empty, `s_cmp_lg_u64 m, 0` (and the 64-bit logic ops, s_bcnt1)
    when it is not: the side a vote runs when some lane votes yes is the not-taken side of `s_cbranch_scc1` after
    cmp_eq and the taken one after cmp_lg."""
    tier = Classification.tier
    run_both = Execution.run_both
    HEAD = ["v_cmp_gt_u32_e32 vcc, 20, v0", "s_and_b64 s[20:21], exec, vcc"]

    def test_cmp_eq_jump_to_narrowing_is_c(self):
        # scc1 (mask empty) jumps to the narrowing, the "yes" side skips it: a half with no lane < 20 empties EXEC,
        # the wave never does
        tier, whole, halves = self.run_both("vs_cmp_eq_wrong", vs(self.HEAD + [
            "s_cmp_eq_u64 s[20:21], 0", "s_cbranch_scc1 1", "s_branch 1", "s_and_b64 exec, exec, s[20:21]",
            "v_add_nc_u32_e32 v6, 3, v0"]))
        self.assertEqual([w[0] for w in whole], [3 + k for k in range(64)])
        self.assertEqual([w[0] for w in halves], [3 + k for k in range(32)] + [0] * 32)
        self.assertEqual(tier, "C")

    def test_cmp_eq_skip_when_empty_is_a(self):
        # scc1 (mask empty) skips the narrowed region: the yes side narrows EXEC to the mask, the no side is the join
        for name, cmp_, br in (("vs_cmp_eq_skip", "s_cmp_eq_u64 s[20:21], 0", "s_cbranch_scc1 2"),
                               ("vs_cmp_eq_skip0", "s_cmp_eq_u64 0, s[20:21]", "s_cbranch_scc1 2"),
                               ("vs_cmp_lg_skip", "s_cmp_lg_u64 s[20:21], 0", "s_cbranch_scc0 2")):
            tier, whole, halves = self.run_both(name, vs(self.HEAD + [
                cmp_, br, "s_and_b64 exec, exec, s[20:21]", "v_add_nc_u32_e32 v6, 3, v0",
                "s_mov_b64 exec, s[10:11]"]))
            self.assertEqual(tier, "A", name)
            self.assertEqual(whole, halves, name)
            self.assertEqual([w[0] for w in whole], [3 + k if k < 20 else 0 for k in range(64)], name)

    def test_exec_restore_by_mov_is_not_an_exec_vote(self):
        # s_mov_b64 writes no SCC: the branch after restoring EXEC reads the cmp_lg vote on the mask, not EXEC's
        tier, whole, halves = self.run_both("vs_mov_exec_scc", vs(self.HEAD + [
            "s_cmp_lg_u64 s[20:21], 0", "s_mov_b64 exec, s[10:11]", "s_cbranch_scc0 1",
            "v_add_nc_u32_e32 v6, 3, v0"]))
        self.assertEqual([w[0] for w in whole], [3 + k for k in range(64)])
        self.assertEqual([w[0] for w in halves], [3 + k for k in range(32)] + [0] * 32)
        self.assertEqual(tier, "C")

    def test_mov_of_the_mask_keeps_the_cmp_eq_vote(self):
        # s_mov_b64 writes no SCC: scc0 after cmp_eq is "mask not empty", the wrong side to skip the narrowing. Tagged
        # as a vote on the copy (non-empty polarity) the same skip looked like the exact one: a false tier A
        tier, whole, halves = self.run_both("vs_mov_keeps_vote", vs(self.HEAD + [
            "s_cmp_eq_u64 s[20:21], 0", "s_mov_b64 s[22:23], vcc", "s_cbranch_scc0 2",
            "s_and_b64 exec, exec, s[22:23]", "v_add_nc_u32_e32 v6, 3, v0", "s_mov_b64 exec, s[10:11]"]))
        self.assertEqual(tier, "C")
        self.assertEqual(whole, halves)       # equal on this input; the tier is about the branch sense

    def test_cselect_result_is_no_vote_execution(self):
        # s_cselect_b64 reads SCC and writes none: scc1 after cmp_eq skips when the mask is empty; the narrowing by the
        # cselect result is the wrong thing to skip and the halves differ from the wave
        tier, whole, halves = self.run_both("vs_cselect_keeps_vote", vs(self.HEAD + [
            "s_cmp_eq_u64 s[20:21], 0", "s_cselect_b64 s[22:23], s[20:21], exec", "s_cbranch_scc1 2",
            "s_and_b64 exec, exec, s[22:23]", "v_add_nc_u32_e32 v6, 3, v0", "s_mov_b64 exec, s[10:11]"]))
        self.assertEqual(tier, "C")
        self.assertNotEqual(whole, halves)

    def test_cselect_and_mov_keep_the_scc_vote(self):
        # neither s_cselect_b64 nor s_mov_b64 of a mask writes SCC: the vote stays on the compared mask
        for name, mid in (("vs_keep_cselect", "s_cselect_b64 s[22:23], s[20:21], exec"),
                          ("vs_keep_mov_vcc", "s_mov_b64 s[22:23], vcc")):
            self.tier(name, vs(self.HEAD + [
                "s_cmp_eq_u64 s[20:21], 0", mid, "s_cbranch_scc1 1", "s_branch 1", "s_and_b64 exec, exec, s[20:21]",
                "v_add_nc_u32_e32 v6, 3, v0"]), "C", contains=["unsafe-vote-branch"])
            self.tier(name + "_a", vs(self.HEAD + [
                "s_cmp_eq_u64 s[20:21], 0", mid, "s_cbranch_scc1 2", "s_and_b64 exec, exec, s[20:21]",
                "v_add_nc_u32_e32 v6, 3, v0", "s_mov_b64 exec, s[10:11]"]), "A")

    def test_cmp_eq_wrong_polarity_classification(self):
        # the same skip with the branch sense flipped: scc0 after cmp_eq jumps over the narrowing when lanes vote yes
        self.tier("vs_cmp_eq_scc0", vs(self.HEAD + [
            "s_cmp_eq_u64 s[20:21], 0", "s_cbranch_scc0 2", "s_and_b64 exec, exec, s[20:21]",
            "v_add_nc_u32_e32 v6, 3, v0", "s_mov_b64 exec, s[10:11]"]), "C", contains=["unsafe-vote-branch"])


UNIFORM = [[0, 0, 0, 0, 9], [0, 0, 0, 0, 1]]     # s4 = 9: the compare 5 < s4 says yes; s4 = 1: no
VCCZ_SIDES = ["s_cbranch_vccz 3", "v_mov_b32_e32 v6, 7", "s_mov_b32 s20, 1", "s_branch 2", "v_mov_b32_e32 v6, 5",
              "s_mov_b32 s20, 2", "v_add_nc_u32_e32 v7, s20, v0"]


class UniformVote(unittest.TestCase):
    """A vote on a compare of a wave-uniform value made while EXEC is the launch mask is no lane vote: the mask is every
    launch lane or none, so each half votes as the wave does, and both sides may have code (scalars included) of their
    own. The near misses vote differently in the two halves and must stay tier C."""
    run_both = Execution.run_both

    def invariant(self, name, body, tier, idioms, want):
        for push in UNIFORM:
            t, whole, halves = self.run_both(name, vs(body), push)
            (_, reasons, got_idioms), _ = meta_of(name, vs(body))
            self.assertEqual((t, reasons, got_idioms), (tier, [], idioms), name)
            self.assertEqual(whole, halves, (name, push))
            self.assertEqual([w[:2] for w in whole], [want(push[4], k) for k in range(64)], (name, push))

    def differs(self, name, body, reason, push=UNIFORM[0]):
        t, whole, halves = self.run_both(name, vs(body), push)
        (_, reasons, _), _ = meta_of(name, vs(body))
        self.assertEqual(t, "C", name)
        self.assertIn(reason, reasons, name)
        self.assertNotEqual(whole, halves, name)
        return whole, halves

    def test_a_vccz_if_else_with_scalars_read_after_the_join(self):
        self.invariant("vs_uv_vccz", ["v_mov_b32_e32 v1, s4", "v_cmp_lt_u32_e32 vcc, 5, v1"] + VCCZ_SIDES, "A", [],
                       lambda s4, k: (7, 1 + k) if s4 > 5 else (5, 2 + k))

    def test_a_llvm_and_exec_vccnz(self):
        self.invariant("vs_uv_and_exec", [
            "v_mov_b32_e32 v1, s4", "v_cmp_lt_u32_e64 s[14:15], 5, v1", "s_and_b64 vcc, exec, s[14:15]",
            "s_cbranch_vccnz 3", "v_mov_b32_e32 v6, 5", "s_mov_b32 s20, 2", "s_branch 2", "v_mov_b32_e32 v6, 7",
            "s_mov_b32 s20, 1", "v_add_nc_u32_e32 v7, s20, v0"], "A", [],
            lambda s4, k: (7, 1 + k) if s4 > 5 else (5, 2 + k))

    def test_a_uniform_loop_back_edge(self):
        # s20 counts the trips in an SGPR across the back edge: every half runs s4 of them, as the wave does
        self.invariant("vs_uv_loop", [
            "s_mov_b32 s20, 0", "s_add_u32 s20, s20, 1", "v_add_nc_u32_e32 v6, 3, v6", "v_mov_b32_e32 v1, s20",
            "v_cmp_gt_u32_e32 vcc, s4, v1", "s_cbranch_vccnz -5", "v_add_nc_u32_e32 v7, s20, v0"], "A", [],
            lambda s4, k: (3 * s4, s4 + k))

    def test_b_readfirstlane_of_a_uniform_value_feeds_the_vote(self):
        self.invariant("vs_uv_rfl", ["v_mov_b32_e32 v1, s4", "v_readfirstlane_b32 s21, v1", "v_mov_b32_e32 v2, s21",
                                     "v_cmp_lt_u32_e32 vcc, 5, v2"] + VCCZ_SIDES, "B", ["readfirstlane-uniform"],
                       lambda s4, k: (7, 1 + k) if s4 > 5 else (5, 2 + k))

    def test_c_compare_under_a_narrowed_exec(self):
        # EXEC narrowed to lanes 0..31: the second half has no active lane, votes no and sets s20 = 2, which the lanes
        # of that half read once EXEC is restored; the wave voted yes (s20 = 1)
        whole, halves = self.differs("vs_uv_narrowed", [
            "v_mov_b32_e32 v1, s4", "v_cmp_gt_u32_e32 vcc, 32, v0", "s_and_saveexec_b64 s[12:13], vcc",
            "v_cmp_lt_u32_e32 vcc, 5, v1", "s_cbranch_vccz 3", "v_mov_b32_e32 v6, 7", "s_mov_b32 s20, 1",
            "s_branch 2", "v_mov_b32_e32 v6, 5", "s_mov_b32 s20, 2", "s_mov_b64 exec, s[12:13]",
            "v_add_nc_u32_e32 v7, s20, v0"], "unsafe-vote-branch")
        self.assertEqual([w[1] for w in whole[32:]], [1 + k for k in range(32, 64)])
        self.assertEqual([w[1] for w in halves[32:]], [2 + k for k in range(32, 64)])

    def test_c_half_of_a_uniform_mask_rewritten(self):
        # s15 (the high half of the uniform mask) is rewritten before the mask becomes VCC: the wave still sees the low
        # half's bits and votes yes, the second half sees s15 = 0 and votes no
        name = "vs_uv_half_rewritten"
        body = vs(["v_mov_b32_e32 v1, s4", "v_cmp_lt_u32_e64 s[14:15], 5, v1", "s_mov_b32 s15, 0",
                   "s_mov_b64 vcc, s[14:15]"] + VCCZ_SIDES)
        (t, reasons, _), _ = meta_of(name, body)
        self.assertEqual(t, "C", name)
        self.assertIn("unsafe-vote-branch", reasons, name)

    def test_c_readfirstlane_of_a_varying_value(self):
        # the first lane is vertex 0 in the wave and vertex 32 in the second half
        self.differs("vs_uv_rfl_varying", ["v_readfirstlane_b32 s21, v0", "v_mov_b32_e32 v1, s21",
                                           "v_cmp_eq_u32_e32 vcc, 0, v1"] + VCCZ_SIDES, "tainted-branch")

    def test_c_compare_on_a_lane_count(self):
        # 64 lanes in the wave, 32 in a half: 40 < count is yes for the wave, no for each half
        self.differs("vs_uv_bcnt", ["s_bcnt1_i32_b64 s21, exec", "v_mov_b32_e32 v1, s21",
                                    "v_cmp_lt_u32_e32 vcc, 40, v1"] + VCCZ_SIDES, "lane-count")

    def test_c_scc_of_a_lane_mask_with_a_uniform_vcc_written_after_it(self):
        # SCC is the vote of the AND (vertex 1 on: the wave and the first half say yes); the uniform v_cmp after it
        # writes VCC but not SCC, so the branch still votes on the lane mask; since SHADER-820 (#4620) the new VCC
        # ends SCC's tag as a vote, so the branch is reported as a tainted one
        self.differs("vs_uv_scc_stale", ["v_cmp_eq_u32_e64 s[16:17], 1, v0", "s_and_b64 vcc, exec, s[16:17]",
                                         "v_mov_b32_e32 v1, s4", "v_cmp_lt_u32_e32 vcc, 5, v1", "s_cbranch_scc0 3"]
                     + VCCZ_SIDES[1:], "tainted-branch")

    def test_c_uniform_mask_anded_with_a_lane_mask(self):
        # only vertex 1 survives the AND: the wave and the first half vote yes, the second half no
        self.differs("vs_uv_and_lane", ["v_mov_b32_e32 v1, s4", "v_cmp_lt_u32_e32 vcc, 5, v1",
                                        "v_cmp_eq_u32_e64 s[16:17], 1, v0", "s_and_b64 vcc, vcc, s[16:17]"]
                     + VCCZ_SIDES, "unsafe-vote-branch")

    def test_c_partial_write_keeps_varying_bits(self):
        # v1 = lane << 8, then its low byte (SDWA, UNUSED_PRESERVE) or low half (16-bit add) from s4: the sources are
        # uniform but bits the write keeps are not, so 0x2000 > v1 is yes for lanes 0..31 only (found by @shdann)
        sides = ["s_cbranch_vccz 3", "v_mov_b32_e32 v6, 7", "s_mov_b32 s20, 1", "s_branch 2", "v_mov_b32_e32 v6, 5",
                 "s_mov_b32 s20, 2", "v_add_nc_u32_e32 v7, s20, v0"]
        self.differs("vs_uv_sdwa_partial", [
            "v_lshlrev_b32_e32 v1, 8, v0",
            "v_mov_b32_sdwa v1, s4 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD",
            "v_cmp_gt_u32_e32 vcc, 0x2000, v1"] + sides, "unsafe-vote-branch", push=[0, 0, 0, 0, 9])
        self.differs("vs_uv_u16_partial", [
            "v_lshlrev_b32_e32 v1, 16, v0", "v_add_u16_e64 v1, 1, s4",
            "v_cmp_gt_u32_e32 vcc, 0x200000, v1"] + sides, "unsafe-vote-branch", push=[0, 0, 0, 0, 9])

    def test_a_whole_register_16_bit_forms_stay_uniform(self):
        # a packed convert, a SAD and a VOP2 16-bit integer op write the whole VGPR: uniform sources, uniform result
        for name, op in (("vs_uv_pkrtz", "v_cvt_pkrtz_f16_f32_e32 v1, s4, v2"), ("vs_uv_sad", "v_sad_u16 v1, s4, 0, v2"),
                         ("vs_uv_add_u16_e32", "v_add_u16_e32 v1, s4, v2")):
            (t, reasons, _), _ = meta_of(name, vs(["v_mov_b32_e32 v2, s4", op, "v_cmp_lt_u32_e32 vcc, 5, v1"]
                                                  + VCCZ_SIDES))
            self.assertEqual((t, reasons), ("A", []), name)

    def test_a_partial_write_over_a_uniform_register(self):
        # the kept bits come from s4 too, so the register stays uniform
        name = "vs_uv_sdwa_uniform"
        (t, reasons, _), _ = meta_of(name, vs([
            "v_mov_b32_e32 v1, s4", "v_mov_b32_sdwa v1, s4 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD",
            "v_cmp_lt_u32_e32 vcc, 5, v1"] + VCCZ_SIDES))
        self.assertEqual((t, reasons), ("A", []), name)

class KillExit(unittest.TestCase):
    """The early exit of a pixel kill (`s_andn2_b64 live, live, exec/vcc`, `s_cbranch_scc0` to a block that only
    empties EXEC from that mask, exports and ends): a half whose live mask is empty writes nothing there, and in the
    wave its lanes stay off on the run side to the end. Tier A only when the exit is inert, the run side narrows EXEC
    into the mask's quads before anything else and never widens it, and the vote is the mask's own."""
    tier = Classification.tier
    run_both = Execution.run_both

    # -- tier A
    def test_a_kill_wqm_and_exec(self):
        self.tier("ps_kill_wqm", kill(KILL, RUN_WQM + ["v_add_nc_u32_e32 v6, 3, v0"], EXIT_PS), "A", [], [])

    def test_a_kill_cmpx_wqm_mov_exec(self):
        self.tier("ps_kill_cmpx", kill(
            ["s_mov_b64 s[20:21], exec", "v_cmpx_gt_u32_e32 40, v0", "s_andn2_b64 s[20:21], s[20:21], exec"],
            ["s_wqm_b64 s[22:23], s[20:21]", "s_mov_b64 exec, s[22:23]", "v_add_nc_u32_e32 v6, 3, v0"],
            EXIT_PS), "A", [], [])

    def test_a_kill_direct_narrowing(self):
        for name, run in (("ps_kill_and", ["s_and_b64 exec, exec, s[20:21]"]),
                          ("ps_kill_mov", ["s_mov_b64 exec, s[20:21]"]),
                          ("ps_kill_wqm_exec", ["s_wqm_b64 exec, s[20:21]"])):
            self.tier(name, kill(KILL, run + ["v_add_nc_u32_e32 v6, 3, v0"], EXIT_PS), "A", [], [])
        self.tier("ps_kill_exit0", kill(KILL, RUN_WQM, ["s_mov_b64 exec, 0"] + EXIT_PS[1:]), "A", [], [])

    def test_a_kill_then_if_endif(self):
        # a saveexec / restore pair after the narrowing restores a mask taken inside it: EXEC stays in the quads
        self.tier("ps_kill_if", kill(KILL, RUN_WQM + [
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execz 1",
            "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[12:13]"], EXIT_PS), "A", [], [])

    def test_a_kill_saveexec_first(self):
        # the run side's first narrowing is a SAVEEXEC: EXEC = wqm mask & EXEC (inside), s[30:31] = the old EXEC
        self.tier("ps_kill_saveexec", kill(KILL, ["s_wqm_b64 s[22:23], s[20:21]", "s_and_saveexec_b64 s[30:31], "
                                                  "s[22:23]", "v_add_nc_u32_e32 v6, 3, v0"], EXIT_PS), "A", [], [])

    # -- near misses: tier C unsafe-vote-branch
    def test_c_exit_with_code_of_its_own(self):
        for name, exit_ in (
                ("ps_kill_exit_vmov", ["v_mov_b32_e32 v6, 1"] + EXIT_PS),
                ("ps_kill_exit_exp_first", ["exp mrt0 v6, v7, v5, v5 done vm", "s_mov_b64 exec, s[20:21]",
                                            "s_endpgm"]),
                ("ps_kill_exit_store", EXIT_PS[:1] + ["buffer_store_dword v6, v5, s[0:3], 0 offen"] + EXIT_PS[1:]),
                ("ps_kill_exit_sgpr", EXIT_PS[:1] + ["s_mov_b32 s5, 1"] + EXIT_PS[1:]),
                ("ps_kill_exit_other_mask", ["s_mov_b64 exec, s[10:11]"] + EXIT_PS[1:]),
                ("ps_kill_exit_jumps_back", EXIT_PS[:2] + ["s_branch -8"])):      # into the run side's s_and
            self.tier(name, kill(KILL, RUN_WQM, exit_), "C", contains=["unsafe-vote-branch"])

    def test_c_run_side_does_not_narrow(self):
        self.tier("ps_kill_run_code", kill(KILL, ["v_mov_b32_e32 v6, 5"] + RUN_WQM, EXIT_PS),
                  "C", contains=["unsafe-vote-branch"])
        self.tier("ps_kill_run_other", kill(KILL, ["s_and_b64 exec, exec, s[10:11]"], EXIT_PS),
                  "C", contains=["unsafe-vote-branch"])

    def test_c_saveexec_that_does_not_narrow(self):
        # s_or_saveexec: EXEC = mask | EXEC, not inside; restoring the saved (old) EXEC widens again
        self.tier("ps_kill_or_saveexec", kill(KILL, ["s_wqm_b64 s[22:23], s[20:21]", "s_or_saveexec_b64 s[30:31], "
                                                     "s[22:23]", "v_add_nc_u32_e32 v6, 3, v0"], EXIT_PS),
                  "C", contains=["unsafe-vote-branch"])
        self.tier("ps_kill_saveexec_restore", kill(KILL, ["s_wqm_b64 s[22:23], s[20:21]", "s_and_saveexec_b64 "
                                                          "s[30:31], s[22:23]", "s_mov_b64 exec, s[30:31]",
                                                          "v_add_nc_u32_e32 v6, 3, v0"], EXIT_PS),
                  "C", contains=["unsafe-vote-branch"])

    def test_c_run_side_widens_again(self):
        self.tier("ps_kill_widen", kill(KILL, RUN_WQM + ["s_mov_b64 exec, s[10:11]", "v_mov_b32_e32 v6, 9"],
                                        EXIT_PS), "C", contains=["unsafe-vote-branch"])
        self.tier("ps_kill_or", kill(KILL, RUN_WQM + ["s_or_b64 exec, exec, s[10:11]"], EXIT_PS),
                  "C", contains=["unsafe-vote-branch"])

    def test_c_mask_redefined_or_not_the_vote(self):
        # the mask is rewritten after the vote (s_mov leaves SCC alone): the exit reads another value. Since
        # SHADER-820 (#4620) the new value ends SCC's tag as a vote on that mask, so the branch is a tainted one.
        self.tier("ps_kill_redef", kill(KILL + ["s_mov_b64 s[20:21], exec"], RUN_WQM, EXIT_PS),
                  "C", contains=["tainted-branch"])
        # SCC = (mask == 0): the branch to the "exit" is taken when the mask is not empty
        self.tier("ps_kill_cmp_eq", kill(KILL + ["s_cmp_eq_u64 s[20:21], 0"], RUN_WQM, EXIT_PS),
                  "C", contains=["unsafe-vote-branch"])

    def test_c_vm_export_before_the_vote(self):
        self.tier("ps_kill_vm_first", kill(["exp mrt1 v6, v7, v5, v5 vm"] + KILL, RUN_WQM, EXIT_PS),
                  "C", contains=["unsafe-vote-branch"])

    def test_shared_exit_of_two_kills_stays_c(self):
        # conservative: the exit sets EXEC from a live mask the second kill redefined behind the first vote (stale)
        second = ["v_cmp_gt_u32_e32 vcc, 50, v0", "s_andn2_b64 s[20:21], s[20:21], vcc"]
        run = RUN_WQM + second + ["s_cbranch_scc0 5"] + RUN_WQM
        self.tier("ps_kill_shared", kill(KILL, run, EXIT_PS), "C", contains=["tainted-exec"])

    def test_mov_of_a_mask_is_not_a_vote(self):
        # s_mov_b64 writes no SCC: the branch reads the lane-count compare before it, not a vote on the copy
        self.tier("ps_mov_scc", ps([
            "v_cmp_eq_u32_e32 vcc, 1, v0", "s_bcnt1_i32_b64 s6, vcc", "s_cmp_eq_u32 s6, 5",
            "s_mov_b64 s[24:25], vcc", "s_cbranch_scc0 3", "s_and_saveexec_b64 s[16:17], s[24:25]",
            "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[16:17]"]), "C", contains=["lane-count"])

    # -- execution: vertex shaders, 64 lanes against two halves of 32 (vertices < 40 killed: all of the first half)
    def test_kill_exit_invariant(self):
        tier, whole, halves = self.run_both("vs_kill", kill(KILL, RUN_WQM + [
            "v_mul_u32_u24_e32 v6, 3, v0", "v_mov_b32_e32 v7, 1"], EXIT_VS, "vs"))
        self.assertEqual(tier, "A")
        self.assertEqual(whole, halves)
        # a vertex that is off writes nothing: the interpreter's outputs start at zero
        self.assertEqual([w[:2] for w in whole], [(0, 0)] * 40 + [(3 * k, 1) for k in range(40, 64)])

    def test_kill_exit_saveexec_invariant(self):
        tier, whole, halves = self.run_both("vs_kill_saveexec", kill(KILL, [
            "s_wqm_b64 s[22:23], s[20:21]", "s_and_saveexec_b64 s[30:31], s[22:23]",
            "v_mul_u32_u24_e32 v6, 3, v0", "v_mov_b32_e32 v7, 1"], EXIT_VS, "vs"))
        self.assertEqual(tier, "A")
        self.assertEqual(whole, halves)
        self.assertEqual([w[:2] for w in whole], [(0, 0)] * 40 + [(3 * k, 1) for k in range(40, 64)])

    def test_kill_exit_that_widens_differs_and_is_c(self):
        tier, whole, halves = self.run_both("vs_kill_widen", kill(KILL, RUN_WQM + [
            "s_mov_b64 exec, s[10:11]", "v_mul_u32_u24_e32 v6, 3, v0", "v_mov_b32_e32 v7, 1"], EXIT_VS, "vs"))
        self.assertEqual(tier, "C")
        self.assertEqual([w[:2] for w in whole], [(3 * k, 1) for k in range(64)])
        self.assertEqual([w[:2] for w in halves[:32]], [(0, 0)] * 32)      # the first half took the exit


class AddTidBuffer(unittest.TestCase):
    """A MUBUF/MTBUF access with no idxen through a literal V# whose word3 has ADD_TID_ENABLE (bit 23) takes the
    wave lane as its index (idxen ignores the bit), so what it reads or writes depends on the lane id: tier C
    `lane-id`. A literal with the bit clear and any access with idxen are tier A. (A run-time V# is assumed to
    have the bit clear: docs/design.md, SHADER-10.)"""
    TID, NO_TID = 0x31816fac, 0x31016fac

    def v_sharp(self, word3):
        # SWIZZLE_ENABLE, stride 4, 8-index swizzle: with ADD_TID_ENABLE, lane i reads the dword at base + 4 * i
        return ["s_mov_b32 s0, 0x1000", "s_mov_b32 s1, 0x80040000", "s_mov_b32 s2, 0x1000",
                f"s_mov_b32 s3, {word3:#x}", "v_mov_b32_e32 v9, 0"]

    def program(self, word3, access):
        return vs(self.v_sharp(word3) + [access, "s_waitcnt vmcnt(0)"])

    def test_literal_add_tid_without_idxen_is_lane_dependent(self):
        for load in ("buffer_load_dword v6, off, s[0:3], 0", "buffer_load_format_x v6, off, s[0:3], 0"):
            lines = self.program(self.TID, load)
            (tier, reasons, _), _ = meta_of("vs_add_tid", lines)
            self.assertEqual((tier, "lane-id" in reasons), ("C", True), load)

    def test_literal_add_tid_clear_is_tier_a(self):
        lines = self.program(self.NO_TID, "buffer_load_dword v6, off, s[0:3], 0")
        (tier, reasons, _), _ = meta_of("vs_no_tid", lines)
        self.assertEqual((tier, reasons), ("A", []))

    def test_idxen_ignores_add_tid(self):
        lines = self.program(self.TID, "buffer_load_dword v6, v9, s[0:3], 0 idxen")
        (tier, reasons, _), _ = meta_of("vs_idxen_tid", lines)
        self.assertEqual((tier, reasons), ("A", []))

    def test_literal_copied_through_another_sgpr_still_counts(self):
        lines = vs(self.v_sharp(self.NO_TID)[:3] + ["s_mov_b32 s20, 0x31816fac", "s_mov_b32 s3, s20",
                   "buffer_load_dword v6, off, s[0:3], 0"])
        self.assertEqual(meta_of("vs_tid_copy", lines)[0][0], "C")

    def test_word3_built_by_other_scalar_ops_counts(self):
        # rdna2spirv folds these into the literal word3; the analysis must see the same bit
        head = ["s_mov_b32 s0, 0x1000", "s_mov_b32 s1, 0x80040000", "s_mov_b32 s2, 0x1000"]
        load = "buffer_load_dword v6, off, s[0:3], 0"
        for name, build, tier in (
                ("mov64_set", ["s_mov_b64 s[2:3], 0x31816fac00001000"], "C"),
                ("mov64_clear", ["s_mov_b64 s[2:3], 0x31016fac00001000"], "A"),
                ("mov64_pair_copy", ["s_mov_b64 s[20:21], 0x31816fac00001000", "s_mov_b64 s[2:3], s[20:21]"], "C"),
                ("pack_set", ["s_pack_ll_b32_b16 s3, 0x6fac, 0x3181"], "C"),
                ("pack_clear", ["s_pack_ll_b32_b16 s3, 0x6fac, 0x3101"], "A"),
                ("bfm_set", ["s_bfm_b32 s3, 9, 23"], "C"),
                ("bfm_clear", ["s_bfm_b32 s3, 3, 12"], "A"),
                ("movk_clear", ["s_movk_i32 s3, 0x1234"], "A")):
            lines = vs(head + (["s_mov_b32 s3, 0"] if name.startswith("mov64") else []) + build
                       + ["v_mov_b32_e32 v9, 0", load, "s_waitcnt vmcnt(0)"])
            self.assertEqual(meta_of("vs_w3_" + name, lines)[0][0], tier, name)

    def test_offen_tbuffer_and_store_format_follow_the_bit(self):
        for access in ("buffer_load_dword v6, v9, s[0:3], 0 offen",
                       "tbuffer_load_format_x v6, off, s[0:3], 0 format:[BUF_FMT_32_FLOAT]",
                       "buffer_store_format_x v6, off, s[0:3], 0"):
            (tier, reasons, _), _ = meta_of("vs_tid_variant", self.program(self.TID, access))
            self.assertEqual((tier, "lane-id" in reasons), ("C", True), access)
            (tier, reasons, _), _ = meta_of("vs_notid_variant", self.program(self.NO_TID, access))
            self.assertEqual((tier, reasons), ("A", []), access)

    def test_stores_and_atomics_by_lane_are_tier_c(self):
        for access in ("buffer_store_dword v6, off, s[0:3], 0", "buffer_atomic_add v6, off, s[0:3], 0"):
            (tier, reasons, _), _ = meta_of("vs_tid_write", self.program(self.TID, access))
            self.assertEqual((tier, "lane-id" in reasons), ("C", True), access)
            (tier, _, _), _ = meta_of("vs_notid_write", self.program(self.NO_TID, access))
            self.assertEqual(tier, "A", access)

    def test_run_time_v_sharp_is_assumed_to_have_the_bit_clear(self):
        # the word3 comes from user data: nothing in the code says the bit is set (design.md, SHADER-10)
        lines = vs(["buffer_load_dword v6, off, s[4:7], 0", "s_waitcnt vmcnt(0)"])
        self.assertEqual(meta_of("vs_runtime_vsharp", lines)[0][0], "A")


class Refusal(unittest.TestCase):
    def test_error_mode_refuses_tier_c(self):
        old = rdna2spirv.WAVE64_GRAPHICS
        rdna2spirv.WAVE64_GRAPHICS = "error"
        try:
            with self.assertRaises(rdna2spirv.Unsupported) as cm:
                translate_listing("ps_c_cnt", listing("ps_c_cnt", ps(["v_cmp_eq_u32_e32 vcc, 1, v0",
                                                                      "s_bcnt1_i32_b64 s6, vcc",
                                                                      "v_mov_b32_e32 v6, s6"])))
            self.assertIn("lane-count", str(cm.exception))
        finally:
            rdna2spirv.WAVE64_GRAPHICS = old

    def test_off_mode_has_no_tier(self):
        old = rdna2spirv.WAVE64_GRAPHICS
        rdna2spirv.WAVE64_GRAPHICS = "off"
        try:
            translate_listing("ps_c_cnt", listing("ps_c_cnt", ps(["v_cmp_eq_u32_e32 vcc, 1, v0",
                                                                  "s_bcnt1_i32_b64 s6, vcc", "v_mov_b32_e32 v6, s6"])))
            self.assertIsNone(rdna2spirv.translate.last_gen.w64g)
        finally:
            rdna2spirv.WAVE64_GRAPHICS = old


class GeometryRing(unittest.TestCase):
    """TEST-251: a legacy GS's LDS write is an output slot from its offset alone (rdna2spirv.gs_ds drops the VGPR
    address, the lane's GSVS area), so a lane id that only forms that address is no partition dependence; the data
    written, and a lane id in a memory store's address, still are. GPU twins: gs_w64_a_ring, gs_w64_c_lane_id."""
    PRO = ["s_bfe_i32 s0, s3, 0x80008", "s_bfm_b64 exec, s0, 0", "s_cmp_eq_u32 s0, 64", "s_cselect_b64 exec, -1, exec",
           "v_mbcnt_lo_u32_b32 v13, -1, 0", "v_mbcnt_hi_u32_b32 v13, -1, v13"]
    EPI = ["s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)", "s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)",
           "s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)", "s_endpgm"]

    def tier(self, body):
        sys.path.insert(0, os.path.join(HERE, "..", "..", "runtime", "test", "shaders"))
        import build                                       # gs_container: gs_tri's ISG1/OSG1
        name = "gs_w64_ring_test"
        translate_listing(name, listing(name, self.PRO + body + self.EPI), build.gs_container())
        w = rdna2spirv.translate.last_gen.w64g
        return w["tier"], w["reasons"]

    def test_lane_id_in_the_ring_address_is_tier_a(self):
        self.assertEqual(self.tier(["v_mul_u32_u24_e32 v12, 0x60, v13", "v_mov_b32_e32 v14, 0",
                                    "ds_write2_b32 v12, v14, v14 offset0:0 offset1:1"]), ("A", []))

    def test_lane_id_written_to_the_ring_is_tier_c(self):
        self.assertEqual(self.tier(["v_mov_b32_e32 v12, 0", "ds_write2_b32 v12, v13, v13 offset0:0 offset1:1"]),
                         ("C", ["lane-id"]))

    def test_lane_id_in_a_memory_store_address_is_tier_c(self):
        self.assertEqual(self.tier(["v_mov_b32_e32 v14, 0", "v_lshlrev_b32_e32 v12, 2, v13",
                                    "buffer_store_dword v14, v12, s[4:7], 0 offen",
                                    "ds_write2_b32 v14, v14, v14 offset0:0 offset1:1"]), ("C", ["lane-id"]))




def _ps_xy(lane):
    """8x8 2x2-quad pixel coordinates per lane: lane 4k+{0,1,2,3} is TL, TR, BL, BR."""
    q, c = divmod(lane, 4)
    return 2 * (q % 4) + (c & 1), 2 * (q // 4) + (c >> 1)


_SWIZZLE_OFFSETS = {
    "offset:0x801b": "offset:swizzle(QUAD_PERM,3,2,1,0)",
    "offset:0x41f": "offset:swizzle(SWAP,1)",
    "offset:0x401f": "offset:swizzle(SWAP,16)",
    "offset:0x1c": "offset:swizzle(BROADCAST,4,0)",
}


def _load_ps_w64_lines(name):
    asm_path = os.path.join(os.path.dirname(HERE), "..", "runtime", "test", "shaders", f"{name}.asm")
    out = []
    with open(asm_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            for hex_off, sym_off in _SWIZZLE_OFFSETS.items():
                line = line.replace(hex_off, sym_off)
            out.append(line)
    return out


def _run_ps_cut(asm, wave, push=None, memory=None):
    words = []
    for half in range(64 // wave):
        base = half * wave
        res = wi.run_wave(
            asm,
            wave=wave,
            push_constants=push,
            inputs={0: lambda lane, b=base: ((_ps_xy(b + lane)[0] + 0.5) / 64.0,
                                             (_ps_xy(b + lane)[1] + 0.5) / 64.0, 0.0, 0.0)},
            memory=memory,
        )
        for o in res["outputs"]:
            if o is None:                 # a killed lane exports nothing; the attachment keeps its clear value
                words.append(None)
                continue
            mrt = o["%out_mrt0"]
            words.append(tuple(v.bits if isinstance(v, wi.F) else v for v in mrt))
    return words


class PixelShaders(unittest.TestCase):
    """TEST-722: run every runtime/test/shaders/ps_w64_*.asm fixture on spirv_wave_interp (wave 64 vs 2x32)."""

    def test_ps_w64_fixtures(self):
        sys.path.insert(0, os.path.join(HERE, "..", "..", "runtime", "test", "shaders"))
        import w64_graphics as wg

        ps_names = sorted(n for n in wg.FIXTURES if n.startswith("ps_w64_"))
        self.assertGreaterEqual(len(ps_names), 17)   # 17 when TEST-722 landed; new fixtures run here too
        for name in ps_names:
            with self.subTest(fixture=name):
                ref_tier, ref_fn, env = wg.FIXTURES[name]
                (tier, _, _), asm = meta_of(name, _load_ps_w64_lines(name))
                self.assertEqual(tier, ref_tier)
                push = [int(x, 0) for x in env["VKH_SGPRS"].split(",")] if "VKH_SGPRS" in env else None
                mem64 = {0: 0} if "VKH_HEAP" in env else None
                mem32 = {0: 0} if "VKH_HEAP" in env else None
                w64 = _run_ps_cut(asm, 64, push=push, memory=mem64)
                for lane in range(64):
                    x, y = _ps_xy(lane)
                    exp = ref_fn(y * 64 + x)
                    if w64[lane] is None:
                        self.assertEqual(exp, (wg.CLEAR,) * 4, f"{name} wave64 lane {lane} was killed")
                        continue
                    for c in range(4):
                        if exp[c] is not None:
                            self.assertEqual(w64[lane][c], exp[c], f"{name} wave64 lane {lane} c{c}")
                if name == "ps_w64_c_readlane":
                    with self.assertRaises(wi.UnsupportedConstruct):
                        _run_ps_cut(asm, 32, push=push)
                    continue
                w32 = _run_ps_cut(asm, 32, push=push, memory=mem32)
                if tier in ("A", "B"):
                    self.assertEqual(w32, w64)
                else:
                    self.assertNotEqual(w32, w64)
                if "VKH_HEAP" in env:
                    self.assertEqual(mem64[0], 16)
                    self.assertEqual(mem32[0], 16)
                    slots64 = {mem64[0x100 + 4 * k] for k in range(16)}
                    slots32 = {mem32[0x100 + 4 * k] for k in range(16)}
                    self.assertEqual(len(slots64), 16)
                    self.assertEqual(slots64, slots32)


if __name__ == "__main__":
    unittest.main()
