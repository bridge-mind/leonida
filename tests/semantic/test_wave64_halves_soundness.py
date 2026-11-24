"""Soundness regression tests for tools/wave64_halves.py.

Each regression checks a bug where an unsafe wave64 VS/PS/GS program used to pass as Tier A or B
(or lose a genuine vote SCC), together with clean control cases that must remain Tier A or B.
Vertex-shader cases using integer arithmetic also execute on spirv_wave_interp.py as one 64-lane
wave vs two 32-lane waves to prove the two runs actually diverge (or agree for clean controls).

    python -I tests/semantic/test_wave64_halves_soundness.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (sets the tools path)
import rdna2spirv  # noqa: E402
from test_wave64_halves import (  # noqa: E402
    AGG_HEAD,
    AGG_RANK,
    meta_of,
    out_param,
    ps,
    run_cut,
    vs,
)

ATOMIC = ["v_mov_b32_e32 v11, s6", "v_mov_b32_e32 v12, 0",
          "buffer_atomic_add v11, v12, s[0:3], 0 offen glc",
          "s_or_b64 exec, exec, s[14:15]"]
SLOT = ["v_readfirstlane_b32 s7, v11", "v_add_nc_u32_e32 v13, s7, {rank}",
        "v_lshlrev_b32_e32 v13, 2, v13",
        "buffer_store_dword v0, v13, s[0:3], 0 offen offset:256"]
ELECT_NARROWED = ["v_cmp_eq_u32_e32 vcc, 0, v10", "s_and_saveexec_b64 s[14:15], vcc"]


class SoundnessRegressions(unittest.TestCase):
    def assert_tier_c_and_diverges(self, name, body, contains=None, push=None):
        lines = vs(body)
        (tier, reasons, _), asm = meta_of(name, lines)
        self.assertEqual(tier, "C", f"{name}: expected tier C, got {tier} (reasons={reasons})")
        if contains is not None:
            self.assertTrue(set(contains) <= set(reasons), f"{name}: reasons {reasons} lack {contains}")
        whole = out_param(run_cut(asm, 64, push))
        halves = out_param(run_cut(asm, 32, push))
        self.assertNotEqual(whole, halves, f"{name}: expected 1x64 and 2x32 outputs to diverge")

    def assert_ps_tier(self, name, body, tier, reasons=None, contains=None, idioms=None):
        (got_tier, got_reasons, got_idioms), _ = meta_of(name, ps(body))
        self.assertEqual(got_tier, tier, f"{name}: got tier {got_tier}, reasons={got_reasons}, idioms={got_idioms}")
        if reasons is not None:
            self.assertEqual(sorted(got_reasons), sorted(reasons), name)
        if contains is not None:
            self.assertTrue(set(contains) <= set(got_reasons), f"{name}: reasons {got_reasons} lack {contains}")
        if idioms is not None:
            self.assertEqual(sorted(got_idioms), sorted(idioms), name)

    # -- RMW destinations and s_pack
    def test_s_addk_i32_preserves_lane_count_taint(self):
        self.assert_tier_c_and_diverges("vs_rmw_addk", [
            "v_cmp_lt_u32_e32 vcc, 4, v0",
            "s_bcnt1_i32_b64 s6, vcc",
            "s_addk_i32 s6, 1",
            "v_mov_b32_e32 v6, s6",
        ], contains=["lane-count"])

    def test_s_mulk_i32_preserves_lane_count_taint(self):
        self.assert_tier_c_and_diverges("vs_rmw_mulk", [
            "v_cmp_lt_u32_e32 vcc, 4, v0",
            "s_bcnt1_i32_b64 s6, vcc",
            "s_mulk_i32 s6, 3",
            "v_mov_b32_e32 v6, s6",
        ], contains=["lane-count"])

    def test_s_bitset_preserves_lane_count_taint(self):
        self.assert_tier_c_and_diverges("vs_rmw_bitset1", [
            "v_cmp_lt_u32_e32 vcc, 4, v0",
            "s_bcnt1_i32_b64 s6, vcc",
            "s_bitset1_b32 s6, 7",
            "v_mov_b32_e32 v6, s6",
        ], contains=["lane-count"])
        self.assert_tier_c_and_diverges("vs_rmw_bitset0", [
            "v_cmp_lt_u32_e32 vcc, 4, v0",
            "s_bcnt1_i32_b64 s6, vcc",
            "s_bitset0_b32 s6, 0",
            "v_mov_b32_e32 v6, s6",
        ], contains=["lane-count"])

    def test_s_cmovk_i32_reads_vote_scc(self):
        self.assert_tier_c_and_diverges("vs_rmw_cmovk", [
            "v_cmp_eq_u32_e32 vcc, 1, v0",
            "s_cmp_lg_u64 vcc, 0",
            "s_mov_b32 s5, 0",
            "s_cmovk_i32 s5, 9",
            "v_mov_b32_e32 v6, s5",
        ], contains=["cross-lane"])

    def test_s_cmov_b32_keeps_tainted_dst_when_scc_false(self):
        self.assert_tier_c_and_diverges("vs_rmw_cmov_dst", [
            "v_cmp_lt_u32_e32 vcc, 4, v0",
            "s_bcnt1_i32_b64 s6, vcc",
            "s_mov_b32 s4, 0",
            "s_cmp_eq_u32 s4, 1",
            "s_cmov_b32 s6, 9",
            "v_mov_b32_e32 v6, s6",
        ], contains=["lane-count"])

    def test_s_pack_is_not_no_dest(self):
        self.assert_tier_c_and_diverges("vs_spack", [
            "v_cmp_lt_u32_e32 vcc, 4, v0",
            "s_bcnt1_i32_b64 s6, vcc",
            "s_pack_ll_b32_b16 s5, s6, 1",
            "v_mov_b32_e32 v6, s5",
        ], contains=["lane-count"])

    def test_s_pack_behind_vote_is_tracked_as_stale_scalar(self):
        self.assert_tier_c_and_diverges("vs_spack_stale", [
            "v_cmp_eq_u32_e32 vcc, 1, v0",
            "s_mov_b32 s5, 7",
            "s_cbranch_vccz 1",
            "s_pack_ll_b32_b16 s5, 9, 2",
            "v_mov_b32_e32 v6, s5",
        ], contains=["stale-scalar"])

    def test_v_dot4c_i32_i8_preserves_lane_id_taint(self):
        self.assert_tier_c_and_diverges("vs_rmw_dot4c", [
            "v_mbcnt_lo_u32_b32 v6, -1, 0",
            "v_mbcnt_hi_u32_b32 v6, -1, v6",
            "v_dot4c_i32_i8 v6, v5, v5",
        ], contains=["lane-id"])

    def test_v_fmac_f32_on_lane_id_does_not_become_uniform(self):
        self.assert_ps_tier("ps_rmw_fmac_uniform", [
            "v_mbcnt_lo_u32_b32 v6, -1, 0",
            "v_mbcnt_hi_u32_b32 v6, -1, v6",
            "v_fmac_f32_e32 v6, 0, v5",
            "v_readfirstlane_b32 s5, v6",
            "v_mov_b32_e32 v6, s5",
        ], "C", contains=["lane-id"])

    def test_v_swap_b32_swaps_taint_into_exported_vgpr(self):
        self.assert_tier_c_and_diverges("vs_vswap", [
            "v_mbcnt_lo_u32_b32 v7, -1, 0",
            "v_mbcnt_hi_u32_b32 v7, -1, v7",
            "v_swap_b32 v6, v7",
        ], contains=["lane-id"])

    # -- SCC-preserving scalar ops
    def test_s_mul_i32_does_not_clear_vote_scc(self):
        self.assert_tier_c_and_diverges("vs_scc_mul", [
            "v_cmp_eq_u32_e32 vcc, 1, v0",
            "s_cmp_lg_u64 vcc, 0",
            "s_mul_i32 s8, 3, 4",
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32_e32 v6, s5",
        ], contains=["cross-lane"])

    def test_s_ff1_i32_b64_does_not_turn_tainted_scc_into_exec_vote(self):
        self.assert_tier_c_and_diverges("vs_ff1_preserves_scc", [
            "v_cmp_lt_u32_e32 vcc, 4, v0",
            "s_bcnt1_i32_b64 s6, vcc",
            "s_cmp_eq_u32 s6, 59",
            "s_ff1_i32_b64 s7, exec",
            "s_cbranch_scc0 1",
            "v_mov_b32_e32 v6, 9",
        ], contains=["lane-count"])

    # -- s_bfm_b64 uses per-half wave_lane (0..31)
    def test_s_bfm_b64_lane_mask_is_lane_id_tainted(self):
        self.assert_tier_c_and_diverges("vs_sbfm64", [
            "s_bfm_b64 s[14:15], 4, 0",
            "v_cndmask_b32_e64 v6, 0, 1, s[14:15]",
        ], contains=["lane-id"])

    # -- Dynamic register indexing (movrel)
    def test_v_movrels_b32_with_tainted_m0(self):
        self.assert_tier_c_and_diverges("vs_vmovrels_m0", [
            "v_and_b32_e32 v4, 15, v0",
            "v_cmp_eq_u32_e32 vcc, 1, v4",
            "s_bcnt1_i32_b64 s6, vcc",
            "v_mov_b32_e32 v10, 10",
            "v_mov_b32_e32 v12, 12",
            "s_mov_b32 m0, s6",
            "v_movrels_b32 v6, v8",
        ], contains=["lane-count"])

    def test_s_movrels_b32_reads_tainted_and_stale_sgpr(self):
        self.assert_ps_tier("ps_smovrels_tainted", [
            "v_cmp_eq_u32_e32 vcc, 1, v0",
            "s_bcnt1_i32_b64 s9, vcc",
            "s_mov_b32 m0, 3",
            "s_movrels_b32 s5, s6",
            "v_mov_b32_e32 v6, s5",
        ], "C", contains=["lane-count"])
        lines = [
            "s_wqm_b64 exec, exec",
            "v_mov_b32_e32 v5, 0",
            "v_mov_b32_e32 v6, 0",
            "v_mov_b32_e32 v7, 0",
            "v_cmp_eq_u32_e32 vcc, 1, v0",
            "s_mov_b32 s9, 1",
            "s_cbranch_vccz 1",
            "s_mov_b32 s9, 2",
            "s_mov_b32 m0, 3",
            "s_movrels_b32 s5, s6",
            "v_mov_b32_e32 v6, s5",
            "exp mrt0 v6, v7, v5, v5 done vm",
            "s_endpgm",
        ]
        (got_tier, got_reasons, _), _ = meta_of("ps_smovrels_stale", lines)
        self.assertEqual(got_tier, "C", f"ps_smovrels_stale: got {got_tier}, reasons={got_reasons}")
        self.assertIn("stale-scalar", got_reasons)

    def test_v_movreld_b32_with_tainted_m0_or_src(self):
        self.assert_tier_c_and_diverges("vs_vmovreld_m0", [
            "v_and_b32_e32 v4, 15, v0",
            "v_cmp_eq_u32_e32 vcc, 1, v4",
            "s_bcnt1_i32_b64 s6, vcc",
            "s_sub_i32 s6, s6, 2",
            "s_mov_b32 m0, s6",
            "v_mov_b32_e32 v5, 9",
            "v_movreld_b32 v6, v5",
        ], contains=["lane-count"])
        self.assert_tier_c_and_diverges("vs_vmovreld_src", [
            "v_mbcnt_lo_u32_b32 v5, -1, 0",
            "v_mbcnt_hi_u32_b32 v5, -1, v5",
            "s_mov_b32 m0, 0",
            "v_movreld_b32 v6, v5",
        ], contains=["lane-id"])

    def test_v_movreld_b32_nonzero_offset_propagates_taint(self):
        self.assert_tier_c_and_diverges("vs_vmovreld_src_off1", [
            "v_mbcnt_lo_u32_b32 v8, -1, 0",
            "v_mbcnt_hi_u32_b32 v8, -1, v8",
            "s_mov_b32 m0, 1",
            "v_movreld_b32 v5, v8",
        ], contains=["lane-id"])
        w64g = rdna2spirv.translate.last_gen.w64g
        self.assertIn("exp", [op for _, _, op in w64g["sites"]],
                      f"expected downstream exp sink to observe propagated taint: {w64g['sites']}")

    def test_v_add_co_ci_u32_reads_implicit_vcc(self):
        self.assert_tier_c_and_diverges("vs_add_co_ci_vcc", [
            "v_mbcnt_lo_u32_b32 v4, -1, 0",
            "v_mbcnt_hi_u32_b32 v4, -1, v4",
            "v_cmp_lt_u32_e32 vcc, 31, v4",
            "v_add_co_ci_u32_e32 v6, vcc, 0, v6",
        ], contains=["lane-id"])

    def test_v_swap_b32_under_narrowed_exec_is_not_uniform(self):
        self.assert_tier_c_and_diverges("vs_vswap_masked_uniform", [
            "v_mov_b32_e32 v5, v0",
            "v_mov_b32_e32 v7, 11",
            "v_cmp_eq_u32_e32 vcc, 0, v0",
            "s_and_saveexec_b64 s[12:13], vcc",
            "v_swap_b32 v5, v7",
            "s_mov_b64 exec, s[10:11]",
            "v_readfirstlane_b32 s20, v5",
            "v_mov_b32_e32 v6, s20",
        ], contains=["cross-lane"])

    def test_s_movrels_b32_reads_mask_sgpr_as_data(self):
        self.assert_tier_c_and_diverges("vs_smovrels_mask", [
            "s_mov_b32 m0, 5",
            "s_movrels_b32 s5, s6",
            "v_mov_b32_e32 v6, s5",
        ], contains=["cross-lane"])

    # -- saveexec / wrexec / s_andn2_b64 / _b32 mask-half
    def test_s_andn1_saveexec_on_elected_mask_is_refused(self):
        body = (AGG_HEAD + ["s_bcnt1_i32_b64 s6, exec"] + AGG_RANK
                + ["v_cmp_eq_u32_e32 vcc, 0, v10", "s_andn1_saveexec_b64 s[14:15], vcc"]
                + ATOMIC + [x.format(rank="v10") for x in SLOT] + ["s_mov_b64 exec, s[10:11]"])
        self.assert_ps_tier("ps_andn1_saveexec_elect", body, "C", contains=["tainted-exec"])

    def test_s_andn2_saveexec_flips_out_of_elected_region(self):
        body = (AGG_HEAD + ["s_bcnt1_i32_b64 s6, exec"] + AGG_RANK + ELECT_NARROWED
                + ["s_andn2_saveexec_b64 s[16:17], s[14:15]"]
                + ATOMIC + [x.format(rank="v10") for x in SLOT] + ["s_mov_b64 exec, s[10:11]"])
        self.assert_ps_tier("ps_andn2_saveexec_flip", body, "C", contains=["lane-count"])

    def test_s_andn1_wrexec_on_elected_mask_is_refused(self):
        body = (AGG_HEAD + ["s_bcnt1_i32_b64 s6, exec"] + AGG_RANK
                + ["v_cmp_eq_u32_e32 vcc, 0, v10", "s_andn1_wrexec_b64 s[14:15], vcc"]
                + ATOMIC + [x.format(rank="v10") for x in SLOT] + ["s_mov_b64 exec, s[10:11]"])
        self.assert_ps_tier("ps_andn1_wrexec_elect", body, "C", contains=["tainted-exec"])

    def test_s_andn2_wrexec_writes_new_exec_not_old(self):
        self.assert_tier_c_and_diverges("vs_andn2_wrexec_saved", [
            "v_mov_b32_e32 v5, v0",
            "v_cmp_eq_u32_e32 vcc, 0, v0",
            "s_andn2_wrexec_b64 s[12:13], vcc",
            "s_mov_b64 exec, s[12:13]",
            "v_mov_b32_e32 v5, 7",
            "s_mov_b64 exec, s[10:11]",
            "v_readfirstlane_b32 s20, v5",
            "v_mov_b32_e32 v6, s20",
        ], contains=["cross-lane"])

    def test_s_andn1_wrexec_writes_new_exec_not_old(self):
        self.assert_tier_c_and_diverges("vs_andn1_wrexec_saved", [
            "v_mov_b32_e32 v5, v0",
            "v_cmp_eq_u32_e32 vcc, 0, v0",
            "s_andn1_wrexec_b64 s[12:13], vcc",
            "s_mov_b64 exec, s[12:13]",
            "v_mov_b32_e32 v5, 7",
            "s_mov_b64 exec, s[10:11]",
            "v_readfirstlane_b32 s20, v5",
            "v_mov_b32_e32 v6, s20",
        ], contains=["cross-lane"])

    def test_s_and_saveexec_b32_is_mask_half(self):
        self.assert_ps_tier("ps_saveexec_b32", [
            "s_mov_b32 s14, 1",
            "s_and_saveexec_b32 s12, s14",
        ], "C", contains=["mask-half"])

    def test_s_andn2_b64_exec_rules(self):
        self.assert_ps_tier("ps_andn2_exec_elect", (
            AGG_HEAD + AGG_RANK + ["v_cmp_eq_u32_e32 vcc, 0, v10", "s_andn2_b64 exec, exec, vcc",
                                   "s_mov_b64 exec, s[10:11]"]
        ), "C", contains=["tainted-exec"])
        self.assert_ps_tier("ps_andn2_exec_flip", (
            AGG_HEAD + ["s_bcnt1_i32_b64 s6, exec"] + AGG_RANK + ELECT_NARROWED
            + ["s_andn2_b64 exec, s[14:15], exec"]
            + ATOMIC + [x.format(rank="v10") for x in SLOT] + ["s_mov_b64 exec, s[10:11]"]
        ), "C", contains=["lane-count"])

    def test_s_andn2_exec_in_vote_branch_is_unsafe(self):
        for tag, op in (("saveexec", "s_andn2_saveexec_b64"), ("wrexec", "s_andn2_wrexec_b64")):
            self.assert_tier_c_and_diverges(f"vs_vote_{tag}_andn2", [
                "v_cmp_eq_u32_e32 vcc, 1, v0",
                "s_cbranch_vccz 1",
                f"{op} s[14:15], vcc",
                "v_mov_b32_e32 v6, 9",
                "s_mov_b64 exec, s[10:11]",
            ], contains=["unsafe-vote-branch"])

    def test_s_andn1_exec_in_vote_branch_is_unsafe(self):
        for tag, op in (("saveexec", "s_andn1_saveexec_b64"), ("wrexec", "s_andn1_wrexec_b64")):
            self.assert_tier_c_and_diverges(f"vs_vote_{tag}_andn1", [
                "v_cmp_eq_u32_e32 vcc, 1, v0",
                "s_cbranch_vccz 3",
                f"{op} s[14:15], vcc",
                "v_mov_b32_e32 v6, 9",
                "s_mov_b64 exec, s[10:11]",
            ], contains=["unsafe-vote-branch"])

    # -- Clean controls (no false positives)
    # SHADER-820 (@shdann): a vote's SCC tag survived a new value in the voted mask register. v_cmp into the pair
    # or VCC and s_mov_b64 don't write SCC, so s_cbranch_scc0 votes on the old mask while the run side narrows
    # EXEC to the new one: one 64-lane wave writes v6 in lanes 20..63, two 32-lane waves skip the upper half.
    STALE_VOTE_HEAD = ["v_cmp_gt_u32_e32 vcc, 20, v0", "s_and_b64 s[20:21], exec, vcc", "s_cmp_lg_u64 s[20:21], 0"]
    STALE_VOTE_TAIL = ["s_cbranch_scc0 3", "s_and_b64 exec, exec, s[20:21]", "v_add_nc_u32_e32 v6, 3, v0",
                       "s_mov_b64 exec, s[10:11]"]

    def test_vote_on_a_rewritten_pair_is_unsafe(self):
        self.assert_tier_c_and_diverges("vs_stale_vote_vcmp", self.STALE_VOTE_HEAD + [
            "v_cmp_gt_u32_e64 s[20:21], 64, v0"] + self.STALE_VOTE_TAIL)
        self.assert_tier_c_and_diverges("vs_stale_vote_mov", self.STALE_VOTE_HEAD + [
            "s_mov_b64 s[20:21], exec"] + self.STALE_VOTE_TAIL)

    def test_vote_on_a_rewritten_vcc_is_unsafe(self):
        self.assert_tier_c_and_diverges("vs_stale_vote_vcc", [
            "v_cmp_gt_u32_e32 vcc, 20, v0", "s_and_b64 vcc, exec, vcc", "v_cmp_gt_u32_e32 vcc, 64, v0",
            "s_cbranch_scc0 3", "s_and_saveexec_b64 s[18:19], vcc", "v_add_nc_u32_e32 v6, 3, v0",
            "s_mov_b64 exec, s[18:19]"])

    def test_vote_on_an_unchanged_or_revoted_mask_stays_a(self):
        for name, body in (
                ("vs_vote_unchanged", self.STALE_VOTE_HEAD + self.STALE_VOTE_TAIL),
                # the second s_and_b64 writes s[20:21] and votes on it (SCC = new mask != 0): its tag is the new one
                ("vs_vote_revoted", self.STALE_VOTE_HEAD + ["v_cmp_gt_u32_e32 vcc, 40, v0",
                                                            "s_and_b64 s[20:21], exec, vcc"] + self.STALE_VOTE_TAIL)):
            with self.subTest(name=name):
                (tier, reasons, _), asm = meta_of(name, vs(body))
                self.assertEqual(tier, "A", f"{name}: got {tier}, reasons={reasons}")
                self.assertEqual(out_param(run_cut(asm, 64)), out_param(run_cut(asm, 32)), name)

    def test_clean_controls_stay_invariant(self):
        for name, body in (
                ("vs_clean_spack", ["s_pack_ll_b32_b16 s5, s4, 7", "v_mov_b32_e32 v6, s5"]),
                ("vs_clean_rmw", ["s_mov_b32 s5, 3", "s_addk_i32 s5, 4", "s_mulk_i32 s5, 2",
                                  "s_bitset1_b32 s5, 4", "s_cmp_eq_u32 s5, 14", "s_cmovk_i32 s5, 9",
                                  "v_mov_b32_e32 v6, s5"]),
                ("vs_clean_vswap", ["v_mbcnt_lo_u32_b32 v6, -1, 0", "v_mbcnt_hi_u32_b32 v6, -1, v6",
                                    "v_mov_b32_e32 v8, 11", "v_swap_b32 v6, v8"]),
                ("vs_clean_vmovrels", ["v_mov_b32_e32 v8, 5", "s_mov_b32 m0, 0", "v_movrels_b32 v6, v8"]),
                ("vs_clean_vmovreld", ["v_mov_b32_e32 v5, 9", "s_mov_b32 m0, 0", "v_movreld_b32 v6, v5"]),
                ("vs_clean_vmovreld_off1", ["v_mov_b32_e32 v8, 9", "s_mov_b32 m0, 1", "v_movreld_b32 v5, v8"]),
                ("vs_clean_andn2_saveexec", ["v_cmp_eq_u32_e32 vcc, 1, v0",
                                             "s_and_saveexec_b64 s[12:13], vcc",
                                             "v_mov_b32_e32 v6, 5",
                                             "s_andn2_saveexec_b64 s[12:13], s[12:13]",
                                             "v_mov_b32_e32 v6, 7",
                                             "s_mov_b64 exec, s[10:11]"]),
        ):
            lines = vs(body)
            (tier, reasons, _), asm = meta_of(name, lines)
            self.assertEqual(tier, "A", f"{name}: got {tier}, reasons={reasons}")
            self.assertEqual(out_param(run_cut(asm, 64)), out_param(run_cut(asm, 32)), name)

        lines = vs(["s_mov_b32 s8, 5", "s_mov_b32 m0, 0", "s_movrels_b32 s5, s8", "v_mov_b32_e32 v6, s5"],
                   prologue=["s_wqm_b64 exec, exec"])
        (tier, reasons, _), asm = meta_of("vs_clean_smovrels", lines)
        self.assertEqual(tier, "A", f"vs_clean_smovrels: got {tier}, reasons={reasons}")
        self.assertEqual(out_param(run_cut(asm, 64)), out_param(run_cut(asm, 32)), "vs_clean_smovrels")

        self.assert_ps_tier("ps_clean_fmac_uniform", [
            "v_mov_b32_e32 v5, s4",
            "v_fmac_f32_e32 v5, 0, v5",
            "v_readfirstlane_b32 s5, v5",
            "v_mov_b32_e32 v6, s5",
        ], "B", reasons=[], idioms=["readfirstlane-uniform"])


if __name__ == "__main__":
    unittest.main()
