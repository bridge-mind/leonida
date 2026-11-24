"""End-to-end translator tests for lane_masks operand invalidation and SDWA/DPP guards.

Each test translates a synthetic RDNA2 compute shader through tools/rdna2spirv.py and executes all
4 lanes via spirv_mini_interp.run_workgroup to confirm that lane_masks.py neither leaves stale
VGPR/parked-mask state nor forces write_all_lanes on SDWA/DPP ops nor wipes masks on stores.
"""
import struct
import unittest

import spirv_mini_interp as si
import support

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 4, 1, 1)  # 4 user SGPRs, workgroup id in s4, 4x1x1
FSSM = bytes(FSSM)

OUT_BASE = 0x1000
SCRATCH_BASE = 0x2000


def run_cs(name, asm, nlanes=4):
    spv = support.translate_lines(name, asm, fssm=FSSM)
    mem = {}
    si.run_workgroup(
        spv,
        nlanes,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=[OUT_BASE, 0, SCRATCH_BASE, 0] + [0] * 32,
        memory=mem,
    )
    return spv, [mem.get(OUT_BASE + 4 * i, 0) for i in range(nlanes)]


class LaneMasksEndToEndTests(unittest.TestCase):
    def test_v_mov_of_a_mask_keeps_inactive_lanes(self):
        """A plain v_mov_b32 of a mask SGPR is an ordinary VALU write: the lane that is off in EXEC keeps its VGPR."""
        _, got = run_cs("cs_vmov_mask_inactive", [
            "v_cmp_lt_u32_e64 s6, v0, 2",
            "v_mov_b32_e32 v2, 0xdeadbeef",
            "s_mov_b32 exec_lo, 7",
            "v_mov_b32_e32 v2, s6",
            "s_mov_b32 exec_lo, 0xf",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "global_store_dword v3, v2, s[0:1]",
            "s_endpgm",
        ])
        # Lane 3 was off while v2 was written. The active lanes read s6 as data, which needs #293's mask
        # materialization, so only the inactive lane is asserted.
        self.assertEqual(got[3], 0xdeadbeef)

    def test_dpp_bitwise_respects_row_mask_on_mask_operands(self):
        """v_or_b32_dpp must not enter mask_writes and bypass dpp_guard (row_mask:0x0 keeps old v2)."""
        _, got = run_cs("cs_dpp_mask", [
            "s_mov_b32 s6, exec_lo",
            "v_mov_b32_e32 v1, s6",
            "v_mov_b32_e32 v2, 0x1234",
            "v_or_b32_dpp v2, v1, v1 row_shr:1 row_mask:0x0 bank_mask:0xf",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "global_store_dword v3, v2, s[0:1]",
            "s_endpgm",
        ])
        self.assertEqual(got, [0x1234] * 4)

    def test_sdwa_bitwise_respects_exec_on_mask_operands(self):
        """v_or_b32_sdwa must not enter mask_writes and bypass EXEC on inactive lanes."""
        _, got = run_cs("cs_sdwa_mask", [
            "v_cmp_lt_u32_e64 s6, v0, 3",
            "v_mov_b32_e32 v2, 0xdeadbeef",
            "s_mov_b32 exec_lo, s6",
            "v_or_b32_sdwa v2, s6, s6 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_0 src1_sel:BYTE_0",
            "s_mov_b32 exec_lo, 0xf",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "global_store_dword v3, v2, s[0:1]",
            "s_endpgm",
        ])
        # Only the inactive lane tests this EXEC regression. Active lanes read s6 as
        # ordinary data, which needs mask materialization (#293); do not enshrine
        # the translator's per-lane 0xff representation instead of hardware's 0x07.
        self.assertEqual(got[3], 0xdeadbeef)

    def test_writelane_invalidates_vgpr_mask_before_readfirstlane(self):
        """Overwriting lane 0 of a mask-holding VGPR via v_writelane_b32 must end ("v", r) tracking."""
        _, got = run_cs("cs_wl_after_or", [
            "v_cmp_eq_u32_e64 s6, v0, 2",
            "v_or_b32_e32 v5, s6, s6",
            "s_mov_b32 s7, 0x55aa",
            "v_writelane_b32 v5, s7, 0",
            "v_readfirstlane_b32 s8, v5",
            "v_mov_b32_e32 v2, s8",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "global_store_dword v3, v2, s[0:1]",
            "s_endpgm",
        ])
        self.assertEqual(got, [0x55aa] * 4)

    def test_v_swap_b32_invalidates_second_operand_parked_mask(self):
        """v_swap_b32 writes a[1] as well as a[0]; reading back a[1] must not fetch a stale parked mask."""
        _, got = run_cs("cs_swap_parked", [
            "v_cmp_lt_u32_e64 s6, v0, 2",
            "v_writelane_b32 v5, s6, 0",
            "v_mov_b32_e32 v4, 0x7777",
            "v_swap_b32_e32 v4, v5",
            "v_readlane_b32 s7, v5, 0",
            "v_mov_b32_e32 v2, s7",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "global_store_dword v3, v2, s[0:1]",
            "s_endpgm",
        ])
        self.assertEqual(got, [0x7777] * 4)

    def test_v_movreld_and_v_swaprel_invalidate_parked_masks_above_base(self):
        """Relative VGPR writes can overwrite any VGPR >= base and must invalidate parked masks there."""
        _, got_movrel = run_cs("cs_movreld_parked", [
            "v_cmp_lt_u32_e64 s6, v0, 2",
            "v_writelane_b32 v6, s6, 0",
            "v_mov_b32_e32 v7, 0x8888",
            "s_mov_b32 m0, 2",
            "v_movreld_b32_e32 v4, v7",
            "v_readlane_b32 s7, v6, 0",
            "v_mov_b32_e32 v2, s7",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "global_store_dword v3, v2, s[0:1]",
            "s_endpgm",
        ])
        self.assertEqual(got_movrel, [0x8888] * 4)

        _, got_swaprel = run_cs("cs_swaprel_parked", [
            "v_cmp_lt_u32_e64 s6, v0, 2",
            "v_writelane_b32 v6, s6, 0",
            "v_mov_b32_e32 v4, 0x4444",
            "s_mov_b32 m0, 0x00000002",  # dst offset 0 (v4), src offset 2 (v4+2 = v6)
            "v_swaprel_b32_e32 v4, v4",
            "v_readlane_b32 s7, v6, 0",
            "v_mov_b32_e32 v2, s7",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "global_store_dword v3, v2, s[0:1]",
            "s_endpgm",
        ])
        self.assertEqual(got_swaprel, [0x4444] * 4)

    def test_store_and_non_returning_atomic_preserve_parked_mask_in_a0(self):
        """Using a VGPR as a[0] of global_store_dword or non-returning ds_add_u32 must keep its parked mask."""
        _, got_gstore = run_cs("cs_store_parked", [
            "v_cmp_lt_u32_e64 s6, v0, 3",
            "v_lshlrev_b32_e32 v5, 2, v0",
            "v_writelane_b32 v5, s6, 3",
            "v_mov_b32_e32 v6, 0x9999",
            "global_store_dword v5, v6, s[2:3]",
            "v_readlane_b32 s7, v5, 3",
            "s_bcnt1_i32_b32 s8, s7",
            "v_mov_b32_e32 v2, s8",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "global_store_dword v3, v2, s[0:1]",
            "s_endpgm",
        ])
        self.assertEqual(got_gstore, [3, 3, 3, 3])

        _, got_ds = run_cs("cs_ds_add_parked", [
            "v_cmp_lt_u32_e64 s6, v0, 3",
            "v_lshlrev_b32_e32 v5, 2, v0",
            "v_writelane_b32 v5, s6, 3",
            "v_mov_b32_e32 v6, 1",
            "ds_add_u32 v5, v6",
            "v_readlane_b32 s7, v5, 3",
            "s_bcnt1_i32_b32 s8, s7",
            "v_mov_b32_e32 v2, s8",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "global_store_dword v3, v2, s[0:1]",
            "s_endpgm",
        ])
        self.assertEqual(got_ds, [3, 3, 3, 3])


if __name__ == "__main__":
    unittest.main()
