; DPP on a VOP2 op with an invalid source and bound_ctrl:1: lane 0 uses 0 as src0, so it writes op(0, 10);
; the other lanes use the lane below them. Lanes 0-7 store [10,10,11,12,13,14,15,16] to out[0..7].
; Run: vk_compute_test cs_dpp_bound.spv 1 dump -> out[0..7] = 10 10 11 12 13 14 15 16
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mov_b32 v1, 10
v_mov_b32 v10, 100
v_add_nc_u32_dpp v10, v0, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_lshlrev_b32 v2, 2, v0
buffer_store_dword v10, v2, s[0:3], 0 offen
s_endpgm
