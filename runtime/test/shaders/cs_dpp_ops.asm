; DPP on a VOP2 op with an invalid source and no bound_ctrl: lane 0 reads outside its row, so it must not be
; written and keeps 100. The other lanes compute op(source lane, 10). Lanes 0-7 store
; [100,10,11,12,13,14,15,16] to out[0..7].
; Run: vk_compute_test cs_dpp_ops.spv 1 dump -> out[0..7] = 100 10 11 12 13 14 15 16
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mov_b32 v1, 10
v_mov_b32 v10, 100
v_add_nc_u32_dpp v10, v0, v1 row_shr:1 row_mask:0xf bank_mask:0xf
v_lshlrev_b32 v2, 2, v0
buffer_store_dword v10, v2, s[0:3], 0 offen
s_endpgm
