; DPP row_shr:1 with bound_ctrl:1: each lane reads the lane below it in its 16-lane row; lane 0 (and lane 16,
; 32, 48) has an invalid source and reads zero. Lanes 0-7 store [0,0,1,2,3,4,5,6] to out[0..7].
; Run: vk_compute_test cs_dpp_row.spv 1 dump -> out[0..7] = 0 0 1 2 3 4 5 6
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mov_b32_dpp v1, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_lshlrev_b32 v2, 2, v0
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
