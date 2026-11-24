; DPP row_mask/bank_mask: quad_perm:[1,0,3,2] with row 0 and bank 0 enabled only. Lanes 0-3 get the
; permutation; lanes 4-7 keep the output buffer's initial 0xEE. Lanes 0-3 store to out[0..3] only.
; Run: vk_compute_test cs_dpp_mask.spv 1 dump -> out[0..7] = 1 0 3 2 4008636142 4008636142 4008636142 4008636142
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2] row_mask:0x1 bank_mask:0x1
v_lshlrev_b32 v2, 2, v0
v_cmpx_lt_u32 v0, 4
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
