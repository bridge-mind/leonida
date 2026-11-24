; DPP quad_perm: v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2], with v0 = lane id. Lanes 0-7 store
; [1,0,3,2,5,4,7,6] to out[0..7]. Same FSSM header as cs_lds.
; Run: vk_compute_test cs_dpp.spv 1 dump -> out[0..7] = 1 0 3 2 5 4 7 6
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_lshlrev_b32 v2, 2, v0
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
