; DPP bank_mask on a VOP1 op: bank 0 is enabled, so lanes 0-3 compute ~lane; lanes 4-7 are not written and
; keep 7. Lanes 0-7 store [0xffffffff,0xfffffffe,0xfffffffd,0xfffffffc,7,7,7,7] to out[0..7].
; Run: vk_compute_test cs_dpp_bank.spv 1 dump -> out[0..7] = 4294967295 4294967294 4294967293 4294967292 7 7 7 7
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mov_b32 v10, 7
v_not_b32_dpp v10, v0 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0x1
v_lshlrev_b32 v2, 2, v0
buffer_store_dword v10, v2, s[0:3], 0 offen
s_endpgm
