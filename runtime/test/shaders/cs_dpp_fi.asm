; DPP16 fi:1 with EXEC 0x77777777 (lane 3 of each quad off). lane 2 reads the inactive lane 3; lane 3 is
; itself inactive and keeps 0x7777. Lanes 0-7 store [1,2,3,0x7777,5,6,7,0x7777] to out[0..7].
; Run: vk_compute_test cs_dpp_fi.spv 1 dump -> out[0..7] = 1 2 3 30583 5 6 7 30583
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mov_b32 v10, 0x7777
s_mov_b32 exec_lo, 0x77777777
v_mov_b32_dpp v10, v0 quad_perm:[1,2,3,0] row_mask:0xf bank_mask:0xf fi:1
s_mov_b32 exec_lo, -1
v_lshlrev_b32 v2, 2, v0
buffer_store_dword v10, v2, s[0:3], 0 offen
s_endpgm
