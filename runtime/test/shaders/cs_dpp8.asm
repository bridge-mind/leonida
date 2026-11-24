; DPP8 reverse within 8-lane groups: dpp8:[7,6,5,4,3,2,1,0], fi:1. Lanes 0-7 store
; [7,6,5,4,3,2,1,0] to out[0..7].
; Run: vk_compute_test cs_dpp8.spv 1 dump -> out[0..7] = 7 6 5 4 3 2 1 0
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mov_b32_dpp v1, v0 dpp8:[7,6,5,4,3,2,1,0] fi:1
v_lshlrev_b32 v2, 2, v0
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
