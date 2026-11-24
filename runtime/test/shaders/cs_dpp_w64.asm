; Wave64 DPP: the lane id needs both halves, so this is a wave64 shader. Without emulation it needs a 64-wide
; subgroup; LEONIDA_W64_EMULATE=1 sends the DPP source/EXEC exchange through %x_buf and runs on any subgroup.
; Run: vk_compute_test cs_dpp_w64.spv 1 dump -> out[0..7] = 1 0 3 2 5 4 7 6.
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_lshlrev_b32 v2, 2, v0
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
