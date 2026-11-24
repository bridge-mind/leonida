; group: 8,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; CMPX class keeps qNaN and +0 across both compares. E32 (mask 0x242) drops -0, E64 (mask 0x62)
; drops +infinity. Neither compare can reactivate lane 0, which was disabled even though it holds qNaN.
; VCC selects the distinctive value in lane 3 and must survive both EXEC-only compares.
; Run: vk_compute_test cs_cmpx_class_f32.spv 1 dump
; out[0..7] = EEEEEEEE EEEEEEEE EEEEEEEE 00000022 00000011 00000011 00000011 00000011 (hex)
v_mov_b32_e32 v1, 0x7fc00000
v_mov_b32_e32 v2, 0x7f800000
v_cmp_eq_u32_e32 vcc_lo, 1, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_mov_b32_e32 v2, 0x80000000
v_cmp_eq_u32_e32 vcc_lo, 2, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_mov_b32_e32 v2, 0
v_cmp_eq_u32_e32 vcc_lo, 3, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_mov_b32_e32 v2, 0x242
v_mov_b32_e32 v7, 0x62
v_mov_b32_e32 v3, 0x11
v_mov_b32_e32 v4, 0x22
v_lshlrev_b32_e32 v5, 2, v0
v_cmpx_ne_u32_e32 0, v0
v_cmpx_class_f32_e32 v1, v2
v_cmpx_class_f32_e64 v1, v7
v_cndmask_b32_e32 v6, v3, v4, vcc_lo
buffer_store_dword v6, v5, s[0:3], 0 offen
s_endpgm
