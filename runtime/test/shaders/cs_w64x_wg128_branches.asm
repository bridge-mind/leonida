; 128-thread workgroup = two wave64s (v0 = thread id 0-127, v1 = lane in its wave). The waves take DIFFERENT
; wave-uniform branches (s6 = wave id from v_readfirstlane of v0 >> 6, s_cmp_eq + s_cbranch_scc0) and each does
; cross-half work INSIDE its branch. v3 = 3 * tid + 1, v10 = 0.
; Wave 0: s8 = readlane(v3, 40) = 3 * 40 + 1 = 121; ballot (49 < lane) = lanes 50-63 -> s9 = 14;
;         v_writelane(v10, 77, lane 44) then readlane(v10, 44) = 77; s10 = 100 + (77 << 8) = 19812.
; Wave 1: s8 = readlane(v3, 50) = 3 * (64 + 50) + 1 = 343; ballot (32 < lane) = lanes 33-63 -> s9 = 31;
;         writelane 88 into lane 55, readlane -> 88; s10 = 200 + (88 << 8) = 22728.
; Stores (by thread id): out[0] t0 s8 = 121, out[1] t45 s8 = 121, out[2] t64 s8 = 343, out[3] t127 s8 = 343,
;   out[4] t33 s9 = 14, out[5] t100 s9 = 31, out[6] t1 s10 = 19812, out[7] t127 s10 = 22728.
; A wave decided per 32-lane half, or a readlane >= 32 that wraps, changes these. Needs a 128x1x1 group header
; (the group directive below; build.py and run_gpu_tests.py read it).
; Run: vk_compute_test cs_w64x_wg128_branches.spv 1 dump --require-wave64
;   -> out[0..7] = 121 121 343 343 14 31 19812 22728
; mattr: +wavefrontsize64,-wavefrontsize32
; group: 128,1,1
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_lshrrev_b32_e32 v2, 6, v0
v_readfirstlane_b32 s6, v2
v_mad_u32_u24 v3, v0, 3, 1
v_mov_b32_e32 v10, 0
s_cmp_eq_u32 s6, 0
s_cbranch_scc0 14
v_readlane_b32 s8, v3, 40
v_cmp_lt_u32_e64 s[4:5], 49, v1
s_bcnt1_i32_b64 s9, s[4:5]
s_mov_b32 s11, 77
v_writelane_b32 v10, s11, 44
v_readlane_b32 s12, v10, 44
s_mov_b32 s10, 100
s_branch 13
v_readlane_b32 s8, v3, 50
v_cmp_lt_u32_e64 s[4:5], 32, v1
s_bcnt1_i32_b64 s9, s[4:5]
s_mov_b32 s11, 88
v_writelane_b32 v10, s11, 55
v_readlane_b32 s12, v10, 55
s_mov_b32 s10, 200
s_lshl_b32 s12, s12, 8
s_add_u32 s10, s10, s12
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 45, v0
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 64, v0
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 127, v0
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 33, v0
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 100, v0
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 1, v0
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 127, v0
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
