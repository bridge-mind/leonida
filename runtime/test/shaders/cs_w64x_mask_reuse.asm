; The same lane mask read twice, then rewritten, then read again: --wave64-emulate exchanges a mask once per value, so
; a stale reuse after the register or EXEC changed would show here.
;   s[4:5] = lanes with (lane & 7) == 5 (8 lanes): s6 = s_bcnt1 = 8, v3 = v_mbcnt_lo/hi of it = selected lanes below.
;   s[4:5] = lanes < 40 (v_cmp_gt_u32 40, lane): s7 = s_bcnt1 = 40, v4 = v_mbcnt_lo/hi of it = min(lane, 40).
;   EXEC: s8 = popcount(exec) = 64, v_cmpx lane < 10 -> s9 = 10, exec = -1 again -> s10 = 64.
; Stores: out[0] s6, out[1] s7, out[2] lane 63 v3 (= 8), out[3] lane 63 v4 (= 40), out[4] lane 20 v3 (= 2),
;   out[5] lane 20 v4 (= 20), out[6] s9, out[7] s10.
; Run: vk_compute_test cs_w64x_mask_reuse.spv 1 dump --require-wave64
;   -> out[0..7] = 8 40 8 40 2 20 10 64
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_and_b32_e32 v2, 7, v1
v_cmp_eq_u32_e64 s[4:5], 5, v2
s_bcnt1_i32_b64 s6, s[4:5]
v_mbcnt_lo_u32_b32 v3, s4, 0
v_mbcnt_hi_u32_b32 v3, s5, v3
s_bcnt1_i32_b64 s11, s[4:5]
v_cmp_gt_u32_e64 s[4:5], 40, v1
s_bcnt1_i32_b64 s7, s[4:5]
v_mbcnt_lo_u32_b32 v4, s4, 0
v_mbcnt_hi_u32_b32 v4, s5, v4
s_bcnt1_i32_b64 s8, exec
v_cmpx_gt_u32_e32 10, v1
s_bcnt1_i32_b64 s9, exec
s_mov_b64 exec, -1
s_bcnt1_i32_b64 s10, exec
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s7
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, v3
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 20, v1
v_mov_b32_e32 v20, v3
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
