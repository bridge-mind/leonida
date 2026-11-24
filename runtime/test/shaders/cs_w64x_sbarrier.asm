; 128 threads = two wave64s around s_barrier. Wave 0 runs a loop of 5 trips with a cross-half v_readlane (lane 40)
; BEFORE the barrier; wave 1 runs a loop of 2 trips (readlane of lane 50), then writes LDS[lane] = 3000 + lane
; and waits for it (s_waitcnt lgkmcnt(0)). Both reach the one s_barrier, then every lane reads
; LDS[lane ^ 32] (wave 0 reads what wave 1 wrote; its own half-crossing readlane loop must not deadlock or
; desynchronise the barrier). v3 = 3 * tid + 1.
;   wave 0: s21 = 5 * v3[lane 40] = 5 * (3 * 40 + 1) = 605
;   wave 1: s21 = 2 * v3[lane 50] = 2 * (3 * (64 + 50) + 1) = 686
;   v10 = 3000 + (lane ^ 32): t0 -> 3032, t45 -> 3000 + 13 = 3013, t64 (lane 0) -> 3032, t127 (lane 63) -> 3031.
; Stores (by thread id): out[0] t0 s21, out[1] t45 s21, out[2] t64 s21, out[3] t127 s21,
;   out[4] t0 v10, out[5] t45 v10, out[6] t64 v10, out[7] t127 v10.
; Run: vk_compute_test cs_w64x_sbarrier.spv 1 dump --require-wave64
;   -> out[0..7] = 605 605 686 686 3032 3013 3032 3031
; mattr: +wavefrontsize64,-wavefrontsize32
; group: 128,1,1
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_lshrrev_b32_e32 v2, 6, v0
v_readfirstlane_b32 s6, v2
v_mad_u32_u24 v3, v0, 3, 1
s_cmp_eq_u32 s6, 0
s_cbranch_scc0 9
s_mov_b32 s20, 0
s_mov_b32 s21, 0
v_readlane_b32 s22, v3, 40
s_add_u32 s21, s21, s22
s_add_u32 s20, s20, 1
s_cmp_lt_u32 s20, 5
s_cbranch_scc1 -6
s_branch 14
s_mov_b32 s20, 0
s_mov_b32 s21, 0
v_readlane_b32 s22, v3, 50
s_add_u32 s21, s21, s22
s_add_u32 s20, s20, 1
s_cmp_lt_u32 s20, 2
s_cbranch_scc1 -6
v_add_nc_u32_e32 v9, 3000, v1
v_lshlrev_b32_e32 v8, 2, v1
ds_write_b32 v8, v9
s_waitcnt lgkmcnt(0)
s_barrier
v_xor_b32_e32 v11, 32, v1
v_lshlrev_b32_e32 v11, 2, v11
ds_read_b32 v10, v11
s_waitcnt lgkmcnt(0)
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s21
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 45, v0
v_mov_b32_e32 v20, s21
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 64, v0
v_mov_b32_e32 v20, s21
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 127, v0
v_mov_b32_e32 v20, s21
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 45, v0
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 64, v0
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 127, v0
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
