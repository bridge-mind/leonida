; Wave64 LDS write-after-read, the warp-synchronous shift idiom: LDS[lane] = 1000 + lane; then in one wave
;   v5 = LDS[(lane + 32) & 63]   (ds_read_b32)    LDS[lane] = 2000 + lane   (ds_write_b32, issued right after)
; The read of the whole wave executes before the write of the whole wave, so lanes 0-31 read the OLD values of
; lanes 32-63 (and lanes 32-63 the old values of lanes 0-31). Run as two 32-wide subgroups, the upper half's
; write can land before the lower half's read. A second read of the same addresses sees the new values.
;   v5 = 1000 + ((lane + 32) & 63): out[0] lane 0 -> 1032   out[1] lane 31 -> 1063   out[2] lane 32 -> 1000
;                                   out[3] lane 63 -> 1031
;   v8 = 2000 + ((lane + 32) & 63): out[4] lane 0 -> 2032   out[5] lane 31 -> 2063   out[6] lane 32 -> 2000
;                                   out[7] lane 63 -> 2031
; Run: vk_compute_test cs_w64x_lds_war.spv 1 dump --require-wave64
;   -> out[0..7] = 1032 1063 1000 1031 2032 2063 2000 2031
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_add_nc_u32_e32 v2, 1000, v1
v_lshlrev_b32_e32 v3, 2, v1
ds_write_b32 v3, v2
v_add_nc_u32_e32 v4, 32, v1
v_and_b32_e32 v4, 63, v4
v_lshlrev_b32_e32 v4, 2, v4
ds_read_b32 v5, v4
v_add_nc_u32_e32 v6, 2000, v1
ds_write_b32 v3, v6
s_waitcnt lgkmcnt(0)
ds_read_b32 v8, v4
s_waitcnt lgkmcnt(0)
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 31, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 32, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, v8
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 31, v1
v_mov_b32_e32 v20, v8
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 32, v1
v_mov_b32_e32 v20, v8
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, v8
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
