; Wave64 LDS read-after-write across the halves of ONE wave, no s_barrier: LDS operations of a wave execute in
; order, so a ds_read issued after a ds_write sees every lane's write, including the other half's. A wave64 run as
; two 32-wide subgroups needs the halves ordered (or a barrier) for this to hold.
; v2 = 100 + lane, ds_write_b32 LDS[lane * 4] = v2, then ds_read_b32 v5 = LDS[(lane ^ 32) * 4] = 100 + (lane ^ 32).
;   out[0] lane 0 -> 132    out[1] lane 5 -> 137    out[2] lane 32 -> 100    out[3] lane 40 -> 108
;   out[4] lane 63 -> 131   out[5] lane 33 -> 101   out[6] lane 17 -> 149    out[7] lane 50 -> 118
; (A buffer_store/buffer_load variant is not included: the 64-word output buffer holds only the observed words.)
; Run: vk_compute_test cs_w64x_lds_raw.spv 1 dump --require-wave64
;   -> out[0..7] = 132 137 100 108 131 101 149 118
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_add_nc_u32_e32 v2, 100, v1
v_lshlrev_b32_e32 v3, 2, v1
ds_write_b32 v3, v2
v_xor_b32_e32 v4, 32, v1
v_lshlrev_b32_e32 v4, 2, v4
ds_read_b32 v5, v4
s_waitcnt lgkmcnt(0)
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 5, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 32, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 40, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 33, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 17, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 50, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
