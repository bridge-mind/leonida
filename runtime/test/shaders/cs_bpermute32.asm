; Wave32 ds_bpermute_b32: RDNA2 ISA 10.4.4 - EXEC is honoured for both reading the source and writing the destination,
; reading from a disabled lane returns zero, and only index bits [6:2] are used (src lane = (addr / 4) & 31).
; A translator that shuffles the raw value hands a disabled source lane's data to the reader instead of 0.
; v2 = 7 * lane + 1 (data), v6 preset to 99.
; (a) EXEC = all but lanes 8-15, addr = (lane + 3) * 4, src = (lane + 3) & 31:
;     lane 6 -> 9 (disabled) -> 0. Lane 4 -> 7 -> 50. Lane 20 -> 23 -> 162. Lane 2 -> 5 -> 36.
;     Lane 31 -> 34 & 31 = 2 -> 15. Lane 10 is disabled: its destination keeps 99.
; (b) full EXEC, addr = (63 - lane) * 4, index 32..63 wraps: src = 31 - lane. Lane 0 -> 31 -> 218. Lane 5 -> 26 -> 183.
; out[0..7] = (a) lane 6, 4, 20; (a) lane 10; (a) lane 2; (b) lane 0, 5; (a) lane 31.
; Run: vk_compute_test cs_bpermute32.spv 1 dump
;   -> out[0..7] = 0 50 162 99 36 218 183 15
; mattr: -wavefrontsize64,+wavefrontsize32
; group: 32,1,1
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mad_u32_u24 v2, v1, 7, 1
v_add_nc_u32_e32 v3, 3, v1
v_lshlrev_b32_e32 v3, 2, v3
v_sub_nc_u32_e32 v5, 63, v1
v_lshlrev_b32_e32 v5, 2, v5
v_mov_b32_e32 v6, 99
s_mov_b32 exec_lo, 0xffff00ff
ds_bpermute_b32 v6, v3, v2
s_waitcnt lgkmcnt(0)
s_mov_b32 exec_lo, -1
ds_bpermute_b32 v7, v5, v2
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 6, v1
v_mov_b32_e32 v20, v6
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 4, v1
v_mov_b32_e32 v20, v6
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 20, v1
v_mov_b32_e32 v20, v6
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 10, v1
v_mov_b32_e32 v20, v6
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 2, v1
v_mov_b32_e32 v20, v6
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, v7
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 5, v1
v_mov_b32_e32 v20, v7
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 31, v1
v_mov_b32_e32 v20, v6
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
