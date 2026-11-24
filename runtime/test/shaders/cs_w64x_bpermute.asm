; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Wave64 ds_bpermute_b32: RDNA2 ISA 10.4.4 - it works on each 32-lane half as two independent wave32s (only index
; bits [6:2] are used, a half references its own lanes 0-31) and reads from a disabled lane return 0. So there is NO
; cross-half read: src lane = (lane & 32) | ((addr / 4) & 31). Crosses: the upper half must read from ITS OWN half
; (a translator that shuffles to idx & 31 of the subgroup reads the low half there), an index >= 32 wraps inside
; the half, and a source lane that is inactive in EXEC gives 0.
; v2 = 7 * lane + 1 (data).
; (a) addr = (lane + 5) * 4: src = half | ((lane + 5) & 31). Lane 0 -> 5 -> 36. Lane 40 -> 32 | 13 = 45 -> 316.
;     Lane 63 -> 32 | (68 & 31 = 4) = 36 -> 253.
; (b) addr = (63 - (lane & 31)) * 4, index 32..63 wraps: src = half | (31 - (lane & 31)). Lane 0 -> 31 -> 218.
;     Lane 33 -> 32 | 30 = 62 -> 435. Lane 63 -> 32 | 0 = 32 -> 225.
; (c) EXEC = all but lanes 32-47 (v6 preset to 99), addr = (lane + 14) * 4: lane 50 -> 32 | (64 & 31 = 0) = 32, inactive
;     -> 0. Lane 49 -> 32 | (63 & 31 = 31) = 63, active -> 442.
; out[0..7] = (a) lane 0, 40, 63; (b) lane 0, 33, 63; (c) lane 50, 49.
; Run: vk_compute_test cs_w64x_bpermute.spv 1 dump --require-wave64
;   -> out[0..7] = 36 316 253 218 435 225 0 442
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_mad_u32_u24 v2, v1, 7, 1
v_add_nc_u32_e32 v3, 5, v1
v_lshlrev_b32_e32 v3, 2, v3
ds_bpermute_b32 v4, v3, v2
v_and_b32_e32 v5, 31, v1
v_sub_nc_u32_e32 v5, 63, v5
v_lshlrev_b32_e32 v5, 2, v5
ds_bpermute_b32 v5, v5, v2
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v6, 99
v_add_nc_u32_e32 v7, 14, v1
v_lshlrev_b32_e32 v7, 2, v7
s_mov_b32 exec_lo, -1
s_mov_b32 exec_hi, 0xffff0000
ds_bpermute_b32 v6, v7, v2
s_waitcnt lgkmcnt(0)
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 40, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 33, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 50, v1
v_mov_b32_e32 v20, v6
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 49, v1
v_mov_b32_e32 v20, v6
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
