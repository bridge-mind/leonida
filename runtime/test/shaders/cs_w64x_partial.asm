; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Partial wave64: a 40x1x1 group is one wave with lanes 0-39 enabled, lanes 40-63 disabled (EXEC = 0xff_ffffffff).
; Only what the ISA defines is asserted (no readlane of a nonexistent lane): EXEC and ballot counts, ds_bpermute
; reads from a disabled lane return 0, readfirstlane on the enabled lanes of the upper half.
; v2 = 500 + lane (data). s6:s7 = EXEC at entry.
;   s8 = s_bcnt1 of EXEC = 40.   s11 = s_bcnt1 of v_cmp_eq(v1, v1) = 40 (the ballot has no bits of disabled lanes).
;   v4 = ds_bpermute with addr = (lane + 8) * 4, src = half | ((lane + 8) & 31):
;        lane 20 -> 28 -> 528;  lane 33 -> 32 | 9 = 41 (disabled) -> 0;  lane 39 -> 32 | 15 = 47 (disabled) -> 0.
;   v5 = ds_bpermute with addr = (lane ^ 1) * 4: lane 33 -> 32 -> 532;  lane 38 -> 39 -> 539.
;   Then EXEC = lanes 32-39 (v_cmpx 31 < lane): s9 = v_readfirstlane(v2) = v2[32] = 532, s10 = s_bcnt1 EXEC = 8,
;   packed s9 + (s10 << 16) = 532 + 524288 = 524820.
; Stores: out[0] lane 0 s8, out[1] lane 39 s11, out[2] lane 20 v4, out[3] lane 33 v4, out[4] lane 39 v4,
;   out[5] lane 33 v5, out[6] lane 38 v5, out[7] lane 2 s9.
; The group has 40 threads, so --require-wave64 (full subgroups) cannot be used; check.py asks for subgroup size 64
; with VKC_SUBGROUP_SIZE=64 instead.
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_w64x_partial.spv 1 dump
;   -> out[0..7] = 40 40 528 0 0 532 539 524820
; mattr: +wavefrontsize64,-wavefrontsize32
; group: 40,1,1
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_add_nc_u32_e32 v2, 500, v1
s_mov_b64 s[6:7], exec
s_bcnt1_i32_b64 s8, exec
v_cmp_eq_u32_e64 s[4:5], v1, v1
s_bcnt1_i32_b64 s11, s[4:5]
v_add_nc_u32_e32 v3, 8, v1
v_lshlrev_b32_e32 v3, 2, v3
ds_bpermute_b32 v4, v3, v2
v_xor_b32_e32 v6, 1, v1
v_lshlrev_b32_e32 v6, 2, v6
ds_bpermute_b32 v5, v6, v2
s_waitcnt lgkmcnt(0)
v_cmpx_lt_u32_e32 31, v1
v_readfirstlane_b32 s9, v2
s_bcnt1_i32_b64 s10, exec
s_mov_b64 exec, s[6:7]
s_lshl_b32 s10, s10, 16
s_add_u32 s9, s9, s10
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 39, v1
v_mov_b32_e32 v20, s11
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 20, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 33, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 39, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 33, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 38, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 2, v1
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
