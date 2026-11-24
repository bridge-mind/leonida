; Partial wave32: s_cmp of a lane mask against -1 (all ones) compares the real mask word, not a vote.
; A 20x1x1 group is one wave32 with lanes 0-19 enabled and lanes 20-31 never launched.
; vcc_lo = v_cmp_eq(v1, v1) has a bit per launched lane only, so vcc_lo = 0x000fffff, and EXEC_LO = 0x000fffff.
; A vote over the launched lanes would call both "all ones".
; Lane 0 stores SCC as 0/1 for each compare:
;   out[0] s_cmp_eq_u32 vcc_lo, -1        = 0  (0xfffff != 0xffffffff)
;   out[1] s_cmp_lg_u32 vcc_lo, 0         = 1
;   out[2] s_cmp_eq_u32 vcc_lo, s14       = 1  (s14 = 0xfffff)
;   out[3] s_cmp_eq_u32 exec_lo, -1       = 0
;   out[4] s_cmp_lg_u32 vcc_lo, -1        = 1
;   out[5] s_cmp_eq_u32 vcc_lo, exec_lo   = 1
;   out[6] s_cmp_lg_u32 exec_lo, -1       = 1
;   out[7] s_cmp_eq_u32 exec_lo, s14      = 1
; Until 2026-11-22 the compares with -1 were a vote: out[0] and out[3] read 1, out[4] and out[6] read 0.
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_partial_scmp.spv 1 dump
;   -> out[0..7] = 0 1 1 0 1 1 1 1
; mattr: -wavefrontsize64,+wavefrontsize32
; group: 20,1,1
v_mbcnt_lo_u32_b32 v1, -1, 0
s_mov_b32 s6, exec_lo
v_cmp_eq_u32_e32 vcc_lo, v1, v1
s_mov_b32 s14, 0xfffff
s_cmp_eq_u32 vcc_lo, -1
s_cselect_b32 s20, 1, 0
s_cmp_lg_u32 vcc_lo, 0
s_cselect_b32 s21, 1, 0
s_cmp_eq_u32 vcc_lo, s14
s_cselect_b32 s22, 1, 0
s_cmp_eq_u32 exec_lo, -1
s_cselect_b32 s23, 1, 0
s_cmp_lg_u32 vcc_lo, -1
s_cselect_b32 s24, 1, 0
s_cmp_eq_u32 vcc_lo, exec_lo
s_cselect_b32 s25, 1, 0
s_cmp_lg_u32 exec_lo, -1
s_cselect_b32 s26, 1, 0
s_cmp_eq_u32 exec_lo, s14
s_cselect_b32 s27, 1, 0
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s20
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s21
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s22
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s23
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s24
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s25
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s26
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s27
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, s6
s_endpgm
