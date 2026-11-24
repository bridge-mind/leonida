; Wave64 wave-wide branches decided by one half only. VCC / EXEC are wave registers, so s_cbranch_* is taken by the
; whole wave: both halves must run (or skip) the same scalar instructions. A 32-wide subgroup decides per half.
; (A) VCC = lane 40 only (upper half), s_cbranch_vccnz taken -> `s_mov_b32 s6, 1` is skipped in ALL lanes: s6 = 0.
;     (per-half: the low half sees VCC = 0, falls through, s6 = 1 there)
; (B) VCC = lane 3 only (low half), s_cbranch_vccnz taken -> s7 = 0 everywhere (upper half would fall through).
; (C) EXEC = lanes 32-39 (exec_lo = 0, exec_hi = 0xff), s_cbranch_execz NOT taken (EXEC != 0) -> `s_mov_b32 s9, 5` runs
;     in all lanes: s9 = 5 (per-half: the low half has EXEC = 0, takes the branch, s9 = 0 there)
; (D) VCC = lane 40 only, s_cbranch_vccz NOT taken -> s10 = 9 everywhere (low half would take it, s10 = 0)
; Stores: out[0] lane 0 s6, out[1] lane 40 s6, out[2] lane 40 s7, out[3] lane 63 s7, out[4] lane 0 s9, out[5] lane 40 s9,
;   out[6] lane 0 s10, out[7] lane 63 s10.
; Run: vk_compute_test cs_w64x_branch.spv 1 dump --require-wave64
;   -> out[0..7] = 0 0 0 0 5 5 9 9
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
s_mov_b32 s6, 0
v_cmp_eq_u32_e32 vcc, 40, v1
s_cbranch_vccnz 1
s_mov_b32 s6, 1
s_mov_b32 s7, 0
v_cmp_eq_u32_e32 vcc, 3, v1
s_cbranch_vccnz 1
s_mov_b32 s7, 1
s_mov_b32 s9, 0
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0xff
s_cbranch_execz 1
s_mov_b32 s9, 5
s_mov_b64 exec, -1
s_mov_b32 s10, 0
v_cmp_eq_u32_e32 vcc, 40, v1
s_cbranch_vccz 1
s_mov_b32 s10, 9
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 40, v1
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 40, v1
v_mov_b32_e32 v20, s7
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, s7
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 40, v1
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
