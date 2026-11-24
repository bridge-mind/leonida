; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Wave64 cross-half v_readlane_b32 (RDNA2 ISA: lane = S1 % 64, EXEC ignored). Crosses: lane >= 32 sources read by
; lanes of both halves, and a lane < 32 source read by an upper-half lane. A 32-wide subgroup wraps the lane
; select modulo 32 (the issue #94 symptom), so lane 40 would return lane 8's value.
; v2 = 3 * lane + 1. s6 = v2[40] = 121, s7 = v2[5] = 16, s8 = v2[32] = 97, s9 = v2[97 % 64 = 33] = 100.
; Each word is stored by one lane (v_cmpx), so both halves observe the scalar:
;   out[0] lane 0 stores s6 = 121    out[1] lane 33 stores s6 = 121   out[2] lane 63 stores s7 = 16
;   out[3] lane 0 stores s8 = 97     out[4] lane 45 stores s7 = 16    out[5] lane 7 stores s9 = 100
;   out[6] lane 60 stores s9 = 100   out[7] EXEC = lane 63 only, readlane of lane 40 (inactive): 121 (EXEC ignored)
; Same FSSM header as cs_lds. Run: vk_compute_test cs_w64x_readlane.spv 1 dump --require-wave64
;   -> out[0..7] = 121 121 16 97 16 100 100 121
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_mad_u32_u24 v2, v1, 3, 1
v_readlane_b32 s6, v2, 40
v_readlane_b32 s7, v2, 5
v_readlane_b32 s8, v2, 32
s_mov_b32 s10, 97
v_readlane_b32 s9, v2, s10
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 33, v1
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, s7
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 45, v1
v_mov_b32_e32 v20, s7
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 7, v1
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 60, v1
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_readlane_b32 s11, v2, 40
v_mov_b32_e32 v20, s11
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
