; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Wave64 v_readfirstlane_b32 with the first active lane in the upper half (lowest set EXEC bit; lane 0 if EXEC = 0).
; A 32-wide subgroup sees only its own half's EXEC bits and returns lane 0 of the low half instead.
; v2 = lane + 100, v1 = lane. EXEC (hi:lo) and result:
;   s6: lanes 40-63 (0xffffff00:0)          -> v2[40] = 140     s7: lane 63 only (0x80000000:0) -> v2[63] = 163
;   s8: lane 32 only (1:0)                  -> v2[32] = 132     s9: EXEC = 0 -> lane 0 -> v2[0] = 100
;   s10: lanes 5,33,34,35 (0xe:0x20)        -> v2[5] = 105 (control: lowest lane is in the low half)
;   s11: lanes 33-35 (0xe:0)                -> v2[33] = 133     s12: lanes 48-63, v1 -> 48
; Stores (EXEC restored, one lane per word): out[0] lane 0 s6, out[1] lane 63 s6, out[2] lane 5 s7, out[3] lane 63 s8,
;   out[4] lane 40 s9, out[5] lane 34 s10, out[6] lane 0 s11, out[7] lane 49 s12.
; Run: vk_compute_test cs_w64x_readfirstlane.spv 1 dump --require-wave64
;   -> out[0..7] = 140 140 163 132 100 105 133 48
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_add_nc_u32_e32 v2, 100, v1
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0xffffff00
v_readfirstlane_b32 s6, v2
s_mov_b64 exec, -1
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0x80000000
v_readfirstlane_b32 s7, v2
s_mov_b64 exec, -1
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 1
v_readfirstlane_b32 s8, v2
s_mov_b64 exec, -1
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0
v_readfirstlane_b32 s9, v2
s_mov_b64 exec, -1
s_mov_b32 exec_lo, 0x20
s_mov_b32 exec_hi, 0xe
v_readfirstlane_b32 s10, v2
s_mov_b64 exec, -1
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0xe
v_readfirstlane_b32 s11, v2
s_mov_b64 exec, -1
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0xffff0000
v_readfirstlane_b32 s12, v1
s_mov_b64 exec, -1
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 5, v1
v_mov_b32_e32 v20, s7
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 40, v1
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 34, v1
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s11
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 49, v1
v_mov_b32_e32 v20, s12
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
