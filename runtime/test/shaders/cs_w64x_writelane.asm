; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Wave64 cross-half v_writelane_b32 (writes lane S1 % 64 of the VGPR, EXEC ignored). Crosses: writes into lanes
; >= 32 (a 32-wide subgroup never matches lane select >= 32, so those writes are lost), the dynamic lane select
; 96 % 64 = 32 (first lane of the upper half), and a readlane of the written upper lanes from the low half.
; v10 = 0 everywhere, then v10[40] = 0x1234 (4660), v10[3] = 77, v10[63] = 99, v10[96 % 64 = 32] = 0xabc (2748).
;   out[0] lane 40 stores v10 = 4660   out[1] lane 3 stores v10 = 77    out[2] lane 63 stores v10 = 99
;   out[3] lane 32 stores v10 = 2748   out[4] lane 39 stores v10 = 0    out[5] lane 41 stores v10 = 0 (neighbours untouched)
;   out[6] lane 0 stores readlane(v10, 40) = 4660   out[7] lane 8 stores readlane(v10, 63) = 99
; Run: vk_compute_test cs_w64x_writelane.spv 1 dump --require-wave64
;   -> out[0..7] = 4660 77 99 2748 0 0 4660 99
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_mov_b32_e32 v10, 0
s_mov_b32 s6, 0x1234
s_mov_b32 s7, 77
s_mov_b32 s8, 99
s_mov_b32 s11, 0xabc
s_mov_b32 s9, 96
v_writelane_b32 v10, s6, 40
v_writelane_b32 v10, s7, 3
v_writelane_b32 v10, s8, 63
v_writelane_b32 v10, s11, s9
v_readlane_b32 s12, v10, 40
v_readlane_b32 s13, v10, 63
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 40, v1
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 3, v1
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 32, v1
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 39, v1
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 41, v1
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s12
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 8, v1
v_mov_b32_e32 v20, s13
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
