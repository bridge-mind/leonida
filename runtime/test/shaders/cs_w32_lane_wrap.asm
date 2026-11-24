; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; SHADER-52: wave32 v_readlane_b32 / v_writelane_b32 mask the lane select with 31 (S1[4:0]), not 63.
; v0 = LocalInvocationId.x (0..31). v2 = 3 * lane + 1.
; readlane(v2, 40) must equal lane (40 & 31) = 8 -> 3*8+1 = 25. With & 63 the shuffle asks for lane 40
; (outside a 32-wide subgroup) and fails.
; writelane 777 into lane 40 lands on lane 8; readlane of 8 and of 40 both return 777.
; Lane 0 stores: out[0]=25 out[1]=16 (readlane 5) out[2]=777 out[3]=777.
; group: 32,1,1
; Run: vk_compute_test cs_w32_lane_wrap.spv 1 dump -> out[0..3] = 25 16 777 777
; mattr: +wavefrontsize32,-wavefrontsize64
v_mad_u32_u24 v2, v0, 3, 1
v_readlane_b32 s6, v2, 40
v_readlane_b32 s7, v2, 5
v_mov_b32_e32 v10, 0
s_mov_b32 s8, 777
v_writelane_b32 v10, s8, 40
v_readlane_b32 s9, v10, 8
v_readlane_b32 s10, v10, 40
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s7
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
