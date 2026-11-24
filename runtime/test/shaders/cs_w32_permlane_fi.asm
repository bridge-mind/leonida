; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; SHADER-74: v_permlane16_b32 / v_permlanex16_b32 with the SOURCE lane off in EXEC (wave32). Hardware measurement on
; RDNA2 by aslan2142 (issue #374): op_sel:[FI, BOUND_CTRL] (LLVM: llvm.amdgcn.permlane16 <fi> <bound_control> -> op_sel
; bits 0, 1). FI=1: the value of the off source lane. FI=0, BC=0: the destination lane is not written. FI=0, BC=1:
; the destination lane gets 0. A source lane that is on gives its value whatever FI / BC are.
; EXEC = 0x55555555 (even lanes on). v2 = 3 * lane + 1, destinations start as 1000 + lane. s4/s5 select lane
; (i + 1) & 15 of the row (an odd lane: off), s6/s7 select the lane itself (even: on). Lane 0 stores:
;   out[0] permlane16  FI=0 BC=0, source lane 1 (off)  -> 1000 (not written)
;   out[1] permlane16  FI=0 BC=1, source lane 1 (off)  -> 0
;   out[2] permlane16  FI=1 BC=0, source lane 1 (off)  -> 4  (3*1+1)
;   out[3] permlane16  FI=0 BC=0, source lane 0 (on)   -> 1
;   out[4] permlanex16 FI=0 BC=0, source lane 17 (off) -> 1000 (not written)
;   out[5] permlanex16 FI=0 BC=1, source lane 17 (off) -> 0
;   out[6] permlanex16 FI=1 BC=0, source lane 17 (off) -> 52 (3*17+1)
;   out[7] permlanex16 FI=0 BC=0, source lane 16 (on)  -> 49 (3*16+1)
; group: 32,1,1
; Run: vk_compute_test cs_w32_permlane_fi.spv 1 dump -> out[0..7] = 1000 0 4 1 1000 0 52 49
; mattr: +wavefrontsize32,-wavefrontsize64
v_mad_u32_u24 v2, v0, 3, 1
v_add_nc_u32_e32 v10, 1000, v0
v_add_nc_u32_e32 v11, 1000, v0
v_add_nc_u32_e32 v12, 1000, v0
v_add_nc_u32_e32 v13, 1000, v0
v_add_nc_u32_e32 v14, 1000, v0
v_add_nc_u32_e32 v15, 1000, v0
v_add_nc_u32_e32 v16, 1000, v0
v_add_nc_u32_e32 v17, 1000, v0
s_mov_b32 s4, 0x87654321
s_mov_b32 s5, 0x0fedcba9
s_mov_b32 s6, 0x76543210
s_mov_b32 s7, 0xfedcba98
s_mov_b32 exec_lo, 0x55555555
v_permlane16_b32 v10, v2, s4, s5
v_permlane16_b32 v11, v2, s4, s5 op_sel:[0,1]
v_permlane16_b32 v12, v2, s4, s5 op_sel:[1,0]
v_permlane16_b32 v13, v2, s6, s7
v_permlanex16_b32 v14, v2, s4, s5
v_permlanex16_b32 v15, v2, s4, s5 op_sel:[0,1]
v_permlanex16_b32 v16, v2, s4, s5 op_sel:[1,0]
v_permlanex16_b32 v17, v2, s6, s7
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v11
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v12
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v13
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v14
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v15
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v16
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v17
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
