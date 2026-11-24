; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; s_andn2_saveexec_b32 and s_andn2_wrexec_b32 in a 32-thread wave32 (v0 = lane 0..31). Neither had a test.
; RDNA2 ISA:  s_andn2_saveexec_b32: D = EXEC; EXEC = S0 & ~EXEC; SCC = (EXEC != 0)
;             s_andn2_wrexec_b32:   EXEC = S0 & ~EXEC; D = EXEC; SCC = (EXEC != 0)   (D gets the new EXEC)
; S0 is a compare result in VCC, written with all 32 lanes on. The masks are only looked at through mask-aware
; instructions (s_bcnt1, s_ff1, EXEC writes), not read as data.
; 1) VCC = lanes with bit 3 clear = 0x00ff00ff; EXEC = 0x0000ffff. saveexec: D = 0x0000ffff,
;    EXEC = 0x00ff00ff & 0xffff0000 = 0x00ff0000 (lanes 16..23), SCC 1.
;    out[0] = s_cselect 1/2 = 1; out[1] = s_bcnt1(EXEC) = 8; out[2] = s_bcnt1(EXEC after EXEC = D) = 16;
;    v1 = 1 under the new EXEC only, so the lanes with v1 != 0 are 16..23: out[3] = bcnt << 8 | ff1 = 8 << 8 | 16 = 2064.
; 2) VCC = lanes 16..31 = 0xffff0000; EXEC = 0x0f0f0f0f. wrexec: EXEC = 0xffff0000 & 0xf0f0f0f0 = 0xf0f00000
;    (lanes 20-23, 28-31), D = 0xf0f00000, SCC 1.
;    out[4] = bcnt << 8 | ff1 of EXEC = 8 << 8 | 20 = 2068; out[5] = s_cselect 3/4 = 3;
;    out[6] = s_ff1(D) = 20 (D = the old EXEC 0x0f0f0f0f would give 0).
; 3) VCC = lanes 0..7. EXEC = 0x0000ffff, saveexec: EXEC = 0x000000ff & 0xffff0000 = 0, SCC 0 -> s_cselect 5/6 = 6;
;    EXEC = -1, wrexec: EXEC = 0x000000ff & 0 = 0, SCC 0 -> s_cselect 7/8 = 8. out[7] = 6 << 8 | 8 = 1544.
; Lane 0 stores the eight words.
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_andn2_exec.spv 1 dump -> out[0..7] = 1 8 16 2064 2068 3 20 1544
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0
v_and_b32_e32 v2, 8, v0
v_cmp_eq_u32_e32 vcc_lo, 0, v2
s_mov_b32 exec_lo, 0xffff
s_andn2_saveexec_b32 s7, vcc_lo
s_cselect_b32 s8, 1, 2
s_bcnt1_i32_b32 s9, exec_lo
v_mov_b32_e32 v1, 1
s_mov_b32 exec_lo, s7
s_bcnt1_i32_b32 s10, exec_lo
s_mov_b32 exec_lo, -1
v_cmp_ne_u32_e32 vcc_lo, 0, v1
s_bcnt1_i32_b32 s11, vcc_lo
s_ff1_i32_b32 s12, vcc_lo
s_lshl_b32 s11, s11, 8
s_or_b32 s11, s11, s12
v_cmp_le_u32_e32 vcc_lo, 16, v0
s_mov_b32 exec_lo, 0xf0f0f0f
s_andn2_wrexec_b32 s13, vcc_lo
s_cselect_b32 s14, 3, 4
s_bcnt1_i32_b32 s15, exec_lo
s_ff1_i32_b32 s16, exec_lo
s_lshl_b32 s15, s15, 8
s_or_b32 s15, s15, s16
s_mov_b32 exec_lo, -1
s_ff1_i32_b32 s17, s13
v_cmp_gt_u32_e32 vcc_lo, 8, v0
s_mov_b32 exec_lo, 0xffff
s_andn2_saveexec_b32 s18, vcc_lo
s_cselect_b32 s19, 5, 6
s_mov_b32 exec_lo, -1
s_andn2_wrexec_b32 s20, vcc_lo
s_cselect_b32 s21, 7, 8
s_mov_b32 exec_lo, -1
s_lshl_b32 s19, s19, 8
s_or_b32 s19, s19, s21
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s11
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s15
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s14
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s17
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s19
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
