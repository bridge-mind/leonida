; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; s_andn2_saveexec_b64 and s_andn2_wrexec_b64 in one 64-lane wave (v1 = lane 0..63). Neither had a test; the _b32
; forms are in cs_w32_andn2_exec.asm.
; RDNA2 ISA:  s_andn2_saveexec_b64: D = EXEC; EXEC = S0 & ~EXEC; SCC = (EXEC != 0)
;             s_andn2_wrexec_b64:   EXEC = S0 & ~EXEC; D = EXEC; SCC = (EXEC != 0)   (D gets the new EXEC)
; S0 is a compare result in VCC, written with all 64 lanes on. The masks are only looked at through mask-aware
; instructions (s_bcnt1, s_ff1, EXEC writes), not read as data. Each result has lanes in one half of the wave
; only, so an EXEC or an SCC taken from the other half alone shows.
; 1) VCC = lanes 40..63; EXEC = lanes 16..47 (exec_lo 0xffff0000, exec_hi 0x0000ffff).
;    saveexec: D = lanes 16..47, EXEC = lanes 40..63 without 16..47 = lanes 48..63 (the low half is empty), SCC 1.
;    out[0] = s_cselect 1/2 = 1; out[1] = s_bcnt1(EXEC) = 16; out[2] = s_bcnt1(EXEC after EXEC = D) = 32;
;    v2 = 1 under the new EXEC only, so the lanes with v2 != 0 are 48..63: out[3] = bcnt << 8 | ff1 = 16 << 8 | 48 = 4144.
; 2) VCC = lanes with bit 3 clear (0x00ff00ff in each half); EXEC = exec_lo 0x0f0f0f0f, exec_hi 0xffffffff.
;    wrexec: EXEC = lanes 4-7 and 20-23 (0x00f000f0; the high half is empty), D = the same, SCC 1.
;    out[4] = bcnt << 8 | ff1 of EXEC = 8 << 8 | 4 = 2052; out[5] = s_cselect 3/4 = 3;
;    out[6] = s_bcnt1(D) = 8 (D = the old EXEC would give 48).
; 3) VCC = lanes 0..7. EXEC = lanes 0..47, saveexec: EXEC = 0, SCC 0 -> s_cselect 5/6 = 6;
;    EXEC = all 64, wrexec: EXEC = 0, SCC 0 -> s_cselect 7/8 = 8. out[7] = 6 << 8 | 8 = 1544.
; Lane 0 stores the eight words.
; Run: vk_compute_test cs_w64x_andn2_exec.spv 1 dump --require-wave64
;   -> out[0..7] = 1 16 32 4144 2052 3 8 1544
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_mov_b32_e32 v2, 0
v_cmp_le_u32_e32 vcc, 40, v1
s_mov_b32 exec_lo, 0xffff0000
s_mov_b32 exec_hi, 0xffff
s_andn2_saveexec_b64 s[6:7], vcc
s_cselect_b32 s8, 1, 2
s_bcnt1_i32_b64 s9, exec
v_mov_b32_e32 v2, 1
s_mov_b64 exec, s[6:7]
s_bcnt1_i32_b64 s10, exec
s_mov_b64 exec, -1
v_cmp_ne_u32_e32 vcc, 0, v2
s_bcnt1_i32_b64 s11, vcc
s_ff1_i32_b64 s12, vcc
s_lshl_b32 s11, s11, 8
s_or_b32 s11, s11, s12
v_and_b32_e32 v3, 8, v1
v_cmp_eq_u32_e32 vcc, 0, v3
s_mov_b32 exec_lo, 0xf0f0f0f
s_mov_b32 exec_hi, -1
s_andn2_wrexec_b64 s[14:15], vcc
s_cselect_b32 s16, 3, 4
s_bcnt1_i32_b64 s17, exec
s_ff1_i32_b64 s18, exec
s_lshl_b32 s17, s17, 8
s_or_b32 s17, s17, s18
s_mov_b64 exec, -1
s_bcnt1_i32_b64 s19, s[14:15]
v_cmp_gt_u32_e32 vcc, 8, v1
s_mov_b32 exec_lo, -1
s_mov_b32 exec_hi, 0xffff
s_andn2_saveexec_b64 s[20:21], vcc
s_cselect_b32 s22, 5, 6
s_mov_b64 exec, -1
s_andn2_wrexec_b64 s[24:25], vcc
s_cselect_b32 s23, 7, 8
s_mov_b64 exec, -1
s_lshl_b32 s22, s22, 8
s_or_b32 s22, s22, s23
v_cmpx_eq_u32_e32 0, v1
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
v_mov_b32_e32 v20, s17
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s16
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s19
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s22
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
