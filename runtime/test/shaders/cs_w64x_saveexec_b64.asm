; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; s_and_saveexec_b64 and s_or_saveexec_b64 with a compare result in VCC or in an SGPR pair, and s_andn2_saveexec_b64 /
; s_andn2_wrexec_b64 with an SGPR-pair source (their VCC form is in cs_w64x_andn2_exec.asm; constant sources are in
; cs_w64_mask_imm.asm), in one 64-lane wave (v1 = lane 0..63).
; RDNA2 ISA:  s_and_saveexec_b64:   D = EXEC; EXEC = S0 & EXEC;  SCC = (EXEC != 0)
;             s_or_saveexec_b64:    D = EXEC; EXEC = S0 | EXEC;  SCC = (EXEC != 0)
;             s_andn2_saveexec_b64: D = EXEC; EXEC = S0 & ~EXEC; SCC = (EXEC != 0)
;             s_andn2_wrexec_b64:   EXEC = S0 & ~EXEC; D = EXEC; SCC = (EXEC != 0)   (D gets the new EXEC)
; S0 is written by a v_cmp with all 64 lanes on. The masks are only looked at through mask-aware instructions
; (s_bcnt1, s_ff1, v_cmp into VCC), not read as data. A mask is reported as bcnt << 8 | ff1 (population count,
; lowest lane). Old EXEC and S0 differ between the two halves in every case, so a swap of the halves moves the
; lowest lane and a result cut to 32 bits loses lanes. EXEC is measured per lane: v2 = 1 under the new EXEC, then
; with all lanes on, the lanes with v2 != 0.
; 1) s_and_saveexec_b64 s[20:21], s[10:11]: S0 = lanes 0..39 (v_cmp_gt_u32_e64 40 > lane);
;    old EXEC = lanes 16..63 (exec_lo 0xffff0000, exec_hi 0xffffffff).
;    EXEC = lanes 16..39 (24 lanes, lowest 16): out[0] = 24 << 8 | 16 = 6160. SCC 1.
;    D = lanes 16..63 (48 lanes, lowest 16): out[1] = 48 << 8 | 16 = 12304.
; 2) s_or_saveexec_b64 s[22:23], vcc: S0 = lanes 0..11 (v_cmp_gt_u32_e32 12 > lane);
;    old EXEC = lanes 40..47 (exec_lo 0, exec_hi 0x0000ff00).
;    EXEC = lanes 0..11 and 40..47 (20 lanes, lowest 0): out[2] = 20 << 8 = 5120. SCC 1.
;    (halves swapped: lanes 8..15 and 32..43, lowest 8; the low word only: 12 lanes.)
;    D = lanes 40..47: out[3] = 8 << 8 | 40 = 2088.
; 3a) s_or_saveexec_b64 s[24:25], s[12:13]: S0 = no lane (0 > lane), old EXEC = 0: EXEC = 0, SCC 0.
; 3b) s_and_saveexec_b64 s[26:27], vcc: S0 = lanes 0..31, old EXEC = lanes 32..63: EXEC = 0, SCC 0
;     (with the halves of either swapped it would be 32 lanes, SCC 1).
; 3c) s_and_saveexec_b64 s[34:35], vcc: S0 = lanes 48..63, old EXEC = lanes 0..15 and 32..63:
;     EXEC = lanes 48..63 (the low half is empty), SCC 1 (an SCC from the low word alone would be 0).
; 4a) s_andn2_saveexec_b64 s[28:29], s[14:15]: S0 = lanes 24..63 (v_cmp_le_u32_e64 24 <= lane);
;     old EXEC = lanes 0..15 and 48..63 (exec_lo 0xffff, exec_hi 0xffff0000), ~EXEC = lanes 16..47.
;     EXEC = lanes 24..47 (24 lanes, lowest 24): out[4] = 24 << 8 | 24 = 6168. SCC 1.
;     D = the old EXEC (32 lanes, lowest 0): out[5] = 32 << 8 = 8192.
; 4b) s_andn2_wrexec_b64 s[30:31], s[16:17]: S0 = lanes 0..35 (v_cmp_gt_u32_e64 36 > lane);
;     old EXEC = lanes 0..7 (exec_lo 0xff, exec_hi 0). EXEC = lanes 8..35 (28 lanes, lowest 8), SCC 1.
;     D = the new EXEC: out[6] = 28 << 8 | 8 = 7176 (the old EXEC would give 8 << 8 = 2048).
; out[7] = SCC bits (1: case 1, 2: case 2, 4: 3a, 8: 3b, 16: 3c, 32: 4a, 64: 4b) = 1 + 2 + 16 + 32 + 64 = 115,
;          plus (bcnt << 8 | ff1 of EXEC after 4b, read with s_bcnt1/s_ff1 on exec) << 8 = 7176 << 8:
;          115 + 1837056 = 1837171.
; Lane 0 stores the eight words.
; Run: vk_compute_test cs_w64x_saveexec_b64.spv 1 dump --require-wave64
;   -> out[0..7] = 6160 12304 5120 2088 6168 8192 7176 1837171
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_mov_b32_e32 v2, 0
; 1) s_and_saveexec_b64, SGPR-pair source
v_cmp_gt_u32_e64 s[10:11], 40, v1
s_mov_b32 exec_lo, 0xffff0000
s_mov_b32 exec_hi, -1
s_and_saveexec_b64 s[20:21], s[10:11]
s_cselect_b32 s40, 1, 0
v_mov_b32_e32 v2, 1
s_mov_b64 exec, -1
v_cmp_ne_u32_e32 vcc, 0, v2
s_bcnt1_i32_b64 s41, vcc
s_ff1_i32_b64 s42, vcc
s_lshl_b32 s41, s41, 8
s_or_b32 s41, s41, s42
s_bcnt1_i32_b64 s43, s[20:21]
s_ff1_i32_b64 s44, s[20:21]
s_lshl_b32 s43, s43, 8
s_or_b32 s43, s43, s44
v_mov_b32_e32 v2, 0
; 2) s_or_saveexec_b64, VCC source
v_cmp_gt_u32_e32 vcc, 12, v1
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0xff00
s_or_saveexec_b64 s[22:23], vcc
s_cselect_b32 s45, 2, 0
v_mov_b32_e32 v2, 1
s_mov_b64 exec, -1
v_cmp_ne_u32_e32 vcc, 0, v2
s_bcnt1_i32_b64 s46, vcc
s_ff1_i32_b64 s47, vcc
s_lshl_b32 s46, s46, 8
s_or_b32 s46, s46, s47
s_bcnt1_i32_b64 s48, s[22:23]
s_ff1_i32_b64 s49, s[22:23]
s_lshl_b32 s48, s48, 8
s_or_b32 s48, s48, s49
v_mov_b32_e32 v2, 0
s_or_b32 s40, s40, s45
; 3a) s_or_saveexec_b64 to an empty EXEC
v_cmp_gt_u32_e64 s[12:13], 0, v1
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0
s_or_saveexec_b64 s[24:25], s[12:13]
s_cselect_b32 s45, 4, 0
s_mov_b64 exec, -1
s_or_b32 s40, s40, s45
; 3b) s_and_saveexec_b64 of two halves that do not meet
v_cmp_gt_u32_e32 vcc, 32, v1
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, -1
s_and_saveexec_b64 s[26:27], vcc
s_cselect_b32 s45, 8, 0
s_mov_b64 exec, -1
s_or_b32 s40, s40, s45
; 3c) s_and_saveexec_b64 with a result in the high half only
v_cmp_le_u32_e32 vcc, 48, v1
s_mov_b32 exec_lo, 0xffff
s_mov_b32 exec_hi, -1
s_and_saveexec_b64 s[34:35], vcc
s_cselect_b32 s45, 16, 0
s_mov_b64 exec, -1
s_or_b32 s40, s40, s45
; 4a) s_andn2_saveexec_b64, SGPR-pair source
v_cmp_le_u32_e64 s[14:15], 24, v1
s_mov_b32 exec_lo, 0xffff
s_mov_b32 exec_hi, 0xffff0000
s_andn2_saveexec_b64 s[28:29], s[14:15]
s_cselect_b32 s45, 32, 0
v_mov_b32_e32 v2, 1
s_mov_b64 exec, -1
s_or_b32 s40, s40, s45
v_cmp_ne_u32_e32 vcc, 0, v2
s_bcnt1_i32_b64 s50, vcc
s_ff1_i32_b64 s51, vcc
s_lshl_b32 s50, s50, 8
s_or_b32 s50, s50, s51
s_bcnt1_i32_b64 s52, s[28:29]
s_ff1_i32_b64 s53, s[28:29]
s_lshl_b32 s52, s52, 8
s_or_b32 s52, s52, s53
; 4b) s_andn2_wrexec_b64, SGPR-pair source
v_cmp_gt_u32_e64 s[16:17], 36, v1
s_mov_b32 exec_lo, 0xff
s_mov_b32 exec_hi, 0
s_andn2_wrexec_b64 s[30:31], s[16:17]
s_cselect_b32 s45, 64, 0
s_bcnt1_i32_b64 s54, exec
s_ff1_i32_b64 s55, exec
s_mov_b64 exec, -1
s_or_b32 s40, s40, s45
s_lshl_b32 s54, s54, 8
s_or_b32 s54, s54, s55
s_lshl_b32 s54, s54, 8
s_or_b32 s40, s40, s54
s_bcnt1_i32_b64 s56, s[30:31]
s_ff1_i32_b64 s57, s[30:31]
s_lshl_b32 s56, s56, 8
s_or_b32 s56, s56, s57
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s41
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s43
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s46
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s48
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s50
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s52
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s56
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s40
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
