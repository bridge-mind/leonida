; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Constant lane masks in wave32 (s_and/or/andn2_saveexec_b32, s_andn2_wrexec_b32).
; A constant read as a lane mask gives lane L its bit L (RDNA2 ISA: EXEC is one bit per lane, and an inline
; constant or literal source of these 32-bit ops is that mask; -16 is 0xfffffff0). Until 2026-11-22 the translator
; gave every lane bit 0 of the constant (Gen.mask_src), so 0xff00 turned all lanes off and 5 turned all 32 on.
; v0 = lane (LocalInvocationId.x, 0..31). Each case sets EXEC, runs the op and marks the lanes it leaves on (a VALU
; write under the new EXEC); with EXEC back to all lanes, v_cmp turns the marks into a lane mask and lane 0 stores
; word = (set lanes) | (lowest set lane) << 8, from s_bcnt1 and s_ff1 of that mask (no lane: 0xffffff00).
;   out[0] EXEC=all,    s_and_saveexec_b32 0xff00        -> 0x0000ff00 = lanes 8-15             8 | 8<<8  = 2056
;   out[1] EXEC=0xff,   s_or_saveexec_b32 0x55555555     -> 0x555555ff (4+4+4+8 lanes)          20 | 0    = 20
;   out[2] EXEC=all,    s_and_saveexec_b32 5 (inline)    -> lanes 0, 2                          2 | 0     = 2
;   out[3] EXEC=0xffff, s_andn2_saveexec_b32 0xff00ff00  -> & ~0xffff = 0xff000000, lanes 24-31  8 | 24<<8 = 6152
;   out[4] EXEC=0xff,   s_andn2_wrexec_b32 0x3ff         -> 0x3ff & ~0xff = lanes 8, 9          2 | 8<<8  = 2050
;   out[5] EXEC=0xffff, s_andn2_saveexec_b32 -16         -> 0xfffffff0 & ~0xffff = lanes 16-31  16 | 16<<8 = 4112
;   out[6] EXEC=0,      s_or_saveexec_b32 0x80000002     -> lanes 1, 31                         2 | 1<<8  = 258
;   out[7] EXEC=15,     s_or_saveexec_b32 -1             -> all 32 lanes (as before the fix)    32 | 0    = 32
; Bit 0 for every lane (the old translation) gives 0xffffff00 32 32 0xffffff00 2072 0xffffff00 0xffffff00 32.
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_mask_imm.spv 1 dump
;   -> out[0..7] = 2056 20 2 6152 2050 4112 258 32
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v14, 0
v_mov_b32_e32 v15, 0
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0
s_mov_b32 exec_lo, -1
s_and_saveexec_b32 s7, 0xff00
v_mov_b32_e32 v10, 1
s_mov_b32 exec_lo, 0xff
s_or_saveexec_b32 s7, 0x55555555
v_mov_b32_e32 v11, 1
s_mov_b32 exec_lo, -1
s_and_saveexec_b32 s7, 5
v_mov_b32_e32 v12, 1
s_mov_b32 exec_lo, 0xffff
s_andn2_saveexec_b32 s7, 0xff00ff00
v_mov_b32_e32 v13, 1
s_mov_b32 exec_lo, 0xff
s_andn2_wrexec_b32 s7, 0x3ff
v_mov_b32_e32 v14, 1
s_mov_b32 exec_lo, 0xffff
s_andn2_saveexec_b32 s7, -16
v_mov_b32_e32 v15, 1
s_mov_b32 exec_lo, 0
s_or_saveexec_b32 s7, 0x80000002
v_mov_b32_e32 v16, 1
s_mov_b32 exec_lo, 15
s_or_saveexec_b32 s7, -1
v_mov_b32_e32 v17, 1
s_mov_b32 exec_lo, -1
v_cmp_ne_u32_e32 vcc_lo, 0, v10
s_bcnt1_i32_b32 s20, vcc_lo
s_ff1_i32_b32 s21, vcc_lo
s_lshl_b32 s21, s21, 8
s_or_b32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, 0, v11
s_bcnt1_i32_b32 s22, vcc_lo
s_ff1_i32_b32 s23, vcc_lo
s_lshl_b32 s23, s23, 8
s_or_b32 s22, s22, s23
v_cmp_ne_u32_e32 vcc_lo, 0, v12
s_bcnt1_i32_b32 s24, vcc_lo
s_ff1_i32_b32 s25, vcc_lo
s_lshl_b32 s25, s25, 8
s_or_b32 s24, s24, s25
v_cmp_ne_u32_e32 vcc_lo, 0, v13
s_bcnt1_i32_b32 s26, vcc_lo
s_ff1_i32_b32 s27, vcc_lo
s_lshl_b32 s27, s27, 8
s_or_b32 s26, s26, s27
v_cmp_ne_u32_e32 vcc_lo, 0, v14
s_bcnt1_i32_b32 s28, vcc_lo
s_ff1_i32_b32 s29, vcc_lo
s_lshl_b32 s29, s29, 8
s_or_b32 s28, s28, s29
v_cmp_ne_u32_e32 vcc_lo, 0, v15
s_bcnt1_i32_b32 s30, vcc_lo
s_ff1_i32_b32 s31, vcc_lo
s_lshl_b32 s31, s31, 8
s_or_b32 s30, s30, s31
v_cmp_ne_u32_e32 vcc_lo, 0, v16
s_bcnt1_i32_b32 s32, vcc_lo
s_ff1_i32_b32 s33, vcc_lo
s_lshl_b32 s33, s33, 8
s_or_b32 s32, s32, s33
v_cmp_ne_u32_e32 vcc_lo, 0, v17
s_bcnt1_i32_b32 s34, vcc_lo
s_ff1_i32_b32 s35, vcc_lo
s_lshl_b32 s35, s35, 8
s_or_b32 s34, s34, s35
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, s20
v_mov_b32_e32 v2, 0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s22
v_mov_b32_e32 v2, 4
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s24
v_mov_b32_e32 v2, 8
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s26
v_mov_b32_e32 v2, 12
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s28
v_mov_b32_e32 v2, 16
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s30
v_mov_b32_e32 v2, 20
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s32
v_mov_b32_e32 v2, 24
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s34
v_mov_b32_e32 v2, 28
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
