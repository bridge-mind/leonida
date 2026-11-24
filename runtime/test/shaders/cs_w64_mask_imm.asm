; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Constant lane masks in wave64 (s_and/or/andn2_saveexec_b64, s_andn2_wrexec_b64, s_and_b64 / s_not_b64 into EXEC,
; s_mov_b64 exec/vcc) and s_bcnt1_i32_b64 of a constant. A constant read as a 64-bit lane mask gives lane L bit L
; of the 64-bit value: an inline constant is sign-extended (-16 = lanes 4-63), a literal is the low word with a
; zero high word (as Gen.src64_data reads 64-bit constants; the literals here are below 0x80000000, so a
; sign-extending reading would give the same words). Until 2026-11-22 the translator gave every lane bit 0 of the
; constant (Gen.mask_src), and s_bcnt1_i32_b64 counted that lane's copy.
; Each case sets EXEC, runs the op and marks the lanes it leaves on; with EXEC back to all lanes, v_cmp turns the
; marks into a lane mask and lane 0 stores word = (set lanes) | (lowest set lane) << 8 (no lane: 0xffffff00).
;   out[0] EXEC=all,  s_and_saveexec_b64 0xff00           -> lanes 8-15                      8 | 8<<8  = 2056
;   out[1] EXEC=all,  s_and_b64 exec, exec, -16           -> lanes 4-63                      60 | 4<<8 = 1084
;   out[2] s_mov_b64 exec, 0x10; s_andn2_saveexec_b64 -8  -> bits 3-63 & ~bit 4 = 3, 5-63    60 | 3<<8 = 828
;   out[3] s_mov_b64 exec, 0xf0f0; s_andn2_wrexec_b64 0xff00 -> 0x0f00 = lanes 8-11          4 | 8<<8  = 2052
;   out[4] s_mov_b64 exec, 0; s_or_saveexec_b64 -2        -> lanes 1-63                      63 | 1<<8 = 319
;   out[5] s_not_b64 exec, 0xff                           -> lanes 8-63                      56 | 8<<8 = 2104
;   out[6] s_mov_b64 vcc, 0x3c; v_cndmask_b32_e32 v, 0, v(=1), vcc -> lanes 2-5             4 | 2<<8  = 516
;   out[7] s_bcnt1_i32_b64 0xff00ff | s_ff1_i32_b64 0xff00 << 8 (scalar data)                16 | 8<<8 = 2064
; Bit 0 for every lane (the old translation) gives 0xffffff00 in out[0..6] and 2080 in out[7].
; (llvm-mc rejects a constant as the lane-mask operand of v_cndmask_b32_e64 / v_add_co_ci_u32_e64, and prints one
; it decodes as "Invalid immediate", so those VALU forms never reach the translator with a constant.)
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_w64_mask_imm.spv 1 dump --require-wave64
;   (or translated with --wave64-emulate, on any subgroup size) -> out[0..7] = 2056 1084 828 2052 319 2104 516 2064
; mattr: +wavefrontsize64,-wavefrontsize32
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v14, 0
v_mov_b32_e32 v15, 0
v_mov_b32_e32 v16, 0
s_mov_b64 exec, -1
s_and_saveexec_b64 s[8:9], 0xff00
v_mov_b32_e32 v10, 1
s_mov_b64 exec, -1
s_and_b64 exec, exec, -16
v_mov_b32_e32 v11, 1
s_mov_b64 exec, 0x10
s_andn2_saveexec_b64 s[8:9], -8
v_mov_b32_e32 v12, 1
s_mov_b64 exec, 0xf0f0
s_andn2_wrexec_b64 s[8:9], 0xff00
v_mov_b32_e32 v13, 1
s_mov_b64 exec, 0
s_or_saveexec_b64 s[8:9], -2
v_mov_b32_e32 v14, 1
s_not_b64 exec, 0xff
v_mov_b32_e32 v15, 1
s_mov_b64 exec, -1
s_mov_b64 vcc, 0x3c
v_mov_b32_e32 v18, 1
v_cndmask_b32_e32 v16, 0, v18, vcc
s_mov_b64 exec, -1
v_cmp_ne_u32_e32 vcc, 0, v10
s_bcnt1_i32_b64 s20, vcc
s_ff1_i32_b64 s21, vcc
s_lshl_b32 s21, s21, 8
s_or_b32 s20, s20, s21
v_cmp_ne_u32_e32 vcc, 0, v11
s_bcnt1_i32_b64 s22, vcc
s_ff1_i32_b64 s23, vcc
s_lshl_b32 s23, s23, 8
s_or_b32 s22, s22, s23
v_cmp_ne_u32_e32 vcc, 0, v12
s_bcnt1_i32_b64 s24, vcc
s_ff1_i32_b64 s25, vcc
s_lshl_b32 s25, s25, 8
s_or_b32 s24, s24, s25
v_cmp_ne_u32_e32 vcc, 0, v13
s_bcnt1_i32_b64 s26, vcc
s_ff1_i32_b64 s27, vcc
s_lshl_b32 s27, s27, 8
s_or_b32 s26, s26, s27
v_cmp_ne_u32_e32 vcc, 0, v14
s_bcnt1_i32_b64 s28, vcc
s_ff1_i32_b64 s29, vcc
s_lshl_b32 s29, s29, 8
s_or_b32 s28, s28, s29
v_cmp_ne_u32_e32 vcc, 0, v15
s_bcnt1_i32_b64 s30, vcc
s_ff1_i32_b64 s31, vcc
s_lshl_b32 s31, s31, 8
s_or_b32 s30, s30, s31
v_cmp_ne_u32_e32 vcc, 0, v16
s_bcnt1_i32_b64 s32, vcc
s_ff1_i32_b64 s33, vcc
s_lshl_b32 s33, s33, 8
s_or_b32 s32, s32, s33
s_bcnt1_i32_b64 s34, 0xff00ff
s_ff1_i32_b64 s35, 0xff00
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
