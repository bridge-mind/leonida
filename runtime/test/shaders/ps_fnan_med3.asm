; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_med3_f32 / v_min3_f32 / v_max3_f32 with a quiet-NaN operand, inputs read from memory at run time (not inline
; constants, which the driver could fold). The V# in s[0:3] points at 0x1000 (num_records 32); the test seeds
; 0x1000..0x1014 with a..f = 0x7fc10001 0x40e00000 0x40a00000 0x7fc10001 0x40600000 0xc0200000
; (NaN, 7.0, 5.0, NaN, 3.5, -2.5).
; Expected words, measured on RDNA2 hardware (AMD Radeon RX 6800, gfx1030, hw-oracle tool from the AnyPS5
; project, all four IEEE/DX10_CLAMP mode combinations, wave32 and wave64):
;   med3(NaN, 7.0, 5.0)   = 0x40a00000  (any NaN operand: min3 of the operands, NaN ignored)
;   med3(NaN, 3.5, -2.5)  = 0xc0200000
;   min3(NaN, -2.5, 3.5)  = 0xc0200000  (min/max: a NaN operand gives the other operand)
;   max3(NaN, -2.5, 3.5)  = 0x40600000
; Expected mrt0 bits: 0x40a00000, 0xc0200000, 0xc0200000, 0x40600000
; The translator before the med3 NaN rule gave 0x40e00000 and 0x40600000 for the two med3 words on an RX 6800.
s_buffer_load_dwordx8 s[4:11], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
v_mov_b32_e32 v4, s7
v_mov_b32_e32 v5, s8
v_mov_b32_e32 v6, s9
v_med3_f32 v7, v1, v2, v3
v_med3_f32 v8, v4, v5, v6
v_min3_f32 v9, v1, v6, v5
v_max3_f32 v10, v1, v6, v5
exp mrt0 v7, v8, v9, v10 done vm
s_endpgm
