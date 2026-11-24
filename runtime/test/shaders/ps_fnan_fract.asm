; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_fract_f32 / v_fract_f16 next to 1.0 and v_max_f32 / v_min_f32 with a quiet-NaN operand, inputs read from memory
; at run time (not inline constants, which the driver could fold). The V# in s[0:3] points at 0x1000
; (num_records 16); the test seeds 0x1000..0x100c with a..d = 0xaedbe6ff 0x00008800 0x7fc10001 0xc0200000
; (-1e-10, f16 -2^-13 in the low half, NaN, -2.5).
; Expected words, measured on RDNA2 hardware (AMD Radeon RX 6800, gfx1030, hw-oracle tool from the AnyPS5
; project, all four IEEE/DX10_CLAMP mode combinations, wave32 and wave64):
;   fract_f32(-1e-10)       = 0x3f7fffff  (1 - 1e-10 rounds to 1.0; the hardware gives the largest float below 1)
;   fract_f16(-2^-13)       = 0x3bff      (likewise in f16; the high half of the destination, 0, is kept)
;   max_f32(NaN, -2.5)      = 0xc0200000  (a NaN operand gives the other operand)
;   min_f32(-2.5, NaN)      = 0xc0200000
; Expected mrt0 bits: 0x3f7fffff, 0x00003bff, 0xc0200000, 0xc0200000
s_buffer_load_dwordx4 s[4:7], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
v_mov_b32_e32 v4, s7
v_fract_f32_e32 v5, v1
v_fract_f16_e32 v2, v2
v_max_f32_e32 v6, v3, v4
v_min_f32_e32 v7, v4, v3
exp mrt0 v5, v2, v6, v7 done vm
s_endpgm
