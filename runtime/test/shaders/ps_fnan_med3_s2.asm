; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_med3_f32 with a quiet NaN in S2, or in two operands: the cases ps_fnan_med3 does not reach (it puts the NaN in S0
; only). Inputs are read from memory at run time (not inline constants, which the driver could fold). The V# in s[0:3]
; points at 0x1000 (num_records 16); the test seeds 0x1000..0x100c with a..d = 0x40a00000 0x40e00000 0x7fc10001
; 0xc0200000 (5.0, 7.0, NaN, -2.5).
; Expected words, from the rule measured on RDNA2 hardware (docs/design.md, "Float min/max/med3 with NaN": with any
; NaN operand the result is min3(s0, s1, s2); min/max: one NaN operand gives the other, both NaN give src0):
;   med3(5.0, 7.0, NaN)   = min(min(5.0, 7.0), NaN)  = 5.0   0x40a00000
;   med3(NaN, NaN, 5.0)   = min(min(NaN, NaN), 5.0)  = 5.0   0x40a00000
;   med3(NaN, -2.5, NaN)  = min(min(NaN, -2.5), NaN) = -2.5  0xc0200000
;   med3(7.0, NaN, NaN)   = min(min(7.0, NaN), NaN)  = 7.0   0x40e00000
; Expected mrt0 bits: 0x40a00000, 0x40a00000, 0xc0200000, 0x40e00000
; The first two rows are the ones suggested in the #2029 review. The all-NaN case is left out: its NaN payload is not
; modelled (docs/design.md).
s_buffer_load_dwordx4 s[4:7], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
v_mov_b32_e32 v4, s7
v_med3_f32 v5, v1, v2, v3
v_med3_f32 v6, v3, v3, v1
v_med3_f32 v7, v3, v4, v3
v_med3_f32 v8, v2, v3, v3
exp mrt0 v5, v6, v7, v8 done vm
s_endpgm
