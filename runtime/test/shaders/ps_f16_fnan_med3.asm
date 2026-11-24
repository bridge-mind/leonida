; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_med3_f16 with a quiet NaN in S2, or in two operands: the f16 form of ps_fnan_med3_s2. Inputs are read from memory
; at run time; each word holds one f16 in its low half. The V# in s[0:3] points at 0x1000 (num_records 16); the test
; seeds 0x1000..0x100c with a..d = 0x00004500 0x00004700 0x00007e01 0x0000c100 (5.0, 7.0, NaN, -2.5 as f16).
; v_med3_f16 (op_sel 0) writes the low half of its destination and keeps the high half (docs/design.md, "16-bit
; results"); destinations start as 0xaaaa5555.
; Expected words, from the rule measured on RDNA2 hardware (docs/design.md, "Float min/max/med3 with NaN": with any
; NaN operand the result is min3(s0, s1, s2); min/max: one NaN operand gives the other, both NaN give src0):
;   med3(5.0, 7.0, NaN)   = 5.0   0x4500
;   med3(NaN, NaN, 5.0)   = 5.0   0x4500
;   med3(NaN, -2.5, NaN)  = -2.5  0xc100
;   med3(7.0, NaN, NaN)   = 7.0   0x4700
; Expected mrt0 bits: 0xaaaa4500, 0xaaaa4500, 0xaaaac100, 0xaaaa4700
; The all-NaN case is left out: its NaN payload is not modelled (docs/design.md).
s_buffer_load_dwordx4 s[4:7], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
v_mov_b32_e32 v4, s7
v_mov_b32 v5, 0xaaaa5555
v_mov_b32 v6, 0xaaaa5555
v_mov_b32 v7, 0xaaaa5555
v_mov_b32 v8, 0xaaaa5555
v_med3_f16 v5, v1, v2, v3
v_med3_f16 v6, v3, v3, v1
v_med3_f16 v7, v3, v4, v3
v_med3_f16 v8, v2, v3, v3
exp mrt0 v5, v6, v7, v8 done vm
s_endpgm
