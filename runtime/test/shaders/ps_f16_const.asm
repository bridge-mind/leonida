; Inline float constants in f16 ops are f16 values: the constant sits in the low half of the operand and the high
; half is zero (so the high half that op_sel selects from a constant reads +0.0). They are not f32 bit patterns.
; v0 = 0.25 | 3.0 << 16 (low | high half); destinations start as 0xaaaa5555.
;   v_add_f16 1.0 + 0.25 = 1.25
;   v_med3_f16 v0, -4.0, 0.5 [1,0,0,1]: med(3, -4, 0.5) = 0.5 into the high half
;   v_fma_f16 v0, 2.0, 0.5 [0,0,1,0]: 0.25 * 2 + (high half of the constant = 0) = 0.5
;   v_cvt_f32_f16 0.5 = 0x3f000000
; The words are the ones an RX 6800 (gfx1030) gave for this body run as a raw kernel, wave32 and wave64.
; Expected mrt0 bits: 0xaaaa3d00, 0x38005555, 0xaaaa3800, 0x3f000000
v_mov_b32 v0, 0x42003400
v_mov_b32 v2, 0xaaaa5555
v_mov_b32 v3, 0xaaaa5555
v_mov_b32 v4, 0xaaaa5555
v_mov_b32 v5, 0xaaaa5555
v_add_f16 v2, 1.0, v0
v_med3_f16 v3, v0, -4.0, 0.5 op_sel:[1,0,0,1]
v_fma_f16 v4, v0, 2.0, 0.5 op_sel:[0,0,1,0]
v_cvt_f32_f16 v5, 0.5
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
