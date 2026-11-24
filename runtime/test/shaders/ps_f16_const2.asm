; Inline float constants in packed f16 ops, v_cos_f16 and an f16 compare: an f16 value in the low half of the
; operand, zero in the high half (see ps_f16_const).
; v0 = 0.25 | 3.0 << 16 (low | high half).
;   v_pk_add_f16 1.0, v0: low 1.0 + 0.25 = 1.25, high 0 + 3.0 = 3.0
;   the same with op_sel:[1,1] op_sel_hi:[0,0]: low 0 + 3.0 = 3.0, high 1.0 + 0.25 = 1.25
;   v_cos_f16 0.5 (turns) = -1.0, high half of 0xaaaa5555 kept
;   v_cmp_lt_f16 0.5 < 0.25 is false, so v_cndmask_b32 picks 3 (it would pick 7 if the constant were read as 0.0)
; The words are the ones an RX 6800 (gfx1030) gave for this body run as a raw kernel, wave32 and wave64.
; Expected mrt0 bits: 0x42003d00, 0x3d004200, 0xaaaabc00, 3
v_mov_b32 v0, 0x42003400
v_mov_b32 v4, 0xaaaa5555
v_pk_add_f16 v2, 1.0, v0
v_pk_add_f16 v3, 1.0, v0 op_sel:[1,1] op_sel_hi:[0,0]
v_cos_f16 v4, 0.5
v_cmp_lt_f16_e64 s1, 0.5, v0
v_cndmask_b32_e64 v5, 3, 7, s1
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
