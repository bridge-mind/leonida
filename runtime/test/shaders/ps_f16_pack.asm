; v_pack_b32_f16 with inline float constants and op_sel: dst = half of src0 | half of src1 << 16. A constant is
; an f16 value in the low half of the operand with a zero high half (see ps_f16_const).
; v0 = 0.25 | 3.0 << 16, v1 = 4.0 | 0.25 << 16 (low | high half).
;   1.0, v0: 0x3c00 | 0x3400 << 16      v0, 0.5: 0x3400 | 0x3800 << 16
;   2.0, v0 op_sel:[1,0,0]: the high half of the constant (0) | 0x3400 << 16
;   v0, v1 op_sel:[1,0,0]: the high half of v0 (0x4200) | the low half of v1 (0x4400) << 16
; The words are the ones an RX 6800 (gfx1030) gave for this body run as a raw kernel, wave32 and wave64.
; Expected mrt0 bits: 0x34003c00, 0x38003400, 0x34000000, 0x44004200
v_mov_b32 v0, 0x42003400
v_mov_b32 v1, 0x34004400
v_pack_b32_f16 v2, 1.0, v0
v_pack_b32_f16 v3, v0, 0.5
v_pack_b32_f16 v4, 2.0, v0 op_sel:[1,0,0]
v_pack_b32_f16 v5, v0, v1 op_sel:[1,0,0]
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
