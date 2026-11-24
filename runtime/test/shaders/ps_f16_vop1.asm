; f16 unary (VOP1) rounding ops read the low half and keep the high half of the destination.
; v0 = 2.5 | -1.5 << 16, v1 = -0.75 | 3.5 << 16 (low | high half); destinations start as 0xaaaa5555.
;   v_rndne_f16 2.5 -> 2.0 (ties to even)    v_floor_f16 -0.75 -> -1.0    v_ceil_f16_e64 -v0: ceil(-2.5) = -2.0
;   v_trunc_f16 -0.75 -> -0.0 (0x8000: the zero keeps the sign)
; The words are the ones an RX 6800 (gfx1030) gave for this body run as a raw kernel, wave32 and wave64.
; Expected mrt0 bits: 0xaaaa4000, 0xaaaabc00, 0xaaaac000, 0xaaaa8000
v_mov_b32 v0, 0xbe004100
v_mov_b32 v1, 0x4300ba00
v_mov_b32 v2, 0xaaaa5555
v_mov_b32 v3, 0xaaaa5555
v_mov_b32 v4, 0xaaaa5555
v_mov_b32 v5, 0xaaaa5555
v_rndne_f16 v2, v0
v_floor_f16 v3, v1
v_ceil_f16_e64 v4, -v0
v_trunc_f16 v5, v1
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
