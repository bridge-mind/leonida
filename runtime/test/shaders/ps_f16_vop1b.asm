; f16 unary (VOP1) ops read the low half and keep the high half of the destination.
; v0 = 4.0 | 0.25 << 16, v1 = -2.75 | 1.5 << 16 (low | high half); destinations start as 0xaaaa5555.
;   v_fract_f16 -2.75 - floor(-2.75) = 0.25    v_sqrt_f16 sqrt(4) = 2.0    v_rcp_f16_e64 -v0: 1 / -4 = -0.25
;   v_rsq_f16 1 / sqrt(4) = 0.5
; The words are the ones an RX 6800 (gfx1030) gave for this body run as a raw kernel, wave32 and wave64.
; Expected mrt0 bits: 0xaaaa3400, 0xaaaa4000, 0xaaaab400, 0xaaaa3800
v_mov_b32 v0, 0x34004400
v_mov_b32 v1, 0x3e00c180
v_mov_b32 v2, 0xaaaa5555
v_mov_b32 v3, 0xaaaa5555
v_mov_b32 v4, 0xaaaa5555
v_mov_b32 v5, 0xaaaa5555
v_fract_f16 v2, v1
v_sqrt_f16 v3, v0
v_rcp_f16_e64 v4, -v0
v_rsq_f16 v5, v0
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
