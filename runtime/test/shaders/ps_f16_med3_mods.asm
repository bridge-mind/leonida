; v_med3_f16 op_sel with neg/abs (they apply to the selected half), clamp, and a destination that is also a source.
; v0 = 0.5 | -2.5 << 16, v1 = 2.0 | -8.0 << 16, v2 = 1.0 | 0.75 << 16 (low | high half).
;   -v0, |v1|, v2 [1,1,0,1]          med(2.5, 8, 1) = 2.5 into the high half of 0xaaaa5555
;   |v0|, |v1|, v2 [1,1,1,1] clamp   med(2.5, 8, 0.75) = 2.5, clamped to 1.0, into the high half of 0xaaaa5555
;   v5 = v0; -v5, v1, |v1| [1,0,1,1] med(2.5, 2, 8) = 2.5 into the high half of v5, which was just read; low half kept
;   v0, v1, -|v2| [0,1,1,0]          med(0.5, -8, -0.75) = -0.75 into the low half of 0xaaaa5555
; The words are the ones an RX 6800 (gfx1030) gave for this body run as a raw kernel, wave32 and wave64.
; Expected mrt0 bits: 0x41005555, 0x3c005555, 0x41003800, 0xaaaaba00
v_mov_b32 v0, 0xc1003800
v_mov_b32 v1, 0xc8004000
v_mov_b32 v2, 0x3a003c00
v_mov_b32 v3, 0xaaaa5555
v_mov_b32 v4, 0xaaaa5555
v_mov_b32 v5, v0
v_mov_b32 v6, 0xaaaa5555
v_med3_f16 v3, -v0, |v1|, v2 op_sel:[1,1,0,1]
v_med3_f16 v4, |v0|, |v1|, v2 op_sel:[1,1,1,1] clamp
v_med3_f16 v5, -v5, v1, |v1| op_sel:[1,0,1,1]
v_med3_f16 v6, v0, v1, -|v2| op_sel:[0,1,1,0]
exp mrt0 v3, v4, v5, v6 done vm
s_endpgm
