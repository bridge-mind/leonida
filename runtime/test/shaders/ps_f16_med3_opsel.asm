; v_med3_f16 op_sel, one bit at a time: bit i reads the high half of source i, the last bit writes the high half
; of the destination; the other half of the destination is kept.
; v0 = 1.0 | 2.5 << 16, v1 = 2.0 | 8.0 << 16, v2 = 3.0 | 1.5 << 16 (low | high half); destinations start as 0xaaaa5555.
;   [1,0,0,0] med(2.5, 2, 3) = 2.5    [0,1,0,0] med(1, 8, 3) = 3    [0,0,1,0] med(1, 2, 1.5) = 1.5
;   [0,0,0,1] med(1, 2, 3) = 2 into the high half
; The words are the ones an RX 6800 (gfx1030) gave for this body run as a raw kernel, wave32 and wave64.
; Expected mrt0 bits: 0xaaaa4100, 0xaaaa4200, 0xaaaa3e00, 0x40005555
v_mov_b32 v0, 0x41003c00
v_mov_b32 v1, 0x48004000
v_mov_b32 v2, 0x3e004200
v_mov_b32 v3, 0xaaaa5555
v_mov_b32 v4, 0xaaaa5555
v_mov_b32 v5, 0xaaaa5555
v_mov_b32 v6, 0xaaaa5555
v_med3_f16 v3, v0, v1, v2 op_sel:[1,0,0,0]
v_med3_f16 v4, v0, v1, v2 op_sel:[0,1,0,0]
v_med3_f16 v5, v0, v1, v2 op_sel:[0,0,1,0]
v_med3_f16 v6, v0, v1, v2 op_sel:[0,0,0,1]
exp mrt0 v3, v4, v5, v6 done vm
s_endpgm
