; v_cndmask_b32_e64 source modifiers are sign-bit moves (docs/design.md "v_cndmask_b32 source modifiers"): abs clears
; bit 31 and neg flips it on the raw bits, so a signalling NaN stays signalling and a denormal isn't flushed.
;   x: s4 = 0 selects src0, -v1 of the sNaN 0x7f800001 = 0xff800001 (a float negate may quiet it to 0xffc00001)
;   y: s4 = 0 selects src0, |v6| of the sNaN 0xff800001 = 0x7f800001
;   z: s4 = 0 selects src0, -|v7| of the denormal 0x00000001 = 0x80000001 (a flushing float op gives 0x80000000)
;   w: s5 = all lanes selects src1, -v6 of 0xff800001 = 0x7f800001
; Expected mrt0 bits: 0xff800001, 0x7f800001, 0x80000001, 0x7f800001
v_mov_b32_e32 v1, 0x7f800001
v_mov_b32_e32 v6, 0xff800001
v_mov_b32_e32 v7, 1
s_mov_b32 s4, 0
s_mov_b32 s5, -1
v_cndmask_b32_e64 v2, -v1, v6, s4
v_cndmask_b32_e64 v3, |v6|, v1, s4
v_cndmask_b32_e64 v4, -|v7|, v1, s4
v_cndmask_b32_e64 v5, v1, -v6, s5
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
