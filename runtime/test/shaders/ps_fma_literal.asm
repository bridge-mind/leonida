; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Which operand the literal K multiplies. v_fmamk_f32/v_madmk_f32 (printed vdst, src0, K, vsrc1): D = S0 * K + S1.
; v_fmaak_f32/v_madak_f32 (printed vdst, src0, vsrc1, K): D = S0 * S1 + K. All values are exact in f32, and each
; word changes if K and vsrc1 trade places: fmamk 2*10+3 = 23, fmaak 2*3+10 = 16, madmk 4*0.5+5 = 7, madak 4*5+0.5 = 20.5.
; Expected mrt0 bits: 0x41b80000, 0x41800000, 0x40e00000, 0x41a40000
v_mov_b32 v1, 2.0
v_mov_b32 v3, 0x40400000
v_fmamk_f32 v5, v1, 0x41200000, v3
v_fmaak_f32 v6, v1, v3, 0x41200000
v_mov_b32 v2, 4.0
v_mov_b32 v4, 0x40a00000
v_madmk_f32 v7, v2, 0x3f000000, v4
v_madak_f32 v8, v2, v4, 0x3f000000
exp mrt0 v5, v6, v7, v8 done vm
s_endpgm
