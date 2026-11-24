; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Two-operand integer min/max (VOP2), signed and unsigned on the same pair. vN holds hN.
; Inputs come from memory at run time (not inline constants, which the driver could fold). The V# in s[0:3]
; points at 0x1000 (num_records 32); the test seeds 0x1000..0x101c with h0..h7 =
;   h0 0xab800000  bit 23 set, bits 24-31 garbage: i24 = -2^23, u24 = 0x800000
;   h1 0x12fffffe  i24 = -2, u24 = 0xfffffe
;   h2 0x00000021  33: a shift or field size of 32 or more, taken mod 32 (= 1)
;   h3 0x80000001  negative as i32, large as u32
;   h4 0x7ffffffe  positive either way
;   h5 0xf0f0f0f0
;   h6 0x12345678
;   h7 0xffffffff
; Each word is derived by hand from the RDNA2 ISA pseudocode quoted next to it, never from the translator.
;   v_max_i32  D = S0.i >= S1.i ? S0 : S1      v_min_i32  D = S0.i < S1.i ? S0 : S1
;   v_max_u32  D = S0.u >= S1.u ? S0 : S1      v_min_u32  D = S0.u < S1.u ? S0 : S1
;   S0 = h3 = 0x80000001 (-2147483647 signed, 2147483649 unsigned), S1 = h4 = 0x7ffffffe (2147483646 either way)
;     v_max_i32 = h4 = 0x7ffffffe   v_min_i32 = h3 = 0x80000001
;     v_max_u32 = h3 = 0x80000001   v_min_u32 = h4 = 0x7ffffffe
;   (a signed/unsigned mix-up swaps the words of each pair)
; Expected mrt0 bits: 0x7ffffffe, 0x80000001, 0x80000001, 0x7ffffffe
s_buffer_load_dwordx8 s[4:11], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v0, s4
v_mov_b32_e32 v1, s5
v_mov_b32_e32 v2, s6
v_mov_b32_e32 v3, s7
v_mov_b32_e32 v4, s8
v_mov_b32_e32 v5, s9
v_mov_b32_e32 v6, s10
v_mov_b32_e32 v7, s11
v_max_i32_e32 v8, v3, v4
v_min_i32_e32 v9, v3, v4
v_max_u32_e32 v10, v3, v4
v_min_u32_e32 v11, v3, v4
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
