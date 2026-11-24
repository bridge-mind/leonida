; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Bit-field mask and bit reverse: v_bfm_b32 (VOP3), v_bfrev_b32. vN holds hN.
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
;   v_bfm_b32       D = ((1 << S0[4:0]) - 1) << S1[4:0]   (S0 the field size, S1 the offset)
;     size h7, offset h2 = size 0xffffffff & 31 = 31, offset 33 & 31 = 1: 0x7fffffff << 1  = 0xfffffffe
;     size h2, offset h1 = size 33 & 31 = 1, offset 0x12fffffe & 31 = 30: 1 << 30           = 0x40000000
;   v_bfrev_b32     D[31:0] = S0[0:31]
;     h6 = 0001 0010 0011 0100 0101 0110 0111 1000 reversed = 0001 1110 0110 1010 0010 1100 0100 1000 = 0x1e6a2c48
;     h0 = 0xab800000: its top nine bits 1010 1011 1 land in bits 0..8 reversed, 1 1101 0101             = 0x000001d5
; Expected mrt0 bits: 0xfffffffe, 0x40000000, 0x1e6a2c48, 0x000001d5
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
v_bfm_b32 v8, v7, v2
v_bfm_b32 v9, v2, v1
v_bfrev_b32_e32 v10, v6
v_bfrev_b32_e32 v11, v0
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
