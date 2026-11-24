; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Three-operand integer logic (VOP3): v_add_lshl_u32, v_and_or_b32, v_xor3_b32, v_bfi_b32. vN holds hN.
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
;   v_add_lshl_u32  D = (S0 + S1) << S2[4:0]
;     (h3 + h4) << h2 = (0x80000001 + 0x7ffffffe) << (33 & 31) = 0xffffffff << 1          = 0xfffffffe
;     (shifting before adding gives 0x80000000; an unmasked shift by 33 is undefined in SPIR-V)
;   v_and_or_b32    D = (S0 & S1) | S2
;     (h6 & h5) | h0 = (0x12345678 & 0xf0f0f0f0) | 0xab800000 = 0x10305070 | 0xab800000   = 0xbbb05070
;   v_xor3_b32      D = S0 ^ S1 ^ S2
;     h5 ^ h6 ^ h7 = 0xf0f0f0f0 ^ 0x12345678 ^ 0xffffffff = 0xe2c4a688 ^ 0xffffffff       = 0x1d3b5977
;   v_bfi_b32       D = (S0 & S1) | (~S0 & S2), S0 the mask
;     (h5 & h6) | (~h5 & h1) = 0x10305070 | (0x0f0f0f0f & 0x12fffffe) = 0x10305070 | 0x020f0f0e = 0x123f5f7e
; Expected mrt0 bits: 0xfffffffe, 0xbbb05070, 0x1d3b5977, 0x123f5f7e
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
v_add_lshl_u32 v8, v3, v4, v2
v_and_or_b32 v9, v6, v5, v0
v_xor3_b32 v10, v5, v6, v7
v_bfi_b32 v11, v5, v6, v1
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
