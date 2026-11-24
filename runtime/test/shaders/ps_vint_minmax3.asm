; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Three-operand integer min/max (VOP3): v_max3_u32, v_min3_i32. vN holds hN.
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
;   v_max3_u32  D = max(S0.u, S1.u, S2.u)        v_min3_i32  D = min(S0.i, S1.i, S2.i)
;     max3_u32(h4, h3, h5) = max(0x7ffffffe, 0x80000001, 0xf0f0f0f0)                        = 0xf0f0f0f0
;       (signed: 0x7ffffffe)
;     max3_u32(h0, h6, h3) = max(0xab800000, 0x12345678, 0x80000001)                        = 0xab800000
;       (signed: 0x12345678)
;     min3_i32(h5, h4, h6): 0xf0f0f0f0 = -252645136 is the only negative one                = 0xf0f0f0f0
;       (unsigned: 0x12345678)
;     min3_i32(h7, h6, h0): -1, 0x12345678, 0xab800000 = -1417674752                        = 0xab800000
;       (unsigned: 0x12345678)
;   The extreme operand sits in S2, S0, S0 and S2.
; Expected mrt0 bits: 0xf0f0f0f0, 0xab800000, 0xf0f0f0f0, 0xab800000
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
v_max3_u32 v8, v4, v3, v5
v_max3_u32 v9, v0, v6, v3
v_min3_i32 v10, v5, v4, v6
v_min3_i32 v11, v7, v6, v0
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
