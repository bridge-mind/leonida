; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; 24-bit multiplies: v_mad_i32_i24 (VOP3, no clamp), v_mul_hi_u32_u24 (VOP2). vN holds hN.
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
;   v_mad_i32_i24   D = S0.i[23:0] * S1.i[23:0] + S2.i, the 24-bit fields sign-extended from bit 23, low 32 bits
;     h0, h1, h6: (-2^23) * (-2) + 0x12345678 = 0x01000000 + 0x12345678                   = 0x13345678
;       (zero-extended fields give 0x800000 * 0xfffffe = 0x7fffff000000, low word 0xff000000 + h6 = 0x11345678)
;     h4, h3, h7: i24(0x7ffffffe) = -2, i24(0x80000001) = 1: -2 * 1 + (-1) = -3            = 0xfffffffd
;       (zero-extended: 0xfffffe + 0xffffffff = 0x00fffffd)
;   v_mul_hi_u32_u24  D = (S0.u[23:0] * S1.u[23:0]) >> 32, the 24-bit fields zero-extended
;     h0, h1: 0x800000 * 0xfffffe = 0x7fffff000000, >> 32                                  = 0x00007fff
;       (sign-extended fields give (-2^23) * (-2) = 2^24, high word 0)
;     h7, h7: 0xffffff * 0xffffff = 0xfffffe000001, >> 32                                  = 0x0000ffff
;       (sign-extended: (-1) * (-1) = 1, high word 0)
; Expected mrt0 bits: 0x13345678, 0xfffffffd, 0x00007fff, 0x0000ffff
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
v_mad_i32_i24 v8, v0, v1, v6
v_mad_i32_i24 v9, v4, v3, v7
v_mul_hi_u32_u24_e32 v10, v0, v1
v_mul_hi_u32_u24_e32 v11, v7, v7
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
