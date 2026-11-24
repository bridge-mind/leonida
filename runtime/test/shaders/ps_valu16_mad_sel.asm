; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_mad_u32_u16 / v_mad_i32_i16 with op_sel on S2 or D and with clamp, the forms the translator used to refuse.
; Measured on an AMD Radeon iGPU (gfx1036, RDNA2, Windows driver 26.10.1) with raw kernels through tools/hw_oracle,
; wave32 and wave64 alike (EGEN-HW-MAD-U32-U16):
;   op_sel bit 2 (S2) adds S2[31:16] zero-extended, in the i16 op too (S2 = 0xffff8000 adds 0xffff, not -1);
;   op_sel bit 3 (D) writes the 32-bit result shifted left by 16: its low half to D's high half, 0 to D's low half;
;   clamp saturates the 32-bit sum, to [0, 0xffffffff] (u16) or [0x80000000, 0x7fffffff] (i16), before that shift.
; Otherwise as ps_valu16_mad: D = S0.half * S1.half + S2 mod 2^32, the halves zero- (u16) or sign-extended (i16),
; op_sel bit 0 / 1 picks the high half of S0 / S1; the destinations' 0xaaaa5555 is never kept.
; v0 = 0x1234ffff, v1 = 0x5678fffe, v3 = 0x8001fff0 (high half 0x8001: negative as an i16)
; v5 = 0x00018000 (low -32768), v6 = 0xffff7fff (low 32767), v7 = 0x80000000
;   x: v_mad_i32_i16 v2, v0, v1, v3 op_sel:[0,0,1,0]: (-1) * (-2) = 2; + 0x8001 = 0x00008003
;      (S2 high half sign-extended: 0xffff8003; whole S2: 0x8001fff2)
;   y: v_mad_u32_u16 v4, v0, v1, v3 op_sel:[1,0,0,1]: 0x1234 * 0xfffe = 0x1233db98; + 0x8001fff0 = 0x9235db88;
;      << 16 = 0xdb880000
;   z: v_mad_u32_u16 v8, v0, v0, v3 clamp: 0xffff * 0xffff = 0xfffe0001; + 0x8001fff0 = 0x1_7ffffff1 > 0xffffffff,
;      so 0xffffffff (wrapping: 0x7ffffff1)
;   w: v_mad_i32_i16 v9, v5, v6, v7 clamp: -32768 * 32767 = -1073709056; + (-2^31) < -2^31, so 0x80000000
;      (wrapping: 0x40008000)
; Expected mrt0 bits: 0x00008003, 0xdb880000, 0xffffffff, 0x80000000
v_mov_b32 v0, 0x1234ffff
v_mov_b32 v1, 0x5678fffe
v_mov_b32 v3, 0x8001fff0
v_mov_b32 v5, 0x00018000
v_mov_b32 v6, 0xffff7fff
v_mov_b32 v7, 0x80000000
v_mov_b32 v2, 0xaaaa5555
v_mov_b32 v4, 0xaaaa5555
v_mov_b32 v8, 0xaaaa5555
v_mov_b32 v9, 0xaaaa5555
v_mad_i32_i16 v2, v0, v1, v3 op_sel:[0,0,1,0]
v_mad_u32_u16 v4, v0, v1, v3 op_sel:[1,0,0,1]
v_mad_u32_u16 v8, v0, v0, v3 clamp
v_mad_i32_i16 v9, v5, v6, v7 clamp
exp mrt0 v2, v4, v8, v9 done vm
s_endpgm
