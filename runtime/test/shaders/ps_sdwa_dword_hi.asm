; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; 16-bit SDWA results written with dst_sel:DWORD. The non-SDWA f16 VOP1/VOP2 forms keep the destination's high
; half (docs/design.md, 16-bit results), but an SDWA destination with dst_sel:DWORD takes the whole dword: the
; 16-bit result zero-extended, whatever dst_unused says. Measured natively on an AMD Radeon iGPU (gfx1036, RDNA2,
; Windows driver 26.10.1) through tools/hw_oracle with this exact program: 0x00004000, 0x00000003, 0x00004700,
; 0x0000c000; the same four ops with dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE gave 0xdead4000, 0xdead0003,
; 0xdead4700, 0xdeadc000 (the high half kept), so the zero comes from dst_sel:DWORD.
; Every destination starts as 0xdeadbeef. Sources: v2 = 0x12343c00 (WORD_0 = 1.0), v4 = 0xc0004200 (WORD_0 = 3.0,
; WORD_1 = -2.0), v6 = 0x00070000 (WORD_1 = 7).
;   x: v_add_f16    1.0 + 1.0 = 2.0 = 0x4000, UNUSED_PRESERVE          -> 0x00004000
;   y: v_cvt_u16_f16 of 3.0 = 3 = 0x0003, UNUSED_PAD                    -> 0x00000003
;   z: v_cvt_f16_u16 of 7 = 7.0 = 0x4700 (exact), UNUSED_PRESERVE      -> 0x00004700
;   w: v_mul_f16    1.0 * -2.0 = -2.0 = 0xc000, UNUSED_SEXT            -> 0x0000c000 (no sign copies above it)
; All four are small positive denormal floats, kept bit-exact by the RGBA32F target (as 0x000000f1 in ps_sdwa).
; Expected mrt0 bits: 0x00004000, 0x00000003, 0x00004700, 0x0000c000
v_mov_b32 v1, 0xdeadbeef
v_mov_b32 v2, 0x12343c00
v_mov_b32 v3, 0xdeadbeef
v_mov_b32 v4, 0xc0004200
v_mov_b32 v5, 0xdeadbeef
v_mov_b32 v6, 0x70000
v_mov_b32 v7, 0xdeadbeef
v_add_f16_sdwa v1, v2, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_cvt_u16_f16_sdwa v3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0
v_cvt_f16_u16_sdwa v5, v6 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_mul_f16_sdwa v7, v2, v4 dst_sel:DWORD dst_unused:UNUSED_SEXT src0_sel:WORD_0 src1_sel:WORD_1
exp mrt0 v1, v3, v5, v7 done vm
s_endpgm
