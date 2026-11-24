; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; SDWA source select on a float inline constant in an f16 op. Assemble with llvm-mc -mcpu=gfx1013 -show-encoding,
; translate, run with vk_harness. In a 16-bit float op an inline float constant is the f16 value in bits 15:0 over a
; zero high half (docs/design.md, "f16 inline constants"); SDWA src<i>_sel then narrows those 32 bits like a
; register's (zero-extended), before the op. An SGPR source and an integer op reading the f32 bits of 1.0 are
; narrowed the same way. dst_sel:WORD_0 with UNUSED_PRESERVE keeps the destination's 0xaaaa high half.
; v1 = 0.5 | 1024.0 << 16 (0x64003800), s8 = 0x3c000000 (1.0 in the high half), v2..v5 start as 0xaaaa5555.
;   x: v_add_f16_sdwa v2, 1.0, v1 src0_sel:WORD_1 src1_sel:WORD_0
;        1.0 = 0x00003c00, WORD_1 = 0x0000 = +0.0; +0.0 + 0.5 = 0.5 = 0x3800 -> 0xaaaa3800
;        (the unselected constant would give 1.5 = 0x3e00)
;   y: v_mul_f16_sdwa v3, 2.0, v1 src0_sel:BYTE_1 src1_sel:WORD_1
;        2.0 = 0x00004000, BYTE_1 = 0x0040 = 64 * 2^-24 = 2^-18 (f16 denormal); * 1024.0 = 2^-8 = 0x1c00
;        -> 0xaaaa1c00 (the unselected constant would give 2048.0 = 0x6800)
;   z: v_add_f16_sdwa v4, s8, v1 src0_sel:WORD_1 src1_sel:WORD_0
;        s8 WORD_1 = 0x3c00 = 1.0; 1.0 + 0.5 = 1.5 = 0x3e00 -> 0xaaaa3e00
;   w: v_cvt_f16_u16_sdwa v5, 1.0 src0_sel:WORD_1 (integer source: the f32 bits 0x3f800000)
;        WORD_1 = 0x3f80 = 16256 = 2^13 * (1 + 1008/1024) -> f16 exponent 13 + 15 = 28: 0x7000 | 0x3f0 = 0x73f0
;        -> 0xaaaa73f0
; Expected mrt0 bits: 0xaaaa3800, 0xaaaa1c00, 0xaaaa3e00, 0xaaaa73f0
; Measured: the same four words on an AMD Radeon iGPU (gfx1036, Ryzen 9 9950X3D, driver 26.10.1) through
; tools/hw_oracle (raw HIP kernel, wave32), the same instructions writing v8..v11.
v_mov_b32 v1, 0x64003800
v_mov_b32 v2, 0xaaaa5555
v_mov_b32 v3, 0xaaaa5555
v_mov_b32 v4, 0xaaaa5555
v_mov_b32 v5, 0xaaaa5555
s_mov_b32 s8, 0x3c000000
v_add_f16_sdwa v2, 1.0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_0
v_mul_f16_sdwa v3, 2.0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_1 src1_sel:WORD_1
v_add_f16_sdwa v4, s8, v1 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_0
v_cvt_f16_u16_sdwa v5, 1.0 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
