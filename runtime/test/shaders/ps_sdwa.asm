; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; SDWA GPU check. Assemble with llvm-mc -mcpu=gfx1013 -show-encoding, translate, run with vk_harness.
; Expected mrt0 bits: 0x000000f1, 0xbffefff2, 0x3c003344, 0xbff70000
; XOR bit 30 before export to encode the integer result as a finite RGBA32F value.
v_mov_b32 v0, 0x8081f0f7
v_mov_b32 v1, 0x00010005
v_add_nc_u32_sdwa v2, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:WORD_1
v_subrev_nc_u32_sdwa v3, v1, sext(v0) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
v_mov_b32 v4, 0x11223344
v_mov_b32 v5, 1.0
v_cvt_f16_f32_sdwa v4, v5 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
v_or_b32_sdwa v6, v0, v1 dst_sel:BYTE_2 dst_unused:UNUSED_SEXT src0_sel:BYTE_0 src1_sel:BYTE_0
v_xor_b32_e32 v3, 0x40000000, v3
v_xor_b32_e32 v6, 0x40000000, v6
exp mrt0 v2, v3, v4, v6 done vm
s_endpgm
