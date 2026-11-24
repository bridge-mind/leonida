; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_mac_f32 is the VOP2 form of v_mad_f32: the product is rounded before the add (not fused). VOP3 clamp and
; output modifiers on conversions: v_cvt_f32_ubyte1 with clamp, v_cvt_f32_u32 with mul:4, v_cvt_f32_f16 with
; div:2 and v_rcp_iflag_f32 with mul:4, the last two summed (0.5 + 1.0; each modifier missing gives a different sum).
; Expected mrt0 bits: 0xc0256a01, 0x3f800000, 0x41e00000, 0x3fc00000
v_mov_b32_e32 v0, 0xbc500c59
v_mov_b32_e32 v1, 0x435a2099
v_mov_b32_e32 v2, 0x3e3db015
v_mac_f32_e32 v2, v0, v1
v_mov_b32_e32 v3, 0x12347856
v_cvt_f32_ubyte1_e64 v4, v3 clamp
v_mov_b32_e32 v5, 7
v_cvt_f32_u32_e64 v6, v5 mul:4
v_mov_b32_e32 v7, 0x3c00
v_cvt_f32_f16_e64 v9, v7 div:2
v_mov_b32_e32 v8, 0x40800000
v_rcp_iflag_f32_e64 v10, v8 mul:4
v_add_f32_e32 v11, v9, v10
exp mrt0 v2, v4, v6, v11 done vm
s_endpgm
