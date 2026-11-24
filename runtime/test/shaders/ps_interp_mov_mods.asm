; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; VOP3 output modifiers on v_interp_p2_f32_e64 and v_interp_mov_f32_e64 (docs/design.md "v_fmac_f16,
; v_interp_p2/mov_f32 output modifiers"). Drawn with vs_interp_consts: attr2 = (0.25, 3.0, 0, 0) interpolated,
; attr1 = (1.5, -0.5, 0.25, 4.0) read only with v_interp_mov (so its location is Flat).
;   x: interp 0.25 mul:4 = 1.0; y: interp 3.0 div:2 = 1.5; z: mov 4.0 div:2 = 2.0;
;   w: mov 1.5 div:2 = 0.75, then clamp (clamping first would give 0.5)
; Run: vk_harness vs_interp_consts.spv ps_interp_mov_mods.spv out.raw
; Expected mrt0 bits: 0x3f800000, 0x3fc00000, 0x40000000, 0x3f400000
v_interp_p1_f32_e32 v2, v0, attr2.x
v_interp_p2_f32_e64 v2, v1, attr2.x mul:4
v_interp_p1_f32_e32 v3, v0, attr2.y
v_interp_p2_f32_e64 v3, v1, attr2.y div:2
v_interp_mov_f32_e64 v4, p0, attr1.w div:2
v_interp_mov_f32_e64 v5, p0, attr1.x clamp div:2
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
