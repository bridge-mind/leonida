; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; VOP3 output modifiers on v_interp_p2_f32_e64 (docs/design.md "v_fmac_f16, v_interp_p2/mov_f32 output modifiers"),
; omod before clamp. Drawn with vs_interp_consts: attr0 = (1.5, -0.25, 0.375, 0.625) at every pixel.
;   x: clamp(1.5) = 1.0; y: clamp(-0.25) = 0.0; z: 0.375 mul:2 = 0.75; w: 0.625 mul:2 = 1.25, clamp: 1.0
;   (clamping before the multiply would give 1.25)
; Run: vk_harness vs_interp_consts.spv ps_interp_f32_mods.spv out.raw
; Expected mrt0 bits: 0x3f800000, 0x00000000, 0x3f400000, 0x3f800000
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e64 v2, v1, attr0.x clamp
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e64 v3, v1, attr0.y clamp
v_interp_p1_f32_e32 v4, v0, attr0.z
v_interp_p2_f32_e64 v4, v1, attr0.z mul:2
v_interp_p1_f32_e32 v5, v0, attr0.w
v_interp_p2_f32_e64 v5, v1, attr0.w clamp mul:2
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
