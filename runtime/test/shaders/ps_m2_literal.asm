; M2 literal sampler: ps_m2.asm with its S# built from immediates instead of loaded from the sampler table, as game
; shaders do (docs/design.md, Literal samplers): CLAMP_LAST_TEXEL on all axes, bilinear min/mag, point mip, MAX_LOD
; 12. No heap slot names this S#, so the layer registers it when a pipeline uses the shader; otherwise the shader's
; hash lookup finds the fallback sampler. Run: d3d12_m2_textured_test.
;   s[0:1] = console VA of the SRV table's first slot (the T#).   attr0.xy = (u, v).   mrt0 = the sampled texel.
; fssm-at: 0x10
; fssm-data: 0x324d5350 0x4c544c20
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_movk_i32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
image_sample v[4:7], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
