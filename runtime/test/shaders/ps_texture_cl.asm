; image_sample_cl on vk_texture_test's two-mip texture (--two-mip: 128x128 quadrants, solid cyan mip 1) drawn on 64x64:
; the coordinates are u / 4, v / 4, so the unclamped LOD is -1 (mip 0, the top-left quadrant, red). The min-LOD clamp
; after the coordinates is 0.0 left of x = 32 and 1.0 from there on: the left half must read mip 0 (red), the right half
; mip 1 (cyan). A clamp read from the wrong register, or ignored, makes the right half red too.
; Address v[4:6] = (u / 4, v / 4, clamp); MinLod on an implicit-LOD sample (OpImageSampleImplicitLod ... MinLod).
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
v_cmp_le_f32_e32 vcc_lo, 0.5, v2
v_mul_f32_e32 v4, 0.25, v2
v_mul_f32_e32 v5, 0.25, v3
v_cndmask_b32_e64 v6, 0, 1.0, vcc_lo
image_sample_cl v[8:11], v[4:6], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
