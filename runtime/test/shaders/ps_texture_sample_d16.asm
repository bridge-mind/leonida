; Texture sampling test with d16: samples a 2D texture into packed f16 pairs (v4=r|g<<16, v5=b|a<<16),
; then unpacks the four f16 halves back to f32 and outputs to mrt0.
; Run: vk_texture_test vs_ids.spv ps_texture_sample_d16.spv out.raw 0 0 0 0 0
;   -> 64x64 RGBA32F readback: each component is the nearest gradient texel's UNORM8 value rounded to f16, then widened.
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
image_sample v[4:5], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D d16
s_waitcnt vmcnt(0)
v_cvt_f32_f16_e32 v6, v4
v_lshrrev_b32_e32 v7, 16, v4
v_cvt_f32_f16_e32 v7, v7
v_cvt_f32_f16_e32 v8, v5
v_lshrrev_b32_e32 v9, 16, v5
v_cvt_f32_f16_e32 v9, v9
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
