; Depth-compare sampling test: image_sample_c_lz of an 8x8 D32 texture whose texel x holds depth (x + 0.5) / 8, with
; the reference 0.5 against a LESS_OR_EQUAL compare sampler (S# DEPTH_COMPARE_FUNC 3 at s[16:19]), at u, v from attr0.
; The same sample without and with d16; the d16 result (an f16 in the low half) is widened with v_cvt_f32_f16.
; T# at s[8:15] loaded via s[0:1]. mrt0 = (f32 result, widened d16 result, u, v).
; Run: vk_texture_test dref -> every pixel's first two components are 1.0 where its texel has x >= 4, else 0.0
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32_e32 v4, 0.5
v_mov_b32_e32 v5, v2
v_mov_b32_e32 v6, v3
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x3092
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
image_sample_c_lz v9, v[4:6], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_c_lz v10, v[4:6], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D d16
s_waitcnt vmcnt(0)
v_cvt_f32_f16_e32 v11, v10
exp mrt0 v9, v11, v2, v3 done vm
s_endpgm
