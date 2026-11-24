; Depth-compare sampling with a16: as ps_sample_c_lz (the same D32 texture, reference 0.5, LESS_OR_EQUAL sampler), but
; the z-compare keeps a whole VGPR as an f32 and only u, v after it are packed f16. Read as pairs, the reference
; would be the low half of 0.5 (0.0) and the coordinates a half early. mrt0 = (f32 result, widened d16 result, u, v).
; Run: vk_texture_test dref_a16 -> every pixel's first two components are 1.0 where its texel has x >= 4, else 0.0
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32_e32 v4, 0.5
v_cvt_pkrtz_f16_f32 v5, v2, v3
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x3092
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
image_sample_c_lz v9, v[4:5], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D a16
image_sample_c_lz v10, v[4:5], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D a16 d16
s_waitcnt vmcnt(0)
v_cvt_f32_f16_e32 v11, v10
exp mrt0 v9, v11, v2, v3 done vm
s_endpgm
