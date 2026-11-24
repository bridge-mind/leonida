; Point-sample the R32_UINT and R32_SINT images at each texel centre and add the D16 results.
; vk_store_test load-d16 checks the signed/unsigned saturation boundaries in the last row,
; the sum (100 - 8y - x) elsewhere, and the untouched UINT image.
; The unregistered S# selects the harness's nearest, clamp-to-edge sampler.
v_and_b32_e32 v1, 7, v0
v_lshrrev_b32_e32 v2, 3, v0
v_cvt_f32_u32_e32 v3, v1
v_cvt_f32_u32_e32 v4, v2
v_add_f32_e32 v8, 0.5, v3
v_add_f32_e32 v9, 0.5, v4
v_mul_f32_e32 v8, 0x3e000000, v8
v_mul_f32_e32 v9, 0x3e000000, v9
s_mov_b32 s32, 0
s_mov_b32 s33, 0
s_mov_b32 s34, 0
s_mov_b32 s35, 0
v_mul_f32_e32 v3, 0x3e124925, v3
v_mul_f32_e32 v4, 0x3e124925, v4
v_mov_b32_e32 v5, 0.5
v_mov_b32_e32 v6, 1.0
s_mov_b32 s8, 0x1000
s_mov_b32 s9, 0x3800000
s_mov_b32 s10, 0x11
s_mov_b32 s11, 0x11
s_mov_b32 s12, 0x11
s_mov_b32 s13, 0x11
s_mov_b32 s14, 0x11
s_mov_b32 s15, 0x11
image_store v[3:6], v[1:2], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
s_mov_b32 s16, 0x2000
s_mov_b32 s17, 0x1400000
s_mov_b32 s18, 0x22
s_mov_b32 s19, 0x22
s_mov_b32 s20, 0x22
s_mov_b32 s21, 0x22
s_mov_b32 s22, 0x22
s_mov_b32 s23, 0x22
image_sample_lz v7, v[8:9], s[16:23], s[32:35] dmask:0x1 dim:SQ_RSRC_IMG_2D d16
s_mov_b32 s24, 0x3000
s_mov_b32 s25, 0x1500000
s_mov_b32 s26, 0x33
s_mov_b32 s27, 0x33
s_mov_b32 s28, 0x33
s_mov_b32 s29, 0x33
s_mov_b32 s30, 0x33
s_mov_b32 s31, 0x33
image_sample_lz v11, v[8:9], s[24:31], s[32:35] dmask:0x1 dim:SQ_RSRC_IMG_2D d16
s_waitcnt vmcnt(0)
v_and_b32_e32 v7, 0xffff, v7
v_bfe_i32 v11, v11, 0, 16
v_add_nc_u32_e32 v11, v7, v11
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v14, 0
image_store v[11:14], v[1:2], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
s_endpgm
